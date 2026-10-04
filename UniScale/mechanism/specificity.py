"""Registered rule matching, site robustness and cross-model interventions."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np

from .data import key
from .protocol import SIGNS, digest_file, write_arrays, write_json
from .transfer import InterventionSession, OBJECTIVES


def task_plan(config: dict) -> list[dict]:
    tasks = []
    for model, kind in (("timesfm25", "specificity_timesfm"),
                        ("chronos2", "specificity_development")):
        for site, process in product(range(3), config["processes"]):
            tasks.append({"id": f"specificity-{model}-{process}-site{site}", "kind": kind,
                          "model": model, "process": process, "site": site,
                          "environment": "TSFM", "slot_cost": 2, "depends": []})
    development = [task["id"] for task in tasks if task["kind"] == "specificity_development"]
    tasks.append({"id": "specificity-select", "kind": "specificity_selection",
                  "environment": "TSFM", "slot_cost": 1, "depends": development})
    for process in config["processes"]:
        tasks.append({"id": f"specificity-chronos2-confirm-{process}",
                      "kind": "specificity_confirmation", "model": "chronos2", "process": process,
                      "environment": "TSFM", "slot_cost": 2, "depends": ["specificity-select"]})
    return tasks


def _select(config: dict, root: Path, task: dict) -> None:
    records, hashes = [], {}
    for identifier in task["depends"]:
        directory = root / "tasks" / identifier
        index_path, metrics_path = directory / "intervention.json", directory / "point_metrics.json"
        report, metrics = json.loads(index_path.read_text()), json.loads(metrics_path.read_text())
        for path in (index_path, metrics_path):
            hashes[str(path.relative_to(root))] = digest_file(path)
        for item in report["prediction_index"]:
            if item["split"] != "development":
                raise ValueError("Selection must not read confirmation predictions")
            if item["condition"] != "intervention":
                continue
            score = metrics["conditions"][item["key"]]["conditional_mean"][0]["relative"]["mase"]
            records.append({"layer": item["candidate"]["layer"],
                            "strength": item["candidate"]["strength"], "score": score})
    candidates = sorted({(row["layer"], row["strength"]) for row in records})
    expected = len(config["processes"]) * len(config["magnitudes"]) * config["activation_transfer"]["donor_repeats"] * len(SIGNS) * len(OBJECTIVES)
    if any(sum((row["layer"], row["strength"]) == candidate for row in records) != expected
           for candidate in candidates):
        raise ValueError("Candidate selection requires complete, balanced development conditions")
    scores = [{"layer": layer, "strength": strength,
               "score": float(np.mean([row["score"] for row in records
                                       if (row["layer"], row["strength"]) == (layer, strength)]))}
              for layer, strength in candidates]
    if len(scores) != 9 or not all(np.isfinite(item["score"]) for item in scores):
        raise ValueError("Development comparison must contain nine finite candidate scores")
    selected = min(scores, key=lambda item: (item["score"], item["layer"], item["strength"]))
    write_json(root, f"tasks/{task['id']}/selection.json", {
        "selected": selected, "candidates": scores, "source_sha256": hashes,
        "criterion": config["activation_design"]["selection"],
        "test_access": "No test predictions used; one setting shared across all processes and objectives",
    })


def run_task(config: dict, root: Path, task: dict, model_root: Path) -> None:
    if task["kind"] == "specificity_selection":
        _select(config, root, task)
        return
    from .activation import chronos2_forward, forward
    from .score_predictions import score_synthetic_task
    from .tsfm import load_model

    specification = config["activation_design"]
    transfer = config["activation_transfer"]
    model = load_model(config, task["model"], model_root)
    model.module.eval()
    selection_sha = None
    if task["kind"] == "specificity_confirmation":
        selection_path = root / "tasks/specificity-select/selection.json"
        chosen = json.loads(selection_path.read_text())["selected"]
        layer, strengths = chosen["layer"], [chosen["strength"]]
        selection_sha = digest_file(selection_path)
    else:
        if task["model"] == "timesfm25":
            layer = specification["timesfm_layers"][task["site"]]
        else:
            depth = len(model.module.encoder.block)
            fraction = specification["chronos_depth_fractions"][task["site"]]
            layer = int(round(depth * fraction)) - 1
        strengths = specification["strengths"]
    if task["model"] == "chronos2":
        hook_forward, patch_size = chronos2_forward, model.module.chronos_config.input_patch_size
        if patch_size != 16 or len(model.module.encoder.block) != 12:
            raise ValueError("Chronos checkpoint token/depth layout changed")
    else:
        hook_forward, patch_size = forward, model.module.p
        if patch_size != 32:
            raise ValueError("TimesFM checkpoint patch layout changed")
    split = "development" if task["kind"] == "specificity_development" else "test"
    with np.load(root / "data/transfer_evidence.npz", allow_pickle=False) as source:
        evidence = {name: source[name] for name in source.files}
    arrays, index = {}, []
    forward_check = None
    candidate = {"method": "residual_short", "layer": layer,
                 "length": transfer["candidate"]["length"],
                 "donor_count": transfer["candidate"]["donor_count"], "strength": 1.0}
    experiment = {**config, "activation_transfer": {
        **config["activation_transfer"], "candidate": candidate}}
    for magnitude, repeat in product(config["magnitudes"], range(transfer["donor_repeats"])):
        query = evidence[f"{key(task['process'], magnitude)}_{split}_query"]
        session = InterventionSession(experiment, model, task["process"], magnitude, query, repeat,
                                      forward_fn=hook_forward, patch_size=patch_size)
        # Independent donor streams prevent development/confirmation donor reuse.
        if split == "development":
            session.settings = {**session.settings, "donor_seed": transfer["donor_seed"] + 100000}
        # Keep batch composition identical when checking the two forward paths.
        if magnitude == config["magnitudes"][0] and repeat == 0:
            ordinary, _ = model.infer(query, normalization="fixed_population")
            forward_check = {"queries": len(query), "batch_size": model.batch_size,
                             "max_absolute_difference": float(np.max(np.abs(ordinary - session.local)))}
            if not np.allclose(ordinary, session.local, rtol=1e-4, atol=1e-5):
                raise RuntimeError(f"Unpatched intervention path differs from registered inference: {forward_check}")
            print(f"{task['id']}: forward equivalence passed {forward_check}", flush=True)
        recipients = {sign: session.recipient(sign) for sign in SIGNS}
        pools = {sign: session.donor_pool(sign) for sign in SIGNS}
        for strength in strengths:
            current_candidate = {**candidate, "strength": strength}
            session.settings = {**session.settings, "candidate": current_candidate}
            for sign, objective in product(SIGNS, OBJECTIVES):
                recipient, pool = recipients[sign], pools[sign]
                prediction = session.predict(objective, recipient, pool["delta"])
                outputs = session.controls(recipient, pool, objective, prediction)
                outputs["opposite_rule"] = session.predict(objective, recipient, pools[-sign]["delta"])
                for condition, values in outputs.items():
                    if not np.isfinite(values).all():
                        raise FloatingPointError("Nonfinite rule-specificity output")
                    name = f"p{len(index):05d}"
                    arrays[name] = values
                    index.append({"key": name, "model": task["model"], "process": task["process"],
                                  "split": split, "magnitude": magnitude, "repeat": repeat,
                                  "sign": sign, "objective": objective, "condition": condition,
                                  "candidate": current_candidate,
                                  "donor_sign": -sign if condition == "opposite_rule" else sign})
        print(f"{task['id']}: a={magnitude}, repetition={repeat} complete", flush=True)
    write_arrays(root, f"tasks/{task['id']}/predictions.npz", **arrays)
    write_json(root, f"tasks/{task['id']}/intervention.json", {
        "task": task, "model": model.metadata, "split": split, "prediction_index": index,
        "effective_batch_size": model.batch_size, "oom_recoveries": model.recoveries,
        "patch_size": patch_size, "query_token_count": 96 // patch_size,
        "selection_sha256": selection_sha,
        "forward_equivalence": forward_check,
        "token_scope": "Final 96 observed values; REG and future tokens are not replaced",
        "recipient_matching": "Identical query and long recipient history for matched/opposite donor comparisons",
        "donor_matching": "Same carriers and conditional sampling random numbers across rule signs",
        "normalization": "Fixed population statistics, unique Chronos group IDs per query",
        "selection": "All TimesFM settings retained; Chronos confirmation uses development-selected shared setting",
        "restoration": "Exact natural-state and zero-change checks for every condition",
    })
    score_synthetic_task(config, root, task)
