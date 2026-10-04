"""GIFT scores from the saved predictions of every synthetic mechanism task."""

from __future__ import annotations

import json
import re
import numpy as np

from .data import key
from .metrics import GiftPointAccumulator, SCHEMA, naive_point, scoring_scale
from .protocol import write_json


def score_synthetic_task(config, root, task):
    kind = task["kind"]
    if kind.startswith("specificity_"):
        kind = "activation_transfer"
    evidence_file = {"functional": "evidence", "weights": "state_evidence",
                     "activation_transfer": "transfer_evidence"}[kind]
    with np.load(root / f"data/{evidence_file}.npz", allow_pickle=False) as source:
        evidence = {name: source[name] for name in source.files}
    directory = root / "tasks" / task["id"]
    with np.load(directory / "predictions.npz", allow_pickle=False) as source:
        predictions = {name: source[name] for name in source.files}
    index = {}
    if kind == "activation_transfer":
        index = {item["key"]: item for item in json.loads((directory / "intervention.json").read_text())["prediction_index"]}
    conditions, cache = {}, {}
    for name, prediction in predictions.items():
        if kind == "functional":
            match = re.fullmatch(r"(.+_(?:test|development))_(?:native_statistics|fixed_population)_L\d+_r\d+_s([+-]\d+)", name)
            if match is None:
                raise ValueError(f"Unregistered functional prediction key: {name}")
            prefix, sign = match[1], int(match[2])
        elif kind == "weights":
            match = re.fullmatch(r"(test|development)_(own_s|calibrated_parameters_recipient)([+-]\d+)", name)
            if match is None:
                raise ValueError(f"Unregistered parameter prediction key: {name}")
            prefix = f"{key(task['process'], config['phi_magnitude'])}_{match[1]}"
            sign = int(match[3]) * (-1 if match[2].endswith("recipient") else 1)
        else:
            record = index[name]
            prefix = f"{key(task['process'], record['magnitude'])}_{record['split']}"
            sign = record["sign"]
        signs = (-1, 1) if sign == 0 else (sign,)
        if prefix not in cache:
            query = evidence[prefix + "_query"]
            cache[prefix] = (scoring_scale(query, 1), naive_point(query, 3, 1))
        scales, baseline = cache[prefix]
        item = {}
        target_types = ["conditional_mean"]
        if kind == "weights" and "_own_" in name:
            target_types.append("conditional_mean_opposite_rule")
        for target_type in target_types:
            results = []
            for h in range(3):
                scorer = GiftPointAccumulator()
                for s in signs:
                    target_sign = -s if target_type.endswith("_opposite_rule") else s
                    labels = evidence[f"{prefix}_oracle{target_sign:+d}"]
                    scorer.update(s, labels[:, h:h+1], prediction[:, h:h+1], scales,
                                  baseline[:, h:h+1])
                _, result = scorer.finish()
                results.append({"horizon": h + 1, **result})
            item[target_type] = results
        conditions[name] = item
    write_json(root, f"tasks/{task['id']}/point_metrics.json", {
        "schema": SCHEMA, "seasonality": 1,
        "scoring_history": "Common 96-observation query; identical scaling and naive forecast across paired interventions",
        "horizon_scope": "Each horizon endpoint is scored separately, not cumulatively",
        "targets": "Conditional mean at each horizon endpoint",
        "point_forecast": "Saved deterministic point used for both mean and median metrics",
        "conditions": conditions,
    })
