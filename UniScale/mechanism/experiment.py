"""Immutable registered task matrix and functional interventions."""

from pathlib import Path

from .data import context, key
from .protocol import SIGNS, load_evidence, write_arrays, write_json


def task_plan(config: dict) -> list[dict]:
    from .specificity import task_plan as specificity_plan
    from .controls import controls_plan
    tasks = []
    for process in config["processes"]:
        for model in config["models"]:
            tasks.append({"id": f"functional-{process}-{model['name']}", "kind": "functional",
                          "process": process, "model": model["name"], "environment": model["environment"], "depends": []})
    for length in sorted(config["training_history_lengths"], reverse=True):
        for seed in config["training_seeds"]:
            for process in config["processes"]:
                for architecture in config["architectures"]:
                    tasks.append({"id": f"weights-{process}-{architecture}-L{length}-seed{seed}",
                                  "kind": "weights", "process": process,
                                  "architecture": architecture, "history_length": length,
                                  "seed": seed, "environment": "toto", "depends": []})
    return tasks + controls_plan(config) + specificity_plan(config)


def run_functional(config: dict, root: Path, task: dict, model_root: Path) -> None:
    from .tsfm import load_model

    model = load_model(config, task["model"], model_root)
    evidence, arrays = load_evidence(root), {}
    process = task["process"]
    for magnitude in config["magnitudes"]:
        for split, label in enumerate(("development", "test")):
            if label == "development" and magnitude != config["phi_magnitude"]:
                continue
            name = f"{key(process, magnitude)}_{label}"
            query = evidence[f"{name}_query"]
            for normalization in ("native_statistics", "fixed_population"):
                for length in config["context_lengths"]:
                    for repeat in range(config["prefix_repeats"] if length > 96 else 1):
                        for sign in (SIGNS if length > 96 else (0,)):
                            values = query if length == 96 else context(config, process, magnitude, query, length, sign, repeat, split)
                            prediction, _ = model.infer(values, normalization=normalization)
                            arrays[f"{name}_{normalization}_L{length}_r{repeat}_s{sign:+d}"] = prediction
                    print(f"{task['id']}: a={magnitude} {label} {normalization} L={length}", flush=True)
    write_arrays(root, f"tasks/{task['id']}/predictions.npz", **arrays)
    write_json(root, f"tasks/{task['id']}/model.json", {
        **model.metadata, "effective_batch_size": model.batch_size,
        "oom_recoveries": model.recoveries, "process": process,
    })


def run_task(config: dict, root: Path, task: dict, model_root: Path) -> None:
    if task["kind"].startswith("specificity_"):
        from .specificity import run_task as run_specificity

        run_specificity(config, root, task, model_root)
        return
    if task["kind"] == "crossed_history":
        from .crossed_history import run_crossed_history

        run_crossed_history(config, root, task, model_root)
    elif task["kind"] == "functional":
        run_functional(config, root, task, model_root)
    elif task["kind"] == "weights":
        from .weights import run_weights

        run_weights(config, root, task)
    else:
        raise ValueError(f"Unknown registered task: {task['kind']}")
    if task["kind"] != "crossed_history":
        from .score_predictions import score_synthetic_task

        score_synthetic_task(config, root, task)
