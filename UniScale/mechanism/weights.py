"""Validated predictors and shared-history calibrated parameter transfer."""

from __future__ import annotations

import copy
import numpy as np

from .data import generator, key, stationary
from .protocol import SIGNS, write_arrays, write_json


def predict(model, values: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    chunks = []
    with torch.inference_mode():
        for offset in range(0, len(values), batch_size):
            x = torch.as_tensor(values[offset:offset + batch_size], device="cuda")[:, :, None]
            chunks.append(model(x)[:, :, 0].cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite trained predictor output")
    return result


def calibrated_transfer(recipient, donor, calibration, batch_size):
    """Copy parameters and recalibrate BN on the same shared observed histories."""
    import torch

    model = copy.deepcopy(recipient).eval()
    parameters = dict(donor.named_parameters())
    state = model.state_dict()
    for name, value in parameters.items():
        state[name] = value.detach().clone()
    model.load_state_dict(state, strict=True)
    modules = [module for module in model.modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    if modules:
        with torch.no_grad():
            for offset in range(0, len(calibration), batch_size):
                model(torch.as_tensor(calibration[offset:offset + batch_size, :, None], device="cuda"))
    model.eval()
    if any(not torch.equal(value, parameters[name]) for name, value in model.named_parameters()):
        raise RuntimeError("Calibration changed transferred parameters")
    return model, len(modules)


def run_weights(config, root, task):
    from .training import train_pair

    models, training = train_pair(config, root, task)
    spec = config["state_transfer"]
    sampling = {**config, "data_seed": spec["calibration_seed"]}
    calibration = np.concatenate([
        stationary(generator(sampling, task["process"], config["phi_magnitude"], 801, task["seed"]),
                   task["process"], spec["calibration_windows_per_rule"], 96,
                   sign * config["phi_magnitude"]).astype(np.float32)
        for sign in SIGNS
    ])
    arrays, module_counts = {}, {}
    with np.load(root / "data/state_evidence.npz", allow_pickle=False) as evidence:
        for recipient in SIGNS:
            transferred, count = calibrated_transfer(models[recipient], models[-recipient], calibration,
                                                       spec["calibration_batch_size"])
            module_counts[str(recipient)] = count
            for split in ("development", "test"):
                query = evidence[f"{key(task['process'], config['phi_magnitude'])}_{split}_query"]
                arrays[f"{split}_own_s{recipient:+d}"] = predict(models[recipient], query, config["inference_batch_size"])
                arrays[f"{split}_calibrated_parameters_recipient{recipient:+d}"] = predict(
                    transferred, query, config["inference_batch_size"])
            del transferred
    write_arrays(root, f"tasks/{task['id']}/predictions.npz", **arrays)
    write_json(root, f"tasks/{task['id']}/training.json", {**task, **training})
    write_json(root, f"tasks/{task['id']}/intervention.json", {
        "task": task, "method": spec["method"], "batchnorm_module_counts": module_counts,
        "calibration_seed": spec["calibration_seed"], "calibration_windows": len(calibration),
        "calibration_batch_size": spec["calibration_batch_size"], "calibration_length": 96,
        "shared_across_directions": True, "future_targets_used": False, "optimizer_updates": 0,
        "no_batchnorm_behavior": "Parameter transfer; no calibration forward passes required",
    })
