"""Train the registered validation-selected synthetic predictors."""

from __future__ import annotations

import math
import numpy as np

from .data import generator, stationary
from .protocol import SIGNS


def train_pair(config, root, task):
    import torch
    from UniScale.vendor.timeseries_library.models import build_model

    spec = config["state_transfer"]
    process, length, seed = task["process"], task["history_length"], task["seed"]
    architecture, magnitude = task["architecture"], config["phi_magnitude"]
    split, budget = int(length * spec["training_fraction"]), config["training_updates"]
    hyper = dict(config["patchtst"]) if architecture == "PatchTST" else {}
    batch_size = config["training_batch_size"]
    variant = {"lr": config["training_learning_rate"],
               "weight_decay": spec["patchtst_weight_decay"] if architecture == "PatchTST" else 0.0,
               "schedule": spec["patchtst_schedule"] if architecture == "PatchTST" else "constant"}
    models, records = {}, {}
    initial = None

    def cpu_state(model):
        return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    def validation_mse(model, x, y):
        model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for offset in range(0, len(x), batch_size):
                error = (model(x[offset:offset + batch_size])[:, 0] - y[offset:offset + batch_size, 0]) ** 2
                total += float(error.sum())
                count += error.numel()
        score = total / count
        if not math.isfinite(score):
            raise RuntimeError("Nonfinite validation error")
        return score

    for sign in SIGNS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = build_model(architecture, 96, 3, hyper).cuda()
        state = cpu_state(model)
        if initial is None:
            initial = state
        elif any(not torch.equal(initial[name], value) for name, value in state.items()):
            raise RuntimeError("Opposite rules must share initialization")
        rng = generator(config, process, magnitude, 40, seed)
        history = stationary(rng, process, 1, max(config["training_history_lengths"]), sign * magnitude)[0, :length].astype(np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(history, 99).copy()
        values = torch.as_tensor(windows, device="cuda")
        train = values[:split - 98]
        validation = values[split - 96:]
        x, y = train[:, :96, None], train[:, 96:, None]
        optimizer = torch.optim.Adam(model.parameters(), lr=variant["lr"], weight_decay=variant["weight_decay"])
        sampler = torch.Generator(device="cuda").manual_seed(seed + 10000)
        best_score, best_step, best_state = math.inf, 0, None
        trace = []
        for step in range(1, budget + 1):
            lr = variant["lr"]
            if variant["schedule"] == "cosine":
                lr *= 0.01 + 0.99 * (1 + math.cos(math.pi * (step - 1) / max(1, budget - 1))) / 2
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            indices = torch.randint(len(x), (batch_size,), generator=sampler, device="cuda")
            loss = torch.mean((model(x[indices]) - y[indices]) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss: {task['id']} sign={sign} step={step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % spec["validation_interval"] == 0 or step == budget:
                score = validation_mse(model, validation[:, :96, None], validation[:, 96:, None])
                if score < best_score - spec["validation_min_delta"]:
                    best_score, best_step, best_state = score, step, cpu_state(model)
                trace.append({"step": step, "train_batch_mse_h123": float(loss.detach()),
                              "validation_mse_h1": score, "lr": lr})
            if step == 1 or step % 500 == 0:
                print(f"{task['id']} sign={sign:+d} step={step}/{budget} train_mse={float(loss.detach()):.6f}", flush=True)
        model.eval()
        if best_state is None:
            raise RuntimeError("No validation-selected model was produced")
        model.load_state_dict(best_state, strict=True)
        selected_step = best_step
        models[sign] = model
        records[str(sign)] = {"trace": trace, "actual_updates": step, "selected_step": selected_step,
                              "best_validation_mse_h1": best_score if math.isfinite(best_score) else None,
                              "early_stopped": False, "sampled_window_exposures": step * batch_size / len(train)}

    return models, {"rules": records, "hyperparameters": hyper,
                    "training_record_seed": config["data_seed"], "split_index": split,
                    "training_windows": split - 98, "validation_windows": length - split - 2,
                    "maximum_updates": budget, "batch_size": batch_size, "refit": False,
                    "selection_metric": "observed validation MSE, horizon 1",
                    "loss": "observed future MSE, horizons 1/2/3"}
