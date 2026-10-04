"""Cross real training history and immediate context on the full fixed task panel.

All contexts share training target starts and chronological validation targets.
The initial TimesFM predictor is evaluated before any updates. Training never
reads test labels for selection, and no source checkpoint is overwritten.
"""

from __future__ import annotations

import json
import hashlib
import math
import time
from types import SimpleNamespace

import numpy as np

from .controls import project_file
from .protocol import write_arrays, write_json, output_path


def causal_preparation(histories, context, horizon, settings, seasonality):
    """Preserve declared policies with target-disjoint input imputation."""
    from UniScale.experiments.full_shot import (
        prepare_full_shot_data, _scalar_histories, _dataset_scalar_fallback,
        chronological_selection_layout, interpolate_missing, TemporalCoverageWindowPool,
    )

    if settings.normalization_policy == "series_window_std_v1":
        return prepare_full_shot_data(histories, context, horizon, settings, seasonality)
    scalar = _scalar_histories(histories)
    layouts = [chronological_selection_layout(len(values), context, horizon,
               settings.validation_origins_per_series, settings.minimum_training_windows) for values in scalar]
    prefixes = [values[:layout.first_validation_target_start] for values, layout in zip(scalar, layouts)]
    fallback = _dataset_scalar_fallback(prefixes)
    clean = [interpolate_missing(values, fallback)[0] for values in prefixes]
    inputs, targets, observed, series = [], [], [], []
    for i, (raw, layout) in enumerate(zip(scalar, layouts)):
        for origin in layout.validation_target_starts:
            available = interpolate_missing(raw[:origin], fallback)[0]
            target = raw[origin:origin + horizon]
            mask = np.isfinite(target)
            inputs.append(available[-context:])
            targets.append(np.where(mask, target, 0.0))
            observed.append(mask)
            series.append(i)
    return SimpleNamespace(
        selection_pool=TemporalCoverageWindowPool(clean, [item.training_window_count for item in layouts],
                        context, horizon, settings.temporal_coverage_stride, [len(value) for value in scalar]),
        validation_inputs=np.asarray(inputs, dtype=np.float32),
        validation_targets=np.asarray(targets, dtype=np.float32),
        validation_observed=np.asarray(observed), validation_series_indices=np.asarray(series),
        validation_error_scales=None, fallback_values=fallback, selection_history_scales=np.ones(1),
    )


def run_crossed_history(config, root, task, model_root):
    import torch
    from gluonts.time_feature import get_seasonality
    from UniScale.experiments.data import ControlledDataset
    from UniScale.experiments.matched_history import _earliest_training_histories, _target_time_dimensions
    from UniScale.experiments.full_shot import (
        FullShotTrainingSettings, _normalize_batch,
        _validation_score, forecast_with_checkpoint,
    )
    from UniScale.vendor.timeseries_library.models import build_model
    from .tsfm import load_model
    from .metrics import GiftPointAccumulator, scoring_scale, naive_point, SCHEMA, COLUMNS

    x = config["controls"]["crossed"]
    declared = json.loads(project_file(x["training_config"]).read_text())
    model_name = task["model"]
    architecture = "PatchTST" if model_name == "PatchTST-compact" else model_name
    model_spec = next((m for m in declared["models"] if m["name"] == architecture), None)
    overrides = {**declared.get("dataset_training_overrides", {}),
                 **(model_spec or {}).get("dataset_training_overrides", {})}
    payload = {**declared["training"], **overrides.get(task["dataset"], {}),
               "seed": task["seed"], "learning_rate": x["learning_rates"][model_name]}
    settings = FullShotTrainingSettings.from_config(payload, x["batch_size"])
    context, horizon = task["input_length"], x["horizon"]
    source = ControlledDataset(name=task["dataset"], term="short", to_univariate=False,
                               prediction_length=horizon, origin_horizon=x["origin_horizon"])
    identifiers = []
    histories, available_min, available_max = _earliest_training_histories(
        source, task["history_length"], identifiers)
    training_digest = hashlib.sha256()
    for identifier, values in zip(identifiers, histories):
        training_digest.update(str(identifier).encode())
        training_digest.update(str(values.shape).encode())
        training_digest.update(values.astype(np.float32).tobytes())
    seasonality = get_seasonality(source.freq)
    # Building once at Cmax keeps training starts and validation targets identical
    # across C. Cropping only the inputs changes immediate information, not labels.
    prepared = causal_preparation(histories, max(x["input_lengths"]), horizon, settings, seasonality)
    torch.manual_seed(task["seed"])
    torch.cuda.manual_seed_all(task["seed"])
    frozen_metadata = None
    if model_name == "TimesFM":
        wrapped = load_model(config, "timesfm25", model_root)
        frozen_metadata = wrapped.metadata

        class PointForecaster(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.backbone = module

            def forward(self, values):
                patches = values[..., 0].reshape(len(values), -1, self.backbone.p)
                (_, _, outputs, _), _ = self.backbone(patches, torch.zeros_like(patches, dtype=torch.bool))
                return outputs[:, -1].reshape(len(values), self.backbone.o, self.backbone.q)[:, :horizon, 5:6]

        model = PointForecaster(wrapped.module).cuda()
        model.requires_grad_(True)
        # The unused spread head is outside the optimized median forecast path.
        model.backbone.output_projection_quantiles.requires_grad_(False)
        hyperparameters = {"checkpoint": frozen_metadata["weights_sha256"], "output": "native median head"}
    else:
        hyperparameters = dict(model_spec["hyperparameters"])
        if architecture == "PatchTST":
            hyperparameters["internal_normalization"] = False
            if model_name == "PatchTST-compact":
                hyperparameters.update(width=32, feedforward_width=64, encoder_layers=1)
        model = build_model(architecture, context, horizon, hyperparameters).cuda()
    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=settings.learning_rate,
                                 weight_decay=x["weight_decay"] if architecture == "PatchTST" else 0.0)
    checkpoint = SimpleNamespace(
        model=model, normalization_policy=settings.normalization_policy,
        input_length=context, horizon=horizon, fallback_values=prepared.fallback_values,
        history_record_ids=tuple(identifiers), history_scales=prepared.selection_history_scales,
    )
    device = torch.device("cuda")
    rng = np.random.default_rng(task["seed"] + 1)
    traces, costs, evaluation_rows = [], [], []
    metric_arrays, metric_results, scoring_inputs = {}, {}, {}
    best_score, best_step, best_state = math.inf, None, None
    training_seconds, examples_seen = 0.0, 0
    effective_batch = x["batch_size"]
    microbatch = x["microbatch_size"]
    evaluation_signature = None
    evaluation_contexts = None

    def validation():
        if prepared.validation_error_scales is None:
            model.eval()
            totals = np.zeros(len(prepared.selection_pool.histories), dtype=np.float64)
            counts = np.zeros_like(totals)
            with torch.no_grad():
                for start in range(0, len(prepared.validation_inputs), microbatch):
                    inputs = torch.as_tensor(prepared.validation_inputs[start:start + microbatch, -context:], device=device)
                    targets = torch.as_tensor(prepared.validation_targets[start:start + microbatch], device=device)
                    mask = torch.as_tensor(prepared.validation_observed[start:start + microbatch], device=device)
                    normalized, mean, std = _normalize_batch(inputs, settings.normalization_policy)
                    errors = (model(normalized) - (targets - mean) / std).abs()
                    n = mask.sum(dim=(1, 2))
                    losses = (errors * mask).sum(dim=(1, 2)) / n.clamp_min(1)
                    if not torch.isfinite(losses).all():
                        raise FloatingPointError("Nonfinite crossed validation loss")
                    ids = prepared.validation_series_indices[start:start + microbatch]
                    valid = n.cpu().numpy() > 0
                    np.add.at(totals, ids[valid], losses.cpu().numpy()[valid])
                    np.add.at(counts, ids[valid], 1)
            valid = counts > 0
            if not valid.any():
                raise ValueError("No observed validation targets")
            return float(np.mean(totals[valid] / counts[valid]))
        return _validation_score(model, prepared.validation_inputs[:, -context:],
                                 prepared.validation_targets, microbatch, device,
                                 settings.normalization_policy, prepared.validation_error_scales,
                                 prepared.validation_observed, prepared.validation_series_indices)

    def evaluate(label, updates):
        nonlocal evaluation_signature, evaluation_contexts
        model.eval()
        started = time.monotonic()
        contexts = []
        scorer = GiftPointAccumulator()
        fingerprint = hashlib.sha256()
        for index, (entry, target_entry) in enumerate(zip(source.test_data.input, source.test_data.label)):
            past = _target_time_dimensions(entry)
            target = _target_time_dimensions(target_entry)
            fingerprint.update(str((entry["item_id"], len(past), target.shape)).encode())
            fingerprint.update(past[-max(x["input_lengths"]):].astype(np.float32).tobytes())
            fingerprint.update(target.astype(np.float32).tobytes())
            predictions, _ = forecast_with_checkpoint(
                checkpoint, [past], batch_size=microbatch, device_name="cuda",
                record_ids=[str(entry["item_id"])],
            )
            predicted = predictions[0]
            if predicted.shape != target.shape:
                raise ValueError("Forecast and target shape mismatch")
            if index not in scoring_inputs:
                histories = past.T
                scoring_inputs[index] = (scoring_scale(histories, seasonality),
                                         naive_point(histories, horizon, seasonality))
            scales, scoring_baseline = scoring_inputs[index]
            scorer.update(index, target.T, predicted.T, scales, scoring_baseline)
            contexts.append(len(past))
        if not contexts:
            raise ValueError("Crossed task has no evaluation origins")
        if evaluation_signature is not None and evaluation_signature != fingerprint.hexdigest():
            raise RuntimeError("Evaluation inputs or targets changed between update budgets")
        evaluation_signature = fingerprint.hexdigest()
        evaluation_contexts = contexts
        metric_arrays[label], metric_results[label] = scorer.finish()
        # Loss sums are raw evaluation outputs. Dataset/seed aggregation is local.
        evaluation_rows.append({"condition": label, "updates": updates,
                                "origin_records": len(contexts), "evaluation_seconds": time.monotonic() - started})

    score = validation()
    best_score, best_step = score, 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    traces.append({"step": 0, "validation": score, "training_objective": None})
    evaluate("step0", 0)
    iterator = iter(prepared.selection_pool.iter_epoch(effective_batch, rng))
    step = 0
    while step < x["updates"]:
        pieces_x, pieces_y, remaining = [], [], effective_batch
        while remaining:
            try:
                bx, by = next(iterator)
            except StopIteration:
                iterator = iter(prepared.selection_pool.iter_epoch(effective_batch, rng))
                bx, by = next(iterator)
            # Each epoch's final batch may be short; complete it before an update.
            take = min(remaining, len(bx))
            pieces_x.append(bx[:take, -context:])
            pieces_y.append(by[:take])
            remaining -= take
        bx, by = np.concatenate(pieces_x), np.concatenate(pieces_y)
        # Retry only before any optimizer step; preserve RNG and BN state on OOM.
        while True:
            cuda_rng = torch.cuda.get_rng_state()
            cpu_rng = torch.get_rng_state()
            saved_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
            torch.cuda.synchronize()
            start = time.monotonic()
            try:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                lr = settings.learning_rate * (0.01 + 0.99 * (1 + math.cos(math.pi * step / x["updates"])) / 2)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                for offset in range(0, effective_batch, microbatch):
                    inputs = torch.as_tensor(bx[offset:offset + microbatch], device=device)
                    targets = torch.as_tensor(by[offset:offset + microbatch], device=device)
                    normalized, mean, std = _normalize_batch(inputs, settings.normalization_policy)
                    error = model(normalized) - (targets - mean) / std
                    loss = (error.abs() if settings.training_loss == "mae" else error.square()).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite crossed-history training objective")
                    (loss * len(inputs) / effective_batch).backward()
                    loss_sum += float(loss.detach()) * len(inputs) / effective_batch
                torch.nn.utils.clip_grad_norm_(trainable, settings.gradient_clip_norm, error_if_nonfinite=True)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                inputs = targets = normalized = mean = std = error = loss = None
                with torch.no_grad():
                    for name, value in model.named_buffers():
                        value.copy_(saved_buffers[name])
                torch.cuda.set_rng_state(cuda_rng)
                torch.set_rng_state(cpu_rng)
                if microbatch <= 2:
                    raise
                microbatch //= 2
                torch.cuda.empty_cache()
                print(f"{task['id']}: retry update with microbatch={microbatch}", flush=True)
                continue
            # An OOM inside the optimizer is a task failure, never a partial-step retry.
            optimizer.step()
            torch.cuda.synchronize()
            training_seconds += time.monotonic() - start
            break
        step += 1
        examples_seen += effective_batch
        if step % x["validation_interval"] == 0 or step == x["updates"]:
            score = validation()
            traces.append({"step": step, "validation": score, "training_objective": loss_sum, "lr": lr})
            if score < best_score - settings.selection_improvement_tolerance:
                best_score, best_step = score, step
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if step in x["report_updates"]:
            costs.append({"step": step, "training_gpu_seconds": training_seconds,
                          "examples_seen": examples_seen, "input_tokens_seen": examples_seen * context,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
            evaluate(f"step{step}", step)
            print(f"{task['id']}: updates={step}, validation={score:.6g}", flush=True)
    model.load_state_dict(best_state, strict=True)
    evaluate("selected", best_step)
    if x["save_checkpoints"]:
        torch.save({"state_dict": best_state, "task": task, "selected_step": best_step},
                   output_path(root, f"tasks/{task['id']}/selected.pt"))
    write_arrays(root, f"tasks/{task['id']}/point_statistics.npz", **metric_arrays)
    write_json(root, f"tasks/{task['id']}/point_metrics.json", {
        "schema": SCHEMA, "columns": COLUMNS, "seasonality": seasonality,
        "scoring_history": "Complete raw history before each registered forecast origin",
        "point_forecast": "Saved deterministic point used for both mean and median metrics",
        "conditions": metric_results,
    })
    write_json(root, f"tasks/{task['id']}/training.json", {
        "task": task, "hyperparameters": hyperparameters, "trace": traces, "costs": costs,
        "raw_training_histories_sha256": training_digest.hexdigest(),
        "evaluation_inputs_targets_sha256": evaluation_signature,
        "selected_step": best_step, "selected_validation": best_score,
        "validation_includes_unmodified_predictor": True, "refit": False,
        "normalization_policy": settings.normalization_policy, "training_loss": settings.training_loss,
        "training_history_available_min": available_min, "training_history_available_max": available_max,
        "effective_training_history_lengths": [len(history) for history in histories],
        "training_windows": prepared.selection_pool.total_window_count,
        "validation_examples": len(prepared.validation_inputs), "scalar_series": len(prepared.selection_pool.histories),
        "common_target_context": max(x["input_lengths"]), "immediate_context": context,
        "missing_data": "Training imputation and fallback use pre-validation history; validation input imputation stops at each origin",
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "optimized_parameters": sum(value.numel() for value in trainable),
        "microbatch": microbatch, "effective_batch": effective_batch, "frozen_source": frozen_metadata,
        "forecast_path": "Per-window normalization; first native median block for TimesFM; no rollout or carried state",
        "shared_origins": source.origin_policy, "evaluations": evaluation_rows,
        "evaluation_available_history_lengths": evaluation_contexts,
        "columns": COLUMNS,
        "statistics_scope": "GIFT per-variable/per-origin point and SeasonalNaive losses with masked observation counts",
        "compute_scope": "Measured fine-tuning/training costs; historical pretraining compute is not equated",
        "source_weights_modified": False,
    })
