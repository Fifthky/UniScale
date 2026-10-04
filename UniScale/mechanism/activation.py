"""Registered residual-state capture and replacement in frozen TimesFM."""

from __future__ import annotations

import numpy as np
import torch


def forward(model, values: np.ndarray, tokens: list[int], layers: list[int] = (),
            intervention: dict | None = None) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Capture or replace residual query tokens, retaining downstream computation."""
    predictions, states = [], {layer: [] for layer in layers}
    offset = 0
    while offset < len(values):
        width = min(model.batch_size, len(values) - offset)
        captured, handles = {}, []
        try:
            def make_hook(layer):
                def hook(module, arguments, output):
                    hidden = output[0]
                    if layer in layers:
                        captured[layer] = hidden[:, tokens].detach().cpu().numpy().copy()
                    if intervention is not None and layer == intervention["layer"]:
                        supplied = intervention["values"][offset:offset + width]
                        patched = torch.as_tensor(supplied, device=hidden.device, dtype=hidden.dtype)
                        hidden = hidden.clone()
                        hidden[:, tokens] = patched
                    return hidden, output[1]
                return hook
            active_layers = set(layers) | ({intervention["layer"]} if intervention is not None else set())
            for layer in sorted(active_layers):
                if not 0 <= layer < model.module.x - 1:
                    raise ValueError("A downstream Transformer layer is required")
                handles.append(model.module.stacked_xf[layer].register_forward_hook(make_hook(layer)))
            with torch.inference_mode():
                tensor = torch.as_tensor(values[offset:offset + width], device="cuda", dtype=torch.float32)
                patches = tensor.reshape(width, -1, model.module.p)
                (_, _, output, _), _ = model.module(patches, torch.zeros_like(patches, dtype=torch.bool))
                prediction = output[:, -1].reshape(width, model.module.o, model.module.q)[:, :3, 5]
            prediction = prediction.cpu().numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError("Nonfinite activation intervention output")
            predictions.append(prediction)
            for layer in layers:
                states[layer].append(captured[layer])
            offset += width
        except torch.cuda.OutOfMemoryError:
            if width == 1:
                raise
            model.batch_size = max(1, width // 2)
            model.recoveries.append({"offset": offset, "batch_size": model.batch_size})
            torch.cuda.empty_cache()
        finally:
            for handle in handles:
                handle.remove()
    return np.concatenate(predictions), {layer: np.concatenate(chunks) for layer, chunks in states.items()}


def chronos2_forward(model, values: np.ndarray, tokens: list[int], layers: list[int] = (),
                    intervention: dict | None = None) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Patch aligned observed-query tokens, excluding Chronos REG/future tokens."""
    from dataclasses import replace

    settings = model.module.chronos_config
    patch = settings.input_patch_size
    if settings.input_patch_stride != patch or values.shape[1] % patch:
        raise ValueError("Chronos query alignment requires disjoint complete patches")
    if values.shape[1] > settings.context_length or not tokens:
        raise ValueError("Invalid Chronos context or empty intervention span")
    if min(tokens) < 0 or max(tokens) >= values.shape[1] // patch:
        raise ValueError("Chronos interventions must stay in observed context tokens")
    predictions, states = [], {layer: [] for layer in layers}
    offset = 0
    while offset < len(values):
        width = min(model.batch_size, len(values) - offset)
        captured, handles = {}, []
        try:
            def make_hook(layer):
                def hook(module, arguments, output):
                    hidden = output.hidden_states
                    if layer in layers:
                        captured[layer] = hidden[:, tokens].detach().cpu().numpy().copy()
                    if intervention is not None and layer == intervention["layer"]:
                        supplied = torch.as_tensor(intervention["values"][offset:offset + width],
                                                   device=hidden.device, dtype=hidden.dtype)
                        if supplied.shape != hidden[:, tokens].shape:
                            raise ValueError("Chronos replacement shape mismatch")
                        hidden = hidden.clone()
                        hidden[:, tokens] = supplied
                        return replace(output, hidden_states=hidden)
                    return output
                return hook
            active = set(layers) | ({intervention["layer"]} if intervention is not None else set())
            for layer in sorted(active):
                if not 0 <= layer < len(model.module.encoder.block) - 1:
                    raise ValueError("Chronos intervention requires a downstream encoder block")
                handles.append(model.module.encoder.block[layer].register_forward_hook(make_hook(layer)))
            with torch.inference_mode(), model.fixed_statistics(True):
                tensor = torch.as_tensor(values[offset:offset + width], device="cuda", dtype=torch.float32)
                output = model.module(context=tensor,
                    group_ids=torch.arange(width, device="cuda", dtype=torch.long), num_output_patches=1)
                prediction = output.quantile_preds[:, model.median_index, :3].detach().cpu().numpy()
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite Chronos intervention prediction")
            predictions.append(prediction)
            for layer in layers:
                states[layer].append(captured[layer])
            offset += width
        except torch.cuda.OutOfMemoryError:
            if width == 1:
                raise
            model.batch_size = max(1, width // 2)
            model.recoveries.append({"offset": offset, "batch_size": model.batch_size})
            torch.cuda.empty_cache()
        finally:
            for handle in handles:
                handle.remove()
    return np.concatenate(predictions), {layer: np.concatenate(chunks) for layer, chunks in states.items()}
