"""Read-only frozen checkpoints and mechanism-only normalization interventions."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .protocol import REPOSITORY, digest_file


class TimesFM:
    """Use the vendored forward graph without changing any model parameters."""

    def __init__(self, checkpoint: Path, batch_size: int):
        from UniScale.vendor.timesfm25.timesfm_2p5.timesfm_2p5_torch import (
            TimesFM_2p5_200M_torch_module,
        )

        self.module = TimesFM_2p5_200M_torch_module()
        self.module.load_checkpoint(str(checkpoint / "model.safetensors"), torch_compile=False)
        self.module.requires_grad_(False)
        self.batch_size = batch_size
        self.recoveries = []
        self.metadata = {
            "name": "timesfm25", "checkpoint": checkpoint.name,
            "weights_sha256": digest_file(checkpoint / "model.safetensors"),
            "point_output": "native median head, channel 5",
            "native_block": self.module.o,
            "loaded_parameters_m": sum(parameter.numel() for parameter in self.module.parameters()) / 1e6,
            "statistics_bridge": "native running-patch statistics vs fixed population (0,1)",
            "inference": "first native output block; no rollout, no outer normalization or flip ensemble",
            "cache": "no state carried across examples or batches",
        }

    def infer(self, values: np.ndarray, normalization: str = "fixed_population"
              ) -> tuple[np.ndarray, dict]:
        if values.ndim != 2 or values.shape[1] % self.module.p:
            raise ValueError("TimesFM mechanism inputs must be dense patch-aligned batches")
        if normalization not in {"fixed_population", "native_statistics"}:
            raise ValueError("Unknown normalization intervention")
        forecasts = []
        offset = 0
        while offset < len(values):
            width = min(self.batch_size, len(values) - offset)
            try:
                tensor = torch.as_tensor(values[offset:offset + width], device="cuda", dtype=torch.float32)
                with torch.inference_mode():
                    if normalization == "fixed_population":
                        patches = tensor.reshape(width, -1, self.module.p)
                        mask = torch.zeros_like(patches, dtype=torch.bool)
                        (_, _, output, _), _ = self.module(patches, mask)
                        prediction = output[:, -1].reshape(width, self.module.o, self.module.q)[:, :3, 5]
                    else:
                        output, _, _ = self.module.decode(3, tensor, torch.zeros_like(tensor, dtype=torch.bool))
                        prediction = output[:, -1, :3, 5]
                result = prediction.detach().cpu().numpy()
                if not np.isfinite(result).all():
                    raise RuntimeError("Nonfinite frozen TimesFM prediction")
                forecasts.append(result)
                offset += width
            except torch.cuda.OutOfMemoryError:
                if width == 1:
                    raise
                self.batch_size = max(1, width // 2)
                self.recoveries.append({"offset": offset, "new_batch_size": self.batch_size})
                print(f"TimesFM OOM: reducing batch to {self.batch_size}", flush=True)
                torch.cuda.empty_cache()
        return np.concatenate(forecasts), {}



class Chronos2:
    """Use separate group IDs so batching cannot communicate between examples."""

    def __init__(self, checkpoint: Path, batch_size: int):
        from chronos import BaseChronosPipeline, Chronos2Pipeline

        pipeline = BaseChronosPipeline.from_pretrained(
            str(checkpoint), device_map="cuda", torch_dtype=torch.float32, local_files_only=True,
        )
        if not isinstance(pipeline, Chronos2Pipeline):
            raise TypeError("The checkpoint did not load as Chronos 2")
        self.module = pipeline.model.eval().requires_grad_(False)
        self.batch_size = batch_size
        self.recoveries = []
        self.median_index = list(self.module.chronos_config.quantiles).index(0.5)
        self.metadata = {
            "name": "chronos2", "checkpoint": checkpoint.name,
            "weights_sha256": {path.name: digest_file(path) for path in checkpoint.glob("*.safetensors")},
            "point_output": "native median quantile", "grouping": "unique group ID per example",
            "native_block": self.module.chronos_config.output_patch_size,
            "statistics_bridge": "native instance statistics vs fixed population (0,1); arcsinh retained",
            "inference": "one output patch; no rollout or cross-example context",
            "cache": "no state carried across examples or batches",
        }

    @contextmanager
    def fixed_statistics(self, enabled: bool) -> Iterator[None]:
        # The hook changes only the arguments of this private in-memory instance.
        handle = None
        if enabled:
            def hook(module, arguments):
                values = arguments[0]
                zeros = torch.zeros_like(values[:, :1], dtype=torch.float32)
                return values, (zeros, torch.ones_like(zeros))
            handle = self.module.instance_norm.register_forward_pre_hook(hook)
        try:
            yield
        finally:
            if handle is not None:
                handle.remove()

    def infer(self, values: np.ndarray, normalization: str = "fixed_population"
              ) -> tuple[np.ndarray, dict]:
        if values.shape[1] > self.module.chronos_config.context_length:
            raise ValueError("Refusing silent context truncation in the mechanism experiment")
        if normalization not in {"fixed_population", "native_statistics"}:
            raise ValueError("Unknown normalization intervention")
        predictions = []
        offset = 0
        while offset < len(values):
            width = min(self.batch_size, len(values) - offset)
            try:
                tensor = torch.as_tensor(values[offset:offset + width], device="cuda", dtype=torch.float32)
                with torch.inference_mode(), self.fixed_statistics(normalization == "fixed_population"):
                    output = self.module(
                        context=tensor, group_ids=torch.arange(width, device="cuda", dtype=torch.long),
                        num_output_patches=1,
                    ).quantile_preds[:, self.median_index, :3]
                result = output.detach().cpu().numpy()
                if not np.isfinite(result).all():
                    raise RuntimeError("Nonfinite frozen Chronos 2 prediction")
                predictions.append(result)
                offset += width
            except torch.cuda.OutOfMemoryError:
                if width == 1:
                    raise
                self.batch_size = max(1, width // 2)
                self.recoveries.append({"offset": offset, "new_batch_size": self.batch_size})
                print(f"Chronos 2 OOM: reducing batch to {self.batch_size}", flush=True)
                torch.cuda.empty_cache()
        return np.concatenate(predictions), {}


def load_model(config: dict, name: str, model_root: Path):
    record = next(item for item in config["models"] if item["name"] == name)
    checkpoint = (model_root / record["checkpoint_subdir"]).resolve(strict=True)
    if model_root.resolve() not in checkpoint.parents:
        raise ValueError("Checkpoint resolves outside the approved model directory")
    registry = json.loads((REPOSITORY / "UniScale/configs/learning_mechanism_models.json").read_text())[name]
    if record["checkpoint_subdir"] != registry["checkpoint_subdir"]:
        raise ValueError("Checkpoint directory differs from the registered mechanism model")
    for filename, expected in registry["files"].items():
        if digest_file(checkpoint / filename) != expected:
            raise RuntimeError(f"Registered checkpoint checksum mismatch: {name}/{filename}")
    constructors = {"timesfm25": TimesFM, "chronos2": Chronos2}
    return constructors[name](checkpoint, config["inference_batch_size"])
