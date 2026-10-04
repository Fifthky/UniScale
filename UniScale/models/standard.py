"""Strict notebook-aligned model profiles with an H/L-only forecast interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from UniScale.experiments.pipelines import ContextLimitedPredictor
from UniScale.experiments.predictors import (
    PredictorSettings,
    create_predictor,
    load_backend,
)
from UniScale.model_registry import (
    DEFAULT_CATALOGUE,
    configured_model_root,
    load_catalogue,
    model_index,
    resolve_checkpoint,
)


DEFAULT_BATCH_SIZE = 256
MAX_BATCH_SIZE = 256


@dataclass(frozen=True)
class ExecutionOptions:
    """Infrastructure settings kept outside the scientific H/L intervention."""

    device: str = "cuda"
    torch_dtype: str = "float32"
    model_root: Path | None = None
    catalogue: Path = DEFAULT_CATALOGUE


@dataclass(frozen=True)
class PreparedForecast:
    """A notebook-aligned predictor resolved for one dataset and one H/L cell."""

    predictor: Any
    base_predictor: Any
    settings: PredictorSettings
    H: int
    L: int
    H_source: str
    L_source: str
    model_call_horizon: int
    inference: dict[str, Any]

    @property
    def effective_batch_size(self) -> int:
        return int(
            getattr(self.base_predictor, "effective_batch_size", self.settings.batch_size)
        )

    @property
    def effective_samples_per_batch(self) -> int:
        return int(
            getattr(
                self.base_predictor,
                "effective_samples_per_batch",
                self.settings.num_samples,
            )
        )

    @property
    def batch_recovery_events(self) -> list[dict[str, Any]]:
        return list(getattr(self.base_predictor, "recovery_events", []))


def resolve_standard_inference(model: dict[str, Any]) -> dict[str, Any]:
    """Return frozen inference settings transcribed from the fixed upstream notebook."""
    notebook = model.get("notebook_inference", {})

    def optional_int(name: str) -> int | None:
        value = notebook.get(name)
        return None if value is None else int(value)

    deterministic = notebook.get("deterministic_algorithms")
    configured_batch_size = int(notebook.get("batch_size", DEFAULT_BATCH_SIZE))
    if configured_batch_size <= 0:
        raise ValueError("notebook batch_size must be positive")
    return {
        "batch_size": min(configured_batch_size, MAX_BATCH_SIZE),
        "per_core_batch_size": optional_int("per_core_batch_size"),
        "batch_size_policy": str(notebook.get("batch_size_policy", "fixed")),
        "batch_size_cap": optional_int("batch_size_cap"),
        "num_samples": int(notebook.get("num_samples", 100)),
        "seed": optional_int("seed"),
        "deterministic_algorithms": (
            None if deterministic is None else bool(deterministic)
        ),
        "float32_matmul_precision": notebook.get("float32_matmul_precision"),
    }


def _resolve_context_length(
    requested: int,
    model: dict[str, Any],
    dataset: Any,
    minimum_context: int,
) -> int:
    if requested <= 0:
        raise ValueError("L must be positive")
    maximum = model["context"].get("maximum")
    if maximum is not None and requested > int(maximum):
        raise ValueError(
            f"Configured L={requested} exceeds {model['id']} maximum {maximum}"
        )
    policy = model.get("notebook_inference", {}).get("context_policy")
    if policy != "dataset_max_without_padding":
        return max(requested, minimum_context)
    available = dataset._min_series_length - (
        dataset.windows + 1
    ) * dataset.prediction_length
    if available <= 0:
        raise ValueError(f"Dataset has no positive no-padding context: {available}")
    return max(min(requested, int(available)), minimum_context)




class StandardModel:
    """One frozen checkpoint with notebook defaults and controllable H and L only."""

    def __init__(
        self,
        record: dict[str, Any],
        checkpoint: str,
        execution: ExecutionOptions,
    ):
        adapter = record.get("adapter")
        if not adapter:
            raise ValueError(f"Model {record['id']} has no verified experiment adapter")
        self.record = record
        self.checkpoint = checkpoint
        self.adapter = str(adapter)
        self.execution = execution
        self.inference = resolve_standard_inference(record)
        self._backend: Any | None = None
        self._minimum_context = 1

    def _base_settings(
        self,
        H: int,
        L: int,
        dataset: Any,
        frequency: str,
        domain: str | None,
        has_daily_cycle: bool,
    ) -> PredictorSettings:
        return PredictorSettings(
            checkpoint=self.checkpoint,
            prediction_length=H,
            context_length=L,
            batch_size=self.inference["batch_size"],
            per_core_batch_size=self.inference["per_core_batch_size"],
            device=self.execution.device,
            torch_dtype=self.execution.torch_dtype,
            num_samples=self.inference["num_samples"],
            past_feat_dynamic_real_dim=dataset.past_feat_dynamic_real_dim,
            frequency=frequency,
            domain=domain,
            has_daily_cycle=has_daily_cycle,
            target_dim=dataset.target_dim,
            seed=self.inference["seed"],
            deterministic_algorithms=self.inference["deterministic_algorithms"],
            float32_matmul_precision=self.inference["float32_matmul_precision"],
        )

    def _ensure_backend(self, settings: PredictorSettings) -> Any:
        if self._backend is None:
            self._backend = load_backend(self.adapter, settings)
        return self._backend

    def context_without_backend(self, L: int | None, dataset: Any = None) -> int | None:
        """Resolve a skip key when context policy does not require model loading."""
        requested = self.record["context"].get("gift") if L is None else L
        if requested is None:
            raise ValueError(f"Model {self.record['id']} has no notebook-aligned default L")
        policy = self.record.get("notebook_inference", {}).get("context_policy")
        if dataset is None and policy == "dataset_max_without_padding":
            return None
        return _resolve_context_length(
            int(requested), self.record, dataset, self._minimum_context
        )

    def prepare(
        self,
        dataset: Any,
        frequency: str,
        H: int | None = None,
        L: int | None = None,
        domain: str | None = None,
        has_daily_cycle: bool = True,
    ) -> PreparedForecast:
        """Create a frozen model call with the selected context limit."""
        scored_horizon = dataset.prediction_length if H is None else int(H)
        if scored_horizon != dataset.prediction_length:
            raise ValueError(
                "H must be applied when constructing ControlledDataset so its labels "
                "and declared origin protocol match the scored horizon"
            )
        if scored_horizon <= 0:
            raise ValueError("H must be positive")

        default_context = self.record["context"].get("gift")
        if L is None and default_context is None:
            raise ValueError(f"Model {self.record['id']} has no notebook-aligned default L")
        requested_context = int(default_context if L is None else L)
        initial_settings = self._base_settings(
            scored_horizon,
            requested_context,
            dataset,
            frequency,
            domain,
            has_daily_cycle,
        )
        backend = self._ensure_backend(initial_settings)
        context_length = _resolve_context_length(
            requested_context,
            self.record,
            dataset,
            minimum_context=self._minimum_context,
        )
        if self.inference["batch_size_policy"] != "fixed":
            raise ValueError("Formal checkpoint inference requires a fixed batch profile")
        batch_size = int(self.inference["batch_size"])
        settings = PredictorSettings(
            **{
                **initial_settings.__dict__,
                "context_length": context_length,
                "batch_size": batch_size,
            }
        )
        base_predictor = create_predictor(self.adapter, settings, backend=backend)
        predictor = ContextLimitedPredictor(base_predictor, context_length)
        return PreparedForecast(
            predictor=predictor,
            base_predictor=base_predictor,
            settings=settings,
            H=scored_horizon,
            L=context_length,
            H_source="gift_default" if H is None else "controlled",
            L_source="notebook_default" if L is None else "controlled",
            model_call_horizon=scored_horizon,
            inference=dict(self.inference),
        )


def load_standard_model(
    model_id: str,
    execution: ExecutionOptions | None = None,
) -> StandardModel:
    """Load one registered checkpoint behind the uniform standard-model facade."""
    resolved_execution = execution or ExecutionOptions()
    catalogue = load_catalogue(resolved_execution.catalogue)
    models = model_index(catalogue)
    if model_id not in models:
        raise KeyError(f"Unknown model_id: {model_id}")
    record = models[model_id]
    checkpoint = resolve_checkpoint(
        record=record,
        model_root=configured_model_root(resolved_execution.model_root),
        explicit_checkpoint=None,
        allow_remote=False,
    )
    return StandardModel(record, checkpoint, resolved_execution)
