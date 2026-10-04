"""Exact stationary processes and conditional histories for the registered experiment."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .protocol import SIGNS, config_digest, write_arrays, write_json


PROCESSES = ("ar1", "lag8", "threshold")


def key(process: str, magnitude: float) -> str:
    return f"{process}_a{round(magnitude * 1000):03d}"


def generator(config: dict, process: str, magnitude: float, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([
        config["data_seed"], PROCESSES.index(process), round(magnitude * 1000), *keys,
    ]))


def stationary(rng: np.random.Generator, process: str, count: int, length: int,
               coefficient: float | np.ndarray) -> np.ndarray:
    coefficients = np.broadcast_to(np.asarray(coefficient), (count,))
    magnitude = np.abs(coefficients)
    sigma = np.sqrt(1 - coefficients ** 2)
    noise = rng.standard_normal((count, length))
    values = np.empty_like(noise)
    lag = 8 if process == "lag8" else 1
    if process == "threshold":
        values[:, 0] = rng.choice([-1, 1], count) * magnitude + sigma * noise[:, 0]
    else:
        values[:, :lag] = noise[:, :lag]
    for index in range(lag, length):
        source = values[:, index - lag]
        if process == "threshold":
            source = np.sign(source)
        values[:, index] = coefficients * source + sigma * noise[:, index]
    return values


def llr(values: np.ndarray, process: str, magnitude: float) -> np.ndarray:
    lag = 8 if process == "lag8" else 1
    source = values[:, :-lag]
    if process == "threshold":
        source = np.sign(source)
    return 2 * magnitude / (1 - magnitude ** 2) * np.sum(values[:, lag:] * source, axis=1)


def queries(config: dict, process: str, magnitude: float, split: int, count: int
            ) -> tuple[np.ndarray, dict]:
    rng = generator(config, process, magnitude, 10, split)
    chunks = []
    candidates = accepted = 0
    while accepted < count:
        batch = min(4096, config["maximum_query_candidates"] - candidates)
        if batch <= 0:
            raise RuntimeError("Registered conditional-query budget exhausted")
        values = stationary(rng, process, batch, 96, rng.choice([-magnitude, magnitude], batch))
        chosen = values[np.abs(llr(values, process, magnitude)) <= config["query_log_likelihood_bound"]]
        chunks.append(chosen)
        candidates += batch
        accepted += len(chosen)
    values = np.concatenate(chunks)[:count].astype(np.float32)
    paired = np.stack([values, -values], axis=1).reshape(-1, 96)
    return paired, {"independent_queries": count, "antithetic_rows": len(paired),
                    "candidates": candidates, "accepted_in_candidates": accepted,
                    "maximum_abs_llr": float(np.max(np.abs(llr(values, process, magnitude))))}


def context(config: dict, process: str, magnitude: float, query: np.ndarray,
            length: int, sign: int, repeat: int, split: int) -> np.ndarray:
    """Backward conditional sampling with nested, common-random-number prefixes."""
    if sign not in SIGNS or len(query) % 2 or length < query.shape[1]:
        raise ValueError("Invalid paired conditional context")
    prefix_length = length - query.shape[1]
    if not prefix_length:
        return query.copy()
    maximum = max(config["context_lengths"]) - 96
    if prefix_length > maximum:
        raise ValueError("Context exceeds the immutable generation plan")
    rng = generator(config, process, magnitude, 20, split, repeat)
    count = len(query) // 2
    noise = rng.standard_normal((count, maximum))
    uniforms = rng.random((count, maximum)) if process == "threshold" else None
    mixture = rng.choice([-1, 1], (count, maximum)) if process == "threshold" else None
    values = np.empty((count, prefix_length + query.shape[1]), dtype=np.float64)
    values[:, prefix_length:] = query[::2]
    sigma = np.sqrt(1 - magnitude ** 2)
    lag = 8 if process == "lag8" else 1
    for step, position in enumerate(range(prefix_length - 1, -1, -1)):
        successor = values[:, position + lag]
        if process == "threshold":
            odds = np.clip(2 * sign * magnitude * successor / sigma ** 2, -700, 700)
            previous_sign = np.where(uniforms[:, step] < 1 / (1 + np.exp(-odds)), 1, -1)
            amplitude = np.abs(magnitude * mixture[:, step] + sigma * noise[:, step])
            values[:, position] = previous_sign * amplitude
        else:
            values[:, position] = sign * magnitude * successor + sigma * noise[:, step]
    values = values.astype(np.float32)
    return np.stack([values, -values], axis=1).reshape(len(query), length)


def conditional_mean(query: np.ndarray, process: str, magnitude: float, sign: int) -> np.ndarray:
    if process == "ar1":
        return query[:, -1, None] * (sign * magnitude) ** np.arange(1, 4)[None, :]
    if process == "lag8":
        return sign * magnitude * query[:, -8:-5]
    if process == "threshold":
        c = math.erf(magnitude / np.sqrt(2 * (1 - magnitude ** 2)))
        return np.sign(query[:, -1, None]) * magnitude * sign ** np.arange(1, 4)[None, :] * c ** np.arange(3)[None, :]
    raise ValueError(f"Unknown process: {process}")


def prepare_data(config: dict, root: Path, *, state_queries: bool = False,
                 transfer_queries: bool = False) -> None:
    arrays, diagnostics = {}, {}
    sampling = config
    if state_queries:
        sampling = {**config, "data_seed": config["state_transfer"]["evaluation_seed"]}
    elif transfer_queries:
        sampling = {**config, "data_seed": config["activation_transfer"]["query_seed"]}
    for process in config["processes"]:
        for magnitude in config["magnitudes"]:
            for split, label in enumerate(("development", "test")):
                if label == "development" and magnitude != config["phi_magnitude"] and not transfer_queries:
                    continue
                name = f"{key(process, magnitude)}_{label}"
                values, diagnostic = queries(sampling, process, magnitude, split, config[f"{label}_queries"])
                arrays[f"{name}_query"] = values
                for sign in SIGNS:
                    arrays[f"{name}_oracle{sign:+d}"] = conditional_mean(values, process, magnitude, sign)
                diagnostics[name] = diagnostic
    prefix = "state_" if state_queries else "transfer_" if transfer_queries else ""
    write_arrays(root, f"data/{prefix}evidence.npz", **arrays)
    write_json(root, f"data/{prefix}generation.json", {
        "kind": "declared synthetic dynamic processes, separate from all GIFT results",
        "config_sha256": config_digest(config), "conditions": diagnostics,
        "population_mean": 0, "population_variance": 1,
        "oracle": "conditional mean; not a claim of median optimality at h2 or h3",
        "null_contrast": "h2 for ar1 and threshold only; lag8 has nonzero contrast at all endpoints",
        "uncertainty_unit": "independent query with both antithetic rows, rules, lengths, and repeats paired",
        "training_unit": "three independent trajectory and initialization seeds",
    })
    if not state_queries and not transfer_queries:
        prepare_data(config, root, state_queries=True)
        prepare_data(config, root, transfer_queries=True)
