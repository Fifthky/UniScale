"""Resolve configured dataset-frequency-term entries to canonical result ids."""

from __future__ import annotations

from typing import Any


PRETTY_DATASET_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


def dataset_configuration(
    name: str,
    term: str,
    properties: dict[str, Any],
) -> tuple[str, str]:
    if "/" in name:
        raw_key, frequency = name.split("/", maxsplit=1)
        key = PRETTY_DATASET_NAMES.get(raw_key.lower(), raw_key.lower())
    else:
        key = PRETTY_DATASET_NAMES.get(name.lower(), name.lower())
        if key not in properties:
            raise KeyError(f"No dataset properties found for {name}")
        frequency = properties[key]["frequency"]
    return f"{key}/{frequency}/{term}", key


def configured_dataset_ids(
    datasets: list[dict[str, Any]],
    properties: dict[str, Any],
) -> set[str]:
    return {
        dataset_configuration(str(spec["name"]), str(term), properties)[0]
        for spec in datasets
        for term in spec.get("terms", ["short"])
    }
