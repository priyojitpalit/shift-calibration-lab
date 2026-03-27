from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SplitConfig:
    train: float
    validation: float
    calibration: float
    test: float


@dataclass(slots=True)
class DatasetConfig:
    name: str
    normalize: bool
    split: SplitConfig


@dataclass(slots=True)
class ModelConfig:
    name: str
    hidden_layer_sizes: list[int] = field(default_factory=list)
    alpha: float = 0.0001
    max_iter: int = 300
    early_stopping: bool = True


@dataclass(slots=True)
class ScalarTemperatureConfig:
    enabled: bool
    search_bounds: tuple[float, float]
    grid_points: int = 200


@dataclass(slots=True)
class AdaptiveTemperatureConfig:
    enabled: bool
    learning_rate: float = 0.05
    max_iter: int = 500
    l2: float = 0.001
    include_severity: bool = False


@dataclass(slots=True)
class CalibrationConfig:
    scalar_temperature: ScalarTemperatureConfig
    adaptive_temperature: AdaptiveTemperatureConfig


@dataclass(slots=True)
class ShiftFamilyConfig:
    name: str
    severities: list[float]


@dataclass(slots=True)
class ShiftConfig:
    families: list[ShiftFamilyConfig]


@dataclass(slots=True)
class ReportingConfig:
    reliability_bins: int = 15
    save_probabilities: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    output_dir: str
    seed: int


@dataclass(slots=True)
class AppConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    calibration: CalibrationConfig
    shift: ShiftConfig
    reporting: ReportingConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: Path) -> AppConfig:
    raw = _load_yaml(path)
    return AppConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        dataset=DatasetConfig(
            name=raw["dataset"]["name"],
            normalize=raw["dataset"].get("normalize", True),
            split=SplitConfig(**raw["dataset"]["split"]),
        ),
        model=ModelConfig(**raw["model"]),
        calibration=CalibrationConfig(
            scalar_temperature=ScalarTemperatureConfig(
                enabled=raw["calibration"]["scalar_temperature"]["enabled"],
                search_bounds=tuple(raw["calibration"]["scalar_temperature"]["search_bounds"]),
                grid_points=raw["calibration"]["scalar_temperature"].get("grid_points", 200),
            ),
            adaptive_temperature=AdaptiveTemperatureConfig(**raw["calibration"]["adaptive_temperature"]),
        ),
        shift=ShiftConfig(
            families=[ShiftFamilyConfig(**family) for family in raw.get("shift", {}).get("families", [])]
        ),
        reporting=ReportingConfig(**raw.get("reporting", {})),
    )
