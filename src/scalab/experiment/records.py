from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class EvaluationRecord:
    dataset_partition: str
    condition: str
    shift_family: str
    severity: float
    method: str
    accuracy: float
    nll: float
    brier: float
    ece: float
    mce: float
    mean_confidence: float

    def to_dict(self) -> dict:
        return asdict(self)
