"""Additive score values shared by the independent workflow scorers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Precision:
    correct: int = 0
    total: int = 0

    def __add__(self, other: Precision) -> Precision:
        return Precision(self.correct + other.correct, self.total + other.total)

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "correct": self.correct,
            "total": self.total,
            "precision": self.correct / self.total if self.total else None,
        }


@dataclass(frozen=True)
class FieldScore:
    correct: int = 0
    predicted: int = 0
    gold: int = 0

    def __add__(self, other: FieldScore) -> FieldScore:
        return FieldScore(
            self.correct + other.correct, self.predicted + other.predicted, self.gold + other.gold
        )

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "correct": self.correct,
            "predicted": self.predicted,
            "gold": self.gold,
            "precision": self.correct / self.predicted if self.predicted else None,
            "recall": self.correct / self.gold if self.gold else None,
        }


@dataclass(frozen=True)
class StageScore:
    stage: str
    metrics: dict[str, Precision]

    def __add__(self, other: StageScore) -> StageScore:
        if self.stage != other.stage or self.metrics.keys() != other.metrics.keys():
            raise ValueError("Cannot combine different stage scores")
        return StageScore(
            self.stage, {key: value + other.metrics[key] for key, value in self.metrics.items()}
        )

    def as_dict(self) -> dict[str, object]:
        return {"stage": self.stage, "metrics": {key: value.as_dict() for key, value in self.metrics.items()}}
