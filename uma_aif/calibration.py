"""Baseline calibration for raw detector scores.

This module exists because of the single most serious defect in v1.0.0.0: the
aggregate threat level was

    0.4 * smuggling + 0.4 * awareness + 0.2 * drift,   clamped to [0, 1]

where ``smuggling`` was a Renyi mutual information bounded above by ``log2 N``
(6 bits at a sequence length of 64), ``awareness`` was an unbounded coefficient
of variation, and ``drift`` was a ratio in ``[0, 1]``. The weights sum to 1,
which looks like a convex combination, but the inputs live on incomparable
scales. In practice the smuggling term alone exceeds 2.5 on benign input, the
clamp pins the aggregate at 1.0, and the 0.35 / 0.70 thresholds fire on
everything. A detector that always fires has no true-positive rate worth
quoting.

The fix is not better weights. It is calibration: observe each raw statistic on
known-benign traffic, then report how far the live value sits from that
baseline in robust standard deviations, and squash that to ``[0, 1]``. The
weights then combine quantities that actually mean the same thing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

__all__ = ["RunningStat", "ScoreCalibrator", "CalibrationError"]


class CalibrationError(RuntimeError):
    """Raised when a calibrated score is requested before a baseline exists."""


@dataclass
class RunningStat:
    """Welford online mean/variance with a sample counter."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def zscore(self, value: float, floor: float) -> float:
        return (value - self.mean) / max(self.std, floor)


@dataclass
class ScoreCalibrator:
    """Turns raw detector statistics into comparable ``[0, 1]`` scores.

    Usage is two-phase. During ``observe`` (the baseline phase) the calibrator
    accumulates statistics and returns nothing actionable. After ``freeze`` it
    scores live values against that baseline.

    Args:
        min_samples: Baseline observations required before ``score`` will
            produce a value. Below this, ``score`` raises.
        std_floor: Lower bound on the baseline standard deviation, so a metric
            that happened to be constant during calibration does not produce
            infinite z-scores.
        std_floors: Per-metric absolute lower bounds on the baseline standard
            deviation. These matter more than they look: several UMA-AIF
            statistics have a baseline sigma around 5e-3, so a benign
            distribution shift of one percent registers as a two-sigma event.
            Set a floor reflecting the smallest change you consider meaningful
            for that metric, or the detector will be brittle to drift in the
            traffic rather than sensitive to anomalies in the model.
        steepness: Logistic gain.
        midpoint: Absolute z-score that maps to 0.5.
        one_sided: If True, only deviations *above* the baseline raise the
            score. Defaults to False, because an anomaly can move a statistic
            in either direction -- empirically, collapsing a residual stream
            onto a low-rank subspace *lowers* cross-layer mutual information,
            and a one-sided test discards that signal entirely.
    """

    min_samples: int = 32
    std_floor: float = 1e-6
    std_floors: Dict[str, float] = field(default_factory=dict)
    steepness: float = 1.0
    midpoint: float = 3.0
    one_sided: bool = False
    stats: Dict[str, RunningStat] = field(default_factory=dict)
    frozen: bool = False

    def observe(self, values: Dict[str, float]) -> None:
        """Fold one baseline observation into the running statistics."""
        if self.frozen:
            raise CalibrationError("calibrator is frozen; call thaw() to resume")
        for name, value in values.items():
            self.stats.setdefault(name, RunningStat()).update(float(value))

    def observe_many(self, batches: Iterable[Dict[str, float]]) -> None:
        for batch in batches:
            self.observe(batch)

    def freeze(self) -> None:
        """Lock the baseline in. Raises if any metric is under-sampled."""
        thin = {
            name: st.count
            for name, st in self.stats.items()
            if st.count < self.min_samples
        }
        if thin:
            raise CalibrationError(
                f"under-sampled metrics (need {self.min_samples}): {thin}"
            )
        self.frozen = True

    def thaw(self) -> None:
        self.frozen = False

    def is_ready(self, name: str) -> bool:
        st = self.stats.get(name)
        return st is not None and st.count >= self.min_samples

    def zscore(self, name: str, value: float) -> float:
        st = self.stats.get(name)
        if st is None or st.count < self.min_samples:
            raise CalibrationError(
                f"metric {name!r} has {0 if st is None else st.count} baseline "
                f"samples, need {self.min_samples}"
            )
        floor = self.std_floors.get(name, self.std_floor)
        return st.zscore(float(value), floor)

    def score(self, name: str, value: float) -> float:
        """Map a raw value to ``[0, 1]`` against its baseline."""
        z = self.zscore(name, value)
        z = max(z, 0.0) if self.one_sided else abs(z)
        # Guard against overflow for the very large |z| that a low-variance
        # baseline can produce.
        exponent = -self.steepness * (z - self.midpoint)
        if exponent < -700:
            return 1.0
        if exponent > 700:
            return 0.0
        return 1.0 / (1.0 + math.exp(exponent))

    def score_all(self, values: Dict[str, float]) -> Dict[str, float]:
        return {name: self.score(name, v) for name, v in values.items()}

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {"count": st.count, "mean": st.mean, "std": st.std}
            for name, st in sorted(self.stats.items())
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "min_samples": self.min_samples,
            "std_floor": self.std_floor,
            "std_floors": dict(self.std_floors),
            "steepness": self.steepness,
            "midpoint": self.midpoint,
            "one_sided": self.one_sided,
            "frozen": self.frozen,
            "stats": {
                n: {"count": s.count, "mean": s.mean, "m2": s.m2}
                for n, s in self.stats.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "ScoreCalibrator":
        cal = cls(
            min_samples=int(state["min_samples"]),
            std_floor=float(state["std_floor"]),
            steepness=float(state["steepness"]),
            midpoint=float(state["midpoint"]),
            one_sided=bool(state["one_sided"]),
            std_floors=dict(state.get("std_floors", {})),  # type: ignore[arg-type]
        )
        cal.stats = {
            n: RunningStat(count=int(d["count"]), mean=float(d["mean"]), m2=float(d["m2"]))
            for n, d in state["stats"].items()  # type: ignore[union-attr]
        }
        cal.frozen = bool(state["frozen"])
        return cal
