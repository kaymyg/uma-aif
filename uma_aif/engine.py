"""Audit orchestration and response policy.

Structural changes relative to ``UMA_AIF_v1_0_0_0.audit_step``:

* **No zero-masking.** The original multiplied unmonitored layers by 0 and then
  fed them to the Gram-matrix path, where ``F.normalize`` returned zero rows,
  the trace became 0, and ``K / trace`` produced NaN. Because the guard was
  ``if trace <= 0`` and ``NaN <= 0`` is False, the NaN propagated silently
  through every downstream metric into the threat score. Layers are now
  *selected*, never zeroed.
* **No data-dependent branching on tensor values.** The original guarded with
  ``if torch.sum(monitored[l]) != 0``, a host-side branch on a device tensor.
  It forced a synchronisation on every layer and reintroduced exactly the
  timing side channel the "constant-time" scheduler existed to remove.
* **Calibrated aggregation.** See ``calibration.py``.
* **Mitigation sign corrected.** The original computed
  ``act - act @ Q @ Q.T``, i.e. it *discarded* the component lying in the
  reference-consensus subspace and returned the residual -- the exact opposite
  of its stated intent, and the opposite of how the same projection was used
  one loop earlier to define drift. ``retain_consensus`` is now the default and
  the alternative is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .calibration import CalibrationError, ScoreCalibrator
from .information import CrossLayerMutualInformation
from .perturbation import EnvelopeNoiseInjector, LayerSampler, LocalSensitivityProbe
from .subspace import GrassmannianConsensus

__all__ = [
    "AuditMetrics",
    "ResponseAction",
    "ResponsePolicy",
    "AlignmentAuditor",
    "ThreatWeights",
]


class AuditMetrics(NamedTuple):
    """Raw and calibrated outputs of one audit step."""

    raw: Dict[str, float]
    calibrated: Dict[str, float]
    per_pair_mi: Dict[Tuple[int, int], float]
    per_layer_drift: Dict[int, float]
    audited_layers: List[int]
    aggregate_threat: Optional[float]
    calibration_ready: bool


class ResponseAction(NamedTuple):
    threat_level: Optional[float]
    intercept_triggered: bool
    action_type: str
    mitigation: Dict[int, torch.Tensor]
    reason: str


@dataclass(frozen=True)
class ThreatWeights:
    """Weights over calibrated (hence comparable) subscores."""

    smuggling: float = 0.4
    sensitivity: float = 0.4
    drift: float = 0.2

    def __post_init__(self) -> None:
        total = self.smuggling + self.sensitivity + self.drift
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total}")

    def as_dict(self) -> Dict[str, float]:
        return {
            "smuggling": self.smuggling,
            "sensitivity": self.sensitivity,
            "drift": self.drift,
        }


@dataclass
class ResponsePolicy:
    """Maps a calibrated threat level to an action.

    Args:
        audit_threshold: Above this, sampling frequency should be raised.
        intercept_threshold: Above this, activations are projected.
        consecutive_required: Number of consecutive steps above
            ``intercept_threshold`` before intercepting. A single-step trigger
            on a stochastic statistic is a false-positive generator; requiring
            persistence trades detection latency for precision.
        project_mode: ``"retain_consensus"`` keeps the component of the
            activation lying in the reference subspace (the aligned part).
            ``"remove_consensus"`` keeps the orthogonal residual, which is what
            v1.0.0.0 did.
    """

    audit_threshold: float = 0.35
    intercept_threshold: float = 0.70
    consecutive_required: int = 2
    project_mode: str = "retain_consensus"
    _streak: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.audit_threshold <= self.intercept_threshold <= 1.0:
            raise ValueError(
                "require 0 <= audit_threshold <= intercept_threshold <= 1, got "
                f"{self.audit_threshold} / {self.intercept_threshold}"
            )
        if self.project_mode not in ("retain_consensus", "remove_consensus"):
            raise ValueError(f"unknown project_mode {self.project_mode!r}")

    def reset(self) -> None:
        self._streak = 0

    @torch.no_grad()
    def evaluate(
        self,
        threat: Optional[float],
        activations: Mapping[int, torch.Tensor],
        consensus: torch.Tensor,
    ) -> ResponseAction:
        if threat is None:
            self._streak = 0
            return ResponseAction(
                threat_level=None,
                intercept_triggered=False,
                action_type="CALIBRATING",
                mitigation={},
                reason="baseline statistics not yet established",
            )

        if threat < self.audit_threshold:
            self._streak = 0
            return ResponseAction(
                threat, False, "NOMINAL", {}, "threat below audit threshold"
            )

        if threat < self.intercept_threshold:
            self._streak = 0
            return ResponseAction(
                threat,
                False,
                "RAISE_AUDIT_RATE",
                {},
                "threat in the elevated band; increase sampling budget",
            )

        self._streak += 1
        if self._streak < self.consecutive_required:
            return ResponseAction(
                threat,
                False,
                "ARMED",
                {},
                f"above intercept threshold {self._streak}/"
                f"{self.consecutive_required} consecutive steps",
            )

        basis = consensus
        mitigation: Dict[int, torch.Tensor] = {}
        for idx, act in activations.items():
            b = basis.to(device=act.device, dtype=act.dtype)
            projected = (act @ b) @ b.T
            mitigation[idx] = (
                projected if self.project_mode == "retain_consensus" else act - projected
            )
        return ResponseAction(
            threat,
            True,
            "SUBSPACE_PROJECTION",
            mitigation,
            f"sustained threat; applied {self.project_mode}",
        )


class AlignmentAuditor(nn.Module):
    """Sampling-based internal-state auditor for transformer activations.

    What this measures, stated without embellishment:

    1. **Cross-layer normalised mutual information** between adjacent audited
       layers. Elevated values mean the representation is carrying more shared
       structure between those layers than the baseline did. That is a
       *correlate* of unusual information routing; it is not a proof of
       "smuggling", and it cannot attribute a cause.
    2. **Local sensitivity** to a small native-scale perturbation. Elevated
       values mean the input sits in a sharp region of the represented
       function. It is not a measurement of deception or audit-awareness.
    3. **Subspace drift** from a consensus of reference models. Elevated values
       mean the activation has more energy outside the reference span than the
       baseline did.

    All three are anomaly signals relative to an observed baseline, and all
    three require a calibration phase on known-benign traffic before any
    threshold means anything.

    Args:
        num_layers: Total layers in the audited model.
        k_components: Rank of the consensus reference subspace.
        audit_budget: Layers sampled per step. Must be >= 2 for the MI term.
        eta: Relative perturbation magnitude.
        alpha: Renyi order for the entropy estimator.
        seed: Seed for both the sampler and the noise injector.
    """

    def __init__(
        self,
        num_layers: int,
        k_components: int,
        audit_budget: int = 4,
        eta: float = 0.005,
        alpha: float = 2.0,
        trim: int = 1,
        seed: int = 0,
        weights: Optional[ThreatWeights] = None,
        calibrator: Optional[ScoreCalibrator] = None,
        policy: Optional[ResponsePolicy] = None,
    ) -> None:
        super().__init__()
        if audit_budget < 2:
            raise ValueError(
                f"audit_budget must be >= 2 so at least one layer pair exists, "
                f"got {audit_budget}"
            )
        self.num_layers = num_layers
        self.eta = float(eta)

        self.consensus = GrassmannianConsensus(k_components=k_components, trim=trim)
        self.mutual_information = CrossLayerMutualInformation(alpha=alpha)
        self.sampler = LayerSampler(num_layers, budget=audit_budget, seed=seed)
        self.injector = EnvelopeNoiseInjector(eta=eta, seed=seed)
        self.sensitivity = LocalSensitivityProbe()

        self.weights = weights or ThreatWeights()
        self.calibrator = calibrator or ScoreCalibrator()
        self.policy = policy or ResponsePolicy()

    # ------------------------------------------------------------------
    # metric computation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _raw_metrics(
        self,
        activations: Mapping[int, torch.Tensor],
        perturbed: Mapping[int, torch.Tensor],
        consensus: torch.Tensor,
    ) -> Tuple[Dict[str, float], Dict[Tuple[int, int], float], Dict[int, float]]:
        per_pair = {
            pair: float(v)
            for pair, v in self.mutual_information(dict(activations)).items()
        }

        drift: Dict[int, float] = {}
        sensitivities: List[float] = []
        dispersions: List[float] = []
        for idx, act in activations.items():
            b = consensus.to(device=act.device, dtype=act.dtype)
            residual = act - (act @ b) @ b.T
            drift[idx] = float(
                residual.norm() / act.norm().clamp_min(torch.finfo(act.dtype).tiny)
            )
            if idx in perturbed:
                sensitivities.append(
                    float(self.sensitivity(act, perturbed[idx], self.eta))
                )
                dispersions.append(
                    float(self.sensitivity.dispersion(act, perturbed[idx]))
                )

        def _mean(xs: Sequence[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        raw = {
            "smuggling": max(per_pair.values()) if per_pair else 0.0,
            "sensitivity": _mean(sensitivities),
            "drift": _mean(list(drift.values())),
            "sensitivity_dispersion": _mean(dispersions),
        }
        return raw, per_pair, drift

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def audit_step(
        self,
        layer_activations: Mapping[int, torch.Tensor],
        reference_bases: torch.Tensor,
        layer_gains: Optional[Mapping[int, torch.Tensor]] = None,
        perturbed_activations: Optional[Mapping[int, torch.Tensor]] = None,
        calibrating: bool = False,
    ) -> Tuple[AuditMetrics, ResponseAction]:
        """Run one audit cycle.

        Args:
            layer_activations: ``{layer_index: [seq_len, d_model]}`` from the
                clean forward pass. Only the sampled subset is read.
            reference_bases: ``[M, d_model, k]`` reference subspace bases.
            layer_gains: Optional ``{layer_index: [d_model]}`` norm gains, used
                to shape the perturbation across channels.
            perturbed_activations: Activations from a genuine second forward
                pass with the input perturbed. **Strongly preferred.** If
                omitted, the perturbation is applied to the recorded
                activations directly, which measures the sensitivity of nothing
                but the identity map and will read ~1.0 by construction --
                useful only as a smoke test. v1.0.0.0 only ever did the latter,
                which is why its "audit awareness" score carried no signal
                about the model at all.
            calibrating: If True, fold this step's raw metrics into the
                baseline and return no threat level.

        Returns:
            ``(metrics, action)``.
        """
        if not layer_activations:
            raise ValueError("layer_activations is empty")

        available = sorted(layer_activations)
        selected = self.sampler.sample(candidates=available)
        if len(selected) < 2:
            raise ValueError(
                f"need at least 2 audited layers, got {len(selected)} from "
                f"{len(available)} available"
            )

        clean = {i: layer_activations[i] for i in selected}
        for i, act in clean.items():
            if act.ndim != 2:
                raise ValueError(
                    f"layer {i}: expected [seq_len, d_model], got {tuple(act.shape)}"
                )
            if act.shape[0] < 2:
                raise ValueError(
                    f"layer {i}: sequence length {act.shape[0]} is too short for a "
                    f"Gram-matrix entropy estimate"
                )

        if perturbed_activations is not None:
            perturbed = {
                i: perturbed_activations[i] for i in selected if i in perturbed_activations
            }
        else:
            perturbed = {
                i: self.injector(
                    act, None if layer_gains is None else layer_gains.get(i)
                )
                for i, act in clean.items()
            }

        consensus = self.consensus(reference_bases)
        raw, per_pair, drift = self._raw_metrics(clean, perturbed, consensus)

        if calibrating:
            self.calibrator.observe(raw)
            metrics = AuditMetrics(
                raw=raw,
                calibrated={},
                per_pair_mi=per_pair,
                per_layer_drift=drift,
                audited_layers=selected,
                aggregate_threat=None,
                calibration_ready=False,
            )
            return metrics, self.policy.evaluate(None, clean, consensus)

        weight_map = self.weights.as_dict()
        try:
            calibrated = {name: self.calibrator.score(name, raw[name]) for name in weight_map}
            aggregate = sum(weight_map[n] * calibrated[n] for n in weight_map)
            ready = True
        except CalibrationError:
            calibrated, aggregate, ready = {}, None, False

        metrics = AuditMetrics(
            raw=raw,
            calibrated=calibrated,
            per_pair_mi=per_pair,
            per_layer_drift=drift,
            audited_layers=selected,
            aggregate_threat=aggregate,
            calibration_ready=ready,
        )
        return metrics, self.policy.evaluate(aggregate, clean, consensus)

    def calibrate(
        self,
        traffic: Sequence[Mapping[int, torch.Tensor]],
        reference_bases: torch.Tensor,
        layer_gains: Optional[Mapping[int, torch.Tensor]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Fold a sequence of known-benign activation snapshots into the baseline."""
        for snapshot in traffic:
            self.audit_step(
                snapshot,
                reference_bases,
                layer_gains=layer_gains,
                calibrating=True,
            )
        return self.calibrator.summary()

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.num_layers}, budget={self.sampler.budget}, "
            f"eta={self.eta}"
        )
