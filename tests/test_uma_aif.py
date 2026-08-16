"""Test suite for UMA-AIF.

Split into three groups:

* ``TestUnits`` -- properties each component must satisfy in isolation.
* ``TestRegressions`` -- one test per defect found in v1.0.0.0, each asserting
  the corrected behaviour. Named after the defect.
* ``TestEndToEnd`` -- does the calibrated pipeline actually separate benign
  activations from perturbed ones, or does it just fire on everything?
"""

from __future__ import annotations

import math

import pytest
import torch

from uma_aif import (
    AlignmentAuditor,
    CalibrationError,
    EnvelopeNoiseInjector,
    GrassmannianConsensus,
    LayerSampler,
    LocalSensitivityProbe,
    MatrixRenyiEntropy,
    ResponsePolicy,
    ScoreCalibrator,
    ThreatWeights,
    chordal_distance_matrix,
    gram_matrix,
)
from uma_aif.information import CrossLayerMutualInformation

D_MODEL = 64
SEQ = 32
LAYERS = 12
K = 8


def make_bases(
    m: int = 5, d: int = D_MODEL, k: int = K, seed: int = 0, spread: float = 0.05
) -> torch.Tensor:
    """Reference bases clustered around a shared subspace.

    Consensus is only a meaningful operation when the references broadly agree.
    ``spread`` controls how far each reference wanders from the shared span;
    at ``spread=1.0`` they are mutually independent and there is no consensus
    to compute.
    """
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(d, k, generator=g)
    return shared.unsqueeze(0) + spread * torch.randn(m, d, k, generator=g)


def residual_stream(
    n_layers: int = LAYERS,
    seq: int = SEQ,
    d: int = D_MODEL,
    seed: int = 0,
    drift: float = 0.05,
) -> dict[int, torch.Tensor]:
    """A crude but structurally honest stand-in for a residual stream.

    Each layer is the previous layer plus a small update, so adjacent layers are
    highly correlated -- exactly the regime where a naive MI detector saturates.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(seq, d, generator=g)
    out = {}
    for i in range(n_layers):
        x = x + drift * torch.randn(seq, d, generator=g)
        out[i] = x.clone()
    return out


# ======================================================================
# Units
# ======================================================================
class TestUnits:
    def test_gram_matrix_is_a_density_matrix(self):
        k = gram_matrix(torch.randn(SEQ, D_MODEL))
        assert k.shape == (SEQ, SEQ)
        assert math.isclose(float(k.diagonal().sum()), 1.0, rel_tol=1e-5)
        assert torch.allclose(k, k.T, atol=1e-6)
        assert float(torch.linalg.eigvalsh(k).min()) > -1e-5

    def test_gram_matrix_rejects_zero_rows(self):
        x = torch.randn(SEQ, D_MODEL)
        x[3] = 0.0
        with pytest.raises(ValueError, match="zero-norm"):
            gram_matrix(x)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0, 4.0])
    def test_entropy_is_bounded_and_finite(self, alpha):
        h = MatrixRenyiEntropy(alpha=alpha)(gram_matrix(torch.randn(SEQ, D_MODEL)))
        assert torch.isfinite(h)
        assert 0.0 <= float(h) <= math.log2(SEQ) + 1e-4

    def test_entropy_of_maximally_mixed_state_is_log2_n(self):
        k = torch.eye(SEQ) / SEQ
        for alpha in (0.5, 1.0, 2.0, 3.0):
            h = MatrixRenyiEntropy(alpha=alpha)(k)
            assert math.isclose(float(h), math.log2(SEQ), abs_tol=1e-3)

    def test_entropy_of_pure_state_is_zero(self):
        v = torch.randn(SEQ, 1)
        v = v / v.norm()
        h = MatrixRenyiEntropy(alpha=2.0)(v @ v.T)
        assert float(h) < 1e-3

    def test_consensus_basis_is_orthonormal(self):
        q = GrassmannianConsensus(k_components=K)(make_bases())
        assert q.shape == (D_MODEL, K)
        assert torch.allclose(q.T @ q, torch.eye(K), atol=1e-4)

    def test_chordal_distance_is_zero_for_identical_subspaces(self):
        b = make_bases(m=3)
        q, _ = torch.linalg.qr(torch.cat([b[:1], b[:1], b[:1]], dim=0))
        d = chordal_distance_matrix(q)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-4)

    def test_sampler_draws_a_constant_number_of_distinct_layers(self):
        s = LayerSampler(LAYERS, budget=4, seed=7)
        draws = [s.sample() for _ in range(50)]
        assert all(len(set(d)) == 4 for d in draws)
        assert all(0 <= i < LAYERS for d in draws for i in d)

    def test_sampler_is_reproducible_from_its_seed(self):
        a = LayerSampler(LAYERS, budget=4, seed=11)
        b = LayerSampler(LAYERS, budget=4, seed=11)
        assert [a.sample() for _ in range(20)] == [b.sample() for _ in range(20)]

    def test_injector_produces_the_requested_relative_magnitude(self):
        x = torch.randn(SEQ, D_MODEL) * 17.0
        y = EnvelopeNoiseInjector(eta=0.01, seed=3)(x)
        rel = ((y - x).norm(dim=-1) / x.norm(dim=-1))
        assert torch.allclose(rel, torch.full_like(rel, 0.01), atol=1e-5)

    def test_injector_gain_shaping_does_not_change_magnitude(self):
        x = torch.randn(SEQ, D_MODEL)
        gain = torch.rand(D_MODEL) * 5.0
        y = EnvelopeNoiseInjector(eta=0.02, seed=3)(x, gain)
        rel = ((y - x).norm(dim=-1) / x.norm(dim=-1))
        assert torch.allclose(rel, torch.full_like(rel, 0.02), atol=1e-5)

    def test_calibrator_refuses_to_score_before_baseline(self):
        cal = ScoreCalibrator(min_samples=10)
        cal.observe({"m": 1.0})
        with pytest.raises(CalibrationError):
            cal.score("m", 1.0)

    def test_calibrator_scores_baseline_low_and_outlier_high(self):
        cal = ScoreCalibrator(min_samples=20)
        g = torch.Generator().manual_seed(0)
        for v in torch.randn(200, generator=g).tolist():
            cal.observe({"m": v})
        assert cal.score("m", 0.0) < 0.2
        assert cal.score("m", 8.0) > 0.9

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            ThreatWeights(0.5, 0.5, 0.5)


# ======================================================================
# Regressions against v1.0.0.0
# ======================================================================
class TestRegressions:
    def test_alpha_one_does_not_divide_by_zero(self):
        """v1: ``1/(1-alpha)`` was evaluated unguarded at alpha=1."""
        h = MatrixRenyiEntropy(alpha=1.0)(gram_matrix(torch.randn(SEQ, D_MODEL)))
        assert torch.isfinite(h)

    def test_unmonitored_layers_do_not_produce_nan(self):
        """v1: masked layers were multiplied by 0, then normalised and divided
        by a zero trace, producing NaN that passed the ``trace <= 0`` guard."""
        auditor = AlignmentAuditor(LAYERS, K, audit_budget=4, seed=0)
        acts = residual_stream()
        bases = make_bases()
        metrics, _ = auditor.audit_step(acts, bases, calibrating=True)
        assert all(math.isfinite(v) for v in metrics.raw.values()), metrics.raw

    def test_consensus_rejects_a_poisoned_reference(self):
        """v1: eigenvalue clamping cannot move eigenvectors, so the stated
        Byzantine resilience was a no-op."""
        clean = make_bases(m=5, seed=1)
        poisoned = clean.clone()
        g = torch.Generator().manual_seed(99)
        poisoned[0] = torch.randn(D_MODEL, K, generator=g) * 50.0

        model = GrassmannianConsensus(k_components=K, trim=1)
        q_clean = model(clean)
        q_poisoned = model(poisoned)

        # Distance between the two consensus subspaces, in squared chordal terms.
        overlap = float((q_clean.T @ q_poisoned).pow(2).sum())
        assert overlap > K - 0.5, f"poisoning moved the consensus (overlap {overlap})"

    def test_mutual_information_is_bounded_in_unit_interval(self):
        """v1: raw MI up to log2(N) bits was summed with weight 0.4 and then
        clamped, pinning the aggregate threat at 1.0 for any realistic input."""
        mi = CrossLayerMutualInformation(alpha=2.0)
        acts = residual_stream(n_layers=4)
        for pair, v in mi(acts).items():
            assert 0.0 <= float(v) <= 1.0, (pair, float(v))

    def test_sensitivity_is_stable_for_tiny_perturbations(self):
        """v1: ``1 - cos`` in fp32 at cos ~ 1-1e-5, then std/mean, is noise."""
        probe = LocalSensitivityProbe()
        x = torch.randn(SEQ, D_MODEL)
        for eta in (1e-2, 1e-3, 1e-4, 1e-5):
            y = EnvelopeNoiseInjector(eta=eta, seed=1)(x)
            ratio = float(probe(x, y, eta))
            assert math.isfinite(ratio)
            # A pure additive perturbation of an identity map: the angular
            # displacement of the unit vector is at most eta, and is eta times
            # the sine of the angle to the radial direction.
            assert 0.5 < ratio < 1.5, (eta, ratio)

    def test_mitigation_retains_the_consensus_component(self):
        """v1: returned ``act - proj``, i.e. it kept the deviation and threw
        away the part that agreed with the reference consensus."""
        policy = ResponsePolicy(
            audit_threshold=0.05, intercept_threshold=0.1, consecutive_required=1
        )
        basis = GrassmannianConsensus(k_components=K)(make_bases())
        act = torch.randn(SEQ, D_MODEL)
        action = policy.evaluate(0.99, {0: act}, basis)
        assert action.intercept_triggered
        out = action.mitigation[0]
        # The output must lie inside the consensus span.
        residual = out - (out @ basis) @ basis.T
        assert float(residual.norm() / out.norm()) < 1e-4

    def test_no_host_side_branch_on_activation_sums(self):
        """v1 branched on ``torch.sum(act) != 0`` per layer. Assert we can run
        a step where a layer sums to exactly zero without changing behaviour."""
        auditor = AlignmentAuditor(LAYERS, K, audit_budget=4, seed=0)
        acts = residual_stream()
        # Make one layer mean-zero so its sum is ~0 but it is not degenerate.
        acts[5] = acts[5] - acts[5].mean()
        metrics, _ = auditor.audit_step(acts, make_bases(), calibrating=True)
        assert all(math.isfinite(v) for v in metrics.raw.values())

    def test_degenerate_input_raises_instead_of_returning_a_score(self):
        """v1 would silently return NaN. Failing loudly is the correct
        behaviour for a security control."""
        auditor = AlignmentAuditor(LAYERS, K, audit_budget=4, seed=0)
        acts = residual_stream()
        for i in acts:
            acts[i] = torch.zeros_like(acts[i])
        with pytest.raises(ValueError):
            auditor.audit_step(acts, make_bases(), calibrating=True)


# ======================================================================
# End to end
# ======================================================================
class TestEndToEnd:
    def _calibrated_auditor(self, n: int = 60):
        auditor = AlignmentAuditor(
            LAYERS,
            K,
            audit_budget=4,
            seed=0,
            calibrator=ScoreCalibrator(min_samples=40),
        )
        bases = make_bases()
        traffic = [residual_stream(seed=100 + i) for i in range(n)]
        auditor.calibrate(traffic, bases)
        return auditor, bases

    def test_benign_traffic_scores_low_after_calibration(self):
        auditor, bases = self._calibrated_auditor()
        scores = []
        for i in range(20):
            m, action = auditor.audit_step(residual_stream(seed=900 + i), bases)
            assert m.calibration_ready
            scores.append(m.aggregate_threat)
            assert action.action_type in ("NOMINAL", "RAISE_AUDIT_RATE", "ARMED")
        mean = sum(scores) / len(scores)
        assert mean < 0.35, f"benign traffic mean threat {mean:.3f} -- detector saturating"

    def test_anomalous_traffic_scores_higher_than_benign(self):
        auditor, bases = self._calibrated_auditor()

        benign, anomalous = [], []
        for i in range(20):
            benign.append(auditor.audit_step(residual_stream(seed=900 + i), bases)[0])
            # Anomaly: collapse the residual stream onto a low-rank subspace,
            # which raises cross-layer MI and moves the drift profile.
            acts = residual_stream(seed=900 + i)
            g = torch.Generator().manual_seed(7)
            proj_basis, _ = torch.linalg.qr(torch.randn(D_MODEL, 4, generator=g))
            for idx in acts:
                acts[idx] = (acts[idx] @ proj_basis) @ proj_basis.T + 1e-3 * acts[idx]
            anomalous.append(auditor.audit_step(acts, bases)[0])

        b = sum(m.aggregate_threat for m in benign) / len(benign)
        a = sum(m.aggregate_threat for m in anomalous) / len(anomalous)
        assert a > b + 0.2, f"no separation: benign {b:.3f} vs anomalous {a:.3f}"

    def test_policy_requires_persistence_before_intercepting(self):
        policy = ResponsePolicy(consecutive_required=3)
        basis = GrassmannianConsensus(k_components=K)(make_bases())
        acts = {0: torch.randn(SEQ, D_MODEL)}
        kinds = [policy.evaluate(0.95, acts, basis).action_type for _ in range(4)]
        assert kinds == ["ARMED", "ARMED", "SUBSPACE_PROJECTION", "SUBSPACE_PROJECTION"]

    def test_audit_step_is_reproducible_from_seed(self):
        bases = make_bases()
        acts = residual_stream(seed=1)
        out = []
        for _ in range(2):
            a = AlignmentAuditor(LAYERS, K, audit_budget=4, seed=42)
            m, _ = a.audit_step(acts, bases, calibrating=True)
            out.append((m.audited_layers, m.raw))
        assert out[0][0] == out[1][0]
        for key in out[0][1]:
            assert math.isclose(out[0][1][key], out[1][1][key], rel_tol=1e-6)

    def test_uncalibrated_auditor_reports_calibrating_not_a_threat(self):
        auditor = AlignmentAuditor(LAYERS, K, audit_budget=4, seed=0)
        m, action = auditor.audit_step(residual_stream(), make_bases())
        assert m.aggregate_threat is None
        assert not m.calibration_ready
        assert action.action_type == "CALIBRATING"
