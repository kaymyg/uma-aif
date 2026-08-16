"""Empirical demonstration of the v1.0.0.0 defects.

Every claim in ``docs/REVIEW.md`` that says "in practice X happens" is produced
by this script. Run: ``python examples/demo_v1_failures.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# `legacy` is deliberately not a packaged module -- it exists only for these
# reproductions. Make it importable when the demo is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legacy.uma_aif_v1 import (  # noqa: E402
    AuditAwarenessDetector,
    ByzantineResilientGrassmannianMean,
    ConstantTimeStochasticHookScheduler,
    CrossLayerSmugglingDetector,
    ShrinkageRenyiEntropy,
    UMA_AIF_v1_0_0_0,
)

D_MODEL, SEQ, LAYERS, K = 64, 32, 12, 8
RULE = "-" * 72


def banner(n: int, title: str) -> None:
    print(f"\n{RULE}\n[{n}] {title}\n{RULE}")


def residual_stream(seed: int = 0, drift: float = 0.05):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(SEQ, D_MODEL, generator=g)
    out = {}
    for i in range(LAYERS):
        x = x + drift * torch.randn(SEQ, D_MODEL, generator=g)
        out[i] = x.clone()
    return out


def clustered_bases(m: int = 5, seed: int = 0, spread: float = 0.05):
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(D_MODEL, K, generator=g)
    return shared.unsqueeze(0) + spread * torch.randn(m, D_MODEL, K, generator=g)


def demo_scheduler_is_deterministic() -> None:
    banner(1, "The 'cryptographic stochastic' scheduler is a strict alternation")
    sched = ConstantTimeStochasticHookScheduler(LAYERS)
    for step in range(6):
        mask = sched.generate_layer_masks(torch.device("cpu"))
        print(f"  step {step}: {mask.to(torch.int).tolist()}")
    print("\n  mask[i] reduces to (i & 1) XOR (state & 1): the multiplier")
    print("  2654435761 is odd, so it never touches the low bit. Zero entropy.")


def demo_spectral_clamping_is_a_noop() -> None:
    banner(2, "'Spectral clamping' cannot change the returned basis")
    bases = clustered_bases()
    a = ByzantineResilientGrassmannianMean(D_MODEL, K, max_influence=0.25)(bases)
    b = ByzantineResilientGrassmannianMean(D_MODEL, K, max_influence=99.0)(bases)
    print(f"  max |basis(max_influence=0.25) - basis(max_influence=99.0)| = "
          f"{float((a - b).abs().max()):.3e}")
    print("  Clamping eigenvalues leaves eigenvectors untouched; the clamped")
    print("  vector `eigvals_normalized` is computed and then never used.")

    print("\n  Consensus displacement under a poisoning coalition, M = 7 refs.")
    print("  Squared chordal distance between the consensus computed on clean")
    print(f"  references and on poisoned ones (0 = unmoved, {2 * K} = orthogonal).")
    print("  Each estimator is compared against its own clean baseline, so the")
    print("  numbers isolate the effect of poisoning, not of using fewer refs.\n")
    print(f"  {'poisoned':>9s} {'v1 (clamping)':>15s} {'v2 (trimmed)':>14s}")

    from uma_aif import GrassmannianConsensus

    clean7 = clustered_bases(m=7, seed=3)
    g = torch.Generator().manual_seed(9)
    v1 = ByzantineResilientGrassmannianMean(D_MODEL, K)
    a7 = v1(clean7)
    for n_bad in (1, 2, 3, 4, 5):
        poisoned = clean7.clone()
        for j in range(n_bad):
            poisoned[j] = torch.randn(D_MODEL, K, generator=g) * 50
        v2 = GrassmannianConsensus(k_components=K, trim=n_bad)
        d_v1 = 2 * K - 2 * float((a7.T @ v1(poisoned)).pow(2).sum())
        d_v2 = 2 * K - 2 * float((v2(clean7).T @ v2(poisoned)).pow(2).sum())
        note = "  <- coalition is the majority" if n_bad > 3 else ""
        print(f"  {n_bad:>9d} {d_v1:>15.4f} {d_v2:>14.4f}{note}")
    print("\n  v1 degrades monotonically because it has no mechanism at all:")
    print("  the only line mentioning resilience is the dead assignment to")
    print("  `eigvals_normalized`.")
    print("\n  v2 holds even past the majority, but only because `trim` is set")
    print("  to the true coalition size here. That is the real limitation and")
    print("  it should be stated plainly: the guarantee is conditional on")
    print("  knowing an upper bound on how many references are compromised,")
    print("  and on at least two honest references remaining. Set trim too")
    print("  low and poisoned references survive the cut.")


def demo_shrinkage_saturates() -> None:
    banner(3, "The 'adaptive Ledoit-Wolf' shrinkage is a constant 0.20 ridge")
    for seq in (8, 32, 128):
        x = torch.randn(seq, D_MODEL)
        xn = torch.nn.functional.normalize(x, p=2, dim=-1)
        k = xn @ xn.T
        k = k / torch.trace(k)
        centered = k - (torch.trace(k) / seq) * torch.eye(seq)
        fro = float((centered**2).sum())
        mu = min(max(fro / (fro + 1e-4), 0.001), 0.20)
        print(f"  seq_len={seq:4d}  ||K_c||_F^2={fro:.3e}  ->  mu={mu:.3f}")
    print("\n  mu = f/(f+1e-4) exceeds 0.20 whenever f > 2.5e-5, i.e. always.")
    print("  It is a fixed ridge that biases every entropy toward log2(N).")


def demo_alpha_one_divides_by_zero() -> None:
    banner(4, "alpha == 1 divides by zero")
    x = torch.randn(SEQ, D_MODEL)
    xn = torch.nn.functional.normalize(x, p=2, dim=-1)
    k = xn @ xn.T
    k = k / torch.trace(k)
    try:
        ShrinkageRenyiEntropy(alpha=1.0)(k)
        print("  no error raised (unexpected)")
    except ZeroDivisionError as exc:
        print(f"  ShrinkageRenyiEntropy(alpha=1.0) raises {type(exc).__name__}: {exc}")
    print("  alpha=1 is the Shannon / von Neumann case -- the single most")
    print("  likely order a user would reach for -- and it is unguarded.")


def demo_mi_is_unbounded() -> None:
    banner(5, "Raw MI is unbounded, so the threat score saturates at 1.0")
    det = CrossLayerSmugglingDetector(D_MODEL, LAYERS)
    for seq in (16, 64, 256):
        a = torch.randn(seq, D_MODEL)
        b = a + 0.05 * torch.randn(seq, D_MODEL)
        mi = float(det.compute_pairwise_mi(a, b))
        print(f"  seq_len={seq:4d}  raw MI = {mi:6.3f} bits  ->  "
              f"0.4 * MI = {0.4 * mi:.3f}")
    print("\n  The aggregate is 0.4*MI + 0.4*awareness + 0.2*drift, clamped to")
    print("  [0, 1]. The first term alone exceeds 1.0. Every input is maximal")
    print("  threat and the 0.35 / 0.70 thresholds never discriminate.")


def demo_awareness_is_float_noise() -> None:
    banner(6, "The awareness statistic is dominated by float error")
    det = AuditAwarenessDetector(D_MODEL)
    x = torch.randn(SEQ, D_MODEL)
    print("  Same input, same eta, five different noise draws:")
    for i in range(5):
        torch.manual_seed(i)
        y = x + 0.005 * torch.abs(torch.ones(D_MODEL)) * torch.randn_like(x)
        print(f"    draw {i}: divergence = {float(det(x, y)):.4f}")
    print("\n  It computes 1 - cos in fp32 where cos ~ 1 - 1e-5, losing about")
    print("  five decimal digits, then divides std by that mean.")


def demo_nan_from_masked_layers() -> None:
    banner(7, "Unmonitored layers inject NaN into every downstream metric")
    det = CrossLayerSmugglingDetector(D_MODEL, LAYERS)
    zeroed = torch.zeros(SEQ, D_MODEL)
    k = det._compute_gram_matrix(zeroed)
    print(f"  Gram matrix of a masked-out layer: all-NaN = {bool(k.isnan().all())}")
    print(f"  trace = {float(torch.trace(k))}, and the guard is `if trace <= 0`;")
    print(f"  NaN <= 0 evaluates to {float('nan') <= 0}, so the guard never fires.")
    print(f"  Downstream clamp: min(max(nan, 0.0), 1.0) = "
          f"{min(max(float('nan'), 0.0), 1.0)}")

    print("\n  Running the full v1 audit_step on a normal residual stream:")
    model = UMA_AIF_v1_0_0_0(D_MODEL, LAYERS, K)
    acts = residual_stream()
    gammas = {i: torch.ones(D_MODEL) for i in range(LAYERS)}
    try:
        metrics, response = model.audit_step(acts, gammas, clustered_bases(m=4))
        print(f"    smuggling  = {float(metrics.smuggling_score)}")
        print(f"    awareness  = {float(metrics.audit_awareness_score)}")
        print(f"    drift      = {float(metrics.subspace_drift_score)}")
        print(f"    aggregate  = {metrics.aggregate_threat_level}")
        print(f"    action     = {response.action_type}")
    except Exception as exc:
        print(f"    {type(exc).__name__}: {str(exc).splitlines()[0]}")
        print("\n  It does not even reach the NaN: the degenerate Gram matrix")
        print("  takes down the eigensolver first. A security control that")
        print("  raises on its own nominal path has an availability failure")
        print("  before it has a detection story.")


def demo_mitigation_sign_is_inverted() -> None:
    banner(8, "Mitigation keeps the deviation and discards the consensus")
    model = UMA_AIF_v1_0_0_0(D_MODEL, LAYERS, K)
    basis = ByzantineResilientGrassmannianMean(D_MODEL, K)(clustered_bases())
    act = torch.randn(SEQ, D_MODEL)
    action = model.response_protocol.evaluate(0.99, {0: act}, basis)
    out = action.mitigation_payload[0]
    inside = float(((out @ basis) @ basis.T).norm() / out.norm())
    print(f"  fraction of the 'mitigated' output lying inside the consensus "
          f"subspace: {inside:.3e}")
    print("  It returns `act - proj`: the component that AGREES with the")
    print("  reference models is removed, and the anomalous residual is what")
    print("  gets passed downstream. One loop earlier the same residual is")
    print("  used as the drift score, i.e. as the definition of the problem.")


def main() -> None:
    torch.manual_seed(0)
    demo_scheduler_is_deterministic()
    demo_spectral_clamping_is_a_noop()
    demo_shrinkage_saturates()
    demo_alpha_one_divides_by_zero()
    demo_mi_is_unbounded()
    demo_awareness_is_float_noise()
    demo_nan_from_masked_layers()
    demo_mitigation_sign_is_inverted()
    print(f"\n{RULE}\nAll eight reproduce. See docs/REVIEW.md for the fixes.\n{RULE}")


if __name__ == "__main__":
    main()
