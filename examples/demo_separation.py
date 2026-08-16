"""Does the calibrated auditor actually separate benign from anomalous?

Run: ``python examples/demo_separation.py``

The synthetic traffic is a crude residual-stream stand-in: each layer is the
previous one plus a small update, so adjacent layers are highly correlated.
That is the regime real transformers live in, and it is the regime where an
uncalibrated MI detector saturates.
"""

from __future__ import annotations

import statistics

import torch

from uma_aif import AlignmentAuditor, ScoreCalibrator

D_MODEL, SEQ, LAYERS, K = 64, 32, 12, 8


def bases(m: int = 5, seed: int = 0, spread: float = 0.05) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(D_MODEL, K, generator=g)
    return shared.unsqueeze(0) + spread * torch.randn(m, D_MODEL, K, generator=g)


def residual_stream(seed: int, drift: float = 0.05) -> dict[int, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(SEQ, D_MODEL, generator=g)
    out = {}
    for i in range(LAYERS):
        x = x + drift * torch.randn(SEQ, D_MODEL, generator=g)
        out[i] = x.clone()
    return out


def low_rank_collapse(seed: int, rank: int = 4) -> dict[int, torch.Tensor]:
    """Anomaly: force the stream through a narrow subspace."""
    acts = residual_stream(seed)
    g = torch.Generator().manual_seed(7)
    proj, _ = torch.linalg.qr(torch.randn(D_MODEL, rank, generator=g))
    return {i: (a @ proj) @ proj.T + 1e-3 * a for i, a in acts.items()}


def main() -> None:
    ref = bases()
    auditor = AlignmentAuditor(
        LAYERS, K, audit_budget=4, seed=0, calibrator=ScoreCalibrator(min_samples=40)
    )

    print("calibrating on 60 benign snapshots...")
    summary = auditor.calibrate([residual_stream(100 + i) for i in range(60)], ref)
    for name, st in summary.items():
        print(f"  {name:26s} mean={st['mean']:.4f}  std={st['std']:.5f}")

    print("\nscoring 25 benign and 25 anomalous snapshots")
    benign = [auditor.audit_step(residual_stream(900 + i), ref)[0] for i in range(25)]
    anomalous = [auditor.audit_step(low_rank_collapse(900 + i), ref)[0] for i in range(25)]

    def col(ms, key):
        return statistics.mean(m.raw[key] for m in ms)

    def cal(ms, key):
        return statistics.mean(m.calibrated[key] for m in ms)

    print(f"\n{'metric':26s} {'benign raw':>11s} {'anom raw':>10s} "
          f"{'benign cal':>11s} {'anom cal':>10s}")
    for key in ("smuggling", "sensitivity", "drift"):
        print(f"{key:26s} {col(benign, key):11.4f} {col(anomalous, key):10.4f} "
              f"{cal(benign, key):11.3f} {cal(anomalous, key):10.3f}")

    b = statistics.mean(m.aggregate_threat for m in benign)
    a = statistics.mean(m.aggregate_threat for m in anomalous)
    print(f"\naggregate threat   benign {b:.3f}   anomalous {a:.3f}   "
          f"separation {a - b:+.3f}")

    actions = [auditor.audit_step(low_rank_collapse(50 + i), ref)[1] for i in range(4)]
    print("\nresponse trajectory on sustained anomaly:")
    for i, act in enumerate(actions):
        print(f"  step {i}: {act.action_type:22s} {act.reason}")


if __name__ == "__main__":
    main()
