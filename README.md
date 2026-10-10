# UMA-AIF

[![tests](https://github.com/kaymyg/uma-aif/actions/workflows/tests.yml/badge.svg)](https://github.com/kaymyg/uma-aif/actions/workflows/tests.yml)
[![CodeQL](https://github.com/kaymyg/uma-aif/actions/workflows/codeql.yml/badge.svg)](https://github.com/kaymyg/uma-aif/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sampling-based internal-state auditing for transformer activations.

A research instrument, not a product. It samples a fixed budget of layers per
step and reports three anomaly signals against a baseline calibrated on
known-benign traffic:

* **Cross-layer mutual information** — matrix-based Rényi MI between adjacent
  audited layers, normalised to `[0, 1]`. Elevated values mean those layers
  share more structure than the baseline did.
* **Local sensitivity** — how much a layer amplifies a small native-scale
  perturbation. ~1.0 for a locally isometric map.
* **Subspace drift** — energy outside the span that a set of reference models
  agrees on.

None of these detects deception. They detect *distribution shift relative to an
observed baseline*, which is a weaker and more honest claim. See
[What this does not do](#what-this-does-not-do).

## Status

This is version 0.2.0, a rewrite of a v1.0.0.0 draft that did not work. The
original is preserved verbatim in `legacy/` because the failure modes are
instructive, and `docs/REVIEW.md` walks through all twelve of them with
reproductions. Three were fatal:

* The threat score summed a quantity in bits, an unbounded ratio, and a fraction,
  then clamped to `[0, 1]` — so it read 1.0 on every input and the thresholds
  never discriminated.
* Unmonitored layers were zeroed rather than skipped, producing a degenerate
  Gram matrix that crashed the eigensolver on the nominal path.
* The mitigation step discarded the consensus-aligned component and forwarded the
  anomalous residual — the inverse of its stated purpose.

Reproduce all of them:

```bash
python examples/demo_v1_failures.py
```

## Install

```bash
pip install -e .
pytest
```

Requires `torch >= 2.0`. CPU is fine.

## Use

```python
import torch
from uma_aif import AlignmentAuditor, ScoreCalibrator

auditor = AlignmentAuditor(
    num_layers=32,
    k_components=16,
    audit_budget=6,          # layers sampled per step; constant cost
    eta=0.005,               # relative perturbation size
    trim=1,                  # poisoned references tolerated
    seed=0,                  # reproducible audits
    calibrator=ScoreCalibrator(min_samples=64),
)

# 1. Baseline on traffic you believe is benign. This is not optional --
#    without it every threshold is arbitrary.
auditor.calibrate(benign_snapshots, reference_bases)

# 2. Audit. Supply activations from a real perturbed forward pass when you can.
metrics, action = auditor.audit_step(
    layer_activations,          # {layer: [seq_len, d_model]}
    reference_bases,            # [M, d_model, k]
    perturbed_activations=...,  # strongly preferred
)

print(metrics.raw)               # uncalibrated statistics
print(metrics.calibrated)        # [0, 1] deviations from baseline
print(metrics.aggregate_threat)  # None until the baseline is established
print(action.action_type)        # CALIBRATING | NOMINAL | RAISE_AUDIT_RATE
                                 # | ARMED | SUBSPACE_PROJECTION
```

`aggregate_threat` is `None` and the action is `CALIBRATING` until the baseline
has enough samples. The auditor will not invent a threat level it has no basis
for.

## Does it separate anything?

`python examples/demo_separation.py`, on synthetic residual-stream activations
with an injected low-rank collapse:

```
metric                      benign raw   anom raw  benign cal   anom cal
smuggling                       0.8793     0.5495       0.112      1.000
sensitivity                     0.9923     0.9918       0.095      0.126
drift                           0.9362     0.9522       0.118      0.423

aggregate threat   benign 0.106   anomalous 0.535   separation +0.429
```

Two things in that table are worth reading carefully.

Benign normalised MI is **0.88** — adjacent layers of a residual stream are
nearly identical, so a raw MI detector is saturated before anything goes wrong.
This is why the baseline matters more than the statistic.

The anomaly *lowers* MI rather than raising it. A one-sided test would have
scored it 0.0 and missed it entirely, which is why the calibrator is two-sided
by default.

## Design notes

**Calibration is the architecture.** Raw statistics on incomparable scales
cannot be combined by weighting, however carefully the weights are chosen. Each
statistic is z-scored against a benign baseline and squashed through a logistic
before the weights ever apply. `ScoreCalibrator` serialises, so a baseline is a
versionable artefact.

**Reference consensus uses Krum-style trimmed aggregation** (Blanchard et al.,
2017), scoring each reference by summed chordal distance to its nearest
neighbours and dropping the `trim` worst. Conditional on `trim` bounding the
true coalition size and two honest references surviving. It never materialises
a `d × d` projector: the top-k eigenvectors of `Σ wᵢQᵢQᵢᵀ` are the top-k left
singular vectors of `[√w₁Q₁ | … | √w_MQ_M]`, so cost is `O(d(Mk)²)` not `O(d³)`.

**Sampling is seeded, not secure.** `LayerSampler` uses a `torch.Generator`. The
per-step FLOP cost is constant because the subset size is fixed, but which
layers get read is still visible through memory-access timing, and an adversary
observing enough draws can predict the schedule. If that is in your threat
model, drive selection from `secrets` or ChaCha20. The v1 draft claimed
cryptographic scheduling while emitting a strict alternating pattern; claiming a
property you don't have is worse than not having it.

**Failures are loud.** Degenerate activations raise `ValueError` rather than
propagating NaN. A monitoring control that silently reports NOMINAL on a failed
measurement is worse than no control.

## What this does not do

* It does not detect deception, sandbagging, or "audit awareness". The
  sensitivity probe measures local curvature. Nothing here connects that to
  intent.
* None of the statistics have been evaluated against a labelled dataset of
  adversarial model behaviour. The separation figure above is synthetic — it
  shows the pipeline *can* discriminate a distribution shift, not that it
  detects misalignment.
* No transformer has been instrumented. Everything operates on activation
  tensors supplied by the caller.
* Baseline standard deviations run around 5e-3 for several metrics, so a ~1%
  benign distribution shift registers as a two-sigma event. `std_floors` blunts
  this; choosing the floors is an open problem.
* Interception is a blunt instrument. Projecting activations onto a reference
  subspace mid-forward-pass will degrade the model in ways nobody here has
  measured.

## Relationship to AMES

[ames](https://github.com/kaymyg/ames) contains an agent from the host side and
asks whether it left the box. UMA-AIF sits inside the forward pass and asks
whether the computation drifted from its references. They are complements, and
`docs/INTEGRATION.md` explains why merging them would break AMES's
evidence/testimony separation — UMA-AIF's output is guest-internal, which AMES
deliberately excludes from enforcement.

## Layout

```
uma_aif/
  subspace.py       trimmed Grassmannian consensus
  information.py    matrix Rényi entropy, normalised cross-layer MI
  perturbation.py   layer sampling, noise injection, sensitivity probe
  calibration.py    baseline statistics and score normalisation
  engine.py         orchestration and response policy
legacy/             v1.0.0.0, verbatim, for the regression demos
examples/           reproductions and the separation demo
docs/REVIEW.md      full review of v1 with reproductions
docs/INTEGRATION.md fit assessment against ames and kaysentinel
tests/              31 tests: units, one per v1 defect, end-to-end
```

## References

* Giraldo, Rao & Príncipe (2015), *Measures of entropy from data using
  infinitely divisible kernels* — the matrix-based Rényi entropy and the
  Hadamard-product joint entropy.
* Blanchard, El Mhamdi, Guerraoui & Stainer (2017), *Machine Learning with
  Adversaries: Byzantine Tolerant Gradient Descent* — the Krum aggregation rule.
* Ledoit & Wolf (2004), *A well-conditioned estimator for large-dimensional
  covariance matrices* — cited here mainly to note that the v1 "adaptive
  Ledoit–Wolf shrinkage" was not this.

## Licence

MIT.
