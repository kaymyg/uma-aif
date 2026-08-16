# Code review: UMA-AIF v1.0.0.0

Every finding below is reproduced by `python examples/demo_v1_failures.py`, and
each has a corresponding named test in `tests/test_uma_aif.py::TestRegressions`.
The original source is preserved verbatim in `legacy/uma_aif_v1.py`.

## Summary

The v1 file is well-organised and the underlying ideas are real ones — the
matrix-based Rényi entropy construction is Giraldo–Rao–Príncipe (2015), and the
Hadamard-product joint entropy is the correct form for it. The problems are not
in the choice of literature. They are that several mechanisms do not perform the
operation their names and docstrings describe, and that the pipeline's final
score cannot discriminate.

Ranked by severity:

| # | Finding | Severity |
|---|---------|----------|
| 1 | Threat aggregation sums quantities on incomparable scales, then clamps — the detector saturates at maximum on benign input | Critical |
| 2 | Masked layers produce a degenerate Gram matrix; the eigensolver raises, and on the path where it doesn't, NaN passes the `trace <= 0` guard | Critical |
| 3 | Mitigation returns the *anomalous residual* and discards the consensus-aligned component | Critical |
| 4 | The "cryptographic stochastic" scheduler is a deterministic alternating pattern with zero entropy | High |
| 5 | "Byzantine-resilient spectral clamping" is mathematically incapable of changing its own output | High |
| 6 | `alpha == 1` raises `ZeroDivisionError` | High |
| 7 | The audit-awareness statistic is catastrophic cancellation in fp32 | High |
| 8 | "Adaptive Ledoit–Wolf shrinkage" is a constant 0.20 ridge in every realistic regime | Medium |
| 9 | The perturbation envelope is scaled by a quantity unrelated to the activations it perturbs | Medium |
| 10 | Data-dependent host-side branching defeats the stated constant-time property | Medium |
| 11 | Perturbation is applied to recorded activations, not to a second forward pass, so the sensitivity term measures nothing about the model | Medium |
| 12 | No `no_grad`, no seeding, no tests, dead parameters | Low |

---

## 1. The aggregate threat score cannot discriminate — Critical

```python
aggregate_threat = (0.4 * smuggling_score.item()
                  + 0.4 * awareness_score.item()
                  + 0.2 * drift_score.item())
aggregate_threat = min(max(aggregate_threat, 0.0), 1.0)
```

The weights sum to 1.0, which reads as a convex combination. But:

* `smuggling_score` is a raw Rényi mutual information in **bits**, bounded above
  by `log2(N)` — 5 bits at a sequence length of 32, 8 at 256.
* `awareness_score` is an unbounded coefficient of variation.
* `drift_score` is the only term actually in `[0, 1]`.

Measured on the demo's synthetic residual stream:

```
seq_len=  16  raw MI =  3.623 bits  ->  0.4 * MI = 1.449
seq_len=  64  raw MI =  4.564 bits  ->  0.4 * MI = 1.825
seq_len= 256  raw MI =  4.516 bits  ->  0.4 * MI = 1.807
```

The first term alone exceeds 1.0 on benign input. The clamp pins the aggregate
at maximum, `SUBSPACE_INTERCEPTION_CLAMP` fires on everything, and the 0.35 /
0.70 thresholds never separate anything. A detector with a 100% firing rate has
no true-positive rate worth reporting.

**Fix.** This is not a weight-tuning problem. Each raw statistic needs a
baseline observed on known-benign traffic; the live value is then reported as a
deviation from that baseline in robust standard deviations and squashed to
`[0, 1]`. Only then do the weights combine comparable things. See
`uma_aif/calibration.py`. Mutual information is additionally normalised by
`min(S_i, S_j)` so it is a unitless quantity in `[0, 1]` before calibration ever
sees it.

The corrected pipeline, calibrated on 60 benign snapshots and evaluated on 25
benign and 25 anomalous ones:

```
aggregate threat   benign 0.106   anomalous 0.535   separation +0.429
```

## 2. Masked layers destroy the measurement — Critical

`audit_step` zeroes unmonitored layers rather than skipping them:

```python
monitored_activations[l_idx] = act * mask
```

Those zeroed tensors are then handed to `_compute_gram_matrix`, where
`F.normalize` returns zero rows, `K` is the zero matrix, and `K / torch.trace(K)`
is `0/0`:

```
Gram matrix of a masked-out layer: all-NaN = True
trace = nan, and the guard is `if trace <= 0`;
NaN <= 0 evaluates to False, so the guard never fires.
Downstream clamp: min(max(nan, 0.0), 1.0) = nan
```

In practice it does not even reach the NaN — `torch.linalg.eigvalsh` raises
`_LinAlgError: the algorithm failed to converge` on the degenerate matrix, so
`audit_step` throws on its own nominal path. A monitoring control that raises
during normal operation has an availability failure before it has a detection
story.

**Fix.** Select layers, never zero them (`uma_aif/perturbation.py::LayerSampler`
returns indices). Validate inputs at the boundary and raise a clear `ValueError`
on genuinely degenerate activations rather than propagating NaN — for a security
control, failing loudly is correct and failing silently is not.

## 3. The mitigation has its sign inverted — Critical

```python
projection = act @ consensus_subspace @ consensus_subspace.T
mitigated_activations[layer_idx] = act - projection
```

`consensus_subspace` is the span the reference models agree on. `act - proj` is
the component *outside* that span. So the intervention discards everything the
references agree with and forwards the deviation — the exact opposite of the
docstring's "strip alignment-breaching components".

The contradiction is visible within the same file: one loop earlier, the
identical residual is used as the definition of the drift *score*, i.e. as the
measure of the problem. Measured on the mitigated output:

```
fraction of the 'mitigated' output lying inside the consensus subspace: 2.191e-07
```

**Fix.** `ResponsePolicy(project_mode="retain_consensus")` is now the default and
projects onto the consensus span. `"remove_consensus"` preserves the old
behaviour for anyone who wants it, but it has to be asked for by name.

## 4. The stochastic scheduler is not stochastic — High

```python
hashes = torch.bitwise_xor(layer_indices * 2654435761, state)
masks = (hashes % 2).to(torch.float32)
```

2654435761 is odd, so the low bit of `i * 2654435761` is the low bit of `i`.
The mask reduces to `(i & 1) XOR (state & 1)`:

```
step 0: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
step 1: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
step 2: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
```

Zero entropy, perfectly predictable one step ahead, despite a docstring claiming
ChaCha20/AES. Any adversary modelling the audit schedule gets it exactly right
from two observations. The mask also selects a variable number of layers in
general, so the audit cost is not constant either — undercutting the second half
of the class name.

**Fix.** `LayerSampler` draws a fixed-size subset from a seeded
`torch.Generator`, so the FLOP cost per step is constant and the schedule is
reproducible for forensic replay. The docstring now says plainly that a
Mersenne Twister is **not** cryptographically secure, and points at `secrets` /
ChaCha20 for threat models where the adversary adapts to the schedule. Claiming
a property is worse than not having it.

## 5. Spectral clamping cannot affect the output — High

```python
eigvals_clamped = torch.clamp(eigvals, max=self.max_influence * M)
eigvals_normalized = (eigvals_clamped / torch.sum(eigvals_clamped)) * self.k_components
consensus_subspace = eigvecs[:, :self.k_components]
return consensus_subspace
```

`eigvals_normalized` is assigned and never read. It could not have mattered
anyway: clamping eigenvalues leaves eigenvectors untouched by construction, and
the return value is built only from eigenvectors.

```
max |basis(max_influence=0.25) - basis(max_influence=99.0)| = 0.000e+00
```

So the entire Byzantine-resilience claim rests on a dead statement. Poisoning
one reference moves the consensus by exactly as much as it would with no
defence at all.

**Fix.** `GrassmannianConsensus` uses Krum-style trimmed aggregation (Blanchard
et al., 2017): score each reference by the summed chordal distance to its
`n_keep - 1` nearest neighbours, drop the `trim` worst, then average what
survives. Consensus displacement under poisoning, M = 7 references:

```
 poisoned   v1 (clamping)   v2 (trimmed)
        1          0.0493         0.0025
        2          0.1654         0.0081
        3          0.4255         0.0127
        4          0.8034         0.0206
        5          3.4355         0.0233
```

The honest caveat, stated in the module docstring and the demo: this holds
because `trim` is set to the true coalition size. The guarantee is conditional
on knowing an upper bound on how many references are compromised and on at
least two honest references remaining.

As a side effect the rewrite also stopped materialising `M` dense `d × d`
projection operators. At `d_model = 4096` with `M = 4` that was 268 MB in fp32
plus a 4096×4096 eigendecomposition, per audit step. The top-k eigenvectors of
`Σ wᵢ QᵢQᵢᵀ` are the top-k left singular vectors of `[√w₁Q₁ | … | √w_MQ_M]`, so
the cost drops from `O(d³)` to `O(d·(Mk)²)` with no approximation.

## 6. `alpha == 1` divides by zero — High

```python
entropy = (1.0 / (1.0 - self.alpha)) * torch.log2(...)
```

`alpha = 1` is the Shannon / von Neumann case — the most likely order a user
reaches for — and it is unguarded:

```
ShrinkageRenyiEntropy(alpha=1.0) raises ZeroDivisionError: float division by zero
```

**Fix.** `MatrixRenyiEntropy` falls back to `-Σ λ log₂ λ` at `alpha = 1`, and
the constructor validates `alpha > 0`.

Related, in the same function: eigenvalues are floored with `clamp(min=1e-8)`
but not renormalised afterwards, so the spectrum no longer sums to 1 and the
entropy drifts outside `[0, log₂N]`. The rewrite renormalises after flooring.

## 7. The awareness statistic is float noise — High

```python
cosine_distance = 1.0 - dot_product
divergence = torch.std(cosine_distance) / (torch.mean(cosine_distance) + 1e-6)
```

With `eta = 0.005`, the cosine between clean and perturbed activations is about
`1 - 1e-5`. Computing `1 - cos` in fp32 at that magnitude discards roughly five
of the seven available decimal digits; the standard deviation of the result is
then divided by a mean that is itself mostly rounding error. Five different
noise draws on identical input:

```
draw 0: divergence = 0.2390
draw 1: divergence = 0.2876
draw 2: divergence = 0.2906
draw 3: divergence = 0.2872
draw 4: divergence = 0.2292
```

A 25% spread with the signal held fixed.

**Fix.** `LocalSensitivityProbe` computes `‖â − b̂‖` on unit vectors in float64 —
algebraically `√(2(1−cos))` but with no cancellation — and reports it as an
*amplification ratio* against the injected perturbation size. That number is
~1.0 for a locally isometric map, which is interpretable, rather than merely
large. It is stable from `eta = 1e-2` down to `1e-5`, which the regression test
asserts.

## 8. The shrinkage is a constant — Medium

```python
mu = torch.clamp(frobenius_norm / (frobenius_norm + 1e-4), min=0.001, max=0.20)
```

`f / (f + 1e-4)` exceeds 0.20 whenever `f > 2.5e-5`. For a trace-normalised
cosine Gram matrix, `f` is 1e-2 or larger at every sequence length tested:

```
seq_len=   8  ||K_c||_F^2=9.436e-03  ->  mu=0.200
seq_len=  32  ||K_c||_F^2=1.519e-02  ->  mu=0.200
seq_len= 128  ||K_c||_F^2=1.558e-02  ->  mu=0.200
```

So the "adaptive" parameter is pinned at its ceiling universally. It is also not
Ledoit–Wolf, which estimates an optimal shrinkage intensity from the sample
variance of the eigenvalues relative to a target — this is an arbitrary
saturating function of the Frobenius norm. What it actually does is mix 20% of
the maximally-mixed state into every Gram matrix, biasing every entropy upward
toward `log₂N`. `frobenius_norm` is also a misnomer: the variable holds the
*squared* norm.

**Fix.** Removed. `MatrixRenyiEntropy` exposes `fallback_ridge`, applied only as
a retry when the eigensolver actually raises, documented as a bias rather than a
denoiser.

## 9. The perturbation envelope is mis-scaled — Medium

```python
scale = self.eta * torch.abs(norm_gamma)
noise = torch.randn_like(x) * scale
```

`gamma` is the RMSNorm gain. It multiplies the *normalised* activation, so it
carries no information about the magnitude of the pre-norm residual stream that
`x` actually is. Scaling noise by `|gamma|` therefore does not keep the
perturbation inside the "native activation manifold" the docstring invokes; on a
residual stream with norm 17 and a gain near 1, it injects a perturbation five
orders of magnitude smaller than intended, and the effective `eta` varies
per-model.

**Fix.** `EnvelopeNoiseInjector` scales by the per-token RMS of the activation
itself, so the relative perturbation is exactly `eta` by construction (asserted
in the tests). The gain is used only to *shape* noise across channels, and is
normalised to unit mean so it cannot change the magnitude.

## 10. Constant time is defeated by a host-side branch — Medium

```python
if torch.sum(monitored_activations[l_idx]) != 0:
```

This is a Python `if` on a device tensor. It forces a device→host
synchronisation per layer, and its outcome is exactly the secret the
constant-time scheduler exists to hide — an observer timing the audit learns
which layers were sampled. The three `.item()` calls in the aggregation add
three more syncs.

**Fix.** No branching on tensor values; layer selection happens on indices in
Python before any tensor work. The remaining honest caveat is in the
`LayerSampler` docstring: FLOP cost is constant, but *which* layers are read
still varies and remains visible through memory-access timing. That channel is
declared out of scope rather than papered over.

## 11. The sensitivity term measures the identity map — Medium

v1 perturbs the *recorded* activations and compares them to themselves:

```python
noised_act = self.noise_injector(act, gamma)
...
score = self.awareness_detector(monitored_activations[l_idx], noised_activations[l_idx])
```

No forward pass happens in between. Whatever this measures, it is a property of
the additive noise, not of the model — the model never sees the perturbation and
never responds to it. The claim that this detects "self-preservation or
concealment behaviours" has no mechanism connecting it to the computation.

**Fix.** `audit_step` accepts `perturbed_activations` from a genuine second
forward pass and documents that as the intended use. The in-place fallback is
retained as a smoke test, with the docstring stating that it reads ~1.0 by
construction and carries no signal. The class is renamed
`LocalSensitivityProbe` and its docstring says what it measures — local
curvature — and explicitly disclaims the deception interpretation.

## 12. Smaller items — Low

* No `torch.no_grad()` anywhere. Auditing a live model would build a graph over
  every activation. All metric paths are now decorated.
* No seeding, so an audit result could not be reproduced from its inputs. Both
  the sampler and the injector now take a seed, and a regression test asserts
  bit-level reproducibility of a step.
* Dead parameters: `sensitivity_threshold`, `d_model` on
  `AuditAwarenessDetector`, `d_model`/`num_layers` on
  `CrossLayerSmugglingDetector`, `reference_models_count` on the top-level
  class, `max_influence`. Unused imports: `math`, `Optional`, `List`.
* No input validation. Missing `layer_gammas[l_idx]` is a bare `KeyError`;
  mismatched sequence lengths, wrong ranks, and non-finite activations all pass
  silently into the linear algebra.
* Version baked into the class name (`UMA_AIF_v1_0_0_0`). Version belongs in
  package metadata; a rename should not be a breaking API change.
* `ResponseAction.mitigation_payload` is annotated `Dict[str, torch.Tensor]` but
  populated with integer keys.
* No tests.

## What was kept

The Hadamard-product joint entropy is the correct construction for matrix-based
Rényi mutual information and is unchanged. The overall shape — sample layers,
perturb, measure information flow and subspace drift, respond by policy — is
sound, and the rewrite preserves it. The three-term decomposition of the threat
score is also kept; the problem was never the decomposition, only the scales.

## What remains unvalidated

Stated so nobody reads more into the numbers than is in them:

* None of these statistics have been evaluated against a labelled dataset of
  adversarial model behaviour. The separation figure above is on synthetic
  activations with an injected low-rank collapse — evidence that the pipeline
  *can* discriminate a distribution shift, not that it detects misalignment.
* No transformer has been instrumented yet. Everything here runs on activation
  tensors supplied by the caller.
* Baseline standard deviations for several metrics are around 5e-3, so a benign
  distribution shift of ~1% registers as a two-sigma event. `std_floors` exists
  to blunt this, but choosing those floors is an open calibration problem.
* The perturbation-response path has not been exercised with a real second
  forward pass.
