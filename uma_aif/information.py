"""Matrix-based Renyi entropy and cross-layer mutual information.

Follows Giraldo, Rao & Principe (2015), "Measures of entropy from data using
infinitely divisible kernels", which defines the entropy of a normalised Gram
matrix from its eigenspectrum and the joint entropy of two such matrices from
their Hadamard product.

Fixes relative to v1.0.0.0:

* ``alpha == 1`` no longer divides by zero; it falls back to the von Neumann
  (Shannon) limit.
* Eigenvalues are renormalised after flooring, so the spectrum still sums to 1
  and the entropy stays in ``[0, log2 N]``.
* The "Ledoit-Wolf adaptive shrinkage" of the original saturated at its upper
  clamp for essentially every realistic input (``||K_c||_F^2 >> 1e-4``), making
  it a constant 0.20 ridge that inflated every entropy toward ``log2 N``. It is
  replaced by an explicit, off-by-default ridge that is only applied as a
  fallback when the eigensolver fails to converge.
* Mutual information is normalised by ``min(S_i, S_j)`` so it lands in
  ``[0, 1]`` and can be compared across layers and sequence lengths. The raw
  value is unbounded above by ``log2 N``, which in the original flowed into a
  weighted sum that was then clamped to ``[0, 1]`` -- saturating the threat
  score at maximum for any sequence longer than a few tokens.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MatrixRenyiEntropy", "gram_matrix", "CrossLayerMutualInformation"]

_SOLVER_DTYPE = torch.float32
_LOG2 = 0.6931471805599453


@torch.no_grad()
def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """Trace-normalised cosine Gram matrix of ``x`` (``[N, d]`` -> ``[N, N]``).

    Rows are L2-normalised first, so the Gram matrix has unit diagonal and
    trace N; dividing by N yields a trace-1 PSD matrix (a density matrix).

    Raises:
        ValueError: if ``x`` contains no finite rows with non-zero norm.
    """
    if x.ndim != 2:
        raise ValueError(f"expected [N, d], got shape {tuple(x.shape)}")
    x = x.to(_SOLVER_DTYPE)
    norms = x.norm(dim=-1)
    if not torch.isfinite(x).all():
        raise ValueError("activations contain non-finite values")
    if (norms <= 0).any():
        raise ValueError(
            "activation matrix contains zero-norm rows; the Gram matrix would "
            "be rank-deficient with a zero trace"
        )
    xn = F.normalize(x, p=2.0, dim=-1)
    k = xn @ xn.T
    # Symmetrise against accumulation asymmetry before the eigensolver sees it.
    k = 0.5 * (k + k.T)
    return k / k.diagonal().sum()


class MatrixRenyiEntropy(nn.Module):
    """Matrix-based Renyi alpha-entropy of a trace-1 PSD Gram matrix.

    Args:
        alpha: Renyi order. ``alpha == 1`` uses the von Neumann limit.
        floor: Lower clamp applied to eigenvalues before the log, guarding
            against ``log(0)`` from numerically negative eigenvalues.
        fallback_ridge: Ridge mixed toward the maximally-mixed state and
            retried if the eigensolver raises. Set to 0 to disable the retry.
            This *biases the entropy upward*; it is a last resort, not a
            denoiser.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        floor: float = 1e-12,
        fallback_ridge: float = 1e-3,
    ) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if not 0.0 <= fallback_ridge < 1.0:
            raise ValueError(f"fallback_ridge must be in [0, 1), got {fallback_ridge}")
        self.alpha = float(alpha)
        self.floor = float(floor)
        self.fallback_ridge = float(fallback_ridge)

    @staticmethod
    def _eigvals(k: torch.Tensor) -> torch.Tensor:
        return torch.linalg.eigvalsh(k)

    @torch.no_grad()
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Entropy in bits. Returns a 0-dim tensor in ``[0, log2 N]``."""
        if k.ndim != 2 or k.shape[0] != k.shape[1]:
            raise ValueError(f"expected a square matrix, got {tuple(k.shape)}")
        n = k.shape[0]
        k = k.to(_SOLVER_DTYPE)
        trace = k.diagonal().sum()
        if not torch.isfinite(trace) or trace <= 0:
            raise ValueError(f"Gram matrix has non-positive trace ({trace!r})")
        k = k / trace

        try:
            eigvals = self._eigvals(k)
        except Exception:
            if self.fallback_ridge <= 0:
                raise
            eye = torch.eye(n, device=k.device, dtype=k.dtype) / n
            k = (1.0 - self.fallback_ridge) * k + self.fallback_ridge * eye
            eigvals = self._eigvals(k)

        eigvals = eigvals.clamp_min(self.floor)
        eigvals = eigvals / eigvals.sum()  # flooring perturbs the trace

        if abs(self.alpha - 1.0) < 1e-8:
            entropy = -(eigvals * eigvals.log()).sum() / _LOG2
        else:
            entropy = torch.log2((eigvals**self.alpha).sum()) / (1.0 - self.alpha)
        # Numerical slack can push the result marginally outside the interval.
        return entropy.clamp(0.0, float(torch.log2(torch.tensor(float(n)))))


class CrossLayerMutualInformation(nn.Module):
    """Normalised matrix-based Renyi MI between successive layer activations.

    ``forward`` returns a dict keyed by the ordered layer pair, plus the raw
    per-pair values, so a calibrator can baseline each pair independently.
    Reporting a single ``max`` over pairs -- as v1.0.0.0 did -- discards the
    information needed to tell "this pair is unusual" from "this pair is always
    high", and in a residual-stream transformer *every* adjacent pair is always
    high because the residual stream changes slowly between layers.
    """

    def __init__(self, alpha: float = 2.0) -> None:
        super().__init__()
        self.entropy = MatrixRenyiEntropy(alpha=alpha)

    @torch.no_grad()
    def pairwise(
        self, act_i: torch.Tensor, act_j: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(normalised_mi, raw_mi_bits)`` for two ``[N, d]`` tensors."""
        if act_i.shape[0] != act_j.shape[0]:
            raise ValueError(
                f"sequence length mismatch: {act_i.shape[0]} vs {act_j.shape[0]}"
            )
        k_i = gram_matrix(act_i)
        k_j = gram_matrix(act_j)

        s_i = self.entropy(k_i)
        s_j = self.entropy(k_j)

        joint_raw = k_i * k_j  # Hadamard product -> joint density matrix
        joint_trace = joint_raw.diagonal().sum()
        if joint_trace <= 0:
            zero = torch.zeros((), device=act_i.device, dtype=_SOLVER_DTYPE)
            return zero, zero
        s_joint = self.entropy(joint_raw / joint_trace)

        raw_mi = (s_i + s_j - s_joint).clamp_min(0.0)
        # I(X;Y) <= min(H(X), H(Y)); normalise into [0, 1].
        denom = torch.minimum(s_i, s_j).clamp_min(1e-6)
        return (raw_mi / denom).clamp(0.0, 1.0), raw_mi

    @torch.no_grad()
    def forward(
        self, layer_activations: Dict[int, torch.Tensor]
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """Normalised MI for each adjacent pair of the supplied layers."""
        indices: List[int] = sorted(layer_activations)
        out: Dict[Tuple[int, int], torch.Tensor] = {}
        for a, b in zip(indices, indices[1:]):
            nmi, _ = self.pairwise(layer_activations[a], layer_activations[b])
            out[(a, b)] = nmi
        return out
