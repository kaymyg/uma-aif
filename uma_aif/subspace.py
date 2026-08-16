"""Robust reference-subspace consensus on the Grassmannian.

The original v1.0.0.0 implementation formed M dense d x d projection operators,
averaged them, ran a full symmetric eigendecomposition, and then "spectrally
clamped" the eigenvalues -- an operation that provably cannot change the
returned basis, because clamping eigenvalues leaves eigenvectors untouched.
The Byzantine-resilience mechanism was therefore a no-op.

This module replaces it with trimmed aggregation on the Grassmannian, which is
an actual robust estimator: references whose subspaces sit far from the
consensus (in chordal distance) are dropped before averaging, so a coalition of
size <= ``trim`` cannot steer the result.

It also never materialises a d x d matrix. The top-k eigenvectors of
``P = sum_i w_i Q_i Q_i^T`` are the top-k left singular vectors of the
d x (M*k) matrix ``[sqrt(w_1) Q_1 | ... | sqrt(w_M) Q_M]``, so the cost drops
from O(d^3) to O(d * (M*k)^2).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GrassmannianConsensus", "chordal_distance_matrix"]

# eigh/svd are not implemented for half precision on most backends.
_SOLVER_DTYPE = torch.float32


def _orthonormalise(bases: torch.Tensor) -> torch.Tensor:
    """Batch-orthonormalise ``[M, d, k]`` reference bases via reduced QR."""
    q, _ = torch.linalg.qr(bases, mode="reduced")
    return q


def chordal_distance_matrix(q: torch.Tensor) -> torch.Tensor:
    """Pairwise squared chordal distance between the subspaces spanned by ``q``.

    For orthonormal bases ``Q_i, Q_j`` of equal rank k::

        ||Q_i Q_i^T - Q_j Q_j^T||_F^2 = 2k - 2 ||Q_i^T Q_j||_F^2

    so the d x d projectors never need to be formed.

    Args:
        q: ``[M, d, k]`` batch of orthonormal bases.

    Returns:
        ``[M, M]`` matrix of squared chordal distances, zero on the diagonal.
    """
    m, _, k = q.shape
    # cross[i, j] = Q_i^T Q_j  ->  [M, M, k, k]
    cross = torch.einsum("idk,jdl->ijkl", q, q)
    overlap = (cross**2).sum(dim=(-2, -1))
    dist2 = 2.0 * k - 2.0 * overlap
    return dist2.clamp_min(0.0)


class GrassmannianConsensus(nn.Module):
    """Trimmed consensus subspace across M reference models.

    Args:
        k_components: Dimension of the consensus subspace to return.
        trim: Number of most-deviant references to discard before averaging.
            Guarantees resilience to a coalition of at most ``trim`` poisoned
            references. Must satisfy ``trim < M - 1`` at call time.
        weighting: ``"uniform"`` weights surviving references equally;
            ``"inverse_distance"`` additionally down-weights references by their
            median chordal distance to the others.
    """

    def __init__(
        self,
        k_components: int,
        trim: int = 1,
        weighting: str = "inverse_distance",
    ) -> None:
        super().__init__()
        if k_components < 1:
            raise ValueError(f"k_components must be >= 1, got {k_components}")
        if trim < 0:
            raise ValueError(f"trim must be >= 0, got {trim}")
        if weighting not in ("uniform", "inverse_distance"):
            raise ValueError(f"unknown weighting {weighting!r}")
        self.k_components = k_components
        self.trim = trim
        self.weighting = weighting

    @torch.no_grad()
    def forward(self, reference_bases: torch.Tensor) -> torch.Tensor:
        """Compute the consensus subspace.

        Args:
            reference_bases: ``[M, d_model, k]`` bases from M reference models.
                They need not be orthonormal; QR is applied first.

        Returns:
            ``[d_model, k_components]`` orthonormal consensus basis, in the
            dtype and on the device of ``reference_bases``.
        """
        if reference_bases.ndim != 3:
            raise ValueError(
                f"expected [M, d_model, k], got shape {tuple(reference_bases.shape)}"
            )
        m, d_model, k = reference_bases.shape
        if self.k_components > min(k * m, d_model):
            raise ValueError(
                f"k_components={self.k_components} exceeds the rank available "
                f"from M={m} references of rank k={k} in d_model={d_model}"
            )
        out_dtype = reference_bases.dtype
        work = reference_bases.to(_SOLVER_DTYPE)

        q = _orthonormalise(work)

        if m == 1:
            weights = torch.ones(1, dtype=_SOLVER_DTYPE, device=q.device)
            keep = torch.zeros(1, dtype=torch.long, device=q.device)
        else:
            dist2 = chordal_distance_matrix(q)
            off_diag = dist2 + torch.diag(
                torch.full((m,), float("inf"), device=q.device, dtype=_SOLVER_DTYPE)
            )
            n_keep = max(1, m - min(self.trim, m - 1))

            # Krum-style centrality (Blanchard et al., 2017): score each
            # reference by the sum of distances to its n_keep - 1 nearest
            # neighbours, not by its distance to all of them. An honest
            # reference has a tight cluster of honest neighbours and scores
            # near zero; a poisoned one does not, even when the poisoned
            # references outnumber the honest ones. Using the median instead
            # -- as an earlier draft of this module did -- fails exactly when
            # the coalition reaches half the population, because then the
            # median neighbour of an honest reference is a poisoned one.
            neighbours = max(1, min(n_keep - 1, m - 1))
            centrality = torch.topk(
                off_diag, neighbours, dim=1, largest=False
            ).values.sum(dim=1)
            keep = torch.topk(centrality, n_keep, largest=False).indices

            if self.weighting == "uniform":
                weights = torch.ones(n_keep, dtype=_SOLVER_DTYPE, device=q.device)
            else:
                kept = centrality[keep]
                weights = 1.0 / (kept + kept.median().clamp_min(1e-6))
            weights = weights / weights.sum()

        # Stack sqrt(w_i) Q_i horizontally: [d_model, n_keep * k].
        stacked = (q[keep] * weights.sqrt().view(-1, 1, 1)).permute(1, 0, 2)
        stacked = stacked.reshape(d_model, -1)

        u, _, _ = torch.linalg.svd(stacked, full_matrices=False)
        consensus = u[:, : self.k_components]
        return consensus.to(out_dtype)

    def extra_repr(self) -> str:
        return (
            f"k_components={self.k_components}, trim={self.trim}, "
            f"weighting={self.weighting!r}"
        )
