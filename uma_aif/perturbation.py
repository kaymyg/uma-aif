"""Layer sampling, perturbation injection, and local-sensitivity measurement.

Three corrections to v1.0.0.0 live here.

**The scheduler was not stochastic.** The original derived its mask from
``(layer_index * 2654435761) XOR (seed + counter) % 2``. The multiplier is odd,
so the low bit of ``i * 2654435761`` equals the low bit of ``i``, and the mask
reduced to ``(i & 1) ^ (state & 1)`` -- a strict alternating pattern that
flipped on every step. Zero entropy, fully predictable, despite the docstring
claiming ChaCha20/AES. It also selected a variable number of layers per step,
so the audit cost was not constant either.

**The perturbation envelope was mis-scaled.** Noise was scaled by ``|gamma|``,
the RMSNorm gain. Gamma multiplies the *normalised* activation, so it carries
no information about the magnitude of the pre-norm residual stream the noise is
actually being added to. Scaling is now driven by the per-token RMS of the
activation itself, with the gain used only to shape the noise across channels.

**The sensitivity statistic was catastrophic cancellation.** The original
computed ``1 - cos`` in fp32 for a perturbation of relative size ~5e-3, where
``cos ~ 1 - 1e-5``, then divided the standard deviation by the mean. Both the
subtraction and the ratio are dominated by float error in that regime. The
distance is now computed as a squared difference of unit vectors (no
cancellation) in float64, and reported as an amplification ratio that is ~1 for
a locally isometric map, which is interpretable rather than merely large.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LayerSampler", "EnvelopeNoiseInjector", "LocalSensitivityProbe"]


class LayerSampler:
    """Samples a fixed-size subset of layers per audit step.

    The subset size is constant, so the FLOP cost of an audit step does not
    depend on the draw. Which layers are read still varies, and that remains
    observable through cache and memory-access timing -- this class does not
    claim to close that channel.

    The generator is a standard Mersenne-Twister ``torch.Generator``. It is
    reproducible from a seed, which matters for forensic replay, but it is
    **not** cryptographically secure: an adversary who observes enough draws can
    predict future ones. If the threat model includes an adversary that adapts
    to the audit schedule, drive ``budget`` selection from ``secrets`` or a
    ChaCha20 stream instead.

    Args:
        num_layers: Total number of layers available to sample from.
        budget: How many layers to audit per step. Clamped to ``num_layers``.
        seed: Generator seed, for reproducible audit trails.
    """

    def __init__(self, num_layers: int, budget: int = 4, seed: int = 0) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        self.num_layers = num_layers
        self.budget = min(budget, num_layers)
        self.seed = seed
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self.step = 0

    def reset(self, seed: Optional[int] = None) -> None:
        """Restart the schedule, optionally from a new seed."""
        if seed is not None:
            self.seed = seed
        self._generator.manual_seed(self.seed)
        self.step = 0

    def sample(self, candidates: Optional[list[int]] = None) -> list[int]:
        """Draw ``budget`` distinct layer indices, sorted ascending.

        Args:
            candidates: Optional pool to draw from. Defaults to
                ``range(num_layers)``.
        """
        pool = list(range(self.num_layers)) if candidates is None else list(candidates)
        if not pool:
            raise ValueError("candidate pool is empty")
        take = min(self.budget, len(pool))
        perm = torch.randperm(len(pool), generator=self._generator)[:take]
        self.step += 1
        return sorted(pool[i] for i in perm.tolist())


class EnvelopeNoiseInjector(nn.Module):
    """Adds perturbations scaled to the native magnitude of the activations.

    Args:
        eta: Relative perturbation size. The injected noise has expected norm
            ``eta * ||x||`` per token.
        seed: Seed for the internal generator. Reproducible perturbations are
            required for an audit record to be re-derivable.
    """

    def __init__(self, eta: float = 0.005, seed: int = 0) -> None:
        super().__init__()
        if not 0.0 < eta < 1.0:
            raise ValueError(f"eta must be in (0, 1), got {eta}")
        self.eta = float(eta)
        self.seed = seed
        self._generators: dict[torch.device, torch.Generator] = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        gen = self._generators.get(device)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self._generators[device] = gen
        return gen

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed every per-device generator so a step can be replayed."""
        if seed is not None:
            self.seed = seed
        for device, gen in self._generators.items():
            gen.manual_seed(self.seed)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, norm_gain: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perturb ``x`` (``[N, d]``).

        Args:
            x: Activations.
            norm_gain: Optional ``[d]`` LayerNorm/RMSNorm gain. Used only to
                *shape* the noise across channels; it is normalised to unit mean
                so it cannot change the overall perturbation magnitude.
        """
        if x.ndim != 2:
            raise ValueError(f"expected [N, d], got shape {tuple(x.shape)}")
        d = x.shape[-1]
        gen = self._generator(x.device)
        noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)

        if norm_gain is not None:
            if norm_gain.shape != (d,):
                raise ValueError(
                    f"norm_gain must have shape ({d},), got {tuple(norm_gain.shape)}"
                )
            shape = norm_gain.abs().to(x.dtype)
            shape = shape / shape.mean().clamp_min(1e-6)
            noise = noise * shape

        # Rescale so the perturbation is exactly eta in relative norm per token.
        noise = F.normalize(noise, p=2.0, dim=-1)
        token_norm = x.norm(dim=-1, keepdim=True)
        return x + self.eta * token_norm * noise


class LocalSensitivityProbe(nn.Module):
    """Measures how much a layer amplifies a small input perturbation.

    Returns an *amplification ratio*: the relative output displacement divided
    by the relative input displacement. A locally isometric map scores ~1.0;
    values well above 1 indicate the activation sits near a sharp region of the
    represented function.

    This is a curvature/sensitivity measurement. It is **not** evidence of
    "self-preservation" or "concealment behaviour", which the v1.0.0.0
    docstring asserted without any mechanism connecting the two. Treat a high
    score as "this input sits somewhere unstable", nothing more.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = float(eps)

    @torch.no_grad()
    def forward(
        self,
        clean: torch.Tensor,
        perturbed: torch.Tensor,
        input_relative_delta: float,
    ) -> torch.Tensor:
        """Amplification ratio, averaged over tokens.

        Args:
            clean: ``[N, d]`` unperturbed activations.
            perturbed: ``[N, d]`` activations from the perturbed forward pass.
            input_relative_delta: The relative size of the perturbation that was
                injected upstream (i.e. ``eta``). Must be > 0.
        """
        if input_relative_delta <= 0:
            raise ValueError("input_relative_delta must be > 0")
        if clean.shape != perturbed.shape:
            raise ValueError(
                f"shape mismatch: {tuple(clean.shape)} vs {tuple(perturbed.shape)}"
            )
        # float64 throughout: the quantities of interest are O(eta) ~ 1e-3 and
        # differencing them in fp32 destroys most of the significand.
        a = F.normalize(clean.to(torch.float64), p=2.0, dim=-1)
        b = F.normalize(perturbed.to(torch.float64), p=2.0, dim=-1)
        # ||a - b||^2 = 2(1 - cos) with no catastrophic cancellation.
        angular = (a - b).norm(dim=-1)
        ratio = angular / input_relative_delta
        return ratio.mean().to(torch.float32)

    @torch.no_grad()
    def dispersion(self, clean: torch.Tensor, perturbed: torch.Tensor) -> torch.Tensor:
        """Coefficient of variation of per-token displacement, in float64.

        High dispersion means the perturbation response is concentrated on a few
        tokens rather than spread across the sequence.
        """
        a = F.normalize(clean.to(torch.float64), p=2.0, dim=-1)
        b = F.normalize(perturbed.to(torch.float64), p=2.0, dim=-1)
        disp = (a - b).norm(dim=-1)
        mean = disp.mean()
        if mean <= self.eps or disp.numel() < 2:
            return torch.zeros((), dtype=torch.float32)
        return (disp.std(unbiased=True) / mean).to(torch.float32)
