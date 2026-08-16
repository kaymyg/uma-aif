"""
UMA-AIF v1.0.0.0 Core Production Framework -- PRESERVED VERBATIM.

Kept only so the regression tests and `examples/demo_v1_failures.py` can
demonstrate the defects empirically. Do not import this from new code.
"""

import math
from typing import Dict, List, Optional, Tuple, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AuditMetrics(NamedTuple):
    smuggling_score: torch.Tensor
    audit_awareness_score: torch.Tensor
    subspace_drift_score: torch.Tensor
    aggregate_threat_level: float


class ResponseAction(NamedTuple):
    threat_level: float
    intercept_triggered: bool
    action_type: str
    mitigation_payload: Dict[str, torch.Tensor]


class ByzantineResilientGrassmannianMean(nn.Module):
    def __init__(self, d_model: int, k_components: int, max_influence: float = 0.25):
        super().__init__()
        self.d_model = d_model
        self.k_components = k_components
        self.max_influence = max_influence

    def forward(self, reference_bases: torch.Tensor) -> torch.Tensor:
        M, d_model, k = reference_bases.shape
        device = reference_bases.device
        dtype = reference_bases.dtype
        Q, _ = torch.linalg.qr(reference_bases)
        P = torch.bmm(Q, Q.transpose(1, 2))
        P_avg = torch.mean(P, dim=0)
        eigvals, eigvecs = torch.linalg.eigh(P_avg)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        eigvals_clamped = torch.clamp(eigvals, max=self.max_influence * M)
        eigvals_normalized = (eigvals_clamped / torch.sum(eigvals_clamped)) * self.k_components
        consensus_subspace = eigvecs[:, :self.k_components]
        return consensus_subspace


class ShrinkageRenyiEntropy(nn.Module):
    def __init__(self, alpha: float = 2.0, floor_eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.floor_eps = floor_eps

    def forward(self, K: torch.Tensor) -> torch.Tensor:
        N = K.shape[0]
        device = K.device
        dtype = K.dtype
        trace_K = torch.trace(K)
        if trace_K <= 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        identity = torch.eye(N, device=device, dtype=dtype)
        K_centered = K - (trace_K / N) * identity
        frobenius_norm = torch.sum(K_centered ** 2)
        mu = torch.clamp(frobenius_norm / (frobenius_norm + 1e-4), min=0.001, max=0.20)
        K_shrunk = (1.0 - mu) * K + mu * (trace_K / N) * identity
        K_norm = K_shrunk / torch.trace(K_shrunk)
        eigvals = torch.linalg.eigvalsh(K_norm)
        eigvals = torch.clamp(eigvals, min=self.floor_eps)
        entropy = (1.0 / (1.0 - self.alpha)) * torch.log2(torch.sum(eigvals ** self.alpha))
        return entropy


class CrossLayerSmugglingDetector(nn.Module):
    def __init__(self, d_model: int, num_layers: int, alpha: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.renyi_entropy = ShrinkageRenyiEntropy(alpha=alpha)

    def _compute_gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        K = x_norm @ x_norm.T
        return K / torch.trace(K)

    def compute_pairwise_mi(self, act_i: torch.Tensor, act_j: torch.Tensor) -> torch.Tensor:
        K_i = self._compute_gram_matrix(act_i)
        K_j = self._compute_gram_matrix(act_j)
        S_i = self.renyi_entropy(K_i)
        S_j = self.renyi_entropy(K_j)
        K_joint_raw = K_i * K_j
        trace_joint = torch.trace(K_joint_raw)
        if trace_joint <= 0:
            return torch.tensor(0.0, device=act_i.device)
        K_joint = K_joint_raw / trace_joint
        S_joint = self.renyi_entropy(K_joint)
        mi = S_i + S_j - S_joint
        return torch.clamp(mi, min=0.0)

    def forward(self, layer_activations: Dict[int, torch.Tensor]) -> torch.Tensor:
        layer_indices = sorted(list(layer_activations.keys()))
        if len(layer_indices) < 2:
            return torch.tensor(0.0, device=next(iter(layer_activations.values())).device)
        mi_scores = []
        for idx in range(len(layer_indices) - 1):
            i = layer_indices[idx]
            j = layer_indices[idx + 1]
            mi = self.compute_pairwise_mi(layer_activations[i], layer_activations[j])
            mi_scores.append(mi)
        mi_tensor = torch.stack(mi_scores)
        return torch.max(mi_tensor)


class ConstantTimeStochasticHookScheduler(nn.Module):
    def __init__(self, num_layers: int, seed: int = 0xDEADBEEF):
        super().__init__()
        self.num_layers = num_layers
        self.register_buffer("seed", torch.tensor(seed, dtype=torch.int64))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))

    def generate_layer_masks(self, device: torch.device) -> torch.Tensor:
        self.counter += 1
        state = self.seed + self.counter
        layer_indices = torch.arange(self.num_layers, device=device, dtype=torch.int64)
        hashes = torch.bitwise_xor(layer_indices * 2654435761, state)
        masks = (hashes % 2).to(torch.float32)
        return masks


class NativeEnvelopeNoiseInjector(nn.Module):
    def __init__(self, eta: float = 0.005):
        super().__init__()
        self.eta = eta

    def forward(self, x: torch.Tensor, norm_gamma: torch.Tensor) -> torch.Tensor:
        scale = self.eta * torch.abs(norm_gamma)
        noise = torch.randn_like(x) * scale
        return x + noise


class AuditAwarenessDetector(nn.Module):
    def __init__(self, d_model: int, sensitivity_threshold: float = 2.5):
        super().__init__()
        self.d_model = d_model
        self.sensitivity_threshold = sensitivity_threshold

    def forward(self, clean_activations: torch.Tensor, noised_activations: torch.Tensor) -> torch.Tensor:
        clean_norm = F.normalize(clean_activations, p=2, dim=-1)
        noised_norm = F.normalize(noised_activations, p=2, dim=-1)
        dot_product = torch.sum(clean_norm * noised_norm, dim=-1)
        cosine_distance = 1.0 - dot_product
        divergence = torch.std(cosine_distance) / (torch.mean(cosine_distance) + 1e-6)
        return divergence


class AdaptiveResponseProtocol:
    def __init__(self, threshold_audit: float = 0.35, threshold_intercept: float = 0.70):
        self.threshold_audit = threshold_audit
        self.threshold_intercept = threshold_intercept

    def evaluate(self, threat_score: float, current_activations: Dict[int, torch.Tensor],
                 consensus_subspace: torch.Tensor) -> ResponseAction:
        if threat_score < self.threshold_audit:
            return ResponseAction(threat_score, False, "NOMINAL_PASS", {})
        elif threat_score < self.threshold_intercept:
            return ResponseAction(threat_score, False, "INCREASE_AUDIT_FREQUENCY", {})
        else:
            mitigated_activations = {}
            for layer_idx, act in current_activations.items():
                projection = act @ consensus_subspace @ consensus_subspace.T
                mitigated_activations[layer_idx] = act - projection
            return ResponseAction(threat_score, True, "SUBSPACE_INTERCEPTION_CLAMP", mitigated_activations)


class UMA_AIF_v1_0_0_0(nn.Module):
    def __init__(self, d_model: int, num_layers: int, k_components: int,
                 reference_models_count: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.k_components = k_components
        self.grassmannian_mean = ByzantineResilientGrassmannianMean(d_model, k_components)
        self.smuggling_detector = CrossLayerSmugglingDetector(d_model, num_layers)
        self.hook_scheduler = ConstantTimeStochasticHookScheduler(num_layers)
        self.noise_injector = NativeEnvelopeNoiseInjector()
        self.awareness_detector = AuditAwarenessDetector(d_model)
        self.response_protocol = AdaptiveResponseProtocol()
        self.w_smuggling = 0.4
        self.w_awareness = 0.4
        self.w_drift = 0.2

    def audit_step(self, layer_activations, layer_gammas, reference_bases):
        device = next(iter(layer_activations.values())).device
        consensus_subspace = self.grassmannian_mean(reference_bases)
        layer_masks = self.hook_scheduler.generate_layer_masks(device)
        monitored_activations = {}
        noised_activations = {}
        for idx, (l_idx, act) in enumerate(layer_activations.items()):
            mask = layer_masks[l_idx % self.num_layers]
            gamma = layer_gammas[l_idx]
            noised_act = self.noise_injector(act, gamma)
            monitored_activations[l_idx] = act * mask
            noised_activations[l_idx] = noised_act * mask
        smuggling_score = self.smuggling_detector(monitored_activations)
        awareness_scores = []
        drift_scores = []
        for l_idx in monitored_activations.keys():
            if torch.sum(monitored_activations[l_idx]) != 0:
                score = self.awareness_detector(monitored_activations[l_idx], noised_activations[l_idx])
                awareness_scores.append(score)
                act = monitored_activations[l_idx]
                proj = act @ consensus_subspace @ consensus_subspace.T
                drift = torch.norm(act - proj) / (torch.norm(act) + 1e-6)
                drift_scores.append(drift)
        awareness_score = torch.mean(torch.stack(awareness_scores)) if awareness_scores else torch.tensor(0.0, device=device)
        drift_score = torch.mean(torch.stack(drift_scores)) if drift_scores else torch.tensor(0.0, device=device)
        aggregate_threat = (
            self.w_smuggling * smuggling_score.item() +
            self.w_awareness * awareness_score.item() +
            self.w_drift * drift_score.item()
        )
        aggregate_threat = min(max(aggregate_threat, 0.0), 1.0)
        metrics = AuditMetrics(smuggling_score, awareness_score, drift_score, aggregate_threat)
        response = self.response_protocol.evaluate(aggregate_threat, layer_activations, consensus_subspace)
        return metrics, response
