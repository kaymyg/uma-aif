"""UMA-AIF: sampling-based internal-state auditing for transformer activations.

This package is a research instrument. It reports anomaly signals relative to a
calibrated benign baseline. It does not detect deception, and none of its
statistics have been validated against a labelled adversarial dataset.
"""

from .calibration import CalibrationError, RunningStat, ScoreCalibrator
from .engine import (
    AlignmentAuditor,
    AuditMetrics,
    ResponseAction,
    ResponsePolicy,
    ThreatWeights,
)
from .information import CrossLayerMutualInformation, MatrixRenyiEntropy, gram_matrix
from .perturbation import EnvelopeNoiseInjector, LayerSampler, LocalSensitivityProbe
from .subspace import GrassmannianConsensus, chordal_distance_matrix

__version__ = "0.2.0"

__all__ = [
    "AlignmentAuditor",
    "AuditMetrics",
    "CalibrationError",
    "CrossLayerMutualInformation",
    "EnvelopeNoiseInjector",
    "GrassmannianConsensus",
    "LayerSampler",
    "LocalSensitivityProbe",
    "MatrixRenyiEntropy",
    "ResponseAction",
    "ResponsePolicy",
    "RunningStat",
    "ScoreCalibrator",
    "ThreatWeights",
    "chordal_distance_matrix",
    "gram_matrix",
    "__version__",
]
