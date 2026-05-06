"""
AISec linear risk scorer.

Implements the risk model from the paper:

    R(x) = sigmoid(w^T x + b)

Where:
    x  — feature vector (8 dimensions, all in [0.0, 1.0])
    w  — weight vector (one weight per dimension)
    b  — bias term
    R  — normalised risk score in (0.0, 1.0)

Design principles:
    - Weights are explicit and documented — no black box.
    - Each weight has a written rationale tied to the paper.
    - The model is intentionally simple and interpretable.
    - Sigmoid normalisation keeps output bounded in (0.0, 1.0).
    - Scenario-specific weight sets reflect different threat models.

Paper reference:
    Section 5 — Formalization and Enforceable Control Mechanisms.
    R(x) = w^T x + b, normalised to [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from aisec.storage.models import FeatureVector, Scenario


# ── Weight sets ───────────────────────────────────────────────────────────────
#
# Feature vector dimension order (must match models.FeatureVector.dimensions):
#   0  action_type_encoding
#   1  keyword_risk_score
#   2  frequency_score
#   3  api_call_flag
#   4  file_access_flag
#   5  network_access_flag
#   6  sensitive_path_flag
#   7  privilege_flag
#
# Weight rationale:
#   Trading AI — financial impact dominates.
#     High weight on keyword_risk (trading-specific danger words),
#     api_call (all trades go via external API),
#     privilege (self-modifying risk limits is catastrophic).
#
#   Urban AI — infrastructure impact dominates.
#     High weight on sensitive_path (protected city systems),
#     network_access (city-wide broadcasts are high risk),
#     privilege (overriding safety systems).

TRADING_AI_WEIGHTS = np.array([
    0.15,   # action_type_encoding  — moderate; type alone is not decisive
    0.25,   # keyword_risk_score    — strong signal; "manipulate", "override"
    0.10,   # frequency_score       — burst trading is suspicious
    0.20,   # api_call_flag         — all external trades are elevated risk
    0.05,   # file_access_flag      — low relevance for trading
    0.10,   # network_access_flag   — relevant for data exfiltration
    0.05,   # sensitive_path_flag   — low relevance for trading
    0.10,   # privilege_flag        — risk limit overrides are high risk
], dtype=np.float64)

TRADING_AI_BIAS = -0.05   # Slight negative bias — trading agents act frequently

URBAN_AI_WEIGHTS = np.array([
    0.10,   # action_type_encoding  — moderate signal
    0.15,   # keyword_risk_score    — "curfew", "shutdown", "lockdown"
    0.10,   # frequency_score       — repeated commands are suspicious
    0.10,   # api_call_flag         — external API calls matter less here
    0.10,   # file_access_flag      — config file access is concerning
    0.15,   # network_access_flag   — city-wide broadcasts are high risk
    0.20,   # sensitive_path_flag   — targeting protected systems
    0.10,   # privilege_flag        — overriding safety systems
], dtype=np.float64)

URBAN_AI_BIAS = -0.05   # Slight negative bias — routine city operations are common

# Default weights used when scenario is unknown
DEFAULT_WEIGHTS = np.array([0.125] * 8, dtype=np.float64)
DEFAULT_BIAS    = 0.0

SCENARIO_WEIGHTS: dict[Scenario, tuple[np.ndarray, float]] = {
    Scenario.TRADING_AI: (TRADING_AI_WEIGHTS, TRADING_AI_BIAS),
    Scenario.URBAN_AI:   (URBAN_AI_WEIGHTS,   URBAN_AI_BIAS),
}


# ── Scorer ────────────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    """
    Output of the risk scorer for a single feature vector.

    Attributes:
        risk_score:     Final normalised score in (0.0, 1.0).
                        0.0 = completely safe, 1.0 = critical risk.
        raw_score:      w^T x + b before sigmoid normalisation.
        weights_used:   Name of the weight set applied.
        explanation:    Human-readable breakdown for analysts.
    """
    risk_score:   float
    raw_score:    float
    weights_used: str
    explanation:  str


class RiskScorer:
    """
    Computes R(x) = sigmoid(w^T x + b) for a given feature vector.

    The sigmoid function maps any real-valued raw score into (0.0, 1.0),
    making the output safe to compare against fixed thresholds regardless
    of how large or small the individual features are.

    Usage:
        scorer = RiskScorer()
        result = scorer.score(feature_vector, scenario=Scenario.TRADING_AI)
        print(result.risk_score)   # e.g. 0.87
        print(result.explanation)
    """

    def score(
        self,
        fv: FeatureVector,
        scenario: Scenario = Scenario.UNKNOWN,
    ) -> ScoreResult:
        """
        Compute the risk score for a feature vector.

        Args:
            fv:       The 8-dimensional feature vector to score.
            scenario: The threat scenario — selects the weight set.

        Returns:
            ScoreResult with risk_score, raw_score, and explanation.

        Raises:
            ValueError: If the feature vector has wrong dimensions.
        """
        if len(fv.vector) != FeatureVector.EXPECTED_DIMENSIONS:
            raise ValueError(
                f"Expected {FeatureVector.EXPECTED_DIMENSIONS} features, "
                f"got {len(fv.vector)}"
            )

        weights, bias = SCENARIO_WEIGHTS.get(
            scenario,
            (DEFAULT_WEIGHTS, DEFAULT_BIAS),
        )
        weights_name = scenario.value

        x         = np.array(fv.vector, dtype=np.float64)
        raw_score = float(np.dot(weights, x) + bias)
        risk      = self._sigmoid(raw_score)

        explanation = self._build_explanation(x, weights, bias, raw_score, risk)

        return ScoreResult(
            risk_score=risk,
            raw_score=raw_score,
            weights_used=weights_name,
            explanation=explanation,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Apply the sigmoid function: σ(x) = 1 / (1 + e^(-x)).

        Maps any real number to (0.0, 1.0).
        Large positive x → close to 1.0 (high risk).
        Large negative x → close to 0.0 (low risk).
        """
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _build_explanation(
        x: np.ndarray,
        w: np.ndarray,
        b: float,
        raw: float,
        risk: float,
    ) -> str:
        """
        Build a human-readable explanation of the score breakdown.

        Shows the contribution of each dimension to the raw score
        so analysts can understand why a particular score was produced.
        """
        dimension_names = FeatureVector.__dataclass_fields__[
            "dimensions"
        ].default_factory()

        contributions = []
        for name, weight, value in zip(dimension_names, w, x):
            contrib = weight * value
            if contrib > 0.01:
                contributions.append(f"{name}={value:.2f}×{weight:.2f}={contrib:.3f}")

        contrib_str = ", ".join(contributions) if contributions else "no significant contributions"
        return (
            f"raw={raw:.4f}, sigmoid={risk:.4f} | "
            f"top contributors: {contrib_str} | bias={b:.2f}"
        )