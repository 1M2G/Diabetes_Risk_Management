"""
Recommendation generator: map predictions to clinical recommendations, dosage suggestions, high-risk flags.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config.schema import RecommendationConfig

logger = logging.getLogger(__name__)

# Map insulin category (model output) to clinical recommendation text and dosage adjustment magnitude
INSULIN_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "down": {
        "summary": "Consider reducing insulin dosage.",
        "action": "Decrease",
        "magnitude": "Moderate",
        "detail": "Model suggests downward adjustment. Review recent glucose trends and hypoglycemia risk before reducing.",
    },
    "up": {
        "summary": "Consider increasing insulin dosage.",
        "action": "Increase",
        "magnitude": "Moderate",
        "detail": "Model suggests upward adjustment. Consider HbA1c and fasting glucose before increasing.",
    },
    "steady": {
        "summary": "Maintain current insulin regimen.",
        "action": "Maintain",
        "magnitude": "None",
        "detail": "No change recommended. Continue monitoring.",
    },
    "no": {
        "summary": "No insulin adjustment indicated.",
        "action": "None",
        "magnitude": "None",
        "detail": "Current regimen appears appropriate; no change suggested.",
    },
}


@dataclass
class DosageSuggestion:
    """Dosage adjustment suggestion with magnitude and confidence."""

    action: str  # Increase | Decrease | Maintain | None
    magnitude: str  # None | Small | Moderate | Large
    confidence: float
    summary: str
    detail: str


@dataclass
class ClinicalRecommendation:
    """Full clinical recommendation for one prediction."""

    predicted_class: str
    confidence: float
    uncertainty_entropy: float
    dosage_suggestion: DosageSuggestion
    is_high_risk: bool
    high_risk_reason: Optional[str] = None
    probability_breakdown: Optional[Dict[str, float]] = None


def _magnitude_from_confidence(confidence: float) -> str:
    """Map confidence to suggestion magnitude (for display)."""
    if confidence >= 0.8:
        return "Large"
    if confidence >= 0.6:
        return "Moderate"
    if confidence >= 0.4:
        return "Small"
    return "None"


class RecommendationGenerator:
    """Maps model predictions to clinical recommendations with dosage suggestions and risk flags."""

    def __init__(self, config: Optional[RecommendationConfig] = None):
        self._cfg = config or RecommendationConfig()
        self._rec_map = INSULIN_RECOMMENDATIONS

    def is_high_risk(self, confidence: float, entropy: float) -> Tuple[bool, Optional[str]]:
        """Flag for clinician review: low confidence or high uncertainty."""
        reasons = []
        if confidence < self._cfg.confidence_threshold:
            reasons.append(f"Low confidence ({confidence:.0%} < {self._cfg.confidence_threshold:.0%})")
        if entropy > self._cfg.uncertainty_entropy_threshold:
            reasons.append(f"High uncertainty (entropy {entropy:.2f})")
        if reasons:
            return True, "; ".join(reasons)
        return False, None

    def generate(
        self,
        predicted_class: str,
        confidence: float,
        uncertainty_entropy: float,
        probability_breakdown: Optional[Dict[str, float]] = None,
    ) -> ClinicalRecommendation:
        """
        Build clinical recommendation from prediction and confidence/uncertainty.
        probability_breakdown: optional dict class_name -> prob for display.
        """
        rec = self._rec_map.get(str(predicted_class).lower(), self._rec_map.get("steady", {}))
        if not rec:
            rec = {"summary": "Review model output.", "action": "Maintain", "magnitude": "None", "detail": "Manual review recommended."}

        magnitude = rec.get("magnitude", "Moderate")
        if magnitude == "Moderate":
            magnitude = _magnitude_from_confidence(confidence)

        dosage = DosageSuggestion(
            action=rec.get("action", "Maintain"),
            magnitude=magnitude,
            confidence=confidence,
            summary=rec.get("summary", ""),
            detail=rec.get("detail", ""),
        )
        is_risk, reason = self.is_high_risk(confidence, uncertainty_entropy)
        return ClinicalRecommendation(
            predicted_class=predicted_class,
            confidence=confidence,
            uncertainty_entropy=uncertainty_entropy,
            dosage_suggestion=dosage,
            is_high_risk=is_risk,
            high_risk_reason=reason,
            probability_breakdown=probability_breakdown,
        )


