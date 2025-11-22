"""
Clinical Rule Engine for Diabetes Risk Assessment
Implements evidence-based clinical rules that can override ML predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RuleResult:
    """Result from a clinical rule evaluation"""
    rule_name: str
    triggered: bool
    risk_level: RiskLevel
    override_score: Optional[float] = None
    explanation: str = ""
    recommendation: str = ""
    confidence: float = 1.0


class ClinicalRuleEngine:
    """
    Rule-based clinical decision support engine
    Implements evidence-based rules for diabetes risk assessment
    """
    
    def __init__(self):
        self.rules = []
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default clinical rules"""
        self.rules = [
            self._rule_critical_glucose,
            self._rule_hypoglycemia_risk,
            self._rule_hyperglycemia_risk,
            self._rule_hba1c_critical,
            self._rule_blood_pressure_crisis,
            self._rule_ketoacidosis_risk,
            self._rule_rapid_glucose_change,
            self._rule_multiple_risk_factors,
            self._rule_medication_noncompliance,
        ]
    
    def evaluate_all_rules(self, patient_data: Dict) -> List[RuleResult]:
        """
        Evaluate all clinical rules for a patient
        
        Args:
            patient_data: Dictionary containing patient features
            
        Returns:
            List of RuleResult objects
        """
        results = []
        for rule_func in self.rules:
            try:
                result = rule_func(patient_data)
                if result.triggered:
                    results.append(result)
            except Exception as e:
                # Log error but continue with other rules
                print(f"Error evaluating rule {rule_func.__name__}: {e}")
                continue
        
        # Sort by risk level (critical > high > moderate > low)
        risk_priority = {RiskLevel.CRITICAL: 4, RiskLevel.HIGH: 3, 
                        RiskLevel.MODERATE: 2, RiskLevel.LOW: 1}
        results.sort(key=lambda x: risk_priority.get(x.risk_level, 0), reverse=True)
        
        return results
    
    def get_highest_risk_override(self, patient_data: Dict) -> Optional[RuleResult]:
        """
        Get the highest risk rule that was triggered
        
        Returns:
            RuleResult with highest risk, or None if no rules triggered
        """
        results = self.evaluate_all_rules(patient_data)
        return results[0] if results else None
    
    # ========== CLINICAL RULES ==========
    
    def _rule_critical_glucose(self, data: Dict) -> RuleResult:
        """Critical glucose levels requiring immediate intervention"""
        glucose = data.get('Glucose_Level', None)
        
        if glucose is None:
            return RuleResult("critical_glucose", False, RiskLevel.LOW)
        
        if glucose < 54:  # Severe hypoglycemia
            return RuleResult(
                rule_name="critical_glucose",
                triggered=True,
                risk_level=RiskLevel.CRITICAL,
                override_score=0.95,
                explanation=f"CRITICAL: Severe hypoglycemia detected (Glucose: {glucose:.1f} mg/dL). "
                           f"Patient is at immediate risk of hypoglycemic emergency.",
                recommendation="IMMEDIATE ACTION REQUIRED: Administer fast-acting glucose (15-20g), "
                             "monitor every 15 minutes, consider glucagon if unconscious. "
                             "Contact emergency services if severe symptoms present.",
                confidence=1.0
            )
        elif glucose > 400:  # Severe hyperglycemia
            return RuleResult(
                rule_name="critical_glucose",
                triggered=True,
                risk_level=RiskLevel.CRITICAL,
                override_score=0.90,
                explanation=f"CRITICAL: Severe hyperglycemia detected (Glucose: {glucose:.1f} mg/dL). "
                           f"Risk of diabetic ketoacidosis (DKA) or hyperosmolar hyperglycemic state (HHS).",
                recommendation="IMMEDIATE ACTION REQUIRED: Check ketones, administer rapid-acting insulin "
                             "as per protocol, ensure adequate hydration, monitor closely. "
                             "Consider emergency care if ketones elevated or patient symptomatic.",
                confidence=1.0
            )
        
        return RuleResult("critical_glucose", False, RiskLevel.LOW)
    
    def _rule_hypoglycemia_risk(self, data: Dict) -> RuleResult:
        """Moderate hypoglycemia risk"""
        glucose = data.get('Glucose_Level', None)
        activity = data.get('Activity_Level', 0)
        medication = data.get('Medication_Intake', 0)
        
        if glucose is None:
            return RuleResult("hypoglycemia_risk", False, RiskLevel.LOW)
        
        if 54 <= glucose < 70:  # Moderate hypoglycemia
            risk_factors = []
            if activity > 7:
                risk_factors.append("high activity level")
            if medication > 80:
                risk_factors.append("recent medication intake")
            
            factors_text = f" Risk factors: {', '.join(risk_factors)}." if risk_factors else ""
            
            return RuleResult(
                rule_name="hypoglycemia_risk",
                triggered=True,
                risk_level=RiskLevel.HIGH,
                override_score=0.75,
                explanation=f"HIGH RISK: Hypoglycemia detected (Glucose: {glucose:.1f} mg/dL).{factors_text}",
                recommendation="Administer 15-20g fast-acting carbohydrates. Recheck glucose in 15 minutes. "
                             "If still <70 mg/dL, repeat treatment. Review insulin dosing and meal timing.",
                confidence=0.9
            )
        
        return RuleResult("hypoglycemia_risk", False, RiskLevel.LOW)
    
    def _rule_hyperglycemia_risk(self, data: Dict) -> RuleResult:
        """Moderate to severe hyperglycemia"""
        glucose = data.get('Glucose_Level', None)
        hba1c = data.get('HbA1c', None)
        
        if glucose is None:
            return RuleResult("hyperglycemia_risk", False, RiskLevel.LOW)
        
        if 250 <= glucose <= 400:
            hba1c_text = f" HbA1c: {hba1c:.1f}%." if hba1c and hba1c > 7.5 else ""
            
            return RuleResult(
                rule_name="hyperglycemia_risk",
                triggered=True,
                risk_level=RiskLevel.HIGH,
                override_score=0.70,
                explanation=f"HIGH RISK: Significant hyperglycemia (Glucose: {glucose:.1f} mg/dL).{hba1c_text}",
                recommendation="Check ketones. Administer correction dose of rapid-acting insulin per protocol. "
                             "Review recent meals, activity, and medication adherence. "
                             "Consider adjusting basal insulin if pattern persists.",
                confidence=0.85
            )
        
        return RuleResult("hyperglycemia_risk", False, RiskLevel.LOW)
    
    def _rule_hba1c_critical(self, data: Dict) -> RuleResult:
        """Critical HbA1c levels indicating poor long-term control"""
        hba1c = data.get('HbA1c', None)
        
        if hba1c is None:
            return RuleResult("hba1c_critical", False, RiskLevel.LOW)
        
        if hba1c >= 10.0:
            return RuleResult(
                rule_name="hba1c_critical",
                triggered=True,
                risk_level=RiskLevel.HIGH,
                override_score=0.80,
                explanation=f"HIGH RISK: Critically elevated HbA1c ({hba1c:.1f}%) indicates poor long-term "
                           f"glycemic control. Increased risk of complications.",
                recommendation="Urgent diabetes management review required. Consider intensifying insulin therapy, "
                             "referral to endocrinologist, comprehensive diabetes education, and evaluation "
                             "for complications. Review medication adherence and lifestyle factors.",
                confidence=0.95
            )
        elif hba1c >= 8.5:
            return RuleResult(
                rule_name="hba1c_critical",
                triggered=True,
                risk_level=RiskLevel.MODERATE,
                override_score=0.60,
                explanation=f"MODERATE RISK: Elevated HbA1c ({hba1c:.1f}%) above target (<7.0%).",
                recommendation="Review and optimize diabetes management plan. Consider medication adjustments, "
                             "lifestyle modifications, and enhanced glucose monitoring.",
                confidence=0.85
            )
        
        return RuleResult("hba1c_critical", False, RiskLevel.LOW)
    
    def _rule_blood_pressure_crisis(self, data: Dict) -> RuleResult:
        """Hypertensive crisis or hypotension"""
        sys_bp = data.get('Blood_Pressure_Systolic', None)
        dia_bp = data.get('Blood_Pressure_Diastolic', None)
        
        if sys_bp is None or dia_bp is None:
            return RuleResult("blood_pressure_crisis", False, RiskLevel.LOW)
        
        if sys_bp >= 180 or dia_bp >= 120:  # Hypertensive crisis
            return RuleResult(
                rule_name="blood_pressure_crisis",
                triggered=True,
                risk_level=RiskLevel.CRITICAL,
                override_score=0.85,
                explanation=f"CRITICAL: Hypertensive crisis (BP: {sys_bp}/{dia_bp} mmHg). "
                           f"Immediate risk of cardiovascular complications.",
                recommendation="IMMEDIATE ACTION: Assess for end-organ damage. Administer antihypertensive "
                             "medication per protocol. Monitor closely. Consider emergency care if symptomatic.",
                confidence=1.0
            )
        elif sys_bp < 90:  # Hypotension
            return RuleResult(
                rule_name="blood_pressure_crisis",
                triggered=True,
                risk_level=RiskLevel.HIGH,
                override_score=0.70,
                explanation=f"HIGH RISK: Hypotension detected (Systolic BP: {sys_bp} mmHg). "
                           f"May indicate dehydration, medication effect, or other complications.",
                recommendation="Assess for dehydration, review medications, check for signs of shock. "
                             "Consider fluid replacement if appropriate. Monitor closely.",
                confidence=0.9
            )
        
        return RuleResult("blood_pressure_crisis", False, RiskLevel.LOW)
    
    def _rule_ketoacidosis_risk(self, data: Dict) -> RuleResult:
        """Risk factors for diabetic ketoacidosis"""
        glucose = data.get('Glucose_Level', None)
        hba1c = data.get('HbA1c', None)
        stress = data.get('Stress_Level', 0)
        
        if glucose is None:
            return RuleResult("ketoacidosis_risk", False, RiskLevel.LOW)
        
        # High glucose + high stress + poor control = DKA risk
        if glucose > 250 and stress > 70 and (hba1c is None or hba1c > 8.0):
            return RuleResult(
                rule_name="ketoacidosis_risk",
                triggered=True,
                risk_level=RiskLevel.HIGH,
                override_score=0.75,
                explanation=f"HIGH RISK: Multiple DKA risk factors present (Glucose: {glucose:.1f} mg/dL, "
                           f"Stress: {stress:.0f}, HbA1c: {hba1c:.1f}% if available).",
                recommendation="URGENT: Check blood/urine ketones immediately. If ketones present, "
                             "administer rapid-acting insulin, ensure hydration, monitor closely. "
                             "Consider emergency care if ketones elevated or patient symptomatic.",
                confidence=0.8
            )
        
        return RuleResult("ketoacidosis_risk", False, RiskLevel.LOW)
    
    def _rule_rapid_glucose_change(self, data: Dict) -> RuleResult:
        """Rapid glucose changes indicating instability"""
        # This would require historical data - simplified for single point
        glucose = data.get('Glucose_Level', None)
        glucose_trend = data.get('glucose_trend', None)  # Would be calculated from history
        
        if glucose is None:
            return RuleResult("rapid_glucose_change", False, RiskLevel.LOW)
        
        # If trend data available, check for rapid changes
        if glucose_trend is not None and abs(glucose_trend) > 50:  # >50 mg/dL change
            direction = "increase" if glucose_trend > 0 else "decrease"
            return RuleResult(
                rule_name="rapid_glucose_change",
                triggered=True,
                risk_level=RiskLevel.MODERATE,
                override_score=0.65,
                explanation=f"MODERATE RISK: Rapid glucose {direction} detected ({glucose_trend:+.1f} mg/dL). "
                           f"Indicates unstable glycemic control.",
                recommendation="Review recent insulin doses, meals, and activity. Consider adjusting "
                             "insulin sensitivity factor or correction factor. Monitor closely.",
                confidence=0.75
            )
        
        return RuleResult("rapid_glucose_change", False, RiskLevel.LOW)
    
    def _rule_multiple_risk_factors(self, data: Dict) -> RuleResult:
        """Multiple moderate risk factors compounding"""
        risk_count = 0
        factors = []
        
        glucose = data.get('Glucose_Level', None)
        hba1c = data.get('HbA1c', None)
        bmi = data.get('BMI', None)
        stress = data.get('Stress_Level', 0)
        activity = data.get('Activity_Level', 0)
        
        if glucose and (180 <= glucose < 250):
            risk_count += 1
            factors.append(f"elevated glucose ({glucose:.1f} mg/dL)")
        
        if hba1c and hba1c >= 7.5:
            risk_count += 1
            factors.append(f"elevated HbA1c ({hba1c:.1f}%)")
        
        if bmi and bmi >= 30:
            risk_count += 1
            factors.append(f"obesity (BMI: {bmi:.1f})")
        
        if stress > 60:
            risk_count += 1
            factors.append(f"high stress ({stress:.0f})")
        
        if activity < 2:
            risk_count += 1
            factors.append("low activity level")
        
        if risk_count >= 3:
            return RuleResult(
                rule_name="multiple_risk_factors",
                triggered=True,
                risk_level=RiskLevel.MODERATE,
                override_score=0.55,
                explanation=f"MODERATE RISK: Multiple risk factors present ({risk_count}): "
                           f"{'; '.join(factors)}.",
                recommendation="Comprehensive diabetes management review recommended. Address modifiable "
                             "risk factors: optimize glucose control, increase physical activity, "
                             "stress management, weight management if applicable.",
                confidence=0.7
            )
        
        return RuleResult("multiple_risk_factors", False, RiskLevel.LOW)
    
    def _rule_medication_noncompliance(self, data: Dict) -> RuleResult:
        """Low medication intake indicating potential non-compliance"""
        medication = data.get('Medication_Intake', None)
        glucose = data.get('Glucose_Level', None)
        
        if medication is None:
            return RuleResult("medication_noncompliance", False, RiskLevel.LOW)
        
        # Low medication intake + high glucose suggests non-compliance
        if medication < 30 and glucose and glucose > 180:
            return RuleResult(
                rule_name="medication_noncompliance",
                triggered=True,
                risk_level=RiskLevel.MODERATE,
                override_score=0.50,
                explanation=f"MODERATE RISK: Low medication intake ({medication:.0f}%) with elevated glucose "
                           f"({glucose:.1f} mg/dL) suggests potential medication non-compliance.",
                recommendation="Review medication adherence. Provide patient education on importance of "
                             "consistent medication intake. Consider reminder systems or medication review.",
                confidence=0.65
            )
        
        return RuleResult("medication_noncompliance", False, RiskLevel.LOW)

