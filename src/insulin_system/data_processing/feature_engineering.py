"""
Feature Engineering Module: orchestrates base derived, interactions,
polynomial, aggregate, and temporal features.
"""

import logging
from typing import List, Optional

import pandas as pd

from ..config.schema import DataSchema, FeatureEngineeringConfig
from ..exceptions import DataValidationError
from .interaction_features import InteractionFeatureCreator
from .polynomial_features import PolynomialFeatureCreator
from .aggregate_features import AggregateFeatureCreator
from .temporal_features import TemporalFeatureCreator

logger = logging.getLogger(__name__)

# Exported for pipeline: columns that need encoding (categorical) or scaling (numeric)
DERIVED_CATEGORICAL = ["bmi_category", "glucose_risk", "hba1c_control"]
DERIVED_NUMERIC_BASE = ["bmi_glucose_interaction", "weight_bmi_ratio"]


def _derived_numeric_from_config(config: FeatureEngineeringConfig) -> List[str]:
    """All derived numeric column names from config (interactions, polynomial, aggregate, temporal)."""
    out = list(DERIVED_NUMERIC_BASE)
    for a, b in config.INTERACTION_PAIRS:
        out.append(f"{a}_{b}_interaction")
    for c in config.POLYNOMIAL_COLUMNS:
        out.append(f"{c}_poly2")
    for agg_name, _ in config.AGGREGATE_WEIGHTS:
        out.append(agg_name)
    if config.TEMPORAL_ORDER_COLUMN:
        out.extend(["temporal_rank", "temporal_segment"])
    return out


class FeatureEngineer:
    """
    Orchestrates all feature engineering steps in order:
    base derived -> interaction -> polynomial -> aggregate -> temporal.
    """

    def __init__(
        self,
        schema: Optional[DataSchema] = None,
        config: Optional[FeatureEngineeringConfig] = None,
    ) -> None:
        self._schema = schema or DataSchema()
        self._config = config or FeatureEngineeringConfig()
        self._interaction = InteractionFeatureCreator(schema=self._schema, config=self._config)
        self._polynomial = PolynomialFeatureCreator(schema=self._schema, config=self._config)
        self._aggregate = AggregateFeatureCreator(schema=self._schema, config=self._config)
        self._temporal = TemporalFeatureCreator(schema=self._schema, config=self._config)

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        for col in ("BMI", "glucose_level", "HbA1c", "weight"):
            if col not in df.columns:
                raise DataValidationError("FeatureEngineering requires column: " + col)
        self._interaction.fit(df)
        self._polynomial.fit(df)
        self._aggregate.fit(df)
        self._temporal.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = self._add_base_derived(out)
        out = self._interaction.transform(out)
        out = self._polynomial.transform(out)
        out = self._aggregate.transform(out)
        out = self._temporal.transform(out)
        return out

    def _add_base_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "BMI" in out.columns:
            out["bmi_category"] = pd.cut(
                out["BMI"],
                bins=[0, 18.5, 25, 30, 100],
                labels=["under", "normal", "over", "obese"],
            ).astype(str)
        if "glucose_level" in out.columns:
            out["glucose_risk"] = pd.cut(
                out["glucose_level"],
                bins=[0, 70, 100, 125, 200, 1000],
                labels=["low", "normal", "prediabetic", "elevated", "high"],
            ).astype(str)
        if "HbA1c" in out.columns:
            out["hba1c_control"] = (out["HbA1c"] <= 7.0).map(
                {True: "controlled", False: "uncontrolled"}
            )
        if "BMI" in out.columns and "glucose_level" in out.columns:
            out["bmi_glucose_interaction"] = (out["BMI"] / 30.0) * (out["glucose_level"] / 100.0)
        if "weight" in out.columns and "BMI" in out.columns:
            out["weight_bmi_ratio"] = out["weight"] / (out["BMI"] + 1e-5)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @staticmethod
    def derived_numeric_columns(config: Optional[FeatureEngineeringConfig] = None) -> List[str]:
        """Return list of all derived numeric feature names (for encoder/scaler config)."""
        cfg = config or FeatureEngineeringConfig()
        return _derived_numeric_from_config(cfg)
