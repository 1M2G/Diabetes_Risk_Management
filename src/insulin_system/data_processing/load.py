"""
Data loading and validation module.

Single responsibility: load CSV and validate schema/types so that
downstream steps receive a validated DataFrame (or raise clear errors).
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config.schema import DataSchema
from ..exceptions import DataLoadError, DataValidationError

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads the insulin dosage dataset from CSV and validates structure.

    Dependency injection: schema and path are injectable for testing.
    """

    def __init__(
        self,
        schema: Optional[DataSchema] = None,
        file_path: Optional[Path] = None,
    ) -> None:
        self._schema = schema or DataSchema()
        self._file_path = file_path

    def load(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load dataset from CSV.

        Args:
            file_path: Override path for this call. If None, uses instance path.

        Returns:
            Raw DataFrame as read from CSV.

        Raises:
            DataLoadError: If file is missing or cannot be read.
        """
        path = file_path or self._file_path
        if path is None:
            raise DataLoadError("No file path provided to load.")

        path = Path(path)
        if not path.exists():
            raise DataLoadError(f"Dataset file not found: {path}")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise DataLoadError(f"Failed to read CSV from {path}: {e}") from e

        logger.info("Loaded dataset from %s, shape=%s", path, df.shape)
        return df

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate that DataFrame has required columns and non-empty.

        Raises:
            DataValidationError: If validation fails.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame.")

        if df.empty:
            raise DataValidationError("DataFrame is empty.")

        required = set(self._schema.all_columns)
        missing = required - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Missing required columns: {sorted(missing)}. "
                f"Present: {sorted(df.columns)}"
            )

        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            raise DataValidationError("Duplicate column names are not allowed.")

        logger.debug("Validation passed for DataFrame with columns %s", list(df.columns))

    def load_and_validate(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load CSV and validate schema. Convenience method.

        Returns:
            Validated raw DataFrame.
        """
        df = self.load(file_path)
        self.validate(df)
        return df


def load_and_validate(
    file_path: Path,
    schema: Optional[DataSchema] = None,
) -> pd.DataFrame:
    """
    Pure function entry point: load and validate in one call.

    Useful for tests and scripts that do not need a loader instance.
    """
    loader = DataLoader(schema=schema, file_path=file_path)
    return loader.load_and_validate(file_path)
