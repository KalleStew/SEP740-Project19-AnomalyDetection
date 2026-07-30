"""Data loading and preprocessing utilities for the anomaly detection pipeline.

These utilities are designed to operate on the KDD Cup 1999 network
intrusion detection dataset (Stolfo, S. J., Fan, W., Lee, W., Prodromidis,
A., & Chan, P. K., 1999; distributed via the UCI Machine Learning
Repository), but the cleaning logic itself (whitespace normalization,
missing-value handling, imputation, and encoding) is dataset-agnostic and
reusable for other tabular datasets.

Imputation (`sklearn.impute.SimpleImputer`) and categorical encoding
(`sklearn.preprocessing.OrdinalEncoder`) are implemented with scikit-learn
(Pedregosa et al., 2011).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

PathLike = Union[str, Path]


def resolve_project_root() -> Path:
    """Return the repository root directory.

    Returns:
        The absolute path to the repository root.
    """

    return Path(__file__).resolve().parents[1]


def load_data(path: PathLike, sep: str = ",", **kwargs: Any) -> pd.DataFrame:
    """Load tabular data from a delimited file.

    Args:
        path: Input file path.
        sep: Field separator used by the file.
        **kwargs: Additional arguments passed to ``pandas.read_csv``.

    Returns:
        A DataFrame containing the loaded dataset.
    """

    return pd.read_csv(Path(path), sep=sep, **kwargs)


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column labels by trimming whitespace and collapsing spaces."""

    cleaned = df.copy()
    cleaned.columns = [re.sub(r"\s+", "_", str(col).strip()) for col in cleaned.columns]
    return cleaned


def _normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common missing-value markers to ``np.nan``."""

    missing_values = ["?", "NA", "N/A", "na", "n/a", "None", ""]
    return df.replace(missing_values, np.nan)


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace from string-like columns."""

    cleaned = df.copy()
    string_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column_name in string_columns:
        cleaned[column_name] = cleaned[column_name].astype("string").str.strip()
    return cleaned


def _drop_sparse_columns(df: pd.DataFrame, missing_threshold: float) -> pd.DataFrame:
    """Remove columns whose missing-value fraction exceeds the threshold."""

    missing_ratio = df.isna().mean()
    columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    if not columns_to_drop:
        return df
    return df.drop(columns=columns_to_drop)


def _impute_numeric_columns(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Impute numeric columns using the requested strategy."""

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        return df

    cleaned = df.copy()
    imputer = SimpleImputer(strategy=strategy)
    cleaned[numeric_columns] = imputer.fit_transform(cleaned[numeric_columns])
    return cleaned


def _impute_categorical_columns(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Impute categorical columns using the requested strategy."""

    categorical_columns = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if not categorical_columns:
        return df

    cleaned = df.copy()
    imputer = SimpleImputer(strategy=strategy, fill_value="missing")
    cleaned[categorical_columns] = imputer.fit_transform(cleaned[categorical_columns])
    return cleaned


def _encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns with ordinal integer labels."""

    categorical_columns = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if not categorical_columns:
        return df

    cleaned = df.copy()
    encoder = OrdinalEncoder(dtype=int)
    cleaned[categorical_columns] = encoder.fit_transform(cleaned[categorical_columns])
    return cleaned


def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    missing_threshold: float = 0.5,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    drop_columns: Optional[Sequence[str]] = None,
    encode_categorical: bool = False,
) -> pd.DataFrame:
    """Clean a tabular dataset with deterministic, reusable preprocessing steps.

    Args:
        df: Raw input DataFrame.
        drop_duplicates: Whether to remove duplicate rows.
        missing_threshold: Maximum allowable missing-value fraction before a column is dropped.
        numeric_strategy: Imputation strategy for numeric features.
        categorical_strategy: Imputation strategy for categorical features.
        drop_columns: Optional explicit list of columns to remove.
        encode_categorical: Whether to ordinal-encode categorical features after imputation.

    Returns:
        A cleaned DataFrame ready for downstream modeling.
    """

    cleaned = _clean_column_names(df)
    cleaned = _strip_string_columns(cleaned)
    cleaned = _normalize_missing_values(cleaned)

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    if drop_columns:
        columns_to_drop = [column for column in drop_columns if column in cleaned.columns]
        if columns_to_drop:
            cleaned = cleaned.drop(columns=columns_to_drop)

    cleaned = _drop_sparse_columns(cleaned, missing_threshold=missing_threshold)
    cleaned = _impute_numeric_columns(cleaned, strategy=numeric_strategy)
    cleaned = _impute_categorical_columns(cleaned, strategy=categorical_strategy)

    if encode_categorical:
        cleaned = _encode_categorical_columns(cleaned)

    return cleaned


def save_clean_data(df: pd.DataFrame, input_path: PathLike | None = None, output_dir: PathLike | None = None) -> Path:
    """Persist cleaned data to the repository's processed-data area.

    Args:
        df: Cleaned DataFrame to save.
        input_path: Optional original input path used to derive the output filename.
        output_dir: Optional override for the output directory.

    Returns:
        The absolute path to the saved CSV file.
    """

    resolved_output_dir = Path(output_dir) if output_dir is not None else resolve_project_root() / "data" / "clean"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    if input_path is None:
        output_path = resolved_output_dir / "cleaned_data.csv"
    else:
        source_path = Path(input_path)
        output_path = resolved_output_dir / f"{source_path.stem}_cleaned.csv"

    df.to_csv(output_path, index=False)
    return output_path


@dataclass(slots=True)
class TabularDataPreprocessor:
    """Reusable preprocessing pipeline for tabular anomaly detection data."""

    drop_duplicates: bool = True
    missing_threshold: float = 0.5
    numeric_strategy: str = "median"
    categorical_strategy: str = "most_frequent"
    drop_columns: Optional[Sequence[str]] = None
    encode_categorical: bool = False
    output_dir: Path = field(default_factory=lambda: resolve_project_root() / "data" / "clean")

    def load(self, path: PathLike, sep: str = ",", **kwargs: Any) -> pd.DataFrame:
        """Load raw tabular data from disk."""

        return load_data(path=path, sep=sep, **kwargs)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured cleaning operations to a DataFrame."""

        return clean_data(
            df=df,
            drop_duplicates=self.drop_duplicates,
            missing_threshold=self.missing_threshold,
            numeric_strategy=self.numeric_strategy,
            categorical_strategy=self.categorical_strategy,
            drop_columns=self.drop_columns,
            encode_categorical=self.encode_categorical,
        )

    def fit_transform(self, path: PathLike, sep: str = ",", **kwargs: Any) -> pd.DataFrame:
        """Load data, clean it, and return the transformed DataFrame."""

        raw_df = self.load(path=path, sep=sep, **kwargs)
        return self.transform(raw_df)

    def save(self, df: pd.DataFrame, input_path: PathLike | None = None) -> Path:
        """Save a cleaned DataFrame to the configured output directory."""

        return save_clean_data(df=df, input_path=input_path, output_dir=self.output_dir)


def prepare_data(path: PathLike, sep: str = ",", **kwargs: Any) -> pd.DataFrame:
    """Load, clean, and save a dataset in a single call.

    Args:
        path: Input file path.
        sep: Field separator used by the source file.
        **kwargs: Cleaning options forwarded to ``clean_data``.

    Returns:
        The cleaned DataFrame.
    """

    preprocessor = TabularDataPreprocessor(**kwargs)
    cleaned_df = preprocessor.fit_transform(path=path, sep=sep)
    preprocessor.save(cleaned_df, input_path=path)
    return cleaned_df


if __name__ == "__main__":
    """Run a minimal preprocessing pass when executed as a script."""

    sample_path = resolve_project_root() / "data" / "raw" / "kddcup.data_10_percent_corrected"
    if sample_path.exists():
        cleaned_dataset = prepare_data(sample_path)
        print(f"Saved cleaned dataset with shape {cleaned_dataset.shape}")
    else:
        print(f"Input data not found at {sample_path}")

