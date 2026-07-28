import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Data loading
def load_data(path: Union[str, Path], sep: str = ',', **kwargs) -> pd.DataFrame:
    """Load tabular data from a CSV file."""
    path = Path(path)
    return pd.read_csv(path, sep=sep, **kwargs)

# column name cleaning function
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by stripping whitespace and replacing spaces with underscores."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(col).strip()) for col in df.columns]
    return df

# Missing value handling functions
def _normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common missing value markers to np.nan."""
    missing_values = ["?", "NA", "N/A", "na", "n/a", "None", ""]
    return df.replace(missing_values, np.nan)

# String column cleaning functions
def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from string columns."""
    df = df.copy()
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].astype("string").str.strip()
    return df

# Main data cleaning function
def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    missing_threshold: float = 0.5,
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'most_frequent',
    drop_columns: Optional[Sequence[str]] = None,
    encode_categorical: bool = False,
) -> pd.DataFrame:
    """Perform dataset cleaning and basic imputation."""
    df = _clean_column_names(df)
    df = _strip_string_columns(df)
    df = _normalize_missing_values(df)

    if drop_duplicates:
        df = df.drop_duplicates()

    if drop_columns:
        df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy=numeric_strategy)
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=categorical_strategy, fill_value='missing')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    if encode_categorical and categorical_cols:
        encoder = OrdinalEncoder(dtype=int)
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    return df

# Combined data loading and cleaning function
def prepare_data(
    path: Union[str, Path],
    sep: str = ',',
    **kwargs,
) -> pd.DataFrame:
    """Load and clean dataset in one step."""
    df = load_data(path, sep=sep)
    df = clean_data(df, **kwargs)

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "clean"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(path)
    if input_path.is_file():
        output_path = output_dir / f"{input_path.stem}_cleaned.csv"
    else:
        output_path = output_dir / "cleaned_data.csv"
    df.to_csv(output_path, index=False)

    return df

