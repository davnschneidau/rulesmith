"""Dataset loading and slicing utilities."""

from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_dataset(path: str, format: Optional[str] = None) -> Any:
    """
    Load dataset from file.

    Args:
        path: File path
        format: Optional format (auto-detected from extension if None)

    Returns:
        Dataset (pandas DataFrame)
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for dataset loading")

    if format is None:
        if path.endswith(".csv"):
            format = "csv"
        elif path.endswith((".json", ".jsonl")):
            format = "json"
        elif path.endswith(".parquet"):
            format = "parquet"
        else:
            raise ValueError(f"Unknown file format: {path}")

    if format == "csv":
        return pd.read_csv(path)
    elif format == "json":
        return pd.read_json(path)
    elif format == "jsonl":
        return pd.read_json(path, lines=True)
    elif format == "parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def slice_dataset(
    data: Any,
    filters: Dict[str, Any],
) -> Any:
    """
    Slice dataset based on filters.

    Args:
        data: Dataset (pandas DataFrame)
        filters: Dictionary of column: value filters

    Returns:
        Sliced dataset
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for dataset slicing")

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be pandas DataFrame")

    sliced = data.copy()

    for column, value in filters.items():
        if column in sliced.columns:
            if isinstance(value, (list, tuple)):
                sliced = sliced[sliced[column].isin(value)]
            else:
                sliced = sliced[sliced[column] == value]

    return sliced


def create_evaluation_slices(
    data: Any,
    slice_columns: List[str],
) -> Dict[str, Any]:
    """
    Create evaluation slices for fairness/disparate impact analysis.

    Args:
        data: Dataset
        slice_columns: Columns to slice by

    Returns:
        Dictionary mapping slice names to sliced datasets
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for evaluation slicing")

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be pandas DataFrame")

    slices = {}

    for column in slice_columns:
        if column in data.columns:
            unique_values = data[column].unique()
            for value in unique_values:
                slice_name = f"{column}={value}"
                slices[slice_name] = data[data[column] == value]

    return slices
