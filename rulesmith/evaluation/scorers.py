"""Custom scorers for evaluation."""

from typing import Any, Dict, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def accuracy_scorer(data: Any, predictions: Any, targets: Optional[str] = None) -> float:
    """
    Compute accuracy score.

    Args:
        data: Input data
        predictions: Model predictions
        targets: Target column name

    Returns:
        Accuracy score
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for accuracy scorer")

    if targets is None:
        raise ValueError("targets must be specified for accuracy scorer")

    if isinstance(predictions, pd.DataFrame):
        pred_values = predictions.iloc[:, 0] if len(predictions.columns) > 0 else predictions.iloc[0]
    elif isinstance(predictions, (list, tuple)):
        pred_values = pd.Series(predictions)
    else:
        pred_values = pd.Series([predictions])

    if isinstance(data, pd.DataFrame):
        true_values = data[targets]
    else:
        raise ValueError("data must be pandas DataFrame")

    # Compute accuracy
    correct = (pred_values == true_values).sum()
    total = len(true_values)
    return float(correct / total) if total > 0 else 0.0


def precision_scorer(data: Any, predictions: Any, targets: Optional[str] = None, positive_label: Any = 1) -> float:
    """
    Compute precision score.

    Args:
        data: Input data
        predictions: Model predictions
        targets: Target column name
        positive_label: Positive class label

    Returns:
        Precision score
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for precision scorer")

    if targets is None:
        raise ValueError("targets must be specified for precision scorer")

    if isinstance(predictions, pd.DataFrame):
        pred_values = predictions.iloc[:, 0] if len(predictions.columns) > 0 else predictions.iloc[0]
    elif isinstance(predictions, (list, tuple)):
        pred_values = pd.Series(predictions)
    else:
        pred_values = pd.Series([predictions])

    if isinstance(data, pd.DataFrame):
        true_values = data[targets]
    else:
        raise ValueError("data must be pandas DataFrame")

    # Compute precision: TP / (TP + FP)
    true_positives = ((pred_values == positive_label) & (true_values == positive_label)).sum()
    predicted_positives = (pred_values == positive_label).sum()

    return float(true_positives / predicted_positives) if predicted_positives > 0 else 0.0


def recall_scorer(data: Any, predictions: Any, targets: Optional[str] = None, positive_label: Any = 1) -> float:
    """
    Compute recall score.

    Args:
        data: Input data
        predictions: Model predictions
        targets: Target column name
        positive_label: Positive class label

    Returns:
        Recall score
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for recall scorer")

    if targets is None:
        raise ValueError("targets must be specified for recall scorer")

    if isinstance(predictions, pd.DataFrame):
        pred_values = predictions.iloc[:, 0] if len(predictions.columns) > 0 else predictions.iloc[0]
    elif isinstance(predictions, (list, tuple)):
        pred_values = pd.Series(predictions)
    else:
        pred_values = pd.Series([predictions])

    if isinstance(data, pd.DataFrame):
        true_values = data[targets]
    else:
        raise ValueError("data must be pandas DataFrame")

    # Compute recall: TP / (TP + FN)
    true_positives = ((pred_values == positive_label) & (true_values == positive_label)).sum()
    actual_positives = (true_values == positive_label).sum()

    return float(true_positives / actual_positives) if actual_positives > 0 else 0.0


def f1_scorer(data: Any, predictions: Any, targets: Optional[str] = None, positive_label: Any = 1) -> float:
    """
    Compute F1 score.

    Args:
        data: Input data
        predictions: Model predictions
        targets: Target column name
        positive_label: Positive class label

    Returns:
        F1 score
    """
    precision = precision_scorer(data, predictions, targets, positive_label)
    recall = recall_scorer(data, predictions, targets, positive_label)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def business_kpi_scorer(
    data: Any,
    predictions: Any,
    targets: Optional[str] = None,
    kpi_function: Optional[Any] = None,
) -> float:
    """
    Compute custom business KPI score.

    Args:
        data: Input data
        predictions: Model predictions
        targets: Optional target column name
        kpi_function: Custom KPI computation function

    Returns:
        KPI score
    """
    if kpi_function:
        return float(kpi_function(data, predictions, targets))
    else:
        # Default: accuracy
        return accuracy_scorer(data, predictions, targets)


def llm_as_judge_scorer(
    data: Any,
    predictions: Any,
    targets: Optional[str] = None,
    judge_model: Optional[str] = None,
    prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use LLM as a judge for evaluation.

    Args:
        data: Input data
        predictions: Model predictions
        targets: Optional target column name
        judge_model: LLM model to use as judge
        prompt_template: Prompt template for judging

    Returns:
        Dictionary with judgment results
    """
    # Placeholder - would integrate with GenAI wrapper
    # For now, return structure
    return {
        "score": 0.0,
        "reasoning": "LLM-as-judge not yet fully implemented",
        "judge_model": judge_model,
    }
