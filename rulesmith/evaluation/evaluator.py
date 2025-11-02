"""Evaluator wrapper for MLflow.evaluate()."""

from typing import Any, Dict, List, Optional

import mlflow
from mlflow.models import EvaluationMetric


def evaluate_rulebook(
    model_uri: str,
    data: Any,
    targets: Optional[str] = None,
    evaluators: Optional[List[str]] = None,
    extra_metrics: Optional[List[EvaluationMetric]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a rulebook model using MLflow.evaluate().

    Args:
        model_uri: MLflow model URI
        data: Evaluation dataset (pandas DataFrame or path)
        targets: Optional target column name
        evaluators: Optional list of evaluator names
        extra_metrics: Optional list of custom EvaluationMetric objects

    Returns:
        Evaluation results dictionary
    """
    try:
        # Use MLflow's evaluate function
        result = mlflow.evaluate(
            model=model_uri,
            data=data,
            targets=targets,
            evaluators=evaluators,
            extra_metrics=extra_metrics,
        )

        return result
    except Exception as e:
        return {
            "error": str(e),
            "success": False,
        }


def evaluate_with_custom_scorers(
    model_uri: str,
    data: Any,
    scorers: List[Any],
    targets: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate model with custom scorers.

    Args:
        model_uri: MLflow model URI
        data: Evaluation dataset
        scorers: List of custom scorer functions
        targets: Optional target column name

    Returns:
        Evaluation results with custom scores
    """
    try:
        import pandas as pd

        # Load model
        model = mlflow.pyfunc.load_model(model_uri)

        # Load data if path
        if isinstance(data, str):
            df = pd.read_csv(data) if data.endswith(".csv") else pd.read_json(data)
        else:
            df = data

        # Run predictions
        predictions = model.predict(df)

        # Evaluate with custom scorers
        scores = {}
        for scorer in scorers:
            if hasattr(scorer, "__call__"):
                score = scorer(df, predictions, targets)
                scorer_name = getattr(scorer, "__name__", "scorer")
                scores[scorer_name] = score

        return {
            "scores": scores,
            "predictions": predictions,
            "success": True,
        }

    except Exception as e:
        return {
            "error": str(e),
            "success": False,
        }
