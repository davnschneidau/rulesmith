"""Evaluation system modules."""

from rulesmith.evaluation.datasets import create_evaluation_slices, load_dataset, slice_dataset
from rulesmith.evaluation.evaluator import evaluate_rulebook, evaluate_with_custom_scorers
from rulesmith.evaluation.scorers import (
    accuracy_scorer,
    business_kpi_scorer,
    f1_scorer,
    llm_as_judge_scorer,
    precision_scorer,
    recall_scorer,
)

__all__ = [
    "evaluate_rulebook",
    "evaluate_with_custom_scorers",
    "accuracy_scorer",
    "precision_scorer",
    "recall_scorer",
    "f1_scorer",
    "business_kpi_scorer",
    "llm_as_judge_scorer",
    "load_dataset",
    "slice_dataset",
    "create_evaluation_slices",
]

