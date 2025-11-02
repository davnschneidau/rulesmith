"""Backtesting framework for running rules against historical data."""

from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class BacktestMetrics:
    """Metrics from a backtest run."""

    def __init__(
        self,
        true_positives: int = 0,
        false_positives: int = 0,
        true_negatives: int = 0,
        false_negatives: int = 0,
        financial_impact: Optional[float] = None,
        conversion_lift: Optional[float] = None,
        conversion_drop: Optional[float] = None,
        total_decisions: int = 0,
    ):
        """
        Initialize backtest metrics.

        Args:
            true_positives: Number of true positives
            false_positives: Number of false positives
            true_negatives: Number of true negatives
            false_negatives: Number of false negatives
            financial_impact: Estimated financial impact ($)
            conversion_lift: Conversion rate improvement
            conversion_drop: Conversion rate decrease
            total_decisions: Total number of decisions made
        """
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives
        self.financial_impact = financial_impact
        self.conversion_lift = conversion_lift
        self.conversion_drop = conversion_drop
        self.total_decisions = total_decisions

    @property
    def precision(self) -> float:
        """Calculate precision."""
        tp_fp = self.true_positives + self.false_positives
        return self.true_positives / tp_fp if tp_fp > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall."""
        tp_fn = self.true_positives + self.false_negatives
        return self.true_positives / tp_fn if tp_fn > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        if self.total_decisions == 0:
            return 0.0
        correct = self.true_positives + self.true_negatives
        return correct / self.total_decisions

    @property
    def false_positive_rate(self) -> float:
        """Calculate false positive rate."""
        fp_tn = self.false_positives + self.true_negatives
        return self.false_positives / fp_tn if fp_tn > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "false_positive_rate": self.false_positive_rate,
            "financial_impact": self.financial_impact,
            "conversion_lift": self.conversion_lift,
            "conversion_drop": self.conversion_drop,
            "total_decisions": self.total_decisions,
        }


class BacktestReport:
    """Report from a backtest run."""

    def __init__(
        self,
        rulebook_name: str,
        rulebook_version: str,
        metrics: BacktestMetrics,
        data_period: Optional[str] = None,
        sample_size: int = 0,
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize backtest report.

        Args:
            rulebook_name: Name of rulebook tested
            rulebook_version: Version of rulebook
            metrics: Backtest metrics
            data_period: Time period of historical data
            sample_size: Number of samples tested
            errors: List of errors encountered
            metadata: Additional metadata
        """
        self.rulebook_name = rulebook_name
        self.rulebook_version = rulebook_version
        self.metrics = metrics
        self.data_period = data_period
        self.sample_size = sample_size
        self.errors = errors or []
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rulebook_name": self.rulebook_name,
            "rulebook_version": self.rulebook_version,
            "metrics": self.metrics.to_dict(),
            "data_period": self.data_period,
            "sample_size": self.sample_size,
            "errors": self.errors,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Backtest Report: {self.rulebook_name} v{self.rulebook_version}",
            f"Sample Size: {self.sample_size}",
            f"Period: {self.data_period or 'N/A'}",
            "",
            "Metrics:",
            f"  Precision: {self.metrics.precision:.2%}",
            f"  Recall: {self.metrics.recall:.2%}",
            f"  Accuracy: {self.metrics.accuracy:.2%}",
            f"  False Positive Rate: {self.metrics.false_positive_rate:.2%}",
        ]
        
        if self.metrics.financial_impact is not None:
            lines.append(f"  Financial Impact: ${self.metrics.financial_impact:,.2f}")
        
        if self.metrics.conversion_lift is not None:
            lines.append(f"  Conversion Lift: {self.metrics.conversion_lift:.2%}")
        
        if self.metrics.conversion_drop is not None:
            lines.append(f"  Conversion Drop: {self.metrics.conversion_drop:.2%}")
        
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")
        
        return "\n".join(lines)


class BacktestRunner:
    """
    Run backtests against historical data.
    
    Simulates running a rulebook against historical data to estimate
    performance before deploying to production.
    """

    def __init__(self, rulebook: Any):
        """
        Initialize backtest runner.

        Args:
            rulebook: Rulebook instance to test
        """
        self.rulebook = rulebook

    def run(
        self,
        historical_data: Any,
        ground_truth_column: Optional[str] = None,
        financial_impact_column: Optional[str] = None,
        conversion_column: Optional[str] = None,
        sample_limit: Optional[int] = None,
    ) -> BacktestReport:
        """
        Run backtest against historical data.

        Args:
            historical_data: Historical dataset (pandas DataFrame or list of dicts)
            ground_truth_column: Column name with ground truth labels
            financial_impact_column: Column name with financial impact per decision
            conversion_column: Column name with conversion outcomes
            sample_limit: Optional limit on number of samples to test

        Returns:
            BacktestReport with metrics
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for backtesting")

        # Convert to DataFrame if needed
        if isinstance(historical_data, list):
            df = pd.DataFrame(historical_data)
        elif isinstance(historical_data, pd.DataFrame):
            df = historical_data.copy()
        else:
            raise ValueError("historical_data must be pandas DataFrame or list of dicts")

        # Apply sample limit
        if sample_limit and len(df) > sample_limit:
            df = df.sample(n=sample_limit, random_state=42)

        # Initialize metrics
        metrics = BacktestMetrics()
        errors = []

        # Run rulebook on each historical sample
        for idx, row in df.iterrows():
            try:
                # Convert row to dict (excluding ground truth columns)
                payload = row.to_dict()
                
                # Remove ground truth columns from payload
                if ground_truth_column and ground_truth_column in payload:
                    ground_truth = payload.pop(ground_truth_column)
                else:
                    ground_truth = None
                
                # Run rulebook
                result = self.rulebook.run(payload.copy(), enable_mlflow=False)
                
                # Determine prediction (assuming binary classification)
                # Look for common decision fields
                prediction = None
                for key in ["approved", "eligible", "blocked", "decision", "result"]:
                    if key in result:
                        prediction = bool(result[key])
                        break
                
                if prediction is None:
                    # Try to infer from result
                    prediction = result.get("output", False) if isinstance(result.get("output"), bool) else False

                # Calculate metrics if ground truth available
                if ground_truth is not None:
                    gt_bool = bool(ground_truth)
                    
                    if prediction and gt_bool:
                        metrics.true_positives += 1
                    elif prediction and not gt_bool:
                        metrics.false_positives += 1
                    elif not prediction and not gt_bool:
                        metrics.true_negatives += 1
                    else:  # not prediction and gt_bool
                        metrics.false_negatives += 1

                # Track financial impact
                if financial_impact_column and financial_impact_column in row:
                    impact = float(row[financial_impact_column])
                    if metrics.financial_impact is None:
                        metrics.financial_impact = 0.0
                    metrics.financial_impact += impact if prediction else 0.0

                metrics.total_decisions += 1

            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")

        # Calculate conversion metrics if available
        if conversion_column and conversion_column in df.columns:
            # This would require comparing conversion rates
            # For now, placeholder
            pass

        report = BacktestReport(
            rulebook_name=self.rulebook.name,
            rulebook_version=self.rulebook.version,
            metrics=metrics,
            sample_size=len(df),
            errors=errors if errors else None,
        )

        return report

