"""Model risk metrics tracking (PSI, drift, calibration, ROC/AUC)."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class ModelRiskMetric:
    """
    Track model health and risk metrics.
    
    Metrics: ROC/AUC, calibration, Population Stability Index (PSI),
    model drift detection, performance degradation.
    """

    def __init__(
        self,
        metric_name: str,
        value: float,
        model_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize model risk metric.

        Args:
            metric_name: Metric name (e.g., "roc_auc", "psi", "calibration_error")
            value: Metric value
            model_name: Optional model name
            timestamp: Optional timestamp
            metadata: Additional metadata
        """
        self.metric_name = metric_name
        self.value = value
        self.model_name = model_name
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


def calculate_psi(expected: List[float], actual: List[float], bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures how much a distribution has shifted over time.
    PSI < 0.1: No significant shift
    PSI 0.1-0.25: Some shift (monitor)
    PSI > 0.25: Significant shift (investigate)

    Args:
        expected: Expected distribution (baseline)
        actual: Actual distribution (current)
        bins: Number of bins for histogram

    Returns:
        PSI value
    """
    if not HAS_NUMPY:
        # Fallback calculation without numpy
        # Simple histogram-based PSI
        def create_histogram(values, bins):
            min_val, max_val = min(values), max(values)
            bin_width = (max_val - min_val) / bins if max_val > min_val else 1.0
            hist = [0] * bins
            
            for val in values:
                bin_idx = min(int((val - min_val) / bin_width), bins - 1) if bin_width > 0 else 0
                hist[bin_idx] += 1
            
            # Normalize
            total = sum(hist)
            return [h / total if total > 0 else 0.0 for h in hist]

        expected_hist = create_histogram(expected, bins)
        actual_hist = create_histogram(actual, bins)
        
        psi = 0.0
        for i in range(bins):
            if expected_hist[i] > 0:
                ratio = actual_hist[i] / expected_hist[i] if expected_hist[i] > 0 else 1.0
                if ratio > 0:
                    psi += (actual_hist[i] - expected_hist[i]) * np.log(ratio) if HAS_NUMPY else (actual_hist[i] - expected_hist[i]) * (np.log(ratio) if ratio > 0 else 0)
        
        return psi

    # Use numpy for efficient calculation
    expected_array = np.array(expected)
    actual_array = np.array(actual)

    # Create bins based on expected distribution
    _, bin_edges = np.histogram(expected_array, bins=bins)
    
    # Calculate histograms
    expected_hist, _ = np.histogram(expected_array, bins=bin_edges)
    actual_hist, _ = np.histogram(actual_array, bins=bin_edges)
    
    # Normalize to probabilities
    expected_probs = expected_hist / expected_hist.sum() if expected_hist.sum() > 0 else expected_hist
    actual_probs = actual_hist / actual_hist.sum() if actual_hist.sum() > 0 else actual_hist
    
    # Calculate PSI
    psi = 0.0
    for i in range(len(expected_probs)):
        if expected_probs[i] > 0:
            ratio = actual_probs[i] / expected_probs[i]
            if ratio > 0:
                psi += (actual_probs[i] - expected_probs[i]) * np.log(ratio)
    
    return float(psi)


def calculate_roc_auc(
    y_true: List[bool],
    y_score: List[float],
) -> float:
    """
    Calculate ROC-AUC score.
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores/probabilities

    Returns:
        ROC-AUC value (0.0-1.0)
    """
    if not HAS_NUMPY:
        # Fallback calculation
        # Simple implementation without sklearn
        # Sort by score descending
        sorted_pairs = sorted(zip(y_score, y_true), reverse=True)
        
        # Count positives and negatives
        positives = sum(1 for _, label in sorted_pairs if label)
        negatives = len(sorted_pairs) - positives
        
        if positives == 0 or negatives == 0:
            return 0.5  # No discrimination possible
        
        # Calculate AUC using trapezoidal rule
        auc = 0.0
        tp = 0  # True positives
        fp = 0  # False positives
        
        for _, label in sorted_pairs:
            if label:
                tp += 1
            else:
                fp += 1
                auc += tp  # Area under curve
        
        return auc / (positives * negatives) if (positives * negatives) > 0 else 0.0

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        # Fallback if sklearn not available
        return calculate_roc_auc(y_true, y_score)  # Use our implementation


def calculate_calibration_error(
    y_true: List[bool],
    y_pred_proba: List[float],
    bins: int = 10,
) -> float:
    """
    Calculate calibration error (ECE - Expected Calibration Error).
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        bins: Number of bins

    Returns:
        Calibration error (lower is better, 0.0 = perfectly calibrated)
    """
    if not HAS_NUMPY:
        # Fallback: simple bin-based calculation
        bin_counts = [0] * bins
        bin_correct = [0] * bins
        
        for true_label, proba in zip(y_true, y_pred_proba):
            bin_idx = min(int(proba * bins), bins - 1)
            bin_counts[bin_idx] += 1
            if true_label:
                bin_correct[bin_idx] += 1
        
        ece = 0.0
        total = len(y_true)
        for i in range(bins):
            if bin_counts[i] > 0:
                accuracy = bin_correct[i] / bin_counts[i]
                prob_avg = (i + 0.5) / bins
                ece += (bin_counts[i] / total) * abs(accuracy - prob_avg)
        
        return ece

    # Use numpy for efficient calculation
    y_true_array = np.array(y_true, dtype=float)
    y_pred_proba_array = np.array(y_pred_proba)
    
    # Bin probabilities
    bin_indices = np.digitize(y_pred_proba_array, np.linspace(0, 1, bins + 1)) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    
    # Calculate calibration error
    ece = 0.0
    total = len(y_true)
    
    for i in range(bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = y_true_array[mask].mean()
            bin_prob = y_pred_proba_array[mask].mean()
            bin_weight = mask.sum() / total
            ece += bin_weight * abs(bin_accuracy - bin_prob)
    
    return float(ece)


class ModelRiskMetricsTracker:
    """
    Track and aggregate model risk metrics over time.
    """

    def __init__(self):
        self.metrics: List[ModelRiskMetric] = []
        self.baseline_distributions: Dict[str, List[float]] = {}

    def record(
        self,
        metric_name: str,
        value: float,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a model risk metric.

        Args:
            metric_name: Metric name
            value: Metric value
            model_name: Optional model name
            metadata: Optional metadata
        """
        metric = ModelRiskMetric(metric_name, value, model_name, metadata=metadata)
        self.metrics.append(metric)

    def record_roc_auc(
        self,
        roc_auc: float,
        model_name: Optional[str] = None,
    ) -> None:
        """Record ROC-AUC score."""
        metadata = {}
        self.record("roc_auc", roc_auc, model_name, metadata)

    def record_psi(
        self,
        psi: float,
        model_name: Optional[str] = None,
        feature_name: Optional[str] = None,
    ) -> None:
        """Record Population Stability Index."""
        metadata = {}
        if feature_name:
            metadata["feature"] = feature_name
        self.record("psi", psi, model_name, metadata)

    def record_calibration_error(
        self,
        calibration_error: float,
        model_name: Optional[str] = None,
    ) -> None:
        """Record calibration error."""
        metadata = {}
        self.record("calibration_error", calibration_error, model_name, metadata)

    def detect_drift(
        self,
        feature_name: str,
        current_distribution: List[float],
        model_name: Optional[str] = None,
        bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Detect model drift by comparing distributions.

        Args:
            feature_name: Feature name
            current_distribution: Current feature distribution
            model_name: Optional model name
            bins: Number of bins for PSI calculation

        Returns:
            Drift detection result with PSI value
        """
        baseline_key = f"{model_name}:{feature_name}" if model_name else feature_name
        baseline = self.baseline_distributions.get(baseline_key)

        if not baseline:
            # Store as baseline
            self.baseline_distributions[baseline_key] = current_distribution.copy()
            return {
                "drift_detected": False,
                "psi": 0.0,
                "message": "Baseline established",
            }

        # Calculate PSI
        psi = calculate_psi(baseline, current_distribution, bins)

        # Determine drift severity
        if psi < 0.1:
            severity = "none"
            drift_detected = False
        elif psi < 0.25:
            severity = "moderate"
            drift_detected = True
        else:
            severity = "severe"
            drift_detected = True

        # Record PSI
        self.record_psi(psi, model_name, feature_name)

        return {
            "drift_detected": drift_detected,
            "psi": psi,
            "severity": severity,
            "message": f"PSI={psi:.3f}: {severity} drift detected" if drift_detected else f"PSI={psi:.3f}: No significant drift",
        }

    def set_baseline(
        self,
        feature_name: str,
        baseline_distribution: List[float],
        model_name: Optional[str] = None,
    ) -> None:
        """Set baseline distribution for drift detection."""
        baseline_key = f"{model_name}:{feature_name}" if model_name else feature_name
        self.baseline_distributions[baseline_key] = baseline_distribution.copy()

    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        model_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ModelRiskMetric]:
        """
        Get metrics, optionally filtered.

        Args:
            metric_name: Optional metric name filter
            model_name: Optional model name filter
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            List of matching metrics
        """
        filtered = self.metrics.copy()

        if metric_name:
            filtered = [m for m in filtered if m.metric_name == metric_name]

        if model_name:
            filtered = [m for m in filtered if m.model_name == model_name]

        if start_time:
            filtered = [m for m in filtered if m.timestamp >= start_time]

        if end_time:
            filtered = [m for m in filtered if m.timestamp <= end_time]

        return filtered


# Global model risk metrics tracker
model_risk_metrics_tracker = ModelRiskMetricsTracker()

