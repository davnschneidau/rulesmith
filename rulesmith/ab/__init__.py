"""A/B testing system modules."""

from rulesmith.ab.metrics import (
    ABMetricsCollector,
    ArmMetrics,
    ComparativeMetrics,
    metrics_collector,
)
from rulesmith.ab.policies import (
    EpsilonGreedyPolicy,
    HashPolicy,
    RandomPolicy,
    ThompsonSamplingPolicy,
    TrafficPolicy,
    UCBPolicy,
)
from rulesmith.ab.lineage import (
    ABLineageTracker,
    ForkLineage,
    lineage_tracker,
)
from rulesmith.ab.reasoning import (
    ForkReason,
    explain_fork_selection,
    explain_why_not,
)
from rulesmith.ab.traffic import hash_bucket, pick_arm

__all__ = [
    # Traffic management
    "hash_bucket",
    "pick_arm",
    # Policies
    "TrafficPolicy",
    "HashPolicy",
    "RandomPolicy",
    "ThompsonSamplingPolicy",
    "UCBPolicy",
    "EpsilonGreedyPolicy",
    # Metrics
    "ABMetricsCollector",
    "ArmMetrics",
    "ComparativeMetrics",
    "metrics_collector",
    # Reasoning
    "ForkReason",
    "explain_fork_selection",
    "explain_why_not",
    # Lineage
    "ABLineageTracker",
    "ForkLineage",
    "lineage_tracker",
]

