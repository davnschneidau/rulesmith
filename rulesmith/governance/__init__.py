"""Governance modules for promotion, diff, lineage, and audit."""

from rulesmith.governance.audit import (
    AuditLogEntry,
    AuditLogger,
    audit_logger,
    hash_rulebook_spec,
    log_deployment,
    log_promotion,
)
from rulesmith.governance.diff import RulebookDiff, diff_rulebook_uris, diff_rulebooks
from rulesmith.governance.lineage import LineageGraph, build_lineage, get_model_lineage
from rulesmith.governance.promotion import (
    PromotionPolicy,
    SLO,
    check_slos,
    get_model_metrics,
    promote_model,
)

__all__ = [
    # Promotion
    "SLO",
    "PromotionPolicy",
    "promote_model",
    "check_slos",
    "get_model_metrics",
    # Diff
    "RulebookDiff",
    "diff_rulebooks",
    "diff_rulebook_uris",
    # Lineage
    "LineageGraph",
    "build_lineage",
    "get_model_lineage",
    # Audit
    "AuditLogger",
    "AuditLogEntry",
    "audit_logger",
    "hash_rulebook_spec",
    "log_promotion",
    "log_deployment",
]

