"""Governance modules for promotion, diff, lineage, audit, signing, PII, review, and policy-as-code."""

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
from rulesmith.governance.pii_minimization import (
    PIIField,
    PIIMinimizer,
    PrivacyBudget,
    pii_minimizer,
)
from rulesmith.governance.policy_as_code import (
    CedarPolicyEngine,
    OPAPolicyEngine,
    PolicyEngine,
    PolicyNode,
    cedar_engine,
    opa_engine,
)
from rulesmith.governance.promotion import (
    PromotionPolicy,
    SLO,
    check_slos,
    get_model_metrics,
    promote_model,
)
from rulesmith.governance.review import (
    Review,
    ReviewManager,
    ReviewRequest,
    ReviewStatus,
    review_manager,
)
from rulesmith.governance.signing import (
    Attestation,
    AttestationStore,
    RulebookSigner,
    Signature,
    attestation_store,
    rulebook_signer,
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
    # Phase 7: Signing & Attestation
    "RulebookSigner",
    "Signature",
    "Attestation",
    "AttestationStore",
    "rulebook_signer",
    "attestation_store",
    # Phase 7: PII Minimization
    "PIIField",
    "PIIMinimizer",
    "PrivacyBudget",
    "pii_minimizer",
    # Phase 7: Two-Person Review
    "ReviewManager",
    "ReviewRequest",
    "Review",
    "ReviewStatus",
    "review_manager",
    # Phase 7: Policy-as-Code
    "PolicyEngine",
    "OPAPolicyEngine",
    "CedarPolicyEngine",
    "PolicyNode",
    "opa_engine",
    "cedar_engine",
]

