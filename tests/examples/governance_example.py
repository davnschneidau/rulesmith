"""Examples demonstrating governance and promotion features."""

from rulesmith import rule, Rulebook
from rulesmith.governance.audit import AuditLogger, log_deployment, log_promotion
from rulesmith.governance.diff import diff_rulebooks
from rulesmith.governance.lineage import build_lineage
from rulesmith.governance.promotion import PromotionPolicy, SLO, promote_model


@rule(name="rule_v1", inputs=["x"], outputs=["y"])
def rule_v1(x: int) -> dict:
    """Version 1 of a rule."""
    return {"y": x * 2}


@rule(name="rule_v2", inputs=["x"], outputs=["y"])
def rule_v2(x: int) -> dict:
    """Version 2 of a rule (optimized)."""
    return {"y": x << 1}  # Bit shift (same as * 2)


def example_promotion():
    """Example model promotion with SLO checks."""
    print("=" * 60)
    print("Example: Model Promotion with SLO Checks")
    print("=" * 60)

    # Define SLOs
    slos = [
        SLO(metric_name="accuracy", threshold=0.95, operator=">="),
        SLO(metric_name="latency_ms", threshold=100, operator="<="),
        SLO(metric_name="error_rate", threshold=0.01, operator="<="),
    ]

    # Create promotion policy
    policy = PromotionPolicy(
        name="production_policy",
        slos=slos,
        require_all=True,
        min_samples=100,
    )

    print("Promotion Policy:")
    print(f"  - Name: {policy.name}")
    print(f"  - SLOs: {len(policy.slos)}")
    print(f"  - Require all: {policy.require_all}")
    print()
    print("SLOs:")
    for slo in slos:
        print(f"  - {slo.metric_name} {slo.operator} {slo.threshold}")
    print()
    print("Usage:")
    print("  result = promote_model('my_model', '@staging', '@prod', policy=policy)")
    print()


def example_diff():
    """Example rulebook diff."""
    print("=" * 60)
    print("Example: Rulebook Diff")
    print("=" * 60)

    rb1 = Rulebook(name="credit_check", version="1.0.0")
    rb1.add_rule(rule_v1, as_name="calculate")

    rb2 = Rulebook(name="credit_check", version="1.1.0")
    rb2.add_rule(rule_v2, as_name="calculate")  # Updated rule

    spec1 = rb1.to_spec()
    spec2 = rb2.to_spec()

    diff = diff_rulebooks(spec1, spec2)

    print(f"Diff: {spec1.name} v{spec1.version} -> {spec2.name} v{spec2.version}")
    print()
    print("Changes:")
    print(diff.to_string())
    print()


def example_lineage():
    """Example lineage tracking."""
    print("=" * 60)
    print("Example: Lineage Tracking")
    print("=" * 60)

    rb = Rulebook(name="credit_check", version="1.0.0")
    rb.add_rule(rule_v1, as_name="calculate")
    spec = rb.to_spec()

    lineage = build_lineage(spec)

    print("Lineage Graph:")
    print(f"  - Rulebook: {lineage.spec.name} v{lineage.spec.version}")
    print(f"  - Nodes: {len(lineage.nodes)}")
    print()
    print("Node Lineage:")
    for name, ref in lineage.nodes.items():
        print(f"  - {name}:")
        print(f"      Kind: {ref.kind}")
        if ref.code_hash:
            print(f"      Code Hash: {ref.code_hash[:16]}...")
        if ref.uri:
            print(f"      Model URI: {ref.uri}")
    print()


def example_audit_logging():
    """Example audit logging."""
    print("=" * 60)
    print("Example: Audit Logging")
    print("=" * 60)

    logger = AuditLogger()

    # Log promotion
    entry1 = log_promotion(
        logger,
        "credit_check_model",
        "@staging",
        "@prod",
        actor="alice",
        slo_results={"overall_passed": True},
    )

    # Log deployment
    rb = Rulebook(name="credit_check", version="1.0.0")
    spec = rb.to_spec()

    entry2 = log_deployment(
        logger,
        spec,
        actor="alice",
        environment="production",
    )

    print("Audit Log Entries:")
    print(f"  1. {entry1.action} - {entry1.entity_type}:{entry1.entity_id}")
    print(f"     Actor: {entry1.actor}")
    print(f"     Metadata: {entry1.metadata}")
    print()
    print(f"  2. {entry2.action} - {entry2.entity_type}:{entry2.entity_id}")
    print(f"     Actor: {entry2.actor}")
    print(f"     Spec Hash: {entry2.metadata.get('spec_hash', '')[:16]}...")
    print()
    print("Query audit log:")
    print("  entries = logger.get_entries(entity_type='model', action='promote')")
    print()


def example_complete_governance():
    """Example complete governance workflow."""
    print("=" * 60)
    print("Example: Complete Governance Workflow")
    print("=" * 60)

    logger = AuditLogger()

    # 1. Build rulebook
    rb = Rulebook(name="credit_check", version="1.0.0")
    rb.add_rule(rule_v1, as_name="calculate")
    spec = rb.to_spec()

    # 2. Compute spec hash
    from rulesmith.governance.audit import hash_rulebook_spec

    spec_hash = hash_rulebook_spec(spec)
    print(f"1. Rulebook spec hash: {spec_hash[:16]}...")

    # 3. Build lineage
    lineage = build_lineage(spec)
    print(f"2. Lineage tracked: {len(lineage.nodes)} nodes")

    # 4. Log deployment
    log_deployment(logger, spec, actor="system", environment="staging")
    print("3. Deployed to staging (audit logged)")

    # 5. Define promotion policy
    policy = PromotionPolicy(
        name="prod_policy",
        slos=[
            SLO("accuracy", 0.95),
            SLO("latency_ms", 100, operator="<="),
        ],
    )

    # 6. Promote (with SLO checks)
    # result = promote_model("credit_check_model", "@staging", "@prod", policy=policy)
    print("4. Promote to production (would check SLOs)")

    # 7. Log promotion
    log_promotion(logger, "credit_check_model", "@staging", "@prod", actor="alice")
    print("5. Promotion logged in audit trail")

    print()
    print("Complete workflow:")
    print("  ✓ Spec hashing for integrity")
    print("  ✓ Lineage tracking for provenance")
    print("  ✓ Audit logging for accountability")
    print("  ✓ SLO-gated promotion for quality")
    print()


if __name__ == "__main__":
    example_promotion()
    example_diff()
    example_lineage()
    example_audit_logging()
    example_complete_governance()

