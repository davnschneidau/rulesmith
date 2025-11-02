"""Examples demonstrating guardrails system."""

from rulesmith import rule, Rulebook
from rulesmith.guardrails.packs import (
    HALLUCINATION_PACK,
    OUTPUT_VALIDATION_PACK,
    PII_PACK,
    TOXICITY_PACK,
)
from rulesmith.guardrails.policy import GuardAction, GuardPolicy


@rule(name="generate_text", inputs=["prompt"], outputs=["output"])
def generate_text(prompt: str) -> dict:
    """Generate text output."""
    return {"output": f"Generated: {prompt}"}


@rule(name="process_user_input", inputs=["user_text"], outputs=["processed"])
def process_user_input(user_text: str) -> dict:
    """Process user input."""
    return {"processed": user_text.upper()}


def example_pii_guards():
    """Example with PII detection guards."""
    print("=" * 60)
    print("Example: PII Detection Guards")
    print("=" * 60)

    rb = Rulebook(name="pii_guard_example", version="1.0.0")
    rb.add_rule(process_user_input, as_name="process")

    # Attach PII guard pack
    rb.attach_guard("process", PII_PACK)

    print("PII Guard Pack:")
    print("  - Detects: emails, phone numbers, SSNs")
    print("  - Action: BLOCK (default)")
    print()
    print("Usage:")
    print("  rb.attach_guard('node_name', PII_PACK)")
    print()


def example_toxicity_guards():
    """Example with toxicity detection guards."""
    print("=" * 60)
    print("Example: Toxicity Detection Guards")
    print("=" * 60)

    rb = Rulebook(name="toxicity_guard_example", version="1.0.0")
    rb.add_rule(generate_text, as_name="generate")

    # Attach toxicity guard pack
    rb.attach_guard("generate", TOXICITY_PACK)

    print("Toxicity Guard Pack:")
    print("  - Detects: toxic keywords")
    print("  - Action: BLOCK")
    print()
    print("Note: Basic implementation uses keyword matching.")
    print("For production, use a proper toxicity detection model.")
    print()


def example_hallucination_guards():
    """Example with hallucination detection guards."""
    print("=" * 60)
    print("Example: Hallucination Detection Guards")
    print("=" * 60)

    rb = Rulebook(name="hallucination_guard_example", version="1.0.0")
    rb.add_rule(generate_text, as_name="generate")

    # Attach hallucination guard pack
    rb.attach_guard("generate", HALLUCINATION_PACK)

    print("Hallucination Guard Pack:")
    print("  - Checks: citations, confidence markers")
    print("  - Action: FLAG (warns but allows)")
    print()
    print("Features:")
    print("  - Detects missing citations")
    print("  - Flags high-confidence claims without evidence")
    print()


def example_output_validation():
    """Example with output validation guards."""
    print("=" * 60)
    print("Example: Output Validation Guards")
    print("=" * 60)

    rb = Rulebook(name="output_validation_example", version="1.0.0")
    rb.add_rule(generate_text, as_name="generate")

    # Attach output validation pack
    rb.attach_guard("generate", OUTPUT_VALIDATION_PACK)

    print("Output Validation Guard Pack:")
    print("  - Checks: length, format")
    print("  - Action: FLAG")
    print()
    print("Features:")
    print("  - Validates output length (min/max)")
    print("  - Validates format (JSON, text, etc.)")
    print()


def example_custom_guard_policy():
    """Example with custom guard policy."""
    print("=" * 60)
    print("Example: Custom Guard Policy")
    print("=" * 60)

    from rulesmith.guardrails.policy import GuardPolicy, GuardAction

    # Create custom policy
    policy = GuardPolicy(
        name="custom_policy",
        checks=["pii_email", "toxicity_basic"],
        on_fail=GuardAction.FLAG,  # Flag instead of block
    )

    rb = Rulebook(name="custom_guard_example", version="1.0.0")
    rb.add_rule(process_user_input, as_name="process")
    rb.attach_guard("process", policy)

    print("Custom Guard Policy:")
    print("  - Checks: pii_email, toxicity_basic")
    print("  - Action: FLAG (on failure)")
    print()
    print("Actions:")
    print("  - BLOCK: Block execution entirely")
    print("  - FLAG: Add warning but continue")
    print("  - OVERRIDE: Replace output with safe template")
    print("  - ALLOW: Allow through (default)")
    print()


def example_multiple_guards():
    """Example with multiple guard policies."""
    print("=" * 60)
    print("Example: Multiple Guard Policies")
    print("=" * 60)

    rb = Rulebook(name="multi_guard_example", version="1.0.0")
    rb.add_rule(generate_text, as_name="generate")

    # Attach multiple guard packs
    rb.attach_guard("generate", PII_PACK)  # Block PII
    rb.attach_guard("generate", TOXICITY_PACK)  # Block toxicity
    rb.attach_guard("generate", HALLUCINATION_PACK)  # Flag hallucinations

    print("Multiple Guards on Same Node:")
    print("  - PII guards: BLOCK")
    print("  - Toxicity guards: BLOCK")
    print("  - Hallucination guards: FLAG")
    print()
    print("Execution order:")
    print("  1. All guards evaluated")
    print("  2. Blocking guards stop execution if failed")
    print("  3. Flagging guards add warnings but continue")
    print()


def example_guard_with_override():
    """Example with guard override action."""
    print("=" * 60)
    print("Example: Guard Override Action")
    print("=" * 60)

    policy = GuardPolicy(
        name="override_policy",
        checks=["toxicity_basic"],
        on_fail=GuardAction.OVERRIDE,
        override_template={"output": "[Content filtered]"},
    )

    rb = Rulebook(name="override_example", version="1.0.0")
    rb.add_rule(generate_text, as_name="generate")
    rb.attach_guard("generate", policy)

    print("Guard Override Policy:")
    print("  - On failure: Replace output with safe template")
    print("  - Template: {'output': '[Content filtered]'}")
    print()
    print("Use cases:")
    print("  - Filter sensitive content")
    print("  - Replace with sanitized version")
    print("  - Maintain execution flow")
    print()


if __name__ == "__main__":
    example_pii_guards()
    example_toxicity_guards()
    example_hallucination_guards()
    example_output_validation()
    example_custom_guard_policy()
    example_multiple_guards()
    example_guard_with_override()

