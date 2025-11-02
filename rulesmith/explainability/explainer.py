"""Decision explanation and reason code generation."""

from typing import Any, Dict, List, Optional


class ReasonCode:
    """
    Structured reason code for explaining a decision.
    
    Used for adverse actions, consumer-facing explanations,
    and regulatory compliance (e.g., lending regulations).
    """

    def __init__(
        self,
        code: str,
        description: str,
        severity: Optional[str] = None,
        rule_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a reason code.

        Args:
            code: Reason code (e.g., "AGE_TOO_LOW", "INCOME_INSUFFICIENT")
            description: Human-readable description
            severity: Optional severity ("low", "medium", "high", "critical")
            rule_name: Optional rule that generated this reason
            metadata: Additional metadata
        """
        self.code = code
        self.description = description
        self.severity = severity
        self.rule_name = rule_name
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "description": self.description,
            "severity": self.severity,
            "rule_name": self.rule_name,
            "metadata": self.metadata,
        }

    def to_consumer_message(self) -> str:
        """Generate consumer-facing message."""
        return self.description


class RuleTrace:
    """Trace of which rules fired and their outputs."""

    def __init__(
        self,
        rule_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        fired: bool = True,
    ):
        """
        Initialize a rule trace.

        Args:
            rule_name: Rule name
            inputs: Input values to the rule
            outputs: Output values from the rule
            fired: Whether the rule fired/executed
        """
        self.rule_name = rule_name
        self.inputs = inputs
        self.outputs = outputs
        self.fired = fired

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "fired": self.fired,
        }


class DecisionExplainer:
    """
    Generate explanations for decisions.
    
    Creates rule traces, reason codes, and consumer-facing
    explanations for compliance (GDPR, lending regulations).
    """

    def __init__(self):
        self.rule_traces: List[RuleTrace] = []
        self.reason_codes: List[ReasonCode] = []

    def trace_rule(
        self,
        rule_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        fired: bool = True,
    ) -> None:
        """
        Record a rule trace.

        Args:
            rule_name: Rule name
            inputs: Rule inputs
            outputs: Rule outputs
            fired: Whether rule fired
        """
        trace = RuleTrace(rule_name, inputs, outputs, fired)
        self.rule_traces.append(trace)

    def add_reason_code(
        self,
        code: str,
        description: str,
        severity: Optional[str] = None,
        rule_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a reason code.

        Args:
            code: Reason code
            description: Description
            severity: Optional severity
            rule_name: Optional rule name
            metadata: Optional metadata
        """
        reason = ReasonCode(code, description, severity, rule_name, metadata)
        self.reason_codes.append(reason)

    def generate_explanation(
        self,
        decision: str,
        consumer_facing: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate decision explanation.

        Args:
            decision: Decision made (e.g., "approved", "rejected")
            consumer_facing: If True, generate consumer-friendly explanation

        Returns:
            Explanation dictionary
        """
        explanation = {
            "decision": decision,
            "rules_executed": [trace.to_dict() for trace in self.rule_traces],
            "reason_codes": [code.to_dict() for code in self.reason_codes],
            "rule_count": len(self.rule_traces),
            "reason_count": len(self.reason_codes),
        }

        if consumer_facing:
            explanation["consumer_message"] = self._generate_consumer_message(decision)

        return explanation

    def _generate_consumer_message(self, decision: str) -> str:
        """Generate consumer-facing explanation message."""
        if decision.lower() in ("rejected", "denied", "blocked"):
            if self.reason_codes:
                # Use top reason code
                top_reason = self.reason_codes[0]
                return top_reason.to_consumer_message()
            else:
                return "Your application did not meet our criteria."
        else:
            return f"Your application has been {decision}."

    def get_reason_codes(self, severity: Optional[str] = None) -> List[ReasonCode]:
        """Get reason codes, optionally filtered by severity."""
        if severity:
            return [code for code in self.reason_codes if code.severity == severity]
        return self.reason_codes.copy()

    def get_rule_traces(self, rule_name: Optional[str] = None) -> List[RuleTrace]:
        """Get rule traces, optionally filtered by rule name."""
        if rule_name:
            return [trace for trace in self.rule_traces if trace.rule_name == rule_name]
        return self.rule_traces.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_traces": [trace.to_dict() for trace in self.rule_traces],
            "reason_codes": [code.to_dict() for code in self.reason_codes],
        }


def explain_decision(
    decision: str,
    rule_traces: List[RuleTrace],
    reason_codes: Optional[List[ReasonCode]] = None,
    consumer_facing: bool = False,
) -> Dict[str, Any]:
    """
    Generate decision explanation from traces and reason codes.

    Args:
        decision: Decision made
        rule_traces: List of rule traces
        reason_codes: Optional list of reason codes
        consumer_facing: If True, generate consumer-friendly explanation

    Returns:
        Explanation dictionary
    """
    explainer = DecisionExplainer()
    explainer.rule_traces = rule_traces
    explainer.reason_codes = reason_codes or []
    return explainer.generate_explanation(decision, consumer_facing)

