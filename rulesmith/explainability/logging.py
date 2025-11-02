"""Enhanced decision logging for full decision rationale."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from rulesmith.explainability.explainer import DecisionExplainer, ReasonCode, RuleTrace
from rulesmith.governance.audit import AuditLogger, audit_logger


class DecisionLog:
    """
    Log full decision with input, rationale, and output.
    
    Required for disputes, regulators, and compliance.
    Stores everything needed to reconstruct a decision.
    """

    def __init__(
        self,
        decision_id: str,
        rulebook_name: str,
        rulebook_version: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        explainer: Optional[DecisionExplainer] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize decision log.

        Args:
            decision_id: Unique decision ID
            rulebook_name: Rulebook name
            rulebook_version: Rulebook version
            inputs: Input payload
            outputs: Output/decision
            explainer: Optional decision explainer with traces
            metadata: Additional metadata
        """
        self.decision_id = decision_id
        self.rulebook_name = rulebook_name
        self.rulebook_version = rulebook_version
        self.inputs = inputs
        self.outputs = outputs
        self.explainer = explainer or DecisionExplainer()
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()

    def get_explanation(self, consumer_facing: bool = False) -> Dict[str, Any]:
        """Get decision explanation."""
        decision = self.outputs.get("decision", self.outputs.get("approved", "unknown"))
        return self.explainer.generate_explanation(str(decision), consumer_facing)

    def get_reason_codes(self) -> List[ReasonCode]:
        """Get all reason codes."""
        return self.explainer.get_reason_codes()

    def to_dict(self, include_rationale: bool = True) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Args:
            include_rationale: If True, include full rule traces and rationale

        Returns:
            Dictionary representation
        """
        result = {
            "decision_id": self.decision_id,
            "rulebook_name": self.rulebook_name,
            "rulebook_version": self.rulebook_version,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
        }

        if include_rationale:
            result["explanation"] = self.explainer.to_dict()
            result["reason_codes"] = [code.to_dict() for code in self.explainer.reason_codes]

        return result

    def to_consumer_explanation(self) -> Dict[str, Any]:
        """Generate consumer-facing explanation (for GDPR/compliance)."""
        explanation = self.get_explanation(consumer_facing=True)
        return {
            "decision": explanation.get("decision"),
            "consumer_message": explanation.get("consumer_message"),
            "reason_codes": [
                {
                    "code": code.code,
                    "description": code.description,
                }
                for code in self.explainer.reason_codes
            ],
        }


class DecisionLogStore:
    """
    Store and retrieve decision logs.
    
    Provides programmatic access to decision history.
    """

    def __init__(self):
        self.logs: Dict[str, DecisionLog] = {}

    def store(self, decision_log: DecisionLog) -> None:
        """Store a decision log."""
        self.logs[decision_log.decision_id] = decision_log

        # Also log to audit system
        audit_logger.log(
            action="decision_logged",
            entity_type="decision",
            entity_id=decision_log.decision_id,
            metadata={
                "rulebook": decision_log.rulebook_name,
                "version": decision_log.rulebook_version,
            },
        )

    def get(self, decision_id: str) -> Optional[DecisionLog]:
        """Get a decision log by ID."""
        return self.logs.get(decision_id)

    def query(
        self,
        rulebook_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        decision: Optional[str] = None,
    ) -> List[DecisionLog]:
        """
        Query decision logs.

        Args:
            rulebook_name: Optional rulebook filter
            start_time: Optional start time
            end_time: Optional end time
            decision: Optional decision value filter

        Returns:
            List of matching decision logs
        """
        filtered = list(self.logs.values())

        if rulebook_name:
            filtered = [log for log in filtered if log.rulebook_name == rulebook_name]

        if start_time:
            filtered = [log for log in filtered if log.timestamp >= start_time]

        if end_time:
            filtered = [log for log in filtered if log.timestamp <= end_time]

        if decision:
            filtered = [
                log
                for log in filtered
                if str(log.outputs.get("decision", log.outputs.get("approved", ""))).lower()
                == decision.lower()
            ]

        return filtered

    def get_for_user(self, user_id: str) -> List[DecisionLog]:
        """Get all decision logs for a specific user (for GDPR data portability)."""
        return [
            log
            for log in self.logs.values()
            if user_id in str(log.inputs.get("user_id", log.inputs.get("identity", "")))
        ]


# Global decision log store
decision_log_store = DecisionLogStore()

