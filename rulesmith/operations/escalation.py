"""Escalation paths for high-risk or regulatory issues."""

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from rulesmith.governance.audit import audit_logger
from rulesmith.hitl.base import HITLQueue


class EscalationLevel(str, Enum):
    """Escalation severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    REGULATORY = "regulatory"


class EscalationRule:
    """Defines when to escalate."""

    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        level: EscalationLevel = EscalationLevel.MEDIUM,
        description: Optional[str] = None,
    ):
        """
        Initialize escalation rule.

        Args:
            name: Escalation rule name
            condition: Function that returns True if should escalate
                       Takes decision/outputs dictionary as input
            level: Escalation level
            description: Optional description
        """
        self.name = name
        self.condition = condition
        self.level = level
        self.description = description

    def should_escalate(self, outputs: Dict[str, Any]) -> bool:
        """Check if should escalate based on outputs."""
        try:
            return self.condition(outputs)
        except Exception:
            return False  # Don't escalate on condition errors


class EscalationStep:
    """Single step in escalation path."""

    def __init__(
        self,
        step_name: str,
        queue: Optional[HITLQueue] = None,
        team: Optional[str] = None,
        timeout_hours: Optional[float] = None,
        auto_escalate_on_timeout: bool = True,
    ):
        """
        Initialize escalation step.

        Args:
            step_name: Step name
            queue: Optional HITL queue for this step
            team: Optional team name (for routing)
            timeout_hours: Optional timeout before auto-escalating
            auto_escalate_on_timeout: Whether to auto-escalate on timeout
        """
        self.step_name = step_name
        self.queue = queue
        self.team = team
        self.timeout_hours = timeout_hours
        self.auto_escalate_on_timeout = auto_escalate_on_timeout


class EscalationPath:
    """Multi-level escalation workflow."""

    def __init__(
        self,
        name: str,
        steps: List[EscalationStep],
        description: Optional[str] = None,
    ):
        """
        Initialize escalation path.

        Args:
            name: Path name
            steps: List of escalation steps (ordered by level)
            description: Optional description
        """
        self.name = name
        self.steps = steps
        self.description = description

    def get_step(self, level: int) -> Optional[EscalationStep]:
        """Get step by level (0-based)."""
        if 0 <= level < len(self.steps):
            return self.steps[level]
        return None


class EscalationManager:
    """
    Manages escalation workflows.
    
    Integrates with HITL queues for automatic escalation.
    """

    def __init__(self):
        self.escalation_rules: Dict[str, EscalationRule] = {}
        self.escalation_paths: Dict[str, EscalationPath] = {}
        self.active_escalations: Dict[str, Dict[str, Any]] = {}

    def register_rule(self, rule: EscalationRule) -> None:
        """Register an escalation rule."""
        self.escalation_rules[rule.name] = rule

    def register_path(self, path: EscalationPath) -> None:
        """Register an escalation path."""
        self.escalation_paths[path.name] = path

    def check_and_escalate(
        self,
        decision_id: str,
        outputs: Dict[str, Any],
        default_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if decision should be escalated and escalate if needed.

        Args:
            decision_id: Decision ID
            outputs: Decision outputs
            default_path: Optional default escalation path name

        Returns:
            Escalation result or None if no escalation
        """
        # Check all escalation rules
        triggered_rules = []
        max_level = None

        for rule_name, rule in self.escalation_rules.items():
            if rule.should_escalate(outputs):
                triggered_rules.append(rule_name)
                # Track highest level
                if max_level is None or self._compare_levels(rule.level, max_level) > 0:
                    max_level = rule.level

        if not triggered_rules:
            return None  # No escalation needed

        # Determine escalation path
        path_name = default_path or "default"
        path = self.escalation_paths.get(path_name)

        if not path or not path.steps:
            # No path configured, just log escalation
            audit_logger.log(
                action="escalation_triggered",
                entity_type="decision",
                entity_id=decision_id,
                metadata={
                    "triggered_rules": triggered_rules,
                    "level": max_level.value if max_level else None,
                },
            )
            return {
                "escalated": True,
                "triggered_rules": triggered_rules,
                "level": max_level.value if max_level else None,
            }

        # Escalate to first step
        first_step = path.get_step(0)
        if first_step and first_step.queue:
            # Submit to HITL queue
            from rulesmith.hitl.base import ReviewRequest

            request = ReviewRequest(
                id=decision_id,
                node="escalation",
                payload={
                    "decision_id": decision_id,
                    "outputs": outputs,
                    "escalation_level": max_level.value if max_level else None,
                    "triggered_rules": triggered_rules,
                },
            )

            try:
                request_id = first_step.queue.submit(request)
                self.active_escalations[decision_id] = {
                    "path": path_name,
                    "current_level": 0,
                    "triggered_rules": triggered_rules,
                    "level": max_level.value if max_level else None,
                    "request_id": request_id,
                    "started_at": datetime.utcnow().isoformat(),
                }

                audit_logger.log(
                    action="escalation_submitted",
                    entity_type="decision",
                    entity_id=decision_id,
                    metadata={
                        "path": path_name,
                        "level": max_level.value if max_level else None,
                    },
                )

                return {
                    "escalated": True,
                    "path": path_name,
                    "level": 0,
                    "triggered_rules": triggered_rules,
                    "request_id": request_id,
                }
            except Exception as e:
                return {
                    "escalated": True,
                    "error": str(e),
                }

        return {
            "escalated": True,
            "triggered_rules": triggered_rules,
            "level": max_level.value if max_level else None,
        }

    def _compare_levels(self, level1: EscalationLevel, level2: EscalationLevel) -> int:
        """Compare escalation levels (returns >0 if level1 > level2)."""
        order = {
            EscalationLevel.LOW: 1,
            EscalationLevel.MEDIUM: 2,
            EscalationLevel.HIGH: 3,
            EscalationLevel.CRITICAL: 4,
            EscalationLevel.REGULATORY: 5,
        }
        return order.get(level1, 0) - order.get(level2, 0)

    def get_active_escalations(self) -> Dict[str, Dict[str, Any]]:
        """Get all active escalations."""
        return self.active_escalations.copy()


# Global escalation manager
escalation_manager = EscalationManager()

