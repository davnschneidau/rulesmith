"""Phased rollout system for gradual rule deployment."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from rulesmith.ab.traffic import pick_arm
from rulesmith.governance.promotion import PromotionPolicy, SLO, check_slos
from rulesmith.io.ser import ABArm


class Cohort:
    """Represents a user cohort for phased rollout."""

    def __init__(
        self,
        name: str,
        criteria: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize a cohort.

        Args:
            name: Cohort name (e.g., "internal_users", "beta_testers")
            criteria: Optional criteria dictionary for matching (e.g., {"segment": "premium"})
            description: Optional description
        """
        self.name = name
        self.criteria = criteria or {}
        self.description = description

    def matches(self, payload: Dict[str, Any]) -> bool:
        """Check if payload matches cohort criteria."""
        if not self.criteria:
            return True  # Empty criteria matches all

        for key, value in self.criteria.items():
            if key not in payload:
                return False
            if payload[key] != value:
                return False

        return True


class RolloutStage:
    """Single stage in a phased rollout."""

    def __init__(
        self,
        stage_name: str,
        percentage: float,
        cohorts: Optional[List[Cohort]] = None,
        duration_hours: Optional[int] = None,
        auto_advance_slos: Optional[List[SLO]] = None,
    ):
        """
        Initialize a rollout stage.

        Args:
            stage_name: Stage name (e.g., "canary_1", "canary_2", "full")
            percentage: Traffic percentage for this stage (0.0-1.0)
            cohorts: Optional list of cohorts to target
            duration_hours: Optional duration before auto-advancing
            auto_advance_slos: Optional SLOs that must pass to auto-advance
        """
        self.stage_name = stage_name
        self.percentage = percentage
        self.cohorts = cohorts or []
        self.duration_hours = duration_hours
        self.auto_advance_slos = auto_advance_slos or []
        self.started_at: Optional[datetime] = None

    def should_advance(self, model_uri: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """Check if stage should advance to next stage."""
        # Check duration
        if self.duration_hours and self.started_at:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds() / 3600
            if elapsed >= self.duration_hours:
                return True, f"Duration threshold reached ({self.duration_hours}h)"

        # Check SLOs
        if self.auto_advance_slos and model_uri:
            policy = PromotionPolicy(
                name="rollout_advance_policy",
                slos=self.auto_advance_slos,
                require_all=True,
            )
            slo_check = check_slos(model_uri, policy)
            if slo_check["overall_passed"]:
                return True, "SLO checks passed"

        return False, None


class RolloutPlan:
    """
    Defines a phased rollout plan for gradually deploying rules.
    
    Starts with small percentage (canary) and gradually increases
    based on SLO checks or time-based progression.
    """

    def __init__(
        self,
        name: str,
        stages: List[RolloutStage],
        model_uri: Optional[str] = None,
    ):
        """
        Initialize rollout plan.

        Args:
            name: Plan name
            stages: List of rollout stages (ordered by progression)
            model_uri: Optional model URI for SLO checking
        """
        self.name = name
        self.stages = stages
        self.model_uri = model_uri
        self.current_stage_idx = 0
        self.created_at = datetime.utcnow()

    def get_current_stage(self) -> RolloutStage:
        """Get current rollout stage."""
        return self.stages[self.current_stage_idx]

    def advance_stage(self) -> bool:
        """Advance to next stage if conditions are met."""
        current = self.get_current_stage()
        should_advance, reason = current.should_advance(self.model_uri)

        if should_advance and self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            new_stage = self.get_current_stage()
            new_stage.started_at = datetime.utcnow()
            return True

        return False

    def get_traffic_percentage(self, payload: Optional[Dict[str, Any]] = None) -> float:
        """Get current traffic percentage, considering cohorts."""
        stage = self.get_current_stage()

        # Check if payload matches any cohort
        if stage.cohorts and payload:
            for cohort in stage.cohorts:
                if cohort.matches(payload):
                    return stage.percentage

            # If payload doesn't match any cohort, return 0% (not in rollout)
            return 0.0

        return stage.percentage


class CohortRollout:
    """
    Manages phased rollout with cohort-based traffic splitting.
    
    Integrates with A/B testing system to route traffic gradually.
    """

    def __init__(self, rollout_plan: RolloutPlan, new_rulebook: Rulebook, production_rulebook: Rulebook):
        """
        Initialize cohort rollout.

        Args:
            rollout_plan: Rollout plan
            new_rulebook: New rulebook being rolled out
            production_rulebook: Current production rulebook
        """
        self.rollout_plan = rollout_plan
        self.new_rulebook = new_rulebook
        self.production_rulebook = production_rulebook

    def should_route_to_new(
        self,
        payload: Dict[str, Any],
        identity: Optional[str] = None,
    ) -> bool:
        """
        Determine if request should be routed to new rulebook.

        Args:
            payload: Request payload
            identity: Optional user identity

        Returns:
            True if should route to new rulebook
        """
        # Get current traffic percentage
        percentage = self.rollout_plan.get_traffic_percentage(payload)

        if percentage == 0.0:
            return False

        if percentage >= 1.0:
            return True

        # Use hash-based routing for deterministic allocation
        if identity:
            # Use hash to deterministically route
            import hashlib
            hash_val = int(hashlib.md5(identity.encode()).hexdigest(), 16)
            normalized = (hash_val % 10000) / 10000.0
            return normalized < percentage
        else:
            # Random routing if no identity
            import random
            return random.random() < percentage

    def execute(
        self,
        payload: Dict[str, Any],
        identity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute request with rollout routing.

        Args:
            payload: Request payload
            identity: Optional user identity

        Returns:
            Execution result (from new or production rulebook)
        """
        route_to_new = self.should_route_to_new(payload, identity)

        if route_to_new:
            result = self.new_rulebook.run(payload, enable_mlflow=True)
            result["_rollout_stage"] = self.rollout_plan.get_current_stage().stage_name
            result["_rollout_source"] = "new"
            return result
        else:
            result = self.production_rulebook.run(payload, enable_mlflow=True)
            result["_rollout_stage"] = self.rollout_plan.get_current_stage().stage_name
            result["_rollout_source"] = "production"
            return result

    def check_and_advance(self) -> bool:
        """Check if rollout should advance and advance if conditions met."""
        return self.rollout_plan.advance_stage()

