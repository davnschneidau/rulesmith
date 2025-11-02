"""Playbook system for manual review decision trees."""

from typing import Any, Callable, Dict, List, Optional


class PlaybookStep:
    """Individual step in a playbook."""

    def __init__(
        self,
        step_id: str,
        description: str,
        action: Optional[Callable] = None,
        next_steps: Optional[Dict[str, str]] = None,
        required: bool = True,
    ):
        """
        Initialize a playbook step.

        Args:
            step_id: Step identifier
            description: Step description/instructions
            action: Optional action function to execute
            next_steps: Optional mapping of outcomes to next step IDs
            required: Whether step is required
        """
        self.step_id = step_id
        self.description = description
        self.action = action
        self.next_steps = next_steps or {}
        self.required = required

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "next_steps": self.next_steps,
            "required": self.required,
        }


class Playbook:
    """
    Decision tree for manual review workflows.
    
    Guides reviewers through structured decision processes.
    """

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        start_step_id: Optional[str] = None,
    ):
        """
        Initialize a playbook.

        Args:
            name: Playbook name
            description: Optional description
            start_step_id: Optional starting step ID
        """
        self.name = name
        self.description = description
        self.steps: Dict[str, PlaybookStep] = {}
        self.start_step_id = start_step_id

    def add_step(
        self,
        step_id: str,
        description: str,
        action: Optional[Callable] = None,
        next_steps: Optional[Dict[str, str]] = None,
        required: bool = True,
    ) -> "Playbook":
        """
        Add a step to the playbook.

        Args:
            step_id: Step identifier
            description: Step description
            action: Optional action function
            next_steps: Optional outcome-to-next-step mapping
            required: Whether step is required

        Returns:
            Self for chaining
        """
        step = PlaybookStep(step_id, description, action, next_steps, required)
        self.steps[step_id] = step

        # Set as start step if first step
        if self.start_step_id is None:
            self.start_step_id = step_id

        return self

    def get_step(self, step_id: str) -> Optional[PlaybookStep]:
        """Get a step by ID."""
        return self.steps.get(step_id)

    def get_next_step(self, current_step_id: str, outcome: str) -> Optional[str]:
        """Get next step ID based on outcome."""
        step = self.steps.get(current_step_id)
        if step and step.next_steps:
            return step.next_steps.get(outcome)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "start_step_id": self.start_step_id,
            "steps": {step_id: step.to_dict() for step_id, step in self.steps.items()},
        }


class PlaybookExecutor:
    """
    Execute playbooks programmatically.
    
    Integrates with HITL system for guided reviews.
    """

    def __init__(self, playbook: Playbook):
        """
        Initialize playbook executor.

        Args:
            playbook: Playbook to execute
        """
        self.playbook = playbook
        self.current_step_id: Optional[str] = None
        self.execution_history: List[Dict[str, Any]] = []

    def start(self) -> PlaybookStep:
        """Start playbook execution."""
        if not self.playbook.start_step_id:
            raise ValueError("Playbook has no start step")

        self.current_step_id = self.playbook.start_step_id
        return self.playbook.steps[self.current_step_id]

    def execute_step(
        self,
        step_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a playbook step.

        Args:
            step_id: Step ID (uses current step if None)
            inputs: Optional inputs for step action

        Returns:
            Step execution result
        """
        step_id = step_id or self.current_step_id
        if not step_id:
            raise ValueError("No step specified and playbook not started")

        step = self.playbook.steps.get(step_id)
        if not step:
            raise ValueError(f"Step '{step_id}' not found")

        result = {
            "step_id": step_id,
            "description": step.description,
            "executed": True,
        }

        # Execute action if provided
        if step.action and inputs:
            try:
                action_result = step.action(**inputs)
                result["action_result"] = action_result
            except Exception as e:
                result["action_error"] = str(e)
                result["executed"] = False

        # Track execution
        self.execution_history.append(result)
        self.current_step_id = step_id

        # Show next steps
        if step.next_steps:
            result["next_steps"] = step.next_steps

        return result

    def advance(self, outcome: str) -> Optional[PlaybookStep]:
        """
        Advance to next step based on outcome.

        Args:
            outcome: Outcome from current step

        Returns:
            Next step or None if playbook complete
        """
        if not self.current_step_id:
            return None

        next_step_id = self.playbook.get_next_step(self.current_step_id, outcome)
        if next_step_id:
            self.current_step_id = next_step_id
            return self.playbook.steps[next_step_id]

        return None

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history.copy()

    def is_complete(self) -> bool:
        """Check if playbook execution is complete."""
        if not self.current_step_id:
            return False

        step = self.playbook.steps.get(self.current_step_id)
        if not step or not step.next_steps:
            return True  # No next steps = complete

        return False  # Has next steps = not complete

