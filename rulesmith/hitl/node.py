"""HITL node implementation and active learning integration."""

from typing import Any, Dict, Optional
from uuid import uuid4

from rulesmith.dag.nodes import Node
from rulesmith.hitl.base import ReviewDecision, ReviewRequest


class HITLNode(Node):
    """Node that submits requests for human review and awaits decisions."""

    def __init__(
        self,
        name: str,
        queue: Any,
        timeout: Optional[float] = None,
        async_mode: bool = False,
        active_learning_threshold: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize HITL node.

        Args:
            name: Node name
            queue: HITL queue instance (HITLQueue)
            timeout: Optional timeout in seconds (None = wait indefinitely)
            async_mode: If True, don't block execution (returns pending state)
            active_learning_threshold: Optional confidence threshold for active learning
            params: Optional parameters
        """
        super().__init__(name, "hitl")
        self.queue = queue
        self.timeout = timeout
        self.async_mode = async_mode
        self.active_learning_threshold = active_learning_threshold
        self.params = params or {}

    def _should_request_review(self, state: Dict[str, Any]) -> bool:
        """
        Determine if review is needed based on active learning threshold.

        Args:
            state: Current state

        Returns:
            True if review should be requested
        """
        if self.active_learning_threshold is None:
            # Always request if no threshold
            return True

        # Check confidence score in state
        confidence = state.get("confidence", state.get("score", 1.0))
        if isinstance(confidence, dict):
            confidence = confidence.get("value", 1.0)

        # Request review if confidence below threshold
        return confidence < self.active_learning_threshold

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute HITL node - submit review request and await decision."""
        # Check if review is needed (active learning)
        if not self._should_request_review(state):
            # Skip review, use model output
            return {
                "output": state.get("output", state.get("result", {})),
                "_hitl_skipped": True,
                "_hitl_reason": "confidence_above_threshold",
            }

        # Create review request
        request_id = str(uuid4())

        # Prepare payload for review
        review_payload = {
            "node": self.name,
            "inputs": {k: v for k, v in state.items() if not k.startswith("_")},
            "model_output": state.get("output", state.get("result", {})),
        }

        # Add suggestions if available
        suggestions = self.params.get("suggestions", {})

        # Create request
        request = ReviewRequest(
            id=request_id,
            node=self.name,
            payload=review_payload,
            suggestions=suggestions,
        )

        # Submit to queue
        try:
            submitted_id = self.queue.submit(request)

            # Log HITL request if context supports it
            if hasattr(context, "on_hitl"):
                context.on_hitl(self.name, state, context, submitted_id)

            if self.async_mode:
                # Return immediately with pending state
                return {
                    "_hitl_pending": True,
                    "_hitl_request_id": submitted_id,
                    "output": state.get("output", {}),
                }

            # Wait for decision (blocking)
            decision = self.queue.get_decision(submitted_id, timeout=self.timeout)

            if decision is None:
                # Timeout - return timeout state
                return {
                    "_hitl_timeout": True,
                    "_hitl_request_id": submitted_id,
                    "output": state.get("output", {}),
                }

            # Process decision
            if decision.approved:
                # Use edited output if provided, otherwise use original
                output = decision.edited_output if decision.edited_output else state.get("output", {})
                return {
                    "output": output,
                    "_hitl_approved": True,
                    "_hitl_reviewer": decision.reviewer,
                    "_hitl_comment": decision.comment,
                }
            else:
                # Rejected - return rejection state
                return {
                    "_hitl_rejected": True,
                    "_hitl_reviewer": decision.reviewer,
                    "_hitl_comment": decision.comment,
                    "output": state.get("output", {}),
                }

        except Exception as e:
            # Error submitting or getting decision
            return {
                "_hitl_error": True,
                "_hitl_error_message": str(e),
                "output": state.get("output", {}),
            }
