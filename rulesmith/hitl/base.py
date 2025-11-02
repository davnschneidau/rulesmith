"""HITL base classes and interfaces."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional


class ReviewRequest:
    """Human review request."""

    def __init__(
        self,
        id: str,
        node: str,
        payload: Dict[str, Any],
        suggestions: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ):
        self.id = id
        self.node = node
        self.payload = payload
        self.suggestions = suggestions or {}
        self.expires_at = expires_at


class ReviewDecision:
    """Human review decision."""

    def __init__(
        self,
        id: str,
        approved: bool,
        edited_output: Optional[Dict[str, Any]] = None,
        comment: Optional[str] = None,
        reviewer: Optional[str] = None,
    ):
        self.id = id
        self.approved = approved
        self.edited_output = edited_output or {}
        self.comment = comment
        self.reviewer = reviewer


class HITLQueue(ABC):
    """Abstract interface for HITL queues."""

    @abstractmethod
    def submit(self, request: ReviewRequest) -> str:
        """Submit a review request."""
        pass

    @abstractmethod
    def get_decision(self, request_id: str, timeout: Optional[float] = None) -> Optional[ReviewDecision]:
        """Get decision for a request."""
        pass

