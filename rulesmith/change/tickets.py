"""Change control tickets for tracking rule changes."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from rulesmith.governance.audit import audit_logger


class ChangeStatus(str, Enum):
    """Change ticket status."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class ChangeTicket:
    """
    Track rule changes with full audit trail.
    
    Records who changed what and why (required for compliance).
    """

    def __init__(
        self,
        ticket_id: str,
        change_type: str,  # "rule_add", "rule_modify", "rule_delete", "rulebook_deploy"
        entity_type: str,  # "rule", "rulebook"
        entity_name: str,
        description: str,
        changed_by: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize change ticket.

        Args:
            ticket_id: Unique ticket ID
            change_type: Type of change
            entity_type: Type of entity being changed
            entity_name: Name of entity
            description: Description of change
            changed_by: Person/team making change
            reason: Reason for change (why)
            metadata: Additional metadata
        """
        self.ticket_id = ticket_id
        self.change_type = change_type
        self.entity_type = entity_type
        self.entity_name = entity_name
        self.description = description
        self.changed_by = changed_by
        self.reason = reason
        self.metadata = metadata or {}
        self.status = ChangeStatus.DRAFT
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.approvals: List[Dict[str, Any]] = []
        self.rollback_plan: Optional["RollbackPlan"] = None

    def submit(self) -> None:
        """Submit ticket for review."""
        self.status = ChangeStatus.SUBMITTED
        self.updated_at = datetime.utcnow()

        audit_logger.log(
            action="change_ticket_submitted",
            entity_type="change_ticket",
            entity_id=self.ticket_id,
            actor=self.changed_by,
            metadata={
                "change_type": self.change_type,
                "entity_type": self.entity_type,
                "entity_name": self.entity_name,
            },
        )

    def approve(self, approver: str, comments: Optional[str] = None) -> None:
        """Approve the change."""
        self.status = ChangeStatus.APPROVED
        self.updated_at = datetime.utcnow()
        self.approvals.append({
            "approver": approver,
            "comments": comments,
            "approved_at": datetime.utcnow().isoformat(),
        })

        audit_logger.log(
            action="change_ticket_approved",
            entity_type="change_ticket",
            entity_id=self.ticket_id,
            actor=approver,
        )

    def reject(self, reviewer: str, reason: Optional[str] = None) -> None:
        """Reject the change."""
        self.status = ChangeStatus.REJECTED
        self.updated_at = datetime.utcnow()

        audit_logger.log(
            action="change_ticket_rejected",
            entity_type="change_ticket",
            entity_id=self.ticket_id,
            actor=reviewer,
            metadata={"rejection_reason": reason},
        )

    def mark_deployed(self) -> None:
        """Mark change as deployed."""
        self.status = ChangeStatus.DEPLOYED
        self.updated_at = datetime.utcnow()

        audit_logger.log(
            action="change_ticket_deployed",
            entity_type="change_ticket",
            entity_id=self.ticket_id,
            actor=self.changed_by,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticket_id": self.ticket_id,
            "change_type": self.change_type,
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "description": self.description,
            "changed_by": self.changed_by,
            "reason": self.reason,
            "status": self.status.value,
            "metadata": self.metadata,
            "approvals": self.approvals,
            "rollback_plan": self.rollback_plan.to_dict() if self.rollback_plan else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class RollbackPlan:
    """Define rollback procedures for a change."""

    def __init__(
        self,
        description: str,
        steps: List[str],
        rollback_to_version: Optional[str] = None,
        estimated_time_minutes: Optional[int] = None,
    ):
        """
        Initialize rollback plan.

        Args:
            description: Description of rollback procedure
            steps: List of rollback steps
            rollback_to_version: Optional version to roll back to
            estimated_time_minutes: Optional estimated rollback time
        """
        self.description = description
        self.steps = steps
        self.rollback_to_version = rollback_to_version
        self.estimated_time_minutes = estimated_time_minutes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "steps": self.steps,
            "rollback_to_version": self.rollback_to_version,
            "estimated_time_minutes": self.estimated_time_minutes,
        }


class ChangeApproval:
    """
    Approval workflow for changes.
    
    Supports multi-level approvals, required reviewers, etc.
    """

    def __init__(
        self,
        ticket_id: str,
        required_approvers: List[str],
        optional_approvers: Optional[List[str]] = None,
    ):
        """
        Initialize change approval.

        Args:
            ticket_id: Change ticket ID
            required_approvers: List of required approvers
            optional_approvers: Optional list of optional approvers
        """
        self.ticket_id = ticket_id
        self.required_approvers = required_approvers
        self.optional_approvers = optional_approvers or []
        self.approvals: Dict[str, Dict[str, Any]] = {}

    def approve(self, approver: str, comments: Optional[str] = None) -> bool:
        """
        Record an approval.

        Args:
            approver: Approver name
            comments: Optional comments

        Returns:
            True if all required approvals received
        """
        self.approvals[approver] = {
            "approved_at": datetime.utcnow().isoformat(),
            "comments": comments,
        }

        # Check if all required approvals received
        approved_required = all(
            approver in self.approvals for approver in self.required_approvers
        )

        return approved_required

    def is_fully_approved(self) -> bool:
        """Check if all required approvals received."""
        return all(
            approver in self.approvals for approver in self.required_approvers
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticket_id": self.ticket_id,
            "required_approvers": self.required_approvers,
            "optional_approvers": self.optional_approvers,
            "approvals": self.approvals,
            "is_fully_approved": self.is_fully_approved(),
        }


class ChangeTicketRegistry:
    """Registry for managing change tickets."""

    def __init__(self):
        self.tickets: Dict[str, ChangeTicket] = {}
        self.approvals: Dict[str, ChangeApproval] = {}

    def register(self, ticket: ChangeTicket) -> None:
        """Register a change ticket."""
        self.tickets[ticket.ticket_id] = ticket

    def get(self, ticket_id: str) -> Optional[ChangeTicket]:
        """Get a ticket by ID."""
        return self.tickets.get(ticket_id)

    def list(
        self,
        status: Optional[ChangeStatus] = None,
        entity_type: Optional[str] = None,
    ) -> List[ChangeTicket]:
        """
        List tickets, optionally filtered.

        Args:
            status: Optional status filter
            entity_type: Optional entity type filter

        Returns:
            List of matching tickets
        """
        filtered = list(self.tickets.values())

        if status:
            filtered = [t for t in filtered if t.status == status]

        if entity_type:
            filtered = [t for t in filtered if t.entity_type == entity_type]

        return filtered

    def register_approval(self, approval: ChangeApproval) -> None:
        """Register a change approval."""
        self.approvals[approval.ticket_id] = approval


# Global change ticket registry
change_ticket_registry = ChangeTicketRegistry()

