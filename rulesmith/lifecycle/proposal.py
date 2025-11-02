"""Rule proposal system for documenting and reviewing new rules."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from rulesmith.governance.audit import AuditLogger, audit_logger


class ProposalStatus(str, Enum):
    """Proposal review status."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"


class RuleProposal:
    """
    Document a proposed rule change with rationale and expected impact.
    
    This class helps teams document the "why" behind rule changes before
    they go into production.
    """

    def __init__(
        self,
        rule_name: str,
        rulebook_name: str,
        rationale: str,
        expected_lift: Optional[float] = None,
        expected_false_positive_impact: Optional[str] = None,
        business_justification: Optional[str] = None,
        proposed_by: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a rule proposal.

        Args:
            rule_name: Name of the rule being proposed
            rulebook_name: Name of the rulebook
            rationale: Rationale for the rule (why is it needed?)
            expected_lift: Expected improvement in key metric (e.g., 0.15 for 15% lift)
            expected_false_positive_impact: Description of expected FP impact
            business_justification: Business case for the rule
            proposed_by: Person/team proposing the rule
            tags: Optional tags for categorization
            metadata: Additional metadata
        """
        self.rule_name = rule_name
        self.rulebook_name = rulebook_name
        self.rationale = rationale
        self.expected_lift = expected_lift
        self.expected_false_positive_impact = expected_false_positive_impact
        self.business_justification = business_justification
        self.proposed_by = proposed_by
        self.tags = tags or {}
        self.metadata = metadata or {}
        
        self.status = ProposalStatus.DRAFT
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.proposal_id = f"{rulebook_name}:{rule_name}:{self.created_at.isoformat()}"

    def submit(self) -> None:
        """Submit proposal for review."""
        self.status = ProposalStatus.SUBMITTED
        self.updated_at = datetime.utcnow()
        
        # Log submission
        audit_logger.log(
            action="proposal_submitted",
            entity_type="rule_proposal",
            entity_id=self.proposal_id,
            actor=self.proposed_by,
            metadata={
                "rule_name": self.rule_name,
                "rulebook_name": self.rulebook_name,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "rule_name": self.rule_name,
            "rulebook_name": self.rulebook_name,
            "rationale": self.rationale,
            "expected_lift": self.expected_lift,
            "expected_false_positive_impact": self.expected_false_positive_impact,
            "business_justification": self.business_justification,
            "proposed_by": self.proposed_by,
            "status": self.status.value,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ProposalReview:
    """Review of a rule proposal."""

    def __init__(
        self,
        proposal_id: str,
        reviewer: str,
        approved: bool,
        comments: Optional[str] = None,
        concerns: Optional[List[str]] = None,
    ):
        """
        Initialize a proposal review.

        Args:
            proposal_id: ID of the proposal being reviewed
            reviewer: Name/ID of reviewer
            approved: Whether proposal is approved
            comments: Review comments
            concerns: List of concerns raised
        """
        self.proposal_id = proposal_id
        self.reviewer = reviewer
        self.approved = approved
        self.comments = comments
        self.concerns = concerns or []
        self.reviewed_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "reviewer": self.reviewer,
            "approved": self.approved,
            "comments": self.comments,
            "concerns": self.concerns,
            "reviewed_at": self.reviewed_at.isoformat(),
        }


class ProposalRegistry:
    """Registry for managing rule proposals."""

    def __init__(self):
        self._proposals: Dict[str, RuleProposal] = {}
        self._reviews: Dict[str, List[ProposalReview]] = {}

    def register(self, proposal: RuleProposal) -> None:
        """Register a proposal."""
        self._proposals[proposal.proposal_id] = proposal

    def get(self, proposal_id: str) -> Optional[RuleProposal]:
        """Get a proposal by ID."""
        return self._proposals.get(proposal_id)

    def submit_review(self, review: ProposalReview) -> None:
        """Submit a review for a proposal."""
        if review.proposal_id not in self._reviews:
            self._reviews[review.proposal_id] = []
        self._reviews[review.proposal_id].append(review)

        # Update proposal status based on review
        proposal = self._proposals.get(review.proposal_id)
        if proposal:
            if review.approved:
                proposal.status = ProposalStatus.APPROVED
            else:
                proposal.status = ProposalStatus.REJECTED
            proposal.updated_at = datetime.utcnow()

        # Log review
        audit_logger.log(
            action="proposal_reviewed",
            entity_type="rule_proposal",
            entity_id=review.proposal_id,
            actor=review.reviewer,
            metadata={
                "approved": review.approved,
                "has_concerns": len(review.concerns) > 0,
            },
        )

    def get_reviews(self, proposal_id: str) -> List[ProposalReview]:
        """Get all reviews for a proposal."""
        return self._reviews.get(proposal_id, [])

    def list(self, status: Optional[ProposalStatus] = None) -> List[RuleProposal]:
        """List proposals, optionally filtered by status."""
        proposals = list(self._proposals.values())
        if status:
            proposals = [p for p in proposals if p.status == status]
        return proposals


# Global proposal registry
proposal_registry = ProposalRegistry()

