"""Two-person review for protected flows."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ReviewStatus(Enum):
    """Status of a review."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class ReviewRequest:
    """Review request for a change."""
    
    request_id: str
    entity_type: str  # "rulebook", "promotion", "deployment"
    entity_id: str
    entity_version: Optional[str] = None
    requester: str
    reviewers: List[str] = field(default_factory=list)
    required_approvals: int = 2
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "entity_version": self.entity_version,
            "requester": self.requester,
            "reviewers": self.reviewers,
            "required_approvals": self.required_approvals,
            "status": self.status.value,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewRequest":
        """Create from dictionary."""
        return cls(
            request_id=data["request_id"],
            entity_type=data["entity_type"],
            entity_id=data["entity_id"],
            entity_version=data.get("entity_version"),
            requester=data["requester"],
            reviewers=data.get("reviewers", []),
            required_approvals=data.get("required_approvals", 2),
            status=ReviewStatus(data.get("status", "pending")),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Review:
    """Individual review from a reviewer."""
    
    review_id: str
    request_id: str
    reviewer: str
    approved: bool
    comments: Optional[str] = None
    reviewed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "review_id": self.review_id,
            "request_id": self.request_id,
            "reviewer": self.reviewer,
            "approved": self.approved,
            "comments": self.comments,
            "reviewed_at": self.reviewed_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Review":
        """Create from dictionary."""
        return cls(
            review_id=data["review_id"],
            request_id=data["request_id"],
            reviewer=data["reviewer"],
            approved=data["approved"],
            comments=data.get("comments"),
            reviewed_at=data.get("reviewed_at", datetime.utcnow().isoformat()),
        )


class ReviewManager:
    """Manages two-person review process."""
    
    def __init__(self):
        self.requests: Dict[str, ReviewRequest] = {}  # request_id -> ReviewRequest
        self.reviews: Dict[str, List[Review]] = {}  # request_id -> List[Review]
        self.protected_entities: Dict[str, bool] = {}  # entity_id -> is_protected
    
    def mark_protected(self, entity_id: str, entity_type: str = "rulebook") -> None:
        """Mark an entity as requiring review."""
        key = f"{entity_type}:{entity_id}"
        self.protected_entities[key] = True
    
    def is_protected(self, entity_id: str, entity_type: str = "rulebook") -> bool:
        """Check if an entity requires review."""
        key = f"{entity_type}:{entity_id}"
        return self.protected_entities.get(key, False)
    
    def create_review_request(
        self,
        entity_type: str,
        entity_id: str,
        requester: str,
        entity_version: Optional[str] = None,
        required_approvals: int = 2,
        reviewers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReviewRequest:
        """
        Create a review request.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity identifier
            requester: Requester identifier
            entity_version: Optional version
            required_approvals: Number of approvals required
            reviewers: Optional list of reviewers
            metadata: Optional metadata
        
        Returns:
            ReviewRequest
        """
        import uuid
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        
        request = ReviewRequest(
            request_id=request_id,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_version=entity_version,
            requester=requester,
            reviewers=reviewers or [],
            required_approvals=required_approvals,
            metadata=metadata or {},
        )
        
        self.requests[request_id] = request
        self.reviews[request_id] = []
        
        return request
    
    def submit_review(
        self,
        request_id: str,
        reviewer: str,
        approved: bool,
        comments: Optional[str] = None,
    ) -> Review:
        """
        Submit a review.
        
        Args:
            request_id: Review request ID
            reviewer: Reviewer identifier
            approved: Whether approved
            comments: Optional comments
        
        Returns:
            Review object
        
        Raises:
            ValueError: If request not found or already completed
        """
        if request_id not in self.requests:
            raise ValueError(f"Review request '{request_id}' not found")
        
        request = self.requests[request_id]
        
        if request.status != ReviewStatus.PENDING:
            raise ValueError(f"Review request '{request_id}' is not pending")
        
        # Check if reviewer is in the list
        if request.reviewers and reviewer not in request.reviewers:
            raise ValueError(f"Reviewer '{reviewer}' not authorized for request '{request_id}'")
        
        # Check if reviewer already reviewed
        existing_reviews = self.reviews.get(request_id, [])
        if any(r.reviewer == reviewer for r in existing_reviews):
            raise ValueError(f"Reviewer '{reviewer}' has already reviewed request '{request_id}'")
        
        # Create review
        import uuid
        review_id = f"rev_{uuid.uuid4().hex[:8]}"
        review = Review(
            review_id=review_id,
            request_id=request_id,
            reviewer=reviewer,
            approved=approved,
            comments=comments,
        )
        
        # Add review
        if request_id not in self.reviews:
            self.reviews[request_id] = []
        self.reviews[request_id].append(review)
        
        # Update request status
        approvals = sum(1 for r in self.reviews[request_id] if r.approved)
        rejections = sum(1 for r in self.reviews[request_id] if not r.approved)
        
        if rejections > 0:
            request.status = ReviewStatus.REJECTED
        elif approvals >= request.required_approvals:
            request.status = ReviewStatus.APPROVED
        
        return review
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """
        Get status of a review request.
        
        Args:
            request_id: Review request ID
        
        Returns:
            Status dictionary
        """
        if request_id not in self.requests:
            raise ValueError(f"Review request '{request_id}' not found")
        
        request = self.requests[request_id]
        reviews = self.reviews.get(request_id, [])
        
        approvals = sum(1 for r in reviews if r.approved)
        rejections = sum(1 for r in reviews if not r.approved)
        
        return {
            "request_id": request_id,
            "status": request.status.value,
            "required_approvals": request.required_approvals,
            "current_approvals": approvals,
            "current_rejections": rejections,
            "reviews": [r.to_dict() for r in reviews],
        }
    
    def is_approved(self, request_id: str) -> bool:
        """Check if a review request is approved."""
        if request_id not in self.requests:
            return False
        
        request = self.requests[request_id]
        return request.status == ReviewStatus.APPROVED
    
    def list_pending_requests(
        self,
        reviewer: Optional[str] = None,
    ) -> List[ReviewRequest]:
        """List pending review requests."""
        pending = [
            req for req in self.requests.values()
            if req.status == ReviewStatus.PENDING
        ]
        
        if reviewer:
            # Filter to requests where reviewer is in the list or list is empty
            pending = [
                req for req in pending
                if not req.reviewers or reviewer in req.reviewers
            ]
        
        return sorted(pending, key=lambda r: r.created_at)


# Global review manager
review_manager = ReviewManager()

