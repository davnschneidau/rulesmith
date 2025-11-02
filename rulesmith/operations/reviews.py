"""Review cadences and scheduled reviews."""

import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ReviewFrequency(str, Enum):
    """Review frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class ReviewSchedule:
    """
    Define review cadences for rules.
    
    Supports weekly rule reviews, quarterly governance, etc.
    """

    def __init__(
        self,
        name: str,
        frequency: ReviewFrequency,
        entity_type: str,  # "rule", "rulebook", etc.
        entity_name: str,
        next_review: Optional[datetime] = None,
        expiration_date: Optional[datetime] = None,
        reviewer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize review schedule.

        Args:
            name: Schedule name
            frequency: Review frequency
            entity_type: Type of entity to review
            entity_name: Name of entity
            next_review: Optional next review date
            expiration_date: Optional expiration date (for rule retirement)
            reviewer: Optional assigned reviewer
            metadata: Additional metadata
        """
        self.name = name
        self.frequency = frequency
        self.entity_type = entity_type
        self.entity_name = entity_name
        self.next_review = next_review or self._calculate_next_review(frequency)
        self.expiration_date = expiration_date
        self.reviewer = reviewer
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()

    def _calculate_next_review(self, frequency: ReviewFrequency) -> datetime:
        """Calculate next review date based on frequency."""
        now = datetime.utcnow()

        if frequency == ReviewFrequency.DAILY:
            return now + timedelta(days=1)
        elif frequency == ReviewFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif frequency == ReviewFrequency.MONTHLY:
            return now + timedelta(days=30)
        elif frequency == ReviewFrequency.QUARTERLY:
            return now + timedelta(days=90)
        elif frequency == ReviewFrequency.YEARLY:
            return now + timedelta(days=365)
        else:
            return now + timedelta(days=7)  # Default to weekly

    def is_due(self) -> bool:
        """Check if review is due."""
        return datetime.utcnow() >= self.next_review

    def is_expired(self) -> bool:
        """Check if rule/entity is expired."""
        if self.expiration_date:
            return datetime.utcnow() >= self.expiration_date
        return False

    def advance_next_review(self) -> None:
        """Advance next review date based on frequency."""
        self.next_review = self._calculate_next_review(self.frequency)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "frequency": self.frequency.value,
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "next_review": self.next_review.isoformat(),
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "reviewer": self.reviewer,
            "metadata": self.metadata,
            "is_due": self.is_due(),
            "is_expired": self.is_expired(),
            "created_at": self.created_at.isoformat(),
        }


class ReviewTask:
    """Represents a scheduled review task."""

    def __init__(
        self,
        schedule: ReviewSchedule,
        status: str = "pending",
        completed_at: Optional[datetime] = None,
        completed_by: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """
        Initialize review task.

        Args:
            schedule: Review schedule
            status: Task status ("pending", "in_progress", "completed", "skipped")
            completed_at: Optional completion time
            completed_by: Optional reviewer who completed
            notes: Optional review notes
        """
        self.schedule = schedule
        self.status = status
        self.completed_at = completed_at
        self.completed_by = completed_by
        self.notes = notes
        self.created_at = datetime.utcnow()

    def complete(self, reviewer: str, notes: Optional[str] = None) -> None:
        """Mark task as completed."""
        self.status = "completed"
        self.completed_by = reviewer
        self.completed_at = datetime.utcnow()
        self.notes = notes

        # Advance schedule
        self.schedule.advance_next_review()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schedule": self.schedule.to_dict(),
            "status": self.status,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "completed_by": self.completed_by,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }


class ReviewScheduler:
    """
    Manage review schedules and generate review tasks.
    
    Handles weekly rule reviews, quarterly governance, rule expiration.
    """

    def __init__(self):
        self.schedules: Dict[str, ReviewSchedule] = {}
        self.tasks: List[ReviewTask] = []

    def register_schedule(self, schedule: ReviewSchedule) -> None:
        """Register a review schedule."""
        self.schedules[schedule.name] = schedule

    def get_due_tasks(self) -> List[ReviewTask]:
        """Get all due review tasks."""
        due_tasks = []

        for schedule in self.schedules.values():
            if schedule.is_due() and not schedule.is_expired():
                # Check if task already exists
                existing = next(
                    (
                        task
                        for task in self.tasks
                        if task.schedule.name == schedule.name
                        and task.status == "pending"
                    ),
                    None,
                )

                if not existing:
                    task = ReviewTask(schedule)
                    self.tasks.append(task)
                    due_tasks.append(task)
                else:
                    due_tasks.append(existing)

        return due_tasks

    def get_expired_entities(self) -> List[ReviewSchedule]:
        """Get all expired entities (rules/rulebooks that should be retired)."""
        return [schedule for schedule in self.schedules.values() if schedule.is_expired()]

    def complete_task(
        self,
        schedule_name: str,
        reviewer: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Complete a review task.

        Args:
            schedule_name: Schedule name
            reviewer: Reviewer name
            notes: Optional review notes

        Returns:
            True if task found and completed
        """
        task = next(
            (
                task
                for task in self.tasks
                if task.schedule.name == schedule_name and task.status == "pending"
            ),
            None,
        )

        if task:
            task.complete(reviewer, notes)
            return True

        return False

    def get_all_tasks(
        self,
        status: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> List[ReviewTask]:
        """
        Get all tasks, optionally filtered.

        Args:
            status: Optional status filter
            entity_type: Optional entity type filter

        Returns:
            List of tasks
        """
        filtered = self.tasks.copy()

        if status:
            filtered = [task for task in filtered if task.status == status]

        if entity_type:
            filtered = [
                task
                for task in filtered
                if task.schedule.entity_type == entity_type
            ]

        return filtered


# Global review scheduler
review_scheduler = ReviewScheduler()

