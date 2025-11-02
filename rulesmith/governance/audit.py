"""Audit logs and signed hashes."""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from rulesmith.io.ser import RulebookSpec


class AuditLogEntry:
    """Single audit log entry."""

    def __init__(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        actor: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp = datetime.utcnow()
        self.action = action  # e.g., "promote", "deploy", "modify"
        self.entity_type = entity_type  # e.g., "rulebook", "model"
        self.entity_id = entity_id  # e.g., model name or rulebook name
        self.actor = actor  # e.g., user ID or system
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "actor": self.actor,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class AuditLogger:
    """Audit logger with signed hashes."""

    def __init__(self, signing_key: Optional[str] = None):
        """
        Initialize audit logger.

        Args:
            signing_key: Optional key for signing audit entries
        """
        self.signing_key = signing_key
        self._entries: List[AuditLogEntry] = []

    def log(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        actor: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLogEntry:
        """
        Log an audit event.

        Args:
            action: Action performed
            entity_type: Type of entity
            entity_id: Entity identifier
            actor: Actor performing the action
            metadata: Optional metadata

        Returns:
            AuditLogEntry object
        """
        entry = AuditLogEntry(action, entity_type, entity_id, actor, metadata)
        self._entries.append(entry)
        return entry

    def sign_entry(self, entry: AuditLogEntry) -> str:
        """
        Generate signed hash for audit entry.

        Args:
            entry: Audit log entry

        Returns:
            SHA256 hash (signed if key provided)
        """
        data = json.dumps(entry.to_dict(), sort_keys=True)
        if self.signing_key:
            data += self.signing_key

        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def verify_entry(self, entry: AuditLogEntry, expected_hash: str) -> bool:
        """
        Verify audit entry hash.

        Args:
            entry: Audit log entry
            expected_hash: Expected hash

        Returns:
            True if hash matches
        """
        actual_hash = self.sign_entry(entry)
        return actual_hash == expected_hash

    def get_entries(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        action: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """
        Get audit entries with optional filters.

        Args:
            entity_type: Filter by entity type
            entity_id: Filter by entity ID
            action: Filter by action

        Returns:
            List of matching audit entries
        """
        entries = self._entries

        if entity_type:
            entries = [e for e in entries if e.entity_type == entity_type]

        if entity_id:
            entries = [e for e in entries if e.entity_id == entity_id]

        if action:
            entries = [e for e in entries if e.action == action]

        return entries

    def export(self) -> List[Dict[str, Any]]:
        """Export all audit entries."""
        return [entry.to_dict() for entry in self._entries]


def hash_rulebook_spec(spec: RulebookSpec) -> str:
    """
    Compute SHA256 hash of rulebook specification.

    Args:
        spec: Rulebook specification

    Returns:
        SHA256 hex digest
    """
    spec_dict = spec.model_dump()
    spec_json = json.dumps(spec_dict, sort_keys=True)
    return hashlib.sha256(spec_json.encode("utf-8")).hexdigest()


def log_promotion(
    logger: AuditLogger,
    model_name: str,
    from_stage: str,
    to_stage: str,
    actor: Optional[str] = None,
    slo_results: Optional[Dict[str, Any]] = None,
) -> AuditLogEntry:
    """
    Log a promotion event.

    Args:
        logger: Audit logger
        model_name: Model name
        from_stage: Source stage
        to_stage: Target stage
        actor: Actor performing promotion
        slo_results: Optional SLO check results

    Returns:
        AuditLogEntry
    """
    metadata = {
        "from_stage": from_stage,
        "to_stage": to_stage,
    }

    if slo_results:
        metadata["slo_check"] = slo_results

    return logger.log("promote", "model", model_name, actor=actor, metadata=metadata)


def log_deployment(
    logger: AuditLogger,
    rulebook_spec: RulebookSpec,
    actor: Optional[str] = None,
    environment: Optional[str] = None,
) -> AuditLogEntry:
    """
    Log a deployment event.

    Args:
        logger: Audit logger
        rulebook_spec: Rulebook specification
        actor: Actor performing deployment
        environment: Deployment environment

    Returns:
        AuditLogEntry
    """
    spec_hash = hash_rulebook_spec(rulebook_spec)
    metadata = {
        "rulebook_name": rulebook_spec.name,
        "rulebook_version": rulebook_spec.version,
        "spec_hash": spec_hash,
    }

    if environment:
        metadata["environment"] = environment

    return logger.log("deploy", "rulebook", rulebook_spec.name, actor=actor, metadata=metadata)


# Global audit logger instance
audit_logger = AuditLogger()
