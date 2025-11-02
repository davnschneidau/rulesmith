"""Data retention and privacy management (GDPR/CCPA compliance)."""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional


class RetentionPolicy:
    """
    Define data retention rules.
    
    Automatically expire/delete old data for GDPR/CCPA compliance.
    """

    def __init__(
        self,
        name: str,
        entity_type: str,  # "decision_log", "audit_log", etc.
        retention_days: int,
        deletion_action: Optional[Callable[[str], None]] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize retention policy.

        Args:
            name: Policy name
            entity_type: Type of entity to retain
            retention_days: Number of days to retain
            deletion_action: Optional function to call for deletion
            description: Optional description
        """
        self.name = name
        self.entity_type = entity_type
        self.retention_days = retention_days
        self.deletion_action = deletion_action
        self.description = description

    def should_retain(self, created_at: datetime) -> bool:
        """Check if entity should still be retained."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        return created_at >= cutoff

    def get_expiration_date(self, created_at: datetime) -> datetime:
        """Get expiration date for entity."""
        return created_at + timedelta(days=self.retention_days)


class DataRetentionManager:
    """
    Automatically expire/delete old data.
    
    Handles GDPR right to deletion, CCPA compliance, data portability.
    """

    def __init__(self):
        self.policies: Dict[str, RetentionPolicy] = {}
        self.deleted_entities: List[Dict[str, Any]] = []

    def register_policy(self, policy: RetentionPolicy) -> None:
        """Register a retention policy."""
        self.policies[policy.name] = policy

    def apply_policy(
        self,
        policy_name: str,
        entities: List[Dict[str, Any]],
        created_at_key: str = "created_at",
        entity_id_key: str = "id",
    ) -> Dict[str, Any]:
        """
        Apply retention policy to entities.

        Args:
            policy_name: Policy name
            entities: List of entity dictionaries
            created_at_key: Key for created_at timestamp
            entity_id_key: Key for entity ID

        Returns:
            Summary of deletion results
        """
        policy = self.policies.get(policy_name)
        if not policy:
            return {
                "error": f"Policy '{policy_name}' not found",
                "deleted_count": 0,
            }

        deleted_count = 0
        deleted_ids = []

        for entity in entities:
            created_at_str = entity.get(created_at_key)
            if not created_at_str:
                continue

            # Parse timestamp
            try:
                if isinstance(created_at_str, str):
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                else:
                    created_at = created_at_str
            except Exception:
                continue  # Skip if can't parse

            # Check if should retain
            if not policy.should_retain(created_at):
                entity_id = entity.get(entity_id_key)
                if entity_id:
                    # Execute deletion action if provided
                    if policy.deletion_action:
                        try:
                            policy.deletion_action(entity_id)
                        except Exception:
                            pass  # Continue with other entities

                    deleted_ids.append(entity_id)
                    deleted_count += 1

                    # Track deletion
                    self.deleted_entities.append({
                        "policy": policy_name,
                        "entity_type": policy.entity_type,
                        "entity_id": entity_id,
                        "deleted_at": datetime.utcnow().isoformat(),
                    })

        return {
            "policy": policy_name,
            "deleted_count": deleted_count,
            "deleted_ids": deleted_ids,
        }

    def get_retention_summary(self) -> Dict[str, Any]:
        """Get summary of retention policies and deletions."""
        return {
            "policies": {
                name: {
                    "entity_type": policy.entity_type,
                    "retention_days": policy.retention_days,
                }
                for name, policy in self.policies.items()
            },
            "total_deleted": len(self.deleted_entities),
        }


class GDPRCompliance:
    """
    GDPR compliance utilities.
    
    Right to deletion, data portability, privacy-preserving logging.
    """

    @staticmethod
    def delete_user_data(user_id: str, data_sources: List[Any]) -> Dict[str, Any]:
        """
        Delete all data for a user (GDPR right to deletion).

        Args:
            user_id: User ID
            data_sources: List of data sources (DecisionLogStore, etc.)

        Returns:
            Deletion summary
        """
        deleted_count = 0
        errors = []

        for source in data_sources:
            try:
                # Try to get user-specific data and delete
                if hasattr(source, "get_for_user"):
                    user_data = source.get_for_user(user_id)
                    if hasattr(source, "delete"):
                        for item in user_data:
                            source.delete(item.id if hasattr(item, "id") else str(item))
                            deleted_count += 1
                elif hasattr(source, "query"):
                    # Try query-based deletion
                    user_items = source.query(user_id=user_id)
                    if hasattr(source, "delete"):
                        for item in user_items:
                            source.delete(item.id if hasattr(item, "id") else str(item))
                            deleted_count += 1
            except Exception as e:
                errors.append(str(e))

        return {
            "user_id": user_id,
            "deleted_count": deleted_count,
            "errors": errors,
            "deleted_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def export_user_data(user_id: str, data_sources: List[Any]) -> Dict[str, Any]:
        """
        Export all data for a user (GDPR data portability).

        Args:
            user_id: User ID
            data_sources: List of data sources

        Returns:
            Exported data dictionary
        """
        exported_data = {}

        for source in data_sources:
            try:
                if hasattr(source, "get_for_user"):
                    user_data = source.get_for_user(user_id)
                    source_name = source.__class__.__name__
                    exported_data[source_name] = [
                        item.to_dict() if hasattr(item, "to_dict") else str(item)
                        for item in user_data
                    ]
            except Exception:
                pass  # Continue with other sources

        return {
            "user_id": user_id,
            "exported_at": datetime.utcnow().isoformat(),
            "data": exported_data,
        }


# Global data retention manager
data_retention_manager = DataRetentionManager()

