"""PII minimization and privacy budget enforcement."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from rulesmith.security.redaction import RedactionConfig, redact_dict


@dataclass
class PIIField:
    """PII field definition."""
    
    field_name: str
    pii_type: str  # e.g., "email", "phone", "ssn", "address", "custom"
    mask_method: str = "redact"  # "redact", "hash", "tokenize", "none"
    sensitivity_level: int = 1  # 1-5, higher = more sensitive


@dataclass
class PrivacyBudget:
    """Privacy budget for a user or tenant."""
    
    user_id: str
    budget_per_day: float = 100.0  # Budget units per day
    budget_per_month: float = 3000.0  # Budget units per month
    used_today: float = 0.0
    used_this_month: float = 0.0
    last_reset_date: str = field(default_factory=lambda: datetime.utcnow().date().isoformat())
    
    def reset_if_needed(self) -> None:
        """Reset budget if date has changed."""
        today = datetime.utcnow().date().isoformat()
        last_reset = datetime.fromisoformat(self.last_reset_date).date()
        
        if today != last_reset:
            # Check if it's a new month
            if last_reset.month != datetime.utcnow().month or last_reset.year != datetime.utcnow().year:
                self.used_this_month = 0.0
            
            self.used_today = 0.0
            self.last_reset_date = today
    
    def consume(self, amount: float) -> bool:
        """
        Consume budget.
        
        Args:
            amount: Amount to consume
        
        Returns:
            True if budget available, False otherwise
        """
        self.reset_if_needed()
        
        if self.used_today + amount > self.budget_per_day:
            return False
        
        if self.used_this_month + amount > self.budget_per_month:
            return False
        
        self.used_today += amount
        self.used_this_month += amount
        return True
    
    def remaining_today(self) -> float:
        """Get remaining budget for today."""
        self.reset_if_needed()
        return max(0.0, self.budget_per_day - self.used_today)
    
    def remaining_this_month(self) -> float:
        """Get remaining budget for this month."""
        self.reset_if_needed()
        return max(0.0, self.budget_per_month - self.used_this_month)


class PIIMinimizer:
    """Enforces PII minimization policies."""
    
    def __init__(self):
        self.pii_fields: Dict[str, PIIField] = {}  # field_name -> PIIField
        self.privacy_budgets: Dict[str, PrivacyBudget] = {}  # user_id -> PrivacyBudget
        self.blocked_llm_nodes: Set[str] = set()  # Nodes that block PII
    
    def register_pii_field(self, field: PIIField) -> None:
        """Register a PII field."""
        self.pii_fields[field.field_name] = field
    
    def register_pii_fields(self, fields: List[PIIField]) -> None:
        """Register multiple PII fields."""
        for field in fields:
            self.register_pii_field(field)
    
    def get_user_budget(self, user_id: str) -> PrivacyBudget:
        """Get or create privacy budget for a user."""
        if user_id not in self.privacy_budgets:
            self.privacy_budgets[user_id] = PrivacyBudget(user_id=user_id)
        
        return self.privacy_budgets[user_id]
    
    def check_pii_in_inputs(
        self,
        inputs: Dict[str, Any],
        node_name: str,
    ) -> Dict[str, Any]:
        """
        Check if PII is present in inputs for a node.
        
        Args:
            inputs: Input dictionary
            node_name: Name of node (to check if it's an LLM node)
        
        Returns:
            Dictionary with check results:
            - has_pii: bool
            - pii_fields: List of field names with PII
            - blocked: bool (if node blocks PII)
        """
        pii_fields_found = []
        
        for field_name, field_def in self.pii_fields.items():
            if field_name in inputs:
                value = inputs[field_name]
                if value and str(value).strip():
                    pii_fields_found.append(field_name)
        
        # Check if this is an LLM node that blocks PII
        blocked = node_name in self.blocked_llm_nodes
        
        return {
            "has_pii": len(pii_fields_found) > 0,
            "pii_fields": pii_fields_found,
            "blocked": blocked,
            "should_mask": blocked and len(pii_fields_found) > 0,
        }
    
    def mask_pii(
        self,
        inputs: Dict[str, Any],
        node_name: str,
    ) -> Dict[str, Any]:
        """
        Mask PII in inputs before sending to LLM node.
        
        Args:
            inputs: Input dictionary
            node_name: Name of node
        
        Returns:
            Masked inputs dictionary
        """
        check_result = self.check_pii_in_inputs(inputs, node_name)
        
        if not check_result["should_mask"]:
            return inputs
        
        # Create redaction config based on registered PII fields
        redact_config = RedactionConfig()
        
        # Mask PII fields
        masked_inputs = inputs.copy()
        for field_name in check_result["pii_fields"]:
            field_def = self.pii_fields.get(field_name)
            if field_def:
                if field_def.mask_method == "redact":
                    # Use redaction
                    if isinstance(masked_inputs[field_name], str):
                        from rulesmith.security.redaction import redact_text
                        masked_inputs[field_name] = redact_text(masked_inputs[field_name], redact_config)
                    elif isinstance(masked_inputs[field_name], dict):
                        masked_inputs[field_name] = redact_dict(masked_inputs[field_name], redact_config)
                elif field_def.mask_method == "hash":
                    # Hash the value
                    import hashlib
                    value_str = str(masked_inputs[field_name])
                    masked_inputs[field_name] = hashlib.sha256(value_str.encode()).hexdigest()[:16]
                elif field_def.mask_method == "tokenize":
                    # Replace with token (simplified)
                    masked_inputs[field_name] = f"[TOKEN_{field_name.upper()}]"
                # "none" means don't mask
        
        return masked_inputs
    
    def enforce_privacy_budget(
        self,
        user_id: str,
        pii_fields: List[str],
    ) -> tuple[bool, Optional[str]]:
        """
        Enforce privacy budget for a user.
        
        Args:
            user_id: User identifier
            pii_fields: List of PII field names being accessed
        
        Returns:
            (allowed, error_message)
        """
        budget = self.get_user_budget(user_id)
        
        # Calculate cost based on sensitivity
        cost = 0.0
        for field_name in pii_fields:
            field_def = self.pii_fields.get(field_name)
            if field_def:
                cost += field_def.sensitivity_level
        
        # Check budget
        if not budget.consume(cost):
            error_msg = (
                f"Privacy budget exhausted for user {user_id}. "
                f"Remaining today: {budget.remaining_today()}, "
                f"this month: {budget.remaining_this_month()}"
            )
            return False, error_msg
        
        return True, None
    
    def block_llm_node(self, node_name: str) -> None:
        """Mark an LLM node as blocking PII (requires masking)."""
        self.blocked_llm_nodes.add(node_name)
    
    def allow_llm_node(self, node_name: str) -> None:
        """Allow an LLM node to receive unmasked PII (removes blocking)."""
        self.blocked_llm_nodes.discard(node_name)


# Global PII minimizer
pii_minimizer = PIIMinimizer()

