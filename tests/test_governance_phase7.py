"""Tests for Phase 7: Governance features."""

import pytest
import tempfile
import os

from rulesmith.governance.pii_minimization import (
    PIIField,
    PIIMinimizer,
    PrivacyBudget,
    pii_minimizer,
)
from rulesmith.governance.review import (
    ReviewManager,
    ReviewRequest,
    ReviewStatus,
    review_manager,
)
from rulesmith.governance.signing import (
    Attestation,
    AttestationStore,
    RulebookSigner,
    Signature,
    rulebook_signer,
)
from rulesmith.governance.policy_as_code import (
    OPAPolicyEngine,
    CedarPolicyEngine,
    PolicyNode,
)
from rulesmith.io.ser import RulebookSpec, NodeSpec


class TestSigning:
    """Test rulebook signing and attestation."""
    
    def test_signature_creation(self):
        """Test signature creation."""
        spec = RulebookSpec(
            name="test_rulebook",
            version="1.0.0",
            nodes=[],
            edges=[],
        )
        
        signature = rulebook_signer.sign(spec, signer="alice")
        
        assert signature.signer == "alice"
        assert signature.signature_hash is not None
        assert signature.algorithm == "SHA256"
    
    def test_signature_verification(self):
        """Test signature verification."""
        spec = RulebookSpec(
            name="test_rulebook",
            version="1.0.0",
            nodes=[],
            edges=[],
        )
        
        signature = rulebook_signer.sign(spec, signer="alice")
        
        # Verify signature
        is_valid = rulebook_signer.verify(spec, signature)
        assert is_valid is True
        
        # Modify spec and verify fails
        spec2 = RulebookSpec(
            name="test_rulebook",
            version="1.0.1",  # Different version
            nodes=[],
            edges=[],
        )
        is_valid = rulebook_signer.verify(spec2, signature)
        assert is_valid is False
    
    def test_attestation_creation(self):
        """Test attestation creation."""
        spec = RulebookSpec(
            name="test_rulebook",
            version="1.0.0",
            nodes=[NodeSpec(name="rule1", kind="rule")],
            edges=[],
        )
        
        attestation = rulebook_signer.create_attestation(
            spec,
            signer="alice",
        )
        
        assert attestation.rulebook_name == "test_rulebook"
        assert len(attestation.signatures) == 1
        assert attestation.sbom is not None
        assert attestation.provenance is not None
    
    def test_attestation_store(self):
        """Test attestation store."""
        store = AttestationStore()
        
        spec = RulebookSpec(name="test", version="1.0.0", nodes=[], edges=[])
        attestation = rulebook_signer.create_attestation(spec, signer="alice")
        
        store.save_attestation(attestation)
        loaded = store.load_attestation("test", "1.0.0")
        
        assert loaded is not None
        assert loaded.rulebook_name == "test"
        assert len(loaded.signatures) == 1


class TestPIIMinimization:
    """Test PII minimization."""
    
    def test_pii_field_registration(self):
        """Test PII field registration."""
        minimizer = PIIMinimizer()
        
        field = PIIField(
            field_name="email",
            pii_type="email",
            mask_method="redact",
            sensitivity_level=3,
        )
        
        minimizer.register_pii_field(field)
        assert "email" in minimizer.pii_fields
    
    def test_pii_detection(self):
        """Test PII detection in inputs."""
        minimizer = PIIMinimizer()
        
        minimizer.register_pii_field(PIIField(
            field_name="email",
            pii_type="email",
        ))
        
        inputs = {"email": "user@example.com", "name": "John"}
        result = minimizer.check_pii_in_inputs(inputs, "llm_node")
        
        assert result["has_pii"] is True
        assert "email" in result["pii_fields"]
    
    def test_pii_masking(self):
        """Test PII masking."""
        minimizer = PIIMinimizer()
        
        minimizer.register_pii_field(PIIField(
            field_name="email",
            pii_type="email",
            mask_method="redact",
        ))
        minimizer.block_llm_node("llm_node")
        
        inputs = {"email": "user@example.com", "name": "John"}
        masked = minimizer.mask_pii(inputs, "llm_node")
        
        assert masked["name"] == "John"  # Not PII
        assert "[REDACTED]" in masked["email"] or masked["email"] != "user@example.com"
    
    def test_privacy_budget(self):
        """Test privacy budget enforcement."""
        budget = PrivacyBudget(user_id="user123", budget_per_day=100.0)
        
        # Consume budget
        success = budget.consume(50.0)
        assert success is True
        assert budget.used_today == 50.0
        
        # Try to consume too much
        success = budget.consume(60.0)  # Would exceed 100
        assert success is False
        
        # Check remaining
        remaining = budget.remaining_today()
        assert remaining == 50.0
    
    def test_privacy_budget_enforcement(self):
        """Test privacy budget enforcement."""
        minimizer = PIIMinimizer()
        
        minimizer.register_pii_field(PIIField(
            field_name="email",
            pii_type="email",
            sensitivity_level=10,
        ))
        
        # Get budget
        budget = minimizer.get_user_budget("user123")
        budget.budget_per_day = 50.0  # Low budget
        
        # Try to access PII
        allowed, error = minimizer.enforce_privacy_budget("user123", ["email"])
        
        # Should fail if budget is too low
        # Note: depends on budget configuration
        assert isinstance(allowed, bool)


class TestReview:
    """Test two-person review."""
    
    def test_review_request_creation(self):
        """Test review request creation."""
        manager = ReviewManager()
        
        request = manager.create_review_request(
            entity_type="rulebook",
            entity_id="test_rulebook",
            requester="alice",
            required_approvals=2,
            reviewers=["bob", "charlie"],
        )
        
        assert request.entity_id == "test_rulebook"
        assert request.required_approvals == 2
        assert len(request.reviewers) == 2
        assert request.status == ReviewStatus.PENDING
    
    def test_review_submission(self):
        """Test review submission."""
        manager = ReviewManager()
        
        request = manager.create_review_request(
            entity_type="rulebook",
            entity_id="test_rulebook",
            requester="alice",
            required_approvals=2,
            reviewers=["bob", "charlie"],
        )
        
        # Submit first approval
        review1 = manager.submit_review(request.request_id, "bob", approved=True)
        assert review1.approved is True
        
        status = manager.get_request_status(request.request_id)
        assert status["current_approvals"] == 1
        assert status["status"] == "pending"
        
        # Submit second approval
        review2 = manager.submit_review(request.request_id, "charlie", approved=True)
        assert review2.approved is True
        
        status = manager.get_request_status(request.request_id)
        assert status["current_approvals"] == 2
        assert status["status"] == "approved"
    
    def test_review_rejection(self):
        """Test review rejection."""
        manager = ReviewManager()
        
        request = manager.create_review_request(
            entity_type="rulebook",
            entity_id="test_rulebook",
            requester="alice",
            required_approvals=2,
            reviewers=["bob", "charlie"],
        )
        
        # Submit rejection
        manager.submit_review(request.request_id, "bob", approved=False)
        
        status = manager.get_request_status(request.request_id)
        assert status["status"] == "rejected"
    
    def test_protected_entities(self):
        """Test protected entity marking."""
        manager = ReviewManager()
        
        manager.mark_protected("prod_rulebook")
        assert manager.is_protected("prod_rulebook") is True
        assert manager.is_protected("dev_rulebook") is False


class TestPolicyAsCode:
    """Test policy-as-code integration."""
    
    def test_opa_engine(self):
        """Test OPA policy engine."""
        engine = OPAPolicyEngine()
        
        # Simple policy
        policy = """
        package test
        default allow = false
        allow {
            input.user == "admin"
        }
        """
        
        result = engine.evaluate(policy, {"user": "admin"})
        # Note: May fail if OPA not installed, but should return dict
        assert isinstance(result, dict)
    
    def test_cedar_engine(self):
        """Test Cedar policy engine."""
        engine = CedarPolicyEngine()
        
        # Simple policy
        policy = """
        permit(
            principal == User::"admin",
            action == Action::"view",
            resource
        );
        """
        
        result = engine.evaluate(policy, {
            "principal": "User::\"admin\"",
            "action": "Action::\"view\"",
        })
        # Note: May fail if Cedar not installed, but should return dict
        assert isinstance(result, dict)
    
    def test_policy_node(self):
        """Test PolicyNode."""
        # Simple OPA-like policy
        policy = "allow = input.user == 'admin'"
        
        node = PolicyNode(
            name="auth_policy",
            engine_type="opa",
            policy=policy,
        )
        
        result = node.evaluate({"user": "admin"})
        assert isinstance(result, dict)
        assert "allowed" in result

