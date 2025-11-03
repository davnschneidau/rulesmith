"""Policy-as-code interop with OPA/Rego and Cedar."""

from typing import Any, Dict, Optional


class PolicyEngine:
    """Base class for policy engines."""
    
    def evaluate(
        self,
        policy: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a policy.
        
        Args:
            policy: Policy definition
            input_data: Input data
        
        Returns:
            Policy evaluation result
        """
        raise NotImplementedError


class OPAPolicyEngine(PolicyEngine):
    """Open Policy Agent (OPA) Rego policy engine."""
    
    def __init__(self):
        """Initialize OPA engine."""
        self._opa_available = False
        try:
            # Try to import OPA client
            # In production, would use opa-python-client or similar
            import subprocess
            result = subprocess.run(["opa", "version"], capture_output=True, text=True)
            if result.returncode == 0:
                self._opa_available = True
        except Exception:
            pass
    
    def evaluate(
        self,
        policy: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a Rego policy.
        
        Args:
            policy: Rego policy string
            input_data: Input data
        
        Returns:
            Policy evaluation result
        """
        if not self._opa_available:
            # Fallback: simple policy evaluation
            return self._evaluate_simple(policy, input_data)
        
        try:
            import subprocess
            import json
            import tempfile
            import os
            
            # Write policy to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as f:
                f.write(policy)
                policy_file = f.name
            
            # Write input to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(input_data, f)
                input_file = f.name
            
            try:
                # Run OPA eval
                result = subprocess.run(
                    ["opa", "eval", "-f", "json", "-d", policy_file, "-i", input_file, "data"],
                    capture_output=True,
                    text=True,
                )
                
                if result.returncode == 0:
                    output = json.loads(result.stdout)
                    return {
                        "allowed": True,
                        "result": output,
                    }
                else:
                    return {
                        "allowed": False,
                        "error": result.stderr,
                    }
            finally:
                os.unlink(policy_file)
                os.unlink(input_file)
        except Exception as e:
            return {
                "allowed": False,
                "error": str(e),
            }
    
    def _evaluate_simple(
        self,
        policy: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simple policy evaluation (fallback when OPA not available)."""
        # Very basic Rego-like evaluation
        # This is a simplified implementation - real OPA would be much more sophisticated
        
        # Check for common patterns
        if "allow" in policy.lower() and "input" in policy.lower():
            # Try to extract condition
            # This is a placeholder - real implementation would parse Rego properly
            return {
                "allowed": True,
                "result": {"message": "Simple policy evaluation (OPA not available)"},
            }
        
        return {
            "allowed": False,
            "error": "OPA not available and policy could not be evaluated",
        }


class CedarPolicyEngine(PolicyEngine):
    """AWS Cedar policy engine."""
    
    def __init__(self):
        """Initialize Cedar engine."""
        self._cedar_available = False
        try:
            # Try to import Cedar
            # In production, would use cedar-python or similar
            import subprocess
            result = subprocess.run(["cedar", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self._cedar_available = True
        except Exception:
            pass
    
    def evaluate(
        self,
        policy: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a Cedar policy.
        
        Args:
            policy: Cedar policy string
            input_data: Input data (principal, action, resource, context)
        
        Returns:
            Policy evaluation result
        """
        if not self._cedar_available:
            return {
                "allowed": False,
                "error": "Cedar not available",
            }
        
        try:
            import subprocess
            import json
            import tempfile
            import os
            
            # Cedar typically uses principal, action, resource, context
            principal = input_data.get("principal", "User::\"anonymous\"")
            action = input_data.get("action", "Action::\"view\"")
            resource = input_data.get("resource", "Resource::\"default\"")
            context = input_data.get("context", {})
            
            # Write policy to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cedar', delete=False) as f:
                f.write(policy)
                policy_file = f.name
            
            # Write context to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(context, f)
                context_file = f.name
            
            try:
                # Run Cedar authorize
                result = subprocess.run(
                    [
                        "cedar", "authorize",
                        "--principal", principal,
                        "--action", action,
                        "--resource", resource,
                        "--policies", policy_file,
                        "--context", context_file,
                    ],
                    capture_output=True,
                    text=True,
                )
                
                if result.returncode == 0:
                    return {
                        "allowed": True,
                        "result": {"decision": "Allow"},
                    }
                else:
                    return {
                        "allowed": False,
                        "error": result.stderr,
                    }
            finally:
                os.unlink(policy_file)
                os.unlink(context_file)
        except Exception as e:
            return {
                "allowed": False,
                "error": str(e),
            }


class PolicyNode:
    """Node that evaluates policies using external engines."""
    
    def __init__(
        self,
        name: str,
        engine_type: str,  # "opa" or "cedar"
        policy: str,
    ):
        """
        Initialize policy node.
        
        Args:
            name: Node name
            engine_type: Policy engine type ("opa" or "cedar")
            policy: Policy definition string
        """
        self.name = name
        self.engine_type = engine_type
        
        if engine_type == "opa":
            self.engine = OPAPolicyEngine()
        elif engine_type == "cedar":
            self.engine = CedarPolicyEngine()
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
        
        self.policy = policy
    
    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate policy with inputs.
        
        Args:
            inputs: Input data
        
        Returns:
            Policy evaluation result
        """
        result = self.engine.evaluate(self.policy, inputs)
        
        # Convert to standard format
        return {
            "allowed": result.get("allowed", False),
            "policy_result": result.get("result"),
            "error": result.get("error"),
        }


# Global policy engines
opa_engine = OPAPolicyEngine()
cedar_engine = CedarPolicyEngine()

