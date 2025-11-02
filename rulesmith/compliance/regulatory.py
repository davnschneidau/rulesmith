"""Regulatory compliance validation (lending, explainability, audit)."""

from typing import Any, Dict, List, Optional

from rulesmith.explainability.explainer import DecisionExplainer, ReasonCode


class ComplianceChecker:
    """
    Validate rules against regulations.
    
    Enforces explainability requirements, audit trail requirements,
    adverse action requirements (lending regulations).
    """

    def __init__(self):
        self.checks: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def register_check(
        self,
        check_name: str,
        check_function: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """
        Register a compliance check.

        Args:
            check_name: Check name
            check_function: Function that performs check
                           Takes decision/explanation dict, returns compliance result
        """
        self.checks[check_name] = check_function

    def check_explainability(
        self,
        decision_id: str,
        explainer: Optional[DecisionExplainer] = None,
        reason_codes: Optional[List[ReasonCode]] = None,
    ) -> Dict[str, Any]:
        """
        Check explainability requirements (required for adverse actions).

        Args:
            decision_id: Decision ID
            explainer: Optional decision explainer
            reason_codes: Optional list of reason codes

        Returns:
            Compliance check result

        Raises:
            ValueError: If decision_id is empty
        """
        if not decision_id:
            raise ValueError("decision_id cannot be empty")

        issues = []

        # Check for reason codes (required for adverse actions)
        if not reason_codes and (not explainer or not explainer.reason_codes):
            issues.append("No reason codes provided for decision")

        # Check for rule traces (explainability)
        if not explainer or not explainer.rule_traces:
            issues.append("No rule traces available for decision")

        # Check that reason codes have descriptions
        if reason_codes:
            for code in reason_codes:
                if not code.description:
                    issues.append(f"Reason code '{code.code}' missing description")

        compliant = len(issues) == 0
        
        if not compliant:
            logger.warning(
                f"Explainability compliance check failed for decision {decision_id}",
                extra={"issues": issues, "decision_id": decision_id},
            )

        return {
            "check": "explainability",
            "decision_id": decision_id,
            "compliant": compliant,
            "issues": issues,
        }

    def check_adverse_action_requirements(
        self,
        decision: str,
        reason_codes: Optional[List[ReasonCode]] = None,
    ) -> Dict[str, Any]:
        """
        Check adverse action requirements (lending regulations).

        Args:
            decision: Decision made
            reason_codes: Optional reason codes

        Returns:
            Compliance check result
        """
        issues = []

        # Adverse actions require reason codes
        if decision.lower() in ("rejected", "denied", "blocked"):
            if not reason_codes or len(reason_codes) == 0:
                issues.append("Adverse action requires reason codes")

            # Reason codes must have consumer-facing descriptions
            if reason_codes:
                for code in reason_codes:
                    if not code.description:
                        issues.append(f"Reason code '{code.code}' missing consumer description")

        compliant = len(issues) == 0

        return {
            "check": "adverse_action_requirements",
            "decision": decision,
            "compliant": compliant,
            "issues": issues,
        }

    def check_audit_trail(
        self,
        decision_id: str,
        has_log: bool = False,
        has_inputs: bool = False,
        has_outputs: bool = False,
        has_explanation: bool = False,
    ) -> Dict[str, Any]:
        """
        Check audit trail requirements.

        Args:
            decision_id: Decision ID
            has_log: Whether decision log exists
            has_inputs: Whether inputs are logged
            has_outputs: Whether outputs are logged
            has_explanation: Whether explanation is logged

        Returns:
            Compliance check result
        """
        issues = []

        if not has_log:
            issues.append("Decision log missing")
        if not has_inputs:
            issues.append("Decision inputs not logged")
        if not has_outputs:
            issues.append("Decision outputs not logged")
        if not has_explanation:
            issues.append("Decision explanation not logged")

        compliant = len(issues) == 0

        return {
            "check": "audit_trail",
            "decision_id": decision_id,
            "compliant": compliant,
            "issues": issues,
        }

    def validate_decision(
        self,
        decision_id: str,
        decision: str,
        explainer: Optional[DecisionExplainer] = None,
        has_log: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform full compliance validation for a decision.

        Args:
            decision_id: Decision ID
            decision: Decision made
            explainer: Optional decision explainer
            has_log: Whether decision log exists

        Returns:
            Comprehensive compliance validation result
        """
        results = []

        # Run all checks
        results.append(
            self.check_explainability(
                decision_id,
                explainer,
                explainer.reason_codes if explainer else None,
            )
        )

        results.append(
            self.check_adverse_action_requirements(
                decision,
                explainer.reason_codes if explainer else None,
            )
        )

        results.append(
            self.check_audit_trail(
                decision_id,
                has_log=has_log,
                has_inputs=has_log,
                has_outputs=has_log,
                has_explanation=explainer is not None,
            )
        )

        # Run custom checks
        for check_name, check_function in self.checks.items():
            try:
                result = check_function({
                    "decision_id": decision_id,
                    "decision": decision,
                    "explainer": explainer,
                })
                results.append(result)
            except Exception as e:
                results.append({
                    "check": check_name,
                    "error": str(e),
                    "compliant": False,
                })

        # Aggregate result
        all_compliant = all(r.get("compliant", False) for r in results)
        all_issues = [issue for r in results for issue in r.get("issues", [])]

        return {
            "decision_id": decision_id,
            "decision": decision,
            "overall_compliant": all_compliant,
            "check_results": results,
            "all_issues": all_issues,
        }


# Global compliance checker
compliance_checker = ComplianceChecker()

