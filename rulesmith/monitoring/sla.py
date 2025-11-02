"""SLA tracking and monitoring."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from rulesmith.governance.promotion import SLO


class SLADefinition:
    """
    Defines an SLA for a rule or rulebook.
    
    Tracks precision/recall targets, latency SLAs, review queue depth, etc.
    """

    def __init__(
        self,
        name: str,
        entity_type: str,  # "rule" or "rulebook"
        entity_name: str,
        slos: List[SLO],
        description: Optional[str] = None,
    ):
        """
        Initialize SLA definition.

        Args:
            name: SLA name
            entity_type: Type of entity ("rule" or "rulebook")
            entity_name: Name of rule/rulebook
            slos: List of SLOs to track
            description: Optional description
        """
        self.name = name
        self.entity_type = entity_type
        self.entity_name = entity_name
        self.slos = slos
        self.description = description

    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate SLA compliance against metrics.

        Args:
            metrics: Metrics dictionary

        Returns:
            Dictionary with SLO evaluation results
        """
        results = []
        all_passed = True

        for slo in self.slos:
            metric_value = metrics.get(slo.metric_name)
            if metric_value is None:
                results.append({
                    "slo": slo.metric_name,
                    "passed": False,
                    "reason": "metric_not_found",
                    "value": None,
                    "threshold": slo.threshold,
                })
                all_passed = False
            else:
                passed = slo.evaluate(metric_value)
                results.append({
                    "slo": slo.metric_name,
                    "passed": passed,
                    "value": metric_value,
                    "threshold": slo.threshold,
                    "operator": slo.operator,
                })
                if not passed:
                    all_passed = False

        return {
            "sla_name": self.name,
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "all_passed": all_passed,
            "slo_results": results,
        }


class SLATracker:
    """
    Monitor SLA compliance and generate reports.
    
    Tracks SLAs over time and generates compliance reports.
    """

    def __init__(self):
        self.sla_definitions: Dict[str, SLADefinition] = {}
        self.compliance_history: List[Dict[str, Any]] = []

    def register_sla(self, sla: SLADefinition) -> None:
        """Register an SLA definition."""
        self.sla_definitions[sla.name] = sla

    def check_compliance(
        self,
        sla_name: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check SLA compliance.

        Args:
            sla_name: SLA name
            metrics: Current metrics

        Returns:
            Compliance evaluation result
        """
        sla = self.sla_definitions.get(sla_name)
        if not sla:
            return {
                "error": f"SLA '{sla_name}' not found",
                "compliant": False,
            }

        result = sla.evaluate(metrics)
        result["checked_at"] = datetime.utcnow().isoformat()

        # Store in history
        self.compliance_history.append(result)

        return result

    def generate_report(
        self,
        sla_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate SLA compliance report.

        Args:
            sla_name: Optional SLA name to filter
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            Compliance report
        """
        # Filter history
        history = self.compliance_history.copy()

        if sla_name:
            history = [h for h in history if h.get("sla_name") == sla_name]

        if start_time:
            history = [
                h
                for h in history
                if datetime.fromisoformat(h["checked_at"]) >= start_time
            ]

        if end_time:
            history = [
                h
                for h in history
                if datetime.fromisoformat(h["checked_at"]) <= end_time
            ]

        # Calculate compliance rate
        total_checks = len(history)
        compliant_checks = sum(1 for h in history if h.get("all_passed", False))
        compliance_rate = compliant_checks / total_checks if total_checks > 0 else 0.0

        # Aggregate SLO results
        slo_summary = {}
        for record in history:
            for slo_result in record.get("slo_results", []):
                slo_name = slo_result["slo"]
                if slo_name not in slo_summary:
                    slo_summary[slo_name] = {
                        "total_checks": 0,
                        "passed": 0,
                        "failed": 0,
                    }
                slo_summary[slo_name]["total_checks"] += 1
                if slo_result.get("passed", False):
                    slo_summary[slo_name]["passed"] += 1
                else:
                    slo_summary[slo_name]["failed"] += 1

        # Calculate pass rates
        for slo_name, summary in slo_summary.items():
            summary["pass_rate"] = (
                summary["passed"] / summary["total_checks"]
                if summary["total_checks"] > 0
                else 0.0
            )

        return {
            "sla_name": sla_name,
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "total_checks": total_checks,
            "compliant_checks": compliant_checks,
            "compliance_rate": compliance_rate,
            "slo_summary": slo_summary,
            "history": history[-100:] if len(history) > 100 else history,  # Last 100 records
        }


# Global SLA tracker instance
sla_tracker = SLATracker()

