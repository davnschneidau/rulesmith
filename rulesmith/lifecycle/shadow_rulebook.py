"""Enhanced shadow mode for rulebook-level execution."""

from typing import Any, Dict, List, Optional

from rulesmith.dag.graph import Rulebook


class ShadowRulebookExecutor:
    """
    Execute rulebook in shadow mode against live traffic.
    
    Runs a rulebook in parallel with production without affecting
    production decisions. Used for safe testing and comparison.
    """

    def __init__(self, shadow_rulebook: Rulebook, production_rulebook: Optional[Rulebook] = None):
        """
        Initialize shadow executor.

        Args:
            shadow_rulebook: Rulebook to run in shadow mode
            production_rulebook: Optional production rulebook for comparison
        """
        self.shadow_rulebook = shadow_rulebook
        self.production_rulebook = production_rulebook
        self.comparisons: List[Dict[str, Any]] = []

    def execute_shadow(
        self,
        payload: Dict[str, Any],
        compare_with_production: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute shadow rulebook on a payload.

        Args:
            payload: Input payload
            compare_with_production: If True and production_rulebook provided, compare results

        Returns:
            Shadow execution results with optional comparison
        """
        # Execute shadow rulebook
        shadow_result = self.shadow_rulebook.run(payload.copy(), enable_mlflow=True)

        result = {
            "shadow_result": shadow_result,
            "shadow_rulebook": self.shadow_rulebook.name,
            "shadow_version": self.shadow_rulebook.version,
        }

        # Compare with production if requested
        if compare_with_production and self.production_rulebook:
            try:
                production_result = self.production_rulebook.run(payload.copy(), enable_mlflow=False)
                
                # Compare outputs
                comparison = self._compare_results(production_result, shadow_result)
                result["comparison"] = comparison
                result["production_result"] = production_result
                
                # Track comparison
                self.comparisons.append({
                    "payload": payload,
                    "production": production_result,
                    "shadow": shadow_result,
                    "comparison": comparison,
                })

            except Exception as e:
                result["comparison_error"] = str(e)

        return result

    def _compare_results(
        self,
        production: Dict[str, Any],
        shadow: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare production and shadow results."""
        # Find differences
        differences = {}
        
        # Compare common keys
        all_keys = set(production.keys()) | set(shadow.keys())
        
        for key in all_keys:
            if key.startswith("_"):
                continue  # Skip internal keys
            
            prod_val = production.get(key)
            shadow_val = shadow.get(key)
            
            if prod_val != shadow_val:
                differences[key] = {
                    "production": prod_val,
                    "shadow": shadow_val,
                }

        # Calculate agreement metrics
        total_keys = len([k for k in all_keys if not k.startswith("_")])
        matching_keys = total_keys - len(differences)
        agreement_rate = matching_keys / total_keys if total_keys > 0 else 1.0

        return {
            "differences": differences,
            "agreement_rate": agreement_rate,
            "matching_keys": matching_keys,
            "total_keys": total_keys,
            "has_differences": len(differences) > 0,
        }

    def get_disagreement_rate(self) -> float:
        """Calculate overall disagreement rate from comparisons."""
        if not self.comparisons:
            return 0.0

        total = len(self.comparisons)
        disagreements = sum(
            1 for comp in self.comparisons
            if comp["comparison"]["has_differences"]
        )

        return disagreements / total if total > 0 else 0.0

    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get summary of all comparisons."""
        if not self.comparisons:
            return {"total_comparisons": 0}

        disagreement_rate = self.get_disagreement_rate()
        
        # Aggregate difference patterns
        difference_patterns = {}
        for comp in self.comparisons:
            if comp["comparison"]["has_differences"]:
                diff_keys = list(comp["comparison"]["differences"].keys())
                pattern = tuple(sorted(diff_keys))
                difference_patterns[pattern] = difference_patterns.get(pattern, 0) + 1

        return {
            "total_comparisons": len(self.comparisons),
            "disagreement_rate": disagreement_rate,
            "agreement_rate": 1.0 - disagreement_rate,
            "difference_patterns": {str(k): v for k, v in difference_patterns.items()},
        }

