"""A/B testing lineage and traceability."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ForkLineage:
    """Lineage information for a fork/AB test."""
    
    fork_name: str
    policy: str
    selected_arm: str
    all_arms: List[str]
    arm_weights: Dict[str, float]
    identity: Optional[str] = None
    reasoning: Optional[Dict[str, Any]] = None
    downstream_nodes: Dict[str, List[str]] = field(default_factory=dict)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    metrics_ref: Optional[Dict[str, Any]] = None


class ABLineageTracker:
    """Tracks lineage for A/B tests across execution."""
    
    def __init__(self):
        self.forks: Dict[str, ForkLineage] = {}
        self.arm_executions: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_fork(
        self,
        fork_name: str,
        policy: str,
        selected_arm: str,
        all_arms: List[str],
        arm_weights: Dict[str, float],
        identity: Optional[str] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        metrics_ref: Optional[Dict[str, Any]] = None,
    ) -> ForkLineage:
        """
        Record a fork selection.
        
        Args:
            fork_name: Name of the fork
            policy: Policy used
            selected_arm: Selected arm
            all_arms: List of all arms
            arm_weights: Weights for each arm
            identity: Optional identity
            reasoning: Optional reasoning explanation
            metrics_ref: Optional metrics reference
        
        Returns:
            ForkLineage object
        """
        lineage = ForkLineage(
            fork_name=fork_name,
            policy=policy,
            selected_arm=selected_arm,
            all_arms=all_arms,
            arm_weights=arm_weights,
            identity=identity,
            reasoning=reasoning,
            metrics_ref=metrics_ref,
        )
        
        self.forks[fork_name] = lineage
        return lineage
    
    def record_arm_execution(
        self,
        fork_name: str,
        arm_name: str,
        node_name: str,
        outcome: Optional[Any] = None,
        latency_ms: float = 0.0,
        cost: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """
        Record execution of a node in an arm.
        
        Args:
            fork_name: Name of the fork
            arm_name: Name of the arm
            node_name: Name of the node executed
            outcome: Outcome value
            latency_ms: Execution latency
            cost: Execution cost
            error: Error if execution failed
        """
        key = f"{fork_name}:{arm_name}"
        if key not in self.arm_executions:
            self.arm_executions[key] = []
        
        execution = {
            "node": node_name,
            "outcome": outcome,
            "latency_ms": latency_ms,
            "cost": cost,
            "error": error,
        }
        
        self.arm_executions[key].append(execution)
        
        # Update downstream nodes in lineage
        if fork_name in self.forks:
            if arm_name not in self.forks[fork_name].downstream_nodes:
                self.forks[fork_name].downstream_nodes[arm_name] = []
            if node_name not in self.forks[fork_name].downstream_nodes[arm_name]:
                self.forks[fork_name].downstream_nodes[arm_name].append(node_name)
    
    def record_arm_outcome(
        self,
        fork_name: str,
        arm_name: str,
        outcome: Any,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record final outcome for an arm.
        
        Args:
            fork_name: Name of the fork
            arm_name: Name of the arm
            outcome: Final outcome
            metrics: Optional metrics
        """
        if fork_name in self.forks:
            self.forks[fork_name].outcomes[arm_name] = {
                "outcome": outcome,
                "metrics": metrics or {},
            }
    
    def get_fork_lineage(self, fork_name: str) -> Optional[ForkLineage]:
        """Get lineage for a fork."""
        return self.forks.get(fork_name)
    
    def get_arm_executions(self, fork_name: str, arm_name: str) -> List[Dict[str, Any]]:
        """Get all executions for an arm."""
        key = f"{fork_name}:{arm_name}"
        return self.arm_executions.get(key, [])
    
    def get_complete_lineage(self, fork_name: str) -> Optional[Dict[str, Any]]:
        """
        Get complete lineage for a fork including all executions.
        
        Args:
            fork_name: Name of the fork
        
        Returns:
            Complete lineage dictionary
        """
        if fork_name not in self.forks:
            return None
        
        lineage = self.forks[fork_name]
        
        result = {
            "fork_name": lineage.fork_name,
            "policy": lineage.policy,
            "selected_arm": lineage.selected_arm,
            "all_arms": lineage.all_arms,
            "arm_weights": lineage.arm_weights,
            "identity": lineage.identity,
            "reasoning": lineage.reasoning,
            "downstream_nodes": lineage.downstream_nodes,
            "outcomes": lineage.outcomes,
            "executions": {},
        }
        
        # Add executions for each arm
        for arm_name in lineage.all_arms:
            executions = self.get_arm_executions(fork_name, arm_name)
            result["executions"][arm_name] = executions
        
        return result
    
    def get_all_forks(self) -> Dict[str, ForkLineage]:
        """Get all recorded forks."""
        return self.forks.copy()


# Global lineage tracker
lineage_tracker = ABLineageTracker()

