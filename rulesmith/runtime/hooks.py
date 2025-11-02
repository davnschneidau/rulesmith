"""Plugin hook system."""

from typing import Any, Callable, Dict, Protocol


class Hooks(Protocol):
    """Protocol for execution hooks."""

    def before_node(self, node_name: str, state: Dict[str, Any], context: Any) -> None:
        """Called before node execution."""
        ...

    def after_node(self, node_name: str, state: Dict[str, Any], context: Any, outputs: Dict[str, Any]) -> None:
        """Called after node execution."""
        ...

    def on_error(self, node_name: str, state: Dict[str, Any], context: Any, error: Exception) -> None:
        """Called on node execution error."""
        ...

    def on_guard(self, node_name: str, state: Dict[str, Any], context: Any, guard_result: Any) -> None:
        """Called on guard evaluation."""
        ...

    def on_hitl(self, node_name: str, state: Dict[str, Any], context: Any, request_id: str) -> None:
        """Called on HITL request submission."""
        ...


class HookRegistry:
    """Registry for multiple hooks."""

    def __init__(self):
        self._hooks: list[Hooks] = []

    def register(self, hook: Hooks) -> None:
        """Register a hook."""
        self._hooks.append(hook)

    def before_node(self, node_name: str, state: Dict[str, Any], context: Any) -> None:
        """Call all before_node hooks."""
        for hook in self._hooks:
            try:
                hook.before_node(node_name, state, context)
            except Exception:
                pass  # Don't fail execution on hook errors

    def after_node(self, node_name: str, state: Dict[str, Any], context: Any, outputs: Dict[str, Any]) -> None:
        """Call all after_node hooks."""
        for hook in self._hooks:
            try:
                hook.after_node(node_name, state, context, outputs)
            except Exception:
                pass

    def on_error(self, node_name: str, state: Dict[str, Any], context: Any, error: Exception) -> None:
        """Call all on_error hooks."""
        for hook in self._hooks:
            try:
                hook.on_error(node_name, state, context, error)
            except Exception:
                pass

    def on_guard(self, node_name: str, state: Dict[str, Any], context: Any, guard_result: Any) -> None:
        """Call all on_guard hooks."""
        for hook in self._hooks:
            try:
                hook.on_guard(node_name, state, context, guard_result)
            except Exception:
                pass

    def on_hitl(self, node_name: str, state: Dict[str, Any], context: Any, request_id: str) -> None:
        """Call all on_hitl hooks."""
        for hook in self._hooks:
            try:
                hook.on_hitl(node_name, state, context, request_id)
            except Exception:
                pass


# Global hook registry
hook_registry = HookRegistry()

