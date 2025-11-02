"""Global registries for rules and rulebooks."""

from typing import Callable, Dict, Optional

from rulesmith.io.ser import RuleSpec, RulebookSpec


class RuleRegistry:
    """Global registry for rule functions and their specifications."""

    def __init__(self):
        self._rules: Dict[str, Callable] = {}
        self._specs: Dict[str, RuleSpec] = {}

    def register(self, spec: RuleSpec, func: Callable) -> None:
        """
        Register a rule function with its specification.

        Args:
            spec: Rule specification
            func: Rule function
        """
        self._rules[spec.name] = func
        self._specs[spec.name] = spec

    def get(self, name: str) -> Optional[Callable]:
        """
        Get a registered rule function by name.

        Args:
            name: Rule name

        Returns:
            Rule function or None if not found
        """
        return self._rules.get(name)

    def get_spec(self, name: str) -> Optional[RuleSpec]:
        """
        Get a rule specification by name.

        Args:
            name: Rule name

        Returns:
            RuleSpec or None if not found
        """
        return self._specs.get(name)

    def list(self) -> list[str]:
        """List all registered rule names."""
        return list(self._rules.keys())

    def clear(self) -> None:
        """Clear all registered rules (mainly for testing)."""
        self._rules.clear()
        self._specs.clear()


class RulebookRegistry:
    """Global registry for rulebooks."""

    def __init__(self):
        self._rulebooks: Dict[str, RulebookSpec] = {}

    def register(self, name: str, spec: RulebookSpec) -> None:
        """
        Register a rulebook specification.

        Args:
            name: Rulebook name
            spec: Rulebook specification
        """
        self._rulebooks[name] = spec

    def get(self, name: str) -> Optional[RulebookSpec]:
        """
        Get a rulebook specification by name.

        Args:
            name: Rulebook name

        Returns:
            RulebookSpec or None if not found
        """
        return self._rulebooks.get(name)

    def list(self) -> list[str]:
        """List all registered rulebook names."""
        return list(self._rulebooks.keys())

    def clear(self) -> None:
        """Clear all registered rulebooks (mainly for testing)."""
        self._rulebooks.clear()


# Global registries
rule_registry = RuleRegistry()
rulebook_registry = RulebookRegistry()

