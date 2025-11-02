"""Tests for plugin system."""

import pytest

from rulesmith.runtime.hooks import Hooks
from rulesmith.runtime.plugins import Plugin, PluginRegistry


class TestPlugin(Plugin, Hooks):
    """Test plugin implementation."""

    def __init__(self):
        super().__init__(name="test_plugin", version="1.0.0")
        self.activated = False

    def activate(self) -> None:
        """Activate plugin."""
        self.activated = True

    def deactivate(self) -> None:
        """Deactivate plugin."""
        self.activated = False

    def before_node(self, node_name: str, state, context) -> None:
        """Hook implementation."""
        pass

    def after_node(self, node_name: str, state, context, outputs) -> None:
        """Hook implementation."""
        pass

    def on_error(self, node_name: str, state, context, error) -> None:
        """Hook implementation."""
        pass

    def on_guard(self, node_name: str, state, context, guard_result) -> None:
        """Hook implementation."""
        pass

    def on_hitl(self, node_name: str, state, context, request_id: str) -> None:
        """Hook implementation."""
        pass


class TestPluginSystem:
    """Test plugin system."""

    def test_plugin_registry(self):
        """Test plugin registry."""
        registry = PluginRegistry()
        plugin = TestPlugin()

        registry.register(plugin)
        assert plugin.activated is True
        assert "test_plugin" in registry.list()

        registry.unregister("test_plugin")
        assert plugin.activated is False
        assert "test_plugin" not in registry.list()

    def test_plugin_as_hooks(self):
        """Test plugin implementing hooks."""
        registry = PluginRegistry()
        plugin = TestPlugin()

        registry.register(plugin)
        # Plugin should be registered as hooks too
        assert len(registry._hooks_plugins) == 1

