"""Plugin system for extensibility."""

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from rulesmith.runtime.hooks import Hooks, hook_registry


class Plugin:
    """Base class for plugins."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version

    def activate(self) -> None:
        """Called when plugin is activated."""
        pass

    def deactivate(self) -> None:
        """Called when plugin is deactivated."""
        pass


class PluginRegistry:
    """Registry for plugins."""

    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._hooks_plugins: List[Hooks] = []

    def register(self, plugin: Plugin) -> None:
        """Register a plugin."""
        self._plugins[plugin.name] = plugin

        # If plugin implements Hooks, register it
        if isinstance(plugin, Hooks):
            hook_registry.register(plugin)
            self._hooks_plugins.append(plugin)

        plugin.activate()

    def unregister(self, plugin_name: str) -> None:
        """Unregister a plugin."""
        if plugin_name in self._plugins:
            plugin = self._plugins[plugin_name]
            plugin.deactivate()

            if plugin in self._hooks_plugins:
                # Note: HookRegistry doesn't have unregister, would need to add
                self._hooks_plugins.remove(plugin)

            del self._plugins[plugin_name]

    def get(self, plugin_name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(plugin_name)

    def list(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())


def load_plugin_from_module(module_path: str, plugin_class_name: str = "Plugin") -> Plugin:
    """
    Load a plugin from a Python module.

    Args:
        module_path: Module path (e.g., "my_package.my_plugin")
        plugin_class_name: Name of plugin class in module

    Returns:
        Plugin instance
    """
    module = importlib.import_module(module_path)
    plugin_class = getattr(module, plugin_class_name)
    return plugin_class()


def discover_plugins(plugin_dir: str) -> List[Type[Plugin]]:
    """
    Discover plugins in a directory.

    Args:
        plugin_dir: Directory path to search

    Returns:
        List of Plugin classes
    """
    plugins = []
    plugin_path = Path(plugin_dir)

    if not plugin_path.exists():
        return plugins

    for file in plugin_path.glob("*.py"):
        if file.name.startswith("_"):
            continue

        try:
            module_name = file.stem
            spec = importlib.util.spec_from_file_location(module_name, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find Plugin subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Plugin) and obj != Plugin:
                        plugins.append(obj)
        except Exception:
            continue

    return plugins


# Global plugin registry
plugin_registry = PluginRegistry()

