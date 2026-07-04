from __future__ import annotations

import inspect
from typing import Any, Mapping

from .actions import register_ai_actions
from .config import AiConfig
from .runtime import AiRuntime


class AiPackage:
    namespace = "ai"

    def build_runtime(self, app, config: Mapping[str, Any]):
        del app
        return AiRuntime(**self._runtime_kwargs(_normalize_config(config)))

    def services(self, app, runtime: AiRuntime):
        container = _ensure_container(app)
        container.register(AiRuntime, lambda: runtime)
        return runtime

    def actions(self, app, runtime: AiRuntime, *, config: AiConfig):
        del runtime
        register_ai_actions(app, transports=config.transports)
        return True

    def init(self, app, config):
        package_config = _normalize_config(config or {})
        runtime = AiRuntime(
            key=package_config.key,
            provider=package_config.provider,
            model_name=package_config.model_name,
            options=package_config.options,
            top_k_default=package_config.top_k_default,
            top_k_max=package_config.top_k_max,
        )
        self.services(app, runtime)
        self.actions(app, runtime, config=package_config)
        return runtime

    @staticmethod
    def _runtime_kwargs(config: AiConfig):
        return {
            "key": config.key,
            "provider": config.provider,
            "model_name": config.model_name,
            "options": config.options,
            "top_k_default": config.top_k_default,
            "top_k_max": config.top_k_max,
        }


def init_package(app, config: Mapping[str, Any] | None):
    """
    Muscles package entry point.
    """
    package = AiPackage()
    runtime = package.init(app, config or {})
    installable = _resolve_install_hook()
    if installable is not None:
        try:
            # Future extension point once core package lifecycle lands.
            return installable(app=app, config=config, package=package)  # type: ignore[call-arg]
        except Exception:
            pass
    return runtime


def _normalize_config(config) -> AiConfig:
    if not isinstance(config, Mapping):
        if hasattr(config, "_object") and isinstance(getattr(config, "_object"), Mapping):
            config = dict(getattr(config, "_object"))
        elif hasattr(config, "to_dict") and callable(config.to_dict):
            raw = config.to_dict()
            if isinstance(raw, Mapping):
                config = dict(raw)
            else:
                config = dict(config)
        elif hasattr(config, "__dict__"):
            config = dict(getattr(config, "__dict__") or {})
        else:
            config = {}
    if "key" not in config and "name" in config:
        config = dict(config)
        config["key"] = config["name"]
    return AiConfig.from_raw(config, init_key="ai")


def _ensure_container(app):
    container = getattr(app, "container", None)
    if container is None:
        container = _dependency_container()
        setattr(app, "container", container)
    return container


def _resolve_install_hook():
    """
    Backward-compatible bridge for future `muscles` core lifecycle.
    """
    try:
        from muscles.core.lifecycle import install_package  # type: ignore[import-not-found]
        return install_package
    except Exception:
        pass
    try:
        from muscles.lifecycle import install_package  # type: ignore[import-not-found]
        return install_package
    except Exception:
        return None


def _dependency_container():
    try:
        from muscles.core import DependencyContainer  # type: ignore[import-not-found]
        return DependencyContainer()
    except Exception:  # pragma: no cover
        try:
            from muscles.core import DependencyContainer  # type: ignore[import-not-found]
            return DependencyContainer()
        except Exception:
            return _LegacyContainer()


class _LegacyContainer:
    """
    Fallback container for older Muscles versions that do not expose DependencyContainer.
    """

    def __init__(self):
        self._entries: dict[type, tuple[Any, tuple[Any, ...], dict[str, Any]]] = {}

    def register(self, interface: type, provider: Any, *args: Any, **kwargs: Any):
        self._entries[interface] = (provider, args, kwargs)

    def resolve(self, interface: type):
        if interface not in self._entries:
            raise KeyError(f"Dependency {interface.__name__} not registered")
        provider, args, kwargs = self._entries[interface]
        if inspect.isclass(provider):
            return provider(*args, **kwargs)
        if callable(provider):
            return provider(*args, **kwargs)
        return provider
