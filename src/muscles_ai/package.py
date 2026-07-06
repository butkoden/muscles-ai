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
        del app
        return [_package_service(AiRuntime, lambda: runtime)]

    def actions(self, app, runtime: AiRuntime, *, config: AiConfig | Mapping[str, Any]):
        del runtime
        package_config = config if isinstance(config, AiConfig) else _normalize_config(config)
        register_ai_actions(app, transports=package_config.transports)
        return []

    def inspection_provider(self, app, runtime: AiRuntime, config: AiConfig | Mapping[str, Any] | None = None):
        del app, config
        return runtime.capabilities

    def doctor_provider(self, app, runtime: AiRuntime, config: AiConfig | Mapping[str, Any] | None = None):
        del app, config

        def doctor_ai() -> dict[str, Any]:
            return {
                "status": "ok",
                "checks": [
                    {
                        "name": "ai.runtime.exists",
                        "status": "ok" if runtime is not None else "failed",
                    }
                ],
            }

        return doctor_ai

    def init(self, app, config):
        package_config = _normalize_config(config or {})
        runtime = self.build_runtime(app, package_config)
        _apply_services(app, self.services(app, runtime))
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
    installable = _resolve_install_hook()
    if installable is not None:
        try:
            return installable(app=app, config=config, package=package)  # type: ignore[call-arg]
        except Exception:
            pass
    return package.init(app, config or {})


def _normalize_config(config) -> AiConfig:
    if isinstance(config, AiConfig):
        return config
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


def _package_service(interface: type, provider: Any):
    try:
        from muscles import PackageService  # type: ignore[import-not-found]

        return PackageService(interface=interface, provider=provider)
    except Exception:
        return {"interface": interface, "provider": provider}


def _apply_services(app, services: Any) -> None:
    container = _ensure_container(app)
    for service in services or []:
        if isinstance(service, Mapping):
            container.register(
                service["interface"],
                service["provider"],
                *tuple(service.get("args", ())),
                scope=service.get("scope", getattr(container, "APP", "app")),
                **dict(service.get("kwargs", {})),
            )
            continue
        container.register(
            service.interface,
            service.provider,
            *tuple(getattr(service, "args", ())),
            scope=getattr(service, "scope", getattr(container, "APP", "app")),
            **dict(getattr(service, "kwargs", {})),
        )


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
