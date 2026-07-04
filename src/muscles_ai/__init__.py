from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .actions import AskActionResult, SearchActionResult
    from .config import AiConfig
    from .package import AiPackage
    from .runtime import AiRuntime

__all__ = [
    "AiConfig",
    "init_package",
    "AiPackage",
    "AiRuntime",
    "AskActionResult",
    "SearchActionResult",
]


def __getattr__(name: str):
    if name == "AiConfig":
        from .config import AiConfig

        return AiConfig
    if name == "AiPackage":
        from .package import AiPackage

        return AiPackage
    if name == "init_package":
        from .package import init_package

        return init_package
    if name == "AiRuntime":
        from .runtime import AiRuntime

        return AiRuntime
    if name == "AskActionResult":
        from .actions import AskActionResult

        return AskActionResult
    if name == "SearchActionResult":
        from .actions import SearchActionResult

        return SearchActionResult
    raise AttributeError(name)


def __dir__():
    return sorted(__all__)
