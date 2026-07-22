from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AiConfig:
    """
    Minimal normalized package config.
    """

    key: str
    transports: list[str] = field(default_factory=lambda: ["http", "mcp", "cli"])
    provider: str = "noop"
    model_name: str | None = None
    top_k_default: int = 5
    top_k_max: int = 20
    options: dict[str, Any] = field(default_factory=dict)
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    defaults: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, value: Mapping[str, Any], *, init_key: str = "ai") -> "AiConfig":
        data = dict(value or {})
        if "key" not in data:
            data["key"] = data.get("name", init_key)
        return cls(
            key=str(data["key"]),
            transports=list(data.get("transports", ["http", "mcp", "cli"])),
            provider=str(data.get("provider", "noop")),
            model_name=data.get("model_name"),
            top_k_default=int(data.get("top_k_default", 5)),
            top_k_max=int(data.get("top_k_max", 20)),
            options=dict(data.get("options", {}) or {}),
            providers={str(key): dict(value or {}) for key, value in dict(data.get("providers", {}) or {}).items()},
            models={str(key): dict(value or {}) for key, value in dict(data.get("models", {}) or {}).items()},
            defaults={str(key): str(value) for key, value in dict(data.get("defaults", {}) or {}).items()},
        )
