from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


AI_ARCHITECTURE_CONTRACT: dict[str, Any] = {
    "role": "Framework-level AI/RAG orchestration through runtime, ports, adapters and Muscles actions.",
    "preferred_patterns": [
        "Use AiRuntime for orchestration and keep project data access in registered source ports/adapters.",
        "Expose AI behavior through Muscles action contracts and discover it through inspect_application.",
        "Keep retrieval, reranking and telemetry metadata deterministic and privacy-safe.",
        "Use MCP as a thin projection over the core action and inspection contracts.",
    ],
    "allowed_patterns": [
        "In-process model providers behind ModelGateway.",
        "Project-owned vector, keyword, parent-fetch and index adapters.",
        "Read-only AI actions for retrieval and diagnostics.",
        "Explicit confirmation metadata for state-changing actions.",
    ],
    "forbidden_patterns": [
        "Opening database or vector connections from the AI transport layer.",
        "Duplicating action registry, validation, dispatch or business rules in MCP.",
        "Putting raw questions, prompts, answers or excerpts into telemetry attributes.",
        "Treating an LLM as the enforcement mechanism for architecture rules.",
    ],
    "rules": [
        {
            "id": "ai.action.metadata",
            "severity": "error",
            "summary": "Every ai.* action declares AI, architecture and MCP safety metadata.",
        },
        {
            "id": "ai.state_change.confirmation",
            "severity": "error",
            "summary": "State-changing AI actions require explicit confirmation metadata.",
        },
        {
            "id": "ai.mcp.safety",
            "severity": "error",
            "summary": "MCP read-only and state-changing flags must agree with architecture metadata.",
        },
    ],
}


_READ_ONLY_ARCHITECTURE = {
    "side_effects": [],
    "state_changing": False,
    "requires_confirmation": False,
}
_READ_ONLY_MCP = {
    "exposed": True,
    "read_only": True,
    "destructive": False,
    "requires_confirmation": False,
}


AI_ACTION_METADATA: dict[str, dict[str, Any]] = {
    "ai.ask": {
        "ai": {
            "operation": "answer",
            "uses_retrieval": True,
            "uses_model": True,
            "telemetry_safe": True,
        },
        "architecture": {
            **_READ_ONLY_ARCHITECTURE,
            "side_effects": ["model_inference"],
        },
        "mcp": dict(_READ_ONLY_MCP),
    },
    "ai.search": {
        "ai": {
            "operation": "search",
            "uses_retrieval": True,
            "uses_model": False,
            "telemetry_safe": True,
        },
        "architecture": dict(_READ_ONLY_ARCHITECTURE),
        "mcp": dict(_READ_ONLY_MCP),
    },
    "ai.retrieve_context": {
        "ai": {
            "operation": "retrieve_context",
            "uses_retrieval": True,
            "uses_model": False,
            "telemetry_safe": True,
        },
        "architecture": dict(_READ_ONLY_ARCHITECTURE),
        "mcp": dict(_READ_ONLY_MCP),
    },
    "ai.sources.list": {
        "ai": {"operation": "source_discovery", "uses_retrieval": False, "uses_model": False, "telemetry_safe": True},
        "architecture": dict(_READ_ONLY_ARCHITECTURE),
        "mcp": dict(_READ_ONLY_MCP),
    },
    "ai.source.inspect": {
        "ai": {"operation": "source_inspection", "uses_retrieval": False, "uses_model": False, "telemetry_safe": True},
        "architecture": dict(_READ_ONLY_ARCHITECTURE),
        "mcp": dict(_READ_ONLY_MCP),
    },
    "ai.documents.inspect": {
        "ai": {"operation": "document_inspection", "uses_retrieval": False, "uses_model": False, "telemetry_safe": True},
        "architecture": dict(_READ_ONLY_ARCHITECTURE),
        "mcp": dict(_READ_ONLY_MCP),
    },
    "ai.index.request": {
        "ai": {
            "operation": "index_request",
            "uses_retrieval": False,
            "uses_model": False,
            "telemetry_safe": True,
        },
        "architecture": {
            "side_effects": ["index_write"],
            "state_changing": True,
            "requires_confirmation": True,
        },
        "mcp": {
            "exposed": True,
            "read_only": False,
            "destructive": False,
            "requires_confirmation": True,
        },
    },
    "ai.inspect": {
        "ai": {"operation": "runtime_inspection", "uses_retrieval": False, "uses_model": False, "telemetry_safe": True},
        "architecture": dict(_READ_ONLY_ARCHITECTURE),
        "mcp": dict(_READ_ONLY_MCP),
    },
    "ai.doctor": {
        "ai": {"operation": "runtime_diagnostics", "uses_retrieval": False, "uses_model": False, "telemetry_safe": True},
        "architecture": dict(_READ_ONLY_ARCHITECTURE),
        "mcp": {"exposed": False, "read_only": True, "destructive": False, "requires_confirmation": False},
    },
}


def architecture_contract() -> dict[str, Any]:
    """Return a detached, machine-readable copy for inspection consumers."""
    return deepcopy(AI_ARCHITECTURE_CONTRACT)


def action_metadata(name: str) -> dict[str, Any]:
    """Return safety metadata without sharing mutable package-level state."""
    metadata = AI_ACTION_METADATA.get(name)
    if metadata is None:
        metadata = {
            "ai": {"operation": name.removeprefix("ai."), "telemetry_safe": True},
            "architecture": dict(_READ_ONLY_ARCHITECTURE),
            "mcp": dict(_READ_ONLY_MCP),
        }
    return deepcopy(metadata)


def check_ai_architecture(app: Any) -> dict[str, Any]:
    """Validate AI action safety metadata without invoking models or providers."""
    actions = _application_actions(app)
    violations_by_rule = {
        "ai.action.metadata": [],
        "ai.state_change.confirmation": [],
        "ai.mcp.safety": [],
    }

    for action in actions:
        name = _action_value(action, "name")
        metadata = _action_value(action, "metadata")
        if not isinstance(metadata, Mapping) or not all(
            isinstance(metadata.get(key), Mapping) for key in ("ai", "architecture", "mcp")
        ):
            violations_by_rule["ai.action.metadata"].append(
                {"action": name, "reason": "missing ai, architecture or mcp metadata"}
            )
            continue

        architecture = metadata["architecture"]
        mcp = metadata["mcp"]
        if architecture.get("state_changing") and not architecture.get("requires_confirmation"):
            violations_by_rule["ai.state_change.confirmation"].append(
                {"action": name, "reason": "state-changing action does not require confirmation"}
            )
        if bool(architecture.get("state_changing")) == bool(mcp.get("read_only")):
            violations_by_rule["ai.mcp.safety"].append(
                {"action": name, "reason": "architecture.state_changing conflicts with mcp.read_only"}
            )
        if bool(architecture.get("requires_confirmation")) != bool(mcp.get("requires_confirmation")):
            violations_by_rule["ai.mcp.safety"].append(
                {"action": name, "reason": "confirmation requirement differs between architecture and mcp metadata"}
            )

    checks = []
    for rule in AI_ARCHITECTURE_CONTRACT["rules"]:
        violations = violations_by_rule[rule["id"]]
        checks.append(
            {
                **rule,
                "name": rule["id"],
                "status": "failed" if violations else "ok",
                "violations": violations,
            }
        )
    return {
        "status": "error" if any(check["status"] == "failed" for check in checks) else "ok",
        "checks": checks,
    }


def _application_actions(app: Any) -> list[Any]:
    try:
        from muscles.core import get_application_registry

        registry = get_application_registry(app, create=False)
    except Exception:
        return []
    return [
        action
        for action in getattr(registry, "actions", []) or []
        if str(_action_value(action, "name") or "").startswith("ai.")
    ]


def _action_value(action: Any, key: str, default: Any = None) -> Any:
    if isinstance(action, Mapping):
        return action.get(key, default)
    return getattr(action, key, default)
