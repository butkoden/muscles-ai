from __future__ import annotations

from typing import Any

try:
    from muscles import ActionContext
except Exception:  # pragma: no cover
    from muscles.core.core import ActionContext

try:
    from muscles import register_action  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    register_action = None


def _register_action(app, **kwargs):
    if register_action is not None:
        return register_action(app, **kwargs)
    from muscles.core.core import ActionContract, get_application_registry
    return get_application_registry(app).add_action(
        ActionContract(
            name=kwargs["name"],
            description=kwargs.get("description", ""),
            input_schema=kwargs.get("input_schema", None),
            output_schema=kwargs.get("output_schema", None),
            rules=kwargs.get("rules", []),
            handler_ref=kwargs.get("handler_ref", None),
            transports=kwargs.get("transports", []),
            stream_output=kwargs.get("stream_output", False),
            stream_metadata=kwargs.get("stream_metadata", None) or {},
            metadata=kwargs.get("metadata", None) or {},
            handler=kwargs.get("handler"),
        )
    )

from .runtime import AskResult, SearchResult


ASK_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
        "source": {"type": "string"},
    },
    "required": ["question"],
    "additionalProperties": False,
}


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
        "source": {"type": "string"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
    },
    "additionalProperties": False,
}


AskActionResult = dict[str, Any]
SearchActionResult = dict[str, Any]


def register_ai_actions(app, *, transports: list[str]) -> list[tuple[str, str]]:
    _register_action(
        app,
        name="ai.ask",
        description="Ask a question to AI runtime.",
        input_schema=ASK_SCHEMA,
        transports=transports,
        handler=_ask,
    )
    _register_action(
        app,
        name="ai.search",
        description="Search in AI retrieval layer.",
        input_schema=SEARCH_SCHEMA,
        transports=transports,
        handler=_search,
    )
    _register_action(
        app,
        name="ai.sources.list",
        description="List registered AI sources.",
        input_schema=SIMPLE_SCHEMA,
        transports=transports,
        handler=_sources_list,
    )
    _register_action(
        app,
        name="ai.documents.inspect",
        description="Inspect a source for doc/ chunk availability.",
        input_schema=SIMPLE_SCHEMA,
        transports=transports,
        handler=_documents_inspect,
    )
    _register_action(
        app,
        name="ai.index.request",
        description="Request index sync/refresh action.",
        input_schema=SIMPLE_SCHEMA,
        transports=transports,
        handler=_index_request,
    )
    _register_action(
        app,
        name="ai.inspect",
        description="Inspect AI package runtime metadata (safe to show).",
        input_schema=SIMPLE_SCHEMA,
        transports=["cli", "http", "mcp"],
        handler=_inspect,
    )
    _register_action(
        app,
        name="ai.doctor",
        description="Run lightweight runtime health check.",
        input_schema=SIMPLE_SCHEMA,
        transports=["cli"],
        handler=_doctor,
    )
    return [
        ("ai.ask", "done"),
        ("ai.search", "done"),
        ("ai.sources.list", "done"),
        ("ai.documents.inspect", "done"),
        ("ai.index.request", "done"),
        ("ai.inspect", "done"),
        ("ai.doctor", "done"),
    ]


def _resolve_runtime(context: ActionContext):
    container = getattr(context.application, "container", None)
    if container is None:
        raise RuntimeError("AI package runtime is not initialized")
    try:
        runtime = container.resolve(_runtime_type())
    except KeyError as exc:
        raise RuntimeError("AI package runtime is not registered") from exc
    return runtime


def _runtime_type():
    from .runtime import AiRuntime

    return AiRuntime


def _to_contract(result: Any) -> dict[str, Any]:
    if isinstance(result, AskResult):
        return {
            "question": result.question,
            "answer": result.answer,
            "sources": [chunk.__dict__ for chunk in result.sources],
        }
    if isinstance(result, SearchResult):
        return {
            "query": result.query,
            "hits": [hit.__dict__ for hit in result.hits],
        }
    return result if isinstance(result, dict) else {"value": result}


def _ask(payload: dict[str, Any], context: ActionContext) -> AskActionResult:
    runtime = _resolve_runtime(context)
    return _to_contract(
        runtime.ask(
            payload["question"],
            top_k=payload.get("top_k", 5),
            source=payload.get("source", "default"),
        )
    )


def _search(payload: dict[str, Any], context: ActionContext) -> SearchActionResult:
    runtime = _resolve_runtime(context)
    return _to_contract(
        runtime.search(
            payload["query"],
            top_k=payload.get("top_k", 5),
            source=payload.get("source", "default"),
        )
    )


def _sources_list(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    del payload
    runtime = _resolve_runtime(context)
    return {"sources": runtime.list_sources()}


def _documents_inspect(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    runtime = _resolve_runtime(context)
    source = payload.get("source")
    return runtime.inspect_documents(source=source)


def _index_request(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    runtime = _resolve_runtime(context)
    return runtime.request_index()


def _inspect(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    runtime = _resolve_runtime(context)
    return runtime.capabilities()


def _doctor(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    runtime = _resolve_runtime(context)
    return {
        "status": "ok",
        "checks": [
            {
                "name": "ai.runtime.exists",
                "status": "ok" if runtime is not None else "failed",
            }
        ],
    }
