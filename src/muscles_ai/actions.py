from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, Iterator

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

from .contracts import AskResult, ContextResult, SearchResult


ASK_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
        "source": {"type": "string"},
        "mode": {"type": "string", "enum": ["keyword", "vector", "hybrid"]},
        "filters": {"type": "object"},
        "metadata": {"type": "object"},
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
        "mode": {"type": "string", "enum": ["keyword", "vector", "hybrid"]},
        "filters": {"type": "object"},
        "metadata": {"type": "object"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


RETRIEVE_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
        "source": {"type": "string"},
        "mode": {"type": "string", "enum": ["keyword", "vector", "hybrid"]},
        "filters": {"type": "object"},
        "metadata": {"type": "object"},
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


INDEX_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "dry_run": {"type": "boolean"},
        "metadata": {"type": "object"},
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
        name="ai.retrieve_context",
        description="Retrieve block-level AI context with citations.",
        input_schema=RETRIEVE_CONTEXT_SCHEMA,
        transports=transports,
        handler=_retrieve_context,
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
        name="ai.source.inspect",
        description="Inspect a registered AI source.",
        input_schema=SIMPLE_SCHEMA,
        transports=transports,
        handler=_source_inspect,
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
        input_schema=INDEX_SCHEMA,
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
        ("ai.retrieve_context", "done"),
        ("ai.sources.list", "done"),
        ("ai.source.inspect", "done"),
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
    if isinstance(result, (AskResult, SearchResult, ContextResult)):
        return asdict(result)
    if is_dataclass(result):
        return asdict(result)
    return result if isinstance(result, dict) else {"value": result}


def _ask(payload: dict[str, Any], context: ActionContext) -> AskActionResult:
    runtime = _resolve_runtime(context)
    telemetry = _telemetry(context)
    attrs = _ai_attributes(runtime)
    top_k = payload.get("top_k", runtime.top_k_default)
    source = payload.get("source", "default")

    with telemetry.span(
        "muscles.ai.retrieve",
        **attrs,
        **{"ai.retriever": "runtime", "ai.documents.retrieved": int(top_k)},
    ):
        pass
    with telemetry.span("muscles.ai.rerank", **attrs, **{"ai.documents.retrieved": int(top_k)}):
        pass
    with telemetry.span("muscles.ai.prompt.build", **attrs, **{"ai.documents.retrieved": int(top_k)}):
        pass
    with telemetry.span("muscles.ai.generate", **attrs):
        result = runtime.ask(
            payload["question"],
            top_k=top_k,
            source=source,
            mode=payload.get("mode", "hybrid"),
            filters=payload.get("filters", {}),
            metadata=payload.get("metadata", {}),
        )
    with telemetry.span("muscles.ai.answer", **attrs, **{"ai.citations.count": len(result.sources)}):
        return _to_contract(result)


def _search(payload: dict[str, Any], context: ActionContext) -> SearchActionResult:
    runtime = _resolve_runtime(context)
    telemetry = _telemetry(context)
    attrs = _ai_attributes(runtime)
    top_k = payload.get("top_k", runtime.top_k_default)

    with telemetry.span(
        "muscles.ai.retrieve",
        **attrs,
        **{"ai.retriever": "runtime", "ai.documents.retrieved": int(top_k)},
    ):
        result = runtime.search(
            payload["query"],
            top_k=top_k,
            source=payload.get("source", "default"),
            mode=payload.get("mode", "hybrid"),
            filters=payload.get("filters", {}),
            metadata=payload.get("metadata", {}),
        )
    with telemetry.span("muscles.ai.rerank", **attrs, **{"ai.documents.retrieved": len(result.hits)}):
        return _to_contract(result)


def _retrieve_context(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    runtime = _resolve_runtime(context)
    telemetry = _telemetry(context)
    attrs = _ai_attributes(runtime)
    top_k = payload.get("top_k", runtime.top_k_default)
    with telemetry.span(
        "muscles.ai.retrieve",
        **attrs,
        **{"ai.retriever": "runtime", "ai.documents.retrieved": int(top_k)},
    ):
        result = runtime.retrieve_context(
            payload["query"],
            top_k=top_k,
            source=payload.get("source", "default"),
            mode=payload.get("mode", "hybrid"),
            filters=payload.get("filters", {}),
            metadata=payload.get("metadata", {}),
        )
    with telemetry.span("muscles.ai.rerank", **attrs, **{"ai.context.blocks.count": len(result.context)}):
        return _to_contract(result)


def _sources_list(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    del payload
    runtime = _resolve_runtime(context)
    return {"sources": runtime.list_sources(), "items": runtime.list_source_details()}


def _source_inspect(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    runtime = _resolve_runtime(context)
    return runtime.inspect_source(source=payload.get("source"))


def _documents_inspect(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    runtime = _resolve_runtime(context)
    source = payload.get("source")
    return runtime.inspect_documents(source=source)


def _index_request(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    runtime = _resolve_runtime(context)
    telemetry = _telemetry(context)
    with telemetry.span(
        "muscles.ai.embed",
        **_ai_attributes(runtime),
        **{"ai.embedding.model": runtime.model_name or "default"},
    ):
        return runtime.request_index(
            source=payload.get("source"),
            dry_run=bool(payload.get("dry_run", False)),
            metadata=payload.get("metadata", {}),
        )


def _inspect(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    runtime = _resolve_runtime(context)
    return runtime.capabilities()


def _doctor(payload: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    del payload
    runtime = _resolve_runtime(context)
    return runtime.doctor()


def _telemetry(context: ActionContext):
    try:
        from muscles import resolve_telemetry  # type: ignore[import-not-found]

        return resolve_telemetry(context.application)
    except Exception:
        return _NoopTelemetry()


def _ai_attributes(runtime) -> dict[str, Any]:
    return {
        "ai.provider": runtime.provider,
        "ai.model": runtime.model_name or "default",
    }


class _NoopTelemetry:
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        del name, attributes
        yield
