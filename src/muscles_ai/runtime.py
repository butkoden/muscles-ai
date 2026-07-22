from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    AskResult,
    ContextResult,
    RetrievalPolicy,
    SearchQuery,
    SearchResult,
)
from .gateway import ModelGateway
from .pipeline import RagPipeline, SourceRegistry, default_source, source_capabilities


class AiRuntime:
    """
    Framework-level RAG runtime.

    The runtime owns orchestration and contracts, while project code owns data
    adapters. It never opens database/vector/search connections by itself.
    """

    def __init__(
        self,
        *,
        key: str,
        provider: str = "noop",
        model_name: str | None = None,
        options: dict[str, Any] | None = None,
        top_k_default: int = 5,
        top_k_max: int = 20,
        providers: Mapping[str, Mapping[str, Any]] | None = None,
        models: Mapping[str, Mapping[str, Any]] | None = None,
        defaults: Mapping[str, str] | None = None,
        model_gateway: ModelGateway | None = None,
    ) -> None:
        self.key = key
        self.provider = provider
        self.model_name = model_name
        self.options = dict(options or {})
        self.options.setdefault("timeout_ms", 30_000)
        self.top_k_default = top_k_default
        self.top_k_max = top_k_max
        self.default_policy = RetrievalPolicy(limit_max=top_k_max)
        self.source_registry = SourceRegistry()
        self.model_gateway = model_gateway or self._build_model_gateway(
            providers=providers,
            models=models,
            defaults=defaults,
        )
        self.llm_provider = self.model_gateway.as_llm_provider()
        self.pipeline = RagPipeline(self.source_registry, policy=self.default_policy, llm_provider=self.llm_provider)
        self.register_source("default", default_source("default"))
        self.register_source("documents", default_source("documents"))

    def register_source(self, name: str, source: Any, *, policy: RetrievalPolicy | None = None) -> None:
        self.source_registry.register(name, source, policy=policy)

    def ask(
        self,
        question: str,
        *,
        top_k: int = 5,
        source: str = "default",
        mode: str = "hybrid",
        filters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> AskResult:
        query = self._query(question, source=source, mode=mode, top_k=top_k, filters=filters, metadata=metadata)
        self.pipeline.llm_provider = self.llm_provider
        return self.pipeline.ask(query, options=self.options, policy=policy)

    def search(
        self,
        query: str | SearchQuery,
        *,
        top_k: int = 5,
        source: str = "default",
        mode: str = "hybrid",
        filters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> SearchResult:
        search_query = query if isinstance(query, SearchQuery) else self._query(
            query,
            source=source,
            mode=mode,
            top_k=top_k,
            filters=filters,
            metadata=metadata,
        )
        return self.pipeline.search(search_query, policy=policy)

    def retrieve_context(
        self,
        query: str | SearchQuery,
        *,
        top_k: int = 5,
        source: str = "default",
        mode: str = "hybrid",
        filters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> ContextResult:
        search_query = query if isinstance(query, SearchQuery) else self._query(
            query,
            source=source,
            mode=mode,
            top_k=top_k,
            filters=filters,
            metadata=metadata,
        )
        return self.pipeline.retrieve_context(search_query, policy=policy)

    def list_sources(self) -> list[str]:
        return self.source_registry.names()

    def list_source_details(self) -> list[dict[str, Any]]:
        return self.source_registry.list_safe()

    def inspect_source(self, source: str | None = None) -> dict[str, Any]:
        return self.source_registry.inspect(source or "default")

    def inspect_documents(self, source: str | None = None) -> dict[str, Any]:
        payload = self.inspect_source(source or "documents")
        payload.setdefault("note", "ai.documents.inspect is a compatibility alias for ai.source.inspect")
        return payload

    def request_index(
        self,
        *,
        source: str | None = None,
        dry_run: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_name = source or "default"
        adapter = self.source_registry.resolve(source_name)
        if adapter is None or not hasattr(adapter, "request_index"):
            return {"status": "not_supported", "source": source_name}
        return adapter.request_index(source=source_name, dry_run=dry_run, metadata=metadata or {})

    def capabilities(self) -> dict[str, Any]:
        return {
            "namespace": self.key,
            "provider": self.provider,
            "model_name": self.model_name,
            "features": [
                "ask",
                "search",
                "retrieve_context",
                "source.inspect",
                "documents.inspect",
                "index.request",
            ],
            "sources": {
                "count": len(self.list_sources()),
                "items": self.list_source_details(),
            },
            "default_policy": self.default_policy.__dict__,
            "models": self.model_gateway.inspect(),
        }

    def doctor(self) -> dict[str, Any]:
        checks = [
            {"name": "ai.runtime.exists", "status": "ok"},
            {
                "name": "ai.llm_provider.exists",
                "status": "ok" if self.llm_provider is not None else "failed",
                "provider": self.provider,
            },
        ]
        model_doctor = self.model_gateway.doctor()
        checks.extend(model_doctor.get("checks", []))
        for source_name in self.list_sources():
            adapter = self.source_registry.resolve(source_name)
            capabilities = source_capabilities(adapter)
            status = "ok" if adapter is not None and capabilities else "warning"
            if adapter is not None and hasattr(adapter, "healthcheck"):
                try:
                    health = adapter.healthcheck()
                    status = str(health.get("status", status))
                except Exception:
                    status = "error"
            checks.append(
                {
                    "name": f"ai.source.{source_name}",
                    "status": status,
                    "capabilities": capabilities,
                }
            )
        overall = "ok" if all(check["status"] in {"ok", "ready", "configured"} for check in checks) else "warning"
        return {"status": overall, "checks": checks}

    def _query(
        self,
        text: str,
        *,
        source: str,
        mode: str,
        top_k: int,
        filters: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
    ) -> SearchQuery:
        return SearchQuery(
            text=text,
            source=source,
            mode=mode,  # type: ignore[arg-type]
            limit=max(1, min(int(top_k or self.top_k_default), self.top_k_max)),
            filters=filters or {},
            metadata=metadata or {},
        )

    def _build_model_gateway(
        self,
        *,
        providers: Mapping[str, Mapping[str, Any]] | None,
        models: Mapping[str, Mapping[str, Any]] | None,
        defaults: Mapping[str, str] | None,
    ) -> ModelGateway:
        if providers or models:
            return ModelGateway.from_config(providers=providers, models=models, defaults=defaults)
        return ModelGateway.from_legacy(
            provider=self.provider,
            model_name=self.model_name,
            options=self.options,
        )
