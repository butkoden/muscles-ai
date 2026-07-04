from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class SourceChunk:
    source: str
    excerpt: str
    score: float | None = None


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    source: str
    score: float
    excerpt: str


@dataclass(frozen=True)
class AskResult:
    question: str
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)


@dataclass(frozen=True)
class SearchResult:
    query: str
    hits: list[SearchHit] = field(default_factory=list)


class AiRuntime:
    """
    Placeholder runtime class for stage-1 AI package.
    It intentionally keeps behavior explicit and deterministic until provider adapters
    and prompt pipelines are introduced in follow-up steps.
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
    ) -> None:
        self.key = key
        self.provider = provider
        self.model_name = model_name
        self.options = dict(options or {})
        self.top_k_default = top_k_default
        self.top_k_max = top_k_max

    def ask(self, question: str, *, top_k: int = 5, source: str = "default") -> AskResult:
        limit = max(1, min(int(top_k), self.top_k_max))
        return AskResult(
            question=question,
            answer=(
                f"AI runtime '{self.key}' received question: {question}"
                f" (provider={self.provider}, model={self.model_name or 'default'})"
            ),
            sources=[
                SourceChunk(
                    source=source,
                    excerpt=f"No indexed chunks available for '{source}' in MVP runtime.",
                    score=0.0,
                )
            ] * limit,
        )

    def search(self, query: str, *, top_k: int = 5, source: str = "default") -> SearchResult:
        limit = max(1, min(int(top_k), self.top_k_max))
        return SearchResult(
            query=query,
            hits=[
                SearchHit(
                    chunk_id=f"{source}:{idx}",
                    source=source,
                    score=0.0,
                    excerpt="MVP search stub result. Real vector retrieval is planned.",
                )
                for idx in range(limit)
            ],
        )

    def list_sources(self) -> list[str]:
        return ["default", "documents"]

    def inspect_documents(self, source: str | None = None) -> dict[str, Any]:
        return {
            "source": source or "default",
            "status": "not_connected",
            "note": "documents package integration is optional for MVP",
        }

    def request_index(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "provider": self.provider,
            "provider_model": self.model_name or "default",
            "message": "Index request was accepted in MVP mode",
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "namespace": self.key,
            "provider": self.provider,
            "model_name": self.model_name,
            "features": [
                "ask",
                "search",
                "documents.inspect",
                "index.request",
            ],
        }
