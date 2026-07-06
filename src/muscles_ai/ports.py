from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .contracts import ContextBlock, LLMResponse, RetrievedChunk, SearchQuery


class VectorSearchPort(Protocol):
    def search_vector(self, query: SearchQuery) -> Sequence[RetrievedChunk]:
        ...


class KeywordSearchPort(Protocol):
    def search_keyword(self, query: SearchQuery) -> Sequence[RetrievedChunk]:
        ...


class SectionSearchPort(Protocol):
    def search_sections(self, query: SearchQuery) -> Sequence[str]:
        ...


class ParentFetchPort(Protocol):
    def fetch_parent_blocks(self, chunks: Sequence[RetrievedChunk]) -> Sequence[ContextBlock]:
        ...


class IndexRequestPort(Protocol):
    def request_index(self, *, source: str | None = None, dry_run: bool = False, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        ...


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, options: Mapping[str, Any] | None = None, context: Sequence[ContextBlock] | None = None) -> LLMResponse:
        ...
