from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence


SENSITIVE_KEY_PARTS = (
    "secret",
    "token",
    "password",
    "credential",
    "authorization",
    "api_key",
    "private_key",
    "dsn",
    "url",
)


@dataclass(frozen=True)
class RetrievalPolicy:
    vector_weight: float = 0.5
    keyword_weight: float = 0.3
    exact_match_weight: float = 0.15
    metadata_weight: float = 0.05
    max_chunks_per_source: int = 3
    max_context_tokens: int = 6000
    fail_on_partial_search: bool = False
    limit_max: int = 20


@dataclass(frozen=True)
class SearchQuery:
    text: str
    source: str = "default"
    mode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    limit: int = 20
    filters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.text or "").strip():
            raise ValueError("SearchQuery.text must not be empty")
        if self.mode not in {"keyword", "vector", "hybrid"}:
            raise ValueError("SearchQuery.mode must be keyword, vector, or hybrid")
        object.__setattr__(self, "limit", max(1, int(self.limit)))
        object.__setattr__(self, "filters", dict(self.filters or {}))
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


@dataclass(frozen=True)
class Citation:
    citation_id: str
    chunk_id: str
    source: str
    title: str | None = None
    section_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    score: float | None = None
    parent_id: str | None = None
    section_path: str | None = None
    title: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    score_breakdown: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("RetrievedChunk.chunk_id must not be empty")
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))
        object.__setattr__(self, "score_breakdown", dict(self.score_breakdown or {}))


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
    title: str | None = None
    section_path: str | None = None
    parent_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    score_breakdown: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))
        object.__setattr__(self, "score_breakdown", dict(self.score_breakdown or {}))


@dataclass(frozen=True)
class ContextBlock:
    block_id: str
    title: str | None
    text: str
    source: str
    citations: Sequence[Citation] = field(default_factory=list)
    relevance: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "citations", list(self.citations or []))
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


RetrievedBlock = ContextBlock


@dataclass(frozen=True)
class SearchResult:
    query: str
    hits: list[SearchHit] = field(default_factory=list)
    total: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "total", self.total or len(self.hits))
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


@dataclass(frozen=True)
class ContextResult:
    query: str
    context: list[ContextBlock] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


@dataclass(frozen=True)
class AskResult:
    question: str
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    context: list[ContextBlock] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    model_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))
        object.__setattr__(self, "model_metadata", sanitize_mapping(self.model_metadata or {}))


@dataclass(frozen=True)
class LLMResponse:
    answer: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


def sanitize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, item in dict(value or {}).items():
        key_text = str(key)
        if _is_sensitive_key(key_text):
            continue
        if isinstance(item, Mapping):
            safe[key_text] = sanitize_mapping(item)
        elif isinstance(item, list):
            safe[key_text] = [sanitize_mapping(child) if isinstance(child, Mapping) else child for child in item]
        else:
            safe[key_text] = item
    return safe


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def citation_from_chunk(index: int, chunk: RetrievedChunk) -> Citation:
    return Citation(
        citation_id=f"C{index}",
        chunk_id=chunk.chunk_id,
        source=chunk.source,
        title=chunk.title,
        section_path=chunk.section_path,
        metadata=chunk.metadata,
    )


def hit_from_chunk(chunk: RetrievedChunk) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.chunk_id,
        source=chunk.source,
        score=float(chunk.score or 0.0),
        excerpt=chunk.text[:240],
        title=chunk.title,
        section_path=chunk.section_path,
        parent_id=chunk.parent_id,
        metadata=chunk.metadata,
        score_breakdown=chunk.score_breakdown,
    )
