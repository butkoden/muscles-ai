from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .contracts import ContextBlock, LLMResponse, RetrievedChunk, SearchQuery, sanitize_mapping


class InMemoryRagSource:
    def __init__(
        self,
        name: str,
        *,
        chunks: Sequence[RetrievedChunk] | None = None,
        parents: Mapping[str, ContextBlock] | None = None,
        inspect_payload: Mapping[str, Any] | None = None,
        enabled: bool = True,
        fallback_on_empty: bool = False,
    ) -> None:
        self.name = name
        self.enabled = enabled
        self._chunks = list(chunks or [])
        self._parents = dict(parents or {})
        self._inspect_payload = sanitize_mapping(inspect_payload or {})
        self._fallback_on_empty = fallback_on_empty

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "keyword": True,
            "vector": True,
            "parent_fetch": bool(self._parents),
            "index_request": True,
        }

    def search_keyword(self, query: SearchQuery) -> Sequence[RetrievedChunk]:
        terms = _tokens(query.text)
        scored = []
        for chunk in self._chunks:
            chunk_terms = _tokens(chunk.text)
            matches = sum(1 for term in terms if term in chunk_terms)
            exact = 1 if query.text.lower() in chunk.text.lower() else 0
            if matches or exact:
                scored.append(replace(chunk, score=float(matches + exact * 2)))
        return self._with_fallback(scored, query)

    def search_vector(self, query: SearchQuery) -> Sequence[RetrievedChunk]:
        terms = set(_tokens(query.text))
        scored = []
        for chunk in self._chunks:
            chunk_terms = set(_tokens(chunk.text))
            if not terms or not chunk_terms:
                continue
            overlap = len(terms & chunk_terms)
            if overlap:
                score = overlap / max(len(terms | chunk_terms), 1)
                scored.append(replace(chunk, score=float(score)))
        return self._with_fallback(scored, query)

    def fetch_parent_blocks(self, chunks: Sequence[RetrievedChunk]) -> Sequence[ContextBlock]:
        seen = set()
        blocks = []
        for chunk in chunks:
            parent_id = chunk.parent_id
            if not parent_id or parent_id in seen or parent_id not in self._parents:
                continue
            seen.add(parent_id)
            blocks.append(self._parents[parent_id])
        return blocks

    def request_index(
        self,
        *,
        source: str | None = None,
        dry_run: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "planned" if dry_run else "accepted",
            "source": source or self.name,
            "chunks": len(self._chunks),
            "dry_run": bool(dry_run),
            "metadata": sanitize_mapping(metadata or {}),
        }

    def inspect(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "status": "ready" if self.enabled else "disabled",
            "chunks": len(self._chunks),
            "capabilities": self.capabilities,
        }
        payload.update(self._inspect_payload)
        return sanitize_mapping(payload)

    def healthcheck(self) -> dict[str, Any]:
        return {"status": "ok" if self.enabled else "disabled", "source": self.name}

    def _with_fallback(self, scored: list[RetrievedChunk], query: SearchQuery) -> list[RetrievedChunk]:
        if not scored and self._fallback_on_empty and self._chunks:
            scored = [replace(self._chunks[0], score=0.01)]
        return sorted(scored, key=lambda item: (-(item.score or 0.0), item.chunk_id))[: query.limit]


class FakeLLMProvider:
    def __init__(self, *, answer_prefix: str = "Fake answer") -> None:
        self.answer_prefix = answer_prefix

    def generate(
        self,
        prompt: str,
        *,
        options: Mapping[str, Any] | None = None,
        context: Sequence[ContextBlock] | None = None,
    ) -> LLMResponse:
        del prompt
        context_count = len(context or [])
        return LLMResponse(
            answer=f"{self.answer_prefix} using {context_count} context block(s).",
            metadata={"provider": "fake", "context_blocks": context_count, **sanitize_mapping(options or {})},
        )


class NoopLLMProvider:
    def __init__(self, *, runtime_key: str, provider: str = "noop", model_name: str | None = None) -> None:
        self.runtime_key = runtime_key
        self.provider = provider
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        *,
        options: Mapping[str, Any] | None = None,
        context: Sequence[ContextBlock] | None = None,
    ) -> LLMResponse:
        del prompt, context
        return LLMResponse(
            answer=(
                f"AI runtime '{self.runtime_key}' generated a deterministic answer"
                f" (provider={self.provider}, model={self.model_name or 'default'})"
            ),
            metadata={"provider": self.provider, "model": self.model_name or "default", **sanitize_mapping(options or {})},
        )


def _tokens(value: str) -> list[str]:
    current = []
    tokens: list[str] = []
    for char in value.lower():
        if char.isalnum() or char in {"_", "-", "."}:
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens
