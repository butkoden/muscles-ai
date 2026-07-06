from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Mapping, Sequence

from .contracts import (
    AskResult,
    Citation,
    ContextBlock,
    ContextResult,
    LLMResponse,
    RetrievedChunk,
    RetrievalPolicy,
    SearchHit,
    SearchQuery,
    SearchResult,
    SourceChunk,
    citation_from_chunk,
    hit_from_chunk,
    sanitize_mapping,
)
from .memory import InMemoryRagSource, NoopLLMProvider


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, Any] = {}
        self._policies: dict[str, RetrievalPolicy] = {}

    def register(self, name: str, source: Any, *, policy: RetrievalPolicy | None = None) -> None:
        if not name:
            raise ValueError("source name must not be empty")
        if name in self._sources:
            raise ValueError(f"AI source '{name}' is already registered")
        self._sources[name] = source
        if policy is not None:
            self._policies[name] = policy

    def names(self) -> list[str]:
        return sorted(self._sources)

    def resolve(self, name: str) -> Any | None:
        return self._sources.get(name)

    def policy_for(self, name: str, default: RetrievalPolicy) -> RetrievalPolicy:
        return self._policies.get(name, default)

    def list_safe(self) -> list[dict[str, Any]]:
        return [self.inspect(name) for name in self.names()]

    def inspect(self, name: str) -> dict[str, Any]:
        source = self.resolve(name)
        if source is None:
            return {"name": name, "status": "not_found", "capabilities": {}}
        if hasattr(source, "inspect"):
            payload = source.inspect()
        else:
            payload = {"name": name, "status": "not_supported"}
        payload = sanitize_mapping(payload if isinstance(payload, Mapping) else {"value": payload})
        payload.setdefault("name", name)
        payload.setdefault("status", "ready" if getattr(source, "enabled", True) else "disabled")
        payload.setdefault("capabilities", source_capabilities(source))
        return payload


class RagPipeline:
    def __init__(self, registry: SourceRegistry, *, policy: RetrievalPolicy, llm_provider: Any) -> None:
        self.registry = registry
        self.policy = policy
        self.llm_provider = llm_provider

    def search(self, query: SearchQuery, policy: RetrievalPolicy | None = None) -> SearchResult:
        active_policy = policy or self.registry.policy_for(query.source, self.policy)
        query = replace(query, limit=min(query.limit, active_policy.limit_max))
        source = self.registry.resolve(query.source) or self.registry.resolve("default")
        if source is None or getattr(source, "enabled", True) is False:
            return SearchResult(query=query.text, hits=[], metadata={"source": query.source, "mode": query.mode})

        keyword_hits = self._call_search(source, "search_keyword", query, query.mode in {"keyword", "hybrid"}, active_policy)
        vector_hits = self._call_search(source, "search_vector", query, query.mode in {"vector", "hybrid"}, active_policy)
        merged = merge_search_results(
            keyword_hits=keyword_hits,
            vector_hits=vector_hits,
            query=query,
            policy=active_policy,
        )
        reranked = rerank_chunks(query, merged, active_policy)[: query.limit]
        return SearchResult(
            query=query.text,
            hits=[hit_from_chunk(chunk) for chunk in reranked],
            metadata={"source": query.source, "mode": query.mode},
        )

    def retrieve_context(self, query: SearchQuery, policy: RetrievalPolicy | None = None) -> ContextResult:
        active_policy = policy or self.registry.policy_for(query.source, self.policy)
        search = self.search(query, active_policy)
        source = self.registry.resolve(query.source) or self.registry.resolve("default")
        chunks = [_chunk_from_hit(hit, query.text) for hit in search.hits]
        blocks = expand_parent_blocks(chunks, source)
        context = assemble_context(blocks, active_policy)
        citations = [citation for block in context for citation in block.citations]
        return ContextResult(
            query=query.text,
            context=context,
            citations=citations,
            metadata={**dict(search.metadata), "chunks": len(search.hits), "context_blocks": len(context)},
        )

    def ask(self, query: SearchQuery, *, options: Mapping[str, Any] | None = None, policy: RetrievalPolicy | None = None) -> AskResult:
        context = self.retrieve_context(query, policy)
        if not context.context:
            return AskResult(
                question=query.text,
                answer="No context available for this question.",
                sources=[],
                citations=[],
                context=[],
                metadata={"source": query.source, "mode": query.mode, "no_context": True},
            )

        prompt = build_prompt(query.text, context.context)
        response = self.llm_provider.generate(prompt, options=options or {}, context=context.context)
        if not isinstance(response, LLMResponse):
            response = LLMResponse(answer=str(response), metadata={})
        return AskResult(
            question=query.text,
            answer=response.answer,
            sources=[
                SourceChunk(source=citation.source, excerpt=citation.title or citation.chunk_id, score=None)
                for citation in context.citations
            ],
            citations=context.citations,
            context=context.context,
            metadata=context.metadata,
            model_metadata=response.metadata,
        )

    def _call_search(
        self,
        source: Any,
        method_name: str,
        query: SearchQuery,
        enabled: bool,
        policy: RetrievalPolicy,
    ) -> list[RetrievedChunk]:
        if not enabled or not hasattr(source, method_name):
            return []
        try:
            return list(getattr(source, method_name)(query) or [])
        except Exception:
            if policy.fail_on_partial_search:
                raise
            return []


def merge_search_results(
    *,
    keyword_hits: Sequence[RetrievedChunk],
    vector_hits: Sequence[RetrievedChunk],
    query: SearchQuery,
    policy: RetrievalPolicy,
) -> list[RetrievedChunk]:
    del query
    keyword_scores = _normalize_scores(keyword_hits)
    vector_scores = _normalize_scores(vector_hits)
    merged: dict[str, RetrievedChunk] = {}
    breakdowns: dict[str, dict[str, float]] = defaultdict(dict)

    for chunk in keyword_hits:
        merged.setdefault(chunk.chunk_id, chunk)
        breakdowns[chunk.chunk_id]["keyword"] = keyword_scores.get(chunk.chunk_id, 0.0)
    for chunk in vector_hits:
        merged.setdefault(chunk.chunk_id, chunk)
        breakdowns[chunk.chunk_id]["vector"] = vector_scores.get(chunk.chunk_id, 0.0)

    output = []
    for chunk_id, chunk in merged.items():
        breakdown = {
            "keyword": breakdowns[chunk_id].get("keyword", 0.0),
            "vector": breakdowns[chunk_id].get("vector", 0.0),
        }
        score = (breakdown["keyword"] * policy.keyword_weight) + (breakdown["vector"] * policy.vector_weight)
        output.append(replace(chunk, score=score, score_breakdown=breakdown))
    return output


def rerank_chunks(query: SearchQuery, chunks: Sequence[RetrievedChunk], policy: RetrievalPolicy) -> list[RetrievedChunk]:
    per_source: dict[str, int] = defaultdict(int)
    reranked = []
    query_text = query.text.lower()
    query_terms = set(_tokens(query.text))
    for chunk in chunks:
        metadata = sanitize_mapping(chunk.metadata)
        exact = policy.exact_match_weight if query_text and query_text in chunk.text.lower() else 0.0
        tag_boost = 0.0
        tags = metadata.get("tags", [])
        if isinstance(tags, list) and query_terms:
            tag_boost = policy.metadata_weight * (len(query_terms & {str(tag).lower() for tag in tags}) / max(len(query_terms), 1))
        deprecated_penalty = 0.25 if metadata.get("deprecated") else 0.0
        score = max(0.0, float(chunk.score or 0.0) + exact + tag_boost - deprecated_penalty)
        reranked.append(replace(chunk, score=score, metadata=metadata))

    output = []
    for chunk in sorted(reranked, key=lambda item: (-(item.score or 0.0), item.source, item.chunk_id)):
        if per_source[chunk.source] >= policy.max_chunks_per_source:
            continue
        per_source[chunk.source] += 1
        output.append(chunk)
    return output


def expand_parent_blocks(chunks: Sequence[RetrievedChunk], source: Any | None) -> list[ContextBlock]:
    parent_blocks: dict[str, ContextBlock] = {}
    if source is not None and hasattr(source, "fetch_parent_blocks"):
        fetched = source.fetch_parent_blocks(chunks)
        parent_blocks = {block.block_id: block for block in fetched or []}

    blocks = []
    seen = set()
    for index, chunk in enumerate(chunks, start=1):
        citation = citation_from_chunk(index, chunk)
        parent_id = chunk.parent_id
        if parent_id and parent_id in parent_blocks:
            parent = parent_blocks[parent_id]
            if parent.block_id in seen:
                continue
            seen.add(parent.block_id)
            blocks.append(
                ContextBlock(
                    block_id=parent.block_id,
                    title=parent.title,
                    text=parent.text,
                    source=parent.source,
                    citations=[citation],
                    relevance=chunk.score,
                    metadata=parent.metadata,
                )
            )
            continue
        block_id = parent_id or chunk.chunk_id
        if block_id in seen:
            continue
        seen.add(block_id)
        blocks.append(
            ContextBlock(
                block_id=block_id,
                title=chunk.title,
                text=chunk.text,
                source=chunk.source,
                citations=[citation],
                relevance=chunk.score,
                metadata=chunk.metadata,
            )
        )
    return blocks


def assemble_context(blocks: Sequence[ContextBlock], policy: RetrievalPolicy) -> list[ContextBlock]:
    budget = max(1, policy.max_context_tokens * 4)
    output = []
    used = 0
    for block in sorted(blocks, key=lambda item: (-(item.relevance or 0.0), item.source, item.block_id)):
        if used >= budget:
            break
        remaining = budget - used
        text = block.text[:remaining]
        used += len(text)
        output.append(
            ContextBlock(
                block_id=block.block_id,
                title=block.title,
                text=text,
                source=block.source,
                citations=block.citations,
                relevance=block.relevance,
                metadata=block.metadata,
            )
        )
    return output


def build_prompt(question: str, context: Sequence[ContextBlock]) -> str:
    lines = [
        "Answer using only the provided context.",
        "Cite facts with citation ids.",
        "",
        f"Question: {question}",
        "",
        "Context:",
    ]
    for block in context:
        citation_ids = ", ".join(citation.citation_id for citation in block.citations)
        lines.append(f"[{citation_ids}] {block.title or block.block_id}: {block.text}")
    return "\n".join(lines)


def source_capabilities(source: Any) -> dict[str, bool]:
    if hasattr(source, "capabilities"):
        capabilities = source.capabilities
        return dict(capabilities() if callable(capabilities) else capabilities)
    return {
        "keyword": hasattr(source, "search_keyword"),
        "vector": hasattr(source, "search_vector"),
        "parent_fetch": hasattr(source, "fetch_parent_blocks"),
        "index_request": hasattr(source, "request_index"),
    }


def default_source(name: str = "default") -> InMemoryRagSource:
    return InMemoryRagSource(
        name,
        chunks=[
            RetrievedChunk(
                chunk_id=f"{name}:overview",
                text=(
                    "MVP search stub result for tests and smoke examples. "
                    "Real project adapters can register keyword, vector, and parent fetch ports."
                ),
                source=name,
                score=1.0,
                title="Default AI source",
                metadata={"tags": ["ai", "rag"]},
            )
        ],
        fallback_on_empty=True,
    )


def _normalize_scores(chunks: Sequence[RetrievedChunk]) -> dict[str, float]:
    max_score = max((float(chunk.score or 0.0) for chunk in chunks), default=0.0)
    if max_score <= 0:
        return {chunk.chunk_id: 0.0 for chunk in chunks}
    return {chunk.chunk_id: float(chunk.score or 0.0) / max_score for chunk in chunks}


def _chunk_from_hit(hit: SearchHit, query: str) -> RetrievedChunk:
    del query
    return RetrievedChunk(
        chunk_id=hit.chunk_id,
        text=hit.excerpt,
        source=hit.source,
        score=hit.score,
        parent_id=hit.parent_id,
        section_path=hit.section_path,
        title=hit.title,
        metadata=hit.metadata,
        score_breakdown=hit.score_breakdown,
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
