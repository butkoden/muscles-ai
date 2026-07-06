# muscles-ai RAG toolkit

`muscles-ai` provides framework-level RAG orchestration without owning project
databases or protocol routes.

## Owned by muscles-ai

- DTOs: `SearchQuery`, `RetrievedChunk`, `ContextBlock`, `Citation`,
  `SearchResult`, `ContextResult`, `AskResult`.
- Ports: `VectorSearchPort`, `KeywordSearchPort`, `ParentFetchPort`,
  `IndexRequestPort`, `LLMProvider`.
- Runtime source registry.
- Deterministic hybrid merge and rerank.
- Parent-block context expansion.
- Prompt assembly.
- Fake/in-memory implementations for tests and examples.

## Not owned by muscles-ai

- SQL sessions, migrations, DSNs or tables.
- Qdrant/Elasticsearch/OpenSearch collections and mappings.
- Document parsing and chunking.
- HTTP, MCP, SSE, JSON-RPC or CLI routes.
- Telemetry exporters.

## Default flow

```text
SearchQuery
  -> source keyword/vector ports
  -> score normalization and merge
  -> deterministic rerank without LLM
  -> parent block expansion
  -> context budget trim
  -> prompt
  -> LLMProvider
  -> AskResult with citations
```

The default reranker uses deterministic signals only: keyword/vector scores,
exact phrase match, tag metadata, deprecated penalties and source diversity.

## Actions

- `ai.search` retrieves chunks and never calls an LLM.
- `ai.retrieve_context` returns block-level context and citations.
- `ai.ask` builds prompt context and calls the configured `LLMProvider`.
- `ai.index.request` delegates to a project adapter and does not perform long
  indexing synchronously.
- `ai.inspect` and `ai.doctor` return safe machine-readable diagnostics.

`ai.documents.inspect` remains as a compatibility alias for `ai.source.inspect`.
