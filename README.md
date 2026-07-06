# muscles-ai

Framework-level AI package for the Muscles ecosystem.

## Purpose

- Keep reusable AI/RAG contracts, runtimes, and actions in one framework package.
- Provide read-only primitives for question answering and retrieval.
- Stay transport-agnostic: transport adapters (HTTP/CLI/MCP/JSON-RPC/SSE) call `Muscles actions`.

## Ecosystem Position

`muscles-ai` is a framework extension, not an application template and not a
transport adapter. It registers AI-oriented actions in a Muscles app and lets
other packages project those actions through HTTP, CLI, MCP, JSON-RPC or SSE.

Related repositories:

- [`muscles`](https://github.com/butkoden/muscles) - core action contracts, dispatcher, inspect contract and canonical documentation.
- [`muscles-documents`](https://github.com/butkoden/muscles-documents) - document loading/parsing/chunking actions that AI flows can inspect or compose with.
- [`muscles-mcp`](https://github.com/butkoden/muscles-mcp) - MCP projection for AI tools.
- [`muscles-sse`](https://github.com/butkoden/muscles-sse) - streaming projection for long AI output.
- [`muscles-benchmarks`](https://github.com/butkoden/muscles-benchmarks) - regression coverage for AI extension contracts.

## Installation

```bash
pip install git+https://github.com/butkoden/muscles-ai.git
```

The canonical ecosystem install matrix lives in
[`muscles/docs/installation.md`](https://github.com/butkoden/muscles/blob/master/docs/installation.md).

The package expects to be loaded as a Muscles module:

```yaml
modules:
  ai:
    package: muscles_ai
    providers:
      provider: "noop"
    transports: ["http", "cli", "mcp"]
```

## Public API

Importing package symbols uses lazy `__getattr__` to keep package startup lightweight:

- `AiPackage` — package installer for `init_package` integration.
- `AiRuntime` — runtime container for RAG execution.
- `SearchQuery`, `RetrievedChunk`, `ContextBlock`, `Citation`, `RetrievalPolicy`.
- `SearchResult`, `ContextResult`, `AskResult`, `SearchHit`, `SourceChunk`.
- `VectorSearchPort`, `KeywordSearchPort`, `ParentFetchPort`, `IndexRequestPort`, `LLMProvider`.
- `InMemoryRagSource`, `FakeLLMProvider`, `NoopLLMProvider` for tests and examples.
- `AiConfig`.
- `init_package(app, config)` entry point for Muscles.

## Default actions

The package registers the following actions:

- `ai.ask`
- `ai.search`
- `ai.retrieve_context`
- `ai.sources.list`
- `ai.source.inspect`
- `ai.documents.inspect`
- `ai.index.request`
- `ai.inspect`
- `ai.doctor`

## RAG Toolkit

`muscles-ai` owns RAG orchestration, not project data storage:

```text
query -> retrieval -> merge -> deterministic rerank -> context -> prompt -> LLM -> answer + citations
```

Projects register sources/adapters on `AiRuntime`:

```python
from muscles_ai import InMemoryRagSource, RetrievedChunk

runtime.register_source(
    "kb",
    InMemoryRagSource(
        "kb",
        chunks=[
            RetrievedChunk(
                chunk_id="flowwow-1",
                text="Flowwow backend used PostgreSQL and Kafka.",
                source="kb",
                parent_id="flowwow",
                title="Flowwow",
            )
        ],
    ),
)
```

Real projects can implement the same ports over Qdrant, PostgreSQL, Elasticsearch
or another store. This package does not own DSNs, collections, mappings,
migrations or document parsing.

## Notes

- `muscles-ai` intentionally does not open HTTP routes.
- Runtime clients must be registered in DI and used from actions/context, not kept in `ApplicationRegistry`.
- Transport packages should discover `ai.*` actions through `inspect_application(app)` and execute through `ActionDispatcher`.
- Document ingestion belongs in `muscles-documents`; AI should consume document contracts instead of duplicating parsers.
- Telemetry is resolved through the neutral Muscles `TelemetryProvider`; this
  package does not import `muscles-otel` directly.

## Telemetry

When a project registers a `TelemetryProvider`, `muscles-ai` emits safe spans:

- `muscles.ai.embed`
- `muscles.ai.retrieve`
- `muscles.ai.rerank`
- `muscles.ai.prompt.build`
- `muscles.ai.generate`
- `muscles.ai.answer`

Allowed attributes include provider/model names, retriever name, retrieved
document count and citation count. Raw queries, prompts, answers, excerpts,
chunks, request bodies and API keys must not be stored in span attributes.

## Examples

### Direct package initialization

Run an end-to-end smoke scenario with `muscles_ai` actions:

```bash
PYTHONPATH=src python examples/run_ai_smoke.py
```

### Action calls through a dispatcher

```bash
PYTHONPATH=src python examples/run_ai_configured.py
```

Both examples:

- initialize the package via `init_package(app, config)`;
- register all ai actions;
- call actions through `ActionDispatcher`;
- demonstrate the neutral telemetry provider hook without requiring
  `muscles-otel`.
