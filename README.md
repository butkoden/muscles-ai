# muscles-ai

Framework-level AI package for the Muscles ecosystem.

## Purpose

- Keep reusable AI/RAG contracts, runtimes, and actions in one framework package.
- Provide read-only primitives for question answering and retrieval.
- Stay transport-agnostic: transport adapters (HTTP/CLI/MCP/JSON-RPC/SSE) call `Muscles actions`.

## Installation

```bash
pip install muscles-ai
```

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
- `AiConfig`, `AskRequest`, `SearchResult`, `AiAnswer`.
- `init_package(app, config)` entry point for Muscles.

## Default actions

The package registers the following actions:

- `ai.ask`
- `ai.search`
- `ai.sources.list`
- `ai.documents.inspect`
- `ai.index.request`
- `ai.inspect`
- `ai.doctor`

## Notes

- `muscles-ai` intentionally does not open HTTP routes.
- Runtime clients must be registered in DI and used from actions/context, not kept in `ApplicationRegistry`.
