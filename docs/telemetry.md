# muscles-ai telemetry

`muscles-ai` uses the neutral Muscles telemetry contract:

```python
from muscles import resolve_telemetry

telemetry = resolve_telemetry(app)
with telemetry.span("muscles.ai.retrieve"):
    ...
```

The package never imports `muscles_otel`. A project may install any provider
that implements `span(name, **attributes)`, including `muscles-otel`.

## Spans

- `muscles.ai.embed`
- `muscles.ai.retrieve`
- `muscles.ai.rerank`
- `muscles.ai.prompt.build`
- `muscles.ai.generate`
- `muscles.ai.answer`

## Safe attributes

- `ai.provider`
- `ai.model`
- `ai.embedding.model`
- `ai.retriever`
- `ai.documents.retrieved`
- `ai.citations.count`

Do not add raw user queries, prompts, generated answers, excerpts, document
chunks, provider request/response bodies or API keys to span attributes.
