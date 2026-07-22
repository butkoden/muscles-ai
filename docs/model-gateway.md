# Model Gateway

`ModelGateway` is the model integration boundary for Muscles applications. It
keeps business actions independent from OpenAI SDKs, local inference libraries,
and model-specific request formats.

## Design

```text
business action
  -> ModelGateway.invoke(typed request)
  -> named model and capability routing
  -> provider adapter
  -> external SDK/API or local in-process runtime
```

There is no mandatory Ollama or model server. A provider may call an external
API directly, or it may load a local model in the application process.

## Typed capabilities

The gateway is one public facade, but requests are typed by capability:

- `TextGenerationRequest` -> `TextGenerationResult`;
- `ImageGenerationRequest` -> `ImageGenerationResult`;
- `EmbeddingRequest` -> `EmbeddingResult`;
- future requests can cover reranking, transcription, speech synthesis and
  other operations without changing existing request contracts.

Do not expose a generic `execute(dict)` to business code. A provider declares
the capabilities it supports, and the registry rejects unsupported operations
before calling the backend.

## Configuration

Providers describe SDK/runtime adapters. Models describe named model aliases and
their capabilities:

```yaml
modules:
  ai:
    providers:
      local:
        type: llama_cpp
        options:
          model_path: ./models/text.gguf
      images:
        type: openai
        options:
          api_key_env: OPENAI_API_KEY
    models:
      text.local:
        provider: local
        model: text-model
        capabilities: [text.generate]
      image.remote:
        provider: images
        model: dall-e-3
        capabilities: [image.generate]
    defaults:
      text.generate: text.local
```

Only configured model aliases may be selected by an action. Arbitrary SDK
options, provider URLs and credentials must not arrive through public action
payloads.

## Optional providers

The base package keeps heavyweight dependencies optional:

```bash
pip install "muscles-ai[openai]"
pip install "muscles-ai[llama-cpp]"
```

The `python` adapter is dependency-free and is intended for project-owned
Transformers, Diffusers, MLX or other model objects. Provider modules import
their SDK lazily, so unused backends do not affect package startup.

For example, the OpenAI adapter can use an image model such as `dall-e-3` by
calling the OpenAI API directly. That is a remote provider call, but it still
does not require an Ollama or application-side model server.

## RAG compatibility

The existing `LLMProvider` contract is bridged to the gateway's
`text.generate` capability. Existing `ai.ask` and RAG flows continue to work;
new business actions can resolve `ModelGateway` from Muscles DI and choose a
configured alias explicitly.

```python
from muscles_ai import ModelGateway, TextGenerationRequest

gateway = context.application.container.resolve(ModelGateway)
result = gateway.invoke(
    TextGenerationRequest(prompt="Summarize this document"),
    model="text.local",
)
```

## Artifact ownership

Image and other binary results use `Artifact`. The adapter normalizes bytes or a
provider URI; the application decides whether to return, cache, or persist the
artifact in an object store. Core does not write files or own model output
storage.
