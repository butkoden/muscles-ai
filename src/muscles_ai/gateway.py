from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .contracts import LLMResponse, sanitize_mapping
from .models import (
    EmbeddingRequest,
    EmbeddingResult,
    ImageGenerationRequest,
    ImageGenerationResult,
    ModelCapability,
    ModelConfig,
    ModelProviderConfig,
    ModelRequest,
    ModelResponse,
    TextGenerationRequest,
    TextGenerationResult,
)


class ModelGatewayError(RuntimeError):
    """Base error for model selection and adapter failures."""


class ModelNotFoundError(ModelGatewayError, LookupError):
    pass


class ProviderNotFoundError(ModelGatewayError, LookupError):
    pass


class OptionalProviderDependencyError(ModelGatewayError, ImportError):
    pass


class ModelProviderAdapter(Protocol):
    @property
    def capabilities(self) -> frozenset[str]:
        ...

    def invoke(self, request: ModelRequest, *, model_id: str | None, options: Mapping[str, Any]) -> ModelResponse:
        ...

    def inspect(self) -> Mapping[str, Any]:
        ...


class ModelProviderFactory(Protocol):
    def create(self, provider: ModelProviderConfig, model: ModelConfig) -> ModelProviderAdapter:
        ...


ModelHandler = Callable[[ModelRequest, str | None], ModelResponse]


class PythonModelAdapter:
    """Adapter for project-owned callables and arbitrary local runtimes."""

    def __init__(self, handlers: Mapping[str, ModelHandler]) -> None:
        self._handlers = dict(handlers)

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(self._handlers)

    def invoke(self, request: ModelRequest, *, model_id: str | None, options: Mapping[str, Any]) -> ModelResponse:
        del options
        handler = self._handlers.get(request.capability)
        if handler is None:
            raise ValueError(f"Python model adapter does not support capability '{request.capability}'")
        return handler(request, model_id)

    def inspect(self) -> Mapping[str, Any]:
        return {"type": "python", "capabilities": sorted(self.capabilities), "status": "ready"}


class FakeModelAdapter(PythonModelAdapter):
    def __init__(self, *, answer_prefix: str = "Fake answer") -> None:
        def generate(request: ModelRequest, model_id: str | None) -> TextGenerationResult:
            del request
            return TextGenerationResult(
                text=f"{answer_prefix} using model {model_id or 'default'}.",
                model=model_id,
                metadata={"provider": "fake"},
            )

        def image(request: ModelRequest, model_id: str | None) -> ImageGenerationResult:
            del request
            from .models import Artifact

            return ImageGenerationResult(
                artifacts=[Artifact(media_type="image/png", data=b"fake-image")],
                model=model_id,
                metadata={"provider": "fake"},
            )

        def embed(request: ModelRequest, model_id: str | None) -> EmbeddingResult:
            count = len(getattr(request, "texts", []))
            return EmbeddingResult(vectors=[[0.0] for _ in range(count)], model=model_id, metadata={"provider": "fake"})

        super().__init__(
            {
                ModelCapability.TEXT_GENERATE: generate,
                ModelCapability.IMAGE_GENERATE: image,
                ModelCapability.TEXT_EMBED: embed,
            }
        )

    def inspect(self) -> Mapping[str, Any]:
        return {"type": "fake", "capabilities": sorted(self.capabilities), "status": "ready"}


class NoopModelAdapter(PythonModelAdapter):
    def __init__(self, *, provider_name: str = "noop", model_name: str | None = None) -> None:
        def generate(request: ModelRequest, model_id: str | None) -> TextGenerationResult:
            del request
            selected_model = model_id or model_name or "default"
            return TextGenerationResult(
                text=f"AI runtime generated a deterministic answer (provider={provider_name}, model={selected_model})",
                model=selected_model,
                metadata={"provider": provider_name, "model": selected_model},
            )

        super().__init__({ModelCapability.TEXT_GENERATE: generate})

    def inspect(self) -> Mapping[str, Any]:
        return {"type": "noop", "capabilities": sorted(self.capabilities), "status": "ready"}


class _NoopFactory:
    def create(self, provider: ModelProviderConfig, model: ModelConfig) -> ModelProviderAdapter:
        return NoopModelAdapter(
            provider_name=str(provider.options.get("provider_name", provider.type)),
            model_name=model.model_id,
        )


class _FakeFactory:
    def create(self, provider: ModelProviderConfig, model: ModelConfig) -> ModelProviderAdapter:
        del model
        return FakeModelAdapter(answer_prefix=str(provider.options.get("answer_prefix", "Fake answer")))


class _PythonFactory:
    def create(self, provider: ModelProviderConfig, model: ModelConfig) -> ModelProviderAdapter:
        del model
        adapter = provider.options.get("adapter")
        if adapter is None or not hasattr(adapter, "invoke"):
            raise ValueError("python provider requires an adapter object in options['adapter']")
        return adapter


class ModelProviderCatalog:
    def __init__(self) -> None:
        self._factories: dict[str, ModelProviderFactory] = {}

    @classmethod
    def with_defaults(cls) -> "ModelProviderCatalog":
        catalog = cls()
        catalog.register("noop", _NoopFactory())
        catalog.register("fake", _FakeFactory())
        catalog.register("python", _PythonFactory())

        # These modules only import optional SDKs when a configured model is used.
        from .providers.llama_cpp import LlamaCppModelFactory
        from .providers.openai import OpenAIModelFactory

        catalog.register("llama_cpp", LlamaCppModelFactory())
        catalog.register("openai", OpenAIModelFactory())
        return catalog

    def register(self, provider_type: str, factory: ModelProviderFactory) -> None:
        if not provider_type:
            raise ValueError("provider_type must not be empty")
        if provider_type in self._factories:
            raise ValueError(f"Model provider type '{provider_type}' is already registered")
        self._factories[provider_type] = factory

    def create(self, provider: ModelProviderConfig, model: ModelConfig) -> ModelProviderAdapter:
        factory = self._factories.get(provider.type)
        if factory is None:
            raise ProviderNotFoundError(f"Model provider type '{provider.type}' is not registered")
        return factory.create(provider, model)

    def names(self) -> list[str]:
        return sorted(self._factories)


@dataclass
class _ModelBinding:
    config: ModelConfig
    adapter: ModelProviderAdapter | None = None


class ModelGateway:
    """Universal in-process facade for configured model capabilities."""

    def __init__(self, *, catalog: ModelProviderCatalog | None = None) -> None:
        self.catalog = catalog or ModelProviderCatalog.with_defaults()
        self._providers: dict[str, ModelProviderConfig] = {}
        self._models: dict[str, _ModelBinding] = {}
        self._defaults: dict[str, str] = {}

    @classmethod
    def from_legacy(
        cls,
        *,
        provider: str = "noop",
        model_name: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> "ModelGateway":
        gateway = cls()
        provider_type = provider if provider in gateway.catalog.names() else "noop"
        provider_options = dict(options or {})
        provider_options.setdefault("provider_name", provider)
        gateway.register_provider("legacy", provider_type, options=provider_options)
        gateway.register_model_config(
            "text.default",
            provider="legacy",
            model_id=model_name,
            capabilities=[ModelCapability.TEXT_GENERATE],
        )
        gateway.set_default(ModelCapability.TEXT_GENERATE, "text.default")
        return gateway

    @classmethod
    def from_config(
        cls,
        *,
        providers: Mapping[str, Mapping[str, Any]] | None = None,
        models: Mapping[str, Mapping[str, Any]] | None = None,
        defaults: Mapping[str, str] | None = None,
        catalog: ModelProviderCatalog | None = None,
    ) -> "ModelGateway":
        gateway = cls(catalog=catalog)
        for name, raw in (providers or {}).items():
            gateway.register_provider(name, str(raw.get("type", name)), options=raw.get("options", {}))
        for name, raw in (models or {}).items():
            gateway.register_model_config(
                name,
                provider=str(raw.get("provider", "")),
                model_id=raw.get("model", raw.get("model_id")),
                capabilities=raw.get("capabilities", []),
                options=raw.get("options", {}),
            )
        for capability, model in (defaults or {}).items():
            gateway.set_default(str(capability), str(model))
        return gateway

    def register_provider(self, name: str, provider_type: str, *, options: Mapping[str, Any] | None = None) -> None:
        if name in self._providers:
            raise ValueError(f"Model provider '{name}' is already registered")
        self._providers[name] = ModelProviderConfig(name=name, type=provider_type, options=options or {})

    def register_model(
        self,
        name: str,
        adapter: ModelProviderAdapter,
        *,
        model_id: str | None = None,
        capabilities: list[str] | None = None,
        default: bool = False,
        provider: str = "custom",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered")
        declared = frozenset(capabilities or adapter.capabilities)
        config = ModelConfig(name=name, provider=provider, model_id=model_id, capabilities=declared, options=options or {})
        self._models[name] = _ModelBinding(config=config, adapter=adapter)
        if default:
            for capability in declared:
                self.set_default(capability, name)

    def register_model_config(
        self,
        name: str,
        *,
        provider: str,
        model_id: str | None = None,
        capabilities: list[str] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered")
        self._models[name] = _ModelBinding(
            config=ModelConfig(
                name=name,
                provider=provider,
                model_id=model_id,
                capabilities=frozenset(capabilities or []),
                options=options or {},
            )
        )

    def set_default(self, capability: str, model: str) -> None:
        if model not in self._models:
            raise ModelNotFoundError(f"Model '{model}' is not registered")
        self._defaults[capability] = model

    def invoke(self, request: ModelRequest, *, model: str | None = None) -> ModelResponse:
        model_name = model or self._defaults.get(request.capability)
        if not model_name:
            raise ModelNotFoundError(f"No model is configured for capability '{request.capability}'")
        binding = self._models.get(model_name)
        if binding is None:
            raise ModelNotFoundError(f"Model '{model_name}' is not registered")
        if request.capability not in binding.config.capabilities:
            raise ValueError(
                f"Model '{model_name}' does not support capability '{request.capability}'"
            )
        adapter = self._resolve_adapter(binding)
        if request.capability not in adapter.capabilities:
            raise ValueError(
                f"Provider adapter for model '{model_name}' does not support capability '{request.capability}'"
            )
        return adapter.invoke(
            request,
            model_id=binding.config.model_id,
            options={**dict(binding.config.options), **dict(getattr(request, "options", {}) or {})},
        )

    def generate_text(self, prompt: str, *, model: str | None = None, options: Mapping[str, Any] | None = None) -> TextGenerationResult:
        result = self.invoke(TextGenerationRequest(prompt=prompt, options=options or {}), model=model)
        if not isinstance(result, TextGenerationResult):
            raise TypeError("text.generate provider returned an incompatible result")
        return result

    def generate_image(self, prompt: str, *, model: str | None = None, options: Mapping[str, Any] | None = None) -> ImageGenerationResult:
        result = self.invoke(ImageGenerationRequest(prompt=prompt, options=options or {}), model=model)
        if not isinstance(result, ImageGenerationResult):
            raise TypeError("image.generate provider returned an incompatible result")
        return result

    def embed(self, texts: list[str], *, model: str | None = None, options: Mapping[str, Any] | None = None) -> EmbeddingResult:
        result = self.invoke(EmbeddingRequest(texts=texts, options=options or {}), model=model)
        if not isinstance(result, EmbeddingResult):
            raise TypeError("text.embed provider returned an incompatible result")
        return result

    def as_llm_provider(self) -> "GatewayTextGenerationProvider":
        return GatewayTextGenerationProvider(self)

    def inspect(self) -> dict[str, Any]:
        return {
            "providers": [
                {"name": config.name, "type": config.type, "options": sanitize_mapping(config.options)}
                for config in self._providers.values()
            ],
            "models": [
                {
                    "name": binding.config.name,
                    "provider": binding.config.provider,
                    "model": binding.config.model_id,
                    "capabilities": sorted(binding.config.capabilities),
                    "adapter_loaded": binding.adapter is not None,
                }
                for binding in self._models.values()
            ],
            "defaults": dict(self._defaults),
            "provider_types": self.catalog.names(),
        }

    def doctor(self) -> dict[str, Any]:
        checks = []
        for binding in self._models.values():
            try:
                adapter = self._resolve_adapter(binding)
                payload = dict(adapter.inspect())
                checks.append({"name": f"ai.model.{binding.config.name}", **sanitize_mapping(payload)})
            except Exception as exc:
                checks.append({"name": f"ai.model.{binding.config.name}", "status": "error", "error": type(exc).__name__})
        return {"status": "ok" if all(item.get("status") in {"ok", "ready", "configured"} for item in checks) else "warning", "checks": checks}

    def _resolve_adapter(self, binding: _ModelBinding) -> ModelProviderAdapter:
        if binding.adapter is None:
            provider = self._providers.get(binding.config.provider)
            if provider is None:
                raise ProviderNotFoundError(f"Model provider '{binding.config.provider}' is not registered")
            binding.adapter = self.catalog.create(provider, binding.config)
        return binding.adapter

class GatewayTextGenerationProvider:
    """Compatibility bridge from ModelGateway to the existing RAG LLM port."""

    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway

    def generate(self, prompt: str, *, options: Mapping[str, Any] | None = None, context=None) -> LLMResponse:
        del context
        result = self.gateway.invoke(TextGenerationRequest(prompt=prompt, options=options or {}))
        if not isinstance(result, TextGenerationResult):
            raise TypeError("text.generate provider returned an incompatible result")
        return LLMResponse(answer=result.text, metadata=result.metadata)
