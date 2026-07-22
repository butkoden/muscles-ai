from __future__ import annotations

from typing import Any, Mapping

from ..gateway import ModelProviderAdapter, OptionalProviderDependencyError
from ..models import ModelCapability, ModelConfig, ModelProviderConfig, ModelRequest, TextGenerationRequest, TextGenerationResult


class LlamaCppModelFactory:
    def create(self, provider: ModelProviderConfig, model: ModelConfig) -> ModelProviderAdapter:
        return LlamaCppModelAdapter(provider.options, model)


class LlamaCppModelAdapter:
    capabilities = frozenset({ModelCapability.TEXT_GENERATE})

    def __init__(self, provider_options: Mapping[str, Any], model: ModelConfig) -> None:
        self.provider_options = dict(provider_options)
        self.model = model
        self._model: Any = None

    def invoke(self, request: ModelRequest, *, model_id: str | None, options: Mapping[str, Any]) -> TextGenerationResult:
        if request.capability != ModelCapability.TEXT_GENERATE:
            raise ValueError(f"llama_cpp provider does not support capability '{request.capability}'")
        runtime = self._ensure_model()
        request = request if isinstance(request, TextGenerationRequest) else TextGenerationRequest(prompt=str(request))
        model_options = dict(options)
        if hasattr(runtime, "create_chat_completion"):
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": request.prompt})
            response = runtime.create_chat_completion(messages=messages, **model_options)
            choice = response["choices"][0]
            text = choice.get("message", {}).get("content") or choice.get("text", "")
        else:
            response = runtime(request.prompt, **model_options)
            text = response["choices"][0].get("text", "")
        return TextGenerationResult(text=str(text), model=model_id or self.model.model_id, metadata={"provider": "llama_cpp"})

    def inspect(self) -> Mapping[str, Any]:
        if self._model is None:
            try:
                import llama_cpp  # type: ignore[import-not-found,unused-import]
            except ImportError:
                return {
                    "type": "llama_cpp",
                    "status": "error",
                    "error": "optional_dependency",
                    "capabilities": sorted(self.capabilities),
                }
        return {
            "type": "llama_cpp",
            "status": "ready" if self._model is not None else "configured",
            "capabilities": sorted(self.capabilities),
        }

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as exc:
            raise OptionalProviderDependencyError(
                "Provider 'llama_cpp' requires the optional dependency: pip install 'muscles-ai[llama-cpp]'"
            ) from exc
        model_path = self.provider_options.get("model_path")
        if not model_path:
            raise ValueError("llama_cpp provider requires options.model_path")
        llama_options = dict(self.provider_options.get("llama_options", {}) or {})
        self._model = Llama(model_path=str(model_path), **llama_options)
        return self._model
