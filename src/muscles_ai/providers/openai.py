from __future__ import annotations

import base64
import os
from typing import Any, Mapping

from ..gateway import ModelProviderAdapter, OptionalProviderDependencyError
from ..models import (
    Artifact,
    EmbeddingRequest,
    EmbeddingResult,
    ImageGenerationRequest,
    ImageGenerationResult,
    ModelCapability,
    ModelConfig,
    ModelProviderConfig,
    ModelRequest,
    TextGenerationRequest,
    TextGenerationResult,
)


class OpenAIModelFactory:
    def create(self, provider: ModelProviderConfig, model: ModelConfig) -> ModelProviderAdapter:
        client = provider.options.get("client")
        return OpenAIModelAdapter(provider.options, model, client=client)


class OpenAIModelAdapter:
    capabilities = frozenset(
        {
            ModelCapability.TEXT_GENERATE,
            ModelCapability.IMAGE_GENERATE,
            ModelCapability.TEXT_EMBED,
        }
    )

    def __init__(self, provider_options: Mapping[str, Any], model: ModelConfig, *, client: Any = None) -> None:
        self.provider_options = dict(provider_options)
        self.model = model
        self._client = client

    def invoke(self, request: ModelRequest, *, model_id: str | None, options: Mapping[str, Any]) -> Any:
        client = self._ensure_client()
        selected_model = model_id or self.model.model_id
        if not selected_model:
            raise ValueError("openai provider requires a model id")
        if request.capability == ModelCapability.TEXT_GENERATE:
            return self._generate_text(client, request, selected_model, options)
        if request.capability == ModelCapability.IMAGE_GENERATE:
            return self._generate_image(client, request, selected_model, options)
        if request.capability == ModelCapability.TEXT_EMBED:
            return self._embed(client, request, selected_model, options)
        raise ValueError(f"openai provider does not support capability '{request.capability}'")

    def inspect(self) -> Mapping[str, Any]:
        if self._client is None:
            try:
                import openai  # type: ignore[import-not-found,unused-import]
            except ImportError:
                return {
                    "type": "openai",
                    "status": "error",
                    "error": "optional_dependency",
                    "capabilities": sorted(self.capabilities),
                }
        return {
            "type": "openai",
            "status": "ready" if self._client is not None else "configured",
            "capabilities": sorted(self.capabilities),
        }

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise OptionalProviderDependencyError(
                "Provider 'openai' requires the optional dependency: pip install 'muscles-ai[openai]'"
            ) from exc

        options = dict(self.provider_options)
        api_key = options.pop("api_key", None)
        api_key_env = options.pop("api_key_env", "OPENAI_API_KEY")
        if api_key is None and api_key_env:
            api_key = os.getenv(str(api_key_env))
        client_options: dict[str, Any] = {}
        for key in ("base_url", "organization", "project", "timeout", "max_retries"):
            if key in options:
                client_options[key] = options[key]
        self._client = OpenAI(api_key=api_key, **client_options)
        return self._client

    @staticmethod
    def _generate_text(client: Any, request: TextGenerationRequest, model_id: str, options: Mapping[str, Any]) -> TextGenerationResult:
        request_options = dict(options)
        if hasattr(client, "responses"):
            response_options = {"model": model_id, "input": request.prompt, **request_options}
            if request.system:
                response_options["instructions"] = request.system
            response = client.responses.create(**response_options)
            text = getattr(response, "output_text", None)
            if text is None:
                text = _extract_response_text(response)
        else:
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": request.prompt})
            response = client.chat.completions.create(model=model_id, messages=messages, **request_options)
            text = response.choices[0].message.content
        return TextGenerationResult(text=str(text or ""), model=model_id, metadata={"provider": "openai", "model": model_id})

    @staticmethod
    def _generate_image(client: Any, request: ImageGenerationRequest, model_id: str, options: Mapping[str, Any]) -> ImageGenerationResult:
        response = client.images.generate(model=model_id, prompt=request.prompt, n=request.count, **dict(options))
        artifacts = []
        for item in getattr(response, "data", []) or []:
            encoded = getattr(item, "b64_json", None)
            uri = getattr(item, "url", None)
            data = base64.b64decode(encoded) if encoded else None
            if data is not None or uri:
                artifacts.append(Artifact(media_type="image/png", data=data, uri=uri))
        return ImageGenerationResult(artifacts=artifacts, model=model_id, metadata={"provider": "openai", "model": model_id})

    @staticmethod
    def _embed(client: Any, request: EmbeddingRequest, model_id: str, options: Mapping[str, Any]) -> EmbeddingResult:
        response = client.embeddings.create(input=list(request.texts), model=model_id, **dict(options))
        data = sorted(getattr(response, "data", []) or [], key=lambda item: int(getattr(item, "index", 0)))
        vectors = [list(getattr(item, "embedding", [])) for item in data]
        return EmbeddingResult(vectors=vectors, model=model_id, metadata={"provider": "openai", "model": model_id})


def _extract_response_text(response: Any) -> str:
    output = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "".join(parts)
