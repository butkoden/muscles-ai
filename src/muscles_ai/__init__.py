from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .actions import AskActionResult, SearchActionResult
    from .config import AiConfig
    from .contracts import (
        AskResult,
        Citation,
        ContextBlock,
        ContextResult,
        RetrievedBlock,
        RetrievedChunk,
        RetrievalPolicy,
        SearchHit,
        SearchQuery,
        SearchResult,
        SourceChunk,
    )
    from .memory import FakeLLMProvider, InMemoryRagSource, NoopLLMProvider
    from .gateway import (
        GatewayTextGenerationProvider,
        ModelGateway,
        ModelGatewayError,
        ModelNotFoundError,
        ModelProviderAdapter,
        ModelProviderCatalog,
        ModelProviderFactory,
        OptionalProviderDependencyError,
        ProviderNotFoundError,
        PythonModelAdapter,
    )
    from .models import (
        Artifact,
        EmbeddingRequest,
        EmbeddingResult,
        ImageGenerationRequest,
        ImageGenerationResult,
        ModelCapability,
        ModelConfig,
        ModelProviderConfig,
        TextGenerationRequest,
        TextGenerationResult,
    )
    from .package import AiPackage
    from .ports import (
        IndexRequestPort,
        KeywordSearchPort,
        LLMProvider,
        ParentFetchPort,
        SectionSearchPort,
        VectorSearchPort,
    )
    from .runtime import AiRuntime

__all__ = [
    "AiConfig",
    "init_package",
    "AiPackage",
    "AiRuntime",
    "SearchQuery",
    "RetrievedChunk",
    "RetrievedBlock",
    "ContextBlock",
    "Citation",
    "RetrievalPolicy",
    "SearchResult",
    "ContextResult",
    "AskResult",
    "SearchHit",
    "SourceChunk",
    "InMemoryRagSource",
    "FakeLLMProvider",
    "NoopLLMProvider",
    "VectorSearchPort",
    "KeywordSearchPort",
    "SectionSearchPort",
    "ParentFetchPort",
    "IndexRequestPort",
    "LLMProvider",
    "AskActionResult",
    "SearchActionResult",
    "ModelGateway",
    "ModelGatewayError",
    "ModelNotFoundError",
    "ProviderNotFoundError",
    "OptionalProviderDependencyError",
    "ModelProviderAdapter",
    "ModelProviderFactory",
    "ModelProviderCatalog",
    "GatewayTextGenerationProvider",
    "PythonModelAdapter",
    "ModelCapability",
    "ModelConfig",
    "ModelProviderConfig",
    "Artifact",
    "TextGenerationRequest",
    "TextGenerationResult",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "EmbeddingRequest",
    "EmbeddingResult",
]


def __getattr__(name: str):
    if name == "AiConfig":
        from .config import AiConfig

        return AiConfig
    if name == "AiPackage":
        from .package import AiPackage

        return AiPackage
    if name == "init_package":
        from .package import init_package

        return init_package
    if name == "AiRuntime":
        from .runtime import AiRuntime

        return AiRuntime
    if name in {
        "SearchQuery",
        "RetrievedChunk",
        "RetrievedBlock",
        "ContextBlock",
        "Citation",
        "RetrievalPolicy",
        "SearchResult",
        "ContextResult",
        "AskResult",
        "SearchHit",
        "SourceChunk",
    }:
        from . import contracts

        return getattr(contracts, name)
    if name in {"InMemoryRagSource", "FakeLLMProvider", "NoopLLMProvider"}:
        from . import memory

        return getattr(memory, name)
    if name in {
        "ModelGateway",
        "ModelGatewayError",
        "ModelNotFoundError",
        "ProviderNotFoundError",
        "OptionalProviderDependencyError",
        "ModelProviderAdapter",
        "ModelProviderFactory",
        "ModelProviderCatalog",
        "GatewayTextGenerationProvider",
        "PythonModelAdapter",
    }:
        from . import gateway

        return getattr(gateway, name)
    if name in {
        "ModelCapability",
        "ModelConfig",
        "ModelProviderConfig",
        "Artifact",
        "TextGenerationRequest",
        "TextGenerationResult",
        "ImageGenerationRequest",
        "ImageGenerationResult",
        "EmbeddingRequest",
        "EmbeddingResult",
    }:
        from . import models

        return getattr(models, name)
    if name in {
        "VectorSearchPort",
        "KeywordSearchPort",
        "SectionSearchPort",
        "ParentFetchPort",
        "IndexRequestPort",
        "LLMProvider",
    }:
        from . import ports

        return getattr(ports, name)
    if name == "AskActionResult":
        from .actions import AskActionResult

        return AskActionResult
    if name == "SearchActionResult":
        from .actions import SearchActionResult

        return SearchActionResult
    raise AttributeError(name)


def __dir__():
    return sorted(__all__)
