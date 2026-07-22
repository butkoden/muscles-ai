from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from .contracts import sanitize_mapping


class ModelCapability:
    """Stable names for operations a model adapter may implement."""

    TEXT_GENERATE: ClassVar[str] = "text.generate"
    TEXT_EMBED: ClassVar[str] = "text.embed"
    TEXT_RERANK: ClassVar[str] = "text.rerank"
    IMAGE_GENERATE: ClassVar[str] = "image.generate"
    IMAGE_EDIT: ClassVar[str] = "image.edit"
    AUDIO_TRANSCRIBE: ClassVar[str] = "audio.transcribe"
    AUDIO_SYNTHESIZE: ClassVar[str] = "audio.synthesize"


@dataclass(frozen=True)
class Artifact:
    """A model-produced binary or remote artifact.

    Providers normalize SDK-specific base64, URLs and local objects to this
    contract. Persistence remains the application's responsibility.
    """

    media_type: str
    data: bytes | None = None
    uri: str | None = None
    filename: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.media_type:
            raise ValueError("Artifact.media_type must not be empty")
        if self.data is None and not self.uri:
            raise ValueError("Artifact requires data or uri")
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


@dataclass(frozen=True)
class TextGenerationRequest:
    prompt: str
    system: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    capability: ClassVar[str] = ModelCapability.TEXT_GENERATE

    def __post_init__(self) -> None:
        if not str(self.prompt or "").strip():
            raise ValueError("TextGenerationRequest.prompt must not be empty")
        object.__setattr__(self, "options", dict(self.options or {}))
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    count: int = 1
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    capability: ClassVar[str] = ModelCapability.IMAGE_GENERATE

    def __post_init__(self) -> None:
        if not str(self.prompt or "").strip():
            raise ValueError("ImageGenerationRequest.prompt must not be empty")
        if int(self.count) < 1:
            raise ValueError("ImageGenerationRequest.count must be positive")
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "options", dict(self.options or {}))
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


@dataclass(frozen=True)
class EmbeddingRequest:
    texts: Sequence[str]
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    capability: ClassVar[str] = ModelCapability.TEXT_EMBED

    def __post_init__(self) -> None:
        normalized = [str(text) for text in self.texts]
        if not normalized or any(not text.strip() for text in normalized):
            raise ValueError("EmbeddingRequest.texts must contain non-empty strings")
        object.__setattr__(self, "texts", normalized)
        object.__setattr__(self, "options", dict(self.options or {}))
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


ModelRequest = TextGenerationRequest | ImageGenerationRequest | EmbeddingRequest


@dataclass(frozen=True)
class TextGenerationResult:
    text: str
    model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


@dataclass(frozen=True)
class ImageGenerationResult:
    artifacts: list[Artifact] = field(default_factory=list)
    model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", list(self.artifacts or []))
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]] = field(default_factory=list)
    model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vectors", [list(vector) for vector in self.vectors or []])
        object.__setattr__(self, "metadata", sanitize_mapping(self.metadata or {}))


ModelResponse = Any


@dataclass(frozen=True)
class ModelProviderConfig:
    name: str
    type: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelProviderConfig.name must not be empty")
        if not self.type:
            raise ValueError("ModelProviderConfig.type must not be empty")
        object.__setattr__(self, "options", dict(self.options or {}))


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model_id: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelConfig.name must not be empty")
        if not self.provider:
            raise ValueError("ModelConfig.provider must not be empty")
        object.__setattr__(self, "capabilities", frozenset(str(item) for item in self.capabilities))
        object.__setattr__(self, "options", dict(self.options or {}))
