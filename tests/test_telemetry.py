from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from muscles import ActionDispatcher, DependencyContainer, TelemetryProvider

from muscles_ai import AiPackage
from muscles_ai.package import init_package


class RecordingTelemetry:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        self.records.append({"name": name, "attributes": dict(attributes)})
        yield


def _app_with_telemetry() -> tuple[SimpleNamespace, RecordingTelemetry]:
    app = SimpleNamespace(container=DependencyContainer())
    telemetry = RecordingTelemetry()
    app.container.register(TelemetryProvider, lambda: telemetry)
    return app, telemetry


def test_ai_actions_emit_safe_framework_spans_without_otel_dependency():
    app, telemetry = _app_with_telemetry()
    AiPackage().init(app, {"key": "ai", "provider": "noop", "model_name": "stub", "top_k_max": 3})
    dispatcher = ActionDispatcher(app)

    ask = dispatcher.execute(
        "ai.ask",
        {"question": "raw private question", "top_k": 2, "source": "manual"},
    )
    search = dispatcher.execute(
        "ai.search",
        {"query": "raw private search", "top_k": 2, "source": "manual"},
    )
    index_request = dispatcher.execute("ai.index.request", {})

    assert ask.value["sources"]
    assert search.value["hits"]
    assert index_request.value["status"] in {"accepted", "planned", "not_supported"}
    assert {record["name"] for record in telemetry.records} >= {
        "muscles.ai.embed",
        "muscles.ai.retrieve",
        "muscles.ai.rerank",
        "muscles.ai.prompt.build",
        "muscles.ai.generate",
        "muscles.ai.answer",
    }

    for record in telemetry.records:
        attributes = record["attributes"]
        assert "question" not in attributes
        assert "query" not in attributes
        assert "prompt" not in attributes
        assert "answer" not in attributes
        assert "excerpt" not in attributes
        assert "raw private" not in repr(attributes)


def test_ai_init_package_uses_core_lifecycle_when_available():
    app = SimpleNamespace()

    runtime = init_package(app, {"key": "ai", "provider": "noop"})

    assert runtime is app.container.resolve(type(runtime))


def test_ai_source_does_not_import_muscles_otel_directly():
    source_root = Path(__file__).resolve().parents[1] / "src" / "muscles_ai"
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in source_root.rglob("*.py"))

    assert "muscles_otel" not in source_text
