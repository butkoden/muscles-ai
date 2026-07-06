from __future__ import annotations

"""Smoke example for muscles-ai package-level initialization and usage.

Run:
  PYTHONPATH=src python examples/run_ai_smoke.py
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

from muscles import ActionDispatcher, DependencyContainer, TelemetryProvider, inspect_application
from muscles_ai import ContextBlock, InMemoryRagSource, RetrievedChunk, init_package


class MemoryTelemetry:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        self.records.append((name, dict(attributes)))
        yield


def main() -> None:
    telemetry = MemoryTelemetry()
    app = SimpleNamespace(container=DependencyContainer())
    app.container.register(TelemetryProvider, lambda: telemetry)

    runtime = init_package(
        app,
        {
            "key": "ai",
            "provider": "fake",
            "model_name": "stub-mini",
            "top_k_default": 2,
            "top_k_max": 5,
            "options": {"temperature": 0.0},
        },
    )
    runtime.register_source(
        "manual",
        InMemoryRagSource(
            "manual",
            chunks=[
                RetrievedChunk(
                    chunk_id="manual:flowwow",
                    text="Flowwow backend used PostgreSQL, Kafka and deterministic delivery workflows.",
                    source="manual",
                    parent_id="flowwow",
                    title="Flowwow backend",
                    metadata={"tags": ["backend", "highload"]},
                )
            ],
            parents={
                "flowwow": ContextBlock(
                    block_id="flowwow",
                    title="Flowwow",
                    text="Flowwow parent block: backend architecture, PostgreSQL, Kafka and delivery reliability.",
                    source="manual",
                    citations=[],
                    relevance=0.0,
                )
            },
        ),
    )

    contract = inspect_application(app)
    action_names = {item["name"] for item in contract["actions"]}
    print("Registered actions:", ", ".join(sorted(action_names)))

    dispatcher = ActionDispatcher(app)

    ask = dispatcher.execute("ai.ask", {"question": "Что известно про Flowwow Kafka?", "top_k": 2, "source": "manual"})
    print("ai.ask ->", ask.value["answer"])
    print("ai.ask citations:", [citation["chunk_id"] for citation in ask.value["citations"]])

    search = dispatcher.execute("ai.search", {"query": "Flowwow Kafka", "top_k": 3, "source": "manual"})
    print("ai.search hits:", len(search.value["hits"]))

    context = dispatcher.execute("ai.retrieve_context", {"query": "Flowwow Kafka", "top_k": 2, "source": "manual"})
    print("ai.retrieve_context blocks:", len(context.value["context"]))

    sources = dispatcher.execute("ai.sources.list", {})
    print("ai.sources.list ->", sources.value)

    inspect_payload = dispatcher.execute("ai.inspect", {})
    print("ai.inspect ->", inspect_payload.value["features"])
    print("telemetry spans:", [name for name, _ in telemetry.records])


if __name__ == "__main__":
    main()
