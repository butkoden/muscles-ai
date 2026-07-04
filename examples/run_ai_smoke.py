from __future__ import annotations

"""Smoke example for muscles-ai package-level initialization and usage.

Run:
  PYTHONPATH=src python examples/run_ai_smoke.py
"""

from types import SimpleNamespace

from muscles import ActionDispatcher, inspect_application
from muscles_ai import init_package


def main() -> None:
    app = SimpleNamespace()
    init_package(
        app,
        {
            "key": "ai",
            "provider": "noop",
            "model_name": "stub-mini",
            "top_k_default": 2,
            "top_k_max": 5,
            "options": {"temperature": 0.0},
        },
    )

    contract = inspect_application(app)
    action_names = {item["name"] for item in contract["actions"]}
    print("Registered actions:", ", ".join(sorted(action_names)))

    dispatcher = ActionDispatcher(app)

    ask = dispatcher.execute("ai.ask", {"question": "Что входит в MVP пакета?", "top_k": 2, "source": "manual"})
    print("ai.ask ->", ask.value["answer"])

    search = dispatcher.execute("ai.search", {"query": "stub", "top_k": 3, "source": "manual"})
    print("ai.search hits:", len(search.value["hits"]))

    sources = dispatcher.execute("ai.sources.list", {})
    print("ai.sources.list ->", sources.value)

    inspect_payload = dispatcher.execute("ai.inspect", {})
    print("ai.inspect ->", inspect_payload.value["features"])


if __name__ == "__main__":
    main()

