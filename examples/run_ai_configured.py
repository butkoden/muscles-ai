from __future__ import annotations

"""Configured example with explicit transport restrictions.

Run:
  PYTHONPATH=src python examples/run_ai_configured.py
"""

from types import SimpleNamespace

from muscles import ActionDispatcher, inspect_application
from muscles_ai import init_package


def main() -> None:
    app = SimpleNamespace()
    init_package(
        app,
        {
            "name": "ai",
            "provider": "noop",
            "model_name": "stub-small",
            "transports": ["cli", "http"],
            "top_k_default": 1,
            "top_k_max": 2,
            "options": {"timeout_ms": 1500},
        },
    )

    contract = inspect_application(app)
    print("Loaded packages:", len(contract.get("packages", [])))

    dispatcher = ActionDispatcher(app)
    doctor = dispatcher.execute("ai.doctor", {})
    print("ai.doctor ->", doctor.value)

    documents = dispatcher.execute("ai.sources.list", {})
    if "default" not in documents.value["sources"]:
        raise RuntimeError("Unexpected AI source default list")

    inspected = dispatcher.execute("ai.documents.inspect", {"source": "documents"})
    print("ai.documents.inspect ->", inspected.value)

    index_request = dispatcher.execute("ai.index.request", {})
    print("ai.index.request ->", index_request.value)


if __name__ == "__main__":
    main()
