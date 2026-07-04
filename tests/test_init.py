from __future__ import annotations

from types import SimpleNamespace

from muscles import inspect_application

import muscles_ai
from muscles import ActionDispatcher
from muscles_ai import AiPackage
from muscles_ai.runtime import AiRuntime


def _create_app():
    return SimpleNamespace()


def test_muscles_ai_init_registers_actions_and_runtime():
    package = AiPackage()
    app = _create_app()
    runtime = package.init(app, {"key": "ai", "provider": "noop"})
    assert isinstance(runtime, AiRuntime)
    contract = inspect_application(app)
    action_names = {action["name"] for action in contract["actions"]}
    assert "ai.ask" in action_names
    assert "ai.search" in action_names
    assert "ai.sources.list" in action_names
    dispatcher = ActionDispatcher(app)
    ask_result = dispatcher.execute("ai.ask", {"question": "test", "top_k": 2})
    assert ask_result.value["question"] == "test"
    assert ask_result.value["sources"], "ask should return source diagnostics"


def test_muscles_ai_public_exports():
    assert hasattr(muscles_ai, "AiRuntime")
    assert hasattr(muscles_ai, "AiPackage")
