from __future__ import annotations

from types import SimpleNamespace

from muscles import doctor_application, inspect_application

import muscles_ai
from muscles import ActionDispatcher
from muscles_ai import AiPackage, ModelGateway
from muscles_ai.package import init_package
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
    assert hasattr(muscles_ai, "ModelGateway")


def test_package_registers_model_gateway_in_di():
    app = _create_app()
    runtime = AiPackage().init(app, {"key": "ai", "provider": "fake"})

    assert app.container.resolve(ModelGateway) is runtime.model_gateway


def test_ai_inspection_exposes_machine_readable_architecture_contract():
    app = _create_app()

    init_package(app, {"key": "ai", "provider": "noop"})

    architecture = inspect_application(app)["capabilities"]["ai"]["architecture"]

    assert architecture["role"]
    assert architecture["preferred_patterns"]
    assert architecture["allowed_patterns"]
    assert architecture["forbidden_patterns"]
    assert {rule["id"] for rule in architecture["rules"]} >= {
        "ai.action.metadata",
        "ai.state_change.confirmation",
    }
    assert all({"id", "severity", "summary"} <= set(rule) for rule in architecture["rules"])


def test_ai_actions_expose_safety_metadata_and_index_requires_confirmation():
    app = _create_app()

    init_package(app, {"key": "ai", "provider": "noop"})

    actions = {
        item["name"]: item
        for item in inspect_application(app)["actions"]
        if item["name"] in {"ai.ask", "ai.search", "ai.retrieve_context", "ai.index.request"}
    }

    assert set(actions) == {"ai.ask", "ai.search", "ai.retrieve_context", "ai.index.request"}
    for action in actions.values():
        assert {"ai", "architecture", "mcp"} <= set(action["metadata"])

    index_metadata = actions["ai.index.request"]["metadata"]
    assert index_metadata["architecture"]["state_changing"] is True
    assert index_metadata["architecture"]["requires_confirmation"] is True
    assert index_metadata["mcp"]["read_only"] is False
    assert index_metadata["mcp"]["requires_confirmation"] is True


def test_ai_doctor_enforces_metadata_without_model_or_network_calls():
    app = _create_app()

    init_package(app, {"key": "ai", "provider": "noop"})
    action = app.__muscles_registry__.get_action("ai.index.request")
    action.metadata["architecture"].pop("requires_confirmation")

    doctor = doctor_application(app)["packages"]["ai"]

    assert doctor["status"] == "error"
    failed = [check for check in doctor["checks"] if check["status"] == "failed"]
    assert any(check["id"] == "ai.state_change.confirmation" for check in failed)
