from __future__ import annotations

from types import SimpleNamespace

from muscles import ActionDispatcher, inspect_application

from muscles_ai import (
    AiPackage,
    ContextBlock,
    FakeLLMProvider,
    InMemoryRagSource,
    RetrievedChunk,
    RetrievalPolicy,
    SearchQuery,
)
from muscles_ai.runtime import AiRuntime


def _runtime() -> AiRuntime:
    runtime = AiRuntime(key="ai", provider="fake", model_name="fake-rag")
    runtime.register_source(
        "kb",
        InMemoryRagSource(
            "kb",
            chunks=[
                RetrievedChunk(
                    chunk_id="flowwow-1",
                    text="Flowwow backend architecture used PostgreSQL and Kafka for highload delivery.",
                    source="kb",
                    score=0.9,
                    parent_id="flowwow",
                    section_path="Resume / Experience / Flowwow",
                    title="Flowwow backend",
                    metadata={"tags": ["backend", "highload"], "secret_token": "hidden"},
                ),
                RetrievedChunk(
                    chunk_id="ai-1",
                    text="AI resume generator uses retrieval augmented generation and citations.",
                    source="kb",
                    score=0.8,
                    parent_id="ai",
                    section_path="Projects / AI",
                    title="AI resume generator",
                    metadata={"tags": ["ai", "rag"]},
                ),
                RetrievedChunk(
                    chunk_id="old-1",
                    text="Deprecated legacy PHP snippet.",
                    source="kb",
                    score=0.95,
                    section_path="Archive",
                    title="Legacy",
                    metadata={"deprecated": True},
                ),
            ],
            parents={
                "flowwow": ContextBlock(
                    block_id="flowwow",
                    title="Flowwow",
                    text="Full Flowwow parent block: backend architecture, PostgreSQL, Kafka, delivery reliability.",
                    source="kb",
                    citations=[],
                    relevance=0.0,
                    metadata={"company": "Flowwow"},
                ),
                "ai": ContextBlock(
                    block_id="ai",
                    title="AI projects",
                    text="Full AI project parent block: RAG, prompt assembly, citations.",
                    source="kb",
                    citations=[],
                    relevance=0.0,
                    metadata={"kind": "project"},
                ),
            },
            inspect_payload={"status": "ready", "dsn": "postgres://secret@localhost/db"},
        ),
    )
    return runtime


def test_search_query_validation_and_source_registration():
    runtime = _runtime()

    assert runtime.list_sources() == ["default", "documents", "kb"]
    assert runtime.inspect_source("kb")["capabilities"]["keyword"] is True
    assert "dsn" not in repr(runtime.inspect_source("kb"))

    try:
        SearchQuery(text="")
    except ValueError as exc:
        assert "text" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty query should fail")


def test_keyword_vector_and_hybrid_search_are_deterministic_without_llm():
    runtime = _runtime()

    keyword = runtime.search("Flowwow PostgreSQL", source="kb", mode="keyword", top_k=5)
    vector = runtime.search("retrieval citations", source="kb", mode="vector", top_k=5)
    hybrid_a = runtime.search("Flowwow Kafka backend", source="kb", mode="hybrid", top_k=5)
    hybrid_b = runtime.search("Flowwow Kafka backend", source="kb", mode="hybrid", top_k=5)

    assert keyword.hits[0].chunk_id == "flowwow-1"
    assert vector.hits[0].chunk_id == "ai-1"
    assert [hit.chunk_id for hit in hybrid_a.hits] == [hit.chunk_id for hit in hybrid_b.hits]
    assert hybrid_a.hits[0].score_breakdown["keyword"] >= 0
    assert hybrid_a.hits[0].score_breakdown["vector"] >= 0
    assert "secret_token" not in hybrid_a.hits[0].metadata


def test_retrieve_context_expands_parents_preserves_citations_and_budget():
    runtime = _runtime()

    context = runtime.retrieve_context(
        "Flowwow PostgreSQL Kafka",
        source="kb",
        top_k=5,
        policy=RetrievalPolicy(max_context_tokens=12),
    )

    assert context.context
    assert context.context[0].block_id == "flowwow"
    assert context.context[0].citations[0].chunk_id == "flowwow-1"
    assert context.citations[0].source == "kb"
    assert len(context.context[0].text) <= 12 * 4


def test_ask_uses_fake_llm_and_returns_citations_and_no_context_path():
    runtime = _runtime()
    runtime.llm_provider = FakeLLMProvider(answer_prefix="Answer")
    runtime.register_source("empty", InMemoryRagSource("empty", chunks=[]))

    answer = runtime.ask("What did Flowwow use?", source="kb", top_k=3)
    no_context = runtime.ask("Nothing here", source="empty", top_k=3)

    assert answer.answer.startswith("Answer")
    assert answer.citations
    assert answer.context
    assert no_context.answer == "No context available for this question."
    assert no_context.citations == []


def test_actions_expose_search_retrieve_context_ask_and_index_request():
    app = SimpleNamespace()
    runtime = AiPackage().init(app, {"key": "ai", "provider": "fake", "model_name": "fake-rag"})
    runtime.register_source("kb", InMemoryRagSource("kb", chunks=[RetrievedChunk("c1", "Kafka backend", "kb")]))
    dispatcher = ActionDispatcher(app)

    actions = {item["name"] for item in inspect_application(app)["actions"]}
    assert "ai.retrieve_context" in actions
    assert "ai.source.inspect" in actions

    search = dispatcher.execute("ai.search", {"query": "Kafka", "source": "kb", "top_k": 2}).value
    context = dispatcher.execute("ai.retrieve_context", {"query": "Kafka", "source": "kb", "top_k": 2}).value
    answer = dispatcher.execute("ai.ask", {"question": "Kafka?", "source": "kb", "top_k": 2}).value
    inspect = dispatcher.execute("ai.inspect", {}).value
    doctor = dispatcher.execute("ai.doctor", {}).value
    index = dispatcher.execute("ai.index.request", {"source": "kb", "dry_run": True}).value

    assert search["hits"][0]["chunk_id"] == "c1"
    assert context["context"][0]["citations"][0]["chunk_id"] == "c1"
    assert answer["citations"][0]["chunk_id"] == "c1"
    assert inspect["sources"]["count"] >= 1
    assert doctor["status"] == "ok"
    assert index["status"] in {"planned", "not_supported"}
