from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents import graph, nodes
from app.agents.state import create_initial_state
from app.api import chat as chat_api
from app.api.deps import get_current_user
from app.core.config import DEFAULT_SECRET_KEY, Settings, validate_security_settings
from app.core.database import Base, get_db
from app.core.security import decode_access_token
from app.core.security import settings as security_settings
from app.models import ChatSession, Message, User
from app.schemas.chat import ChatRequest


class FailingRouterLlm:
    def invoke(self, *_args, **_kwargs):
        raise RuntimeError("router offline")


def test_router_failure_is_blocked_instead_of_direct_answer(monkeypatch):
    monkeypatch.setattr(nodes, "HelloAgentsLLM", FailingRouterLlm)
    state = create_initial_state("PAML 的 omega 是什么？")

    routed = nodes.router_node(state)
    merged = {**state, **routed}
    answered = nodes.answer_node(merged)

    assert routed["intent"] is None
    assert routed["router_error"]["type"] == "router_unavailable"
    assert "未执行知识检索" in answered["final_answer"]
    assert answered["sources"] == []


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"intent":"unknown","tool_name":null,"tool_input":{}}',
        '{"intent":"direct_answer","tool_name":null,"arguments":{}}',
    ],
)
def test_invalid_router_contract_is_blocked(monkeypatch, response):
    class InvalidRouterLlm:
        def invoke(self, *_args, **_kwargs):
            return response

    monkeypatch.setattr(nodes, "HelloAgentsLLM", InvalidRouterLlm)
    state = create_initial_state("专业问题")

    result = nodes.router_node(state)

    assert result["router_error"]["type"] == "router_invalid_response"
    assert result["tool_name"] is None


def test_stream_mode_does_not_execute_graph_twice(monkeypatch):
    class FakeGraph:
        invoke_calls = 0

        def stream(self, initial_state, stream_mode):
            assert stream_mode == "values"
            yield {**initial_state, "final_answer": "done"}

        def invoke(self, _initial_state):
            self.invoke_calls += 1
            raise AssertionError("stream mode must not invoke the graph again")

    fake = FakeGraph()
    monkeypatch.setattr(graph, "get_compiled_graph", lambda: fake)

    result = graph.run_agent("hello", stream=True, user_id=1)

    assert result["final_answer"] == "done"
    assert fake.invoke_calls == 0


def test_tool_call_limit_counts_executed_tools_not_router_rounds():
    state = create_initial_state("需要多个工具")
    state.update(intent="fasta_analysis", tool_name="parse_fasta_stats")
    state["iteration"] = 3
    state["tool_history"] = [
        {"tool": "search_pdf_knowledge"},
        {"tool": "search_pubmed"},
    ]

    assert graph.should_call_tool(state) == "tool"

    state["tool_history"].append({"tool": "query_ncbi_gene"})
    assert graph.should_call_tool(state) == "answer"


def test_answer_receives_every_tool_result(monkeypatch):
    captured = {}

    class CapturingLlm:
        def invoke(self, prompt, **_kwargs):
            captured["prompt"] = prompt
            return "combined answer"

    monkeypatch.setattr(nodes, "HelloAgentsLLM", CapturingLlm)
    state = create_initial_state("组合分析")
    state["intent"] = "direct_answer"
    state["tool_history"] = [
        {
            "tool": "search_pdf_knowledge",
            "intent": "literature_search",
            "status": "success",
            "result": {"context": "PAML evidence", "count": 1},
        },
        {
            "tool": "parse_fasta_stats",
            "intent": "fasta_analysis",
            "status": "success",
            "result": {"num_sequences": 2, "total_length": 100},
        },
    ]
    state["tool_result"] = state["tool_history"][-1]["result"]

    result = nodes.answer_node(state)

    assert result["final_answer"] == "combined answer"
    assert "PAML evidence" in captured["prompt"]
    assert "序列数量: 2" in captured["prompt"]


def test_access_token_decoder_rejects_non_access_token(monkeypatch):
    monkeypatch.setattr(
        security_settings, "secret_key", "test-secret-key-with-at-least-32-chars"
    )
    token = jwt.encode(
        {"sub": "1", "type": "refresh"},
        security_settings.secret_key,
        algorithm=security_settings.algorithm,
    )

    assert decode_access_token(token) is None


def test_production_rejects_default_jwt_secret():
    value = Settings(app_env="production", secret_key=DEFAULT_SECRET_KEY)

    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        validate_security_settings(value)


def test_production_rejects_wildcard_cors():
    value = Settings(
        app_env="production",
        secret_key="a-secure-production-secret-with-32-characters",
        cors_origins=("*",),
    )

    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        validate_security_settings(value)


@contextmanager
def _chat_test_app():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    Base.metadata.create_all(engine)

    db = testing_session()
    user = User(
        username="test-user",
        email="test@example.com",
        hashed_password="unused",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    api = FastAPI()
    api.include_router(chat_api.router)

    def override_db():
        request_db = testing_session()
        try:
            yield request_db
        finally:
            request_db.close()

    api.dependency_overrides[get_db] = override_db
    api.dependency_overrides[get_current_user] = lambda: user

    try:
        yield TestClient(api), testing_session, user
    finally:
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_chat_passes_bounded_history_and_persists_actual_tool(monkeypatch):
    with _chat_test_app() as (client, session_factory, user):
        db = session_factory()
        chat_session = ChatSession(user_id=user.id, title="history")
        db.add(chat_session)
        db.flush()
        db.add_all(
            [
                Message(session_id=chat_session.id, role="user", content="上一问"),
                Message(session_id=chat_session.id, role="assistant", content="上一答"),
            ]
        )
        db.commit()
        session_id = chat_session.id
        db.close()

        captured = {}

        def fake_run_agent(query, messages, user_id):
            captured.update(query=query, messages=messages, user_id=user_id)
            return {
                "final_answer": "本轮回答",
                "sources": [],
                "tool_name": None,
                "tool_result": {"count": 1},
                "tool_history": [{"tool": "search_pdf_knowledge", "status": "success"}],
                "execution_log": [{"node": "tool", "status": "success"}],
            }

        monkeypatch.setattr(chat_api, "run_agent", fake_run_agent)

        response = client.post(
            "/api/chat",
            json={"message": "继续问", "session_id": session_id},
        )

        assert response.status_code == 200
        assert captured == {
            "query": "继续问",
            "messages": [
                {"role": "user", "content": "上一问"},
                {"role": "assistant", "content": "上一答"},
            ],
            "user_id": user.id,
        }
        assert response.json()["tool_name"] == "search_pdf_knowledge"

        verify_db = session_factory()
        assistant = (
            verify_db.query(Message)
            .filter(Message.session_id == session_id, Message.role == "assistant")
            .order_by(Message.id.desc())
            .first()
        )
        assert assistant.tool_name == "search_pdf_knowledge"
        verify_db.close()


def test_chat_persists_failure_trace(monkeypatch):
    with _chat_test_app() as (client, session_factory, _user):

        def fail_agent(*_args, **_kwargs):
            raise RuntimeError("unexpected failure")

        monkeypatch.setattr(chat_api, "run_agent", fail_agent)

        response = client.post("/api/chat", json={"message": "测试失败记录"})

        assert response.status_code == 502
        verify_db = session_factory()
        messages = verify_db.query(Message).order_by(Message.id.asc()).all()
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[-1].execution_log[0]["error_type"] == "agent_execution_error"
        verify_db.close()


def test_public_chat_request_does_not_advertise_fake_streaming():
    assert "stream" not in ChatRequest.model_fields
