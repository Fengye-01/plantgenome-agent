"""Regression tests for authenticated vector-search isolation."""

from app.agents.nodes import tool_node
from app.agents.state import create_initial_state
from app.agents.tool_registry import tool_executor
from app.tools import search_pdf_knowledge as search_module


def test_search_only_returns_current_users_documents(monkeypatch):
    records = [
        {
            "text": "private content owned by user A",
            "metadata": {"user_id": 1, "filename": "user-a.pdf", "page_num": 1},
            "distance": 0.1,
        },
        {
            "text": "private content owned by user B",
            "metadata": {"user_id": 2, "filename": "user-b.pdf", "page_num": 2},
            "distance": 0.2,
        },
    ]

    class FakeVectorStore:
        def search(self, query, top_k, filter_metadata):
            assert query == "private research"
            assert filter_metadata == {"user_id": 1}
            return [
                record
                for record in records
                if record["metadata"]["user_id"] == filter_metadata["user_id"]
            ][:top_k]

    monkeypatch.setattr(search_module, "VectorStore", FakeVectorStore)

    result = search_module.search_pdf_knowledge(
        query="private research",
        top_k=3,
        user_id=1,
    )

    assert result["has_result"] is True
    assert [source["filename"] for source in result["sources"]] == ["user-a.pdf"]
    assert "user-b.pdf" not in result["context"]


def test_tool_node_overrides_llm_supplied_user_id(monkeypatch):
    captured_input = {}

    def fake_search(**tool_input):
        captured_input.update(tool_input)
        return {"context": "", "sources": [], "count": 0, "has_result": False}

    monkeypatch.setitem(
        tool_executor.tools,
        "search_pdf_knowledge",
        {
            **tool_executor.tools["search_pdf_knowledge"],
            "func": fake_search,
        },
    )

    state = create_initial_state("private research", user_id=7)
    state["tool_name"] = "search_pdf_knowledge"
    state["tool_input"] = {
        "query": "private research",
        "user_id": 999,
    }

    tool_node(state)

    assert captured_input["user_id"] == 7
