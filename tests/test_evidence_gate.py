from app.agents import nodes
from app.agents.state import create_initial_state
from app.tools import search_pdf_knowledge as search_module


def test_search_rejects_low_relevance_results(monkeypatch):
    class FakeVectorStore:
        def search(self, query, top_k, filter_metadata):
            return [
                {
                    "text": "A superficially related passage.",
                    "metadata": {"filename": "weak.pdf", "page_num": 1},
                    "distance": 1.2,
                }
            ]

    monkeypatch.setattr(search_module, "VectorStore", FakeVectorStore)

    result = search_module.search_pdf_knowledge("positive selection")

    assert result["has_result"] is False
    assert result["evidence_status"] == "insufficient"
    assert result["evidence_reason"] == "low_relevance_or_empty_content"
    assert result["sources"] == []
    assert result["context"] == ""


def test_answer_abstains_without_calling_llm(monkeypatch):
    class FailIfCalledLLM:
        def __init__(self, *args, **kwargs):
            raise AssertionError("LLM must not be created when evidence is insufficient")

    monkeypatch.setattr(nodes, "HelloAgentsLLM", FailIfCalledLLM)

    state = create_initial_state("Does this paper prove positive selection?")
    state["intent"] = "literature_search"
    state["tool_result"] = {
        "context": "",
        "sources": [],
        "count": 0,
        "has_result": False,
        "evidence_status": "insufficient",
        "evidence_reason": "no_results",
    }

    result = nodes.answer_node(state)

    assert result["final_answer"] == nodes.INSUFFICIENT_EVIDENCE_ANSWER
    assert result["sources"] == []
    assert result["execution_log"][-1]["status"] == "abstained"
    assert result["execution_log"][-1]["reason"] == "no_results"
