import pytest

from app.agents import nodes
from app.agents.graph import should_call_tool
from app.agents.state import create_initial_state
from app.agents.tool_registry import tool_executor
from app.agents.tool_schemas import validate_tool_input


class FakeRouterLlm:
    def __init__(self, response: str):
        self.response = response

    def invoke(self, *_args, **_kwargs) -> str:
        return self.response


@pytest.mark.parametrize(
    ("tool_name", "missing_field"),
    [
        ("search_pdf_knowledge", "query"),
        ("parse_fasta_stats", "fasta_text"),
        ("scan_cpg_islands", "sequence"),
        ("suggest_pipeline", "research_goal"),
        ("search_pubmed", "keyword"),
        ("query_ncbi_gene", "gene_name"),
    ],
)
def test_each_tool_rejects_missing_required_input(tool_name, missing_field):
    validated, errors = validate_tool_input(tool_name, {})

    assert validated is None
    assert errors
    assert missing_field in errors[0]["loc"]


@pytest.mark.parametrize(
    ("tool_name", "payload"),
    [
        ("search_pdf_knowledge", {"query": "x" * 10_001}),
        ("parse_fasta_stats", {"fasta_text": "A" * 1_000_001}),
        ("scan_cpg_islands", {"sequence": "A" * 1_000_001}),
        ("search_pubmed", {"keyword": "x" * 2_001}),
    ],
)
def test_tool_inputs_reject_unbounded_payloads(tool_name, payload):
    validated, errors = validate_tool_input(tool_name, payload)

    assert validated is None
    assert errors[0]["type"] == "string_too_long"


def test_router_blocks_arguments_only_payload(monkeypatch):
    response = """
    {
      "intent": "pipeline_suggest",
      "tool_name": "suggest_pipeline",
      "arguments": {"research_goal": "mTERF gene family"}
    }
    """
    monkeypatch.setattr(nodes, "HelloAgentsLLM", lambda: FakeRouterLlm(response))
    state = create_initial_state("请推荐 mTERF 基因家族分析流程")

    result = nodes.router_node(state)

    assert result["tool_name"] is None
    assert result["router_error"]["type"] == "router_invalid_response"
    assert result["tool_result"]["error_type"] == "router_invalid_response"
    assert should_call_tool({**state, **result}) == "answer"


def test_tool_node_does_not_execute_invalid_input(monkeypatch):
    called = False

    def fake_tool(**_kwargs):
        nonlocal called
        called = True
        return {"unexpected": True}

    monkeypatch.setattr(tool_executor, "getTool", lambda _name: fake_tool)
    state = create_initial_state("请推荐一个分析流程")
    state["tool_name"] = "suggest_pipeline"
    state["tool_input"] = {}

    result = nodes.tool_node(state)

    assert called is False
    assert result["tool_result"]["error_type"] == "tool_validation_error"
    assert result["execution_log"][-1]["status"] == "validation_failed"


def test_valid_tool_input_is_normalized():
    validated, errors = validate_tool_input(
        "search_pdf_knowledge",
        {"query": "  PAML omega  ", "top_k": "2", "user_id": 7},
    )

    assert errors == []
    assert validated == {
        "query": "PAML omega",
        "top_k": 2,
        "user_id": 7,
        "enable_hybrid": True,
    }


def test_unknown_fields_are_rejected():
    validated, errors = validate_tool_input(
        "suggest_pipeline",
        {"research_goal": "gene family analysis", "unexpected": True},
    )

    assert validated is None
    assert errors
    assert errors[0]["type"] == "extra_forbidden"
