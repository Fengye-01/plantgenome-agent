"""Pydantic schemas for validating Agent tool inputs before execution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchPdfKnowledgeInput(ToolInput):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)
    user_id: int | None = Field(default=None, ge=1)
    enable_hybrid: bool = True


class ParseFastaStatsInput(ToolInput):
    fasta_text: str = Field(min_length=1)


class ScanCpgIslandsInput(ToolInput):
    sequence: str = Field(min_length=1)
    window_size: int = Field(default=200, ge=1)
    step: int = Field(default=100, ge=1)
    gc_threshold: float = Field(default=0.5, ge=0, le=1)
    oe_threshold: float = Field(default=0.6, ge=0)


class SuggestPipelineInput(ToolInput):
    research_goal: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


class SearchPubmedInput(ToolInput):
    keyword: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1, le=10)


class QueryNcbiGeneInput(ToolInput):
    gene_name: str = Field(min_length=1)
    organism: str | None = None
    max_results: int = Field(default=3, ge=1, le=5)


TOOL_INPUT_SCHEMAS: dict[str, type[ToolInput]] = {
    "search_pdf_knowledge": SearchPdfKnowledgeInput,
    "parse_fasta_stats": ParseFastaStatsInput,
    "scan_cpg_islands": ScanCpgIslandsInput,
    "suggest_pipeline": SuggestPipelineInput,
    "search_pubmed": SearchPubmedInput,
    "query_ncbi_gene": QueryNcbiGeneInput,
}


def validate_tool_input(
    tool_name: str,
    tool_input: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return normalized input or structured validation errors."""
    schema = TOOL_INPUT_SCHEMAS.get(tool_name)
    if schema is None:
        return None, [
            {
                "type": "unknown_tool",
                "loc": ["tool_name"],
                "msg": f"未注册工具输入 Schema: {tool_name}",
                "input": tool_name,
            }
        ]

    try:
        validated = schema.model_validate(tool_input)
    except ValidationError as exc:
        return None, exc.errors(include_url=False)

    return validated.model_dump(exclude_none=True), []
