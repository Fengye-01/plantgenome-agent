from app.worker.tasks import _chunk_content, _chunk_page_number, _chunk_section


def test_chunker_output_maps_to_persistence_fields():
    chunk = {
        "text": "正文内容",
        "page_num": 7,
        "metadata": {"heading_path": "# Results > ## CpG islands"},
    }

    assert _chunk_content(chunk) == "正文内容"
    assert _chunk_page_number(chunk) == 7
    assert _chunk_section(chunk) == "# Results > ## CpG islands"


def test_chunk_mapping_supports_legacy_content_and_limits_section_length():
    chunk = {
        "content": "legacy content",
        "metadata": {"page_start": 3, "heading_path": ["# A", "x" * 600]},
    }

    assert _chunk_content(chunk) == "legacy content"
    assert _chunk_page_number(chunk) == 3
    assert len(_chunk_section(chunk)) == 500
