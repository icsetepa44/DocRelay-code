import pytest

from src.core.exceptions import StructuredOutputError
from src.llm.structured_output import extract_json_object, _salvage_json


def test_valid_json_unchanged():
    assert extract_json_object('{"page_number": 0, "blocks": []}') == {"page_number": 0, "blocks": []}


def test_prose_wrapped_json_recovered():
    raw = 'Sure, here is the JSON:\n```json\n{"page_number": 1, "blocks": []}\n```\nDone.'
    assert extract_json_object(raw) == {"page_number": 1, "blocks": []}


def test_trailing_comma_recovered():
    raw = '{"page_number": 2, "blocks": [{"block_id": "b1"},]}'
    assert extract_json_object(raw) == {"page_number": 2, "blocks": [{"block_id": "b1"}]}


def test_truncated_object_recovers_completed_blocks():
    raw = '{"page_number": 3, "blocks": [{"block_id": "b1", "content": {"text": "hi"}}, {"block_id": "b2", "content": {"text": "wor'
    obj = extract_json_object(raw)
    assert obj["blocks"][0]["block_id"] == "b1"
    assert len(obj["blocks"]) == 1


def test_salvage_never_alters_string_content():
    raw = '{"page_number": 4, "blocks": [{"content": {"text": "a, }"}}]}'
    obj = extract_json_object(raw)
    assert obj["blocks"][0]["content"]["text"] == "a, }"


def test_unrecoverable_raises():
    with pytest.raises(StructuredOutputError):
        extract_json_object("no json here at all")


def test_salvage_returns_none_on_flat_truncation_without_container():
    assert _salvage_json('{"a": "b') is None
