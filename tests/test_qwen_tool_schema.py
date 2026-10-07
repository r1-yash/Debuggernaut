from __future__ import annotations

from typing import Any, Optional, Union

from swe_agent.agent.fixer import list_directory
from swe_agent.llm import _annotation_to_json_schema, _qwen_tool_schema


def test_primitive_annotations_map_to_json_schema_types() -> None:
    assert _annotation_to_json_schema(str) == {"type": "string"}
    assert _annotation_to_json_schema(int) == {"type": "integer"}
    assert _annotation_to_json_schema(float) == {"type": "number"}
    assert _annotation_to_json_schema(bool) == {"type": "boolean"}
    assert _annotation_to_json_schema(list) == {"type": "array", "items": {}}
    assert _annotation_to_json_schema(dict) == {
        "type": "object",
        "additionalProperties": {},
    }


def test_container_optional_union_and_any_annotations() -> None:
    assert _annotation_to_json_schema(Optional[str]) == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }
    assert _annotation_to_json_schema(list[str]) == {
        "type": "array", "items": {"type": "string"}
    }
    assert _annotation_to_json_schema(dict[str, int]) == {
        "type": "object", "additionalProperties": {"type": "integer"}
    }
    assert _annotation_to_json_schema(Union[int, float]) == {
        "anyOf": [{"type": "integer"}, {"type": "number"}]
    }
    assert _annotation_to_json_schema(Any) == {}


def test_debuggernaut_tool_schema_is_openai_compatible() -> None:
    schema = _qwen_tool_schema(list_directory)

    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "list_directory"
    assert isinstance(function["description"], str)
    assert function["parameters"] == {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string"},
            "dir_path": {"type": "string"},
        },
        "required": ["repo_path"],
        "additionalProperties": False,
    }
