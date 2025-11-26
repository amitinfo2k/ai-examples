"""JOLT-like transformer tool and CrewAI BaseTool wrapper.

Provides functions to transform OCSF input using a minimal JOLT "shift" spec
and exposes a CrewAI BaseTool so agents can call it.

Supported features:
- spec is either a list of operations (first op used) or a single dict
- only the "shift" operation is supported
- destination values are dot-paths like "a.b.c"
- source values in spec may be dot-paths; basic array indices like key[0].subkey are supported
"""
from __future__ import annotations

from typing import Any, Dict, List, Union, Tuple, Type
import json
import os

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

Json = Union[Dict[str, Any], List[Any]]


class JoltTransformToolInput(BaseModel):
    """Input schema for JoltTransformTool.

    Provide either JSON strings or file paths for input and spec.
    Precedence: *_json over *_file.
    """
    input_json: str | None = Field(
        default=None, description="OCSF input JSON as a string"
    )
    input_file: str | None = Field(
        default=None, description="Path to OCSF input JSON file"
    )
    jolt_spec_json: str | None = Field(
        default=None, description="JOLT spec as JSON string (list or dict)"
    )
    jolt_spec_file: str | None = Field(
        default=None, description="Path to JOLT spec JSON file"
    )


class JoltTransformTool(BaseTool):
    name: str = "jolt_transform"
    description: str = (
        "Apply a minimal JOLT 'shift' spec to an OCSF input and return the transformed JSON. "
        "Accepts either inline JSON strings or file paths for both input and spec."
    )
    args_schema: Type[BaseModel] = JoltTransformToolInput

    def _run(
        self,
        input_json: str | None = None,
        input_file: str | None = None,
        jolt_spec_json: str | None = None,
        jolt_spec_file: str | None = None,
    ) -> str:
        # Resolve input
        if input_json is not None:
            try:
                inp = json.loads(input_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"input_json is not valid JSON: {e}") from e
        elif input_file is not None:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
            with open(input_file, "r", encoding="utf-8") as f:
                inp = json.load(f)
        else:
            raise ValueError("Provide either input_json or input_file")

        # Resolve spec
        if jolt_spec_json is not None:
            try:
                spec = json.loads(jolt_spec_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"jolt_spec_json is not valid JSON: {e}") from e
        elif jolt_spec_file is not None:
            if not os.path.exists(jolt_spec_file):
                raise FileNotFoundError(f"Spec file not found: {jolt_spec_file}")
            with open(jolt_spec_file, "r", encoding="utf-8") as f:
                spec = json.load(f)
        else:
            raise ValueError("Provide either jolt_spec_json or jolt_spec_file")

        out = transform_with_jolt(inp, spec)
        return json.dumps(out, ensure_ascii=False)


def transform_with_jolt(ocsf_input: Json, jolt_spec: Union[List[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    """Transform input using a minimal JOLT "shift" spec.

    Args:
        ocsf_input: Input JSON (dict or list)
        jolt_spec: JOLT spec as list[op] or single op dict

    Returns:
        Dict[str, Any]: Transformed JSON
    """
    ops: List[Dict[str, Any]]
    if isinstance(jolt_spec, list):
        ops = jolt_spec
    elif isinstance(jolt_spec, dict):
        ops = [jolt_spec]
    else:
        raise ValueError("Invalid JOLT spec type. Expected list or dict.")

    if not ops:
        return {}

    op0 = ops[0]
    if not isinstance(op0, dict) or op0.get("operation") != "shift":
        raise ValueError('Only "shift" operation is supported in this tool')

    spec = op0.get("spec")
    if not isinstance(spec, dict):
        raise ValueError('"spec" must be an object for shift operation')

    out: Dict[str, Any] = {}
    _apply_shift(spec, ocsf_input, out)
    return out


def _apply_shift(spec: Dict[str, Any], src: Json, out: Dict[str, Any], src_prefix: str = "", dst_prefix: str = "") -> None:
    """Recursively apply a simplified shift spec.

    The spec structure here is interpreted as: keys navigate the source, and string values
    define destination dot-paths. Nested dict values continue traversal.
    """
    for key, value in spec.items():
        # Current source path to read from
        current_src_path = f"{src_prefix}.{key}".strip(".")

        if isinstance(value, str):
            # value is destination path
            src_val = _get_by_path(src, current_src_path)
            if src_val is not None:
                _set_by_path(out, value, src_val)
        elif isinstance(value, dict):
            _apply_shift(value, src, out, current_src_path, dst_prefix)
        else:
            # Unsupported spec node, ignore
            continue


def _get_by_path(data: Json, path: str) -> Any:
    """Get value from nested JSON using dot path with optional array indexes like key[0].sub."""
    if path == "":
        return data

    parts = _split_path(path)
    cur: Any = data
    for p_key, p_idx in parts:
        if isinstance(cur, dict):
            if p_key not in cur:
                return None
            cur = cur[p_key]
        else:
            return None
        if p_idx is not None:
            if isinstance(cur, list):
                if p_idx < 0 or p_idx >= len(cur):
                    return None
                cur = cur[p_idx]
            else:
                return None
    return cur


def _set_by_path(obj: Dict[str, Any], path: str, value: Any) -> None:
    """Set value on dict using dot path; creates nested dicts as needed.
    Array notation in destination is ignored (we only handle objects in output).
    """
    parts = path.split(".") if path else []
    cur = obj
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    if parts:
        cur[parts[-1]] = value


def _split_path(path: str) -> List[Tuple[str, int | None]]:
    """Split path like "events[0].message" into [("events",0),("message",None)]."""
    parts: List[Tuple[str, int | None]] = []
    for raw in path.split('.'):
        if '[' in raw and raw.endswith(']'):
            name, idx_str = raw[:-1].split('[', 1)
            try:
                idx = int(idx_str)
            except ValueError:
                idx = None
            parts.append((name, idx))
        else:
            parts.append((raw, None))
    return parts


def transform_files(input_path: str, spec_path: str) -> Dict[str, Any]:
    """Convenience helper to load files and transform.

    Args:
        input_path: path to OCSF JSON
        spec_path: path to JOLT spec JSON (list or dict)
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        inp = json.load(f)
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    return transform_with_jolt(inp, spec)
