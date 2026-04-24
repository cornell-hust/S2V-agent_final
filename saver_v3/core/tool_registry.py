from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from saver_v3.core.initial_observation import mark_initial_global_scan_message
from saver_v3.core.categories import CANONICAL_POLICY_CATEGORIES
from saver_v3.core.schema import SaverEnvironmentState
from saver_v3.core.semantic_answer import augment_finalize_case_schema
from saver_v3.core.self_verification import build_self_verification_tool_schema
from saver_v3.core import tools as saver_tools


DEFAULT_TOOL_NAMES: Tuple[str, ...] = (
    "scan_timeline",
    "seek_evidence",
    "verify_hypothesis",
    "finalize_case",
)


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "scan_timeline",
            "description": "Uniformly inspect a time window to build a global or local overview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "num_frames": {"type": "integer"},
                    "stride_sec": {"type": "number"},
                    "purpose": {"type": "string"},
                },
                "required": ["start_sec", "end_sec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "seek_evidence",
            "description": "Search a time window for frames relevant to a textual anomaly hypothesis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "num_frames": {"type": "integer"},
                    "moment_id": {"type": "string"},
                    "role": {"type": "string"},
                    "query_source": {"type": "string"},
                    "top_k_candidates": {"type": "integer"},
                    "candidate_merge_gap_sec": {"type": "number"},
                },
                "required": ["query", "start_sec", "end_sec", "role"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_hypothesis",
            "description": (
                "Verify whether the currently selected evidence subset is sufficient, necessary enough, "
                "and actionable for the active anomaly hypothesis using a compact policy-produced "
                "self-verification verdict."
            ),
            "parameters": build_self_verification_tool_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_case",
            "description": (
                "Finalize the structured anomaly decision once enough evidence has been gathered. "
                "When available, include summary, rationale, event_chain_summary, and qa_focus_answers "
                "in the same finalize_case call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "existence": {"type": "string", "enum": ["normal", "anomaly"]},
                    "category": {"type": "string", "enum": list(CANONICAL_POLICY_CATEGORIES)},
                },
            },
        },
    },
]

TOOL_IMPLS = {
    "scan_timeline": saver_tools.scan_timeline,
    "seek_evidence": saver_tools.seek_evidence,
    "verify_hypothesis": saver_tools.verify_hypothesis,
    "finalize_case": saver_tools.finalize_case,
}


def get_tool_schemas(*, finalize_case_schema: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    tool_schemas = copy.deepcopy(TOOL_SCHEMAS)
    for tool in tool_schemas:
        function = tool.get("function") or {}
        if function.get("name") != "finalize_case":
            continue
        base_schema = finalize_case_schema if finalize_case_schema else function.get("parameters") or {}
        function["parameters"] = augment_finalize_case_schema(base_schema)
        break
    return tool_schemas


def execute_tool_call(
    params: Dict[str, Any],
    multimodal_cache: Dict[str, Any],
    state: SaverEnvironmentState,
) -> Tuple[Dict[str, Any], SaverEnvironmentState]:
    func = params.get("function", {})
    name = func.get("name")
    arguments = func.get("arguments", {})
    if name not in TOOL_IMPLS:
        raise ValueError(f"Unknown tool name: {name}")
    content, state, tool_trace = TOOL_IMPLS[name](arguments, multimodal_cache, state)
    message = {
        "role": "tool",
        "name": name,
        "arguments": arguments,
        "content": content,
    }
    if isinstance(tool_trace, dict) and bool(tool_trace.get("initial_global_scan")):
        message = mark_initial_global_scan_message(
            message,
            config=multimodal_cache.get("config_snapshot") or {},
        )
    return message, state
