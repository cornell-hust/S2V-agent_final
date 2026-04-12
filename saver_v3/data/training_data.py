from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from convert_to_saver_agent import CanonicalSaverAdapter, ConverterConfig, build_finalize_case_payload

from saver_v3.core.adapter import TimeSearchRolloutAdapter
from saver_v3.data.config import SaverAgentConfig
from saver_v3.core.environment import SaverEnvironmentState, SaverVideoInteraction, parse_actions_and_contents
from saver_v3.core.event_chain import (
    compute_event_chain_score,
)
from saver_v3.core.proposal import normalize_query_text
from saver_v3.core.reward import DEFAULT_COMPONENT_WEIGHTS, PRIMARY_STATUS_REWARD
from saver_v3.core.semantic_answer import (
    build_public_semantic_replay_payload,
    build_public_semantic_replay_scaffold,
    normalize_public_semantic_replay_payload,
)
from saver_v3.core.self_verification import (
    DECISION_TO_PRIMARY_STATUS,
    build_policy_self_verification_payload,
)
from saver_v3.teacher.teacher_judge import (
    compute_teacher_judge_signal,
    has_teacher_judge_labels,
    is_teacher_judge_candidate,
)
from saver_v3.data.prepared_metadata import PREPARED_SFT_FORMAT

THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_AUXILIARY_WEIGHT_MULTIPLIER = 0.1
_EPISODE_LOSS_WEIGHT_BY_KIND = {
    "search": 1.0,
    "verify": 1.25,
    "finalize": 1.5,
    "semantic_replay": 0.2,
    "answer": 0.1,
}
_EPISODE_PASSTHROUGH_KEYS = (
    "multimodal_cache",
    "rollout_context",
    "teacher_judge_scores",
    "teacher_judge_decision",
    "teacher_judge_rationale",
    "teacher_judge_weight_multiplier",
    "teacher_judge_effective_sample_weight",
)
_COMPACT_TRACE_RECORD_FIELDS = (
    "video_id",
    "split",
    "video_path",
    "video_meta",
    "scene",
    "agent_task",
    "structured_target",
    "tool_io",
    "label",
    "temporal",
    "evidence",
    "language",
    "qa_pairs",
    "proposal_supervision",
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def is_compact_trace_sft_record(example: Dict[str, Any]) -> bool:
    return (
        isinstance(example, dict)
        and str(example.get("prepared_format") or "").strip() == str(PREPARED_SFT_FORMAT)
        and isinstance(example.get("oracle_trajectory"), list)
        and str(example.get("video_path") or "").strip() != ""
    )


def _compact_trace_record_payload(item: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field_name in _COMPACT_TRACE_RECORD_FIELDS:
        if field_name == "video_path":
            resolved_video_path = (
                str((item.get("multimodal_cache") or {}).get("video_path") or "")
                or str(item.get("video") or "")
                or str(record.get("video_path") or "")
            )
            payload["video_path"] = resolved_video_path
            continue
        payload[field_name] = copy.deepcopy(record.get(field_name))
    return payload


def build_compact_trace_sft_record(
    item: Dict[str, Any],
    record: Dict[str, Any],
    *,
    config: Optional[SaverAgentConfig] = None,
) -> Dict[str, Any]:
    del config  # compact trace stores the canonical episode trace; prompt/preview live in metadata
    oracle_sft = _ensure_oracle_sft(record)
    compact_record = _compact_trace_record_payload(item, record)
    compact_record["prepared_format"] = str(PREPARED_SFT_FORMAT)
    compact_record["source"] = "oracle_sft_compact_trace_v2"
    compact_record["oracle_trajectory"] = copy.deepcopy(list(oracle_sft.get("trajectory") or []))
    compact_record["oracle_final_decision"] = copy.deepcopy(
        oracle_sft.get("final_decision")
        or record.get("structured_target")
        or {}
    )
    return compact_record


def _clip_text(text: str, *, max_len: int = 160) -> str:
    normalized = " ".join(str(text or "").strip().split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3].rstrip() + "..."


def _format_time_span(start_sec: Any, end_sec: Any) -> str:
    try:
        start_value = float(start_sec)
        end_value = float(end_sec)
        return f"{start_value:.1f}-{end_value:.1f}s"
    except Exception:
        return "this interval"


def _find_evidence_moment(record: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    moment_id = arguments.get("moment_id")
    role = str(arguments.get("role") or "")
    start_sec = arguments.get("start_sec")
    end_sec = arguments.get("end_sec")
    for moment in (record.get("evidence") or {}).get("evidence_moments", []):
        if moment_id is not None and str(moment.get("moment_id")) == str(moment_id):
            return copy.deepcopy(moment)
    for moment in (record.get("evidence") or {}).get("evidence_moments", []):
        if role and str(moment.get("role") or "") == role:
            if (
                start_sec is None
                or end_sec is None
                or (
                    float(moment.get("start_sec") or -1.0) == float(start_sec)
                    and float(moment.get("end_sec") or -1.0) == float(end_sec)
                )
            ):
                return copy.deepcopy(moment)
    return {}


def _tool_reasoning_text(
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    record: Dict[str, Any],
    state: SaverEnvironmentState,
) -> str:
    label = record.get("label") or {}
    structured_target = record.get("structured_target") or {}
    category = str(label.get("category") or structured_target.get("category") or "event")
    task_prompt = str((record.get("agent_task") or {}).get("task_prompt") or "").strip()
    summary = str((record.get("language") or {}).get("summary") or structured_target.get("summary") or "").strip()
    rationale = str((record.get("language") or {}).get("rationale") or structured_target.get("rationale") or "").strip()
    span_text = _format_time_span(arguments.get("start_sec"), arguments.get("end_sec"))

    if tool_name == "scan_timeline":
        if str(arguments.get("purpose") or "") == "global_overview":
            if task_prompt:
                return _clip_text(
                    f"I only have a limited preview, so I should scan {span_text} for a global overview before deciding how to investigate {task_prompt.lower()}."
                )
            return _clip_text(
                f"I only have a limited preview, so I should scan {span_text} for a global overview before making a decision about the clip."
            )
        return _clip_text(f"I should inspect {span_text} to refine my overview before taking the next step.")

    if tool_name == "seek_evidence":
        moment = _find_evidence_moment(record, arguments)
        role = str(arguments.get("role") or moment.get("role") or "evidence")
        description = str(moment.get("description") or "").strip()
        query = str(arguments.get("query") or "").strip()
        if description:
            return _clip_text(
                f"The next useful clue is the {role} around {span_text}; I should inspect it to check whether {description.lower()}."
            )
        if query:
            return _clip_text(
                f"I should inspect {span_text} for the {role} evidence and test whether the video shows {query.lower()}."
            )
        return _clip_text(f"I should inspect {span_text} for more targeted evidence about the suspected {category}.")

    if tool_name == "verify_hypothesis":
        claim = arguments.get("claim") or state.last_claim or {}
        claim_existence = str(claim.get("existence") or structured_target.get("existence") or "current")
        claim_category = str(claim.get("category") or category)
        selected_window_ids = list(arguments.get("selected_window_ids") or [])
        selected_evidence_moment_ids = list(
            arguments.get("selected_evidence_moment_ids")
            or arguments.get("candidate_evidence_moment_ids")
            or arguments.get("evidence_moment_ids")
            or []
        )
        if selected_window_ids:
            candidate_count = len(selected_window_ids)
        else:
            candidate_count = len(selected_evidence_moment_ids)
        candidate_phrase = f"{candidate_count} selected evidence item(s)" if candidate_count > 0 else "the currently gathered evidence"
        return _clip_text(f"I should verify whether {candidate_phrase} are enough for the {claim_existence} claim about {claim_category}.")

    if tool_name == "finalize_case":
        existence = str(arguments.get("existence") or structured_target.get("existence") or "unknown")
        final_category = str(arguments.get("category") or category)
        if existence == "normal":
            return _clip_text("The inspected evidence is enough to conclude that the clip is normal, so I can finalize the case now.")
        if summary:
            return _clip_text(
                f"The searched evidence is sufficient and consistent, so I can finalize this as {final_category}: {summary}"
            )
        return _clip_text(f"The searched evidence is sufficient and consistent, so I can finalize this as {existence} / {final_category}.")

    if rationale:
        return _clip_text(f"I should act on the current evidence and rationale: {rationale}")
    return "I should take the next tool step based on the evidence collected so far."


def _answer_reasoning_text(answer_payload: Dict[str, Any], *, record: Dict[str, Any]) -> str:
    structured_target = record.get("structured_target") or {}
    existence = str(structured_target.get("existence") or "unknown")
    category = str(structured_target.get("category") or "case")
    summary = str(answer_payload.get("summary") or (record.get("structured_target") or {}).get("summary") or "").strip()
    if summary:
        return _clip_text(f"The search is complete, so I can write the post-finalize semantic replay: {summary}")
    if existence == "normal":
        return "The search is complete, so I can write the post-finalize semantic replay that this clip is normal."
    return _clip_text(f"The search is complete, so I can write the post-finalize semantic replay for the {category} case.")


def _assistant_tool_response(
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    record: Dict[str, Any],
    state: SaverEnvironmentState,
) -> str:
    think_text = _tool_reasoning_text(tool_name, arguments, record=record, state=state)
    tool_payload = {"name": tool_name, "arguments": arguments}
    return f"<think>{think_text}</think><tool_call>{_json_dumps(tool_payload)}</tool_call>"


def _assistant_answer_response(answer_payload: Dict[str, Any], *, record: Dict[str, Any]) -> str:
    think_text = _answer_reasoning_text(answer_payload, record=record)
    return f"<think>{think_text}</think><answer>{_json_dumps(answer_payload)}</answer>"


def _answer_payload_from_record(
    record: Dict[str, Any],
    *,
    finalized_case: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    structured_target = copy.deepcopy(record.get("structured_target") or {})
    if isinstance(finalized_case, dict):
        structured_target.update(copy.deepcopy(finalized_case))
    payload = build_public_semantic_replay_payload(
        structured_target=structured_target,
        qa_pairs=record.get("qa_pairs") or [],
        evidence_moments=((record.get("evidence") or {}).get("evidence_moments") or []),
    )
    normalized = normalize_public_semantic_replay_payload(payload)
    return normalized or payload


def _semantic_replay_source_name(source: str) -> str:
    normalized = str(source or "").strip() or "oracle_sft"
    if normalized.endswith("_semantic_replay"):
        return normalized
    return f"{normalized}_semantic_replay"


def _public_replay_payload_from_response_text(response_text: str) -> Optional[Dict[str, Any]]:
    answer_text = TimeSearchRolloutAdapter.parse_answer_text(str(response_text or ""))
    if not answer_text:
        return None
    try:
        payload = json.loads(answer_text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return normalize_public_semantic_replay_payload(payload)


def _build_semantic_replay_messages(
    *,
    record: Dict[str, Any],
    finalized_case: Dict[str, Any],
) -> List[Dict[str, Any]]:
    structured_target = dict(record.get("structured_target") or {})
    evidence_moments = list(((record.get("evidence") or {}).get("evidence_moments") or []))
    task_prompt = str((record.get("agent_task") or {}).get("task_prompt") or "Summarize the finalized anomaly decision.").strip()
    evidence_lines = []
    for moment in evidence_moments:
        role = str(moment.get("role") or "").strip() or "evidence"
        description = str(moment.get("description") or "").strip()
        if not description:
            continue
        evidence_lines.append(f"- {role}: {description}")
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "- no explicit evidence descriptions provided"
    user_text = (
        "This is a separate post-finalize semantic replay.\n"
        "Do not call tools.\n"
        f"Task: {task_prompt}\n"
        f"Finalized decision: {json.dumps(finalized_case, ensure_ascii=False)}\n"
        "Use the provided evidence descriptions only.\n"
        f"Evidence descriptions:\n{evidence_block}\n"
        f"Return exactly one <answer></answer> JSON in this shape: {build_public_semantic_replay_scaffold()}"
    )
    if str(structured_target.get("anomaly_interval_sec") or "").strip():
        user_text += f"\nReference anomaly interval: {json.dumps(structured_target.get('anomaly_interval_sec'), ensure_ascii=False)}"
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are SAVER semantic replay. "
                        "This is a post-finalize explanation pass. "
                        "You must not call tools."
                    ),
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


def _extract_finalize_case_arguments(target_response: str) -> Dict[str, Any]:
    actions, contents = parse_actions_and_contents([target_response])
    if not actions or actions[0] != "tool_call":
        return {}
    function = ((contents[0] or {}).get("function") or {})
    if str(function.get("name") or "") != "finalize_case":
        return {}
    arguments = function.get("arguments") or {}
    return copy.deepcopy(arguments) if isinstance(arguments, dict) else {}


def _raw_finalize_tool_message(finalized_case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": "tool",
        "name": "finalize_case",
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {"status": "finalized", "finalized_case": finalized_case},
                    ensure_ascii=False,
                ),
            }
        ],
    }


def _answer_example_from_finalize(
    finalize_example: Dict[str, Any],
    *,
    record: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    fallback_sample_weight: Optional[float] = None,
    answer_template: Optional[Dict[str, Any]] = None,
    config: Optional[SaverAgentConfig] = None,
) -> Dict[str, Any]:
    answer_example = copy.deepcopy(answer_template or {})
    answer_example["video_id"] = answer_example.get("video_id") or finalize_example.get("video_id")
    answer_example["split"] = answer_example.get("split") or finalize_example.get("split")
    answer_example["step_index"] = int(finalize_example.get("step_index") or 0) + 1
    answer_example["source"] = _semantic_replay_source_name(
        str(source or answer_example.get("source") or finalize_example.get("source") or "teacher_rollout_primary")
    )
    answer_example["target_action"] = "answer"
    answer_example["tool_name"] = None
    if fallback_sample_weight is not None:
        answer_example["sample_weight"] = float(fallback_sample_weight)
    rollout_context = copy.deepcopy(
        answer_example.get("rollout_context")
        or finalize_example.get("rollout_context")
        or {}
    )
    if rollout_context:
        answer_example["rollout_context"] = rollout_context
    finalize_arguments = _extract_finalize_case_arguments(str(finalize_example.get("target_response") or ""))
    target_payload = None
    if answer_template:
        target_payload = _public_replay_payload_from_response_text(str(answer_template.get("target_response") or ""))
    if target_payload is not None:
        target_response = _assistant_answer_response(target_payload, record=record or {"structured_target": finalize_arguments})
    else:
        payload_record = record or {"structured_target": finalize_arguments}
        answer_payload = _answer_payload_from_record(
            payload_record,
            finalized_case=finalize_arguments,
        )
        target_response = _assistant_answer_response(answer_payload, record=payload_record)
    answer_example["target_response"] = target_response
    payload_record = record or {"structured_target": finalize_arguments, "agent_task": {}, "evidence": {}}
    answer_example["messages"] = _build_semantic_replay_messages(
        record=payload_record,
        finalized_case=finalize_arguments,
    )
    answer_example["proposal_supervision"] = copy.deepcopy(answer_example.get("proposal_supervision") or {})
    return answer_example


def is_episode_sft_record(example: Dict[str, Any]) -> bool:
    return isinstance(example.get("messages"), list) and isinstance(example.get("assistant_supervision"), list)


def _episode_loss_weight(kind: str) -> float:
    return float(_EPISODE_LOSS_WEIGHT_BY_KIND.get(str(kind or "").strip().lower(), 1.0))


def _episode_supervision_kind(example: Dict[str, Any]) -> str:
    source_name = str(example.get("source") or "").strip().lower()
    if "semantic_replay" in source_name:
        return "semantic_replay"
    if str(example.get("target_action") or "") == "answer":
        return "answer"
    tool_name = str(example.get("tool_name") or "").strip().lower()
    if tool_name == "verify_hypothesis":
        return "verify"
    if tool_name == "finalize_case":
        return "finalize"
    return "search"


def _episode_source_name(source: str) -> str:
    normalized = str(source or "").strip() or "oracle_sft"
    if normalized.endswith("_episode"):
        return normalized
    return f"{normalized}_episode"


def _step_example_episode_source_name(example: Dict[str, Any]) -> str:
    source = str(example.get("source") or "")
    target_action = str(example.get("target_action") or "").strip().lower()
    if target_action == "answer" and "semantic_replay" not in source.lower():
        source = _semantic_replay_source_name(source)
    return _episode_source_name(source)


def _extract_following_messages_from_step_examples(
    current_example: Dict[str, Any],
    next_example: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if next_example is None:
        return []
    current_messages = list(current_example.get("messages") or [])
    next_messages = list(next_example.get("messages") or [])
    if len(next_messages) <= len(current_messages):
        return []
    suffix = copy.deepcopy(next_messages[len(current_messages) :])
    if (
        suffix
        and str(suffix[0].get("role") or "") == "assistant"
        and _extract_message_text(suffix[0]) == str(current_example.get("target_response") or "")
    ):
        suffix = suffix[1:]
    return suffix


def convert_step_examples_to_episode_records(
    examples: Sequence[Dict[str, Any]],
    *,
    config: Optional[SaverAgentConfig] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = copy.deepcopy(config) if config is not None else SaverAgentConfig()
    ordered_groups: List[tuple[str, str, str]] = []
    grouped_examples: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
    for example in examples:
        if is_episode_sft_record(example):
            key = (str(example.get("video_id") or ""), str(example.get("split") or ""), str(example.get("source") or ""))
        else:
            key = (
                str(example.get("video_id") or ""),
                str(example.get("split") or ""),
                _step_example_episode_source_name(example),
            )
        if key not in grouped_examples:
            grouped_examples[key] = []
            ordered_groups.append(key)
        grouped_examples[key].append(copy.deepcopy(example))

    episode_records: List[Dict[str, Any]] = []
    summary = {
        "num_episode_records": 0,
        "num_upgraded_step_groups": 0,
        "num_passthrough_episode_records": 0,
    }

    def _copy_group_metadata(group_examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        preserved: Dict[str, Any] = {}
        for key in _EPISODE_PASSTHROUGH_KEYS:
            for candidate in group_examples:
                if key in candidate:
                    preserved[key] = copy.deepcopy(candidate[key])
                    break
        return preserved

    for key in ordered_groups:
        group = list(grouped_examples.get(key) or [])
        if not group:
            continue
        if all(is_episode_sft_record(example) for example in group):
            episode_records.extend(copy.deepcopy(group))
            summary["num_episode_records"] += len(group)
            summary["num_passthrough_episode_records"] += len(group)
            continue

        group.sort(key=lambda example: int(example.get("step_index") or 0))
        episode_source = str(key[2] or "")
        base_messages = copy.deepcopy(group[0].get("messages") or [])
        assistant_supervision: List[Dict[str, Any]] = []
        max_sample_weight = 0.0

        for index, example in enumerate(group):
            response_text = str(example.get("target_response") or "")
            if not response_text:
                continue
            base_messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response_text}],
                }
            )
            kind = _episode_supervision_kind(example)
            if kind == "answer" and "semantic_replay" in episode_source:
                kind = "semantic_replay"
            assistant_supervision.append(
                {
                    "assistant_message_index": len(base_messages) - 1,
                    "kind": kind,
                    "loss_weight": _episode_loss_weight(kind),
                }
            )
            max_sample_weight = max(max_sample_weight, float(example.get("sample_weight", 1.0) or 0.0))
            if str(example.get("target_action") or "") != "answer":
                base_messages.extend(
                    _extract_following_messages_from_step_examples(
                        example,
                        group[index + 1] if index + 1 < len(group) else None,
                    )
                )

        episode_weight = max(1.0, float(max_sample_weight or 1.0))
        episode_record = {
            "video_id": group[0].get("video_id"),
            "split": group[0].get("split"),
            "source": episode_source,
            "messages": base_messages,
            "assistant_supervision": assistant_supervision,
            "episode_weight": episode_weight,
            "sample_weight": episode_weight,
        }
        episode_record.update(_copy_group_metadata(group))
        episode_records.append(episode_record)
        summary["num_episode_records"] += 1
        summary["num_upgraded_step_groups"] += 1

    return episode_records, summary


def build_oracle_sft_episode_examples(
    item: Dict[str, Any],
    record: Dict[str, Any],
    *,
    config: Optional[SaverAgentConfig] = None,
    serialize_messages: bool = False,
) -> List[Dict[str, Any]]:
    step_examples = build_oracle_sft_examples(
        item,
        record,
        config=config,
        serialize_messages=serialize_messages,
    )
    episode_records, _ = convert_step_examples_to_episode_records(step_examples, config=config)
    return episode_records


def _extract_teacher_judge_labels(payload: Dict[str, Any]) -> Dict[str, Any]:
    labels: Dict[str, Any] = {}
    for key in ("teacher_judge_scores", "teacher_judge_decision", "teacher_judge_rationale"):
        if key in payload:
            labels[key] = copy.deepcopy(payload[key])
    return labels


def _merge_verify_arguments_with_oracle_feedback(
    arguments: Dict[str, Any],
    oracle_feedback: Any,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(oracle_feedback, dict):
        return build_policy_self_verification_payload(arguments), {}
    merged_source = copy.deepcopy(arguments)
    merged_source.update(copy.deepcopy(oracle_feedback))
    return build_policy_self_verification_payload(merged_source), _extract_teacher_judge_labels(oracle_feedback)


def _apply_oracle_verifier_feedback(
    tool_message: Dict[str, Any],
    *,
    step: Dict[str, Any],
) -> Dict[str, Any]:
    if str(step.get("tool") or "") != "verify_hypothesis":
        return tool_message
    oracle_feedback = step.get("oracle_verifier_feedback")
    if not isinstance(oracle_feedback, dict):
        return tool_message

    arguments = dict(step.get("arguments") or {})
    merged_source = copy.deepcopy(arguments)
    merged_source.update(copy.deepcopy(oracle_feedback))
    payload = build_policy_self_verification_payload(merged_source)
    payload.update(_extract_teacher_judge_labels(oracle_feedback))
    return {
        "role": "tool",
        "name": "verify_hypothesis",
        "content": [{"type": "text", "text": _json_dumps(payload)}],
    }


def _is_anomaly_record(record: Dict[str, Any]) -> bool:
    structured_target = record.get("structured_target") or {}
    existence = str(structured_target.get("existence") or "").strip().lower()
    if existence:
        return existence == "anomaly"
    return bool((record.get("label") or {}).get("is_anomaly"))


def _counterfactual_terminal_multiplier(record: Dict[str, Any]) -> float:
    structured_target = record.get("structured_target") or {}
    counterfactual_type = str(structured_target.get("counterfactual_type") or "none").strip().lower()
    if not counterfactual_type or counterfactual_type == "none":
        return 1.0
    return {
        "remove_actor_interaction": 1.35,
        "remove_dangerous_object": 1.55,
        "restore_safe_context": 1.25,
        "replace_risky_motion_with_normal_motion": 1.7,
        "move_event_out_of_sensitive_area": 1.7,
    }.get(counterfactual_type, 1.35)


def _oracle_step_sample_weight(step: Dict[str, Any], *, record: Dict[str, Any]) -> float:
    tool_name = str(step.get("tool") or "")
    weight = 1.0
    if tool_name == "scan_timeline":
        weight = 0.5
    elif tool_name == "finalize_case":
        weight = 2.0
    elif tool_name == "verify_hypothesis":
        merged = copy.deepcopy(step.get("arguments") or {})
        if isinstance(step.get("oracle_verifier_feedback"), dict):
            merged.update(copy.deepcopy(step.get("oracle_verifier_feedback") or {}))
        verification_decision = str(merged.get("verification_decision") or "").strip().lower()
        recommended_action = str(merged.get("recommended_action") or "").strip().lower()
        if recommended_action == "finalize" or verification_decision == "sufficient":
            weight = 2.0

    if tool_name in {"verify_hypothesis", "finalize_case"} and _is_anomaly_record(record):
        weight *= 2.0
        weight *= _counterfactual_terminal_multiplier(record)
    return float(weight)


def _serialize_message_content(
    content: List[Dict[str, Any]],
    *,
    multimodal_cache: Dict[str, Any],
) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    video_path = str(multimodal_cache.get("video_path") or "")
    for item in content:
        if item.get("type") != "image" or "image" not in item:
            serialized.append(copy.deepcopy(item))
            continue

        image_ref: Dict[str, Any] = {"video_path": video_path}
        if item.get("sampled_frame_index") is not None:
            image_ref["sampled_frame_index"] = int(item["sampled_frame_index"])
        if item.get("raw_frame_index") is not None:
            image_ref["raw_frame_index"] = int(item["raw_frame_index"])
        if item.get("timestamp_sec") is not None:
            image_ref["timestamp_sec"] = float(item["timestamp_sec"])

        serialized_item = copy.deepcopy(item)
        serialized_item.pop("image", None)
        serialized_item.pop("sampled_frame_index", None)
        serialized_item.pop("raw_frame_index", None)
        serialized_item.pop("timestamp_sec", None)
        serialized_item["image_ref"] = image_ref
        serialized.append(serialized_item)
    return serialized


def _serialize_messages(
    messages: List[Dict[str, Any]],
    *,
    multimodal_cache: Dict[str, Any],
) -> List[Dict[str, Any]]:
    serialized_messages: List[Dict[str, Any]] = []
    for message in messages:
        serialized_message = copy.deepcopy(message)
        serialized_message["content"] = _serialize_message_content(
            list(message.get("content") or []),
            multimodal_cache=multimodal_cache,
        )
        serialized_messages.append(serialized_message)
    return serialized_messages


def _serialize_message(
    message: Dict[str, Any],
    *,
    multimodal_cache: Dict[str, Any],
) -> Dict[str, Any]:
    serialized_message = copy.deepcopy(message)
    serialized_message["content"] = _serialize_message_content(
        list(message.get("content") or []),
        multimodal_cache=multimodal_cache,
    )
    return serialized_message


def _prepared_rollout_context(multimodal_cache: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": str(multimodal_cache.get("question") or ""),
        "duration": multimodal_cache.get("duration"),
        "fps": multimodal_cache.get("fps"),
        "video_path": str(multimodal_cache.get("video_path") or ""),
        "frame_indices": list(multimodal_cache.get("frame_indices") or []),
        "tool_io": copy.deepcopy(multimodal_cache.get("tool_io") or {}),
    }


def _extract_message_text(message: Dict[str, Any]) -> str:
    for item in list(message.get("content") or []):
        if isinstance(item, dict) and item.get("type") == "text":
            return str(item.get("text") or "")
    return ""


def _extract_think_text(response_text: str) -> str:
    match = THINK_BLOCK_PATTERN.search(str(response_text or ""))
    if not match:
        return ""
    return str(match.group(1) or "")


def _render_tool_call_with_preserved_think(
    original_response_text: str,
    *,
    tool_name: str,
    arguments: Dict[str, Any],
) -> str:
    think_text = _extract_think_text(original_response_text)
    tool_payload = {"name": tool_name, "arguments": arguments}
    if think_text:
        return f"<think>{think_text}</think><tool_call>{_json_dumps(tool_payload)}</tool_call>"
    return f"<tool_call>{_json_dumps(tool_payload)}</tool_call>"


def _extract_tool_call_arguments(response_text: str) -> Dict[str, Any]:
    actions, contents = parse_actions_and_contents([str(response_text or "")])
    if not actions or actions[0] != "tool_call":
        return {}
    parsed_content = contents[0]
    if not isinstance(parsed_content, dict):
        return {}
    function_payload = parsed_content.get("function") or {}
    arguments = function_payload.get("arguments") or {}
    return copy.deepcopy(arguments) if isinstance(arguments, dict) else {}


def _teacher_decision_to_recommended_action(decision: str) -> str:
    normalized = str(decision or "").strip().lower()
    if normalized == "sufficient":
        return "finalize"
    if normalized == "misaligned":
        return "revise_claim"
    if normalized == "redundant":
        return "refine_evidence"
    return "continue_search"


def _build_teacher_override_verify_payload(example: Dict[str, Any]) -> Dict[str, Any]:
    base_arguments = _extract_tool_call_arguments(example.get("target_response") or "")
    if not base_arguments:
        return {}
    merged = copy.deepcopy(base_arguments)
    teacher_scores = dict(example.get("teacher_judge_scores") or {})
    if "sufficiency" in teacher_scores:
        merged["sufficiency_score"] = teacher_scores.get("sufficiency")
    if "necessity" in teacher_scores:
        merged["necessity_score"] = teacher_scores.get("necessity")
    if "finalize_readiness" in teacher_scores:
        merged["finalize_readiness_score"] = teacher_scores.get("finalize_readiness")
    if "counterfactual_faithfulness" in teacher_scores:
        merged["counterfactual_faithfulness"] = teacher_scores.get("counterfactual_faithfulness")
    teacher_decision = str(example.get("teacher_judge_decision") or "").strip().lower()
    if teacher_decision:
        merged["verification_decision"] = teacher_decision
        merged["recommended_action"] = _teacher_decision_to_recommended_action(teacher_decision)
    rationale = str(example.get("teacher_judge_rationale") or "").strip()
    if rationale:
        merged["rationale"] = rationale
    return build_policy_self_verification_payload(merged)


def _build_verify_tool_observation_message(
    payload: Dict[str, Any],
    *,
    rollout_context: Optional[Dict[str, Any]],
    config: SaverAgentConfig,
) -> Dict[str, Any]:
    adapter = TimeSearchRolloutAdapter(config=config)
    tool_message = {
        "role": "tool",
        "name": "verify_hypothesis",
        "content": [{"type": "text", "text": _json_dumps(payload)}],
    }
    return adapter.adapt_tool_observation(
        tool_message,
        multimodal_cache=copy.deepcopy(rollout_context or {}),
    )


def _patch_verify_history_messages(
    messages: List[Dict[str, Any]],
    *,
    original_response_text: str,
    replacement_assistant_message: Dict[str, Any],
    replacement_tool_message: Dict[str, Any],
) -> List[Dict[str, Any]]:
    patched = copy.deepcopy(messages)
    for idx in range(len(patched) - 1):
        current = patched[idx]
        next_message = patched[idx + 1]
        if str(current.get("role") or "") != "assistant":
            continue
        if str(next_message.get("role") or "") != "tool":
            continue
        if str(next_message.get("name") or "") != "verify_hypothesis":
            continue
        if _extract_message_text(current) != str(original_response_text or ""):
            continue
        patched[idx] = copy.deepcopy(replacement_assistant_message)
        patched[idx + 1] = copy.deepcopy(replacement_tool_message)
    return patched


def _normalize_group_sample_weights(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = [copy.deepcopy(example) for example in examples]
    if not normalized:
        return normalized
    weights = [max(0.0, float(example.get("sample_weight", 1.0) or 0.0)) for example in normalized]
    total = sum(weights)
    if total <= 0.0:
        weights = [1.0 for _ in normalized]
        total = float(len(weights) or 1)
    for example, weight in zip(normalized, weights):
        example["sample_weight"] = float(weight) / float(total)
    return normalized


def _upweight_terminal_teacher_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    updated = [copy.deepcopy(example) for example in examples]
    for example in updated:
        tool_name = str(example.get("tool_name") or "")
        if tool_name == "verify_hypothesis":
            example["sample_weight"] = float(example.get("sample_weight", 1.0) or 0.0) * 1.35
        elif tool_name == "finalize_case":
            example["sample_weight"] = float(example.get("sample_weight", 1.0) or 0.0) * 1.5
    return updated


def rebuild_teacher_rollout_primary_examples(
    examples: Sequence[Dict[str, Any]],
    *,
    config: Optional[SaverAgentConfig] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = copy.deepcopy(config) if config is not None else SaverAgentConfig()
    ordered_groups: List[tuple[str, str]] = []
    grouped_examples: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for example in examples:
        key = (str(example.get("video_id") or ""), str(example.get("split") or ""))
        if key not in grouped_examples:
            grouped_examples[key] = []
            ordered_groups.append(key)
        grouped_examples[key].append(copy.deepcopy(example))

    rebuilt_examples: List[Dict[str, Any]] = []
    summary = {
        "num_teacher_rollout_records": 0,
        "num_teacher_override_examples": 0,
        "num_synthetic_finalize_examples": 0,
        "num_terminal_upweighted_records": 0,
    }

    for key in ordered_groups:
        group = list(grouped_examples.get(key) or [])
        group.sort(key=lambda example: int(example.get("step_index") or 0))
        has_teacher_group = any(
            is_teacher_judge_candidate(example) and has_teacher_judge_labels(example)
            for example in group
        )
        if not has_teacher_group:
            rebuilt_examples.extend(group)
            continue

        adapter = TimeSearchRolloutAdapter(config=config)
        replacements: List[Dict[str, Any]] = []
        rebuilt_group: List[Dict[str, Any]] = []
        group_changed = False
        group_structure_changed = False

        for idx, original_example in enumerate(group):
            example = copy.deepcopy(original_example)
            patched_messages = copy.deepcopy(example.get("messages") or [])
            for replacement in replacements:
                patched_messages = _patch_verify_history_messages(
                    patched_messages,
                    original_response_text=replacement["original_response_text"],
                    replacement_assistant_message=replacement["assistant_message"],
                    replacement_tool_message=replacement["tool_message"],
                )
            example["messages"] = patched_messages
            example["source"] = "teacher_rollout_primary"

            if is_teacher_judge_candidate(original_example) and has_teacher_judge_labels(original_example):
                verify_payload = _build_teacher_override_verify_payload(original_example)
                if verify_payload:
                    updated_response = _render_tool_call_with_preserved_think(
                        str(original_example.get("target_response") or ""),
                        tool_name="verify_hypothesis",
                        arguments=verify_payload,
                    )
                    example["target_response"] = updated_response
                    assistant_message = adapter.build_assistant_message(updated_response)
                    tool_message = _build_verify_tool_observation_message(
                        verify_payload,
                        rollout_context=example.get("rollout_context") or {},
                        config=config,
                    )
                    replacements.append(
                        {
                            "original_response_text": str(original_example.get("target_response") or ""),
                            "assistant_message": assistant_message,
                            "tool_message": tool_message,
                        }
                    )
                    summary["num_teacher_override_examples"] += 1
                    group_changed = True
                    rebuilt_group.append(example)

                    if str(verify_payload.get("recommended_action") or "") == "finalize":
                        finalize_template = None
                        for candidate in group[idx + 1 :]:
                            if str(candidate.get("tool_name") or "") == "finalize_case":
                                finalize_template = copy.deepcopy(candidate)
                                break
                        if finalize_template is not None:
                            finalize_template["messages"] = copy.deepcopy(example["messages"]) + [
                                copy.deepcopy(assistant_message),
                                copy.deepcopy(tool_message),
                            ]
                            finalize_template["source"] = "teacher_rollout_primary"
                            rebuilt_group.append(finalize_template)
                            answer_template = None
                            for candidate in group[idx + 1 :]:
                                if str(candidate.get("target_action") or "") == "answer":
                                    answer_template = copy.deepcopy(candidate)
                                    break
                            if answer_template is not None:
                                patched_answer_messages = copy.deepcopy(answer_template.get("messages") or [])
                                for replacement in replacements:
                                    patched_answer_messages = _patch_verify_history_messages(
                                        patched_answer_messages,
                                        original_response_text=replacement["original_response_text"],
                                        replacement_assistant_message=replacement["assistant_message"],
                                        replacement_tool_message=replacement["tool_message"],
                                    )
                                answer_template["messages"] = patched_answer_messages
                            rebuilt_group.append(
                                _answer_example_from_finalize(
                                    finalize_template,
                                    source="teacher_rollout_primary",
                                    fallback_sample_weight=float(finalize_template.get("sample_weight", 1.0) or 1.0) * ANSWER_AUXILIARY_WEIGHT_MULTIPLIER,
                                    answer_template=answer_template,
                                    config=config,
                                )
                            )
                            summary["num_synthetic_finalize_examples"] += 1
                            group_structure_changed = True
                            break
                    continue

            rebuilt_group.append(example)

        if not group_changed:
            rebuilt_examples.extend(group)
            continue

        has_answer_example = any(str(example.get("target_action") or "") == "answer" for example in rebuilt_group)
        if not has_answer_example:
            finalize_examples = [
                example
                for example in rebuilt_group
                if str(example.get("tool_name") or "") == "finalize_case"
            ]
            if finalize_examples:
                finalize_example = copy.deepcopy(finalize_examples[-1])
                rebuilt_group.append(
                        _answer_example_from_finalize(
                            finalize_example,
                            source="teacher_rollout_primary",
                            fallback_sample_weight=float(finalize_example.get("sample_weight", 1.0) or 1.0) * ANSWER_AUXILIARY_WEIGHT_MULTIPLIER,
                            config=config,
                        )
                    )
                group_structure_changed = True

        should_upweight_terminals = any(
            str(example.get("tool_name") or "") in {"verify_hypothesis", "finalize_case"}
            and (
                str(example.get("teacher_judge_decision") or "").strip().lower() == "sufficient"
                or '"recommended_action":"finalize"' in str(example.get("target_response") or "")
            )
            for example in rebuilt_group
        )
        if should_upweight_terminals:
            rebuilt_group = _upweight_terminal_teacher_examples(rebuilt_group)
            summary["num_terminal_upweighted_records"] += 1
        if group_structure_changed or len(rebuilt_group) != len(group):
            rebuilt_examples.extend(_normalize_group_sample_weights(rebuilt_group))
        else:
            rebuilt_examples.extend(rebuilt_group)
        summary["num_teacher_rollout_records"] += 1

    return rebuilt_examples, summary


def _ensure_agent_train_view(record: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = {"agent_task", "structured_target", "tool_io"}
    if required_keys.issubset(record.keys()):
        return record
    adapter = CanonicalSaverAdapter(ConverterConfig())
    return adapter.convert(record, mode="agent_train")


def _ensure_oracle_sft(record: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(record.get("oracle_sft"), dict):
        return record["oracle_sft"]
    adapter = CanonicalSaverAdapter(ConverterConfig())
    agent_train_view = _ensure_agent_train_view(record)
    return adapter._build_oracle_sft(agent_train_view)  # Reuse the existing oracle warm-start builder for supervision.


def _clamp_score(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _weighted_verifier_turn_credit(turn: Dict[str, Any]) -> float:
    derived = turn.get("verifier_derived_scores") or {}
    consistency = float(derived.get("consistency", 0.0) or 0.0)
    sufficiency = float(derived.get("sufficiency", 0.0) or 0.0)
    necessity = float(derived.get("necessity", 0.0) or 0.0)
    finalize_readiness = float(derived.get("finalize_readiness", 0.0) or 0.0)
    counterfactual = float(derived.get("counterfactual_faithfulness", 0.0) or 0.0)
    primary_status = str(turn.get("verifier_primary_status") or "")

    verification_credit = _clamp_score(
        0.35 * float(PRIMARY_STATUS_REWARD.get(primary_status, 0.0))
        + 0.20 * sufficiency
        + 0.15 * necessity
        + 0.15 * finalize_readiness
        + 0.15 * counterfactual
    )
    stage_credit = _clamp_score(0.5 * consistency + 0.5 * sufficiency)
    return (
        DEFAULT_COMPONENT_WEIGHTS["protocol_finalize_reward"] * verification_credit
        + DEFAULT_COMPONENT_WEIGHTS["stage_necessity_reward"] * stage_credit
    )


def _turn_search_has_progress(turn: Dict[str, Any]) -> bool:
    if str(turn.get("tool_name") or "") not in {"scan_timeline", "seek_evidence"}:
        return False
    if list(turn.get("new_evidence_ids") or []):
        return True
    state_delta = dict(turn.get("state_delta") or {})
    if list(state_delta.get("new_evidence_windows") or []):
        return True
    if list(state_delta.get("new_visited_windows") or []):
        return True
    return False


def _canonical_search_signature_from_turn(turn: Dict[str, Any]) -> Optional[str]:
    tool_name = str(turn.get("tool_name") or "")
    if tool_name not in {"scan_timeline", "seek_evidence"}:
        return None
    parsed_tool_call = dict(turn.get("parsed_tool_call") or {})
    arguments = parsed_tool_call.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    normalized_arguments = {
        str(key): (
            round(float(value), 4)
            if isinstance(value, float)
            else " ".join(value.strip().lower().split())
            if isinstance(value, str)
            else value
        )
        for key, value in sorted(arguments.items())
    }
    return _json_dumps({"name": tool_name, "arguments": normalized_arguments})


def _compute_turn_credit(
    turn: Dict[str, Any],
    *,
    search_bonus: float,
    evidence_bonus: float,
    finalize_bonus: float,
    invalid_penalty: float,
    invalid_attempt_count: int = 0,
    search_has_progress: bool = True,
    repeated_search: bool = False,
    search_after_finalize_recommendation: bool = False,
) -> float:
    valid_action = bool(turn.get("valid_action", turn.get("action") in {"tool_call", "answer"}))
    tool_name = str(turn.get("tool_name") or "")
    step_index = int(turn.get("step_index") or 0)
    turn_credit = 0.0

    if int(invalid_attempt_count) > 0:
        turn_credit -= abs(float(invalid_penalty)) * float(int(invalid_attempt_count))

    if not valid_action:
        turn_credit -= abs(float(invalid_penalty))

    if valid_action and tool_name in {"scan_timeline", "seek_evidence"}:
        if search_has_progress:
            turn_credit += float(search_bonus)
        else:
            turn_credit -= abs(float(search_bonus))
        if repeated_search:
            turn_credit -= 0.25
        if search_after_finalize_recommendation:
            turn_credit -= 0.6

    turn_credit += float(evidence_bonus) * float(len(turn.get("new_evidence_ids") or []))

    if tool_name == "verify_hypothesis":
        turn_credit += _weighted_verifier_turn_credit(turn)

    if tool_name == "finalize_case" and turn.get("new_finalized_case") is not None:
        turn_credit += float(finalize_bonus)

    return float(turn_credit)


def _compute_turn_level_advantages(
    rollout: Dict[str, Any],
    *,
    gamma: float,
    alpha: float,
    search_bonus: float,
    evidence_bonus: float,
    finalize_bonus: float,
    invalid_penalty: float,
) -> List[Dict[str, float]]:
    turns = list(rollout.get("turns") or [])
    if not turns:
        return []
    has_structured_terminal_tool = any(
        str(turn.get("tool_name") or "") in {"verify_hypothesis", "finalize_case"}
        for turn in turns
    )

    invalid_attempts_by_step: Dict[int, int] = {}
    for invalid_attempt in list(rollout.get("invalid_attempts") or []):
        try:
            step_index = int(invalid_attempt.get("step_index") or 0)
        except Exception:
            step_index = 0
        if step_index <= 0:
            continue
        invalid_attempts_by_step[step_index] = invalid_attempts_by_step.get(step_index, 0) + 1

    rollout_advantage = float(
        rollout.get("group_advantage", (rollout.get("reward_summary") or {}).get("total_reward", 0.0)) or 0.0
    )
    turn_credits: List[float] = []
    previous_search_signature = None
    pending_finalize_recommendation = False
    for turn_index, turn in enumerate(turns):
        current_search_signature = _canonical_search_signature_from_turn(turn)
        search_after_finalize_recommendation = bool(
            pending_finalize_recommendation and current_search_signature is not None
        )
        turn_credit = _compute_turn_credit(
            turn,
            search_bonus=search_bonus,
            evidence_bonus=evidence_bonus,
            finalize_bonus=finalize_bonus,
            invalid_penalty=invalid_penalty,
            invalid_attempt_count=invalid_attempts_by_step.get(int(turn.get("step_index") or 0), 0),
            search_has_progress=_turn_search_has_progress(turn),
            repeated_search=bool(
                current_search_signature
                and previous_search_signature
                and current_search_signature == previous_search_signature
            ),
            search_after_finalize_recommendation=search_after_finalize_recommendation,
        )
        tool_name = str(turn.get("tool_name") or "")
        if (
            not has_structured_terminal_tool
            and bool(turn.get("valid_action", turn.get("action") == "answer"))
            and str(turn.get("action") or "") == "answer"
        ):
            turn_credit += float(finalize_bonus)
        turn_credits.append(turn_credit)
        if tool_name == "verify_hypothesis":
            pending_finalize_recommendation = str(turn.get("verifier_recommended_action") or "") == "finalize"
        elif tool_name == "finalize_case":
            pending_finalize_recommendation = False
        previous_search_signature = current_search_signature

    discounted_returns = [0.0 for _ in turn_credits]
    running_return = 0.0
    for idx in range(len(turn_credits) - 1, -1, -1):
        running_return = float(turn_credits[idx]) + float(gamma) * running_return
        discounted_returns[idx] = running_return

    if abs(rollout_advantage) < 1e-8:
        return [
            {
                "rollout_advantage": rollout_advantage,
                "turn_credit": float(turn_credits[idx]),
                "discounted_return": float(discounted_returns[idx]),
                "advantage": 0.0,
            }
            for idx in range(len(turn_credits))
        ]

    mean_turn_credit = sum(turn_credits) / float(len(turn_credits))
    centered_turn_credits = [value - mean_turn_credit for value in turn_credits]
    mean_abs_centered = sum(abs(value) for value in centered_turn_credits) / float(len(centered_turn_credits))

    if mean_abs_centered <= 1e-8:
        return [
            {
                "rollout_advantage": rollout_advantage,
                "turn_credit": float(turn_credits[idx]),
                "discounted_return": float(discounted_returns[idx]),
                "advantage": float(rollout_advantage),
            }
            for idx in range(len(turn_credits))
        ]

    advantages: List[Dict[str, float]] = []
    credit_scale = max(abs(float(rollout_advantage)), 0.25)
    for idx, centered_value in enumerate(centered_turn_credits):
        normalized_centered = float(centered_value) / float(mean_abs_centered)
        turn_advantage = float(rollout_advantage) + float(alpha) * credit_scale * normalized_centered
        advantages.append(
            {
                "rollout_advantage": rollout_advantage,
                "turn_credit": float(turn_credits[idx]),
                "discounted_return": float(discounted_returns[idx]),
                "advantage": float(turn_advantage),
            }
        )
    return advantages


def build_oracle_sft_examples(
    item: Dict[str, Any],
    record: Dict[str, Any],
    *,
    config: Optional[SaverAgentConfig] = None,
    serialize_messages: bool = False,
) -> List[Dict[str, Any]]:
    config = copy.deepcopy(config) if config is not None else SaverAgentConfig()
    adapter = TimeSearchRolloutAdapter(config=config)
    environment = SaverVideoInteraction()
    messages = adapter.build_initial_messages(item)
    multimodal_cache = item["multimodal_cache"]
    serialized_messages = (
        _serialize_messages(messages, multimodal_cache=multimodal_cache)
        if serialize_messages
        else None
    )
    state = SaverEnvironmentState()
    oracle_sft = _ensure_oracle_sft(record)
    trajectory = list(oracle_sft.get("trajectory") or [])
    examples: List[Dict[str, Any]] = []
    final_decision = copy.deepcopy(
        oracle_sft.get("final_decision")
        or record.get("structured_target")
        or state.finalized_case
        or {}
    )
    raw_step_weights = [_oracle_step_sample_weight(step, record=record) for step in trajectory]

    for step, sample_weight in zip(trajectory, raw_step_weights):
        tool_name = str(step.get("tool") or "")
        arguments = copy.deepcopy(step.get("arguments") or {})
        response_arguments = copy.deepcopy(arguments)
        teacher_judge_labels: Dict[str, Any] = {}
        if tool_name == "verify_hypothesis":
            response_arguments, teacher_judge_labels = _merge_verify_arguments_with_oracle_feedback(
                arguments,
                step.get("oracle_verifier_feedback"),
            )
        elif tool_name == "finalize_case":
            response_arguments = build_finalize_case_payload(arguments)
        response_text = _assistant_tool_response(
            tool_name,
            response_arguments,
            record=record,
            state=state,
        )
        example = {
            "video_id": item.get("video_id"),
            "split": item.get("split"),
            "step_index": len(examples) + 1,
            "source": "oracle_sft",
            "target_action": "tool_call",
            "target_response": response_text,
            "messages": (
                copy.deepcopy(serialized_messages)
                if serialize_messages
                else copy.deepcopy(messages)
            ),
                "sample_weight": float(sample_weight),
            "tool_name": tool_name,
            "proposal_supervision": (
                _proposal_supervision_for_query(
                    record,
                    arguments.get("query", ""),
                )
                if tool_name == "seek_evidence"
                else {}
            ),
        }
        if teacher_judge_labels:
            example.update(teacher_judge_labels)
        if serialize_messages:
            example["rollout_context"] = _prepared_rollout_context(multimodal_cache)
        examples.append(example)

        next_obs, _, _, _, next_states = environment.execute_predictions(
            [response_text],
            [multimodal_cache],
            [state],
            [True],
        )
        state = next_states[0]
        assistant_message = adapter.build_assistant_message(response_text)
        messages.append(assistant_message)
        if serialized_messages is not None:
            serialized_messages.append(copy.deepcopy(assistant_message))
        tool_message = next_obs[0]
        if isinstance(tool_message, dict) and tool_message.get("role") == "tool":
            tool_message = _apply_oracle_verifier_feedback(tool_message, step=step)
            adapted_tool_message = adapter.adapt_tool_observation(tool_message, multimodal_cache)
            messages.append(adapted_tool_message)
            if serialized_messages is not None:
                serialized_messages.append(
                    _serialize_message(
                        adapted_tool_message,
                        multimodal_cache=multimodal_cache,
                    )
                )
        if tool_name == "finalize_case":
            answer_example = _answer_example_from_finalize(
                example,
                record=record,
                source="oracle_sft",
                fallback_sample_weight=float(sample_weight) * ANSWER_AUXILIARY_WEIGHT_MULTIPLIER,
                config=config,
            )
            if serialize_messages:
                answer_example["rollout_context"] = _prepared_rollout_context(multimodal_cache)
            examples.append(answer_example)
    return _normalize_group_sample_weights(examples)


def _coerce_state_from_turn(turn: Dict[str, Any], key: str) -> SaverEnvironmentState:
    payload = _sanitize_state_selection_payload(dict(turn.get(key) or {}))
    return SaverEnvironmentState(
        visited_windows=list(payload.get("visited_windows") or []),
        evidence_ledger=list(payload.get("evidence_ledger") or []),
        verification_records=list(payload.get("verification_records") or []),
        finalized_case=dict(payload["finalized_case"]) if isinstance(payload.get("finalized_case"), dict) else None,
        last_claim=dict(payload["last_claim"]) if isinstance(payload.get("last_claim"), dict) else None,
        active_evidence_window_ids=list(payload.get("active_evidence_window_ids") or []),
        verifier_cache=list(payload.get("verifier_cache") or []),
        next_evidence_id=int(payload.get("next_evidence_id") or 1),
        next_window_id=int(payload.get("next_window_id") or 1),
    )


def _known_evidence_window_ids_from_state_payload(state_payload: Dict[str, Any] | SaverEnvironmentState) -> set[str]:
    if isinstance(state_payload, SaverEnvironmentState):
        entries = list(state_payload.evidence_ledger or [])
    else:
        entries = list((state_payload or {}).get("evidence_ledger") or [])
    return {
        str(entry.get("window_id")).strip()
        for entry in entries
        if str(entry.get("window_id") or "").strip()
    }


def _sanitize_window_ids_against_state(
    window_ids: Sequence[str],
    *,
    state_payload: Dict[str, Any] | SaverEnvironmentState,
) -> List[str]:
    known_window_ids = _known_evidence_window_ids_from_state_payload(state_payload)
    sanitized: List[str] = []
    seen = set()
    for value in window_ids or []:
        window_id = str(value).strip()
        if not window_id or window_id not in known_window_ids or window_id in seen:
            continue
        sanitized.append(window_id)
        seen.add(window_id)
    return sanitized


def _sanitize_state_selection_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(payload or {})
    sanitized["active_evidence_window_ids"] = _sanitize_window_ids_against_state(
        sanitized.get("active_evidence_window_ids") or [],
        state_payload=sanitized,
    )
    verification_records = []
    for record in sanitized.get("verification_records") or []:
        updated_record = dict(record or {})
        for key in ("selected_window_ids", "verified_window_ids", "best_effort_window_ids"):
            if key in updated_record:
                updated_record[key] = _sanitize_window_ids_against_state(
                    updated_record.get(key) or [],
                    state_payload=sanitized,
                )
        verification_records.append(updated_record)
    sanitized["verification_records"] = verification_records
    return sanitized


def _latest_claim_from_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    claim = turn.get("latest_claim_after") or (turn.get("state_after") or {}).get("last_claim") or {}
    return copy.deepcopy(claim) if isinstance(claim, dict) else {}


def _proposal_supervision_for_query(
    record: Dict[str, Any],
    query: str,
) -> Dict[str, Any]:
    proposal_supervision = record.get("proposal_supervision") or {}
    candidate_texts = set()
    normalized_query = normalize_query_text(str(query or ""))
    if normalized_query:
        candidate_texts.add(normalized_query)

    best_match: Dict[str, Any] = {}
    best_score = 0.0
    for query_group in proposal_supervision.get("queries") or []:
        normalized_entries = list(query_group.get("normalized_queries") or [])
        normalized_texts = {
            normalize_query_text(str(entry.get("text") or ""))
            for entry in normalized_entries
            if str(entry.get("text") or "").strip()
        }
        query_text = normalize_query_text(str(query_group.get("query_text") or ""))
        if query_text:
            normalized_texts.add(query_text)
        overlap_count = len(candidate_texts & normalized_texts)
        if overlap_count <= 0:
            candidate_tokens = {
                token
                for text in candidate_texts
                for token in normalize_query_text(text).split()
                if token
            }
            normalized_entry_tokens = {
                token
                for text in normalized_texts
                for token in normalize_query_text(text).split()
                if token
            }
            if candidate_tokens and normalized_entry_tokens:
                overlap_count = len(candidate_tokens & normalized_entry_tokens)
        if overlap_count <= 0:
            continue
        weight_bonus = 0.0
        for entry in normalized_entries:
            text = normalize_query_text(str(entry.get("text") or ""))
            if text in candidate_texts:
                weight_bonus = max(weight_bonus, float(entry.get("weight") or 0.0))
        score = float(overlap_count) + weight_bonus
        if score <= best_score:
            continue
        best_score = score
        best_match = {
            "query_id": query_group.get("query_id"),
            "raw_text": query_group.get("raw_text"),
            "normalized_queries": copy.deepcopy(normalized_entries),
            "linked_moment_ids": list(query_group.get("linked_moment_ids") or []),
            "linked_roles": list(query_group.get("linked_roles") or []),
            "linked_windows_sec": copy.deepcopy(query_group.get("linked_windows_sec") or []),
            "alignment_source": query_group.get("alignment_source"),
        }
    return best_match


def _proposal_metadata_from_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    if str(turn.get("tool_name") or "") != "seek_evidence":
        return {}
    return {
        "backend": turn.get("proposal_backend"),
        "feature_cache_used": turn.get("feature_cache_used"),
        "query_raw": turn.get("proposal_query_raw"),
        "query_normalized": turn.get("proposal_query_normalized"),
        "query_source": turn.get("proposal_query_source"),
        "candidate_count": turn.get("proposal_candidate_count"),
        "candidate_frame_indices": list(turn.get("proposal_candidate_frame_indices") or []),
        "candidate_frame_scores": list(turn.get("proposal_candidate_frame_scores") or []),
        "candidate_windows": copy.deepcopy(turn.get("proposal_candidate_windows") or []),
        "selected_frame_indices": list(turn.get("proposal_selected_frame_indices") or []),
        "selected_frame_scores": list(turn.get("proposal_selected_frame_scores") or []),
        "fallback_reason": turn.get("proposal_fallback_reason"),
    }


def _selected_window_ids_from_turn(turn: Dict[str, Any]) -> List[str]:
    state_after = _sanitize_state_selection_payload(dict(turn.get("state_after") or {}))
    values = (
        turn.get("selected_window_ids_after")
        or turn.get("verifier_verified_window_ids")
        or turn.get("verifier_best_effort_window_ids")
        or state_after.get("active_evidence_window_ids")
        or []
    )
    return _sanitize_window_ids_against_state(values, state_payload=state_after)


def _selected_evidence_ids_from_turn(turn: Dict[str, Any]) -> List[str]:
    values = turn.get("selected_evidence_ids_after") or []
    if values:
        return [str(value) for value in values if value]
    state_after = _sanitize_state_selection_payload(dict(turn.get("state_after") or {}))
    selected_window_ids = set(_selected_window_ids_from_turn(turn))
    evidence_ids: List[str] = []
    for entry in state_after.get("evidence_ledger") or []:
        window_id = str(entry.get("window_id") or "")
        evidence_id = entry.get("evidence_id")
        if window_id in selected_window_ids and evidence_id:
            evidence_ids.append(str(evidence_id))
    return evidence_ids


def _clamp_unit_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _clamp_signed_score(value: Any) -> float:
    try:
        return max(-1.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _group_relative_advantages_local(records: List[Dict[str, Any]], *, eps: float = 1e-6) -> List[Dict[str, Any]]:
    if not records:
        return []
    rewards = [float(record.get("branch_reward") or 0.0) for record in records]
    mean_reward = sum(rewards) / float(len(rewards))
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / float(len(rewards))
    std_reward = variance ** 0.5
    updated: List[Dict[str, Any]] = []
    for record, reward in zip(records, rewards):
        local_advantage = 0.0 if std_reward <= eps else (reward - mean_reward) / (std_reward + eps)
        enriched = dict(record)
        enriched["local_advantage"] = round(float(local_advantage), 6)
        enriched["group_reward_mean"] = round(float(mean_reward), 6)
        enriched["group_reward_std"] = round(float(std_reward), 6)
        updated.append(enriched)
    return updated


def _counterfactual_stage_bonus_local(profile: Dict[str, Any]) -> float:
    summary = dict(profile.get("summary") or {}) if isinstance(profile, dict) else {}
    stage_necessity = summary.get("stage_necessity")
    if not isinstance(stage_necessity, dict):
        stage_necessity = profile.get("stage_necessity") or {}
    if not isinstance(stage_necessity, dict) or not stage_necessity:
        return 0.0
    label_scores = {
        "narrative_only": 0.4,
        "decision_critical": 1.0,
        "finalize_critical": 1.0,
        "optional": 0.0,
        "non_critical": -0.4,
        "not_observed": 0.0,
    }
    values = [
        float(label_scores.get(str(value).strip().lower(), 0.0))
        for value in stage_necessity.values()
    ]
    if not values:
        return 0.0
    return _clamp_signed_score(sum(values) / float(len(values)))


def _available_counterfactual_branch(
    branches: Dict[str, Any],
    *names: str,
) -> Dict[str, Any]:
    for name in names:
        branch = branches.get(name) or {}
        if bool(branch.get("available")):
            return dict(branch)
    return {}


def build_reward_weighted_examples(
    item: Dict[str, Any],
    rollout: Dict[str, Any],
    *,
    config: Optional[SaverAgentConfig] = None,
    include_invalid: bool = False,
    serialize_messages: bool = False,
    turn_advantage_gamma: float = 0.9,
    turn_advantage_alpha: float = 0.5,
    turn_search_bonus: float = 0.05,
    turn_evidence_bonus: float = 0.1,
    turn_finalize_bonus: float = 0.2,
    turn_invalid_penalty: float = 0.75,
) -> List[Dict[str, Any]]:
    config = copy.deepcopy(config) if config is not None else SaverAgentConfig()
    adapter = TimeSearchRolloutAdapter(config=config)
    environment = SaverVideoInteraction()
    messages = adapter.build_initial_messages(item)
    multimodal_cache = item["multimodal_cache"]
    state = SaverEnvironmentState()
    sample_weight = float((rollout.get("reward_summary") or {}).get("total_reward") or 0.0)
    turn_advantages = _compute_turn_level_advantages(
        rollout,
        gamma=turn_advantage_gamma,
        alpha=turn_advantage_alpha,
        search_bonus=turn_search_bonus,
        evidence_bonus=turn_evidence_bonus,
        finalize_bonus=turn_finalize_bonus,
        invalid_penalty=turn_invalid_penalty,
    )
    examples: List[Dict[str, Any]] = []
    saw_structured_terminal_tool = False

    for step_index, turn in enumerate(rollout.get("turns") or [], start=1):
        response_text = str(turn.get("assistant_response_raw") or turn.get("response") or "")
        if not response_text:
            continue
        skip_terminal_answer_example = (
            str(turn.get("action") or "") == "answer"
            and saw_structured_terminal_tool
        )
        if skip_terminal_answer_example:
            next_obs, _, _, _, next_states = environment.execute_predictions(
                [response_text],
                [multimodal_cache],
                [state],
                [True],
            )
            state = next_states[0]
            messages.append(adapter.build_assistant_message(response_text))
            tool_message = next_obs[0]
            if isinstance(tool_message, dict) and tool_message.get("role") == "tool":
                messages.append(adapter.adapt_tool_observation(tool_message, multimodal_cache))
            continue
        valid_action = bool(turn.get("valid_action", turn.get("action") in {"tool_call", "answer"}))
        teacher_signal = compute_teacher_judge_signal(turn)
        turn_advantage_info = (
            turn_advantages[step_index - 1]
            if step_index - 1 < len(turn_advantages)
            else {
                "rollout_advantage": sample_weight,
                "turn_credit": 0.0,
                "discounted_return": 0.0,
                "advantage": sample_weight,
            }
        )
        if include_invalid or valid_action:
            example = {
                "video_id": item.get("video_id"),
                "split": item.get("split"),
                "step_index": step_index,
                "source": "reward_weighted_rollout",
                "target_action": turn.get("action"),
                "target_response": response_text,
                "messages": (
                    _serialize_messages(messages, multimodal_cache=multimodal_cache)
                    if serialize_messages
                    else copy.deepcopy(messages)
                ),
                "sample_weight": float(turn_advantage_info["advantage"]),
                "advantage": float(turn_advantage_info["advantage"]),
                "rollout_advantage": float(turn_advantage_info["rollout_advantage"]),
                "advantage_metadata": {
                    "turn_credit": float(turn_advantage_info["turn_credit"]),
                    "discounted_return": float(turn_advantage_info["discounted_return"]),
                    "gamma": float(turn_advantage_gamma),
                    "alpha": float(turn_advantage_alpha),
                    "teacher_judge_reward": float(teacher_signal.get("teacher_judge_reward") or 0.0),
                    "teacher_judge_alignment": teacher_signal.get("teacher_judge_alignment"),
                },
                "tool_name": turn.get("tool_name"),
                "proposal_metadata": _proposal_metadata_from_turn(turn),
                "teacher_judge_present": bool(teacher_signal.get("teacher_judge_present")),
                "teacher_judge_alignment": teacher_signal.get("teacher_judge_alignment"),
                "teacher_judge_score_agreement": float(teacher_signal.get("teacher_judge_score_agreement") or 0.0),
                "teacher_judge_confidence": float(teacher_signal.get("teacher_judge_confidence") or 0.0),
                "teacher_judge_reward": float(teacher_signal.get("teacher_judge_reward") or 0.0),
            }
            if "teacher_judge_scores" in turn:
                example["teacher_judge_scores"] = copy.deepcopy(turn.get("teacher_judge_scores") or {})
            if turn.get("teacher_judge_decision") is not None:
                example["teacher_judge_decision"] = turn.get("teacher_judge_decision")
            if turn.get("teacher_judge_rationale") is not None:
                example["teacher_judge_rationale"] = turn.get("teacher_judge_rationale")
            examples.append(example)

        next_obs, _, _, _, next_states = environment.execute_predictions(
            [response_text],
            [multimodal_cache],
            [state],
            [True],
        )
        state = next_states[0]
        messages.append(adapter.build_assistant_message(response_text))
        tool_message = next_obs[0]
        if isinstance(tool_message, dict) and tool_message.get("role") == "tool":
            messages.append(adapter.adapt_tool_observation(tool_message, multimodal_cache))
        if str(turn.get("tool_name") or "") in {"verify_hypothesis", "finalize_case"}:
            saw_structured_terminal_tool = True
    return examples
