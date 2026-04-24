from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PRIMARY_STATUS_KEYS = ["complete", "incomplete", "redundant", "misaligned", "unknown"]
TEACHER_DECISION_KEYS = ["sufficient", "insufficient", "misaligned", "redundant", "unknown"]
TEACHER_ALIGNMENT_KEYS = ["aligned", "misaligned", "unknown"]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _extract_counterfactual_profile(record: Dict[str, Any]) -> Dict[str, Any]:
    del record
    return {}


def _counterfactual_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    summary = profile.get("summary")
    if isinstance(summary, dict):
        return summary
    return {}


def _latest_turn_verdict(record: Dict[str, Any]) -> Tuple[str, str]:
    turns = record.get("turns") or []
    for turn in reversed(turns):
        primary_status = turn.get("verifier_primary_status")
        if primary_status:
            return str(primary_status or "unknown"), "unknown"
    return "unknown", "unknown"


def extract_verifier_statuses(record: Dict[str, Any]) -> Tuple[str, str]:
    offline_verifier = record.get("offline_verifier")
    if isinstance(offline_verifier, dict):
        primary_status = str(offline_verifier.get("primary_status") or "unknown")
        return primary_status, "unknown"
    return _latest_turn_verdict(record)


def extract_teacher_judge_status(record: Dict[str, Any]) -> Tuple[str, str, Optional[float], float]:
    turns = record.get("turns") or []
    for turn in reversed(turns):
        decision = str(turn.get("teacher_judge_decision") or "").strip().lower()
        alignment = turn.get("teacher_judge_alignment")
        reward = _safe_float(
            turn.get("teacher_judge_reward"),
            _safe_float((record.get("reward_summary") or {}).get("components", {}).get("teacher_judge_reward"), 0.0),
        )
        if decision or alignment is not None or reward != 0.0:
            if decision not in TEACHER_DECISION_KEYS:
                decision = "unknown"
            if alignment is None:
                alignment_key = "unknown"
            elif float(alignment) >= 0.5:
                alignment_key = "aligned"
            else:
                alignment_key = "misaligned"
            return decision or "unknown", alignment_key, alignment, reward
    return "unknown", "unknown", None, 0.0


def _infer_policy_existence(record: Dict[str, Any]) -> str:
    final_answer = record.get("final_answer")
    if isinstance(final_answer, dict):
        return str(final_answer.get("existence") or "").strip().lower()
    state = record.get("state") or {}
    finalized_case = state.get("finalized_case")
    if isinstance(finalized_case, dict):
        return str(finalized_case.get("existence") or "").strip().lower()
    return ""


def _resolve_reference_label_group(
    record: Dict[str, Any],
    *,
    reference_data: Optional[Any] = None,
) -> str:
    if reference_data is not None:
        reference_record = reference_data.by_video_id.get(record.get("video_id"))
        if isinstance(reference_record, dict):
            label = reference_record.get("label") or {}
            if "is_anomaly" in label:
                return "anomaly" if bool(label.get("is_anomaly")) else "normal"
            structured_target = reference_record.get("structured_target") or {}
            existence = str(structured_target.get("existence") or "").strip().lower()
            if existence in {"anomaly", "normal"}:
                return existence
    existence = _infer_policy_existence(record)
    if existence in {"anomaly", "normal"}:
        return existence
    return "unknown"


def _resolve_reference_category(
    record: Dict[str, Any],
    *,
    reference_data: Optional[Any] = None,
) -> str:
    if reference_data is not None:
        reference_record = reference_data.by_video_id.get(record.get("video_id"))
        if isinstance(reference_record, dict):
            label = reference_record.get("label") or {}
            category = str(label.get("category") or "").strip().lower()
            if category:
                return category
            structured_target = reference_record.get("structured_target") or {}
            category = str(structured_target.get("category") or "").strip().lower()
            if category:
                return category

    state = record.get("state") or {}
    for source in (
        record.get("final_answer"),
        state.get("finalized_case"),
        state.get("last_claim"),
        record.get("structured_target"),
    ):
        if isinstance(source, dict):
            category = str(source.get("category") or "").strip().lower()
            if category:
                return category
    return _resolve_reference_label_group(record, reference_data=reference_data)


def _resolve_verification_mode(turn: Dict[str, Any]) -> str:
    return str(turn.get("verifier_mode") or turn.get("verification_mode") or "").strip().lower()


def _resolve_turn_stage(turn: Dict[str, Any]) -> str:
    tool_name = str(turn.get("tool_name") or "").strip().lower()
    action = str(turn.get("action") or "").strip().lower()
    verification_mode = _resolve_verification_mode(turn)
    tags = {
        str(tag).strip().lower()
        for tag in (turn.get("counterfactual_anchor_tags") or [])
        if str(tag).strip()
    }

    if tool_name in {"scan_timeline", "seek_evidence"}:
        return "search"
    if tool_name == "finalize_case":
        return "finalization"
    if action == "answer":
        return "answer"
    if tool_name != "verify_hypothesis":
        return "unknown"
    if verification_mode == "stage_check" or "search_anchor" in tags:
        return "search_verification"
    if verification_mode in {"stage_check", "final_check"}:
        return "evidence_verification"
    if "evidence_anchor" in tags:
        return "evidence_verification"
    return "verification"


def _teacher_disagreement_cases(
    records: Sequence[Dict[str, Any]],
    *,
    reference_data: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for record in records:
        for turn in record.get("turns") or []:
            if str(turn.get("tool_name") or "") != "verify_hypothesis":
                continue
            alignment = turn.get("teacher_judge_alignment")
            decision = str(turn.get("teacher_judge_decision") or "").strip().lower()
            if alignment is None and not decision:
                continue
            if alignment is not None and float(alignment) >= 0.5:
                continue
            policy_decision = str(
                turn.get("self_verification_decision")
                or turn.get("verification_decision")
                or turn.get("verifier_primary_status")
                or ""
            ).strip().lower()
            cases.append(
                {
                    "video_id": record.get("video_id"),
                    "split": record.get("split"),
                    "step_index": int(turn.get("step_index") or 0),
                    "label_group": _resolve_reference_label_group(record, reference_data=reference_data),
                    "category": _resolve_reference_category(record, reference_data=reference_data),
                    "stage": _resolve_turn_stage(turn),
                    "verification_mode": _resolve_verification_mode(turn) or "unknown",
                    "policy_decision": policy_decision or "unknown",
                    "teacher_judge_decision": decision or "unknown",
                    "teacher_judge_alignment": alignment,
                    "teacher_judge_reward": _safe_float(turn.get("teacher_judge_reward"), 0.0),
                }
            )
    return cases


def _counter_to_sorted_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _teacher_disagreement_cluster_rows(
    cases: Sequence[Dict[str, Any]],
    *,
    group_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for case in cases:
        key = tuple(str(case.get(group_key) or "unknown") for group_key in group_keys)
        bucket = grouped.setdefault(
            key,
            {
                "num_cases": 0,
                "teacher_reward_values": [],
                "teacher_decision_counter": Counter(),
                "policy_decision_counter": Counter(),
                "example_video_ids": [],
                "video_id_set": set(),
            },
        )
        bucket["num_cases"] += 1
        bucket["teacher_reward_values"].append(_safe_float(case.get("teacher_judge_reward"), 0.0))
        bucket["teacher_decision_counter"][str(case.get("teacher_judge_decision") or "unknown")] += 1
        bucket["policy_decision_counter"][str(case.get("policy_decision") or "unknown")] += 1
        video_id = str(case.get("video_id") or "")
        if video_id and video_id not in bucket["video_id_set"]:
            bucket["video_id_set"].add(video_id)
            if len(bucket["example_video_ids"]) < 5:
                bucket["example_video_ids"].append(video_id)

    rows: List[Dict[str, Any]] = []
    for key, bucket in grouped.items():
        row = {group_key: value for group_key, value in zip(group_keys, key)}
        row.update(
            {
                "num_cases": int(bucket["num_cases"]),
                "num_unique_videos": int(len(bucket["video_id_set"])),
                "mean_teacher_judge_reward": _mean(bucket["teacher_reward_values"]),
                "teacher_judge_decision_counts": _counter_to_sorted_dict(bucket["teacher_decision_counter"]),
                "policy_decision_counts": _counter_to_sorted_dict(bucket["policy_decision_counter"]),
                "example_video_ids": list(bucket["example_video_ids"]),
            }
        )
        rows.append(row)

    def _sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
        return tuple([-int(row.get("num_cases") or 0)] + [str(row.get(group_key) or "") for group_key in group_keys])

    rows.sort(key=_sort_key)
    return rows


def _verify_turn_coverage(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_records = len(records)
    records_with_verify = 0
    teacher_labeled_records = 0
    total_verify_turns = 0
    teacher_labeled_turns = 0
    for record in records:
        verify_turns = [turn for turn in (record.get("turns") or []) if str(turn.get("tool_name") or "") == "verify_hypothesis"]
        if verify_turns:
            records_with_verify += 1
            total_verify_turns += len(verify_turns)
        teacher_turns = [
            turn
            for turn in verify_turns
            if turn.get("teacher_judge_decision") or turn.get("teacher_judge_alignment") is not None
        ]
        if teacher_turns:
            teacher_labeled_records += 1
            teacher_labeled_turns += len(teacher_turns)
    return {
        "records_with_verify_turn": records_with_verify,
        "record_ratio": (_safe_float(records_with_verify) / total_records if total_records else 0.0),
        "total_verify_turns": total_verify_turns,
        "mean_verify_turns_per_record": (_safe_float(total_verify_turns) / total_records if total_records else 0.0),
        "records_with_teacher_judge": teacher_labeled_records,
        "teacher_record_ratio": (_safe_float(teacher_labeled_records) / total_records if total_records else 0.0),
        "teacher_labeled_verify_turns": teacher_labeled_turns,
    }


def _verify_health_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    parse_mode_counter = Counter()
    total_verify_turns = 0
    invalid_selected_turns = 0
    unresolved_selection_turns = 0
    verify_invalid_turns = 0
    total_invalid_attempts = 0
    invalid_attempt_reason_counts = Counter()
    repeated_search_attempts = 0
    finalize_loop_attempts = 0

    for record in records:
        for turn in record.get("turns") or []:
            if str(turn.get("tool_name") or "") != "verify_hypothesis":
                continue
            total_verify_turns += 1
            parse_mode = str(turn.get("verification_parse_mode") or "unknown")
            parse_mode_counter[parse_mode] += 1
            if list(turn.get("invalid_selected_window_ids") or []):
                invalid_selected_turns += 1
            failure_reasons = {
                str(value)
                for value in (turn.get("verifier_failure_reasons") or [])
                if str(value).strip()
            }
            if (
                "selected_evidence_not_resolved_to_known_windows" in failure_reasons
                or str(turn.get("selection_resolution_source") or "") == "unresolved"
            ):
                unresolved_selection_turns += 1
            if not bool(turn.get("valid_action", turn.get("action") in {"tool_call"})):
                verify_invalid_turns += 1
        for attempt in record.get("invalid_attempts") or []:
            total_invalid_attempts += 1
            reason = (
                str(attempt.get("guardrail_reason") or "").strip()
                or str(attempt.get("action") or "").strip()
                or "unknown"
            )
            invalid_attempt_reason_counts[reason] += 1
            if reason == "repeated_search_signature":
                repeated_search_attempts += 1
            if reason == "tool_after_finalize_case" or str(attempt.get("tool_name") or "") == "finalize_case":
                finalize_loop_attempts += 1

    return {
        "verify_parse_mode_counts": _counter_to_sorted_dict(parse_mode_counter),
        "invalid_attempt_reason_counts": _counter_to_sorted_dict(invalid_attempt_reason_counts),
        "repeated_search_rate": _safe_rate(repeated_search_attempts, total_invalid_attempts),
        "empty_selection_verify_rate": _safe_rate(unresolved_selection_turns, total_verify_turns),
        "finalize_loop_rate": _safe_rate(finalize_loop_attempts, total_invalid_attempts),
        "invalid_selected_window_rate": _safe_rate(invalid_selected_turns, total_verify_turns),
        "unresolved_selection_rate": _safe_rate(unresolved_selection_turns, total_verify_turns),
        "verify_invalid_turn_rate": _safe_rate(verify_invalid_turns, total_verify_turns),
    }


def _teacher_reward_by_label_group(
    records: Sequence[Dict[str, Any]],
    *,
    reference_data: Optional[Any] = None,
) -> Dict[str, Any]:
    buckets: Dict[str, List[float]] = {"anomaly": [], "normal": [], "unknown": []}
    teacher_case_counts: Counter = Counter()
    record_counts: Counter = Counter()
    for record in records:
        label_group = _resolve_reference_label_group(record, reference_data=reference_data)
        record_counts[label_group] += 1
        reward_value = None
        for turn in reversed(record.get("turns") or []):
            if str(turn.get("tool_name") or "") != "verify_hypothesis":
                continue
            if turn.get("teacher_judge_decision") or turn.get("teacher_judge_alignment") is not None:
                reward_value = _safe_float(turn.get("teacher_judge_reward"), 0.0)
                break
        if reward_value is None:
            reward_value = _safe_float((record.get("reward_summary") or {}).get("components", {}).get("teacher_judge_reward"), 0.0)
            if reward_value == 0.0 and not any(
                (turn.get("teacher_judge_decision") or turn.get("teacher_judge_alignment") is not None)
                for turn in (record.get("turns") or [])
            ):
                buckets.setdefault(label_group, []).append(0.0)
                continue
        teacher_case_counts[label_group] += 1
        buckets.setdefault(label_group, []).append(float(reward_value))

    summary: Dict[str, Any] = {}
    for key in ("anomaly", "normal", "unknown"):
        values = buckets.get(key, [])
        summary[key] = {
            "num_records": int(record_counts.get(key, 0)),
            "num_teacher_cases": int(teacher_case_counts.get(key, 0)),
            "mean_teacher_judge_reward": _mean(values),
            "min_teacher_judge_reward": min(values) if values else 0.0,
            "max_teacher_judge_reward": max(values) if values else 0.0,
        }
    return summary


def _initialize_counter(keys: Sequence[str]) -> Dict[str, int]:
    return {key: 0 for key in keys}


def summarize_scored_rollouts(
    records: Sequence[Dict[str, Any]],
    *,
    reference_data: Optional[Any] = None,
    max_teacher_disagreement_cases: int = 50,
) -> Dict[str, Any]:
    num_records = len(records)
    primary_counter = Counter()
    teacher_decision_counter = Counter()
    teacher_alignment_counter = Counter()
    total_rewards: List[float] = []
    num_turns: List[float] = []
    teacher_alignment_values: List[float] = []
    teacher_reward_values: List[float] = []
    component_values: Dict[str, List[float]] = {}

    for record in records:
        primary_status, _ = extract_verifier_statuses(record)
        if primary_status not in PRIMARY_STATUS_KEYS:
            primary_status = "unknown"
        primary_counter[primary_status] += 1
        teacher_decision, teacher_alignment_key, teacher_alignment, teacher_reward = extract_teacher_judge_status(record)
        if teacher_decision not in TEACHER_DECISION_KEYS:
            teacher_decision = "unknown"
        if teacher_alignment_key not in TEACHER_ALIGNMENT_KEYS:
            teacher_alignment_key = "unknown"
        teacher_decision_counter[teacher_decision] += 1
        teacher_alignment_counter[teacher_alignment_key] += 1
        if teacher_alignment is not None:
            teacher_alignment_values.append(_safe_float(teacher_alignment, 0.0))
        if teacher_reward != 0.0 or teacher_decision != "unknown":
            teacher_reward_values.append(_safe_float(teacher_reward, 0.0))

        total_rewards.append(_safe_float((record.get("reward_summary") or {}).get("total_reward"), 0.0))
        num_turns.append(_safe_float(record.get("num_turns"), 0.0))
        reward_summary = record.get("reward_summary") or {}
        reward_components = reward_summary.get("components") or {}
        for key, value in reward_components.items():
            component_values.setdefault(str(key), []).append(_safe_float(value, 0.0))

    primary_counts = _initialize_counter(PRIMARY_STATUS_KEYS)
    primary_counts.update(primary_counter)
    teacher_decision_counts = _initialize_counter(TEACHER_DECISION_KEYS)
    teacher_decision_counts.update(teacher_decision_counter)
    teacher_alignment_counts = _initialize_counter(TEACHER_ALIGNMENT_KEYS)
    teacher_alignment_counts.update(teacher_alignment_counter)

    primary_ratios = {
        key: (_safe_float(primary_counts[key]) / num_records if num_records else 0.0)
        for key in PRIMARY_STATUS_KEYS
    }
    teacher_decision_ratios = {
        key: (_safe_float(teacher_decision_counts[key]) / num_records if num_records else 0.0)
        for key in TEACHER_DECISION_KEYS
    }
    teacher_alignment_ratios = {
        key: (_safe_float(teacher_alignment_counts[key]) / num_records if num_records else 0.0)
        for key in TEACHER_ALIGNMENT_KEYS
    }
    mean_component_values = {key: _mean(values) for key, values in sorted(component_values.items())}
    teacher_disagreement_all = _teacher_disagreement_cases(
        records,
        reference_data=reference_data,
    )
    if max_teacher_disagreement_cases > 0:
        teacher_disagreement_preview = teacher_disagreement_all[: int(max_teacher_disagreement_cases)]
    else:
        teacher_disagreement_preview = teacher_disagreement_all
    verify_health = _verify_health_summary(records)

    return {
        "num_records": num_records,
        "primary_status_counts": primary_counts,
        "primary_status_ratios": primary_ratios,
        "teacher_judge_decision_counts": teacher_decision_counts,
        "teacher_judge_decision_ratios": teacher_decision_ratios,
        "teacher_judge_alignment_counts": teacher_alignment_counts,
        "teacher_judge_alignment_ratios": teacher_alignment_ratios,
        "mean_teacher_judge_alignment": _mean(teacher_alignment_values),
        "mean_teacher_judge_reward": _mean(teacher_reward_values),
        "verify_turn_coverage": _verify_turn_coverage(records),
        "teacher_disagreement_case_total": len(teacher_disagreement_all),
        "teacher_disagreement_cases": teacher_disagreement_preview,
        "teacher_disagreement_by_category": _teacher_disagreement_cluster_rows(
            teacher_disagreement_all,
            group_keys=("category",),
        ),
        "teacher_disagreement_by_stage": _teacher_disagreement_cluster_rows(
            teacher_disagreement_all,
            group_keys=("stage",),
        ),
        "teacher_disagreement_by_category_stage": _teacher_disagreement_cluster_rows(
            teacher_disagreement_all,
            group_keys=("category", "stage"),
        ),
        "teacher_reward_by_label_group": _teacher_reward_by_label_group(
            records,
            reference_data=reference_data,
        ),
        **verify_health,
        "mean_total_reward": _mean(total_rewards),
        "min_total_reward": min(total_rewards) if total_rewards else 0.0,
        "max_total_reward": max(total_rewards) if total_rewards else 0.0,
        "mean_num_turns": _mean(num_turns),
        "min_num_turns": min(num_turns) if num_turns else 0.0,
        "max_num_turns": max(num_turns) if num_turns else 0.0,
        "mean_reward_components": mean_component_values,
    }
