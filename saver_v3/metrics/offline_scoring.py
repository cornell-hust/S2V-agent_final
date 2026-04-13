from __future__ import annotations

import copy
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from saver_v3.core.counterfactual_verification import (
    run_counterfactual_verification,
)
from saver_v3.data.dataset import SaverAgentDataset
from saver_v3.common.runtime import DistributedRuntime, create_progress_bar, runtime_log, should_log_progress
from saver_v3.core.reward import score_rollout_trace
from saver_v3.core.schema import SaverEnvironmentState
from saver_v3.teacher.teacher_judge import compute_teacher_judge_signal
from saver_v3.core.verifier import run_counterfactual_verifier


def load_rollout_records(input_path: str | Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Rollout input path does not exist: {path}. "
            "Replace placeholder paths like /path/to/... with a real rollout .json, .jsonl, or directory."
        )
    if path.is_dir():
        records = []
        files = sorted(list(path.glob("*.json")) + list(path.glob("*.jsonl")))
        for file_path in files:
            if file_path.suffix == ".jsonl":
                records.extend(
                    json.loads(line)
                    for line in file_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            else:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    records.extend(payload)
                else:
                    records.append(payload)
        return records, {"input_kind": "directory", "source_files": [str(file_path) for file_path in files]}

    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return records, {"input_kind": "jsonl", "source_file": str(path)}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {"input_kind": "json", "source_file": str(path), "json_mode": "list"}
    return [payload], {"input_kind": "json", "source_file": str(path), "json_mode": "single"}


def save_rollout_records(
    records: Sequence[Dict[str, Any]],
    output_path: str | Path,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}

    if output_path.suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    if metadata.get("input_kind") == "json" and metadata.get("json_mode") == "single" and len(records) == 1:
        output_path.write_text(json.dumps(records[0], ensure_ascii=False, indent=2), encoding="utf-8")
        return

    output_path.write_text(json.dumps(list(records), ensure_ascii=False, indent=2), encoding="utf-8")


class ReferenceDataProvider:
    def __init__(self, *, data_path: str | Path | None = None, data_root: str | Path = ""):
        self.data_path = Path(data_path) if data_path else None
        self.data_root = Path(data_root) if data_root else Path()
        self.records: List[Dict[str, Any]] = []
        self.by_video_id: Dict[str, Dict[str, Any]] = {}
        self.index_by_video_id: Dict[str, int] = {}
        self._dataset: Optional[SaverAgentDataset] = None
        if self.data_path is not None:
            with self.data_path.open("r", encoding="utf-8") as f:
                self.records = [json.loads(line) for line in f if line.strip()]
            self.by_video_id = {record.get("video_id"): record for record in self.records if record.get("video_id")}
            self.index_by_video_id = {
                record.get("video_id"): idx for idx, record in enumerate(self.records) if record.get("video_id")
            }

    def _ensure_dataset(self) -> SaverAgentDataset:
        if self.data_path is None:
            raise ValueError("A data_path is required to build multimodal caches for offline verification.")
        if self._dataset is None:
            self._dataset = SaverAgentDataset(self.data_path, data_root=self.data_root)
        return self._dataset

    def get_multimodal_cache(
        self,
        video_id: str,
        *,
        verifier_backend: str,
        verifier_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del verifier_backend
        verifier_kwargs = verifier_kwargs or {}
        dataset = self._ensure_dataset()
        if video_id not in self.index_by_video_id:
            raise KeyError(f"Video id {video_id!r} not found in reference data.")
        cache = copy.deepcopy(dataset[self.index_by_video_id[video_id]]["multimodal_cache"])
        cache.update({key: value for key, value in verifier_kwargs.items() if value is not None})
        return cache

    def get_dataset_item(self, video_id: str) -> Dict[str, Any]:
        dataset = self._ensure_dataset()
        if video_id not in self.index_by_video_id:
            raise KeyError(f"Video id {video_id!r} not found in reference data.")
        return dataset[int(self.index_by_video_id[video_id])]


def rollout_state_from_dict(payload: Dict[str, Any]) -> SaverEnvironmentState:
    valid_fields = {field.name for field in fields(SaverEnvironmentState)}
    kwargs = {key: copy.deepcopy(value) for key, value in payload.items() if key in valid_fields}
    return SaverEnvironmentState(**kwargs)


def infer_claim_from_rollout(rollout: Dict[str, Any]) -> Dict[str, Any]:
    state = rollout.get("state") or {}
    finalized_case = state.get("finalized_case")
    if isinstance(finalized_case, dict):
        return copy.deepcopy(finalized_case)

    final_answer = rollout.get("final_answer")
    if isinstance(final_answer, dict):
        return copy.deepcopy(final_answer)

    return {}


def _build_scoring_target_from_reference_record(reference_record: Dict[str, Any]) -> Dict[str, Any]:
    structured_target = reference_record.get("structured_target")
    if isinstance(structured_target, dict) and structured_target:
        return copy.deepcopy(structured_target)
    label = dict(reference_record.get("label") or {})
    temporal = dict(reference_record.get("temporal") or {})
    counterfactual = dict(reference_record.get("counterfactual") or {})
    return {
        "existence": "anomaly" if bool(label.get("is_anomaly")) else "normal",
        "category": label.get("category"),
        "severity": label.get("severity"),
        "anomaly_interval_sec": temporal.get("anomaly_interval_sec"),
        "precursor_interval_sec": temporal.get("precursor_interval_sec"),
        "counterfactual_type": counterfactual.get("type", "none"),
    }


def infer_candidate_window_ids(rollout: Dict[str, Any]) -> List[str]:
    turns = rollout.get("turns") or []
    for turn in reversed(turns):
        if turn.get("verifier_verified_window_ids"):
            return list(turn["verifier_verified_window_ids"])
    state = rollout.get("state") or {}
    if state.get("active_evidence_window_ids"):
        return list(state["active_evidence_window_ids"])
    window_ids = []
    for entry in state.get("evidence_ledger") or []:
        if entry.get("window_id"):
            window_ids.append(entry["window_id"])
    return window_ids


def attach_offline_verifier(
    rollout: Dict[str, Any],
    *,
    reference_data: ReferenceDataProvider,
    verifier_backend: str,
    force_reverify: bool = False,
    verifier_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    verifier_kwargs = verifier_kwargs or {}
    augmented = copy.deepcopy(rollout)
    if not force_reverify and isinstance(augmented.get("offline_verifier"), dict):
        return augmented

    video_id = augmented.get("video_id")
    if not video_id:
        raise ValueError("Rollout record is missing video_id and cannot be reverified offline.")

    multimodal_cache = reference_data.get_multimodal_cache(
        video_id,
        verifier_backend=verifier_backend,
        verifier_kwargs=verifier_kwargs,
    )
    state = rollout_state_from_dict(augmented.get("state") or {})
    claim = infer_claim_from_rollout(augmented)
    candidate_window_ids = infer_candidate_window_ids(augmented)

    verdict = run_counterfactual_verifier(
        state=state,
        multimodal_cache=multimodal_cache,
        verification_mode="final_check",
        claim=claim,
        candidate_window_ids=candidate_window_ids,
        backend=verifier_backend,
        use_reference_supervision=True,
    )
    verdict["reference_conditioned"] = True
    verdict["intended_use"] = "training_or_diagnostic_only"
    augmented["offline_verifier"] = verdict
    return augmented


def attach_teacher_judge_to_records(
    records: Sequence[Dict[str, Any]],
    *,
    reference_data: ReferenceDataProvider,
    judge: Any,
    input_mode: str = "multimodal_visual",
    progress_every: int = 0,
    progress_label: str = "teacher judge",
    runtime: Optional[DistributedRuntime] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from saver_v3.data.training_data import build_reward_weighted_examples

    annotated_records: List[Dict[str, Any]] = []
    total_records = len(records)
    candidate_turns = 0
    annotated_turns = 0

    for completed, record in enumerate(records, start=1):
        augmented = copy.deepcopy(record)
        video_id = str(augmented.get("video_id") or "").strip()
        if not video_id:
            annotated_records.append(augmented)
            continue
        try:
            item = reference_data.get_dataset_item(video_id)
        except Exception:
            annotated_records.append(augmented)
            continue

        per_turn_examples = build_reward_weighted_examples(
            item,
            augmented,
            include_invalid=True,
        )
        verify_examples_by_step = {
            int(example.get("step_index") or 0): example
            for example in per_turn_examples
            if str(example.get("tool_name") or "") == "verify_hypothesis"
        }
        for turn in augmented.get("turns") or []:
            step_index = int(turn.get("step_index") or 0)
            verify_example = verify_examples_by_step.get(step_index)
            if verify_example is None:
                continue
            candidate_turns += 1
            annotated_example = judge.annotate_example(verify_example, input_mode=input_mode)
            if "teacher_judge_scores" in annotated_example:
                turn["teacher_judge_scores"] = copy.deepcopy(annotated_example.get("teacher_judge_scores") or {})
            if "teacher_judge_decision" in annotated_example:
                turn["teacher_judge_decision"] = annotated_example.get("teacher_judge_decision")
            if "teacher_judge_rationale" in annotated_example:
                turn["teacher_judge_rationale"] = annotated_example.get("teacher_judge_rationale")
            turn.update(compute_teacher_judge_signal(turn))
            annotated_turns += 1

        annotated_records.append(augmented)
        if should_log_progress(completed, total_records, int(progress_every)):
            runtime_log(
                (
                    f"{progress_label}: records={completed}/{total_records} "
                    f"teacher_candidates={candidate_turns} teacher_annotated={annotated_turns}"
                ),
                runtime=runtime,
            )

    return annotated_records, {
        "num_records": total_records,
        "num_teacher_judge_candidates": candidate_turns,
        "num_teacher_judge_annotated_turns": annotated_turns,
    }


def _normalize_counterfactual_branch_profile(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "full", "offline_full", "online_core"}:
        return "full" if normalized else ""
    return normalized


def _infer_existing_counterfactual_branch_profile(record: Dict[str, Any]) -> str:
    explicit = _normalize_counterfactual_branch_profile(record.get("counterfactual_branch_profile"))
    if explicit:
        return explicit
    if isinstance(record.get("counterfactual_branches"), dict) and record.get("counterfactual_branches"):
        return "full"
    return ""


def _counterfactual_profile_covers(existing_profile: str, requested_profile: str) -> bool:
    existing = _normalize_counterfactual_branch_profile(existing_profile)
    requested = _normalize_counterfactual_branch_profile(requested_profile) or "full"
    if not existing:
        return False
    return existing == requested


def _build_counterfactual_defaults(branch_profile: str) -> Dict[str, Any]:
    normalized_profile = _normalize_counterfactual_branch_profile(branch_profile) or "full"
    return {
        "counterfactual_branches": {},
        "counterfactual_profile": {
            "summary": {
                "decision_sufficiency": False,
                "minimal_subset_sufficiency": False,
                "stage_necessity": {},
                "negative_specificity_pass": False,
                "counterfactual_type_supported": False,
            },
            "branch_field_matrix": {},
            "branch_delta_matrix": {},
            "stage_packages": {},
            "selection_metadata": {
                "normalized_branch_profile": normalized_profile,
            },
        },
        "counterfactual_profile_source": normalized_profile,
        "counterfactual_branch_profile": normalized_profile,
    }


def attach_counterfactual_verification_to_records(
    records: Sequence[Dict[str, Any]],
    *,
    reference_data: Optional[ReferenceDataProvider],
    policy: Any,
    counterfactual_branch_profile: str = "full",
    counterfactual_max_images: int = 12,
) -> List[Dict[str, Any]]:
    requested_counterfactual_profile = (
        _normalize_counterfactual_branch_profile(counterfactual_branch_profile) or "full"
    )
    attached_records: List[Dict[str, Any]] = []
    for record in records:
        augmented = copy.deepcopy(record)
        counterfactual_defaults = _build_counterfactual_defaults(requested_counterfactual_profile)
        existing_counterfactual_profile = _infer_existing_counterfactual_branch_profile(augmented)
        has_existing_counterfactual = bool(existing_counterfactual_profile)
        if _counterfactual_profile_covers(existing_counterfactual_profile, requested_counterfactual_profile):
            augmented["counterfactual_profile_source"] = existing_counterfactual_profile
            augmented["counterfactual_branch_profile"] = existing_counterfactual_profile
            attached_records.append(augmented)
            continue

        if reference_data is None or policy is None or not hasattr(policy, "generate_from_messages"):
            if has_existing_counterfactual:
                augmented["counterfactual_profile_source"] = existing_counterfactual_profile
                augmented["counterfactual_branch_profile"] = existing_counterfactual_profile
            else:
                augmented.update(counterfactual_defaults)
            attached_records.append(augmented)
            continue

        video_id = str(augmented.get("video_id") or "").strip()
        if not video_id:
            augmented.update(counterfactual_defaults)
            attached_records.append(augmented)
            continue
        try:
            item = reference_data.get_dataset_item(video_id)
            reference_record = reference_data.by_video_id.get(video_id)
            if isinstance(reference_record, dict):
                augmented.update(
                    run_counterfactual_verification(
                        policy,
                        item=item,
                        rollout=augmented,
                        reference_record=reference_record,
                        max_images=int(counterfactual_max_images),
                        branch_profile=requested_counterfactual_profile,
                    )
                )
            else:
                augmented.update(counterfactual_defaults)
        except Exception:
            augmented.update(counterfactual_defaults)
        attached_records.append(augmented)
    return attached_records


def score_rollout_records(
    records: Sequence[Dict[str, Any]],
    *,
    reference_data: Optional[ReferenceDataProvider] = None,
    policy: Any = None,
    verifier_backend: str = "qwen_self_verifier",
    force_reverify: bool = False,
    attach_reference_offline_verifier: bool = False,
    verifier_kwargs: Optional[Dict[str, Any]] = None,
    counterfactual_max_images: int = 12,
    counterfactual_branch_profile: str = "full",
    progress_every: int = 0,
    progress_label: str = "score",
    show_progress_bar: bool = True,
    runtime: Optional[DistributedRuntime] = None,
) -> List[Dict[str, Any]]:
    verifier_kwargs = verifier_kwargs or {}
    requested_counterfactual_profile = (
        _normalize_counterfactual_branch_profile(counterfactual_branch_profile) or "full"
    )
    scored_records: List[Dict[str, Any]] = []
    total_records = len(records)
    progress_bar = (
        create_progress_bar(
            total=total_records,
            desc=str(progress_label or "score"),
            runtime=runtime,
            unit="record",
            leave=True,
        )
        if bool(show_progress_bar)
        else None
    )
    try:
        for completed, record in enumerate(records, start=1):
            augmented = copy.deepcopy(record)
            if reference_data is not None and not any(
                isinstance(augmented.get(key), dict) and augmented.get(key)
                for key in ("scoring_target", "structured_target", "target")
            ):
                video_id = str(augmented.get("video_id") or "").strip()
                reference_record = reference_data.by_video_id.get(video_id) if video_id else None
                if isinstance(reference_record, dict):
                    augmented["scoring_target"] = _build_scoring_target_from_reference_record(reference_record)
            needs_offline_verifier = bool(attach_reference_offline_verifier) and (
                force_reverify
                or not any(turn.get("tool_name") == "verify_hypothesis" for turn in augmented.get("turns") or [])
            )
            if needs_offline_verifier:
                if reference_data is None:
                    raise ValueError(
                        "Reference data is required for offline verifier attachment when force_reverify is enabled "
                        "or verifier turns are absent."
                    )
                augmented = attach_offline_verifier(
                    augmented,
                    reference_data=reference_data,
                    verifier_backend=verifier_backend,
                    force_reverify=bool(force_reverify),
                    verifier_kwargs=verifier_kwargs,
                )
            if reference_data is not None:
                augmented = attach_counterfactual_verification_to_records(
                    [augmented],
                    reference_data=reference_data,
                    policy=policy,
                    counterfactual_branch_profile=requested_counterfactual_profile,
                    counterfactual_max_images=int(counterfactual_max_images),
                )[0]
            augmented["reward_summary"] = score_rollout_trace(augmented, reward_version="timesearch_v3")
            augmented["scoring_metadata"] = {
                "verifier_backend": verifier_backend,
                "force_reverify": bool(force_reverify),
                "attach_offline_verifier": bool(attach_reference_offline_verifier),
                "counterfactual_branch_profile_requested": requested_counterfactual_profile,
                "counterfactual_branch_profile_scored": str(augmented.get("counterfactual_branch_profile") or ""),
                "counterfactual_full_selected_available": bool(
                    (((augmented.get("counterfactual_profile") or {}).get("branch_field_matrix") or {}).get("full_selected") or {}).get("available")
                ),
                "counterfactual_minimal_subset_available": bool(
                    (((augmented.get("counterfactual_profile") or {}).get("branch_field_matrix") or {}).get("minimal_subset") or {}).get("available")
                ),
                "counterfactual_negative_branch_available": bool(
                    (((augmented.get("counterfactual_profile") or {}).get("branch_field_matrix") or {}).get("hard_negative_swap") or {}).get("available")
                ),
                "teacher_judge_present": bool(
                    any(turn.get("teacher_judge_decision") or turn.get("teacher_judge_alignment") is not None for turn in augmented.get("turns") or [])
                ),
            }
            scored_records.append(augmented)
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix_str(
                    f"video_id={augmented.get('video_id', '')}",
                    refresh=False,
                )
            if int(progress_every) > 0 and should_log_progress(completed, total_records, int(progress_every)):
                runtime_log(
                    f"{progress_label}: {completed}/{total_records} video_id={augmented.get('video_id', '')}",
                    runtime=runtime,
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return scored_records
