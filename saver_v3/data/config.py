from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from saver_v3.cli.common import load_yaml_mapping, resolve_path


# Inline finalize_case now carries compact semantic fields in the same terminal
# tool call, so the old 256-token ceiling truncates valid terminal JSON too
# often during rollout/eval.
DEFAULT_POLICY_MAX_NEW_TOKENS = 768
DEFAULT_TOTAL_VISUAL_BUDGET = 28
DEFAULT_RECOMMENDED_MAX_SEQ_LENGTH = 6144
DEFAULT_RECOMMENDED_KEEP_RECENT_TEXT_MESSAGES = 20
DEFAULT_RECOMMENDED_MAX_TOTAL_IMAGES = 28

DEFAULT_INITIAL_USER_TEMPLATE = (
    "Case ID: {public_case_id}\n"
    "Scene: {scene}\n"
    "Duration (sec): {duration}\n"
    "Task: {task_prompt}\n"
    "Success Criteria:\n"
    "{criteria_text}"
)

DEFAULT_PREVIEW_INSTRUCTION = (
    "You are given temporally ordered preview frames from the video. "
    "Use these preview frames to decide the next best action. "
    "If they are insufficient, call a tool to inspect more evidence."
)

DEFAULT_TOOL_RESPONSE_TEMPLATE = (
    "Selected frames: {timestamps}.\n"
    "Choose exactly one next step.\n"
    "Usually call seek_evidence for another stage or verify_hypothesis to judge sufficiency.\n"
    "Call finalize_case only when the gathered evidence is already sufficient.\n"
    "When finalizing, prioritize a compact finalize_case that closes cleanly.\n"
    "If optional summary, rationale, event_chain_summary, or qa_focus_answers would make the tool call too long, omit them.\n"
    "Do not answer before finalize_case returns.\n"
)


@dataclass
class PreviewConfig:
    num_preview_frames: int = 8
    preview_sampling_fps: Optional[float] = None
    max_preview_frames: int = 8


@dataclass
class PromptConfig:
    initial_user_template: str = DEFAULT_INITIAL_USER_TEMPLATE
    preview_instruction: str = DEFAULT_PREVIEW_INSTRUCTION
    tool_response_template: str = DEFAULT_TOOL_RESPONSE_TEMPLATE


@dataclass
class RolloutTraceConfig:
    record_observation_content: bool = False
    record_state_deltas: bool = True
    record_counterfactual_trace: bool = True
    record_message_history: bool = True


@dataclass
class SaverAgentConfig:
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    rollout_trace: RolloutTraceConfig = field(default_factory=RolloutTraceConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return value or {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _anchor_dir(anchor: str | Path | None) -> Path | None:
    if anchor is None:
        return None
    anchor_path = Path(anchor).expanduser().resolve()
    return anchor_path.parent if anchor_path.is_file() else anchor_path


def resolve_saver_config_source(
    saver_config_source: str | Path | None,
    *,
    source_anchor: str | Path | None = None,
) -> Path | None:
    if not saver_config_source:
        return None
    return resolve_path(saver_config_source, anchor=_anchor_dir(source_anchor))


def saver_config_from_mapping(
    mapping: Mapping[str, Any] | None,
    *,
    saver_config_source: str | Path | None = None,
    source_anchor: str | Path | None = None,
) -> SaverAgentConfig:
    merged_mapping: Dict[str, Any] = {}
    resolved_source = resolve_saver_config_source(saver_config_source, source_anchor=source_anchor)
    if resolved_source is not None:
        source_mapping = load_yaml_mapping(resolved_source)
        for section in ("preview", "prompt", "rollout_trace"):
            if section in source_mapping:
                merged_mapping[section] = dict(_mapping(source_mapping.get(section)))

    local_mapping = dict(mapping or {})
    for section in ("preview", "prompt", "rollout_trace"):
        if section in local_mapping:
            section_mapping = dict(_mapping(merged_mapping.get(section)))
            section_mapping.update(dict(_mapping(local_mapping.get(section))))
            merged_mapping[section] = section_mapping

    preview_cfg = dict(_mapping(merged_mapping.get("preview")))
    prompt_cfg = dict(_mapping(merged_mapping.get("prompt")))
    rollout_trace_cfg = dict(_mapping(merged_mapping.get("rollout_trace")))

    default_preview = PreviewConfig()
    default_prompt = PromptConfig()
    default_rollout_trace = RolloutTraceConfig()
    num_preview_frames = int(preview_cfg.get("num_preview_frames", default_preview.num_preview_frames) or default_preview.num_preview_frames)
    max_preview_frames = int(
        preview_cfg.get("max_preview_frames", preview_cfg.get("num_preview_frames", default_preview.max_preview_frames))
        or default_preview.max_preview_frames
    )
    return SaverAgentConfig(
        preview=PreviewConfig(
            num_preview_frames=num_preview_frames,
            preview_sampling_fps=_optional_float(preview_cfg.get("preview_sampling_fps", default_preview.preview_sampling_fps)),
            max_preview_frames=max_preview_frames,
        ),
        prompt=PromptConfig(
            initial_user_template=str(prompt_cfg.get("initial_user_template", default_prompt.initial_user_template)),
            preview_instruction=str(prompt_cfg.get("preview_instruction", default_prompt.preview_instruction)),
            tool_response_template=str(prompt_cfg.get("tool_response_template", default_prompt.tool_response_template)),
        ),
        rollout_trace=RolloutTraceConfig(
            record_observation_content=bool(
                rollout_trace_cfg.get("record_observation_content", default_rollout_trace.record_observation_content)
            ),
            record_state_deltas=bool(rollout_trace_cfg.get("record_state_deltas", default_rollout_trace.record_state_deltas)),
            record_counterfactual_trace=bool(
                rollout_trace_cfg.get("record_counterfactual_trace", default_rollout_trace.record_counterfactual_trace)
            ),
            record_message_history=bool(rollout_trace_cfg.get("record_message_history", default_rollout_trace.record_message_history)),
        ),
    )
