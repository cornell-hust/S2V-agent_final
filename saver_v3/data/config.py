from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


# Inline finalize_case now carries compact semantic fields in the same terminal
# tool call, so the old 256-token ceiling truncates valid terminal JSON too
# often during rollout/eval.
DEFAULT_POLICY_MAX_NEW_TOKENS = 768
DEFAULT_TOTAL_VISUAL_BUDGET = 24
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
