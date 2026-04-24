from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional

from saver_v3.data.config import SaverAgentConfig
from saver_v3.core.protocol_guidance import build_finalize_scaffold
from saver_v3.core.prompts import build_tool_response_prompt
TIMESTAMP_PATTERN = re.compile(r"^\d+(?:\.\d+)?s$")
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


class TimeSearchRolloutAdapter:
    """Adapt SAVER observations into a TimeSearch-R style rollout transcript."""

    def __init__(self, *, config: SaverAgentConfig | None = None):
        self.config = copy.deepcopy(config) if config is not None else SaverAgentConfig()

    def build_initial_messages(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        return copy.deepcopy(item.get("messages", []))

    def build_assistant_message(self, response_text: str) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}],
        }

    def adapt_tool_observation(
        self,
        tool_message: Dict[str, Any],
        multimodal_cache: Dict[str, Any],
    ) -> Dict[str, Any]:
        adapted = {key: value for key, value in dict(tool_message).items() if key != "content"}
        content = [dict(item) if isinstance(item, dict) else item for item in list(tool_message.get("content", []))]
        if adapted.get("name") == "parse_error":
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Retry now with exactly one valid <tool_call>{...}</tool_call> or "
                        "<answer>{...}</answer>. Do not explain the action in plain English."
                    ),
                }
            )
            adapted["content"] = content
            return adapted
        if adapted.get("name") == "verify_hypothesis":
            verification = self._extract_json_payload(content)
            next_tool = str((verification or {}).get("next_tool") or "")
            if next_tool == "finalize_case":
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Call finalize_case next. "
                            + build_finalize_scaffold(
                                verification_payload=verification,
                                finalize_schema=(multimodal_cache.get("tool_io") or {}).get("finalize_case_schema") or {},
                            )
                        ),
                    }
                )
            elif next_tool == "seek_evidence":
                content.append(
                    {
                        "type": "text",
                        "text": "Call seek_evidence next. Cover the next missing stage before finalizing.",
                    }
                )
            adapted["content"] = content
            return adapted
        if adapted.get("name") == "finalize_case":
            content.append(
                {
                    "type": "text",
                    "text": "Finalize recorded. The main rollout is complete. Do not call more tools in this rollout.",
                }
            )
            adapted["content"] = content
            return adapted
        prompt_text = build_tool_response_prompt(
            self._extract_timestamps(content),
            question=str(multimodal_cache.get("question") or ""),
            duration=multimodal_cache.get("duration"),
            prompt_config=self.config.prompt,
        )
        content.append({"type": "text", "text": prompt_text})
        adapted["content"] = content
        return adapted

    @staticmethod
    def parse_answer_text(response_text: str) -> Optional[str]:
        stripped = THINK_BLOCK_PATTERN.sub("", response_text)
        matches = re.findall(r"<answer>(.*?)</answer>", stripped, re.DOTALL)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _extract_timestamps(content: List[Dict[str, Any]]) -> List[str]:
        timestamps: List[str] = []
        for item in content:
            if item.get("type") != "text":
                continue
            text = str(item.get("text", "")).strip()
            if TIMESTAMP_PATTERN.match(text):
                timestamps.append(text)
        return timestamps

    @staticmethod
    def _extract_json_payload(content: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for item in content:
            if item.get("type") != "text":
                continue
            text = str(item.get("text", "")).strip()
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _finalize_required_fields(multimodal_cache: Dict[str, Any]) -> List[str]:
        schema = (multimodal_cache.get("tool_io") or {}).get("finalize_case_schema") or {}
        required = list(schema.get("required") or [])
        return [str(field_name) for field_name in required if str(field_name).strip()]
