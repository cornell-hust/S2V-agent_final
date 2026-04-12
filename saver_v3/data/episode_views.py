from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EpisodeMessagesExample:
    messages: List[Dict[str, Any]]
    video_id: str = ""
    split: str = ""
    target_action: str = "tool_call"
    tool_name: str = ""
    target_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
