from __future__ import annotations

from typing import Any, Mapping, Tuple


INITIAL_GLOBAL_SCAN_MESSAGE_KEY = "initial_global_scan"
PROTECT_INITIAL_GLOBAL_SCAN_KEY = "protect_initial_global_scan"
ERROR_ON_INITIAL_SCAN_SEQ_PRUNE_KEY = "error_on_initial_scan_seq_prune"
_WINDOW_TOLERANCE_SEC = 1e-6
_DEFAULT_MODE = "explicit_first_scan"
_DEFAULT_SCAN_NUM_FRAMES = 8
_DEFAULT_SCAN_PURPOSE = "global_overview"
_DEFAULT_PROTECT_FROM_VISUAL_BUDGET = True
_DEFAULT_ERROR_ON_SEQ_PRUNE = True


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_initial_observation_config(config: Any = None) -> dict[str, Any]:
    if hasattr(config, "initial_observation"):
        return normalize_initial_observation_config(getattr(config, "initial_observation"))
    if hasattr(config, "mode") and hasattr(config, "scan_num_frames"):
        return {
            "mode": str(getattr(config, "mode", _DEFAULT_MODE) or _DEFAULT_MODE),
            "scan_num_frames": max(1, int(getattr(config, "scan_num_frames", _DEFAULT_SCAN_NUM_FRAMES) or _DEFAULT_SCAN_NUM_FRAMES)),
            "scan_purpose": str(getattr(config, "scan_purpose", _DEFAULT_SCAN_PURPOSE) or _DEFAULT_SCAN_PURPOSE),
            "protect_from_visual_budget": bool(
                getattr(config, "protect_from_visual_budget", _DEFAULT_PROTECT_FROM_VISUAL_BUDGET)
            ),
            "error_on_seq_prune": bool(getattr(config, "error_on_seq_prune", _DEFAULT_ERROR_ON_SEQ_PRUNE)),
        }
    if isinstance(config, Mapping):
        payload = dict(config.get("initial_observation") or config)
        return {
            "mode": str(payload.get("mode", _DEFAULT_MODE) or _DEFAULT_MODE),
            "scan_num_frames": max(1, int(payload.get("scan_num_frames", _DEFAULT_SCAN_NUM_FRAMES) or _DEFAULT_SCAN_NUM_FRAMES)),
            "scan_purpose": str(payload.get("scan_purpose", _DEFAULT_SCAN_PURPOSE) or _DEFAULT_SCAN_PURPOSE),
            "protect_from_visual_budget": bool(
                payload.get("protect_from_visual_budget", _DEFAULT_PROTECT_FROM_VISUAL_BUDGET)
            ),
            "error_on_seq_prune": bool(payload.get("error_on_seq_prune", _DEFAULT_ERROR_ON_SEQ_PRUNE)),
        }
    return {
        "mode": _DEFAULT_MODE,
        "scan_num_frames": _DEFAULT_SCAN_NUM_FRAMES,
        "scan_purpose": _DEFAULT_SCAN_PURPOSE,
        "protect_from_visual_budget": _DEFAULT_PROTECT_FROM_VISUAL_BUDGET,
        "error_on_seq_prune": _DEFAULT_ERROR_ON_SEQ_PRUNE,
    }


def is_preview_initial_observation(config: Any = None) -> bool:
    return str(normalize_initial_observation_config(config).get("mode") or "").strip().lower() == "preview"


def is_explicit_first_scan_initial_observation(config: Any = None) -> bool:
    return str(normalize_initial_observation_config(config).get("mode") or "").strip().lower() == "explicit_first_scan"


def expected_initial_scan_window_sec(
    multimodal_cache: Mapping[str, Any],
    *,
    config: Any = None,
) -> Tuple[float, float]:
    del config
    tool_io = dict(multimodal_cache.get("tool_io") or {})
    duration_sec = max(0.0, _coerce_float(multimodal_cache.get("duration"), 0.0))
    window = tool_io.get("initial_scan_window_sec")
    if isinstance(window, (list, tuple)) and len(window) == 2:
        start_sec = max(0.0, _coerce_float(window[0], 0.0))
        end_sec = max(0.0, _coerce_float(window[1], duration_sec))
        if duration_sec > 0.0:
            start_sec = min(start_sec, duration_sec)
            end_sec = min(end_sec, duration_sec)
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        return round(start_sec, 6), round(end_sec, 6)
    return 0.0, round(duration_sec, 6)


def _entry_selected_frame_count(entry: Mapping[str, Any]) -> int:
    selected_frame_count = entry.get("selected_frame_count")
    if selected_frame_count is not None:
        try:
            return max(0, int(selected_frame_count))
        except Exception:
            pass
    selected_timestamps = list(entry.get("selected_timestamps") or [])
    return max(0, int(len(selected_timestamps)))


def is_canonical_initial_scan_entry(
    entry: Mapping[str, Any] | None,
    *,
    arguments: Mapping[str, Any] | None = None,
    multimodal_cache: Mapping[str, Any],
    config: Any = None,
    prior_scan_count: int = 0,
) -> bool:
    if entry is None or not is_explicit_first_scan_initial_observation(config):
        return False
    if int(prior_scan_count) != 0:
        return False

    initial_observation = normalize_initial_observation_config(config)
    purpose = str(
        (arguments or {}).get("purpose")
        or entry.get("query")
        or ""
    ).strip()
    if purpose != str(initial_observation.get("scan_purpose") or "").strip():
        return False

    selected_frame_count = _entry_selected_frame_count(entry)
    if selected_frame_count != int(initial_observation.get("scan_num_frames") or _DEFAULT_SCAN_NUM_FRAMES):
        return False

    expected_start_sec, expected_end_sec = expected_initial_scan_window_sec(
        multimodal_cache,
        config=config,
    )
    actual_start_sec = _coerce_float(entry.get("start_sec"), expected_start_sec)
    actual_end_sec = _coerce_float(entry.get("end_sec"), expected_end_sec)
    return (
        abs(actual_start_sec - expected_start_sec) <= _WINDOW_TOLERANCE_SEC
        and abs(actual_end_sec - expected_end_sec) <= _WINDOW_TOLERANCE_SEC
    )


def mark_initial_global_scan_message(message: dict[str, Any], *, config: Any = None) -> dict[str, Any]:
    initial_observation = normalize_initial_observation_config(config)
    message[INITIAL_GLOBAL_SCAN_MESSAGE_KEY] = True
    message[PROTECT_INITIAL_GLOBAL_SCAN_KEY] = bool(initial_observation.get("protect_from_visual_budget"))
    message[ERROR_ON_INITIAL_SCAN_SEQ_PRUNE_KEY] = bool(initial_observation.get("error_on_seq_prune"))
    return message


def is_initial_global_scan_message(message: Mapping[str, Any] | None) -> bool:
    return bool((message or {}).get(INITIAL_GLOBAL_SCAN_MESSAGE_KEY))


def protect_initial_global_scan_message(message: Mapping[str, Any] | None) -> bool:
    return bool((message or {}).get(PROTECT_INITIAL_GLOBAL_SCAN_KEY))


def error_on_initial_scan_seq_prune(message: Mapping[str, Any] | None) -> bool:
    return bool((message or {}).get(ERROR_ON_INITIAL_SCAN_SEQ_PRUNE_KEY))
