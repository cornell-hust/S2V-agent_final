from __future__ import annotations

import re
from typing import Any, Dict


NORMAL_CATEGORY = "normal"
CANONICAL_ANOMALY_CATEGORIES = (
    "assault",
    "robbery",
    "shooting",
    "fire",
    "explosion",
    "fighting",
    "people_falling",
    "traffic_accident",
    "object_falling",
    "vandalism",
    "water_incident",
)
CANONICAL_POLICY_CATEGORIES = (NORMAL_CATEGORY,) + CANONICAL_ANOMALY_CATEGORIES

SAFE_CATEGORY_ALIASES: Dict[str, str] = {
    "vehicle_collision": "traffic_accident",
    "vehicle_crash": "traffic_accident",
    "vehicle_accident": "traffic_accident",
    "collision": "traffic_accident",
    "car_crash": "traffic_accident",
    "traffic_collision": "traffic_accident",
    "physical_altercation": "fighting",
    "violent_confrontation": "fighting",
    "brawl": "fighting",
    "fall": "people_falling",
    "falling": "people_falling",
    "falling_or_tripping": "people_falling",
    "slip_and_fall": "people_falling",
    "person_with_weapon": "shooting",
    "gun_violence": "shooting",
    "armed_shooting": "shooting",
    "theft": "robbery",
    "bag_snatch": "robbery",
    "bag_theft": "robbery",
    "stealing": "robbery",
}

_SEPARATOR_PATTERN = re.compile(r"[\s\-\/]+")


def _clean_category_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = _SEPARATOR_PATTERN.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def canonicalize_saver_category(
    value: Any,
    *,
    existence: Any = None,
) -> str:
    existence_text = str(existence or "").strip().lower()
    if existence_text == "normal":
        return NORMAL_CATEGORY

    text = _clean_category_text(value)
    if not text:
        return NORMAL_CATEGORY if existence_text == "normal" else ""
    if text in CANONICAL_POLICY_CATEGORIES:
        return text
    if text in SAFE_CATEGORY_ALIASES:
        return SAFE_CATEGORY_ALIASES[text]

    if any(token in text for token in ("traffic", "collision", "crash", "vehicle_accident")):
        return "traffic_accident"
    if any(token in text for token in ("rob", "snatch", "steal", "theft")):
        return "robbery"
    if any(token in text for token in ("shoot", "gun", "weapon", "firearm")):
        return "shooting"
    if any(token in text for token in ("vandal", "graffiti", "property_damage")):
        return "vandalism"
    if "object" in text and any(token in text for token in ("fall", "drop")):
        return "object_falling"
    if any(token in text for token in ("trip", "slip")):
        return "people_falling"
    if "fall" in text and "object" not in text:
        return "people_falling"
    if any(token in text for token in ("fight", "altercation", "brawl", "confrontation")):
        return "fighting"
    if any(token in text for token in ("assault", "attack", "attacked", "punch", "kick", "beating")):
        return "assault"
    if any(token in text for token in ("water", "drown", "drowning")):
        return "water_incident"
    if text != "fire_or_explosion" and any(token in text for token in ("explosion", "blast", "explode")):
        return "explosion"
    if text != "fire_or_explosion" and any(token in text for token in ("fire", "flame", "smoke")):
        return "fire"

    return text


def canonicalize_category_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    has_category = "category" in normalized
    if has_category:
        normalized["category"] = canonicalize_saver_category(
            normalized.get("category"),
            existence=normalized.get("existence"),
        )
    elif str(normalized.get("existence") or "").strip().lower() == "normal":
        normalized["category"] = NORMAL_CATEGORY
    return normalized


def validate_canonical_category_payload(
    payload: Any,
    *,
    payload_name: str = "payload",
    require_category_for_anomaly: bool = False,
) -> Any:
    if not isinstance(payload, dict):
        return payload
    normalized = canonicalize_category_payload(payload)
    existence = str(normalized.get("existence") or "").strip().lower()
    has_category = "category" in normalized
    category = str(normalized.get("category") or "").strip()

    if require_category_for_anomaly and existence == "anomaly" and not category:
        raise ValueError(
            f"{payload_name}.category must use a canonical anomaly label when existence='anomaly'."
        )

    if not has_category or not category:
        return normalized

    if category not in CANONICAL_POLICY_CATEGORIES:
        raise ValueError(
            f"{payload_name}.category must be one of {list(CANONICAL_POLICY_CATEGORIES)}, got {category!r}."
        )
    if existence == "normal" and category != NORMAL_CATEGORY:
        raise ValueError(
            f"{payload_name}.category must be {NORMAL_CATEGORY!r} when existence='normal', got {category!r}."
        )
    if existence == "anomaly" and category == NORMAL_CATEGORY:
        raise ValueError(
            f"{payload_name}.category must be an anomaly label when existence='anomaly', got {category!r}."
        )
    return normalized
