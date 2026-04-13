#!/usr/bin/env python3
"""
Generate hard-normal SFT trajectories for data augmentation.

Problem: All 360 normal trajectories use an identical 4-step pattern with a
constant template query, while anomaly trajectories have varied, descriptive
queries. This creates behavioral asymmetry where the model learns
"finding content = anomaly" because it never sees content-rich normal videos.

Solution: Create 120 hard-normal variants from the 360 normal records with:
  - Scene-specific seek_evidence queries
  - Randomized sufficiency_score in [0.75, 0.88]
  - Enhanced qa_focus_answers with explicit normal reasoning
  - sample_weight=1.2 (vs 1.0 for base normal, 0.6 for anomaly)
"""

import copy
import json
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
INPUT_PATH = (
    "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3"
    "/data_utils/sft_train.compact_trace_v2.jsonl"
)
OUTPUT_PATH = (
    "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3"
    "/data_utils/sft_train.compact_trace_v2.hard_normal_augmented.jsonl"
)

NUM_HARD_NORMALS = 120
HARD_NORMAL_WEIGHT = 1.2
BASE_NORMAL_WEIGHT = 1.0
ANOMALY_WEIGHT = 0.6

# ---------------------------------------------------------------------------
# Scene-specific query templates
# Templates use {scene_activity}, {scene_element}, {scene_description}
# which are populated from the record's scene/scenario field.
# ---------------------------------------------------------------------------
QUERY_TEMPLATES = [
    "check whether {scene_activity} constitutes suspicious or anomalous behavior requiring intervention",
    "investigate if the observed {scene_element} in the footage represents a genuine security concern or routine activity",
    "determine whether {scene_description} indicates any anomalous event or is consistent with normal operations",
    "assess whether activity observed in the {scene_element} footage warrants an alert or reflects ordinary behavior",
    "examine if the {scene_activity} captured in this segment poses any safety or security risk",
    "evaluate whether the {scene_element} scene contains evidence of misconduct or is operating within normal parameters",
    "verify whether {scene_description} is indicative of a reportable incident or standard day-to-day activity",
]

# Per-scenario descriptive expansions for the three template slots
SCENARIO_DESCRIPTORS = {
    "frontdoor": {
        "scene_activity": "movement near the building entrance",
        "scene_element": "front door area",
        "scene_description": "activity at the building entry point",
    },
    "highway": {
        "scene_activity": "vehicle movement on the highway",
        "scene_element": "highway corridor",
        "scene_description": "traffic flow on the highway segment",
    },
    "mall": {
        "scene_activity": "pedestrian movement inside the shopping mall",
        "scene_element": "mall interior",
        "scene_description": "shopper activity within the commercial area",
    },
    "office": {
        "scene_activity": "occupant behavior in the office area",
        "scene_element": "office environment",
        "scene_description": "workplace activity observed in the office",
    },
    "park": {
        "scene_activity": "visitor movement throughout the park",
        "scene_element": "park area",
        "scene_description": "outdoor activity in the park setting",
    },
    "parkinglot": {
        "scene_activity": "vehicle and pedestrian behavior in the parking lot",
        "scene_element": "parking lot",
        "scene_description": "parking lot activity including vehicle movement",
    },
    "pedestrian_street": {
        "scene_activity": "pedestrian flow along the street",
        "scene_element": "pedestrian street",
        "scene_description": "foot traffic on the pedestrian walkway",
    },
    "restaurant": {
        "scene_activity": "patron and staff behavior inside the restaurant",
        "scene_element": "restaurant area",
        "scene_description": "dining activity within the restaurant premises",
    },
    "road": {
        "scene_activity": "vehicular and pedestrian movement on the road",
        "scene_element": "road segment",
        "scene_description": "traffic and street-level activity on the road",
    },
    "shop": {
        "scene_activity": "customer and staff interactions inside the shop",
        "scene_element": "shop interior",
        "scene_description": "retail activity observed within the store",
    },
    "sidewalk": {
        "scene_activity": "pedestrian movement on the sidewalk",
        "scene_element": "sidewalk area",
        "scene_description": "foot traffic along the sidewalk corridor",
    },
    "street_highview": {
        "scene_activity": "street-level activity from an elevated vantage point",
        "scene_element": "street (high-angle view)",
        "scene_description": "overhead-captured movement on the street",
    },
    "train": {
        "scene_activity": "passenger behavior at the train platform or carriage",
        "scene_element": "train environment",
        "scene_description": "transit activity at the train station or onboard",
    },
    "warehouse": {
        "scene_activity": "worker and equipment movement inside the warehouse",
        "scene_element": "warehouse facility",
        "scene_description": "operational activity within the warehouse space",
    },
}

# Fallback for unknown scenarios
DEFAULT_DESCRIPTORS = {
    "scene_activity": "activity observed in the surveillance footage",
    "scene_element": "monitored area",
    "scene_description": "behavior captured in the video segment",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_scenario(record):
    scene = record.get("scene", {})
    if isinstance(scene, dict):
        return scene.get("scenario", "")
    return str(scene)


def get_descriptors(scenario):
    return SCENARIO_DESCRIPTORS.get(scenario, DEFAULT_DESCRIPTORS)


def make_scene_specific_query(scenario, rng):
    template = rng.choice(QUERY_TEMPLATES)
    descriptors = get_descriptors(scenario)
    return template.format(**descriptors)


def build_hard_normal(record, rng):
    """
    Return a deep copy of `record` with hard-normal modifications applied:
      a) seek_evidence query -> scene-specific descriptive query
      b) verify_hypothesis sufficiency_score -> random in [0.75, 0.88]
      c) finalize_case qa_focus_answers -> enhanced existence answer
      d) sample_weight = HARD_NORMAL_WEIGHT
      e) hard_normal flag = True
    """
    rec = copy.deepcopy(record)
    scenario = get_scenario(rec)
    descriptors = get_descriptors(scenario)
    scene_element = descriptors["scene_element"]

    new_trajectory = []
    for step in rec["oracle_trajectory"]:
        step = copy.deepcopy(step)
        tool = step["tool"]

        if tool == "seek_evidence":
            # (a) Replace constant template query with scene-specific query
            step["arguments"]["query"] = make_scene_specific_query(scenario, rng)

        elif tool == "verify_hypothesis":
            # (b) Randomize sufficiency_score to [0.75, 0.88] to simulate deliberation
            new_score = round(rng.uniform(0.75, 0.88), 4)
            step["arguments"]["sufficiency_score"] = new_score
            # Keep oracle_verifier_feedback in sync if present
            if "oracle_verifier_feedback" in step:
                step["oracle_verifier_feedback"]["sufficiency_score"] = new_score

        elif tool == "finalize_case":
            # (c) Enhance qa_focus_answers with explicit normal reasoning
            qa = step["arguments"].get("qa_focus_answers", {})
            existing_existence = qa.get(
                "existence", "No. No anomaly is visible in this video."
            )
            append_text = (
                " After investigating {}, "
                "the evidence confirms routine, non-threatening activity.".format(scene_element)
            )
            qa["existence"] = existing_existence.rstrip() + append_text
            step["arguments"]["qa_focus_answers"] = qa

        new_trajectory.append(step)

    rec["oracle_trajectory"] = new_trajectory

    # Mark as hard-normal variant
    rec["sample_weight"] = HARD_NORMAL_WEIGHT
    if "structured_target" in rec:
        rec["structured_target"]["hard_normal"] = True

    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(RANDOM_SEED)

    # Load all records
    with open(INPUT_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]

    normal_records = [
        r for r in records if r["structured_target"]["existence"] == "normal"
    ]
    anomaly_records = [
        r for r in records if r["structured_target"]["existence"] == "anomaly"
    ]

    print("Loaded {} total records".format(len(records)))
    print("  Normal  : {}".format(len(normal_records)))
    print("  Anomaly : {}".format(len(anomaly_records)))
    assert len(normal_records) == 360, "Expected 360 normals, got {}".format(len(normal_records))
    assert len(anomaly_records) == 120, "Expected 120 anomalies, got {}".format(len(anomaly_records))

    # Select 120 normals randomly to become hard-normal variants
    selected_indices = rng.sample(range(len(normal_records)), NUM_HARD_NORMALS)
    selected_for_hard = [normal_records[i] for i in selected_indices]

    # Build output records
    output_records = []

    # 1. Original 360 normals with sample_weight=1.0
    for rec in normal_records:
        r = copy.deepcopy(rec)
        r["sample_weight"] = BASE_NORMAL_WEIGHT
        output_records.append(r)

    # 2. 120 hard-normal variants with sample_weight=1.2
    hard_normals = [build_hard_normal(rec, rng) for rec in selected_for_hard]
    output_records.extend(hard_normals)

    # 3. Original 120 anomalies with sample_weight=0.6
    for rec in anomaly_records:
        r = copy.deepcopy(rec)
        r["sample_weight"] = ANOMALY_WEIGHT
        output_records.append(r)

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nOutput: {}".format(OUTPUT_PATH))
    print("Total records written: {}".format(len(output_records)))

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    weights = {}
    scenario_counts = {}
    seek_queries = set()
    sufficiency_scores = []

    for rec in output_records:
        w = rec.get("sample_weight", None)
        weights[w] = weights.get(w, 0) + 1

    for rec in hard_normals:
        scenario = get_scenario(rec)
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        for step in rec["oracle_trajectory"]:
            if step["tool"] == "seek_evidence":
                seek_queries.add(step["arguments"]["query"])
            if step["tool"] == "verify_hypothesis":
                sufficiency_scores.append(step["arguments"]["sufficiency_score"])

    print("\n--- Summary Statistics ---")
    print("Records by sample_weight:")
    for w in sorted(weights.keys()):
        print("  weight={}: {} records".format(w, weights[w]))
    print("\nHard-normal variants generated: {}".format(len(hard_normals)))
    print("Unique seek_evidence queries in hard-normals: {}".format(len(seek_queries)))
    if sufficiency_scores:
        print("Sufficiency score range: [{:.4f}, {:.4f}]".format(
            min(sufficiency_scores), max(sufficiency_scores)))
        print("Sufficiency score mean : {:.4f}".format(
            sum(sufficiency_scores) / len(sufficiency_scores)))
    print("\nHard-normal scenario distribution:")
    for scenario in sorted(scenario_counts):
        print("  {}: {}".format(scenario, scenario_counts[scenario]))

    # Spot-check one hard-normal
    print("\n--- Spot-check: first hard-normal ---")
    hn = hard_normals[0]
    print("video_id: {}".format(hn["video_id"]))
    print("scenario: {}".format(get_scenario(hn)))
    print("sample_weight: {}".format(hn["sample_weight"]))
    print("hard_normal flag: {}".format(hn["structured_target"].get("hard_normal")))
    for step in hn["oracle_trajectory"]:
        if step["tool"] == "seek_evidence":
            print("seek_evidence query: {}".format(step["arguments"]["query"]))
        if step["tool"] == "verify_hypothesis":
            print("verify sufficiency_score: {}".format(step["arguments"]["sufficiency_score"]))
        if step["tool"] == "finalize_case":
            print("finalize existence answer: {}".format(
                step["arguments"]["qa_focus_answers"]["existence"]))

    print("\nDone.")


if __name__ == "__main__":
    main()
