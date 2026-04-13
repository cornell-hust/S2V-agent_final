import json

INPUT_PATH = "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3/data_utils/sft_train.compact_trace_v2.jsonl"
OUTPUT_PATH = "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3/data_utils/sft_train.compact_trace_v2.weighted.jsonl"

WEIGHT_MAP = {
    "normal": 1.0,
    "anomaly": 0.6,
}

total = 0
normal_count = 0
anomaly_count = 0
unknown_count = 0

with open(INPUT_PATH, "r") as fin, open(OUTPUT_PATH, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        existence = record.get("structured_target", {}).get("existence", "")
        if existence == "normal":
            weight = 1.0
            normal_count += 1
        elif existence == "anomaly":
            weight = 0.6
            anomaly_count += 1
        else:
            weight = 1.0
            unknown_count += 1
            print(f"WARNING: unknown existence value '{existence}' for record, defaulting weight to 1.0")
        record["sample_weight"] = weight
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        total += 1

mean_weight = (normal_count * 1.0 + anomaly_count * 0.6) / total if total > 0 else 0.0

print(f"Summary:")
print(f"  Total records : {total}")
print(f"  Normal        : {normal_count}  (weight=1.0)")
print(f"  Anomaly       : {anomaly_count}  (weight=0.6)")
if unknown_count:
    print(f"  Unknown       : {unknown_count}  (weight=1.0 default)")
print(f"  Mean weight   : {mean_weight:.4f}")
print(f"Output written to: {OUTPUT_PATH}")
