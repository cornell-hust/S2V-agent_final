import json
from pathlib import Path

from saver_v3.rl.cli_shared import select_iteration_indices
from saver_v3.rl.runtime import RLJobConfig, build_active_rl_trl_argv


def _write_model_config(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(
        "\n".join(
            [
                "base_model: /models/base",
                "torch_dtype: bfloat16",
                "attn_implementation: flash_attention_3",
                "sequence:",
                "  max_length: 8192",
                "vision:",
                "  max_images_per_sample: 28",
                "  max_image_side: 640",
                "  max_image_pixels: 0",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_attention_config(tmp_path: Path) -> Path:
    path = tmp_path / "attention.yaml"
    path.write_text("policy_name: fa3_only\n", encoding="utf-8")
    return path


def _write_rollout_config(tmp_path: Path) -> Path:
    path = tmp_path / "rollout.yaml"
    path.write_text("semantic_metrics:\n  enabled: false\n", encoding="utf-8")
    return path


def _write_runtime_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _write_rl_config(
    tmp_path: Path,
    *,
    train_manifest: Path,
    rollout_count: int,
    materialized_train_items_path: Path | None = None,
    include_splits: str = "train",
    num_iterations: int = 999,
) -> Path:
    path = tmp_path / "config.yaml"
    lines = [
        "run_name: test",
        "output_dir: artifacts/test_rl",
        f"rollout_config: {_write_rollout_config(tmp_path)}",
        "policy_init_from: /models/checkpoint",
        "rollout_backend: vllm",
        "data:",
        f"  train_manifest: {train_manifest}",
        "  eval_manifest: /data/eval.jsonl",
        "  data_root: /data",
        f"  include_splits: {include_splits}",
        "optimization:",
        f"  num_iterations: {int(num_iterations)}",
        "  iteration_budget_multiplier: 1.5",
        f"  rollout_count: {int(rollout_count)}",
        "rewards:",
        "  reward_version: timesearch_v4",
    ]
    if materialized_train_items_path is not None:
        lines.insert(9, f"  materialized_train_items_path: {materialized_train_items_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_select_iteration_indices_uses_full_dataset_cycle_even_with_imbalanced_records():
    records = [
        {"structured_target": {"existence": "anomaly"}},
        {"structured_target": {"existence": "normal"}},
        {"structured_target": {"existence": "normal"}},
        {"structured_target": {"existence": "normal"}},
        {"structured_target": {"existence": "normal"}},
        {"structured_target": {"existence": "normal"}},
    ]

    selected_with_records = select_iteration_indices(
        6,
        4,
        0,
        0,
        seed=7,
        records=records,
    )
    selected_without_records = select_iteration_indices(6, 4, 0, 0, seed=7)

    assert selected_with_records == selected_without_records
    assert len(selected_with_records) == 4
    assert len(set(selected_with_records)) == 4
    assert selected_with_records.count(0) <= 1


def test_rl_job_derives_num_iterations_from_filtered_train_manifest(tmp_path: Path):
    train_manifest = _write_runtime_manifest(
        tmp_path,
        [
            {
                "video_id": f"train-{index}",
                "video_path": f"/data/train-{index}.mp4",
                "split": "train",
            }
            for index in range(10)
        ]
        + [
            {
                "video_id": f"test-{index}",
                "video_path": f"/data/test-{index}.mp4",
                "split": "test",
            }
            for index in range(2)
        ],
    )
    config_path = _write_rl_config(tmp_path, train_manifest=train_manifest, rollout_count=4)

    job = RLJobConfig.from_files(
        config_path=str(config_path),
        model_config_path=str(_write_model_config(tmp_path)),
        attention_config_path=str(_write_attention_config(tmp_path)),
    )
    argv = build_active_rl_trl_argv(job)

    assert job.train_record_count == 10
    assert job.iteration_budget_multiplier == 1.5
    assert job.num_iterations == 4
    assert job.num_iterations_source == "dynamic_train_record_count"
    assert argv[argv.index("--num-iterations") + 1] == "4"


def test_rl_job_prefers_materialized_train_items_for_dynamic_num_iterations(tmp_path: Path):
    source_manifest = _write_runtime_manifest(
        tmp_path,
        [
            {
                "video_id": f"source-{index}",
                "video_path": f"/data/source-{index}.mp4",
                "split": "train",
            }
            for index in range(20)
        ],
    )
    materialized_path = tmp_path / "train.materialized_items_v5.jsonl"
    materialized_rows = [
        {
            "video_id": f"materialized-{index}",
            "split": "train",
            "record": {"split": "train"},
        }
        for index in range(5)
    ]
    materialized_rows.append(
        {
            "video_id": "materialized-test",
            "split": "test",
            "record": {"split": "test"},
        }
    )
    materialized_path.write_text(
        "\n".join(json.dumps(row) for row in materialized_rows) + "\n",
        encoding="utf-8",
    )
    config_path = _write_rl_config(
        tmp_path,
        train_manifest=source_manifest,
        materialized_train_items_path=materialized_path,
        rollout_count=4,
    )

    job = RLJobConfig.from_files(
        config_path=str(config_path),
        model_config_path=str(_write_model_config(tmp_path)),
        attention_config_path=str(_write_attention_config(tmp_path)),
    )

    assert job.train_record_count == 5
    assert job.train_record_count_source.startswith("materialized_train_items_path:")
    assert job.num_iterations == 2
