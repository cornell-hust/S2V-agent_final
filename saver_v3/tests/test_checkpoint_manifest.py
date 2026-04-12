from pathlib import Path

from saver_v3.common.checkpoint_manifest import CheckpointManifest, load_checkpoint_manifest, write_checkpoint_manifest


def test_checkpoint_manifest_round_trip(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    manifest = CheckpointManifest(
        checkpoint_dir=str(checkpoint_dir),
        base_model="/models/qwen3-vl-8b",
        source_stage="sft",
        epoch=1,
        global_step=42,
    )
    path = write_checkpoint_manifest(manifest)
    loaded = load_checkpoint_manifest(path)
    assert loaded.base_model == manifest.base_model
    assert loaded.attn_implementation == "flash_attention_3"
