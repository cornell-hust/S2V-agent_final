from __future__ import annotations

MESSAGE = (
    "This legacy saver_v3.common.training module is retired and no longer runnable. "
    "Use the active SEEK v3 path instead: scripts/prepare_sft_manifest.sh --run, "
    "scripts/run_full_pipeline.sh, saver_v3.cli.train_sft_ds, saver_v3.cli.train_rl_ds, "
    "and saver_v3.sft.training.run_standard_sft. "
    "The only supported data families are compact_trace_v5, materialized_sft_messages_v5, "
    "and materialized_runtime_items_v5."
)


def _raise_retired() -> None:
    raise RuntimeError(MESSAGE)


def __getattr__(name: str):
    del name
    _raise_retired()


def main() -> None:
    raise SystemExit(MESSAGE)


if __name__ == "__main__":
    main()
