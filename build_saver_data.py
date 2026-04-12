#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from convert_to_saver_agent import convert_record, iter_jsonl, write_jsonl
from saver_v3.data.config import PromptConfig, PreviewConfig, RolloutTraceConfig, SaverAgentConfig
from saver_v3.data.prepared_metadata import write_prepared_sft_metadata
from saver_v3.core.proposal import SiglipFeatureEncoder
from saver_v3.common.runtime import distributed_runtime_from_env, runtime_log
from saver_v3.sft.training import validate_prepared_examples
from split_utils import parse_include_splits
from saver_v3.sft.training import build_prepared_sft_examples_from_jsonl


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the simplified SAVER data artifacts: runtime_train/runtime_test episode JSONLs "
            "and optional final SFT train JSONLs."
        )
    )
    parser.add_argument("--input", required=True, help="Canonical SAVER JSONL path.")
    parser.add_argument(
        "--runtime-train-output",
        required=True,
        help="Output episode-level runtime JSONL for train split.",
    )
    parser.add_argument(
        "--runtime-test-output",
        required=True,
        help="Output episode-level runtime JSONL for test split.",
    )
    parser.add_argument(
        "--sft-train-output",
        default="",
        help="Optional output compact_trace_v2 prepared SFT JSONL for train split.",
    )
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument(
        "--adapter",
        default="msad_saver_qwen",
        help="Canonical adapter passed through to convert_to_saver_agent.convert_record(...).",
    )
    parser.add_argument(
        "--train-splits",
        default="train",
        help="Comma-separated source splits to route into runtime_train and sft_train.",
    )
    parser.add_argument(
        "--test-splits",
        default="test",
        help="Comma-separated source splits to route into runtime_test.",
    )
    parser.add_argument(
        "--skip-invalid-jsonl-lines",
        action="store_true",
        help="Skip malformed canonical JSONL rows instead of failing immediately.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log conversion/build progress every N records/examples.",
    )
    parser.add_argument(
        "--proposal-model-path",
        default="",
        help="Optional SigLIP/CLIP path used during runtime->SFT preparation.",
    )
    parser.add_argument(
        "--proposal-torch-dtype",
        default="auto",
        help="Torch dtype for the optional proposal encoder.",
    )
    parser.add_argument(
        "--proposal-device",
        default="",
        help="Device for the optional proposal encoder. Empty means cuda:0 if available else cpu.",
    )
    parser.add_argument(
        "--validate-sft-data",
        action="store_true",
        help="Validate the final SFT JSONL payload before writing it.",
    )
    parser.add_argument(
        "--validate-materialization",
        action="store_true",
        help="During validation, resolve image_ref payloads back to frames.",
    )
    parser.add_argument(
        "--validation-max-examples",
        type=int,
        default=0,
        help="Optional cap on how many prepared examples to materialize during validation.",
    )
    parser.add_argument("--num-preview-frames", type=int, default=8, help="Preview frames for prepared SFT messages.")
    parser.add_argument("--preview-sampling-fps", type=float, default=None, help="Preview sampling fps for prepared SFT messages.")
    parser.add_argument("--initial-user-template", default="", help="Optional custom initial user template for prepared SFT messages.")
    parser.add_argument("--preview-instruction", default="", help="Optional custom preview instruction for prepared SFT messages.")
    parser.add_argument("--tool-response-template", default="", help="Optional custom tool response template for prepared SFT messages.")
    return parser.parse_args(argv)


def _load_canonical_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return list(
        iter_jsonl(
            Path(args.input),
            skip_invalid_lines=args.skip_invalid_jsonl_lines,
        )
    )


def _write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_config(args: argparse.Namespace) -> SaverAgentConfig:
    return SaverAgentConfig(
        preview=PreviewConfig(
            num_preview_frames=int(args.num_preview_frames),
            preview_sampling_fps=args.preview_sampling_fps,
            max_preview_frames=int(args.num_preview_frames),
        ),
        prompt=PromptConfig(
            initial_user_template=args.initial_user_template or PromptConfig().initial_user_template,
            preview_instruction=args.preview_instruction or PromptConfig().preview_instruction,
            tool_response_template=args.tool_response_template or PromptConfig().tool_response_template,
        ),
        rollout_trace=RolloutTraceConfig(),
    )


def _resolve_train_splits(args: argparse.Namespace) -> set[str]:
    return {str(value).strip() for value in parse_include_splits(args.train_splits) or [] if str(value).strip()}


def _resolve_test_splits(args: argparse.Namespace) -> set[str]:
    return {str(value).strip() for value in parse_include_splits(args.test_splits) or [] if str(value).strip()}


def _build_runtime_rows(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    runtime = distributed_runtime_from_env()
    train_splits = _resolve_train_splits(args)
    test_splits = _resolve_test_splits(args)
    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    num_skipped_other_splits = 0

    canonical_records = _load_canonical_records(args)
    total_records = len(canonical_records)
    for index, record in enumerate(canonical_records, start=1):
        converted = convert_record(
            record,
            mode="oracle_sft",
            adapter_name=args.adapter,
        )
        split = str(converted.get("split") or "").strip()
        if split in train_splits:
            train_rows.append(converted)
        elif split in test_splits:
            test_rows.append(converted)
        else:
            num_skipped_other_splits += 1
        if args.progress_every > 0 and (index == 1 or index == total_records or index % args.progress_every == 0):
            runtime_log(
                (
                    f"runtime build progress: records={index}/{total_records} "
                    f"train={len(train_rows)} test={len(test_rows)} skipped_other={num_skipped_other_splits}"
                ),
                runtime=runtime,
                main_process_only=True,
            )
    return (
        train_rows,
        test_rows,
        {
            "num_input_records": total_records,
            "num_runtime_train_records": len(train_rows),
            "num_runtime_test_records": len(test_rows),
            "num_skipped_other_splits": num_skipped_other_splits,
        },
    )


def _resolve_default_proposal_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda:0"


def _build_proposal_runtime(args: argparse.Namespace):
    if not str(args.proposal_model_path or "").strip():
        return None
    return SiglipFeatureEncoder.from_pretrained(
        args.proposal_model_path,
        torch_dtype=args.proposal_torch_dtype,
        device=args.proposal_device or _resolve_default_proposal_device(),
    )


def _run_sft_validation(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not args.validate_sft_data:
        return {}
    validation = validate_prepared_examples(
        rows,
        materialize_images=args.validate_materialization,
        max_materialized_examples=args.validation_max_examples,
        progress_every=args.progress_every,
    )
    if int(validation.get("num_errors", 0)) > 0:
        error_preview = dict(validation)
        error_preview["errors"] = list(validation.get("errors") or [])[:10]
        raise ValueError(
            "Final SFT data validation failed. "
            f"Preview: {json.dumps(error_preview, ensure_ascii=False, indent=2)}"
        )
    return validation
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    runtime = distributed_runtime_from_env()
    config = _build_config(args)

    runtime_train_rows, runtime_test_rows, runtime_summary = _build_runtime_rows(args)
    write_jsonl(Path(args.runtime_train_output), runtime_train_rows)
    write_jsonl(Path(args.runtime_test_output), runtime_test_rows)

    summary: Dict[str, Any] = {
        "input": str(args.input),
        "runtime_train_output": str(args.runtime_train_output),
        "runtime_test_output": str(args.runtime_test_output),
        **runtime_summary,
    }

    if args.sft_train_output:
        proposal_runtime = _build_proposal_runtime(args)
        sft_rows = build_prepared_sft_examples_from_jsonl(
            data_path=args.runtime_train_output,
            data_root=args.data_root,
            include_splits=args.train_splits,
            progress_every=args.progress_every,
            runtime=runtime,
            config=config,
            proposal_runtime=proposal_runtime,
        )
        validation_summary = _run_sft_validation(args, sft_rows)
        _write_jsonl(args.sft_train_output, sft_rows)
        write_prepared_sft_metadata(args.sft_train_output, config=config)
        summary.update(
            {
                "sft_train_output": str(args.sft_train_output),
                "num_sft_train_examples": len(sft_rows),
            }
        )
        if validation_summary:
            summary["sft_validation"] = validation_summary

    runtime_log(
        json.dumps(summary, ensure_ascii=False),
        runtime=runtime,
        main_process_only=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
