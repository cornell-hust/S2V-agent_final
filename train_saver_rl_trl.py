#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from saver_v3.common.experiment_logging import resolve_experiment_log_dir, utc_timestamp, write_json
from saver_v3.common.runtime import distributed_barrier
from saver_v3.rl import cli_shared
from saver_v3.rl.trl_grpo_trainer import build_recovery_vllm_policy_factory, run_trainer_vllm_grpo


def _build_vllm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for colocated vLLM workers.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.35,
        help="GPU memory utilization target for colocated vLLM workers.",
    )
    parser.add_argument(
        "--vllm-guided-decoding-regex",
        default="",
        help="Optional guided decoding regex passed through the vLLM route.",
    )
    parser.add_argument("--materialized-train-items-path", default="", help="Optional materialized runtime-items cache for RL training.")
    parser.add_argument("--materialized-eval-items-path", default="", help="Optional materialized runtime-items cache for RL eval/reference flows.")
    parser.add_argument("--require-materialized-runtime-cache", type=cli_shared._parse_bool_flag, default=False)
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=4,
        help="RL-only explicit max_num_seqs override for the vLLM runtime. Defaults to 4.",
    )
    parser.add_argument(
        "--vllm-fallback-max-num-seqs",
        type=int,
        default=2,
        help="RL-only fallback max_num_seqs used when vLLM init fails due to GPU memory pressure. Defaults to 2.",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    cli_shared.fail_on_removed_active_rl_flags(raw_argv)
    for token in list(raw_argv):
        text = str(token or "")
        if text == "--use-vllm" or text.startswith("--use-vllm="):
            raise SystemExit(
                "--use-vllm has been removed; active RL always uses the colocated vLLM pure-pack route."
            )
        if text == "--vllm-mode" or text.startswith("--vllm-mode="):
            raise SystemExit(
                "--vllm-mode has been removed; active RL only supports the colocated pure-pack route."
            )
        if text in {"--vllm-server-host", "--vllm-server-port", "--vllm-server-timeout"} or any(
            text.startswith(f"{flag}=") for flag in ("--vllm-server-host", "--vllm-server-port", "--vllm-server-timeout")
        ):
            raise SystemExit(
                f"{text.split('=')[0]} has been removed; active RL no longer supports the server/client vLLM path."
            )
    vllm_parser = _build_vllm_parser()
    vllm_args, remaining = vllm_parser.parse_known_args(raw_argv)
    base_args = cli_shared.parse_active_rl_args(
        remaining,
        description="SAVER active RL entrypoint for the trajectory-level TRL + colocated-vLLM GRPO route.",
    )
    merged = argparse.Namespace(**vars(base_args))
    for key, value in vars(vllm_args).items():
        setattr(merged, key, value)
    merged.use_vllm = True
    merged.vllm_mode = "colocate"
    return merged


def run_trainer_trl_vllm(
    *,
    args: Any,
    runtime: Any,
    log_dir: str | Path = "",
    config_builder: Any,
    eval_config_builder: Any,
    reference_model_resolver: Any,
    select_iteration_indices_fn: Any,
) -> Dict[str, Any]:
    return run_trainer_vllm_grpo(
        args=args,
        runtime=runtime,
        log_dir=str(log_dir) if log_dir else "",
        config_builder=config_builder,
        eval_config_builder=eval_config_builder,
        reference_model_resolver=reference_model_resolver,
        select_iteration_indices_fn=select_iteration_indices_fn,
    )


def _write_run_config(
    *,
    args: argparse.Namespace,
    log_dir: Path,
    rollout_eval_output_dir: str,
) -> None:
    write_json(
        log_dir / "train_saver_rl_run_config.json",
        {
            "timestamp_utc": utc_timestamp(),
            "script_entrypoint": "train_saver_rl_trl.py",
            "data": args.data,
            "data_root": args.data_root,
            "materialized_train_items_path": str(getattr(args, "materialized_train_items_path", "") or ""),
            "materialized_eval_items_path": str(getattr(args, "materialized_eval_items_path", "") or ""),
            "require_materialized_runtime_cache": bool(getattr(args, "require_materialized_runtime_cache", False)),
            "include_splits": cli_shared.parse_include_splits(args.include_splits) or [],
            "output_dir": args.output_dir,
            "log_dir": str(log_dir),
            "rollout_eval_output_dir": rollout_eval_output_dir,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "resume_rollout_eval_only": bool(args.resume_rollout_eval_only),
            "inline_rollout_eval": bool(args.inline_rollout_eval),
            "rollout_eval_start_iteration": int(getattr(args, "rollout_eval_start_iteration", 1) or 1),
            "rollout_eval_interval_iterations": int(getattr(args, "rollout_eval_interval_iterations", 1) or 1),
            "final_rollout_eval": bool(getattr(args, "final_rollout_eval", False)),
            "model_path": args.model_path,
            "reference_model_mode": "per_iteration_trainer_init",
            "reference_model_source_path": args.model_path,
            "num_iterations": int(args.num_iterations),
            "rollout_count": int(args.rollout_count),
            "num_generations": int(args.num_generations),
            "rollout_max_turns": int(args.rollout_max_turns),
            "rl_reward_version": str(args.rl_reward_version),
            "kl_beta": float(args.kl_beta),
            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_tensor_parallel_size": int(args.vllm_tensor_parallel_size),
            "vllm_gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
            "vllm_guided_decoding_regex": str(args.vllm_guided_decoding_regex or ""),
            "vllm_max_num_seqs": int(args.vllm_max_num_seqs),
            "vllm_fallback_max_num_seqs": int(args.vllm_fallback_max_num_seqs),
            "rl_rollout_use_cache": bool(args.rl_rollout_use_cache),
            "rl_fecv_use_cache": bool(args.rl_fecv_use_cache),
            "rl_compute_loss_microbatch_size": int(args.rl_compute_loss_microbatch_size),
            "rl_steps_per_generation": int(args.rl_steps_per_generation),
            "use_liger_loss_requested": bool(getattr(args, "use_liger_loss", False)),
            "rollout_stage_batch_size": int(args.rollout_stage_batch_size),
            "fecv_stage_batch_size": int(args.fecv_stage_batch_size),
            "episode_grpo_trajectory_level": True,
            "rl_replay_buffer_supported": False,
            "rl_replay_buffer_mode": "disabled_episode_grpo",
            "rl_fecv_failure_policy": str(args.rl_fecv_failure_policy),
            "rl_log_empty_batch_rank_summary": bool(args.rl_log_empty_batch_rank_summary),
            "rl_reward_config": dict(getattr(args, "rl_reward_config", {}) or {}),
            "deepspeed": str(getattr(args, "deepspeed", "") or ""),
            "proposal_model_path": str(args.proposal_model_path or ""),
            "teacher_judge_model_path": str(args.teacher_judge_model_path or ""),
            "teacher_judge_input_mode": str(args.teacher_judge_input_mode or ""),
            "eval_data": str(args.eval_data or ""),
        },
    )


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if not args.resume_rollout_eval_only and not args.data:
        raise ValueError("--data is required unless --resume-rollout-eval-only is used.")
    if args.resume_rollout_eval_only and not args.resume_from_checkpoint:
        raise ValueError("--resume-rollout-eval-only requires --resume-from-checkpoint.")

    runtime = cli_shared.distributed_runtime_from_env()
    cli_shared.init_torch_distributed(runtime)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = resolve_experiment_log_dir(args.log_dir, output_dir=args.output_dir)
    rollout_eval_output_dir = str(args.rollout_eval_output_dir or "").strip()

    if runtime.is_main_process and log_dir is not None:
        _write_run_config(
            args=args,
            log_dir=Path(log_dir),
            rollout_eval_output_dir=rollout_eval_output_dir,
        )

    if args.resume_rollout_eval_only:
        rollout_eval_config = cli_shared.build_rollout_eval_config(
            args=args,
            current_model_path=args.model_path,
            reference_model_path=args.model_path,
            config=cli_shared.build_saver_config(args),
        )
        if rollout_eval_config is None:
            raise ValueError("--resume-rollout-eval-only requires --eval-data so the missing RL rollout eval can be replayed.")
        checkpoint_path = Path(args.resume_from_checkpoint)
        epoch_index = cli_shared.resolve_resume_epoch_index(checkpoint_path)
        iteration_index = cli_shared.resolve_resume_iteration_index(checkpoint_path)
        recovery_kwargs = dict(
            checkpoint_path=checkpoint_path,
            output_dir=cli_shared.resolve_resume_checkpoint_record_output_dir(
                checkpoint_path,
                fallback_output_dir=output_dir,
            ),
            rollout_eval_config=rollout_eval_config,
            epoch_index=epoch_index,
            model_path=args.model_path,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation or None,
            runtime=runtime,
        )
        if rollout_eval_output_dir:
            recovery_kwargs["rollout_eval_output_dir"] = rollout_eval_output_dir
        recovery_kwargs["policy_factory"] = build_recovery_vllm_policy_factory(args=args)
        from saver_v3.sft.training import run_rollout_eval_from_checkpoint

        result = run_rollout_eval_from_checkpoint(**recovery_kwargs)
        final_summary = {
            "resume_from_checkpoint": str(checkpoint_path),
            "resume_rollout_eval_only": True,
            "resume_iteration_index": int(iteration_index),
            "resume_epoch_index": int(epoch_index),
            "script_entrypoint": "train_saver_rl_trl.py",
            **(result or {}),
        }
        if runtime.is_main_process and log_dir is not None:
            write_json(Path(log_dir) / "train_saver_rl_summary.json", final_summary)
        if runtime.is_main_process:
            print(json.dumps(final_summary, ensure_ascii=False, indent=2))
        return final_summary

    result = run_trainer_trl_vllm(
        args=args,
        runtime=runtime,
        log_dir=log_dir,
        config_builder=cli_shared.build_saver_config,
        eval_config_builder=cli_shared.build_rollout_eval_config,
        reference_model_resolver=cli_shared.resolve_reference_model_path,
        select_iteration_indices_fn=cli_shared.select_iteration_indices,
    )
    final_summary = {"latest_checkpoint": str(result.get("latest_checkpoint", args.model_path))}
    if runtime.is_main_process:
        print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return final_summary


if __name__ == "__main__":
    main()
