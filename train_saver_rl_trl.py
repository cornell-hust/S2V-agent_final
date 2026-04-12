#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import train_saver_rl as legacy_train_saver_rl

from saver_v3.common.experiment_logging import resolve_experiment_log_dir, utc_timestamp, write_json
from saver_v3.common.runtime import distributed_barrier
from saver_v3.rl.trl_grpo_trainer import build_recovery_vllm_policy_factory, run_trainer_vllm_grpo


def _build_vllm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--use-vllm",
        type=legacy_train_saver_rl._parse_bool_flag,
        default=True,
        help="Enable the TimeSearch-R-style vLLM RL route. Defaults to true for the new TRL entrypoint.",
    )
    parser.add_argument(
        "--vllm-mode",
        choices=["colocate"],
        default="colocate",
        help="vLLM execution mode. Defaults to colocate to mirror TimeSearch-R.",
    )
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
    parser.add_argument(
        "--vllm-server-host",
        default="127.0.0.1",
        help="Optional vLLM server host when --vllm-mode server is used.",
    )
    parser.add_argument(
        "--vllm-server-port",
        type=int,
        default=8000,
        help="Optional vLLM server port when --vllm-mode server is used.",
    )
    parser.add_argument(
        "--vllm-server-timeout",
        type=float,
        default=240.0,
        help="Connection timeout in seconds for vLLM server mode.",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    for token in raw_argv:
        if token == "--rl-all-empty-policy" or str(token).startswith("--rl-all-empty-policy="):
            raise SystemExit(
                "--rl-all-empty-policy is not supported by train_saver_rl_trl.py; "
                "the dedicated TRL/vLLM route always uses donor no-op padding plus zero-loss no-op steps."
            )
    vllm_parser = _build_vllm_parser()
    vllm_args, remaining = vllm_parser.parse_known_args(raw_argv)
    base_args = legacy_train_saver_rl.parse_args(remaining)
    merged = argparse.Namespace(**vars(base_args))
    for key, value in vars(vllm_args).items():
        setattr(merged, key, value)
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
            "include_splits": legacy_train_saver_rl.parse_include_splits(args.include_splits) or [],
            "output_dir": args.output_dir,
            "log_dir": str(log_dir),
            "rollout_eval_output_dir": rollout_eval_output_dir,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "resume_rollout_eval_only": bool(args.resume_rollout_eval_only),
            "inline_rollout_eval": bool(args.inline_rollout_eval),
            "model_path": args.model_path,
            "reference_model_path": args.reference_model_path,
            "num_iterations": int(args.num_iterations),
            "rollout_count": int(args.rollout_count),
            "num_generations": int(args.num_generations),
            "rollout_max_turns": int(args.rollout_max_turns),
            "rl_reward_version": str(args.rl_reward_version),
            "kl_beta": float(args.kl_beta),
            "use_vllm": bool(args.use_vllm),
            "vllm_mode": str(args.vllm_mode),
            "vllm_tensor_parallel_size": int(args.vllm_tensor_parallel_size),
            "vllm_gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
            "vllm_guided_decoding_regex": str(args.vllm_guided_decoding_regex or ""),
            "vllm_server_host": str(args.vllm_server_host),
            "vllm_server_port": int(args.vllm_server_port),
            "vllm_server_timeout": float(args.vllm_server_timeout),
            "rl_rollout_use_cache": bool(args.rl_rollout_use_cache),
            "rl_fecv_use_cache": bool(args.rl_fecv_use_cache),
            "rl_compute_loss_microbatch_size": int(args.rl_compute_loss_microbatch_size),
            "rl_steps_per_generation": int(args.rl_steps_per_generation),
            "rl_replay_buffer_enable": bool(args.rl_replay_buffer_enable),
            "rl_replay_buffer_type": str(args.rl_replay_buffer_type),
            "rl_replay_buffer_capacity": int(args.rl_replay_buffer_capacity),
            "rl_replay_buffer_alpha": float(args.rl_replay_buffer_alpha),
            "rl_fecv_failure_policy": str(args.rl_fecv_failure_policy),
            "rl_log_empty_batch_rank_summary": bool(args.rl_log_empty_batch_rank_summary),
            "rl_open_ended_judge_enabled": bool(args.rl_open_ended_judge_enabled),
            "rl_open_ended_judge_base_url": str(args.rl_open_ended_judge_base_url or ""),
            "rl_open_ended_judge_model": str(args.rl_open_ended_judge_model or ""),
            "rl_open_ended_judge_cache_path": str(args.rl_open_ended_judge_cache_path or ""),
            "rl_open_ended_judge_timeout_sec": float(args.rl_open_ended_judge_timeout_sec),
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

    runtime = legacy_train_saver_rl.distributed_runtime_from_env()
    legacy_train_saver_rl.init_torch_distributed(runtime)
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
        rollout_eval_config = legacy_train_saver_rl._build_rollout_eval_config(
            args=args,
            current_model_path=args.model_path,
            reference_model_path=legacy_train_saver_rl.resolve_reference_model_path(
                args.model_path,
                args.reference_model_path,
            ),
            config=legacy_train_saver_rl._build_config(args),
        )
        if rollout_eval_config is None:
            raise ValueError("--resume-rollout-eval-only requires --eval-data so the missing RL rollout eval can be replayed.")
        checkpoint_path = Path(args.resume_from_checkpoint)
        epoch_index = legacy_train_saver_rl._resolve_resume_epoch_index(checkpoint_path)
        iteration_index = legacy_train_saver_rl._resolve_resume_iteration_index(checkpoint_path)
        recovery_kwargs = dict(
            checkpoint_path=checkpoint_path,
            output_dir=legacy_train_saver_rl._resolve_resume_checkpoint_record_output_dir(
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
        if bool(getattr(args, "use_vllm", True)):
            recovery_kwargs["policy_factory"] = build_recovery_vllm_policy_factory(args=args)
        result = legacy_train_saver_rl.run_rollout_eval_from_checkpoint(**recovery_kwargs)
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
        config_builder=legacy_train_saver_rl._build_config,
        eval_config_builder=legacy_train_saver_rl._build_rollout_eval_config,
        reference_model_resolver=legacy_train_saver_rl.resolve_reference_model_path,
        select_iteration_indices_fn=legacy_train_saver_rl.select_iteration_indices,
    )
    final_summary = {"latest_checkpoint": str(result.get("latest_checkpoint", args.model_path))}
    if runtime.is_main_process:
        print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return final_summary


if __name__ == "__main__":
    main()
