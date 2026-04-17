#!/usr/bin/env python3

import argparse
import time


def gib_to_bytes(value):
    return int(value * 1024**3)


def format_gib(value):
    return f"{value / 1024**3:.2f} GiB"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Use PyTorch to occupy about 80 GiB on each CUDA GPU and idle until Ctrl+C."
    )
    parser.add_argument("--gpus", type=int, default=8, help="How many GPUs to occupy.")
    parser.add_argument("--target-gib", type=float, default=80.0, help="Target memory per GPU.")
    parser.add_argument("--chunk-gib", type=float, default=2.0, help="Allocation chunk size per GPU.")
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="How often to print memory status while idling.",
    )
    return parser


def allocate_on_gpu(torch, gpu_id, target_gib, chunk_gib):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    target_bytes = gib_to_bytes(target_gib)
    chunk_bytes = gib_to_bytes(chunk_gib)
    buffers = []

    while torch.cuda.memory_allocated(device) < target_bytes:
        used = torch.cuda.memory_allocated(device)
        remaining = target_bytes - used
        next_chunk = min(chunk_bytes, remaining)

        # uint8 gives 1 byte per element, so numel == bytes.
        buffers.append(torch.empty(next_chunk, dtype=torch.uint8, device=device))
        used = torch.cuda.memory_allocated(device)
        print(f"[GPU {gpu_id}] allocated {format_gib(used)} / {format_gib(target_bytes)}", flush=True)

    return buffers


def print_status(torch, gpus):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for gpu_id in range(gpus):
        device = torch.device(f"cuda:{gpu_id}")
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        print(
            f"[{stamp}] GPU {gpu_id}: allocated={format_gib(allocated)}, reserved={format_gib(reserved)}",
            flush=True,
        )


def main():
    args = build_parser().parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    if torch.cuda.device_count() < args.gpus:
        raise SystemExit(f"Requested {args.gpus} GPUs, but only found {torch.cuda.device_count()}.")

    buffers = []

    try:
        for gpu_id in range(args.gpus):
            buffers.append(allocate_on_gpu(torch, gpu_id, args.target_gib, args.chunk_gib))

        print("All GPUs are occupied. Press Ctrl+C to release memory and exit.", flush=True)

        while True:
            time.sleep(args.heartbeat_seconds)
            print_status(torch, args.gpus)

    except KeyboardInterrupt:
        print("\nInterrupted, releasing GPU memory...", flush=True)
    finally:
        buffers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
