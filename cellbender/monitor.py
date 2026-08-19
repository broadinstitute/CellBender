"""Utility functions for hardware monitoring"""

# Inspiration for the nvidia-smi command comes from here:
# https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/gpu_stats_monitor.html#GPUStatsMonitor
# but here it is stripped down to the absolute minimum

import shutil
import subprocess

import psutil
import torch
from psutil._common import bytes2human


def get_hardware_usage(device: str) -> str:
    """Get a current snapshot of RAM, CPU, GPU memory, and GPU utilization as a string

    Args:
        device: Backend in use, one of 'cpu', 'cuda', 'mps'.
    """

    mem = psutil.virtual_memory()

    if device == "cuda":
        # Run nvidia-smi to get GPU utilization
        gpu_query = "utilization.gpu"
        format = "csv,nounits,noheader"
        result = subprocess.run(
            [shutil.which("nvidia-smi") or "nvidia-smi", f"--query-gpu={gpu_query}", f"--format={format}"],
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
            check=True,
        )
        pct_gpu_util = result.stdout.strip()
        gpu_string = (
            f"Volatile GPU utilization: {pct_gpu_util} %\n"
            f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9} GB\n"
            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9} GB\n"
        )
    elif device == "mps":
        # Metal has no per-process utilization counter comparable to nvidia-smi,
        # so report the two memory figures torch.mps exposes.
        gpu_string = (
            f"GPU memory reserved: {torch.mps.driver_allocated_memory() / 1e9} GB\n"
            f"GPU memory allocated: {torch.mps.current_allocated_memory() / 1e9} GB\n"
        )
    else:
        gpu_string = ""

    cpu_string = (
        f"Avg CPU load over past minute: "
        f"{psutil.getloadavg()[0] / psutil.cpu_count() * 100:.1f} %\n"
        f"RAM in use: {bytes2human(mem.used)} ({mem.percent} %)"
    )

    return gpu_string + cpu_string
