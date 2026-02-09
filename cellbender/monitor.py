"""Utility functions for hardware monitoring"""

# Inspiration for the nvidia-smi command comes from here:
# https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/gpu_stats_monitor.html#GPUStatsMonitor
# but here it is stripped down to the absolute minimum

import torch
import psutil
from psutil._common import bytes2human
import shutil
import subprocess

from cellbender.remove_background.device_utils import is_mps_available


def get_hardware_usage(use_cuda: bool, device: str = None) -> str:
    """Get a current snapshot of RAM, CPU, GPU memory, and GPU utilization as a string.

    Args:
        use_cuda: Whether GPU acceleration was requested
        device: Optional device string ('cuda', 'mps', 'cpu'). If not provided,
                will be inferred from use_cuda and available backends.
    """

    mem = psutil.virtual_memory()
    gpu_string = ''

    # Determine device if not provided
    if device is None:
        if use_cuda:
            if torch.cuda.is_available():
                device = 'cuda'
            elif is_mps_available():
                device = 'mps'
            else:
                device = 'cpu'
        else:
            device = 'cpu'

    if device == 'cuda':
        # Run nvidia-smi to get GPU utilization
        try:
            gpu_query = 'utilization.gpu'
            format = 'csv,nounits,noheader'
            result = subprocess.run(
                [shutil.which("nvidia-smi"), f"--query-gpu={gpu_query}", f"--format={format}"],
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            pct_gpu_util = result.stdout.strip()
            gpu_string = (f'Volatile GPU utilization: {pct_gpu_util} %\n'
                          f'GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB\n'
                          f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n')
        except (subprocess.SubprocessError, TypeError):
            gpu_string = 'CUDA GPU (nvidia-smi not available)\n'

    elif device == 'mps':
        # MPS (Apple Silicon) memory monitoring
        gpu_string = 'Apple Silicon GPU (MPS):\n'
        try:
            # Available in PyTorch 2.0+
            allocated = torch.mps.current_allocated_memory() / 1e9
            gpu_string += f'  MPS memory allocated: {allocated:.2f} GB\n'
        except AttributeError:
            gpu_string += '  (MPS memory stats not available in this PyTorch version)\n'

        try:
            # Available in PyTorch 2.1+
            driver_allocated = torch.mps.driver_allocated_memory() / 1e9
            gpu_string += f'  MPS driver memory: {driver_allocated:.2f} GB\n'
        except AttributeError:
            pass

    cpu_string = (f'Avg CPU load over past minute: '
                  f'{psutil.getloadavg()[0] / psutil.cpu_count() * 100:.1f} %\n'
                  f'RAM in use: {bytes2human(mem.used)} ({mem.percent} %)')

    return gpu_string + cpu_string
