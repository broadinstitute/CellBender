"""Backend device selection and backend-specific helpers.

CellBender can run on three PyTorch backends: CUDA, MPS (Apple Silicon GPUs, via
Metal), and CPU. Everything that differs between them is collected here, so that
the rest of the codebase only passes around a plain device string.

This module deliberately does not import torch at module level. Enabling the MPS
CPU fallback only takes effect if the environment variable is set before torch is
first imported (see ``maybe_enable_mps_fallback``), and this module is imported
from ``cellbender.base_cli`` for precisely that reason. Functions that need torch
import it locally.
"""

import logging
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger("cellbender")

DEVICES = ("cpu", "cuda", "mps")

MPS_FALLBACK_ENV_VAR = "PYTORCH_ENABLE_MPS_FALLBACK"

# Metal kernels for aten::poisson and aten::binomial landed in torch after the
# 2.13 branch cut, so from 2.14 onward every op CellBender uses has a native MPS
# implementation and no CPU fallback is needed.
FIRST_TORCH_VERSION_WITH_NATIVE_MPS_SAMPLING = (2, 14)


def _installed_torch_version() -> Optional[Tuple[int, ...]]:
    """Read the installed torch version without importing torch.

    Returns None if torch is not installed or the version is unparseable.
    """
    try:
        raw = _package_version("torch")
    except PackageNotFoundError:
        return None

    # Strip any local version label, e.g. "2.13.0+cu128".
    numbers: List[int] = []
    for chunk in raw.split("+")[0].split("."):
        if not chunk.isdigit():
            break
        numbers.append(int(chunk))

    return tuple(numbers) if numbers else None


def torch_has_native_mps_sampling() -> bool:
    """True if the installed torch has Metal kernels for every op we sample from."""
    installed = _installed_torch_version()
    if installed is None:
        return False
    return installed >= FIRST_TORCH_VERSION_WITH_NATIVE_MPS_SAMPLING


def maybe_enable_mps_fallback(argv: Sequence[str]) -> bool:
    """Turn on the MPS-to-CPU fallback for ops without a Metal kernel.

    Must be called before anything imports torch: torch reads this environment
    variable when it registers the MPS fallback kernel, so setting it afterwards
    has no effect.

    On torch < 2.14, ``aten::poisson`` and ``aten::binomial`` have no Metal
    kernel, and CellBender samples from both. Those ops raise NotImplementedError
    on MPS unless the fallback is enabled. From torch 2.14 the fallback is not
    needed, and leaving it off means a future gap surfaces as a loud error rather
    than as a silent slow path.

    An explicit setting in the environment is always respected.

    Args:
        argv: Command line arguments, normally ``sys.argv``.

    Returns:
        True if the CPU fallback is enabled for this process.
    """
    if "--mps" not in argv:
        return False

    if MPS_FALLBACK_ENV_VAR in os.environ:
        return os.environ[MPS_FALLBACK_ENV_VAR] == "1"

    if torch_has_native_mps_sampling():
        return False

    os.environ[MPS_FALLBACK_ENV_VAR] = "1"
    return True


def resolve_device(use_cuda: bool, use_mps: bool) -> str:
    """Pick the backend to run on, checking that it is actually usable.

    Args:
        use_cuda: True if the user passed --cuda.
        use_mps: True if the user passed --mps.

    Returns:
        One of 'cuda', 'mps', 'cpu'.

    Raises:
        ValueError: If both backends are requested, or the requested one is
            unavailable.
    """
    import torch

    if use_cuda and use_mps:
        raise ValueError("Specify at most one of --cuda and --mps.")

    if use_cuda:
        if not torch.cuda.is_available():
            raise ValueError("Trying to use CUDA, but CUDA is not available.")
        return "cuda"

    if use_mps:
        if not torch.backends.mps.is_built():
            raise ValueError("Trying to use MPS, but this PyTorch installation was not built with MPS enabled.")
        if not torch.backends.mps.is_available():
            raise ValueError(
                "Trying to use MPS, but no MPS-enabled device is available. "
                "MPS requires macOS 12.3 or later on Apple Silicon."
            )
        return "mps"

    return "cpu"


def available_devices() -> List[str]:
    """Backends usable on this machine, fastest first. CPU is always included."""
    import torch

    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def seed_all(seed: int, device: str) -> None:
    """Seed the RNG of the host and of the given backend."""
    import torch

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device == "mps":
        torch.mps.manual_seed(seed)


def get_device_rng_state(device: str) -> Optional[Any]:
    """Return the backend RNG state, or None for backends that have none of their own."""
    import torch

    if device == "cuda":
        return torch.cuda.get_rng_state_all()
    if device == "mps":
        return torch.mps.get_rng_state()
    return None


def set_device_rng_state(device: str, state: Any) -> None:
    """Restore a backend RNG state produced by :func:`get_device_rng_state`."""
    import torch

    if device == "cuda":
        torch.cuda.set_rng_state_all(state)
    elif device == "mps":
        torch.mps.set_rng_state(state)


def empty_cache(device: str) -> None:
    """Release cached device memory back to the driver."""
    import torch

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def checkpoint_map_location(device: str) -> str:
    """Device string to load a checkpoint onto."""
    return "cuda:0" if device == "cuda" else device


def supports_float64(device: str) -> bool:
    """False for MPS: Metal Shading Language has no double type.

    This is a hardware limitation rather than a missing kernel, so
    PYTORCH_ENABLE_MPS_FALLBACK does not work around it. Computations that would
    normally use float64 run in float32 on MPS.
    """
    return device != "mps"


def float_dtype(device: str) -> Any:
    """Widest float dtype the backend supports."""
    import torch

    return torch.float32 if device == "mps" else torch.float64
