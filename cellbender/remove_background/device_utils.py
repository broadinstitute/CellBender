"""Device utilities for supporting CUDA, MPS (Apple Silicon), and CPU backends."""

import logging
from typing import Optional

import pyro
import torch

logger = logging.getLogger('cellbender')


def get_available_device() -> str:
    """Detect the best available device.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def is_gpu_available() -> bool:
    """Check if any GPU (CUDA or MPS) is available."""
    return torch.cuda.is_available() or (
        hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    )


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon GPU) is available."""
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def get_device_for_args(use_cuda: bool, use_mps: bool = False) -> str:
    """Get the appropriate device string based on use_cuda and use_mps flags.

    Args:
        use_cuda: The value of args.use_cuda from command line
        use_mps: The value of args.use_mps from command line (explicit MPS request)

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if use_mps:
        if not is_mps_available():
            raise RuntimeError(
                "Trying to use MPS acceleration (--mps flag), but MPS "
                "(Apple Silicon) is not available on this system."
            )
        return 'mps'

    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Trying to use CUDA acceleration (--cuda flag), but CUDA "
                "is not available. Use --mps for Apple Silicon GPU."
            )
        return 'cuda'

    return 'cpu'


def empty_cache_if_available(device: str) -> None:
    """Empty GPU cache if using CUDA. MPS does not have this method."""
    if device == 'cuda':
        torch.cuda.empty_cache()


def set_manual_seed(seed: int, device: str) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        device: Device string to determine which backend seeds to set
    """
    torch.manual_seed(seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_memory_info(device: str) -> dict:
    """Get memory information for the device if available.

    Args:
        device: Device string

    Returns:
        Dictionary with memory info (empty dict for CPU/MPS)
    """
    if device == 'cuda':
        return {
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        }
    elif device == 'mps':
        try:
            return {
                'allocated_gb': torch.mps.current_allocated_memory() / 1e9,
            }
        except AttributeError:
            return {}
    return {}


def get_rng_state(device: str) -> dict:
    """Get random number generator states for checkpointing.

    Args:
        device: Device string

    Returns:
        Dictionary of RNG states
    """
    states = {
        'torch': torch.get_rng_state(),
    }
    if device == 'cuda' and torch.cuda.is_available():
        states['cuda'] = torch.cuda.get_rng_state_all()
    return states


def set_rng_state(states: dict, device: str) -> None:
    """Restore random number generator states from checkpoint.

    Args:
        states: Dictionary of RNG states
        device: Device string
    """
    if 'torch' in states:
        torch.set_rng_state(states['torch'])
    if device == 'cuda' and 'cuda' in states:
        torch.cuda.set_rng_state_all(states['cuda'])


def resolve_posterior_device(preference: str,
                             training_device: Optional[str],
                             model_device: Optional[str]) -> str:
    """Resolve which device should be used for posterior computations.

    Args:
        preference: User preference string (auto, model, cpu, cuda, mps)
        training_device: Device used during training (if known)
        model_device: Device where the model currently lives

    Returns:
        Device string for posterior calculations.
    """
    requested = (preference or 'auto').lower()
    fallback_device = model_device or training_device or get_available_device()
    training_device = training_device or fallback_device

    if requested == 'auto':
        if training_device == 'mps':
            return 'cpu'
        return fallback_device

    if requested == 'model':
        return fallback_device

    if requested == 'cpu':
        return 'cpu'

    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("Requested posterior_device=cuda but CUDA is unavailable.")
        return 'cuda'

    if requested == 'mps':
        if not is_mps_available():
            raise RuntimeError("Requested posterior_device=mps but MPS is unavailable.")
        return 'mps'

    raise ValueError(f"Unknown posterior_device preference: {preference}")


def move_param_store_to_device(target_device: str) -> None:
    """Move all tensors in the Pyro param store to the requested device."""
    store = pyro.get_param_store()
    for name in list(store.keys()):
        value = pyro.param(name)
        if not torch.is_tensor(value):
            continue
        if value.device.type == target_device:
            continue
        store[name] = value.detach().to(target_device)


def move_model_to_device(model: torch.nn.Module, target_device: str) -> torch.nn.Module:
    """Move RemoveBackgroundPyroModel and the Pyro param store to a new device."""
    if model is None:
        return None

    current_device = str(getattr(model, 'device', 'cpu'))
    if current_device == target_device:
        return model

    model.to(target_device)

    encoder = getattr(model, 'encoder', None)
    if isinstance(encoder, dict):
        for module in encoder.values():
            if hasattr(module, 'to'):
                module.to(target_device)

    decoder = getattr(model, 'decoder', None)
    if decoder is not None and hasattr(decoder, 'to'):
        decoder.to(target_device)

    # Move plain tensor attributes (priors, init tensors) that are not
    # registered as nn.Parameter or buffers and thus missed by .to().
    for attr_name in list(vars(model).keys()):
        val = getattr(model, attr_name)
        if torch.is_tensor(val) and val.device != torch.device(target_device):
            setattr(model, attr_name, val.to(target_device))

    if hasattr(model, 'device'):
        model.device = target_device
    if hasattr(model, 'use_cuda'):
        model.use_cuda = (target_device == 'cuda')
    if hasattr(model, 'use_mps'):
        model.use_mps = (target_device == 'mps')
    if hasattr(model, '_pyro_use_cuda'):
        model._pyro_use_cuda = bool(getattr(model, 'use_cuda', False))

    if target_device == 'mps' and hasattr(model, '_make_all_parameters_contiguous'):
        model._make_all_parameters_contiguous()

    move_param_store_to_device(target_device)
    return model
