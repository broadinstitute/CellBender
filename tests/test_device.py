"""Tests for backend device selection and backend-specific helpers."""

import os

import pytest
import torch
from conftest import DEVICES

from cellbender.device import (
    MPS_FALLBACK_ENV_VAR,
    available_devices,
    checkpoint_map_location,
    empty_cache,
    float_dtype,
    get_device_rng_state,
    maybe_enable_mps_fallback,
    resolve_device,
    seed_all,
    set_device_rng_state,
    supports_float64,
    torch_has_native_mps_sampling,
)

MPS_AVAILABLE = "mps" in DEVICES
CUDA_AVAILABLE = "cuda" in DEVICES


def test_available_devices_always_includes_cpu():
    assert "cpu" in available_devices()


def test_available_devices_ordered_fastest_first():
    """CPU is the fallback, so it must come last."""
    assert available_devices()[-1] == "cpu"


def test_resolve_device_defaults_to_cpu():
    assert resolve_device(use_cuda=False, use_mps=False) == "cpu"


def test_resolve_device_rejects_both_backends():
    with pytest.raises(ValueError, match="at most one"):
        resolve_device(use_cuda=True, use_mps=True)


@pytest.mark.skipif(CUDA_AVAILABLE, reason="CUDA is available on this machine")
def test_resolve_device_raises_when_cuda_unavailable():
    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_device(use_cuda=True, use_mps=False)


@pytest.mark.skipif(MPS_AVAILABLE, reason="MPS is available on this machine")
def test_resolve_device_raises_when_mps_unavailable():
    with pytest.raises(ValueError, match="MPS"):
        resolve_device(use_cuda=False, use_mps=True)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires MPS")
def test_resolve_device_returns_mps_when_available():
    assert resolve_device(use_cuda=False, use_mps=True) == "mps"


def test_maybe_enable_mps_fallback_ignores_other_commands():
    assert maybe_enable_mps_fallback(["cellbender", "remove-background", "--cuda"]) is False


def test_maybe_enable_mps_fallback_respects_explicit_env(monkeypatch):
    """An operator who set the variable by hand keeps control of it."""
    monkeypatch.setenv(MPS_FALLBACK_ENV_VAR, "0")
    assert maybe_enable_mps_fallback(["cellbender", "remove-background", "--mps"]) is False
    assert os.environ[MPS_FALLBACK_ENV_VAR] == "0"


def test_maybe_enable_mps_fallback_tracks_torch_version(monkeypatch):
    """The fallback is only needed while torch lacks Metal sampling kernels."""
    monkeypatch.delenv(MPS_FALLBACK_ENV_VAR, raising=False)
    enabled = maybe_enable_mps_fallback(["cellbender", "remove-background", "--mps"])
    assert enabled is not torch_has_native_mps_sampling()
    if enabled:
        assert os.environ[MPS_FALLBACK_ENV_VAR] == "1"


def test_checkpoint_map_location():
    assert checkpoint_map_location("cuda") == "cuda:0"
    assert checkpoint_map_location("mps") == "mps"
    assert checkpoint_map_location("cpu") == "cpu"


def test_supports_float64():
    """Metal Shading Language has no double type."""
    assert supports_float64("cpu")
    assert supports_float64("cuda")
    assert not supports_float64("mps")


def test_float_dtype():
    assert float_dtype("cpu") == torch.float64
    assert float_dtype("cuda") == torch.float64
    assert float_dtype("mps") == torch.float32


@pytest.mark.parametrize("device", DEVICES)
def test_float_dtype_is_constructible_on_device(device):
    """The dtype we advertise for a backend must actually work on it."""
    tensor = torch.ones(3, dtype=float_dtype(device), device=device)
    assert tensor.device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_seed_all_is_reproducible(device):
    seed_all(0, device)
    first = torch.randn(5, device=device)
    seed_all(0, device)
    second = torch.randn(5, device=device)
    assert torch.equal(first, second)


@pytest.mark.parametrize("device", DEVICES)
def test_device_rng_state_roundtrip(device):
    state = get_device_rng_state(device)
    if device == "cpu":
        assert state is None
        return
    expected = torch.randn(5, device=device)
    set_device_rng_state(device, state)
    assert torch.equal(torch.randn(5, device=device), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_cache_runs(device):
    empty_cache(device)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires MPS")
def test_every_distribution_the_model_samples_works_on_mps():
    """Guard against a regression in MPS coverage of the model's sampling sites.

    The generative model and guide sample from Gamma (phi, epsilon), Beta (rho),
    LogNormal (d_cell, d_empty), Normal (z), Bernoulli (y) and Poisson (counts).
    All but Poisson sampling have Metal kernels as of torch 2.13.

    Note that this cannot check the CPU fallback: torch reads
    PYTORCH_ENABLE_MPS_FALLBACK when it registers the fallback kernel at import
    time, and torch is already imported by the time this test runs.
    """
    import torch.distributions as dist

    ones = torch.ones(5, device="mps")

    assert dist.Gamma(ones * 2, ones).rsample().device.type == "mps"
    assert dist.Beta(ones * 2, ones * 3).rsample().device.type == "mps"
    assert dist.Dirichlet(ones).rsample().device.type == "mps"
    assert dist.LogNormal(torch.zeros(5, device="mps"), ones).rsample().device.type == "mps"
    assert dist.Normal(torch.zeros(5, device="mps"), ones).rsample().device.type == "mps"
    assert dist.Bernoulli(logits=torch.zeros(5, device="mps")).sample().device.type == "mps"
    assert dist.Poisson(ones * 3).log_prob(ones).device.type == "mps"


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires MPS")
@pytest.mark.skipif(
    os.environ.get(MPS_FALLBACK_ENV_VAR) == "1",
    reason="CPU fallback was enabled before torch was imported",
)
def test_poisson_sampling_on_mps_matches_torch_version():
    """aten::poisson gained a Metal kernel after the torch 2.13 branch cut.

    Below that version CellBender needs the CPU fallback, which is why
    maybe_enable_mps_fallback exists. This pins which side of that line we are on
    so the fallback can be dropped once the floor moves past it.
    """
    import torch.distributions as dist

    rate = torch.ones(5, device="mps") * 3

    if torch_has_native_mps_sampling():
        assert dist.Poisson(rate).sample().device.type == "mps"
    else:
        with pytest.raises(NotImplementedError, match="aten::poisson"):
            dist.Poisson(rate).sample()
