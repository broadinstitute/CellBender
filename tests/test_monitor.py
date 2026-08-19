"""Tests for monitoring function."""

import pytest
from conftest import DEVICES

from cellbender.monitor import get_hardware_usage


@pytest.mark.parametrize("device", DEVICES)
def test_get_hardware_usage(device):
    """Check that a hardware usage snapshot can be produced for each backend."""

    print(get_hardware_usage(device=device))
