"""Tests for monitoring function."""

import pytest
from conftest import DEVICE_PARAMS

from cellbender.monitor import get_hardware_usage


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_get_hardware_usage(device):
    """Check and see if restarting from a checkpoint picks up in the same place
    we left off.  Use our model and dataloader.
    """

    print(get_hardware_usage(device=device))
