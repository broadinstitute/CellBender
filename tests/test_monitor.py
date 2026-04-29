"""Tests for monitoring function."""

import pytest
import torch

from cellbender.monitor import get_hardware_usage

USE_CUDA = torch.cuda.is_available()


@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
def test_get_hardware_usage(cuda):
    """Check and see if restarting from a checkpoint picks up in the same place
    we left off.  Use our model and dataloader.
    """

    print(get_hardware_usage(use_cuda=cuda))
