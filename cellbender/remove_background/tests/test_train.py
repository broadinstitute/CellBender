"""Test functionality in train.py"""

import pytest
import scipy.sparse as sp
import numpy as np
import unittest.mock


@pytest.mark.skip
def test_epoch_elbo_fail_restart():
    """Trigger failure and ensure a new attempt is made"""
    pass


@pytest.mark.skip
def test_final_elbo_fail_restart():
    """Trigger failure and ensure a new attempt is made"""
    pass


@pytest.mark.skip
def test_restart_matches_scratch():
    """Ensure that an auto-restart gives same ELBO as a start-from-scratch"""
    # https://stackoverflow.com/questions/50964786/mock-exception-raised-in-function-using-pytest
    pass


@pytest.mark.skip
def test_num_training_attempts():
    """Make sure the number of training attempts matches input arg"""
    pass
