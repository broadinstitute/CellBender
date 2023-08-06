"""Test functions in dataset.py"""

import pytest
import scipy.sparse as sp
import numpy as np

from .conftest import sparse_matrix_equal


@pytest.mark.skip
def test_heuristic_priors():
    pass


@pytest.mark.skip
def test_feature_type_exclusion():
    # TODO there seems to be an error
    # TODO see https://github.com/broadinstitute/CellBender/issues/121#issuecomment-1443995082
    pass


@pytest.mark.skip
def test_barcode_trimming():
    pass


@pytest.mark.skip
def test_feature_trimming():
    pass


@pytest.mark.skip
def test_restore_eliminated_features_in_cells():
    pass


@pytest.mark.skip
def test_remove_zero_count_cells():
    """Functionality yet to be written too"""
    pass
