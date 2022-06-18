"""Test functions in dataset.py"""

import pytest
import scipy.sparse as sp
import numpy as np
import torch

from cellbender.remove_background.data.dataset import \
    _overwrite_matrix_with_columns_from_another, _csr_set_rows_to_zero
import cellbender.remove_background.consts as consts
from cellbender.remove_background.gmm import GMM

from .conftest import sparse_matrix_equal, USE_CUDA


@pytest.mark.parametrize('mat1, mat2, col_inds',
                         [(sp.csc_matrix([[1, 2], [3, 4]]),
                           sp.csc_matrix([[0, 0], [0, 0]]),
                           [0]),
                          (sp.csc_matrix([[1, 2], [3, 4], [5, 6]]),
                           sp.csc_matrix([[0, 0], [0, 0], [0, 0]]),
                           [1]),
                          (sp.csc_matrix(np.random.poisson(lam=2., size=(10, 10))),
                           sp.csc_matrix(np.random.poisson(lam=2., size=(10, 10))),
                           [0, 1, 2, 3])])
def test_overwrite_matrix_with_columns_from_another(mat1: sp.csc_matrix,
                                                    mat2: sp.csc_matrix,
                                                    col_inds: np.ndarray):
    """test dataset.py _overwrite_matrix_with_columns_from_another()"""

    out = _overwrite_matrix_with_columns_from_another(mat1=mat1, mat2=mat2, column_inds=col_inds)
    excluded_col_inds = [i for i in range(mat1.shape[1]) if i not in col_inds]

    print('col_inds')
    print(col_inds)
    print('mat1')
    print(mat1.todense())
    print('mat2')
    print(mat2.todense())
    print('out')
    print(out.todense())

    print('assertion')
    print(out[:, excluded_col_inds].todense())
    print(mat2[:, excluded_col_inds].todense())

    print('assertion')
    print(out[:, col_inds].todense())
    print(mat1[:, col_inds].todense())

    # excluded columns should be replaced with new values
    assert sparse_matrix_equal(out[:, excluded_col_inds], mat2[:, excluded_col_inds])

    # included columns should be left alone
    assert sparse_matrix_equal(out[:, col_inds], mat1[:, col_inds])


@pytest.mark.parametrize('mat, row_inds',
                         [(sp.csc_matrix([[1, 2], [3, 4]]),
                           [0]),
                          (sp.csc_matrix([[1, 2], [3, 4], [5, 6]]),
                           [1]),
                          (sp.csc_matrix(np.random.poisson(lam=2., size=(10, 10))),
                           [0, 1, 2, 3])])
def test_csr_set_rows_to_zero(mat: sp.csr_matrix, row_inds: np.ndarray):
    """test dataset.py _overwrite_matrix_with_columns_from_another()"""

    out = _csr_set_rows_to_zero(csr=mat, row_inds=row_inds)
    other_row_inds = [i for i in range(mat.shape[0]) if i not in row_inds]

    print('row_inds')
    print(row_inds)
    print('mat')
    print(mat.todense())
    print('out')
    print(out.todense())

    print('assertion')
    print(out[other_row_inds, :].todense())
    print(mat[other_row_inds, :].todense())

    # other rows should be left alone
    assert sparse_matrix_equal(out[other_row_inds, :], mat[other_row_inds, :])

    # specified rows should be all zero
    assert out[row_inds, :].sum() == 0


@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
def test_gmm(cuda):
    """Dummy data no error; and also specific test cases"""

    counts = np.concatenate((np.random.rand(1000) + 2, 2 * np.random.rand(100) + 4))

    # Fit a Gaussian mixture model to the counts.
    gmm = GMM(torch.tensor(counts).float().to('cuda' if cuda else 'cpu'),
              n_components=consts.GMM_COMPONENTS,
              alpha_prior=consts.GMM_ALPHA_PRIOR,
              use_cuda=cuda,
              verbose=False)
    gmm.train(epochs=consts.GMM_EPOCHS)
    map_est = gmm.map_estimate()


@pytest.mark.skip
def test_heuristic_priors():
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


@pytest.mark.skip
def test_saving():
    """Dummy data, and ensure saving all outputs does not error"""
    pass
