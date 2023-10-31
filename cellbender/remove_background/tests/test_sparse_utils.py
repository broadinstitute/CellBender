import pytest
import scipy.sparse as sp
import numpy as np
import torch

from cellbender.remove_background.sparse_utils import todense_fill, \
    csr_set_rows_to_zero, dense_to_sparse_op_torch, log_prob_sparse_to_dense, \
    overwrite_matrix_with_columns_from_another
from cellbender.remove_background.data.dataprep import DataLoader
from .conftest import sparse_matrix_equal


USE_CUDA = torch.cuda.is_available()


@pytest.mark.parametrize('val', [0, 1, np.nan, np.inf, -np.inf])
def test_todense_fill(val):
    """Test densification of scipy sparse COO matrix with arbitrary fill value"""

    mat = np.array(
        [[0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 1, 2, 0],
         [1, 2, 3, 4]],
        dtype=float,
    )
    coo = sp.coo_matrix(mat)

    print('original')
    print(mat)
    print('sparse version')
    print(coo)

    print(f'densified using {val}')
    dense = todense_fill(coo=coo, fill_value=val)
    print(dense)
    brute_force = mat.copy()
    brute_force[brute_force == 0] = val
    np.testing.assert_array_equal(brute_force, dense)


@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
def test_dense_to_sparse_op_torch(simulated_dataset, cuda):
    """test infer.py BasePosterior.dense_to_sparse_op_torch()"""

    d = simulated_dataset
    data_loader = DataLoader(
        d['matrix'],
        empty_drop_dataset=None,
        batch_size=5,
        fraction_empties=0.,
        shuffle=False,
        use_cuda=cuda,
    )

    barcodes = []
    genes = []
    counts = []
    ind = 0

    for data in data_loader:
        dense_counts = data  # just make it the same!

        # Convert to sparse.
        bcs_i_chunk, genes_i, counts_i = \
            dense_to_sparse_op_torch(dense_counts)

        # Barcode index in the dataloader.
        bcs_i = bcs_i_chunk + ind

        # Obtain the real barcode index after unsorting the dataloader.
        bcs_i = data_loader.unsort_inds(bcs_i)

        # Add sparse matrix values to lists.
        barcodes.append(bcs_i.detach().cpu())
        genes.append(genes_i.detach().cpu())
        counts.append(counts_i.detach().cpu())

        # Increment barcode index counter.
        ind += data.shape[0]  # Same as data_loader.batch_size

    # Convert the lists to numpy arrays.
    counts = np.concatenate(counts).astype(np.uint32)
    barcodes = np.concatenate(barcodes).astype(np.uint32)
    genes = np.concatenate(genes).astype(np.uint32)  # uint16 is too small!

    # Put the counts into a sparse csc_matrix.
    out = sp.csc_matrix((counts, (barcodes, genes)),
                        shape=d['matrix'].shape)

    assert sparse_matrix_equal(out, d['matrix'])


def test_log_prob_sparse_to_dense():
    """Test densification of log prob sparse posteriors filling with -np.inf"""

    data = np.array([0, -1, -2, 0, -4, -2, 0])
    row = np.array([0, 0, 2, 2, 3, 3, 4])
    col = np.array([0, 1, 0, 1, 0, 1, 0])
    coo = sp.coo_matrix((data, (row, col)), shape=[5, 2])
    print(coo)
    print('scipy .todense()')
    print(coo.todense())
    print('log prob densification')
    mat = log_prob_sparse_to_dense(coo=coo)
    print(mat)
    print('nan densification')
    mat_nan = todense_fill(coo=coo, fill_value=np.nan)
    print(mat_nan)
    print('8 densification')
    mat_8 = todense_fill(coo=coo, fill_value=8.)
    print(mat_8)

    truth = lambda x: np.array(
        [[ 0., -1.],
         [ x,   x ],
         [-2.,  0.],
         [-4., -2.],
         [ 0.,  x ]],
    )
    print('truth with -np.inf')
    print(truth(-np.inf))

    np.testing.assert_array_equal(mat, truth(-np.inf))
    np.testing.assert_array_equal(mat_nan, truth(np.nan))
    np.testing.assert_array_equal(mat_8, truth(8.))


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
    """test overwrite_matrix_with_columns_from_another()"""

    out = overwrite_matrix_with_columns_from_another(mat1=mat1, mat2=mat2, column_inds=col_inds)
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
    """test csr_set_rows_to_zero()"""

    out = csr_set_rows_to_zero(csr=mat, row_inds=row_inds)
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
