import pytest
import scipy.sparse as sp
import numpy as np
import torch

from cellbender.remove_background.sparse_utils import todense_fill, \
    zero_out_csr_rows, dense_to_sparse_op_torch, log_prob_sparse_to_dense
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
        barcodes.append(bcs_i)
        genes.append(genes_i)
        counts.append(counts_i)

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
