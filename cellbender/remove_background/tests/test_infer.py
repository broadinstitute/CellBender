"""Test functions in infer.py"""

import pytest
import scipy.sparse as sp
import numpy as np
import torch

from cellbender.remove_background.data.dataprep import DataLoader
from cellbender.remove_background.infer import Posterior

from .conftest import sparse_matrix_equal, simulated_dataset


USE_CUDA = torch.cuda.is_available()


@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
def test_dense_to_sparse_op_numpy(simulated_dataset, cuda):
    """test infer.py Posterior.dense_to_sparse_op_numpy()"""

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
            Posterior.dense_to_sparse_op_numpy(dense_counts.detach().cpu().numpy())

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
    counts = np.array(np.concatenate(tuple(counts)), dtype=np.uint32)
    barcodes = np.array(np.concatenate(tuple(barcodes)), dtype=np.uint32)
    genes = np.array(np.concatenate(tuple(genes)), dtype=np.uint32)  # uint16 is too small!

    # Put the counts into a sparse csc_matrix.
    out = sp.csc_matrix((counts, (barcodes, genes)),
                        shape=d['matrix'].shape)

    assert sparse_matrix_equal(out, d['matrix'])


@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
def test_dense_to_sparse_op_torch(simulated_dataset, cuda):
    """test infer.py Posterior.dense_to_sparse_op_torch()"""

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
            Posterior.dense_to_sparse_op_torch(dense_counts)

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
    counts = torch.cat(counts, dim=0).detach().cpu().numpy().astype(np.uint32)
    barcodes = torch.cat(barcodes, dim=0).detach().cpu().numpy().astype(np.uint32)
    genes = torch.cat(genes, dim=0).detach().cpu().numpy().astype(np.uint32)  # uint16 is too small!

    # Put the counts into a sparse csc_matrix.
    out = sp.csc_matrix((counts, (barcodes, genes)),
                        shape=d['matrix'].shape)

    assert sparse_matrix_equal(out, d['matrix'])
