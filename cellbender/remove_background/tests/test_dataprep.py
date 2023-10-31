"""Test functions in dataprep.py"""

import pytest
import scipy.sparse as sp
import numpy as np
import torch

from cellbender.remove_background.data.dataprep import DataLoader
from cellbender.remove_background.sparse_utils import dense_to_sparse_op_torch

from .conftest import sparse_matrix_equal, simulated_dataset


USE_CUDA = torch.cuda.is_available()


@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
def test_dataloader_sorting(simulated_dataset, cuda):
    """test dataset.py _overwrite_matrix_with_columns_from_another()"""

    d = simulated_dataset
    data_loader = DataLoader(
        d['matrix'],
        empty_drop_dataset=None,
        batch_size=5,
        fraction_empties=0.,
        shuffle=False,
        use_cuda=cuda,
    )
    sorted_data_loader = DataLoader(
        d['matrix'],
        empty_drop_dataset=None,
        batch_size=5,
        fraction_empties=0.,
        shuffle=False,
        sort_by=lambda x: -1 * np.array(x.max(axis=1).todense()).squeeze(),
        use_cuda=cuda,
    )

    # try to shuffle and sort at the same time, and expect a failure
    with pytest.raises(AssertionError):
        sorted_data_loader2 = DataLoader(
            d['matrix'],
            empty_drop_dataset=None,
            batch_size=5,
            fraction_empties=0.,
            shuffle=True,
            sort_by=lambda x: -1 * np.array(x.max(axis=1).todense()).squeeze(),
            use_cuda=cuda,
        )

    # this is copied from infer.BasePosterior._get_mean() which is not ideal
    out = []
    for loader in [data_loader, sorted_data_loader]:

        barcodes = []
        genes = []
        counts = []
        ind = 0

        for data in loader:
            dense_counts = data  # just make it the same!

            # Convert to sparse.
            bcs_i_chunk, genes_i, counts_i = dense_to_sparse_op_torch(dense_counts)

            # Barcode index in the dataloader.
            bcs_i = bcs_i_chunk + ind

            # Obtain the real barcode index after unsorting the dataloader.
            bcs_i = loader.unsort_inds(bcs_i)

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

        print('counts')
        print(counts)
        print('barcodes')
        print(barcodes)
        print('genes')
        print(genes)

        # Put the counts into a sparse csc_matrix.
        out.append(sp.csc_matrix((counts, (barcodes, genes)),
                                 shape=d['matrix'].shape))

    assert sparse_matrix_equal(out[0], out[1])
