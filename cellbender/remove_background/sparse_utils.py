"""Utility functions for working with scipy sparse matrices"""

import torch
import numpy as np
import scipy.sparse as sp

from typing import Optional, Tuple


@torch.no_grad()
def dense_to_sparse_op_torch(t: torch.Tensor,
                             tensor_for_nonzeros: Optional[torch.Tensor] = None) \
        -> Tuple[np.ndarray, ...]:
    """Converts dense matrix to sparse COO format tuple of numpy arrays (*indices, data)

    Args:
        t: The Tensor
        tensor_for_nonzeros: If this is not None, then this tensor will be used
            to determine nonzero indices, while the values will come from t.

    Returns:
        Tuple of
        (nonzero_inds_dim0, nonzero_inds_dim1, ...) + (nonzero_values, )

    """

    if tensor_for_nonzeros is None:
        tensor_for_nonzeros = t

    nonzero_inds_tuple = torch.nonzero(tensor_for_nonzeros, as_tuple=True)
    nonzero_values = t[nonzero_inds_tuple].flatten()

    return tuple([ten.cpu().numpy() for ten in (nonzero_inds_tuple + (nonzero_values,))])


def log_prob_sparse_to_dense(coo: sp.coo_matrix) -> np.ndarray:
    """Densify a sparse log prob COO data structure. Same as coo_matrix.todense()
    except it fills missing entries with -np.inf instead of 0, since 0 is a
    very meaningful quantity for log prob.
    """
    return todense_fill(coo=coo, fill_value=-np.inf)


def todense_fill(coo: sp.coo_matrix, fill_value: float) -> np.ndarray:
    """Densify a sparse COO matrix. Same as coo_matrix.todense()
    except it fills missing entries with fill_value instead of 0.
    """
    dummy_value = np.nan if not np.isnan(fill_value) else np.inf
    dummy_check = np.isnan if np.isnan(dummy_value) else np.isinf
    coo = coo.copy().astype(float)
    coo.data[coo.data == 0] = dummy_value
    out = np.array(coo.todense()).squeeze()
    out[out == 0] = fill_value
    out[dummy_check(out)] = 0
    return out


def zero_out_csr_rows(csr: sp.csr_matrix,
                      row_logic: np.ndarray) -> sp.csr_matrix:
    """Given a sparse matrix, set specified rows to zero.
    """
    row_inds = set(np.where(row_logic)[0])
    coo = csr.copy().tocoo()

    # Zero out values that are in the specified rows.
    logic = np.array([i in row_inds for i in coo.row])
    coo.data[logic] = 0
    coo.eliminate_zeros()

    return coo.tocsr()
