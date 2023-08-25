"""Utility functions for working with scipy sparse matrices"""

import torch
import numpy as np
import scipy.sparse as sp

from typing import Optional, Tuple, Iterable


@torch.no_grad()
def dense_to_sparse_op_torch(t: torch.Tensor,
                             tensor_for_nonzeros: Optional[torch.Tensor] = None) \
        -> Tuple[torch.Tensor, ...]:
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
    nonzero_values = t[nonzero_inds_tuple].flatten().clone()

    return nonzero_inds_tuple + (nonzero_values,)


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


def csr_set_rows_to_zero(csr: sp.csr_matrix,
                         row_inds: Iterable[int]) -> sp.csr_matrix:
    """Set all nonzero elements in rows "row_inds" to zero.
    Happens in-place, although output is returned as well.

    https://stackoverflow.com/questions/12129948/scipy-sparse-set-row-to-zeros
    """

    if not isinstance(csr, sp.csr_matrix):
        try:
            csr = csr.tocsr()
        except Exception:
            raise ValueError('Matrix given must be of CSR format.')
    for row in row_inds:
        csr.data[csr.indptr[row]:csr.indptr[row + 1]] = 0
    csr.eliminate_zeros()
    return csr


def overwrite_matrix_with_columns_from_another(mat1: sp.csc_matrix,
                                               mat2: sp.csc_matrix,
                                               column_inds: np.ndarray) -> sp.csc_matrix:
    """Given two sparse matrices of the same shape, replace columns that are not
    in `column_inds` in `mat1` with the entries from `mat2`.
    """
    column_inds = set(column_inds)

    mat1 = mat1.copy().tocsr()
    mat2 = mat2.copy().tocsr()  # failure to copy could overwrite actual count data

    # Zero out values in mat2 that are in the specified columns.
    inds = np.where([i in column_inds for i in mat2.indices])[0]
    mat2.data[inds] = 0
    mat2.eliminate_zeros()

    # Zero out values in mat1 that are not in the specified columns.
    inds = np.where([i not in column_inds for i in mat1.indices])[0]
    mat1.data[inds] = 0
    mat1.eliminate_zeros()

    # Put in the new values by addition.
    output = mat1 + mat2

    return output.tocsc()
