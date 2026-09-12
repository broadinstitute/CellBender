"""Sparse count-matrix storage backends.

Two implementations share a common duck-typed interface:

    shape  -> (n_rows, n_cols)
    __len__ -> n_rows
    __getitem__(row_indices: np.ndarray) -> sp.csr_matrix

``InMemoryBackend`` wraps a scipy CSR matrix held entirely in RAM.
``MmapBackend`` stores the three CSR arrays as memory-mapped .npy files; only
the pages that are accessed get loaded into physical RAM, so the dataset never
has to fit in memory at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.sparse as sp


class InMemoryBackend:
    """Wraps a scipy CSR matrix; all data lives in RAM."""

    def __init__(self, matrix: sp.csr_matrix) -> None:
        self._matrix = matrix.tocsr()

    @property
    def shape(self) -> Tuple[int, int]:
        return self._matrix.shape

    def __len__(self) -> int:
        return self._matrix.shape[0]

    def __getitem__(self, row_indices: np.ndarray) -> sp.csr_matrix:
        return self._matrix[np.asarray(row_indices)]


class MmapBackend:
    """CSR matrix backed by three memory-mapped numpy arrays.

    Use ``MmapBackend.create(matrix, cache_dir)`` to convert a CSR matrix once
    and persist it; subsequent runs call ``MmapBackend(cache_dir)`` directly.

    Row access sorts the requested indices before hitting the mmap files
    (maximising sequential reads) then unshuffles the results, so random-access
    batches are served efficiently regardless of index order.
    """

    _DATA = "mmap_data.npy"
    _INDICES = "mmap_indices.npy"
    _INDPTR = "mmap_indptr.npy"
    _SHAPE = "mmap_shape.npy"

    def __init__(self, cache_dir: Path) -> None:
        d = Path(cache_dir)
        self._data = np.load(d / self._DATA, mmap_mode="r")
        self._indices = np.load(d / self._INDICES, mmap_mode="r")
        self._indptr = np.load(d / self._INDPTR, mmap_mode="r")
        shape_arr = np.load(d / self._SHAPE)
        self._shape: Tuple[int, int] = (int(shape_arr[0]), int(shape_arr[1]))

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, row_indices: np.ndarray) -> sp.csr_matrix:
        row_indices = np.asarray(row_indices)
        order = np.argsort(row_indices)
        sorted_rows = row_indices[order]

        starts = self._indptr[sorted_rows]
        ends = self._indptr[sorted_rows + 1]

        new_data = np.concatenate([self._data[s:e] for s, e in zip(starts, ends)])
        new_col_indices = np.concatenate([self._indices[s:e] for s, e in zip(starts, ends)])
        new_indptr = np.zeros(len(sorted_rows) + 1, dtype=np.int64)
        new_indptr[1:] = np.cumsum(ends - starts)

        sorted_matrix = sp.csr_matrix(
            (new_data, new_col_indices, new_indptr),
            shape=(len(sorted_rows), self._shape[1]),
        )

        # Restore original request order.
        unsort = np.argsort(order)
        return sorted_matrix[unsort]

    @classmethod
    def create(cls, matrix: sp.csr_matrix, cache_dir: Path) -> "MmapBackend":
        """Write CSR arrays to disk as .npy files and return a MmapBackend."""
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        csr = matrix.tocsr()
        np.save(d / cls._DATA, csr.data)
        np.save(d / cls._INDICES, csr.indices)
        np.save(d / cls._INDPTR, csr.indptr)
        np.save(d / cls._SHAPE, np.array(csr.shape))
        return cls(d)

    @classmethod
    def exists(cls, cache_dir: Path) -> bool:
        d = Path(cache_dir)
        return all((d / f).exists() for f in [cls._DATA, cls._INDICES, cls._INDPTR, cls._SHAPE])


class _SlicedBackend:
    """Read-only view into a backend that exposes a fixed subset of rows.

    Wraps any backend (InMemoryBackend or MmapBackend) and maps local indices
    0..len(indices)-1 to the global row positions given in ``indices``.  Used
    by the mmap path so that the mmap files cover the full matrix and
    train/test slicing happens lazily at access time.
    """

    def __init__(self, backend, indices: np.ndarray) -> None:
        self._backend = backend
        self._indices = np.asarray(indices, dtype=np.int64)

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self._indices), self._backend.shape[1])

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, row_indices: np.ndarray) -> sp.csr_matrix:
        return self._backend[self._indices[np.asarray(row_indices)]]
