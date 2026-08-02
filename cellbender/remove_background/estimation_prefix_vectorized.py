"""Vectorized replacements for the *shared prefix* helpers in ``estimation.py``.

Benchmarking the MCKP kernel showed that once the pandas groupby/apply work is
gone, the remaining cost is dominated by three helpers that every estimator
shares.  Line-profiled on the ``heavy`` scenario (54.9M COO entries, 4.98M
m-values), the ~26 s prefix broke down as:

    apply_function_dense_chunks                                23.28 s
      chunked_iterator                                          14.93 s
        np.unique(coo.row)                                        4.90 s
        np.unique(coo.col[logic], return_inverse=True)            4.19 s
        pd.Series(coo.row).isin(set(row_m_values))                2.47 s
        np.unique(coo.row[logic], return_inverse=True)            2.43 s
      len(np.unique(noise_log_prob_coo.row))   <- 2nd full unique  4.95 s
      densify + torch.tensor                                     3.15 s
    _estimation_array_to_csr                                    2.84 s
      [noise_offsets.get(i, 0) for i in m]                        2.10 s

Three separate inefficiencies, all fixed here:

1. ``coo.row`` is passed through ``np.unique`` **three times** per call (once for
   ``array_length``, once for ``unique_m_values``, once per chunk for the compact
   row indices).  We do it once, with ``return_inverse``, and derive everything
   else.  Because ``np.array_split`` hands out *contiguous* slices of the sorted
   unique values, chunk *k*'s compact row index is just
   ``global_inverse - (index of chunk k's first unique value)`` -- no per-chunk
   unique needed at all.

2. ``pd.Series(coo.row).isin(set(row_m_values))`` re-scans the entire ``coo.row``
   once per chunk and builds a Python ``set`` of up to millions of numpy scalars
   each time: O(n_chunks x nnz) where O(nnz) suffices.  Since the chunks are
   contiguous ranges of sorted m-values, assigning every nonzero to its owning
   chunk is a single ``np.searchsorted`` against the chunk start boundaries.

3. ``np.unique(coo.col, return_inverse=True)`` is gone entirely -- not optimized,
   *deleted*.  Compacting the 'c' axis was a genuine correctness bug (the caller
   throws the column mapping away and then reads a compacted column position as
   if it were a true noise count); see the NOTE in ``estimation.chunked_iterator``.
   Both implementations now densify to the full ``coo.shape[1]`` width, which the
   memory budget was already sized for, so the fastest way to handle the 'c' axis
   turned out to be not to touch it.

4. ``[noise_offsets.get(i, 0) for i in m]`` is a Python-level dict lookup per
   m-value.  Replaced with sorted-key ``searchsorted`` (the same pattern as
   ``_map_value_per_entry`` in ``estimation_vectorized``, but defaulting to 0 on
   a miss instead of raising).  The sorted-key arrays are built ONCE per
   ``estimate_noise`` call and reused across chunks, since the dict never
   changes.

Relationship to ``estimation.py``
---------------------------------
These were originally purely additive, with ``estimation.py`` left pristine.
That is no longer true: inefficiencies (1) and (2) and the 'c'-axis bug (3) have
since been fixed in ``estimation.chunked_iterator`` /
``estimation.apply_function_dense_chunks`` themselves, because two hardcoded
call sites in ``posterior.py`` (``compute_mean_target_removal_as_function`` ->
``Mean``, and ``PRmu._binary_search_for_posterior_regularization_factor`` ->
``MAP``, up to ``max_iterations`` times) reach the originals regardless of which
``--estimator`` the user picked, ``mckp-fast`` included.  The two copies are kept
separate so that ``tests/test_estimation_prefix_vectorized.py`` can keep checking
them against each other *and* against an independent naive reference; they must
stay in lockstep.

Order preservation
------------------
The pre-optimization implementation selected chunk members with a boolean mask,
so the original relative order of COO entries is preserved within each chunk.  We
partition with ``np.argsort(chunk_index, kind="stable")``, which reproduces that
order exactly, so each yielded chunk is identical triple-for-triple -- not merely
equivalent as a set.  (``apply_function_dense_chunks`` turns out not to depend on
within-chunk order, because it keys results off the sorted ``unique_row_values``;
but ``chunked_iterator`` is a public-looking helper, and exact identity is
testable, so we preserve it rather than rely on that.)
"""

from typing import Callable, Dict, Generator, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from cellbender.remove_background.estimation import (
    COUNT_DATATYPE,
    MAP,
    N_CELLS_DATATYPE,
    N_GENES_DATATYPE,
    MultipleChoiceKnapsack,
)
from cellbender.remove_background.estimation_vectorized import chunk_estimate_noise_vectorized
from cellbender.remove_background.sparse_utils import log_prob_sparse_to_dense

__all__ = [
    "NoiseOffsetLookup",
    "chunked_iterator_vectorized",
    "apply_function_dense_chunks_vectorized",
    "estimation_array_to_csr_vectorized",
    "gene_chunk_logic_vectorized",
    "MultipleChoiceKnapsackFast",
]


def chunked_iterator_vectorized(
    coo: sp.coo_matrix, max_dense_batch_size_GB: float = 1.0
) -> Generator[Tuple[sp.coo_matrix, np.ndarray, np.ndarray], None, None]:
    """Drop-in replacement for ``estimation.chunked_iterator``.

    Yields ``(chunk_coo, unique_row_values, unique_col_values)`` identically to
    the original, including within-chunk entry order and including the
    degenerate empty-COO case.

    Like the original, the 'c' axis is NOT compacted: chunk column indices are
    true noise counts and the chunk width is ``coo.shape[1]``. This module used to
    replicate the original's c-axis compaction deliberately, for bit-exact parity
    even where the original was buggy; that bug is now fixed in both places, so
    there is nothing left to replicate. (The bincount+LUT helper that made the
    compaction cheap is gone with it -- not compacting is cheaper still.)
    """
    # --- identical batch/chunk arithmetic to the original ---
    n_elements_in_batch = max_dense_batch_size_GB * 1e9 / 4  # torch float32 is 4 bytes
    batch_size = max(1, int(np.floor(n_elements_in_batch / coo.shape[1])))

    # ONE unique over coo.row, with the inverse, reused for everything.
    unique_m_values, inverse = np.unique(coo.row, return_inverse=True)
    inverse = inverse.ravel()  # numpy>=2 may return a shaped inverse

    n_chunks = max(1, len(unique_m_values) // batch_size)
    # Contiguous slices of the sorted uniques -- exactly what np.array_split gives.
    split_points = np.array_split(np.arange(len(unique_m_values)), n_chunks)

    # The true noise count values spanned by every chunk (the axis is not compacted).
    all_col_values = np.arange(coo.shape[1])

    if len(unique_m_values) == 0:
        # Pre-optimization original: isin over an empty set -> all-False mask ->
        # empty chunk_coo. Width is coo.shape[1] now that 'c' is not compacted.
        empty = np.zeros(0, dtype=coo.row.dtype)
        yield (
            sp.coo_matrix(
                (
                    np.zeros(0, dtype=coo.data.dtype),
                    (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)),
                ),
                shape=(0, coo.shape[1]),
            ),
            empty,
            all_col_values,
        )
        return

    # Boundaries: the first unique-value index owned by each chunk. (np.array_split
    # can only emit empty pieces when n_chunks > len(unique_m_values), impossible
    # here since batch_size >= 1; the filter is defensive, and split_points is
    # filtered the same way so the two stay index-aligned.)
    split_points = [s for s in split_points if s.size > 0]
    chunk_first_unique_idx = np.array([s[0] for s in split_points], dtype=np.int64)
    n_real_chunks = chunk_first_unique_idx.size

    if n_real_chunks == 1:
        # Fast path (the common case at production scale: batch_size is ~12.5M
        # m-values for a 20-column posterior, and real chunks have <10M).
        chunk_order_slices = [slice(0, inverse.size)]
        order = None
        chunk_idx_of_entry = None
    else:
        # Assign every nonzero to its owning chunk in one vectorized pass.
        # inverse is the index into unique_m_values, so this is a threshold
        # assignment on the unique-index, no searchsorted over values needed.
        chunk_idx_of_entry = (
            np.searchsorted(chunk_first_unique_idx, inverse, side="right").astype(np.int64) - 1
        )
        order = np.argsort(chunk_idx_of_entry, kind="stable")  # stable => original order kept
        counts = np.bincount(chunk_idx_of_entry, minlength=n_real_chunks)
        edges = np.zeros(n_real_chunks + 1, dtype=np.int64)
        np.cumsum(counts, out=edges[1:])
        chunk_order_slices = [slice(int(edges[k]), int(edges[k + 1])) for k in range(n_real_chunks)]

    for k in range(n_real_chunks):
        uniq_slice = split_points[k]
        lo_unique = int(uniq_slice[0])
        unique_row_values = unique_m_values[uniq_slice]

        sl = chunk_order_slices[k]
        if order is None:
            sel_data = coo.data
            sel_inverse = inverse
            sel_col = coo.col
        else:
            idx = order[sl]
            sel_data = coo.data[idx]
            sel_inverse = inverse[idx]
            sel_col = coo.col[idx]

        # Compact row index within this chunk, without any per-chunk np.unique.
        rows = sel_inverse - lo_unique

        chunk_coo = sp.coo_matrix(
            (sel_data, (rows, sel_col)),
            shape=(len(unique_row_values), coo.shape[1]),
        )
        yield (chunk_coo, unique_row_values, all_col_values)


def apply_function_dense_chunks_vectorized(
    noise_log_prob_coo: sp.coo_matrix, fun: Callable[..., torch.Tensor], device: str = "cpu", **kwargs
) -> Dict[str, np.ndarray]:
    """Drop-in replacement for ``estimation.apply_function_dense_chunks``.

    Identical output. Avoids preallocating via a THIRD full ``np.unique(coo.row)``
    by accumulating into lists and concatenating -- same result, one fewer
    O(n log n) sort over all nnz. (``estimation.apply_function_dense_chunks`` now
    does the same thing; this stays as the ``chunked_iterator_vectorized``-based
    twin so the two can be checked against each other.)
    """
    m_parts = []
    out_parts = []
    for coo, row, _col in chunked_iterator_vectorized(coo=noise_log_prob_coo):
        dense_tensor = torch.tensor(log_prob_sparse_to_dense(coo)).to(device)
        if torch.numel(dense_tensor) == 0:
            # github issue 207
            continue
        s = fun(dense_tensor, **kwargs)
        if s.ndim == 0:
            # avoid "TypeError: len() of a 0-d tensor"
            s_np = np.atleast_1d(s.detach().cpu().numpy())
        else:
            s_np = s.detach().cpu().numpy()
        if s_np.shape[0] != row.shape[0]:
            # Same guard as estimation.apply_function_dense_chunks; see the comment
            # there. Keeps the two in lockstep, including on the failure path.
            raise ValueError(
                f"{getattr(fun, '__name__', fun)} produced {s_np.shape[0]} value(s) for a chunk of "
                f"{row.shape[0]} row(s) (dense chunk shape {tuple(dense_tensor.shape)}). "
                "This function requires one value per row of the dense chunk."
            )
        m_parts.append(np.asarray(row, dtype=np.uint64))
        out_parts.append(s_np)

    if not m_parts:
        # Original returns zero-filled arrays of length len(unique(row)); with
        # nothing processed that length is 0 unless every chunk was empty, in
        # which case the original also leaves zeros. Match the empty case.
        n = len(np.unique(noise_log_prob_coo.row))
        return {"m": np.zeros(n, dtype=np.uint64), "result": np.zeros(n)}

    m = np.concatenate(m_parts).astype(np.uint64, copy=False)
    out = np.concatenate(out_parts).astype(np.float64, copy=False)
    return {"m": m, "result": out}


class NoiseOffsetLookup:
    """Sorted-array form of the ``noise_offsets`` dict for vectorized lookup.

    Built ONCE per ``estimate_noise`` call and reused across gene chunks, since
    the dict is constant for the whole call. Replaces::

        data = data + np.array([noise_offsets.get(i, 0) for i in m])

    A strong reference to the source dict is retained so that identity-based
    cache validation (``lookup.matches(d)``) can never be fooled by ``id()``
    reuse after garbage collection.
    """

    __slots__ = ("_src", "keys", "vals")

    def __init__(self, noise_offsets: Optional[Dict[int, int]]):
        self._src = noise_offsets
        if not noise_offsets:
            self.keys = None
            self.vals = None
            return
        n = len(noise_offsets)
        keys = np.fromiter(noise_offsets.keys(), dtype=np.uint64, count=n)
        # int32 rather than int16: noise offsets are poisson_values_low, which
        # scales with ambient counts per droplet. Issue #248 describes 600+
        # ambient counts per drop; int16 would cap at 32767, which is plausible
        # to exceed in a pathological sample, and this array is only
        # len(nonzero offsets) long so the width costs nothing.
        vals = np.fromiter(noise_offsets.values(), dtype=np.int64, count=n)
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.vals = vals[order]

    def matches(self, noise_offsets) -> bool:
        return noise_offsets is self._src

    def offsets_for(self, m: np.ndarray) -> np.ndarray:
        """Return the offset for each entry of ``m``, 0 where absent."""
        m_u = np.asarray(m, dtype=np.uint64)
        if self.keys is None:
            return np.zeros(m_u.size, dtype=np.int64)
        idx = np.searchsorted(self.keys, m_u, side="left")
        idx_clipped = np.where(idx >= self.keys.size, 0, idx)
        hit = self.keys[idx_clipped] == m_u
        return np.where(hit, self.vals[idx_clipped], np.int64(0))


def estimation_array_to_csr_vectorized(
    index_converter,
    data: np.ndarray,
    m: np.ndarray,
    noise_offsets: Optional[Dict[int, int]],
    dtype=COUNT_DATATYPE,
    offset_lookup: Optional[NoiseOffsetLookup] = None,
) -> sp.csr_matrix:
    """Drop-in replacement for ``estimation._estimation_array_to_csr``.

    ``offset_lookup`` lets a caller hoist the dict->sorted-array conversion out
    of a per-chunk loop. If omitted it is built on the fly (still vectorized,
    just not amortized).
    """
    row, col = index_converter.get_ng_indices(m_inds=np.asarray(m))
    if noise_offsets is not None:
        if offset_lookup is None or not offset_lookup.matches(noise_offsets):
            offset_lookup = NoiseOffsetLookup(noise_offsets)
        data = np.asarray(data) + offset_lookup.offsets_for(m)
    coo = sp.coo_matrix(
        (np.asarray(data).astype(dtype), (row.astype(N_CELLS_DATATYPE), col.astype(N_GENES_DATATYPE))),
        shape=index_converter.matrix_shape,
        dtype=dtype,
    )
    coo.sum_duplicates()
    return coo.tocsr()


def gene_chunk_logic_vectorized(index_converter, noise_log_prob_coo: sp.coo_matrix, n_chunks: int):
    """Drop-in replacement for ``MultipleChoiceKnapsack._gene_chunk_iterator``.

    The original is a third instance of the same O(n_chunks x nnz) pandas
    ``isin`` anti-pattern::

        genes_series = pd.Series(genes)
        gene_chunk_arrays = np.array_split(np.arange(total_n_genes), n_chunks)
        return [genes_series.isin(x).values for x in gene_chunk_arrays]

    ``np.array_split`` yields contiguous gene-index ranges, so membership is a
    simple comparison against each range's bounds -- one pass, no sets.
    """
    _, genes = index_converter.get_ng_indices(m_inds=noise_log_prob_coo.row)
    gene_chunk_arrays = np.array_split(np.arange(index_converter.total_n_genes), n_chunks)
    out = []
    for chunk in gene_chunk_arrays:
        if chunk.size == 0:
            out.append(np.zeros(genes.size, dtype=bool))
            continue
        lo = int(chunk[0])
        hi = int(chunk[-1])
        out.append((genes >= lo) & (genes <= hi))
    return out


class MultipleChoiceKnapsackFast(MultipleChoiceKnapsack):
    """MCKP with BOTH the pandas kernel and the shared prefix vectorized.

    Inherits ``estimate_noise`` unchanged; only the helpers it reaches are
    swapped, plus the per-call ``NoiseOffsetLookup`` cache.
    """

    def __init__(self, index_converter):
        super().__init__(index_converter=index_converter)
        self._offset_lookup: Optional[NoiseOffsetLookup] = None

    def _lookup_for(self, noise_offsets) -> NoiseOffsetLookup:
        if self._offset_lookup is None or not self._offset_lookup.matches(noise_offsets):
            self._offset_lookup = NoiseOffsetLookup(noise_offsets)
        return self._offset_lookup

    def _gene_chunk_iterator(self, noise_log_prob_coo: sp.coo_matrix, n_chunks: int):
        return gene_chunk_logic_vectorized(
            index_converter=self.index_converter,
            noise_log_prob_coo=noise_log_prob_coo,
            n_chunks=n_chunks,
        )

    def _chunk_estimate_noise(
        self,
        noise_log_prob_coo: sp.coo_matrix,
        noise_offsets: Optional[Dict[int, int]],
        noise_targets_per_gene: np.ndarray,
        verbose: bool = False,
    ) -> sp.csr_matrix:
        return chunk_estimate_noise_vectorized(
            index_converter=self.index_converter,
            noise_log_prob_coo=noise_log_prob_coo,
            noise_offsets=noise_offsets,
            noise_targets_per_gene=noise_targets_per_gene,
            verbose=verbose,
            map_fun=apply_function_dense_chunks_vectorized,
            array_to_csr_fun=estimation_array_to_csr_vectorized,
            offset_lookup=self._lookup_for(noise_offsets),
        )


def map_prefix_vectorized(index_converter, coo, noise_offsets, offset_lookup=None):
    """The shared MAP prefix, fully vectorized. Exposed for benchmarking."""
    md = apply_function_dense_chunks_vectorized(
        noise_log_prob_coo=coo, fun=MAP.torch_argmax, device="cpu"
    )
    return estimation_array_to_csr_vectorized(
        index_converter=index_converter,
        data=md["result"],
        m=md["m"],
        noise_offsets=noise_offsets,
        offset_lookup=offset_lookup,
    )
