"""Vectorized (pure numpy/scipy, zero pandas) drop-in replacement for
``MultipleChoiceKnapsack._chunk_estimate_noise``.

This module is a *research prototype* kept deliberately separate from
``estimation.py`` so that the original pandas implementation stays untouched and
both can be benchmarked side by side against identical inputs.

Design contract
---------------
``chunk_estimate_noise_vectorized`` is intended to be **bit-exact** with
``MultipleChoiceKnapsack._chunk_estimate_noise``, including its tie-breaking
behaviour and including two quirks of the original that a "clean rewrite" would
accidentally change:

1. The per-direction ``delta`` is a *global* ``Series.diff()`` over the whole
   ``(m, c)``-sorted sub-frame, **not** a per-``m`` (per-droplet) diff.  The
   first row of each ``m`` group therefore initially carries a delta computed
   against the *previous* droplet's last row.  The original relies on the
   subsequent ``delta = NaN`` assignment at ``c == map`` to wipe exactly those
   rows out (for positive-step genes the MAP row is the first kept row of each
   ``m`` group; for negative-step genes it is the last).  We reproduce the
   global diff + NaN sentinel literally rather than "fixing" it into a grouped
   diff, because if the MAP row happens to be absent from the kept rows for some
   ``m`` (see the note on ``apply_function_dense_chunks`` below) the original
   leaks a cross-droplet delta, and we must leak the same one.

2. ``pandas.DataFrame.groupby("g").apply(lambda x: x.nsmallest(k, "delta"))``
   picks, within each gene, the ``k`` smallest deltas with ``keep="first"``.
   Reading ``pandas.core.methods.selectn.SelectNSeries.compute`` /
   ``SelectNFrame.compute`` (verified against pandas 3.0.3): for a single sort
   column, ``SelectNFrame`` delegates entirely to ``Series.nsmallest``, which is
   ``arr[arr <= kth_val].argsort(kind="stable")[:n]`` in the general case and
   ``sort_values(kind="stable").head(n)`` when ``n >= len(group)``.  Both are
   exactly "stable sort by value, take the first n", i.e. ties are resolved by
   *position within the group as the group was laid out when nsmallest was
   called*.  That layout is ``(m, c)`` ascending (the frame was
   ``sort_values(by=["m", "c"])``-ed, then split into a positive block and a
   negative block, then concatenated; a gene is exclusively positive-step or
   negative-step, never both, so a gene's rows are contiguous within one block
   and remain in ``(m, c)`` order).  We therefore reproduce it as: stable
   ``lexsort`` on ``(delta, g)`` over the ``(m, c)``-sorted rows, then a
   rank-within-gene ``< topk`` cutoff.

Note on the shared MAP prefix
-----------------------------
The initial MAP estimate is computed by calling the *same*
``apply_function_dense_chunks`` / ``MAP.torch_argmax`` / ``_estimation_array_to_csr``
helpers the original uses, unchanged.  That is deliberate: it keeps the two
implementations bit-identical even where the original is arguably buggy
(``chunked_iterator`` compacts the ``c`` axis with ``np.unique`` before the
argmax, so ``map_dict["result"]`` is an index into the *compacted* column space
rather than a true noise-count value ``c`` whenever a chunk's occupied columns
are not ``0..K-1``).  Replicating instead of repairing is the point here.
"""

from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

from cellbender.remove_background.estimation import (
    MAP,
    MultipleChoiceKnapsack,
    _estimation_array_to_csr,
    apply_function_dense_chunks,
)

__all__ = [
    "chunk_estimate_noise_vectorized",
    "MultipleChoiceKnapsackVectorized",
    "stable_topk_positions_per_group",
]

# Narrowed working dtypes. See the "Integer/float widths" section of
# chunk_estimate_noise_vectorized's docstring for the bounds justifying each.
_C_DATATYPE = np.int16  # noise count index 'c', bounded by n_counts_max (<=100)
_GENE_DATATYPE = np.int32  # gene index 'g'; int16 overflows a 33538-gene reference
_RANK_DATATYPE = np.int32  # per-group ranks / group starts; bounded by nnz < 2**31


def _narrow_map_values(map_value: np.ndarray, n_c: int) -> np.ndarray:
    """Narrow float-stored MAP argmax indices to _C_DATATYPE, exactly.

    ``apply_function_dense_chunks`` stores its result in a float64 array even
    though ``MAP.torch_argmax`` produces integers. Casting back is exact, but we
    refuse to do it if the values are not in fact integral and in range, so a
    future non-argmax caller cannot be silently truncated.
    """
    if map_value.size == 0:
        return map_value.astype(_C_DATATYPE, copy=False)
    if np.issubdtype(map_value.dtype, np.integer):
        narrowed = map_value
    else:
        narrowed = map_value.astype(_C_DATATYPE)
        if not np.array_equal(narrowed.astype(map_value.dtype), map_value):
            # Non-integral MAP values: keep full precision rather than truncate.
            return map_value
    lo = int(np.min(narrowed))
    hi = int(np.max(narrowed))
    if lo < np.iinfo(_C_DATATYPE).min or hi > np.iinfo(_C_DATATYPE).max:
        return map_value
    return narrowed.astype(_C_DATATYPE, copy=False)


def _map_value_per_entry(map_m: np.ndarray, map_result: np.ndarray, m_rows: np.ndarray) -> np.ndarray:
    """Vectorized replacement for::

        lookup_map_from_m = dict(zip(map_dict['m'], map_dict['result']))
        df['map'] = df['m'].apply(lambda x: lookup_map_from_m[x])

    ``map_m`` holds unique 'm' values (they come from ``np.unique`` inside
    ``chunked_iterator``), so a sorted-search lookup is exact.

    Raises:
        KeyError: if any entry of ``m_rows`` is absent from ``map_m``, matching
            the ``KeyError`` the original dict lookup would raise.
    """
    # Cast both sides to uint64. 'm' indices are non-negative by construction
    # (IndexConverter.get_ng_indices asserts m >= 0) but coo.row may be int32 /
    # int64 while map_dict['m'] is uint64; mixing those in searchsorted would
    # make numpy fall back to float64 and lose precision above 2**53.
    haystack = np.ascontiguousarray(map_m, dtype=np.uint64)
    needles = np.ascontiguousarray(m_rows, dtype=np.uint64)

    order = np.argsort(haystack, kind="stable")
    sorted_m = haystack[order]
    idx = np.searchsorted(sorted_m, needles, side="left")
    if needles.size:
        # searchsorted can return len(sorted_m) for values beyond the end
        oob = idx >= sorted_m.size
        idx_clipped = np.where(oob, 0, idx)
        missing = oob | (sorted_m[idx_clipped] != needles)
        if missing.any():
            raise KeyError(int(needles[missing][0]))
    return map_result[order[idx]]


def _group_start_positions(sorted_group_keys: np.ndarray) -> np.ndarray:
    """Given group keys already sorted ascending, return for each position the
    index at which its group begins. O(N), no searchsorted.
    """
    n = sorted_group_keys.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    np.not_equal(sorted_group_keys[1:], sorted_group_keys[:-1], out=boundary[1:])
    starts = np.flatnonzero(boundary).astype(_RANK_DATATYPE, copy=False)
    sizes = np.diff(np.append(starts, _RANK_DATATYPE(n)))
    return np.repeat(starts, sizes)


def stable_topk_positions_per_group(
    group: np.ndarray, value: np.ndarray, topk_per_group: np.ndarray
) -> np.ndarray:
    """Exact vectorized equivalent of::

        df.groupby('g', group_keys=False).apply(
            lambda x: x.nsmallest(x['topk'].iat[0], columns='delta'))

    ...in terms of *which rows are selected*.

    Args:
        group: Group key per row (here: gene index 'g'). Must be usable as an
            index into ``topk_per_group``.
        value: Value to rank ascending per group (here: 'delta'). Must contain
            no NaN (the caller filters non-finite deltas first, exactly as the
            original does).
        topk_per_group: How many rows to keep, indexed by group key (here:
            ``abs_additional_noise_counts_per_gene``).

    Returns:
        Positions (into ``group``/``value``) of the selected rows. Ties in
        ``value`` are broken by *ascending original position*, which is what
        pandas' ``keep='first'`` does.
    """
    n = group.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    # np.lexsort: last key is primary. It is documented as an *indirect stable*
    # sort, so rows tied on (group, value) keep their input order -- which is
    # exactly pandas' keep='first' tie-break. (Verified empirically against an
    # explicit arange tie-break key in the test script.)
    if n > np.iinfo(_RANK_DATATYPE).max:
        raise ValueError(f"{n} rows exceeds the {_RANK_DATATYPE.__name__} rank width")
    order = np.lexsort((value, group))
    g_sorted = group[order]
    rank = np.arange(n, dtype=_RANK_DATATYPE) - _group_start_positions(g_sorted)
    keep = rank < np.asarray(topk_per_group)[g_sorted]
    return order[keep]


def chunk_estimate_noise_vectorized(
    index_converter,
    noise_log_prob_coo: sp.coo_matrix,
    noise_offsets: Optional[Dict[int, int]],
    noise_targets_per_gene: np.ndarray,
    verbose: bool = False,
    map_fun=None,
    array_to_csr_fun=None,
    offset_lookup=None,
) -> sp.csr_matrix:
    """Pandas-free equivalent of ``MultipleChoiceKnapsack._chunk_estimate_noise``.

    Args:
        index_converter: The ``IndexConverter`` (m <-> (n, g)).
        noise_log_prob_coo: Noise log prob COO for one gene chunk, (m, c).
        noise_offsets: Noise count offsets keyed by 'm'.
        noise_targets_per_gene: Integer noise count target per gene, length
            ``index_converter.total_n_genes``.
        verbose: Accepted for signature compatibility. The original prints
            intermediate DataFrames; here it prints a few array summaries.
        map_fun: Optional replacement for ``apply_function_dense_chunks``
            (see ``estimation_prefix_vectorized``). Defaults to the original.
        array_to_csr_fun: Optional replacement for ``_estimation_array_to_csr``.
            Must accept the same keywords plus ``offset_lookup``.
        offset_lookup: Optional prebuilt ``NoiseOffsetLookup``, hoisted out of
            the per-chunk loop by the caller.

    Integer/float widths
    --------------------
    In a real run the incoming COO is ``data=float32`` / ``row=int64`` /
    ``col=int64`` (traced through ``Posterior._get_cell_noise_count_posterior_coo``:
    log probs come from ``.float()`` torch tensors, and the COO shape
    ``total_n_cells * total_n_genes`` forces 64-bit indices).  We narrow the
    per-entry working arrays to the smallest width that is provably exact:

      * ``delta`` and the diff temporaries -> ``promote_types(log_prob.dtype, float32)``.
        This *matches what pandas actually produced*: ``Series.diff()`` preserves
        float32 and performs the subtraction in float32 (verified: it disagrees
        with exact float64 arithmetic on 54% of random float32 pairs, and agrees
        with float32 arithmetic on 100%).  Note the arithmetic width is set by
        ``log_prob``, not by the output array, so this narrowing changes storage
        only -- widening float32 to float64 is exact and order-preserving, so the
        tie-break ranking is identical either way.
      * ``c`` (noise count index) -> int16. Bounded by ``n_counts_max``, which is
        20 in the default posterior and at most 100 anywhere in ``posterior.py``.
      * ``g`` (gene index) -> int32, NOT int16: a standard human CellRanger
        reference has 33538-36601 features, which overflows int16's 32767.
      * ``map_value`` -> int16. ``MAP.torch_argmax`` yields exact small integers
        that ``apply_function_dense_chunks`` happens to store in a float64 array;
        casting back is exact, and it makes ``c > map_value`` an integer compare
        instead of materializing a float64 temporary per entry.
      * ``m`` stays int64/uint64: ``total_n_cells * total_n_genes`` reaches ~2e10
        for a 700k-barcode raw matrix, far past int32.

    Returns:
        Estimated noise count matrix for this chunk, CSR, shape
        ``index_converter.matrix_shape``, dtype ``COUNT_DATATYPE`` (int32)
        promoted only by the final ``map_csr + steps_csr`` addition.
    """
    assert noise_targets_per_gene.size == index_converter.total_n_genes, (
        f"The number of noise count targets ({noise_targets_per_gene.size}) "
        f"must match the number of genes ({index_converter.total_n_genes})"
    )

    # ---------------------------------------------------------------- step 1
    # Initial MAP estimate. Same helpers as the original unless the caller
    # injects the vectorized prefix replacements.
    _map_fun = map_fun if map_fun is not None else apply_function_dense_chunks
    map_dict = _map_fun(noise_log_prob_coo=noise_log_prob_coo, fun=MAP.torch_argmax, device="cpu")
    if array_to_csr_fun is None:
        map_csr = _estimation_array_to_csr(
            index_converter=index_converter,
            data=map_dict["result"],
            m=map_dict["m"],
            noise_offsets=noise_offsets,
        )
    else:
        map_csr = array_to_csr_fun(
            index_converter=index_converter,
            data=map_dict["result"],
            m=map_dict["m"],
            noise_offsets=noise_offsets,
            offset_lookup=offset_lookup,
        )
    map_noise_counts_per_gene = np.array(map_csr.sum(axis=0)).squeeze()
    additional_noise_counts_per_gene = (noise_targets_per_gene - map_noise_counts_per_gene).astype(int)
    abs_additional_noise_counts_per_gene = np.abs(additional_noise_counts_per_gene)

    # ---------------------------------------------------------------- step 2
    # Step direction per COO entry, from the gene it belongs to.
    # (Replaces df['g'].apply(lambda gene: gene in set_positive_genes) etc.)
    m_all = noise_log_prob_coo.row
    # c is bounded by n_counts_max (<=20 default, <=100 anywhere in posterior.py)
    c_all = noise_log_prob_coo.col.astype(_C_DATATYPE, copy=False)
    lp_all = noise_log_prob_coo.data
    # g is bounded by total_n_genes; int32 (int16 would overflow a 33538-gene ref)
    _, g_all = index_converter.get_ng_indices(m_inds=m_all)
    g_all = g_all.astype(_GENE_DATATYPE, copy=False)

    direction_per_gene = np.zeros(additional_noise_counts_per_gene.shape, dtype=np.int8)
    direction_per_gene[additional_noise_counts_per_gene > 0] = 1
    direction_per_gene[additional_noise_counts_per_gene < 0] = -1

    step_direction = direction_per_gene[g_all]

    # Remove all 'm' entries corresponding to genes where target is met by MAP.
    keep = step_direction != 0
    if not keep.any():
        # Original: df becomes empty -> both step frames empty -> len(df) == 0.
        return map_csr

    m = m_all[keep]
    c = c_all[keep]
    log_prob = lp_all[keep]
    g = g_all[keep]
    step_direction = step_direction[keep]

    # ---------------------------------------------------------------- step 3
    # Mask (and drop) log probs that represent steps in the wrong direction.
    # Note the original's `df.loc[df['mask'], 'log_prob'] = -np.inf` is a no-op
    # because the masked rows are dropped on the next line; we skip it.
    # MAP.torch_argmax produces exact small integers that
    # apply_function_dense_chunks stores in a float64 array; narrowing to int16
    # is exact and makes the two comparisons below integer compares rather than
    # float64 ones. Guarded so a non-integral 'result' cannot be silently truncated.
    map_value = _map_value_per_entry(map_dict["m"], map_dict["result"], m)
    map_value = _narrow_map_values(map_value, n_c=noise_log_prob_coo.shape[1])
    positive = step_direction > 0
    mask = ((~positive) & (c > map_value)) | (positive & (c < map_value))
    keep = ~mask
    if not keep.any():
        return map_csr
    m = m[keep]
    c = c[keep]
    log_prob = log_prob[keep]
    g = g[keep]
    step_direction = step_direction[keep]
    map_value = map_value[keep]
    positive = positive[keep]

    # ---------------------------------------------------------------- step 4
    # Sort by ('m', 'c'). np.lexsort's last key is primary -> m then c.
    # pandas' sort_values(by=['m','c']) is likewise a stable lexsort.
    order = np.lexsort((c, m))
    m = m[order]
    c = c[order]
    log_prob = log_prob[order]
    g = g[order]
    step_direction = step_direction[order]
    map_value = map_value[order]
    positive = positive[order]

    # ---------------------------------------------------------------- step 5
    # Deltas: |diff| within each direction's sub-block, global (not grouped),
    # then NaN at the MAP row. See module docstring, quirk (1).
    # Match the dtype pandas' Series.diff() would have produced (it preserves
    # float32 and subtracts in float32). Note the *arithmetic* width is set by
    # log_prob regardless of this array's dtype, since `lp[1:] - lp[:-1]` is
    # evaluated on the float32 slices before the ufunc writes into `d`.
    delta_dtype = np.promote_types(log_prob.dtype, np.float32)
    delta = np.full(m.size, np.nan, dtype=delta_dtype)

    pos_pos = np.flatnonzero(positive)
    if pos_pos.size > 0:
        lp = log_prob[pos_pos]
        d = np.empty(lp.size, dtype=delta_dtype)
        d[0] = np.nan  # pandas .diff(periods=1) leaves the first row NaN
        if lp.size > 1:
            np.abs(lp[1:] - lp[:-1], out=d[1:])
        d[c[pos_pos] == map_value[pos_pos]] = np.nan
        delta[pos_pos] = d

    neg_pos = np.flatnonzero(~positive)
    if neg_pos.size > 0:
        lp = log_prob[neg_pos]
        d = np.empty(lp.size, dtype=delta_dtype)
        d[-1] = np.nan  # pandas .diff(periods=-1) leaves the last row NaN
        if lp.size > 1:
            np.abs(lp[:-1] - lp[1:], out=d[:-1])
        d[c[neg_pos] == map_value[neg_pos]] = np.nan
        delta[neg_pos] = d

    # Remove irrelevant entries: those with non-finite delta (NaN sentinel at
    # the MAP row, plus any +-inf).
    finite = np.isfinite(delta)
    if not finite.any():
        return map_csr
    m = m[finite]
    g = g[finite]
    delta = delta[finite]
    # c / log_prob / map_value / positive / step_direction are all dead from here
    # on (step 7 recovers the direction from the gene index instead). Dropping
    # them before the lexsort in step 6 removes ~5 per-entry arrays from the peak.
    del c, log_prob, map_value, positive, step_direction, finite, order, keep

    if verbose:
        print(f"[vectorized] rows entering top-k: {m.size}")

    # ---------------------------------------------------------------- step 6
    # Per-gene k smallest deltas, pandas keep='first' tie-break. Quirk (2).
    selected = stable_topk_positions_per_group(
        group=g, value=delta, topk_per_group=abs_additional_noise_counts_per_gene
    )
    if selected.size == 0:
        return map_csr

    # ---------------------------------------------------------------- step 7
    # Steps per 'm' = how many times each 'm' was selected, signed by direction.
    selected_m = m[selected]
    unique_m, inverse = np.unique(selected_m, return_inverse=True)
    steps = np.bincount(inverse.ravel(), minlength=unique_m.size)
    # Direction is a property of the gene, and every row sharing an 'm' shares a
    # gene, so this equals the original's dict(zip(df['m'], df['step_direction']))
    # lookup by construction.
    _, unique_g = index_converter.get_ng_indices(m_inds=unique_m)
    counts = steps * direction_per_gene[unique_g].astype(np.int64)

    steps_csr = _estimation_array_to_csr(
        index_converter=index_converter,
        data=counts,
        m=unique_m,
        noise_offsets=None,
    )

    if verbose:
        print(f"[vectorized] unique m stepped: {unique_m.size}, total steps: {int(steps.sum())}")
        print("MAP:")
        print(map_csr.todense())

    # The MAP already has the noise offsets, so they are not added to steps_csr.
    return map_csr + steps_csr


class MultipleChoiceKnapsackVectorized(MultipleChoiceKnapsack):
    """``MultipleChoiceKnapsack`` with the pandas per-chunk kernel swapped for
    the numpy one. Everything else (chunking, ``estimate_noise``) is inherited.

    NOTE: only the single-process path is overridden.
    ``estimate_noise(use_multiple_processes=True)`` dispatches to the
    module-level ``_mckp_chunk_estimate_noise`` function in ``estimation.py``
    and is therefore *not* affected by this subclass.
    """

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
        )
