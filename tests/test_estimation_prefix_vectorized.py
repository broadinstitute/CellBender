"""Correctness tests for the vectorized shared-prefix helpers.

Every test here asserts *identity* with the pristine ``estimation.py`` helper it
replaces, not mere equivalence: same chunk contents in the same order, same
arrays, same final CSR.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import sparse_matrix_equal
from test_estimation import log_prob_coo_base, mckp_log_prob_coo  # noqa: F401
from test_estimation_vectorized import MCKP_CASES, MCKP_IDS, _random_posterior

from cellbender.remove_background import estimation as E
from cellbender.remove_background import estimation_prefix_vectorized as PV
from cellbender.remove_background.posterior import IndexConverter


def _coo_triples_equal(a: sp.coo_matrix, b: sp.coo_matrix) -> bool:
    return (
        a.shape == b.shape
        and np.array_equal(a.row, b.row)
        and np.array_equal(a.col, b.col)
        and np.array_equal(a.data, b.data)
    )


# --------------------------------------------------------------------------
# chunked_iterator
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (np.float64, np.float32), ids=["f64", "f32"])
@pytest.mark.parametrize(
    "batch_gb", (1.0, 1e-5, 1e-6), ids=["one_chunk", "several_chunks", "many_chunks"]
)
def test_chunked_iterator_identical(batch_gb, dtype):
    """Chunk-for-chunk, triple-for-triple identity, including within-chunk order."""
    rng = np.random.default_rng(11)
    _conv, coo, _offs = _random_posterior(
        rng, n_cells=40, n_genes=25, n_counts_max=12, p_entry=0.5, log_prob_decimals=16, dtype=dtype
    )
    old = list(E.chunked_iterator(coo=coo, max_dense_batch_size_GB=batch_gb))
    new = list(PV.chunked_iterator_vectorized(coo=coo, max_dense_batch_size_GB=batch_gb))
    assert len(old) == len(new), f"{len(old)} vs {len(new)} chunks"

    # Verify the scenario is what its id claims, derived from the same formula the
    # implementation uses -- rather than assuming a given GB budget splits this
    # particular (small) test COO.
    expected_batch = max(1, int(np.floor(batch_gb * 1e9 / 4 / coo.shape[1])))
    n_unique = len(np.unique(coo.row))
    expected_chunks = max(1, n_unique // expected_batch)
    assert len(old) == expected_chunks, f"expected {expected_chunks} chunks, iterator gave {len(old)}"
    if batch_gb == 1.0:
        assert len(old) == 1
    else:
        assert len(old) > 1, (
            f"batch budget {batch_gb} GB -> {expected_batch} m-values/chunk does not split "
            f"{n_unique} unique m-values; pick a smaller budget for this test to be meaningful"
        )
    for i, ((ca, ra, cla), (cb, rb, clb)) in enumerate(zip(old, new)):
        assert np.array_equal(ra, rb), f"chunk {i} unique_row_values differ"
        assert np.array_equal(cla, clb), f"chunk {i} unique_col_values differ"
        assert cla.dtype == clb.dtype, f"chunk {i} unique_col_values dtype differs"
        assert _coo_triples_equal(ca, cb), f"chunk {i} COO contents/order differ"


def test_chunked_iterator_unsorted_coo_identical():
    """Shuffled COO entry order must still be reproduced exactly."""
    rng = np.random.default_rng(12)
    _conv, coo, _offs = _random_posterior(
        rng, n_cells=30, n_genes=20, n_counts_max=10, p_entry=0.6, log_prob_decimals=16
    )
    perm = rng.permutation(coo.data.size)
    coo = sp.coo_matrix((coo.data[perm], (coo.row[perm], coo.col[perm])), shape=coo.shape)
    saw_multichunk = False
    for gb in (1.0, 1e-6):
        old = list(E.chunked_iterator(coo=coo, max_dense_batch_size_GB=gb))
        new = list(PV.chunked_iterator_vectorized(coo=coo, max_dense_batch_size_GB=gb))
        assert len(old) == len(new)
        saw_multichunk |= len(old) > 1
        for (ca, ra, cla), (cb, rb, clb) in zip(old, new):
            assert np.array_equal(ra, rb) and np.array_equal(cla, clb)
            assert _coo_triples_equal(ca, cb)
    assert saw_multichunk, "shuffled-order test never exercised the multi-chunk path"


def test_chunked_iterator_empty_coo():
    """Degenerate empty COO: both must yield one empty chunk and not raise."""
    coo = sp.coo_matrix((np.zeros(0), (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))), shape=(10, 5))
    old = list(E.chunked_iterator(coo=coo))
    new = list(PV.chunked_iterator_vectorized(coo=coo))
    assert len(old) == len(new) == 1
    assert old[0][0].shape == new[0][0].shape
    assert old[0][0].data.size == new[0][0].data.size == 0


# --------------------------------------------------------------------------
# apply_function_dense_chunks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (np.float64, np.float32), ids=["f64", "f32"])
def test_apply_function_dense_chunks_identical(dtype):
    rng = np.random.default_rng(13)
    _conv, coo, _offs = _random_posterior(
        rng, n_cells=35, n_genes=22, n_counts_max=11, p_entry=0.55, log_prob_decimals=16, dtype=dtype
    )
    a = E.apply_function_dense_chunks(noise_log_prob_coo=coo, fun=E.MAP.torch_argmax, device="cpu")
    b = PV.apply_function_dense_chunks_vectorized(
        noise_log_prob_coo=coo, fun=E.MAP.torch_argmax, device="cpu"
    )
    np.testing.assert_array_equal(a["m"], b["m"])
    np.testing.assert_array_equal(a["result"], b["result"])
    assert a["m"].dtype == b["m"].dtype


# --------------------------------------------------------------------------
# NoiseOffsetLookup / _estimation_array_to_csr
# --------------------------------------------------------------------------


@pytest.mark.parametrize("offsets", [None, {}, {5: 3}, {0: 1, 7: 2, 999999: 9}])
def test_noise_offset_lookup_matches_dict_get(offsets):
    """Must reproduce ``[noise_offsets.get(i, 0) for i in m]`` exactly, incl. misses."""
    m = np.array([0, 1, 5, 7, 8, 999999, 12345], dtype=np.uint64)
    lut = PV.NoiseOffsetLookup(offsets)
    got = lut.offsets_for(m)
    want = np.array([(offsets or {}).get(int(i), 0) for i in m])
    np.testing.assert_array_equal(got, want)


def test_noise_offset_lookup_identity_cache():
    d = {1: 1}
    lut = PV.NoiseOffsetLookup(d)
    assert lut.matches(d)
    assert not lut.matches({1: 1})  # equal but not the same object


@pytest.mark.parametrize("dtype", (np.float64, np.float32), ids=["f64", "f32"])
def test_estimation_array_to_csr_identical(dtype):
    rng = np.random.default_rng(14)
    conv, coo, offs = _random_posterior(
        rng, n_cells=30, n_genes=18, n_counts_max=9, p_entry=0.6, log_prob_decimals=16, dtype=dtype
    )
    md = E.apply_function_dense_chunks(noise_log_prob_coo=coo, fun=E.MAP.torch_argmax, device="cpu")
    for noise_offsets in (None, {}, offs):
        a = E._estimation_array_to_csr(
            index_converter=conv, data=md["result"], m=md["m"], noise_offsets=noise_offsets
        )
        b = PV.estimation_array_to_csr_vectorized(
            index_converter=conv, data=md["result"], m=md["m"], noise_offsets=noise_offsets
        )
        assert sparse_matrix_equal(a, b), f"differs for noise_offsets={type(noise_offsets)}"


# --------------------------------------------------------------------------
# gene chunk logic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_chunks", (1, 2, 3, 7, 40))
def test_gene_chunk_logic_identical(n_chunks):
    rng = np.random.default_rng(15)
    conv, coo, _offs = _random_posterior(
        rng, n_cells=25, n_genes=40, n_counts_max=8, p_entry=0.5, log_prob_decimals=16
    )
    old = E.MultipleChoiceKnapsack(index_converter=conv)._gene_chunk_iterator(coo, n_chunks=n_chunks)
    new = PV.gene_chunk_logic_vectorized(conv, coo, n_chunks=n_chunks)
    assert len(old) == len(new)
    for i, (a, b) in enumerate(zip(old, new)):
        np.testing.assert_array_equal(a, b, err_msg=f"gene chunk {i}")


# --------------------------------------------------------------------------
# End-to-end estimate_noise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (np.float64, np.float32), ids=["f64", "f32"])
@pytest.mark.parametrize("n_chunks", (1, 2), ids=["1chunk", "2chunks"])
@pytest.mark.parametrize("n_cells, target, truth, truth_mat", MCKP_CASES, ids=MCKP_IDS)
def test_estimate_noise_end_to_end_identical(
    mckp_log_prob_coo, n_cells, target, truth, truth_mat, n_chunks, dtype
):
    """Full estimate_noise: original vs fully-vectorized (kernel + prefix)."""
    c = mckp_log_prob_coo["coo"]
    coo = sp.coo_matrix((c.data.astype(dtype), (c.row, c.col)), shape=c.shape)
    converter = IndexConverter(total_n_cells=n_cells, total_n_genes=c.shape[0] // n_cells)
    kwargs = dict(
        noise_log_prob_coo=coo,
        noise_offsets=mckp_log_prob_coo["offsets"],
        noise_targets_per_gene=target,
        verbose=False,
        n_chunks=n_chunks,
        use_multiple_processes=False,
    )
    old = E.MultipleChoiceKnapsack(index_converter=converter).estimate_noise(**kwargs)
    new = PV.MultipleChoiceKnapsackFast(index_converter=converter).estimate_noise(**kwargs)
    assert sparse_matrix_equal(old.tocsr(), new.tocsr())
    # and still matches the fixture's known truth
    np.testing.assert_array_equal(np.array(new.todense()).sum(axis=0), truth)


@pytest.mark.parametrize("dtype", (np.float64, np.float32), ids=["f64", "f32"])
@pytest.mark.parametrize("seed", range(6))
def test_estimate_noise_end_to_end_random(seed, dtype):
    rng = np.random.default_rng(500 + seed)
    n_cells, n_genes = 12, 24
    conv, coo, offs = _random_posterior(
        rng, n_cells, n_genes, n_counts_max=8, p_entry=0.6, log_prob_decimals=1, dtype=dtype
    )
    targets = rng.integers(0, 3 * n_cells, size=n_genes).astype(float)
    kwargs = dict(
        noise_log_prob_coo=coo,
        noise_offsets=offs,
        noise_targets_per_gene=targets,
        verbose=False,
        n_chunks=3,
        use_multiple_processes=False,
    )
    old = E.MultipleChoiceKnapsack(index_converter=conv).estimate_noise(**kwargs)
    new = PV.MultipleChoiceKnapsackFast(index_converter=conv).estimate_noise(**kwargs)
    assert sparse_matrix_equal(old.tocsr(), new.tocsr())


# --------------------------------------------------------------------------
# Regression tests for confirmed upstream bugs, now fixed
# --------------------------------------------------------------------------


def test_subset_coo_empty_chunk_does_not_crash():
    """Regression test for a CONFIRMED UPSTREAM BUG, now fixed.

    This test used to be ``test_documents_preexisting_subset_coo_empty_chunk_crash``
    and asserted that ``_subset_coo`` RAISES on an empty selection: it called
    ``sp.coo_matrix`` with no ``shape=``, so scipy tried to infer the dimensions
    from ``max(row) + 1, max(col) + 1`` and raised "cannot infer dimensions from an
    empty ...". Any real run whose gene chunking produced an empty chunk (e.g. a
    trailing block of features with no analyzed counts) crashed in
    ``MultipleChoiceKnapsack.estimate_noise``.

    ``_subset_coo`` now passes ``shape=coo.shape``, so the empty case yields an
    empty COO of the parent's shape. Documenting the crash has been replaced by
    asserting it cannot come back.
    """
    coo = sp.coo_matrix((np.array([1.0]), (np.array([0]), np.array([0]))), shape=(5, 5))
    out = E._subset_coo(coo, np.array([False]))
    assert out.shape == (5, 5)
    assert out.data.size == 0
    # ...and the shape is preserved for non-empty selections too, rather than being
    # inferred (and silently shrunk) from the selected entries.
    coo2 = sp.coo_matrix((np.array([1.0, 2.0]), (np.array([1, 3]), np.array([0, 2]))), shape=(9, 7))
    kept = E._subset_coo(coo2, np.array([True, False]))
    assert kept.shape == (9, 7)
    np.testing.assert_array_equal(kept.data, np.array([1.0]))
    np.testing.assert_array_equal(kept.row, np.array([1]))
    np.testing.assert_array_equal(kept.col, np.array([0]))


@pytest.mark.parametrize("n_chunks", (1, 2, 3, 4, 6))
def test_mckp_estimate_noise_survives_empty_gene_chunk(n_chunks):
    """End-to-end companion to the above: gene chunking that yields an empty chunk.

    Genes are only occupied in the low half of the index range, so with enough
    chunks at least one chunk selects zero posterior entries -- the exact situation
    that used to crash in ``_subset_coo``. Both the original and the
    ``mckp-fast`` subclass must complete and agree.
    """
    n_cells, n_genes, n_c = 4, 12, 6
    conv = IndexConverter(total_n_cells=n_cells, total_n_genes=n_genes)
    rng = np.random.default_rng(77)
    rows, cols, data = [], [], []
    for n in range(n_cells):
        for gene in range(n_genes // 2):  # upper half of genes: no entries at all
            m = conv.get_m_indices(cell_inds=np.array([n]), gene_inds=np.array([gene]))[0]
            logits = rng.normal(size=n_c)
            lp = logits - np.log(np.exp(logits).sum())
            rows.append(np.full(n_c, m))
            cols.append(np.arange(n_c))
            data.append(lp)
    coo = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_cells * n_genes, n_c),
    )
    _, occupied_genes = conv.get_ng_indices(m_inds=coo.row)
    gene_logic = E.MultipleChoiceKnapsack(index_converter=conv)._gene_chunk_iterator(coo, n_chunks=n_chunks)
    n_empty = sum(1 for logic in gene_logic if not np.any(logic))
    if n_chunks > 2:
        assert n_empty > 0, (
            f"n_chunks={n_chunks} produced no empty gene chunk over genes "
            f"{sorted(set(occupied_genes.tolist()))}; this test would not exercise the fix"
        )

    kwargs = dict(
        noise_log_prob_coo=coo,
        noise_offsets={},
        noise_targets_per_gene=np.ones(n_genes) * 2.0,
        verbose=False,
        n_chunks=n_chunks,
        use_multiple_processes=False,
    )
    old = E.MultipleChoiceKnapsack(index_converter=conv).estimate_noise(**kwargs)
    new = PV.MultipleChoiceKnapsackFast(index_converter=conv).estimate_noise(**kwargs)
    assert sparse_matrix_equal(old.tocsr(), new.tocsr())
