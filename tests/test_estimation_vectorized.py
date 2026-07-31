"""Correctness tests for the numpy/vectorized MCKP chunk kernel.

Validation strategy (deliberately reuses the repo's own fixtures rather than
inventing a new harness):

1. ``test_mckp_vectorized`` -- the *exact* parametrization and known-truth
   matrices from ``test_estimation.test_mckp``, run through
   ``MultipleChoiceKnapsackVectorized``.
2. ``test_mckp_old_vs_new_identical`` -- same fixtures, but asserting the old
   and new full ``estimate_noise`` outputs are element-for-element identical
   sparse matrices (not merely both matching the truth aggregate).
3. ``test_chunk_kernel_old_vs_new_identical`` -- calls the two per-chunk kernels
   directly on the fixture COOs so that the chunk kernel itself, not just the
   summed result, is compared.
4. ``test_random_differential_*`` -- randomized differential tests with
   deliberately dense ``delta`` ties, which is where a naive "k smallest values"
   implementation diverges from pandas' ``keep='first'``.
5. ``test_lexsort_is_stable`` -- verifies (rather than assumes) that
   ``np.lexsort`` breaks ties by input position.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import sparse_matrix_equal

# Reuse the fixtures from the existing test module verbatim.
from test_estimation import log_prob_coo_base, mckp_log_prob_coo  # noqa: F401

from cellbender.remove_background.estimation import MultipleChoiceKnapsack
from cellbender.remove_background.estimation_vectorized import (
    MultipleChoiceKnapsackVectorized,
    chunk_estimate_noise_vectorized,
    stable_topk_positions_per_group,
)
from cellbender.remove_background.posterior import IndexConverter

MCKP_CASES = (
    [1, np.zeros(8), np.array([0, 1, 2, 0, 0, 0, 0, 1]), None],
    [1, np.ones(8), np.array([0, 1, 2, 1, 1, 1, 1, 1]), None],
    [1, np.ones(8) * 2, np.array([0, 1, 2, 2, 2, 2, 2, 2]), None],
    [4, np.zeros(2), np.array([2, 2]), None],
    [4, np.ones(2) * 4, np.array([4, 4]), np.array([[0, 1], [2, 2], [2, 0], [0, 1]])],
    [4, np.ones(2) * 9, np.array([9, 9]), np.array([[0, 1], [2, 3], [4, 2], [3, 3]])],
)
MCKP_IDS = [
    "1_cell_target_0",
    "1_cell_target_1",
    "1_cell_target_2",
    "4_cell_target_0",
    "4_cell_target_4",
    "4_cell_target_9",
]


@pytest.mark.parametrize("n_chunks", (1, 2), ids=["1chunk", "2chunks"])
@pytest.mark.parametrize("n_cells, target, truth, truth_mat", MCKP_CASES, ids=MCKP_IDS)
def test_mckp_vectorized(mckp_log_prob_coo, n_cells, target, truth, truth_mat, n_chunks):
    """Same assertions as test_estimation.test_mckp, vectorized estimator."""
    converter = IndexConverter(total_n_cells=n_cells, total_n_genes=mckp_log_prob_coo["coo"].shape[0] // n_cells)
    estimator = MultipleChoiceKnapsackVectorized(index_converter=converter)
    noise_csr = estimator.estimate_noise(
        noise_log_prob_coo=mckp_log_prob_coo["coo"],
        noise_offsets=mckp_log_prob_coo["offsets"],
        noise_targets_per_gene=target,
        verbose=False,
        n_chunks=n_chunks,
        use_multiple_processes=False,
    )
    assert noise_csr.shape == (converter.total_n_cells, converter.total_n_genes)
    out_mat = np.array(noise_csr.todense())
    if truth_mat is not None:
        np.testing.assert_array_equal(out_mat, truth_mat)
    np.testing.assert_array_equal(out_mat.sum(axis=0), truth)


@pytest.mark.parametrize("n_chunks", (1, 2), ids=["1chunk", "2chunks"])
@pytest.mark.parametrize("n_cells, target, truth, truth_mat", MCKP_CASES, ids=MCKP_IDS)
def test_mckp_old_vs_new_identical(mckp_log_prob_coo, n_cells, target, truth, truth_mat, n_chunks):
    """Old pandas and new numpy estimators must agree element-for-element."""
    converter = IndexConverter(total_n_cells=n_cells, total_n_genes=mckp_log_prob_coo["coo"].shape[0] // n_cells)
    kwargs = dict(
        noise_log_prob_coo=mckp_log_prob_coo["coo"],
        noise_offsets=mckp_log_prob_coo["offsets"],
        noise_targets_per_gene=target,
        verbose=False,
        n_chunks=n_chunks,
        use_multiple_processes=False,
    )
    old = MultipleChoiceKnapsack(index_converter=converter).estimate_noise(**kwargs)
    new = MultipleChoiceKnapsackVectorized(index_converter=converter).estimate_noise(**kwargs)
    assert old.shape == new.shape
    assert old.dtype == new.dtype
    assert sparse_matrix_equal(old.tocsr(), new.tocsr())


@pytest.mark.parametrize("n_cells, target, truth, truth_mat", MCKP_CASES, ids=MCKP_IDS)
def test_chunk_kernel_old_vs_new_identical(mckp_log_prob_coo, n_cells, target, truth, truth_mat):
    """Compare the per-chunk kernels directly (whole COO as a single chunk)."""
    converter = IndexConverter(total_n_cells=n_cells, total_n_genes=mckp_log_prob_coo["coo"].shape[0] // n_cells)
    coo = mckp_log_prob_coo["coo"]
    offsets = mckp_log_prob_coo["offsets"]
    old = MultipleChoiceKnapsack(index_converter=converter)._chunk_estimate_noise(
        noise_log_prob_coo=coo, noise_offsets=offsets, noise_targets_per_gene=target
    )
    new = chunk_estimate_noise_vectorized(
        index_converter=converter,
        noise_log_prob_coo=coo,
        noise_offsets=offsets,
        noise_targets_per_gene=target,
    )
    assert sparse_matrix_equal(old.tocsr(), new.tocsr())


# --------------------------------------------------------------------------
# Randomized differential testing
# --------------------------------------------------------------------------


def _random_posterior(rng, n_cells, n_genes, n_counts_max, p_entry, log_prob_decimals, dtype=np.float64):
    """A structurally valid random noise-log-prob COO.

    Mirrors the real structure from Posterior._get_cell_noise_count_posterior_coo:
    rows are 'm' = n * total_n_genes + g, columns are noise count values
    0..n_counts_max-1, values are normalized log probabilities, and only some
    (n, g) pairs are present (those with a nonzero raw count).

    ``log_prob_decimals`` rounds the log probs, which manufactures ties in the
    resulting ``delta`` values -- the whole point of this test.
    """
    converter = IndexConverter(total_n_cells=n_cells, total_n_genes=n_genes)
    present = rng.random((n_cells, n_genes)) < p_entry
    ns, gs = np.nonzero(present)
    if ns.size == 0:
        ns, gs = np.array([0]), np.array([0])
    m_vals = ns.astype(np.uint64) * np.uint64(n_genes) + gs.astype(np.uint64)

    rows, cols, data = [], [], []
    for m in m_vals:
        width = int(rng.integers(1, n_counts_max + 1))
        start = int(rng.integers(0, n_counts_max - width + 1))
        logits = rng.normal(size=width) * 1.5
        lp = logits - np.log(np.exp(logits).sum())
        lp = np.round(lp, log_prob_decimals)
        rows.append(np.full(width, m, dtype=np.uint64))
        cols.append(np.arange(start, start + width))
        data.append(lp)
    coo = sp.coo_matrix(
        (np.concatenate(data).astype(dtype), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_cells * n_genes, n_counts_max),
    )
    assert coo.data.dtype == dtype
    offsets = {int(m): int(rng.integers(0, 3)) for m in m_vals}
    return converter, coo, offsets


@pytest.mark.parametrize("dtype", (np.float64, np.float32), ids=["f64", "f32"])
@pytest.mark.parametrize("log_prob_decimals", (16, 2, 1), ids=["no_ties", "some_ties", "many_ties"])
@pytest.mark.parametrize("seed", range(12))
def test_random_differential_chunk(seed, log_prob_decimals, dtype):
    """Old vs new on random structurally-valid posteriors, with tie pressure.

    float32 is the dtype a REAL run produces (traced through
    Posterior._get_cell_noise_count_posterior_coo: log probs come from .float()
    torch tensors, and --posterior-regularization defaults to None so the
    float64-producing regularized path is not the default). float64 covers the
    regularized path, whose COO is built from a Python list via .tolist().
    """
    rng = np.random.default_rng(seed)
    n_cells = int(rng.integers(2, 12))
    n_genes = int(rng.integers(2, 15))
    converter, coo, offsets = _random_posterior(
        rng, n_cells, n_genes, n_counts_max=8, p_entry=0.6, log_prob_decimals=log_prob_decimals, dtype=dtype
    )
    # targets spanning "MAP already exceeds", "MAP matches", and "need more"
    targets = rng.integers(0, 4 * n_cells, size=n_genes).astype(float)

    old = MultipleChoiceKnapsack(index_converter=converter)._chunk_estimate_noise(
        noise_log_prob_coo=coo, noise_offsets=offsets, noise_targets_per_gene=targets
    )
    new = chunk_estimate_noise_vectorized(
        index_converter=converter,
        noise_log_prob_coo=coo,
        noise_offsets=offsets,
        noise_targets_per_gene=targets,
    )
    assert sparse_matrix_equal(old.tocsr(), new.tocsr()), (
        f"mismatch seed={seed} decimals={log_prob_decimals} dtype={np.dtype(dtype).name}"
        f"\nold:\n{old.todense()}\nnew:\n{new.todense()}"
    )


@pytest.mark.parametrize("dtype", (np.float64, np.float32), ids=["f64", "f32"])
@pytest.mark.parametrize("seed", range(8))
def test_random_differential_unsorted_coo(seed, dtype):
    """Same, but with the COO entries shuffled (as in the 'unsorted' fixture)."""
    rng = np.random.default_rng(1000 + seed)
    converter, coo, offsets = _random_posterior(
        rng, n_cells=6, n_genes=7, n_counts_max=8, p_entry=0.7, log_prob_decimals=1, dtype=dtype
    )
    perm = rng.permutation(coo.data.size)
    coo = sp.coo_matrix((coo.data[perm], (coo.row[perm], coo.col[perm])), shape=coo.shape)
    targets = rng.integers(0, 20, size=7).astype(float)
    old = MultipleChoiceKnapsack(index_converter=converter)._chunk_estimate_noise(
        noise_log_prob_coo=coo, noise_offsets=offsets, noise_targets_per_gene=targets
    )
    new = chunk_estimate_noise_vectorized(
        index_converter=converter,
        noise_log_prob_coo=coo,
        noise_offsets=offsets,
        noise_targets_per_gene=targets,
    )
    assert sparse_matrix_equal(old.tocsr(), new.tocsr())


def test_topk_selection_matches_pandas_exactly():
    """Directly compare selected *row positions* against pandas groupby-nsmallest.

    This is stronger than comparing the output matrices, which could coincide
    even if different rows were chosen.
    """
    import pandas as pd

    rng = np.random.default_rng(7)
    for trial in range(200):
        n = int(rng.integers(1, 400))
        n_groups = int(rng.integers(1, 20))
        g = rng.integers(0, n_groups, size=n)
        # heavy ties: only a few distinct delta values
        delta = rng.integers(0, 4, size=n).astype(float)
        topk = rng.integers(1, 8, size=n_groups)

        mine = np.sort(stable_topk_positions_per_group(g, delta, topk))

        df = pd.DataFrame({"pos": np.arange(n), "g": g, "delta": delta})
        df["topk"] = topk[g]
        theirs = (
            df.groupby("g", group_keys=False)
            .apply(lambda x: x.nsmallest(x["topk"].iat[0], columns="delta"), include_groups=False)["pos"]
            .to_numpy()
        )
        theirs = np.sort(theirs)
        np.testing.assert_array_equal(mine, theirs, err_msg=f"trial {trial}")


def test_lexsort_is_stable():
    """np.lexsort must break (group, value) ties by input position."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 500))
        g = rng.integers(0, 5, size=n)
        v = rng.integers(0, 3, size=n).astype(float)
        two_key = np.lexsort((v, g))
        three_key = np.lexsort((np.arange(n), v, g))
        np.testing.assert_array_equal(two_key, three_key)


@pytest.mark.parametrize("n_cells, target, truth, truth_mat", MCKP_CASES, ids=MCKP_IDS)
def test_chunk_kernel_float32_old_vs_new(mckp_log_prob_coo, n_cells, target, truth, truth_mat):
    """The repo's own fixtures, but cast to float32 -- the production dtype.

    Every pre-existing fixture in tests/test_estimation.py is float64 (it builds
    the COO from a float64 numpy array via torch), so float32 -- what a real run
    actually produces -- had no coverage at all before this test.
    """
    converter = IndexConverter(total_n_cells=n_cells, total_n_genes=mckp_log_prob_coo["coo"].shape[0] // n_cells)
    c = mckp_log_prob_coo["coo"]
    coo32 = sp.coo_matrix((c.data.astype(np.float32), (c.row, c.col)), shape=c.shape)
    assert coo32.data.dtype == np.float32
    offsets = mckp_log_prob_coo["offsets"]
    old = MultipleChoiceKnapsack(index_converter=converter)._chunk_estimate_noise(
        noise_log_prob_coo=coo32, noise_offsets=offsets, noise_targets_per_gene=target
    )
    new = chunk_estimate_noise_vectorized(
        index_converter=converter,
        noise_log_prob_coo=coo32,
        noise_offsets=offsets,
        noise_targets_per_gene=target,
    )
    assert sparse_matrix_equal(old.tocsr(), new.tocsr())


def _find_f32_delta_collision(rng, n_needed=6, pool=2_000_000):
    """Find float32 log-prob pairs whose float32 difference is IDENTICAL but whose
    exact (float64) difference DIFFERS.

    ``hi`` is drawn near 0 and ``lo`` near -10 (both inside the real posterior's
    ``smallest_log_probability = -10.0`` window). Their exact difference needs
    ~1e-10 granularity while a float32 result of magnitude ~10 only carries
    ~9.5e-7, so the subtraction necessarily rounds -- which is exactly the regime
    where float32 and float64 arithmetic disagree on ordering.

    Returns (hi, lo, d32, d64) for n_needed pairs, ordered so that the exact
    difference DESCENDS with index (so that a float64 implementation would pick
    the opposite end of the group from a float32 one).
    """
    hi = (-rng.random(pool) * 0.5).astype(np.float32)
    lo = (-10.0 + rng.random(pool) * 0.5).astype(np.float32)
    d32 = hi - lo  # float32 arithmetic (rounds)
    d64 = hi.astype(np.float64) - lo.astype(np.float64)  # exact
    vals, inv, counts = np.unique(d32, return_inverse=True, return_counts=True)
    for vi in np.flatnonzero(counts >= n_needed):
        idx = np.flatnonzero(inv.ravel() == vi)
        _, first = np.unique(d64[idx], return_index=True)
        if first.size < n_needed:
            continue
        sel = idx[np.sort(first)][:n_needed]
        order = np.argsort(-d64[sel], kind="stable")  # exact difference descending
        sel = sel[order]
        return hi[sel], lo[sel], d32[sel], d64[sel]
    return None


def test_adversarial_float32_tie_break_at_rounding_boundary():
    """THE adversarial case: deltas that tie exactly in float32 but not in float64.

    Construction: N droplets, gene 0 gets two entries each (c=0 with log prob
    ``hi``, c=1 with ``lo``), chosen so every droplet's delta ``|lo - hi|`` is the
    SAME float32 number while the exact differences are all distinct. Gene 1 is
    filler whose MAP already meets its target (step_direction 0).

    With target k < N for gene 0, k of the N tied droplets must be chosen purely
    by tie-break. If our narrowed float32 delta disagreed with pandas -- or if we
    had upcast log_prob to float64 before diffing -- a different set of droplets
    would get the noise count, i.e. a different exact integer matrix.
    """
    rng = np.random.default_rng(20240730)
    found = _find_f32_delta_collision(rng, n_needed=6)
    assert found is not None, "could not construct a float32 delta collision"
    hi, lo, d32, d64 = found

    # The construction must genuinely have the adversarial property.
    assert np.all(d32 == d32[0]), "float32 deltas are not exactly tied"
    assert np.unique(d64).size == d64.size, "exact deltas are not distinct"
    assert np.all(np.diff(d64) < 0), "exact deltas should descend with droplet index"
    print(f"tied float32 delta = {d32[0]!r}; distinct exact deltas = {d64}")

    n_cells, n_genes, k = len(hi), 2, 2
    rows, cols, data = [], [], []
    for i in range(n_cells):
        m0 = i * n_genes + 0  # tie gene
        rows += [m0, m0]
        cols += [0, 1]
        data += [hi[i], lo[i]]
        m1 = i * n_genes + 1  # filler gene, single entry -> MAP 0, target 0
        rows.append(m1)
        cols.append(0)
        data.append(np.float32(-0.25))
    coo = sp.coo_matrix(
        (np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))),
        shape=(n_cells * n_genes, 20),
    )
    assert coo.data.dtype == np.float32
    converter = IndexConverter(total_n_cells=n_cells, total_n_genes=n_genes)
    targets = np.array([float(k), 0.0])

    old = MultipleChoiceKnapsack(index_converter=converter)._chunk_estimate_noise(
        noise_log_prob_coo=coo, noise_offsets={}, noise_targets_per_gene=targets
    )
    new = chunk_estimate_noise_vectorized(
        index_converter=converter,
        noise_log_prob_coo=coo,
        noise_offsets={},
        noise_targets_per_gene=targets,
    )
    old_d = np.array(old.todense())
    new_d = np.array(new.todense())
    print("old (pandas float32):\n", old_d.T)
    print("new (vectorized, narrowed):\n", new_d.T)

    # (1) The real requirement: exact integer equality with the pandas original.
    assert sparse_matrix_equal(old.tocsr(), new.tocsr()), (
        f"tie-break diverged under float32\nold:\n{old_d}\nnew:\n{new_d}"
    )
    # Sanity: exactly k droplets were stepped, and they are the FIRST k (tie-break).
    assert old_d[:, 0].sum() == k
    np.testing.assert_array_equal(np.flatnonzero(old_d[:, 0]), np.arange(k))

    # (2) Negative control: the same selection driven by the EXACT float64 deltas
    # picks the opposite end of the group, so this test genuinely discriminates
    # precision handling rather than passing for free.
    g = np.zeros(n_cells, dtype=np.int32)
    topk = np.array([k, 0])
    sel_f32 = np.sort(stable_topk_positions_per_group(g, d32.astype(np.float32), topk))
    sel_f64 = np.sort(stable_topk_positions_per_group(g, d64, topk))
    print(f"selection from tied float32 deltas: {sel_f32}; from exact float64 deltas: {sel_f64}")
    assert not np.array_equal(sel_f32, sel_f64), (
        "negative control failed: float32 and float64 deltas select the same rows, "
        "so this case does not actually probe the precision boundary"
    )

    # (3) Supplementary continuous check: the delta VALUES agree to within the
    # narrower dtype's eps, even where the discrete selection could differ.
    eps = np.finfo(np.float32).eps
    assert np.allclose(d32.astype(np.float64), d64, rtol=eps * 4, atol=0.0)


def test_column_compaction_quirk_is_replicated():
    """Adversarial: force ``chunked_iterator``'s c-axis compaction to shift the MAP.

    ``chunked_iterator`` compacts the column axis with ``np.unique`` before
    ``MAP.torch_argmax`` runs, and ``apply_function_dense_chunks`` throws the
    ``unique_col_values`` away. So when no entry occupies c=0, ``map_dict['result']``
    is off by one from the true noise count, the ``c == map`` NaN sentinel lands on
    the WRONG row, and the original's global ``.diff()`` leaks a cross-droplet
    delta. The vectorized version must leak the identical one.
    """
    n_cells, n_genes, n_c = 5, 4, 7
    rng = np.random.default_rng(4242)
    rows, cols, data = [], [], []
    for n in range(n_cells):
        for gene in range(n_genes):
            m = n * n_genes + gene
            # never occupy c = 0 -> np.unique(col) starts at 1 -> compaction shift
            start = 1
            width = int(rng.integers(3, n_c - start + 1))
            logits = rng.normal(size=width)
            lp = logits - np.log(np.exp(logits).sum())
            rows.append(np.full(width, m))
            cols.append(np.arange(start, start + width))
            data.append(lp)
    coo = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_cells * n_genes, n_c),
    )
    assert coo.col.min() == 1, "test setup must exclude column 0"
    converter = IndexConverter(total_n_cells=n_cells, total_n_genes=n_genes)
    old_est = MultipleChoiceKnapsack(index_converter=converter)
    for targets in (np.zeros(n_genes), np.ones(n_genes) * 6, np.ones(n_genes) * 20):
        old = old_est._chunk_estimate_noise(
            noise_log_prob_coo=coo, noise_offsets={}, noise_targets_per_gene=targets
        )
        new = chunk_estimate_noise_vectorized(
            index_converter=converter,
            noise_log_prob_coo=coo,
            noise_offsets={},
            noise_targets_per_gene=targets,
        )
        assert sparse_matrix_equal(old.tocsr(), new.tocsr()), (
            f"targets={targets}\nold:\n{old.todense()}\nnew:\n{new.todense()}"
        )


def test_empty_and_degenerate_inputs():
    """Early-return paths: no gene needs steps; and target exactly met by MAP."""
    converter = IndexConverter(total_n_cells=2, total_n_genes=3)
    dense = np.array(
        [
            [np.log(0.5), np.log(0.5), -np.inf],
            [np.log(0.9), np.log(0.1), -np.inf],
            [-np.inf, -np.inf, -np.inf],
            [np.log(0.2), np.log(0.8), -np.inf],
            [np.log(0.7), np.log(0.3), -np.inf],
            [-np.inf, -np.inf, -np.inf],
        ]
    )
    r, c = np.nonzero(np.isfinite(dense))
    coo = sp.coo_matrix((dense[r, c], (r, c)), shape=dense.shape)
    offsets = {}

    old_est = MultipleChoiceKnapsack(index_converter=converter)
    # 1) targets equal to the MAP result -> step_direction == 0 everywhere
    map_only = old_est._chunk_estimate_noise(
        noise_log_prob_coo=coo, noise_offsets=offsets, noise_targets_per_gene=np.zeros(3)
    )
    per_gene_map = np.array(map_only.sum(axis=0)).squeeze().astype(float)
    for targets in (per_gene_map, np.zeros(3), np.ones(3) * 50):
        old = old_est._chunk_estimate_noise(
            noise_log_prob_coo=coo, noise_offsets=offsets, noise_targets_per_gene=targets
        )
        new = chunk_estimate_noise_vectorized(
            index_converter=converter,
            noise_log_prob_coo=coo,
            noise_offsets=offsets,
            noise_targets_per_gene=targets,
        )
        assert sparse_matrix_equal(old.tocsr(), new.tocsr()), f"targets={targets}"
