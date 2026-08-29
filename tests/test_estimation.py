"""Test functions in estimation.py"""

from pathlib import Path
from typing import Dict, Union

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp
import torch

import cellbender.remove_background.estimation as est_mod
from cellbender.remove_background.data.io import (
    POSTERIOR_SCHEMA,
    _make_duckdb_conn,
    write_posterior_batch_to_parquet,
)
from cellbender.remove_background.estimation import (
    MAP,
    Mean,
    MultipleChoiceKnapsack,
    SingleSample,
    ThresholdCDF,
    _mckp_chunk_size,
    estimate_mean_noise_per_gene,
)
from cellbender.remove_background.posterior import compute_mean_target_removal_as_function
from cellbender.remove_background.sparse_utils import dense_to_sparse_op_torch, log_prob_sparse_to_dense


@pytest.fixture(scope="module")
def log_prob_coo_base() -> Dict[str, Union[sp.coo_matrix, np.ndarray]]:
    n = -np.inf
    m = np.array(
        [
            [0, n, n, n, n, n, n, n, n, n],  # map 0, mean 0
            [n, 0, n, n, n, n, n, n, n, n],  # map 1, mean 1
            [n, n, 0, n, n, n, n, n, n, n],  # map 2, mean 2
            [-6, -2, np.log(1.0 - 2 * np.exp(-2) - np.exp(-6)), -2, n, n, n, n, n, n],
            [-2.5, -1.5, -0.5, -3, np.log(1.0 - np.exp([-2.5, -1.5, -0.5, -3]).sum()), n, n, n, n, n],
            [-0.74, -1, -2, -4, np.log(1.0 - np.exp([-0.74, -1, -2, -4]).sum()), n, n, n, n, n],
            [-1, -0.74, -2, -4, np.log(1.0 - np.exp([-0.74, -1, -2, -4]).sum()), n, n, n, n, n],
            [-2, -1, -0.74, -4, np.log(1.0 - np.exp([-0.74, -1, -2, -4]).sum()), n, n, n, n, n],
        ]
    )
    # make m sparse, i.e. zero probability entries are absent
    rows, cols, vals = dense_to_sparse_op_torch(torch.tensor(m), tensor_for_nonzeros=torch.tensor(m).exp())
    # make it a bit more difficult by having an empty row at the beginning
    rows = rows + 1
    shape = list(m.shape)
    shape[0] = shape[0] + 1

    # The last dense row (m row 7, COO row 8) has a noise count offset of 1 in the old
    # representation. We encode that offset directly into the COO column values (absolute
    # noise counts), shifting cols for that row by +1.
    last_row_mask = (rows == 8).numpy()
    cols = cols.clone()
    cols[last_row_mask] = cols[last_row_mask] + 1

    # Absolute MAP and CDF values (offset already embedded in COO cols above).
    maps = np.argmax(m, axis=1)
    shifts = np.array([0] * 7 + [1])
    maps = maps + shifts
    cdf_logic = torch.logcumsumexp(torch.tensor(m), dim=-1) > np.log(0.5)
    cdfs_list = [np.where(a)[0][0] for a in cdf_logic]
    cdfs: np.ndarray = np.array(cdfs_list) + shifts

    return {
        "coo": sp.coo_matrix((vals, (rows, cols)), shape=shape),
        "maps": np.array([0] + maps.tolist()),
        "cdfs": np.array([0] + cdfs.tolist()),
    }


@pytest.fixture(scope="module")
def log_prob_parquet(log_prob_coo_base, tmp_path_factory) -> Dict[str, Union[Path, np.ndarray, sp.coo_matrix]]:
    """Write the base COO fixture to a parquet file for SQL-path tests.

    One cell, n_genes genes: cell_id=0 for all entries, gene_id = COO row value.
    """
    coo = log_prob_coo_base["coo"]
    n_genes = coo.shape[0]

    tmp_dir = tmp_path_factory.mktemp("parquet")
    parquet_path = tmp_dir / "posterior.parquet"

    cell_ids = np.zeros(len(coo.row), dtype=np.int32)
    gene_ids = coo.row.astype(np.int32)
    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        c_vals=coo.col.astype(np.int32),
        log_probs=coo.data.astype(np.float32),
    )
    writer.close()

    return {
        "path": parquet_path,
        "n_genes": n_genes,
        "coo": coo,
        "maps": log_prob_coo_base["maps"],
        "cdfs": log_prob_coo_base["cdfs"],
    }


def _read_noise_parquet(path: Path) -> np.ndarray:
    """Read (cell_id, gene_id, noise_count) parquet and return flat noise_count array indexed by gene_id.
    Assumes a single-cell (cell_id=0) result and returns a 1-D array over gene_ids."""
    df = duckdb.connect().execute(f"SELECT gene_id, noise_count FROM read_parquet('{path}') ORDER BY gene_id").df()
    if len(df) == 0:
        return np.array([])
    max_gene = int(df["gene_id"].max())
    out = np.zeros(max_gene + 1)
    out[df["gene_id"].values] = df["noise_count"].values
    return out


# ---------------------------------------------------------------------------
# SQL / parquet-path tests for MAP, Mean, ThresholdCDF, SingleSample
# ---------------------------------------------------------------------------


def test_map_from_parquet(log_prob_parquet, tmp_path):
    """MAP from parquet path (SQL argmax) matches known truth."""
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    estimator = MAP(n_cells=1, n_genes=n_genes)
    output = tmp_path / "noise.parquet"
    estimator.estimate_noise_to_parquet(path, output)
    out_per_m = _read_noise_parquet(output)
    # Pad to n_genes in case zero-count genes are absent
    result = np.zeros(n_genes)
    result[: len(out_per_m)] = out_per_m
    np.testing.assert_array_equal(result, log_prob_parquet["maps"])


def test_mean_from_parquet(log_prob_parquet, tmp_path):
    """Mean from parquet path (SQL weighted sum) matches brute-force truth."""
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    coo = log_prob_parquet["coo"]
    estimator = Mean(n_cells=1, n_genes=n_genes)
    output = tmp_path / "noise.parquet"
    estimator.estimate_noise_to_parquet(path, output)
    out_per_m = _read_noise_parquet(output)
    result = np.zeros(n_genes)
    result[: len(out_per_m)] = out_per_m

    dense = log_prob_sparse_to_dense(coo)
    brute_force = np.matmul(np.arange(dense.shape[1]), np.exp(dense).T)
    # rtol=1e-4: log_probs stored as float32 in parquet vs float64 brute force
    np.testing.assert_allclose(result, brute_force, rtol=1e-4)


def test_cdf_from_parquet(log_prob_parquet, tmp_path):
    """ThresholdCDF from parquet path (SQL window cumsum) matches known truth."""
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    estimator = ThresholdCDF(n_cells=1, n_genes=n_genes)
    output = tmp_path / "noise.parquet"
    estimator.estimate_noise_to_parquet(path, output, q=0.5)
    out_per_m = _read_noise_parquet(output)
    result = np.zeros(n_genes)
    result[: len(out_per_m)] = out_per_m
    np.testing.assert_array_equal(result, log_prob_parquet["cdfs"])


def test_single_sample_from_parquet(log_prob_parquet, tmp_path):
    """SingleSample from parquet path (Gumbel-max trick) returns valid noise counts."""
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    coo = log_prob_parquet["coo"]
    estimator = SingleSample(n_cells=1, n_genes=n_genes)
    output = tmp_path / "noise.parquet"
    estimator.estimate_noise_to_parquet(path, output)
    out_per_m = _read_noise_parquet(output)

    dense = log_prob_sparse_to_dense(coo)
    for i in range(1, coo.shape[0]):
        row = dense[i, :]
        if not np.any(row > -np.inf):
            continue
        allowed_vals = np.arange(dense.shape[1])[row > -np.inf]
        assert out_per_m[i] in allowed_vals, f"sample {out_per_m[i]} not in allowed set for m={i}"


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_cdf_parquet_coalesce_fallback(tmp_path_factory):
    """ThresholdCDF returns max(c) when the whole distribution's mass is below q."""
    tmp_dir = tmp_path_factory.mktemp("cdf_edge")
    parquet_path = tmp_dir / "posterior.parquet"
    output_path = tmp_dir / "noise.parquet"

    log_prob = float(np.log(0.3))
    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.array([0, 0], dtype=np.int32),
        gene_ids=np.array([0, 0], dtype=np.int32),
        c_vals=np.array([5, 10], dtype=np.int32),
        log_probs=np.array([log_prob, log_prob], dtype=np.float32),
    )
    writer.close()

    estimator = ThresholdCDF(n_cells=1, n_genes=1)
    estimator.estimate_noise_to_parquet(parquet_path, output_path, q=0.9)
    result = duckdb.connect().execute(f"SELECT noise_count FROM read_parquet('{output_path}')").fetchone()[0]
    assert result == 10


def test_mean_parquet_numerical_stability(tmp_path_factory):
    """Mean SQL path uses log-max shift so extreme log_probs don't cause NaN."""
    tmp_dir = tmp_path_factory.mktemp("mean_stability")
    parquet_path = tmp_dir / "posterior.parquet"
    output_path = tmp_dir / "noise.parquet"

    cs = np.array([3, 5], dtype=np.int32)
    log_probs = np.array([-1000.0, -1001.0], dtype=np.float32)

    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.zeros(2, dtype=np.int32),
        gene_ids=np.zeros(2, dtype=np.int32),
        c_vals=cs,
        log_probs=log_probs,
    )
    writer.close()

    Mean(n_cells=1, n_genes=1).estimate_noise_to_parquet(parquet_path, output_path)
    result = duckdb.connect().execute(f"SELECT noise_count FROM read_parquet('{output_path}')").fetchone()[0]

    assert np.isfinite(result), "Mean returned NaN/Inf for extreme log_probs"
    denom = 1.0 + np.exp(-1.0)
    expected = 3.0 / denom + 5.0 * np.exp(-1.0) / denom
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_compute_mean_target_removal_from_parquet(log_prob_parquet):
    """compute_mean_target_removal_as_function accepts a parquet Path."""
    path = log_prob_parquet["path"]
    n_genes = log_prob_parquet["n_genes"]
    raw_counts = sp.csr_matrix(np.ones((1, n_genes), dtype=np.float32))

    target_fun = compute_mean_target_removal_as_function(
        noise_count_posterior_coo=path,
        n_genes=n_genes,
        raw_count_csr_for_cells=raw_counts,
        n_cells=1,
        device="cpu",
        per_gene=True,
    )
    result = target_fun(0.01).numpy()
    assert result.shape == (n_genes,)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# MCKP tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mckp_parquet(log_prob_coo_base, tmp_path_factory) -> Dict:
    """Write the base COO (minus empty row 0) to parquet for MCKP tests.

    Row 0 in log_prob_coo_base is empty; MCKP needs a clean 1-cell / n_genes layout.
    One cell, n_genes genes: cell_id=0 for all entries, gene_id = re-indexed row value.
    """
    coo = log_prob_coo_base["coo"]
    # Drop the empty row 0 and re-index rows to 0-based
    mask = coo.row > 0
    new_row = coo.row[mask] - 1
    new_col = coo.col[mask]
    new_data = coo.data[mask]
    n_genes = coo.shape[0] - 1  # original shape[0] was 9 (8 rows + 1 empty)
    new_coo = sp.coo_matrix((new_data, (new_row, new_col)), shape=(n_genes, coo.shape[1]))

    tmp_dir = tmp_path_factory.mktemp("mckp_parquet")
    parquet_path = tmp_dir / "posterior.parquet"

    cell_ids = np.zeros(len(new_coo.row), dtype=np.int32)
    gene_ids = new_coo.row.astype(np.int32)
    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        c_vals=new_coo.col.astype(np.int32),
        log_probs=new_data.astype(np.float32),
    )
    writer.close()

    return {"path": parquet_path, "n_genes": n_genes, "coo": new_coo}


@pytest.mark.parametrize(
    "target, truth",
    (
        [np.zeros(8), np.array([0, 1, 2, 0, 0, 0, 0, 1])],
        [np.ones(8), np.array([0, 1, 2, 1, 1, 1, 1, 1])],
        [np.ones(8) * 2, np.array([0, 1, 2, 2, 2, 2, 2, 2])],
    ),
    ids=["target_0", "target_1", "target_2"],
)
def test_mckp_from_parquet(mckp_parquet, target, truth, tmp_path):
    """MCKP from parquet produces correct per-gene totals."""
    path = mckp_parquet["path"]
    n_genes = mckp_parquet["n_genes"]
    estimator = MultipleChoiceKnapsack(n_cells=1, n_genes=n_genes)
    output = tmp_path / "noise.parquet"
    estimator.estimate_noise_to_parquet(path, output, noise_targets_per_gene=target)

    df = duckdb.connect().execute(f"SELECT gene_id, noise_count FROM read_parquet('{output}') ORDER BY gene_id").df()
    out_per_gene = np.zeros(n_genes)
    out_per_gene[df["gene_id"].values] = df["noise_count"].values
    np.testing.assert_array_equal(out_per_gene, truth)


# ---------------------------------------------------------------------------
# int32 c-value regression: values that would overflow int16
# ---------------------------------------------------------------------------


def test_c_large_value_no_overflow(tmp_path_factory):
    """c values > 32767 (former int16 max) must round-trip without corruption.

    In the old int16 schema, c=33000 would silently become -32536.  With int32
    this must pass: MAP should return 33000, not a corrupted negative value.
    """
    tmp_dir = tmp_path_factory.mktemp("c_overflow")
    parquet_path = tmp_dir / "posterior.parquet"
    output_path = tmp_dir / "noise.parquet"

    # Two c values for one (cell, gene): 33000 is the MAP (higher log_prob).
    c_vals = np.array([32767, 33000], dtype=np.int32)
    log_probs = np.array([-1.0, -0.1], dtype=np.float32)

    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.zeros(2, dtype=np.int32),
        gene_ids=np.zeros(2, dtype=np.int32),
        c_vals=c_vals,
        log_probs=log_probs,
    )
    writer.close()

    MAP(n_cells=1, n_genes=1).estimate_noise_to_parquet(parquet_path, output_path)
    result = duckdb.connect().execute(f"SELECT noise_count FROM read_parquet('{output_path}')").fetchone()[0]
    assert result == 33000, f"Expected MAP=33000, got {result} (int16 overflow would give -32536)"


# ---------------------------------------------------------------------------
# MCKP: multi-cell and early-exit
# ---------------------------------------------------------------------------


def test_mckp_multi_cell(tmp_path_factory):
    """MCKP correctly ranks step candidates across multiple cells.

    3 cells × 2 genes.  Gene 0: MAP per-cell = [0, 1, 2], target total = 2.
    Gene 1: MAP per-cell = [3, 3, 3], target total = 7 (step down by 2 total).
    The knapsack must pick the two cells with smallest cost to decrease by 1 each.
    """
    tmp_dir = tmp_path_factory.mktemp("mckp_multicell")
    parquet_path = tmp_dir / "posterior.parquet"
    output_path = tmp_dir / "noise.parquet"

    # Gene 0: cell 0 → MAP=0 (only c=0); cell 1 → MAP=1 (only c=1);
    #          cell 2 → MAP=2 (c=1 log=-2, c=2 log=-0.1, c=3 log=-3).
    # Target total for gene 0 = 2:  MAP total = 0+1+2 = 3, need to step down by 1.
    # Only cell 2 can step down (has c < MAP).  Expected output: cell 2 → 1.
    # → per-cell gene-0 noise: [0, 1, 1], sum=2.

    # Gene 1: all cells have c=2 (log=-1) and c=3 (log=-0.1) [MAP=3].
    # Target total = 7, MAP total = 9, step down by 2.
    # Each cell can step from 3 to 2; cost = abs(-1 - (-0.1)) = 0.9 for each.
    # We pick 2 of the 3 cells — any 2 (all equal cost). Expected sum = 7.
    # → one cell stays at 3, two cells drop to 2.

    cell_ids, gene_ids, c_vals, log_probs = [], [], [], []

    def _add(cell, gene, c_arr, lp_arr):
        for c, lp in zip(c_arr, lp_arr):
            cell_ids.append(cell)
            gene_ids.append(gene)
            c_vals.append(c)
            log_probs.append(lp)

    # gene 0
    _add(0, 0, [0], [0.0])  # MAP=0, can't move
    _add(1, 0, [1], [0.0])  # MAP=1, can't move
    _add(2, 0, [1, 2, 3], [-2.0, -0.1, -3.0])  # MAP=2, can step down to 1

    # gene 1
    for cell in range(3):
        _add(cell, 1, [2, 3], [-1.0, -0.1])  # MAP=3 for all

    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.array(cell_ids, dtype=np.int32),
        gene_ids=np.array(gene_ids, dtype=np.int32),
        c_vals=np.array(c_vals, dtype=np.int32),
        log_probs=np.array(log_probs, dtype=np.float32),
    )
    writer.close()

    targets = np.array([2.0, 7.0])
    estimator = MultipleChoiceKnapsack(n_cells=3, n_genes=2)
    estimator.estimate_noise_to_parquet(parquet_path, output_path, noise_targets_per_gene=targets)

    df = (
        duckdb.connect()
        .execute(
            f"SELECT gene_id, SUM(noise_count) AS total "
            f"FROM read_parquet('{output_path}') GROUP BY gene_id ORDER BY gene_id"
        )
        .df()
    )
    totals = {int(row.gene_id): int(row.total) for row in df.itertuples()}

    assert totals[0] == 2, f"Gene 0 total should be 2, got {totals[0]}"
    assert totals[1] == 7, f"Gene 1 total should be 7, got {totals[1]}"


def test_mckp_early_exit_when_map_matches_targets(tmp_path_factory):
    """When MAP totals already equal all targets, the early-exit path is taken."""
    tmp_dir = tmp_path_factory.mktemp("mckp_early")
    parquet_path = tmp_dir / "posterior.parquet"
    output_path = tmp_dir / "noise.parquet"

    # 2 genes, 1 cell: MAP = [3, 5].  Targets = [3.0, 5.0] → no adjustment needed.
    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.zeros(4, dtype=np.int32),
        gene_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        c_vals=np.array([2, 3, 4, 5], dtype=np.int32),
        log_probs=np.array([-1.0, -0.1, -1.0, -0.1], dtype=np.float32),
    )
    writer.close()

    targets = np.array([3.0, 5.0])
    estimator = MultipleChoiceKnapsack(n_cells=1, n_genes=2)
    estimator.estimate_noise_to_parquet(parquet_path, output_path, noise_targets_per_gene=targets)

    df = (
        duckdb.connect()
        .execute(f"SELECT gene_id, noise_count FROM read_parquet('{output_path}') ORDER BY gene_id")
        .df()
    )
    assert list(df["noise_count"]) == [3, 5], f"Expected [3, 5], got {list(df['noise_count'])}"


@pytest.mark.parametrize("chunk_size", [1, 2], ids=["chunk_1", "chunk_2"])
def test_mckp_chunked_correctness(tmp_path_factory, monkeypatch, chunk_size):
    """MCKP gives correct, sorted results for different chunk sizes.

    Uses the same 3-cell × 2-gene fixture as test_mckp_multi_cell.

    chunk_size=1: one gene per chunk — exercises the chunked-loop path, including
      the step-down case (gene 1) and the early-exit case (gene 0, MAP matches
      target=2 after step-down of one cell).
    chunk_size=2 (= n_genes): single chunk — exercises the "no real chunking"
      path that should produce the same result as the original monolithic approach.

    Both cases also verify that the output parquet rows are sorted by
    (gene_id, cell_id), since the chunked writer concatenates without a final sort.
    """
    monkeypatch.setattr(est_mod, "_mckp_chunk_size", lambda source, n_genes: chunk_size)

    tmp_dir = tmp_path_factory.mktemp(f"mckp_chunked_{chunk_size}")
    parquet_path = tmp_dir / "posterior.parquet"
    output_path = tmp_dir / "noise.parquet"

    cell_ids, gene_ids, c_vals, log_probs = [], [], [], []

    def _add(cell, gene, c_arr, lp_arr):
        for c, lp in zip(c_arr, lp_arr):
            cell_ids.append(cell)
            gene_ids.append(gene)
            c_vals.append(c)
            log_probs.append(lp)

    # gene 0: MAP per cell = [0, 1, 2], target total = 2 → step-down cell 2 by 1
    _add(0, 0, [0], [0.0])
    _add(1, 0, [1], [0.0])
    _add(2, 0, [1, 2, 3], [-2.0, -0.1, -3.0])

    # gene 1: MAP per cell = [3, 3, 3], target total = 7 → step-down 2 of 3 cells
    for cell in range(3):
        _add(cell, 1, [2, 3], [-1.0, -0.1])

    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.array(cell_ids, dtype=np.int32),
        gene_ids=np.array(gene_ids, dtype=np.int32),
        c_vals=np.array(c_vals, dtype=np.int32),
        log_probs=np.array(log_probs, dtype=np.float32),
    )
    writer.close()

    targets = np.array([2.0, 7.0])
    estimator = MultipleChoiceKnapsack(n_cells=3, n_genes=2)
    estimator.estimate_noise_to_parquet(parquet_path, output_path, noise_targets_per_gene=targets)

    # Correctness: per-gene totals.
    totals_df = (
        duckdb.connect()
        .execute(
            f"SELECT gene_id, SUM(noise_count) AS total "
            f"FROM read_parquet('{output_path}') GROUP BY gene_id ORDER BY gene_id"
        )
        .df()
    )
    totals = {int(row.gene_id): int(row.total) for row in totals_df.itertuples()}
    assert totals[0] == 2, f"Gene 0 total should be 2, got {totals[0]}"
    assert totals[1] == 7, f"Gene 1 total should be 7, got {totals[1]}"

    # Sort order: read raw parquet row order (no ORDER BY) and verify.
    raw = pq.read_table(output_path)
    gene_arr = raw.column("gene_id").to_pylist()
    cell_arr = raw.column("cell_id").to_pylist()
    assert gene_arr == sorted(gene_arr), f"chunk_size={chunk_size}: output not sorted by gene_id: {gene_arr}"
    for g in sorted(set(gene_arr)):
        cells_for_gene = [cell_arr[i] for i, gv in enumerate(gene_arr) if gv == g]
        assert cells_for_gene == sorted(cells_for_gene), (
            f"chunk_size={chunk_size}: cell_id not sorted within gene {g}: {cells_for_gene}"
        )


def test_mckp_chunk_size_estimation(tmp_path_factory):
    """_mckp_chunk_size returns a value in [1, n_genes]."""
    tmp_dir = tmp_path_factory.mktemp("chunk_size")
    parquet_path = tmp_dir / "posterior.parquet"

    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.zeros(6, dtype=np.int32),
        gene_ids=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        c_vals=np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        log_probs=np.array([-1.0, -2.0, -1.0, -2.0, -1.0, -2.0], dtype=np.float32),
    )
    writer.close()

    n_genes = 3
    cs = _mckp_chunk_size(parquet_path, n_genes)
    assert 1 <= cs <= n_genes, f"chunk_size={cs} out of [1, {n_genes}]"

    # Edge case: single gene
    assert _mckp_chunk_size(parquet_path, 1) == 1


# ---------------------------------------------------------------------------
# ThresholdCDF boundary q values
# ---------------------------------------------------------------------------


def test_cdf_q_zero(tmp_path_factory):
    """q=0.0 returns the minimum c in the stored posterior window."""
    tmp_dir = tmp_path_factory.mktemp("cdf_q0")
    parquet_path = tmp_dir / "posterior.parquet"
    output_path = tmp_dir / "noise.parquet"

    # c values 5, 10, 15 with most mass at c=15.  q=0 → min c = 5.
    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.zeros(3, dtype=np.int32),
        gene_ids=np.zeros(3, dtype=np.int32),
        c_vals=np.array([5, 10, 15], dtype=np.int32),
        log_probs=np.array([-3.0, -2.0, -0.1], dtype=np.float32),
    )
    writer.close()

    ThresholdCDF(n_cells=1, n_genes=1).estimate_noise_to_parquet(parquet_path, output_path, q=0.0)
    result = duckdb.connect().execute(f"SELECT noise_count FROM read_parquet('{output_path}')").fetchone()[0]
    assert result == 5, f"q=0.0 should return minimum c=5, got {result}"


def test_cdf_q_one(tmp_path_factory):
    """q=1.0 triggers the COALESCE fallback and returns the maximum c."""
    tmp_dir = tmp_path_factory.mktemp("cdf_q1")
    parquet_path = tmp_dir / "posterior.parquet"
    output_path = tmp_dir / "noise.parquet"

    # Three entries that sum to < 1.0 (windowed posterior).  q=1.0 → COALESCE → max c.
    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.zeros(3, dtype=np.int32),
        gene_ids=np.zeros(3, dtype=np.int32),
        c_vals=np.array([5, 10, 15], dtype=np.int32),
        log_probs=np.array([-3.0, -2.0, -0.1], dtype=np.float32),
    )
    writer.close()

    ThresholdCDF(n_cells=1, n_genes=1).estimate_noise_to_parquet(parquet_path, output_path, q=1.0)
    result = duckdb.connect().execute(f"SELECT noise_count FROM read_parquet('{output_path}')").fetchone()[0]
    assert result == 15, f"q=1.0 should return maximum c=15 via COALESCE, got {result}"


# ---------------------------------------------------------------------------
# Gene absent from posterior → absent from estimation output
# ---------------------------------------------------------------------------


def test_gene_absent_from_posterior_is_absent_from_output(tmp_path_factory):
    """A gene with no posterior entries should not appear in the estimation output.

    Downstream code treats missing genes as 0 noise.  This test confirms the
    contract: absent gene → absent row (not a zero row).
    """
    tmp_dir = tmp_path_factory.mktemp("absent_gene")
    parquet_path = tmp_dir / "posterior.parquet"
    output_path = tmp_dir / "noise.parquet"

    # Only gene 0 has posterior entries; gene 1 is absent.
    writer = pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA)
    write_posterior_batch_to_parquet(
        writer=writer,
        cell_ids=np.zeros(2, dtype=np.int32),
        gene_ids=np.zeros(2, dtype=np.int32),
        c_vals=np.array([0, 1], dtype=np.int32),
        log_probs=np.array([-0.5, -1.0], dtype=np.float32),
    )
    writer.close()

    MAP(n_cells=1, n_genes=2).estimate_noise_to_parquet(parquet_path, output_path)
    df = duckdb.connect().execute(f"SELECT gene_id FROM read_parquet('{output_path}')").df()
    present_genes = set(df["gene_id"].tolist())
    assert present_genes == {0}, f"Only gene 0 should be in output, got {present_genes}"
    assert 1 not in present_genes, "Gene 1 has no posterior entries and must be absent from output"


# ---------------------------------------------------------------------------
# _make_duckdb_conn: auto memory limit
# ---------------------------------------------------------------------------


def test_make_duckdb_conn_auto_detects_memory_limit(tmp_path):
    """_make_duckdb_conn without memory_limit sets a limit well below total RAM."""

    conn = _make_duckdb_conn(str(tmp_path))
    row = conn.execute("SELECT current_setting('memory_limit')").fetchone()
    assert row is not None, "memory_limit not set"
    # DuckDB reports the limit as a human-readable string (e.g. '7.5GB').
    # We just verify it was set (non-empty and not the DuckDB sentinel '-1').
    limit_str = str(row[0])
    assert limit_str not in ("", "-1"), f"memory_limit looks unset: {limit_str!r}"

    # Verify explicit override is respected.
    conn2 = _make_duckdb_conn(str(tmp_path), memory_limit="512MB")
    row2 = conn2.execute("SELECT current_setting('memory_limit')").fetchone()
    # DuckDB may report "512MB" as "488.3MiB" or similar; just verify it's non-default.
    limit_str2 = str(row2[0])
    assert limit_str2 not in ("", "-1"), f"Explicit limit not applied: {row2[0]!r}"


# ---------------------------------------------------------------------------
# estimate_mean_noise_per_gene: streaming scan correctness
# ---------------------------------------------------------------------------


def _make_sorted_posterior(tmp_dir, rows):
    """Write rows as a sorted (gene_id, cell_id, c, log_prob) parquet."""
    parquet_path = tmp_dir / "posterior.parquet"
    rows_sorted = sorted(rows, key=lambda r: (r[0], r[1], r[2]))  # gene, cell, c
    gene_ids = np.array([r[0] for r in rows_sorted], dtype=np.int32)
    cell_ids = np.array([r[1] for r in rows_sorted], dtype=np.int32)
    c_vals = np.array([r[2] for r in rows_sorted], dtype=np.int32)
    lp_vals = np.array([r[3] for r in rows_sorted], dtype=np.float32)
    with pq.ParquetWriter(str(parquet_path), POSTERIOR_SCHEMA) as writer:
        write_posterior_batch_to_parquet(writer, cell_ids, gene_ids, c_vals, lp_vals)
    return parquet_path


def _reference_mean_noise(rows: list[tuple[int, int, int, float]], n_genes: int) -> np.ndarray:
    """Brute-force per-gene total mean noise (Python reference, no parquet)."""
    from collections import defaultdict

    pairs: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for gene_id, cell_id, c, lp in rows:
        pairs[(gene_id, cell_id)].append((c, lp))
    result = np.zeros(n_genes, dtype=np.float64)
    for (gene_id, cell_id), entries in pairs.items():
        cs = np.array([e[0] for e in entries], dtype=np.float64)
        lps = np.array([e[1] for e in entries], dtype=np.float64)
        max_lp = np.max(lps)
        shifted = np.exp(lps - max_lp)
        result[gene_id] += float(np.sum(cs * shifted) / np.sum(shifted))
    return result


@pytest.mark.parametrize("batch_size", [1, 3, 1_000_000], ids=["batch1", "batch3", "batchlarge"])
def test_estimate_mean_noise_per_gene_streaming(tmp_path_factory, batch_size, monkeypatch):
    """Streaming scan matches brute-force reference at various batch sizes.

    batch_size=1 forces a new batch boundary after every single row, exercising
    the cross-batch pending-state logic on every step.
    batch_size=3 causes boundaries mid-group for some groups.
    batch_size=large exercises the fully-vectorised path.
    """
    import cellbender.remove_background.estimation as est_mod

    monkeypatch.setattr(est_mod, "_STREAM_BATCH_SIZE", batch_size)

    tmp_dir = tmp_path_factory.mktemp("stream_mean")

    # 4 genes, 3 cells; hand-crafted log_prob values (not normalised to 1).
    # The log-sum-exp trick must handle absolute log_probs as low as -500.
    rows = [
        # gene 0, cell 0: two c values
        (0, 0, 0, -1.0),
        (0, 0, 1, -0.5),
        # gene 0, cell 1: single c value → mean = that c
        (0, 1, 3, -0.1),
        # gene 0, cell 2: very negative absolute log_prob (tests log-sum-exp)
        (0, 2, 2, -500.0),
        (0, 2, 4, -501.0),
        # gene 1, cell 0: single entry
        (1, 0, 5, -2.0),
        # gene 1, cell 1: two entries
        (1, 1, 1, -3.0),
        (1, 1, 2, -1.0),
        # gene 2, cell 2 only
        (2, 2, 7, -0.3),
        (2, 2, 8, -0.8),
        # gene 3: absent from posterior → should be zero
    ]
    n_genes = 4

    parquet_path = _make_sorted_posterior(tmp_dir, rows)
    expected = _reference_mean_noise(rows, n_genes)
    result = estimate_mean_noise_per_gene(parquet_path, n_genes)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-5,
        err_msg=f"Streaming result mismatch at batch_size={batch_size}",
    )
    assert result[3] == 0.0, "Gene 3 has no posterior entries; result must be 0"
