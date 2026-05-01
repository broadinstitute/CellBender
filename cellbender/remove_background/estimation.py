"""Classes and methods for estimation of noise counts, given a posterior."""

import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Generator, Optional, Tuple, Union

if TYPE_CHECKING:
    from cellbender.remove_background.posterior import IndexConverter

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp
import torch
from torch.distributions.categorical import Categorical

from cellbender.remove_background.sparse_utils import log_prob_sparse_to_dense

logger = logging.getLogger("cellbender")

N_CELLS_DATATYPE = np.int32
N_GENES_DATATYPE = np.int32
COUNT_DATATYPE = np.int32

PosteriorSource = Union[sp.coo_matrix, Path]


class EstimationMethod(ABC):
    """Base class for estimation of noise counts, given a posterior."""

    def __init__(self, index_converter: "IndexConverter"):
        """Instantiate the EstimationMethod.
        Args:
            index_converter: The IndexConverter that can be used to translate
                back and forth between count matrix (n, g) indices, and the
                flattened, generalized 'm' indices that index the posterior COO.
        """
        self.index_converter = index_converter
        super(EstimationMethod, self).__init__()

    @abstractmethod
    def estimate_noise(
        self, noise_log_prob_coo: sp.coo_matrix, noise_offsets: Optional[Dict[int, int]], **kwargs
    ) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts.
        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_offsets: Noise count offset values keyed by 'm'.
        """
        pass

    def _estimation_array_to_csr(
        self, data: np.ndarray, m: np.ndarray, noise_offsets: Optional[Dict[int, int]], dtype=COUNT_DATATYPE
    ) -> sp.csr_matrix:
        """Say you have point estimates for each count matrix element (data) and
        you have the 'm'-indices for each value (m). This returns a CSR matrix
        that has the shape of the count matrix, where duplicate entries have
        been summed.

        Args:
            data: Point estimates for each nonzero entry of the count matrix, in
                a flat format, indexed by 'm'.
            m: Array of the same length as data, where each entry is an m-index.
            noise_offsets: Noise count offset values keyed by 'm'.
            dtype: Data type for sparse matrix. Int32 is too small for 'm' indices.

        Results:
            noise_csr: Noise point estimate, as a CSR sparse matrix.

        """
        return _estimation_array_to_csr(
            index_converter=self.index_converter,
            data=data,
            m=m,
            noise_offsets=noise_offsets,
            dtype=dtype,
        )
        # row, col = self.index_converter.get_ng_indices(m_inds=m)
        # if noise_offsets is not None:
        #     data = data + np.array([noise_offsets[i] for i in m])
        # coo = sp.coo_matrix((data.astype(dtype), (row.astype(dtype), col.astype(dtype))),
        #                     shape=self.index_converter.matrix_shape, dtype=dtype)
        # coo.sum_duplicates()
        # return coo.tocsr()


class SingleSample(EstimationMethod):
    """A single sample from the noise count posterior"""

    @torch.no_grad()
    def estimate_noise(
        self, noise_log_prob_coo: sp.coo_matrix, noise_offsets: Optional[Dict[int, int]], device: str = "cpu", **kwargs
    ) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts by
        taking a single sample from each probability distribution.

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_offsets: Noise count offset values keyed by 'm'.
            device: ['cpu', 'cuda'] - whether to perform the pytorch sampling
                operation on CPU or GPU. It's pretty fast on CPU already.

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """

        def _torch_sample(x):
            return Categorical(logits=x, validate_args=False).sample()

        result = apply_function_dense_chunks(noise_log_prob_coo=noise_log_prob_coo, fun=_torch_sample, device=device)
        return self._estimation_array_to_csr(data=result["result"], m=result["m"], noise_offsets=noise_offsets)


class Mean(EstimationMethod):
    """Posterior mean"""

    def estimate_noise(
        self, noise_log_prob_coo: sp.coo_matrix, noise_offsets: Optional[Dict[int, int]], device: str = "cpu", **kwargs
    ) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts by
        taking the mean of each probability distribution.

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_offsets: Noise count offset values keyed by 'm'.

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """
        # c = torch.arange(noise_log_prob_coo.shape[1], dtype=float).to(device).t()

        def _torch_mean(x):
            c = torch.arange(x.shape[1], dtype=float).to(x.device)
            return torch.matmul(x.exp(), c.t())

        result = apply_function_dense_chunks(noise_log_prob_coo=noise_log_prob_coo, fun=_torch_mean, device=device)
        return self._estimation_array_to_csr(
            data=result["result"], m=result["m"], noise_offsets=noise_offsets, dtype=np.float32
        )


class MAP(EstimationMethod):
    """The canonical maximum a posteriori"""

    @staticmethod
    def torch_argmax(x):
        return x.argmax(dim=-1)

    def estimate_noise(
        self, noise_log_prob_coo: sp.coo_matrix, noise_offsets: Optional[Dict[int, int]], device: str = "cpu", **kwargs
    ) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts by
        taking the maximum a posteriori (MAP) of each probability distribution.

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_offsets: Noise count offset values keyed by 'm'.
            device: ['cpu', 'cuda'] - whether to perform the pytorch argmax
                operation on CPU or GPU. It's pretty fast on CPU already.

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """
        result = apply_function_dense_chunks(
            noise_log_prob_coo=noise_log_prob_coo, fun=self.torch_argmax, device=device
        )
        return self._estimation_array_to_csr(data=result["result"], m=result["m"], noise_offsets=noise_offsets)


class ThresholdCDF(EstimationMethod):
    """Noise estimation via thresholding the noise count CDF"""

    @staticmethod
    def torch_cdf_fun(x: torch.Tensor, q: float):
        return (x.exp().cumsum(dim=-1) <= q).sum(dim=-1)

    def estimate_noise(
        self,
        noise_log_prob_coo: sp.coo_matrix,
        noise_offsets: Optional[Dict[int, int]],
        q: float = 0.5,
        device: str = "cpu",
        **kwargs,
    ) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_offsets: Noise count offset values keyed by 'm'.
            q: The CDF threshold value.

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """
        result = apply_function_dense_chunks(
            noise_log_prob_coo=noise_log_prob_coo, fun=self.torch_cdf_fun, device=device, q=q
        )
        return self._estimation_array_to_csr(data=result["result"], m=result["m"], noise_offsets=noise_offsets)


def _estimation_array_to_csr(
    index_converter, data: np.ndarray, m: np.ndarray, noise_offsets: Optional[Dict[int, int]], dtype=COUNT_DATATYPE
) -> sp.csr_matrix:
    """Say you have point estimates for each count matrix element (data) and
    you have the 'm'-indices for each value (m). This returns a CSR matrix
    that has the shape of the count matrix, where duplicate entries have
    been summed.

    Args:
        data: Point estimates for each nonzero entry of the count matrix, in
            a flat format, indexed by 'm'.
        m: Array of the same length as data, where each entry is an m-index.
        noise_offsets: Noise count offset values keyed by 'm'.
        dtype: Data type for values of sparse matrix

    Results:
        noise_csr: Noise point estimate, as a CSR sparse matrix.

    """
    row, col = index_converter.get_ng_indices(m_inds=m)
    if noise_offsets is not None:
        data = data + np.array([noise_offsets.get(i, 0) for i in m])
    coo = sp.coo_matrix(
        (data.astype(dtype), (row.astype(N_CELLS_DATATYPE), col.astype(N_GENES_DATATYPE))),
        shape=index_converter.matrix_shape,
        dtype=dtype,
    )
    coo.sum_duplicates()
    return coo.tocsr()


def _register_posterior(
    conn: "duckdb.DuckDBPyConnection",
    source: "PosteriorSource",
    noise_offsets: Dict[int, int],
    index_converter: "IndexConverter",
) -> None:
    """Register the posterior data as a DuckDB relation named 'posterior'.

    When *source* is a :class:`pathlib.Path` the parquet file is registered as
    a zero-copy view (offsets already embedded as the ``offset`` column).
    When *source* is a :class:`scipy.sparse.coo_matrix` the matrix is
    converted to a PyArrow Table and registered in-process.
    """
    if isinstance(source, Path):
        conn.execute(f"CREATE OR REPLACE VIEW posterior AS SELECT * FROM read_parquet('{source}')")
    else:
        # COO matrix path: build Arrow table with embedded offsets
        coo = source
        rows = coo.row.astype("int32")
        cols = coo.col.astype("int16")
        data = coo.data.astype("float32")
        cell_ids, gene_ids = index_converter.get_ng_indices(m_inds=rows)
        offsets = np.array([noise_offsets.get(int(m), 0) for m in rows], dtype="int16")
        table = pa.table(
            {
                "cell_id": pa.array(cell_ids.astype("int32"), type=pa.int32()),
                "gene_id": pa.array(gene_ids.astype("int32"), type=pa.int32()),
                "c": pa.array(cols, type=pa.int16()),
                "noise_offset": pa.array(offsets, type=pa.int16()),
                "log_prob": pa.array(data, type=pa.float32()),
                "regularized": pa.array(np.zeros(len(rows), dtype=bool), type=pa.bool_()),
            }
        )
        conn.register("posterior", table)


def _ng_arrays_to_csr(
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    data: np.ndarray,
    shape: Tuple[int, int],
    dtype=COUNT_DATATYPE,
) -> sp.csr_matrix:
    """Build a CSR sparse matrix from flat (cell_id, gene_id, data) arrays."""
    coo = sp.coo_matrix(
        (data.astype(dtype), (cell_ids.astype(np.int32), gene_ids.astype(np.int32))),
        shape=shape,
        dtype=dtype,
    )
    coo.sum_duplicates()
    return coo.tocsr()


class MultipleChoiceKnapsack(EstimationMethod):
    """Noise estimation via the multiple-choice knapsack problem, solved with DuckDB SQL.

    DuckDB executes out-of-core by default so this scales to datasets with
    millions of non-zero posterior entries without incurring OOM errors.
    """

    def estimate_noise(
        self,
        noise_log_prob_coo: "PosteriorSource",
        noise_offsets: Optional[Dict[int, int]],
        noise_targets_per_gene: Optional[np.ndarray] = None,
        verbose: bool = False,
        n_chunks: Optional[int] = None,
        use_multiple_processes: bool = False,
        duckdb_memory_limit: str = "32GB",
        **kwargs,
    ) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts via MCKP.

        Args:
            noise_log_prob_coo: Either a (m, c) COO matrix with log probabilities,
                or a Path to a posterior parquet file written by CellBender.
            noise_offsets: Noise count offset values keyed by 'm' (only used when
                source is a COO matrix; ignored for parquet since offsets are embedded).
            noise_targets_per_gene: Integer noise count target per gene (required).
            verbose: Print intermediate DuckDB results for debugging.
            n_chunks: Ignored; DuckDB handles memory management internally.
            use_multiple_processes: Ignored; DuckDB is multi-threaded internally.
            duckdb_memory_limit: DuckDB memory cap (e.g. '8GB'). DuckDB spills to
                disk when this limit is reached.

        Returns:
            noise_count_csr: Estimated noise count CSR matrix.
        """
        assert noise_targets_per_gene is not None, (
            "noise_targets_per_gene is required for MultipleChoiceKnapsack.estimate_noise"
        )

        # When source is a parquet Path, offsets are embedded — no external dict needed.
        if isinstance(noise_log_prob_coo, Path):
            noise_offsets = {}

        if noise_offsets is None:
            noise_offsets = {}

        t0 = time.time()

        conn = duckdb.connect()
        conn.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        _register_posterior(conn, noise_log_prob_coo, noise_offsets, self.index_converter)

        # Step 1: MAP estimate — argmax of log_prob per (cell_id, gene_id)
        map_df = conn.execute("""
            SELECT
                cell_id,
                gene_id,
                CAST(argmax(c, log_prob) AS INTEGER) AS map_c,
                CAST(any_value(noise_offset) AS INTEGER) AS noise_offset
            FROM posterior
            WHERE NOT regularized
            GROUP BY cell_id, gene_id
        """).df()

        if verbose:
            logger.debug("MAP head:\n%s", map_df.head(10).to_string())

        # MAP noise counts per gene (offset-adjusted)
        map_csr = _ng_arrays_to_csr(
            cell_ids=map_df["cell_id"].values,
            gene_ids=map_df["gene_id"].values,
            data=(map_df["map_c"].values + map_df["noise_offset"].values).astype(COUNT_DATATYPE),
            shape=self.index_converter.matrix_shape,
        )
        map_noise_per_gene = np.asarray(map_csr.sum(axis=0)).squeeze()
        additional = (noise_targets_per_gene - map_noise_per_gene).astype(int)
        step_dir = np.sign(additional).astype(np.int32)
        topk = np.abs(additional).astype(np.int64)

        # Step 2: Build gene-level targets table
        all_gene_ids = np.arange(self.index_converter.total_n_genes, dtype=np.int32)
        gene_targets_df = pd.DataFrame(
            {
                "gene_id": all_gene_ids,
                "step_direction": step_dir,
                "topk": topk,
            }
        )
        gene_targets_df = gene_targets_df[gene_targets_df["step_direction"] != 0].reset_index(drop=True)

        if len(gene_targets_df) == 0:
            logger.info("MCKP: MAP already matches targets for all genes.")
            logger.info("Total MCKP time = %.2f sec", time.time() - t0)
            return map_csr

        conn.register("gene_targets", gene_targets_df)
        conn.register("map_estimates", map_df[["cell_id", "gene_id", "map_c"]])

        if verbose:
            logger.debug(
                "Positive-step genes: %d  Negative-step genes: %d",
                (step_dir > 0).sum(),
                (step_dir < 0).sum(),
            )

        # Step 3: Compute deltas and select cheapest topk steps per gene
        steps_df = conn.execute("""
            WITH directed AS (
                SELECT
                    p.cell_id,
                    p.gene_id,
                    p.c,
                    p.log_prob,
                    t.step_direction,
                    t.topk,
                    m.map_c,
                    LAG(p.log_prob)  OVER w AS lag_lp,
                    LEAD(p.log_prob) OVER w AS lead_lp
                FROM posterior p
                JOIN gene_targets  t ON p.gene_id = t.gene_id
                JOIN map_estimates m ON p.cell_id  = m.cell_id
                                    AND p.gene_id  = m.gene_id
                WHERE NOT p.regularized
                  AND (
                        (t.step_direction =  1 AND p.c >= m.map_c)
                     OR (t.step_direction = -1 AND p.c <= m.map_c)
                  )
                WINDOW w AS (PARTITION BY p.cell_id, p.gene_id ORDER BY p.c)
            ),
            deltas AS (
                SELECT
                    cell_id,
                    gene_id,
                    step_direction,
                    topk,
                    ABS(
                        CASE WHEN step_direction =  1 THEN log_prob - lag_lp
                             ELSE                          log_prob - lead_lp
                        END
                    ) AS delta
                FROM directed
                WHERE (step_direction =  1 AND lag_lp  IS NOT NULL)
                   OR (step_direction = -1 AND lead_lp IS NOT NULL)
            ),
            ranked AS (
                SELECT
                    cell_id,
                    gene_id,
                    step_direction,
                    topk,
                    ROW_NUMBER() OVER (PARTITION BY gene_id ORDER BY delta) AS rn
                FROM deltas
            )
            SELECT
                cell_id,
                gene_id,
                CAST(COUNT(*) AS INTEGER) * any_value(step_direction) AS step_counts
            FROM ranked
            WHERE rn <= topk
            GROUP BY cell_id, gene_id
        """).df()

        if verbose:
            logger.debug("Steps head:\n%s", steps_df.head(10).to_string())

        logger.info("Total MCKP estimation time = %.2f sec", time.time() - t0)

        if len(steps_df) == 0:
            return map_csr

        steps_csr = _ng_arrays_to_csr(
            cell_ids=steps_df["cell_id"].values,
            gene_ids=steps_df["gene_id"].values,
            data=steps_df["step_counts"].values,
            shape=self.index_converter.matrix_shape,
        )
        return map_csr + steps_csr


def chunked_iterator(
    coo: sp.coo_matrix, max_dense_batch_size_GB: float = 1.0
) -> Generator[Tuple[sp.coo_matrix, np.ndarray, np.ndarray], None, None]:
    """Return an iterator which yields the full dataset in chunks.

    NOTE: Idea is to prevent memory overflow. The use case is for worst-case
    scenario algorithms that have to make things into dense matrix chunks in
    order to do their compute.

    Args:
        coo: Sparse COO matrix with rows as generalized 'm'-indices and
            columns as noise count values.
        max_dense_batch_size_GB: Size of a batch on disk, in gigabytes.

    Returns:
        A generator that yields compact CSR sparse matrices until the whole dataset
        has been yielded. "Compact" in the sense that if they are made dense, there
        will be no all-zero rows.
            Tuple[chunk csr, actual row values in the full matrix]

    """
    n_elements_in_batch = max_dense_batch_size_GB * 1e9 / 4  # torch float32 is 4 bytes
    batch_size = max(1, int(np.floor(n_elements_in_batch / coo.shape[1])))

    # COO rows are not necessarily contiguous or in order
    unique_m_values = np.unique(coo.row)
    n_chunks = max(1, len(unique_m_values) // batch_size)
    row_m_value_chunks = np.array_split(unique_m_values, n_chunks)
    coo_row_series = pd.Series(coo.row)

    for row_m_values in row_m_value_chunks:
        logic = coo_row_series.isin(set(row_m_values))
        # Map these row values to a compact set of unique integers
        unique_row_values, rows = np.unique(coo.row[logic], return_inverse=True)
        unique_col_values, cols = np.unique(coo.col[logic], return_inverse=True)
        chunk_coo = sp.coo_matrix(
            (coo.data[logic], (rows, cols)),
            shape=(len(unique_row_values), len(unique_col_values)),
        )
        yield (chunk_coo, unique_row_values, unique_col_values)


def apply_function_dense_chunks(
    noise_log_prob_coo: sp.coo_matrix, fun: Callable[..., torch.Tensor], device: str = "cpu", **kwargs
) -> Dict[str, np.ndarray]:
    """Uses chunked_iterator to densify chunked portions of a COO sparse
    matrix and then applies a function to the dense chunks, keeping track
    of the results per row.

    NOTE: The function should produce one value per row of the dense matrix.
          The COO should contain log probability in data.

    Args:
        noise_log_prob_coo: The posterior noise count log prob data structure,
            indexed by 'm' as rows
        fun: Pytorch function that operates on a dense tensor and produces
            one value per row
        device: ['cpu', 'cuda'] - whether to perform the pytorch sampling
            operation on CPU or GPU. It's pretty fast on CPU already.
        **kwargs: Passed to fun

    Returns:
        Dict containing
            'm': np.ndarray of indices
            'result': the values computed by the function

    """
    array_length = len(np.unique(noise_log_prob_coo.row))

    m = np.zeros(array_length, dtype=np.uint64)
    out = np.zeros(array_length)
    a = 0

    for coo, row, col in chunked_iterator(coo=noise_log_prob_coo):
        dense_tensor = torch.tensor(log_prob_sparse_to_dense(coo)).to(device)
        if torch.numel(dense_tensor) == 0:
            # github issue 207
            continue
        s = fun(dense_tensor, **kwargs)
        if s.ndim == 0:
            # avoid "TypeError: len() of a 0-d tensor"
            len_s = 1
        else:
            len_s = len(s)
        m[a : (a + len_s)] = row
        out[a : (a + len_s)] = s.detach().cpu().numpy()
        a = a + len_s

    return {"m": m, "result": out}


def pandas_grouped_apply(
    coo: sp.coo_matrix,
    fun: Callable[[pd.DataFrame], Union[int, float]],
    extra_data: Optional[Dict[str, np.ndarray]] = None,
    sort_first: bool = False,
    parallel: bool = False,
) -> Dict[str, np.ndarray]:
    """Apply function on a sparse COO format (noise log probs) to compute output
    noise counts using pandas groupby and apply operations on CPU.

    TODO: consider numpy-groupies or np.ufunc.reduceat or similar
    https://stackoverflow.com/questions/31790819/scipy-sparse-csr-matrix-how-to-get-top-ten-values-and-indices

    Args:
        coo: COO data structure: (m, c) with 'm'-indexing.
        fun: Function to be applied with pandas .apply(): turns
            all the values for a matrix entry (m, :) into one number.
        extra_data: To include extra information other than 'm', 'c', and the
            'log_prob', you can pass in a dict here with array values the same
            length as 'm' and in the same order.
        sort_first: Sort the whole dataframe by ['m', 'c'] before applying
            function (much faster than sorting inside the function call).
        parallel: Use multiprocessing to run on all cores. This is 4x slower for
            a dataset of 300 cells.  Not clear if it's faster for larger data.

    Returns:
        output_csr: The aggregated sparse matrix, in row format
    """
    df = pd.DataFrame(data={"m": coo.row, "c": coo.col, "log_prob": coo.data})
    if extra_data is not None:
        for k, v in extra_data.items():
            df[k] = v
    if sort_first:
        df = df.sort_values(by=["m", "c"], ascending=[True, True])
    if parallel:
        m, result_per_m = _parallel_pandas_apply(df_grouped=df.groupby("m"), fun=fun)
    else:
        df = df.groupby("m").apply(fun).reset_index()
        m = df["m"].values
        result_per_m = df[0].values
    return {"m": m, "result": result_per_m}


def _parallel_pandas_apply(
    df_grouped: pd.core.groupby.DataFrameGroupBy, fun: Callable[[pd.DataFrame], Union[int, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Use multiprocessing to apply a function to a grouped dataframe in pandas.

    Args:
        df_grouped: Grouped dataframe, as in df.groupby('m')
        fun: Function to be applied to dataframe, as in df.groupby('m').apply(fun)

    Returns:
        Tuple of (groupby_value, groupby_apply_output_for_each_value)

    NOTE: see https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
    """
    groupby_val, group_df_list = zip(*[(group_val, group_df) for group_val, group_df in df_grouped])
    with mp.Pool(mp.cpu_count()) as p:
        output_list = p.map(fun, group_df_list)
    return np.array(groupby_val), np.array(output_list)


def _subset_coo(coo: sp.coo_matrix, logic: np.ndarray) -> sp.coo_matrix:
    return sp.coo_matrix((coo.data[logic], (coo.row[logic], coo.col[logic])))


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
