"""Handle input parsing and output writing."""

import gzip
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import anndata
import duckdb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.io as io
import scipy.sparse as sp
import tables

from cellbender.remove_background import consts

logger = logging.getLogger("cellbender")


class IngestedData(dict):
    """Small container object for the results of file loading. This is a way to
    ensure that all filetypes are loaded into the same general format that can
    be used in dataset.py

    NOTE: This really exists to ensure all these fields are present, and to
    force each loader to specify each field
    """

    def __init__(self, matrix, barcodes, gene_names, gene_ids, feature_types, genomes, **kwargs):
        # Fill in some fields no matter the input source (for loading in scanpy)
        blank_array = np.array(["NA"] * len(gene_names))
        if genomes is None:
            genomes = blank_array
        if gene_ids is None:
            gene_ids = blank_array
        if feature_types is None:
            feature_types = blank_array

        # Warn if file looks filtered.
        if len(barcodes) < consts.MINIMUM_BARCODES_H5AD:
            logger.warning(
                f"WARNING: Only {len(barcodes)} barcodes in the input file. "
                f"Ensure this is a raw (unfiltered) file with all barcodes, "
                f"including the empty droplets."
            )

        # Required values, some of which can be None
        super().__init__(
            [
                ("matrix", matrix),
                ("barcodes", barcodes),
                ("gene_names", gene_names),
                ("gene_ids", gene_ids),
                ("feature_types", feature_types),
                ("genomes", genomes),
            ]
        )
        self.update(**kwargs)  # cellranger version, for example, is optional


class FileLoader:
    """Make explicit guarantees about what a file-loading method yields."""

    def __init__(self, load_fn):
        self.load_fn = load_fn

    def load(self, file) -> IngestedData:
        data = self.load_fn(file)
        return IngestedData(**data)


def write_matrix_to_cellranger_h5(
    cellranger_version: int,
    output_file: str,
    gene_names: np.ndarray,
    barcodes: np.ndarray,
    count_matrix: sp.csc_matrix,
    feature_types: Optional[np.ndarray] = None,
    gene_ids: Optional[np.ndarray] = None,
    genomes: Optional[np.ndarray] = None,
    local_latents: Dict[str, Optional[np.ndarray]] = {},
    global_latents: Dict[str, Optional[np.ndarray]] = {},
    metadata: Dict[str, Optional[Union[np.ndarray, int, float, str, Dict]]] = {},
) -> bool:
    """Write count matrix data to output HDF5 file using CellRanger format.

    Args:
        cellranger_version: Either 2 or 3. Determines the format of the output
            h5 file.
        output_file: Path to output .h5 file (e.g., 'output.h5').
        gene_names: Name of each gene (column of count matrix).
        gene_ids: Ensembl ID of each gene (column of count matrix).
        genomes: Name of the genome that each gene comes from.
        feature_types: Type of each feature (column of count matrix).
        barcodes: Name of each barcode (row of count matrix).
        count_matrix: Count matrix to be written to file, in sparse
            format.  Rows are barcodes, columns are genes.
        local_latents: Local latent variables. Should include one key called
            'barcodes' which specifies the droplets being referred to.
        global_latents: Global latent variables.
        metadata: Other metadata like loss per epoch and FPR, etc.

    Note:
        To match the CellRanger .h5 files, the matrix is stored as its
        transpose, with rows as genes and cell barcodes as columns.

    """

    assert isinstance(count_matrix, sp.csc_matrix), (
        "The count matrix must be csc_matrix format in order to write to HDF5."
    )

    assert gene_names.size == count_matrix.shape[1], (
        "The number of gene names must match the number of columns in the count matrix."
    )

    if gene_ids is not None:
        assert gene_names.size == gene_ids.size, (
            f"The number of gene_names {gene_names.shape} must match the number of gene_ids {gene_ids.shape}."
        )

    if feature_types is not None:
        assert gene_names.size == feature_types.size, (
            f"The number of gene_names {gene_names.shape} must match the number of feature_types {feature_types.shape}."
        )

    if genomes is not None:
        assert gene_names.size == genomes.size, "The number of gene_names must match the number of genome designations."

    assert barcodes.size == count_matrix.shape[0], (
        "The number of barcodes must match the number of rows in the count matrix."
    )

    # This reverses the role of rows and columns, to match CellRanger format.
    count_matrix = count_matrix.transpose().tocsc()

    # Write to output file.
    filters = tables.Filters(complevel=1, complib="zlib", shuffle=True)
    filter_noshuffle = tables.Filters(complevel=1, complib="zlib", shuffle=False)
    with tables.open_file(output_file, "w", title="CellBender remove-background output") as f:
        if cellranger_version == 2:
            # Create the group where count data will be stored
            group = f.create_group("/", "matrix_v2", "Counts after background correction")

            # Create arrays within that group for gene info.
            f.create_carray(group, "gene_names", obj=gene_names, filters=filters)
            if gene_ids is None:
                # some R loaders require unique values here
                gene_ids = np.array([f"NA_{i}" for i in range(gene_names.size)])
            f.create_carray(group, "genes", obj=gene_ids, filters=filters)
            if genomes is None:
                genomes = np.array(["NA"] * gene_names.size)
            f.create_carray(group, "genome", obj=genomes, filters=filters)

        elif cellranger_version == 3:
            # Create the group where count data will be stored
            group = f.create_group("/", "matrix", "Counts after background correction")

            # Create a sub-group called "features"
            feature_group = f.create_group(group, "features", "Genes and other features measured")

            # Create arrays within that group for feature info.
            f.create_carray(feature_group, "name", obj=gene_names, filters=filters)
            if gene_ids is None:
                # some R loaders require unique values here
                gene_ids = np.array([f"NA_{i}" for i in range(gene_names.size)])
            f.create_carray(feature_group, "id", obj=gene_ids, filters=filters)
            if feature_types is None:
                feature_types = np.array(["Gene Expression"] * gene_names.size)
            f.create_carray(feature_group, "feature_type", obj=feature_types, filters=filters)
            if genomes is None:
                genomes = np.array(["NA"] * gene_names.size)
            f.create_carray(feature_group, "genome", obj=genomes, filters=filters)

            # TODO: Copy the other extraneous information from the input file.
            # (Some user might need it for some reason.)

        else:
            raise ValueError(f"Trying to save to CellRanger v{cellranger_version} format, which is not implemented.")

        # Code for both versions.
        f.create_carray(group, "barcodes", obj=barcodes, filters=filter_noshuffle)

        # Create arrays to store the count data.
        f.create_carray(group, "data", obj=count_matrix.data, filters=filters)
        f.create_carray(group, "indices", obj=count_matrix.indices, filters=filters)
        f.create_carray(group, "indptr", obj=count_matrix.indptr, filters=filters)
        f.create_carray(
            group, "shape", atom=tables.Int32Atom(), obj=np.array(count_matrix.shape, dtype=np.int32), filters=filters
        )

        # Store local latent variables.
        droplet_latent_group = f.create_group("/", "droplet_latents", "Latent variables per droplet")
        for key, value in local_latents.items():
            if value is not None:
                f.create_carray(droplet_latent_group, key, obj=value, filters=filters)

        # Store global latent variables.
        global_group = f.create_group("/", "global_latents", "Global latent variables")
        for key, value in global_latents.items():
            if value is not None:
                f.create_array(global_group, key, value)

        def create_nonscalar_metadata_array(f, group, k, v):
            """Wrap scalar or string values in lists"""
            if v is None:
                return
            if isinstance(v, (list, np.ndarray)):
                f.create_array(group, k, v)
            else:
                f.create_array(group, k, [v])

        # Store metadata.
        metadata_group = f.create_group("/", "metadata", "Metadata")
        for meta_key, meta_value in metadata.items():
            for k, v in unravel_dict(meta_key, meta_value).items():
                create_nonscalar_metadata_array(f, metadata_group, k, v)

    logger.info(f"Succeeded in writing CellRanger format output to file {output_file}")

    return True


# ---------------------------------------------------------------------------
# Parquet-based posterior IO
# ---------------------------------------------------------------------------

POSTERIOR_SCHEMA = pa.schema(
    [
        pa.field("cell_id", pa.int32()),
        pa.field("gene_id", pa.int32()),
        pa.field("c", pa.int32()),
        pa.field("log_prob", pa.float32()),
    ]
)


def _make_duckdb_conn(
    tmp_dir: str,
    memory_limit: Optional[str] = None,
) -> "duckdb.DuckDBPyConnection":
    """Create a DuckDB connection configured for safe spill-to-disk behaviour.

    Always sets ``temp_directory`` so spill files land in the output directory
    rather than ``/tmp`` (which may be RAM-backed inside Docker).  When
    ``memory_limit`` is not provided, auto-detects 50 % of currently-available
    system RAM so DuckDB proactively spills before the OS OOM-killer fires.
    """
    conn = duckdb.connect()
    conn.execute(f"SET temp_directory='{tmp_dir}'")
    if memory_limit is not None:
        conn.execute(f"SET memory_limit='{memory_limit}'")
    else:
        available_bytes = psutil.virtual_memory().available
        limit_bytes = int(available_bytes * 0.5)
        conn.execute(f"SET memory_limit='{limit_bytes}B'")
        logger.debug("DuckDB memory_limit auto-set to %.1f GB", limit_bytes / 2**30)
    return conn


def write_posterior_batch_to_parquet(
    writer: pq.ParquetWriter, cell_ids: np.ndarray, gene_ids: np.ndarray, c_vals: np.ndarray, log_probs: np.ndarray
) -> None:
    """Stream one batch of posterior rows into an open ParquetWriter."""
    batch = pa.table(
        {
            "cell_id": pa.array(cell_ids.astype(np.int32), type=pa.int32()),
            "gene_id": pa.array(gene_ids.astype(np.int32), type=pa.int32()),
            "c": pa.array(c_vals.astype(np.int32), type=pa.int32()),
            "log_prob": pa.array(log_probs.astype(np.float32), type=pa.float32()),
        }
    )
    writer.write_table(batch)


def sort_posterior_parquet(path: Path, duckdb_memory_limit: Optional[str] = None) -> None:
    """Re-write the posterior parquet sorted by (gene_id, cell_id) for efficient DuckDB scans."""
    tmp = Path(str(path) + ".tmp")
    path_str = str(path).replace("'", "''")
    tmp_str = str(tmp).replace("'", "''")
    tmp_dir = str(path.parent).replace("'", "''")
    conn = _make_duckdb_conn(tmp_dir, memory_limit=duckdb_memory_limit)
    conn.execute(
        f"COPY (SELECT * FROM read_parquet('{path_str}') ORDER BY gene_id, cell_id) "
        f"TO '{tmp_str}' (FORMAT PARQUET, COMPRESSION 'snappy')"
    )
    tmp.rename(path)


class IncrementalH5Writer:
    """Context manager for streaming a sparse CSC count matrix to a CellRanger-format HDF5 file.

    Writes gene column by gene column using PyTables extensible arrays so that
    the full data/indices arrays never need to be held in RAM simultaneously.

    Usage::

        with IncrementalH5Writer(output_file, ...) as writer:
            for gene_idx in range(n_genes):
                bc_inds, counts = get_gene_data(gene_idx)   # small arrays
                writer.append_gene(gene_idx, bc_inds, counts)

    The ``indptr`` array (length ``n_genes + 1``) is written at ``__exit__``
    time, when the cumulative non-zero count is known for every gene.
    """

    def __init__(
        self,
        output_file: str,
        n_genes: int,
        n_barcodes: int,
        gene_names: np.ndarray,
        barcodes: np.ndarray,
        gene_ids: Optional[np.ndarray] = None,
        feature_types: Optional[np.ndarray] = None,
        genomes: Optional[np.ndarray] = None,
        local_latents: Optional[Dict[str, Optional[np.ndarray]]] = None,
        global_latents: Optional[Dict[str, Optional[np.ndarray]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.output_file = output_file
        self.n_genes = n_genes
        self.n_barcodes = n_barcodes
        self.gene_names = gene_names
        self.barcodes = barcodes
        self.gene_ids = gene_ids
        self.feature_types = feature_types
        self.genomes = genomes
        self.local_latents = local_latents or {}
        self.global_latents = global_latents or {}
        self.metadata = metadata or {}

        self._f = None
        self._group = None
        self._data_arr = None
        self._indices_arr = None
        self._indptr: List[int] = [0]  # cumulative nnz per gene column

    def __enter__(self) -> "IncrementalH5Writer":
        filters = tables.Filters(complevel=1, complib="zlib", shuffle=True)
        filter_noshuffle = tables.Filters(complevel=1, complib="zlib", shuffle=False)

        self._f = tables.open_file(self.output_file, "w", title="CellBender remove-background output")
        f = self._f
        assert f is not None  # mypy

        # Feature group (CellRanger v3 format — the only format we write).
        group = f.create_group("/", "matrix", "Counts after background correction")
        self._group = group
        feature_group = f.create_group(group, "features", "Genes and other features measured")

        gene_ids = self.gene_ids if self.gene_ids is not None else np.array([f"NA_{i}" for i in range(self.n_genes)])
        ft = self.feature_types if self.feature_types is not None else np.array(["Gene Expression"] * self.n_genes)
        genomes = self.genomes if self.genomes is not None else np.array(["NA"] * self.n_genes)

        f.create_carray(feature_group, "name", obj=self.gene_names, filters=filters)
        f.create_carray(feature_group, "id", obj=gene_ids, filters=filters)
        f.create_carray(feature_group, "feature_type", obj=ft, filters=filters)
        f.create_carray(feature_group, "genome", obj=genomes, filters=filters)
        f.create_carray(group, "barcodes", obj=self.barcodes, filters=filter_noshuffle)

        # Extensible arrays for non-zeros (gene-column-major order = CellRanger CSC transposed).
        self._data_arr = f.create_earray(group, "data", tables.Int32Atom(), shape=(0,), filters=filters)
        self._indices_arr = f.create_earray(group, "indices", tables.Int32Atom(), shape=(0,), filters=filters)

        return self

    def append_barcode(self, barcode_idx: int, gene_indices: np.ndarray, counts: np.ndarray) -> None:
        """Append non-zero entries for one barcode column.

        CellRanger H5 stores a CSC matrix of shape ``(n_genes, n_barcodes)``,
        so barcodes are the CSC columns.  Call this method once per barcode
        in ascending order; pass empty arrays for barcodes with no counts.

        Args:
            barcode_idx: Output row index (0-based, position in the written
                barcodes array).  Not actually stored — used only for ordering.
            gene_indices: Absolute gene indices of the non-zeros for this barcode.
            counts: Non-zero count values for this barcode.
        """
        assert self._data_arr is not None and self._indices_arr is not None
        if len(counts) > 0:
            self._data_arr.append(counts.astype(np.int32))
            self._indices_arr.append(gene_indices.astype(np.int32))
        self._indptr.append(self._indptr[-1] + len(counts))

    def append_batch(self, batch: "sp.csr_matrix") -> None:
        """Append non-zero entries for a batch of barcodes in a single EArray call.

        ``batch`` must be a CSR matrix of shape ``(batch_size, n_genes)`` where
        non-cell rows have already been zeroed.  All rows advance ``_indptr``
        regardless of their nnz count.  Call batches in ascending barcode order.
        """
        assert self._data_arr is not None and self._indices_arr is not None
        if batch.nnz > 0:
            self._data_arr.append(batch.data.astype(np.int32))
            self._indices_arr.append(batch.indices.astype(np.int32))
        row_nnzs = np.diff(batch.indptr)
        last = self._indptr[-1]
        self._indptr.extend((last + np.cumsum(row_nnzs)).tolist())

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._f is None:
            return
        try:
            if exc_type is None:
                f = self._f
                group = self._group
                filters = tables.Filters(complevel=1, complib="zlib", shuffle=True)

                # Pad indptr for any barcodes not covered by append_barcode calls.
                last = self._indptr[-1]
                while len(self._indptr) < self.n_barcodes + 1:
                    self._indptr.append(last)

                f.create_carray(group, "indptr", obj=np.array(self._indptr, dtype=np.int64), filters=filters)
                # CellRanger convention: shape [n_genes, n_barcodes] — the stored
                # CSC has genes as rows and barcodes as columns; anndata_from_h5
                # reads this and transposes to get barcodes × genes.
                f.create_carray(
                    group,
                    "shape",
                    atom=tables.Int32Atom(),
                    obj=np.array([self.n_genes, self.n_barcodes], dtype=np.int32),
                    filters=filters,
                )

                # Local latent variables.
                droplet_latent_group = f.create_group("/", "droplet_latents", "Latent variables per droplet")
                for key, value in self.local_latents.items():
                    if value is not None:
                        f.create_carray(droplet_latent_group, key, obj=value, filters=filters)

                # Global latent variables.
                global_group = f.create_group("/", "global_latents", "Global latent variables")
                for key, value in self.global_latents.items():
                    if value is not None:
                        f.create_array(global_group, key, value)

                def _write_meta(f, grp, k, v):
                    if v is None:
                        return
                    if isinstance(v, (list, np.ndarray)):
                        f.create_array(grp, k, v)
                    else:
                        f.create_array(grp, k, [v])

                metadata_group = f.create_group("/", "metadata", "Metadata")
                for meta_key, meta_value in self.metadata.items():
                    for k, v in unravel_dict(meta_key, meta_value).items():
                        _write_meta(f, metadata_group, k, v)

                logger.info(f"Succeeded in writing CellRanger format output to file {self.output_file}")
        finally:
            self._f.close()
            self._f = None


def compute_noise_totals_per_barcode(
    noise_parquet_path: Path,
    total_n_barcodes: int,
    duckdb_memory_limit: Optional[str] = None,
) -> np.ndarray:
    """Aggregate total noise counts per barcode from the noise parquet.

    Runs a single ``GROUP BY cell_id`` query in DuckDB — very fast since it
    returns only one row per cell rather than one row per (cell, gene).

    Args:
        noise_parquet_path: Parquet with ``(cell_id, gene_id, noise_count)``
            sorted by ``(gene_id, cell_id)``.
        total_n_barcodes: Length of the output array.
        duckdb_memory_limit: DuckDB memory cap.

    Returns:
        noise_per_barcode: 1-D float64 array of shape ``(total_n_barcodes,)``.
    """
    import duckdb as _duckdb

    conn = _duckdb.connect()
    tmp_dir = str(noise_parquet_path.parent).replace("'", "''")
    conn.execute(f"SET temp_directory='{tmp_dir}'")
    if duckdb_memory_limit is not None:
        conn.execute(f"SET memory_limit='{duckdb_memory_limit}'")
    noise_str = str(noise_parquet_path).replace("'", "''")
    df = conn.execute(
        f"SELECT cell_id, SUM(noise_count) AS total_noise FROM read_parquet('{noise_str}') GROUP BY cell_id"
    ).df()
    result = np.zeros(total_n_barcodes, dtype=np.float64)
    if len(df) > 0:
        result[df["cell_id"].values.astype(np.int64)] = df["total_noise"].values
    return result


def stream_denoised_to_cellranger_h5(
    noise_parquet_path: Path,
    raw_count_matrix: "sp.csr_matrix",
    output_file: str,
    cell_logic: np.ndarray,
    analyzed_barcode_inds: np.ndarray,
    gene_names: np.ndarray,
    barcodes: np.ndarray,
    gene_ids: Optional[np.ndarray] = None,
    feature_types: Optional[np.ndarray] = None,
    genomes: Optional[np.ndarray] = None,
    local_latents: Optional[Dict[str, Optional[np.ndarray]]] = None,
    global_latents: Optional[Dict[str, Optional[np.ndarray]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    barcode_subset: Optional[np.ndarray] = None,
    barcode_batch_size: int = 1000,
    duckdb_memory_limit: Optional[str] = None,
) -> np.ndarray:
    """Compute denoised counts and write to CellRanger H5 barcode-by-barcode.

    Workflow:
    1. Re-sort the noise parquet by ``(cell_id, gene_id)`` using DuckDB so
       that barcode-range queries are efficient via row-group min/max pushdown.
    2. Iterate barcodes in ``barcode_subset`` in batches.  For each batch:
       a. Slice the raw CSR matrix once: ``raw_csr[batch_bcs, :]``.
       b. Query noise parquet for the batch's local cell-ID range.
       c. Build a noise CSR matrix in absolute-gene column space.
       d. Subtract, apply a per-row cell mask (zeros empty-barcode rows),
          clamp negatives to zero.
       e. Write the whole batch to H5 via :meth:`IncrementalH5Writer.append_batch`.
    3. For each barcode: subtract noise from raw (analyzed genes); keep raw
       for non-analyzed genes; zero out non-cells.
    4. The full denoised matrix is never held in RAM simultaneously.

    The noise parquet uses *absolute* indices:
    - ``cell_id``: absolute barcode index (row in the full raw count matrix)
    - ``gene_id``: absolute gene index (column in the full raw count matrix)

    Args:
        noise_parquet_path: Parquet with ``(cell_id, gene_id, noise_count)``
            sorted by ``(gene_id, cell_id)`` (absolute indices).
        raw_count_matrix: Full raw count matrix, shape
            ``(total_barcodes, total_genes)`` — CSR or CSC.
        output_file: Destination ``.h5`` path.
        cell_logic: Boolean mask of length ``len(analyzed_barcode_inds)``
            where True marks a cell.
        analyzed_barcode_inds: Absolute barcode indices that were analysed.
        gene_names: Gene name array of length ``total_genes``.
        barcodes: Barcode names for the rows written.  When
            ``barcode_subset`` is given this should already be the subset.
        barcode_subset: Absolute barcode indices to include. ``None`` → all.
        barcode_batch_size: Number of barcodes per DuckDB batch query.
        duckdb_memory_limit: DuckDB memory cap.
        local_latents, global_latents, metadata: Passed to
            :class:`IncrementalH5Writer`.

    Returns:
        denoised_per_barcode: 1-D int64 array of total denoised counts per
            output-row barcode, shape ``(len(barcode_subset),)``.
    """
    import duckdb as _duckdb

    total_barcodes = raw_count_matrix.shape[0]
    total_genes = raw_count_matrix.shape[1]

    # Set of absolute barcode indices that are cells (for noise-subtraction masking).
    cell_absolute: set = set(analyzed_barcode_inds[cell_logic].tolist())

    if barcode_subset is None:
        barcode_subset = np.arange(total_barcodes, dtype=np.int64)
    n_out_barcodes = len(barcode_subset)

    denoised_per_barcode = np.zeros(n_out_barcodes, dtype=np.int64)
    raw_csr = raw_count_matrix.tocsr()

    # Step 1: Re-sort noise parquet by (cell_id, gene_id) so that per-barcode-batch
    # DuckDB queries can skip row groups via min/max statistics.
    bc_sorted_path = noise_parquet_path.parent / (noise_parquet_path.stem + "_bcsorted.parquet")
    noise_str = str(noise_parquet_path).replace("'", "''")
    bc_sorted_str = str(bc_sorted_path).replace("'", "''")
    tmp_dir = str(noise_parquet_path.parent).replace("'", "''")

    conn = _duckdb.connect()
    conn.execute(f"SET temp_directory='{tmp_dir}'")
    if duckdb_memory_limit is not None:
        conn.execute(f"SET memory_limit='{duckdb_memory_limit}'")

    logger.debug("Re-sorting noise parquet by cell_id for streaming H5 write")
    conn.execute(
        f"COPY (SELECT * FROM read_parquet('{noise_str}') ORDER BY cell_id, gene_id)"
        f" TO '{bc_sorted_str}' (FORMAT PARQUET, COMPRESSION 'snappy')"
    )

    # Step 2: Stream barcodes in batches, write to H5.
    try:
        with IncrementalH5Writer(
            output_file=output_file,
            n_genes=total_genes,
            n_barcodes=n_out_barcodes,
            gene_names=gene_names,
            barcodes=barcodes,
            gene_ids=gene_ids,
            feature_types=feature_types,
            genomes=genomes,
            local_latents=local_latents,
            global_latents=global_latents,
            metadata=metadata,
        ) as writer:
            for batch_start in range(0, n_out_barcodes, barcode_batch_size):
                batch_end = min(batch_start + barcode_batch_size, n_out_barcodes)
                batch_bcs = barcode_subset[batch_start:batch_end]
                batch_size = batch_end - batch_start

                # One sparse matrix slice for the entire batch.
                raw_batch = raw_csr[batch_bcs, :]  # (batch_size, total_genes)

                # Map absolute cell barcode → row position within this batch.
                local_to_batch_row: Dict[int, int] = {}
                for i, bc in enumerate(batch_bcs.tolist()):
                    if bc in cell_absolute:
                        local_to_batch_row[bc] = i

                # Build noise sparse matrix from DuckDB query.
                noise_csr: sp.csr_matrix = sp.csr_matrix((batch_size, total_genes), dtype=raw_batch.dtype)
                if local_to_batch_row:
                    min_lid = min(local_to_batch_row)
                    max_lid = max(local_to_batch_row)
                    noise_df = conn.execute(
                        f"SELECT cell_id, gene_id, noise_count "
                        f"FROM read_parquet('{bc_sorted_str}') "
                        f"WHERE cell_id >= {min_lid} AND cell_id <= {max_lid}"
                    ).df()
                    if len(noise_df) > 0:
                        # Range query may include cell IDs between min and max that are
                        # not actually in this batch; filter them out.
                        in_batch = noise_df["cell_id"].isin(local_to_batch_row)
                        if in_batch.any():
                            noise_df = noise_df[in_batch]
                            noise_rows = noise_df["cell_id"].map(local_to_batch_row).to_numpy()
                            noise_cols = noise_df["gene_id"].to_numpy()
                            noise_data = noise_df["noise_count"].to_numpy().astype(raw_batch.dtype)
                            noise_csr = sp.csr_matrix(
                                (noise_data, (noise_rows, noise_cols)),
                                shape=(batch_size, total_genes),
                            )
                    del noise_df

                # Subtract noise from raw counts.
                denoised_batch: sp.csr_matrix = (raw_batch - noise_csr).tocsr()

                # Zero non-cell rows: multiply each stored value by its row's
                # cell mask (1 for cells, 0 for empties) using vectorised indexing.
                if denoised_batch.nnz > 0:
                    cell_mask = np.zeros(batch_size, dtype=denoised_batch.dtype)
                    for row_i in local_to_batch_row.values():
                        cell_mask[row_i] = 1
                    row_of_each_nnz = np.repeat(
                        np.arange(batch_size, dtype=np.intp),
                        np.diff(denoised_batch.indptr),
                    )
                    denoised_batch.data *= cell_mask[row_of_each_nnz]

                # Clamp negatives to zero and drop explicit zeros.
                np.clip(denoised_batch.data, 0, None, out=denoised_batch.data)
                denoised_batch.eliminate_zeros()

                # Accumulate denoised totals per output barcode.
                row_sums = np.asarray(denoised_batch.sum(axis=1)).ravel()
                denoised_per_barcode[batch_start:batch_end] = row_sums.astype(np.int64)

                # Batch write — one EArray.append call for the whole batch.
                writer.append_batch(denoised_batch)
    finally:
        conn.close()
        try:
            bc_sorted_path.unlink()
        except Exception:
            logger.warning(f"Could not remove temp parquet: {bc_sorted_path}")

    return denoised_per_barcode


def _split_latents(latents: Dict[str, np.ndarray]) -> tuple[Dict[str, np.ndarray], Dict[str, list]]:
    """Split latents into per-barcode arrays and global (scalar/small) entries.

    Returns:
        (per_barcode, global_latents) where *per_barcode* contains 1D/2D arrays
        with the barcode count on axis-0, and *global_latents* contains everything
        else (e.g. phi_loc_scale which is a 2-element list).
    """
    # Determine per-barcode length from the first long-enough array.
    n_barcodes: int | None = None
    for val in latents.values():
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] > 2:
            n_barcodes = arr.shape[0]
            break

    per_barcode: Dict[str, np.ndarray] = {}
    global_lats: Dict[str, list] = {}
    for key, val in latents.items():
        if val is None:
            continue
        arr = np.asarray(val)
        if n_barcodes is not None and (arr.ndim == 0 or arr.shape[0] != n_barcodes):
            global_lats[key] = np.asarray(val).tolist()
        else:
            per_barcode[key] = arr
    return per_barcode, global_lats


def write_posterior_latents_csv(path: Path, latents: Dict[str, np.ndarray]) -> None:
    """Save per-barcode posterior MAP latents to a gzip-compressed CSV.

    Global (scalar/small) entries such as *phi_loc_scale* are skipped here and
    must be written separately via :func:`write_posterior_global_latents_json`.
    """
    import pandas as pd

    per_barcode, _global = _split_latents(latents)

    data: Dict[str, np.ndarray] = {}
    for key, arr in per_barcode.items():
        if arr.ndim == 1:
            data[key] = arr
        elif arr.ndim == 2:
            for i in range(arr.shape[1]):
                data[f"{key}_{i}"] = arr[:, i]
        else:
            logger.warning(f"Skipping latent '{key}' with shape {arr.shape}: only 1D/2D supported")
    pd.DataFrame(data).to_csv(str(path), index=False, compression="gzip", float_format="%.17g")


def write_posterior_global_latents_json(path: Path, latents: Dict[str, np.ndarray]) -> None:
    """Save global (non-per-barcode) posterior latents to a JSON sidecar.

    This captures entries such as *phi_loc_scale* = [phi_loc, phi_scale] that
    are model-wide rather than per-barcode.
    """
    _per_barcode, global_lats = _split_latents(latents)
    if not global_lats:
        return
    with open(str(path), "w") as f:
        json.dump(global_lats, f)


def load_posterior_global_latents_json(path: Path) -> Dict[str, np.ndarray]:
    """Load global posterior latents from a JSON sidecar.

    Returns an empty dict if the file does not exist.
    """
    if not path.exists():
        return {}
    with open(str(path)) as f:
        raw = json.load(f)
    return {k: np.asarray(v) for k, v in raw.items()}


def load_posterior_latents_csv(path: Path) -> Dict[str, np.ndarray]:
    """Load posterior MAP latents from a gzip-compressed CSV file."""
    import pandas as pd

    df = pd.read_csv(str(path), compression="gzip")
    latents: Dict[str, np.ndarray] = {}
    z_cols = sorted(
        [c for c in df.columns if c.startswith("z_") and c.split("_", 1)[1].isdigit()],
        key=lambda x: int(x.split("_", 1)[1]),
    )
    other_cols = [c for c in df.columns if c not in z_cols]
    for col in other_cols:
        latents[col] = df[col].values
    if z_cols:
        latents["z"] = df[z_cols].values
    return latents


def unravel_dict(pref: str, d: Any) -> Dict:
    """Unravel a nested dict, returning a dict with values that are not dicts"""

    if not isinstance(d, dict):
        return {pref: d}
    out_d = {}
    for k, v in d.items():
        out_d.update({pref + "_" + key: val for key, val in unravel_dict(k, v).items()})
    return out_d


def load_data(input_file: str) -> Dict[str, Union[sp.csr_matrix, List[np.ndarray], np.ndarray]]:
    """Load a dataset into the SingleCellRNACountsDataset object from
    the self.input_file"""

    # Detect input data type.
    load_fn = choose_data_loader(input_file=input_file)

    # Load data using the appropriate loader.
    logger.info(f"Loading data from {input_file}")
    data = FileLoader(load_fn).load(input_file)

    return data


def choose_data_loader(input_file: str) -> Callable:
    """Detect the type of input data and return the relevant load function."""

    # Error if no input data file has been specified.
    assert input_file is not None, "Attempting to load data, but no input file was specified."

    file_ext = os.path.splitext(input_file)[1]

    # Detect type.
    if os.path.isdir(input_file):
        return get_matrix_from_cellranger_mtx

    elif file_ext == ".h5":
        return get_matrix_from_cellranger_h5

    elif input_file.endswith(".txt.gz") or input_file.endswith(".txt"):
        return get_matrix_from_dropseq_dge

    elif input_file.endswith(".csv.gz") or input_file.endswith(".csv"):
        return get_matrix_from_bd_rhapsody

    elif file_ext == ".h5ad":
        return get_matrix_from_anndata

    elif file_ext == ".loom":
        return get_matrix_from_loom

    elif file_ext == ".npz":
        return get_matrix_from_npz

    else:
        raise ValueError(
            "Failed to determine input file type for "
            + input_file
            + "\n"
            + "This must either be: a directory that contains "
            "CellRanger-format MTX outputs; a single CellRanger "
            '".h5" file; a DropSeq-format DGE ".txt.gz" file; '
            'a BD-Rhapsody-format ".csv" file; a ".h5ad" file '
            "produced by anndata (include all barcodes); a "
            '".loom" file (include all barcodes); or a ".npz" '
            "sparse matrix file"
        )


def detect_cellranger_version_mtx(filedir: str) -> int:
    """Detect which version of CellRanger (2 or 3) created this mtx directory.

    Args:
        filedir: string path to .mtx file that contains the raw gene
            barcode matrix in a sparse coo text format.

    Returns:
        CellRanger version, either 2 or 3, as an integer.

    """

    assert os.path.isdir(filedir), f"The directory {filedir} is not accessible."

    if os.path.isfile(os.path.join(filedir, "features.tsv.gz")):
        return 3

    else:
        return 2


def detect_cellranger_version_h5(filename: str) -> int:
    """Detect which version of CellRanger (2 or 3) created this h5 file.

    Args:
        filename: string path to .mtx file that contains the raw gene
            barcode matrix in a sparse coo text format.

    Returns:
        version: CellRanger version, either 2 or 3, as an integer.

    """

    with tables.open_file(filename, "r") as f:
        # For CellRanger v2, each group in the table (other than root)
        # contains a genome.
        # For CellRanger v3, there is a 'matrix' group that contains 'features'.

        version = 2

        try:
            # This works for version 3 but not for version 2.
            getattr(f.root.matrix, "features")
            version = 3

        except tables.NoSuchNodeError:
            pass

    return version


def get_matrix_from_cellranger_mtx(filedir: str) -> Dict[str, Union[sp.csr_matrix, List[np.ndarray], np.ndarray]]:
    """Load a count matrix from an mtx directory from CellRanger's output.

    For CellRanger v2:
    The directory must contain three files:
        matrix.mtx
        barcodes.tsv
        genes.tsv

    For CellRanger v3:
    The directory must contain three files:
        matrix.mtx.gz
        barcodes.tsv.gz
        features.tsv.gz

    This function returns a dictionary that includes the count matrix, the gene
    names (which correspond to columns of the count matrix), and the barcodes
    (which correspond to rows of the count matrix).

    Args:
        filedir: string path to .mtx file that contains the raw gene
            barcode matrix in a sparse coo text format.

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string Ensembl ID of genes in the genome, which
             also correspond to the columns in the out['matrix'].
        out['feature_types']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string feature types of genes (or possibly
             antibody capture reads), which also correspond to the columns
             in the out['matrix'].

    """

    assert os.path.isdir(filedir), "The directory {filedir} is not accessible."

    # Decide whether data is CellRanger v2 or v3.
    cellranger_version = detect_cellranger_version_mtx(filedir=filedir)
    logger.info(f"CellRanger v{cellranger_version} format")

    # CellRanger version 3
    if cellranger_version == 3:
        matrix_file = os.path.join(filedir, "matrix.mtx.gz")
        gene_file = os.path.join(filedir, "features.tsv.gz")
        barcode_file = os.path.join(filedir, "barcodes.tsv.gz")

        # Read in feature names.
        features = np.genfromtxt(fname=gene_file, delimiter="\t", skip_header=0, dtype=str)

        # Read in gene expression and feature data.
        gene_ids = features[:, 0].squeeze()  # first column
        gene_names = features[:, 1].squeeze()  # second column
        feature_types = features[:, 2].squeeze()  # third column

    # CellRanger version 2
    elif cellranger_version == 2:
        # Read in the count matrix using scipy.
        matrix_file = os.path.join(filedir, "matrix.mtx")
        gene_file = os.path.join(filedir, "genes.tsv")
        barcode_file = os.path.join(filedir, "barcodes.tsv")

        # Read in gene names.
        gene_data = np.genfromtxt(fname=gene_file, delimiter="\t", skip_header=0, dtype=str)
        if len(gene_data.shape) == 1:  # custom file format with just gene names
            gene_names = gene_data.squeeze()
            gene_ids = None
        else:  # the 10x CellRanger v2 format with two columns
            gene_names = gene_data[:, 1].squeeze()  # second column
            gene_ids = gene_data[:, 0].squeeze()  # first column
        feature_types = None

    else:
        raise NotImplementedError(
            "MTX format was not identifiable as CellRanger v2 or v3.  Please check 10x Genomics formatting."
        )

    # For both versions:

    # Read in sparse count matrix.
    count_matrix = io.mmread(matrix_file).tocsr().transpose()

    # Read in barcode names.
    barcodes = np.genfromtxt(fname=barcode_file, delimiter="\t", skip_header=0, dtype=str)

    # Issue warnings if necessary, based on dimensions matching.
    if count_matrix.shape[1] != len(gene_names):
        logger.warning(
            f"Number of gene names in {filedir}/genes.tsv does not match the number expected from the count matrix."
        )
    if count_matrix.shape[0] != len(barcodes):
        logger.warning(
            f"Number of barcodes in {filedir}/barcodes.tsv does not match the number expected from the count matrix."
        )

    return {
        "matrix": count_matrix,
        "gene_names": gene_names,
        "feature_types": feature_types,
        "gene_ids": gene_ids,
        "genomes": None,
        "barcodes": barcodes,
        "cellranger_version": cellranger_version,
    }


def get_matrix_from_cellranger_h5(filename: str) -> Dict[str, Union[sp.csr_matrix, np.ndarray]]:
    """Load a count matrix from an h5 file from CellRanger's output.

    The file needs to be a _raw_gene_bc_matrices_h5.h5 file.  This function
    returns a dictionary that includes the count matrix, the gene names (which
    correspond to columns of the count matrix), and the barcodes (which
    correspond to rows of the count matrix).

    This function works for CellRanger v2 and v3 HDF5 formats.

    Args:
        filename: string path to .h5 file that contains the raw gene
            barcode matrices

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string Ensembl ID of genes in the genome, which
             also correspond to the columns in the out['matrix'].
        out['feature_types']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string feature types of genes (or possibly
             antibody capture reads), which also correspond to the columns
             in the out['matrix'].

    """

    # Detect CellRanger version.
    cellranger_version = detect_cellranger_version_h5(filename=filename)
    logger.info(f"CellRanger v{cellranger_version} format")

    with tables.open_file(filename, "r") as f:
        # Warn if the file looks like h5ad saved with a .h5 extension.
        if hasattr(f.root, "obs") and hasattr(f.root, "var"):
            logger.warning(
                f"The file '{filename}' appears to be in AnnData (.h5ad) format, "
                "but has a .h5 extension.  CellBender will attempt to read it as "
                "CellRanger format, which will likely fail.  If this file is an "
                "AnnData object, rename it with a .h5ad extension and re-run."
            )

        # Initialize empty lists.
        csc_list = []
        barcodes: np.ndarray | None = None
        feature_ids: np.ndarray | None = None
        feature_types: np.ndarray | None = None
        genomes: np.ndarray | None = None
        feature_names: np.ndarray = np.array([])

        # CellRanger v2:
        # Each group in the table (other than root) contains a genome,
        # so walk through the groups to get data for each genome.
        if cellranger_version == 2:
            feature_names_list: list = []
            feature_ids_list: list = []
            genomes_list: list = []

            for group in f.walk_groups():
                try:
                    # Read in data for this genome, and put it into a
                    # scipy.sparse.csc.csc_matrix
                    barcodes = getattr(group, "barcodes").read()
                    data = getattr(group, "data").read()
                    indices = getattr(group, "indices").read()
                    indptr = getattr(group, "indptr").read()
                    shape = getattr(group, "shape").read()
                    csc_list.append(sp.csc_matrix((data, indices, indptr), shape=shape))
                    fnames_this_genome = getattr(group, "gene_names").read()
                    feature_names_list.extend(fnames_this_genome)
                    feature_ids_list.extend(getattr(group, "genes").read())
                    genomes_list.extend([group._g_gettitle()] * fnames_this_genome.size)

                except tables.NoSuchNodeError:
                    # This exists to bypass the root node, which has no data.
                    pass

            # Create numpy arrays.
            feature_names = np.array(feature_names_list, dtype=str)
            genomes = np.array(genomes_list, dtype=str)
            if len(feature_ids_list) > 0:
                feature_ids = np.array(feature_ids_list)
            else:
                feature_ids = None

        # CellRanger v3:
        # There is only the 'matrix' group.
        elif cellranger_version == 3:
            # Read in data for this genome, and put it into a
            # scipy.sparse.csc.csc_matrix
            barcodes = getattr(f.root.matrix, "barcodes").read()
            data = getattr(f.root.matrix, "data").read()
            indices = getattr(f.root.matrix, "indices").read()
            indptr = getattr(f.root.matrix, "indptr").read()
            shape = getattr(f.root.matrix, "shape").read()
            csc_list.append(sp.csc_matrix((data, indices, indptr), shape=shape))

            # Read in 'feature' information
            feature_group = f.get_node(f.root.matrix, "features")
            feature_names = getattr(feature_group, "name").read()

            try:
                feature_types = getattr(feature_group, "feature_type").read()
            except tables.NoSuchNodeError:
                # This exists in case someone produced a file without feature_type.
                pass
            try:
                feature_ids = getattr(feature_group, "id").read()
            except tables.NoSuchNodeError:
                # This exists in case someone produced a file without feature id.
                pass
            try:
                genomes = getattr(feature_group, "genome").read()
            except tables.NoSuchNodeError:
                # This exists in case someone produced a file without feature genome.
                pass

    # Put the data together (possibly from several genomes for v2 datasets).
    count_matrix = sp.vstack(csc_list, format="csc")
    count_matrix = count_matrix.transpose().tocsr()

    assert barcodes is not None, "barcodes not loaded from HDF5 file"

    # Issue warnings if necessary, based on dimensions matching.
    if count_matrix.shape[1] != feature_names.size:
        logger.warning(
            f"Number of gene names ({feature_names.size}) in {filename} "
            f"does not match the number expected from the count "
            f"matrix ({count_matrix.shape[1]})."
        )
    if count_matrix.shape[0] != barcodes.size:
        logger.warning(
            f"Number of barcodes ({barcodes.size}) in {filename} "
            f"does not match the number expected from the count "
            f"matrix ({count_matrix.shape[0]})."
        )

    return {
        "matrix": count_matrix,
        "gene_names": feature_names,
        "gene_ids": feature_ids,
        "genomes": genomes,
        "feature_types": feature_types,
        "barcodes": barcodes,
        "cellranger_version": cellranger_version,
    }


def get_matrix_from_dropseq_dge(filename: str) -> Dict[str, Union[sp.csr_matrix, np.ndarray]]:
    """Load a count matrix from a DropSeq DGE matrix file.

    The file needs to be a gzipped text file in DGE format.  This function
    returns a dictionary that includes the count matrix, the gene names (which
    correspond to columns of the count matrix), and the barcodes (which
    correspond to rows of the count matrix).  Reads in the file line by line
    instead of trying to read in an entire dense matrix at once, which might
    require quite a bit of memory.

    Args:
        filename: string path to .txt.gz file that contains the raw gene
            barcode matrix

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string Ensembl ID of genes in the genome, which
            also correspond to the columns in the out['matrix'].

    """

    logger.info("DropSeq DGE format")

    load_fcn = gzip.open if filename.endswith(".gz") else open

    with load_fcn(filename, "rt") as f:
        # Skip the comment '#' lines in header
        for header in f:
            if header[0] == "#":
                continue
            else:
                break

        # Read in first row with droplet barcodes
        barcodes = header.split("\n")[0].split("\t")[1:]

        # Gene names are first entry per row
        gene_names = []

        # Arrays used to construct a sparse matrix
        row: list = []
        col: list = []
        data: list = []

        # Read in rest of file row by row
        for i, line in enumerate(f):
            # Parse row into gene name and count data
            parsed_line = line.split("\n")[0].split("\t")
            gene_names.append(parsed_line[0])
            counts = np.array(parsed_line[1:], dtype=int)

            # Create sparse version of data and add to arrays
            nonzero_col_inds = np.nonzero(counts)[0]
            row.extend([i] * nonzero_col_inds.size)
            col.extend(nonzero_col_inds)
            data.extend(counts[nonzero_col_inds])

    count_matrix = sp.csc_matrix((data, (row, col)), shape=(len(gene_names), len(barcodes)), dtype=float).transpose()

    return {
        "matrix": count_matrix,
        "gene_names": np.array(gene_names),
        "gene_ids": None,
        "genomes": None,
        "feature_types": None,
        "barcodes": np.array(barcodes),
    }


def get_matrix_from_bd_rhapsody(filename: str) -> Dict[str, Union[sp.csr_matrix, np.ndarray]]:
    """Load a count matrix from a BD Rhapsody MolsPerCell.csv file.

    The file needs to be in MolsPerCell_Unfiltered format, which is comma
    separated, where rows are barcodes and columns are genes.  Can be gzipped
    or not.  This function returns a dictionary that includes the count matrix,
    the gene names (which correspond to columns of the count matrix), and the
    barcodes (which correspond to rows of the count matrix).  Reads in the file
    line by line instead of trying to read in an entire dense matrix at once,
    which might require quite a bit of memory.

    Args:
        filename: string path to .csv file that contains the raw gene
            barcode matrix MolsPerCell_Unfiltered.csv

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string Ensembl ID of genes in the genome, which
            also correspond to the columns in the out['matrix'].

    """

    logger.info("BD Rhapsody MolsPerCell_Unfiltered.csv format")

    load_fcn = gzip.open if filename.endswith(".gz") else open

    with load_fcn(filename, "rt") as f:
        # Skip the comment '#' lines in header
        for header in f:
            if header[0] == "#":
                continue
            else:
                break

        # Read in first row with gene names
        gene_names = header.split("\n")[0].split(",")[1:]

        # Barcode names are first entry per row
        barcodes = []

        # Arrays used to construct a sparse matrix
        row: list = []
        col: list = []
        data: list = []

        # Read in rest of file row by row
        for i, line in enumerate(f):
            # Parse row into gene name and count data
            parsed_line = line.split("\n")[0].split(",")
            barcodes.append(parsed_line[0])
            counts = np.array(parsed_line[1:], dtype=np.int_)

            # Create sparse version of data and add to arrays
            nonzero_col_inds = np.nonzero(counts)[0]
            row.extend([i] * nonzero_col_inds.size)
            col.extend(nonzero_col_inds)
            data.extend(counts[nonzero_col_inds])

    count_matrix = sp.csc_matrix((data, (row, col)), shape=(len(barcodes), len(gene_names)), dtype=float)

    return {
        "matrix": count_matrix,
        "gene_names": np.array(gene_names),
        "gene_ids": None,
        "genomes": None,
        "feature_types": None,
        "barcodes": np.array(barcodes),
    }


def get_matrix_from_npz(filename: str) -> Dict[str, Union[sp.csr_matrix, np.ndarray]]:
    """Load a count matrix from a sparse NPZ file, accompanied by barcode and
    gene NPY files.
    NOTE: This format is one output of the Optimus pipeline. It loads much
    faster than a Loom file. The NPZ file requires two accompanying files:
    'col_index.npy' and 'row_index.npy', named exactly as shown, and in the
    same directory as the NPZ file.
    Args:
        filename: string path to .h5ad file that contains the raw gene
            barcode matrices
    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string Ensembl ID of genes in the genome, which
             also correspond to the columns in the out['matrix'].
        out['feature_types']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string feature types of genes (or possibly
             antibody capture reads), which also correspond to the columns
             in the out['matrix'].
    """
    logger.info("Optimus sparse NPZ format")
    try:
        count_matrix = sp.load_npz(file=filename)
        file_dir, _ = os.path.split(filename)
        gene_ids = np.load(os.path.join(file_dir, "col_index.npy"))
        barcodes = np.load(os.path.join(file_dir, "row_index.npy"))
    except IOError as e:
        logger.error(
            "Loading an NPZ file requires two additional files in the "
            f"same directory ({file_dir}): "
            'one called "col_index.npy" that contains genes, and one '
            'called "row_index.npy" that contains barcodes.'
        )
        logger.error(traceback.format_exc())
        raise e
    return {
        "matrix": count_matrix,
        "gene_names": gene_ids,  # that's all we have access to, so we'll use it
        "gene_ids": gene_ids,
        "genomes": None,
        "feature_types": None,
        "barcodes": barcodes,
    }


def get_matrix_from_anndata(filename: str) -> Dict[str, Union[sp.csr_matrix, np.ndarray]]:
    """Load a count matrix from an h5ad AnnData file.
    The file needs to contain raw counts for all measured barcodes in the
    `.X` attribute or a `.layer[{'counts', 'spliced'}]` attribute.  This function
    returns a dictionary that includes the count matrix, the gene names (which
    correspond to columns of the count matrix), and the barcodes (which
    correspond to rows of the count matrix).
    This function works for any AnnData object meeting the above requirements,
    as generated by alignment methods like `kallisto | bustools`.
    Args:
        filename: string path to .h5ad file that contains the raw gene
            barcode matrices
    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string Ensembl ID of genes in the genome, which
             also correspond to the columns in the out['matrix'].
        out['feature_types']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string feature types of genes (or possibly
             antibody capture reads), which also correspond to the columns
             in the out['matrix'].
    """
    logger.info("AnnData format")
    try:
        adata = anndata.read_h5ad(filename)
    except anndata._io.utils.AnnDataReadError as e:
        logger.error(f"A call to anndata.read_h5ad() with anndata {anndata.__version__} threw AnnDataReadError: ")
        logger.error(traceback.format_exc())
        raise e
    return _dict_from_anndata(adata)


def get_matrix_from_loom(filename: str) -> Dict[str, Union[sp.csr_matrix, np.ndarray]]:
    """Load a count matrix from a loom file.
    The file needs to contain raw counts for all measured barcodes in the
    layer '', as in
    https://broadinstitute.github.io/warp/docs/Pipelines/Optimus_Pipeline/Loom_schema/
    Returns a dictionary that includes the count matrix, the gene names (which
    correspond to columns of the count matrix), and the barcodes (which
    correspond to rows of the count matrix).

    Args:
        filename: string path to .h5ad file that contains the raw gene
            barcode matrices

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string Ensembl ID of genes in the genome, which
             also correspond to the columns in the out['matrix'].
        out['feature_types']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string feature types of genes (or possibly
             antibody capture reads), which also correspond to the columns
             in the out['matrix'].
    """
    logger.info("Loom format, expecting Optimus pipeline conventions")
    try:
        adata = anndata.read_loom(filename, sparse=True, X_name="")
    except anndata._io.utils.AnnDataReadError as e:
        logger.error(f"A call to anndata.read_loom() with anndata {anndata.__version__} threw AnnDataReadError: ")
        logger.error(traceback.format_exc())
        raise e
    return _dict_from_anndata(adata)


def _dict_from_anndata(adata: anndata.AnnData) -> Dict[str, Union[sp.csr_matrix, np.ndarray]]:
    """Extract relevant information from AnnData and format it as a dict

    Args:
        adata: AnnData object

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string Ensembl ID of genes in the genome, which
             also correspond to the columns in the out['matrix'].
        out['feature_types']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string feature types of genes (or possibly
             antibody capture reads), which also correspond to the columns
             in the out['matrix']."""

    if "counts" in adata.layers.keys():
        # this is a common manual setting for users of scVI
        # given the manual convention, we prefer this matrix to
        # .X since it is less likely to represent something other
        # than counts
        logger.info("Found `.layers['counts']`. Using for count data.")
        count_matrix = adata.layers["counts"]
    elif "spliced" in adata.layers.keys() and adata.X is None:
        # alignment using kallisto | bustools with intronic counts
        # does not populate `.X` by default, but does populate
        # `.layers['spliced'], .layers['unspliced']`.
        # we use spliced counts for analysis
        logger.info("Found `.layers['spliced']`. Using for count data.")
        count_matrix = adata.layers["spliced"]
    else:
        logger.info("Using `.X` for count data.")
        count_matrix = adata.X

    # check that `count_matrix` contains a large number of barcodes,
    # consistent with a raw single cell experiment
    if count_matrix.shape[0] < consts.MINIMUM_BARCODES_H5AD:
        # this experiment might be prefiltered
        logger.warning(
            f"Only {count_matrix.shape[0]} barcodes were found.\n"
            "This suggests the matrix was prefiltered.\n"
            "CellBender requires a raw, unfiltered [Barcodes, Genes] matrix."
        )

    # AnnData is [Cells, Genes], no need to transpose
    # we typecast explicitly in the off chance `count_matrix` was dense.
    count_matrix = sp.csr_matrix(count_matrix)
    # feature names and ids are not consistently delineated in AnnData objects
    # so we attempt to find relevant features using common values.
    feature_names = np.array(adata.var_names, dtype=str)
    barcodes = np.array(adata.obs_names, dtype=str)

    # Make an attempt to find feature_IDs if they are present.
    feature_ids = None
    for key in ["gene_id", "gene_ids", "ensembl_ids"]:
        if key in adata.var.keys():
            feature_ids = np.array(adata.var[key].values, dtype=str)

    # Make an attempt to find feature_types if they are present.
    feature_types = None
    for key in ["feature_type", "feature_types"]:
        if key in adata.var.keys():
            feature_types = np.array(adata.var[key].values, dtype=str)

    # Make an attempt to find genomes if they are present.
    genomes = None
    for key in ["genome", "genomes"]:
        if key in adata.var.keys():
            genomes = np.array(adata.var[key].values, dtype=str)

    # Issue warnings if necessary, based on dimensions matching.
    if count_matrix.shape[1] != feature_names.size:
        logger.warning(
            f"Number of gene names ({feature_names.size}) "
            f"does not match the number expected from the count "
            f"matrix ({count_matrix.shape[1]})."
        )
    if count_matrix.shape[0] != barcodes.size:
        logger.warning(
            f"Number of barcodes ({barcodes.size}) "
            f"does not match the number expected from the count "
            f"matrix ({count_matrix.shape[0]})."
        )

    return {
        "matrix": count_matrix,
        "gene_names": feature_names,
        "gene_ids": feature_ids,
        "genomes": genomes,
        "feature_types": feature_types,
        "barcodes": barcodes,
    }
