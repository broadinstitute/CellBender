"""Tests for run.py: streaming H5 output correctness."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
import tables


def _write_empty_noise_parquet(path: Path) -> None:
    """Write a zero-row noise parquet (gene-sorted format expected by streaming writer)."""
    pq.write_table(
        pa.table(
            {
                "cell_id": pa.array([], type=pa.int32()),
                "gene_id": pa.array([], type=pa.int32()),
                "noise_count": pa.array([], type=pa.int32()),
            }
        ),
        path,
    )


class _MockEstimator:
    """Writes an empty noise parquet to the requested output path."""

    def estimate_noise_to_parquet(self, noise_log_prob_coo, output_path, **kwargs):
        _write_empty_noise_parquet(output_path)


def _make_posterior_and_args(tmp_path: Path):
    """Build a minimal synthetic posterior + args for _write_streaming_denoised_outputs.

    Layout:
        20 total barcodes, 5 genes.
        Analyzed barcodes (8): absolute indices [0, 2, 5, 8, 12, 15, 17, 19].
        Cells (3): absolute indices [5, 8, 17]  — deliberately scattered so they
        are *not* in consecutive order relative to the analyzed-barcode list.

    Each cell is given a unique, easily-verified cell_probability value:
        BC005 → p=0.99,  BC008 → p=0.95,  BC017 → p=0.91
    and a unique z-encoding row so that latent-to-barcode alignment can be
    confirmed after reading back the written H5.
    """
    n_barcodes = 20
    n_genes = 5
    n_analyzed = 8

    analyzed_barcode_inds = np.array([0, 2, 5, 8, 12, 15, 17, 19], dtype=np.int64)
    # Positions within analyzed_barcode_inds that are cells:
    #   index 2 → abs barcode 5
    #   index 3 → abs barcode 8
    #   index 6 → abs barcode 17
    cell_logic = np.array([False, False, True, True, False, False, True, False])

    rng = np.random.default_rng(42)
    raw_data = rng.integers(0, 10, size=(n_barcodes, n_genes), dtype=np.int32)
    raw_csr = sp.csr_matrix(raw_data)

    # Per-analyzed-barcode latents (length n_analyzed).
    p_analyzed = np.array([0.1, 0.2, 0.99, 0.95, 0.3, 0.4, 0.91, 0.15])
    d_analyzed = np.array([100.0, 50.0, 500.0, 400.0, 80.0, 70.0, 350.0, 60.0])
    eps_analyzed = np.array([0.9, 0.8, 0.75, 0.70, 0.85, 0.88, 0.65, 0.82])
    z_dim = 4
    z_analyzed = rng.random((n_analyzed, z_dim)).astype(np.float32)
    # Overwrite cell rows with one-hot vectors so alignment errors are obvious.
    z_analyzed[2] = [1.0, 0.0, 0.0, 0.0]  # abs barcode 5
    z_analyzed[3] = [0.0, 1.0, 0.0, 0.0]  # abs barcode 8
    z_analyzed[6] = [0.0, 0.0, 1.0, 0.0]  # abs barcode 17

    barcodes_arr = np.array([f"BC{i:03d}".encode() for i in range(n_barcodes)])
    gene_names = np.array([f"Gene{i}" for i in range(n_genes)])

    dataset_obj = SimpleNamespace(
        data={
            "matrix": raw_csr,
            "barcodes": barcodes_arr,
            "gene_names": gene_names,
        },
        analyzed_barcode_inds=analyzed_barcode_inds,
        analyzed_gene_inds=np.arange(n_genes, dtype=np.int64),
    )

    # posterior_path only needs a valid .parent directory; the actual file need
    # not exist because the mock estimator writes directly to output_path.
    posterior = SimpleNamespace(
        posterior_path=tmp_path / "posterior.parquet",
        dataset_obj=dataset_obj,
        latents_map={
            "p": p_analyzed,
            "d": d_analyzed,
            "epsilon": eps_analyzed,
            "z": z_analyzed,
        },
        model_loss={},
        n_cells=3,
        n_genes=n_genes,
        ensure_posterior_computed=lambda: None,
    )

    args = argparse.Namespace(
        cdf_threshold_q=None,
        duckdb_memory_limit=None,
        estimator="map",
    )

    return posterior, args, cell_logic, barcodes_arr, p_analyzed, d_analyzed, eps_analyzed, z_analyzed


def test_filtered_h5_has_only_cell_barcodes(tmp_path):
    """Filtered H5 must contain exactly the cell barcodes and no others."""
    from cellbender.remove_background.run import _write_streaming_denoised_outputs

    posterior, args, *_, barcodes_arr, p_analyzed, d_analyzed, eps_analyzed, z_analyzed = _make_posterior_and_args(
        tmp_path
    )

    full_file = str(tmp_path / "out.h5")
    filt_file = str(tmp_path / "out_filtered.h5")

    full_ok, filt_ok, _ = _write_streaming_denoised_outputs(
        posterior=posterior,
        estimator_obj=_MockEstimator(),
        noise_targets=None,
        args=args,
        fpr=0.01,
        fpr_output_filename=full_file,
        filtered_output_file=filt_file,
        global_latents={},
    )

    assert full_ok, "Full H5 write failed"
    assert filt_ok, "Filtered H5 write failed"

    expected_cell_barcodes = {b"BC005", b"BC008", b"BC017"}

    with tables.open_file(filt_file, "r") as f:
        filt_barcodes = set(f.root.matrix.barcodes[:])
        filt_shape = f.root.matrix.shape[:]  # [n_genes, n_cells] (CellRanger convention)

    assert filt_shape[1] == 3, f"Filtered H5 should have 3 barcodes, got {filt_shape[1]}"
    assert filt_barcodes == expected_cell_barcodes, (
        f"Filtered H5 barcodes {filt_barcodes} != expected {expected_cell_barcodes}"
    )


def test_full_h5_has_all_barcodes(tmp_path):
    """Full H5 must contain all barcodes (cells + empties)."""
    from cellbender.remove_background.run import _write_streaming_denoised_outputs

    posterior, args, *_ = _make_posterior_and_args(tmp_path)
    n_barcodes = posterior.dataset_obj.data["matrix"].shape[0]

    full_file = str(tmp_path / "out.h5")
    filt_file = str(tmp_path / "out_filtered.h5")

    full_ok, _, _ = _write_streaming_denoised_outputs(
        posterior=posterior,
        estimator_obj=_MockEstimator(),
        noise_targets=None,
        args=args,
        fpr=0.01,
        fpr_output_filename=full_file,
        filtered_output_file=filt_file,
        global_latents={},
    )

    assert full_ok

    with tables.open_file(full_file, "r") as f:
        full_shape = f.root.matrix.shape[:]

    assert full_shape[1] == n_barcodes, f"Full H5 should have {n_barcodes} barcodes, got {full_shape[1]}"


def test_filtered_h5_latent_alignment(tmp_path):
    """Latent variables in the filtered H5 must be aligned to the correct barcodes.

    This is the key regression test for the sort-before-DuckDB-batch fix: if the
    per-cell latent arrays are reordered without being consistently permuted, a
    cell's cell_probability (or z-encoding) will appear under the wrong barcode.
    """
    from cellbender.remove_background.run import _write_streaming_denoised_outputs

    (
        posterior,
        args,
        cell_logic,
        barcodes_arr,
        p_analyzed,
        d_analyzed,
        eps_analyzed,
        z_analyzed,
    ) = _make_posterior_and_args(tmp_path)

    filt_file = str(tmp_path / "out_filtered.h5")

    _, filt_ok, _ = _write_streaming_denoised_outputs(
        posterior=posterior,
        estimator_obj=_MockEstimator(),
        noise_targets=None,
        args=args,
        fpr=0.01,
        fpr_output_filename=str(tmp_path / "out.h5"),
        filtered_output_file=filt_file,
        global_latents={},
    )

    assert filt_ok

    with tables.open_file(filt_file, "r") as f:
        filt_barcodes = f.root.matrix.barcodes[:]
        cell_prob = f.root.droplet_latents.cell_probability[:]
        cell_size = f.root.droplet_latents.cell_size[:]
        droplet_eff = f.root.droplet_latents.droplet_efficiency[:]
        z_enc = f.root.droplet_latents.gene_expression_encoding[:]
        bc_inds = f.root.droplet_latents.barcode_indices_for_latents[:]

    # Ground-truth latent values per absolute barcode index (from _make_posterior_and_args).
    # analyzed positions: [0,2,5,8,12,15,17,19]; cells at pos 2→abs5, pos 3→abs8, pos 6→abs17
    expected_p = {b"BC005": 0.99, b"BC008": 0.95, b"BC017": 0.91}
    expected_d = {b"BC005": 500.0, b"BC008": 400.0, b"BC017": 350.0}
    expected_eps = {b"BC005": 0.75, b"BC008": 0.70, b"BC017": 0.65}
    expected_z = {
        b"BC005": np.array([1.0, 0.0, 0.0, 0.0]),
        b"BC008": np.array([0.0, 1.0, 0.0, 0.0]),
        b"BC017": np.array([0.0, 0.0, 1.0, 0.0]),
    }

    for i, bc in enumerate(filt_barcodes):
        assert np.isclose(cell_prob[i], expected_p[bc], atol=1e-5), (
            f"cell_probability mismatch for {bc}: got {cell_prob[i]:.4f}, expected {expected_p[bc]}"
        )
        assert np.isclose(cell_size[i], expected_d[bc], atol=1e-2), (
            f"cell_size mismatch for {bc}: got {cell_size[i]}, expected {expected_d[bc]}"
        )
        assert np.isclose(droplet_eff[i], expected_eps[bc], atol=1e-5), (
            f"droplet_efficiency mismatch for {bc}: got {droplet_eff[i]:.4f}, expected {expected_eps[bc]}"
        )
        assert np.allclose(z_enc[i], expected_z[bc], atol=1e-5), (
            f"z encoding mismatch for {bc}: got {z_enc[i]}, expected {expected_z[bc]}"
        )

    # barcode_indices_for_latents must be exactly the absolute cell indices.
    assert set(bc_inds.tolist()) == {5, 8, 17}, (
        f"barcode_indices_for_latents should be {{5, 8, 17}}, got {set(bc_inds.tolist())}"
    )


def test_filtered_h5_no_negative_counts(tmp_path):
    """Denoised counts in the filtered H5 must be non-negative."""
    from cellbender.remove_background.run import _write_streaming_denoised_outputs

    posterior, args, *_ = _make_posterior_and_args(tmp_path)

    filt_file = str(tmp_path / "out_filtered.h5")

    _, filt_ok, _ = _write_streaming_denoised_outputs(
        posterior=posterior,
        estimator_obj=_MockEstimator(),
        noise_targets=None,
        args=args,
        fpr=0.01,
        fpr_output_filename=str(tmp_path / "out.h5"),
        filtered_output_file=filt_file,
        global_latents={},
    )

    assert filt_ok

    with tables.open_file(filt_file, "r") as f:
        data = f.root.matrix.data[:]
        indices = f.root.matrix.indices[:]
        indptr = f.root.matrix.indptr[:]
        shape = f.root.matrix.shape[:]

    matrix = sp.csc_matrix((data, indices, indptr), shape=tuple(shape)).toarray()
    assert np.all(matrix >= 0), "Filtered H5 contains negative denoised counts"
