"""Single run of remove-background, given input arguments."""

import argparse
import gc
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union, cast

import matplotlib
import matplotlib.backends.backend_pdf  # issue #287
import numpy as np
import pandas as pd
import psutil
import pyro
import scipy.sparse as sp
import torch
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO, Trace_ELBO, TraceEnum_ELBO
from pyro.optim import ClippedAdam

import cellbender
import cellbender.remove_background.consts as consts
from cellbender.remove_background.checkpoint import (
    attempt_load_checkpoint,
    create_workflow_hashcode,
    load_optim_from_bytes,
    save_checkpoint,
)
from cellbender.remove_background.data.dataprep import DataLoader, reconstruct_loader
from cellbender.remove_background.data.dataprep import prep_sparse_data_for_training as prep_data_for_training
from cellbender.remove_background.data.dataset import SingleCellRNACountsDataset, get_dataset_obj
from cellbender.remove_background.estimation import MAP, Mean, MultipleChoiceKnapsack, SingleSample, ThresholdCDF
from cellbender.remove_background.exceptions import ElboException
from cellbender.remove_background.model import RemoveBackgroundPyroModel
from cellbender.remove_background.posterior import (
    Posterior,
    compute_mean_target_removal_as_function,
    load_or_stream_posterior,
    sort_and_save_posterior,
)
from cellbender.remove_background.report import plot_summary, run_notebook_make_html
from cellbender.remove_background.sparse_utils import csr_set_rows_to_zero
from cellbender.remove_background.train import run_training
from cellbender.remove_background.vae.decoder import Decoder
from cellbender.remove_background.vae.encoder import CompositeEncoder, EncodeNonZLatents, EncodeZ

matplotlib.use("Agg")


logger = logging.getLogger("cellbender")


def run_remove_background(args: argparse.Namespace) -> None:
    """The full script for the command line tool to remove background RNA.

    Args:
        args: Inputs from the command line, already parsed using argparse.

    Note: Returns nothing, but writes output to a file(s) specified from
        command line.

    """

    # Set up checkpointing by creating a unique workflow hash.
    hashcode = create_workflow_hashcode(
        module_path=os.path.dirname(cellbender.__file__),
        args_to_remove=(
            [
                "output_file",
                "fpr",
                "input_checkpoint_tarball",
                "debug",
                "posterior_batch_size",
                "checkpoint_min",
                "truth_file",
                "cdf_threshold_q",
                "estimator",
                "use_multiprocessing_estimation",
                "cpu_threads",
                # The following settings do not affect the results, and can change when retrying,
                # so remove them.
                "epoch_elbo_fail_fraction",
                "final_elbo_fail_fraction",
                "num_failed_attempts",
                "checkpoint_filename",
            ]
            + (["epochs"] if args.constant_learning_rate else [])
        ),
        args=args,
    )[:10]
    args.checkpoint_filename = hashcode  # store this in args
    logger.info(f"(Workflow hash {hashcode})")

    # Handle initial random state.
    pyro.util.set_rng_seed(consts.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(consts.RANDOM_SEED)

    # Load dataset, run inference, and write the output to a file.

    # Log the start time.
    logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Running remove-background")

    # Run pytorch multithreaded if running on CPU: but this makes little difference in runtime.
    if not args.use_cuda:
        if args.n_threads is not None:
            n_jobs = args.n_threads
        else:
            n_jobs = psutil.cpu_count(logical=True)
        torch.set_num_threads(n_jobs)
        logger.debug(f"Set pytorch to use {n_jobs} threads")

    # Load data from file and choose barcodes and genes to analyze.
    try:
        dataset_obj = get_dataset_obj(args=args)

    except OSError:
        logger.error(f"OSError: Unable to open file {args.input_file}.")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Instantiate latent variable model and run full inference procedure.
    if args.model == "naive":
        inferred_model = None
    else:
        inferred_model, _sched, _train_loader, _test_loader = run_inference(
            dataset_obj=dataset_obj,
            args=args,
            output_checkpoint_tarball=args.input_checkpoint_tarball,
        )
        inferred_model.eval()
        # Training DataLoaders hold a copy of the count matrix that is no longer
        # needed. Free them now so that copy is eligible for GC.
        del _sched, _train_loader, _test_loader
        gc.collect()

    try:
        file_dir, file_base = os.path.split(args.output_file)
        file_name = os.path.splitext(os.path.basename(file_base))[0]

        # Stream the posterior to parquet (no sort yet).
        posterior = load_or_stream_posterior(
            dataset_obj=dataset_obj,
            inferred_model=inferred_model,
            args=args,
        )
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Drop all references to the model so GC can free it before the sort.
        # posterior.model_loss already has the cached loss curve.
        logger.info("Streaming complete. Freeing model memory before posterior sort...")
        del inferred_model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Model memory freed.")

        # Save output plots (uses cached model_loss; inferred_model no longer needed).
        save_output_plots(
            file_dir=file_dir,
            file_name=file_name,
            dataset_obj=dataset_obj,
            loss=posterior.model_loss,
            p=posterior.latents_map["p"],
            z=posterior.latents_map["z"],
        )

        # Sort parquet, save to checkpoint, apply regularization.
        sort_and_save_posterior(posterior=posterior, args=args)
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S\n"))

        # Save cell barcodes in a CSV file.
        analyzed_barcode_logic = posterior.latents_map["p"] > consts.CELL_PROB_CUTOFF
        assert dataset_obj.data is not None
        cell_barcodes = dataset_obj.data["barcodes"][dataset_obj.analyzed_barcode_inds[analyzed_barcode_logic]]
        bc_file_name = os.path.join(file_dir, file_name + "_cell_barcodes.csv")
        write_cell_barcodes_csv(bc_file_name=bc_file_name, cell_barcodes=cell_barcodes)

        # Compute estimates of denoised count matrix for each FPR and save them.
        compute_output_denoised_counts_and_metrics(
            posterior=posterior,
            args=args,
            file_dir=file_dir,
            file_name=file_name,
        )

        # Free large in-memory objects before generating reports.
        logger.info("Freeing large in-memory objects before generating reports...")
        del dataset_obj
        del posterior
        gc.collect()
        torch.cuda.empty_cache()

        # Generate HTML reports (reads only already-written output files).
        if not args.no_report:
            _generate_output_reports(args=args, file_dir=file_dir, file_name=file_name)

        logger.info("Completed remove-background.")
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S\n"))

        return None

    # The exception allows user to end inference prematurely with CTRL-C.
    except KeyboardInterrupt:
        # If partial output has been saved, delete it.
        full_file = args.output_file

        # Name of the filtered (cells only) file.
        file_dir, file_base = os.path.split(full_file)
        file_name = os.path.splitext(os.path.basename(file_base))[0]
        filtered_file = os.path.join(file_dir, file_name + "_filtered.h5")

        if os.path.exists(full_file):
            os.remove(full_file)

        if os.path.exists(filtered_file):
            os.remove(filtered_file)

        logger.info("Keyboard interrupt.  Terminated without saving.\n")
        sys.exit(1)


def save_output_plots(
    file_dir: str,
    file_name: str,
    dataset_obj: SingleCellRNACountsDataset,
    loss: Optional[dict],
    p: np.ndarray,
    z: np.ndarray,
) -> bool:
    """Save the UMI histogram and the three-panel output summary PDF"""

    try:
        # File naming.
        summary_fig_name = os.path.join(file_dir, file_name + ".pdf")

        # Three-panel output summary plot.
        counts = np.array(dataset_obj.get_count_matrix().sum(axis=1)).squeeze()
        fig = plot_summary(loss=loss or {}, umi_counts=counts, p=p, z=z)
        fig.savefig(summary_fig_name, bbox_inches="tight", format="pdf")
        logger.info(f"Saved summary plots as {summary_fig_name}")
        return True

    except Exception:
        logger.warning("Unable to save all plots.")
        logger.warning(traceback.format_exc())
        return False


def compute_output_denoised_counts_and_metrics(
    posterior: "Posterior",
    args: argparse.Namespace,
    file_dir: str,
    file_name: str,
) -> bool:
    """Write H5 outputs and metrics CSVs for every FPR. Does NOT generate reports.

    Extracts Pyro params once at entry and immediately clears the param store,
    freeing GPU tensors before any disk I/O begins.  For MCKP, frees the
    cell-count sparse matrix as soon as the noise-target closure is built.

    Args:
        posterior: Posterior object with computed posterior parquet.
        args: Parsed command-line arguments.
        file_dir: Directory for output files.
        file_name: Base filename (no extension).

    Returns:
        True iff all H5 and metric files were written successfully.
    """
    from cellbender.remove_background.estimation import EstimationMethod

    posterior.ensure_posterior_computed()
    assert posterior.dataset_obj is not None and posterior.dataset_obj.data is not None
    dataset_obj = posterior.dataset_obj

    # --- Extract Pyro params once, then clear the store ---
    ambient_expression_trimmed = pyro.param("chi_ambient").detach().cpu().numpy()
    assert dataset_obj.data is not None  # mypy
    total_genes_all = dataset_obj.data["matrix"].shape[1]
    ambient_expression = np.zeros(total_genes_all)
    ambient_expression[dataset_obj.analyzed_gene_inds] = ambient_expression_trimmed
    del ambient_expression_trimmed

    rho = None
    if ("rho_alpha" in pyro.get_param_store().keys()) and ("rho_beta" in pyro.get_param_store().keys()):
        rho = np.array(
            [
                pyro.param("rho_alpha").detach().cpu().numpy().item(),
                pyro.param("rho_beta").detach().cpu().numpy().item(),
            ]
        )

    global_latents: Dict[str, Any] = {
        "ambient_expression": ambient_expression,
        "empty_droplet_size_lognormal_loc": np.array(pyro.param("d_empty_loc").item()),
        "empty_droplet_size_lognormal_scale": np.array(pyro.param("d_empty_scale").item()),
        "cell_size_lognormal_std": np.array(pyro.param("d_cell_scale").item()),
        "swapping_fraction_dist_params": rho,
    }
    pyro.clear_param_store()
    logger.debug("Pyro param store cleared after extracting global latents.")

    # --- Choose estimator ---
    estimator: type[EstimationMethod]
    noise_target_fun = None
    noise_target_fun_per_cell = None

    analyzed_barcode_logic = posterior.latents_map["p"] > consts.CELL_PROB_CUTOFF
    cell_inds = dataset_obj.analyzed_barcode_inds[analyzed_barcode_logic]

    if args.estimator == "map":
        estimator = MAP
    elif args.estimator == "mean":
        estimator = Mean
    elif args.estimator == "sample":
        estimator = SingleSample
    elif args.estimator == "cdf":
        estimator = ThresholdCDF
    elif args.estimator == "mckp":
        estimator = MultipleChoiceKnapsack

        logger.info("Computing target noise counts per gene for MCKP estimator (two-pass GROUP BY)")
        count_matrix = dataset_obj.data["matrix"]
        empty_inds = set(range(count_matrix.shape[0])) - set(cell_inds)
        cell_counts = csr_set_rows_to_zero(csr=count_matrix, row_inds=empty_inds)

        assert posterior.posterior_path is not None, "Posterior must be computed before MCKP target estimation."
        assert posterior.n_genes is not None, "Posterior must have n_genes set before MCKP target estimation."
        noise_target_fun_per_cell = compute_mean_target_removal_as_function(
            noise_count_posterior_coo=posterior.posterior_path,
            n_genes=posterior.n_genes,
            raw_count_csr_for_cells=cell_counts,
            n_cells=len(cell_inds),
            device="cuda" if args.use_cuda else "cpu",
            per_gene=True,
        )
        del cell_counts  # closure does not capture it
        logger.info("Target noise counts per gene computed successfully")

        def noise_target_fun(x):
            return noise_target_fun_per_cell(x) * len(cell_inds)

    else:
        raise ValueError('Input --estimator must be one of ["map", "mean", "sample", "cdf", "mckp"]')

    # --- FPR loop: write H5s and metrics ---
    success = True
    for fpr in args.fpr:
        logger.debug(f"Working on FPR {fpr}")

        if noise_target_fun is not None:
            noise_targets = noise_target_fun(fpr).detach().cpu().numpy()
            logger.debug(f"Computed noise targets for FPR {fpr}:\n{noise_targets}")
            logger.info(f"Using MCKP noise targets computed for FPR {fpr}")
        else:
            noise_targets = None

        name_suffix = f"_FPR_{fpr}" if len(args.fpr) > 1 else ""
        fpr_output_filename = os.path.join(file_dir, file_name + name_suffix + ".h5")
        filtered_output_file = os.path.join(file_dir, file_name + name_suffix + "_filtered.h5")

        total_denoised_counts: Optional[float] = None

        logger.info(f"Computing denoised counts using {args.estimator} estimator (streaming)")
        assert posterior.n_cells is not None and posterior.n_genes is not None
        estimator_obj = estimator(n_cells=posterior.n_cells, n_genes=posterior.n_genes)
        full_ok, filt_ok, total_denoised_counts = _write_streaming_denoised_outputs(
            posterior=posterior,
            estimator_obj=estimator_obj,
            noise_targets=noise_targets,
            args=args,
            fpr=fpr,
            fpr_output_filename=fpr_output_filename,
            filtered_output_file=filtered_output_file,
            global_latents=global_latents,
        )
        success = success and full_ok and filt_ok

        try:
            df = collect_output_metrics(
                dataset_obj=dataset_obj,
                fpr=fpr,
                cell_logic=(posterior.latents_map["p"] >= consts.CELL_PROB_CUTOFF),
                loss=posterior.model_loss,
                total_denoised_counts=total_denoised_counts,
            )
            metrics_file_name = os.path.join(file_dir, file_name + name_suffix + "_metrics.csv")
            df.to_csv(metrics_file_name, index=True, header=False, float_format="%.3f")
            logger.info(f"Saved output metrics as {metrics_file_name}")
        except Exception:
            logger.warning("Unable to collect output metrics.")
            logger.warning(traceback.format_exc())

    if noise_target_fun is not None:
        del noise_target_fun, noise_target_fun_per_cell

    return success


def _generate_output_reports(
    args: argparse.Namespace,
    file_dir: str,
    file_name: str,
) -> None:
    """Generate HTML reports for every FPR by executing the report notebook.

    Reads only already-written output files — no large in-memory objects needed.

    Args:
        args: Parsed command-line arguments (needs ``fpr``, ``input_file``,
            ``output_file``, ``truth_file``).
        file_dir: Directory containing output files.
        file_name: Base filename (no extension).
    """
    for fpr in args.fpr:
        name_suffix = f"_FPR_{fpr}" if len(args.fpr) > 1 else ""
        fpr_output_filename = os.path.join(file_dir, file_name + name_suffix + ".h5")
        html_report_file = os.path.join(file_dir, file_name + name_suffix + "_report.html")
        try:
            os.environ["INPUT_FILE"] = os.path.abspath(os.path.join(os.getcwd(), args.input_file))
            os.environ["OUTPUT_FILE"] = os.path.abspath(os.path.join(os.getcwd(), fpr_output_filename))
            if args.truth_file is not None:
                os.environ["TRUTH_FILE"] = os.path.abspath(os.path.join(os.getcwd(), args.truth_file))
            run_notebook_make_html(
                file=os.path.abspath(os.path.join(os.path.dirname(__file__), "report.ipynb")),
                output=html_report_file,
            )
            logger.info(f"Succeeded in writing report to {html_report_file}")
        except Exception:
            logger.warning("Unable to create report.")
            logger.warning(traceback.format_exc())


def _write_streaming_denoised_outputs(
    posterior: Posterior,
    estimator_obj: "Any",
    noise_targets: Optional[np.ndarray],
    args: argparse.Namespace,
    fpr: float,
    fpr_output_filename: str,
    filtered_output_file: str,
    global_latents: Dict[str, Any],
) -> Tuple[bool, bool, float]:
    """Write full and filtered denoised H5 files via barcode-streaming.

    No full denoised CSC matrix is ever built in RAM.  Noise counts are
    written to a temp parquet, then ``stream_denoised_to_cellranger_h5``
    processes barcodes in batches and writes to H5 incrementally.

    Returns:
        (full_write_succeeded, filtered_write_succeeded, total_denoised_counts)
    """
    import tables as _tables

    from cellbender.remove_background.data.io import stream_denoised_to_cellranger_h5

    assert posterior.posterior_path is not None
    assert posterior.dataset_obj is not None and posterior.dataset_obj.data is not None
    dataset_obj = posterior.dataset_obj

    # Step 1: Write noise to a temp parquet (gene-sorted).
    noise_parquet = posterior.posterior_path.parent / (
        os.path.basename(fpr_output_filename).replace(".h5", "") + "_noise_tmp.parquet"
    )
    logger.info("Writing noise counts to parquet for streaming output")
    estimator_obj.estimate_noise_to_parquet(
        noise_log_prob_coo=posterior.posterior_path,
        output_path=noise_parquet,
        noise_targets_per_gene=noise_targets,
        q=args.cdf_threshold_q,
        duckdb_memory_limit=args.duckdb_memory_limit,
    )

    analyzed_barcode_inds = dataset_obj.analyzed_barcode_inds
    analyzed_barcode_logic = posterior.latents_map["p"] > consts.CELL_PROB_CUTOFF
    cell_inds = analyzed_barcode_inds[analyzed_barcode_logic]

    # Step 2: Build latents and metadata.
    latents = posterior.latents_map

    assert dataset_obj.data is not None  # mypy

    metadata: Dict[str, Any] = {
        "learning_curve": posterior.model_loss,
        "barcodes_analyzed": dataset_obj.data["barcodes"][analyzed_barcode_inds],
        "barcodes_analyzed_inds": analyzed_barcode_inds,
        "features_analyzed_inds": dataset_obj.analyzed_gene_inds,
        "fraction_data_used_for_testing": 1.0 - consts.TRAINING_FRACTION,
        "target_false_positive_rate": fpr,
    }
    metadata["estimator"] = [args.estimator]
    if args.cdf_threshold_q is not None:
        metadata["estimator_kwargs"] = {"q": args.cdf_threshold_q}

    # For the full H5: latents cover ALL analyzed barcodes (cells + empties).
    # anndata_from_h5 will subset the matrix to analyzed barcodes, so
    # latent arrays must have length n_analyzed_barcodes to appear in adata.obs.
    local_latents_full: Dict[str, Optional[np.ndarray]] = {
        "barcode_indices_for_latents": analyzed_barcode_inds,
        "gene_expression_encoding": latents["z"],
        "cell_size": latents["d"],
        "cell_probability": latents["p"],
        "droplet_efficiency": latents["epsilon"],
        # background_fraction appended after the write (needs denoised totals)
    }

    # For the filtered H5: latents cover cells only, sorted by ascending absolute
    # barcode index so that each DuckDB batch spans a tight cell_id range and can
    # skip the majority of parquet row groups (vs. near-full-table scans when
    # cells are in UMI-count order with widely scattered absolute indices).
    _cell_sort_perm = np.argsort(cell_inds)
    _cell_inds_sorted = cell_inds[_cell_sort_perm]
    local_latents_filtered: Dict[str, Optional[np.ndarray]] = {
        "barcode_indices_for_latents": _cell_inds_sorted,
        "gene_expression_encoding": latents["z"][analyzed_barcode_logic, :][_cell_sort_perm, :],
        "cell_size": latents["d"][analyzed_barcode_logic][_cell_sort_perm],
        "cell_probability": latents["p"][analyzed_barcode_logic][_cell_sort_perm],
        "droplet_efficiency": latents["epsilon"][analyzed_barcode_logic][_cell_sort_perm],
        # background_fraction appended after the write
    }

    filters_h5 = _tables.Filters(complevel=1, complib="zlib", shuffle=True)
    raw_matrix = dataset_obj.data["matrix"]
    raw_counts_analyzed = np.array(raw_matrix[analyzed_barcode_inds, :].sum(axis=1)).squeeze()

    shared_kwargs: Dict[str, Any] = dict(
        noise_parquet_path=noise_parquet,
        raw_count_matrix=raw_matrix,
        cell_logic=analyzed_barcode_logic,
        analyzed_barcode_inds=analyzed_barcode_inds,
        gene_names=dataset_obj.data["gene_names"],
        gene_ids=dataset_obj.data.get("gene_ids"),
        feature_types=dataset_obj.data.get("feature_types"),
        genomes=dataset_obj.data.get("genomes"),
        global_latents=global_latents,
        metadata=metadata,
        duckdb_memory_limit=args.duckdb_memory_limit,
    )

    # Step 3: Stream full H5 (all barcodes).
    full_write_succeeded = False
    total_denoised_counts = 0.0
    try:
        logger.info(f"Streaming denoised counts to {fpr_output_filename}")
        denoised_per_bc_full = stream_denoised_to_cellranger_h5(
            output_file=fpr_output_filename,
            barcodes=dataset_obj.data["barcodes"],
            barcode_subset=None,
            local_latents=local_latents_full,
            **shared_kwargs,
        )
        total_denoised_counts = float(denoised_per_bc_full.sum())
        full_write_succeeded = True
        # Compute and append background_fraction to full H5.
        # denoised_per_bc_full is indexed by absolute barcode; 0 for empties.
        out_analyzed = denoised_per_bc_full[analyzed_barcode_inds].astype(np.float64)
        bg_frac_full = (raw_counts_analyzed - out_analyzed) / (raw_counts_analyzed + 0.001)
        with _tables.open_file(fpr_output_filename, "a") as f:
            grp = f.get_node("/droplet_latents")
            f.create_carray(grp, "background_fraction", obj=bg_frac_full.astype(np.float64), filters=filters_h5)
    except Exception:
        logger.error("Failed to write full streaming H5.")
        logger.error(traceback.format_exc())

    # Step 4: Stream filtered H5 (cells only).
    # barcodes and barcode_subset are in ascending absolute-index order so that
    # each DuckDB range query spans a compact region of the sorted noise parquet.
    filtered_write_succeeded = False
    try:
        logger.info(f"Streaming denoised counts to {filtered_output_file}")
        denoised_per_bc_filt = stream_denoised_to_cellranger_h5(
            output_file=filtered_output_file,
            barcodes=dataset_obj.data["barcodes"][_cell_inds_sorted],
            barcode_subset=_cell_inds_sorted,
            local_latents=local_latents_filtered,
            **shared_kwargs,
        )
        filtered_write_succeeded = True
        # Compute and append background_fraction for cells.
        # raw_counts_cells must match the sorted order used above.
        raw_counts_cells = raw_counts_analyzed[analyzed_barcode_logic][_cell_sort_perm]
        out_cells = denoised_per_bc_filt.astype(np.float64)
        bg_frac_filt = (raw_counts_cells - out_cells) / (raw_counts_cells + 0.001)
        with _tables.open_file(filtered_output_file, "a") as f:
            grp = f.get_node("/droplet_latents")
            f.create_carray(grp, "background_fraction", obj=bg_frac_filt.astype(np.float64), filters=filters_h5)
    except Exception:
        logger.error("Failed to write filtered streaming H5.")
        logger.error(traceback.format_exc())

    # Step 5: Delete temp noise parquet.
    try:
        noise_parquet.unlink()
    except Exception:
        logger.warning(f"Could not delete temp noise parquet: {noise_parquet}")

    return full_write_succeeded, filtered_write_succeeded, total_denoised_counts


def collect_output_metrics(
    dataset_obj: SingleCellRNACountsDataset,
    fpr: Union[float, str],
    cell_logic,
    loss,
    inferred_count_matrix: Optional[sp.csr_matrix] = None,
    total_denoised_counts: Optional[float] = None,
) -> pd.DataFrame:
    """Create a table with a few output metrics. The idea is for these to
    potentially be used by people creating automated pipelines.

    Either ``inferred_count_matrix`` or ``total_denoised_counts`` must be given.
    When ``total_denoised_counts`` is supplied (streaming path), the matrix is
    not needed and should be omitted.
    """

    assert dataset_obj.data is not None
    assert inferred_count_matrix is not None or total_denoised_counts is not None, (
        "Provide either inferred_count_matrix or total_denoised_counts"
    )

    # Compute some metrics
    input_count_matrix = dataset_obj.data["matrix"][dataset_obj.analyzed_barcode_inds, :]
    total_raw_counts = dataset_obj.data["matrix"].sum()
    if total_denoised_counts is None:
        assert inferred_count_matrix is not None
        total_denoised_counts = float(inferred_count_matrix.sum())
    total_output_counts = total_denoised_counts
    total_counts_removed = total_raw_counts - total_output_counts
    fraction_counts_removed = total_counts_removed / total_raw_counts
    total_raw_counts_in_nonempty_droplets = input_count_matrix[cell_logic].sum()
    total_counts_removed_from_nonempty_droplets = total_raw_counts_in_nonempty_droplets - total_output_counts
    fraction_counts_removed_from_nonempty_droplets = (
        total_counts_removed_from_nonempty_droplets / total_raw_counts_in_nonempty_droplets
    )
    average_counts_removed_per_nonempty_droplet = total_counts_removed_from_nonempty_droplets / cell_logic.sum()
    expected_cells = dataset_obj.priors["expected_cells"]
    found_cells = cell_logic.sum()
    average_counts_per_cell = total_output_counts / found_cells
    ratio_of_found_cells_to_expected_cells = None if (expected_cells is None) else (found_cells / expected_cells)
    found_empties = len(dataset_obj.analyzed_barcode_inds) - found_cells
    fraction_of_analyzed_droplets_that_are_nonempty = found_cells / len(dataset_obj.analyzed_barcode_inds)
    if len(loss["train"]["elbo"]) > 20:
        # compare mean ELBO increase over last 3 steps to the typical end(ish) fluctuations
        convergence_indicator = np.mean(
            np.abs([(loss["train"]["elbo"][i] - loss["train"]["elbo"][i - 1]) for i in range(-3, -1)])
        ) / np.std(loss["train"]["elbo"][-20:])
    else:
        convergence_indicator = "not enough training epochs to compute (requires more than 20)"
    if len(loss["train"]["elbo"]) > 0:
        overall_change_in_train_elbo = loss["train"]["elbo"][-1] - loss["train"]["elbo"][0]
    else:
        overall_change_in_train_elbo = 0  # zero epoch initialization

    all_metrics_dict = {
        "total_raw_counts": total_raw_counts,
        "total_output_counts": total_output_counts,
        "total_counts_removed": total_counts_removed,
        "fraction_counts_removed": fraction_counts_removed,
        "total_raw_counts_in_cells": total_raw_counts_in_nonempty_droplets,
        "total_counts_removed_from_cells": total_counts_removed_from_nonempty_droplets,
        "fraction_counts_removed_from_cells": fraction_counts_removed_from_nonempty_droplets,
        "average_counts_removed_per_cell": average_counts_removed_per_nonempty_droplet,
        "target_fpr": fpr,
        "expected_cells": expected_cells,
        "found_cells": found_cells,
        "output_average_counts_per_cell": average_counts_per_cell,
        "ratio_of_found_cells_to_expected_cells": ratio_of_found_cells_to_expected_cells,
        "found_empties": found_empties,
        "fraction_of_analyzed_droplets_that_are_nonempty": fraction_of_analyzed_droplets_that_are_nonempty,
        "convergence_indicator": convergence_indicator,
        "overall_change_in_train_elbo": overall_change_in_train_elbo,
    }

    return pd.DataFrame(data=all_metrics_dict, index=["metric"]).transpose()


def write_cell_barcodes_csv(bc_file_name: str, cell_barcodes: np.ndarray):
    """Write the cell barcode CSV file.

    Args:
        bc_file_name: Output CSV file
        cell_barcodes: Array of the cell barcode names

    """

    # Save barcodes determined to contain cells as _cell_barcodes.csv
    try:
        barcode_names = np.array([str(cell_barcodes[i], encoding="UTF-8") for i in range(cell_barcodes.size)])
    except UnicodeDecodeError:
        # necessary if barcodes are ints
        barcode_names = cell_barcodes
    except TypeError:
        # necessary if barcodes are already decoded
        barcode_names = cell_barcodes
    np.savetxt(bc_file_name, barcode_names, delimiter=",", fmt="%s")
    logger.info(f"Saved cell barcodes in {bc_file_name}")


def get_optimizer(
    n_batches: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    constant_learning_rate: bool,
    total_epochs_for_testing_only: Optional[int] = None,
) -> Union[pyro.optim.PyroOptim, pyro.optim.lr_scheduler.PyroLRScheduler]:
    """Get optimizer or learning rate scheduler (if using one)"""

    # Set up the optimizer.
    optimizer = pyro.optim.clipped_adam.ClippedAdam  # just ClippedAdam does not work
    optimizer_args = {"lr": learning_rate, "clip_norm": 10.0}

    # Set up a learning rate scheduler.
    if total_epochs_for_testing_only is not None:
        total_steps = n_batches * total_epochs_for_testing_only
    else:
        total_steps = n_batches * epochs
    scheduler_args = {
        "optimizer": optimizer,
        "max_lr": learning_rate * 10,
        "total_steps": total_steps,
        "optim_args": optimizer_args,
    }
    OneCycleLR_cls = getattr(pyro.optim, "OneCycleLR")
    scheduler = cast(pyro.optim.PyroOptim, OneCycleLR_cls(scheduler_args))

    # Constant learning rate overrides the above and uses no scheduler.
    if constant_learning_rate:
        logger.info(
            "Using ClippedAdam --constant-learning-rate rather than "
            "the OneCycleLR schedule. This is not usually recommended."
        )
        scheduler = ClippedAdam(optimizer_args)

    return scheduler


def _build_model(
    count_matrix: sp.csr_matrix,
    args: argparse.Namespace,
    dataset_obj: SingleCellRNACountsDataset,
) -> RemoveBackgroundPyroModel:
    """Construct a fresh RemoveBackgroundPyroModel from dataset priors and args.
    Used by both fresh-start and checkpoint-restart branches of run_inference."""
    assert dataset_obj.data is not None
    encoder_z = EncodeZ(
        input_dim=count_matrix.shape[1],
        hidden_dims=args.z_hidden_dims,
        output_dim=args.z_dim,
        use_batch_norm=False,
        use_layer_norm=False,
        input_transform="normalize",
    )
    encoder_other = EncodeNonZLatents(
        n_genes=count_matrix.shape[1],
        z_dim=args.z_dim,
        log_count_crossover=dataset_obj.priors["log_counts_crossover"],
        prior_log_cell_counts=np.log1p(dataset_obj.priors["cell_counts"]),
        empty_log_count_threshold=np.log1p(dataset_obj.empty_UMI_threshold),
        prior_logit_cell_prob=dataset_obj.priors["cell_logit"],
        input_transform="log_normalize",
    )
    encoder = CompositeEncoder({"z": encoder_z, "other": encoder_other})
    decoder = Decoder(
        input_dim=args.z_dim,
        hidden_dims=args.z_hidden_dims[::-1],
        use_batch_norm=True,
        use_layer_norm=False,
        output_dim=count_matrix.shape[1],
    )
    return RemoveBackgroundPyroModel(
        model_type=args.model,
        encoder=encoder,
        decoder=decoder,
        dataset_obj_priors=dataset_obj.priors,
        n_analyzed_genes=dataset_obj.analyzed_gene_inds.size,
        n_droplets=dataset_obj.analyzed_barcode_inds.size,
        analyzed_gene_names=dataset_obj.data["gene_names"][dataset_obj.analyzed_gene_inds],
        empty_UMI_threshold=dataset_obj.empty_UMI_threshold,
        log_counts_crossover=dataset_obj.priors["log_counts_crossover"],
        use_cuda=args.use_cuda,
        z_hidden_dims=args.z_hidden_dims,
    )


def run_inference(
    dataset_obj: SingleCellRNACountsDataset,
    args: argparse.Namespace,
    output_checkpoint_tarball: str = consts.CHECKPOINT_FILE_NAME,
    total_epochs_for_testing_only: Optional[int] = None,
) -> Tuple[RemoveBackgroundPyroModel, pyro.optim.PyroOptim, DataLoader, DataLoader]:
    """Run a full inference procedure, training a latent variable model.

    Args:
        dataset_obj: Input data in the form of a SingleCellRNACountsDataset
            object.
        args: Input command line parsed arguments.
        output_checkpoint_tarball: Intended checkpoint tarball filepath.
        total_epochs_for_testing_only: Hack for testing code using LR scheduler

    Returns:
         model: cellbender.model.RemoveBackgroundPyroModel that has had
            inference run.

    """

    assert dataset_obj.data is not None

    # Get the checkpoint file base name with hash, which we stored in args.
    checkpoint_filename = args.checkpoint_filename

    # Configure pyro options (skip validations to improve speed).
    pyro.enable_validation(False)
    pyro.distributions.enable_validation(False)

    # Set random seed, updating global state of python, numpy, and torch RNGs.
    pyro.clear_param_store()
    pyro.set_rng_seed(consts.RANDOM_SEED)
    if args.use_cuda:
        torch.cuda.manual_seed_all(consts.RANDOM_SEED)

    # Attempt to load from a previously-saved checkpoint.
    ckpt = attempt_load_checkpoint(
        filebase=checkpoint_filename,
        tarball_name=args.input_checkpoint_tarball,
        force_device="cuda:0" if args.use_cuda else "cpu",
        force_use_checkpoint=args.force_use_checkpoint,
    )
    ckpt_loaded = ckpt["loaded"]  # True if a checkpoint was loaded successfully

    # Always load the count matrix — needed for both fresh start and ckpt reconstruction.
    count_matrix = dataset_obj.get_count_matrix()
    empty_matrix = dataset_obj.get_count_matrix_empties()

    # Backed mode: mmap files live next to the output file so they survive between runs.
    mmap_cache_dir = None
    if getattr(args, "backed_mode", False):
        from pathlib import Path

        mmap_cache_dir = Path(args.output_file).with_suffix("").parent / (Path(args.output_file).stem + "_mmap")
        logger.info(f"Backed mode enabled: mmap cache at {mmap_cache_dir}")

    if ckpt_loaded:
        logger.info("Reconstructing model and dataloaders from checkpoint state...")

        map_loc = torch.device("cuda:0") if args.use_cuda else torch.device("cpu")

        # Phase 2: rebuild model architecture from scratch, then load saved weights.
        # Save/restore torch RNG around _build_model so that weight-init draws do
        # not advance torch past the checkpoint-restored state.  Training resumes
        # from the exact same torch state as when the checkpoint was saved.
        _torch_state_before_build = torch.get_rng_state()
        model = _build_model(count_matrix, args, dataset_obj)
        torch.set_rng_state(_torch_state_before_build)
        model.load_state_dict(ckpt["model_state_dict"])

        # The Pyro param store was already populated from "_params.pyro" by
        # attempt_load_checkpoint above.  "_params.pyro" now contains ONLY
        # scalar (non-module) params (phi_loc, phi_scale, etc.) when the
        # checkpoint was written by a current version of CellBender.  For
        # backwards compatibility, also purge any encoder/decoder entries that
        # may be present in checkpoints written by older versions — their
        # values are superseded by model.load_state_dict above.
        _ps = pyro.get_param_store()
        _model_pyro_prefixes = ("encoder_z$$$", "encoder_other$$$", "decoder$$$")
        for _k in list(_ps._params.keys()):
            if _k.startswith(_model_pyro_prefixes):
                _p = _ps._params.pop(_k)
                _ps._param_to_name.pop(_p, None)
                _ps._constraints.pop(_k, None)

        # Pre-register encoder/decoder parameters in the param store NOW,
        # before any svi.step() call.  In the one-shot training path, these
        # params are registered on the very first svi.step() (epoch 1, step 1)
        # and remain registered for all subsequent steps.  Without this block,
        # the checkpoint-resume path registers them for the first time INSIDE
        # the first svi.step() of resumed training.  Although the parameter
        # values are identical in both cases, the timing difference causes a
        # subtle divergence in the TraceEnum_ELBO computation (the
        # poutine.trace handler sees param effects behave differently when a
        # param is being registered for the first time vs. looked up), leading
        # to non-bit-for-bit reproducibility across checkpoints.
        # By pre-registering here, the param store state at the start of
        # resumed training exactly matches the one-shot state, making
        # checkpoint-resume produce bit-for-bit identical results.
        for _enc_name, _enc_module in model.encoder.items():
            pyro.module("encoder_" + _enc_name, _enc_module, update_module_params=False)
        pyro.module("decoder", model.decoder, update_module_params=False)
        del _ps

        if "model_meta" in ckpt:
            model.loss = ckpt["model_meta"]["loss"]

        # Phase 1: reconstruct dataloaders from compact index state.
        train_loader = reconstruct_loader(
            ckpt["train_loader_state"],
            count_matrix,
            empty_matrix,
            args.use_cuda,
            mmap_cache_dir,
            getattr(args, "dataloader_workers", 0),
        )
        test_loader = reconstruct_loader(
            ckpt["test_loader_state"],
            count_matrix,
            empty_matrix,
            args.use_cuda,
            mmap_cache_dir,
            getattr(args, "dataloader_workers", 0),
        )

        # Phase 3: rebuild optimizer from args + restore saved state.
        scheduler = get_optimizer(
            n_batches=len(train_loader),
            batch_size=train_loader.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            constant_learning_rate=args.constant_learning_rate,
            total_epochs_for_testing_only=total_epochs_for_testing_only,
        )
        load_optim_from_bytes(scheduler, ckpt["optim_state_bytes"], map_location=map_loc)

        if hasattr(ckpt.get("args", argparse.Namespace()), "num_failed_attempts"):
            args.num_failed_attempts = ckpt["args"].num_failed_attempts
        logger.info("Checkpoint loaded successfully.")

    else:
        logger.info("No checkpoint loaded.")

        # Set up the variational autoencoder using the shared helper.
        model = _build_model(count_matrix, args, dataset_obj)

        # Load the dataset into DataLoaders.
        frac = args.training_fraction  # Fraction of barcodes to use for training
        batch_size = int(min(consts.MAX_BATCH_SIZE, frac * dataset_obj.analyzed_barcode_inds.size / 2))

        # Set up dataloaders.
        train_loader, test_loader = prep_data_for_training(
            dataset=count_matrix,
            empty_drop_dataset=empty_matrix,
            batch_size=batch_size,
            training_fraction=frac,
            fraction_empties=args.fraction_empties,
            shuffle=True,
            use_cuda=args.use_cuda,
            mmap_cache_dir=mmap_cache_dir,
            num_workers=getattr(args, "dataloader_workers", 0),
        )

        # Set up optimizer (optionally wrapped in a learning rate scheduler).
        scheduler = get_optimizer(
            n_batches=len(train_loader),
            batch_size=train_loader.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            constant_learning_rate=args.constant_learning_rate,
            total_epochs_for_testing_only=total_epochs_for_testing_only,
        )

    # Determine the loss function.
    if args.use_jit:
        # Call guide() once as a warm-up.
        # model.guide(torch.zeros([10, dataset_obj.analyzed_gene_inds.size]).to(model.device))

        if args.model == "simple":
            loss_function: JitTrace_ELBO | JitTraceEnum_ELBO | Trace_ELBO | TraceEnum_ELBO = JitTrace_ELBO()
        else:
            loss_function = JitTraceEnum_ELBO(max_plate_nesting=1, strict_enumeration_warning=False)
    else:
        if args.model == "simple":
            loss_function = Trace_ELBO()
        else:
            loss_function = TraceEnum_ELBO(max_plate_nesting=1)

    # Set up the inference process.
    svi = SVI(model.model, model.guide, scheduler, loss=loss_function)

    # Run training.
    if args.epochs == 0:
        logger.info("Zero epochs specified... will only initialize the model.")
        _init_batch = train_loader.__next__().to(train_loader.device, non_blocking=True)
        model.guide(_init_batch)
        train_loader.reset_ptr()

        # Even though it's not much of a checkpoint, we still need one for subsequent steps.
        save_checkpoint(
            filebase=checkpoint_filename,
            tarball_name=output_checkpoint_tarball,
            args=args,
            model_obj=model,
            scheduler=svi.optim,
            train_loader=train_loader,
            test_loader=test_loader,
        )

    else:
        logger.info("Running inference...")
        try:
            run_training(
                model=model,
                args=args,
                svi=svi,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                test_freq=5,
                output_filename=checkpoint_filename,
                ckpt_tarball_name=output_checkpoint_tarball,
                checkpoint_freq=args.checkpoint_min,
                epoch_elbo_fail_fraction=args.epoch_elbo_fail_fraction,
                final_elbo_fail_fraction=args.final_elbo_fail_fraction,
            )

        except ElboException:
            logger.warning(traceback.format_exc())

            # Keep track of number of failed attempts.
            if not hasattr(args, "num_failed_attempts"):
                args.num_failed_attempts = 1
            else:
                args.num_failed_attempts = args.num_failed_attempts + 1
            logger.debug(f"Training failed, and the number of failed attempts on record is {args.num_failed_attempts}")

            # Retry training with reduced learning rate, if indicated by user.
            logger.debug(f"Number of times to retry training is {args.num_training_tries}")
            if args.num_failed_attempts < args.num_training_tries:
                args.learning_rate = args.learning_rate * args.learning_rate_retry_mult
                logger.info(
                    f"Restarting training: attempt {args.num_failed_attempts + 1}, learning_rate = {args.learning_rate}"
                )
                run_remove_background(args)  # start from scratch
                sys.exit(0)
            else:
                logger.info(
                    "No more attempts are specified by --num-training-tries. "
                    "Therefore the workflow will run once more without ELBO restrictions."
                )
                args.epoch_elbo_fail_fraction = None
                args.final_elbo_fail_fraction = None
                run_remove_background(args)  # start from scratch
                # non-zero exit status in order to draw user's attention to the fact that ELBO tests
                # were never satisfied.
                sys.exit(1)

        logger.info("Inference procedure complete.")

    return model, scheduler, train_loader, test_loader
