"""Script to enable memory usage profiling via memory-profiler"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from cellbender.remove_background import consts
from cellbender.remove_background.checkpoint import load_from_checkpoint
from cellbender.remove_background.data.dataset import SingleCellRNACountsDataset
from cellbender.remove_background.estimation import MultipleChoiceKnapsack
from cellbender.remove_background.posterior import Posterior, compute_mean_target_removal_as_function
from cellbender.remove_background.sparse_utils import csr_set_rows_to_zero


def get_parser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(
        description="Run memory profiling on output count matrix generation. "
        "NOTE that you have to decorate "
        "MultipleChoiceKnapsack.estimate_noise_to_parquet() with memory_profiler's "
        "@profile() decorator manually.",
    )
    parser_.add_argument(
        "-f",
        "--checkpoint-file",
        type=str,
        required=True,
        dest="input_checkpoint_tarball",
        help="Saved CellBender checkpoint file ckpt.tar.gz",
    )
    parser_.add_argument("-i", "--input", type=str, required=True, dest="input_file", help="Input data file")
    return parser_


def get_noise_targets(posterior, fpr=0.01):
    count_matrix = posterior.dataset_obj.data["matrix"]  # all barcodes
    cell_inds = posterior.dataset_obj.analyzed_barcode_inds[posterior.latents_map["p"] > consts.CELL_PROB_CUTOFF]
    non_cell_row_logic = np.array([i not in cell_inds for i in range(count_matrix.shape[0])])
    cell_counts = csr_set_rows_to_zero(csr=count_matrix, row_logic=non_cell_row_logic)

    assert posterior.posterior_path is not None, "Posterior must be computed before target estimation."
    noise_target_fun_per_cell = compute_mean_target_removal_as_function(
        noise_count_posterior_coo=posterior.posterior_path,
        n_genes=posterior.n_genes,
        raw_count_csr_for_cells=cell_counts,
        n_cells=len(cell_inds),
        device="cpu",
        per_gene=True,
    )

    def noise_target_fun(x):
        return noise_target_fun_per_cell(x) * len(cell_inds)

    noise_targets = noise_target_fun(fpr).detach().cpu().numpy()
    return noise_targets


if __name__ == "__main__":
    # handle input arguments
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])

    # load checkpoint
    ckpt = load_from_checkpoint(
        tarball_name=args.input_checkpoint_tarball,
        filebase=None,
        to_load=["model", "posterior", "args"],
        force_device="cpu",
    )

    # load dataset
    dataset_obj = SingleCellRNACountsDataset(
        input_file=args.input_file,
        expected_cell_count=ckpt["args"].expected_cell_count,
        total_droplet_barcodes=ckpt["args"].total_droplets,
        fraction_empties=ckpt["args"].fraction_empties,
        model_name=ckpt["args"].model,
        gene_blacklist=ckpt["args"].blacklisted_genes,
        exclude_features=ckpt["args"].exclude_features,
        low_count_threshold=ckpt["args"].low_count_threshold,
        ambient_counts_in_cells_low_limit=ckpt["args"].ambient_counts_in_cells_low_limit,
        fpr=ckpt["args"].fpr,
    )

    # load posterior
    posterior = Posterior(
        dataset_obj=dataset_obj,
        vi_model=ckpt["model"],
        posterior_batch_size=ckpt["args"].posterior_batch_size,
        debug=False,
    )
    posterior.load(file=ckpt["posterior_file"])

    # run output count matrix generation
    noise_targets = get_noise_targets(posterior=posterior, fpr=0.01)
    assert isinstance(posterior.n_cells, int) and isinstance(posterior.n_genes, int)  # mypy
    estimator = MultipleChoiceKnapsack(n_cells=posterior.n_cells, n_genes=posterior.n_genes)
    with tempfile.NamedTemporaryFile(suffix="_noise.parquet", delete=False) as f:
        output_path = Path(f.name)
    assert posterior.posterior_path is not None
    estimator.estimate_noise_to_parquet(
        noise_log_prob_coo=posterior.posterior_path,
        output_path=output_path,
        noise_targets_per_gene=noise_targets,
    )

    sys.exit(0)
