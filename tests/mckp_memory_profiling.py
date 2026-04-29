"""Script to enable memory usage profiling via memory-profiler"""

from cellbender.remove_background.estimation import MultipleChoiceKnapsack
from cellbender.remove_background.sparse_utils import csr_set_rows_to_zero
from cellbender.remove_background.checkpoint import load_from_checkpoint
from cellbender.remove_background.posterior import Posterior, \
    compute_mean_target_removal_as_function
from cellbender.remove_background.data.dataset import SingleCellRNACountsDataset
from cellbender.remove_background import consts

import scipy.sparse as sp
import numpy as np
from memory_profiler import profile

import argparse
import sys


def get_parser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(
        description="Run memory profiling on output count matrix generation. "
                    "NOTE that you have to decorate "
                    "MultipleChoiceKnapsack.estimate_noise() with memory_profiler's "
                    "@profile() decorator manually.",
    )
    parser_.add_argument('-f', '--checkpoint-file',
                         type=str,
                         required=True,
                         dest='input_checkpoint_tarball',
                         help='Saved CellBender checkpoint file ckpt.tar.gz')
    parser_.add_argument('-i', '--input',
                         type=str,
                         required=True,
                         dest='input_file',
                         help='Input data file')
    return parser_


def compute_noise_counts(posterior,
                         fpr: float,
                         estimator_constructor: 'EstimationMethod',
                         **kwargs) -> sp.csr_matrix:
    """Probably the most important method: computation of the clean output count matrix.

    Args:
        estimator_constructor: A noise count estimator class derived from
            the EstimationMethod base class, and implementing the
            .estimate_noise() method, which creates a point estimate of
            noise. Pass in the constructor, not an object.
        **kwargs: Keyword arguments for estimator_constructor().estimate_noise()

    Returns:
        denoised_counts: Denoised output CSC sparse matrix (CSC for saving)

    """

    # Only compute using defaults if the cache is empty.
    if posterior._noise_count_regularized_posterior_coo is not None:
        # Priority is taken by a regularized posterior, since presumably
        # the user computed it for a reason.
        posterior_coo = posterior._noise_count_regularized_posterior_coo
    else:
        # Use exact posterior if a regularized version is not computed.
        posterior_coo = (posterior._noise_count_posterior_coo
                         if (posterior._noise_count_posterior_coo is not None)
                         else posterior.cell_noise_count_posterior_coo())

    # Instantiate Estimator object.
    estimator = estimator_constructor(index_converter=posterior.index_converter)

    # Compute point estimate of noise in cells.
    noise_targets = get_noise_targets(posterior=posterior, fpr=fpr)
    noise_csr = estimator.estimate_noise(
        estimator=estimator,
        noise_log_prob_coo=posterior_coo,
        noise_offsets=posterior._noise_count_posterior_coo_offsets,
        noise_targets_per_gene=noise_targets,
        **kwargs,
    )

    return noise_csr


def get_noise_targets(posterior, fpr=0.01):
    count_matrix = posterior.dataset_obj.data['matrix']  # all barcodes
    cell_inds = posterior.dataset_obj.analyzed_barcode_inds[posterior.latents_map['p']
                                                            > consts.CELL_PROB_CUTOFF]
    non_cell_row_logic = np.array([i not in cell_inds
                                   for i in range(count_matrix.shape[0])])
    cell_counts = csr_set_rows_to_zero(csr=count_matrix, row_logic=non_cell_row_logic)

    noise_target_fun_per_cell = compute_mean_target_removal_as_function(
        noise_count_posterior_coo=posterior._noise_count_posterior_coo,
        noise_offsets=posterior._noise_count_posterior_coo_offsets,
        index_converter=posterior.index_converter,
        raw_count_csr_for_cells=cell_counts,
        n_cells=len(cell_inds),
        device='cpu',
        per_gene=True,
    )
    noise_target_fun = lambda x: noise_target_fun_per_cell(x) * len(cell_inds)
    noise_targets = noise_target_fun(fpr).detach().cpu().numpy()
    return noise_targets


if __name__ == "__main__":

    # handle input arguments
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])

    # load checkpoint
    ckpt = load_from_checkpoint(tarball_name=args.input_checkpoint_tarball,
                                filebase=None,
                                to_load=['model', 'posterior', 'args'],
                                force_device='cpu')

    # load dataset
    dataset_obj = \
        SingleCellRNACountsDataset(input_file=args.input_file,
                                   expected_cell_count=ckpt['args'].expected_cell_count,
                                   total_droplet_barcodes=ckpt['args'].total_droplets,
                                   fraction_empties=ckpt['args'].fraction_empties,
                                   model_name=ckpt['args'].model,
                                   gene_blacklist=ckpt['args'].blacklisted_genes,
                                   exclude_features=ckpt['args'].exclude_features,
                                   low_count_threshold=ckpt['args'].low_count_threshold,
                                   ambient_counts_in_cells_low_limit=ckpt['args'].ambient_counts_in_cells_low_limit,
                                   fpr=ckpt['args'].fpr)

    # load posterior
    posterior = Posterior(
        dataset_obj=dataset_obj,
        vi_model=ckpt['model'],
        posterior_batch_size=ckpt['args'].posterior_batch_size,
        debug=False,
    )
    posterior.load(file=ckpt['posterior_file'])

    # run output count matrix generation
    compute_noise_counts(posterior=posterior,
                         fpr=0.01,
                         estimator_constructor=MultipleChoiceKnapsack,
                         approx_gb=0.1)

    sys.exit(0)
