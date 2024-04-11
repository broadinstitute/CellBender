"""Single run of remove-background, given input arguments."""

import cellbender
from cellbender.remove_background.model import RemoveBackgroundPyroModel
from cellbender.remove_background.data.dataset import get_dataset_obj, \
    SingleCellRNACountsDataset
from cellbender.remove_background.vae.decoder import Decoder
from cellbender.remove_background.vae.encoder \
    import EncodeZ, CompositeEncoder, EncodeNonZLatents
from cellbender.remove_background.data.dataprep import \
    prep_sparse_data_for_training as prep_data_for_training
from cellbender.remove_background.checkpoint import attempt_load_checkpoint, \
    save_checkpoint
import cellbender.remove_background.consts as consts
from cellbender.remove_background.data.dataprep import DataLoader
from cellbender.remove_background.train import run_training
from cellbender.remove_background.posterior import Posterior, PRq, PRmu, \
    compute_mean_target_removal_as_function, load_or_compute_posterior_and_save
from cellbender.remove_background.estimation import MultipleChoiceKnapsack, \
    MAP, Mean, SingleSample, ThresholdCDF
from cellbender.remove_background.exceptions import ElboException
from cellbender.remove_background.sparse_utils import csr_set_rows_to_zero
from cellbender.remove_background.data.io import write_matrix_to_cellranger_h5
from cellbender.remove_background.report import run_notebook_make_html, plot_summary
from cellbender.remove_background.checkpoint import create_workflow_hashcode

import pyro
from pyro.infer import SVI, JitTraceEnum_ELBO, JitTrace_ELBO, \
    TraceEnum_ELBO, Trace_ELBO
from pyro.optim import ClippedAdam
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch

import os
import sys
import logging
import argparse
import traceback
import psutil
from datetime import datetime
from typing import Tuple, Optional, Dict, Union

import matplotlib
import matplotlib.backends.backend_pdf  # issue #287
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # This needs to be after matplotlib.use('Agg')


logger = logging.getLogger('cellbender')


def run_remove_background(args: argparse.Namespace) -> Posterior:
    """The full script for the command line tool to remove background RNA.

    Args:
        args: Inputs from the command line, already parsed using argparse.

    Note: Returns nothing, but writes output to a file(s) specified from
        command line.

    """

    # Set up checkpointing by creating a unique workflow hash.
    hashcode = create_workflow_hashcode(
        module_path=os.path.dirname(cellbender.__file__),
        args_to_remove=(['output_file', 'fpr', 'input_checkpoint_tarball', 'debug',
                         'posterior_batch_size', 'checkpoint_min', 'truth_file',
                         'posterior_regularization', 'cdf_threshold_q', 'prq_alpha',
                         'estimator', 'use_multiprocessing_estimation', 'cpu_threads',
                         # The following settings do not affect the results, and can change when retrying,
                         # so remove them.
                         'epoch_elbo_fail_fraction', 'final_elbo_fail_fraction',
                         'num_failed_attempts', 'checkpoint_filename']
                        + (['epochs'] if args.constant_learning_rate else [])),
        args=args)[:10]
    args.checkpoint_filename = hashcode  # store this in args
    logger.info(f'(Workflow hash {hashcode})')

    # Handle initial random state.
    pyro.util.set_rng_seed(consts.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(consts.RANDOM_SEED)

    # Load dataset, run inference, and write the output to a file.

    # Log the start time.
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("Running remove-background")

    # Run pytorch multithreaded if running on CPU: but this makes little difference in runtime.
    if not args.use_cuda:
        if args.n_threads is not None:
            n_jobs = args.n_threads
        else:
            n_jobs = psutil.cpu_count(logical=True)
        torch.set_num_threads(n_jobs)
        logger.debug(f'Set pytorch to use {n_jobs} threads')

    # Load data from file and choose barcodes and genes to analyze.
    try:
        dataset_obj = get_dataset_obj(args=args)

    except OSError:
        logger.error(f"OSError: Unable to open file {args.input_file}.")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Instantiate latent variable model and run full inference procedure.
    if args.model == 'naive':
        inferred_model = None
    else:
        inferred_model, _, _, _ = run_inference(dataset_obj=dataset_obj, args=args)
        inferred_model.eval()

    try:

        file_dir, file_base = os.path.split(args.output_file)
        file_name = os.path.splitext(os.path.basename(file_base))[0]

        # Create the posterior and save it.
        posterior = load_or_compute_posterior_and_save(
            dataset_obj=dataset_obj,
            inferred_model=inferred_model,
            args=args,
        )
        logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S\n'))

        # Save output plots.
        save_output_plots(
            file_dir=file_dir,
            file_name=file_name,
            dataset_obj=dataset_obj,
            inferred_model=inferred_model,
            p=posterior.latents_map['p'],
            z=posterior.latents_map['z'],
        )

        # Save cell barcodes in a CSV file.
        analyzed_barcode_logic = (posterior.latents_map['p'] > consts.CELL_PROB_CUTOFF)
        cell_barcodes = (dataset_obj.data['barcodes']
                         [dataset_obj.analyzed_barcode_inds[analyzed_barcode_logic]])
        bc_file_name = os.path.join(file_dir, file_name + "_cell_barcodes.csv")
        write_cell_barcodes_csv(bc_file_name=bc_file_name, cell_barcodes=cell_barcodes)

        # Compute estimates of denoised count matrix for each FPR and save them.
        compute_output_denoised_counts_reports_metrics(
            posterior=posterior,
            args=args,
            file_dir=file_dir,
            file_name=file_name,
        )

        logger.info("Completed remove-background.")
        logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S\n'))

        return posterior

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


def save_output_plots(file_dir: str,
                      file_name: str,
                      dataset_obj: SingleCellRNACountsDataset,
                      inferred_model: RemoveBackgroundPyroModel,
                      p: np.ndarray,
                      z: np.ndarray) -> bool:
    """Save the UMI histogram and the three-panel output summary PDF"""

    try:
        # File naming.
        summary_fig_name = os.path.join(file_dir, file_name + ".pdf")

        # Three-panel output summary plot.
        counts = np.array(dataset_obj.get_count_matrix().sum(axis=1)).squeeze()
        fig = plot_summary(loss=inferred_model.loss, umi_counts=counts, p=p, z=z)
        fig.savefig(summary_fig_name, bbox_inches='tight', format='pdf')
        logger.info(f"Saved summary plots as {summary_fig_name}")
        return True

    except Exception:
        logger.warning("Unable to save all plots.")
        logger.warning(traceback.format_exc())
        return False


def compute_output_denoised_counts_reports_metrics(posterior: Posterior,
                                                   args: argparse.Namespace,
                                                   file_dir: str,
                                                   file_name: str) -> bool:
    """Handle the estimation of the output denoised count matrix, given a
    posterior.  Compute the estimate for all specified FPR values.  Save each
    as we go.  Create reports and save, and aggregate metrics and save as we go.

    Args:
        posterior: Posterior object
        args: Argparser namespace with all parsed input args
        file_dir:
        file_name:

    Returns:
        True iff all files write correctly

    """

    # Ensure that the posterior distribution has been computed.
    # TODO: when we make the properties non-private, then this should become obsolete
    posterior.cell_noise_count_posterior_coo()

    # Choose output count matrix estimation method.
    noise_target_fun = None
    if args.estimator == 'map':
        estimator = MAP
    elif args.estimator == 'mean':
        estimator = Mean
    elif args.estimator == 'sample':
        estimator = SingleSample
    elif args.estimator == 'cdf':
        estimator = ThresholdCDF
    elif args.estimator == 'mckp':
        estimator = MultipleChoiceKnapsack

        # Prep specific for MCKP: target estimation.
        logger.info('Computing target noise counts per gene for MCKP estimator')
        count_matrix = posterior.dataset_obj.data['matrix']  # all barcodes
        cell_inds = posterior.dataset_obj.analyzed_barcode_inds[posterior.latents_map['p']
                                                                > consts.CELL_PROB_CUTOFF]
        empty_inds = set(range(count_matrix.shape[0])) - set(cell_inds)
        cell_counts = csr_set_rows_to_zero(csr=count_matrix, row_inds=empty_inds)

        noise_target_fun_per_cell = compute_mean_target_removal_as_function(
            noise_count_posterior_coo=posterior._noise_count_posterior_coo,
            noise_offsets=posterior._noise_count_posterior_coo_offsets,
            index_converter=posterior.index_converter,
            raw_count_csr_for_cells=cell_counts,
            n_cells=len(cell_inds),
            device='cuda' if args.use_cuda else 'cpu',  # TODO check this
            per_gene=True,
        )
        noise_target_fun = lambda x: noise_target_fun_per_cell(x) * len(cell_inds)
    else:
        raise ValueError(f'Input --estimator must be one of '
                         f'["map", "mean", "sample", "cdf", "mckp"]')

    # Save denoised count matrix outputs (for each FPR if applicable).
    success = True
    for fpr in args.fpr:

        logger.debug(f'Working on FPR {fpr}')

        # Regularize posterior again at this FPR, if needed.
        if args.posterior_regularization in ['PRmu', 'PRmu_gene']:
            # TODO: currently this re-calculates the first FPR which is not necessary
            posterior.regularize_posterior(
                regularization=PRmu,
                raw_count_matrix=posterior.dataset_obj.data['matrix'],
                fpr=fpr,
                per_gene=True if (args.posterior_regularization == 'PRmu_gene') else False,
                device='cuda',
            )
        else:
            # Other posterior regularizations were already performed in
            # posterior.load_or_compute_posterior_and_save()
            pass

        # If using MCKP, recompute noise targets for this FPR.
        if noise_target_fun is not None:
            noise_targets = noise_target_fun(fpr).detach().cpu().numpy()
            logger.debug(f'Computed noise targets for FPR {fpr}:\n{noise_targets}')
            logger.info(f'Using MCKP noise targets computed for FPR {fpr}')
        else:
            noise_targets = None

        # Compute denoised counts.
        logger.info(f'Computing denoised counts using {args.estimator} estimator')
        denoised_counts = posterior.compute_denoised_counts(
            estimator_constructor=estimator,
            noise_targets_per_gene=noise_targets,
            q=args.cdf_threshold_q,
            alpha=args.prq_alpha,
            device='cuda' if args.use_cuda else 'cpu',
            use_multiple_processes=args.use_multiprocessing_estimation,
        )

        # Restore eliminated features in cells.
        logger.debug('Restoring eliminated features in cells')
        denoised_counts = posterior.dataset_obj.restore_eliminated_features_in_cells(
            denoised_counts,
            posterior.latents_map['p'],
        )

        # Failsafe to ensure no negative counts.
        assert np.all(denoised_counts.data >= 0), 'Negative count matrix entries in output'

        # TODO: correct cell probabilities so that any zero-count droplet becomes "empty"

        # Save denoised count matrix.
        name_suffix = (f'_FPR_{fpr}' if len(args.fpr) > 1 else '')
        fpr_output_filename = os.path.join(file_dir, file_name + name_suffix + '.h5')
        filtered_output_file = os.path.join(file_dir, file_name + name_suffix + '_filtered.h5')

        def _writer_helper(file, **kwargs) -> bool:
            return write_denoised_count_matrix(
                file=file,
                denoised_count_matrix=denoised_counts,
                posterior_regularization=args.posterior_regularization,
                posterior_regularization_kwargs=posterior._noise_count_regularized_posterior_kwargs,
                estimator=args.estimator,
                estimator_kwargs=None if (args.cdf_threshold_q is None) else {'q': args.cdf_threshold_q},
                latents=posterior.latents_map,
                dataset_obj=posterior.dataset_obj,
                learning_curve=posterior.vi_model.loss,
                fpr=fpr,
                **kwargs,
            )

        # Full count matrix.
        write_succeeded = _writer_helper(file=fpr_output_filename)
        success = success and write_succeeded

        # Count matrix filtered to cells only.
        analyzed_barcode_logic = (posterior.latents_map['p'] > consts.CELL_PROB_CUTOFF)
        write_succeeded = _writer_helper(
            file=filtered_output_file,
            analyzed_barcode_logic=analyzed_barcode_logic,
            barcode_inds=posterior.dataset_obj.analyzed_barcode_inds[analyzed_barcode_logic],
        )
        success = success and write_succeeded

        # Compile and save metrics.
        try:
            df = collect_output_metrics(
                dataset_obj=posterior.dataset_obj,
                inferred_count_matrix=denoised_counts,
                fpr=fpr,
                cell_logic=(posterior.latents_map['p'] >= consts.CELL_PROB_CUTOFF),
                loss=posterior.vi_model.loss,
            )
            metrics_file_name = os.path.join(file_dir, file_name + name_suffix + '_metrics.csv')
            df.to_csv(metrics_file_name, index=True, header=False, float_format='%.3f')
            logger.info(f'Saved output metrics as {metrics_file_name}')
        except Exception:
            logger.warning("Unable to collect output metrics.")
            logger.warning(traceback.format_exc())

        # Create report.
        try:
            os.environ['INPUT_FILE'] = os.path.abspath(os.path.join(os.getcwd(), args.input_file))
            os.environ['OUTPUT_FILE'] = os.path.abspath(os.path.join(os.getcwd(), fpr_output_filename))
            if args.truth_file is not None:
                os.environ['TRUTH_FILE'] = os.path.abspath(os.path.join(os.getcwd(), args.truth_file))
            html_report_file = os.path.join(file_dir, file_name + name_suffix + '_report.html')
            run_notebook_make_html(
                file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'report.ipynb')),
                output=html_report_file,
            )
            logger.info(f'Succeeded in writing report to {html_report_file}')

        except Exception:
            logger.warning("Unable to create report.")
            logger.warning(traceback.format_exc())

    return success


def write_denoised_count_matrix(file: str,
                                denoised_count_matrix: sp.csr_matrix,
                                posterior_regularization: Optional[str],
                                posterior_regularization_kwargs: Optional[Dict[str, float]],
                                estimator: str,
                                estimator_kwargs: Optional[Dict[str, float]],
                                latents: Dict[str, np.ndarray],
                                dataset_obj: SingleCellRNACountsDataset,
                                learning_curve: Dict[str, np.ndarray],  # inferred_model.loss
                                fpr: float,
                                analyzed_barcode_logic: np.ndarray = ...,
                                barcode_inds: np.ndarray = ...) -> bool:
    """Helper function for writing output h5 files"""

    z = latents['z'][analyzed_barcode_logic, :]
    d = latents['d'][analyzed_barcode_logic]
    p = latents['p'][analyzed_barcode_logic]
    epsilon = latents['epsilon'][analyzed_barcode_logic]

    # Inferred ambient gene expression vector.
    ambient_expression_trimmed = pyro.param('chi_ambient').detach().cpu().numpy()

    # Convert the indices from trimmed gene set to original gene indices.
    ambient_expression = np.zeros(dataset_obj.data['matrix'].shape[1])
    ambient_expression[dataset_obj.analyzed_gene_inds] = ambient_expression_trimmed

    # Some summary statistics:
    # Fraction of counts in each droplet that were removed.
    raw_count_matrix = dataset_obj.data['matrix'][dataset_obj.analyzed_barcode_inds, :]  # need all genes
    raw_counts_droplet = np.array(raw_count_matrix.sum(axis=1)).squeeze()
    out_counts_droplet = np.array(denoised_count_matrix[dataset_obj.analyzed_barcode_inds, :]
                                  .sum(axis=1)).squeeze()
    background_fraction = ((raw_counts_droplet - out_counts_droplet) /
                           (raw_counts_droplet + 0.001))[analyzed_barcode_logic]

    # Handle the optional rho parameters.
    rho = None
    if (('rho_alpha' in pyro.get_param_store().keys())
            and ('rho_beta' in pyro.get_param_store().keys())):
        rho = np.array([pyro.param('rho_alpha').detach().cpu().numpy().item(),
                        pyro.param('rho_beta').detach().cpu().numpy().item()])

    # Determine metadata fields.
    # Wrap in lists to avoid scanpy loading bug
    # which may already be fixed by https://github.com/scverse/scanpy/pull/2344
    metadata = {'learning_curve': learning_curve,
                'barcodes_analyzed': dataset_obj.data['barcodes'][dataset_obj.analyzed_barcode_inds],
                'barcodes_analyzed_inds': dataset_obj.analyzed_barcode_inds,
                'features_analyzed_inds': dataset_obj.analyzed_gene_inds,
                'fraction_data_used_for_testing': 1. - consts.TRAINING_FRACTION,
                'target_false_positive_rate': fpr}
    for k in ['posterior_regularization', 'posterior_regularization_kwargs',
              'estimator', 'estimator_kwargs']:
        val = locals().get(k)  # give me the input variable with this name
        if val is not None:
            if type(val) != dict:
                if type(val) != list:
                    val = [val]  # wrap in a list, unless it's a dict
            metadata.update({k: val})

    # Write h5.
    write_succeeded = write_matrix_to_cellranger_h5(
        cellranger_version=3,  # always write v3 format output
        output_file=file,
        gene_names=dataset_obj.data['gene_names'],
        gene_ids=dataset_obj.data['gene_ids'],
        feature_types=dataset_obj.data['feature_types'],
        genomes=dataset_obj.data['genomes'],
        barcodes=dataset_obj.data['barcodes'][barcode_inds],
        count_matrix=denoised_count_matrix[barcode_inds, :],
        local_latents={'barcode_indices_for_latents': dataset_obj.analyzed_barcode_inds,
                       'gene_expression_encoding': z,
                       'cell_size': d,
                       'cell_probability': p,
                       'droplet_efficiency': epsilon,
                       'background_fraction': background_fraction},
        global_latents={'ambient_expression': ambient_expression,
                        'empty_droplet_size_lognormal_loc': pyro.param('d_empty_loc').item(),
                        'empty_droplet_size_lognormal_scale': pyro.param('d_empty_scale').item(),
                        'cell_size_lognormal_std': pyro.param('d_cell_scale').item(),
                        'swapping_fraction_dist_params': rho},
        metadata=metadata,
    )
    return write_succeeded


def collect_output_metrics(dataset_obj: SingleCellRNACountsDataset,
                           inferred_count_matrix: sp.csr_matrix,
                           fpr: Union[float, str],
                           cell_logic,
                           loss) -> pd.DataFrame:
    """Create a table with a few output metrics. The idea is for these to
    potentially be used by people creating automated pipelines."""

    # Compute some metrics
    input_count_matrix = dataset_obj.data['matrix'][dataset_obj.analyzed_barcode_inds, :]
    total_raw_counts = dataset_obj.data['matrix'].sum()
    total_output_counts = inferred_count_matrix.sum()
    total_counts_removed = total_raw_counts - total_output_counts
    fraction_counts_removed = total_counts_removed / total_raw_counts
    total_raw_counts_in_nonempty_droplets = input_count_matrix[cell_logic].sum()
    total_counts_removed_from_nonempty_droplets = \
        total_raw_counts_in_nonempty_droplets - inferred_count_matrix.sum()
    fraction_counts_removed_from_nonempty_droplets = \
        total_counts_removed_from_nonempty_droplets / total_raw_counts_in_nonempty_droplets
    average_counts_removed_per_nonempty_droplet = \
        total_counts_removed_from_nonempty_droplets / cell_logic.sum()
    expected_cells = dataset_obj.priors['expected_cells']
    found_cells = cell_logic.sum()
    average_counts_per_cell = inferred_count_matrix.sum() / found_cells
    ratio_of_found_cells_to_expected_cells = \
        None if (expected_cells is None) else (found_cells / expected_cells)
    found_empties = len(dataset_obj.analyzed_barcode_inds) - found_cells
    fraction_of_analyzed_droplets_that_are_nonempty = \
        found_cells / len(dataset_obj.analyzed_barcode_inds)
    if len(loss['train']['elbo']) > 20:
        # compare mean ELBO increase over last 3 steps to the typical end(ish) fluctuations
        convergence_indicator = (np.mean(np.abs([(loss['train']['elbo'][i]
                                                  - loss['train']['elbo'][i - 1])
                                                 for i in range(-3, -1)]))
                                 / np.std(loss['train']['elbo'][-20:]))
    else:
        convergence_indicator = 'not enough training epochs to compute (requires more than 20)'
    if len(loss['train']['elbo']) > 0:
        overall_change_in_train_elbo = loss['train']['elbo'][-1] - loss['train']['elbo'][0]
    else:
        overall_change_in_train_elbo = 0  # zero epoch initialization

    all_metrics_dict = \
        {'total_raw_counts': total_raw_counts,
         'total_output_counts': total_output_counts,
         'total_counts_removed': total_counts_removed,
         'fraction_counts_removed': fraction_counts_removed,
         'total_raw_counts_in_cells':
             total_raw_counts_in_nonempty_droplets,
         'total_counts_removed_from_cells':
             total_counts_removed_from_nonempty_droplets,
         'fraction_counts_removed_from_cells':
             fraction_counts_removed_from_nonempty_droplets,
         'average_counts_removed_per_cell':
             average_counts_removed_per_nonempty_droplet,
         'target_fpr': fpr,
         'expected_cells': expected_cells,
         'found_cells': found_cells,
         'output_average_counts_per_cell': average_counts_per_cell,
         'ratio_of_found_cells_to_expected_cells':
             ratio_of_found_cells_to_expected_cells,
         'found_empties': found_empties,
         'fraction_of_analyzed_droplets_that_are_nonempty':
             fraction_of_analyzed_droplets_that_are_nonempty,
         'convergence_indicator': convergence_indicator,
         'overall_change_in_train_elbo': overall_change_in_train_elbo}

    return pd.DataFrame(data=all_metrics_dict,
                        index=['metric']).transpose()


def write_cell_barcodes_csv(bc_file_name: str, cell_barcodes: np.ndarray):
    """Write the cell barcode CSV file.

    Args:
        bc_file_name: Output CSV file
        cell_barcodes: Array of the cell barcode names

    """

    # Save barcodes determined to contain cells as _cell_barcodes.csv
    try:
        barcode_names = np.array([str(cell_barcodes[i], encoding='UTF-8')
                                  for i in range(cell_barcodes.size)])
    except UnicodeDecodeError:
        # necessary if barcodes are ints
        barcode_names = cell_barcodes
    except TypeError:
        # necessary if barcodes are already decoded
        barcode_names = cell_barcodes
    np.savetxt(bc_file_name, barcode_names, delimiter=',', fmt='%s')
    logger.info(f"Saved cell barcodes in {bc_file_name}")


def get_optimizer(n_batches: int,
                  batch_size: int,
                  epochs: int,
                  learning_rate: float,
                  constant_learning_rate: bool,
                  total_epochs_for_testing_only: Optional[int] = None) \
        -> Union[pyro.optim.PyroOptim, pyro.optim.lr_scheduler.PyroLRScheduler]:
    """Get optimizer or learning rate scheduler (if using one)"""

    # Set up the optimizer.
    optimizer = pyro.optim.clipped_adam.ClippedAdam  # just ClippedAdam does not work
    optimizer_args = {'lr': learning_rate, 'clip_norm': 10.}

    # Set up a learning rate scheduler.
    if total_epochs_for_testing_only is not None:
        total_steps = n_batches * total_epochs_for_testing_only
    else:
        total_steps = n_batches * epochs
    scheduler_args = {'optimizer': optimizer,
                      'max_lr': learning_rate * 10,
                      'total_steps': total_steps,
                      'optim_args': optimizer_args}
    scheduler = pyro.optim.OneCycleLR(scheduler_args)

    # Constant learning rate overrides the above and uses no scheduler.
    if constant_learning_rate:
        logger.info('Using ClippedAdam --constant-learning-rate rather than '
                    'the OneCycleLR schedule. This is not usually recommended.')
        scheduler = ClippedAdam(optimizer_args)

    return scheduler


def run_inference(dataset_obj: SingleCellRNACountsDataset,
                  args: argparse.Namespace,
                  output_checkpoint_tarball: str = consts.CHECKPOINT_FILE_NAME,
                  total_epochs_for_testing_only: Optional[int] = None)\
        -> Tuple[RemoveBackgroundPyroModel, pyro.optim.PyroOptim, DataLoader, DataLoader]:
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
    ckpt = attempt_load_checkpoint(filebase=checkpoint_filename,
                                   tarball_name=args.input_checkpoint_tarball,
                                   force_device='cuda:0' if args.use_cuda else 'cpu',
                                   force_use_checkpoint=args.force_use_checkpoint)
    ckpt_loaded = ckpt['loaded']  # True if a checkpoint was loaded successfully

    if ckpt_loaded:

        model = ckpt['model']
        scheduler = ckpt['optim']
        train_loader = ckpt['train_loader']
        test_loader = ckpt['test_loader']
        if hasattr(ckpt['args'], 'num_failed_attempts'):
            # update this from the checkpoint file, if present
            args.num_failed_attempts = ckpt['args'].num_failed_attempts
        for obj in [model, scheduler, train_loader, test_loader, args]:
            assert obj is not None, \
                f'Expected checkpoint to contain model, scheduler, train_loader, ' \
                f'test_loader, and args; but some are None:\n{ckpt}'
        logger.info('Checkpoint loaded successfully.')

    else:

        logger.info('No checkpoint loaded.')

        # Get the trimmed count matrix (transformed if called for).
        count_matrix = dataset_obj.get_count_matrix()

        # Set up the variational autoencoder:

        # Encoder.
        encoder_z = EncodeZ(input_dim=count_matrix.shape[1],
                            hidden_dims=args.z_hidden_dims,
                            output_dim=args.z_dim,
                            use_batch_norm=False,
                            use_layer_norm=False,
                            input_transform='normalize')

        encoder_other = EncodeNonZLatents(n_genes=count_matrix.shape[1],
                                          z_dim=args.z_dim,
                                          log_count_crossover=dataset_obj.priors['log_counts_crossover'],
                                          prior_log_cell_counts=np.log1p(dataset_obj.priors['cell_counts']),
                                          empty_log_count_threshold=np.log1p(dataset_obj.empty_UMI_threshold),
                                          prior_logit_cell_prob=dataset_obj.priors['cell_logit'],
                                          input_transform='log_normalize')

        encoder = CompositeEncoder({'z': encoder_z,
                                    'other': encoder_other})

        # Decoder.
        decoder = Decoder(input_dim=args.z_dim,
                          hidden_dims=args.z_hidden_dims[::-1],
                          use_batch_norm=True,
                          use_layer_norm=False,
                          output_dim=count_matrix.shape[1])

        # Set up the pyro model for variational inference.
        model = RemoveBackgroundPyroModel(model_type=args.model,
                                          encoder=encoder,
                                          decoder=decoder,
                                          dataset_obj_priors=dataset_obj.priors,
                                          n_analyzed_genes=dataset_obj.analyzed_gene_inds.size,
                                          n_droplets=dataset_obj.analyzed_barcode_inds.size,
                                          analyzed_gene_names=dataset_obj.data['gene_names'][dataset_obj.analyzed_gene_inds],
                                          empty_UMI_threshold=dataset_obj.empty_UMI_threshold,
                                          log_counts_crossover=dataset_obj.priors['log_counts_crossover'],
                                          use_cuda=args.use_cuda)

        # Load the dataset into DataLoaders.
        frac = args.training_fraction  # Fraction of barcodes to use for training
        batch_size = int(min(consts.MAX_BATCH_SIZE, frac * dataset_obj.analyzed_barcode_inds.size / 2))

        # Set up dataloaders.
        train_loader, test_loader = \
            prep_data_for_training(dataset=count_matrix,
                                   empty_drop_dataset=dataset_obj.get_count_matrix_empties(),
                                   batch_size=batch_size,
                                   training_fraction=frac,
                                   fraction_empties=args.fraction_empties,
                                   shuffle=True,
                                   use_cuda=args.use_cuda)

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
            loss_function = JitTrace_ELBO()
        else:
            loss_function = JitTraceEnum_ELBO(max_plate_nesting=1,
                                              strict_enumeration_warning=False)
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
        model.guide(train_loader.__next__())
        train_loader.reset_ptr()

        # Even though it's not much of a checkpoint, we still need one for subsequent steps.
        save_checkpoint(filebase=checkpoint_filename,
                        tarball_name=output_checkpoint_tarball,
                        args=args,
                        model_obj=model,
                        scheduler=svi.optim,
                        train_loader=train_loader,
                        test_loader=test_loader)

    else:
        logger.info("Running inference...")
        try:
            run_training(model=model,
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
                         final_elbo_fail_fraction=args.final_elbo_fail_fraction)

        except ElboException:

            logger.warning(traceback.format_exc())

            # Keep track of number of failed attempts.
            if not hasattr(args, 'num_failed_attempts'):
                args.num_failed_attempts = 1
            else:
                args.num_failed_attempts = args.num_failed_attempts + 1
            logger.debug(f'Training failed, and the number of failed attempts '
                         f'on record is {args.num_failed_attempts}')

            # Retry training with reduced learning rate, if indicated by user.
            logger.debug(f'Number of times to retry training is {args.num_training_tries}')
            if args.num_failed_attempts < args.num_training_tries:
                args.learning_rate = args.learning_rate * args.learning_rate_retry_mult
                logger.info(f'Restarting training: attempt {args.num_failed_attempts + 1}, '
                            f'learning_rate = {args.learning_rate}')
                run_remove_background(args)  # start from scratch
                sys.exit(0)
            else:
                logger.info(f'No more attempts are specified by --num-training-tries. '
                            f'Therefore the workflow will run once more without ELBO restrictions.')
                args.epoch_elbo_fail_fraction = None
                args.final_elbo_fail_fraction = None
                run_remove_background(args)  # start from scratch
                # non-zero exit status in order to draw user's attention to the fact that ELBO tests
                # were never satisfied.
                sys.exit(1)

        logger.info("Inference procedure complete.")

    return model, scheduler, train_loader, test_loader
