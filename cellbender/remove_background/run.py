"""Single run of remove-background, given input arguments."""
# separated from CLI due to import dependency issues

from cellbender.remove_background.model import RemoveBackgroundPyroModel
from cellbender.remove_background.data.dataset import SingleCellRNACountsDataset
from cellbender.remove_background.vae.decoder import Decoder
from cellbender.remove_background.vae.encoder \
    import EncodeZ, CompositeEncoder, EncodeNonZLatents
from cellbender.remove_background.data.dataprep import \
    prep_sparse_data_for_training as prep_data_for_training
from cellbender.remove_background.checkpoint import attempt_load_checkpoint
import cellbender.remove_background.consts as consts
from cellbender.remove_background.data.dataprep import DataLoader
from cellbender.remove_background.train import run_training
from cellbender.remove_background.exceptions import ElboException

import pyro
from pyro.infer import SVI, JitTraceEnum_ELBO, JitTrace_ELBO, \
    TraceEnum_ELBO, Trace_ELBO
from pyro.optim import ClippedAdam
import numpy as np
import torch

import os
import sys
import logging
from datetime import datetime
from typing import Tuple, Optional


logger = logging.getLogger('cellbender')


def run_remove_background(args):
    """The full script for the command line tool to remove background RNA.

    Args:
        args: Inputs from the command line, already parsed using argparse.

    Note: Returns nothing, but writes output to a file(s) specified from
        command line.

    """

    # Load dataset, run inference, and write the output to a file.

    # Log the start time.
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("Running remove-background")

    # Load data from file and choose barcodes and genes to analyze.
    try:
        dataset_obj = \
            SingleCellRNACountsDataset(input_file=args.input_file,
                                       expected_cell_count=args.expected_cell_count,
                                       total_droplet_barcodes=args.total_droplets,
                                       fraction_empties=args.fraction_empties,
                                       model_name=args.model,
                                       gene_blacklist=args.blacklisted_genes,
                                       exclude_features=args.exclude_features,
                                       low_count_threshold=args.low_count_threshold,
                                       ambient_counts_in_cells_low_limit=args.ambient_counts_in_cells_low_limit,
                                       fpr=args.fpr)

    except OSError:
        logger.error(f"OSError: Unable to open file {args.input_file}.")
        sys.exit(1)

    # Instantiate latent variable model and run full inference procedure.
    if args.model == 'naive':
        inferred_model = None
    else:
        inferred_model, _, _, _ = run_inference(dataset_obj=dataset_obj, args=args)

    # Generate posterior.
    dataset_obj.calculate_posterior(inferred_model=inferred_model,
                                    posterior_batch_size=args.posterior_batch_size)

    # Write outputs to file.
    try:
        dataset_obj.save_to_output_file(output_file=args.output_file,
                                        inferred_model=inferred_model,
                                        save_plots=True,
                                        create_report=True,
                                        truth_file=args.truth_file)

        logger.info("Completed remove-background.")
        logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S\n'))

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


def run_inference(dataset_obj: SingleCellRNACountsDataset,
                  args,
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
                                   force_device='cuda:0' if args.use_cuda else 'cpu')
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
                            input_transform='normalize')

        encoder_other = EncodeNonZLatents(n_genes=count_matrix.shape[1],
                                          z_dim=args.z_dim,
                                          hidden_dims=consts.ENC_HIDDEN_DIMS,
                                          log_count_crossover=dataset_obj.priors['log_counts_crossover'],
                                          prior_log_cell_counts=np.log1p(dataset_obj.priors['cell_counts']),
                                          input_transform='normalize')

        encoder = CompositeEncoder({'z': encoder_z,
                                    'other': encoder_other})

        # Decoder.
        decoder = Decoder(input_dim=args.z_dim,
                          hidden_dims=args.z_hidden_dims[::-1],
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
        batch_size = int(min(300, frac * dataset_obj.analyzed_barcode_inds.size / 2))

        # pretrain_loader = DataLoader(dataset=count_matrix[np.array(count_matrix.sum(axis=1)).squeeze() >
        #                                                   np.exp(dataset_obj.priors['log_counts_crossover']), :],
        #                              empty_drop_dataset=None,
        #                              batch_size=int(min(100, batch_size)),
        #                              fraction_empties=0.,
        #                              shuffle=True,
        #                              use_cuda=args.use_cuda)
        #
        # Set up the optimizer for pre-training.
        adam_args = {'lr': args.learning_rate}
        # pretrain_optimizer = ClippedAdam(adam_args)
        #
        # # Set up the inference process.
        # pretrain_svi = SVI(model.vae_model, model.vae_guide, pretrain_optimizer, loss=Trace_ELBO())
        #
        # # Pre-train z autoencoder.
        # logger.info("Pre-training VAE...")
        # run_training(model=None,  # because I don't want to keep track of the loss # TODO: refactor?
        #              args=args,
        #              svi=pretrain_svi,
        #              train_loader=pretrain_loader,
        #              test_loader=None,
        #              epochs=max(10, int(100000 / dataset_obj.priors['n_cells'])),
        #              output_filename=checkpoint_filename,
        #              checkpoint_freq=0)
        # logger.info("Pre-training of VAE complete.")

        # # Pre-train d and epsilon encoders.
        # logger.info("Pre-training cell and droplet size inference...")
        # # GMM inference using UMI count histogram.  # TODO: to dataset.py
        # # TODO
        # logger.info("Pre-training complete.")

        # Set up the optimizer.
        optimizer = pyro.optim.clipped_adam.ClippedAdam  # just ClippedAdam does not work
        optimizer_args = {'lr': args.learning_rate, 'clip_norm': 10.}

        # Set up dataloaders.
        train_loader, test_loader = \
            prep_data_for_training(dataset=count_matrix,
                                   empty_drop_dataset=dataset_obj.get_count_matrix_empties(),
                                   batch_size=batch_size,
                                   training_fraction=frac,
                                   fraction_empties=args.fraction_empties,
                                   shuffle=True,
                                   use_cuda=args.use_cuda)

        # Set up a learning rate scheduler.
        minibatches_per_epoch = int(np.ceil(len(train_loader) / train_loader.batch_size).item())
        total_epochs = args.epochs if (total_epochs_for_testing_only is None) else total_epochs_for_testing_only
        scheduler_args = {'optimizer': optimizer,
                          'max_lr': args.learning_rate * 10,
                          'steps_per_epoch': minibatches_per_epoch,
                          'epochs': total_epochs,
                          'optim_args': optimizer_args}
        scheduler = pyro.optim.OneCycleLR(scheduler_args)
        if args.constant_learning_rate:
            logger.info('Using ClippedAdam --constant-learning-rate rather than '
                        'the OneCycleLR schedule. This is not usually recommended.')
            scheduler = ClippedAdam(adam_args)

    # Determine the loss function.
    if args.use_jit:

        # Call guide() once as a warm-up.
        model.guide(torch.zeros([10, dataset_obj.analyzed_gene_inds.size]).to(model.device))

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
            # TODO: ensure this works: trigger a new run with half the learning rate

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
                            f'Therefore the workflow will abort here.')
                sys.exit(1)

        logger.info("Inference procedure complete.")

    return model, scheduler, train_loader, test_loader
