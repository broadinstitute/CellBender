"""Helper functions for training."""

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, JitTrace_ELBO, \
    TraceEnum_ELBO, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro.util import ignore_jit_warnings

from cellbender.remove_background.model import RemoveBackgroundPyroModel
from cellbender.remove_background.vae.decoder import Decoder
from cellbender.remove_background.vae.encoder \
    import EncodeZ, CompositeEncoder, EncodeNonZLatents
from cellbender.remove_background.data.dataset import SingleCellRNACountsDataset
from cellbender.remove_background.data.dataprep import \
    prep_sparse_data_for_training as prep_data_for_training
from cellbender.remove_background.data.dataprep import DataLoader
from cellbender.remove_background.exceptions import NanException
import cellbender.remove_background.consts as consts

import numpy as np
import torch

from typing import Tuple, List, Optional
import logging
import time
from datetime import datetime


def train_epoch(svi: SVI,
                train_loader: DataLoader,
                epoch: Optional[int] = None) -> float:
    """Train a single epoch.

    Args:
        svi: The pyro object used for stochastic variational inference.
        train_loader: Dataloader for training set.
        epoch: Epoch used by the learning rate scheduler.  Use None for normal
            schedule.  Can co-opt the schedule by setting a value (cool off).

    Returns:
        total_epoch_loss_train: The loss for this epoch of training, which is
            -ELBO, normalized by the number of items in the training set.

    """

    # Initialize loss accumulator and training set size.
    epoch_loss = 0.
    normalizer_train = 0.

    # Train an epoch by going through each mini-batch.
    for x_cell_batch in train_loader:

        # Perform gradient descent step and accumulate loss.
        epoch_loss += svi.step(x_cell_batch)
        svi.optim.step(epoch=epoch)  # for LR scheduling
        normalizer_train += x_cell_batch.size(0)

    # Return epoch loss.
    total_epoch_loss_train = epoch_loss / normalizer_train

    return total_epoch_loss_train


def evaluate_epoch(svi: pyro.infer.SVI,
                   test_loader: DataLoader) -> float:
    """Evaluate loss on tests set for a single epoch.

        Args:
            svi: The pyro object used for stochastic variational inference.
            test_loader: Dataloader for tests set.

        Returns:
            total_epoch_loss_train: The loss for this epoch of training, which
                is -ELBO, normalized by the number of items in the training set.

    """

    # Initialize loss accumulator and training set size.
    test_loss = 0.
    normalizer_test = 0.

    # Compute the loss over the entire tests set.
    for x_cell_batch in test_loader:

        # Accumulate loss.
        test_loss += svi.evaluate_loss(x_cell_batch)
        normalizer_test += x_cell_batch.size(0)

    # Return epoch loss.
    if normalizer_test > 0:
        total_epoch_loss_test = test_loss / normalizer_test
    else:
        total_epoch_loss_test = np.nan

    return total_epoch_loss_test


@ignore_jit_warnings()
def run_training(model: RemoveBackgroundPyroModel,
                 svi: pyro.infer.SVI,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 epochs: int,
                 test_freq: int = 10,
                 final_elbo_fail_fraction: float = None,
                 epoch_elbo_fail_fraction: float = None) -> Tuple[List[float],
                                                                  List[float], bool]:
    """Run an entire course of training, evaluating on a tests set periodically.

        Args:
            model: The model, here in order to store train and tests loss.
            svi: The pyro object used for stochastic variational inference.
            train_loader: Dataloader for training set.
            test_loader: Dataloader for tests set.
            epochs: Number of epochs to run training.
            test_freq: Test set loss is calculated every test_freq epochs of
                training.
            final_elbo_fail_fraction: fail if final test ELBO >= best ELBO * (1+this value)
            epoch_elbo_fail_fraction: fail if current test ELBO >= previous ELBO * (1+this value)

        Returns:
            Tuple(
             list of training ELBO for each epoch,
             list of test ELBO for each epoch for which testing is done,
             boolean: False indicates training failed
             )
    """

    logging.info("Running inference...")

    # Initialize train and tests ELBO with empty lists.
    train_elbo = []
    test_elbo = []
    succeeded = True
    # Run training loop.  Use try to allow for keyboard interrupt.
    try:
        for epoch in range(1, epochs + 1):

            # Display duration of an epoch (use 2 to avoid initializations).
            if epoch == 2:
                t = time.time()

            total_epoch_loss_train = train_epoch(svi, train_loader)

            train_elbo.append(-total_epoch_loss_train)
            model.loss['train']['epoch'].append(epoch)
            model.loss['train']['elbo'].append(-total_epoch_loss_train)

            if epoch == 2:
                logging.info("[epoch %03d]  average training loss: %.4f  (%.1f seconds per epoch)"
                             % (epoch, total_epoch_loss_train, time.time() - t))
            else:
                logging.info("[epoch %03d]  average training loss: %.4f"
                             % (epoch, total_epoch_loss_train))

            # If there is no test data (training_fraction == 1.), skip test.
            if len(test_loader) == 0:
                continue

            # Every test_freq epochs, evaluate tests loss.
            if epoch % test_freq == 0:
                total_epoch_loss_test = evaluate_epoch(svi, test_loader)
                test_elbo.append(-total_epoch_loss_test)
                model.loss['test']['epoch'].append(epoch)
                model.loss['test']['elbo'].append(-total_epoch_loss_test)
                logging.info("[epoch %03d] average test loss: %.4f"
                             % (epoch, total_epoch_loss_test))
                if epoch_elbo_fail_fraction is not None and len(test_elbo) > 1 and \
                        -test_elbo[-1] >= -test_elbo[-2] * (1 + epoch_elbo_fail_fraction):
                    logging.info(
                        "Training failed because this test loss (%.4f) exceeds previous test loss(%.4f) by >= %.2f%%" %
                        (test_elbo[-1], test_elbo[-2], 100*epoch_elbo_fail_fraction))
                    succeeded = False
                    break

        logging.info("Inference procedure complete.")

        if succeeded and final_elbo_fail_fraction is not None and len(test_elbo) > 1:
            best_test_elbo = max(test_elbo)
            if -test_elbo[-1] >= -best_test_elbo * (1 + final_elbo_fail_fraction):
                logging.info(
                    "Training failed because final test loss (%.4f) exceeds "
                    "best test loss(%.4f) by >= %.2f%%" %
                    (test_elbo[-1], best_test_elbo, 100*final_elbo_fail_fraction))
                succeeded = False

    # Exception allows program to continue after ending inference prematurely.
    except KeyboardInterrupt:

        logging.info("Inference procedure stopped by keyboard interrupt.")

    # Exception allows program to produce output when terminated by a NaN.
    except NanException as nan:

        print(nan.message)
        logging.info(f"Inference procedure terminated early due to a NaN value in: {nan.param}\n\n"
                     f"The suggested fix is to reduce the learning rate.\n\n")
        succeeded = False

    if succeeded:
        logging.info("Training succeeded")
    logging.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return train_elbo, test_elbo, succeeded


def run_inference(dataset_obj: SingleCellRNACountsDataset,
                  args) -> RemoveBackgroundPyroModel:
    """Run a full inference procedure, training a latent variable model.

    Args:
        dataset_obj: Input data in the form of a SingleCellRNACountsDataset
            object.
        args: Input command line parsed arguments.

    Returns:
         model: cellbender.model.RemoveBackgroundPyroModel that has had
            inference run.

    """

    # Get the trimmed count matrix (transformed if called for).
    count_matrix = dataset_obj.get_count_matrix()

    # Configure pyro options (skip validations to improve speed).
    pyro.enable_validation(False)
    pyro.distributions.enable_validation(False)

    # Load the dataset into DataLoaders.
    frac = args.training_fraction  # Fraction of barcodes to use for training
    batch_size = int(min(300, frac * dataset_obj.analyzed_barcode_inds.size / 2))

    for attempt in range(args.num_training_tries):
        # set seed for every attempt for reproducibility
        pyro.set_rng_seed(0)
        pyro.clear_param_store()

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
                                          dataset_obj=dataset_obj,
                                          use_cuda=args.use_cuda)

        train_loader, test_loader = \
            prep_data_for_training(dataset=count_matrix,
                                   empty_drop_dataset=
                                   dataset_obj.get_count_matrix_empties(),
                                   random_state=dataset_obj.random,
                                   batch_size=batch_size,
                                   training_fraction=frac,
                                   fraction_empties=args.fraction_empties,
                                   shuffle=True,
                                   use_cuda=args.use_cuda)

        # Set up the optimizer.
        optimizer = pyro.optim.clipped_adam.ClippedAdam
        optimizer_args = {'lr': args.learning_rate, 'clip_norm': 10.}

        # Set up a learning rate scheduler.
        minibatches_per_epoch = int(np.ceil(len(train_loader) / train_loader.batch_size).item())
        scheduler_args = {'optimizer': optimizer,
                          'max_lr': args.learning_rate * 10,
                          'steps_per_epoch': minibatches_per_epoch,
                          'epochs': args.epochs,
                          'optim_args': optimizer_args}
        scheduler = pyro.optim.OneCycleLR(scheduler_args)

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
        svi = SVI(model.model, model.guide, scheduler,
                  loss=loss_function)

        # Run training.
        train_elbo, test_elbo, succeeded = run_training(model, svi, train_loader, test_loader,
                                                        epochs=args.epochs, test_freq=5,
                                                        final_elbo_fail_fraction=args.final_elbo_fail_fraction,
                                                        epoch_elbo_fail_fraction=args.epoch_elbo_fail_fraction)
        if succeeded or attempt+1 >= args.num_training_tries:
            break
        else:
            args.learning_rate = args.learning_rate * args.learning_rate_retry_mult
            logging.info("Learning failed.  Retrying with learning-rate %.8f" % args.learning_rate)

    return model
