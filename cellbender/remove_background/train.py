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
    import EncodeZ, EncodeD, EncodeNonEmptyDropletLogitProb, CompositeEncoder
from cellbender.remove_background.data.dataset import SingleCellRNACountsDataset
from cellbender.remove_background.data.dataprep import \
    prep_sparse_data_for_training as prep_data_for_training
from cellbender.remove_background.data.dataprep import DataLoader

import numpy as np

from typing import Tuple, List
import logging


def train_epoch(svi: SVI,
                train_loader: DataLoader) -> float:
    """Train a single epoch.

    Args:
        svi: The pyro object used for stochastic variational inference.
        train_loader: Dataloader for training set.

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
                 test_freq: int = 10) -> Tuple[List[float],
                                               List[float]]:
    """Run an entire course of training, evaluating on a tests set periodically.

        Args:
            model: The model, here in order to store train and tests loss.
            svi: The pyro object used for stochastic variational inference.
            train_loader: Dataloader for training set.
            test_loader: Dataloader for tests set.
            epochs: Number of epochs to run training.
            test_freq: Test set loss is calculated every test_freq epochs of
                training.

        Returns:
            total_epoch_loss_train: The loss for this epoch of training, which
                is -ELBO, normalized by the number of items in the training set.

    """

    logging.info("Running inference...")

    # Initialize train and tests ELBO with empty lists.
    train_elbo = []
    test_elbo = []

    # Run training loop.  Use try to allow for keyboard interrupt.
    try:
        for epoch in range(epochs):

            # Train, and keep track of training loss.
            total_epoch_loss_train = train_epoch(svi, train_loader)
            train_elbo.append(-total_epoch_loss_train)
            model.loss['train']['epoch'].append(epoch)
            model.loss['train']['elbo'].append(-total_epoch_loss_train)
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

        logging.info("Inference procedure complete.")

    # Exception allows program to continue after ending inference prematurely.
    except KeyboardInterrupt:

        logging.info("Inference procedure stopped by keyboard interrupt.")

    return train_elbo, test_elbo


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
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    # Set up the variational autoencoder:

    # Encoder.
    encoder_z = EncodeZ(input_dim=count_matrix.shape[1],
                        hidden_dims=args.z_hidden_dims,
                        output_dim=args.z_dim,
                        input_transform='normalize')

    encoder_d = EncodeD(input_dim=count_matrix.shape[1],
                        hidden_dims=args.d_hidden_dims,
                        output_dim=1,
                        log_count_crossover=
                        dataset_obj.priors['log_counts_crossover'])

    if args.model == "simple":

        # If using the simple model, there is no need for p.
        encoder = CompositeEncoder({'z': encoder_z, 'd_loc': encoder_d})

    else:

        # Models that include empty droplets.
        encoder_p = EncodeNonEmptyDropletLogitProb(
            input_dim=count_matrix.shape[1],
            hidden_dims=args.p_hidden_dims,
            output_dim=1,
            input_transform='normalize',
            log_count_crossover=dataset_obj.priors['log_counts_crossover'])
        encoder = CompositeEncoder({'z': encoder_z,
                                    'd_loc': encoder_d,
                                    'p_y': encoder_p})

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

    # Set up the optimizer.
    adam_args = {"lr": args.learning_rate}
    optimizer = ClippedAdam(adam_args)

    # Determine the loss function.
    if args.use_jit:
        loss_function = JitTraceEnum_ELBO(max_plate_nesting=1,
                                          strict_enumeration_warning=False)
    else:
        loss_function = TraceEnum_ELBO(max_plate_nesting=1)

    if args.model == "simple":
        if args.use_jit:
            loss_function = JitTrace_ELBO()
        else:
            loss_function = Trace_ELBO()

    # Set up the inference process.
    svi = SVI(model.model, model.guide, optimizer,
              loss=loss_function)

    # Load the dataset into DataLoaders.
    frac = args.training_fraction  # Fraction of barcodes to use for training
    batch_size = int(min(500,
                         frac * dataset_obj.analyzed_barcode_inds.size / 2))
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

    # Run training.
    run_training(model, svi, train_loader, test_loader,
                 epochs=args.epochs, test_freq=10)

    return model
