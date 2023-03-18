"""Helper functions for training."""

import pyro
from pyro.util import ignore_jit_warnings
from pyro.infer import SVI
import torch

from cellbender.remove_background.model import RemoveBackgroundPyroModel
from cellbender.remove_background.data.dataprep import DataLoader
from cellbender.remove_background.exceptions import NanException, ElboException
import cellbender.remove_background.consts as consts
from cellbender.remove_background.checkpoint import save_checkpoint
from cellbender.monitor import get_hardware_usage

import numpy as np

import argparse
from typing import Tuple, List, Optional
import logging
import time
from datetime import datetime


logger = logging.getLogger('cellbender')


def is_scheduler(optim):
    """Currently is_scheduler is on a pyro dev branch"""
    if hasattr(optim, 'is_scheduler'):
        return optim.is_scheduler()
    else:
        return type(optim) == pyro.optim.lr_scheduler.PyroLRScheduler


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
        normalizer_train += x_cell_batch.size(0)

        # Despite this being the "wrong" place to step the scheduler in general
        # (see https://docs.pyro.ai/en/stable/_modules/pyro/optim/lr_scheduler.html#PyroLRScheduler)
        # it ends up producing the right learning rate schedule for the OneCycleLR scheduler
        # (see https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
        if is_scheduler(svi.optim):
            svi.optim.step(epoch=epoch)  # for LR scheduling

    # Return epoch loss.
    total_epoch_loss_train = epoch_loss / normalizer_train

    if is_scheduler(svi.optim):
        logger.debug(f'Learning rate scheduler: LR = '
                     f'{list(svi.optim.optim_objs.values())[0].get_last_lr()[0]:.2e}')

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
                 args: argparse.Namespace,
                 svi: pyro.infer.SVI,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 epochs: int,
                 output_filename: str,
                 test_freq: int = 10,
                 final_elbo_fail_fraction: float = None,
                 epoch_elbo_fail_fraction: float = None,
                 ckpt_tarball_name: str = consts.CHECKPOINT_FILE_NAME,
                 checkpoint_freq: int = 10) -> Tuple[List[float], List[float]]:
    """Run an entire course of training, evaluating on a tests set periodically.

        Args:
            model: The model, here in order to store train and tests loss.
            args: Parsed arguments, which get saved to checkpoints.
            svi: The pyro object used for stochastic variational inference.
            train_loader: Dataloader for training set.
            test_loader: Dataloader for tests set.
            epochs: Number of epochs to run training.
            output_filename: User-specified output file, used to construct
                checkpoint filenames.
            test_freq: Test set loss is calculated every test_freq epochs of
                training.
            final_elbo_fail_fraction: Fail if final test ELBO >=
                best ELBO * (1 + this value)
            epoch_elbo_fail_fraction: Fail if current test ELBO >=
                previous ELBO * (1 + this value)
            ckpt_tarball_name: Name of saved tarball for checkpoint.
            checkpoint_freq: Checkpoint after this many minutes

        Returns:
            total_epoch_loss_train: The loss for this epoch of training, which
                is -ELBO, normalized by the number of items in the training set.

    """

    # Initialize train and tests ELBO with empty lists.
    train_elbo = []
    test_elbo = []
    epoch_checkpoint_freq = 1000  # a large number... it will be recalculated

    # Run training loop.  Use try to allow for keyboard interrupt.
    try:

        start_epoch = (1 if (model is None) or (len(model.loss['train']['epoch']) == 0)
                       else model.loss['train']['epoch'][-1] + 1)

        for epoch in range(start_epoch, epochs + 1):

            # In debug mode, log hardware load every epoch.
            if args.debug:
                # Don't spend time pinging usage stats if we will not use the log.
                # TODO: use multiprocessing to sample these stats DURING training...
                logger.debug('\n' + get_hardware_usage(use_cuda=model.use_cuda))

            # Display duration of an epoch (use 2 to avoid initializations).
            if epoch == start_epoch + 1:
                t = time.time()

            model.train()
            total_epoch_loss_train = train_epoch(svi, train_loader)

            train_elbo.append(-total_epoch_loss_train)

            if model is not None:
                model.loss['train']['epoch'].append(epoch)
                model.loss['train']['elbo'].append(-total_epoch_loss_train)

            if epoch == start_epoch + 1:
                time_per_epoch = time.time() - t
                logger.info("[epoch %03d]  average training loss: %.4f  (%.1f seconds per epoch)"
                            % (epoch, total_epoch_loss_train, time_per_epoch))
                epoch_checkpoint_freq = int(np.ceil(60 * checkpoint_freq / time_per_epoch))
                if (epoch_checkpoint_freq > 0) and (epoch_checkpoint_freq < epochs):
                    logger.info(f"Will checkpoint every {epoch_checkpoint_freq} epochs")
                elif epoch_checkpoint_freq >= epochs:
                    logger.info(f"Will not checkpoint due to projected run "
                                f"completion in under {checkpoint_freq} min")
            else:
                logger.info("[epoch %03d]  average training loss: %.4f"
                            % (epoch, total_epoch_loss_train))

            # If there is no test data (training_fraction == 1.), skip test.
            if (test_loader is not None) and (len(test_loader) > 0):

                # Every test_freq epochs, evaluate tests loss.
                if epoch % test_freq == 0:
                    model.eval()
                    total_epoch_loss_test = evaluate_epoch(svi, test_loader)
                    test_elbo.append(-total_epoch_loss_test)
                    model.loss['test']['epoch'].append(epoch)
                    model.loss['test']['elbo'].append(-total_epoch_loss_test)
                    logger.info("[epoch %03d] average test loss: %.4f"
                                % (epoch, total_epoch_loss_test))

            # Check on whether ELBO has spiked beyond specified conditions.
            if ((epoch_elbo_fail_fraction is not None)
                    and (len(model.loss['train']['elbo']) > 1)
                    and (-model.loss['train']['elbo'][-1]
                         >= -model.loss['train']['elbo'][-2] * (1 + epoch_elbo_fail_fraction))):
                raise ElboException(f'Training failed because train loss '
                                    f'({-model.loss["train"]["elbo"][-1]:.4f}) '
                                    f'exceeds previous train loss '
                                    f'({-model.loss["train"]["elbo"][-2]:.4f}) by >= '
                                    f'{100 * epoch_elbo_fail_fraction:.1f}%')

            # Checkpoint after final epoch.
            if ((ckpt_tarball_name != 'none')
                and (((checkpoint_freq > 0) and (epoch % epoch_checkpoint_freq == 0))
                     or (epoch == epochs))):  # checkpoint at final epoch
                save_checkpoint(filebase=output_filename,
                                tarball_name=ckpt_tarball_name,
                                args=args,
                                model_obj=model,
                                scheduler=svi.optim,
                                train_loader=train_loader,
                                test_loader=test_loader)

    # Exception allows program to continue after ending inference prematurely.
    except KeyboardInterrupt:
        logger.info("Inference procedure stopped by keyboard interrupt... "
                    "will save a checkpoint.")
        save_checkpoint(filebase=output_filename,
                        tarball_name=ckpt_tarball_name,
                        args=args,
                        model_obj=model,
                        scheduler=svi.optim,
                        train_loader=train_loader,
                        test_loader=test_loader)

    except ElboException as e:
        logger.info(e.message)
        raise e  # re-raise the exception to pass it back to caller function

    # Exception allows program to produce output when terminated by a NaN.
    except NanException as nan:
        print(nan.message)
        logger.info(f"Inference procedure terminated early due to a NaN value in: {nan.param}\n\n"
                    f"The suggested fix is to reduce the learning rate by a factor of two.\n\n")

    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Check final ELBO meets conditions.
    if (final_elbo_fail_fraction is not None) and (len(test_elbo) > 1):
        best_test_elbo = max(test_elbo)
        if -test_elbo[-1] >= -best_test_elbo * (1 + final_elbo_fail_fraction):
            raise ElboException(f'Training failed because final test loss ({-test_elbo[-1]:.4f}) '
                                f'exceeds best test loss ({-best_test_elbo:.4f}) by >= '
                                f'{100 * final_elbo_fail_fraction:.1f}%')

    # Free up all the GPU memory we can once training is complete.
    torch.cuda.empty_cache()

    return train_elbo, test_elbo
