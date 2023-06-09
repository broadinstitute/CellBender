"""Test functionality in train.py"""

import torch
import pyro.distributions as dist
import pyro.infer.trace_elbo
import pytest
import scipy.sparse as sp
from pyro.infer.svi import SVI
import unittest.mock

from cellbender.remove_background.run import get_optimizer
from cellbender.remove_background.data.dataprep import prep_sparse_data_for_training \
    as prep_data_for_training
from cellbender.remove_background.train import train_epoch, evaluate_epoch
from .conftest import USE_CUDA


@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
@pytest.mark.parametrize('dropped_minibatch', [False, True], ids=['', 'dropped_minibatch'])
def test_one_cycle_scheduler(dropped_minibatch, cuda):

    # if there is a minibatch so small that it's below consts.SMALLEST_ALLOWED_BATCH
    # then the minibatch gets skipped. make sure this works with the scheduler.

    pyro.clear_param_store()
    device = 'cuda' if cuda else 'cpu'

    n_cells = 3580
    n_empties = 50000
    n_genes = 100
    learning_rate = 1e-4
    batch_size = 512
    if dropped_minibatch:
        n_cells += 2
    epochs = 50

    count_matrix = sp.random(n_cells, n_genes, density=0.01, format='csr')
    empty_matrix = sp.random(n_empties, n_genes, density=0.0001, format='csr')

    # Set up dataloaders.
    train_loader, _ = \
        prep_data_for_training(dataset=count_matrix,
                               empty_drop_dataset=empty_matrix,
                               batch_size=batch_size,
                               training_fraction=1.,
                               fraction_empties=0.3,
                               shuffle=True,
                               use_cuda=cuda)

    print(f'epochs = {epochs}')
    print(f'len(train_loader), i.e. number of minibatches = {len(train_loader)}')
    print(f'train_loader.batch_size = {train_loader.batch_size}')
    print(f'train_loader.cell_batch_size = {train_loader.cell_batch_size}')

    # Set up optimizer.
    scheduler = get_optimizer(
        n_batches=len(train_loader),
        batch_size=train_loader.batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        constant_learning_rate=False,
        total_epochs_for_testing_only=None,
    )

    # Set up SVI dummy.
    def _model(x):
        y_mean = pyro.param("y_mean", torch.zeros(1).to(device))
        pyro.sample("y_obs", dist.Normal(loc=y_mean, scale=1), obs=x.sum())

    def _guide(x):
        pass

    svi = SVI(model=_model,
              guide=_guide,
              optim=scheduler,
              loss=pyro.infer.trace_elbo.Trace_ELBO())

    # Looking for a RuntimeError raised by the scheduler trying to step too much
    lr = []
    for _ in range(epochs):
        train_epoch(svi=svi, train_loader=train_loader)
        lr.append(list(svi.optim.optim_objs.values())[0].get_last_lr()[0])

    print('learning rates at each epoch:')
    print('\n'.join([f'[{i + 1:03d}]  {rate:.2e}' for i, rate in enumerate(lr)]))

    # Ensure learning rates are numerically correct
    # scheduler args include 'max_lr': learning_rate * 10
    max_lr = 10 * learning_rate
    initial_lr = max_lr / 25
    final_lr = initial_lr / 1e4
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    assert lr[len(lr) // 2] > lr[0], 'Cosine LR schedule should have middle > start'
    assert lr[len(lr) // 2] > lr[-1], 'Cosine LR schedule should have middle > end'
    torch.testing.assert_close(lr[0], initial_lr + (max_lr - initial_lr) / epochs, rtol=1, atol=1e-7), \
        'Starting learning rate for scheduler seems off'
    torch.testing.assert_close(lr[-1], final_lr, rtol=1, atol=1e-7), \
        'Final learning rate for scheduler seems off'
    torch.testing.assert_close(max(lr), max_lr, rtol=1, atol=1e-7), \
        'Max learning rate in scheduler seems off'

    # And one more step ought to raise the error
    with pytest.raises(ValueError,
                       match=r"Tried to step .* times. The specified number "
                             r"of total steps is .*"):
        svi.optim.step()


@pytest.mark.skip
def test_epoch_elbo_fail_restart():
    """Trigger failure and ensure a new attempt is made"""
    pass


@pytest.mark.skip
def test_final_elbo_fail_restart():
    """Trigger failure and ensure a new attempt is made"""
    pass


@pytest.mark.skip
def test_restart_matches_scratch():
    """Ensure that an auto-restart gives same ELBO as a start-from-scratch"""
    # https://stackoverflow.com/questions/50964786/mock-exception-raised-in-function-using-pytest
    pass


@pytest.mark.skip
def test_num_training_attempts():
    """Make sure the number of training attempts matches input arg"""
    pass
