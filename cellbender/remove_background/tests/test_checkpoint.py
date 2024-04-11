"""Tests for checkpointing functions."""

import pytest
import random
import numpy as np
import torch
from torch.distributions import constraints
import pyro
import pyro.optim as optim

import cellbender
from cellbender.remove_background.checkpoint import make_tarball, unpack_tarball, \
    create_workflow_hashcode, save_checkpoint, load_checkpoint, \
    save_random_state, load_random_state
from cellbender.remove_background.vae.encoder import EncodeZ, EncodeNonZLatents, \
    CompositeEncoder
from cellbender.remove_background.vae.decoder import Decoder
from cellbender.remove_background.data.extras.simulate import \
    generate_sample_dirichlet_dataset, get_dataset_dict_as_anndata
from cellbender.remove_background.data.dataprep import prep_sparse_data_for_training
from cellbender.remove_background.model import RemoveBackgroundPyroModel
from cellbender.remove_background.data.dataset import SingleCellRNACountsDataset
from cellbender.remove_background.run import run_inference
from .conftest import USE_CUDA

import os
import argparse
import shutil
import subprocess
from typing import List


class RandomState:
    def __init__(self, use_cuda=False):
        self.python = random.randint(0, 100000)
        self.numpy = np.random.randint(0, 100000, size=1).item()
        self.torch = torch.randint(low=0, high=100000, size=[1], device='cpu').item()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda = torch.randint(low=0, high=100000, size=[1], device='cuda').item()

    def __repr__(self):
        if self.use_cuda:
            return f'python {self.python}; numpy {self.numpy}; torch {self.torch}; torch_cuda {self.cuda}'
        else:
            return f'python {self.python}; numpy {self.numpy}; torch {self.torch}'


def test_create_workflow_hashcode():
    """Ensure workflow hashcodes are behaving as expected"""

    tmp_args1 = argparse.Namespace(epochs=100, expected_cells=1000, use_cuda=True)
    tmp_args2 = argparse.Namespace(epochs=200, expected_cells=1000, use_cuda=True)
    tmp_args3 = argparse.Namespace(epochs=100, expected_cells=500, use_cuda=True)
    hashcode1 = create_workflow_hashcode(module_path=os.path.dirname(cellbender.__file__),
                                         args=tmp_args1)
    hashcode2 = create_workflow_hashcode(module_path=os.path.dirname(cellbender.__file__),
                                         args=tmp_args2)
    hashcode3 = create_workflow_hashcode(module_path=os.path.dirname(cellbender.__file__),
                                         args=tmp_args3)

    # make sure it changes for different arguments
    assert hashcode1 != hashcode3

    # but that the "epochs" argument has no effect
    assert hashcode1 == hashcode2


def create_random_state_blank_slate(seed, use_cuda=USE_CUDA):
    """Establish a base random state
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.util.set_rng_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def perturb_random_state(n, use_cuda=USE_CUDA):
    """Perturb the base random state by drawing random numbers"""
    for _ in range(n):
        random.randint(0, 10)
        np.random.randint(0, 10, 1)
        torch.randn((1,), device='cpu')
        if use_cuda:
            torch.randn((1,), device='cuda')


@pytest.fixture(scope='function', params=[0, 1, 1234], ids=lambda x: f'seed{x}')
def random_state_blank_slate(request):
    create_random_state_blank_slate(request.param)
    return request.param


@pytest.fixture(scope='function', params=[0, 1, 10])
def perturbed_random_state_dict(request, random_state_blank_slate):
    perturb_random_state(request.param)
    return {'state': RandomState(),
            'seed': random_state_blank_slate,
            'n': request.param}


@pytest.fixture(scope='function', params=[0], ids=lambda x: f'setseed{x}')
def random_state_blank_slate0(request):
    create_random_state_blank_slate(request.param)
    return request.param


@pytest.fixture(scope='function', params=[10], ids=lambda x: f'set{x}')
def perturbed_random_state0_dict(request, random_state_blank_slate0):
    perturb_random_state(request.param)
    return {'state': RandomState(),
            'seed': random_state_blank_slate0,
            'n': request.param}


def test_perturbedrandomstate_fixture_meets_expectations(perturbed_random_state0_dict,
                                                         perturbed_random_state_dict):
    """Test the setup of these randomstate fixtures.

    The state0 fixture is one set of params.
    The state fixture is a combinatorial set of params, only one of which matches
        the state0 setup.

    We want to make sure that when we have fixtures set up the same way, then
        randomness behaves the same (and different when set up differently).
    """
    prs = perturbed_random_state_dict['state']
    params = (perturbed_random_state_dict['seed'], perturbed_random_state_dict['n'])
    prs0 = perturbed_random_state0_dict['state']
    params0 = (perturbed_random_state0_dict['seed'], perturbed_random_state0_dict['n'])

    if params == params0:
        # this is the only case in which we expect the two random states to be equal
        assert str(prs0) == str(prs)
    else:
        assert str(prs0) != str(prs)


def test_that_randomstate_plus_perturb_gives_perturbedrandomstate(
        perturbed_random_state0_dict):
    """Make sure perturbation is working as intended"""

    # recreate and make sure we end up in the same place
    # the "recipe" is in perturbed_random_state0_tuple[1]
    create_random_state_blank_slate(perturbed_random_state0_dict['seed'])
    perturb_random_state(perturbed_random_state0_dict['n'])
    this_prs0 = RandomState()

    # check equality
    prs0 = perturbed_random_state0_dict['state']
    assert str(prs0) == str(this_prs0)


@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
def test_save_and_load_random_state(tmpdir_factory, perturbed_random_state_dict, cuda):
    """Test whether random states are being preserved correctly.
    perturbed_random_state_dict is important since it initializes the state."""

    # save the random states
    filebase = tmpdir_factory.mktemp('random_states').join('tmp_checkpoint')
    save_random_state(filebase=filebase)

    # see "what would have happened had we continued"
    counterfactual = RandomState(use_cuda=cuda)
    incorrect = RandomState(use_cuda=cuda)  # a second draw

    # load the random states and check random number generators
    load_random_state(filebase=filebase)
    actual = RandomState(use_cuda=cuda)

    # check equality
    assert str(counterfactual) == str(actual)
    assert str(incorrect) != str(actual)


def new_train_loader(data: torch.Tensor, batch_size: int, shuffle: bool = True):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)


class PyroModel(torch.nn.Module):

    def __init__(self, dim=8, hidden_layer=4, z_dim=2):
        super(PyroModel, self).__init__()
        create_random_state_blank_slate(0)
        pyro.clear_param_store()
        self.encoder = EncodeZ(input_dim=dim, hidden_dims=[hidden_layer], output_dim=z_dim)
        self.decoder = Decoder(input_dim=z_dim, hidden_dims=[hidden_layer], output_dim=dim)
        self.z_dim = z_dim
        self.use_cuda = torch.cuda.is_available()
        self.normal = pyro.distributions.Normal
        self.loss = []
        # self.to(device='cuda' if self.use_cuda else 'cpu')  # CUDA not tested

    def model(self, x: torch.FloatTensor):
        pyro.module("decoder", self.decoder, update_module_params=True)
        with pyro.plate('plate', size=x.shape[0]):
            z = pyro.sample('z', self.normal(loc=0., scale=1.).expand_by([x.shape[0], self.z_dim]).to_event(1))
            x_rec = self.decoder(z)
            pyro.sample('obs', self.normal(loc=x_rec, scale=0.1).to_event(1), obs=x)

    def guide(self, x: torch.FloatTensor):
        pyro.module("encoder", self.encoder, update_module_params=True)
        with pyro.plate('plate', size=x.shape[0]):
            enc = self.encoder(x)
            pyro.sample('z', self.normal(loc=enc['loc'], scale=enc['scale']).to_event(1))


def train_pyro(n_epochs: int,
               data_loader: torch.utils.data.DataLoader,
               svi: pyro.infer.SVI):
    """Run training"""
    loss_per_epoch = []
    for _ in range(n_epochs):
        loss = 0
        norm = 0
        for data in data_loader:
            loss += svi.step(data)
            norm += data.size(0)
        loss_per_epoch.append(loss / norm)
    return loss_per_epoch


def _check_all_close(tensors1, tensors2) -> bool:
    """For two lists of tensors, check that they are all close"""
    assert len(tensors1) == len(tensors2), \
        'Must pass in same number of tensors to check if they are equal'
    equal = True
    for t1, t2 in zip(tensors1, tensors2):
        equal = equal and torch.allclose(t1, t2)
    return equal


def _get_params(module: torch.nn.Module) -> List[torch.Tensor]:
    return [p.data.clone() for p in module.parameters()]


@pytest.mark.parametrize('batch_size_n', [32, 128], ids=lambda n: f'batch{n}')
def test_save_and_load_pyro_checkpoint(tmpdir_factory, batch_size_n):
    """Check and see if restarting from a checkpoint picks up in the same place
    we left off.  Use a dataloader.
    """

    filedir = tmpdir_factory.mktemp('ckpt')
    filebase = filedir.join('ckpt')
    dim = 8
    epochs = 3
    epochs2 = 3
    lr = 1e-2

    # data and dataloader
    dataset = torch.randn((128, dim))
    train_loader = new_train_loader(data=dataset, batch_size=batch_size_n)

    # create an ML model
    initial_model = PyroModel(dim=dim)

    # set up the inference process
    scheduler = optim.ClippedAdam({'lr': lr, 'clip_norm': 10.})
    svi = pyro.infer.SVI(initial_model.model, initial_model.guide, scheduler, loss=pyro.infer.Trace_ELBO())
    w1 = _get_params(initial_model.encoder)

    print('initial weight matrix =========================')
    print('\n'.join([str(t) for t in w1]))

    # train in two parts: part 1
    initial_model.loss.extend(train_pyro(n_epochs=epochs, data_loader=train_loader, svi=svi))

    print('no_ckpt trained round 1 (saved) ===============')
    # print(initial_model.encoder.linears[0].weight.data)
    print('\n'.join([str(t) for t in _get_params(initial_model.encoder)]))

    # save
    save_successful = save_checkpoint(filebase=str(filebase),
                                      args=argparse.Namespace(),
                                      model_obj=initial_model,  # TODO
                                      scheduler=scheduler,
                                      train_loader=train_loader,
                                      tarball_name=str(filebase) + '.tar.gz')
    assert save_successful, 'Failed to save checkpoint during test_save_and_load_checkpoint'

    # load from checkpoint
    create_random_state_blank_slate(0)
    pyro.clear_param_store()
    ckpt = load_checkpoint(filebase=str(filebase),
                           tarball_name=str(filebase) + '.tar.gz',
                           force_device='cpu')
    model_ckpt = ckpt['model']
    scheduler_ckpt = ckpt['optim']
    train_loader = ckpt['train_loader']
    s = ckpt['loaded']
    print('model_ckpt loaded =============================')
    print('\n'.join([str(t) for t in _get_params(model_ckpt.encoder)]))

    matches = _check_all_close(_get_params(model_ckpt.encoder),
                               _get_params(initial_model.encoder))
    print(f'model_ckpt loaded matches data from (saved) trained round 1: {matches}')

    # clean up before most assertions... hokey now due to lack of fixture usage here
    shutil.rmtree(str(filedir))

    # and continue training
    assert s is True, 'Checkpoint loading failed during test_save_and_load_checkpoint'
    svi_ckpt = pyro.infer.SVI(model_ckpt.model, model_ckpt.guide, scheduler_ckpt, loss=pyro.infer.Trace_ELBO())
    rng_ckpt = RandomState()
    guide_trace_ckpt = pyro.poutine.trace(svi_ckpt.guide).get_trace(x=dataset)
    model_ckpt.loss.extend(train_pyro(n_epochs=epochs2, data_loader=train_loader, svi=svi_ckpt))

    print('model_ckpt after round 2 ======================')
    print('\n'.join([str(t) for t in _get_params(model_ckpt.encoder)]))

    # one-shot training straight through
    model_one_shot = PyroModel(dim=dim)  # resets random state
    scheduler = optim.ClippedAdam({'lr': lr, 'clip_norm': 10.})
    train_loader = new_train_loader(data=dataset, batch_size=batch_size_n)
    svi_one_shot = pyro.infer.SVI(model_one_shot.model, model_one_shot.guide, scheduler, loss=pyro.infer.Trace_ELBO())
    model_one_shot.loss.extend(train_pyro(n_epochs=epochs, data_loader=train_loader, svi=svi_one_shot))
    rng_one_shot = RandomState()
    guide_trace_one_shot = pyro.poutine.trace(svi_one_shot.guide).get_trace(x=dataset)
    model_one_shot.loss.extend(train_pyro(n_epochs=epochs2, data_loader=train_loader, svi=svi_one_shot))

    assert str(rng_one_shot) == str(rng_ckpt), \
        'Random states of the checkpointed and non-checkpointed versions at ' \
        'the start of training round 2 do not match.'

    print('model_one_shot ================================')
    print('\n'.join([str(t) for t in _get_params(model_one_shot.encoder)]))

    print(model_one_shot.loss)
    print(model_ckpt.loss)
    print([l1 == l2 for l1, l2 in zip(model_one_shot.loss, model_ckpt.loss)])

    # training should be doing something to the initial weight matrix
    assert (w1[0] != _get_params(model_one_shot.encoder)[0]).sum().item() > 0, \
        'Training is not changing the weight matrix in test_save_and_load_checkpoint'
    assert (w1[0] != _get_params(model_ckpt.encoder)[0]).sum().item() > 0, \
        'Training is not changing the checkpointed weight matrix in test_save_and_load_checkpoint'

    # see if we end up where we should
    assert _check_all_close(_get_params(model_one_shot.encoder),
                            _get_params(model_ckpt.encoder)), \
        'Checkpointed restart does not agree with one-shot training'

    # check guide traces
    print('checking guide trace nodes for agreement:')
    for name in guide_trace_one_shot.nodes:
        if (name != '_RETURN') and ('value' in guide_trace_one_shot.nodes[name].keys()):
            print(name)
            disagreement = (guide_trace_one_shot.nodes[name]['value'].data
                            != guide_trace_ckpt.nodes[name]['value'].data).sum()
            print(f'number of values that disagree: {disagreement}')
            assert disagreement == 0, \
                'Guide traces disagree with and without checkpoint restart'


@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
@pytest.mark.parametrize('scheduler',
                         [False, True],
                         ids=lambda b: 'OneCycleLR' if b else 'Adam')
def test_save_and_load_cellbender_checkpoint(tmpdir_factory, cuda, scheduler):
    """Check and see if restarting from a checkpoint picks up in the same place
    we left off.  Use our model and dataloader.
    """

    filedir = tmpdir_factory.mktemp('ckpt')

    epochs = 5
    epochs2 = 5

    # data
    n_genes = 2000
    dataset = generate_sample_dirichlet_dataset(n_genes=n_genes, cells_of_each_type=[100],
                                                n_droplets=2000, model_type='ambient',
                                                cell_mean_umi=[5000])
    adata = get_dataset_dict_as_anndata(dataset)
    adata_file = os.path.join(filedir, 'sim.h5ad')
    adata.write(adata_file)
    dataset_obj = \
        SingleCellRNACountsDataset(input_file=adata_file,
                                   expected_cell_count=100,
                                   total_droplet_barcodes=1000,
                                   fraction_empties=0.1,
                                   model_name='ambient',
                                   gene_blacklist=[],
                                   exclude_features=[],
                                   low_count_threshold=15,
                                   fpr=[0.01])

    # set up the inference process
    args = argparse.Namespace()
    args.output_file = os.path.join(filedir, 'out.h5')
    args.z_dim = 10
    args.z_hidden_dims = [50]
    args.model = 'ambient'
    args.use_cuda = cuda
    args.use_jit = False
    args.learning_rate = 1e-3
    args.training_fraction = 0.9
    args.fraction_empties = 0.1
    args.checkpoint_filename = 'test01234_'
    args.checkpoint_min = 5
    args.epoch_elbo_fail_fraction = None
    args.final_elbo_fail_fraction = None
    args.constant_learning_rate = not scheduler
    args.debug = False
    args.input_checkpoint_tarball = 'none'
    args.force_use_checkpoint = False

    create_random_state_blank_slate(0)
    pyro.clear_param_store()
    args.epochs = -1  # I am hacking my way around an error induced by saving a checkpoint for 0 epoch runs
    initial_model, scheduler, _, _ = run_inference(dataset_obj=dataset_obj,
                                                   args=args)
    w1 = _get_params(initial_model.encoder['z'])
    print('encoder structure ============')
    print(initial_model.encoder['z'])

    print('initial weight matrix =========================')
    print('\n'.join([str(t) for t in w1]))

    # train in two parts: part 1
    pyro.clear_param_store()
    create_random_state_blank_slate(0)
    args.epochs = epochs
    initial_model, scheduler, train_loader, test_loader = \
        run_inference(dataset_obj=dataset_obj, args=args,
                      output_checkpoint_tarball='none',
                      total_epochs_for_testing_only=epochs + epochs2)

    print('no_ckpt trained round 1 (saved) ===============')
    print('\n'.join([str(t) for t in _get_params(initial_model.encoder['z'])]))
    # print(scheduler.get_state().keys())
    # print(list(scheduler.get_state().values())[0].keys())
    # print(list(list(scheduler.optim_objs.values())[0].optimizer.state_dict().values())[0].values())
    # assert 0

    # save
    hashcode = create_workflow_hashcode(module_path=os.path.dirname(cellbender.__file__),
                                        args=args)[:10]
    checkpoint_filename = os.path.basename(args.output_file).split('.')[0] + '_' + hashcode
    args.checkpoint_filename = checkpoint_filename
    filebase = filedir.join(checkpoint_filename)
    save_successful = save_checkpoint(filebase=str(filebase),
                                      args=args,
                                      model_obj=initial_model,  # TODO
                                      scheduler=scheduler,
                                      tarball_name=str(filebase) + '.tar.gz',
                                      train_loader=train_loader,
                                      test_loader=test_loader)
    assert save_successful, 'Failed to save checkpoint during test_save_and_load_checkpoint'
    assert os.path.exists(str(filebase) + '.tar.gz'), 'Checkpoint should exist but does not'

    # load from checkpoint (automatically) and run
    pyro.clear_param_store()
    create_random_state_blank_slate(0)
    args.epochs = -1
    args.input_checkpoint_tarball = str(filebase) + '.tar.gz'
    model_ckpt, scheduler, _, _ = run_inference(dataset_obj=dataset_obj, args=args,
                                                output_checkpoint_tarball='none')

    print('model_ckpt loaded =============================')
    print('\n'.join([str(t) for t in _get_params(model_ckpt.encoder['z'])]))

    matches = _check_all_close(_get_params(model_ckpt.encoder['z']),
                               _get_params(initial_model.encoder['z']))
    print(f'model_ckpt loaded matches data from (saved) trained round 1: {matches}')

    # and continue training
    create_random_state_blank_slate(0)
    pyro.clear_param_store()
    args.epochs = epochs + epochs2
    model_ckpt, scheduler, _, _ = run_inference(dataset_obj=dataset_obj, args=args,
                                                output_checkpoint_tarball='none')

    print('model_ckpt after round 2 ======================')
    print('\n'.join([str(t) for t in _get_params(model_ckpt.encoder['z'])]))

    # clean up the temp directory to remove checkpoint before running the one-shot
    shutil.rmtree(str(filedir))

    # one-shot training straight through
    pyro.clear_param_store()
    create_random_state_blank_slate(0)
    args.epochs = epochs + epochs2
    args.input_checkpoint_tarball = 'none'
    model_one_shot, scheduler, _, _ = run_inference(dataset_obj=dataset_obj, args=args,
                                                    output_checkpoint_tarball='none')

    print('model_one_shot ================================')
    print('\n'.join([str(t) for t in _get_params(model_one_shot.encoder['z'])]))

    print('loss for model_one_shot:')
    print(model_one_shot.loss)
    print('loss for model_ckpt:')
    print(model_ckpt.loss)
    print([l1 == l2 for l1, l2 in zip(model_one_shot.loss, model_ckpt.loss)])

    # training should be doing something to the initial weight matrix
    assert (w1[0] != _get_params(model_one_shot.encoder['z'])[0]).sum().item() > 0, \
        'Training is not changing the weight matrix in test_save_and_load_checkpoint'
    assert (w1[0] != _get_params(model_ckpt.encoder['z'])[0]).sum().item() > 0, \
        'Training is not changing the checkpointed weight matrix in test_save_and_load_checkpoint'

    # see if we end up where we should
    assert _check_all_close(_get_params(model_one_shot.encoder['z']),
                            _get_params(model_ckpt.encoder['z'])), \
        'Checkpointed restart does not agree with one-shot training'

    # # check guide traces
    # print('checking guide trace nodes for agreement:')
    # for name in guide_trace_one_shot.nodes:
    #     if (name != '_RETURN') and ('value' in guide_trace_one_shot.nodes[name].keys()):
    #         print(name)
    #         disagreement = (guide_trace_one_shot.nodes[name]['value'].data
    #                         != guide_trace_ckpt.nodes[name]['value'].data).sum()
    #         print(f'number of values that disagree: {disagreement}')
    #         assert disagreement == 0, \
    #             'Guide traces disagree with and without checkpoint restart'


@pytest.mark.parametrize(
    "Optim, config",
    [
        (optim.ClippedAdam, {"lr": 0.01}),
        (optim.ExponentialLR, {"optimizer": torch.optim.SGD,
                               "optim_args": {"lr": 0.01},
                               "gamma": 0.9}),
    ],
)
def test_optimizer_checkpoint_restart(Optim, config, tmpdir_factory):
    """This code has been copied from a version of a pyro test by Fritz Obermeyer"""

    tempdir = tmpdir_factory.mktemp('ckpt')

    def model():
        x_scale = pyro.param("x_scale", torch.tensor(1.0),
                             constraint=constraints.positive)
        z = pyro.sample("z", pyro.distributions.Normal(0, 1))
        return pyro.sample("x", pyro.distributions.Normal(z, x_scale), obs=torch.tensor(0.1))

    def guide():
        z_loc = pyro.param("z_loc", torch.tensor(0.0))
        z_scale = pyro.param(
         "z_scale", torch.tensor(0.5), constraint=constraints.positive
        )
        pyro.sample("z", pyro.distributions.Normal(z_loc, z_scale))

    store = pyro.get_param_store()

    def get_snapshot(optimizer):
        s = {k: v.data.clone() for k, v in store.items()}
        if type(optimizer) == optim.lr_scheduler.PyroLRScheduler:
            lr = list(optimizer.optim_objs.values())[0].get_last_lr()[0]
        else:
            lr = list(optimizer.optim_objs.values())[0].param_groups[0]['lr']
        s.update({'lr': f'{lr:.4f}'})
        return s

    # Try without a checkpoint.
    expected = []
    store.clear()
    pyro.set_rng_seed(20210811)
    optimizer = Optim(config.copy())
    svi = pyro.infer.SVI(model, guide, optimizer, pyro.infer.Trace_ELBO())
    for _ in range(5 + 10):
        svi.step()
        expected.append(get_snapshot(optimizer))
        if type(optimizer) == optim.lr_scheduler.PyroLRScheduler:
            svi.optim.step()
    del svi, optimizer

    # Try with a checkpoint.
    actual = []
    store.clear()
    pyro.set_rng_seed(20210811)
    optimizer = Optim(config.copy())
    svi = pyro.infer.SVI(model, guide, optimizer, pyro.infer.Trace_ELBO())
    for _ in range(5):
        svi.step()
        actual.append(get_snapshot(optimizer))
        if type(optimizer) == optim.lr_scheduler.PyroLRScheduler:
            svi.optim.step()

    # checkpoint
    optim_filename = os.path.join(tempdir, "optimizer_state.pt")
    param_filename = os.path.join(tempdir, "param_store.pt")
    optimizer.save(optim_filename)
    store.save(param_filename)
    del optimizer, svi
    store.clear()

    # load from checkpoint
    store.load(param_filename)
    optimizer = Optim(config.copy())
    optimizer.load(optim_filename)
    svi = pyro.infer.SVI(model, guide, optimizer, pyro.infer.Trace_ELBO())
    for _ in range(10):
        svi.step()
        actual.append(get_snapshot(optimizer))
        if type(optimizer) == optim.lr_scheduler.PyroLRScheduler:
            svi.optim.step()

    # display learning rates and actual/expected values for z_loc
    print(repr(optimizer))
    print('epoch\t\tlr\t\tactual\t\t\t\texpected')
    print('-' * 100)
    for i, (ac, ex) in enumerate(zip(actual, expected)):
        print('\t\t'.join([f'{x}' for x in [i, ac['lr'], ac['z_loc'], ex['z_loc']]]))
        if i == 4:
            print('-' * 100)

    # ensure actual matches expected
    for actual_t, expected_t in zip(actual, expected):
        actual_t.pop('lr')
        expected_t.pop('lr')
        for actual_value, expected_value in zip(actual_t.values(), expected_t.values()):
            assert torch.allclose(actual_value, expected_value)
