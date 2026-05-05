import argparse
import os
import random
import shutil
import tempfile
from typing import List

import numpy as np
import pyro
import pyro.optim as optim
import pytest
import scipy.sparse as sp
import torch
from conftest import USE_CUDA
from torch.distributions import constraints

import cellbender
from cellbender.remove_background.checkpoint import (
    create_workflow_hashcode,
    flush_pending_checkpoint,
    load_checkpoint,
    load_optim_from_bytes,
    load_param_store,
    load_random_state,
    save_checkpoint,
    save_checkpoint_async,
    save_random_state,
)
from cellbender.remove_background.data.dataprep import DataLoader
from cellbender.remove_background.data.dataset import SingleCellRNACountsDataset
from cellbender.remove_background.data.extras.simulate import (
    generate_sample_dirichlet_dataset,
    get_dataset_dict_as_anndata,
)
from cellbender.remove_background.run import run_inference
from cellbender.remove_background.vae.decoder import Decoder
from cellbender.remove_background.vae.encoder import EncodeZ


class RandomState:
    def __init__(self, use_cuda=False):
        self.python = random.randint(0, 100000)
        self.numpy = np.random.randint(0, 100000, size=1).item()
        self.torch = torch.randint(low=0, high=100000, size=[1], device="cpu").item()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda = torch.randint(low=0, high=100000, size=[1], device="cuda").item()

    def __repr__(self):
        if self.use_cuda:
            return f"python {self.python}; numpy {self.numpy}; torch {self.torch}; torch_cuda {self.cuda}"
        else:
            return f"python {self.python}; numpy {self.numpy}; torch {self.torch}"


def test_create_workflow_hashcode():
    """Ensure workflow hashcodes are behaving as expected"""

    tmp_args1 = argparse.Namespace(epochs=100, expected_cells=1000, use_cuda=True)
    tmp_args2 = argparse.Namespace(epochs=200, expected_cells=1000, use_cuda=True)
    tmp_args3 = argparse.Namespace(epochs=100, expected_cells=500, use_cuda=True)
    hashcode1 = create_workflow_hashcode(module_path=os.path.dirname(cellbender.__file__), args=tmp_args1)
    hashcode2 = create_workflow_hashcode(module_path=os.path.dirname(cellbender.__file__), args=tmp_args2)
    hashcode3 = create_workflow_hashcode(module_path=os.path.dirname(cellbender.__file__), args=tmp_args3)

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
        torch.randn((1,), device="cpu")
        if use_cuda:
            torch.randn((1,), device="cuda")


@pytest.fixture(scope="function", params=[0, 1, 1234], ids=lambda x: f"seed{x}")
def random_state_blank_slate(request):
    create_random_state_blank_slate(request.param)
    return request.param


@pytest.fixture(scope="function", params=[0, 1, 10])
def perturbed_random_state_dict(request, random_state_blank_slate):
    perturb_random_state(request.param)
    return {"state": RandomState(), "seed": random_state_blank_slate, "n": request.param}


@pytest.fixture(scope="function", params=[0], ids=lambda x: f"setseed{x}")
def random_state_blank_slate0(request):
    create_random_state_blank_slate(request.param)
    return request.param


@pytest.fixture(scope="function", params=[10], ids=lambda x: f"set{x}")
def perturbed_random_state0_dict(request, random_state_blank_slate0):
    perturb_random_state(request.param)
    return {"state": RandomState(), "seed": random_state_blank_slate0, "n": request.param}


def test_perturbedrandomstate_fixture_meets_expectations(perturbed_random_state0_dict, perturbed_random_state_dict):
    """Test the setup of these randomstate fixtures.

    The state0 fixture is one set of params.
    The state fixture is a combinatorial set of params, only one of which matches
        the state0 setup.

    We want to make sure that when we have fixtures set up the same way, then
        randomness behaves the same (and different when set up differently).
    """
    prs = perturbed_random_state_dict["state"]
    params = (perturbed_random_state_dict["seed"], perturbed_random_state_dict["n"])
    prs0 = perturbed_random_state0_dict["state"]
    params0 = (perturbed_random_state0_dict["seed"], perturbed_random_state0_dict["n"])

    if params == params0:
        # this is the only case in which we expect the two random states to be equal
        assert str(prs0) == str(prs)
    else:
        assert str(prs0) != str(prs)


def test_that_randomstate_plus_perturb_gives_perturbedrandomstate(perturbed_random_state0_dict):
    """Make sure perturbation is working as intended"""

    # recreate and make sure we end up in the same place
    # the "recipe" is in perturbed_random_state0_tuple[1]
    create_random_state_blank_slate(perturbed_random_state0_dict["seed"])
    perturb_random_state(perturbed_random_state0_dict["n"])
    this_prs0 = RandomState()

    # check equality
    prs0 = perturbed_random_state0_dict["state"]
    assert str(prs0) == str(this_prs0)


@pytest.mark.parametrize(
    "cuda",
    [False, pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA, reason="requires CUDA"))],
    ids=lambda b: "cuda" if b else "cpu",
)
def test_save_and_load_random_state(tmpdir_factory, perturbed_random_state_dict, cuda):
    """Test whether random states are being preserved correctly.
    perturbed_random_state_dict is important since it initializes the state."""

    # save the random states
    filebase = tmpdir_factory.mktemp("random_states").join("tmp_checkpoint")
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


def new_train_loader(data: sp.csr_matrix, batch_size: int, shuffle: bool = True):
    """Wrap a sparse matrix in our DataLoader for checkpointing tests."""
    n = data.shape[0]
    return DataLoader(
        dataset=data,
        empty_drop_dataset=None,
        batch_size=batch_size,
        fraction_empties=0.0,
        shuffle=shuffle,
        use_cuda=False,
        original_cell_indices=np.arange(n),
        original_empty_indices=np.array([], dtype=np.int64),
    )


class PyroModel(torch.nn.Module):
    def __init__(self, dim=8, hidden_layer=4, z_dim=2, clear_state=True):
        super(PyroModel, self).__init__()
        if clear_state:
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
        with pyro.plate("plate", size=x.shape[0]):
            z = pyro.sample("z", self.normal(loc=0.0, scale=1.0).expand_by([x.shape[0], self.z_dim]).to_event(1))
            x_rec = self.decoder(z)
            pyro.sample("obs", self.normal(loc=x_rec, scale=0.1).to_event(1), obs=x)

    def guide(self, x: torch.FloatTensor):
        pyro.module("encoder", self.encoder, update_module_params=True)
        with pyro.plate("plate", size=x.shape[0]):
            enc = self.encoder(x)
            pyro.sample("z", self.normal(loc=enc["loc"], scale=enc["scale"]).to_event(1))


def train_pyro(n_epochs: int, data_loader: torch.utils.data.DataLoader, svi: pyro.infer.SVI):
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
    assert len(tensors1) == len(tensors2), "Must pass in same number of tensors to check if they are equal"
    equal = True
    for t1, t2 in zip(tensors1, tensors2):
        equal = equal and torch.allclose(t1, t2)
    return equal


def _get_params(module: torch.nn.Module) -> List[torch.Tensor]:
    return [p.data.clone() for p in module.parameters()]


@pytest.mark.parametrize("batch_size_n", [32, 128], ids=lambda n: f"batch{n}")
def test_save_and_load_pyro_checkpoint(tmpdir_factory, batch_size_n):
    """Check that restarting from a checkpoint picks up in the same place.

    The correct comparison is not "checkpoint restart == one-shot training",
    because the new compact checkpoint format only saves the end-of-epoch
    DataLoader state, so the first-epoch shuffle ordering from the original
    training is not reproducible from scratch.  Instead we compare:

        PATH_DIRECT  — continue training on the original objects (no checkpoint)
        PATH_CKPT    — save → load checkpoint → continue training on new objects

    Both paths start from identical state (model weights, optimizer moments, RNG,
    DataLoader position), so they must produce identical weight trajectories.
    """

    filedir = tmpdir_factory.mktemp("ckpt")
    filebase = filedir.join("ckpt")
    dim = 8
    epochs = 3
    epochs2 = 3
    lr = 1e-2

    # data and dataloader (sparse format required by our DataLoader)
    dataset_dense = torch.randn((128, dim))
    dataset_sparse = sp.csr_matrix(dataset_dense.numpy())
    train_loader = new_train_loader(data=dataset_sparse, batch_size=batch_size_n)

    # create an ML model
    initial_model = PyroModel(dim=dim)
    w1 = _get_params(initial_model.encoder)

    # set up inference
    scheduler = optim.ClippedAdam({"lr": lr, "clip_norm": 10.0})
    svi = pyro.infer.SVI(initial_model.model, initial_model.guide, scheduler, loss=pyro.infer.Trace_ELBO())

    # train round 1
    initial_model.loss.extend(train_pyro(n_epochs=epochs, data_loader=train_loader, svi=svi))

    # Save checkpoint
    save_successful = save_checkpoint(
        filebase=str(filebase),
        args=argparse.Namespace(),
        model_obj=initial_model,
        scheduler=scheduler,
        train_loader=train_loader,
        tarball_name=str(filebase) + ".tar.gz",
    )
    assert save_successful, "Failed to save checkpoint during test_save_and_load_checkpoint"

    # Capture round-1 weights before PATH_DIRECT training modifies initial_model.
    w_round1 = _get_params(initial_model.encoder)

    # ---- PATH_DIRECT: continue without checkpoint ----
    # Capture RNG state right here so both paths start from the same point.
    rng_direct = RandomState()
    direct_loss = train_pyro(n_epochs=epochs2, data_loader=train_loader, svi=svi)
    direct_weights = _get_params(initial_model.encoder)

    # ---- PATH_CKPT: load from checkpoint and continue ----
    create_random_state_blank_slate(0)
    pyro.clear_param_store()
    ckpt = load_checkpoint(filebase=str(filebase), tarball_name=str(filebase) + ".tar.gz", force_device="cpu")

    # RNG must be restored to the same state as rng_direct (both captured from the
    # same post-epoch-3 state, rng_direct before any reconstruction side-effects).
    rng_ckpt = RandomState()

    # Reconstruct model.  clear_state=False preserves the param store that
    # load_checkpoint just restored so PyroOptim can load Adam moments lazily.
    # Save/restore torch RNG around model construction to avoid weight-init draws
    # advancing torch past the restored state.
    _torch_state = torch.get_rng_state()
    model_ckpt = PyroModel(dim=dim, clear_state=False)
    torch.set_rng_state(_torch_state)
    model_ckpt.load_state_dict(ckpt["model_state_dict"])

    # Reconstruct optimizer.
    scheduler_ckpt = optim.ClippedAdam({"lr": lr, "clip_norm": 10.0})
    load_optim_from_bytes(scheduler_ckpt, ckpt["optim_state_bytes"])

    # Reconstruct dataloader.  Use shuffle=False during construction to avoid an
    # extra numpy draw from _reset(); restore shuffle=True before set_state.
    loader_state = ckpt["train_loader_state"]
    cell_inds = loader_state["original_cell_indices"]
    train_loader_ckpt = DataLoader(
        dataset=dataset_sparse[cell_inds, :],
        empty_drop_dataset=None,
        batch_size=int(loader_state["batch_size"]),
        fraction_empties=float(loader_state["fraction_empties"]),
        shuffle=False,  # avoid numpy draw in __init__; restored below
        use_cuda=bool(loader_state["use_cuda"]),
        original_cell_indices=cell_inds,
        original_empty_indices=loader_state["original_empty_indices"],
    )
    train_loader_ckpt.shuffle = bool(loader_state["shuffle"])
    train_loader_ckpt.set_state(ind_list=loader_state["ind_list"], ptr=int(loader_state["ptr"]))

    s = ckpt["loaded"]

    # Verify model weights match at load time (compare against round-1 snapshot,
    # not against initial_model which PATH_DIRECT has already updated).
    assert _check_all_close(_get_params(model_ckpt.encoder), w_round1), (
        "Loaded checkpoint weights do not match the saved weights"
    )

    # Clean up checkpoint files.
    shutil.rmtree(str(filedir))

    assert s is True, "Checkpoint loading failed during test_save_and_load_checkpoint"

    # RNG must have been correctly restored by load_checkpoint.
    assert str(rng_direct) == str(rng_ckpt), (
        "RNG state after load_checkpoint does not match the state just before PATH_DIRECT training. "
        "This means load_checkpoint did not restore the random state correctly."
    )

    # Continue training from checkpoint.
    svi_ckpt = pyro.infer.SVI(model_ckpt.model, model_ckpt.guide, scheduler_ckpt, loss=pyro.infer.Trace_ELBO())
    ckpt_loss = train_pyro(n_epochs=epochs2, data_loader=train_loader_ckpt, svi=svi_ckpt)
    ckpt_weights = _get_params(model_ckpt.encoder)

    # Training should change the weights in both paths.
    assert (w1[0] != direct_weights[0]).sum().item() > 0, (
        "Training is not changing the weight matrix (PATH_DIRECT)"
    )
    assert (w1[0] != ckpt_weights[0]).sum().item() > 0, (
        "Training is not changing the checkpointed weight matrix (PATH_CKPT)"
    )

    # PATH_CKPT must reproduce PATH_DIRECT's weights exactly.
    assert _check_all_close(direct_weights, ckpt_weights), (
        "Checkpointed restart (PATH_CKPT) does not agree with direct continuation (PATH_DIRECT)"
    )


@pytest.mark.parametrize(
    "cuda",
    [False, pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA, reason="requires CUDA"))],
    ids=lambda b: "cuda" if b else "cpu",
)
@pytest.mark.parametrize("scheduler", [False, True], ids=lambda b: "OneCycleLR" if b else "Adam")
def test_save_and_load_cellbender_checkpoint(tmpdir_factory, cuda, scheduler):
    """Check that restarting from a checkpoint produces weights identical to
    uninterrupted one-shot training.

    ONE-SHOT PATH  — train all (epochs + epochs2) epochs in a single run.
    CHECKPOINT PATH — train epochs epochs, save a checkpoint, then resume and
                      train epochs2 more epochs from the checkpoint.

    Both paths start from the same RANDOM_SEED (run_inference resets it every
    call), so the only difference is whether training is continuous or
    interrupted.  If checkpoint save/restore is correct the final model weights
    must be bit-for-bit identical.
    """

    filedir = tmpdir_factory.mktemp("ckpt")

    epochs = 5
    epochs2 = 5

    # data
    n_genes = 2000
    dataset = generate_sample_dirichlet_dataset(
        n_genes=n_genes, cells_of_each_type=[100], n_droplets=2000, model_type="ambient", cell_mean_umi=[5000]
    )
    adata = get_dataset_dict_as_anndata(dataset)
    adata_file = os.path.join(filedir, "sim.h5ad")
    adata.write(adata_file)
    dataset_obj = SingleCellRNACountsDataset(
        input_file=adata_file,
        expected_cell_count=100,
        total_droplet_barcodes=1000,
        fraction_empties=0.1,
        model_name="ambient",
        gene_blacklist=[],
        exclude_features=[],
        low_count_threshold=15,
        fpr=[0.01],
    )

    # set up the inference process
    args = argparse.Namespace()
    args.output_file = os.path.join(filedir, "out.h5")
    args.z_dim = 10
    args.z_hidden_dims = [50]
    args.model = "ambient"
    args.use_cuda = cuda
    args.use_jit = False
    args.learning_rate = 1e-3
    args.training_fraction = 0.9
    args.fraction_empties = 0.1
    args.checkpoint_min = 5
    args.epoch_elbo_fail_fraction = None
    args.final_elbo_fail_fraction = None
    args.constant_learning_rate = not scheduler
    args.debug = False
    args.force_use_checkpoint = False

    # Compute the hashcode-based checkpoint filename once (excludes 'epochs').
    hashcode = create_workflow_hashcode(module_path=os.path.dirname(cellbender.__file__), args=args)[:10]
    checkpoint_filename = os.path.basename(args.output_file).split(".")[0] + "_" + hashcode
    filebase = filedir.join(checkpoint_filename)
    args.checkpoint_filename = checkpoint_filename

    # --- ONE-SHOT PATH ---
    # run_inference resets the RNG to RANDOM_SEED on every call, so this path
    # and the checkpoint round-1 path below start from identical initial state.
    args.input_checkpoint_tarball = "none"
    args.epochs = epochs + epochs2
    model_oneshot, _, _, _ = run_inference(
        dataset_obj=dataset_obj,
        args=args,
        output_checkpoint_tarball="none",
        total_epochs_for_testing_only=epochs + epochs2,
    )
    weights_oneshot = _get_params(model_oneshot.encoder["z"])

    # --- CHECKPOINT PATH, ROUND 1 ---
    # Train the first `epochs` epochs.  Identical to one-shot up to epoch
    # `epochs` because both calls start from the same RANDOM_SEED.
    args.input_checkpoint_tarball = "none"
    args.epochs = epochs
    model_r1, scheduler_r1, train_loader, test_loader = run_inference(
        dataset_obj=dataset_obj,
        args=args,
        output_checkpoint_tarball="none",
        total_epochs_for_testing_only=epochs + epochs2,
    )

    # Save checkpoint after round 1.
    save_successful = save_checkpoint(
        filebase=str(filebase),
        args=args,
        model_obj=model_r1,
        scheduler=scheduler_r1,
        tarball_name=str(filebase) + ".tar.gz",
        train_loader=train_loader,
        test_loader=test_loader,
    )
    assert save_successful, "Failed to save checkpoint during test_save_and_load_cellbender_checkpoint"
    assert os.path.exists(str(filebase) + ".tar.gz"), "Checkpoint tarball should exist but does not"

    # --- CHECKPOINT PATH, ROUND 2 ---
    # Resume from checkpoint and train epochs2 more epochs.
    # run_inference resets RNG to RANDOM_SEED first, then load_checkpoint
    # overwrites RNG with the saved post-round-1 state, so training resumes
    # from exactly where round 1 left off.
    args.input_checkpoint_tarball = str(filebase) + ".tar.gz"
    args.epochs = epochs + epochs2
    model_resumed, _, _, _ = run_inference(
        dataset_obj=dataset_obj,
        args=args,
        output_checkpoint_tarball="none",
        total_epochs_for_testing_only=epochs + epochs2,
    )
    weights_resumed = _get_params(model_resumed.encoder["z"])

    # Clean up all checkpoint files now that both runs are done.
    shutil.rmtree(str(filedir))

    # Training must change the weights relative to initial (one-shot as proxy).
    w1 = _get_params(model_r1.encoder["z"])  # weights after round-1 training
    assert (weights_oneshot[0] != w1[0]).sum().item() > 0, (
        "Training is not changing the weight matrix in test_save_and_load_cellbender_checkpoint"
    )

    # One-shot and checkpoint-resume must produce bit-for-bit identical weights.
    assert _check_all_close(weights_oneshot, weights_resumed), (
        "One-shot training and checkpoint-resume training produced different weights — "
        "checkpoint save/restore is not fully deterministic"
    )


# ---------------------------------------------------------------------------
# Fixtures and helpers shared by async checkpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def reset_async_state():
    """Reset module-level async checkpoint state before and after each test."""
    import cellbender.remove_background.checkpoint as ckpt_module

    ckpt_module._pending_ckpt = None
    yield
    # Best-effort cleanup: drain any future left over by a failing test.
    if ckpt_module._pending_ckpt is not None:
        try:
            ckpt_module._pending_ckpt.result(timeout=10)
        except Exception:
            pass
        ckpt_module._pending_ckpt = None


def _make_trained_model(dim: int = 8, n_epochs: int = 2, batch_size: int = 32):
    """Return (model, scheduler, train_loader) after a short training run."""
    import pyro.optim as optim

    dataset_dense = torch.randn((64, dim))
    dataset_sparse = sp.csr_matrix(dataset_dense.numpy())
    loader = new_train_loader(data=dataset_sparse, batch_size=batch_size)
    create_random_state_blank_slate(0)
    pyro.clear_param_store()
    model = PyroModel(dim=dim)
    scheduler = optim.ClippedAdam({"lr": 1e-2, "clip_norm": 10.0})
    svi = pyro.infer.SVI(model.model, model.guide, scheduler, loss=pyro.infer.Trace_ELBO())
    train_pyro(n_epochs=n_epochs, data_loader=loader, svi=svi)
    return model, scheduler, loader


# ---------------------------------------------------------------------------
# Tests for async checkpoint functions
# ---------------------------------------------------------------------------


def test_compact_dataloader_state_roundtrip(tmpdir_factory):
    """DataLoader.get_state() contains all fields needed to reconstruct the loader,
    and restoring via set_state() + slicing puts the loader back in the same position."""
    import io as _io

    dim = 4
    n = 64
    dataset_dense = torch.randn((n, dim))
    dataset_sparse = sp.csr_matrix(dataset_dense.numpy())

    loader = new_train_loader(data=dataset_sparse, batch_size=16)
    # Advance the pointer by consuming one batch so state is non-trivial.
    _ = next(iter(loader))

    state = loader.get_state()
    assert "original_cell_indices" in state
    assert "ind_list" in state
    assert "ptr" in state
    assert "batch_size" in state

    # Round-trip through npz bytes (same as checkpoint.py does).
    buf = _io.BytesIO()
    np.savez_compressed(buf, **state)
    buf.seek(0)
    restored_state = dict(np.load(buf, allow_pickle=False))

    cell_inds = restored_state["original_cell_indices"]
    loader2 = DataLoader(
        dataset=dataset_sparse[cell_inds, :],
        empty_drop_dataset=None,
        batch_size=int(restored_state["batch_size"]),
        fraction_empties=float(restored_state["fraction_empties"]),
        shuffle=bool(restored_state["shuffle"]),
        use_cuda=bool(restored_state["use_cuda"]),
        original_cell_indices=cell_inds,
        original_empty_indices=restored_state["original_empty_indices"],
    )
    loader2.set_state(ind_list=restored_state["ind_list"], ptr=int(restored_state["ptr"]))

    # Both loaders must yield the same first batch from here.
    batch1 = next(iter(loader))
    batch2 = next(iter(loader2))
    assert torch.allclose(batch1, batch2), "Restored DataLoader must yield the same batch as the original"


def test_flush_pending_checkpoint_no_op_when_nothing_pending(reset_async_state):
    """flush_pending_checkpoint returns True immediately when nothing is queued."""
    result = flush_pending_checkpoint()
    assert result is True


def test_save_checkpoint_async_returns_future_and_produces_tarball(tmpdir_factory, reset_async_state):
    """save_checkpoint_async should return a Future and write a readable tarball."""
    filedir = tmpdir_factory.mktemp("async_ckpt")
    filebase = str(filedir.join("ckpt"))
    tarball = filebase + ".tar.gz"

    model, scheduler, train_loader = _make_trained_model()

    future = save_checkpoint_async(
        filebase=filebase,
        args=argparse.Namespace(),
        model_obj=model,
        scheduler=scheduler,
        train_loader=train_loader,
        tarball_name=tarball,
    )
    assert future is not None, "save_checkpoint_async should return a Future"

    success = flush_pending_checkpoint()
    assert success is True, "flush_pending_checkpoint should return True on success"
    assert os.path.exists(tarball), "Tarball must exist after flush"

    # The tarball must be loadable and contain the correct model weights.
    pyro.clear_param_store()
    ckpt = load_checkpoint(filebase=filebase, tarball_name=tarball, force_device="cpu")
    assert ckpt["loaded"] is True
    fresh_model = PyroModel(dim=8)
    fresh_model.load_state_dict(ckpt["model_state_dict"])
    assert _check_all_close(
        _get_params(fresh_model.encoder),
        _get_params(model.encoder),
    ), "Loaded model weights must match saved model weights"


def test_flush_pending_checkpoint_returns_true_on_success(tmpdir_factory, reset_async_state):
    """flush_pending_checkpoint returns True when the background write succeeded."""
    filedir = tmpdir_factory.mktemp("flush_true")
    filebase = str(filedir.join("ckpt"))
    model, scheduler, train_loader = _make_trained_model()
    save_checkpoint_async(
        filebase=filebase,
        args=argparse.Namespace(),
        model_obj=model,
        scheduler=scheduler,
        train_loader=train_loader,
        tarball_name=filebase + ".tar.gz",
    )
    result = flush_pending_checkpoint()
    assert result is True
    # A second flush with nothing pending should also return True.
    assert flush_pending_checkpoint() is True


def test_save_checkpoint_async_second_call_waits_for_first(tmpdir_factory, reset_async_state):
    """Calling save_checkpoint_async twice in a row must complete both writes."""
    filedir = tmpdir_factory.mktemp("async_seq")
    filebase = str(filedir.join("ckpt"))
    tarball1 = filebase + "_1.tar.gz"
    tarball2 = filebase + "_2.tar.gz"

    model, scheduler, train_loader = _make_trained_model()

    # First async save.
    save_checkpoint_async(
        filebase=filebase,
        args=argparse.Namespace(),
        model_obj=model,
        scheduler=scheduler,
        train_loader=train_loader,
        tarball_name=tarball1,
    )
    # Second async save — implementation must drain _pending_ckpt first.
    save_checkpoint_async(
        filebase=filebase,
        args=argparse.Namespace(),
        model_obj=model,
        scheduler=scheduler,
        train_loader=train_loader,
        tarball_name=tarball2,
    )
    flush_pending_checkpoint()

    assert os.path.exists(tarball1), "First checkpoint tarball must exist"
    assert os.path.exists(tarball2), "Second checkpoint tarball must exist"


def test_async_checkpoint_matches_sync_checkpoint(tmpdir_factory, reset_async_state):
    """Async and sync checkpoints of the same state should produce equivalent model weights."""
    filedir = tmpdir_factory.mktemp("async_vs_sync")
    filebase = str(filedir.join("ckpt"))
    tarball_sync = filebase + "_sync.tar.gz"
    tarball_async = filebase + "_async.tar.gz"

    model, scheduler, train_loader = _make_trained_model()

    # Synchronous save.
    save_checkpoint(
        filebase=filebase,
        args=argparse.Namespace(),
        model_obj=model,
        scheduler=scheduler,
        tarball_name=tarball_sync,
    )
    # Async save of the *same* model state.
    save_checkpoint_async(
        filebase=filebase,
        args=argparse.Namespace(),
        model_obj=model,
        scheduler=scheduler,
        tarball_name=tarball_async,
    )
    flush_pending_checkpoint()

    pyro.clear_param_store()
    ckpt_sync = load_checkpoint(filebase=filebase, tarball_name=tarball_sync, force_device="cpu")
    pyro.clear_param_store()
    ckpt_async = load_checkpoint(filebase=filebase, tarball_name=tarball_async, force_device="cpu")

    fresh_sync = PyroModel(dim=8)
    fresh_sync.load_state_dict(ckpt_sync["model_state_dict"])
    fresh_async = PyroModel(dim=8)
    fresh_async.load_state_dict(ckpt_async["model_state_dict"])
    assert _check_all_close(
        _get_params(fresh_sync.encoder),
        _get_params(fresh_async.encoder),
    ), "Async checkpoint encoder weights must match sync checkpoint encoder weights"


@pytest.mark.parametrize(
    "Optim, config",
    [
        (optim.ClippedAdam, {"lr": 0.01}),
        (getattr(optim, "ExponentialLR"), {"optimizer": torch.optim.SGD, "optim_args": {"lr": 0.01}, "gamma": 0.9}),
    ],
)
def test_optimizer_checkpoint_restart(Optim, config, tmpdir_factory):
    """This code has been copied from a version of a pyro test by Fritz Obermeyer"""

    tempdir = tmpdir_factory.mktemp("ckpt")

    def model():
        x_scale = pyro.param("x_scale", torch.tensor(1.0), constraint=constraints.positive)
        z = pyro.sample("z", pyro.distributions.Normal(0, 1))
        return pyro.sample("x", pyro.distributions.Normal(z, x_scale), obs=torch.tensor(0.1))

    def guide():
        z_loc = pyro.param("z_loc", torch.tensor(0.0))
        z_scale = pyro.param("z_scale", torch.tensor(0.5), constraint=constraints.positive)
        pyro.sample("z", pyro.distributions.Normal(z_loc, z_scale))

    store = pyro.get_param_store()

    def get_snapshot(optimizer):
        s = {k: v.data.clone() for k, v in store.items()}
        if isinstance(optimizer, optim.lr_scheduler.PyroLRScheduler):
            lr = list(optimizer.optim_objs.values())[0].get_last_lr()[0]
        else:
            lr = list(optimizer.optim_objs.values())[0].param_groups[0]["lr"]
        s.update({"lr": f"{lr:.4f}"})
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
        if isinstance(optimizer, optim.lr_scheduler.PyroLRScheduler):
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
        if isinstance(optimizer, optim.lr_scheduler.PyroLRScheduler):
            svi.optim.step()

    # checkpoint
    optim_filename = os.path.join(tempdir, "optimizer_state.pt")
    param_filename = os.path.join(tempdir, "param_store.pt")
    optimizer.save(optim_filename)
    store.save(param_filename)
    del optimizer, svi
    store.clear()

    # load from checkpoint
    load_param_store(param_filename)
    optimizer = Optim(config.copy())
    optimizer.load(optim_filename)
    svi = pyro.infer.SVI(model, guide, optimizer, pyro.infer.Trace_ELBO())
    for _ in range(10):
        svi.step()
        actual.append(get_snapshot(optimizer))
        if isinstance(optimizer, optim.lr_scheduler.PyroLRScheduler):
            svi.optim.step()

    # display learning rates and actual/expected values for z_loc
    print(repr(optimizer))
    print("epoch\t\tlr\t\tactual\t\t\t\texpected")
    print("-" * 100)
    for i, (ac, ex) in enumerate(zip(actual, expected)):
        print("\t\t".join([f"{x}" for x in [i, ac["lr"], ac["z_loc"], ex["z_loc"]]]))
        if i == 4:
            print("-" * 100)

    # ensure actual matches expected
    for actual_t, expected_t in zip(actual, expected):
        actual_t.pop("lr")
        expected_t.pop("lr")
        for actual_value, expected_value in zip(actual_t.values(), expected_t.values()):
            assert torch.allclose(actual_value, expected_value)
