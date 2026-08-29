"""Tests for dataprep.py and backends.py."""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from cellbender.remove_background.data.backends import InMemoryBackend, MmapBackend
from cellbender.remove_background.data.dataprep import (
    _make_loader,
    make_simple_dataloader,
    prep_sparse_data_for_training,
    reconstruct_loader,
)

USE_CUDA = torch.cuda.is_available()


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

N_CELLS = 200
N_EMPTIES = 80
N_GENES = 40
BATCH_SIZE = 32
FRACTION_EMPTIES = 0.2


@pytest.fixture(scope="module")
def matrices():
    rng = np.random.RandomState(0)
    cell_mat = sp.random(N_CELLS, N_GENES, density=0.1, format="csr", dtype=np.float32, random_state=rng)
    empty_mat = sp.random(N_EMPTIES, N_GENES, density=0.05, format="csr", dtype=np.float32, random_state=rng)
    return cell_mat, empty_mat


# ──────────────────────────────────────────────────────────────────────────────
# Backends
# ──────────────────────────────────────────────────────────────────────────────


def test_in_memory_backend_row_fetch(matrices):
    cell_mat, _ = matrices
    backend = InMemoryBackend(cell_mat)
    assert len(backend) == N_CELLS
    inds = np.array([0, 5, 99, 150])
    fetched = np.asarray(backend[inds].todense(), dtype=np.float32)
    expected = np.asarray(cell_mat[inds].todense(), dtype=np.float32)
    assert np.allclose(fetched, expected)


def test_mmap_backend_round_trip(matrices):
    cell_mat, _ = matrices
    ref = np.asarray(cell_mat.todense(), dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        backend = MmapBackend.create(cell_mat, Path(d))
        assert MmapBackend.exists(Path(d))
        inds = np.array([3, 0, 150, 7, 99])
        fetched = np.asarray(backend[inds].todense(), dtype=np.float32)
        assert np.allclose(fetched, ref[inds])


# ──────────────────────────────────────────────────────────────────────────────
# make_simple_dataloader
# ──────────────────────────────────────────────────────────────────────────────


def test_simple_dataloader_emits_all_rows(matrices):
    cell_mat, _ = matrices
    loader = make_simple_dataloader(cell_mat, batch_size=16, use_cuda=False)
    total = sum(b.shape[0] for b in loader)
    assert total == N_CELLS


def test_simple_dataloader_batch_shape(matrices):
    cell_mat, _ = matrices
    loader = make_simple_dataloader(cell_mat, batch_size=16, use_cuda=False)
    for batch in loader:
        assert batch.shape[1] == N_GENES
        break


# ──────────────────────────────────────────────────────────────────────────────
# prep_sparse_data_for_training
# ──────────────────────────────────────────────────────────────────────────────


def test_train_loader_basic(matrices):
    cell_mat, empty_mat = matrices
    np.random.seed(1)
    torch.manual_seed(1)
    train, test = prep_sparse_data_for_training(
        dataset=cell_mat,
        empty_drop_dataset=empty_mat,
        batch_size=BATCH_SIZE,
        fraction_empties=FRACTION_EMPTIES,
        shuffle=True,
        use_cuda=False,
    )
    assert train.batch_size == BATCH_SIZE
    assert train.device == "cpu"
    assert len(train) > 0
    # batch shape
    for batch in train:
        assert batch.shape[1] == N_GENES
        break


def test_train_loader_analytical_length_matches_iteration(matrices):
    cell_mat, empty_mat = matrices
    np.random.seed(2)
    torch.manual_seed(2)
    train, _ = prep_sparse_data_for_training(
        dataset=cell_mat,
        empty_drop_dataset=empty_mat,
        batch_size=BATCH_SIZE,
        fraction_empties=FRACTION_EMPTIES,
        shuffle=True,
        use_cuda=False,
    )
    reported_len = len(train)
    actual_len = sum(1 for _ in train)
    assert reported_len == actual_len


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint determinism
# ──────────────────────────────────────────────────────────────────────────────


def test_checkpoint_determinism_epoch_boundary(matrices):
    """Two loaders restored to the same epoch-boundary state produce identical batches."""
    cell_mat, empty_mat = matrices
    np.random.seed(3)
    torch.manual_seed(3)
    train, _ = prep_sparse_data_for_training(
        dataset=cell_mat,
        empty_drop_dataset=empty_mat,
        batch_size=BATCH_SIZE,
        fraction_empties=FRACTION_EMPTIES,
        shuffle=True,
        use_cuda=False,
    )

    # Run one full epoch so the state is at an epoch boundary.
    for _ in train:
        pass
    state = train.get_state()

    # Restore the state into two independent loaders.
    train.set_state(state)
    train2 = reconstruct_loader(state, cell_mat, empty_mat, use_cuda=False)

    for b1, b2 in zip(train, train2):
        assert torch.equal(b1, b2), "Batch mismatch after epoch-boundary restore"


def test_checkpoint_determinism_mid_epoch(matrices):
    """Restore mid-epoch (ptr > 0) and verify remaining batches are identical."""
    cell_mat, empty_mat = matrices
    np.random.seed(4)
    torch.manual_seed(4)
    train, _ = prep_sparse_data_for_training(
        dataset=cell_mat,
        empty_drop_dataset=empty_mat,
        batch_size=BATCH_SIZE,
        fraction_empties=FRACTION_EMPTIES,
        shuffle=True,
        use_cuda=False,
    )

    # Consume a few batches manually to land mid-epoch.
    n_skip = 3
    it = iter(train)
    _ = [next(it) for _ in range(n_skip)]
    state = train.get_state()
    assert int(state["ptr"]) == n_skip

    # Restore both loaders to the mid-epoch state.
    train.set_state(state)
    train2 = reconstruct_loader(state, cell_mat, empty_mat, use_cuda=False)

    for b1, b2 in zip(train, train2):
        assert torch.equal(b1, b2), "Batch mismatch after mid-epoch restore"


def test_checkpoint_npz_round_trip(matrices):
    """State survives numpy npz serialization (as checkpoint.py does it)."""
    cell_mat, empty_mat = matrices
    np.random.seed(5)
    torch.manual_seed(5)
    train, _ = prep_sparse_data_for_training(
        dataset=cell_mat,
        empty_drop_dataset=empty_mat,
        batch_size=BATCH_SIZE,
        fraction_empties=FRACTION_EMPTIES,
        shuffle=True,
        use_cuda=False,
    )

    # Advance one batch.
    next(iter(train))
    state = train.get_state()

    # Serialize to npz (no pickle) and load back.
    buf = io.BytesIO()
    np.savez_compressed(buf, **state)
    buf.seek(0)
    loaded_state = dict(np.load(buf, allow_pickle=False))

    train.set_state(state)
    train2 = reconstruct_loader(loaded_state, cell_mat, empty_mat, use_cuda=False)

    for b1, b2 in zip(train, train2):
        assert torch.equal(b1, b2), "npz round-trip mismatch"


def test_mmap_backend_in_dataloader(matrices):
    """MmapBackend in a DataLoader produces identical batches to InMemoryBackend."""
    cell_mat, empty_mat = matrices
    cell_batch_size = BATCH_SIZE - int(BATCH_SIZE * FRACTION_EMPTIES)
    n_empty_per_batch = BATCH_SIZE - cell_batch_size
    seed = 99

    with tempfile.TemporaryDirectory() as d:
        cell_mmap = MmapBackend.create(cell_mat, Path(d) / "cell")
        empty_mmap = MmapBackend.create(empty_mat, Path(d) / "empty")

        mem_loader = _make_loader(
            cell_backend=InMemoryBackend(cell_mat),
            empty_backend=InMemoryBackend(empty_mat),
            cell_batch_size=cell_batch_size,
            n_empty_per_batch=n_empty_per_batch,
            generator=torch.Generator().manual_seed(seed),
            shuffle=True,
            use_cuda=False,
        )
        mmap_loader = _make_loader(
            cell_backend=cell_mmap,
            empty_backend=empty_mmap,
            cell_batch_size=cell_batch_size,
            n_empty_per_batch=n_empty_per_batch,
            generator=torch.Generator().manual_seed(seed),
            shuffle=True,
            use_cuda=False,
        )

        for b_mem, b_mmap in zip(mem_loader, mmap_loader):
            assert torch.equal(b_mem, b_mmap), "InMemory vs MmapBackend batch mismatch"


def test_checkpoint_backward_compat_old_ind_list(matrices):
    """reconstruct_loader handles old-style state with ind_list instead of perm."""
    cell_mat, empty_mat = matrices
    np.random.seed(6)
    torch.manual_seed(6)
    train, _ = prep_sparse_data_for_training(
        dataset=cell_mat,
        empty_drop_dataset=empty_mat,
        batch_size=BATCH_SIZE,
        fraction_empties=FRACTION_EMPTIES,
        shuffle=True,
        use_cuda=False,
    )
    # Force one epoch so perm is populated.
    next(iter(train))
    state = train.get_state()

    # Simulate an old checkpoint: rename perm→ind_list, drop new-style keys.
    old_state = dict(state)
    old_state["ind_list"] = old_state.pop("perm")
    old_state.pop("gen_state", None)
    old_state.pop("cell_batch_size", None)
    old_state.pop("n_empty_per_batch", None)

    loader = reconstruct_loader(old_state, cell_mat, empty_mat, use_cuda=False)
    assert loader.batch_size == BATCH_SIZE
    # Should be able to iterate without error.
    batches = list(loader)
    assert len(batches) > 0
