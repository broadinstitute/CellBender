"""DataLoader construction for CellBender training.

Replaces the previous hand-rolled background thread with a standard
torch.utils.data.DataLoader backed by a CellEmptyBatchSampler that owns a
torch.Generator for all shuffle and empty-drop randomness.

Key properties:
- Deterministic checkpoint/resume: save generator.get_state() + epoch permutation.
- Pluggable storage: InMemoryBackend (default) or MmapBackend for out-of-RAM datasets.
- Backward-compatible interface: the DataLoader wrapper exposes the same attributes
  (device, batch_size, get_state, set_state, reset_ptr) used by train.py, run.py,
  and checkpoint.py.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data

import cellbender.remove_background.consts as consts
from cellbender.remove_background.data.backends import InMemoryBackend, MmapBackend, _SlicedBackend

logger = logging.getLogger("cellbender")


def _get_or_create_mmap(matrix: sp.csr_matrix, cache_dir: Path) -> MmapBackend:
    """Return a MmapBackend for *matrix*, creating mmap files if absent."""
    if MmapBackend.exists(cache_dir):
        logger.info(f"Reusing mmap cache at {cache_dir}")
        return MmapBackend(cache_dir)
    logger.info(f"Writing mmap cache to {cache_dir} (one-time cost)...")
    backend = MmapBackend.create(matrix, cache_dir)
    logger.info("Mmap cache written.")
    return backend


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────


class _SCRNADataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping two matrix backends (cells and empties).

    Indices 0..n_cells-1 address the cell backend; n_cells.. address the
    empty-drop backend.  __getitem__ returns a dense float32 row tensor.
    """

    def __init__(self, cell_backend, empty_backend=None) -> None:
        self._cell = cell_backend
        self._empty = empty_backend
        self.n_cells: int = len(cell_backend)
        self.n_empties: int = len(empty_backend) if empty_backend is not None else 0

    def __len__(self) -> int:
        return self.n_cells + self.n_empties

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self), self._cell.shape[1])

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < self.n_cells:
            row = self._cell[np.array([idx])]
        else:
            row = self._empty[np.array([idx - self.n_cells])]
        return torch.from_numpy(np.asarray(row.todense(), dtype=np.float32).squeeze(0))


# ──────────────────────────────────────────────────────────────────────────────
# Batch sampler
# ──────────────────────────────────────────────────────────────────────────────


class CellEmptyBatchSampler(torch.utils.data.Sampler):
    """Yields mixed batches of cell and empty-drop indices.

    Cells are shuffled once per epoch using a owned torch.Generator.
    Empty indices are sampled with replacement from that same generator each
    batch.  All randomness is fully contained in the generator, so saving
    generator.get_state() + the current epoch permutation + the batch pointer
    is sufficient to resume training bit-identically from any checkpoint.

    Indices emitted are into the combined _SCRNADataset space:
        0 .. n_cells-1           → cells
        n_cells .. n_cells+n_empties-1  → empties
    """

    def __init__(
        self,
        n_cells: int,
        n_empties: int,
        cell_batch_size: int,
        n_empty_per_batch: int,
        generator: torch.Generator,
        shuffle: bool = True,
        original_cell_indices: Optional[np.ndarray] = None,
        original_empty_indices: Optional[np.ndarray] = None,
    ) -> None:
        self._n_cells = n_cells
        self._n_empties = n_empties
        self._cell_batch_size = cell_batch_size
        self._n_empty_per_batch = n_empty_per_batch
        self._gen = generator
        self._shuffle = shuffle
        self.original_cell_indices = original_cell_indices
        self.original_empty_indices = original_empty_indices
        # Mutable epoch state.
        self._perm: Optional[np.ndarray] = None  # cell permutation for current epoch
        self._ptr: int = 0  # batches already emitted this epoch

    # ------------------------------------------------------------------
    # Length (analytical — no iteration needed)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        n = self._n_cells // self._cell_batch_size
        if self._n_cells % self._cell_batch_size >= consts.SMALLEST_ALLOWED_BATCH:
            n += 1
        return n

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        # Generate permutation if not already set (new epoch or first run).
        if self._perm is None:
            if self._shuffle:
                self._perm = torch.randperm(self._n_cells, generator=self._gen).numpy()
            else:
                self._perm = np.arange(self._n_cells, dtype=np.int64)

        # Start from ptr to allow mid-epoch resume.
        i = self._ptr * self._cell_batch_size

        while i < len(self._perm):
            end = min(i + self._cell_batch_size, len(self._perm))
            cell_chunk = self._perm[i:end]
            if len(cell_chunk) < consts.SMALLEST_ALLOWED_BATCH:
                logger.debug(f"Dropped last minibatch of {len(cell_chunk)} cells")
                break

            if self._n_empties > 0 and self._n_empty_per_batch > 0:
                empty_local = torch.randint(self._n_empties, (self._n_empty_per_batch,), generator=self._gen).numpy()
                batch = np.concatenate([cell_chunk, empty_local + self._n_cells]).tolist()
            else:
                batch = cell_chunk.tolist()

            self._ptr += 1
            yield batch
            i += self._cell_batch_size

        # End of epoch: reset for next epoch.
        self._perm = None
        self._ptr = 0

    # ------------------------------------------------------------------
    # Checkpoint state
    # ------------------------------------------------------------------

    def get_state(self) -> Dict:
        batch_size = self._cell_batch_size + self._n_empty_per_batch
        fraction_empties = self._n_empty_per_batch / batch_size if batch_size > 0 else 0.0
        state: Dict = {
            "perm": self._perm if self._perm is not None else np.array([], dtype=np.int64),
            "ptr": np.int64(self._ptr),
            "batch_size": np.int64(batch_size),
            "cell_batch_size": np.int64(self._cell_batch_size),
            "n_empty_per_batch": np.int64(self._n_empty_per_batch),
            "fraction_empties": np.float64(fraction_empties),
            "shuffle": np.bool_(self._shuffle),
            "gen_state": self._gen.get_state().numpy(),
        }
        if self.original_cell_indices is not None:
            state["original_cell_indices"] = self.original_cell_indices
        if self.original_empty_indices is not None:
            state["original_empty_indices"] = self.original_empty_indices
        return state

    def set_state(self, state: Dict) -> None:
        perm = state.get("perm", np.array([], dtype=np.int64))
        self._perm = perm if len(perm) > 0 else None
        self._ptr = int(state["ptr"])
        if "gen_state" in state:
            self._gen.set_state(torch.tensor(state["gen_state"]))


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader wrapper
# ──────────────────────────────────────────────────────────────────────────────


class DataLoader:
    """Thin wrapper around torch.utils.data.DataLoader.

    Preserves the interface expected by train.py, run.py, and checkpoint.py:
    ``device``, ``batch_size``, ``get_state``/``set_state``, ``reset_ptr``.

    Iteration is epoch-scoped: each for-loop over this object covers one
    training epoch and raises StopIteration at epoch end.
    """

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        sampler: Optional[CellEmptyBatchSampler],
        use_cuda: bool,
        batch_size: int,
        fraction_empties: float = 0.0,
    ) -> None:
        self._torch_loader = torch_loader
        self._sampler = sampler
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.fraction_empties = fraction_empties
        self._device = "cuda" if use_cuda else "cpu"
        self._inner_iter: Optional[Iterator[torch.Tensor]] = None
        self._length: Optional[int] = None

    @property
    def dataset(self) -> torch.utils.data.Dataset:
        return self._torch_loader.dataset

    @property
    def cell_batch_size(self) -> int:
        if self._sampler is not None:
            return self._sampler._cell_batch_size
        return self.batch_size

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # Iteration (epoch-scoped)
    # ------------------------------------------------------------------

    def __iter__(self):
        self._inner_iter = iter(self._torch_loader)
        return self

    def __next__(self) -> torch.Tensor:
        if self._inner_iter is None:
            self._inner_iter = iter(self._torch_loader)
        try:
            return next(self._inner_iter)
        except StopIteration:
            raise

    def __len__(self) -> int:
        if self._length is None:
            if self._sampler is not None:
                self._length = len(self._sampler)
            else:
                self._length = len(self._torch_loader)
        return self._length

    # ------------------------------------------------------------------
    # Compatibility methods
    # ------------------------------------------------------------------

    def reset_ptr(self) -> None:
        """Reset to the start of the current epoch permutation without re-shuffling.

        Called in the zero-epoch path after consuming one batch for model init.
        Keeps the saved permutation so a subsequent checkpoint captures the same
        epoch order.
        """
        self._inner_iter = None
        if self._sampler is not None:
            self._sampler._ptr = 0
            # Intentionally keep _perm so the epoch order is preserved in the checkpoint.

    def get_state(self) -> Dict:
        state = self._sampler.get_state() if self._sampler is not None else {}
        state["use_cuda"] = np.bool_(self.use_cuda)
        if self._length is not None:
            state["_length"] = np.int64(self._length)
        return state

    def set_state(self, state: Dict) -> None:
        if self._sampler is not None:
            self._sampler.set_state(state)
        self._inner_iter = None
        if "_length" in state:
            self._length = int(state["_length"])

    def close(self) -> None:
        self._inner_iter = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────────────────────────────────────


def _make_loader(
    cell_backend,
    empty_backend,
    cell_batch_size: int,
    n_empty_per_batch: int,
    generator: torch.Generator,
    shuffle: bool,
    use_cuda: bool,
    original_cell_indices: Optional[np.ndarray] = None,
    original_empty_indices: Optional[np.ndarray] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Internal factory: given backends and parameters, build a DataLoader."""
    dataset = _SCRNADataset(cell_backend, empty_backend)
    sampler = CellEmptyBatchSampler(
        n_cells=cell_backend.__len__(),
        n_empties=empty_backend.__len__() if empty_backend is not None else 0,
        cell_batch_size=cell_batch_size,
        n_empty_per_batch=n_empty_per_batch,
        generator=generator,
        shuffle=shuffle,
        original_cell_indices=original_cell_indices,
        original_empty_indices=original_empty_indices,
    )
    torch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    total_batch_size = cell_batch_size + n_empty_per_batch
    fraction_empties = n_empty_per_batch / total_batch_size if total_batch_size > 0 else 0.0
    return DataLoader(
        torch_loader=torch_loader,
        sampler=sampler,
        use_cuda=use_cuda,
        batch_size=total_batch_size,
        fraction_empties=fraction_empties,
    )


def prep_sparse_data_for_training(
    dataset: sp.csr_matrix,
    empty_drop_dataset: sp.csr_matrix,
    training_fraction: float = consts.TRAINING_FRACTION,
    fraction_empties: float = consts.FRACTION_EMPTIES,
    batch_size: int = consts.DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    use_cuda: bool = True,
    mmap_cache_dir: Optional[Path] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders from sparse count matrices.

    Args:
        dataset: Cell/droplet count matrix [barcodes × genes].
        empty_drop_dataset: Surely-empty droplet count matrix.
        training_fraction: Fraction of barcodes used for training.
        fraction_empties: Fraction of each minibatch from empty droplets.
        batch_size: Total minibatch size (cells + empties).
        shuffle: Shuffle cell order each epoch.
        use_cuda: Pin tensors for non-blocking GPU transfer.
        mmap_cache_dir: If given, matrices are stored as memory-mapped files
            under this directory (``cells/`` and ``empties/`` subdirs).  The
            DataLoaders then access the data via mmap rather than keeping
            submatrices in RAM.

    Returns:
        (train_loader, test_loader)
    """
    cell_batch_size = int(batch_size * (1.0 - fraction_empties))
    n_empty_per_batch = batch_size - cell_batch_size

    # Train/test split (uses global numpy RNG seeded by pyro.set_rng_seed).
    training_mask = np.random.rand(dataset.shape[0]) < training_fraction
    training_indices = np.where(training_mask)[0]
    test_indices = np.where(~training_mask)[0]

    training_mask_empty = np.random.rand(empty_drop_dataset.shape[0]) < training_fraction
    training_indices_empty = np.where(training_mask_empty)[0]
    test_indices_empty = np.where(~training_mask_empty)[0]

    # Seed generators from the current torch state so they are deterministic.
    train_gen = torch.Generator().manual_seed(torch.initial_seed())
    test_gen = torch.Generator().manual_seed(torch.initial_seed() + 1)

    if mmap_cache_dir is not None:
        cell_mmap = _get_or_create_mmap(dataset, mmap_cache_dir / "cells")
        empty_mmap = _get_or_create_mmap(empty_drop_dataset, mmap_cache_dir / "empties")
        train_cell_backend: _SlicedBackend | InMemoryBackend = _SlicedBackend(cell_mmap, training_indices)
        train_empty_backend: _SlicedBackend | InMemoryBackend = _SlicedBackend(empty_mmap, training_indices_empty)
        test_cell_backend: _SlicedBackend | InMemoryBackend = _SlicedBackend(cell_mmap, test_indices)
        test_empty_backend: _SlicedBackend | InMemoryBackend = _SlicedBackend(empty_mmap, test_indices_empty)
    else:
        train_cell_backend = InMemoryBackend(dataset[training_indices])
        train_empty_backend = InMemoryBackend(empty_drop_dataset[training_indices_empty])
        test_cell_backend = InMemoryBackend(dataset[test_indices])
        test_empty_backend = InMemoryBackend(empty_drop_dataset[test_indices_empty])

    train_loader = _make_loader(
        cell_backend=train_cell_backend,
        empty_backend=train_empty_backend,
        cell_batch_size=cell_batch_size,
        n_empty_per_batch=n_empty_per_batch,
        generator=train_gen,
        shuffle=shuffle,
        use_cuda=use_cuda,
        original_cell_indices=training_indices,
        original_empty_indices=training_indices_empty,
        num_workers=num_workers,
    )
    test_loader = _make_loader(
        cell_backend=test_cell_backend,
        empty_backend=test_empty_backend,
        cell_batch_size=cell_batch_size,
        n_empty_per_batch=n_empty_per_batch,
        generator=test_gen,
        shuffle=shuffle,
        use_cuda=use_cuda,
        original_cell_indices=test_indices,
        original_empty_indices=test_indices_empty,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def make_simple_dataloader(
    matrix: sp.csr_matrix,
    batch_size: int,
    use_cuda: bool = True,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader for a single matrix (no empty-drop mixing).

    Used by SingleCellRNACountsDataset.get_dataloader() for inference passes.
    All rows are emitted (no dropped tails).

    Args:
        matrix: Count matrix to load.
        batch_size: Rows per batch.
        use_cuda: Pin tensors for GPU transfer.
        shuffle: Shuffle row order.
    """
    backend = InMemoryBackend(matrix)
    dataset = _SCRNADataset(cell_backend=backend)

    torch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    return DataLoader(
        torch_loader=torch_loader,
        sampler=None,
        use_cuda=use_cuda,
        batch_size=batch_size,
    )


def reconstruct_loader(
    state: Dict,
    count_matrix: sp.csr_matrix,
    empty_matrix: sp.csr_matrix,
    use_cuda: bool,
    mmap_cache_dir: Optional[Path] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Reconstruct a DataLoader from a checkpoint state dict.

    Handles checkpoints written by both the new code (perm + gen_state) and
    old checkpoints (ind_list, no gen_state) for backward compatibility.

    Args:
        state: Checkpoint state dict from ``DataLoader.get_state()``.
        count_matrix: Full cell count matrix (used to build the backend).
        empty_matrix: Full empty-drop count matrix.
        use_cuda: Pin tensors for GPU transfer.
        mmap_cache_dir: If given, load backends from existing mmap files
            under this directory rather than materialising in-memory subsets.
    """
    cell_inds = state["original_cell_indices"]
    empty_inds = state["original_empty_indices"]

    cell_batch_size = int(state.get("cell_batch_size", state["batch_size"]))
    n_empty_per_batch = int(state.get("n_empty_per_batch", 0))
    shuffle = bool(state.get("shuffle", True))

    # Reconstruct empty count if only old-style batch_size + fraction_empties are stored.
    if "n_empty_per_batch" not in state and "fraction_empties" in state:
        fe = float(state["fraction_empties"])
        total_bs = int(state["batch_size"])
        cell_batch_size = int(total_bs * (1.0 - fe))
        n_empty_per_batch = total_bs - cell_batch_size

    gen = torch.Generator()
    # Default seed; will be overwritten by gen_state if present.
    gen.manual_seed(consts.RANDOM_SEED)

    if mmap_cache_dir is not None:
        cell_mmap = _get_or_create_mmap(count_matrix, mmap_cache_dir / "cells")
        empty_mmap = _get_or_create_mmap(empty_matrix, mmap_cache_dir / "empties")
        cell_backend: _SlicedBackend | InMemoryBackend = _SlicedBackend(cell_mmap, cell_inds)
        empty_backend: _SlicedBackend | InMemoryBackend | None = (
            _SlicedBackend(empty_mmap, empty_inds) if empty_inds.size > 0 else None
        )
    else:
        cell_backend = InMemoryBackend(count_matrix[cell_inds])
        empty_backend = InMemoryBackend(empty_matrix[empty_inds]) if empty_inds.size > 0 else None

    loader = _make_loader(
        cell_backend=cell_backend,
        empty_backend=empty_backend,
        cell_batch_size=cell_batch_size,
        n_empty_per_batch=n_empty_per_batch,
        generator=gen,
        shuffle=shuffle,
        use_cuda=use_cuda,
        original_cell_indices=cell_inds,
        original_empty_indices=empty_inds,
        num_workers=num_workers,
    )

    # Backward compat: old checkpoints store 'ind_list' instead of 'perm'.
    restore_state = dict(state)
    if "perm" not in restore_state and "ind_list" in restore_state:
        restore_state["perm"] = restore_state["ind_list"]

    loader.set_state(restore_state)
    return loader


# ──────────────────────────────────────────────────────────────────────────────
# Legacy utility (kept for callers outside the training path)
# ──────────────────────────────────────────────────────────────────────────────


def sparse_collate(batch: List[sp.csr_matrix]) -> torch.Tensor:
    """Stack a list of sparse CSR matrices into a dense float32 tensor."""
    import scipy.sparse as _sp

    mat = _sp.vstack(batch, format="csr")
    return torch.from_numpy(np.asarray(mat.todense(), dtype=np.float32))
