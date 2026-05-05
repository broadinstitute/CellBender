"""Helper functions for preparing dataloaders, as well as
a class to implement loading data from a sparse matrix in mini-batches.

Intentionally uses global random state, and does not keep its own random number
generator, in order to facilitate checkpointing.  The global random state is
only accessed from a single background prefetch thread during training (the main
thread does not call numpy.random while svi.step() is running), so there is no
concurrent access.
"""

import queue
import threading

import numpy as np
import scipy.sparse as sp

import cellbender.remove_background.consts as consts

import torch
import torch.utils.data

import logging
from typing import Dict, Tuple, List, Optional, Callable


logger = logging.getLogger('cellbender')

# Sentinel object that signals end-of-epoch through the prefetch queue.
_STOP = object()


class DataLoader:
    """Dataloader.

    This dataloader loads a specified fraction of cell barcodes + unknowns, and
    also mixes in a specified fraction of a random sampling of empty barcodes.

    """

    def __init__(self,
                 dataset: sp.csr_matrix,
                 empty_drop_dataset: Optional[sp.csr_matrix],
                 batch_size: int = consts.DEFAULT_BATCH_SIZE,
                 fraction_empties: float = consts.FRACTION_EMPTIES,
                 shuffle: bool = True,
                 sort_by: Optional[Callable[[sp.csr_matrix], np.ndarray]] = None,
                 use_cuda: bool = True,
                 original_cell_indices: Optional[np.ndarray] = None,
                 original_empty_indices: Optional[np.ndarray] = None):
        """
        Args:
            dataset: Droplet count matrix [cell, gene]
            empty_drop_dataset: Surely empty droplets count matrix
            batch_size: Number of droplets in minibatch
            fraction_empties: Fraction of each minibatch that will consist of
                surely empty droplets
            shuffle: True to shuffle data. Incompatible with sort_by.
            sort_by: Lambda function which, when applied to the sparse matrix,
                will return values that can be sorted to give a sort order to
                the dataset. Dataloader will load data in order of increasing
                values. Object attributes sort_order and unsort_order will be
                made available.
            use_cuda: True to load data to GPU
        """
        if shuffle:
            assert sort_by is None, 'Cannot sort_by and shuffle at the same time'
        self.sort_fn = sort_by
        self.dataset = dataset
        if self.sort_fn is not None:
            sort_values = self.sort_fn(self.dataset)
            sort_order = np.argsort(sort_values)
            self.ind_list = sort_order
            self.sort_order = sort_order.copy()
            self._unsort_dict = {i: val for i, val in enumerate(self.sort_order)}
        else:
            self.ind_list = np.arange(self.dataset.shape[0])
            self.sort_order = self.ind_list.copy()
            self._unsort_dict = {i: i for i in self.ind_list}
        self.empty_drop_dataset = empty_drop_dataset
        if self.empty_drop_dataset is None:
            self.empty_ind_list = np.array([])
        else:
            self.empty_ind_list = np.arange(self.empty_drop_dataset.shape[0])
        self.batch_size = batch_size
        self.fraction_empties = fraction_empties
        self.cell_batch_size = int(batch_size * (1. - fraction_empties))
        self.shuffle = shuffle
        self.original_cell_indices = original_cell_indices
        self.original_empty_indices = original_empty_indices
        self._device = 'cpu'
        self.use_cuda = use_cuda
        if self.use_cuda:
            self._device = 'cuda'
        self._length = None

        # Prefetch state — worker is started lazily on first __next__ call.
        self._result_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Initial shuffle and pointer reset (no worker started yet).
        if self.shuffle:
            np.random.shuffle(self.ind_list)
        self.ptr = 0

    # ------------------------------------------------------------------
    # Public device property
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        """Target device string ('cpu' or 'cuda').  Callers should move
        batches to this device with ``batch.to(loader.device, non_blocking=True)``."""
        return self._device

    # ------------------------------------------------------------------
    # Background prefetch worker
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        """Start the background prefetch thread from the current ptr."""
        self._stop_event.clear()
        self._result_queue = queue.Queue(maxsize=2)
        self._worker_thread = threading.Thread(
            target=self._worker_fn,
            args=(self.ind_list.copy(), self.ptr),
            daemon=True,
            name='cellbender-prefetch',
        )
        self._worker_thread.start()

    def _teardown_worker(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        if self._worker_thread is None:
            return
        self._stop_event.set()
        # The worker checks stop_event between put() retries (timeout=0.05 s),
        # so it will exit within one timeout interval after the event is set.
        self._worker_thread.join(timeout=2.0)
        self._worker_thread = None

    def _worker_put(self, item: object) -> None:
        """Put *item* into the result queue, respecting the stop event."""
        while not self._stop_event.is_set():
            try:
                self._result_queue.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def _worker_fn(self, ind_list: np.ndarray, ptr: int) -> None:
        """Background worker: densify batches and push them into result_queue.

        Uses the global numpy random state for empty-droplet sampling (same as
        the original synchronous code).  This is safe because the main thread
        does not call numpy.random while svi.step() is executing.
        """
        try:
            while not self._stop_event.is_set():
                remaining = ind_list.size - ptr
                if remaining < consts.SMALLEST_ALLOWED_BATCH:
                    if remaining > 0:
                        logger.debug(f'Dropped last minibatch of {remaining} cells')
                    self._worker_put(_STOP)
                    return

                next_ptr = min(ind_list.size, ptr + self.cell_batch_size)
                cell_inds = ind_list[ptr:next_ptr]

                n_empties = int(cell_inds.size *
                                (self.fraction_empties /
                                 (1 - self.fraction_empties)))
                if self.empty_ind_list.size > 0:
                    empty_inds = np.random.choice(self.empty_ind_list,
                                                  size=n_empties,
                                                  replace=True)
                    if empty_inds.size > 0:
                        csr_list = [self.dataset[cell_inds, :],
                                    self.empty_drop_dataset[empty_inds, :]]
                    else:
                        csr_list = [self.dataset[cell_inds, :]]
                else:
                    csr_list = [self.dataset[cell_inds, :]]

                dense = sparse_collate(csr_list)
                if self.use_cuda:
                    dense = dense.pin_memory()

                ptr = next_ptr
                self._worker_put((dense, ptr))

        except Exception as exc:
            self._worker_put(exc)

    # ------------------------------------------------------------------
    # Unsort helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def unsort_inds(self, bcs):
        if self.sort_fn is None:
            return bcs  # just for speed
        else:
            return torch.tensor([self._unsort_dict[bc.item()] for bc in bcs], device='cpu')

    # ------------------------------------------------------------------
    # Reset / state management
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Called at end-of-epoch: teardown worker, shuffle, reset ptr.

        The worker for the next epoch is NOT started here — it starts lazily on
        the first ``__next__()`` call.  This keeps the global numpy RNG state
        quiescent at epoch boundaries so checkpoint saves capture a clean state.
        """
        self._teardown_worker()
        if self.shuffle:
            np.random.shuffle(self.ind_list)
        self.ptr = 0

    def get_state(self) -> Dict:
        """Internal state of the data loader, used for checkpointing"""
        state: Dict = {
            'ind_list': self.ind_list,
            'ptr': self.ptr,
            'batch_size': self.batch_size,
            'fraction_empties': self.fraction_empties,
            'shuffle': self.shuffle,
            'use_cuda': self.use_cuda,
        }
        if self.original_cell_indices is not None:
            state['original_cell_indices'] = self.original_cell_indices
        if self.original_empty_indices is not None:
            state['original_empty_indices'] = self.original_empty_indices
        # Cache the length so _reconstruct_loader can restore it without iterating.
        if self._length is not None:
            state['_length'] = np.array(self._length, dtype=np.int64)
        return state

    def set_state(self, ind_list: np.ndarray, ptr: int) -> None:
        """Restore DataLoader position from a checkpoint; worker starts lazily."""
        self._teardown_worker()
        self.ind_list = ind_list
        self.ptr = ptr
        assert self.ptr <= len(self.ind_list), \
            f'Problem setting dataloader state: pointer ({ptr}) is outside the ' \
            f'length of the ind_list ({len(ind_list)})'

    def reset_ptr(self) -> None:
        """Reset the iteration pointer to zero; worker restarts lazily."""
        self._teardown_worker()
        self.ptr = 0

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    @property
    def length(self):
        if self._length is None:
            self._length = self._get_length()
        return self._length

    def _get_length(self):
        # avoid the potential for an off-by-one error by just going through it
        i = 0
        for _ in self:
            i += 1
        return i

    def __len__(self):
        return self.length

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self):
        # Lazily start the worker on the first call after construction,
        # set_state(), or reset_ptr() (avoids unnecessary RNG use at init time).
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._start_worker()

        item = self._result_queue.get()

        if item is _STOP:
            # End of epoch: prepare for the next one and signal the for-loop.
            self._reset()
            raise StopIteration()

        if isinstance(item, BaseException):
            raise item

        dense, ptr = item
        self.ptr = ptr
        # Return a CPU tensor (pinned when use_cuda=True).
        # Callers should move to device with: batch.to(loader.device, non_blocking=True)
        return dense

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop the prefetch worker.  Call when the loader will no longer be used."""
        self._teardown_worker()

    def __del__(self) -> None:
        try:
            self._teardown_worker()
        except Exception:
            pass


def prep_sparse_data_for_training(dataset: sp.csr_matrix,
                                  empty_drop_dataset: sp.csr_matrix,
                                  training_fraction: float = consts.TRAINING_FRACTION,
                                  fraction_empties: float = consts.FRACTION_EMPTIES,
                                  batch_size: int = consts.DEFAULT_BATCH_SIZE,
                                  shuffle: bool = True,
                                  use_cuda: bool = True) -> Tuple[
                                      "DataLoader",
                                      "DataLoader"]:
    """Create DataLoaders for train and test sets.

    The dataset is kept in memory as a sparse matrix and densified batch-by-batch
    in a background prefetch thread.  Each DataLoader returns CPU tensors (pinned
    when use_cuda=True); callers should move batches to the target device with
    ``batch.to(loader.device, non_blocking=True)``.

    Args:
        dataset: Matrix of gene counts, where rows are cell barcodes and
            columns are genes.
        empty_drop_dataset: Matrix of gene counts, where rows are surely-empty
            droplet barcodes and columns are genes.
        training_fraction: Fraction of data to use as the training set.  The
            rest becomes the test set.
        fraction_empties: Fraction of each minibatch to be composed of empty
            droplets.
        batch_size: Number of cell barcodes per mini-batch of data.
        shuffle: Passed as an argument to torch.utils.data.DataLoader.  If
            True, the data is reshuffled at every epoch.
        use_cuda: If True, the data loader will load tensors on GPU.

    Returns:
        train_loader: torch.utils.data.DataLoader object for training set.
        test_loader: torch.utils.data.DataLoader object for tests set.

    Examples:
        train_loader, test_loader = prep_sparse_data_for_training(dataset,
                                        training_fraction=0.9,
                                        batch_size=128, shuffle=True)

    """

    # Choose train and test indices from analysis dataset.
    training_mask = np.random.rand(dataset.shape[0]) < training_fraction
    training_indices = [idx for idx in range(dataset.shape[0])
                        if training_mask[idx]]
    test_indices = [idx for idx in range(dataset.shape[0])
                    if not training_mask[idx]]

    # Choose train and test indices from empty drop dataset.
    training_mask_empty = (np.random.rand(empty_drop_dataset.shape[0]) < training_fraction)
    training_indices_empty = [idx for idx in range(empty_drop_dataset.shape[0])
                              if training_mask_empty[idx]]
    test_indices_empty = [idx for idx in range(empty_drop_dataset.shape[0])
                          if not training_mask_empty[idx]]

    # Set up training dataloader.
    train_dataset = dataset[training_indices, ...]
    train_dataset_empty = empty_drop_dataset[training_indices_empty, ...]
    train_loader = DataLoader(dataset=train_dataset,
                              empty_drop_dataset=train_dataset_empty,
                              batch_size=batch_size,
                              fraction_empties=fraction_empties,
                              shuffle=shuffle,
                              use_cuda=use_cuda,
                              original_cell_indices=np.array(training_indices),
                              original_empty_indices=np.array(training_indices_empty))

    # Set up test dataloader.
    test_dataset = dataset[test_indices, ...]
    test_dataset_empty = empty_drop_dataset[test_indices_empty, ...]
    test_loader = DataLoader(dataset=test_dataset,
                             empty_drop_dataset=test_dataset_empty,
                             batch_size=batch_size,
                             fraction_empties=fraction_empties,
                             shuffle=shuffle,
                             use_cuda=use_cuda,
                             original_cell_indices=np.array(test_indices),
                             original_empty_indices=np.array(test_indices_empty))

    return train_loader, test_loader


def sparse_collate(batch: List[Tuple[sp.csr_matrix]]) -> torch.Tensor:
    """Load a minibatch of sparse data as a dense torch.Tensor in memory.

    Puts each data field into a tensor with leftmost dimension batch size.
    'batch' is a python list of items from the dataset.
    For a scipy.sparse.csr matrix, this is rows of the matrix, but in python
    list form.

    """
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html
    # default_collate()

    # Stack the list of csr matrices.
    mat = sp.vstack(batch, format='csr')
    # Output a dense torch.Tensor wrapped in a tuple.
    # This is fastest if converted in-place using torch.from_numpy().
    a = np.array(mat.todense(), dtype=np.float32)
    return torch.from_numpy(a)
