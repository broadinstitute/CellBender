"""Helper functions for preparing dataloaders, as well as
a class to implement loading data from a sparse matrix in mini-batches.
"""

import numpy as np
import scipy.sparse as sp

import cellbender.remove_background.consts as consts

import torch
import torch.utils.data

from typing import Tuple, List, Optional


class SparseDataset(torch.utils.data.Dataset):
    """torch.utils.data.Dataset wrapping a scipy.sparse.csr.csr_matrix

    Each sample will be retrieved by indexing matrices along the leftmost
    dimension.

    Args:
        *csrs (scipy.sparse.csr.csr_matrix): sparse matrices that have the same
        size in the leftmost dimension.

    """
    # see https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html

    def __init__(self, *csrs):
        assert all(csrs[0].shape[0] == csr.shape[0] for csr in csrs)
        self.csrs = csrs

    def __getitem__(self, index) -> Tuple:
        return tuple(csr[index, ...] for csr in self.csrs)

    def __len__(self) -> int:
        return self.csrs[0].shape[0]


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
                 use_cuda: bool = True):
        self.dataset = dataset
        self.ind_list = np.arange(self.dataset.shape[0])
        self.empty_drop_dataset = empty_drop_dataset
        if self.empty_drop_dataset is None:
            self.empty_ind_list = np.array([])
        else:
            self.empty_ind_list = np.arange(self.empty_drop_dataset.shape[0])
        self.batch_size = batch_size
        self.fraction_empties = fraction_empties
        self.cell_batch_size = int(batch_size * (1. - fraction_empties))
        self.shuffle = shuffle
        self.random = np.random.RandomState(seed=1234)
        self.device = 'cpu'
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.device = 'cuda'
        self._reset()

    def _reset(self):
        if self.shuffle:
            self.random.shuffle(self.ind_list)  # Shuffle cell inds in place
        self.ptr = 0

    def __len__(self):
        return int(self.ind_list.size *
                   (1 + (self.fraction_empties / (1 - self.fraction_empties))))  # ...ish

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr == self.ind_list.size:
            self._reset()
            raise StopIteration()

        else:

            # Move the pointer by the number of cells in this minibatch.
            next_ptr = min(self.ind_list.size, self.ptr + self.cell_batch_size)

            # Decide on cell (+ transition region) indices.
            cell_inds = self.ind_list[self.ptr:next_ptr]

            # Decide on empty droplet indices. (Number changes at end of epoch.)
            n_empties = int(cell_inds.size *
                            (self.fraction_empties /
                             (1 - self.fraction_empties)))
            if self.empty_ind_list.size > 0:
                # This does not happen for 'simple' model.
                empty_inds = self.random.choice(self.empty_ind_list,
                                                size=n_empties,
                                                replace=True)

                if empty_inds.size > 0:
                    csr_list = [self.dataset[cell_inds, :],
                                self.empty_drop_dataset[empty_inds, :]]
                else:
                    csr_list = [self.dataset[cell_inds, :]]
            else:
                csr_list = [self.dataset[cell_inds, :]]

            # Get a dense tensor from the sparse matrix.
            dense_tensor = sparse_collate(csr_list)

            # Increment the pointer and return the minibatch.
            self.ptr = next_ptr
            return dense_tensor.to(device=self.device)


def prep_sparse_data_for_training(dataset: sp.csr.csr_matrix,
                                  empty_drop_dataset: sp.csr.csr_matrix,
                                  random_state: np.random.RandomState,
                                  training_fraction: float = consts.TRAINING_FRACTION,
                                  fraction_empties: float = consts.FRACTION_EMPTIES,
                                  batch_size: int = consts.DEFAULT_BATCH_SIZE,
                                  shuffle: bool = True,
                                  use_cuda: bool = True) -> Tuple[
                                      torch.utils.data.DataLoader,
                                      torch.utils.data.DataLoader]:
    """Create torch.utils.data.DataLoaders for train and tests set.

    The dataset is not loaded into memory as a dense matrix upfront.  Instead
    of using a torch.utils.data.TensorDataset, a SparseDataset is used, which
    only transforms a sparse matrix to a dense one when a minibatch is loaded.
    This is slower, but necessary for datasets which are too large to be
    loaded into memory as a dense matrix all at once.

    Args:
        dataset: Matrix of gene counts, where rows are cell barcodes and
            columns are genes.
        empty_drop_dataset: Matrix of gene counts, where rows are surely-empty
            droplet barcodes and columns are genes.
        random_state: numpy.random.RandomState from the Dataset object, for
            deterministic outputs.
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
    training_mask = random_state.rand(dataset.shape[0]) < training_fraction
    training_indices = [idx for idx in range(dataset.shape[0])
                        if training_mask[idx]]
    test_indices = [idx for idx in range(dataset.shape[0])
                    if not training_mask[idx]]

    # Choose train and test indices from empty drop dataset.
    training_mask_empty = (random_state.rand(empty_drop_dataset.shape[0])
                           < training_fraction)
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
                              use_cuda=use_cuda)

    # Set up test dataloader.
    test_dataset = dataset[test_indices, ...]
    test_dataset_empty = empty_drop_dataset[test_indices_empty, ...]
    test_loader = DataLoader(dataset=test_dataset,
                             empty_drop_dataset=test_dataset_empty,
                             batch_size=batch_size,
                             fraction_empties=fraction_empties,
                             shuffle=shuffle,
                             use_cuda=use_cuda)

    return train_loader, test_loader


def sparse_collate(batch: List[Tuple[sp.csr.csr_matrix]]) -> torch.Tensor:
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
