"""Classes and methods for estimation of noise counts, given a posterior."""

import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
from torch.distributions.categorical import Categorical

from cellbender.remove_background.sparse_utils import log_prob_sparse_to_dense

from abc import ABC, abstractmethod
from functools import partial
from itertools import repeat
import multiprocessing as mp
import concurrent.futures
import time
from datetime import datetime
import logging
from typing import Callable, Union, Dict, Generator, Tuple, List, Optional


logger = logging.getLogger('cellbender')

N_CELLS_DATATYPE = np.int32
N_GENES_DATATYPE = np.int32
COUNT_DATATYPE = np.int32


class EstimationMethod(ABC):
    """Base class for estimation of noise counts, given a posterior."""

    def __init__(self, index_converter: 'IndexConverter'):
        """Instantiate the EstimationMethod.
        Args:
            index_converter: The IndexConverter that can be used to translate
                back and forth between count matrix (n, g) indices, and the
                flattened, generalized 'm' indices that index the posterior COO.
        """
        self.index_converter = index_converter
        super(EstimationMethod, self).__init__()

    @abstractmethod
    def estimate_noise(self,
                       noise_log_prob_coo: sp.coo_matrix,
                       noise_offsets: Dict[int, int],
                       **kwargs) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts.
        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_offsets: Noise count offset values keyed by 'm'.
        """
        pass

    def _estimation_array_to_csr(self,
                                 data: np.ndarray,
                                 m: np.ndarray,
                                 noise_offsets: Optional[Dict[int, int]],
                                 dtype=COUNT_DATATYPE) -> sp.csr_matrix:
        """Say you have point estimates for each count matrix element (data) and
        you have the 'm'-indices for each value (m). This returns a CSR matrix
        that has the shape of the count matrix, where duplicate entries have
        been summed.

        Args:
            data: Point estimates for each nonzero entry of the count matrix, in
                a flat format, indexed by 'm'.
            m: Array of the same length as data, where each entry is an m-index.
            noise_offsets: Noise count offset values keyed by 'm'.
            dtype: Data type for sparse matrix. Int32 is too small for 'm' indices.

        Results:
            noise_csr: Noise point estimate, as a CSR sparse matrix.

        """
        return _estimation_array_to_csr(
            index_converter=self.index_converter,
            data=data,
            m=m,
            noise_offsets=noise_offsets,
            dtype=dtype,
        )
        # row, col = self.index_converter.get_ng_indices(m_inds=m)
        # if noise_offsets is not None:
        #     data = data + np.array([noise_offsets[i] for i in m])
        # coo = sp.coo_matrix((data.astype(dtype), (row.astype(dtype), col.astype(dtype))),
        #                     shape=self.index_converter.matrix_shape, dtype=dtype)
        # coo.sum_duplicates()
        # return coo.tocsr()


class SingleSample(EstimationMethod):
    """A single sample from the noise count posterior"""

    @torch.no_grad()
    def estimate_noise(self,
                       noise_log_prob_coo: sp.coo_matrix,
                       noise_offsets: Dict[int, int],
                       device: str = 'cpu',
                       **kwargs) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts by
        taking a single sample from each probability distribution.

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_offsets: Noise count offset values keyed by 'm'.
            device: ['cpu', 'cuda'] - whether to perform the pytorch sampling
                operation on CPU or GPU. It's pretty fast on CPU already.

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """
        def _torch_sample(x):
            return Categorical(logits=x, validate_args=False).sample()

        result = apply_function_dense_chunks(noise_log_prob_coo=noise_log_prob_coo,
                                             fun=_torch_sample,
                                             device=device)
        return self._estimation_array_to_csr(data=result['result'], m=result['m'],
                                             noise_offsets=noise_offsets)


class Mean(EstimationMethod):
    """Posterior mean"""

    def estimate_noise(self,
                       noise_log_prob_coo: sp.coo_matrix,
                       noise_offsets: Dict[int, int],
                       device: str = 'cpu',
                       **kwargs) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts by
        taking the mean of each probability distribution.

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_offsets: Noise count offset values keyed by 'm'.

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """
        # c = torch.arange(noise_log_prob_coo.shape[1], dtype=float).to(device).t()

        def _torch_mean(x):
            c = torch.arange(x.shape[1], dtype=float).to(x.device)
            return torch.matmul(x.exp(), c.t())

        result = apply_function_dense_chunks(noise_log_prob_coo=noise_log_prob_coo,
                                             fun=_torch_mean,
                                             device=device)
        return self._estimation_array_to_csr(data=result['result'], m=result['m'],
                                             noise_offsets=noise_offsets,
                                             dtype=np.float32)


class MAP(EstimationMethod):
    """The canonical maximum a posteriori"""

    @staticmethod
    def torch_argmax(x):
        return x.argmax(dim=-1)

    def estimate_noise(self,
                       noise_log_prob_coo: sp.coo_matrix,
                       noise_offsets: Dict[int, int],
                       device: str = 'cpu',
                       **kwargs) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts by
        taking the maximum a posteriori (MAP) of each probability distribution.

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_offsets: Noise count offset values keyed by 'm'.
            device: ['cpu', 'cuda'] - whether to perform the pytorch argmax
                operation on CPU or GPU. It's pretty fast on CPU already.

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """
        result = apply_function_dense_chunks(noise_log_prob_coo=noise_log_prob_coo,
                                             fun=self.torch_argmax,
                                             device=device)
        return self._estimation_array_to_csr(data=result['result'], m=result['m'],
                                             noise_offsets=noise_offsets)


class ThresholdCDF(EstimationMethod):
    """Noise estimation via thresholding the noise count CDF"""

    @staticmethod
    def torch_cdf_fun(x: torch.Tensor, q: float):
        return (x.exp().cumsum(dim=-1) <= q).sum(dim=-1)

    def estimate_noise(self,
                       noise_log_prob_coo: sp.coo_matrix,
                       noise_offsets: Dict[int, int],
                       q: float = 0.5,
                       device: str = 'cpu',
                       **kwargs) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_offsets: Noise count offset values keyed by 'm'.
            q: The CDF threshold value.

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """
        result = apply_function_dense_chunks(noise_log_prob_coo=noise_log_prob_coo,
                                             fun=self.torch_cdf_fun,
                                             device=device,
                                             q=q)
        return self._estimation_array_to_csr(data=result['result'], m=result['m'],
                                             noise_offsets=noise_offsets)


def _estimation_array_to_csr(index_converter,
                             data: np.ndarray,
                             m: np.ndarray,
                             noise_offsets: Optional[Dict[int, int]],
                             dtype=COUNT_DATATYPE) -> sp.csr_matrix:
    """Say you have point estimates for each count matrix element (data) and
    you have the 'm'-indices for each value (m). This returns a CSR matrix
    that has the shape of the count matrix, where duplicate entries have
    been summed.

    Args:
        data: Point estimates for each nonzero entry of the count matrix, in
            a flat format, indexed by 'm'.
        m: Array of the same length as data, where each entry is an m-index.
        noise_offsets: Noise count offset values keyed by 'm'.
        dtype: Data type for values of sparse matrix

    Results:
        noise_csr: Noise point estimate, as a CSR sparse matrix.

    """
    row, col = index_converter.get_ng_indices(m_inds=m)
    if noise_offsets is not None:
        data = data + np.array([noise_offsets.get(i, 0) for i in m])
    coo = sp.coo_matrix((data.astype(dtype), (row.astype(N_CELLS_DATATYPE), col.astype(N_GENES_DATATYPE))),
                        shape=index_converter.matrix_shape, dtype=dtype)
    coo.sum_duplicates()
    return coo.tocsr()


def _mckp_chunk_estimate_noise(
        noise_log_prob_coo: sp.coo_matrix,
        index_and_logic: Tuple[int, np.ndarray],
        noise_offsets: Dict[int, int],
        noise_targets_per_gene: np.ndarray,
        index_converter: 'IndexConverter',
        n_chunks: int,
        verbose: bool = False) -> sp.csr_matrix:
    """Given the full probabilistic posterior, compute noise counts. This is
    to be run for a given chunk of genes at a time.

    Args:
        noise_log_prob_coo: The noise log prob data structure: log prob
            values in a (m, c) COO matrix.  One chunk.
        index_and_logic: (chunk_index, logical_coo_indexer) from the chunked
            iterator, as you would get from enumerate()
        noise_targets_per_gene: Integer noise count target for each gene
        noise_offsets: Noise count offset values keyed by 'm'.
        index_converter: IndexConverter to go from 'm' to (n, g)
        n_chunks: Total chunks, for logging purposes only
        verbose: True to print lots of intermediate information (for tests)

    Returns:
        noise_count_csr: Estimated noise count matrix.
    """

    i = index_and_logic[0]

    if i == 0:
        tt = time.time()

    coo = _subset_coo(noise_log_prob_coo, index_and_logic[1])

    assert noise_targets_per_gene.size == index_converter.total_n_genes, \
        f'The number of noise count targets ({noise_targets_per_gene.size}) ' \
        f'must match the number of genes ({index_converter.total_n_genes})'

    # First we need to compute the MAP to find out which direction to go.
    t = time.time()
    map_dict = apply_function_dense_chunks(
        noise_log_prob_coo=coo,
        fun=MAP.torch_argmax,
        device='cpu',
    )
    map_csr = _estimation_array_to_csr(
        data=map_dict['result'],
        m=map_dict['m'],
        noise_offsets=noise_offsets,
        index_converter=index_converter,
    )
    logger.debug(f'{timestamp()} Computed initial MAP estimate')
    logger.debug(f'{timestamp()} Time for MAP calculation = {(time.time() - t):.2f} sec')
    map_noise_counts_per_gene = np.array(map_csr.sum(axis=0)).squeeze()
    additional_noise_counts_per_gene = (noise_targets_per_gene
                                        - map_noise_counts_per_gene).astype(int)
    set_positive_genes = set(np.where(additional_noise_counts_per_gene > 0)[0])
    set_negative_genes = set(np.where(additional_noise_counts_per_gene < 0)[0])  # leave out exact matches
    abs_additional_noise_counts_per_gene = np.abs(additional_noise_counts_per_gene)

    # Determine which genes need to add and which need to subtract noise counts.
    n, g = index_converter.get_ng_indices(m_inds=coo.row)
    df = pd.DataFrame(data={'m': coo.row,
                            'n': n,
                            'g': g,
                            'c': coo.col,
                            'log_prob': coo.data})
    logger.debug(f'{timestamp()} Computing step directions')
    df['positive_step_gene'] = df['g'].apply(lambda gene: gene in set_positive_genes)
    df['negative_step_gene'] = df['g'].apply(lambda gene: gene in set_negative_genes)
    df['step_direction'] = (df['positive_step_gene'].astype(int)
                            - df['negative_step_gene'].astype(int))  # -1 or 1
    logger.debug(f'{timestamp()} Step directions done')

    if verbose:
        pd.set_option('display.width', 120)
        pd.set_option('display.max_columns', 20)
        print(df, end='\n\n')

    # Remove all 'm' entries corresponding to genes where target is met by MAP.
    df = df[df['step_direction'] != 0]

    # Now we mask out log probs (-np.inf) that represent steps in the wrong direction.
    logger.debug(f'{timestamp()} Masking')
    lookup_map_from_m = dict(zip(map_dict['m'], map_dict['result']))
    df['map'] = df['m'].apply(lambda x: lookup_map_from_m[x])
    df['mask'] = ((df['negative_step_gene'] & (df['c'] > df['map']))
                  | (df['positive_step_gene'] & (df['c'] < df['map'])))  # keep MAP
    df.loc[df['mask'], 'log_prob'] = -np.inf
    logger.debug(f'{timestamp()} Masking done')

    # And we remove those entries.
    df = df[~df['mask']]
    df = df[[c for c in df.columns if (c != 'mask')]]

    # Sort
    logger.debug(f'{timestamp()} Sorting')
    df = df.sort_values(by=['m', 'c'])
    logger.debug(f'{timestamp()} Sorting done')

    if verbose:
        print(df, end='\n\n')

    # Do diff for positive and negative separately, without grouping (faster)
    df_positive_steps = df[df['step_direction'] == 1].copy()
    df_negative_steps = df[df['step_direction'] == -1].copy()

    logger.debug(f'{timestamp()} Computing deltas')
    if len(df_positive_steps > 0):
        df_positive_steps.loc[:, 'delta'] = df_positive_steps['log_prob'].diff(periods=1).apply(np.abs)
        df_positive_steps.loc[df_positive_steps['c'] == df_positive_steps['map'], 'delta'] = np.nan
    if len(df_negative_steps > 0):
        df_negative_steps.loc[:, 'delta'] = df_negative_steps['log_prob'].diff(periods=-1).apply(np.abs)
        df_negative_steps.loc[df_negative_steps['c'] == df_negative_steps['map'], 'delta'] = np.nan
    df = pd.concat([df_positive_steps, df_negative_steps], axis=0)
    logger.debug(f'{timestamp()} Computing deltas done')

    if verbose:
        print(df, end='\n\n')

    # if this is an empty dataframe, we are not doing anything here beyond MAP
    if len(df) == 0:
        return map_csr

    # Remove irrelevant entries: those with infinite delta.
    df = df[df['delta'].apply(np.isfinite)]

    # if this is an empty dataframe, we are not doing anything here beyond MAP
    if len(df) == 0:
        return map_csr

    # How many additional noise counts ("steps") we will need for each gene.
    logger.debug(f'{timestamp()} Computing nsmallest')
    df['topk'] = df['g'].apply(lambda gene: abs_additional_noise_counts_per_gene[gene])

    if verbose:
        print(df, end='\n\n')

    # Now we want the smallest additional_noise_counts_per_gene deltas for each gene.
    # https://stackoverflow.com/questions/55179493/
    df_out = df[['m', 'g', 'delta', 'topk']].groupby('g', group_keys=False).apply(
        lambda x: x.nsmallest(x['topk'].iat[0], columns='delta')
    )
    logger.debug(f'{timestamp()} Computing nsmallest done')

    if verbose:
        print(df_out, end='\n\n')

    # if this is an empty dataframe, we are not doing anything here beyond MAP
    if len(df_out) == 0:
        return map_csr

    # And the number by which to increment noise counts per entry 'm' is
    # now the number of times that each m value appears in this dataframe.
    logger.debug(f'{timestamp()} Summarizing steps')
    vc = df_out['m'].value_counts()
    vc_df = pd.DataFrame(data={'m': vc.index, 'steps': vc.values})
    step_direction_lookup_from_m = dict(zip(df['m'], df['step_direction']))
    vc_df['step_direction'] = vc_df['m'].apply(lambda x: step_direction_lookup_from_m[x])
    vc_df['counts'] = vc_df['steps'] * vc_df['step_direction']
    steps_csr = _estimation_array_to_csr(
        data=vc_df['counts'],
        m=vc_df['m'],
        noise_offsets=None,
        index_converter=index_converter,
    )
    logger.debug(f'{timestamp()} Summarizing steps done')

    if verbose:
        print(vc_df, end='\n\n')
        print('MAP:')
        print(map_csr.todense())

    logger.info(f'Completed chunk ({i + 1} / {n_chunks})')
    print(f'Completed chunk ({i + 1} / {n_chunks})')  # because logging from a process does not work right
    if i == 0:
        logger.info(f'    [single chunk took {(time.time() - tt):.2f} mins]')
        print(f'    [single chunk took {(time.time() - tt):.2f} mins]')

    # The final output is those tabulated steps added to the MAP.
    # The MAP already has the noise offsets, so they are not added to steps_csr.
    return map_csr + steps_csr


class MultipleChoiceKnapsack(EstimationMethod):
    """Noise estimation via solving a constrained multiple choice knapsack problem"""

    def estimate_noise(self,
                       noise_log_prob_coo: sp.coo_matrix,
                       noise_offsets: Dict[int, int],
                       noise_targets_per_gene: np.ndarray,
                       verbose: bool = False,
                       n_chunks: Optional[int] = None,
                       use_multiple_processes: bool = False,
                       **kwargs) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_targets_per_gene: Integer noise count target for each gene
            noise_offsets: Noise count offset values keyed by 'm'.
            verbose: True to print lots of intermediate information (for tests)
            n_chunks: Target number of chunks over which to split estimation.
                If None, targets about 5000 genes per chunk.
            use_multiple_processes: True to use multiprocessing. Seems faster
                without using it, not entirely clear why

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """
        if n_chunks is None:
            n_chunks = max(1, self.index_converter.total_n_genes // 5000)
            logger.debug(f'Running MCKP estimator in {n_chunks} chunks')

        t0 = time.time()

        if use_multiple_processes:

            logger.info('Dividing dataset into chunks of genes')
            chunk_logic_list = self._gene_chunk_iterator(
                noise_log_prob_coo=noise_log_prob_coo,
                n_chunks=n_chunks,
            )

            logger.info('Computing the output in asynchronous chunks in parallel...')

            # with mp.get_context('spawn').Pool(processes=mp.cpu_count()) as pool:
            #     csr_matrices = pool.starmap(
            #         _mckp_chunk_estimate_noise,
            #         zip(
            #             repeat(noise_log_prob_coo),
            #             enumerate(chunk_logic_list),
            #             repeat(noise_offsets),
            #             repeat(noise_targets_per_gene),
            #             repeat(self.index_converter),
            #             repeat(n_chunks),
            #             repeat(False),  # verbose
            #         ),
            #     )

            futures = []
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=mp.cpu_count(),
                    mp_context=mp.get_context('spawn')) as executor:
                for i, logic in enumerate(chunk_logic_list):
                    kwargs = {
                        'noise_log_prob_coo': noise_log_prob_coo,
                        'index_and_logic': (i, logic),
                        'noise_offsets': noise_offsets,
                        'noise_targets_per_gene': noise_targets_per_gene,
                        'index_converter': self.index_converter,
                        'n_chunks': n_chunks,
                        'verbose': False,
                    }
                    future = executor.submit(_mckp_chunk_estimate_noise, **kwargs)
                    futures.append(future)

                done, not_done = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.ALL_COMPLETED,
                )
                csr_matrices = [f.result() for f in futures]

        else:

            t = time.time()

            csr_matrices = []
            for i, logic in enumerate(
                    self._gene_chunk_iterator(
                        noise_log_prob_coo=noise_log_prob_coo,
                        n_chunks=n_chunks,
                    )
            ):
                logger.info(f'Working on chunk ({i + 1}/{n_chunks})')
                chunk_csr = self._chunk_estimate_noise(
                    noise_log_prob_coo=_subset_coo(noise_log_prob_coo, logic),
                    noise_offsets=noise_offsets,
                    noise_targets_per_gene=noise_targets_per_gene,
                    verbose=verbose,
                )
                csr_matrices.append(chunk_csr)
                if i == 0:
                    logger.info(f'    [{(time.time() - t) / 60:.2f} mins per chunk]')
                logger.debug(f'{timestamp()} Estimator chunk {i}: shape is {chunk_csr.shape}')

        logger.info(f'{timestamp()} Total MCKP estimation time = {(time.time() - t0):.2f} sec')
        return sum(csr_matrices)

    def _gene_chunk_iterator(self,
                             noise_log_prob_coo: sp.coo_matrix,
                             n_chunks: int) \
            -> List[np.ndarray]:
        """Return a list of logical (size m) arrays used to select gene chunks
        on which to compute the MCKP estimate. These chunks are independent.

        Args:
            noise_log_prob_coo: Full noise log prob posterior COO
            n_chunks: For testing, force this many chunks

        Yields:
            Logical array which indexes elements of coo posterior for the chunk
        """

        # get gene annotations
        _, genes = self.index_converter.get_ng_indices(m_inds=noise_log_prob_coo.row)
        genes_series = pd.Series(genes)

        gene_chunk_arrays = np.array_split(np.arange(self.index_converter.total_n_genes), n_chunks)

        gene_logic_arrays = [genes_series.isin(x).values for x in gene_chunk_arrays]
        return gene_logic_arrays

    def _chunk_estimate_noise(self,
                              noise_log_prob_coo: sp.coo_matrix,
                              noise_offsets: Dict[int, int],
                              noise_targets_per_gene: np.ndarray,
                              verbose: bool = False) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts. This is
        to be run for a given chunk of genes at a time.

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix.  One chunk.
            noise_targets_per_gene: Integer noise count target for each gene
            noise_offsets: Noise count offset values keyed by 'm'.
            verbose: True to print lots of intermediate information (for tests)

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """

        assert noise_targets_per_gene.size == self.index_converter.total_n_genes, \
            f'The number of noise count targets ({noise_targets_per_gene.size}) ' \
            f'must match the number of genes ({self.index_converter.total_n_genes})'

        coo = noise_log_prob_coo.copy()  # we will be modifying values

        # First we need to compute the MAP to find out which direction to go.
        t = time.time()
        map_dict = apply_function_dense_chunks(noise_log_prob_coo=noise_log_prob_coo,
                                               fun=MAP.torch_argmax,
                                               device='cpu')
        map_csr = self._estimation_array_to_csr(data=map_dict['result'],
                                                m=map_dict['m'],
                                                noise_offsets=noise_offsets)
        logger.debug(f'{timestamp()} Computed initial MAP estimate')
        logger.debug(f'{timestamp()} Time for MAP calculation = {(time.time() - t):.2f} sec')
        map_noise_counts_per_gene = np.array(map_csr.sum(axis=0)).squeeze()
        additional_noise_counts_per_gene = (noise_targets_per_gene
                                            - map_noise_counts_per_gene).astype(int)
        set_positive_genes = set(np.where(additional_noise_counts_per_gene > 0)[0])
        set_negative_genes = set(np.where(additional_noise_counts_per_gene < 0)[0])  # leave out exact matches
        abs_additional_noise_counts_per_gene = np.abs(additional_noise_counts_per_gene)

        # Determine which genes need to add and which need to subtract noise counts.
        n, g = self.index_converter.get_ng_indices(m_inds=coo.row)
        df = pd.DataFrame(data={'m': coo.row,
                                'n': n,
                                'g': g,
                                'c': coo.col,
                                'log_prob': coo.data})
        logger.debug(f'{timestamp()} Computing step directions')
        df['positive_step_gene'] = df['g'].apply(lambda gene: gene in set_positive_genes)
        df['negative_step_gene'] = df['g'].apply(lambda gene: gene in set_negative_genes)
        df['step_direction'] = (df['positive_step_gene'].astype(int)
                                - df['negative_step_gene'].astype(int))  # -1 or 1
        logger.debug(f'{timestamp()} Step directions done')

        if verbose:
            pd.set_option('display.width', 120)
            pd.set_option('display.max_columns', 20)
            print(df, end='\n\n')

        # Remove all 'm' entries corresponding to genes where target is met by MAP.
        df = df[df['step_direction'] != 0]

        # Now we mask out log probs (-np.inf) that represent steps in the wrong direction.
        logger.debug(f'{timestamp()} Masking')
        lookup_map_from_m = dict(zip(map_dict['m'], map_dict['result']))
        df['map'] = df['m'].apply(lambda x: lookup_map_from_m[x])
        df['mask'] = ((df['negative_step_gene'] & (df['c'] > df['map']))
                      | (df['positive_step_gene'] & (df['c'] < df['map'])))  # keep MAP
        df.loc[df['mask'], 'log_prob'] = -np.inf
        logger.debug(f'{timestamp()} Masking done')

        # And we remove those entries.
        df = df[~df['mask']]
        df = df[[c for c in df.columns if (c != 'mask')]]

        # Sort
        logger.debug(f'{timestamp()} Sorting')
        df = df.sort_values(by=['m', 'c'])
        logger.debug(f'{timestamp()} Sorting done')

        if verbose:
            print(df, end='\n\n')

        # Do diff for positive and negative separately, without grouping (faster)
        df_positive_steps = df[df['step_direction'] == 1].copy()
        df_negative_steps = df[df['step_direction'] == -1].copy()

        logger.debug(f'{timestamp()} Computing deltas')
        if len(df_positive_steps > 0):
            df_positive_steps.loc[:, 'delta'] = df_positive_steps['log_prob'].diff(periods=1).apply(np.abs)
            df_positive_steps.loc[df_positive_steps['c'] == df_positive_steps['map'], 'delta'] = np.nan
        if len(df_negative_steps > 0):
            df_negative_steps.loc[:, 'delta'] = df_negative_steps['log_prob'].diff(periods=-1).apply(np.abs)
            df_negative_steps.loc[df_negative_steps['c'] == df_negative_steps['map'], 'delta'] = np.nan
        df = pd.concat([df_positive_steps, df_negative_steps], axis=0)
        logger.debug(f'{timestamp()} Computing deltas done')

        if verbose:
            print(df, end='\n\n')

        # if this is an empty dataframe, we are not doing anything here beyond MAP
        if len(df) == 0:
            return map_csr

        # Remove irrelevant entries: those with infinite delta.
        df = df[df['delta'].apply(np.isfinite)]

        # if this is an empty dataframe, we are not doing anything here beyond MAP
        if len(df) == 0:
            return map_csr

        # How many additional noise counts ("steps") we will need for each gene.
        logger.debug(f'{timestamp()} Computing nsmallest')
        df['topk'] = df['g'].apply(lambda gene: abs_additional_noise_counts_per_gene[gene])

        if verbose:
            print(df, end='\n\n')

        # Now we want the smallest additional_noise_counts_per_gene deltas for each gene.
        # https://stackoverflow.com/questions/55179493/
        df_out = df[['m', 'g', 'delta', 'topk']].groupby('g', group_keys=False).apply(
            lambda x: x.nsmallest(x['topk'].iat[0], columns='delta')
        )
        logger.debug(f'{timestamp()} Computing nsmallest done')

        if verbose:
            print(df_out, end='\n\n')

        # if this is an empty dataframe, we are not doing anything here beyond MAP
        if len(df_out) == 0:
            return map_csr

        # And the number by which to increment noise counts per entry 'm' is
        # now the number of times that each m value appears in this dataframe.
        logger.debug(f'{timestamp()} Summarizing steps')
        vc = df_out['m'].value_counts()
        vc_df = pd.DataFrame(data={'m': vc.index, 'steps': vc.values})
        step_direction_lookup_from_m = dict(zip(df['m'], df['step_direction']))
        vc_df['step_direction'] = vc_df['m'].apply(lambda x: step_direction_lookup_from_m[x])
        vc_df['counts'] = vc_df['steps'] * vc_df['step_direction']
        steps_csr = self._estimation_array_to_csr(data=vc_df['counts'],
                                                  m=vc_df['m'],
                                                  noise_offsets=None)
        logger.debug(f'{timestamp()} Summarizing steps done')

        if verbose:
            print(vc_df, end='\n\n')
            print('MAP:')
            print(map_csr.todense())

        # The final output is those tabulated steps added to the MAP.
        # The MAP already has the noise offsets, so they are not added to steps_csr.
        return map_csr + steps_csr


def chunked_iterator(coo: sp.coo_matrix,
                     max_dense_batch_size_GB: float = 1.) \
        -> Generator[Tuple[sp.coo_matrix, np.ndarray], None, None]:
    """Return an iterator which yields the full dataset in chunks.

    NOTE: Idea is to prevent memory overflow. The use case is for worst-case
    scenario algorithms that have to make things into dense matrix chunks in
    order to do their compute.

    Args:
        coo: Sparse COO matrix with rows as generalized 'm'-indices and
            columns as noise count values.
        max_dense_batch_size_GB: Size of a batch on disk, in gigabytes.

    Returns:
        A generator that yields compact CSR sparse matrices until the whole dataset
        has been yielded. "Compact" in the sense that if they are made dense, there
        will be no all-zero rows.
            Tuple[chunk csr, actual row values in the full matrix]

    """
    n_elements_in_batch = max_dense_batch_size_GB * 1e9 / 4  # torch float32 is 4 bytes
    batch_size = max(1, int(np.floor(n_elements_in_batch / coo.shape[1])))

    # COO rows are not necessarily contiguous or in order
    unique_m_values = np.unique(coo.row)
    n_chunks = max(1, len(unique_m_values) // batch_size)
    row_m_value_chunks = np.array_split(unique_m_values, n_chunks)
    coo_row_series = pd.Series(coo.row)

    for row_m_values in row_m_value_chunks:
        logic = coo_row_series.isin(set(row_m_values))
        # Map these row values to a compact set of unique integers
        unique_row_values, rows = np.unique(coo.row[logic], return_inverse=True)
        unique_col_values, cols = np.unique(coo.col[logic], return_inverse=True)
        chunk_coo = sp.coo_matrix(
            (coo.data[logic], (rows, cols)),
            shape=(len(unique_row_values), len(unique_col_values)),
        )
        yield (chunk_coo, unique_row_values, unique_col_values)


def apply_function_dense_chunks(noise_log_prob_coo: sp.coo_matrix,
                                fun: Callable[[torch.Tensor], torch.Tensor],
                                device: str = 'cpu',
                                **kwargs) \
        -> Dict[str, np.ndarray]:
    """Uses chunked_iterator to densify chunked portions of a COO sparse
    matrix and then applies a function to the dense chunks, keeping track
    of the results per row.

    NOTE: The function should produce one value per row of the dense matrix.
          The COO should contain log probability in data.

    Args:
        noise_log_prob_coo: The posterior noise count log prob data structure,
            indexed by 'm' as rows
        fun: Pytorch function that operates on a dense tensor and produces
            one value per row
        device: ['cpu', 'cuda'] - whether to perform the pytorch sampling
            operation on CPU or GPU. It's pretty fast on CPU already.
        **kwargs: Passed to fun

    Returns:
        Dict containing
            'm': np.ndarray of indices
            'result': the values computed by the function

    """
    array_length = len(np.unique(noise_log_prob_coo.row))

    m = np.zeros(array_length, dtype=np.uint64)
    out = np.zeros(array_length)
    a = 0

    for coo, row, col in chunked_iterator(coo=noise_log_prob_coo):
        dense_tensor = torch.tensor(log_prob_sparse_to_dense(coo)).to(device)
        if torch.numel(dense_tensor) == 0:
            # github issue 207
            continue
        s = fun(dense_tensor, **kwargs)
        if s.ndim == 0:
            # avoid "TypeError: len() of a 0-d tensor"
            len_s = 1
        else:
            len_s = len(s)
        m[a:(a + len_s)] = row
        out[a:(a + len_s)] = s.detach().cpu().numpy()
        a = a + len_s

    return {'m': m, 'result': out}


def pandas_grouped_apply(coo: sp.coo_matrix,
                         fun: Callable[[pd.DataFrame], Union[int, float]],
                         extra_data: Optional[Dict[str, np.ndarray]] = None,
                         sort_first: bool = False,
                         parallel: bool = False) -> Dict[str, np.array]:
    """Apply function on a sparse COO format (noise log probs) to compute output
    noise counts using pandas groupby and apply operations on CPU.

    TODO: consider numpy-groupies or np.ufunc.reduceat or similar
    https://stackoverflow.com/questions/31790819/scipy-sparse-csr-matrix-how-to-get-top-ten-values-and-indices

    Args:
        coo: COO data structure: (m, c) with 'm'-indexing.
        fun: Function to be applied with pandas .apply(): turns
            all the values for a matrix entry (m, :) into one number.
        extra_data: To include extra information other than 'm', 'c', and the
            'log_prob', you can pass in a dict here with array values the same
            length as 'm' and in the same order.
        sort_first: Sort the whole dataframe by ['m', 'c'] before applying
            function (much faster than sorting inside the function call).
        parallel: Use multiprocessing to run on all cores. This is 4x slower for
            a dataset of 300 cells.  Not clear if it's faster for larger data.

    Returns:
        output_csr: The aggregated sparse matrix, in row format
    """
    df = pd.DataFrame(data={'m': coo.row, 'c': coo.col, 'log_prob': coo.data})
    if extra_data is not None:
        for k, v in extra_data.items():
            df[k] = v
    if sort_first:
        df = df.sort_values(by=['m', 'c'], ascending=[True, True])
    if parallel:
        m, result_per_m = _parallel_pandas_apply(df_grouped=df.groupby('m'), fun=fun)
    else:
        df = df.groupby('m').apply(fun).reset_index()
        m = df['m'].values
        result_per_m = df[0].values
    return {'m': m, 'result': result_per_m}


def _parallel_pandas_apply(df_grouped: pd.core.groupby.DataFrameGroupBy,
                           fun: Callable[[pd.DataFrame], Union[int, float]])\
        -> Tuple[np.ndarray, np.ndarray]:
    """Use multiprocessing to apply a function to a grouped dataframe in pandas.

    Args:
        df_grouped: Grouped dataframe, as in df.groupby('m')
        fun: Function to be applied to dataframe, as in df.groupby('m').apply(fun)

    Returns:
        Tuple of (groupby_value, groupby_apply_output_for_each_value)

    NOTE: see https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
    """
    groupby_val, group_df_list = zip(*[(group_val, group_df)
                                       for group_val, group_df in df_grouped])
    with mp.Pool(mp.cpu_count()) as p:
        output_list = p.map(fun, group_df_list)
    return np.array(groupby_val), np.array(output_list)


def _subset_coo(coo: sp.coo_matrix, logic: np.ndarray) -> sp.coo_matrix:
    return sp.coo_matrix((coo.data[logic], (coo.row[logic], coo.col[logic])))


def timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
