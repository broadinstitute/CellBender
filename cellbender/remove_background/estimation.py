"""Classes and methods for estimation of noise counts, given a posterior."""

import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
from torch.distributions.categorical import Categorical

from cellbender.remove_background.sparse_utils import log_prob_sparse_to_dense

from abc import ABC, abstractmethod
from functools import partial
import multiprocessing as mp
import time
import logging
from typing import Callable, Union, Dict, Generator, Tuple, List, Optional


logger = logging.getLogger('cellbender')


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
                                 dtype=np.int64) -> sp.csr_matrix:
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
        row, col = self.index_converter.get_ng_indices(m_inds=m)
        if noise_offsets is not None:
            data = data + np.array([noise_offsets[i] for i in m])
        coo = sp.coo_matrix((data.astype(dtype), (row.astype(dtype), col.astype(dtype))),
                            shape=self.index_converter.matrix_shape, dtype=dtype)
        coo.sum_duplicates()
        return coo.tocsr()


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
        c = torch.arange(noise_log_prob_coo.shape[1], dtype=float).to(device).t()

        def _torch_mean(x):
            return torch.matmul(x.exp(), c)

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


class MultipleChoiceKnapsack(EstimationMethod):
    """Noise estimation via solving a constrained multiple choice knapsack problem"""

    def estimate_noise(self,
                       noise_log_prob_coo: sp.coo_matrix,
                       noise_offsets: Dict[int, int],
                       noise_targets_per_gene: np.ndarray,
                       approx_gb: float = 1.,
                       verbose: bool = False,
                       min_chunks: Optional[int] = None,  # just for testing
                       **kwargs) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_targets_per_gene: Integer noise count target for each gene
            noise_offsets: Noise count offset values keyed by 'm'.
            approx_gb: Approximate memory to be used for each gene-wise chunk
                of this computation. This is baseline, not peak memory. Peak
                memory might be 12x this value (on top of whatever else is
                going on)
            verbose: True to print lots of intermediate information (for tests)
            min_chunks: For tests, to ensure chunking works

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """
        csr_matrices = []
        for chunk in self._gene_chunk_iterator(
            noise_log_prob_coo=noise_log_prob_coo,
            approx_gb=approx_gb,
            min_chunks=min_chunks,
        ):
            noise_coo = chunk[0]
            gene_logic = chunk[1]
            chunk_csr = self._chunk_estimate_noise(
                noise_log_prob_coo=noise_coo,
                noise_offsets=noise_offsets,
                noise_targets_per_gene=noise_targets_per_gene,
                verbose=verbose,
            )
            csr_matrices.append(chunk_csr.tocsc()[:, gene_logic].tocsr())
        return sp.hstack(csr_matrices)

    def _gene_chunk_iterator(self,
                             noise_log_prob_coo: sp.coo_matrix,
                             approx_gb: float,
                             min_chunks: Optional[int] = None) \
            -> Generator[Tuple[sp.coo_matrix, np.ndarray], None, None]:
        """Yields chunks of the posterior that can be treated as independent,
        from the standpoint of MCKP count estimation.  That is, they contain all
        matrix entries for any genes they include.

        Args:
            approx_gb: Aim to grab a chunk of this size
            min_chunks: For testing, force this many chunks

        Yields:
            Tuple of (coo posterior, gene_logic) for the chunk
        """

        def _approx_size(n_entries: int) -> float:
            approx_size = (n_entries * 3) * 4 / 1e9  # GB, torch float32 is 4 bytes
            approx_size = approx_size * 20  # empirical fudge from memory profiler
            return approx_size

        if min_chunks is not None:
            # reset approx_gb to target the right number of chunks
            approx_gb = _approx_size((noise_log_prob_coo.data.size - 1) // min_chunks)

        # get gene annotations
        _, genes = self.index_converter.get_ng_indices(m_inds=noise_log_prob_coo.row)

        # things we need to keep track of for each chunk
        current_chunk_genes = []
        entry_logic = np.zeros(noise_log_prob_coo.data.size, dtype=bool)
        gene_logic = np.zeros(self.index_converter.total_n_genes, dtype=bool)

        def _subset_coo(coo: sp.coo_matrix, logic: np.ndarray) -> sp.coo_matrix:
            return sp.coo_matrix((coo.data[logic], (coo.row[logic], coo.col[logic])))

        # go through all genes
        for g in np.unique(genes):

            # append current gene
            current_chunk_genes.append(g)
            entry_logic = entry_logic | (genes == g)
            gene_logic[g] = True

            # figure out when to send out a chunk
            if _approx_size(entry_logic.sum()) >= approx_gb:
                logger.debug('New gene chunk being processed by MCKP')
                yield _subset_coo(noise_log_prob_coo, entry_logic), gene_logic

                # keep track and reset stuff
                entry_logic = np.zeros(noise_log_prob_coo.data.size, dtype=bool)
                gene_logic = np.zeros(self.index_converter.total_n_genes, dtype=bool)
                current_chunk_genes = []

        # last bit
        if entry_logic.sum() > 0:
            logger.debug('New gene chunk being processed by MCKP')
            yield _subset_coo(noise_log_prob_coo, entry_logic), gene_logic

    def _chunk_estimate_noise(self,
                              noise_log_prob_coo: sp.coo_matrix,
                              noise_offsets: Dict[int, int],
                              noise_targets_per_gene: np.ndarray,
                              verbose: bool = False) -> sp.csr_matrix:
        """Given the full probabilistic posterior, compute noise counts. This is
        to be run for a given chunk of genes at a time.

        Args:
            noise_log_prob_coo: The noise log prob data structure: log prob
                values in a (m, c) COO matrix
            noise_targets_per_gene: Integer noise count target for each gene
            noise_offsets: Noise count offset values keyed by 'm'.
            verbose: True to print lots of intermediate information (for tests)

        Returns:
            noise_count_csr: Estimated noise count matrix.
        """

        # TODO: can I split this in chunks of genes and use multiprocessing??
        # TODO: this runs on one CPU at 100% for quite some time
        # TODO: the issue seems to be high memory use (runs out on hgmm12k with 52GB RAM)
        # TODO: do I understand the high memory usage?

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
        logger.info('Computed initial MAP estimate')
        logger.debug(f'Time for MAP calculation = {(time.time() - t) / 60:.2f} mins')
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
        logger.debug('Computing step directions')
        df['positive_step_gene'] = df['g'].apply(lambda gene: gene in set_positive_genes)
        df['negative_step_gene'] = df['g'].apply(lambda gene: gene in set_negative_genes)
        df['step_direction'] = (df['positive_step_gene'].astype(int)
                                - df['negative_step_gene'].astype(int))  # -1 or 1
        logger.debug('Step directions done')

        if verbose:
            pd.set_option('display.width', 120)
            pd.set_option('display.max_columns', 20)
            print(df, end='\n\n')

        # Remove all 'm' entries corresponding to genes where target is met by MAP.
        df = df[df['step_direction'] != 0]

        # Now we mask out log probs (-np.inf) that represent steps in the wrong direction.
        logger.debug('Masking')
        lookup_map_from_m = dict(zip(map_dict['m'], map_dict['result']))
        df['map'] = df['m'].apply(lambda x: lookup_map_from_m[x])
        df['mask'] = ((df['negative_step_gene'] & (df['c'] > df['map']))
                      | (df['positive_step_gene'] & (df['c'] < df['map'])))  # keep MAP
        df.loc[df['mask'], 'log_prob'] = -np.inf
        logger.debug('Masking done')

        # And we remove those entries.
        df = df[~df['mask']]
        df = df[[c for c in df.columns if (c != 'mask')]]

        # Sort
        logger.debug('Sorting')
        df = df.sort_values(by=['m', 'c'])
        logger.debug('Sorting done')

        if verbose:
            print(df, end='\n\n')

        # Do diff for positive and negative separately, without grouping (faster)
        df_positive_steps = df[df['step_direction'] == 1].copy()
        df_negative_steps = df[df['step_direction'] == -1].copy()

        logger.debug('Computing deltas')
        if len(df_positive_steps > 0):
            df_positive_steps.loc[:, 'delta'] = df_positive_steps['log_prob'].diff(periods=1).apply(np.abs)
            df_positive_steps.loc[df_positive_steps['c'] == df_positive_steps['map'], 'delta'] = np.nan
        if len(df_negative_steps > 0):
            df_negative_steps.loc[:, 'delta'] = df_negative_steps['log_prob'].diff(periods=-1).apply(np.abs)
            df_negative_steps.loc[df_negative_steps['c'] == df_negative_steps['map'], 'delta'] = np.nan
        df = pd.concat([df_positive_steps, df_negative_steps], axis=0)
        logger.debug('Computing deltas done')

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
        logger.debug('Computing nsmallest')
        df['topk'] = df['g'].apply(lambda gene: abs_additional_noise_counts_per_gene[gene])

        if verbose:
            print(df, end='\n\n')

        # Now we want the smallest additional_noise_counts_per_gene deltas for each gene.
        # https://stackoverflow.com/questions/55179493/
        df_out = df[['m', 'g', 'delta', 'topk']].groupby('g', group_keys=False).apply(
            lambda x: x.nsmallest(x['topk'].iat[0], columns='delta')
        )
        logger.debug('Computing nsmallest done')

        if verbose:
            print(df_out, end='\n\n')

        # if this is an empty dataframe, we are not doing anything here beyond MAP
        if len(df_out) == 0:
            return map_csr

        # And the number by which to increment noise counts per entry 'm' is
        # now the number of times that each m value appears in this dataframe.
        logger.debug('Summarizing steps')
        vc = df_out['m'].value_counts()
        vc_df = pd.DataFrame(data={'m': vc.index, 'steps': vc.values})
        step_direction_lookup_from_m = dict(zip(df['m'], df['step_direction']))
        vc_df['step_direction'] = vc_df['m'].apply(lambda x: step_direction_lookup_from_m[x])
        vc_df['counts'] = vc_df['steps'] * vc_df['step_direction']
        steps_csr = self._estimation_array_to_csr(data=vc_df['counts'],
                                                  m=vc_df['m'],
                                                  noise_offsets=None)
        logger.debug('Summarizing steps done')

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
    batch_size = int(np.floor(n_elements_in_batch / coo.shape[1]))

    for i in range(0, coo.row.max(), batch_size):
        logic = (coo.row >= i) & (coo.row < (i + batch_size))
        # Map these row values to a compact set of unique integers
        unique_row_values, rows = np.unique(coo.row[logic], return_inverse=True)
        chunk_coo = sp.coo_matrix(
            (coo.data[logic], (rows, coo.col[logic])),
            shape=(len(unique_row_values), coo.shape[1]),
        )
        yield (chunk_coo, unique_row_values)


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

    m = np.zeros(array_length)
    out = np.zeros(array_length)
    a = 0

    for coo, row in chunked_iterator(coo=noise_log_prob_coo):
        dense_tensor = torch.tensor(log_prob_sparse_to_dense(coo)).to(device)
        s = fun(dense_tensor, **kwargs)
        if s.ndim == 0:
            # avoid "TypeError: len() of a 0-d tensor"
            len_s = 1
        else:
            len_s = len(s)
        m[a:(a + len_s)] = row
        out[a:(a + len_s)] = s.detach().cpu().numpy()
        a = a + len_s

    return {'m': m.astype(int), 'result': out}


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
