import numpy as np
import scipy.sparse as sp
import torch

import operator
import pickle
import logging
from typing import List, Union, Dict, Callable, Optional, Tuple

from boltons.cacheutils import cachedproperty, cachedmethod

from sklearn.decomposition import PCA

from cellbender.sampling.fingerprint_sampler import CSRFloatMatrix, CSRBinaryMatrix, \
    SingleCellFingerprintStratifiedSampler

# module-level cache for memoization
_global_cache_dict = dict()


class SingleCellFingerprintBase:
    """
    .. note:: This class is supposed to be close to optimally memory-efficient. As such, dense arrays must
        only be used for quantities that scale either as O(n_cells) or O(n_genes). In particular, the count
        matrix and the fingerprint tensor is stored as sparse matrices.
    """

    def __init__(self,
                 gene_idx_list: List[int],
                 gene_names_list: List[str],
                 max_family_size: int,
                 barcode_list: Union[None, List[int]] = None,
                 csr_fingerprint_list: Union[None, List[sp.csr_matrix]] = None,
                 verbose: bool = True):
        """Initializer.

        :param gene_idx_list: list of gene indices (these indices could correspond to gene identities in an
            external gene list)
        :param gene_names_list: list of gene names (corresponding to gene_idx_list)
        :param max_family_size: maximum allowed family size per molecule
        :param barcode_list: list of barcodes (in an integer representation)
        :param csr_fingerprint_list: list of sparse fingerprint matrices corresponding to the barcodes in
            ``barcode_list``

        .. note:: The (sparse) fingerprint matrix of a barcode is a matrix of shape ``(n_genes, max_family_size)``.
            A element ``(i, j)`` in the matrix denotes the number of unique instances of gene ``i`` that is
            observed ``j + 1`` times in the experiment. In particular, the first column of the fingerprint matrix
            corresponds to the count of singleton ("orphan") molecules.
        """
        assert len(gene_idx_list) > 0, \
            "The fingerprint must have at least one gene!"
        assert len(gene_names_list) == len(gene_idx_list)

        if barcode_list is None:
            barcode_list = list()
        if csr_fingerprint_list is None:
            csr_fingerprint_list = list()
        assert len(barcode_list) == len(csr_fingerprint_list), \
            "The fingerprint list must have as many elements as the cell barcode list!"

        self.gene_idx_list = gene_idx_list
        self.gene_names_list = gene_names_list

        self.max_family_size = max_family_size
        self.csr_fingerprint_dict: Dict[int, sp.csr_matrix] = dict()
        self.barcode_list: List[int] = list()
        self.verbose = verbose
        self._logger = logging.getLogger()
        self._finalized = False

        # populate
        for barcode, csr_fingerprint_list in zip(barcode_list, csr_fingerprint_list):
            self._add_new_barcode(barcode, csr_fingerprint_list)

    def _finalize(self):
        self._finalized = True

    @property
    def finalized(self):
        return self._finalized

    def _add_new_barcode(self, barcode: int, csr_fingerprint: sp.csr_matrix):
        """Adds a new barcode to the collection.
        """
        assert not self.finalized, \
            "The class is in a finalized state, possibly as a matter of accessing " \
            "one of the cached properties."
        assert barcode not in self.csr_fingerprint_dict, \
            f"Cell barcode {barcode} already has a fingerprint!"
        assert csr_fingerprint.shape[0] == self.n_genes, \
            f"The fingerprint matrix must has as many rows ({csr_fingerprint.shape[0]}) as the number " \
                f"of genes ({len(self.gene_idx_list)})!"
        assert csr_fingerprint.shape[1] == self.max_family_size, \
            f"The fingerprint matrix must has as many columns ({csr_fingerprint.shape[1]}) as the maximum " \
                f"family size ({self.max_family_size})!"
        self.csr_fingerprint_dict[barcode] = csr_fingerprint
        self.barcode_list.append(barcode)

    def _log_caching(self, name: str):
        if self.verbose:
            self._logger.warning(f"Calculating and caching \"SingleCellFingerprintBase.{name}\"...")

    def __getitem__(self, barcode: int) -> sp.csr_matrix:
        """Returns the fingerprint for a given barcode (NOTE: not barcode index!)"""
        return self.csr_fingerprint_dict[barcode]

    def save(self, output_path: str):
        """Saves the instance to a .pkl file"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.gene_idx_list, f)
            pickle.dump(self.gene_names_list, f)
            pickle.dump(self.csr_fingerprint_dict, f)

    @staticmethod
    def load(input_path: str) -> 'SingleCellFingerprintBase':
        """Instantiate from a .pkl file"""
        with open(input_path, 'rb') as f:
            loader = pickle.Unpickler(f)
            gene_idx_list = loader.load()
            gene_names_list = loader.load()
            csr_fingerprint_dict = loader.load()

        max_family_size = csr_fingerprint_dict.items().__iter__().__next__()[1].shape[1]
        new = SingleCellFingerprintBase(gene_idx_list, gene_names_list, max_family_size)
        for barcode, csr_fingerprint in csr_fingerprint_dict.items():
            new._add_new_barcode(barcode, csr_fingerprint)
        return new

    @property
    def n_genes(self):
        return len(self.gene_idx_list)

    @property
    def n_cells(self):
        return len(self.barcode_list)

    @cachedproperty
    def eps(self):
        return np.finfo(np.float).eps

    @cachedproperty
    def good_turing_estimator_g(self) -> np.ndarray:
        """Returns the Good-Turing estimator per gene.

        .. note:: yields NaN if a gene has no counts across all barcodes.
        """
        self._finalize()
        self._log_caching("good_turing_estimator_g")
        orphan_reads = np.zeros((self.n_genes,), dtype=np.float)
        all_reads = np.zeros((self.n_genes,), dtype=np.float)
        read_counter = np.arange(1, self.max_family_size + 1)
        for csr_fingerprint in self.csr_fingerprint_dict.values():
            orphan_reads += np.asarray(csr_fingerprint[:, 0].todense()).flatten()
            all_reads += csr_fingerprint.dot(read_counter)
        good_turing_estimator = orphan_reads / (self.eps + all_reads)
        good_turing_estimator[all_reads == 0] = np.nan
        return good_turing_estimator

    @cachedproperty
    def total_molecules_per_gene_g(self) -> np.ndarray:
        self._finalize()
        self._log_caching("total_molecules_per_gene_g")
        total_gene_expression = np.zeros((self.n_genes,))
        for csr_fingerprint in self.csr_fingerprint_dict.values():
            total_gene_expression += np.asarray(np.sum(csr_fingerprint, -1)).flatten()
        return total_gene_expression

    @cachedproperty
    def sparse_count_matrix_csr(self):
        self._finalize()
        self._log_caching("sparse_count_matrix_csr")
        return sp.vstack(
            list(map(
                lambda barcode: sp.csr_matrix(self.csr_fingerprint_dict[barcode].sum(-1)).T,
                self.barcode_list)),
            format='csr')

    def filter_barcodes(self) -> 'SingleCellFingerprintBase':
        raise NotImplementedError

    def filter_genes(self,
                     max_good_turing: float = 0.5,
                     min_total_gene_expression: int = 10,
                     verbose_logging: bool = False) -> 'SingleCellFingerprintBase':
        # calculate summary statistics
        good_turing_estimator_array = self.good_turing_estimator_g
        total_gene_expression_array = self.total_molecules_per_gene_g

        num_failed_good_turing = 0
        num_failed_min_expression = 0
        num_failed_both = 0

        kept_gene_array_idx_list = list()
        for i_gene in range(self.n_genes):
            c_gene_good_turing_estimator = good_turing_estimator_array[i_gene]
            c_gene_total_expression = total_gene_expression_array[i_gene]
            failed_min_expression = False
            failed_good_turing = False
            failed = False

            if c_gene_total_expression < min_total_gene_expression:
                if verbose_logging:
                    self._logger.warning(
                        f"Total expression ({int(c_gene_total_expression)}) of gene "
                        f"({self.gene_idx_list[i_gene]}) is below the minimum allowed threshold "
                        f"({min_total_gene_expression}) -- the gene was dropped from the fingerprint.")
                failed_min_expression = True
                failed = True

            if np.isnan(c_gene_good_turing_estimator) or c_gene_good_turing_estimator > max_good_turing:
                if verbose_logging:
                    if np.isnan(c_gene_good_turing_estimator):
                        self._logger.warning(
                            f"Good-Turing estimator is not defined for gene "
                            f"({self.gene_idx_list[i_gene]}) "
                            f"-- the gene was dropped from the fingerprint.")
                    else:
                        self._logger.warning(
                            f"Good-Turing estimator ({c_gene_good_turing_estimator:.3f}) for "
                            f"gene ({self.gene_idx_list[i_gene]}) is below the maximum allowed threshold "
                            f"({max_good_turing}) -- the gene was dropped from the fingerprint.")
                failed_good_turing = True
                failed = True

            num_failed_good_turing += failed_good_turing
            num_failed_min_expression += failed_min_expression
            num_failed_both += (failed_min_expression and failed_good_turing)

            if not failed:
                kept_gene_array_idx_list.append(i_gene)

        kept_gene_idx_list = [self.gene_idx_list[i_gene] for i_gene in kept_gene_array_idx_list]
        kept_gene_names_list = [self.gene_names_list[i_gene] for i_gene in kept_gene_array_idx_list]
        self._logger.warning(f"Number of genes failed the maximum Good-Turing criterion: {num_failed_good_turing}")
        self._logger.warning(f"Number of genes failed the minimum expression criterion: {num_failed_min_expression}")
        self._logger.warning(f"Number of genes failed both criteria: {num_failed_both}")
        self._logger.warning(f"Number of retained genes: {len(kept_gene_idx_list)}")

        new = SingleCellFingerprintBase(kept_gene_idx_list, kept_gene_names_list, self.max_family_size)
        for barcode, csr_fingerprint in self.csr_fingerprint_dict.items():
            new._add_new_barcode(barcode, csr_fingerprint[kept_gene_array_idx_list, :])
        return new

    def subset_genes(self, subset_gene_idx_list: List[int]) -> 'SingleCellFingerprintBase':
        """Returns a new instance of the collection with subset gene indices.

        .. note:: the order of genes in the new collection is the same as the provided sublist of genes.
        """
        full_gene_idx_set = set(self.gene_idx_list)
        assert all(subset_gene_idx in full_gene_idx_set for subset_gene_idx in subset_gene_idx_list), \
            "Some of the gene indices in ``subset_gene_idx_list`` do not exist in the instance!"
        assert len(set(subset_gene_idx_list)) == len(subset_gene_idx_list), \
            "The subset gene index list contains repeated entries"
        gene_idx_to_old_list_index_map = {
            gene_index: old_list_index for old_list_index, gene_index in enumerate(self.gene_idx_list)}
        keep_index = list(map(gene_idx_to_old_list_index_map.get, subset_gene_idx_list))
        subset_gene_names_list = [self.gene_names_list[idx] for idx in keep_index]
        new = SingleCellFingerprintBase(subset_gene_idx_list, subset_gene_names_list, self.max_family_size)
        for barcode, csr_fingerprint in self.csr_fingerprint_dict.items():
            new._add_new_barcode(barcode, csr_fingerprint[keep_index, :])
        return new

    def subset_barcodes(self, subset_barcode_list: List[int]) -> 'SingleCellFingerprintBase':
        """Returns a new instance of the collection with subset barcodes.

        .. note:: the order of barcodes in the new collection is the same as the provided sublist of barcodes.
        """
        full_barcode_set = set(self.barcode_list)
        assert all(subset_barcode in full_barcode_set for subset_barcode in subset_barcode_list), \
            "Some of the barcodes in ``subset_barcode_list`` do not exist in the instance!"
        assert len(set(subset_barcode_list)) == len(subset_barcode_list), \
            "The subset barcode list contains repeated entries!"
        new = SingleCellFingerprintBase(self.gene_idx_list, self.max_family_size)
        for barcode in subset_barcode_list:
            new._add_new_barcode(barcode, self[barcode])
        return new

    def keep_top_k_genes_by_expression(self, first_rank: int, last_rank: int):
        """Returns a new instance where only genes within a certain rank-range in expression are kept"""
        assert first_rank >= 0
        assert last_rank > first_rank
        assert last_rank <= self.n_genes
        sorted_gene_idx_list_by_expression = list(map(
            operator.itemgetter(0),
            sorted(
                zip(self.gene_idx_list, self.total_molecules_per_gene_g),
                key=operator.itemgetter(1),
                reverse=True)))
        return self.subset_genes(sorted_gene_idx_list_by_expression[first_rank:last_rank])

    def sort_genes_by_expression(self) -> 'SingleCellFingerprintBase':
        """Returns a new instance with genes sorted by their expression in descending order"""
        return self.keep_top_k_genes_by_expression(
            first_rank=0,
            last_rank=self.n_genes)

    def get_internal_gene_indices_by_gene_name_prefix(self, prefix: str) -> List[int]:
        return [gene_index for gene_index in range(self.n_genes)
                if self.gene_names_list[gene_index].find(prefix) == 0]


class SingleCellFingerprintDTM:
    """This class extends the functionality of SingleCellFingerprintBase for Droplet Time Machine (TM)
    training."""

    # minimum over-dispersion of an estimated negative binomial fit
    _min_nb_phi = 1e-2

    def __init__(self,
                 sc_fingerprint_base: SingleCellFingerprintBase,
                 low_family_size_cdf_threshold: float = 0.20,
                 n_cell_pca_features: int = 20,
                 gene_grouping_trans: Callable[[np.ndarray], np.ndarray] = np.log,
                 n_gene_groups: int = 10,
                 allow_dense_int_ndarray: bool = False,
                 verbose: bool = True,
                 device: torch.device = torch.device("cuda"),
                 dtype: torch.dtype = torch.float):

        self.sc_fingerprint_base = sc_fingerprint_base
        self.low_family_size_cdf_threshold = low_family_size_cdf_threshold
        self.n_cell_pca_features = n_cell_pca_features
        self.gene_grouping_trans = gene_grouping_trans
        self.n_gene_groups = n_gene_groups
        self.allow_dense_int_ndarray = allow_dense_int_ndarray
        self.verbose = verbose
        self.device = device
        self.dtype = dtype

        self._logger = logging.getLogger()

        # total observed expression of all genes
        if np.any(sc_fingerprint_base.total_molecules_per_gene_g == 0):
            self._logger.warning("Some of the genes in the provided fingerprint have zero counts in the "
                                 "entire dataset!")

        # placeholder for lazy initialization
        self.highly_variable_gene_indices: Optional[List[int]] = None

    def _log_caching(self, name: str):
        if self.verbose:
            self._logger.warning(f"Calculating and caching \"SingleCellFingerprintDTM.{name}\"...")

    @cachedproperty
    def n_cells(self):
        return self.sc_fingerprint_base.n_cells

    @cachedproperty
    def n_genes(self):
        return self.sc_fingerprint_base.n_genes

    @property
    def max_family_size(self):
        return self.sc_fingerprint_base.max_family_size

    @cachedproperty
    def numpy_dtype(self):
        if self.dtype == torch.float16:
            return np.float16
        if self.dtype == torch.float32:
            return np.float32
        if self.dtype == torch.float64:
            return np.float64
        else:
            raise ValueError("Bad torch dtype -- allowed values are: float16, float32, float64")

    @cachedproperty
    def eps(self):
        return np.finfo(self.numpy_dtype).eps

    def set_highly_variable_gene_indices(self, highly_variable_gene_indices: List[int]):
        # assert that highly variable genes are not specified yet (as changing the indices will invalidate
        # certain cached properties)
        assert self.highly_variable_gene_indices is None, \
            "Highly variable genes are already specified!"
        self.highly_variable_gene_indices = highly_variable_gene_indices

    @cachedproperty
    def dense_fingerprint_ndarray(self) -> np.ndarray:
        self._log_caching("dense_fingerprint_ndarray")
        assert self.allow_dense_int_ndarray
        fingerprint_array = np.zeros((self.n_cells, self.n_genes, self.max_family_size), dtype=np.uint16)
        for i_cell, barcode in enumerate(self.sc_fingerprint_base.barcode_list):
            fingerprint_array[i_cell, :, :] = self.sc_fingerprint_base[barcode].todense()
        return fingerprint_array

    @property
    def sparse_count_matrix_csr(self) -> sp.csr_matrix:
        return self.sc_fingerprint_base.sparse_count_matrix_csr

    @cachedproperty
    def sparse_count_matrix_csc(self) -> sp.csc_matrix:
        self._log_caching("sparse_count_matrix_csc")
        return sp.csc_matrix(self.sc_fingerprint_base.sparse_count_matrix_csr)

    @cachedproperty
    def collapsed_csr_fingerprint_matrix_cython(self) -> CSRFloatMatrix:
        """Returns the collapsed fingerprint matrix.

        .. note:: The collapsed fingerprint matrix is simply the vertically stacked fingerprint matrix
            of each barcode in the order of addition to the collection (i.e. in the same order as
            ``barcode_list``).
        """
        self._log_caching("collapsed_csr_fingerprint_matrix_cython")
        collapsed_csr_fingerprint_matrix_scipy = sp.vstack(list(
            map(self.sc_fingerprint_base.csr_fingerprint_dict.get,
                self.sc_fingerprint_base.barcode_list)))
        collapsed_csr_fingerprint_matrix_cython = CSRFloatMatrix(
            n_rows=collapsed_csr_fingerprint_matrix_scipy.shape[0],
            n_cols=collapsed_csr_fingerprint_matrix_scipy.shape[1],
            indptr=collapsed_csr_fingerprint_matrix_scipy.indptr,
            indices=collapsed_csr_fingerprint_matrix_scipy.indices,
            data=collapsed_csr_fingerprint_matrix_scipy.data.astype(self.numpy_dtype))
        return collapsed_csr_fingerprint_matrix_cython

    def get_single_gene_dense_fingerprint_array(self,
                                                gene_index: int,
                                                buffer: Optional[np.ndarray] = None) -> np.ndarray:
        if buffer is not None:
            assert buffer.shape == (self.n_cells, self.max_family_size)
            buffer.fill(0)
        else:
            buffer = np.zeros((self.n_cells, self.max_family_size), dtype=self.numpy_dtype)
        collapsed_index_array = gene_index + np.arange(self.n_cells, dtype=np.int32) * self.n_genes
        self.collapsed_csr_fingerprint_matrix_cython.copy_rows_to_dense(
            collapsed_index_array, buffer)
        return buffer

    @cachedproperty
    def family_size_threshold_per_gene(self) -> np.ndarray:
        """Estimates per-gene family size cut-off given a lower CDF threshold on the empirical
        family size distribution."""
        self._log_caching("family_size_threshold_per_gene")
        assert 0 <= self.low_family_size_cdf_threshold < 1
        family_size_threshold_g = np.zeros((self.n_genes,), dtype=np.int)
        single_gene_fingerprint_array = np.zeros((self.n_cells, self.max_family_size), dtype=self.numpy_dtype)
        for gene_index in range(self.n_genes):
            _ = self.get_single_gene_dense_fingerprint_array(
                gene_index=gene_index,
                buffer=single_gene_fingerprint_array)
            counts_per_family_size = single_gene_fingerprint_array.sum(0)
            family_size_cdf = np.cumsum(counts_per_family_size) / np.sum(counts_per_family_size)
            try:
                family_size_threshold_g[gene_index] = np.where(
                    family_size_cdf > self.low_family_size_cdf_threshold)[0][0] + 1
            except IndexError:
                family_size_threshold_g[gene_index] = 1
        return family_size_threshold_g

    @cachedproperty
    def sparse_family_size_truncated_count_matrix_csr(self) -> sp.csr_matrix:
        self._log_caching("sparse_family_size_truncated_count_matrix_csr")
        family_size_mask_g = (
                np.arange(0, self.max_family_size)[:, None]
                >= (self.family_size_threshold_per_gene - 1)
        ).T.astype(np.int)

        rows = []
        for barcode in self.sc_fingerprint_base.barcode_list:
            barcode_csr_fingerprint = self.sc_fingerprint_base.csr_fingerprint_dict[barcode]
            row = sp.csr_matrix(np.sum(np.asarray(barcode_csr_fingerprint.todense()) * family_size_mask_g, -1))
            rows.append(row)
        return sp.vstack(rows, format='csr')

    @cachedproperty
    def sparse_family_size_truncated_count_matrix_csc(self) -> sp.csc_matrix:
        self._log_caching("sparse_family_size_truncated_count_matrix_csc")
        return sp.csc_matrix(self.sparse_family_size_truncated_count_matrix_csr)

    @cachedproperty
    def dense_family_size_truncated_count_matrix(self) -> np.ndarray:
        self._log_caching("dense_family_size_truncated_count_matrix")
        return np.asarray(self.sparse_family_size_truncated_count_matrix_csr.todense()).astype(self.numpy_dtype)

    @cachedproperty
    def dense_count_matrix_ndarray(self) -> np.ndarray:
        self._log_caching("dense_count_matrix_ndarray")
        assert self.allow_dense_int_ndarray
        return np.sum(self.dense_fingerprint_ndarray, -1)

    @cachedproperty
    def expressing_cells_per_gene_dict(self) -> Dict[int, List[int]]:
        self._log_caching("expressing_cells_per_gene_dict")
        expressing_cells_per_gene_dict: Dict[int, List[int]] = dict()
        if self.allow_dense_int_ndarray:
            for i_gene in range(self.n_genes):
                expressing_cells_per_gene_dict[i_gene] = np.where(
                    self.dense_count_matrix_ndarray[:, i_gene] > 0)[0].tolist()
        else:
            for i_gene in range(self.n_genes):
                expressing_cells_per_gene_dict[i_gene] = (
                    self.sparse_count_matrix_csc[:, i_gene].nonzero()[0].tolist())
        return expressing_cells_per_gene_dict

    @cachedproperty
    def silent_cells_per_gene_dict(self) -> Dict[int, List[int]]:
        self._log_caching("silent_cells_per_gene_dict")
        silent_cells_per_gene_dict: Dict[int, List[int]] = dict()
        if self.allow_dense_int_ndarray:
            for i_gene in range(self.n_genes):
                silent_cells_per_gene_dict[i_gene] = np.where(
                    self.dense_count_matrix_ndarray[:, i_gene] == 0)[0].tolist()
        else:
            all_cells_set = set(range(0, self.n_cells))
            for i_gene in range(self.n_genes):
                silent_cells_per_gene_dict[i_gene] = list(
                    all_cells_set - set(self.get_expressing_cell_indices(i_gene)))
        return silent_cells_per_gene_dict

    def get_expressing_cell_indices(self, gene_index: int) -> List[int]:
        return self.expressing_cells_per_gene_dict[gene_index]

    def get_silent_cell_indices(self, gene_index: int) -> List[int]:
        return self.silent_cells_per_gene_dict[gene_index]

    @cachedproperty
    def n_expressing_cells_per_gene(self) -> List[int]:
        self._log_caching("n_expressing_cells_per_gene")
        num_expressing_cells = [
            len(self.get_expressing_cell_indices(i_gene)) for i_gene in range(self.n_genes)]
        return num_expressing_cells

    @cachedproperty
    def n_silent_cells_per_gene(self) -> List[int]:
        self._log_caching("n_silent_cells_per_gene")
        num_silent_cells = [
            self.n_cells - self.n_expressing_cells_per_gene[i_gene] for i_gene in range(self.n_genes)]
        return num_silent_cells

    @cachedproperty
    def total_obs_reads_per_cell(self) -> np.ndarray:
        self._log_caching("total_obs_reads_per_cell")
        total_obs_reads_per_cell = np.zeros((self.n_cells,), dtype=self.numpy_dtype)
        family_size_vector = np.arange(1, self.max_family_size + 1)
        for i_cell, barcode in enumerate(self.sc_fingerprint_base.barcode_list):
            total_obs_reads_per_cell[i_cell] = np.sum(self.sc_fingerprint_base[barcode].dot(family_size_vector))
        return total_obs_reads_per_cell

    @cachedproperty
    def total_obs_molecules_per_cell(self) -> np.ndarray:
        self._log_caching("total_obs_molecules_per_cell")
        total_obs_molecules_per_cell = np.zeros((self.n_cells,), dtype=self.numpy_dtype)
        for i_cell, barcode in enumerate(self.sc_fingerprint_base.barcode_list):
            total_obs_molecules_per_cell[i_cell] = np.sum(self.sc_fingerprint_base[barcode])
        return total_obs_molecules_per_cell

    @cachedproperty
    def arithmetic_mean_obs_expr_per_gene(self) -> np.ndarray:
        self._log_caching("arithmetic_mean_obs_expr_per_gene")
        return np.asarray(self.sparse_count_matrix_csr.sum(0)).squeeze(0).astype(self.numpy_dtype) / self.n_cells

    @cachedproperty
    def geometric_mean_obs_expr_per_gene(self) -> np.ndarray:
        self._log_caching("geometric_mean_obs_expr_per_gene")
        log1p_sparse_count_matrix_csr = sp.csr_matrix(
            (np.log1p(self.sparse_count_matrix_csr.data),
             self.sparse_count_matrix_csr.indices,
             self.sparse_count_matrix_csr.indptr))
        return np.asarray(
            np.exp(np.mean(log1p_sparse_count_matrix_csr, axis=0)) - 1).flatten().astype(self.numpy_dtype)

    @cachedproperty
    def empirical_droplet_efficiency(self) -> np.ndarray:
        return self.total_obs_molecules_per_cell / np.mean(self.total_obs_molecules_per_cell)

    @cachedproperty
    def features_per_cell(self) -> np.ndarray:
        self._log_caching("features_per_cell")

        # perform PCA over highly variable genes
        assert self.highly_variable_gene_indices is not None, \
            "Highly variable genes for PCA features are not provided yet -- cannot continue."

        # obtain raw counts of HVG
        raw_count_matrix = np.asarray(
            self.sparse_family_size_truncated_count_matrix_csc[:, self.highly_variable_gene_indices].todense()
        ).astype(self.numpy_dtype)

        # normalize by empirical droplet efficiency (total UMIs per cell)
        normed_count_matrix = raw_count_matrix / self.empirical_droplet_efficiency[:, None]

        # log1p transformation after droplet efficiency normalization
        trans_normed_count_matrix = np.log1p(normed_count_matrix)

        # perform PCA
        pca_features = PCA(n_components=self.n_cell_pca_features, whiten=True).fit_transform(
            trans_normed_count_matrix)

        # log empirical cell size
        log_empirical_cell_size = np.log(self.empirical_droplet_efficiency)

        # concatenate
        full_features = np.concatenate(
            (log_empirical_cell_size[:, None],
             pca_features),
            axis=1)

        return full_features.astype(self.numpy_dtype)

    @cachedproperty
    def gene_groups_dict(self) -> Dict[int, List[int]]:
        self._log_caching("gene_groups_dict")
        # the "weight" of each gene is a monotonic (approximately logarithmic) function of its
        # total observed expression
        total_obs_expr_per_gene = self.n_cells * self.arithmetic_mean_obs_expr_per_gene
        weights = self.gene_grouping_trans(total_obs_expr_per_gene)

        # bucket genes into groups of equal total weight
        sorted_genes_idx_weight = sorted(enumerate(weights), key=operator.itemgetter(1), reverse=True)
        sorted_weights = np.asarray(list(map(operator.itemgetter(1), sorted_genes_idx_weight)))
        weight_per_group = np.sum(weights) / self.n_gene_groups
        sorted_weights_cumsum = np.cumsum(sorted_weights)
        gene_group_start_indices = [
            min(np.where(sorted_weights_cumsum > i_group * weight_per_group)[0][0], self.n_genes)
            for i_group in range(self.n_gene_groups)]
        gene_group_stop_indices = [gene_group_start_indices[j + 1] for j in range(self.n_gene_groups - 1)]
        gene_group_stop_indices.append(self.n_genes)

        gene_groups_dict: Dict[int, List[int]] = dict()
        for i_group in range(self.n_gene_groups):
            gene_group_start_index = gene_group_start_indices[i_group]
            gene_group_stop_index = gene_group_stop_indices[i_group]
            gene_groups_dict[i_group] = list(
                map(operator.itemgetter(0),
                    sorted_genes_idx_weight[gene_group_start_index:gene_group_stop_index]))
        return gene_groups_dict

    @cachedproperty
    def empirical_fsd_params(self) -> np.ndarray:
        self._log_caching("empirical_fsd_params")

        # heuristic quantile-based FS threshold mask with shape (n_genes, max_family_size)
        family_size_mask_g = (
                np.arange(0, self.max_family_size)[:, None]
                    >= (self.family_size_threshold_per_gene - 1)).T.astype(np.int)

        empirical_fsd_params = np.zeros((self.n_genes, 3), dtype=self.numpy_dtype)
        single_gene_fingerprint_array = np.zeros((self.n_cells, self.max_family_size), dtype=self.numpy_dtype)
        for gene_index in range(self.n_genes):
            # obtain the empirical FSD from all cells
            _ = self.get_single_gene_dense_fingerprint_array(
                gene_index=gene_index,
                buffer=single_gene_fingerprint_array)

            # apply the heuristic quantile-based FS threshold to remove some of the chimeras
            counts_per_family_size = single_gene_fingerprint_array.sum(0) * family_size_mask_g[gene_index, :]
            total_kept_molecules = np.sum(counts_per_family_size)
            assert total_kept_molecules > 0

            # calculate empirical moments
            family_size_array = np.arange(1, self.max_family_size + 1)
            family_size_pmf = counts_per_family_size / total_kept_molecules
            family_size_mean = np.sum(family_size_array * family_size_pmf)
            family_size_var = np.sum((family_size_array ** 2) * family_size_pmf) - family_size_mean ** 2

            # estimate negative binomial fit using first two moments
            mu = family_size_mean
            phi = max(self._min_nb_phi, (family_size_var - family_size_mean) / (family_size_mean ** 2))

            # calculate p_obs
            alpha = 1. / phi
            p_obs = 1. - np.exp(alpha * np.log(alpha / (alpha + mu)))

            empirical_fsd_params[gene_index, 0] = mu
            empirical_fsd_params[gene_index, 1] = phi
            empirical_fsd_params[gene_index, 2] = p_obs

        return empirical_fsd_params

    @cachedproperty
    def empirical_fsd_mu_hi(self) -> np.ndarray:
        self._log_caching("empirical_fsd_mu_hi")
        return self.empirical_fsd_params[:, 0]

    @cachedproperty
    def empirical_fsd_phi_hi(self) -> np.ndarray:
        self._log_caching("empirical_fsd_phi_hi")
        return self.empirical_fsd_params[:, 1]

    @cachedproperty
    def empirical_fsd_p_obs(self) -> np.ndarray:
        self._log_caching("empirical_fsd_p_obs")
        return self.empirical_fsd_params[:, 2]

    @cachedmethod(_global_cache_dict)
    def _get_fingerprint_minibatch_ndarray_buffers(
            self,
            genes_per_gene_group: int,
            expressing_cells_per_gene: int,
            silent_cells_per_gene: int) -> Dict[str, np.ndarray]:
        assert genes_per_gene_group > 0
        assert expressing_cells_per_gene > 0
        assert silent_cells_per_gene > 0

        max_buffer_size = genes_per_gene_group * self.n_gene_groups * (
                expressing_cells_per_gene + silent_cells_per_gene)
        fingerprint_array = np.zeros((max_buffer_size, self.max_family_size), dtype=self.numpy_dtype, order='c')
        gene_index_array = np.zeros((max_buffer_size,), dtype=np.int32, order='c')
        cell_index_array = np.zeros((max_buffer_size,), dtype=np.int32, order='c')
        gene_sampling_site_scale_factor_array = np.zeros((max_buffer_size,), dtype=self.numpy_dtype, order='c')
        cell_sampling_site_scale_factor_array = np.zeros((max_buffer_size,), dtype=self.numpy_dtype, order='c')

        assert fingerprint_array.flags['C_CONTIGUOUS']
        assert gene_index_array.flags['C_CONTIGUOUS']
        assert cell_index_array.flags['C_CONTIGUOUS']
        assert gene_sampling_site_scale_factor_array.flags['C_CONTIGUOUS']
        assert cell_sampling_site_scale_factor_array.flags['C_CONTIGUOUS']

        return {'fingerprint_array': fingerprint_array,
                'gene_index_array': gene_index_array,
                'cell_index_array': cell_index_array,
                'gene_sampling_site_scale_factor_array': gene_sampling_site_scale_factor_array,
                'cell_sampling_site_scale_factor_array': cell_sampling_site_scale_factor_array}

    @cachedmethod(_global_cache_dict)
    def _get_counts_minibatch_ndarray_buffers(
            self,
            genes_per_gene_group: int,
            expressing_cells_per_gene: int,
            silent_cells_per_gene: int) -> Dict[str, np.ndarray]:
        assert genes_per_gene_group > 0
        assert expressing_cells_per_gene > 0
        assert silent_cells_per_gene > 0

        max_buffer_size = genes_per_gene_group * self.n_gene_groups * (
                expressing_cells_per_gene + silent_cells_per_gene)
        counts_array = np.zeros((max_buffer_size,), dtype=self.numpy_dtype, order='c')
        gene_index_array = np.zeros((max_buffer_size,), dtype=np.int32, order='c')
        cell_index_array = np.zeros((max_buffer_size,), dtype=np.int32, order='c')
        gene_sampling_site_scale_factor_array = np.zeros((max_buffer_size,), dtype=self.numpy_dtype, order='c')
        cell_sampling_site_scale_factor_array = np.zeros((max_buffer_size,), dtype=self.numpy_dtype, order='c')

        assert counts_array.flags['C_CONTIGUOUS']
        assert gene_index_array.flags['C_CONTIGUOUS']
        assert cell_index_array.flags['C_CONTIGUOUS']
        assert gene_sampling_site_scale_factor_array.flags['C_CONTIGUOUS']
        assert cell_sampling_site_scale_factor_array.flags['C_CONTIGUOUS']

        return {'counts_array': counts_array,
                'gene_index_array': gene_index_array,
                'cell_index_array': cell_index_array,
                'gene_sampling_site_scale_factor_array': gene_sampling_site_scale_factor_array,
                'cell_sampling_site_scale_factor_array': cell_sampling_site_scale_factor_array}

    @cachedproperty
    def gene_expression_csr_binary_matrix(self) -> CSRBinaryMatrix:
        """Returns the (gene, cell) binary expression matrix as :class:`CSRBinaryMatrix`.

        .. note:: we can get the required indptr and indices from the CSC representation of the
            (cell, gene) sparse count matrix.
        """
        return CSRBinaryMatrix(
            n_rows=self.n_genes,
            n_cols=self.n_cells,
            indices_sz=len(self.sparse_count_matrix_csc.indices),
            indptr=self.sparse_count_matrix_csc.indptr.astype(np.int32),
            indices=self.sparse_count_matrix_csc.indices.astype(np.int32))

    @cachedproperty
    def family_size_truncated_expression_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        self._log_caching("family_size_truncated_expression_mean_std")
        # calculate raw mean and std expression
        family_size_truncated_expression_mean_g = np.zeros((self.n_genes,))
        family_size_truncated_expression_std_g = np.zeros((self.n_genes,))

        for gene_index in range(self.n_genes):
            raw_trunc_expression_n = np.asarray(
                self.sparse_family_size_truncated_count_matrix_csc[:, gene_index].todense()).flatten()
            family_size_truncated_expression_mean_g[gene_index] = np.mean(raw_trunc_expression_n)
            family_size_truncated_expression_std_g[gene_index] = np.std(raw_trunc_expression_n)

        return (family_size_truncated_expression_mean_g,
                family_size_truncated_expression_std_g)

    @property
    def family_size_truncated_expression_mean(self):
        return self.family_size_truncated_expression_mean_std[0]

    @property
    def family_size_truncated_expression_std(self):
        return self.family_size_truncated_expression_mean_std[1]

    @cachedproperty
    def gene_groups_csr_binary_matrix(self) -> CSRBinaryMatrix:
        indptr = [0]
        indices = []
        for i_gene_group in range(self.n_gene_groups):
            indices += sorted(self.gene_groups_dict[i_gene_group])
            indptr.append(len(indices))
        return CSRBinaryMatrix(
            n_rows=self.n_gene_groups,
            n_cols=self.n_genes,
            indices_sz=len(indices),
            indptr=np.asarray(indptr, dtype=np.int32),
            indices=np.asarray(indices, dtype=np.int32))

    @cachedmethod(_global_cache_dict)
    def stratified_sampler(self, sampling_strategy: str = 'with_replacement'):
        assert sampling_strategy in ['with_replacement', 'without_replacement']
        return SingleCellFingerprintStratifiedSampler(
            gene_groups_csr=self.gene_groups_csr_binary_matrix,
            expressing_cells_csr=self.gene_expression_csr_binary_matrix,
            sampling_strategy=sampling_strategy)

    def generate_fingerprint_torch_minibatch_data(
            self,
            cell_index_array: np.ndarray,
            gene_index_array: np.ndarray,
            cell_sampling_site_scale_factor_array: np.ndarray,
            gene_sampling_site_scale_factor_array: np.ndarray,
            fingerprint_array: Optional[np.ndarray] = None) \
            -> Dict[str, Optional[torch.Tensor]]:
        mb_size = len(cell_index_array)
        assert cell_index_array.ndim == 1
        assert gene_index_array.ndim == 1
        assert cell_sampling_site_scale_factor_array.ndim == 1
        assert gene_sampling_site_scale_factor_array.ndim == 1
        assert len(gene_index_array) == mb_size
        assert len(cell_sampling_site_scale_factor_array) == mb_size
        assert len(gene_sampling_site_scale_factor_array) == mb_size

        # if a pre-allocated buffer is provided, assert its properties
        if fingerprint_array is not None:
            assert fingerprint_array.ndim == 2
            assert fingerprint_array.shape[0] >= mb_size
            assert fingerprint_array.shape[1] == self.max_family_size

        total_obs_reads_per_cell_array = self.total_obs_reads_per_cell[cell_index_array]
        total_obs_molecules_per_cell_array = self.total_obs_molecules_per_cell[cell_index_array]
        empirical_fsd_mu_hi_array = self.empirical_fsd_mu_hi[gene_index_array]
        arithmetic_mean_obs_expr_per_gene_array = self.arithmetic_mean_obs_expr_per_gene[gene_index_array]
        geometric_mean_obs_expr_per_gene_array = self.geometric_mean_obs_expr_per_gene[gene_index_array]
        empirical_droplet_efficiency_array = self.empirical_droplet_efficiency[cell_index_array]

        collapsed_index_array = cell_index_array * self.n_genes + gene_index_array
        if fingerprint_array is None:
            fingerprint_array = np.zeros((mb_size, self.max_family_size), dtype=self.numpy_dtype)
        fingerprint_array.fill(0)
        self.collapsed_csr_fingerprint_matrix_cython.copy_rows_to_dense(
            collapsed_index_array, fingerprint_array)
        fingerprint_tensor = torch.tensor(
            fingerprint_array[:mb_size, :], device=self.device, dtype=self.dtype)

        cell_features_array = self.features_per_cell[cell_index_array, :]
        cell_features_tensor = torch.tensor(cell_features_array, device=self.device, dtype=self.dtype)

        return {
            'cell_index_tensor': torch.tensor(
                cell_index_array,
                device=self.device).long(),

            'gene_index_tensor': torch.tensor(
                gene_index_array,
                device=self.device).long(),

            'total_obs_reads_per_cell_tensor': torch.tensor(
                total_obs_reads_per_cell_array,
                device=self.device,
                dtype=self.dtype),

            'total_obs_molecules_per_cell_tensor': torch.tensor(
                total_obs_molecules_per_cell_array,
                device=self.device,
                dtype=self.dtype),

            'arithmetic_mean_obs_expr_per_gene_tensor': torch.tensor(
                arithmetic_mean_obs_expr_per_gene_array,
                device=self.device,
                dtype=self.dtype),

            'geometric_mean_obs_expr_per_gene_tensor': torch.tensor(
                geometric_mean_obs_expr_per_gene_array,
                device=self.device,
                dtype=self.dtype),

            'fingerprint_tensor': fingerprint_tensor,

            'empirical_fsd_mu_hi_tensor': torch.tensor(
                empirical_fsd_mu_hi_array,
                device=self.device,
                dtype=self.dtype),

            'empirical_droplet_efficiency_tensor': torch.tensor(
                empirical_droplet_efficiency_array,
                device=self.device,
                dtype=self.dtype),

            'cell_features_tensor': cell_features_tensor,

            'cell_sampling_site_scale_factor_tensor': torch.tensor(
                cell_sampling_site_scale_factor_array,
                device=self.device,
                dtype=self.dtype),

            'gene_sampling_site_scale_factor_tensor': torch.tensor(
                gene_sampling_site_scale_factor_array,
                device=self.device,
                dtype=self.dtype)
        }

    def generate_counts_torch_minibatch_data(
            self,
            cell_index_array: np.ndarray,
            gene_index_array: np.ndarray,
            cell_sampling_site_scale_factor_array: np.ndarray,
            gene_sampling_site_scale_factor_array: np.ndarray,
            counts_array: Optional[np.ndarray] = None) \
            -> Dict[str, Optional[torch.Tensor]]:
        mb_size = len(cell_index_array)
        assert cell_index_array.ndim == 1
        assert gene_index_array.ndim == 1
        assert cell_sampling_site_scale_factor_array.ndim == 1
        assert gene_sampling_site_scale_factor_array.ndim == 1
        assert len(gene_index_array) == mb_size
        assert len(cell_sampling_site_scale_factor_array) == mb_size
        assert len(gene_sampling_site_scale_factor_array) == mb_size

        # if a pre-allocated buffer is provided, assert its properties
        if counts_array is not None:
            assert counts_array.ndim == 1
            assert counts_array.shape[0] >= mb_size

        geometric_mean_obs_expr_per_gene_array = self.geometric_mean_obs_expr_per_gene[gene_index_array]
        empirical_droplet_efficiency_array = self.empirical_droplet_efficiency[cell_index_array]

        if counts_array is None:
            counts_array = np.zeros((mb_size,), dtype=self.numpy_dtype)
        else:
            counts_array.fill(0)
        counts_array[:mb_size] = self.dense_family_size_truncated_count_matrix[cell_index_array, gene_index_array]

        counts_tensor = torch.tensor(
            counts_array[:mb_size], device=self.device, dtype=self.dtype)

        return {
            'cell_index_tensor': torch.tensor(
                cell_index_array,
                device=self.device).long(),

            'gene_index_tensor': torch.tensor(
                gene_index_array,
                device=self.device).long(),

            'cell_sampling_site_scale_factor_tensor': torch.tensor(
                cell_sampling_site_scale_factor_array,
                device=self.device,
                dtype=self.dtype),

            'gene_sampling_site_scale_factor_tensor': torch.tensor(
                gene_sampling_site_scale_factor_array,
                device=self.device,
                dtype=self.dtype),

            'geometric_mean_obs_expr_per_gene_tensor': torch.tensor(
                geometric_mean_obs_expr_per_gene_array,
                device=self.device,
                dtype=self.dtype),

            'empirical_droplet_efficiency_tensor': torch.tensor(
                empirical_droplet_efficiency_array,
                device=self.device,
                dtype=self.dtype),

            'counts_tensor': counts_tensor
        }

    def generate_fingerprint_stratified_sample(
                self,
                genes_per_gene_group: int,
                expressing_cells_per_gene: int,
                silent_cells_per_gene: int,
                sampling_strategy: str) -> Dict[str, torch.Tensor]:

        np_buff_dict = self._get_fingerprint_minibatch_ndarray_buffers(
            genes_per_gene_group=genes_per_gene_group,
            expressing_cells_per_gene=expressing_cells_per_gene,
            silent_cells_per_gene=silent_cells_per_gene)

        sampler = self.stratified_sampler(sampling_strategy)

        n_samples = sampler.draw(
            genes_per_gene_group=genes_per_gene_group,
            expressing_cells_per_gene=expressing_cells_per_gene,
            silent_cells_per_gene=silent_cells_per_gene,
            gene_index_memview=np_buff_dict['gene_index_array'],
            cell_index_memview=np_buff_dict['cell_index_array'],
            gene_sampling_site_scale_factor_memview=np_buff_dict['gene_sampling_site_scale_factor_array'],
            cell_sampling_site_scale_factor_memview=np_buff_dict['cell_sampling_site_scale_factor_array'])

        return self.generate_fingerprint_torch_minibatch_data(
            cell_index_array=np_buff_dict['cell_index_array'][:n_samples],
            gene_index_array=np_buff_dict['gene_index_array'][:n_samples],
            cell_sampling_site_scale_factor_array=np_buff_dict['cell_sampling_site_scale_factor_array'][:n_samples],
            gene_sampling_site_scale_factor_array=np_buff_dict['gene_sampling_site_scale_factor_array'][:n_samples],
            fingerprint_array=np_buff_dict['fingerprint_array'])

    def generate_single_gene_fingerprint_minibatch_data(
            self,
            gene_index: int,
            cell_index_list: List[int],
            n_particles_cell: int) -> Dict[str, torch.Tensor]:
        """Generate model input tensors for a given gene index and the cell index range

        :param gene_index:
        :param cell_index_list:
        :param n_particles_cell: repeat factor for every cell (see the notes)

        .. note: for n_particles_cell > 1, the entire ``cell_index_list`` is repeated that many times

        .. note: The generated minibatch has scale-factor set to 1.0 for all gene and cell sampling
            sites (because they are not necessary for our purposes here). As such, the minibatches
            produced by this method should not be used for training.
        """
        assert n_particles_cell >= 1
        cell_index_array = np.repeat(np.asarray(cell_index_list), n_particles_cell).astype(np.int32)
        gene_index_array = gene_index * np.ones_like(cell_index_array).astype(np.int32)
        cell_sampling_site_scale_factor_array = np.ones_like(
            cell_index_array, dtype=self.numpy_dtype)
        gene_sampling_site_scale_factor_array = np.ones_like(
            cell_index_array, dtype=self.numpy_dtype)
        fingerprint_array = np.zeros(
            (len(cell_index_array), self.max_family_size),
            dtype=self.numpy_dtype)

        return self.generate_fingerprint_torch_minibatch_data(
            cell_index_array=cell_index_array,
            gene_index_array=gene_index_array,
            cell_sampling_site_scale_factor_array=cell_sampling_site_scale_factor_array,
            gene_sampling_site_scale_factor_array=gene_sampling_site_scale_factor_array,
            fingerprint_array=fingerprint_array)

    def generate_counts_stratified_sample(
                self,
                genes_per_gene_group: int,
                expressing_cells_per_gene: int,
                silent_cells_per_gene: int,
                sampling_strategy: str) -> Dict[str, torch.Tensor]:

        np_buff_dict = self._get_counts_minibatch_ndarray_buffers(
            genes_per_gene_group=genes_per_gene_group,
            expressing_cells_per_gene=expressing_cells_per_gene,
            silent_cells_per_gene=silent_cells_per_gene)

        sampler = self.stratified_sampler(sampling_strategy)

        n_samples = sampler.draw(
            genes_per_gene_group=genes_per_gene_group,
            expressing_cells_per_gene=expressing_cells_per_gene,
            silent_cells_per_gene=silent_cells_per_gene,
            gene_index_memview=np_buff_dict['gene_index_array'],
            cell_index_memview=np_buff_dict['cell_index_array'],
            gene_sampling_site_scale_factor_memview=np_buff_dict['gene_sampling_site_scale_factor_array'],
            cell_sampling_site_scale_factor_memview=np_buff_dict['cell_sampling_site_scale_factor_array'])

        return self.generate_counts_torch_minibatch_data(
            cell_index_array=np_buff_dict['cell_index_array'][:n_samples],
            gene_index_array=np_buff_dict['gene_index_array'][:n_samples],
            cell_sampling_site_scale_factor_array=np_buff_dict['cell_sampling_site_scale_factor_array'][:n_samples],
            gene_sampling_site_scale_factor_array=np_buff_dict['gene_sampling_site_scale_factor_array'][:n_samples],
            counts_array=np_buff_dict['counts_array'])

# def downsample_single_fingerprint_numpy(fingerprint_array: np.ndarray, downsampling_rate: float):
#     assert fingerprint_array.ndim == 1
#     assert 0. <= downsampling_rate <= 1.0
#
#     max_family_size = fingerprint_array.shape[0]
#     max_count = int(np.max(fingerprint_array))
#
#     # generate the maximum number of required binomial samples
#     binomial_samples = np.random.binomial(
#         n=np.arange(1, max_family_size + 1)[None, :].repeat(max_count, 0),
#         p=downsampling_rate)
#
#     # generate a mask for which samples to keep
#     keep_mask = (fingerprint_array - np.arange(0, max_count)[:, None]) > 0
#
#     # calculate the histogram of downsampled reads (neglecting 0)
#     downsampled_fingerprint_array = np.histogram(
#         binomial_samples[keep_mask],
#         range=(1, max_family_size),
#         bins=max_family_size)[0]
#
#     return downsampled_fingerprint_array


# def generate_downsampled_minibatch(original_data_dict: Dict[str, torch.Tensor],
#                                    min_downsampling_rate: float,
#                                    max_downsampling_rate: float) -> Dict[str, torch.Tensor]:
#     downsampled_data_dict = dict()
#
#     # copy unchanged data
#     downsampled_data_dict['cell_index_tensor'] = \
#         original_data_dict['cell_index_tensor']
#     downsampled_data_dict['gene_index_tensor'] = \
#         original_data_dict['gene_index_tensor']
#     downsampled_data_dict['cell_sampling_site_scale_factor_tensor'] = \
#         original_data_dict['cell_sampling_site_scale_factor_tensor']
#     downsampled_data_dict['gene_sampling_site_scale_factor_tensor'] = \
#         original_data_dict['gene_sampling_site_scale_factor_tensor']
#
#     # downsampling rate tensor
#     downsampled_data_dict['downsampling_rate_tensor'] = min_downsampling_rate + (
#             (max_downsampling_rate - min_downsampling_rate)
#             * torch.rand_like(original_data_dict['downsampling_rate_tensor']))
#
#     # downsample the total observed reads per cell
#     downsampled_data_dict['total_obs_reads_per_cell_tensor'] = torch.distributions.Binomial(
#         original_data_dict['total_obs_reads_per_cell_tensor'],
#         probs=downsampled_data_dict['downsampling_rate_tensor']).sample()
#
#     # downsample the fingerprint
#     downsampling_rate_numpy = downsampled_data_dict['downsampling_rate_tensor'].cpu().numpy()
#     fingerprint_numpy = original_data_dict['fingerprint_tensor'].cpu().numpy().astype(np.int)
#     downsampled_fingerprint_numpy = np.zeros_like(fingerprint_numpy)
#     for i_cell in range(fingerprint_numpy.shape[0]):
#         downsampled_fingerprint_numpy[i_cell, :] = downsample_single_fingerprint_numpy(
#             fingerprint_numpy[i_cell, :],
#             downsampling_rate_numpy[i_cell])
#     downsampled_data_dict['fingerprint_tensor'] = torch.tensor(
#         downsampled_fingerprint_numpy,
#         dtype=original_data_dict['fingerprint_tensor'].dtype,
#         device=original_data_dict['fingerprint_tensor'].device)
#
#     # downsample the empirical fsd mu hi
#     downsampled_data_dict['empirical_fsd_mu_hi_tensor'] = (
#             downsampled_data_dict['downsampling_rate_tensor'] *
#             original_data_dict['empirical_fsd_mu_hi_tensor'])
#
#     return downsampled_data_dict
