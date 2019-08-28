import numpy as np
import scipy.sparse as sp
import operator
import pickle
import logging
from boltons.cacheutils import cachedproperty
from typing import Tuple, List, Union, Dict, Callable
import torch
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from stats import ApproximateZINBFit
import random


class SingleCellFingerprintBase:
    """
    .. note:: This class is supposed to be close to optimally memory-efficient. As such, dense arrays must
        only be used for quantities that scale either as O(n_cells) or O(n_genes). In particular, the count
        matrix and the fingerprint tensor is stored as sparse matrices.
    """
    _eps = np.finfo(np.float).eps

    def __init__(self,
                 gene_idx_list: List[int],
                 max_family_size: int,
                 barcode_list: Union[None, List[int]] = None,
                 csr_fingerprint_list: Union[None, List[sp.csr_matrix]] = None,
                 verbose: bool = True):
        """Initializer.

        :param gene_idx_list: list of gene indices (these indices could correspond to gene identities in an
            external gene list)
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
        if barcode_list is None:
            barcode_list = list()
        if csr_fingerprint_list is None:
            csr_fingerprint_list = list()
        assert len(barcode_list) == len(csr_fingerprint_list), \
            "The fingerprint list must have as many elements as the cell barcode list!"

        self.gene_idx_list = gene_idx_list
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
            self._logger.warning(f"Calculating and caching {name}...")

    def __getitem__(self, barcode: int) -> sp.csr_matrix:
        """Returns the fingerprint for a given barcode (NOTE: not barcode index!)"""
        return self.csr_fingerprint_dict[barcode]

    def save(self, output_path: str):
        """Saves the instance to a .pkl file"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.gene_idx_list, f)
            pickle.dump(self.csr_fingerprint_dict, f)

    @staticmethod
    def load(input_path: str) -> 'SingleCellFingerprintBase':
        """Instantiate from a .pkl file"""
        with open(input_path, 'rb') as f:
            loader = pickle.Unpickler(f)
            gene_idx_list = loader.load()
            csr_fingerprint_dict = loader.load()

        max_family_size = csr_fingerprint_dict.items().__iter__().__next__()[1].shape[1]
        new = SingleCellFingerprintBase(gene_idx_list, max_family_size)
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
    def collapsed_csr_fingerprint_matrix(self):
        """Returns the collapsed fingerprint matrix.

        .. note:: The collapsed fingerprint matrix is simply the vertically stacked fingerprint matrix
            of each barcode in the order of addition to the collection (i.e. in the same order as
            ``barcode_list``).
        """
        self._finalize()
        self._log_caching("collapsed_csr_fingerprint_matrix")
        return sp.vstack(list(map(self.csr_fingerprint_dict.get, self.barcode_list)))

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
        good_turing_estimator = orphan_reads / (SingleCellFingerprintBase._eps + all_reads)
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

        self._logger.warning(f"Number of genes failed the maximum Good-Turing criterion: {num_failed_good_turing}")
        self._logger.warning(f"Number of genes failed the minimum expression criterion: {num_failed_min_expression}")
        self._logger.warning(f"Number of genes failed both criteria: {num_failed_both}")
        self._logger.warning(f"Number of retained genes: {len(kept_gene_idx_list)}")

        new = SingleCellFingerprintBase(kept_gene_idx_list, self.max_family_size)
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
        new = SingleCellFingerprintBase(subset_gene_idx_list, self.max_family_size)
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


def random_choice(a: List[int], size: int) -> List[int]:
    return random.sample(a, size)


def scaled_log1p_transform(
        nnz_umi_counts: np.ndarray,
        total_umi_scale: float = 10_000,
        eps: float = 1e-6) -> np.ndarray:
    normed_nnz_umi_counts = total_umi_scale * nnz_umi_counts / (eps + np.sum(nnz_umi_counts))
    return np.log1p(normed_nnz_umi_counts)


class SingleCellFingerprintDTM:
    """This class extends the functionality of SingleCellFingerprintBase for Droplet Time Machine (TM)
    training."""
    _eps = np.finfo(np.float).eps

    # minimum over-dispersion of an estimated negative binomial fit
    _min_nb_phi = 1e-2

    def __init__(self,
                 sc_fingerprint_base: SingleCellFingerprintBase,
                 max_estimated_chimera_family_size: int = 0,
                 low_family_size_cdf_threshold: float = 0.05,
                 n_pca_features: int = 50,
                 n_pca_iters: int = 20,
                 count_trans_feature_extraction: Callable[[np.ndarray], np.ndarray] = scaled_log1p_transform,
                 zinb_fitter_kwargs: Union[None, Dict[str, Union[int, float]]] = None,
                 gene_grouping_trans: Callable[[np.ndarray], np.ndarray] = np.log,
                 n_gene_groups: int = 10,
                 allow_dense_int_ndarray: bool = False,
                 verbose: bool = True,
                 device: torch.device = torch.device("cuda"),
                 dtype: torch.dtype = torch.float):
        if zinb_fitter_kwargs is None:
            zinb_fitter_kwargs = dict()

        self.sc_fingerprint_base = sc_fingerprint_base
        self.max_estimated_chimera_family_size = max_estimated_chimera_family_size
        self.low_family_size_cdf_threshold = low_family_size_cdf_threshold
        self.n_pca_features = n_pca_features
        self.n_pca_iters = n_pca_iters
        self.count_trans_feature_extraction = count_trans_feature_extraction
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

        # ZINB fitter
        self.zinb_fitter = ApproximateZINBFit(**zinb_fitter_kwargs)

    def _log_caching(self, name: str):
        if self.verbose:
            self._logger.warning(f"Calculating and caching {name}...")

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
    def dense_fingerprint_ndarray(self) -> np.ndarray:
        self._log_caching("dense_fingerprint_ndarray")
        assert self.allow_dense_int_ndarray
        fingerprint_array = np.zeros((self.n_cells, self.n_genes, self.max_family_size), dtype=np.uint16)
        for i_cell, barcode in enumerate(self.sc_fingerprint_base.barcode_list):
            fingerprint_array[i_cell, :, :] = self.sc_fingerprint_base[barcode].todense()
        return fingerprint_array

    @property
    def sparse_count_matrix_csr(self) -> sp.csr_matrix:
        self._log_caching("sparse_count_matrix_csr")
        return self.sc_fingerprint_base.sparse_count_matrix_csr

    @cachedproperty
    def sparse_count_matrix_csc(self) -> sp.csc_matrix:
        self._log_caching("sparse_count_matrix_csc")
        return sp.csc_matrix(self.sc_fingerprint_base.sparse_count_matrix_csr)

    @cachedproperty
    def family_size_threshold_per_gene(self) -> np.ndarray:
        """Estimates per-gene family size cut-off given a lower CDF threshold on the empirical
        family size distribution."""
        self._log_caching("family_size_threshold_per_gene")
        assert 0 <= self.low_family_size_cdf_threshold < 1
        family_size_threshold_g = np.zeros((self.n_genes,), dtype=np.int)
        for gene_index in range(self.n_genes):
            collapsed_slice = gene_index + self.n_genes * np.arange(self.n_cells)
            counts_per_family_size = np.asarray(
                self.sc_fingerprint_base.collapsed_csr_fingerprint_matrix[collapsed_slice, :].sum(0)).squeeze(0)
            family_size_cdf = np.cumsum(counts_per_family_size) / np.sum(counts_per_family_size)
            try:
                family_size_threshold_g[gene_index] = np.where(
                    family_size_cdf > self.low_family_size_cdf_threshold)[0][0] + 1
            except IndexError:
                family_size_threshold_g[gene_index] = 1
        return family_size_threshold_g

    @cachedproperty
    def sparse_family_size_truncated_count_matrix_csr(self):
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
    def sparse_family_size_truncated_count_matrix_csc(self):
        self._log_caching("sparse_family_size_truncated_count_matrix_csc")
        return sp.csc_matrix(self.sparse_family_size_truncated_count_matrix_csr)

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
        total_obs_reads_per_cell = np.zeros((self.n_cells,), dtype=np.uint64)
        family_size_vector = np.arange(1, self.max_family_size + 1)
        for i_cell, barcode in enumerate(self.sc_fingerprint_base.barcode_list):
            total_obs_reads_per_cell[i_cell] = np.sum(self.sc_fingerprint_base[barcode].dot(family_size_vector))
        return total_obs_reads_per_cell

    @cachedproperty
    def total_obs_molecules_per_cell(self) -> np.ndarray:
        self._log_caching("total_obs_molecules_per_cell")
        total_obs_molecules_per_cell = np.zeros((self.n_cells,), dtype=np.uint64)
        for i_cell, barcode in enumerate(self.sc_fingerprint_base.barcode_list):
            total_obs_molecules_per_cell[i_cell] = np.sum(self.sc_fingerprint_base[barcode])
        return total_obs_molecules_per_cell

    @cachedproperty
    def total_obs_expr_per_gene(self) -> np.ndarray:
        self._log_caching("total_obs_expr_per_gene")
        return np.asarray(self.sparse_count_matrix_csr.sum(0)).squeeze(0)

    @cachedproperty
    def mean_obs_expr_per_gene(self) -> np.ndarray:
        self._log_caching("total_obs_expr_per_gene")
        return self.total_obs_expr_per_gene.astype(np.float) / self.n_cells

    @cachedproperty
    def trans_trunc_sparse_count_matrix_csr(self) -> sp.csr_matrix:
        self._log_caching("trans_trunc_sparse_count_matrix_csr")
        # transformed and family-size-truncated count matrix
        trunc_sparse_count_matrix_csr = self.sparse_family_size_truncated_count_matrix_csr
        return sp.csr_matrix(
            (self.count_trans_feature_extraction(trunc_sparse_count_matrix_csr.data),
             trunc_sparse_count_matrix_csr.indices,
             trunc_sparse_count_matrix_csr.indptr),
            shape=trunc_sparse_count_matrix_csr.shape,
            dtype=np.float)

    @cachedproperty
    def trans_trunc_count_matrix_features_truncated_svd_object(self) -> TruncatedSVD:
        self._log_caching("trans_trunc_count_matrix_features_truncated_svd_object")
        svd = TruncatedSVD(n_components=self.n_pca_features, n_iter=self.n_pca_iters)
        return svd.fit(self.trans_trunc_sparse_count_matrix_csr)

    @cachedproperty
    def svd_feature_loadings_per_cell(self) -> np.ndarray:
        self._log_caching("svd_feature_loadings_per_cell")
        svd = self.trans_trunc_count_matrix_features_truncated_svd_object
        return svd.transform(self.trans_trunc_sparse_count_matrix_csr)

    @cachedproperty
    def svd_feature_z_scores_per_cell(self) -> np.ndarray:
        self._log_caching("svd_feature_z_scores_per_cell")
        features = self.svd_feature_loadings_per_cell
        features_mean, features_std = np.mean(features, 0), np.std(features, 0)
        features_z_score = (features - features_mean[None, :]) / (self._eps + features_std[None, :])
        return features_z_score

    @cachedproperty
    def svd_feature_components(self) -> np.ndarray:
        return self.trans_trunc_count_matrix_features_truncated_svd_object.components_

    @cachedproperty
    def total_obs_molecules_feature_z_scores_per_cell(self) -> np.ndarray:
        self._log_caching("total_obs_molecules_feature_z_scores_per_cell")
        raw_counts = self.total_obs_molecules_per_cell.astype(np.float)
        log1p_counts = np.log1p(raw_counts)
        raw_mean, raw_std = np.mean(raw_counts), np.std(raw_counts)
        log1p_mean, log1p_std = np.mean(log1p_counts), np.std(log1p_counts)
        raw_z_score = (raw_counts - raw_mean) / (self._eps + raw_std)
        log1p_z_score = (log1p_counts - log1p_mean) / (self._eps + log1p_std)
        return np.hstack((raw_z_score[:, None], log1p_z_score[:, None]))

    @cachedproperty
    def feature_z_scores_per_cell(self) -> np.ndarray:
        self._log_caching("feature_z_scores_per_cell")
        return np.hstack(
            (self.svd_feature_z_scores_per_cell,
             self.total_obs_molecules_feature_z_scores_per_cell))

    @cachedproperty
    def gene_groups_dict(self) -> Dict[int, List[int]]:
        self._log_caching("gene_groups_dict")
        # the "weight" of each gene is a monotonic (approximately logarithmic) function of its
        # total observed expression
        weights = self.gene_grouping_trans(self.total_obs_expr_per_gene)

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
        empirical_fsd_params = np.zeros((self.n_genes, 3))
        for gene_index in range(self.n_genes):
            collapsed_slice = gene_index + self.n_genes * np.arange(self.n_cells)
            counts_per_family_size = np.asarray(
                self.sc_fingerprint_base.collapsed_csr_fingerprint_matrix[collapsed_slice, :].sum(0)).squeeze(0)

            # "cap" the empirical histogram as a heuristic for attenuating chimeric counts
            if self.max_estimated_chimera_family_size >= 1:
                counts_per_family_size[:self.max_estimated_chimera_family_size] = counts_per_family_size[
                    self.max_estimated_chimera_family_size]

            family_size_array = np.arange(1, self.max_family_size + 1)
            family_size_pmf = counts_per_family_size / np.sum(counts_per_family_size)
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
    def empirical_e_hi_params(self) -> np.ndarray:
        self._log_caching("empirical_e_hi_params")
        # estimated probability of observing real molecules
        p_obs_g = self.empirical_fsd_params[:, 2]

        # fit zero-inflated negative-binomial
        self._logger.warning("Fitting approximate ZINB to UMI counts (per gene)...")
        empirical_e_hi_params = np.zeros((self.n_genes, 3))
        for gene_index in tqdm(range(self.n_genes)):
            # inflate the counts to account for p_obs
            e_hi_est = np.asarray(self.sparse_count_matrix_csc[:, gene_index].todense()).squeeze(-1) / (
                    p_obs_g[gene_index] + self._eps)
            fit = self.zinb_fitter(e_hi_est)
            if not fit['converged']:
                self._logger.warning(f'ZINB fit to gene (internal index: {gene_index}) was not successful!')
            empirical_e_hi_params[gene_index, 0] = fit['mu']
            empirical_e_hi_params[gene_index, 1] = fit['phi']
            empirical_e_hi_params[gene_index, 2] = fit['p_zero']
        return empirical_e_hi_params

    @cachedproperty
    def empirical_mu_e_hi(self) -> np.ndarray:
        self._log_caching("empirical_mu_e_hi")
        return self.empirical_e_hi_params[:, 0]

    @cachedproperty
    def empirical_phi_e_hi(self) -> np.ndarray:
        self._log_caching("empirical_phi_e_hi")
        return self.empirical_e_hi_params[:, 1]

    @cachedproperty
    def empirical_p_zero_e_hi(self) -> np.ndarray:
        self._log_caching("empirical_p_zero_e_hi")
        return self.empirical_e_hi_params[:, 2]

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

    def _generate_stratified_sample(
            self,
            mb_genes_per_gene_group: int,
            mb_expressing_cells_per_gene: int,
            mb_silent_cells_per_gene: int) -> Dict[str, np.ndarray]:
        mb_cell_indices_per_gene = []
        mb_cell_scale_factors_per_gene = []
        mb_effective_gene_scale_factors_per_cell = []
        mb_gene_indices_per_cell = []

        # select genes
        for i_gene_group in range(self.n_gene_groups):
            gene_size = min(mb_genes_per_gene_group, len(self.gene_groups_dict[i_gene_group]))
            gene_indices = random_choice(self.gene_groups_dict[i_gene_group], gene_size)
            gene_scale_factor = len(self.gene_groups_dict[i_gene_group]) / gene_size

            # select cells
            for gene_index in gene_indices:
                # sample from expressing cells
                size_expressing = min(mb_expressing_cells_per_gene, self.n_expressing_cells_per_gene[gene_index])
                expressing_cell_indices = random_choice(
                    self.get_expressing_cell_indices(gene_index), size_expressing)
                expressing_scale_factor = gene_scale_factor * (
                        self.n_expressing_cells_per_gene[gene_index] / size_expressing)

                # sample from silent cells
                size_silent = min(mb_silent_cells_per_gene, self.n_silent_cells_per_gene[gene_index])
                silent_cell_indices = random_choice(self.get_silent_cell_indices(gene_index), size_silent)
                silent_scale_factor = gene_scale_factor * (
                        self.n_silent_cells_per_gene[gene_index] / size_silent)

                mb_cell_indices_per_gene.append(np.asarray(expressing_cell_indices))
                mb_cell_indices_per_gene.append(np.asarray(silent_cell_indices))
                mb_cell_scale_factors_per_gene.append(expressing_scale_factor * np.ones((size_expressing,)))
                mb_cell_scale_factors_per_gene.append(silent_scale_factor * np.ones((size_silent,)))

                # the effective scale factor for a collapsed gene sampling site is scaled down by the number of cells
                total_cells_per_gene = size_expressing + size_silent
                effective_gene_scale_factor = gene_scale_factor / total_cells_per_gene
                mb_effective_gene_scale_factors_per_cell.append(
                    effective_gene_scale_factor * np.ones((total_cells_per_gene,)))
                mb_gene_indices_per_cell.append(gene_index * np.ones((total_cells_per_gene,), dtype=np.int))

        gene_index_array = np.concatenate(mb_gene_indices_per_cell)
        cell_index_array = np.concatenate(mb_cell_indices_per_gene)
        cell_sampling_site_scale_factor_array = np.concatenate(mb_cell_scale_factors_per_gene)
        gene_sampling_site_scale_factor_array = np.concatenate(mb_effective_gene_scale_factors_per_cell)

        return {'gene_index_array': gene_index_array,
                'cell_index_array': cell_index_array,
                'cell_sampling_site_scale_factor_array': cell_sampling_site_scale_factor_array,
                'gene_sampling_site_scale_factor_array': gene_sampling_site_scale_factor_array}

    def generate_torch_minibatch_data(self,
                                      cell_index_array: np.ndarray,
                                      gene_index_array: np.ndarray,
                                      cell_sampling_site_scale_factor_array: np.ndarray,
                                      gene_sampling_site_scale_factor_array: np.ndarray) -> Dict[str, torch.Tensor]:
        assert cell_index_array.ndim == 1
        assert gene_index_array.ndim == 1
        assert cell_sampling_site_scale_factor_array.ndim == 1
        assert gene_sampling_site_scale_factor_array.ndim == 1
        mb_size = len(cell_index_array)
        assert len(gene_index_array) == mb_size
        assert len(cell_sampling_site_scale_factor_array) == mb_size
        assert len(gene_sampling_site_scale_factor_array) == mb_size

        total_obs_reads_per_cell_array = self.total_obs_reads_per_cell[cell_index_array]
        collapsed_index_array = cell_index_array * self.n_genes + gene_index_array
        fingerprint_array = np.asarray(
            self.sc_fingerprint_base.collapsed_csr_fingerprint_matrix[collapsed_index_array, :].todense())
        empirical_fsd_mu_hi_array = self.empirical_fsd_mu_hi[gene_index_array]
        empirical_mean_obs_expr_per_gene_array = self.mean_obs_expr_per_gene[gene_index_array]
        cell_features_array = self.feature_z_scores_per_cell[cell_index_array, :]

        return {
            'cell_index_tensor': torch.tensor(cell_index_array, device=self.device),
            'gene_index_tensor': torch.tensor(gene_index_array, device=self.device),
            'total_obs_reads_per_cell_tensor': torch.tensor(
                total_obs_reads_per_cell_array.astype(np.int), device=self.device, dtype=self.dtype),
            'empirical_mean_obs_expr_per_gene_tensor': torch.tensor(
                empirical_mean_obs_expr_per_gene_array, device=self.device, dtype=self.dtype),
            'fingerprint_tensor': torch.tensor(
                fingerprint_array.astype(np.int), device=self.device, dtype=self.dtype),
            'empirical_fsd_mu_hi_tensor': torch.tensor(
                empirical_fsd_mu_hi_array, device=self.device, dtype=self.dtype),
            'cell_features_tensor': torch.tensor(
                cell_features_array, device=self.device, dtype=self.dtype),
            'cell_sampling_site_scale_factor_tensor': torch.tensor(
                cell_sampling_site_scale_factor_array, device=self.device, dtype=self.dtype),
            'gene_sampling_site_scale_factor_tensor': torch.tensor(
                gene_sampling_site_scale_factor_array, device=self.device, dtype=self.dtype),
            'downsampling_rate_tensor': torch.ones(mb_size, device=self.device, dtype=self.dtype),
            'fingerprint_obs_log_prob_prefactor': 1.0}

    def generate_stratified_sample_torch(
            self,
            mb_genes_per_gene_group: int,
            mb_expressing_cells_per_gene: int,
            mb_silent_cells_per_gene: int) -> Dict[str, torch.Tensor]:
        sample_dict = self._generate_stratified_sample(
            mb_genes_per_gene_group,
            mb_expressing_cells_per_gene,
            mb_silent_cells_per_gene)
        return self.generate_torch_minibatch_data(
            cell_index_array=sample_dict['cell_index_array'],
            gene_index_array=sample_dict['gene_index_array'],
            cell_sampling_site_scale_factor_array=sample_dict['cell_sampling_site_scale_factor_array'],
            gene_sampling_site_scale_factor_array=sample_dict['gene_sampling_site_scale_factor_array'])


def downsample_single_fingerprint_numpy(fingerprint_array: np.ndarray, downsampling_rate: float):
    assert fingerprint_array.ndim == 1
    assert 0. <= downsampling_rate <= 1.0

    max_family_size = fingerprint_array.shape[0]
    max_count = int(np.max(fingerprint_array))

    # generate the maximum number of required binomial samples
    binomial_samples = np.random.binomial(
        n=np.arange(1, max_family_size + 1)[None, :].repeat(max_count, 0),
        p=downsampling_rate)

    # generate a mask for which samples to keep
    keep_mask = (fingerprint_array - np.arange(0, max_count)[:, None]) > 0

    # calculate the histogram of downsampled reads (neglecting 0)
    downsampled_fingerprint_array = np.histogram(
        binomial_samples[keep_mask],
        range=(1, max_family_size),
        bins=max_family_size)[0]

    return downsampled_fingerprint_array


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
