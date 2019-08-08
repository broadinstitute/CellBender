import numpy as np
import scipy.sparse as sp
import operator
import pickle
import logging
from typing import Tuple, List, Union, Dict, Callable
import torch
from tqdm import tqdm
from stats import ApproximateZINBFit


class SingleCellFingerprint:
    EPS = np.finfo(np.float).eps

    def __init__(self,
                 gene_idx_list: List[int],
                 max_family_size: int,
                 barcode_list: Union[None, List[int]] = None,
                 csr_fingerprint_list: Union[None, List[sp.csr_matrix]] = None):
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
        self._logger = logging.getLogger()
        
        # populate
        for barcode, csr_fingerprint_list in zip(barcode_list, csr_fingerprint_list):
            self.add(barcode, csr_fingerprint_list)

    def add(self, barcode: int, csr_fingerprint: sp.csr_matrix):
        assert barcode not in self.csr_fingerprint_dict, \
            f"Cell barcode {barcode} already has a fingerprint!"
        assert csr_fingerprint.shape[0] == len(self.gene_idx_list), \
            f"The fingerprint matrix must has as many rows ({csr_fingerprint.shape[0]}) as the number "\
            f"of genes ({len(self.gene_idx_list)})!"
        assert csr_fingerprint.shape[1] == self.max_family_size + 1, \
            f"The fingerprint matrix must has as many columns ({csr_fingerprint.shape[1]}) as the maximum "\
            f"family size + 1 ({self.max_family_size + 1})!"
        self.csr_fingerprint_dict[barcode] = csr_fingerprint
        self.barcode_list.append(barcode)

    def __getitem__(self, barcode: int) -> sp.csr_matrix:
        return self.csr_fingerprint_dict[barcode]

    def save(self, output_path: str):
        with open(output_path, 'wb') as f:
            pickle.dump(self.gene_idx_list, f)
            pickle.dump(self.csr_fingerprint_dict, f)
    
    @staticmethod
    def load(input_path: str) -> 'SingleCellFingerprint':
        with open(input_path, 'rb') as f:
            loader = pickle.Unpickler(f)
            gene_idx_list = loader.load()
            csr_fingerprint_dict = loader.load()
        
        max_family_size = csr_fingerprint_dict.items().__iter__().__next__()[1].shape[1] - 1
        new = SingleCellFingerprint(gene_idx_list, max_family_size)
        for barcode, csr_fingerprint in csr_fingerprint_dict.items():
            new.add(barcode, csr_fingerprint)
        return new

    @property
    def num_genes(self):
        return len(self.gene_idx_list)
    
    @property
    def num_cells(self):
        return len(self.barcode_list)

    def _get_good_turing_estimator(self) -> np.ndarray:
        orphan_reads = np.zeros((self.num_genes,), dtype=np.float)
        all_reads = np.zeros((self.num_genes,), dtype=np.float)
        read_counter = np.arange(0, self.max_family_size + 1)
        for csr_fingerprint in self.csr_fingerprint_dict.values():
            orphan_reads += np.asarray(csr_fingerprint[:, 1].todense()).flatten()
            all_reads += csr_fingerprint.dot(read_counter)
        good_turing_estimator = orphan_reads / (SingleCellFingerprint.EPS + all_reads)
        good_turing_estimator[all_reads == 0] = np.nan
        return good_turing_estimator
    
    def _get_total_gene_expression(self) -> np.ndarray:
        total_gene_expression = np.zeros((self.num_genes,))
        for csr_fingerprint in self.csr_fingerprint_dict.values():
            total_gene_expression += np.asarray(np.sum(csr_fingerprint, -1)).flatten()
        return total_gene_expression
    
    def filter_genes(self,
                     max_good_turing: float = 0.5,
                     min_total_gene_expression: int = 10,
                     verbose_logging: bool = False) -> 'SingleCellFingerprint':
        # calculate summary statistics
        good_turing_estimator_array = self._get_good_turing_estimator()
        total_gene_expression_array = self._get_total_gene_expression()
        
        num_failed_good_turing = 0
        num_failed_min_expression = 0
        num_failed_both = 0
        
        kept_gene_array_idx_list = list()
        for i_gene in range(self.num_genes):
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
        
        new = SingleCellFingerprint(kept_gene_idx_list, self.max_family_size)
        for barcode, csr_fingerprint in self.csr_fingerprint_dict.items():
            new.add(barcode, csr_fingerprint[kept_gene_array_idx_list, :])
        return new    


def random_choice(a, size):
    random_indices = np.random.randint(0, len(a), size=size)
    return a[random_indices]


class SingleCellFingerprintDataStore:
    # minimum over-dispersion of an estimated negative binomial fit
    MIN_NB_PHI = 1e-2
    
    def __init__(self,
                 sc_fingerprint: SingleCellFingerprint,
                 top_k_genes: Union[None, int, Tuple[int, int]] = None,
                 gene_idx_list: Union[None, List[int]] = None,
                 max_estimated_chimera_family_size: int = 1,
                 zinb_fitter_kwargs: Union[None, Dict[str, Union[int, float]]] = None,
                 gene_grouping_trans: Callable[[np.ndarray], np.ndarray] = np.log,
                 n_gene_groups: int = 10):
        if zinb_fitter_kwargs is None:
            zinb_fitter_kwargs = dict()
        self.sc_fingerprint = sc_fingerprint
        self.max_estimated_chimera_family_size = max_estimated_chimera_family_size
        self.gene_grouping_trans = gene_grouping_trans
        self.n_gene_groups = n_gene_groups

        self.n_cells = sc_fingerprint.num_cells
        self.max_family_size = sc_fingerprint.max_family_size

        self._logger = logging.getLogger()
        
        # total observed expression of all genes
        total_e_obs_g = np.zeros((sc_fingerprint.num_genes,), dtype=np.uint64)
        for barcode in sc_fingerprint.barcode_list:
            total_e_obs_g += np.asarray(np.sum(sc_fingerprint[barcode], -1)).flatten()
        if np.any(total_e_obs_g == 0):
            self._logger.warning('Some genes in the provided fingerprint have zero expression!')
        
        # select highly expressed genes (if required)
        if top_k_genes is None and gene_idx_list is None:
            self.gene_idx_list_in_fingerprint = np.where(total_e_obs_g > 0)[0]
        elif top_k_genes is None and gene_idx_list is not None:
            fingerprint_gene_idx_map = {
                gene_index: order for order, gene_index in enumerate(sc_fingerprint.gene_idx_list)}
            self.gene_idx_list_in_fingerprint = [
                fingerprint_gene_idx_map[gene_idx] for gene_idx in gene_idx_list]
        elif top_k_genes is not None and gene_idx_list is None:
            sorted_gene_indices = np.asarray(
                list(map(operator.itemgetter(0),
                         sorted(enumerate(total_e_obs_g), key=operator.itemgetter(1), reverse=True))))
            if isinstance(top_k_genes, tuple):
                self.gene_idx_list_in_fingerprint = sorted_gene_indices[top_k_genes[0]:top_k_genes[1]]
            elif isinstance(top_k_genes, int):
                self.gene_idx_list_in_fingerprint = sorted_gene_indices[:top_k_genes]
            else:
                raise ValueError("Bad value for `top_k_genes`")
        else:
            raise ValueError("Cannot specify both 'gene_idx_list' and 'top_k_genes'!")

        self.n_genes = len(self.gene_idx_list_in_fingerprint)
        self.internal_gene_index_to_original_gene_index_map = {
            internal_gene_index: sc_fingerprint.gene_idx_list[gene_index_in_fingerprint]
            for internal_gene_index, gene_index_in_fingerprint in enumerate(self.gene_idx_list_in_fingerprint)}
        
        # ZINB fitter
        self.zinb_fitter = ApproximateZINBFit(**zinb_fitter_kwargs)
        
        # caches
        self._fingerprint_array = None
        self._obs_expr_matrix = None
        self._expressing_cells_dict = None
        self._silent_cells_dict = None
        self._num_expressing_cells = None
        self._num_silent_cells = None
        self._empirical_fsd_params = None
        self._empirical_e_hi_params = None
        self._total_obs_reads_per_cell = None
        self._total_obs_molecules_per_cell = None
        self._mean_obs_expr_per_gene = None
        self._total_obs_expr_per_gene = None
        self._gene_groups_dict = None
    
    # TODO: replace this with pydata.sparse
    @property
    def fingerprint_array(self) -> np.ndarray:
        if self._fingerprint_array is None:
            self._fingerprint_array = np.zeros((self.n_cells, self.n_genes, self.max_family_size), dtype=np.uint16)
            for i_cell, barcode in enumerate(self.sc_fingerprint.barcode_list):
                self._fingerprint_array[i_cell, :, :] = self.sc_fingerprint[barcode][
                    self.gene_idx_list_in_fingerprint, 1:].todense()
                
        return self._fingerprint_array
    
    # TODO: replace this with pydata.sparse
    @property
    def obs_expr_matrix(self) -> np.ndarray:
        if self._obs_expr_matrix is None:
            self._obs_expr_matrix = np.zeros((self.n_cells, self.n_genes), dtype=np.uint32)
            self._obs_expr_matrix[:, :] = np.sum(self.fingerprint_array, -1)
        return self._obs_expr_matrix
    
    @property
    def expressing_cells_dict(self) -> Dict[int, np.ndarray]:
        if self._expressing_cells_dict is None:
            self._expressing_cells_dict = dict()
            for i_gene in range(self.n_genes):
                self._expressing_cells_dict[i_gene] = np.where(self.obs_expr_matrix[:, i_gene] > 0)[0]
        return self._expressing_cells_dict
    
    @property
    def silent_cells_dict(self) -> Dict[int, np.ndarray]:
        if self._silent_cells_dict is None:
            self._silent_cells_dict = dict()
            for i_gene in range(self.n_genes):
                self._silent_cells_dict[i_gene] = np.where(self.obs_expr_matrix[:, i_gene] == 0)[0]
        return self._silent_cells_dict

    @property
    def num_expressing_cells(self) -> List[int]:
        if self._num_expressing_cells is None:
            self._num_expressing_cells = []
            for i_gene in range(self.n_genes):
                self._num_expressing_cells.append(len(self.expressing_cells_dict[i_gene]))
        return self._num_expressing_cells
    
    @property
    def num_silent_cells(self) -> List[int]:
        if self._num_silent_cells is None:
            self._num_silent_cells = [
                self.n_cells - self.num_expressing_cells[i_gene] for i_gene in range(self.n_genes)]
        return self._num_silent_cells
        
    @property
    def total_obs_reads_per_cell(self) -> np.ndarray:
        if self._total_obs_reads_per_cell is None:
            self._total_obs_reads_per_cell = np.zeros((self.n_cells,), dtype=np.uint64)
            family_size_vector = np.arange(0, self.max_family_size + 1)
            for i_cell, barcode in enumerate(self.sc_fingerprint.barcode_list):
                self._total_obs_reads_per_cell[i_cell] = np.sum(self.sc_fingerprint[barcode].dot(family_size_vector))
        return self._total_obs_reads_per_cell

    @property
    def total_obs_molecules_per_cell(self) -> np.ndarray:
        if self._total_obs_molecules_per_cell is None:
            self._total_obs_molecules_per_cell = np.zeros((self.n_cells,), dtype=np.uint64)
            for i_cell, barcode in enumerate(self.sc_fingerprint.barcode_list):
                self._total_obs_molecules_per_cell[i_cell] = np.sum(self.sc_fingerprint[barcode])
        return self._total_obs_molecules_per_cell
    
    @property
    def total_obs_expr_per_gene(self) -> np.ndarray:
        if self._total_obs_expr_per_gene is None:
            self._total_obs_expr_per_gene = np.sum(self.obs_expr_matrix, axis=0)
        return self._total_obs_expr_per_gene

    @property
    def mean_obs_expr_per_gene(self) -> np.ndarray:
        if self._mean_obs_expr_per_gene is None:
            self._mean_obs_expr_per_gene = self.total_obs_expr_per_gene.astype(np.float) / self.n_cells
        return self._mean_obs_expr_per_gene
    
    @property
    def gene_groups_dict(self) -> Dict[int, np.ndarray]:
        if self._gene_groups_dict is None:
            # the "weight" of each gene is a monotonic (approximately logarthimic) function of its
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

            self._gene_groups_dict = dict()
            for i_group in range(self.n_gene_groups):
                gene_group_start_index = gene_group_start_indices[i_group]
                gene_group_stop_index = gene_group_stop_indices[i_group]
                self._gene_groups_dict[i_group] = np.asarray(list(
                    map(operator.itemgetter(0),
                        sorted_genes_idx_weight[gene_group_start_index:gene_group_stop_index])))
        return self._gene_groups_dict        
    
    @property
    def empirical_fsd_params(self) -> np.ndarray:
        if self._empirical_fsd_params is None:
            self._empirical_fsd_params = np.zeros((self.n_genes, 3))
            for gene_index in range(self.n_genes):
                gene_fs_hist = np.sum(self.fingerprint_array[:, gene_index, :], 0)
                
                # "cap" the empirical hisogram as a heuristic for attenuating chimeric counts
                if self.max_estimated_chimera_family_size >= 1:
                    gene_fs_hist[:self.max_estimated_chimera_family_size] = gene_fs_hist[self.max_estimated_chimera_family_size]

                family_size_array = np.arange(1, self.max_family_size + 1)
                family_size_pmf = gene_fs_hist / np.sum(gene_fs_hist)
                family_size_mean = np.sum(family_size_array * family_size_pmf)
                family_size_var = np.sum((family_size_array ** 2) * family_size_pmf) - family_size_mean ** 2

                # calculate NB params using first two moments
                mu = family_size_mean
                phi = max(self.MIN_NB_PHI, (family_size_var - family_size_mean) / (family_size_mean ** 2))

                # calculate p_obs
                alpha = 1. / phi
                p_obs = 1. - np.exp(alpha * np.log(alpha / (alpha + mu)))
                
                self._empirical_fsd_params[gene_index, 0] = mu
                self._empirical_fsd_params[gene_index, 1] = phi
                self._empirical_fsd_params[gene_index, 2] = p_obs
                
        return self._empirical_fsd_params

    @property
    def empirical_e_hi_params(self) -> np.ndarray:
        if self._empirical_e_hi_params is None:
            # total observed expression per gene per cell
            e_obs = np.sum(self.fingerprint_array, -1)
            
            # estimated probability of observing real molecules
            p_obs = self.empirical_fsd_params[:, 2][None, :]
            
            # inflate the counts to account for p_obs
            e_hi_est = e_obs / (1e-12 + p_obs)
            
            # fit ZINB to e_hi_est
            self._empirical_e_hi_params = np.zeros((self.n_genes, 3))
            self._logger.warning("Fitting approximate ZINB to UMI-based gene expression data...")
            for gene_index in tqdm(range(self.n_genes)):
                zinb_fit = self.zinb_fitter(e_hi_est[:, gene_index])
                if not zinb_fit['converged']:
                    self._logger.warning(f'ZINB fit to gene (internal index: {gene_index}) was not successful!')
                self._empirical_e_hi_params[gene_index, 0] = zinb_fit['mu']
                self._empirical_e_hi_params[gene_index, 1] = zinb_fit['phi']
                self._empirical_e_hi_params[gene_index, 2] = zinb_fit['p_zero']
        return self._empirical_e_hi_params
    
    @property
    def empirical_mu_e_hi(self) -> np.ndarray:
        return self.empirical_e_hi_params[:, 0]
        
    @property
    def empirical_phi_e_hi(self) -> np.ndarray:
        return self.empirical_e_hi_params[:, 1]

    @property
    def empirical_p_zero_e_hi(self) -> np.ndarray:
        return self.empirical_e_hi_params[:, 2]

    @property
    def empirical_fsd_mu_hi(self) -> np.ndarray:
        return self.empirical_fsd_params[:, 0]
        
    @property
    def empirical_fsd_phi_hi(self) -> np.ndarray:
        return self.empirical_fsd_params[:, 1]

    @property
    def empirical_fsd_p_obs(self) -> np.ndarray:
        return self.empirical_fsd_params[:, 2]
    
    def _generate_stratified_sample(self,
                                    mb_genes_per_gene_group,
                                    mb_expressing_cells_per_gene,
                                    mb_silent_cells_per_gene) -> Dict[str, np.ndarray]:
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
                size_expressing = min(mb_expressing_cells_per_gene, self.num_expressing_cells[gene_index])
                expressing_cell_indices = random_choice(
                    self.expressing_cells_dict[gene_index], size_expressing).tolist()
                expressing_scale_factor = gene_scale_factor * self.num_expressing_cells[gene_index] / size_expressing

                # sample from silent cells
                size_silent = min(mb_silent_cells_per_gene, self.num_silent_cells[gene_index])
                silent_cell_indices = random_choice(self.silent_cells_dict[gene_index], size_silent)
                silent_scale_factor = gene_scale_factor * self.num_silent_cells[gene_index] / size_silent

                mb_cell_indices_per_gene.append(expressing_cell_indices)
                mb_cell_indices_per_gene.append(silent_cell_indices)
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
                                      cell_index_array,
                                      gene_index_array,
                                      cell_sampling_site_scale_factor_array,
                                      gene_sampling_site_scale_factor_array,
                                      device=torch.device("cuda"),
                                      dtype=torch.float) -> Dict[str, torch.Tensor]:
        assert cell_index_array.ndim == 1
        assert gene_index_array.ndim == 1
        assert cell_sampling_site_scale_factor_array.ndim == 1
        assert gene_sampling_site_scale_factor_array.ndim == 1
        mb_size = len(cell_index_array)
        assert len(gene_index_array) == mb_size
        assert len(cell_sampling_site_scale_factor_array) == mb_size
        assert len(gene_sampling_site_scale_factor_array) == mb_size
        
        total_obs_reads_per_cell_array = self.total_obs_reads_per_cell[cell_index_array]
        fingerprint_array = self.fingerprint_array[cell_index_array, gene_index_array, :]
        empirical_fsd_mu_hi_array = self.empirical_fsd_mu_hi[gene_index_array]
        
        return {
            'cell_index_tensor': torch.tensor(cell_index_array, device=device),
            'gene_index_tensor': torch.tensor(gene_index_array, device=device),
            'total_obs_reads_per_cell_tensor': torch.tensor(
                total_obs_reads_per_cell_array.astype(np.int), device=device, dtype=dtype),
            'fingerprint_tensor': torch.tensor(
                fingerprint_array.astype(np.int), device=device, dtype=dtype),
            'empirical_fsd_mu_hi_tensor': torch.tensor(
                empirical_fsd_mu_hi_array, device=device, dtype=dtype),
            'cell_sampling_site_scale_factor_tensor': torch.tensor(
                cell_sampling_site_scale_factor_array, device=device, dtype=dtype),
            'gene_sampling_site_scale_factor_tensor': torch.tensor(
                gene_sampling_site_scale_factor_array, device=device, dtype=dtype),
            'downsampling_rate_tensor': torch.ones(mb_size, device=device, dtype=dtype),
            'fingerprint_obs_log_prob_prefactor': 1.0}

    def generate_stratified_sample_torch(self,
                                         mb_genes_per_gene_group,
                                         mb_expressing_cells_per_gene,
                                         mb_silent_cells_per_gene,
                                         device=torch.device("cuda"),
                                         dtype=torch.float) -> Dict[str, torch.Tensor]:
        sample_dict = self._generate_stratified_sample(
            mb_genes_per_gene_group,
            mb_expressing_cells_per_gene,
            mb_silent_cells_per_gene)
        return self.generate_torch_minibatch_data(
            sample_dict['cell_index_array'],
            sample_dict['gene_index_array'],
            sample_dict['cell_sampling_site_scale_factor_array'],
            sample_dict['gene_sampling_site_scale_factor_array'],
            device, dtype)

    
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


def generate_downsampled_minibatch(original_data_dict: Dict[str, torch.Tensor],
                                   min_downsampling_rate: float,
                                   max_downsampling_rate: float) -> Dict[str, torch.Tensor]:
    downsampled_data_dict = dict()

    # copy unchanged data
    downsampled_data_dict['cell_index_tensor'] = \
        original_data_dict['cell_index_tensor']
    downsampled_data_dict['gene_index_tensor'] = \
        original_data_dict['gene_index_tensor']
    downsampled_data_dict['cell_sampling_site_scale_factor_tensor'] = \
        original_data_dict['cell_sampling_site_scale_factor_tensor']
    downsampled_data_dict['gene_sampling_site_scale_factor_tensor'] = \
        original_data_dict['gene_sampling_site_scale_factor_tensor']

    # downsampling rate tensor
    downsampled_data_dict['downsampling_rate_tensor'] = min_downsampling_rate + (
        (max_downsampling_rate - min_downsampling_rate)
        * torch.rand_like(original_data_dict['downsampling_rate_tensor']))

    # downsample the total observed reads per cell
    downsampled_data_dict['total_obs_reads_per_cell_tensor'] = torch.distributions.Binomial(
        original_data_dict['total_obs_reads_per_cell_tensor'],
        probs=downsampled_data_dict['downsampling_rate_tensor']).sample()

    # downsample the fingerprint
    downsampling_rate_numpy = downsampled_data_dict['downsampling_rate_tensor'].cpu().numpy()
    fingerprint_numpy = original_data_dict['fingerprint_tensor'].cpu().numpy().astype(np.int)
    downsampled_fingerprint_numpy = np.zeros_like(fingerprint_numpy)
    for i_cell in range(fingerprint_numpy.shape[0]):
        downsampled_fingerprint_numpy[i_cell, :] = downsample_single_fingerprint_numpy(
            fingerprint_numpy[i_cell, :],
            downsampling_rate_numpy[i_cell])
    downsampled_data_dict['fingerprint_tensor'] = torch.tensor(
        downsampled_fingerprint_numpy,
        dtype=original_data_dict['fingerprint_tensor'].dtype,
        device=original_data_dict['fingerprint_tensor'].device)
    
    # downsample the empirical fsd mu hi
    downsampled_data_dict['empirical_fsd_mu_hi_tensor'] = (
        downsampled_data_dict['downsampling_rate_tensor'] * 
        original_data_dict['empirical_fsd_mu_hi_tensor'])
    
    return downsampled_data_dict
