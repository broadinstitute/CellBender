from typing import Optional, Dict
from operator import itemgetter
import time

import numpy as np

from typing import Tuple, List, Any
from boltons.cacheutils import cachedproperty
import torch
import logging

from expr import VSGPGeneExpressionModel, VSGPGeneExpressionModelTrainer

from adjustText import adjust_text
from matplotlib.axes import Axes
from matplotlib.pylab import cm
from matplotlib import pylab

from fingerprint import SingleCellFingerprintDTM


class HighlyVariableGenesSelector:
    DEFAULT_GENE_GROUP_NAME = 'all'

    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 init_params_dict: Dict[str, Any],
                 device: torch.device = torch.device("cuda"),
                 dtype: torch.dtype = torch.float32):
        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.device = device
        self.dtype = dtype

        self.hvg_n_selected_genes: int = init_params_dict[
            'hvg.n_selected_genes']
        self.hvg_neglect_expr_bottom_fraction: float = init_params_dict[
            'hvg.neglect_expr_bottom_fraction']
        self.hvg_gene_group_prefixes: Optional[List[str]] = init_params_dict[
            'hvg.gene_group_prefixes']

        # expression model parameters
        self.hvg_expr_vsgp_n_inducing_points: int = init_params_dict[
            'hvg.expr.vsgp.n_inducing_points']
        self.hvg_expr_vsgp_init_rbf_kernel_variance: int = init_params_dict[
            'hvg.expr.vsgp.init_rbf_kernel_variance']
        self.hvg_expr_vsgp_init_rbf_kernel_lengthscale: int = init_params_dict[
            'hvg.expr.vsgp.init_rbf_kernel_lengthscale']
        self.hvg_expr_vsgp_init_whitenoise_kernel_variance: int = init_params_dict[
            'hvg.expr.vsgp.init_whitenoise_kernel_variance']
        self.hvg_expr_vsgp_init_posterior_scale: int = init_params_dict[
            'hvg.expr.vsgp.init_posterior_scale']
        self.hvg_expr_vsgp_init_mean_intercept: np.ndarray = init_params_dict[
            'hvg.expr.vsgp.init_mean_intercept']
        self.hvg_expr_vsgp_init_mean_slope: np.ndarray = init_params_dict[
            'hvg.expr.vsgp.init_mean_slope']
        self.hvg_expr_vsgp_cholesky_jitter: int = init_params_dict[
            'hvg.expr.vsgp.cholesky_jitter']
        self.hvg_expr_vsgp_min_noise: int = init_params_dict[
            'hvg.expr.vsgp.min_noise']

        # expression model training parameters
        self.hvg_expr_trainer_adam_lr: float = init_params_dict[
            'hvg.expr.trainer.adam_lr']
        self.hvg_expr_trainer_adam_betas: Tuple[float, float] = init_params_dict[
            'hvg.expr.trainer.adam_betas']
        self.hvg_expr_trainer_n_training_iterations: int = init_params_dict[
            'hvg.expr.trainer.n_training_iterations']
        self.hvg_expr_trainer_training_log_frequency: int = init_params_dict[
            'hvg.expr.trainer.training_log_frequency']
        self.hvg_expr_trainer_minibatch_genes_per_gene_group: int = init_params_dict[
            'hvg.expr.trainer.minibatch.genes_per_gene_group']
        self.hvg_expr_trainer_minibatch_expressing_cells_per_gene: int = init_params_dict[
            'hvg.expr.trainer.minibatch.expressing_cells_per_gene']
        self.hvg_expr_trainer_minibatch_silent_cells_per_gene: int = init_params_dict[
            'hvg.expr.trainer.minibatch.silent_cells_per_gene']
        self.hvg_expr_trainer_minibatch_sampling_strategy: str = init_params_dict[
            'hvg.expr.trainer.minibatch.sampling_strategy']

        if self.hvg_gene_group_prefixes is None:
            self.gene_group_internal_indices_dict: Dict[str, np.ndarray] = {
                HighlyVariableGenesSelector.DEFAULT_GENE_GROUP_NAME: np.arange(sc_fingerprint_dtm.n_genes)
            }
        else:
            self.gene_group_internal_indices_dict: Dict[str, np.ndarray] = {
                prefix: sc_fingerprint_dtm.sc_fingerprint_base.get_internal_gene_indices_by_gene_name_prefix(prefix)
                for prefix in self.hvg_gene_group_prefixes
            }

            prev_indices = set()
            for prefix, indices in self.gene_group_internal_indices_dict.items():
                assert len(indices) > 0, \
                    f"No genes could be associated to group '{prefix}' -- please either remove this"\
                    f" group or assert that the prefix is specified correctly."
                assert len(set(indices).intersection(prev_indices)) == 0, \
                    "The provided gene group prefixes do not result in mutually exclusive group of genes!"
                prev_indices = prev_indices.union(set(indices))

            if len(prev_indices) < sc_fingerprint_dtm.n_genes:
                logging.warning("Some of the genes in the fingerprint do not belong to any of the gene groups "
                                "specified for highly variable gene selection!")

        self.n_groups = len(self.gene_group_internal_indices_dict)
        self.gene_group_names = list(self.gene_group_internal_indices_dict.keys())

        self.expr_model_dict: Dict[str, VSGPGeneExpressionModel] = dict()
        self.expr_model_trainer_dict: Dict[str, VSGPGeneExpressionModelTrainer] = dict()
        self.grouped_sc_fingerprint_dtm_dict: Dict[str, SingleCellFingerprintDTM] = dict()

        for gene_group_name, gene_group_fingerprint_indices in self.gene_group_internal_indices_dict.items():

            # partition the fingerprint into gene groups if n_groups > 1
            if self.n_groups > 1:
                grouped_sc_fingerprint_dtm = sc_fingerprint_dtm.new_from(
                    other_sc_fingerprint_base=sc_fingerprint_dtm.sc_fingerprint_base.subset_genes(
                        [sc_fingerprint_dtm.sc_fingerprint_base.gene_idx_list[j]
                         for j in gene_group_fingerprint_indices]))

                # workaround to make sure that all fingerprint share the same empirical droplet
                # efficiency regardless of the included genes
                grouped_sc_fingerprint_dtm.empirical_droplet_efficiency = \
                    sc_fingerprint_dtm.empirical_droplet_efficiency

            else:
                # no need to partition
                grouped_sc_fingerprint_dtm = sc_fingerprint_dtm

            expr_model = VSGPGeneExpressionModel(
                sc_fingerprint_dtm=grouped_sc_fingerprint_dtm,
                gene_group_name=gene_group_name,
                gene_group_indices=gene_group_fingerprint_indices,
                n_inducing_points=self.hvg_expr_vsgp_n_inducing_points,
                init_rbf_kernel_variance=self.hvg_expr_vsgp_init_rbf_kernel_variance,
                init_rbf_kernel_lengthscale=self.hvg_expr_vsgp_init_rbf_kernel_lengthscale,
                init_whitenoise_kernel_variance=self.hvg_expr_vsgp_init_whitenoise_kernel_variance,
                init_posterior_scale=self.hvg_expr_vsgp_init_posterior_scale,
                init_mean_intercept=self.hvg_expr_vsgp_init_mean_intercept,
                init_mean_slope=self.hvg_expr_vsgp_init_mean_slope,
                cholesky_jitter=self.hvg_expr_vsgp_cholesky_jitter,
                min_noise=self.hvg_expr_vsgp_min_noise,
                device=self.device,
                dtype=self.dtype)

            expr_model_trainer = VSGPGeneExpressionModelTrainer(
                vsgp_gene_expression_model=expr_model,
                sc_fingerprint_dtm=grouped_sc_fingerprint_dtm,
                adam_lr=self.hvg_expr_trainer_adam_lr,
                adam_betas=self.hvg_expr_trainer_adam_betas)

            # add to containers
            self.grouped_sc_fingerprint_dtm_dict[gene_group_name] = grouped_sc_fingerprint_dtm
            self.expr_model_dict[gene_group_name] = expr_model
            self.expr_model_trainer_dict[gene_group_name] = expr_model_trainer

            # train
            expr_model_trainer.run_training(
                n_training_iterations=self.hvg_expr_trainer_n_training_iterations,
                training_log_frequency=self.hvg_expr_trainer_training_log_frequency,
                minibatch_genes_per_gene_group=self.hvg_expr_trainer_minibatch_genes_per_gene_group,
                minibatch_expressing_cells_per_gene=self.hvg_expr_trainer_minibatch_expressing_cells_per_gene,
                minibatch_silent_cells_per_gene=self.hvg_expr_trainer_minibatch_silent_cells_per_gene,
                minibatch_sampling_strategy=self.hvg_expr_trainer_minibatch_sampling_strategy)

    @cachedproperty
    def log_pearson_residual_std_per_group(self) -> Dict[str, np.ndarray]:
        log_pearson_residual_std_per_group_dict: Dict[str, np.ndarray] = dict()
        for gene_group_name in self.gene_group_names:

            expr_model = self.expr_model_dict[gene_group_name]
            grouped_sc_fingerprint_dtm = self.grouped_sc_fingerprint_dtm_dict[gene_group_name]

            with torch.no_grad():
                expr_model.vsgp.set_data(
                    X=expr_model.log_mean_obs_expr_g1,
                    y=None)
                beta_loc_rg, _ = expr_model.vsgp.model()
                beta_loc_gr = beta_loc_rg.permute(-1, -2)

            log_eta_n = torch.tensor(
                grouped_sc_fingerprint_dtm.empirical_droplet_efficiency,
                device=self.device, dtype=self.dtype).log()

            log_pearson_res_g = np.zeros((grouped_sc_fingerprint_dtm.n_genes,))
            for gene_index in range(grouped_sc_fingerprint_dtm.n_genes):
                # get FST counts for gene_index
                fst_counts_n = torch.tensor(
                    grouped_sc_fingerprint_dtm.dense_family_size_truncated_count_matrix[:, gene_index],
                    device=self.device, dtype=self.dtype)

                # calculate NB prior loc and scale for each particle
                prior_loc_n = (beta_loc_gr[gene_index, 0] + log_eta_n).exp()
                prior_scale_n = (prior_loc_n + beta_loc_gr[gene_index, 1].exp() * prior_loc_n.pow(2)).sqrt()

                # calculate the one-sided p-value of having excess variance for each particle
                pearson_res_n = (fst_counts_n - prior_loc_n) / prior_scale_n
                log_pearson_res_g[gene_index] = torch.std(pearson_res_n).log().item()

            log_pearson_residual_std_per_group_dict[gene_group_name] = log_pearson_res_g

        return log_pearson_residual_std_per_group_dict

    @cachedproperty
    def highly_variable_gene_indices_per_group(self) -> Dict[str, List[int]]:
        """

        .. note:: the returned indices corresponds to within-group internal indices (not indices
        in the master fingerprint container).

        :return:
        """
        highly_variable_gene_indices_per_group_dict: Dict[str, List[int]] = dict()
        for gene_group_name, log_pearson_residual_std_g in self.log_pearson_residual_std_per_group.items():
            group_n_genes = self.grouped_sc_fingerprint_dtm_dict[gene_group_name].n_genes
            sorted_cutoff_index = int(np.floor(self.hvg_neglect_expr_bottom_fraction * group_n_genes))
            x_data = self.expr_model_dict[gene_group_name].log_mean_obs_expr_g1.detach().cpu().numpy()[:, 0]
            x_cutoff = np.sort(x_data)[sorted_cutoff_index]
            bottom_removed_gene_indices = [
                gene_index for gene_index in range(group_n_genes)
                if x_data[gene_index] >= x_cutoff]
            indexed_log_pearson_residual_std_g = [
                (gene_index, log_pearson_residual_std_g[gene_index])
                for gene_index in bottom_removed_gene_indices]
            group_highly_variable_gene_indices = list(
                map(itemgetter(0),
                    sorted(indexed_log_pearson_residual_std_g,
                           key=itemgetter(1), reverse=True)))
            highly_variable_gene_indices_per_group_dict[gene_group_name] = group_highly_variable_gene_indices
        return highly_variable_gene_indices_per_group_dict

    @cachedproperty
    def highly_variable_gene_indices(self) -> List[int]:
        highly_variable_gene_per_group = int(np.ceil(self.hvg_n_selected_genes / self.n_groups))
        highly_variable_gene_indices = []
        for gene_group_name, log_pearson_residual_std_g in self.log_pearson_residual_std_per_group.items():
            highly_variable_gene_indices_in_group = \
                self.highly_variable_gene_indices_per_group[gene_group_name]
            n_selected_in_group = min(highly_variable_gene_per_group, len(highly_variable_gene_indices_in_group))
            selected_gene_indices_in_group = highly_variable_gene_indices_in_group[:n_selected_in_group]
            gene_indices_in_group = self.gene_group_internal_indices_dict[gene_group_name]
            highly_variable_gene_indices_in_main_fingerprint = [
                gene_indices_in_group[i] for i in selected_gene_indices_in_group]
            highly_variable_gene_indices += highly_variable_gene_indices_in_main_fingerprint
        return highly_variable_gene_indices

    def plot_highly_variable_genes(self, gene_group_name: str, ax: Axes, top_n_annotate=10):
        residual_log_std = self.log_pearson_residual_std_per_group[gene_group_name]
        highly_variable_gene_indices_in_group = \
            self.highly_variable_gene_indices_per_group[gene_group_name]
        indices_to_annotate = highly_variable_gene_indices_in_group[
                              :min(top_n_annotate, len(highly_variable_gene_indices_in_group))]

        data_x = self.expr_model_dict[gene_group_name].log_mean_obs_expr_g1.detach().cpu().numpy()[:, 0]
        gene_names_list_in_group = self.grouped_sc_fingerprint_dtm_dict[gene_group_name]\
            .sc_fingerprint_base.gene_names_list
        data_y = residual_log_std

        # shade out lowly expressed genes that are not included in HVG analysis
        group_n_genes = self.grouped_sc_fingerprint_dtm_dict[gene_group_name].n_genes
        sorted_cutoff_index = int(np.floor(self.hvg_neglect_expr_bottom_fraction * group_n_genes))
        x_cutoff = np.sort(data_x)[sorted_cutoff_index]
        colors = np.zeros((len(data_x), 4))
        colors[:, 3] = 0.5
        colors[data_x < x_cutoff, 0] = 0.5

        # make scatter plot
        ax.scatter(data_x, data_y, s=10, c=colors)

        # add gene names labels
        labels = []
        for gene_index in indices_to_annotate:
            labels.append(
                ax.text(data_x[gene_index], data_y[gene_index], gene_names_list_in_group[gene_index], fontsize=10))
        adjust_text(labels, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

        # add axis labels
        ax.set_xlabel('log mean expression', fontsize=14)
        ax.set_ylabel('log residual std expression', fontsize=14)

    def plot_diagnostics(self, plt: pylab):
        fig, axs_matrix = plt.subplots(nrows=self.n_groups, ncols=3, figsize=(16, self.n_groups * 4))
        if self.n_groups == 1:
            axs_matrix = axs_matrix[None, :]
        for gene_group_name, axs in zip(self.gene_group_internal_indices_dict.keys(), axs_matrix):
            self.expr_model_trainer_dict[gene_group_name].plot_diagnostics(axs)
        plt.tight_layout()

        # HVG selection
        for gene_group_name in self.gene_group_internal_indices_dict.keys():
            plt.figure(figsize=(8, 8))
            ax = plt.gca()
            self.plot_highly_variable_genes(gene_group_name, ax)
            plt.tight_layout()
