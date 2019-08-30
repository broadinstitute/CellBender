import numpy as np
from typing import Tuple, List, Dict, Union, Any, Callable, Generator, Optional
from abc import abstractmethod

import torch
from torch.nn.parameter import Parameter

from pyro_extras import logit, get_log_prob_compl, logaddexp
from fingerprint import SingleCellFingerprintDTM


class GeneExpressionPrior(torch.nn.Module):
    def __init__(self):
        super(GeneExpressionPrior, self).__init__()

    @abstractmethod
    def forward(self,
                gene_index_tensor_n: torch.Tensor,
                cell_index_tensor_n: torch.Tensor,
                cell_features_nf: Optional[torch.Tensor],
                total_obs_reads_per_cell_tensor_n: Optional[torch.Tensor],
                downsampling_rate_tensor_n: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class GeneLevelGeneExpressionPrior(GeneExpressionPrior):
    EPS = 1e-6

    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 init_cell_size_beta: float = 0.5,
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float):
        super(GeneLevelGeneExpressionPrior, self).__init__()
        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.device = device
        self.dtype = dtype
        self.log_mean_total_reads_per_cell: float = np.log(
            np.mean(sc_fingerprint_dtm.total_obs_reads_per_cell)).item()

        init_log_mu_e_hi_g = torch.log(torch.tensor(
            self.EPS + sc_fingerprint_dtm.empirical_mu_e_hi,
            device=device,
            dtype=dtype))
        init_log_phi_e_hi_g = torch.log(torch.tensor(
            self.EPS + sc_fingerprint_dtm.empirical_phi_e_hi,
            device=device,
            dtype=dtype))
        init_logit_p_zero_e_hi_g = logit(torch.tensor(
            sc_fingerprint_dtm.empirical_p_zero_e_hi,
            device=device,
            dtype=dtype))
        self.global_prior_params_gr = torch.nn.Parameter(
            torch.cat((
                init_log_mu_e_hi_g.unsqueeze(-1),
                init_log_phi_e_hi_g.unsqueeze(-1),
                init_logit_p_zero_e_hi_g.unsqueeze(-1)), dim=-1))
        init_logit_cell_size_beta = np.log(init_cell_size_beta) - np.log(1 - init_cell_size_beta)
        self.logit_cell_size_beta = torch.nn.Parameter(
            torch.tensor([init_logit_cell_size_beta], device=device, dtype=dtype))

        # send parameters to device
        self.to(device)

    def forward(self,
                gene_index_tensor_n: torch.Tensor,
                cell_index_tensor_n: torch.Tensor,
                cell_features_nf: Optional[torch.Tensor],
                total_obs_reads_per_cell_tensor_n: Optional[torch.Tensor],
                downsampling_rate_tensor_n: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # calculate cell size correction strength
        log_cell_size_beta = torch.nn.functional.logsigmoid(self.logit_cell_size_beta)
        log_cell_size_alpha = torch.nn.functional.logsigmoid(-self.logit_cell_size_beta)
        log_mu_prefactor_n = logaddexp(
            log_cell_size_alpha,
            log_cell_size_beta
            + total_obs_reads_per_cell_tensor_n.log()
            - self.log_mean_total_reads_per_cell
            - downsampling_rate_tensor_n.log())

        return {
            'log_mu_e_hi_n': log_mu_prefactor_n + self.global_prior_params_gr[gene_index_tensor_n, 0],
            'log_phi_e_hi_n': self.global_prior_params_gr[gene_index_tensor_n, 1],
            'logit_p_zero_e_hi_n': self.global_prior_params_gr[gene_index_tensor_n, 2]}


# class SingleCellFeaturePredictedGeneExpressionPrior(GeneLevelGeneExpressionPrior):
#     EPS = 1e-6
#
#     def __init__(self,
#                  sc_fingerprint_dtm: SingleCellFingerprintDTM,
#                  intermediate_dim: int = 1,
#                  final_hidden_dims: Tuple[int] = (),
#                  init_cell_feature_weight: float = 0.1,
#                  hidden_activation: torch.nn.Module = torch.nn.SELU(),
#                  device: torch.device = torch.device('cuda'),
#                  dtype: torch.dtype = torch.float):
#         super(SingleCellFeaturePredictedGeneExpressionPrior, self).__init__(
#             sc_fingerprint_dtm=sc_fingerprint_dtm,
#             device=device,
#             dtype=dtype)
#         assert intermediate_dim >= 1
#
#         self.hidden_activation = hidden_activation
#
#         # initial batch norm layer
#         n_input_features = sc_fingerprint_dtm.feature_z_scores_per_cell.shape[1]
#
#         # setup the intermediate cell-feature-based readout weight
#         xavier_scale = 1. / np.sqrt(n_input_features)
#         self.intermediate_gene_readout_weight_fgh = torch.nn.Parameter(
#             xavier_scale * torch.randn(
#                 (n_input_features, sc_fingerprint_dtm.n_genes, intermediate_dim),
#                 device=device,
#                 dtype=dtype))
#         self.intermediate_gene_readout_bias_gh = torch.nn.Parameter(
#             xavier_scale * torch.randn(
#                 (sc_fingerprint_dtm.n_genes, intermediate_dim),
#                 device=device,
#                 dtype=dtype))
#
#         # initialize the first channel to the truncated SVD decoder (of the transformed empirical expression)
#         svd_components_fg = sc_fingerprint_dtm.svd_feature_components
#         svd_loadings_nf = sc_fingerprint_dtm.svd_feature_loadings_per_cell
#         svd_mean_loadings_f = np.mean(svd_loadings_nf, 0)
#         svd_std_loadings_f = np.std(svd_loadings_nf, 0)
#         svd_decoder_bias_g = np.dot(svd_components_fg.T, svd_mean_loadings_f)
#         svd_decoder_weights_hg = np.zeros((n_input_features, sc_fingerprint_dtm.n_genes))
#         svd_decoder_weights_hg[:sc_fingerprint_dtm.n_pca_features, :] = (
#                 svd_std_loadings_f[:, None] * svd_components_fg)
#
#         self.intermediate_gene_readout_weight_fgh.data[:, :, 0].copy_(
#             torch.tensor(svd_decoder_weights_hg, device=device, dtype=dtype))
#         self.intermediate_gene_readout_bias_gh.data[:, 0].copy_(
#             torch.tensor(svd_decoder_bias_g, device=device, dtype=dtype))
#
#         final_hidden_dims += (1,)
#         self.final_layers = torch.nn.ModuleList()
#         last_dim = intermediate_dim
#         for hidden_dim in final_hidden_dims:
#             layer = torch.nn.Linear(last_dim, hidden_dim, bias=True)
#             last_dim = hidden_dim
#             self.final_layers.append(layer)
#
#         # global parameters
#
#         # initialize according to init_cell_feature_weight
#         init_logit_p_conf = np.log(init_cell_feature_weight) - np.log(1 - init_cell_feature_weight)
#         self.logit_p_conf_g = torch.nn.Parameter(
#             init_logit_p_conf * torch.ones(
#                 (sc_fingerprint_dtm.n_genes,), device=device, dtype=dtype))
#
#         # initialize to the global value
#         self.log_pred_phi_g = torch.nn.Parameter(self.global_prior_params_gr[:, 1].clone())
#
#         # send parameters to device
#         self.to(device)
#
#     @staticmethod
#     def interp_global_predicted(log_global_mu_n: torch.Tensor,
#                                 log_global_phi_n: torch.Tensor,
#                                 logit_global_p_zero_n: torch.Tensor,
#                                 log_pred_mu_n: torch.Tensor,
#                                 log_pred_phi_n: torch.Tensor,
#                                 logit_p_conf_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         # calculate aux quantities
#         log_global_p_zero_n = torch.nn.functional.logsigmoid(logit_global_p_zero_n)
#         log_global_p_nonzero_n = get_log_prob_compl(log_global_p_zero_n)
#         log_p_conf_n = torch.nn.functional.logsigmoid(logit_p_conf_n)
#         log_p_nonconf_n = get_log_prob_compl(log_p_conf_n)
#
#         # numerically stable interpolation between ZIG and G parameters
#         log_mu_eff_n = (
#                 logaddexp(
#                     log_p_nonconf_n + log_global_p_nonzero_n + log_global_mu_n,
#                     log_p_conf_n + log_pred_mu_n)
#                 - logaddexp(
#                     log_global_p_nonzero_n,
#                     log_p_conf_n + log_global_p_zero_n))
#         log_phi_eff_n = logaddexp(
#             log_p_conf_n + log_pred_phi_n,
#             log_p_nonconf_n + log_global_phi_n)
#         logit_p_zero_eff_n = logit_global_p_zero_n - torch.log1p(
#             logit_p_conf_n.exp() * (1 + logit_global_p_zero_n.exp()))
#
#         return log_mu_eff_n, log_phi_eff_n, logit_p_zero_eff_n
#
#     def forward(self,
#                 gene_index_tensor_n: torch.Tensor,
#                 cell_index_tensor_n: torch.Tensor,
#                 cell_features_nf: Optional[torch.Tensor],
#                 total_obs_reads_per_cell_tensor_n: Optional[torch.Tensor],
#                 downsampling_rate_tensor_n: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Estimate cell-specific ZINB expression parameters."""
#
#         # apply the gene-specific linear transformation
#         intermediate_nh = (
#                 torch.einsum(
#                     "nf,fnh->nh",
#                     [cell_features_nf,
#                      self.intermediate_gene_readout_weight_fgh[:, gene_index_tensor_n, :]])
#                 + self.intermediate_gene_readout_bias_gh[gene_index_tensor_n, :])
#
#         # get a prediction and confidence
#         pred_nh = intermediate_nh
#         for layer in self.final_layers:
#             pred_nh = layer.forward(self.hidden_activation(pred_nh))
#
#         log_pred_mu_n = pred_nh[:, 0]
#         log_pred_phi_n = self.log_pred_phi_g[gene_index_tensor_n]
#         logit_p_conf_n = self.logit_p_conf_g[gene_index_tensor_n]
#
#         log_global_mu_n = self.global_prior_params_gr[gene_index_tensor_n, 0]
#         log_global_phi_n = self.global_prior_params_gr[gene_index_tensor_n, 1]
#         logit_global_p_zero_n = self.global_prior_params_gr[gene_index_tensor_n, 2]
#
#         log_mu_eff_n, log_phi_eff_n, logit_p_zero_eff_n = self.interp_global_predicted(
#             log_global_mu_n=log_global_mu_n,
#             log_global_phi_n=log_global_phi_n,
#             logit_global_p_zero_n=logit_global_p_zero_n,
#             log_pred_mu_n=log_pred_mu_n,
#             log_pred_phi_n=log_pred_phi_n,
#             logit_p_conf_n=logit_p_conf_n)
#
#         return {
#             'log_mu_e_hi_n': log_mu_eff_n,
#             'log_phi_e_hi_n': log_phi_eff_n,
#             'logit_p_zero_e_hi_n': logit_global_p_zero_n}


class SingleCellFeaturePredictedGeneExpressionPriorNew(GeneExpressionPrior):
    EPS = 1e-6

    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 hidden_dims: Tuple[int] = (50,),
                 hidden_activation: torch.nn.Module = torch.nn.SELU(),
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float):
        super(SingleCellFeaturePredictedGeneExpressionPriorNew, self).__init__()

        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.hidden_activation = hidden_activation
        self.device = device
        self.dtype = dtype

        # hidden layers
        n_input_features = sc_fingerprint_dtm.feature_z_scores_per_cell.shape[1]
        self.hidden_layers = torch.nn.ModuleList()
        last_dim = n_input_features
        for hidden_dim in hidden_dims:
            layer = torch.nn.Linear(last_dim, hidden_dim, bias=True)
            last_dim = hidden_dim
            self.hidden_layers.append(layer)

        # setup the final cell-feature-based readout weight
        xavier_scale = 1. / np.sqrt(n_input_features)
        self.readout_weight_hg = torch.nn.Parameter(
            xavier_scale * torch.randn(
                (last_dim, sc_fingerprint_dtm.n_genes),
                device=device, dtype=dtype))
        self.readout_bias_g = torch.nn.Parameter(
            xavier_scale * torch.randn(
                (sc_fingerprint_dtm.n_genes,),
                device=device, dtype=dtype))

        # global parameters
        self.log_phi_e_hi_g = torch.nn.Parameter(
            torch.log(torch.tensor(
                self.EPS + sc_fingerprint_dtm.empirical_phi_e_hi,
                device=device, dtype=dtype)))
        self.logit_p_zero_e_hi_g = torch.nn.Parameter(
            logit(torch.tensor(
                sc_fingerprint_dtm.empirical_p_zero_e_hi,
                device=device, dtype=dtype)))

        self.to(device)

    def forward(self,
                gene_index_tensor_n: torch.Tensor,
                cell_index_tensor_n: torch.Tensor,
                cell_features_nf: Optional[torch.Tensor],
                total_obs_reads_per_cell_tensor_n: Optional[torch.Tensor],
                downsampling_rate_tensor_n: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Estimate cell-specific ZINB expression parameters."""

        activations_nh = cell_features_nf
        for layer in self.hidden_layers:
            activations_nh = self.hidden_activation(layer.forward(activations_nh))

        # gene-specific linear readout of expression rate
        log_pred_mu_n = (
                torch.einsum(
                    "nh,hn->n",
                    [activations_nh,
                     self.readout_weight_hg[:, gene_index_tensor_n]])
                + self.readout_bias_g[gene_index_tensor_n])

        return {
            'log_mu_e_hi_n': log_pred_mu_n,
            'log_phi_e_hi_n': self.log_phi_e_hi_g[gene_index_tensor_n],
            'logit_p_zero_e_hi_n': self.logit_p_zero_e_hi_g[gene_index_tensor_n]}
