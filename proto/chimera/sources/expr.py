import numpy as np
from typing import Tuple, List, Dict, Union, Any, Callable, Generator, Optional
from abc import abstractmethod

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib import autoname
from pyro.contrib.gp.models import VariationalSparseGP
import pyro.contrib.gp.kernels as kernels

from matplotlib import pylab

from pyro_extras import ZeroInflatedNegativeBinomial
from fingerprint import SingleCellFingerprintDTM


class GeneExpressionPrior(torch.nn.Module):
    def __init__(self):
        super(GeneExpressionPrior, self).__init__()

    @abstractmethod
    def model(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def guide(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class VSGPGeneExpressionPrior(GeneExpressionPrior):
    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 n_inducing_points: int,
                 init_rbf_kernel_variance: float,
                 init_rbf_kernel_lengthscale: float,
                 init_linear_kernel_variance: float,
                 init_whitenoise_kernel_variance: float,
                 init_constant_kernel_variance: float,
                 init_beta_mean: np.ndarray,
                 cholesky_jitter: float,
                 min_noise: float,
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float):
        super(VSGPGeneExpressionPrior, self).__init__()
        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.device = device
        self.dtype = dtype

        # feature space
        self.log_mean_obs_expr_g = torch.log(
            torch.tensor(sc_fingerprint_dtm.mean_obs_expr_per_gene, device=device, dtype=dtype))

        # inducing points
        self.inducing_points = torch.linspace(
            torch.min(self.log_mean_obs_expr_g),
            torch.max(self.log_mean_obs_expr_g),
            steps=n_inducing_points,
            device=device, dtype=dtype)

        # GP kernel setup
        input_dim = 1
        
        kernel_rbf = kernels.RBF(
            input_dim=input_dim,
            variance=torch.tensor(init_rbf_kernel_variance, device=device, dtype=dtype),
            lengthscale=torch.tensor(init_rbf_kernel_lengthscale, device=device, dtype=dtype))
        
        kernel_linear = kernels.Linear(
            input_dim=input_dim,
            variance=torch.tensor(init_linear_kernel_variance, device=device, dtype=dtype))
        
        kernel_constant = kernels.Constant(
            input_dim=input_dim,
            variance=torch.tensor(init_constant_kernel_variance, device=device, dtype=dtype))

        kernel_whitenoise = kernels.WhiteNoise(
            input_dim=input_dim,
            variance=torch.tensor(init_whitenoise_kernel_variance, device=device, dtype=dtype))
        kernel_whitenoise.set_constraint("variance", constraints.greater_than(min_noise))
        
        kernel_full = kernels.Sum(
            kernel_rbf,
            kernels.Sum(
                kernel_linear,
                kernels.Sum(
                    kernel_whitenoise,
                    kernel_constant)))

        # mean subtraction
        self.f_mean = torch.nn.Parameter(
            torch.tensor(init_beta_mean, device=device, dtype=dtype).unsqueeze(-1))

        # instantiate VSGP model
        self.vsgp = VariationalSparseGP(
            X=self.log_mean_obs_expr_g,
            y=None,
            kernel=kernel_full,
            Xu=self.inducing_points,
            likelihood=None,
            mean_function=lambda x: self.f_mean,
            latent_shape=torch.Size([4]),
            whiten=True,
            jitter=cholesky_jitter)
        
        # send parameters to device
        self.to(device)

    def model(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        log_mean_obs_expr_n = data['empirical_mean_obs_expr_per_gene_tensor'].log()
        gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']

        # sample all points
        self.vsgp.set_data(X=log_mean_obs_expr_n, y=None)
        beta_loc_rn, beta_var_rn = autoname.scope(prefix="EXPR", fn=self.vsgp.model)()
        beta_loc_nr = beta_loc_rn.permute(-1, -2)
        beta_scale_nr = beta_var_rn.permute(-1, -2).sqrt()

        # sample e_hi prior parameters
        with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
            # sample beta parameters
            beta_nr = pyro.sample(
                "beta_nr",
                dist.Normal(loc=beta_loc_nr, scale=beta_scale_nr).to_event(1))

        return beta_nr

    def guide(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']
        gene_index_tensor_n = data['gene_index_tensor']

        # sample the inducing points from a MVN (see ``VariationalSparseGP.guide``)
        autoname.scope(prefix="EXPR", fn=self.vsgp_gene_expression_prior.vsgp.guide)()

        # sample beta_nr posterior
        beta_posterior_loc_gr = pyro.param(
            "beta_posterior_loc_gr",
            lambda: self.f_mean.detach().clone().squeeze(-1).expand([self.sc_fingerprint_dtm.n_genes, 4]).contiguous())

        with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
            beta_nr = pyro.sample(
                "beta_nr",
                dist.Delta(v=beta_posterior_loc_gr[gene_index_tensor_n, :]).to_event(1))

        return beta_nr


class VSGPGeneExpressionPriorPreTrainer(torch.nn.Module):
    def __init__(
            self,
            vsgp_gene_expression_prior: VSGPGeneExpressionPrior):
        super(VSGPGeneExpressionPriorPreTrainer, self).__init__()
        self.vsgp_gene_expression_prior = vsgp_gene_expression_prior
        self.log_mean_total_molecules_per_cell = np.log(np.mean(
            vsgp_gene_expression_prior.sc_fingerprint_dtm.total_obs_molecules_per_cell))

    def model(self, data: Dict[str, torch.Tensor]):
        fingerprint_tensor_nr = data['fingerprint_tensor']
        cell_sampling_site_scale_factor_tensor_n = data['cell_sampling_site_scale_factor_tensor']
        total_obs_molecules_per_cell_tensor_n = data['total_obs_molecules_per_cell_tensor']

        e_obs_n = fingerprint_tensor_nr.sum(-1)

        pyro.module("vsgp_gene_expression_prior", self.vsgp_gene_expression_prior,
                    update_module_params=True)

        beta_nr = self.vsgp_gene_expression_prior.model(data)
        log_eta_n = (
                total_obs_molecules_per_cell_tensor_n.log()
                - self.log_mean_total_molecules_per_cell)

        # calculate ZINB parameters
        mu_e_hi_n = (beta_nr[:, 0] + beta_nr[:, 1] * log_eta_n).exp()
        phi_e_hi_n = beta_nr[:, 2].exp()
        logit_p_zero_e_hi_n = beta_nr[:, 3]

        with poutine.scale(scale=cell_sampling_site_scale_factor_tensor_n):
            # observe the empirical gene expression
            pyro.sample(
                "e_obs",
                ZeroInflatedNegativeBinomial(
                    logit_p_zero=logit_p_zero_e_hi_n,
                    mu=mu_e_hi_n,
                    phi=phi_e_hi_n),
                obs=e_obs_n)

    def guide(self, data):
        pyro.module("vsgp_gene_expression_prior", self.vsgp_gene_expression_prior,
                    update_module_params=True)
        self.vsgp_gene_expression_prior.guide(data)


##############
# deprecated #
##############

# class SingleCellFeaturePredictedGeneExpressionPrior(GeneExpressionPrior):
#     EPS = 1e-6
#
#     def __init__(self,
#                  sc_fingerprint_dtm: SingleCellFingerprintDTM,
#                  hidden_dims: Tuple[int] = (50,),
#                  hidden_activation: torch.nn.Module = torch.nn.SELU(),
#                  device: torch.device = torch.device('cuda'),
#                  dtype: torch.dtype = torch.float):
#         super(SingleCellFeaturePredictedGeneExpressionPrior, self).__init__()
#
#         self.sc_fingerprint_dtm = sc_fingerprint_dtm
#         self.hidden_activation = hidden_activation
#         self.device = device
#         self.dtype = dtype
#
#         # hidden layers
#         n_input_features = sc_fingerprint_dtm.feature_z_scores_per_cell.shape[1]
#         self.hidden_layers = torch.nn.ModuleList()
#         last_dim = n_input_features
#         for hidden_dim in hidden_dims:
#             layer = torch.nn.Linear(last_dim, hidden_dim, bias=True)
#             last_dim = hidden_dim
#             self.hidden_layers.append(layer)
#
#         # setup the final cell-feature-based readout weight
#         xavier_scale = 1. / np.sqrt(n_input_features)
#         self.readout_weight_hg = torch.nn.Parameter(
#             xavier_scale * torch.randn(
#                 (last_dim, sc_fingerprint_dtm.n_genes),
#                 device=device, dtype=dtype))
#         self.readout_bias_g = torch.nn.Parameter(
#             xavier_scale * torch.randn(
#                 (sc_fingerprint_dtm.n_genes,),
#                 device=device, dtype=dtype))
#
#         # global parameters
#         self.log_phi_e_hi_g = torch.nn.Parameter(
#             torch.log(torch.tensor(
#                 self.EPS + sc_fingerprint_dtm.empirical_phi_e_hi,
#                 device=device, dtype=dtype)))
#         self.logit_p_zero_e_hi_g = torch.nn.Parameter(
#             logit(torch.tensor(
#                 sc_fingerprint_dtm.empirical_p_zero_e_hi,
#                 device=device, dtype=dtype)))
#
#         self.to(device)
#
#     def forward(self,
#                 gene_index_tensor_n: torch.Tensor,
#                 cell_index_tensor_n: torch.Tensor,
#                 eta_n: torch.Tensor,
#                 cell_features_tensor_nf: Optional[torch.Tensor],
#                 total_obs_reads_per_cell_tensor_n: Optional[torch.Tensor],
#                 downsampling_rate_tensor_n: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Estimate cell-specific ZINB expression parameters."""
#
#         activations_nh = cell_features_tensor_nf
#         for layer in self.hidden_layers:
#             activations_nh = self.hidden_activation(layer.forward(activations_nh))
#
#         # gene-specific linear readout of expression rate
#         log_pred_mu_n = (
#                 torch.einsum(
#                     "nh,hn->n",
#                     [activations_nh,
#                      self.readout_weight_hg[:, gene_index_tensor_n]])
#                 + self.readout_bias_g[gene_index_tensor_n])
#
#         return {
#             'log_mu_e_hi_n': log_pred_mu_n + eta_n.log(),
#             'log_phi_e_hi_n': self.log_phi_e_hi_g[gene_index_tensor_n],
#             'logit_p_zero_e_hi_n': self.logit_p_zero_e_hi_g[gene_index_tensor_n]}


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
