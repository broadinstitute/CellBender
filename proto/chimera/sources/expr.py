import numpy as np
from typing import Tuple, List, Dict, Union, Any, Callable, Generator, Optional
from abc import abstractmethod
import logging
import time

import torch
from torch.distributions import constraints

import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.contrib import autoname
from pyro.contrib.gp.models import VariationalSparseGP
from pyro.contrib.gp.parameterized import Parameterized, Parameter
from pyro.infer import SVI, Trace_ELBO
import pyro.contrib.gp.kernels as kernels

from pyro_extras import NegativeBinomial
from fingerprint import SingleCellFingerprintDTM


class GeneExpressionModel(Parameterized):
    def __init__(self):
        super(GeneExpressionModel, self).__init__()

    @abstractmethod
    def model(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def guide(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self,
               beta_nr: torch.Tensor,
               cell_features_nf: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class VSGPGeneExpressionModel(GeneExpressionModel):
    DEFAULT_GENE_GROUP_NAME = 'all genes'
    INPUT_DIM = 1
    LATENT_DIM = 3

    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 gene_group_name: Optional[str],
                 n_inducing_points: int,
                 init_rbf_kernel_variance: float,
                 init_rbf_kernel_lengthscale: float,
                 init_whitenoise_kernel_variance: float,
                 init_posterior_scale: float,
                 init_mean_intercept: np.ndarray,
                 init_mean_slope: np.ndarray,
                 cholesky_jitter: float,
                 min_noise: float,
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float):
        super(VSGPGeneExpressionModel, self).__init__()
        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.device = device
        self.dtype = dtype

        if gene_group_name is not None:
            self.gene_group_name = gene_group_name
        else:
            self.gene_group_name = VSGPGeneExpressionModel.DEFAULT_GENE_GROUP_NAME

        # feature space
        self.log_geometric_mean_obs_expr_g1 = torch.log(
            torch.tensor(sc_fingerprint_dtm.geometric_mean_obs_expr_per_gene,
                         device=device, dtype=dtype)).unsqueeze(-1)

        # inducing points
        lo = torch.min(self.log_geometric_mean_obs_expr_g1).item()
        hi = torch.max(self.log_geometric_mean_obs_expr_g1).item()
        r = hi - lo
        self.inducing_points_k1 = torch.linspace(
            lo - 0.25 * r,
            hi + 0.25 * r,
            steps=n_inducing_points,
            device=device, dtype=dtype).unsqueeze(-1)

        # GP kernel setup
        kernel_rbf = kernels.RBF(
            input_dim=VSGPGeneExpressionModel.INPUT_DIM,
            variance=torch.tensor(init_rbf_kernel_variance, device=device, dtype=dtype),
            lengthscale=torch.tensor(init_rbf_kernel_lengthscale, device=device, dtype=dtype))
        kernel_whitenoise = kernels.WhiteNoise(
            input_dim=VSGPGeneExpressionModel.INPUT_DIM,
            variance=torch.tensor(init_whitenoise_kernel_variance, device=device, dtype=dtype))
        kernel_whitenoise.set_constraint("variance", constraints.greater_than(min_noise))
        kernel_full = kernels.Sum(kernel_rbf, kernel_whitenoise)

        # mean subtraction
        self.f_mean_intercept_r = Parameter(
            torch.tensor(init_mean_intercept, device=device, dtype=dtype))
        self.f_mean_slope_r = Parameter(
            torch.tensor(init_mean_slope, device=device, dtype=dtype))

        def mean_function(x_n1: torch.Tensor):
            return (self.f_mean_intercept_r + self.f_mean_slope_r * x_n1).permute(-1, -2)

        # instantiate VSGP model
        self.vsgp = VariationalSparseGP(
            X=self.log_geometric_mean_obs_expr_g1,
            y=None,
            kernel=kernel_full,
            Xu=self.inducing_points_k1,
            likelihood=None,
            mean_function=mean_function,
            latent_shape=torch.Size([VSGPGeneExpressionModel.LATENT_DIM]),
            whiten=True,
            jitter=cholesky_jitter)

        # posterior parameters
        self.beta_posterior_loc_gr = Parameter(
            mean_function(self.log_geometric_mean_obs_expr_g1).permute(-1, -2).detach().clone())
        self.beta_posterior_scale_gr = Parameter(
            init_posterior_scale * torch.ones(
                (self.sc_fingerprint_dtm.n_genes, VSGPGeneExpressionModel.LATENT_DIM), device=device, dtype=dtype))
        self.set_constraint("beta_posterior_scale_gr", constraints.positive)

        # send parameters to device
        self.to(device)

    @autoname.scope(prefix="expr")
    def model(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.set_mode("model")
        pyro.module("vsgp", self.vsgp, update_module_params=True)

        assert 'geometric_mean_obs_expr_per_gene_tensor' in data
        assert 'gene_sampling_site_scale_factor_tensor' in data

        log_geometric_mean_obs_expr_n1 = data['geometric_mean_obs_expr_per_gene_tensor'].log().unsqueeze(-1)
        gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']

        # sample all points
        self.vsgp.set_data(X=log_geometric_mean_obs_expr_n1, y=None)
        beta_loc_rn, beta_var_rn = self.vsgp.model()
        beta_loc_nr = beta_loc_rn.permute(-1, -2)
        beta_scale_nr = beta_var_rn.permute(-1, -2).sqrt()

        # sample e_hi prior parameters
        with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
            # sample beta parameters
            beta_nr = pyro.sample(
                "beta_nr",
                dist.Normal(loc=beta_loc_nr, scale=beta_scale_nr).to_event(1))

        return beta_nr

    @autoname.scope(prefix="expr")
    def guide(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.set_mode("guide")
        pyro.module("vsgp", self.vsgp, update_module_params=True)

        assert 'gene_sampling_site_scale_factor_tensor' in data
        assert 'gene_index_tensor' in data

        gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']
        gene_index_tensor_n = data['gene_index_tensor']

        # sample the inducing points from a MVN (see ``VariationalSparseGP.guide``)
        self.vsgp.guide()

        with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
            beta_nr = pyro.sample(
                "beta_nr",
                dist.Normal(
                    loc=self.beta_posterior_loc_gr[gene_index_tensor_n, :],
                    scale=self.beta_posterior_scale_gr[gene_index_tensor_n, :]).to_event(1))

        return beta_nr

    def decode(self,
               beta_nr: torch.Tensor,
               cell_features_nf: torch.Tensor) -> Dict[str, torch.Tensor]:

        log_eta_n = cell_features_nf[:, 0]
        log_mu_e_hi_n = beta_nr[:, 0] + beta_nr[:, 1] * log_eta_n
        log_phi_e_hi_n = beta_nr[:, 2]

        return {
            'log_mu_e_hi_n': log_mu_e_hi_n,
            'log_phi_e_hi_n': log_phi_e_hi_n
        }


class VSGPGeneExpressionModelPreTrainer:
    def __init__(
            self,
            vsgp_gene_expression_model: VSGPGeneExpressionModel,
            sc_fingerprint_dtm: SingleCellFingerprintDTM,
            adam_lr: float = 1e-2,
            adam_betas: Tuple[float, float] = (0.9, 0.99)):
        super(VSGPGeneExpressionModelPreTrainer, self).__init__()

        self.vsgp_gene_expression_model = vsgp_gene_expression_model
        self.sc_fingerprint_dtm = sc_fingerprint_dtm

        # training
        self.params = list(vsgp_gene_expression_model.parameters())
        self.optim = pyro.optim.Adam({'lr': adam_lr, 'betas': adam_betas})
        self.svi = SVI(
            model=self.model,
            guide=self.guide,
            optim=self.optim,
            loss=Trace_ELBO())
        self.loss_scale = sc_fingerprint_dtm.n_genes * sc_fingerprint_dtm.n_cells
        self.loss_hist = []
        self.trained = False

    def model(self, data: Dict[str, torch.Tensor]):
        pyro.module(
            "vsgp_gene_expression_model", self.vsgp_gene_expression_model,
            update_module_params=True)
        assert 'fingerprint_tensor' in data
        assert 'cell_sampling_site_scale_factor_tensor' in data
        assert 'empirical_droplet_efficiency_tensor' in data

        fingerprint_tensor_nr = data['fingerprint_tensor']
        cell_sampling_site_scale_factor_tensor_n = data['cell_sampling_site_scale_factor_tensor']
        eta_n = data['empirical_droplet_efficiency_tensor']

        e_obs_n = fingerprint_tensor_nr.sum(-1)

        # sample from GP prior
        beta_nr = self.vsgp_gene_expression_model.model(data)

        # calculate NB parameters
        log_eta_n = eta_n.log()
        cell_features_nf = log_eta_n.unsqueeze(-1)
        e_hi_params_dict = self.vsgp_gene_expression_model.decode(
            beta_nr=beta_nr,
            cell_features_nf=cell_features_nf)
        mu_e_hi_n = e_hi_params_dict['log_mu_e_hi_n'].exp()
        phi_e_hi_n = e_hi_params_dict['log_phi_e_hi_n'].exp()

        with poutine.scale(scale=cell_sampling_site_scale_factor_tensor_n):
            # observe the empirical gene expression
            pyro.sample(
                "e_obs",
                NegativeBinomial(
                    mu=mu_e_hi_n,
                    phi=phi_e_hi_n),
                obs=e_obs_n)

    def guide(self, data):
        pyro.module(
            "vsgp_gene_expression_model", self.vsgp_gene_expression_model,
            update_module_params=True)
        self.vsgp_gene_expression_model.guide(data)

    def run_training(self,
                     n_training_iters: int = 1000,
                     train_log_frequency: int = 100,
                     minibatch_genes_per_gene_group: int = 20,
                     minibatch_expressing_cells_per_gene: int = 20,
                     minibatch_silent_cells_per_gene: int = 5,
                     minibatch_sampling_strategy: str = 'without_replacement'):
        t0 = time.time()
        i_iter = 0
        mb_loss_list = []

        logging.warning(f"[VSGPGeneExpressionModelPreTrainer for "
                        f"{self.vsgp_gene_expression_model.gene_group_name}] training started...")

        while i_iter < n_training_iters:
            mb_data = self.sc_fingerprint_dtm.generate_stratified_sample_for_dtm(
                minibatch_genes_per_gene_group,
                minibatch_expressing_cells_per_gene,
                minibatch_silent_cells_per_gene,
                minibatch_sampling_strategy)

            mb_loss = self.svi.step(mb_data) / self.loss_scale

            mb_loss_list.append(mb_loss)
            self.loss_hist.append(mb_loss)
            i_iter += 1

            if i_iter % train_log_frequency == 0 and i_iter > 0:
                # calculate loss stats
                t1 = time.time()
                mb_loss_mean, mb_loss_std = np.mean(mb_loss_list), np.std(mb_loss_list)

                logging.warning(
                    f'Iteration number: {i_iter}, loss: {mb_loss_mean:.3f} +- {mb_loss_std:.3f}, '
                    f'time: {(t1 - t0):.3f}s')

                # reset
                mb_loss_list = []
                t0 = t1

        logging.warning(f"[VSGPGeneExpressionModelPreTrainer for "
                        f"{self.vsgp_gene_expression_model.gene_group_name}] training finished.")

        self.trained = True


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
