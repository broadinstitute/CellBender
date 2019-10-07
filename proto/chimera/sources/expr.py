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
    def model(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def guide(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def decode_output_to_nb_params_dict(
            self,
            output_dict: Dict[str, torch.Tensor],
            data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class VSGPGeneExpressionModel(GeneExpressionModel):
    DEFAULT_GENE_GROUP_NAME = 'all'
    INPUT_DIM = 1
    LATENT_DIM = 2

    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 gene_group_name: Optional[str],
                 gene_group_indices: Optional[np.ndarray],
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
        assert not ((gene_group_name is None) ^ (gene_group_indices is None))

        if gene_group_name is not None:
            self.gene_group_name = gene_group_name
            self.gene_group_indices = gene_group_indices
        else:
            # default to all genes
            self.gene_group_name = VSGPGeneExpressionModel.DEFAULT_GENE_GROUP_NAME
            self.gene_group_indices = np.arange(sc_fingerprint_dtm.n_genes)

        # feature space
        self.log_mean_obs_expr_g1 = torch.log(
            torch.tensor(sc_fingerprint_dtm.arithmetic_mean_obs_expr_per_gene,
                         device=device, dtype=dtype)).unsqueeze(-1)

        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.device = device
        self.dtype = dtype

        # inducing points
        lo = torch.min(self.log_mean_obs_expr_g1).item()
        hi = torch.max(self.log_mean_obs_expr_g1).item()
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
        assert init_mean_intercept.shape == (self.LATENT_DIM,)
        assert init_mean_slope.shape == (self.LATENT_DIM,)

        self.f_mean_intercept_r = Parameter(
            torch.tensor(init_mean_intercept, device=device, dtype=dtype))
        self.f_mean_slope_r = Parameter(
            torch.tensor(init_mean_slope, device=device, dtype=dtype))

        def mean_function(x_n1: torch.Tensor):
            return (self.f_mean_intercept_r + self.f_mean_slope_r * x_n1).permute(-1, -2)

        # instantiate VSGP model
        self.vsgp = VariationalSparseGP(
            X=self.log_mean_obs_expr_g1,
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
            mean_function(self.log_mean_obs_expr_g1).permute(-1, -2).detach().clone())
        self.beta_posterior_scale_gr = Parameter(
            init_posterior_scale * torch.ones(
                (self.sc_fingerprint_dtm.n_genes, VSGPGeneExpressionModel.LATENT_DIM), device=device, dtype=dtype))
        self.set_constraint("beta_posterior_scale_gr", constraints.positive)

        # send parameters to device
        self.to(device)

    def _model(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.set_mode("model")

        assert 'arithmetic_mean_obs_expr_per_gene_tensor' in data
        assert 'gene_sampling_site_scale_factor_tensor' in data

        log_arithmetic_mean_obs_expr_n1 = data['arithmetic_mean_obs_expr_per_gene_tensor'].log().unsqueeze(-1)
        gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']

        # sample all points
        self.vsgp.set_data(X=log_arithmetic_mean_obs_expr_n1, y=None)
        beta_loc_rn, beta_var_rn = self.vsgp.model()
        beta_loc_nr = beta_loc_rn.permute(-1, -2)
        beta_scale_nr = beta_var_rn.permute(-1, -2).sqrt()

        # sample e_hi prior parameters
        with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
            # sample beta parameters
            beta_nr = pyro.sample(
                "beta_nr",
                dist.Normal(loc=beta_loc_nr, scale=beta_scale_nr).to_event(1))

        return {
            'beta_nr': beta_nr
        }

    def _guide(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.set_mode("guide")

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

        return {
            'beta_nr': beta_nr
        }

    def model(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return autoname.scope(fn=self._model, prefix="expr_" + self.gene_group_name)(data)

    def guide(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return autoname.scope(fn=self._guide, prefix="expr_" + self.gene_group_name)(data)

    def decode_output_to_nb_params_dict(
            self,
            output_dict: Dict[str, torch.Tensor],
            data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'beta_nr' in output_dict
        assert 'empirical_droplet_efficiency_tensor' in data

        beta_nr = output_dict['beta_nr']
        eta_n = data['empirical_droplet_efficiency_tensor']

        log_mu_e_hi_n = beta_nr[:, 0] + eta_n.log()
        log_phi_e_hi_n = beta_nr[:, 1]

        return {
            'log_mu_e_hi_n': log_mu_e_hi_n,
            'log_phi_e_hi_n': log_phi_e_hi_n
        }

    def plot_fit(self, axs):
        # test x range
        lo = self.log_mean_obs_expr_g1.min().item()
        hi = self.log_mean_obs_expr_g1.max().item()
        r = hi - lo
        x_test_t = torch.linspace(
            lo - 0.25 * r,
            hi + 0.25 * r,
            steps=500,
            device=self.device, dtype=self.dtype).unsqueeze(-1)

        with torch.no_grad():
            self.vsgp.set_data(X=x_test_t, y=None)
            beta_loc_rn, beta_var_rn = self.vsgp.model()
            beta_loc_nr = beta_loc_rn.permute(-1, -2)
            beta_scale_nr = beta_var_rn.permute(-1, -2).sqrt()
        f_loc_numpy = beta_loc_nr.cpu().numpy()
        f_scale_numpy = beta_scale_nr.cpu().numpy()

        beta_posterior_loc_gr = self.beta_posterior_loc_gr.detach().cpu().numpy()

        x_test = x_test_t.cpu().numpy()
        for i, ax in enumerate(axs):
            ax.plot(x_test[:, 0], f_loc_numpy[:, i], color='red')
            ax.fill_between(
                x_test[:, 0],
                f_loc_numpy[:, i] - 2.0 * f_scale_numpy[:, i],
                f_loc_numpy[:, i] + 2.0 * f_scale_numpy[:, i],
                color='C0', alpha=0.3)
            gene_colors = np.zeros((beta_posterior_loc_gr.shape[0], 4))
            gene_colors[:, 3] = 1
            x_train = self.log_mean_obs_expr_g1.cpu().numpy()
            ax.scatter(x_train, beta_posterior_loc_gr[:, i], s=1, color=gene_colors)
            ax.set_ylabel(f'$\\beta_{i}$', fontsize=14)
            ax.set_xlabel('$\log \,\, \\tilde{e}$', fontsize=14)


class VSGPGeneExpressionModelTrainer:
    def __init__(
            self,
            vsgp_gene_expression_model: VSGPGeneExpressionModel,
            sc_fingerprint_dtm: SingleCellFingerprintDTM,
            adam_lr: float = 1e-2,
            adam_betas: Tuple[float, float] = (0.9, 0.99)):
        super(VSGPGeneExpressionModelTrainer, self).__init__()

        self.vsgp_gene_expression_model = vsgp_gene_expression_model
        self.sc_fingerprint_dtm = sc_fingerprint_dtm

        # training
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
            "expr_" + self.vsgp_gene_expression_model.gene_group_name,
            self.vsgp_gene_expression_model,
            update_module_params=True)
        assert 'counts_tensor' in data
        assert 'counts_truncation_rate_tensor' in data
        assert 'cell_sampling_site_scale_factor_tensor' in data

        counts_tensor_n = data['counts_tensor']
        counts_truncation_rate_n = data['counts_truncation_rate_tensor']
        cell_sampling_site_scale_factor_tensor_n = data['cell_sampling_site_scale_factor_tensor']

        # sample from GP prior
        model_output_dict = self.vsgp_gene_expression_model.model(data)

        # calculate NB parameters
        e_hi_nb_params_dict = self.vsgp_gene_expression_model.decode_output_to_nb_params_dict(
            output_dict=model_output_dict,
            data=data)
        log_mu_e_hi_n = e_hi_nb_params_dict['log_mu_e_hi_n'] + counts_truncation_rate_n.log()
        log_phi_e_hi_n = e_hi_nb_params_dict['log_phi_e_hi_n']

        with poutine.scale(scale=cell_sampling_site_scale_factor_tensor_n):
            # observe the empirical gene expression
            pyro.sample(
                "e_obs",
                NegativeBinomial(
                    mu=log_mu_e_hi_n.exp(),
                    phi=log_phi_e_hi_n.exp()),
                obs=counts_tensor_n)

    def guide(self, data):
        pyro.module(
            "expr_" + self.vsgp_gene_expression_model.gene_group_name,
            self.vsgp_gene_expression_model,
            update_module_params=True)
        self.vsgp_gene_expression_model.guide(data)

    def run_training(self,
                     n_training_iterations: int = 1000,
                     training_log_frequency: int = 100,
                     minibatch_genes_per_gene_group: int = 20,
                     minibatch_expressing_cells_per_gene: int = 20,
                     minibatch_silent_cells_per_gene: int = 5,
                     minibatch_sampling_strategy: str = 'without_replacement'):
        t0 = time.time()
        i_iter = 0
        mb_loss_list = []

        logging.warning(f"[VSGPGeneExpressionModelTrainer for "
                        f"{self.vsgp_gene_expression_model.gene_group_name}] training started...")

        while i_iter < n_training_iterations:
            mb_data = self.sc_fingerprint_dtm.generate_counts_stratified_sample(
                minibatch_genes_per_gene_group,
                minibatch_expressing_cells_per_gene,
                minibatch_silent_cells_per_gene,
                minibatch_sampling_strategy,
                count_matrix_type='truncated')

            mb_loss = self.svi.step(mb_data) / self.loss_scale

            mb_loss_list.append(mb_loss)
            self.loss_hist.append(mb_loss)
            i_iter += 1

            if i_iter % training_log_frequency == 0 and i_iter > 0:
                # calculate loss stats
                t1 = time.time()
                mb_loss_mean, mb_loss_std = np.mean(mb_loss_list), np.std(mb_loss_list)

                logging.warning(
                    f'Iteration number: {i_iter}, loss: {mb_loss_mean:.3f} +- {mb_loss_std:.3f}, '
                    f'time: {(t1 - t0):.3f}s')

                # reset
                mb_loss_list = []
                t0 = t1

        logging.warning(f"[VSGPGeneExpressionModelTrainer for "
                        f"{self.vsgp_gene_expression_model.gene_group_name}] training finished.")

        self.trained = True

    def plot_diagnostics(self, axs):
        assert len(axs) == 3

        # plot loss history
        ax = axs[0]
        ax.plot(self.loss_hist)
        ax.set_title(f'GP training loss (gene group: '
                     f'{self.vsgp_gene_expression_model.gene_group_name})', fontsize=14)
        ax.set_xlabel('iteration', fontsize=14)
        ax.set_ylabel('loss', fontsize=14)

        # plot fit
        self.vsgp_gene_expression_model.plot_fit(axs[1:])


class FeatureBasedGeneExpressionModel(GeneExpressionModel):
    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 init_features_ard_scale: float = 1.0,
                 phi_scale: float = 0.1,
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float):
        super(FeatureBasedGeneExpressionModel, self).__init__()

        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.device = device
        self.dtype = dtype

        self.init_features_ard_scale = init_features_ard_scale
        self.phi_scale = phi_scale

        self.n_input_features = sc_fingerprint_dtm.features_per_cell.shape[1]

        # log mu weights ARD precisions
        self.gamma_ard_scale_f = Parameter(
            self.init_features_ard_scale * torch.ones((self.n_input_features,), device=device, dtype=dtype))
        self.set_constraint("gamma_ard_scale_f", constraints.positive)

        # log alpha (posterior)
        self.log_alpha_posterior_loc_g = Parameter(
            - np.log(self.phi_scale) * torch.ones((self.sc_fingerprint_dtm.n_genes,), device=device, dtype=dtype))

        # log mu bias (posterior)
        self.beta_posterior_loc_g = Parameter(
            torch.tensor(sc_fingerprint_dtm.arithmetic_mean_obs_expr_per_gene, device=device, dtype=dtype).log())

        # log mu weights (posterior)
        self.gamma_posterior_loc_gf = Parameter(
            torch.zeros((self.sc_fingerprint_dtm.n_genes, self.n_input_features), device=device, dtype=dtype))

        # send parameters to device
        self.to(device)

    @autoname.scope(prefix="feature_based_expr")
    def model(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.set_mode("model")

        assert 'gene_sampling_site_scale_factor_tensor' in data

        gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']
        mb_size = gene_sampling_site_scale_factor_tensor_n.shape[0]

        with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):

            # sample log mu weights
            gamma_nf = pyro.sample(
                "gamma_nf",
                dist.Normal(
                    loc=torch.zeros((mb_size, self.n_input_features), device=self.device, dtype=self.dtype),
                    scale=self.gamma_ard_scale_f.expand((mb_size, self.n_input_features))
                ).to_event(1))

            # sample log alpha
            log_alpha_n = pyro.sample(
                "log_alpha_n",
                dist.Gumbel(
                    loc=-np.log(self.phi_scale) * torch.ones((mb_size,), device=self.device, dtype=self.dtype),
                    scale=torch.ones((mb_size,), device=self.device, dtype=self.dtype)))

        return {
            'gamma_nf': gamma_nf,
            'log_alpha_n': log_alpha_n
        }

    @autoname.scope(prefix="feature_based_expr")
    def guide(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.set_mode("guide")

        assert 'gene_sampling_site_scale_factor_tensor' in data
        assert 'gene_index_tensor' in data

        gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']
        gene_index_tensor_n = data['gene_index_tensor']

        with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):

            # sample log mu weights
            gamma_nf = pyro.sample(
                "gamma_nf",
                dist.Delta(v=self.gamma_posterior_loc_gf[gene_index_tensor_n, :]).to_event(1))

            # sample log alpha
            log_alpha_n = pyro.sample(
                "log_alpha_n",
                dist.Delta(v=self.log_alpha_posterior_loc_g[gene_index_tensor_n]))

        return {
            'gamma_nf': gamma_nf,
            'log_alpha_n': log_alpha_n
        }

    def decode_output_to_nb_params_dict(
            self,
            output_dict: Dict[str, torch.Tensor],
            data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        assert 'gamma_nf' in output_dict
        assert 'log_alpha_n' in output_dict
        assert 'gene_index_tensor' in data
        assert 'cell_features_tensor' in data
        assert 'empirical_droplet_efficiency_tensor' in data

        gamma_nf = output_dict['gamma_nf']
        log_alpha_n = output_dict['log_alpha_n']
        gene_index_tensor_n = data['gene_index_tensor']
        cell_features_nf = data['cell_features_tensor']
        eta_n = data['empirical_droplet_efficiency_tensor']

        log_mu_e_hi_n = (
                self.beta_posterior_loc_g[gene_index_tensor_n]
                + torch.einsum('nf,nf->n', gamma_nf, cell_features_nf)
                + eta_n.log())
        log_phi_e_hi_n = - log_alpha_n

        return {
            'log_mu_e_hi_n': log_mu_e_hi_n,
            'log_phi_e_hi_n': log_phi_e_hi_n
        }


class FeatureBasedGeneExpressionModelTrainer:
    def __init__(
            self,
            feature_based_gene_expression_model: FeatureBasedGeneExpressionModel,
            sc_fingerprint_dtm: SingleCellFingerprintDTM,
            adam_lr: float = 1e-2,
            adam_betas: Tuple[float, float] = (0.9, 0.99)):
        super(FeatureBasedGeneExpressionModelTrainer, self).__init__()

        self.feature_based_gene_expression_model = feature_based_gene_expression_model
        self.sc_fingerprint_dtm = sc_fingerprint_dtm

        # training
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
            "feature_based_expr",
            self.feature_based_gene_expression_model,
            update_module_params=True)

        assert 'counts_tensor' in data
        assert 'counts_truncation_rate_tensor' in data
        assert 'cell_sampling_site_scale_factor_tensor' in data

        counts_tensor_n = data['counts_tensor']
        counts_truncation_rate_n = data['counts_truncation_rate_tensor']
        cell_sampling_site_scale_factor_tensor_n = data['cell_sampling_site_scale_factor_tensor']

        # sample from the prior
        model_output_dict = self.feature_based_gene_expression_model.model(data)

        # calculate NB parameters
        e_hi_nb_params_dict = self.feature_based_gene_expression_model.decode_output_to_nb_params_dict(
            output_dict=model_output_dict,
            data=data)
        log_mu_e_hi_n = e_hi_nb_params_dict['log_mu_e_hi_n'] + counts_truncation_rate_n.log()
        log_phi_e_hi_n = e_hi_nb_params_dict['log_phi_e_hi_n']

        with poutine.scale(scale=cell_sampling_site_scale_factor_tensor_n):
            # observe the empirical gene expression
            pyro.sample(
                "e_obs",
                NegativeBinomial(
                    mu=log_mu_e_hi_n.exp(),
                    phi=log_phi_e_hi_n.exp()),
                obs=counts_tensor_n)

    def guide(self, data):
        pyro.module(
            "feature_based_expr",
            self.feature_based_gene_expression_model,
            update_module_params=True)

        self.feature_based_gene_expression_model.guide(data)

    def run_training(self,
                     n_training_iterations: int = 1000,
                     training_log_frequency: int = 100,
                     minibatch_genes_per_gene_group: int = 20,
                     minibatch_expressing_cells_per_gene: int = 20,
                     minibatch_silent_cells_per_gene: int = 5,
                     minibatch_sampling_strategy: str = 'without_replacement'):
        t0 = time.time()
        i_iter = 0
        mb_loss_list = []

        logging.warning(f"[FeatureBasedGeneExpressionModelTrainer] training started...")

        while i_iter < n_training_iterations:
            mb_data = self.sc_fingerprint_dtm.generate_counts_stratified_sample(
                minibatch_genes_per_gene_group,
                minibatch_expressing_cells_per_gene,
                minibatch_silent_cells_per_gene,
                minibatch_sampling_strategy,
                count_matrix_type='truncated')

            mb_loss = self.svi.step(mb_data) / self.loss_scale

            mb_loss_list.append(mb_loss)
            self.loss_hist.append(mb_loss)
            i_iter += 1

            if i_iter % training_log_frequency == 0 and i_iter > 0:
                # calculate loss stats
                t1 = time.time()
                mb_loss_mean, mb_loss_std = np.mean(mb_loss_list), np.std(mb_loss_list)

                logging.warning(
                    f'Iteration number: {i_iter}, loss: {mb_loss_mean:.3f} +- {mb_loss_std:.3f}, '
                    f'time: {(t1 - t0):.3f}s')

                # reset
                mb_loss_list = []
                t0 = t1

        logging.warning(f"[FeatureBasedGeneExpressionModelTrainer] training finished.")

        self.trained = True

    def plot_diagnostics(self, ax):
        # plot loss history
        ax.plot(self.loss_hist)
        ax.set_title(f'training loss', fontsize=14)
        ax.set_xlabel('iteration', fontsize=14)
        ax.set_ylabel('loss', fontsize=14)
