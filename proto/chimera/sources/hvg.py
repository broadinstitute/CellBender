from typing import Optional, Dict
from operator import itemgetter
import time

import numpy as np

from pyro.infer import TraceMeanField_ELBO
from pyro.contrib import gp
from pyro.infer.util import torch_backward, torch_item

from typing import Tuple, List
import torch
import logging

from adjustText import adjust_text
from matplotlib.axes import Axes
from matplotlib.pylab import cm
from matplotlib import pylab

from fingerprint import SingleCellFingerprintDTM


class ExpressionGPRegression(torch.nn.Module):
    def __init__(self,
                 empirical_mean_expression_g: np.ndarray,
                 empirical_std_expression_g: np.ndarray,
                 gene_names_array_g: np.ndarray,
                 gene_group_name: str,
                 n_inducing_points: int = 20,
                 jitter: float = 1e-5,
                 adam_lr: float = 1e-2,
                 adam_betas: Tuple[float, float] = (0.9, 0.99),
                 init_log_phi: float = 0.0,
                 device: torch.device = torch.device("cuda"),
                 dtype: torch.dtype = torch.float32):
        super(ExpressionGPRegression, self).__init__()

        assert empirical_mean_expression_g.ndim == 1
        assert empirical_std_expression_g.ndim == 1
        assert np.all(empirical_mean_expression_g > 0)
        assert np.all(empirical_std_expression_g > 0)
        assert isinstance(gene_group_name, str)

        data_x = np.log(empirical_mean_expression_g)
        data_y = np.log(empirical_std_expression_g)

        self.data_x_t = torch.tensor(data_x, device=device, dtype=dtype)
        self.data_y_t = torch.tensor(data_y, device=device, dtype=dtype)
        self.gene_names_array_g = gene_names_array_g

        self.num_data = self.data_x_t.shape[0]
        self.device = device
        self.dtype = dtype
        self.gene_group_name = gene_group_name
        self.trained = False

        inducing_points = torch.linspace(
            np.min(data_x), np.max(data_x), steps=n_inducing_points,
            device=device, dtype=dtype)

        kernel = gp.kernels.RBF(input_dim=1)
        likelihood = gp.likelihoods.Gaussian()

        self.log_phi = torch.nn.Parameter(
            data=torch.tensor(init_log_phi, device=device, dtype=dtype))

        def mean_function(x: torch.Tensor):
            return 0.5 * (x.exp() + (2 * x + self.log_phi).exp()).log()

        self.vsgp = gp.models.VariationalSparseGP(
            X=None,
            y=None,
            num_data=self.num_data,
            kernel=kernel,
            Xu=inducing_points,
            likelihood=likelihood,
            jitter=jitter,
            whiten=True,
            mean_function=mean_function).to(device)

        # training
        self.params = list(self.vsgp.parameters()) + [self.log_phi]
        self.optim = torch.optim.Adam(self.params, adam_lr, adam_betas)
        self.loss_fn = TraceMeanField_ELBO().differentiable_loss
        self.loss_hist = []

    def forward(self, x):
        assert self.trained
        self.vsgp.set_data(X=self.data_x_t, y=self.data_y_t)
        y_new_mean, y_new_var = self.vsgp(x, full_cov=False)
        total_y_new_var = y_new_var + self.vsgp.likelihood.variance
        return y_new_mean, total_y_new_var.sqrt()

    def run_training(self,
                     n_training_iters: int,
                     data_subsampling_rate: Optional[float] = 0.1,
                     train_log_frequency: int = 100):

        def closure():
            self.optim.zero_grad()
            loss = self.loss_fn(self.vsgp.model, self.vsgp.guide)
            torch_backward(loss)
            return loss

        if data_subsampling_rate is None:
            self.vsgp.set_data(X=self.data_x_t, y=self.data_y_t)
        else:
            mb_size = int(data_subsampling_rate * self.num_data)

        t0 = time.time()
        i_iter = 0
        mb_loss_list = []

        logging.warning(f"[ExpressionGPRegression for {self.gene_group_name}] training started...")

        while i_iter < n_training_iters:
            if data_subsampling_rate is not None:
                # generate minibatch
                mb_indices = torch.randint(low=0, high=self.num_data, size=[mb_size], device=self.device)
                mb_data_x_t = self.data_x_t[mb_indices]
                mb_data_y_t = self.data_y_t[mb_indices]

                # step
                self.vsgp.set_data(X=mb_data_x_t, y=mb_data_y_t)

            mb_loss = torch_item(self.optim.step(closure)) / self.num_data

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

        logging.warning(f"[ExpressionGPRegression for {self.gene_group_name}] training finished.")
        self.trained = True

    def get_residual_log_std(self) -> np.ndarray:
        assert self.trained
        with torch.no_grad():
            fit_log_std_expr_loc_g, _ = self(self.data_x_t)
        return (self.data_y_t - fit_log_std_expr_loc_g).cpu().numpy()

    def get_highly_variable_indices(self, n_highly_variable: int) -> List[int]:
        assert n_highly_variable > 0
        n_highly_variable = min(n_highly_variable, self.num_data)
        sorted_indexed_scores = sorted(
            list(enumerate(self.get_residual_log_std())),
            key=itemgetter(1), reverse=True)
        return list(map(itemgetter(0), sorted_indexed_scores))[:n_highly_variable]

    def plot_gp_fit(self, ax: Axes, x_padding: float = 2.0, steps: int = 1000):
        x_test = torch.linspace(
            torch.min(self.data_x_t).item() - x_padding,
            torch.max(self.data_x_t).item() + x_padding,
            steps=steps,
            device=self.device, dtype=self.dtype)

        with torch.no_grad():
            y_test_mean, y_test_std = self(x_test)

        ax.scatter(
            self.data_x_t.cpu().numpy(), self.data_y_t.cpu().numpy(),
            s=1, alpha=0.2, color='black', label='data')

        ax.fill_between(
            x_test.cpu().numpy(),
            (y_test_mean - 4.0 * y_test_std).cpu().numpy(),
            (y_test_mean + 4.0 * y_test_std).cpu().numpy(),
            color='C0', alpha=0.3)

        ax.plot(x_test.cpu().numpy(), y_test_mean.cpu().numpy(),
                '-', lw=2, color='red', label='GP full fit')
        ax.set_xlabel('log mean expression', fontsize=14)
        ax.set_ylabel('log std expression', fontsize=14)
        ax.set_title(f'GP regression (gene group: {self.gene_group_name})', fontsize=14)
        ax.legend()

    def plot_gp_loss(self, ax: Axes):
        ax.plot(self.loss_hist)
        ax.set_title(f'GP training loss (gene group: {self.gene_group_name})', fontsize=14)
        ax.set_xlabel('iteration', fontsize=14)
        ax.set_ylabel('loss', fontsize=14)

    def plot_highly_variable_genes(self, ax: Axes, top_n_annotate=50):
        residual_log_std = self.get_residual_log_std()
        highly_variable_indices = self.get_highly_variable_indices(top_n_annotate)

        data_x = self.data_x_t.cpu().numpy()
        data_y = residual_log_std

        # make scatter plot
        ax.scatter(
            data_x,
            data_y,
            s=10, alpha=0.9,
            c=residual_log_std,
            cmap=cm.coolwarm,
            vmin=np.min(residual_log_std),
            vmax=residual_log_std[highly_variable_indices[-1]])

        # add gene names labels
        labels = []
        for gene_index in highly_variable_indices:
            labels.append(
                ax.text(data_x[gene_index], data_y[gene_index], self.gene_names_array_g[gene_index], fontsize=10))
        adjust_text(labels, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

        # add axis labels
        ax.set_xlabel('log mean expression', fontsize=14)
        ax.set_ylabel('log residual std expression', fontsize=14)


class HighlyVariableGenesSelector:
    default_gene_group_name = 'all genes'

    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 hvg_gene_group_prefixes: Optional[List[str]] = None,
                 n_highly_variable_genes: int = 1000,
                 gp_n_inducing_points: int = 20,
                 gp_jitter: float = 1e-3,
                 gp_adam_lr: float = 1e-2,
                 gp_init_log_phi: float = 0.0,
                 gp_n_training_iters: int = 2000,
                 gp_data_subsampling_rate: Optional[float] = None,
                 gp_train_log_frequency: int = 100,
                 device: torch.device = torch.device("cuda"),
                 dtype: torch.dtype = torch.float32):
        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.n_highly_variable_genes = n_highly_variable_genes
        self.device = device
        self.dtype = dtype

        if hvg_gene_group_prefixes is None:
            self.hvg_gene_group_internal_indices_dict: Dict[str, List[int]] = {
                HighlyVariableGenesSelector.default_gene_group_name: list(range(sc_fingerprint_dtm.n_genes))
            }
        else:
            self.hvg_gene_group_internal_indices_dict: Dict[str, List[int]] = {
                prefix: sc_fingerprint_dtm.sc_fingerprint_base.get_internal_gene_indices_by_gene_name_prefix(prefix)
                for prefix in hvg_gene_group_prefixes
            }

        # instantiate GP regressors
        self.expr_reg_dict: Dict[str, ExpressionGPRegression] = dict()
        for gene_group_name, internal_gene_indices in self.hvg_gene_group_internal_indices_dict.items():

            # instantiate
            expr_reg = ExpressionGPRegression(
                empirical_mean_expression_g=sc_fingerprint_dtm.family_size_truncated_expression_mean[
                    internal_gene_indices],
                empirical_std_expression_g=sc_fingerprint_dtm.family_size_truncated_expression_std[
                    internal_gene_indices],
                gene_names_array_g=np.asarray(sc_fingerprint_dtm.sc_fingerprint_base.gene_names_list)[
                    internal_gene_indices],
                gene_group_name=gene_group_name,
                n_inducing_points=gp_n_inducing_points,
                jitter=gp_jitter,
                adam_lr=gp_adam_lr,
                init_log_phi=gp_init_log_phi,
                device=device,
                dtype=dtype)
            self.expr_reg_dict[gene_group_name] = expr_reg

            # train GP
            expr_reg.run_training(
                n_training_iters=gp_n_training_iters,
                data_subsampling_rate=gp_data_subsampling_rate,
                train_log_frequency=gp_train_log_frequency)

    @property
    def n_groups(self):
        return len(self.hvg_gene_group_internal_indices_dict.keys())

    def get_highly_variable_gene_indices(self) -> List[int]:
        highly_variable_gene_per_group = int(np.ceil(self.n_highly_variable_genes / self.n_groups))
        highly_variable_gene_indices = []
        for gene_group_name, expr_reg in self.expr_reg_dict.items():
            highly_variable_gene_indices_in_group = expr_reg.get_highly_variable_indices(
                n_highly_variable=highly_variable_gene_per_group)
            gene_indices_in_group = self.hvg_gene_group_internal_indices_dict[gene_group_name]
            highly_variable_gene_indices_in_fingerprint = [
                gene_indices_in_group[i] for i in highly_variable_gene_indices_in_group]
            highly_variable_gene_indices += highly_variable_gene_indices_in_fingerprint
        return highly_variable_gene_indices

    def plot_diagnostics(self, plt: pylab):
        # GP fits
        fig, axs = plt.subplots(nrows=self.n_groups, ncols=2, figsize=(10, self.n_groups * 4))
        if self.n_groups == 1:
            axs = axs[None, :]
        for gene_group_name, ax in zip(self.hvg_gene_group_internal_indices_dict.keys(), axs):
            self.expr_reg_dict[gene_group_name].plot_gp_loss(ax[0])
            self.expr_reg_dict[gene_group_name].plot_gp_fit(ax[1])
        plt.tight_layout()

        # HVG selection
        for gene_group_name in self.hvg_gene_group_internal_indices_dict.keys():
            plt.figure(figsize=(16, 16))
            ax = plt.gca()
            self.expr_reg_dict[gene_group_name].plot_highly_variable_genes(ax)
            plt.tight_layout()
