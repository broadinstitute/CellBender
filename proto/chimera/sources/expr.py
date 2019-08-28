import numpy as np
from typing import Tuple, List, Dict, Union, Any, Callable, Generator
from abc import abstractmethod

import torch
from torch.nn.parameter import Parameter

from pyro_extras import logit
from fingerprint import SingleCellFingerprintDTM


class GeneExpressionPrior(torch.nn.Module):
    def __init__(self):
        super(GeneExpressionPrior, self).__init__()
        self.unc_to_pos_trans_forward = torch.exp
        self.unc_to_pos_trans_backward = torch.log

    @abstractmethod
    def forward(self,
                gene_index_tensor_n: torch.Tensor,
                cell_index_tensor_n: torch.Tensor,
                downsampling_rate_tensor_n: torch.Tensor,
                total_obs_reads_per_cell_tensor_n: torch.Tensor,
                cell_features_nf: Union[None, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, zeta_nr: torch.Tensor):
        mu_e_hi_n = self.unc_to_pos_trans_forward(zeta_nr[..., 0])
        phi_e_hi_n = self.unc_to_pos_trans_forward(zeta_nr[..., 1])
        logit_p_zero_e_hi_n = zeta_nr[..., 2]
        return mu_e_hi_n, phi_e_hi_n, logit_p_zero_e_hi_n


class GeneLevelGeneExpressionPrior(GeneExpressionPrior):
    EPS = 1e-6

    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float):
        super(GeneLevelGeneExpressionPrior, self).__init__()
        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.device = device
        self.dtype = dtype
        self.mean_total_reads_per_cell: float = np.mean(sc_fingerprint_dtm.total_obs_reads_per_cell).item()

        init_log_mu_e_hi_g = self.unc_to_pos_trans_backward(torch.tensor(
            self.EPS + sc_fingerprint_dtm.empirical_mu_e_hi,
            device=device,
            dtype=dtype))
        init_log_phi_e_hi_g = self.unc_to_pos_trans_backward(torch.tensor(
            self.EPS + sc_fingerprint_dtm.empirical_phi_e_hi,
            device=device,
            dtype=dtype))
        init_logit_p_zero_e_hi_g = logit(torch.tensor(
            sc_fingerprint_dtm.empirical_p_zero_e_hi,
            device=device,
            dtype=dtype))
        self.global_prior_params_gr = torch.nn.Parameter(
            torch.stack((
                init_log_mu_e_hi_g,
                init_log_phi_e_hi_g,
                init_logit_p_zero_e_hi_g)).transpose(-1, -2))

        # send parameters to device
        self.to(device)


class SingleCellFeaturePredictedGeneExpressionPrior(GeneLevelGeneExpressionPrior):
    EPS = 1e-6

    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 intermediate_dim: int = 1,
                 final_hidden_dims: Tuple[int] = (3, 3, 3),
                 init_cell_feature_weight: float = 0.1,
                 max_correction_zeta_space: float = 5.0,
                 hidden_activation=torch.nn.LeakyReLU(),
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float):
        super(SingleCellFeaturePredictedGeneExpressionPrior, self).__init__(
            sc_fingerprint_dtm=sc_fingerprint_dtm,
            device=device,
            dtype=dtype)
        assert intermediate_dim >= 1
        assert max_correction_zeta_space > 0

        self.hidden_activation = hidden_activation
        self.max_correction_zeta_space = max_correction_zeta_space
        self.inv_max_correction_zeta_space = 1.0 / max_correction_zeta_space

        # initial batch norm layer
        n_input_features = sc_fingerprint_dtm.feature_z_scores_per_cell.shape[1]

        # setup the intermediate cell-feature-based readout weight
        xavier_scale = 1. / np.sqrt(n_input_features)
        self.intermediate_gene_readout_weight_fgh = torch.nn.Parameter(
            init_cell_feature_weight * xavier_scale * torch.randn(
                (n_input_features, sc_fingerprint_dtm.n_genes, intermediate_dim),
                device=device,
                dtype=dtype))
        self.intermediate_gene_readout_bias_gh = torch.nn.Parameter(
            init_cell_feature_weight * xavier_scale * torch.randn(
                (sc_fingerprint_dtm.n_genes, intermediate_dim),
                device=device,
                dtype=dtype))

        # initialize the first channel to the truncated SVD decoder (of the transformed empirical expression)
        svd_components_fg = sc_fingerprint_dtm.svd_feature_components
        svd_loadings_nf = sc_fingerprint_dtm.svd_feature_loadings_per_cell
        svd_mean_loadings_f = np.mean(svd_loadings_nf, 0)
        svd_std_loadings_f = np.std(svd_loadings_nf, 0)
        svd_decoder_bias_g = init_cell_feature_weight * np.dot(svd_components_fg.T, svd_mean_loadings_f)
        svd_decoder_weights_hg = np.zeros((n_input_features, sc_fingerprint_dtm.n_genes))
        svd_decoder_weights_hg[:sc_fingerprint_dtm.n_pca_features, :] = init_cell_feature_weight * (
                svd_std_loadings_f[:, None] * svd_components_fg)

        self.intermediate_gene_readout_weight_fgh.data[:, :, 0].copy_(
            torch.tensor(svd_decoder_weights_hg, device=device, dtype=dtype))
        self.intermediate_gene_readout_bias_gh.data[:, 0].copy_(
            torch.tensor(svd_decoder_bias_g, device=device, dtype=dtype))

        final_hidden_dims += (3,)
        self.final_layers = torch.nn.ModuleList()
        last_dim = intermediate_dim + 3
        for hidden_dim in final_hidden_dims:
            layer = torch.nn.Linear(last_dim, hidden_dim, bias=True)
            last_dim = hidden_dim
            self.final_layers.append(layer)

        # send parameters to device
        self.to(device)

    def forward(self,
                gene_index_tensor_n: torch.Tensor,
                cell_index_tensor_n: torch.Tensor,
                downsampling_rate_tensor_n: torch.Tensor,
                total_obs_reads_per_cell_tensor_n: torch.Tensor,
                cell_features_nf: Union[None, torch.Tensor]) -> torch.Tensor:
        """Estimate cell-specific ZINB expression parameters."""

        # apply the gene-specific linear transformation
        intermediate_nh = (
                torch.einsum(
                    "nf,fnh->nh",
                    [cell_features_nf,
                     self.intermediate_gene_readout_weight_fgh[:, gene_index_tensor_n, :]])
                + self.intermediate_gene_readout_bias_gh[gene_index_tensor_n, :])

        # stack the intermediate result and the bias
        global_prior_params_nr = self.global_prior_params_gr[gene_index_tensor_n, :]
        cell_specific_correction_nh = torch.cat((intermediate_nh, global_prior_params_nr), dim=-1)
        for layer in self.final_layers:
            cell_specific_correction_nh = layer.forward(self.hidden_activation(cell_specific_correction_nh))

        # soft clamp on the correction strength
        clamped_cell_specific_correction_nh = self.max_correction_zeta_space * torch.tanh(
            self.inv_max_correction_zeta_space * cell_specific_correction_nh)

        # output residual
        return global_prior_params_nr + clamped_cell_specific_correction_nh
