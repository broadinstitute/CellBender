import numpy as np
from typing import Tuple, List, Dict, Union, Any, Callable, Generator
from abc import abstractmethod

import torch
from torch.nn.parameter import Parameter

from pyro_extras import logit
from fingerprint import SingleCellFingerprintDTM


class GeneExpressionPrior(torch.nn.Module):
    UNCONSTRAINED_SPACE_CLIPPING = 10.

    def __init__(self):
        super(GeneExpressionPrior, self).__init__()
        self.pos_trans_forward = lambda x: torch.exp(
            torch.clamp(
                x,
                min=-self.UNCONSTRAINED_SPACE_CLIPPING,
                max=+self.UNCONSTRAINED_SPACE_CLIPPING))
        self.pos_trans_inverse = torch.log

    @abstractmethod
    def forward(self,
                gene_index_tensor_n: torch.Tensor,
                cell_index_tensor_n: torch.Tensor,
                downsampling_rate_tensor_n: torch.Tensor,
                total_obs_reads_per_cell_tensor_n: torch.Tensor,
                cell_features_nf: Union[None, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, zeta_nr: torch.Tensor):
        mu_e_hi_n = self.pos_trans_forward(zeta_nr[..., 0])
        phi_e_hi_n = self.pos_trans_forward(zeta_nr[..., 1])
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

        init_log_mu_e_hi_g = self.pos_trans_inverse(torch.tensor(
            self.EPS + sc_fingerprint_dtm.empirical_mu_e_hi,
            device=device,
            dtype=dtype))
        init_log_phi_e_hi_g = self.pos_trans_inverse(torch.tensor(
            self.EPS + sc_fingerprint_dtm.empirical_phi_e_hi,
            device=device,
            dtype=dtype))
        init_logit_p_zero_e_hi_g = logit(
            torch.tensor(
                sc_fingerprint_dtm.empirical_p_zero_e_hi,
                device=device,
                dtype=dtype))
        self.readout_bias_gr = torch.nn.Parameter(
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
                 initial_hidden_dims: Tuple[int] = (),
                 intermediate_dim: int = 5,
                 final_hidden_dims: Tuple[int] = (5,),
                 init_cell_feature_weight: float = 0.1,
                 initial_transform=lambda x: x,
                 hidden_activation=torch.nn.LeakyReLU(),
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float):
        super(SingleCellFeaturePredictedGeneExpressionPrior, self).__init__(
            sc_fingerprint_dtm=sc_fingerprint_dtm,
            device=device,
            dtype=dtype)
        self.initial_transform = initial_transform
        self.hidden_activation = hidden_activation

        # setup the initial hidden layers
        self.initial_layers = torch.nn.ModuleList()
        last_dim = sc_fingerprint_dtm.all_features_per_cell.shape[1]
        for hidden_dim in initial_hidden_dims:
            layer = torch.nn.Linear(last_dim, hidden_dim, bias=True)
            last_dim = hidden_dim
            self.initial_layers.append(layer)

        # setup the intermediate cell-feature-based readout weight
        xavier_scale = 1. / np.sqrt(last_dim)
        self.intermediate_gene_readout_weight_fgh = torch.nn.Parameter(
            init_cell_feature_weight * xavier_scale * torch.randn(
                (last_dim, sc_fingerprint_dtm.n_genes, intermediate_dim),
                device=device,
                dtype=dtype))
        self.intermediate_gene_readout_bias_gh = torch.nn.Parameter(
            init_cell_feature_weight * xavier_scale * torch.randn(
                (sc_fingerprint_dtm.n_genes, intermediate_dim),
                device=device,
                dtype=dtype))

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

        # process universally
        processed_features_nf = self.initial_transform(cell_features_nf)
        for layer in self.initial_layers:
            processed_features_nf = self.hidden_activation(layer.forward(processed_features_nf))

        # select by gene
        intermediate_nh = (
                torch.einsum(
                    "nf,fnh->nh",
                    [processed_features_nf,
                     self.intermediate_gene_readout_weight_fgh[:, gene_index_tensor_n, :]])
                + self.intermediate_gene_readout_bias_gh[gene_index_tensor_n, :])

        # stack the bias
        gene_specific_bias_nr = self.readout_bias_gr[gene_index_tensor_n, :]
        hidden_nh = torch.cat((intermediate_nh, gene_specific_bias_nr), dim=-1)
        for layer in self.final_layers:
            hidden_nh = layer.forward(self.hidden_activation(hidden_nh))

        # output residual
        return gene_specific_bias_nr + hidden_nh

