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

    @abstractmethod
    def forward(self,
                gene_index_tensor_n: torch.Tensor,
                cell_index_tensor_n: torch.Tensor,
                downsampling_rate_tensor_n: torch.Tensor,
                total_obs_reads_per_cell_tensor_n: torch.Tensor,
                cell_features_nf: Union[None, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


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

        init_log_mu_e_hi_g = torch.tensor(
            self.EPS + sc_fingerprint_dtm.empirical_mu_e_hi,
            device=device,
            dtype=dtype).log()
        init_log_phi_e_hi_g = torch.tensor(
            self.EPS + sc_fingerprint_dtm.empirical_phi_e_hi,
            device=device,
            dtype=dtype).log()
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

    def forward(self,
                gene_index_tensor_n: torch.Tensor,
                cell_index_tensor_n: torch.Tensor,
                downsampling_rate_tensor_n: torch.Tensor,
                total_obs_reads_per_cell_tensor_n: torch.Tensor,
                cell_features_nf: Union[None, torch.Tensor]) -> torch.Tensor:
        bias_nr = self.readout_bias_gr[gene_index_tensor_n, :]
        cell_size_scale_n = total_obs_reads_per_cell_tensor_n / (
                self.mean_total_reads_per_cell * downsampling_rate_tensor_n)
        expr_params_nr = bias_nr
        expr_params_nr[:, 0] += torch.log(self.EPS + cell_size_scale_n)
        return expr_params_nr


class SingleCellFeaturePredictedGeneExpressionPrior(GeneLevelGeneExpressionPrior):
    EPS = 1e-6

    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 hidden_dims: Tuple[int] = (100, 100),
                 init_cell_feature_weight: float = 0.1,
                 activation=torch.nn.Softplus(),
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float):
        super(SingleCellFeaturePredictedGeneExpressionPrior, self).__init__(
            sc_fingerprint_dtm=sc_fingerprint_dtm,
            device=device,
            dtype=dtype)
        self.hidden_dims = hidden_dims
        self.activation = activation

        # setup the hidden layers
        self.layers = torch.nn.ModuleList()
        n_input_cell_features = sc_fingerprint_dtm.all_features_per_cell.shape[1]
        last_n_features = n_input_cell_features
        for hidden_dim in hidden_dims:
            layer = torch.nn.Linear(last_n_features, hidden_dim, bias=True)
            last_n_features = hidden_dim
            self.layers.append(layer)

        # setup the cell-feature-based readout weight
        read_tensor_xavier_scale = init_cell_feature_weight / np.sqrt(last_n_features)
        self.readout_tensor_fgr = torch.nn.Parameter(
            read_tensor_xavier_scale * torch.randn(
                (last_n_features, sc_fingerprint_dtm.n_genes, 3),
                device=device,
                dtype=dtype))

        # send parameters to device
        self.to(device)

    def forward(self,
                gene_index_tensor_n: torch.Tensor,
                cell_index_tensor_n: torch.Tensor,
                downsampling_rate_tensor_n: torch.Tensor,
                total_obs_reads_per_cell_tensor_n: torch.Tensor,
                cell_features_nf: Union[None, torch.Tensor]) -> torch.Tensor:
        """Estimate cell-specific ZINB expression parameters."""
        bias_nr = self.readout_bias_gr[gene_index_tensor_n, :]
        processed_features_nf = cell_features_nf
        for layer in self.layers:
            processed_features_nf = self.activation(layer.forward(processed_features_nf))
        cell_specific_correction_nr = torch.einsum(
            "nf,fnr->nr",
            [processed_features_nf,
             self.readout_tensor_fgr[:, gene_index_tensor_n, :]])
        return bias_nr + cell_specific_correction_nr
