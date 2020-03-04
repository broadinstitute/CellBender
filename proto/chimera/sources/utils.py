import torch


activation_from_str = {
    'softplus': torch.nn.Softplus(),
    'relu': torch.nn.ReLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'elu': torch.nn.ELU(),
    'selu': torch.nn.SELU(),
}


def get_cell_averaged_from_collapsed_samples(
        input_tensor_n: torch.Tensor,
        gene_index_tensor_n: torch.Tensor,
        cell_sampling_site_scale_factor_tensor_n: torch.Tensor,
        n_genes: int,
        dtype: torch.dtype,
        device: torch.device,
        eps: float = 1e-7) -> torch.Tensor:
    """Calculates cell-averaging of `input_tensor_n` for cells in the minibatch"""

    input_tensor_cell_weighted_sum_g = torch.zeros(n_genes, device=device, dtype=dtype)
    cell_weight_sum_g = torch.zeros(n_genes, device=device, dtype=dtype)

    input_tensor_cell_weighted_sum_g.index_add_(
        dim=0,
        index=gene_index_tensor_n,
        source=input_tensor_n * cell_sampling_site_scale_factor_tensor_n)

    cell_weight_sum_g.index_add_(
        dim=0,
        index=gene_index_tensor_n,
        source=cell_sampling_site_scale_factor_tensor_n)

    # _eps is added to avoid NaNs (on discarded indices)
    input_tensor_cell_averaged_g = input_tensor_cell_weighted_sum_g / (eps + cell_weight_sum_g)

    # gather over genes
    input_tensor_cell_averaged_g = torch.gather(
        input_tensor_cell_averaged_g,
        dim=0,
        index=gene_index_tensor_n)

    return input_tensor_cell_averaged_g


def get_detached_on_non_inducing_genes(
        input_scalar: torch.Tensor,
        inducing_binary_mask_tensor_n: torch.Tensor,
        non_inducing_binary_mask_tensor_n: torch.Tensor) -> torch.Tensor:

    input_scalar_detached = input_scalar.clone().detach()
    return (input_scalar * inducing_binary_mask_tensor_n
            + input_scalar_detached * non_inducing_binary_mask_tensor_n)
