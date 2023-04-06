"""Simulate a basic scRNA-seq count matrix dataset, for tests."""

from cellbender.remove_background.model import calculate_mu, calculate_lambda
from cellbender.remove_background.data.io import write_matrix_to_cellranger_h5
from cellbender.remove_background.checkpoint import load_from_checkpoint

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import torch
import pyro
import random
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Optional


if torch.cuda.is_available():
    USE_CUDA = True
    DEVICE = 'cuda'
else:
    USE_CUDA = False
    DEVICE = 'cpu'


def comprehensive_random_seed(seed, use_cuda=USE_CUDA):
    """Establish a base random state
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.util.set_rng_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def generate_sample_inferred_model_dataset(
        checkpoint_file: str,
        cells_of_each_type: List[int] = [100, 100, 100],
        n_droplets: int = 10000,
        model_type: str = 'full',
        plot_pca: bool = True,
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 10,
        random_seed: int = 0) -> Dict[str, Union[float, np.ndarray, sp.csr_matrix]]:
    """Create a sample dataset for use as 'truth' data, based on a trained checkpoint.

    Args:
        checkpoint_file: ckpt.tar.gz file from a remove-background run
        cells_of_each_type: Number of cells of each cell type to simulate
        n_droplets: Total number of droplets to simulate
        model_type: The --model argument to specify remove-background's model
        plot_pca: True to plot the PCA plot used to define "cell types"
        dbscan_eps: Smaller makes the clusters finer


    """

    # Input checking.
    assert len(cells_of_each_type) > 0, 'cells_of_each_type must be a List of ints'
    n_cell_types = len(cells_of_each_type)

    # Load a trained model from a checkpoint file.
    ckpt = load_from_checkpoint(
        filebase=None,
        tarball_name=checkpoint_file,
        to_load=['model', 'dataloader', 'param_store'],
        force_device='cpu' if not torch.cuda.is_available() else None)
    model = ckpt['model']
    n_genes = model.n_genes

    # Reach in and set model type.
    model.model_type = model_type

    # Seed random number generators.
    comprehensive_random_seed(seed=random_seed)

    # Find z values for cells.
    data_loader = ckpt['train_loader']
    if torch.cuda.is_available():
        data_loader.use_cuda = True
        data_loader.device = 'cuda'
        model.use_cuda = True
        model.device = 'cuda'
    else:
        data_loader.use_cuda = False
        data_loader.device = 'cpu'
        model.use_cuda = False
        model.device = 'cpu'
    z = np.zeros((len(data_loader), model.encoder['z'].output_dim))
    p = np.zeros(len(data_loader))
    chi_ambient = pyro.param('chi_ambient').detach()
    for i, data in enumerate(data_loader):
        enc = model.encoder(x=data,
                            chi_ambient=chi_ambient,
                            cell_prior_log=model.d_cell_loc_prior)
        ind = i * data_loader.batch_size
        z[ind:(ind + data.shape[0]), :] = enc['z']['loc'].detach().cpu().numpy()
        p[ind:(ind + data.shape[0])] = enc['p_y'].sigmoid().detach().cpu().numpy()
    z = z[p > 0.5, :]  # select cells only

    # Cluster cells based on embeddings.
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(z)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters_ < len(cells_of_each_type):
        print(f'WARNING: DBSCAN found only {n_clusters_} clusters, but the input '
              f'cells_of_each_type dictates {len(cells_of_each_type)} cell types '
              f'are required.')

    # Show a plot.
    if plot_pca:
        pca = PCA(n_components=2).fit_transform(z)
        for label in np.unique(labels):
            if label == -1:
                color = 'lightgray'
            else:
                color = None
            plt.plot(pca[labels == label, 0], pca[labels == label, 1],
                     '.', color=color, label=label)
        plt.title('Embedding PCA')
        plt.xlabel('PCA 0')
        plt.ylabel('PCA 1')
        plt.legend()
        plt.show()

        unique_labels = set(np.unique(labels)) - {-1}
        print({label: (labels == label).sum() for label in unique_labels})

    # Sample z values for cells of each cluster based on this posterior.
    z_cluster = []
    for label, n_cells in zip(np.unique(labels)[:len(cells_of_each_type)], cells_of_each_type):
        if n_cells == 0:
            continue
        logic = (labels == label)
        z_tmp = z[logic][torch.randperm(logic.sum())][:n_cells]
        while len(z_tmp) < n_cells:
            z_tmp = np.concatenate((z_tmp, z[logic][torch.randperm(logic.sum())][:n_cells]), axis=0)
        z_cluster.append(z_tmp[:n_cells])

    # Account for the possibility that one cell type has zero cells (useful for selecting populations of interest)
    cells_of_each_type = [c for c in cells_of_each_type if c > 0]
    n_cell_types = len(cells_of_each_type)

    # Prep variables.
    c_real = np.zeros((n_droplets, n_genes))
    c_bkg = np.zeros((n_droplets, n_genes))
    labels = np.zeros(n_droplets)
    epsilon = np.zeros(n_droplets)
    rho = np.zeros(n_droplets)
    d_cell = np.zeros(n_droplets)
    d_empty = np.zeros(n_droplets)
    chi = np.zeros((n_cell_types, n_genes))
    d_cell_type = np.zeros(n_cell_types)

    # Sample counts from cell-containing droplets.
    i = 0
    for t, (num_cells, z_chunk) in enumerate(zip(cells_of_each_type, z_cluster)):
        sim = sample_from_inferred_model(
            num=num_cells,
            model=model,
            z=torch.tensor(z_chunk).to('cuda' if torch.cuda.is_available() else 'cpu').float(),
            y=torch.ones(num_cells).to('cuda' if torch.cuda.is_available() else 'cpu').float(),
        )
        c_real[i:(i + num_cells), :] = sim['counts_real']
        c_bkg[i:(i + num_cells), :] = sim['counts_bkg']
        epsilon[i:(i + num_cells)] = sim['epsilon']
        rho[i:(i + num_cells)] = sim['rho']
        d_cell[i:(i + num_cells)] = sim['cell_counts']
        d_empty[i:(i + num_cells)] = sim['empty_counts']
        labels[i:(i + num_cells)] = t + 1  # label cell types with integers > 0
        chi[t, :] = sim['chi'].sum(axis=0) / sim['chi'].sum()
        d_cell_type[t] = sim['cell_counts'].mean()
        i = i + num_cells

    # Sample counts from empty droplets.
    sim = sample_from_inferred_model(
        num=n_droplets - i,
        model=model,
        y=torch.zeros(n_droplets - i).to('cuda' if torch.cuda.is_available() else 'cpu').float(),
    )
    c_real[i:, :] = sim['counts_real']
    c_bkg[i:, :] = sim['counts_bkg']
    epsilon[i:] = sim['epsilon']
    rho[i:] = sim['rho']

    # Get sparse matrices from dense.
    counts_true = sp.csr_matrix(c_real, shape=c_real.shape, dtype=int)
    counts_bkg = sp.csr_matrix(c_bkg, shape=c_bkg.shape, dtype=int)

    return {'counts_true': counts_true,
            'counts_bkg': counts_bkg,
            'barcodes': np.array([f'bc{n:06d}' for n in np.arange(n_droplets)]),
            'gene_names': model.analyzed_gene_names,
            'chi': {i + 1: chi[i, :] for i in range(chi.shape[0])},
            'chi_ambient': chi_ambient.detach().cpu().numpy(),
            'droplet_labels': labels.astype(int),
            'd_cell': d_cell,
            'd_empty': d_empty,
            'epsilon': epsilon,
            'rho': rho,
            'cell_mean_umi': {i + 1: u for i, u in enumerate(d_cell_type)},
            'cell_lognormal_sigma': model.d_cell_scale_prior.item(),
            'empty_mean_umi': model.d_empty_loc_prior.exp().item(),
            'empty_lognormal_sigma': model.d_empty_scale_prior.item(),
            'epsilon_param': model.epsilon_prior.item(),
            'model': [model_type]}


def generate_sample_dirichlet_dataset(
        n_genes: int = 10000,
        cells_of_each_type: List[int] = [100, 100, 100],
        n_droplets: int = 5000,
        model_type: str = 'full',
        dirichlet_alpha: Union[float, np.ndarray] = 0.05,
        chi: Optional[np.ndarray] = None,
        chi_artificial_similarity: float = 0,
        vector_to_add_to_chi_ambient: Optional[np.ndarray] = None,
        chi_ambient: Optional[np.ndarray] = None,
        cell_mean_umi: List[int] = [5000],
        cell_lognormal_sigma: float = 0.2,
        empty_mean_umi: int = 200,
        empty_lognormal_sigma: float = 0.4,
        alpha0: float = 5000,
        epsilon_param: float = 100,
        rho_alpha: float = 3,
        rho_beta: float = 80,
        random_seed: int = 0) -> Dict[str, Union[float, np.ndarray, sp.csr_matrix]]:
    """Create a sample dataset for use as 'truth' data.
    """

    # Input checking.
    assert len(cells_of_each_type) > 0, 'cells_of_each_type must be a List of ints'
    n_cell_types = len(cells_of_each_type)
    if vector_to_add_to_chi_ambient is not None:
        assert len(vector_to_add_to_chi_ambient) == n_genes, \
            f'vector_to_add_to_chi_ambient {vector_to_add_to_chi_ambient.shape} must ' \
            f'be of length n_genes ({n_genes})'
    if type(dirichlet_alpha) == np.ndarray:
        assert dirichlet_alpha.size == n_genes, \
            f'If you input a vector for dirichlet_alpha, its length ' \
            f'({dirichlet_alpha.size}) must be n_genes ({n_genes})'
    assert cell_lognormal_sigma > 0, 'cell_lognormal_sigma must be > 0'
    assert empty_lognormal_sigma > 0, 'empty_lognormal_sigma must be > 0'
    cell_mean_umi = list(cell_mean_umi)
    if len(cell_mean_umi) == 1:
        cell_mean_umi = cell_mean_umi * n_cell_types  # repeat the entry
    for umi in cell_mean_umi:
        assert umi > empty_mean_umi, \
            f'cell_size_mean_umi ({umi}) must be > empty_mean_umi ({empty_mean_umi})'
    assert len(cell_mean_umi) == len(cells_of_each_type), \
        'cells_of_each_type and cell_mean_umi (if a list) must be of the same length'
    assert (chi_artificial_similarity >= 0) and (chi_artificial_similarity <= 1), \
        'chi_artificial_similarity must be in the range [0, 1]'

    # Seed random number generators.
    comprehensive_random_seed(seed=random_seed)

    # Draw gene expression profiles for each cell type.
    if chi is None:
        chi = np.zeros((n_cell_types, n_genes))
        for i in range(chi.shape[0]):
            chi[i, :] = draw_random_chi(alpha=dirichlet_alpha, n_genes=n_genes).cpu().numpy()
            if i > 0:
                # Make expression profiles similar (if chi_artificial_similarity > 0) via gating.
                chi[i, :] = chi[i, :] * (1 - chi_artificial_similarity) + chi[0, :] * chi_artificial_similarity
    else:
        dirichlet_alpha = ['None: chi was input to generate_sample_dataset()']

    # Get chi_ambient: a weighted average of expression, possibly with extra.
    if chi_ambient is None:
        chi_ambient = np.zeros(n_genes)
        for i in range(n_cell_types):
            chi_ambient = chi_ambient + chi[i, :] * cells_of_each_type[i] * cell_mean_umi[i]
        if vector_to_add_to_chi_ambient is None:
            vector_to_add_to_chi_ambient = np.zeros(n_genes)
        chi_ambient = chi_ambient + vector_to_add_to_chi_ambient
    else:
        if vector_to_add_to_chi_ambient is not None:
            print('You specified both `chi_ambient` and `vector_to_add_to_chi_ambient`. '
                  'Ignoring `vector_to_add_to_chi_ambient` and using `chi_ambient` '
                  'as provided.')
    chi_ambient = chi_ambient / chi_ambient.sum()

    c_real = np.zeros((n_droplets, n_genes))
    c_bkg = np.zeros((n_droplets, n_genes))
    labels = np.zeros(n_droplets)
    epsilon = np.zeros(n_droplets)
    rho = np.zeros(n_droplets)
    d_cell = np.zeros(n_droplets)
    d_empty = np.zeros(n_droplets)

    # Sample counts from cell-containing droplets.
    i = 0
    for t, (num_cells, celltype_umi) in enumerate(zip(cells_of_each_type, cell_mean_umi)):
        sim = sample_from_dirichlet_model(
            num=num_cells,
            alpha=chi[t, :] * alpha0 + 1e-20,  # must be > 0
            d_mu=np.log(celltype_umi),
            d_sigma=cell_lognormal_sigma,
            v_mu=np.log(empty_mean_umi),
            v_sigma=empty_lognormal_sigma,
            y=1,  # cell present
            chi_ambient=chi_ambient,
            eps_param=epsilon_param,
            rho_alpha=rho_alpha,
            rho_beta=rho_beta,
            model_type=model_type,
        )
        c_real[i:(i + num_cells), :] = sim['counts_real']
        c_bkg[i:(i + num_cells), :] = sim['counts_bkg']
        epsilon[i:(i + num_cells)] = sim['epsilon']
        rho[i:(i + num_cells)] = sim['rho']
        d_cell[i:(i + num_cells)] = sim['cell_counts']
        d_empty[i:(i + num_cells)] = sim['empty_counts']
        labels[i:(i + num_cells)] = t + 1  # label cell types with integers > 0
        i = i + num_cells

    # Sample counts from empty droplets.
    sim = sample_from_dirichlet_model(
        num=n_droplets - i,
        alpha=np.ones(n_genes),  # this doesn't get used since y=0
        d_mu=1,  # this doesn't get used since y=0
        d_sigma=1,  # this doesn't get used since y=0
        v_mu=np.log(empty_mean_umi),
        v_sigma=empty_lognormal_sigma,
        y=0,  # no cell present
        chi_ambient=chi_ambient,
        eps_param=epsilon_param,
        rho_alpha=rho_alpha,
        rho_beta=rho_beta,
        model_type=model_type,
    )
    c_real[i:, :] = sim['counts_real']
    c_bkg[i:, :] = sim['counts_bkg']
    epsilon[i:] = sim['epsilon']
    rho[i:] = sim['rho']
    d_empty[i:] = sim['empty_counts']

    # Get sparse matrices from dense.
    counts_true = sp.csr_matrix(c_real, shape=c_real.shape, dtype=int)
    counts_bkg = sp.csr_matrix(c_bkg, shape=c_bkg.shape, dtype=int)

    return {'counts_true': counts_true,
            'counts_bkg': counts_bkg,
            'barcodes': np.array([f'bc{n:06d}' for n in np.arange(n_droplets)]),
            'gene_names': np.array([f'g{n:05d}' for n in np.arange(n_genes)]),
            'chi': {str(i + 1): chi[i, :] for i in range(chi.shape[0])},
            'chi_ambient': chi_ambient,
            'droplet_labels': labels.astype(int),
            'd_cell': d_cell,
            'd_empty': d_empty,
            'epsilon': epsilon,
            'rho': rho,
            'cell_mean_umi': {str(i + 1): u for i, u in enumerate(cell_mean_umi)},
            'cell_lognormal_sigma': cell_lognormal_sigma,
            'empty_mean_umi': empty_mean_umi,
            'empty_lognormal_sigma': empty_lognormal_sigma,
            'alpha0': alpha0,
            'epsilon_param': epsilon_param,
            'dirichlet_alpha': dirichlet_alpha,
            'model': [model_type]}


@torch.no_grad()
def generate_sample_model_dataset(
        n_genes: int = 10000,
        cells_of_each_type: List[int] = [100, 100, 100],
        n_droplets: int = 5000,
        model_type: str = 'full',
        dirichlet_alpha: Union[float, np.ndarray] = 0.05,
        chi: Optional[np.ndarray] = None,
        chi_artificial_similarity: float = 0,
        vector_to_add_to_chi_ambient: Optional[np.ndarray] = None,
        cell_mean_umi: List[int] = [5000],
        cell_lognormal_sigma: float = 0.2,
        empty_mean_umi: int = 200,
        empty_lognormal_sigma: float = 0.4,
        epsilon_param: float = 100,
        rho_alpha: float = 3,
        rho_beta: float = 80,
        phi: float = 0.1,
        random_seed: int = 0) -> Dict[str, Union[float, np.ndarray, sp.csr_matrix]]:
    """Create a sample dataset for use as 'truth' data.
    """

    # Input checking.
    assert len(cells_of_each_type) > 0, 'cells_of_each_type must be a List of ints'
    n_cell_types = len(cells_of_each_type)
    if vector_to_add_to_chi_ambient is not None:
        assert len(vector_to_add_to_chi_ambient) == n_genes, \
            f'vector_to_add_to_chi_ambient {vector_to_add_to_chi_ambient.shape} must ' \
            f'be of length n_genes ({n_genes})'
    if type(dirichlet_alpha) == np.ndarray:
        assert dirichlet_alpha.size == n_genes, \
            f'If you input a vector for dirichlet_alpha, its length ' \
            f'({dirichlet_alpha.size}) must be n_genes ({n_genes})'
    assert cell_lognormal_sigma > 0, 'cell_lognormal_sigma must be > 0'
    assert empty_lognormal_sigma > 0, 'empty_lognormal_sigma must be > 0'
    cell_mean_umi = list(cell_mean_umi)
    if len(cell_mean_umi) == 1:
        cell_mean_umi = cell_mean_umi * n_cell_types  # repeat the entry
    for umi in cell_mean_umi:
        assert umi > empty_mean_umi, \
            f'cell_size_mean_umi ({umi}) must be > empty_mean_umi ({empty_mean_umi})'
    assert len(cell_mean_umi) == len(cells_of_each_type), \
        'cells_of_each_type and cell_mean_umi (if a list) must be of the same length'
    assert (chi_artificial_similarity >= 0) and (chi_artificial_similarity <= 1), \
        'chi_artificial_similarity must be in the range [0, 1]'
    assert phi > 0, 'phi must be > 0'

    # Seed random number generators.
    comprehensive_random_seed(seed=random_seed)

    # Draw gene expression profiles for each cell type.
    if chi is None:
        chi = torch.zeros((n_cell_types, n_genes)).to(DEVICE)
        for i in range(chi.shape[0]):
            chi[i, :] = draw_random_chi(alpha=dirichlet_alpha, n_genes=n_genes)
            if i > 0:
                # Make expression profiles similar (if chi_artificial_similarity > 0) via gating.
                chi[i, :] = chi[i, :] * (1 - chi_artificial_similarity) + chi[0, :] * chi_artificial_similarity
    else:
        dirichlet_alpha = ['None: chi was input to generate_sample_model_dataset()']

    # Get chi_ambient: a weighted average of expression, possibly with extra.
    chi_ambient = torch.zeros(n_genes).to(DEVICE)
    for i in range(n_cell_types):
        chi_ambient = chi_ambient + chi[i, :] * cells_of_each_type[i] * cell_mean_umi[i]
    if vector_to_add_to_chi_ambient is None:
        vector_to_add_to_chi_ambient = torch.zeros(n_genes)
    chi_ambient = chi_ambient + vector_to_add_to_chi_ambient
    chi_ambient = chi_ambient / chi_ambient.sum()

    c_real = torch.zeros((n_droplets, n_genes)).to(DEVICE)
    c_bkg = torch.zeros((n_droplets, n_genes)).to(DEVICE)
    labels = torch.zeros(n_droplets).to(DEVICE)
    epsilon = torch.zeros(n_droplets).to(DEVICE)
    rho = torch.zeros(n_droplets).to(DEVICE)
    d_cell = torch.zeros(n_droplets).to(DEVICE)
    d_empty = torch.zeros(n_droplets).to(DEVICE)

    # Sample counts from cell-containing droplets.
    i = 0
    for t, (num_cells, celltype_umi) in enumerate(zip(cells_of_each_type, cell_mean_umi)):
        sim = sample_from_model(
            num=num_cells,
            chi=chi[t, :],
            d_mu=np.log(celltype_umi),
            d_sigma=cell_lognormal_sigma,
            v_mu=np.log(empty_mean_umi),
            v_sigma=empty_lognormal_sigma,
            y=1,  # cell present
            chi_ambient=chi_ambient,
            eps_param=epsilon_param,
            rho_alpha=rho_alpha,
            rho_beta=rho_beta,
            phi=phi,
            model_type=model_type,
        )
        c_real[i:(i + num_cells), :] = sim['counts_real']
        c_bkg[i:(i + num_cells), :] = sim['counts_bkg']
        epsilon[i:(i + num_cells)] = sim['epsilon']
        rho[i:(i + num_cells)] = sim['rho']
        d_cell[i:(i + num_cells)] = sim['cell_counts']
        d_empty[i:(i + num_cells)] = sim['empty_counts']
        labels[i:(i + num_cells)] = t + 1  # label cell types with integers > 0
        i = i + num_cells

    # Sample counts from empty droplets.
    sim = sample_from_model(
        num=n_droplets - i,
        chi=chi[0, :],  # this doesn't get used since y=0
        d_mu=1,  # this doesn't get used since y=0
        d_sigma=1,  # this doesn't get used since y=0
        v_mu=np.log(empty_mean_umi),
        v_sigma=empty_lognormal_sigma,
        y=0,  # no cell present
        chi_ambient=chi_ambient,
        eps_param=epsilon_param,
        rho_alpha=rho_alpha,
        rho_beta=rho_beta,
        phi=phi,
        model_type=model_type,
    )
    c_real[i:, :] = sim['counts_real']
    c_bkg[i:, :] = sim['counts_bkg']
    epsilon[i:] = sim['epsilon']
    rho[i:] = sim['rho']
    d_empty[i:] = sim['empty_counts']

    # Get sparse matrices from dense.
    counts_true = sp.csr_matrix(c_real.cpu().numpy(), shape=c_real.shape, dtype=int)
    counts_bkg = sp.csr_matrix(c_bkg.cpu().numpy(), shape=c_bkg.shape, dtype=int)

    return {'counts_true': counts_true,
            'counts_bkg': counts_bkg,
            'barcodes': np.array([f'bc{n:06d}' for n in np.arange(n_droplets)]),
            'gene_names': np.array([f'g{n:05d}' for n in np.arange(n_genes)]),
            'chi': {str(i + 1): chi[i, :].cpu().numpy() for i in range(chi.shape[0])},
            'chi_ambient': chi_ambient.cpu().numpy(),
            'droplet_labels': labels.int().cpu().numpy(),
            'd_cell': d_cell.cpu().numpy(),
            'd_empty': d_empty.cpu().numpy(),
            'epsilon': epsilon.cpu().numpy(),
            'rho': rho.cpu().numpy(),
            'cell_mean_umi': {str(i + 1): u for i, u in enumerate(cell_mean_umi)},
            'cell_lognormal_sigma': cell_lognormal_sigma,
            'empty_mean_umi': empty_mean_umi,
            'empty_lognormal_sigma': empty_lognormal_sigma,
            'epsilon_param': epsilon_param,
            'dirichlet_alpha': dirichlet_alpha,
            'phi': phi,
            'model': [model_type]}


def draw_random_chi(alpha: float = 1.,
                    n_genes: int = 10000) -> torch.Tensor:
    """Sample a gene expression vector, chi, from a Dirichlet prior.
    Args:
        alpha: Concentration parameter for Dirichlet distribution, to be
            expanded into a vector to use as the Dirichlet concentration
            parameter.
        n_genes: Number of genes.
    Returns:
        chi: Vector of fractional gene expression, drawn from a Dirichlet
            distribution.
    """

    assert alpha > 0, "Concentration parameter, alpha, must be > 0."
    assert n_genes > 0, "Number of genes, n_genes, must be > 0."

    # Draw gene expression from a Dirichlet distribution.
    chi = torch.distributions.Dirichlet(alpha * torch.ones(n_genes).to(DEVICE)).sample().squeeze()

    return chi


def sample_from_inferred_model(num: int,
                               model: 'RemoveBackgroundPyroModel',
                               z: Optional[torch.Tensor] = None,
                               y: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
    """Sample data from the model, where the model is not completely naive, but
    uses a decoder trained on real data."""

    # Run the model, .
    data = torch.zeros(size=(num, model.n_genes))  # model uses shape only
    tracing_model = model.model
    do_list = []
    if z is not None:
        tracing_model = pyro.poutine.do(tracing_model, data={'z': z})
        do_list.append('z')
    if y is not None:
        tracing_model = pyro.poutine.do(tracing_model, data={'y': y})
        do_list.append('z')

    model_trace = pyro.poutine.trace(tracing_model).get_trace(data)

    # Get outputs of model (mu, alpha, lambda).
    model_output = model_trace.nodes['_RETURN']['value']

    # Get traced parameters.
    params = {name: (model_trace.nodes[name]['value'].unconstrained()
                     if (model_trace.nodes[name]['type'] == 'param')
                     else model_trace.nodes[name]['value'])
              for name in ['chi'] + model_trace.param_nodes + model_trace.stochastic_nodes
              if (not ('$$$' in name) and name not in do_list)}  # the do-samples are dummies

    # Extract latent values.
    epsilon = params['epsilon']
    rho = params['rho']
    d = params['d_cell']
    d_empty = params['d_empty']
    y = y if (y is not None) else params['y']  # the "do" does not show up in the trace, apparently
    chi = params['chi']
    chi_bar = model.avg_gene_expression
    chi_ambient = params['chi_ambient']  # pull from the (loaded) param store

    # Because the model integrates out real counts and noise counts, we sample here.
    logit = torch.log(model_output['mu']) - torch.log(model_output['alpha'])
    c_real = pyro.distributions.NegativeBinomial(total_count=model_output['alpha'],
                                                 logits=logit).to_event(1).sample()
    c_bkg = pyro.distributions.Poisson(model_output['lam']).to_event(1).sample()

    # Output observed counts are the sum, but return them separately.
    return {'counts_real': c_real.detach().cpu().numpy(),
            'counts_bkg': c_bkg.detach().cpu().numpy(),
            'cell_counts': (d * y).detach().cpu().numpy(),
            'empty_counts': d_empty.detach().cpu().numpy(),
            'epsilon': epsilon.detach().cpu().numpy(),
            'rho': rho.detach().cpu().numpy(),
            'chi': chi.detach().cpu().numpy()}


def sample_from_model(
        num: int,
        chi: torch.Tensor,
        d_mu: float,
        d_sigma: float,
        v_mu: float,
        v_sigma: float,
        y: int,
        chi_ambient: torch.Tensor,
        eps_param: float,
        model_type: str = 'full',
        rho_alpha: float = 3,
        rho_beta: float = 80,
        phi: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Draw samples of cell expression profiles using the negative binomial -
    Poisson sum model.

    NOTE: Single cell type with expression chi.
    NOTE: Random number seeding should be done before this function call.

    Args:
        num: Number of expression profiles to draw.
        chi: Cell expression profile, size (n_gene).
        d_mu: Mean of LogNormal cell size distribution.
        d_sigma: Scale parameter of LogNormal cell size distribution.
        v_mu: Mean of LogNormal empty size distribution.
        v_sigma: Scale parameter of LogNormal empty size distribution.
        y: 1 for cell(s), 0 for empties.
        chi_ambient: Ambient gene expression profile (sums to one).
        eps_param: Parameter for gamma distribution of the epsilon droplet
            efficiency factor ~ Gamma(eps_param, 1/eps_param), i.e. mean is 1.
        model_type: ['ambient', 'swapping', 'full']
        rho_alpha: Beta distribution param alpha for swapping fraction.
        rho_beta: Beta distribution param beta for swapping fraction.
        phi: The negative binomial overdispersion parameter.
    Returns:
        Tuple of (c_real, c_bkg)
        c_real: Count matrix (cells, genes) for real cell counts.
        c_bkg: Count matrix (cells, genes) for background RNA.
    """

    # Check inputs.
    assert y in [0, 1], f'y must be 0 or 1, but was {y}'
    assert d_mu > 0, f'd_mu must be > 0, but was {d_mu}'
    assert d_sigma > 0, f'd_sigma must be > 0, but was {d_sigma}'
    assert v_mu > 0, f'v_mu must be > 0, but was {v_mu}'
    assert v_sigma > 0, f'v_sigma must be > 0, but was {v_sigma}'
    assert (chi >= 0).all(), 'all chi values must be >= 0.'
    assert (chi_ambient >= 0).all(), 'all chi_ambient must be >= 0.'
    assert (1. - chi_ambient.sum()).abs() < 1e-6, f'chi_ambient must sum ' \
        f'to 1, but it sums to {chi_ambient.sum()}'
    assert len(chi_ambient.shape) == 1, 'chi_ambient should be 1-dimensional.'
    assert chi.shape[-1] == chi_ambient.shape[-1], 'chi and chi_ambient must ' \
        'be the same size in the rightmost dimension.'
    assert (1. - chi.sum()).abs() < 1e-6, f'chi must sum ' \
        f'to 1, but it sums to {chi.sum()}'
    assert num > 0, f'num must be > 0, but was {num}'
    assert eps_param > 1, f'eps_param must be > 1, but was {eps_param}'

    # Draw epsilon ~ Gamma(eps_param, 1 / eps_param)
    epsilon = torch.distributions.Gamma(concentration=eps_param, rate=eps_param).sample([num])

    # Draw d ~ LogNormal(d_mu, d_sigma)
    d = torch.distributions.LogNormal(loc=d_mu, scale=d_sigma).sample([num])

    # Draw d ~ LogNormal(d_mu, d_sigma)
    v = torch.distributions.LogNormal(loc=v_mu, scale=v_sigma).sample([num])

    # Draw rho ~ Beta(rho_alpha, rho_beta)
    rho = torch.distributions.Beta(rho_alpha, rho_beta).sample([num])

    mu = calculate_mu(epsilon=epsilon,
                      d_cell=d,
                      chi=chi,
                      y=torch.ones(d.shape).to(DEVICE) * y,
                      rho=rho,
                      model_type=model_type) + 1e-30

    # Draw cell counts ~ NB(mean = y * epsilon * d * chi, overdispersion = phi)
    alpha = 1. / phi
    logits = (mu.log() - np.log(alpha))
    c_real = torch.distributions.NegativeBinomial(total_count=alpha,
                                                  logits=logits).sample()

    lam = calculate_lambda(epsilon=epsilon,
                           chi_ambient=chi_ambient,
                           d_cell=d,
                           d_empty=v,
                           y=torch.ones(d.shape).to(DEVICE) * y,
                           rho=rho,
                           chi_bar=chi_ambient,
                           model_type=model_type) + 1e-30

    # Draw empty counts ~ Poisson(epsilon * v * chi_ambient)
    c_bkg = torch.distributions.Poisson(rate=lam).sample()

    # Output observed counts are the sum, but return them separately.
    return {'counts_real': c_real,
            'counts_bkg': c_bkg,
            'cell_counts': d * y,
            'empty_counts': v,
            'epsilon': epsilon,
            'rho': rho}


def sample_from_dirichlet_model(num: int,
                                alpha: np.ndarray,
                                d_mu: float,
                                d_sigma: float,
                                v_mu: float,
                                v_sigma: float,
                                y: int,
                                chi_ambient: np.ndarray,
                                eps_param: float,
                                rng: Optional[np.random.RandomState] = None,
                                model_type: str = 'full',
                                rho_alpha: float = 3,
                                rho_beta: float = 80) -> Dict[str, np.ndarray]:
    """Draw samples of cell expression profiles using the Dirichlet-Poisson,
    Poisson sum model.
    Args:
        num: Number of expression profiles to draw.
        alpha: Dirichlet concentration parameters for cell expression profile,
            size (n_gene).
        d_mu: Mean of LogNormal cell size distribution.
        d_sigma: Scale parameter of LogNormal cell size distribution.
        v_mu: Mean of LogNormal empty size distribution.
        v_sigma: Scale parameter of LogNormal empty size distribution.
        y: 1 for cell(s), 0 for empties.
        chi_ambient: Ambient gene expression profile (sums to one).
        eps_param: Parameter for gamma distribution of the epsilon droplet
            efficiency factor ~ Gamma(eps_param, 1/eps_param), i.e. mean is 1.
        rng: A random number generator.
        model_type: ['ambient', 'swapping', 'full']
        rho_alpha: Beta distribution param alpha for swapping fraction.
        rho_beta: Beta distribution param beta for swapping fraction.
    Returns:
        Tuple of (c_real, c_bkg)
        c_real: Count matrix (cells, genes) for real cell counts.
        c_bkg: Count matrix (cells, genes) for background RNA.
    """

    # Check inputs.
    assert y in [0, 1], f'y must be 0 or 1, but was {y}'
    assert d_mu > 0, f'd_mu must be > 0, but was {d_mu}'
    assert d_sigma > 0, f'd_sigma must be > 0, but was {d_sigma}'
    assert v_mu > 0, f'v_mu must be > 0, but was {v_mu}'
    assert v_sigma > 0, f'v_sigma must be > 0, but was {v_sigma}'
    assert np.all(alpha > 0), 'all alphas must be > 0.'
    assert np.all(chi_ambient >= 0), 'all chi_ambient must be >= 0.'
    assert np.abs(1. - chi_ambient.sum()) < 1e-10, f'chi_ambient must sum ' \
        f'to 1, but it sums to {chi_ambient.sum()}'
    assert len(chi_ambient.shape) == 1, 'chi_ambient should be 1-dimensional.'
    assert alpha.shape[0] == chi_ambient.size, 'alpha and chi_ambient must ' \
        'be the same size in the rightmost dimension.'
    assert num > 0, f'num must be > 0, but was {num}'
    assert eps_param > 1, f'eps_param must be > 1, but was {eps_param}'

    # Seed random number generator.
    if rng is None:
        rng = np.random.RandomState(seed=0)

    # Draw chi ~ Dir(alpha)
    chi = rng.dirichlet(alpha=alpha, size=num)

    # Draw epsilon ~ Gamma(eps_param, 1 / eps_param)
    epsilon = rng.gamma(shape=eps_param, scale=1. / eps_param, size=num)

    # Draw d ~ LogNormal(d_mu, d_sigma)
    d = rng.lognormal(mean=d_mu, sigma=d_sigma, size=num)

    # Draw d ~ LogNormal(d_mu, d_sigma)
    v = rng.lognormal(mean=v_mu, sigma=v_sigma, size=num)

    # Draw rho ~ Beta(rho_alpha, rho_beta)
    rho = rng.beta(a=rho_alpha, b=rho_beta, size=num)

    mu = calculate_mu(epsilon=torch.tensor(epsilon),
                      d_cell=torch.tensor(d),
                      chi=torch.tensor(chi),
                      y=torch.ones(d.shape) * y,
                      rho=torch.tensor(rho),
                      model_type=model_type).numpy()

    # Draw cell counts ~ Poisson(y * epsilon * d * chi)
    c_real = rng.poisson(lam=mu, size=(num, chi_ambient.size))

    lam = calculate_lambda(epsilon=torch.tensor(epsilon),
                           chi_ambient=torch.tensor(chi_ambient),
                           d_cell=torch.tensor(d),
                           d_empty=torch.tensor(v),
                           y=torch.ones(d.shape) * y,
                           rho=torch.tensor(rho),
                           chi_bar=torch.tensor(chi_ambient),
                           model_type=model_type).numpy()

    # Draw empty counts ~ Poisson(epsilon * v * chi_ambient)
    c_bkg = rng.poisson(lam=lam, size=(num, chi_ambient.size))

    # Output observed counts are the sum, but return them separately.
    return {'counts_real': c_real,
            'counts_bkg': c_bkg,
            'cell_counts': d * y,
            'empty_counts': v,
            'epsilon': epsilon,
            'rho': rho}


def get_dataset_dict_as_anndata(
        sample_dataset: Dict[str, Union[float, np.ndarray, sp.csr_matrix]]
) -> 'anndata.AnnData':
    """Return a simulated dataset as an AnnData object."""

    import anndata

    d = sample_dataset.copy()

    # counts
    if 'counts_true' in d.keys() and 'counts_bkg' in d.keys():
        adata = anndata.AnnData(X=(d['counts_true'] + d['counts_bkg']).astype(np.float32),
                                obs={'barcode': d.pop('barcodes')},
                                var={'gene': d.pop('gene_names')})
        adata.layers['counts_true'] = d.pop('counts_true')
        adata.layers['counts'] = adata.layers['counts_true'] + d.pop('counts_bkg')
    elif 'matrix' in d.keys():
        adata = anndata.AnnData(X=d.pop('matrix').astype(np.float32),
                                obs={'barcode': d.pop('barcodes')},
                                var={'gene': d.pop('gene_names')})

    # obs
    obs_keys = ['droplet_labels', 'd_cell', 'epsilon']
    for key in obs_keys:
        adata.obs[key] = d.pop(key, None)

    # uns
    for key, value in d.items():
        adata.uns[key] = value

    return adata


def write_simulated_data_to_h5(output_file: str,
                               d: Dict[str, Union[float, np.ndarray, sp.csr_matrix]],
                               cellranger_version: int = 3) -> bool:
    """Helper function to write the full (noisy) simulate dataset to an H5 file.

    Args:
        output_file: File name
        d: Resulting dict from `generate_sample_dataset`

    Returns:
        True if save was successful.
    """

    assert cellranger_version in [2, 3], 'cellranger_version must be 2 or 3'

    return write_matrix_to_cellranger_h5(
        output_file=output_file,
        gene_names=d['gene_names'],
        feature_types=np.array(['Gene Expression'] * d['gene_names'].size),
        genomes=(np.array(['simulated'] * d['gene_names'].size)
                 if ('genome' not in d.keys()) else d['genome']),
        barcodes=d['barcodes'],
        count_matrix=(d['counts_true'] + d['counts_bkg']).tocsc(),
        cellranger_version=cellranger_version,
    )


def write_simulated_truth_to_h5(output_file: str,
                                d: Dict[str, Union[float, np.ndarray, sp.csr_matrix]],
                                cellranger_version: int = 3) -> bool:
    """Helper function to write the full (noisy) simulate dataset to an H5 file.

    Args:
        output_file: File name
        d: Resulting dict from `generate_sample_dataset`

    Returns:
        True if save was successful.
    """

    assert cellranger_version in [2, 3], 'cellranger_version must be 2 or 3'

    latents = set(d.keys())
    extra_latents = latents - {'gene_names', 'genome', 'barcodes', 'counts_true',
                               'droplet_labels', 'd_cell', 'd_empty', 'epsilon', 'rho',
                               'chi', 'cell_mean_umi', 'counts_bkg'}
    global_latents = {f'truth_{key.replace("chi_ambient", "ambient_expression")}': d[key]
                      for key in extra_latents}

    for i, v in d['chi'].items():
        global_latents.update({f'truth_gene_expression_cell_label_{i}': v})
    for i, v in d['cell_mean_umi'].items():
        global_latents.update({f'truth_mean_umi_cell_label_{i}': v})

    return write_matrix_to_cellranger_h5(
        output_file=output_file,
        gene_names=d['gene_names'],
        feature_types=np.array(['Gene Expression'] * d['gene_names'].size),
        genomes=(np.array(['simulated'] * d['gene_names'].size)
                 if ('genome' not in d.keys()) else d['genome']),
        barcodes=d['barcodes'],
        count_matrix=d['counts_true'].tocsc(),  # just the truth, no background
        cellranger_version=cellranger_version,
        local_latents={'truth_cell_label': d['droplet_labels'],
                       'truth_cell_size': d['d_cell'],
                       'truth_empty_droplet_size': d['d_empty'],
                       'truth_droplet_efficiency': d['epsilon'],
                       'truth_cell_probability': (d['droplet_labels'] != 0).astype(float),
                       'truth_swapping_fraction': d['rho']},
        global_latents=global_latents,
    )
