"""Simulate a basic scRNA-seq count matrix dataset, for unit tests."""

import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple, List, Union


def simulate_dataset_without_ambient_rna(
        n_cells: int = 100,
        clusters: int = 1,
        n_genes: int = 10000,
        cells_in_clusters: Union[List[int], None] = None,
        d_cell: int = 5000) -> Tuple[sp.csr.csr_matrix,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray]:
    """Simulate a dataset with ambient background RNA counts.

    Empty drops have ambient RNA only, while barcodes with cells have cell RNA
    plus some amount of ambient background RNA (in proportion to the sizes of
    cell and droplet).

    Args:
        n_cells: Number of cells.
        clusters: Number of distinct cell types to simulate.
        n_genes: Number of genes.
        d_cell: Cell size scale factor.
        cells_in_clusters: Number of cells of each cell type.  If specified,
            the number of ints in this list must be equal to clusters.

    Returns:
        csr_barcode_gene_synthetic: The simulated barcode by gene matrix of UMI
            counts, as a scipy.sparse.csr.csr_matrix.
        z: The simulated cell type identities.  A numpy array of integers, one
            for each barcode.  The number 0 is used to denote barcodes
            without a cell present.
        chi: The simulated gene expression, one corresponding to each z.
            Access the vector of gene expression for a given z using chi[z, :].
        d: The simulated size scale factors, one for each barcode.

    """

    assert d_cell > 0, "Location parameter, d_cell, of LogNormal " \
                       "distribution must be greater than zero."
    assert clusters > 0, "clusters must be a positive integer."
    assert n_cells > 0, "n_cells must be a positive integer."
    assert n_genes > 0, "n_genes must be a positive integer."

    # Figure out how many cells are in each cell cluster.
    if cells_in_clusters is None:
        # No user input: make equal numbers of each cell type
        cells_in_clusters = np.ones(clusters) * int(n_cells / clusters)
    else:
        assert len(cells_in_clusters) == clusters, "len(cells_in_clusters) " \
                                                   "must equal clusters."
        assert sum(cells_in_clusters) == n_cells, "sum(cells_in_clusters) " \
                                                  "must equal n_cells."

    # Initialize arrays and lists.
    chi = np.zeros((clusters + 1, n_genes))
    csr_list = []
    z = []
    d = []

    # Get chi for cell expression.
    for i in range(clusters):
        chi[i, :] = generate_chi(alpha=1.0, n_genes=n_genes)
        csr, d_n = sample_expression_from(chi[i, :],
                                          n=int(cells_in_clusters[i]),
                                          d_mu=np.log(d_cell).item())
        csr_list.append(csr)
        z = z + [i for _ in range(csr.shape[0])]
        d = d + [j for j in d_n]

    # Package the results.
    csr_barcode_gene_synthetic = sp.vstack(csr_list)
    z = np.array(z)
    d = np.array(d)

    # Permute the barcode order and return results.
    order = np.random.permutation(z.size)
    csr_barcode_gene_synthetic = csr_barcode_gene_synthetic[order, ...]
    z = z[order]
    d = d[order]

    return csr_barcode_gene_synthetic, z, chi, d


def simulate_dataset_with_ambient_rna(
        n_cells: int = 150,
        n_empty: int = 300,
        clusters: int = 3,
        n_genes: int = 10000,
        d_cell: int = 5000,
        d_empty: int = 100,
        cells_in_clusters: Union[List[int], None] = None,
        ambient_different: bool = False,
        chi_input: Union[np.ndarray, None] = None) \
        -> Tuple[sp.csr.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a dataset with ambient background RNA counts.

    Empty drops have ambient RNA only, while barcodes with cells have cell
    RNA plus some amount of ambient background RNA (in proportion to the
    sizes of cell and droplet).

    Args:
        n_cells: Number of cells.
        n_empty: Number of empty droplets with only ambient RNA.
        clusters: Number of distinct cell types to simulate.
        n_genes: Number of genes.
        d_cell: Cell size scale factor.
        d_empty: Empty droplet size scale factor.
        cells_in_clusters: Number of cells of each cell type.  If specified,
            the number of ints in this list must be equal to clusters.
        ambient_different: If False, the gene expression profile of ambient
            RNA is drawn from the sum of cellular gene expression.  If True,
            the ambient RNA expression is completely different from cellular
            gene expression.
        chi_input: Gene expression arrays in a matrix, with rows as clusters and
            columns as genes.  Expression should add to one for each row.
            Setting chi=None will generate new chi randomly according to a
            Dirichlet distribution.

    Returns:
        csr_barcode_gene_synthetic: The simulated barcode by gene matrix of
            UMI counts, as a scipy.sparse.csr.csr_matrix.
        z: The simulated cell type identities.  A numpy array of integers,
            one for each barcode. The number 0 is used to denote barcodes
            without a cell present.
        chi: The simulated gene expression, one corresponding to each z.
            Access the vector of gene expression for a given z using chi[z, :].
        d: The simulated size scale factors, one for each barcode.

    """

    assert d_cell > 0, "Location parameter, d_cell, of LogNormal " \
                       "distribution must be greater than zero."
    assert d_empty > 0, "Location parameter, d_cell, of LogNormal " \
                        "distribution must be greater than zero."
    assert clusters > 0, "clusters must be a positive integer."
    assert n_cells > 0, "n_cells must be a positive integer."
    assert n_empty > 0, "n_empty must be a positive integer."
    assert n_genes > 0, "n_genes must be a positive integer."
    if chi_input is not None:
        assert chi_input.shape[0] == clusters, "Chi was specified, but the " \
                                               "number  of rows must match " \
                                               "the number of clusters."
        assert chi_input.shape[1] == n_genes, "Chi was specified, but the " \
                                              "number of columns must match " \
                                              "the number of genes."

    # Figure out how many cells are in each cell cluster.
    if cells_in_clusters is None:
        # No user input: make equal numbers of each cell type
        cells_in_clusters = (np.ones(clusters, dtype=int)
                             * int(n_cells/clusters))
    else:
        assert len(cells_in_clusters) == clusters, "len(cells_in_clusters) " \
                                                   "must equal clusters."
        assert sum(cells_in_clusters) == n_cells, "sum(cells_in_clusters) " \
                                                  "must equal n_cells."

    # Initialize arrays and lists.
    chi = np.zeros((clusters+1, n_genes))
    csr_list = []
    z = []
    d = []

    if chi_input is not None:

        # Go with the chi that was input.
        chi[1:, :] = chi_input

    else:
    
        # Get chi for cell expression.
        for i in range(1, clusters+1):
            chi[i, :] = generate_chi(alpha=0.01, n_genes=n_genes)
    
    # Get chi for ambient expression.  This becomes chi[0, :].
    if ambient_different:

        # Ambient expression is unrelated to cells, and is itself random.
        chi[0, :] = generate_chi(alpha=0.001, n_genes=n_genes)  # Sparse

    else:

        # Ambient gene expression comes from the sum of cell expression.
        for i in range(1, clusters+1):

            chi[0, :] += cells_in_clusters[i-1] * chi[i, :]  # Weighted sum

    chi[0, :] = chi[0, :] / np.sum(chi[0, :])  # Normalize
    
    # Sample gene expression for ambient.
    csr, d_n = sample_expression_from(chi[0, :],
                                      n=n_empty,
                                      d_mu=np.log(d_empty).item())

    # Add data to lists.
    csr_list.append(csr)
    z = z + [0 for _ in range(csr.shape[0])]
    d = d + [i for i in d_n]
    
    # Sample gene expression for cells.
    for i in range(1, clusters+1):

        # Get chi for cells once ambient expression is added.
        chi_tilde = chi[i, :] * d_cell + chi[0, :] * d_empty
        chi_tilde = chi_tilde / np.sum(chi_tilde)  # Normalize
        csr, d_n = sample_expression_from(chi_tilde, 
                                          n=cells_in_clusters[i-1],
                                          d_mu=np.log(d_cell).item())

        # Add data to lists.
        csr_list.append(csr)
        z = z + [i for _ in range(csr.shape[0])]
        d = d + [j for j in d_n]

    # Package the results.
    csr_barcode_gene_synthetic = sp.vstack(csr_list)
    z = np.array(z)
    d = np.array(d)

    # Permute the barcode order and return results.
    order = np.random.permutation(z.size)
    csr_barcode_gene_synthetic = csr_barcode_gene_synthetic[order, ...]
    z = z[order]
    d = d[order]
    
    return csr_barcode_gene_synthetic, z, chi, d


def generate_chi(alpha: float = 1., n_genes: int = 10000) -> np.ndarray:
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
    chi = np.random.dirichlet(alpha * np.ones(n_genes), size=1).squeeze()
    
    # Normalize gene expression and return result.
    chi = chi / np.sum(chi)
    
    return chi


def sample_expression_from(chi: np.ndarray,
                           n: int = 100,
                           d_mu: float = np.log(5000).item(),
                           d_sigma: float = 0.2,
                           phi: float = 0.3) -> Tuple[sp.csr.csr_matrix,
                                                      np.ndarray]:
    """Generate a count matrix given a mean expression distribution.
    
    Args:
        chi: Normalized gene expression vector (sums to one).
        n: Number of desired cells to simulate.
        d_mu: Log mean number of UMI counts per cell.
        d_sigma: Standard deviation of a normal in log space for the number
            of UMI counts per cell.
        phi: The overdispersion parameter of a negative binomial,
            i.e., variance = mean + phi * mean^2
    
    Returns:
        csr_cell_gene: scipy.sparse.csr_matrix of gene expression
            counts per cell, with cells in axis=0 and genes in axis=1.
    
    Note:
        Draw gene expression from a negative binomial distribution
        counts ~ NB(d*chi, phi)

    """

    assert phi > 0, "Phi must be greater than zero in the negative binomial."
    assert d_sigma > 0, "Scale parameter, d_sigma, of LogNormal distribution " \
                        " must be greater than zero."
    assert d_mu > 0, "Location parameter, d_mu, of LogNormal distribution " \
                     " must be greater than zero."
    assert n > 0, "Number of cells to simulate, n, must be a positive integer."
    assert chi.min() >= 0, "Minimum allowed value in chi vector is zero."

    n_genes = chi.size  # Number of genes

    # Initialize arrays.
    barcodes = np.arange(n)
    genes = np.arange(n_genes)
    predicted_reads = int(np.exp(d_mu) * n * 2)  # Guess array sizes
    coo_bc_list = np.zeros(predicted_reads, dtype=np.uint32)
    coo_gene_list = np.zeros(predicted_reads, dtype=np.uint32)
    coo_count_list = np.zeros(predicted_reads, dtype=np.uint32)
    d = np.zeros(n)
    a = 0

    # Go barcode by barcode, sampling UMI counts per gene.
    for i in range(n):

        # Sample cell size parameter from a LogNormal distribution.
        d[i] = np.exp(np.random.normal(loc=d_mu, scale=d_sigma, size=1))

        # Sample counts from a negative binomial distribution.
        gene_counts = neg_binom(d[i] * chi, phi, size=n_genes)

        # Keep only the non-zero counts to populate the sparse matrix.
        num_nonzeros = np.sum(gene_counts > 0)

        # Check whether arrays need to be re-sized to accommodate more entries.
        if (a + num_nonzeros) < coo_count_list.size:
            # Fill in.
            coo_bc_list[a:a+num_nonzeros] = barcodes[i]
            coo_gene_list[a:a+num_nonzeros] = genes[gene_counts > 0]
            coo_count_list[a:a+num_nonzeros] = gene_counts[gene_counts > 0]
        else:
            # Resize arrays by doubling.
            coo_bc_list = np.resize(coo_bc_list, coo_bc_list.size * 2)
            coo_gene_list = np.resize(coo_gene_list, coo_gene_list.size * 2)
            coo_count_list = np.resize(coo_count_list, coo_count_list.size * 2)
            # Fill in.
            coo_bc_list[a:a+num_nonzeros] = barcodes[i]
            coo_gene_list[a:a+num_nonzeros] = genes[gene_counts > 0]
            coo_count_list[a:a+num_nonzeros] = gene_counts[gene_counts > 0]
            
        a += num_nonzeros

    # Lop off any unused zero entries at the end of the arrays.
    coo_bc_list = coo_bc_list[coo_count_list > 0]
    coo_gene_list = coo_gene_list[coo_count_list > 0]
    coo_count_list = coo_count_list[coo_count_list > 0]
    
    # Package data into a scipy.sparse.coo.coo_matrix.
    count_matrix = sp.coo_matrix((coo_count_list, (coo_bc_list, coo_gene_list)),
                                 shape=(barcodes.size, n_genes),
                                 dtype=np.uint32)
    
    # Convert to a scipy.sparse.csr.csr_matrix and return.
    count_matrix = count_matrix.tocsr()

    return count_matrix, d


def neg_binom(mu: float, phi: float, size: int = 1) -> np.ndarray:
    """Parameterize numpy's negative binomial distribution
    in terms of the mean and the overdispersion.

    Args:
        mu: Mean of the distribution
        phi: Overdispersion, such that variance = mean + phi * mean^2
        size: How many numbers to return

    Returns:
        'size' number of random draws from a negative binomial distribution.

    Note:
        Setting phi=0 turns the negative binomial distribution into a
        Poisson distribution.

    """

    assert phi > 0, "Phi must be greater than zero in the negative binomial."
    assert size > 0, "Number of draws from negative binomial, size, must " \
                     "be a positive integer."

    n = 1. / phi
    p = n / (mu + n)
    return np.random.negative_binomial(n, p, size=size)


def sample_from_dirichlet_model(num: int,
                                alpha: np.ndarray,
                                d_mu: float,
                                d_sigma: float,
                                v_mu: float,
                                v_sigma: float,
                                y: int,
                                chi_ambient: np.ndarray,
                                eps_param: float,
                                random_seed: int = 0,
                                include_swapping: bool = False,
                                rho_alpha: float = 3,
                                rho_beta: float = 80) -> Tuple[np.ndarray,
                                                                         np.ndarray]:
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
        random_seed: Seed a random number generator.
        include_swapping: Whether to include swapping in the model.
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
    assert np.all(chi_ambient > 0), 'all chi_ambient must be > 0.'
    assert np.abs(1. - chi_ambient.sum()) < 1e-10, f'chi_ambient must sum to 1, but it sums ' \
        f'to {chi_ambient.sum()}'
    assert len(chi_ambient.shape) == 1, 'chi_ambient should be 1-dimensional.'
    assert alpha.shape[0] == chi_ambient.size, 'alpha and chi_ambient must ' \
        'be the same size in the rightmost dimension.'
    assert num > 0, f'num must be > 0, but was {num}'
    assert eps_param > 1, f'eps_param must be > 1, but was {eps_param}'

    # Seed random number generator.
    rng = np.random.RandomState(seed=random_seed)

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

    # print(f'eps.shape is {epsilon.shape}')
    # print(f'd.shape is {d.shape}')
    # print(f'chi.shape is {chi.shape}')
    # print(f'rho.shape is {rho.shape}')

    mu = _calculate_mu(model_type='ambient' if not include_swapping else 'full',
                       epsilon=torch.Tensor(epsilon),
                       d_cell=torch.Tensor(d),
                       chi=torch.Tensor(chi),
                       y=torch.ones(d.shape) * y,
                       rho=torch.Tensor(rho)).numpy()

    # Draw cell counts ~ Poisson(y * epsilon * d * chi)
    c_real = rng.poisson(lam=mu,
                         size=(num, chi_ambient.size))

    lam = _calculate_lambda(model_type='ambient' if not include_swapping else 'full',
                            epsilon=torch.Tensor(epsilon),
                            chi_ambient=torch.Tensor(chi_ambient),
                            d_cell=torch.Tensor(d),
                            d_empty=torch.Tensor(v),
                            y=torch.ones(d.shape) * y,
                            rho=torch.Tensor(rho),
                            chi_bar=torch.Tensor(chi_ambient)).numpy()

    # Draw empty counts ~ Poisson(epsilon * v * chi_ambient)
    c_bkg = rng.poisson(lam=lam,
                        size=(num, chi_ambient.size))

    # Output observed counts are the sum, but return them separately.
    return c_real, c_bkg


def _calculate_lambda(model_type: str,
                      epsilon: torch.Tensor,
                      chi_ambient: torch.Tensor,
                      d_empty: torch.Tensor,
                      y: Union[torch.Tensor, None] = None,
                      d_cell: Union[torch.Tensor, None] = None,
                      rho: Union[torch.Tensor, None] = None,
                      chi_bar: Union[torch.Tensor, None] = None):
    """Calculate noise rate based on the model."""

    if model_type == "simple" or model_type == "ambient":
        lam = epsilon.unsqueeze(-1) * d_empty.unsqueeze(-1) * chi_ambient

    elif model_type == "swapping":
        lam = (rho.unsqueeze(-1) * y.unsqueeze(-1)
               * epsilon.unsqueeze(-1) * d_cell.unsqueeze(-1)
               + d_empty.unsqueeze(-1)) * chi_bar

    elif model_type == "full":
        lam = ((1 - rho.unsqueeze(-1)) * d_empty.unsqueeze(-1) * chi_ambient.unsqueeze(0)
               + rho.unsqueeze(-1)
               * (y.unsqueeze(-1) * epsilon.unsqueeze(-1) * d_cell.unsqueeze(-1)
                  + d_empty.unsqueeze(-1)) * chi_bar)
    else:
        raise NotImplementedError(f"model_type was set to {model_type}, "
                                  f"which is not implemented.")

    return lam


def _calculate_mu(model_type: str,
                  epsilon: torch.Tensor,
                  d_cell: torch.Tensor,
                  chi: torch.Tensor,
                  y: Union[torch.Tensor, None] = None,
                  rho: Union[torch.Tensor, None] = None):
    """Calculate mean expression based on the model."""

    if model_type == 'simple':
        mu = epsilon.unsqueeze(-1) * d_cell.unsqueeze(-1) * chi

    elif model_type == 'ambient':
        mu = (y.unsqueeze(-1) * epsilon.unsqueeze(-1)
              * d_cell.unsqueeze(-1) * chi)

    elif model_type == 'swapping' or model_type == 'full':
        mu = ((1 - rho.unsqueeze(-1))
              * y.unsqueeze(-1) * epsilon.unsqueeze(-1)
              * d_cell.unsqueeze(-1) * chi)

    else:
        raise NotImplementedError(f"model_type was set to {model_type}, "
                                  f"which is not implemented.")

    return mu
