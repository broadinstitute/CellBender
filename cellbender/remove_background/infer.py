# Posterior inference.

import pyro
import pyro.distributions as dist
import torch
import numpy as np
import scipy.sparse as sp

import cellbender.remove_background.consts as consts

from typing import Tuple, List, Dict, Optional
from abc import ABC, abstractmethod
import logging


class Posterior(ABC):
    """Base class Posterior handles posterior count inference.

    Args:
        dataset_obj: Dataset object.
        vi_model: Trained RemoveBackgroundPyroModel.
        counts_dtype: Data type of posterior count matrix.  Can be one of
            [np.uint32, np.float]
        float_threshold: For floating point count matrices, counts below
            this threshold will be set to zero, for the purposes of constructing
            a sparse matrix.  Unused if counts_dtype is np.uint32

    Properties:
        mean: Posterior count mean, as a sparse matrix.

    """

    def __init__(self,
                 dataset_obj: 'SingleCellRNACountsDataset',  # Dataset
                 vi_model: 'RemoveBackgroundPyroModel',
                 counts_dtype: np.dtype = np.uint32,
                 float_threshold: float = 0.5):
        self.dataset_obj = dataset_obj
        self.vi_model = vi_model
        self.use_cuda = vi_model.use_cuda
        self.analyzed_gene_inds = dataset_obj.analyzed_gene_inds
        self.count_matrix_shape = dataset_obj.data['matrix'].shape
        self.barcode_inds = np.arange(0, self.count_matrix_shape[0])
        self.dtype = counts_dtype
        self.float_threshold = float_threshold
        self._mean = None
        self._latents = None
        super(Posterior, self).__init__()

    @abstractmethod
    def _get_mean(self):
        """Obtain mean posterior counts and store in self._mean"""
        pass

    @property
    def mean(self) -> sp.csc_matrix:
        if self._mean is None:
            self._get_mean()
        return self._mean

    @property
    def latents(self) -> sp.csc_matrix:
        if self._latents is None:
            self._get_latents()
        return self._latents

    @property
    def variance(self):
        raise NotImplemented("Posterior count variance not implemented.")

    @torch.no_grad()
    def _get_latents(self):
        """Calculate the encoded latent variables."""

        data_loader = self.dataset_obj.get_dataloader(use_cuda=self.use_cuda,
                                                      analyzed_bcs_only=True,
                                                      batch_size=500,
                                                      shuffle=False)

        z = np.zeros((len(data_loader), self.vi_model.encoder['z'].output_dim))
        d = np.zeros(len(data_loader))
        p = np.zeros(len(data_loader))
        epsilon = np.zeros(len(data_loader))

        for i, data in enumerate(data_loader):

            if 'chi_ambient' in pyro.get_param_store().keys():
                chi_ambient = pyro.param('chi_ambient').detach()
            else:
                chi_ambient = None

            enc = self.vi_model.encoder.forward(x=data, chi_ambient=chi_ambient)
            ind = i * data_loader.batch_size
            z[ind:(ind + data.shape[0]), :] = enc['z']['loc'].detach().cpu().numpy()

            phi_loc = pyro.param('phi_loc')
            phi_scale = pyro.param('phi_scale')

            d[ind:(ind + data.shape[0])] = \
                dist.LogNormal(loc=enc['d_loc'],
                               scale=pyro.param('d_cell_scale')).mean.detach().cpu().numpy()

            p[ind:(ind + data.shape[0])] = enc['p_y'].sigmoid().detach().cpu().numpy()

            epsilon[ind:(ind + data.shape[0])] = dist.Gamma(enc['epsilon'] * self.vi_model.epsilon_prior,
                                                            self.vi_model.epsilon_prior).mean.detach().cpu().numpy()

        self._latents = {'z': z, 'd': d, 'p': p,
                         'phi_loc_scale': [phi_loc.item(), phi_scale.item()],
                         'epsilon': epsilon}

    @torch.no_grad()
    def _param_map_estimates(self,
                             data: torch.Tensor,
                             chi_ambient: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate MAP estimates of mu, the mean of the true count matrix, and
        lambda, the rate parameter of the Poisson background counts.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            chi_ambient: Point estimate of inferred ambient gene expression.

        Returns:
            mu_map: Dense tensor of Negative Binomial means for true counts.
            lambda_map: Dense tensor of Poisson rate params for noise counts.
            alpha_map: Dense tensor of Dirichlet concentration params that
                inform the overdispersion of the Negative Binomial.

        """

        # Encode latents.
        enc = self.vi_model.encoder.forward(x=data,
                                            chi_ambient=chi_ambient)
        z_map = enc['z']['loc']

        chi_map = self.vi_model.decoder.forward(z_map)
        phi_loc = pyro.param('phi_loc')
        phi_scale = pyro.param('phi_scale')
        phi_conc = phi_loc.pow(2) / phi_scale.pow(2)
        phi_rate = phi_loc / phi_scale.pow(2)
        alpha_map = 1. / dist.Gamma(phi_conc, phi_rate).mean

        y = (enc['p_y'] > 0).float()
        d_empty = dist.LogNormal(loc=pyro.param('d_empty_loc'),
                                 scale=pyro.param('d_empty_scale')).mean
        d_cell = dist.LogNormal(loc=enc['d_loc'],
                                scale=pyro.param('d_cell_scale')).mean
        epsilon = dist.Gamma(enc['epsilon'] * self.vi_model.epsilon_prior,
                             self.vi_model.epsilon_prior).mean

        if self.vi_model.include_rho:
            rho = pyro.param("rho_alpha") / (pyro.param("rho_alpha")
                                             + pyro.param("rho_beta"))
        else:
            rho = None

        # Calculate MAP estimates of mu and lambda.
        mu_map = self.vi_model.calculate_mu(epsilon=epsilon,
                                            d_cell=d_cell,
                                            chi=chi_map,
                                            y=y,
                                            rho=rho)
        lambda_map = self.vi_model.calculate_lambda(epsilon=epsilon,
                                                    chi_ambient=chi_ambient,
                                                    d_empty=d_empty,
                                                    y=y,
                                                    d_cell=d_cell,
                                                    rho=rho,
                                                    chi_bar=self.vi_model.avg_gene_expression)

        return {'mu': mu_map, 'lam': lambda_map, 'alpha': alpha_map}

    def dense_to_sparse(self,
                        chunk_dense_counts: np.ndarray) -> Tuple[List, List, List]:
        """Distill a batch of dense counts into sparse format.
        Barcode numbering is relative to the tensor passed in.
        """

        # TODO: speed up by keeping it a torch tensor as long as possible

        if chunk_dense_counts.dtype != np.int32:

            if self.dtype == np.uint32:

                # Turn the floating point count estimates into integers.
                decimal_values, _ = np.modf(chunk_dense_counts)  # Stuff after decimal.
                roundoff_counts = np.random.binomial(1, p=decimal_values)  # Bernoulli.
                chunk_dense_counts = np.floor(chunk_dense_counts).astype(dtype=int)
                chunk_dense_counts += roundoff_counts

            elif self.dtype == np.float32:

                # Truncate counts at a threshold value.
                chunk_dense_counts = (chunk_dense_counts *
                                      (chunk_dense_counts > self.float_threshold))

            else:
                raise NotImplementedError(f"Count matrix dtype {self.dtype} is not "
                                          f"supported.  Choose from [np.uint32, "
                                          f"np.float32]")

        # Find all the nonzero counts in this dense matrix chunk.
        nonzero_barcode_inds_this_chunk, nonzero_genes_trimmed = \
            np.nonzero(chunk_dense_counts)
        nonzero_counts = \
            chunk_dense_counts[nonzero_barcode_inds_this_chunk,
                               nonzero_genes_trimmed].flatten(order='C')

        # Get the original gene index from gene index in the trimmed dataset.
        nonzero_genes = self.analyzed_gene_inds[nonzero_genes_trimmed]

        return nonzero_barcode_inds_this_chunk, nonzero_genes, nonzero_counts


class ImputedPosterior(Posterior):
    """Posterior count inference using imputation to infer cell mean (d * chi).

    Args:
        dataset_obj: Dataset object.
        vi_model: Trained RemoveBackgroundPyroModel.
        guide: Variational posterior pyro guide function, optional.  Only
            specify if the required guide function is not vi_model.guide.
        encoder: Encoder that provides encodings of data.
        counts_dtype: Data type of posterior count matrix.  Can be one of
            [np.uint32, np.float]
        float_threshold: For floating point count matrices, counts below
            this threshold will be set to zero, for the purposes of constructing
            a sparse matrix.  Unused if counts_dtype is np.uint32

    Properties:
        mean: Posterior count mean, as a sparse matrix.
        encodings: Encoded latent variables, one per barcode in the dataset.

    """

    def __init__(self,
                 dataset_obj: 'SingleCellRNACountsDataset',  # Dataset
                 vi_model: 'RemoveBackgroundPyroModel',  # Trained variational inference model
                 guide=None,
                 encoder=None,  #: Union[CompositeEncoder, None] = None,
                 counts_dtype: np.dtype = np.uint32,
                 float_threshold: float = 0.5):
        self.vi_model = vi_model
        self.use_cuda = vi_model.use_cuda
        self.guide = guide if guide is not None else vi_model.encoder
        self.encoder = encoder if encoder is not None else vi_model.encoder
        self._encodings = None
        self._mean = None
        super(ImputedPosterior, self).__init__(dataset_obj=dataset_obj,
                                               vi_model=vi_model,
                                               counts_dtype=counts_dtype,
                                               float_threshold=float_threshold)

    @torch.no_grad()
    def _get_mean(self):
        """Send dataset through a guide that returns mean posterior counts.

        Keep track of only what is necessary to distill a sparse count matrix.

        """

        data_loader = self.dataset_obj.get_dataloader(use_cuda=self.use_cuda,
                                                      analyzed_bcs_only=False,
                                                      batch_size=500,
                                                      shuffle=False)

        barcodes = []
        genes = []
        counts = []
        ind = 0

        for data in data_loader:

            # Get return values from guide.
            dense_counts_torch = self._param_map_estimates(data=data,
                                                           chi_ambient=pyro.param("chi_ambient"))
            dense_counts = dense_counts_torch.detach().cpu().numpy()
            bcs_i_chunk, genes_i, counts_i = self.dense_to_sparse(dense_counts)

            # Translate chunk barcode inds to overall inds.
            bcs_i = self.barcode_inds[bcs_i_chunk + ind]

            # Add sparse matrix values to lists.
            barcodes.append(bcs_i)
            genes.append(genes_i)
            counts.append(counts_i)

            # Increment barcode index counter.
            ind += data.shape[0]  # Same as data_loader.batch_size

        # Convert the lists to numpy arrays.
        counts = np.array(np.concatenate(tuple(counts)), dtype=self.dtype)
        barcodes = np.array(np.concatenate(tuple(barcodes)), dtype=np.uint32)
        genes = np.array(np.concatenate(tuple(genes)), dtype=np.uint32)

        # Put the counts into a sparse csc_matrix.
        self._mean = sp.csc_matrix((counts, (barcodes, genes)),
                                   shape=self.count_matrix_shape)


class ProbPosterior(Posterior):
    """Posterior count inference using a noise count probability distribution.

    Args:
        dataset_obj: Dataset object.
        vi_model: Trained model: RemoveBackgroundPyroModel
        fpr: Desired false positive rate for construction of the final regularized
            posterior on true counts. False positives are true counts that are
            (incorrectly) removed from the dataset.
        float_threshold: For floating point count matrices, counts below
            this threshold will be set to zero, for the purposes of constructing
            a sparse matrix.  Unused if counts_dtype is np.uint32

    Properties:
        mean: Posterior count mean, as a sparse matrix.
        encodings: Encoded latent variables, one per barcode in the dataset.

    """

    def __init__(self,
                 dataset_obj: 'SingleCellRNACountsDataset',
                 vi_model: 'RemoveBackgroundPyroModel',
                 fpr: float = 0.01,
                 float_threshold: float = 0.5):
        self.vi_model = vi_model
        self.use_cuda = vi_model.use_cuda
        self.fpr = fpr
        self.lambda_multiplier = None
        self._encodings = None
        self._mean = None
        self.random = np.random.RandomState(seed=1234)
        super(ProbPosterior, self).__init__(dataset_obj=dataset_obj,
                                            vi_model=vi_model,
                                            counts_dtype=np.uint32,
                                            float_threshold=float_threshold)

    @torch.no_grad()
    def _get_mean(self):
        """Send dataset through a guide that returns mean posterior counts.

        Keep track of only what is necessary to distill a sparse count matrix.

        """

        # Get a dataset of ten cells.
        cell_inds = np.where(self.latents['p'] > 0.9)[0]
        lambda_mults = np.zeros(5)

        for i in range(lambda_mults.size):

            n_cells = min(consts.CELLS_POSTERIOR_REG_CALC, cell_inds.size)
            if n_cells == 0:
                raise ValueError('No cells found!  Cannot compute expected FPR.')
            cell_ind_subset = self.random.choice(cell_inds, size=n_cells, replace=False)
            cell_data = (torch.tensor(np.array(self.dataset_obj.get_count_matrix()
                                               [cell_ind_subset, :].todense()).squeeze())
                         .float().to(self.vi_model.device))

            # Get the latents mu, alpha, and lambda for those cells.
            chi_ambient = pyro.param("chi_ambient")
            map_est = self._param_map_estimates(data=cell_data, chi_ambient=chi_ambient)

            # Find the optimal lambda_multiplier value using those cells and target FPR.
            lambda_mult = self._lambda_binary_search_given_fpr(cell_data=cell_data,
                                                               fpr=self.fpr,
                                                               mu_est=map_est['mu'],
                                                               lambda_est=map_est['lam'],
                                                               alpha_est=map_est['alpha'])
            lambda_mults[i] = lambda_mult

        optimal_lambda_mult = np.mean(lambda_mults)
        self.lambda_multiplier = optimal_lambda_mult
        logging.info(f'Optimal posterior regularization factor = {optimal_lambda_mult:.2f}')

        # Compute posterior in mini-batches.
        analyzed_bcs_only = True
        data_loader = self.dataset_obj.get_dataloader(use_cuda=self.use_cuda,
                                                      analyzed_bcs_only=analyzed_bcs_only,
                                                      batch_size=20,
                                                      shuffle=False)
        barcodes = []
        genes = []
        counts = []
        ind = 0

        for data in data_loader:

            # Compute an estimate of the true counts.
            dense_counts = self._compute_true_counts(data=data,
                                                     chi_ambient=chi_ambient,
                                                     lambda_multiplier=optimal_lambda_mult,
                                                     use_map=False,
                                                     n_samples=9)  # must be odd number
            bcs_i_chunk, genes_i, counts_i = self.dense_to_sparse(dense_counts)

            # Translate chunk barcode inds to overall inds.
            if analyzed_bcs_only:
                bcs_i = self.dataset_obj.analyzed_barcode_inds[bcs_i_chunk + ind]
            else:
                bcs_i = self.barcode_inds[bcs_i_chunk + ind]

            # Add sparse matrix values to lists.
            barcodes.append(bcs_i)
            genes.append(genes_i)
            counts.append(counts_i)

            # Increment barcode index counter.
            ind += data.shape[0]  # Same as data_loader.batch_size

        # Convert the lists to numpy arrays.
        counts = np.array(np.concatenate(tuple(counts)), dtype=self.dtype)
        barcodes = np.array(np.concatenate(tuple(barcodes)), dtype=np.uint32)
        genes = np.array(np.concatenate(tuple(genes)), dtype=np.uint32)  # uint16 is too small!

        # Put the counts into a sparse csc_matrix.
        self._mean = sp.csc_matrix((counts, (barcodes, genes)),
                                   shape=self.count_matrix_shape)

    @torch.no_grad()
    def _compute_true_counts(self,
                             data: torch.Tensor,
                             chi_ambient: torch.Tensor,
                             lambda_multiplier: float,
                             use_map: bool = True,
                             n_samples: int = 1) -> np.ndarray:
        """Compute the true de-noised count matrix for this minibatch.

        Can use either a MAP estimate of lambda and mu, or can use a sampling
        approach.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            chi_ambient: Point estimate of inferred ambient gene expression.
            lambda_multiplier: Value by which lambda gets multiplied in order
                to compute regularized posterior true counts.
            use_map: True to use a MAP estimate of lambda and mu.
            n_samples: If not using a MAP estimate, this specifies the number
                of samples to use in calculating the posterior mean.

        Returns:
            dense_counts: Dense matrix of true de-noised counts.

        """

        if use_map:

            # Calculate MAP estimates of mu and lambda.
            est = self._param_map_estimates(data, chi_ambient)
            mu_map = est['mu']
            lambda_map = est['lam']
            alpha_map = est['alpha']

            # Compute the de-noised count matrix given the MAP estimates.
            dense_counts_torch = self._true_counts_from_params(data,
                                                               mu_map,
                                                               lambda_map * lambda_multiplier + 1e-30,
                                                               alpha_map + 1e-30)

            dense_counts = dense_counts_torch.detach().cpu().numpy()

        else:

            assert n_samples > 0, f"Posterior mean estimate needs to be derived " \
                                  f"from at least one sample: here {n_samples} " \
                                  f"samples are called for."

            dense_counts_torch = torch.zeros((data.shape[0],
                                              data.shape[1],
                                              n_samples),
                                             dtype=torch.float32).to(data.device)

            for i in range(n_samples):

                # Sample from mu and lambda.
                mu_sample, lambda_sample, alpha_sample = \
                    self._param_sample(data)

                # Compute the de-noised count matrix given the estimates.
                dense_counts_torch[..., i] = \
                    self._true_counts_from_params(data,
                                                  mu_sample,
                                                  lambda_sample * lambda_multiplier + 1e-30,
                                                  alpha_sample + 1e-30)

            # Take the median of the posterior true count distribution... torch cuda does not implement mode
            dense_counts = dense_counts_torch.median(dim=2, keepdim=False)[0]
            dense_counts = dense_counts.detach().cpu().numpy().astype(np.int32)

        return dense_counts

    @torch.no_grad()
    def _param_sample(self,
                      data: torch.Tensor) -> Tuple[torch.Tensor,
                                                   torch.Tensor,
                                                   torch.Tensor]:
        """Calculate a single sample estimate of mu, the mean of the true count
        matrix, and lambda, the rate parameter of the Poisson background counts.

        Args:
            data: Dense tensor minibatch of cell by gene count data.

        Returns:
            mu_sample: Dense tensor sample of Negative Binomial mean for true
                counts.
            lambda_sample: Dense tensor sample of Poisson rate params for noise
                counts.
            alpha_sample: Dense tensor sample of Dirichlet concentration params
                that inform the overdispersion of the Negative Binomial.

        """

        # Use pyro poutine to trace the guide and sample parameter values.
        guide_trace = pyro.poutine.trace(self.vi_model.guide).get_trace(x=data)
        replayed_model = pyro.poutine.replay(self.vi_model.model, guide_trace)

        # Run the model using these sampled values.
        replayed_model_output = replayed_model(x=data)

        # The model returns mu, alpha, and lambda.
        mu_sample = replayed_model_output['mu']
        lambda_sample = replayed_model_output['lam']
        alpha_sample = replayed_model_output['alpha']

        return mu_sample, lambda_sample, alpha_sample

    @torch.no_grad()
    def _true_counts_from_params(self,
                                 data: torch.Tensor,
                                 mu_est: torch.Tensor,
                                 lambda_est: torch.Tensor,
                                 alpha_est: torch.Tensor) -> torch.Tensor:
        """Calculate a single sample estimate of mu, the mean of the true count
        matrix, and lambda, the rate parameter of the Poisson background counts.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            mu_est: Dense tensor of Negative Binomial means for true counts.
            lambda_est: Dense tensor of Poisson rate params for noise counts.
            alpha_est: Dense tensor of Dirichlet concentration params that
                inform the overdispersion of the Negative Binomial.

        Returns:
            dense_counts_torch: Dense matrix of true de-noised counts.

        """

        # Estimate a reasonable low-end to begin the Poisson summation.
        n = min(100., data.max().item())  # No need to exceed the max value
        poisson_values_low = (lambda_est.detach() - n / 2).int()

        poisson_values_low = torch.clamp(torch.min(poisson_values_low,
                                                   (data - n + 1).int()), min=0).float()

        # Construct a big tensor of possible noise counts per cell per gene,
        # shape (batch_cells, n_genes, max_noise_counts)
        noise_count_tensor = torch.arange(start=0, end=n) \
            .expand([data.shape[0], data.shape[1], -1]) \
            .float().to(device=data.device)
        noise_count_tensor = noise_count_tensor + poisson_values_low.unsqueeze(-1)

        # Compute probabilities of each number of noise counts.
        # NOTE: some values will be outside the support (negative values for NB).
        # The resulting NaNs are ignored by torch.argmax().
        logits = (mu_est.log() - alpha_est.log()).unsqueeze(-1)
        log_prob_tensor = (dist.Poisson(lambda_est.unsqueeze(-1))
                           .log_prob(noise_count_tensor)
                           + dist.NegativeBinomial(total_count=alpha_est.unsqueeze(-1),
                                                   logits=logits)
                           .log_prob(data.unsqueeze(-1) - noise_count_tensor))
        log_prob_tensor = torch.where(noise_count_tensor <= data.unsqueeze(-1),
                                      log_prob_tensor,
                                      torch.ones_like(log_prob_tensor) * -np.inf)

        # Find the most probable number of noise counts per cell per gene.
        noise_count_map = torch.argmax(log_prob_tensor,
                                       dim=-1,
                                       keepdim=False).float()

        # Handle the cases where y = 0 (no cell): all counts are noise.
        noise_count_map = torch.where(mu_est == 0,
                                      data,
                                      noise_count_map)

        # Compute the number of true counts.
        dense_counts_torch = torch.clamp(data - noise_count_map, min=0.)

        return dense_counts_torch

    @torch.no_grad()
    def _calculate_expected_fpr_from_map(self,
                                         data: torch.Tensor,
                                         data_map: torch.Tensor) -> torch.Tensor:
        """Given inferred latent variables and observed total counts,
        generate a MAP estimate for noise counts.  Use that MAP estimate to
        compute the expected false positive rate.

        Args:
            data: Dense tensor tiny minibatch of cell by gene count data.
            data_map: Dense tensor tiny minibatch of MAP output for that data.

        Returns:
            fpr: Expected false positive rate.

        """

        counts_per_cell = data.sum(dim=-1)
        map_counts_per_cell = data_map.sum(dim=-1)
        fraction_counts_removed_per_cell = (counts_per_cell - map_counts_per_cell) / counts_per_cell
        empty_droplet_mean_counts = dist.LogNormal(loc=pyro.param('d_empty_loc'),
                                                   scale=pyro.param('d_empty_scale')).mean
        ambient_fraction = empty_droplet_mean_counts / counts_per_cell
        if self.vi_model.include_rho:
            swapping_fraction = dist.Beta(pyro.param('rho_alpha'), pyro.param('rho_beta')).mean
        else:
            swapping_fraction = 0.

        mean_cell_epsilon = (self.latents['epsilon'][self.latents['p'] > 0.5]).mean()
        target_fraction_counts_removed_per_cell = (ambient_fraction
                                                   + swapping_fraction) / mean_cell_epsilon
        fpr_expectation = (fraction_counts_removed_per_cell - target_fraction_counts_removed_per_cell).mean()

        fpr_expectation = torch.clamp(fpr_expectation, min=0.)

        return fpr_expectation.mean()

    @torch.no_grad()
    def _calculate_expected_fpr_given_lambda_mult(self,
                                                  data: torch.Tensor,
                                                  lambda_mult: float,
                                                  mu_est: torch.Tensor,
                                                  alpha_est: torch.Tensor,
                                                  lambda_est: torch.Tensor) -> torch.Tensor:
        """Given a float lambda_mult, calculate a MAP estimate of output counts,
        and use that estimate to calculate an expected false positive rate.

        Args:
            data: Dense tensor tiny minibatch of cell by gene count data.
            lambda_mult: Value of the lambda multiplier.
            mu_est: Dense tensor of Negative Binomial means for true counts.
            lambda_est: Dense tensor of Poisson rate params for noise counts.
            alpha_est: Dense tensor of Dirichlet concentration params that
                inform the overdispersion of the Negative Binomial.

        Returns:
            fpr: Expected false positive rate.

        """

        # Compute MAP estimate of true de-noised counts.
        data_map = self._true_counts_from_params(data=data,
                                                 mu_est=mu_est,
                                                 lambda_est=lambda_est * lambda_mult,
                                                 alpha_est=alpha_est)

        # Compute expected false positive rate.
        expected_fpr = self._calculate_expected_fpr_from_map(data=data,
                                                             data_map=data_map)

        return expected_fpr

    @torch.no_grad()
    def _lambda_binary_search_given_fpr(self,
                                        cell_data: torch.Tensor,
                                        fpr: float,
                                        mu_est: torch.Tensor,
                                        lambda_est: torch.Tensor,
                                        alpha_est: torch.Tensor,
                                        lam_mult_init: float = 1.,
                                        fpr_tolerance: Optional[float] = None,
                                        max_iterations: int = consts.POSTERIOR_REG_SEARCH_MAX_ITER) -> float:
        """Perform a binary search for the appropriate lambda-multiplier which will
        achieve a desired false positive rate.

        NOTE: It is assumed that
        expected_fpr(lam_mult_bracket[0]) < fpr < expected_fpr(lam_mult_bracket[1]).
        If this is not the case, the algorithm will produce an output close to one
        of the brackets, and FPR control will not be achieved.

        Args:
            cell_data: Data from a fraction of the total number of cells.
            fpr: Desired false positive rate.
            mu_est: Dense tensor of Negative Binomial means for true counts.
            lambda_est: Dense tensor of Poisson rate params for noise counts.
            alpha_est: Dense tensor of Dirichlet concentration params that
                inform the overdispersion of the Negative Binomial.
            lam_mult_init: Initial value of the lambda multiplier, hopefully
                close to the unknown target value.
            fpr_tolerance: Tolerated error in expected false positive rate.  If
                input is None, defaults to 0.001 or fpr / 10, whichever is
                smaller.
            max_iterations: A cutoff to ensure termination. Even if a tolerable
                solution is not found, the algorithm will stop after this many
                iterations and return the best answer so far.

        Returns:
            lam_mult: Value of the lambda-multiplier.

        """

        assert (fpr > 0) and (fpr < 1), "Target FPR should be in the interval (0, 1)."
        if fpr_tolerance is None:
            fpr_tolerance = min(fpr / 10., 0.001)

        # Begin at initial value.
        lam_mult = lam_mult_init

        # Calculate an expected false positive rate for this lam_mult value.
        expected_fpr = self._calculate_expected_fpr_given_lambda_mult(
            data=cell_data,
            lambda_mult=lam_mult,
            mu_est=mu_est,
            lambda_est=lambda_est,
            alpha_est=alpha_est)

        # Find out what direction we need to move in.
        residual = fpr - expected_fpr
        initial_residual_sign = residual.sign()  # plus or minus one
        if initial_residual_sign.item() == 0:
            # In the unlikely event that we hit the value exactly.
            return lam_mult

        # Travel in one direction until the direction of FPR error changes.
        lam_limit = lam_mult_init
        i = 0
        while ((lam_limit < consts.POSTERIOR_REG_MAX)
               and (lam_limit > consts.POSTERIOR_REG_MIN)
               and (residual.sign() == initial_residual_sign)
               and (i < max_iterations)):

            lam_limit = lam_limit * (initial_residual_sign * 2).exp().item()

            # Calculate an expected false positive rate for this lam_mult value.
            expected_fpr = self._calculate_expected_fpr_given_lambda_mult(
                data=cell_data,
                lambda_mult=lam_limit,
                mu_est=mu_est,
                lambda_est=lambda_est,
                alpha_est=alpha_est)

            residual = fpr - expected_fpr
            i = i + 1  # one dataset had this go into an infinite loop, taking lam_limit -> 0

        # Define the values that bracket the correct answer.
        lam_mult_bracket = np.sort(np.array([lam_mult_init, lam_limit]))

        # Binary search algorithm.
        for i in range(max_iterations):

            # Current test value for the lambda-multiplier.
            lam_mult = np.mean(lam_mult_bracket)

            # Calculate an expected false positive rate for this lam_mult value.
            expected_fpr = self._calculate_expected_fpr_given_lambda_mult(
                data=cell_data,
                lambda_mult=lam_mult,
                mu_est=mu_est,
                lambda_est=lambda_est,
                alpha_est=alpha_est)

            # Check on false positive rate and update our bracket values.
            if (expected_fpr < (fpr + fpr_tolerance)) and (expected_fpr > (fpr - fpr_tolerance)):
                break
            elif expected_fpr > fpr:
                lam_mult_bracket[1] = lam_mult
            elif expected_fpr < fpr:
                lam_mult_bracket[0] = lam_mult

        # If we stopped due to iteration limit, take the average lam_mult value.
        if i == max_iterations:
            lam_mult = np.mean(lam_mult_bracket)

        # Check to see if we have achieved the desired FPR control.
        if not ((expected_fpr < (fpr + fpr_tolerance)) and (expected_fpr > (fpr - fpr_tolerance))):
            print(f'FPR control not achieved in {max_iterations} attempts. '
                  f'Output FPR is esitmated to be {expected_fpr.item():.4f}')

        return lam_mult
