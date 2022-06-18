# Posterior inference.

import pyro
import pyro.distributions as dist
import torch
import numpy as np
import scipy.sparse as sp

import cellbender.remove_background.consts as consts
from cellbender.remove_background.model import calculate_mu, calculate_lambda
from cellbender.monitor import get_hardware_usage
from cellbender.remove_background.data.dataprep import DataLoader

from typing import Tuple, List, Dict, Optional, Union, Callable
from abc import ABC, abstractmethod
import logging


logger = logging.getLogger('cellbender')


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
        latents: Posterior latents

    """

    def __init__(self,
                 dataset_obj: 'SingleCellRNACountsDataset',  # Dataset
                 vi_model: Optional['RemoveBackgroundPyroModel'],
                 counts_dtype: np.dtype = np.uint32,
                 float_threshold: Optional[float] = 0.5):
        self.dataset_obj = dataset_obj
        self.vi_model = vi_model
        self.use_cuda = torch.cuda.is_available() if vi_model is None else vi_model.use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.analyzed_gene_inds = None if (dataset_obj is None) else dataset_obj.analyzed_gene_inds
        self.count_matrix_shape = None if (dataset_obj is None) else dataset_obj.data['matrix'].shape
        self.barcode_inds = None if (dataset_obj is None) else np.arange(0, self.count_matrix_shape[0])
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
    def latents(self) -> Dict[str, np.ndarray]:
        if self._latents is None:
            self._get_latents()
        return self._latents

    @property
    def variance(self):
        raise NotImplemented("Posterior count variance not implemented.")

    @torch.no_grad()
    def _get_latents(self):
        """Calculate the encoded latent variables."""

        logger.debug('Computing latent variables')

        if self.vi_model is None:
            self._latents = {'z': None, 'd': None, 'p': None, 'phi_loc_scale': None, 'epsilon': None}
            return None

        data_loader = self.dataset_obj.get_dataloader(use_cuda=self.use_cuda,
                                                      analyzed_bcs_only=True,
                                                      batch_size=500,
                                                      shuffle=False)

        z = np.zeros((len(data_loader), self.vi_model.encoder['z'].output_dim))
        d = np.zeros(len(data_loader))
        p = np.zeros(len(data_loader))
        epsilon = np.zeros(len(data_loader))

        phi_loc = pyro.param('phi_loc')
        phi_scale = pyro.param('phi_scale')
        if 'chi_ambient' in pyro.get_param_store().keys():
            chi_ambient = pyro.param('chi_ambient').detach()
        else:
            chi_ambient = None

        for i, data in enumerate(data_loader):

            enc = self.vi_model.encoder(x=data,
                                        chi_ambient=chi_ambient,
                                        cell_prior_log=self.vi_model.d_cell_loc_prior)
            ind = i * data_loader.batch_size
            z[ind:(ind + data.shape[0]), :] = enc['z']['loc'].detach().cpu().numpy()

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

        logger.debug('Computing MAP esitmate of mu, lambda, alpha')

        # Encode latents.
        enc = self.vi_model.encoder(x=data,
                                    chi_ambient=chi_ambient,
                                    cell_prior_log=self.vi_model.d_cell_loc_prior)
        z_map = enc['z']['loc']

        chi_map = self.vi_model.decoder(z_map)
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
        mu_map = self.vi_model._calculate_mu(  # TODO: use the non-private method?
            epsilon=epsilon,
            d_cell=d_cell,
            chi=chi_map,
            y=y,
            rho=rho,
        )
        lambda_map = self.vi_model._calculate_lambda(
            epsilon=epsilon,
            chi_ambient=chi_ambient,
            d_empty=d_empty,
            y=y,
            d_cell=d_cell,
            rho=rho,
            chi_bar=self.vi_model.avg_gene_expression,
        )

        return {'mu': mu_map, 'lam': lambda_map, 'alpha': alpha_map}

    def dense_to_sparse(self, chunk_dense_counts: torch.Tensor) \
            -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        """Distill a batch of dense counts into sparse format.
        Barcode numbering is relative to the tensor passed in.
        """

        # TODO: speed up by keeping it a torch tensor as long as possible

        if self.name != 'prob':  # ProbPosterior produces ints already

            if self.dtype == np.uint32:

                # Turn the floating point count estimates into integers.
                # TODO: convert this to torch... but it is not used by ProbPosterior
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

        nonzero_barcode_inds_this_chunk, nonzero_genes_trimmed, nonzero_counts = \
            self.dense_to_sparse_op_torch(chunk_dense_counts)

        # Get the original gene index from gene index in the trimmed dataset.
        nonzero_genes = self.analyzed_gene_inds[nonzero_genes_trimmed.cpu()]

        return nonzero_barcode_inds_this_chunk, nonzero_genes, nonzero_counts

    @staticmethod
    def dense_to_sparse_op_numpy(chunk_dense_counts: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """This isn't used directly, but it's used by tests since it's tried and true"""
        # Find all the nonzero counts in this dense matrix chunk.
        nonzero_barcode_inds_this_chunk, nonzero_genes_trimmed = np.nonzero(chunk_dense_counts)
        nonzero_counts = chunk_dense_counts[nonzero_barcode_inds_this_chunk,
                                            nonzero_genes_trimmed].flatten(order='C')
        return nonzero_barcode_inds_this_chunk, nonzero_genes_trimmed, nonzero_counts

    @staticmethod
    @torch.no_grad()
    def dense_to_sparse_op_torch(chunk_dense_counts: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Find all the nonzero counts in this dense matrix chunk.
        nonzero_barcode_inds_this_chunk, nonzero_genes_trimmed = \
            torch.nonzero(chunk_dense_counts, as_tuple=True)
        nonzero_counts = chunk_dense_counts[nonzero_barcode_inds_this_chunk,
                                            nonzero_genes_trimmed].flatten()
        return nonzero_barcode_inds_this_chunk, nonzero_genes_trimmed, nonzero_counts


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
        self.name = 'imputed'
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
        counts = torch.cat(counts, dim=0).detach().cpu().numpy().astype(np.uint32)
        barcodes = np.array(np.concatenate(tuple(barcodes)), dtype=np.uint32)
        genes = np.array(np.concatenate(tuple(genes)), dtype=np.uint32)  # uint16 is too small!

        # Put the counts into a sparse csc_matrix.
        self._mean = sp.csc_matrix((counts, (barcodes, genes)),
                                   shape=self.count_matrix_shape)


class NaivePosterior(Posterior):
    """Posterior count inference using naive ambient subtraction.

    Args:
        dataset_obj: Dataset object.

    Properties:
        mean: Posterior count mean, as a sparse matrix.

    """

    def __init__(self, dataset_obj: 'SingleCellRNACountsDataset'):
        self._mean = None
        self.random = np.random
        self.lambda_multiplier = 1.
        self.name = 'naive'
        super(NaivePosterior, self).__init__(dataset_obj=dataset_obj,
                                             vi_model=None,
                                             counts_dtype=np.uint32)

    @torch.no_grad()
    def _get_mean(self):
        """Perform naive ambient subtraction.

        Keep track of only what is necessary to distill a sparse count matrix.

        """

        # Use the prior for ambient expression, in the absence of inference.
        chi_ambient = self.dataset_obj.priors['chi_ambient']
        ambient_counts = chi_ambient * self.dataset_obj.priors['empty_counts']
        if self.use_cuda:
            ambient_counts = ambient_counts.to(device='cuda') * self.lambda_multiplier

        # Compute posterior in mini-batches.
        analyzed_bcs_only = True
        data_loader = self.dataset_obj.get_dataloader(use_cuda=self.use_cuda,
                                                      analyzed_bcs_only=analyzed_bcs_only,
                                                      batch_size=500,
                                                      shuffle=False)
        barcodes = []
        genes = []
        counts = []
        ind = 0

        for data in data_loader:

            # Compute an estimate of the true counts.
            dense_counts = self._compute_true_counts(data=data,
                                                     ambient_counts=ambient_counts)
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
        counts = torch.cat(counts, dim=0).detach().cpu().numpy().astype(np.uint32)
        barcodes = np.array(np.concatenate(tuple(barcodes)), dtype=np.uint32)
        genes = np.array(np.concatenate(tuple(genes)), dtype=np.uint32)  # uint16 is too small!

        # Put the counts into a sparse csc_matrix.
        self._mean = sp.csc_matrix((counts, (barcodes, genes)),
                                   shape=self.count_matrix_shape)

    @torch.no_grad()
    def _compute_true_counts(self,
                             data: torch.Tensor,
                             ambient_counts: torch.Tensor) -> torch.Tensor:
        """Compute the true de-noised count matrix for this minibatch.

        Naive subtraction of an estimate of ambient counts from empty droplets.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            ambient_counts: Point estimate of inferred ambient gene expression
                times empty droplet size.

        Returns:
            dense_counts: Dense matrix of true de-noised counts.

        """

        # Subtract ambient.
        dense_counts = torch.clamp(data - ambient_counts.unsqueeze(0), min=0)

        return dense_counts


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
                 fpr: Union[str, float] = 0.01,
                 float_threshold: float = 0.5,
                 posterior_batch_size: int = consts.PROB_POSTERIOR_BATCH_SIZE,
                 debug: bool = False):
        self.vi_model = vi_model
        self.fpr = fpr
        self.lambda_multiplier = None
        self.noise_target_G = None
        self._encodings = None
        self._mean = None
        self.posterior_batch_size = posterior_batch_size
        self.random = np.random
        self.debug = debug
        self.name = 'prob'
        super(ProbPosterior, self).__init__(dataset_obj=dataset_obj,
                                            vi_model=vi_model,
                                            counts_dtype=np.uint32,
                                            float_threshold=float_threshold)

    @torch.no_grad()
    def sample(self, data, lambda_multiplier=1., y_map: bool = False) -> torch.Tensor:
        """Draw a single posterior sample for the count matrix conditioned on data

        Args:
            data: Count matrix (slice: some droplets, all genes)
            lambda_multiplier: Posterior regularization multiplier
            y_map: True to enforce the use of the MAP estimate of y, cell or
                no cell. Useful in the case where many samples are collected,
                since typically in those cases it is confusing to have samples
                where a droplet is both cell-containing and empty.

        Returns:
            denoised_output_count_matrix: Single sample of the denoised output
                count matrix, sampling all stochastic latent variables in the model.

        """

        # Sample all the latent variables in the model and get mu, lambda, alpha.
        mu_sample, lambda_sample, alpha_sample = self._param_sample(data, y_map=y_map)

        # Compute the big tensor of log probabilities of possible c_{ng}^{noise} values.
        log_prob_noise_counts_NGC, poisson_values_low_NG = self._log_prob_noise_count_tensor(
            data=data,
            mu_est=mu_sample + 1e-30,
            lambda_est=lambda_sample * lambda_multiplier + 1e-30,
            alpha_est=alpha_sample + 1e-30,
            debug=self.debug,
        )

        # Use those probabilities to draw a sample of c_{ng}^{noise}
        noise_count_increment_NG = dist.Categorical(logits=log_prob_noise_counts_NGC).sample()
        noise_counts_NG = noise_count_increment_NG + poisson_values_low_NG

        # Subtract from data to get the denoised output counts.
        denoised_output_count_matrix = data - noise_counts_NG

        return denoised_output_count_matrix

    @torch.no_grad()
    def map_from_sampled_latents(self,
                                 data,
                                 n_samples: int,
                                 lambda_multiplier: float = 1.,
                                 y_map: bool = False) -> torch.Tensor:
        """Draw posterior samples for all stochastic latent variables in the model
         and use those values to compute a MAP estimate of the denoised count
         matrix conditioned on data.

        Args:
            data: Count matrix (slice: some droplets, all genes)
            lambda_multiplier: Posterior regularization multiplier
            y_map: True to enforce the use of the MAP estimate of y, cell or
                no cell. Useful in the case where many samples are collected,
                since typically in those cases it is confusing to have samples
                where a droplet is both cell-containing and empty.

        Returns:
            denoised_output_count_matrix: MAP estimate of the denoised output
                count matrix, sampling all stochastic latent variables in the model.

        """

        noise_log_pdf, offset_noise_counts = self.noise_log_pdf(
            data=data,
            n_samples=n_samples,
            lambda_multiplier=lambda_multiplier,
            y_map=y_map,
        )

        noise_counts = torch.argmax(noise_log_pdf, dim=-1) + offset_noise_counts
        denoised_output_count_matrix = torch.clamp(data - noise_counts, min=0.)

        return denoised_output_count_matrix

    @torch.no_grad()
    def noise_log_pdf(self,
                      data,
                      n_samples: int = 1,
                      lambda_multiplier=1.,
                      y_map: bool = False,
                      n_counts_max: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior noise-count probability density function
        using n_samples samples. This is a big matrix [n, g, c] where the last
        dimension c is of variable size depending on the computation in
        _log_prob_noise_count_tensor(), but is limited there to be no more than
        100.  The c dimension represents an index to the number of noise counts
        in [n, g]: specifically, the noise count once poisson_values_low_NG is added

        Args:
            data: Count matrix (slice: some droplets, all genes)
            n_samples: Number of samples (of all stochastic latent variables in
                the model) used to generate the CDF
            lambda_multiplier: Posterior regularization multiplier
            y_map: True to enforce the use of the MAP estimate of y, cell or
                no cell. Useful in the case where many samples are collected,
                since typically in those cases it is confusing to have samples
                where a droplet is both cell-containing and empty.
            n_counts_max: Size of count axis (need not start at zero noise
                counts, but should be enough to cover the meat of the posterior)

        Returns:
            noise_log_pdf_NGC: Consensus noise count log_pdf (big tensor) from the samples.
            noise_count_offset_NG: The offset for the noise count axis [n, g].

        """

        noise_log_pdf_NGC = None
        noise_count_offset_NG = None

        for s in range(1, n_samples + 1):

            # Sample all the latent variables in the model and get mu, lambda, alpha.
            mu_sample, lambda_sample, alpha_sample = self._param_sample(data, y_map=y_map)

            # Compute the big tensor of log probabilities of possible c_{ng}^{noise} values.
            log_prob_noise_counts_NGC, noise_count_offset_NG = self._log_prob_noise_count_tensor(
                data=data,
                mu_est=mu_sample + 1e-30,
                lambda_est=lambda_sample * lambda_multiplier + 1e-30,
                alpha_est=alpha_sample + 1e-30,
                n_counts_max=n_counts_max,
                debug=self.debug,
            )

            # Normalize the PDFs (not necessarily normalized over the count range).
            log_prob_noise_counts_NGC = (log_prob_noise_counts_NGC
                                         - torch.logsumexp(log_prob_noise_counts_NGC,
                                                           dim=-1, keepdim=True))

            # Add the probability from this sample to our running total.
            # Update rule is
            # log_prob_total_n = LAE [ log(1 - 1/n) + log_prob_total_{n-1}, log(1/n) + log_prob_sample ]
            if s == 1:
                noise_log_pdf_NGC = log_prob_noise_counts_NGC
            else:
                # This is a (normalized) running sum over samples in log-probability space.
                noise_log_pdf_NGC = torch.logaddexp(
                    noise_log_pdf_NGC + torch.log(torch.tensor(1. - 1. / s).to(device=data.device)),
                    log_prob_noise_counts_NGC + torch.log(torch.tensor(1. / s).to(device=data.device)),
                )

        return noise_log_pdf_NGC, noise_count_offset_NG

    @torch.no_grad()
    def counts_from_cdf_threshold(self,
                                  data: torch.Tensor,
                                  q: Union[float, torch.Tensor],
                                  use_extra_count_sampling: bool = False,
                                  n_samples: int = 5,
                                  lambda_multiplier: float = 1.,
                                  y_map: bool = True):
        """Compute the posterior denoised counts given data, using a threshold, q,
        on the computed posterior CDF.  q is the amount of the noise CDF that
        is covered by the output noise count matrix.  That is, q = 0 results in
        a tensor of zeros, and q = 1 represents the maximum noise counts the
        model can tolerate.

        Args:
            data: Count matrix (slice: some droplets, all genes)
            q: Threshold in CDF, in [0, 1] (can be one value per gene)
            use_extra_count_sampling: False is the "normal" operation here. Counts
                are only deemed noise if the noise count CDF is below the q
                threshold.
                True takes into account how far away we are from the threshold
                and stochastically samples from a Bernoulli so that sometimes
                we remove an extra count.
            n_samples: Number of samples (of all stochastic latent variables in
                the model) used to generate the CDF
            lambda_multiplier: Posterior regularization multiplier
            y_map: True to enforce the use of the MAP estimate of y, cell or
                no cell. Useful in the case where many samples are collected,
                since typically in those cases it is confusing to have samples
                where a droplet is both cell-containing and empty.

        Returns:
            denoised_counts_NG: Consensus denoised counts from the samples, using
                the posterior noise count CDF.

        """

        # Ensure q is the right shape for broadcasting to [n, g, c].
        if type(q) == float:
            q = torch.tensor(q).to(device=data.device)
        else:
            if q.size() == torch.Size([1]):
                q = q.reshape([1, 1, 1])
            else:
                try:
                    q = q.reshape([1, data.shape[1], 1])
                except RuntimeError as e:
                    print(f'The q passed to counts_from_cdf_threshold() had the wrong '
                          f'shape {q.shape} for the data {data.shape}')
                    raise e

        # Obtain posterior PDF over a range of possible noise counts, for each [n, g].
        log_pdf_NGC, offset_NG = self.noise_log_pdf(
            data=data,
            n_samples=n_samples,
            lambda_multiplier=lambda_multiplier,
            y_map=y_map,
        )

        # Find the maximum noise counts that satisfy the threshold q.
        log_cdf_NGC = torch.logcumsumexp(log_pdf_NGC, dim=-1)
        noise_counts_NG = (log_cdf_NGC <= torch.log(q)).sum(dim=-1) + offset_NG

        # Optionally flip a coin to remove an extra count based on distrance from threshold.
        if use_extra_count_sampling:

            raise NotImplementedError('I do not think this actually makes sense to do')

            # CDF in regular probability space
            cdf_NGC = log_cdf_NGC.exp()

            # Subtract off the threshold (negative things are ignored here on).
            # Rescale the CDF (if you don't do something like this, then every single
            # entry in the matrix that had noise probability zero will now have
            # probability q of removing an extra count).
            remove_and_rescale_cdf_NGC = cdf_NGC - q
            remove_and_rescale_cdf_NGC = (remove_and_rescale_cdf_NGC
                                          / remove_and_rescale_cdf_NGC[:, :, -1].unsqueeze(-1))

            # Use the noise count as an index into the rescaled CDF.
            bernoulli_probs_NG = 1. - torch.gather(remove_and_rescale_cdf_NGC, 2,
                                                   noise_counts_NG.long().unsqueeze(-1)).squeeze()
            noise_counts_NG = noise_counts_NG + dist.Bernoulli(probs=bernoulli_probs_NG).sample()

        return data - noise_counts_NG

    @torch.no_grad()
    def log_pdf_cell_counts_noise_counts(self,
                                         data,
                                         n_samples: int = 1,
                                         count_range: int = 50,
                                         y_map: bool = True) -> Dict[str, torch.Tensor]:
        """Compute PDFs for the priors (still conditioned on data via the guide)
        on true counts and noise counts based on n_samples samples of all
        stochastic latent variables in the model.

        NOTE: The PDF returned for cell counts is properly normalized, but it
        will not necessarily sum to 1 over its range (only if the range contains
        all possible count values).

        Args:
            data: Count matrix (slice: some droplets, all genes)
            n_samples: Number of samples used to compute PDF (used to numerically
                marginalize over the other latents).
            count_range: Range of x-axis (count) values for the PDF. Memory-hungry.
            y_map: True to enforce the use of the MAP estimate of y, cell or
                no cell. Useful in the case where many samples are collected,
                since typically in those cases it is confusing to have samples
                where a droplet is both cell-containing and empty.

        Returns:
            Dict containing:
                log_pdf_cell_counts_NGC: log PDF of cell counts, NegBinom(mu_NG, phi)
                log_pdf_cell_counts_offset_NG: Starting point for count axis
                log_pdf_noise_counts_NGC: PDF of noise counts, Poisson(lambda)
                log_pdf_noise_counts_offset_NG: Starting point for count axis

        """

        # Size of x-axis (counts).
        n = min(count_range, data.max().item())  # No need to exceed the max value

        # Draw an initial sample.
        mu_sample, lambda_sample, alpha_sample = self._param_sample(data, y_map=y_map)

        # Estimate a reasonable low-end to begin the Poisson pdf.
        log_pdf_noise_counts_offset_NG = (lambda_sample.detach() - n / 2).int()
        log_pdf_noise_counts_offset_NG = torch.clamp(torch.min(log_pdf_noise_counts_offset_NG,
                                                               (data - n + 1).int()), min=0).float()

        # Construct a big tensor of possible noise counts per cell per gene,
        # shape (batch_cells, n_genes, max_noise_counts)
        noise_count_tensor_NGC = torch.arange(start=0, end=n) \
            .expand([data.shape[0], data.shape[1], -1]) \
            .float().to(device=data.device)
        noise_count_tensor_NGC = noise_count_tensor_NGC + log_pdf_noise_counts_offset_NG.unsqueeze(-1)

        # Construct a big tensor of possible cell counts per cell per gene,
        # shape (batch_cells, n_genes, max_noise_counts)
        cell_count_tensor_NGC = (torch.arange(start=-n + 1, end=1)
                                 .unsqueeze(0).unsqueeze(0)
                                 .float().to(device=data.device)
                                 + data.unsqueeze(-1))
        log_pdf_cell_counts_offset_NG = cell_count_tensor_NGC[:, :, 0].squeeze()

        log_pdf_cell_counts_NGC = None
        log_pdf_noise_counts_NGC = None

        for s in range(n_samples):

            if s > 0:
                # Sample all the latent variables in the model and get mu, lambda, alpha.
                mu_sample, lambda_sample, alpha_sample = self._param_sample(data, y_map=y_map)

            # Compute log_pdf of cell counts.
            logits = (mu_sample.log() - alpha_sample.log()).unsqueeze(-1)
            nb_log_prob_NGC = (dist.NegativeBinomial(total_count=alpha_sample.unsqueeze(-1),
                                                     logits=logits,
                                                     validate_args=False)
                               .log_prob(cell_count_tensor_NGC))
            if log_pdf_cell_counts_NGC is None:
                log_pdf_cell_counts_NGC = nb_log_prob_NGC
            else:
                # Running total, summing probability.
                log_pdf_cell_counts_NGC = torch.logsumexp(
                    torch.cat([log_pdf_cell_counts_NGC.unsqueeze(-1),
                               nb_log_prob_NGC.unsqueeze(-1)], dim=-1),
                    dim=-1,
                )

            # Compute log_pdf of noise counts.
            poisson_log_prob_NGC = (dist.Poisson(lambda_sample.unsqueeze(-1),
                                                 validate_args=False)
                                    .log_prob(noise_count_tensor_NGC))
            if log_pdf_noise_counts_NGC is None:
                log_pdf_noise_counts_NGC = poisson_log_prob_NGC
            else:
                # Running total, summing probability.
                log_pdf_noise_counts_NGC = torch.logsumexp(
                    torch.cat([log_pdf_noise_counts_NGC.unsqueeze(-1),
                               poisson_log_prob_NGC.unsqueeze(-1)], dim=-1),
                    dim=-1,
                )

        # Normalize probabilities.
        log_pdf_cell_counts_NGC = log_pdf_cell_counts_NGC - np.log(n_samples)
        log_pdf_noise_counts_NGC = log_pdf_noise_counts_NGC - np.log(n_samples)

        return {'log_pdf_cell_counts_NGC': log_pdf_cell_counts_NGC,
                'log_pdf_cell_counts_offset_NG': log_pdf_cell_counts_offset_NG,
                'log_pdf_noise_counts_NGC': log_pdf_noise_counts_NGC,
                'log_pdf_noise_counts_offset_NG': log_pdf_noise_counts_offset_NG}

    @torch.no_grad()
    def mckp_denoised_counts(self,
                             data: torch.Tensor,
                             log_prob_NGC: Optional[torch.Tensor] = None,
                             noise_count_offset_NG: Optional[torch.Tensor] = None,
                             carryover_G: Optional[torch.Tensor] = None,
                             n_samples: int = 50,
                             n_counts_max: int = 50,
                             how: str = 'fpr',
                             target: Optional[float] = None):
        """Compute the posterior denoised counts given data by solving a
        multiple-choice knapsack problem to apportion noise counts over cells.

         .....using a threshold, q,
        on the computed posterior CDF.  q is the amount of the noise CDF that
        is covered by the output noise count matrix.  That is, q = 0 results in
        a tensor of zeros, and q = 1 represents the maximum noise counts the
        model can tolerate.

        Args:
            data: Count matrix (slice: some droplets, all genes) [all cells]
            log_prob_NGC: To overwrite the default here, not needed.
            noise_count_offset_NG: To overwrite the default here, not needed.
            carryover_G: Target gene removal counts that were "left over" from
                a previous minibatch
            target: Threshold in CDF or in FPR, depending on "how"
            how: ['fpr', 'cdf'] - how to construct the noise target per gene.
            n_samples: Number of samples (of all stochastic latent variables in
                the model) used to generate the PDF
            n_counts_max: Size of count axis

        Returns:
            denoised_counts_NG: Denoised counts.
            carryover_G: Any carryover of "unused" gene budget for the next batch.

        """

        # Find the noise budget for each gene per cell.
        if self.noise_target_G is None:
            noise_target_fcn_per_cell_G = self._get_noise_budget_per_gene(how=how, bernoulli=True)
            self.noise_target_G = noise_target_fcn_per_cell_G
        target_G = self.noise_target_G(target) * data.shape[0]  # for all cells in batch

        if carryover_G is not None:
            target_G = target_G + carryover_G

        # print(target_G)

        # Round noise budget to integers by sampling.
        bernoulli_prob_G = target_G - target_G.floor()
        bernoulli_draw_G = dist.Bernoulli(bernoulli_prob_G).sample()
        carryover_G = bernoulli_prob_G - bernoulli_draw_G
        target_G = torch.clamp(target_G.floor() + bernoulli_draw_G, min=0.)

        # Compute log_prob noise count tensor via sampling.
        if log_prob_NGC is None:
            log_prob_noise_counts_NGC, noise_count_offset_NG = self.noise_log_pdf(
                data=data,
                n_samples=n_samples,
                y_map=True,
                n_counts_max=n_counts_max,
            )
        else:
            # This is for working with code where the optimum is known and computed elsewhere.
            log_prob_noise_counts_NGC = log_prob_NGC
            noise_count_offset_NG = noise_count_offset_NG

        # Compute noise counts by solving the multiple choice knapsack problem.
        noise_counts_NG, unmet_budget_G = self._mckp_noise_given_log_prob_tensor_fast(
            log_prob_noise_counts_NGC=log_prob_noise_counts_NGC,
            offset_noise_counts_NG=noise_count_offset_NG,
            data_NG=data,
            target_G=target_G,
        )

        # Return the denoised counts and the unmet budget for this minibatch.
        denoised_counts_NG = torch.clamp(data - noise_counts_NG, min=torch.zeros_like(data), max=data)

        return denoised_counts_NG, unmet_budget_G + carryover_G

    @torch.no_grad()
    def _get_noise_budget_per_gene(self,
                                   how: str = 'fpr',
                                   bernoulli: bool = True) -> Callable[[float], torch.Tensor]:
        """Compute the noise budget on a per-gene basis, returned as a function
        that takes a target value and returns counts per gene in one cell"""

        logger.debug('Computing per-gene noise targets')

        if 'chi_ambient' in pyro.get_param_store().keys():
            chi_ambient_G = pyro.param('chi_ambient').detach()
        else:
            chi_ambient_G = 0.

        chi_bar_G = self.vi_model.avg_gene_expression

        if how == 'fpr':

            # Expectation for counts in empty droplets.
            empty_droplet_mean_counts = dist.LogNormal(loc=pyro.param('d_empty_loc'),
                                                       scale=pyro.param('d_empty_scale')).mean
            if self.vi_model.include_rho:
                swapping_fraction = dist.Beta(pyro.param('rho_alpha'), pyro.param('rho_beta')).mean
            else:
                swapping_fraction = 0.
            empty_droplet_mean_counts_G = empty_droplet_mean_counts * chi_ambient_G

            data_loader = self.dataset_obj.get_dataloader(use_cuda=self.use_cuda,
                                                          analyzed_bcs_only=True,
                                                          batch_size=500,
                                                          shuffle=False)

            # Keep a running sum over expected noise counts as we minibatch.
            expected_noise_counts_without_fpr_G = torch.zeros(len(self.dataset_obj.analyzed_gene_inds)).to(self.device)
            expected_real_counts_G = torch.zeros(len(self.dataset_obj.analyzed_gene_inds)).to(self.device)
            expected_cells = 0

            for i, data in enumerate(data_loader):
                enc = self.vi_model.encoder(x=data,
                                            chi_ambient=chi_ambient_G,
                                            cell_prior_log=self.vi_model.d_cell_loc_prior)
                p_batch = enc['p_y'].sigmoid().detach()
                epsilon_batch = dist.Gamma(enc['epsilon'] * self.vi_model.epsilon_prior,
                                           self.vi_model.epsilon_prior).mean.detach()
                empty_droplet_counts_G = data[p_batch <= 0.5].sum(dim=0)
                expected_ambient_counts_in_cells_G = (empty_droplet_mean_counts_G
                                                      * epsilon_batch[p_batch > 0.5].sum())
                expected_swapping_counts_in_cells_G = (swapping_fraction * chi_bar_G *
                                                       (data * epsilon_batch.unsqueeze(-1))[p_batch > 0.5].sum())
                expected_noise_counts_without_fpr_G = (expected_noise_counts_without_fpr_G
                                                       # + empty_droplet_counts_G
                                                       + expected_ambient_counts_in_cells_G
                                                       + expected_swapping_counts_in_cells_G
                                                       )
                expected_real_counts_G = (expected_real_counts_G
                                          + torch.clamp(data[p_batch > 0.5].sum(dim=0)
                                                        - expected_noise_counts_without_fpr_G, min=0.))
                expected_cells = expected_cells + (p_batch > 0.5).sum()

            expected_noise_count_fcn_per_cell_G = \
                lambda x: ((expected_noise_counts_without_fpr_G + expected_real_counts_G * x)  # fpr addition
                           / expected_cells)

        elif how == 'cdf':

            raise NotImplementedError('TODO')

        else:
            raise NotImplementedError(f'No method {how} for _get_noise_budget_per_gene()')

        # TODO: consider - floor() ruined the game (per cell) for the optimal calcs
        # # Deal with converting floats to integer values, since budgets are integers.
        # # Carryover is kind of a moot point here for FPR since we look at the whole dataset.
        # if bernoulli:
        #     remaining_prob_G = expected_noise_counts_per_cell_G - expected_noise_counts_per_cell_G.floor()
        #     bernoulli_draw_G = dist.Bernoulli(remaining_prob_G).sample()
        #     expected_noise_counts_per_cell_G = expected_noise_counts_per_cell_G.floor() + bernoulli_draw_G
        # else:
        #     expected_noise_counts_per_cell_G = expected_noise_counts_per_cell_G.floor()

        return expected_noise_count_fcn_per_cell_G

    @staticmethod
    @torch.no_grad()
    def _mckp_noise_given_log_prob_tensor(log_prob_noise_counts_NGC: torch.Tensor,
                                          offset_noise_counts_NG: torch.Tensor,
                                          data_NG: torch.Tensor,
                                          target_G: torch.Tensor,
                                          debug: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve the multiple-choice knapsack problem that will apportion discrete
        noise counts to each cell, given per-gene target removal constraints.

        Args:
            log_prob_noise_counts_NGC: Log probability of noise count C in each
                cell, gene (N, G)
            offset_noise_counts_NGC: If the axis for noise counts in log prob
                does not start at zero, this can specify it.
            data_NG: The max noise counts in each matrix element (the data).
            target_G: The target removal for each gene in this batch of cells.
            debug: True to write debugging intermediate output.

        Returns:
            noise_counts_NG: Noise count matrix
            unmet_budget_G: Budget of remaining noise counts that was unable to
                be met. Only nonzero if target_G > data_NG.sum(dim=0)

        NOTE: The runtime of this is proportional to how far the argmax solution is
        from the target number of removed counts (for the gene where this is worst).

        """

        max_count_axis = log_prob_noise_counts_NGC.shape[-1] - 1

        # Get the argmax solution.
        max_NG_tuple = torch.max(log_prob_noise_counts_NGC, dim=-1)
        noise_counts_NG = max_NG_tuple[1]
        argmax_log_prob_NG = max_NG_tuple[0]
        argmax_noise_counts_NG = noise_counts_NG.float()
        if debug:
            print(argmax_noise_counts_NG)
            print(argmax_log_prob_NG)
            assert (argmax_noise_counts_NG <= data_NG).all(), \
                'Improper log_prob has finite entries where noise counts exceed data'

        # Trim the target down so that we can reach it.
        if debug:
            print('target_G')
            print(target_G)
        untrimmed_target_G = target_G.clone()
        target_G = torch.clamp(target_G - offset_noise_counts_NG.sum(dim=0),  # offset gets removed at end
                               min=torch.zeros_like(target_G),
                               max=torch.clamp(data_NG, max=max_count_axis).sum(dim=0))
        if debug:
            print('trimmed target')
            print(target_G)

        # Initialize necessary quantities.
        current_noise_counts_NG = argmax_noise_counts_NG
        current_log_prob_NG = argmax_log_prob_NG
        step_direction_G = torch.sign(target_G - argmax_noise_counts_NG.sum(dim=0))

        # Take one step at a time until constraints are met.
        max_n_steps = (argmax_noise_counts_NG.sum(dim=0) - target_G).abs().max().int()
        for i in range(max_n_steps):

            # Find the cost of all possible local moves.
            single_step_noise_counts_NG = current_noise_counts_NG + step_direction_G.unsqueeze(0)
            clamped_single_step_noise_counts_NG = torch.clamp(single_step_noise_counts_NG,
                                                              min=0.,
                                                              max=max_count_axis)
            if debug:
                print('proposal')
                print(single_step_noise_counts_NG)
            log_prob_next_move_NG = torch.gather(
                log_prob_noise_counts_NGC,
                dim=-1,
                index=clamped_single_step_noise_counts_NG.long().unsqueeze(-1),
            ).squeeze()
            log_prob_cost_NG = current_log_prob_NG - log_prob_next_move_NG

            # Make impossible moves impossible to choose.
            log_prob_cost_NG = torch.where(((single_step_noise_counts_NG > data_NG)
                                            | (single_step_noise_counts_NG > max_count_axis)
                                            | (single_step_noise_counts_NG < 0.)),
                                           np.inf * torch.ones_like(log_prob_cost_NG),
                                           log_prob_cost_NG)
            if debug:
                print('moved prob')
                print(log_prob_next_move_NG)
                print('cost')
                print(log_prob_cost_NG)

            # Take the best local move.
            min_G_tuple = torch.min(log_prob_cost_NG, dim=0)  # ties go to first index
            new_log_prob_G = min_G_tuple[0]
            cell_index_of_best_move_G = min_G_tuple[1]
            onehot_best_moves_NG = torch.nn.functional.one_hot(cell_index_of_best_move_G,
                                                               num_classes=data_NG.shape[0]).t()
            # Do not allow an impossible move: still needed since all infs will pick first index.
            onehot_best_moves_NG = torch.where((new_log_prob_G == np.inf).unsqueeze(0),
                                               torch.zeros_like(onehot_best_moves_NG),
                                               onehot_best_moves_NG)
            current_noise_counts_NG = (current_noise_counts_NG
                                       + (step_direction_G.unsqueeze(0) * onehot_best_moves_NG))
            current_log_prob_NG = torch.where(onehot_best_moves_NG == 0.,
                                              current_log_prob_NG,
                                              log_prob_next_move_NG)
            if debug:
                print('step')
                print(current_noise_counts_NG)
                print(current_log_prob_NG)

            # Update step directions for those where constraint is met.
            step_direction_G = torch.sign(target_G - current_noise_counts_NG.sum(dim=0))
            if step_direction_G.abs().sum() == 0:
                logger.debug('MCKP completed early')
                break

            # Also if for some reason we cannot move, we are done.
            if onehot_best_moves_NG.sum() == 0:
                logger.debug('Early stopping during MCKP: no more valid moves')
                break

        noise_counts_NG = current_noise_counts_NG + offset_noise_counts_NG
        unmet_budget_G = untrimmed_target_G - noise_counts_NG.sum(dim=0)

        return noise_counts_NG, unmet_budget_G

    @staticmethod
    @torch.no_grad()
    def _mckp_noise_given_log_prob_tensor_fast(log_prob_noise_counts_NGC: torch.Tensor,
                                               offset_noise_counts_NG: torch.Tensor,
                                               data_NG: torch.Tensor,
                                               target_G: torch.Tensor,
                                               debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve the multiple-choice knapsack problem that will apportion discrete
        noise counts to each cell, given per-gene target removal constraints.

        Args:
            log_prob_noise_counts_NGC: Log probability of noise count C in each
                cell, gene (N, G)
            offset_noise_counts_NGC: If the axis for noise counts in log prob
                does not start at zero, this can specify it.
            data_NG: The max noise counts in each matrix element (the data).
            target_G: The target removal for each gene in this batch of cells.
            debug: True to write debugging intermediate output.

        Returns:
            noise_counts_NG: Noise count matrix
            unmet_budget_G: Budget of remaining noise counts that was unable to
                be met. Only nonzero if target_G > data_NG.sum(dim=0)

        NOTE: The runtime of this is proportional to something like G log(NC).
        Current memory requirements are high.

        """

        max_count_axis = log_prob_noise_counts_NGC.shape[-1] - 1

        # TODO: eliminate genes which have data_NG.sum(dim=0) = 0 or have target_G = 0?
        zero_target_condition_G = (target_G == 0)
        zero_data_condition_G = (data_NG.sum(dim=0) == 0)

        # Get the argmax solution.
        max_NG_tuple = torch.max(log_prob_noise_counts_NGC, dim=-1)
        noise_counts_NG = max_NG_tuple[1]
        argmax_log_prob_NG = max_NG_tuple[0]
        argmax_noise_counts_NG = noise_counts_NG.float()
        if debug:
            print(argmax_noise_counts_NG)
            print(argmax_log_prob_NG)
            assert (argmax_noise_counts_NG <= data_NG).all(), \
                'Improper log_prob has finite entries where noise counts exceed data'

        # Trim the target down so that we can reach it.
        unmodified_target_G = target_G.clone()
        target_G = torch.clamp(target_G - offset_noise_counts_NG.sum(dim=0),  # offset gets removed at end
                               min=torch.zeros_like(target_G),
                               max=torch.clamp(data_NG, max=max_count_axis).sum(dim=0))

        # TODO: eliminate genes which are already meeting targets BUT THEN ADD IN ARGMAX SOLUTION
        target_met_condition_G = (argmax_noise_counts_NG.sum(dim=0) == target_G)
        gene_ind = torch.where(torch.logical_not(zero_data_condition_G
                                                 | zero_target_condition_G
                                                 | target_met_condition_G))[0]
        gene_argmax = torch.where(target_met_condition_G)[0]
        gene_zero = torch.where(zero_data_condition_G | zero_target_condition_G)[0]

        trim_log_prob_noise_counts_NGC = log_prob_noise_counts_NGC[:, gene_ind, :]
        trim_argmax_log_prob_NG = argmax_log_prob_NG[:, gene_ind]
        trim_argmax_noise_counts_NG = argmax_noise_counts_NG[:, gene_ind]
        trim_data_NG = data_NG[:, gene_ind]
        trim_target_G = target_G[gene_ind]

        if debug:
            print('gene_ind')
            print(gene_ind)

        if len(gene_ind) == 0:
            # The only solution is all zeros
            return offset_noise_counts_NG, target_G - offset_noise_counts_NG.sum(dim=0)

        # Initialize necessary quantities.
        trim_penalty_from_argmax_NGC = trim_argmax_log_prob_NG.unsqueeze(-1) - trim_log_prob_noise_counts_NGC
        trim_step_direction_G = torch.sign(trim_target_G - trim_argmax_noise_counts_NG.sum(dim=0))
        step_direction_G = torch.sign(target_G - argmax_noise_counts_NG.sum(dim=0))
        trim_max_steps_G = (trim_argmax_noise_counts_NG.sum(dim=0) - trim_target_G).abs()
        if debug:
            print('max_steps_G')
            print(trim_max_steps_G)

        # Create the big vector (per gene) to sort: the first max_steps deltas for each cell.
        trim_num_possible_steps_NG = torch.where((trim_step_direction_G > 0).unsqueeze(0),
                                                 torch.clamp(trim_data_NG - trim_argmax_noise_counts_NG,
                                                             min=torch.zeros_like(trim_data_NG),
                                                             max=trim_max_steps_G.unsqueeze(0)),
                                                 torch.clamp(trim_argmax_noise_counts_NG,
                                                             max=trim_max_steps_G.unsqueeze(0)))
        if debug:
            print('allowed steps:')
            print(trim_num_possible_steps_NG * trim_step_direction_G.unsqueeze(0))

        trim_c_NGC = (torch.arange(trim_log_prob_noise_counts_NGC.shape[-1]).to(data_NG.device)
                      .unsqueeze(0).unsqueeze(0)
                      .expand(trim_log_prob_noise_counts_NGC.shape))
        trim_unreachable_mask_positive_steps_NGC = \
            ((trim_c_NGC > torch.minimum(trim_data_NG,
                                         trim_argmax_noise_counts_NG + trim_num_possible_steps_NG).unsqueeze(-1))
             | (trim_c_NGC < trim_argmax_noise_counts_NG.unsqueeze(-1)))
        trim_unreachable_mask_negative_steps_NGC = \
            ((trim_c_NGC > trim_argmax_noise_counts_NG.unsqueeze(-1))
             | (trim_c_NGC < (trim_argmax_noise_counts_NG - trim_num_possible_steps_NG).unsqueeze(-1)))
        trim_unreachable_mask_NGC = torch.where((trim_step_direction_G > 0).unsqueeze(0).unsqueeze(-1),
                                                trim_unreachable_mask_positive_steps_NGC,
                                                trim_unreachable_mask_negative_steps_NGC)

        trim_inf_unreachable_mask_NGC = 1. / (1. - trim_unreachable_mask_NGC.float())  # inf at unreachable spots
        trim_delta_penalty_NGC = torch.where(
            (trim_step_direction_G > 0).unsqueeze(0).unsqueeze(-1),
            ((trim_penalty_from_argmax_NGC * trim_inf_unreachable_mask_NGC)[..., 1:]
             - (trim_penalty_from_argmax_NGC * trim_inf_unreachable_mask_NGC)[..., :-1]),
            ((trim_penalty_from_argmax_NGC * trim_inf_unreachable_mask_NGC)[..., :-1]
             - (trim_penalty_from_argmax_NGC * trim_inf_unreachable_mask_NGC)[..., 1:]),
        )
        trim_delta_penalty_NGC[~torch.isfinite(trim_delta_penalty_NGC)] = np.nan

        if debug:
            print('log_probs')
            print(trim_log_prob_noise_counts_NGC)
            print('penalty_from_argmax')
            print(trim_penalty_from_argmax_NGC)
            print('unreachable')
            print(trim_inf_unreachable_mask_NGC)
            print('delta_penalties')
            print(trim_delta_penalty_NGC)

        trim_n_NGC = (torch.arange(trim_data_NG.shape[0]).to(data_NG.device)
                      .unsqueeze(-1).unsqueeze(-1)
                      .expand(trim_delta_penalty_NGC.shape))

        trim_sort_tensor_NGC2 = torch.cat([trim_delta_penalty_NGC.unsqueeze(-1),
                                           trim_n_NGC.unsqueeze(-1)], dim=-1)
        trim_sort_tensor_NCG2 = torch.permute(trim_sort_tensor_NGC2, dims=(0, 2, 1, 3))
        trim_sort_tensor_BG2 = torch.flatten(trim_sort_tensor_NCG2, start_dim=0, end_dim=1)
        if debug:
            print(trim_sort_tensor_BG2[:, 0, :])

        # Sort the big tensor.
        trim_sorted_tensor_BG, trim_sorting_inds_BG = torch.sort(trim_sort_tensor_BG2[..., 0],
                                                                 dim=0, descending=False)

        # Accumulate the results in the spirit of a sparse COO matrix.
        steps_NG = torch.zeros_like(data_NG)
        for g in range(trim_data_NG.shape[1]):
            vals = trim_sort_tensor_BG2[..., 0][trim_sorting_inds_BG[:trim_max_steps_G[g].int(), g], g]
            ns = trim_sort_tensor_BG2[..., 1][trim_sorting_inds_BG[:trim_max_steps_G[g].int(), g], g]
            ns = ns[torch.isfinite(vals)]
            if len(ns) > 0:
                ones = torch.ones_like(ns)
                indices = (ns.long(), (ones * gene_ind[g]).long())  # no longer trim_
                steps_NG.index_put_(indices=indices, values=ones, accumulate=True)

        if debug:
            print('steps')
            print(steps_NG)

        # Obtain noise counts first by filling in argmax on non-ignored genes.
        noise_counts_NG = torch.zeros_like(data_NG)
        noise_counts_NG[:, gene_ind] = trim_argmax_noise_counts_NG
        noise_counts_NG[:, gene_argmax] = argmax_noise_counts_NG[:, gene_argmax]  # not analyzed bc argmax = target
        # Add steps and offset.
        noise_counts_NG = noise_counts_NG + step_direction_G.unsqueeze(0) * steps_NG
        noise_counts_NG = noise_counts_NG + offset_noise_counts_NG
        # Find unmet gene budget.
        unmet_budget_G = unmodified_target_G - noise_counts_NG.sum(dim=0)

        return noise_counts_NG, unmet_budget_G

    @torch.no_grad()
    def _get_mean(self):
        """Compute output counts."""

        logger.debug(f'Re-calculating a posterior count matrix: FPR = {self.fpr}')

        # Compute posterior in mini-batches.
        torch.cuda.empty_cache()

        # Dataloader for cells only.
        analyzed_bcs_only = True
        count_matrix = self.dataset_obj.get_count_matrix()  # analyzed barcodes
        cell_logic = (self.latents['p'] > consts.CELL_PROB_CUTOFF)
        dataloader_index_to_analyzed_bc_index = np.where(cell_logic)[0]
        cell_data_loader = DataLoader(
            count_matrix[cell_logic],
            empty_drop_dataset=None,
            batch_size=self.posterior_batch_size,
            fraction_empties=0.,
            shuffle=False,
            sort_by=None,
            use_cuda=self.use_cuda,
        )

        barcodes = []
        genes = []
        counts = []
        ind = 0
        carryover_G = 0.  # initialization for MCKP

        logger.info('Performing posterior sampling of count matrix in mini-batches...')

        for data in cell_data_loader:

            if self.debug:
                logger.debug(f'Posterior minibatch inference starting with droplet {ind}')
                logger.debug('\n' + get_hardware_usage(use_cuda=self.use_cuda))

            # Compute an estimate of the true counts.
            dense_counts, carryover_G = self.mckp_denoised_counts(
                data=data,
                carryover_G=carryover_G,
                n_samples=20,
                n_counts_max=20,
                how='fpr',
                target=self.fpr,
            )

            bcs_i_chunk, genes_i, counts_i = self.dense_to_sparse(dense_counts)

            # Barcode index in the dataloader.
            bcs_i = bcs_i_chunk + ind

            # Obtain the real barcode index since we only use cells.
            bcs_i = dataloader_index_to_analyzed_bc_index[bcs_i.cpu().numpy()]

            # Translate chunk barcode inds to overall inds.
            if analyzed_bcs_only:
                bcs_i = self.dataset_obj.analyzed_barcode_inds[bcs_i]
            else:
                bcs_i = self.barcode_inds[bcs_i]

            # Add sparse matrix values to lists.
            try:
                barcodes.extend(bcs_i.tolist())
                genes.extend(genes_i.tolist())
            except TypeError as e:
                # edge case of a single value
                barcodes.append(bcs_i)
                genes.append(genes_i)
            counts.append(counts_i)  # leave as a list of torch tensors

            # Increment barcode index counter.
            ind += data.shape[0]  # Same as data_loader.batch_size

        # Convert the lists to numpy arrays.
        counts = torch.cat(counts, dim=0).detach().cpu().numpy().astype(np.uint32)
        barcodes = np.array(barcodes, dtype=np.uint32)
        genes = np.array(genes, dtype=np.uint32)  # uint16 is too small!

        # Put the counts into a sparse csc_matrix.
        self._mean = sp.csc_matrix((counts, (barcodes, genes)),
                                   shape=self.count_matrix_shape)

    # @torch.no_grad()
    # def _get_mean(self):
    #     """Send dataset through a guide that returns mean posterior counts.
    #
    #     Keep track of only what is necessary to distill a sparse count matrix.
    #
    #     """
    #
    #     logger.debug(f'Re-calculating a posterior count matrix: alpha = {self.fpr}')
    #
    #     # TODO mods for q
    #
    #     # beta_star_overall = self._get_posterior_regularization_factor(
    #     #     fpr=self.fpr,
    #     #     per_gene=False,
    #     # )
    #     # self.beta_star = beta_star_overall
    #     # logger.info(f'Optimal posterior regularization factor = {self.beta_star.mean():.3f}')
    #
    #     # print(beta_star_overall)  # TODO
    #     # beta_star_gene = self._get_posterior_regularization_factor(
    #     #     fpr=self.fpr,
    #     #     per_gene=True,
    #     #     max_value=beta_star_overall.item(),
    #     # )
    #     # self.beta_star = beta_star_gene
    #     #
    #     # logger.info(f'Optimal (mean) posterior regularization factor = {self.beta_star.mean():.3f}')
    #
    #     # q_star = self._get_posterior_q_given_fpr(fpr=self.fpr, per_gene=False, max_value=1.)
    #     # logger.info(f'Optimal posterior noise q = {q_star.item():.3f}')
    #
    #     # TODO ======^^^
    #
    #     # TODO: need a way to just do this for cells
    #
    #     # Compute posterior in mini-batches.
    #     torch.cuda.empty_cache()
    #     analyzed_bcs_only = True
    #     sorted_data_loader = self.dataset_obj.get_dataloader(
    #         use_cuda=self.use_cuda,
    #         analyzed_bcs_only=analyzed_bcs_only,
    #         batch_size=self.posterior_batch_size,
    #         shuffle=False,
    #         sort_by=lambda x: -1 * np.array(x.max(axis=1).todense()).squeeze(),  # max entry per droplet
    #     )
    #     barcodes = []
    #     genes = []
    #     counts = []
    #     ind = 0
    #
    #     carryover_G = 0.  # initialization for MCKP
    #
    #     logger.info('Performing posterior sampling of count matrix in mini-batches...')
    #
    #     for data in sorted_data_loader:
    #
    #         if self.debug:
    #             logger.debug(f'Posterior minibatch inference starting with droplet {ind}')
    #             logger.debug('\n' + get_hardware_usage(use_cuda=self.use_cuda))
    #
    #         # Compute an estimate of the true counts.
    #         # TODO: trying out the new alpha- or q-posteior generation
    #         # dense_counts = self.counts_from_alpha_regularized_posterior(
    #         #     alpha=self.fpr,
    #         #     data=data,
    #         #     n_samples=100,
    #         #     lambda_multiplier=1.,
    #         # )
    #         # dense_counts = self.counts_from_cdf_threshold(
    #         #     data=data,
    #         #     q=q_star,
    #         #     n_samples=50,
    #         #     y_map=True,
    #         # )
    #         # dense_counts = self._compute_true_counts(data=data,
    #         #                                          chi_ambient=pyro.param('chi_ambient'),
    #         #                                          lambda_multiplier=self.beta_star,
    #         #                                          use_map=False,
    #         #                                          n_samples=9)  # must be odd number
    #         dense_counts, carryover_G = self.mckp_denoised_counts(
    #             data=data,
    #             carryover_G=carryover_G,
    #             n_samples=50,
    #             how='fpr',
    #             target=self.fpr,
    #         )
    #         # TODO ========^^^
    #         bcs_i_chunk, genes_i, counts_i = self.dense_to_sparse(dense_counts)
    #
    #         # Barcode index in the dataloader.
    #         bcs_i = bcs_i_chunk + ind
    #
    #         # Obtain the real barcode index after unsorting the dataloader.
    #         bcs_i = sorted_data_loader.unsort_inds(bcs_i)
    #
    #         # Translate chunk barcode inds to overall inds.
    #         if analyzed_bcs_only:
    #             bcs_i = self.dataset_obj.analyzed_barcode_inds[bcs_i]
    #         else:
    #             bcs_i = self.barcode_inds[bcs_i]
    #
    #         # Add sparse matrix values to lists.
    #         try:
    #             barcodes.extend(bcs_i.tolist())
    #             genes.extend(genes_i.tolist())
    #         except TypeError as e:
    #             # edge case of a single value
    #             barcodes.append(bcs_i)
    #             genes.append(genes_i)
    #         counts.append(counts_i)  # leave as a list of torch tensors
    #
    #         # Increment barcode index counter.
    #         ind += data.shape[0]  # Same as data_loader.batch_size
    #
    #     # Convert the lists to numpy arrays.
    #     counts = torch.cat(counts, dim=0).detach().cpu().numpy().astype(np.uint32)
    #     barcodes = np.array(barcodes, dtype=np.uint32)
    #     genes = np.array(genes, dtype=np.uint32)  # uint16 is too small!
    #
    #     # Put the counts into a sparse csc_matrix.
    #     self._mean = sp.csc_matrix((counts, (barcodes, genes)),
    #                                shape=self.count_matrix_shape)

    @torch.no_grad()
    def counts_from_alpha_regularized_posterior(self,
                                                alpha,
                                                data,
                                                n_samples=50,
                                                lambda_multiplier=1.):
        """Draw a single sample from the regularized posterior"""

        # compute regularized posterior
        log_regularized_posterior_NGC, offset_noise_counts_NG = self.alpha_regularized_posterior(
            alpha=alpha,
            data=data,
            n_samples=n_samples,
            lambda_multiplier=lambda_multiplier,
        )

        # draw a sample of the noise counts
        # TODO: I am messing around with this for experimentation
        #     noise_counts_NG = torch.distributions.Categorical(logits=log_regularized_posterior_NGC).sample()
        #     noise_counts_NG = torch.distributions.Categorical(logits=2 * log_regularized_posterior_NGC).sample()
        noise_counts_NG = torch.argmax(log_regularized_posterior_NGC, dim=-1)
        # =================

        noise_counts_NG = noise_counts_NG + offset_noise_counts_NG

        # convert to a sample of the remaining counts
        return torch.clamp(data - noise_counts_NG, min=0.)

    @torch.no_grad()
    def alpha_regularized_posterior(self,
                                    alpha,
                                    data,
                                    n_samples=50,
                                    lambda_multiplier=1.):
        """Compute the log of the regularized posterior"""

        device = data.device

        # compute log noise PDF
        log_pdf_noise_counts_NGC, offset_noise_counts_NG = self.noise_log_pdf(
            data=data,
            n_samples=n_samples,
            lambda_multiplier=lambda_multiplier,
            y_map=True,
        )

        # eliminate uninteresting entries and flatten batch dimension
        flat = flatten_non_deltas(log_pdf_noise_counts_NGC)
        log_pdf_noise_counts_BC = flat['tensor']
        flatten = flat['flatten_fcn']
        unflatten = flat['unflatten_fcn']
        offset_noise_counts_B = flatten(offset_noise_counts_NG)

        # compute the expectation for regularization
        log_mu_plus_alpha_sigma_B = log_pdf_log_mean_plus_alpha_std(
            noise_log_pdf_BC=log_pdf_noise_counts_BC,
            offset_noise_counts_B=offset_noise_counts_B,
            alpha=alpha,
        )

        # prepare noise count tensor
        noise_count_BC = (torch.arange(log_pdf_noise_counts_BC.shape[-1]).to(device).unsqueeze(0)
                          + offset_noise_counts_B.unsqueeze(-1))

        # parallel binary search for beta for each entry of count matrix
        beta_B = binary_search(
            evaluate_outcome_given_value=lambda x:
            get_alpha_log_constraint_violation_given_beta(
                beta_B=x,
                log_pdf_noise_counts_BC=log_pdf_noise_counts_BC,
                noise_count_BC=noise_count_BC,
                log_mu_plus_alpha_sigma_B=log_mu_plus_alpha_sigma_B,
            ),
            target_outcome=torch.zeros(noise_count_BC.shape[0]).to(device),
            init_range=(torch.tensor([-100., 100.])
                        .to(device)
                        .unsqueeze(0)
                        .expand((noise_count_BC.shape[0],) + (2,))),
            target_tolerance=0.001,
            max_iterations=100,
        )

        # generate regularized posteriors
        log_pdf_reg_BC = log_pdf_noise_counts_BC + beta_B.unsqueeze(-1) * noise_count_BC
        log_pdf_reg_BC = log_pdf_reg_BC - torch.logsumexp(log_pdf_reg_BC, -1, keepdims=True)
        log_pdf_reg_NGC = unflatten(log_pdf_reg_BC)

        return log_pdf_reg_NGC, offset_noise_counts_NG

    @torch.no_grad()
    def _get_posterior_q_given_fpr(
            self,
            fpr: Union[float, str],
            per_gene: bool,
            max_value: float = consts.POSTERIOR_REG_MAX,
    ) -> torch.Tensor:
        """Compute the posterior regularization factor q, which is how far into
        the noise posterior CDF we want to go [0, 1] to get our output. (0
        represents no removal.) The target is a specific FPR, and the goal here
        is to find the right q(s).

        Args:
            fpr: The target false positive rate, if using the FPR strategy
            per_gene: True to return a vector with one value per gene, False to
                return a single number.
            max_value: Return value(s) cannot exceed this (in [0, 1]).

        """

        assert max_value >= 0., 'The maximum q value must be >= 0'
        assert max_value <= 1., 'The maximum q value must be <= 1'

        # Get a dataset of solid cells.
        cell_inds = np.where(self.latents['p'] > 0.9)[0]
        if len(cell_inds) == 0:
            logger.warning('No cells detected (no droplets with posterior '
                           'cell probability > 0.9)!')
            logger.info('Relaxing the stringency for "cells" in FPR computation... '
                        'realize that the FPR here may be inaccurate.')
            cell_inds = np.argsort(self.latents['p'])[::-1][:200]
        qs = []

        logger.debug(f'Finding optimal posterior regularization factor for FPR = {fpr}')

        for _ in range(5):

            # Get a batch of cells.
            n_cells = min(self.posterior_batch_size, cell_inds.size)
            if n_cells == 0:
                raise RuntimeError('No cells found!  Cannot compute expected FPR.')
            cell_ind_subset = self.random.choice(cell_inds, size=n_cells, replace=False)
            cell_data = (torch.tensor(np.array(self.dataset_obj.get_count_matrix()
                                               [cell_ind_subset, :].todense()).squeeze())
                         .float().to(self.vi_model.device))

            # Find the optimal q value using those cells and target FPR.
            fun = lambda x: self._calculate_expected_fpr_given_q(
                data=cell_data,
                q=x,
                per_gene=per_gene,
            )
            q = binary_search(
                evaluate_outcome_given_value=fun,
                target_outcome=(torch.tensor([fpr] * (cell_data.shape[1] if per_gene else 1))
                                .to(device=cell_data.device)),
                init_range=(torch.tensor([0, max_value])
                            .to(device=cell_data.device)
                            .unsqueeze(0)
                            .expand([cell_data.shape[1] if per_gene else 1, 2])),
            )
            qs.append(q)

        q_star = torch.cat([t.unsqueeze(-1) for t in qs], dim=-1).mean(dim=-1)

        return q_star

    @torch.no_grad()
    def _get_posterior_regularization_factor(
        self,
        fpr: Union[float, str],
        per_gene: bool,
        max_value: float = consts.POSTERIOR_REG_MAX,
    ) -> torch.Tensor:
        """Compute the posterior regularization factor referred to in the paper
        as beta*, which is a multiplier that multiplies the Poisson noise rate
        parameter when the posterior count matrix is computed, as a form of
        approximate posterior regularization.

        Args:
            fpr: The target false positive rate, if using the FPR strategy
            per_gene: True to return a vector with one value per gene, False to
                return a single number.
            max_value: Return value(s) cannot exceed this.

        """

        beta_star = None

        if fpr == 'cohort':
            logger.warning('Using posterior regularization factor 1 in "cohort" mode.')
            beta_star = torch.tensor([1.])

        else:

            # Get a dataset of solid cells.
            cell_inds = np.where(self.latents['p'] > 0.9)[0]
            if len(cell_inds) == 0:
                logger.warning('No cells detected (no droplets with posterior '
                               'cell probability > 0.9)!')
                logger.info('Relaxing the stringency for "cells" in FPR computation... '
                            'realize that the FPR here may be inaccurate.')
                cell_inds = np.argsort(self.latents['p'])[::-1][:200]
            lambda_mults = []

            logger.debug(f'Finding optimal posterior regularization factor for FPR = {fpr}')

            for _ in range(5):

                n_cells = min(self.posterior_batch_size, cell_inds.size)
                if n_cells == 0:
                    raise ValueError('No cells found!  Cannot compute expected FPR.')
                cell_ind_subset = self.random.choice(cell_inds, size=n_cells, replace=False)
                cell_data = (torch.tensor(np.array(self.dataset_obj.get_count_matrix()
                                                   [cell_ind_subset, :].todense()).squeeze())
                             .float().to(self.vi_model.device))

                # Get the latents mu, alpha, and lambda for those cells.
                map_est = self._param_map_estimates(data=cell_data, chi_ambient=pyro.param('chi_ambient'))

                # Find the optimal lambda_multiplier value using those cells and target FPR.
                fun = lambda beta: self._calculate_expected_fpr_given_lambda_mult(
                    data=cell_data,
                    lambda_mult=beta,
                    mu_est=map_est['mu'],
                    lambda_est=map_est['lam'],
                    alpha_est=map_est['alpha'],
                    per_gene=per_gene,
                )
                lambda_mult = binary_search(
                    evaluate_outcome_given_value=fun,
                    target_outcome=(torch.tensor([fpr] * (cell_data.shape[1] if per_gene else 1))
                                    .to(device=cell_data.device)),
                    init_range=(torch.tensor([consts.POSTERIOR_REG_MIN, max_value])
                                .to(device=cell_data.device)
                                .unsqueeze(0)
                                .expand([cell_data.shape[1] if per_gene else 1, 2])),
                )
                # lambda_mult = self._beta_binary_search_given_fpr(cell_data=cell_data,
                #                                                  fpr=fpr,
                #                                                  mu_est=map_est['mu'],
                #                                                  lambda_est=map_est['lam'],
                #                                                  alpha_est=map_est['alpha'],
                #                                                  per_gene=per_gene)
                lambda_mults.append(lambda_mult)

            # beta_star = np.mean(np.concatenate([np.expand_dims(arr, axis=0)
            #                                     for arr in lambda_mults], axis=0), axis=0)
            beta_star = torch.cat([t.unsqueeze(-1) for t in lambda_mults], dim=-1).mean(dim=-1)

        # elif confidence is not None:
        #
        #     raise NotImplementedError

        return beta_star

    @torch.no_grad()
    def _compute_true_counts(self,
                             data: torch.Tensor,
                             chi_ambient: torch.Tensor,
                             lambda_multiplier: float,
                             use_map: bool = True,
                             y_map: bool = False,
                             n_samples: int = 1) -> torch.Tensor:
        """Compute the true de-noised count matrix for this minibatch.

        Can use either a MAP estimate of lambda and mu, or can use a sampling
        approach.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            chi_ambient: Point estimate of inferred ambient gene expression.
            lambda_multiplier: Value by which lambda gets multiplied in order
                to compute regularized posterior true counts.
            use_map: True to use a MAP estimate of lambda and mu.
            y_map: True to enforce the use of a MAP estimate of y rather than
                sampling y, even when sampling all other latents. This prevents
                some samples from having a cell and some not, which can lead
                to strange summary statistics over many samples.
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
            dense_counts_torch = self._map_counts_from_params(data,
                                                              mu_map + 1e-30,
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
                    self._param_sample(data, y_map=y_map)

                # Compute the de-noised count matrix given the estimates.
                dense_counts_torch[..., i] = \
                    self._map_counts_from_params(data,
                                                 mu_sample + 1e-30,
                                                 lambda_sample * lambda_multiplier + 1e-30,
                                                 alpha_sample + 1e-30)

            # Take the median of the posterior true count distribution... torch cuda does not implement mode
            dense_counts = dense_counts_torch.median(dim=2, keepdim=False)[0].detach()

        return dense_counts

    @torch.no_grad()
    def _param_sample(self,
                      data: torch.Tensor,
                      y_map: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate a single sample estimate of mu, the mean of the true count
        matrix, and lambda, the rate parameter of the Poisson background counts.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            y_map: True to enforce the use of a MAP estimate of y rather than
                sampling y. This prevents some samples from having a cell and
                some not, which can lead to strange summary statistics over
                many samples.

        Returns:
            mu_sample: Dense tensor sample of Negative Binomial mean for true
                counts.
            lambda_sample: Dense tensor sample of Poisson rate params for noise
                counts.
            alpha_sample: Dense tensor sample of Dirichlet concentration params
                that inform the overdispersion of the Negative Binomial.

        """

        logger.debug('Replaying model with guide to sample mu, alpha, lambda')

        # Use pyro poutine to trace the guide and sample parameter values.
        guide_trace = pyro.poutine.trace(self.vi_model.guide).get_trace(x=data)

        # TODO: consider replacing p_y with a MAP estimate so that you never get
        # TODO: samples of cell + no cell
        if y_map:
            guide_trace.nodes['y']['value'] = (guide_trace.nodes['p_passback']['value'] > 0).clone().detach()

        replayed_model = pyro.poutine.replay(self.vi_model.model, guide_trace)

        # Run the model using these sampled values.
        replayed_model_output = replayed_model(x=data)

        # The model returns mu, alpha, and lambda.
        mu_sample = replayed_model_output['mu']
        lambda_sample = replayed_model_output['lam']
        alpha_sample = replayed_model_output['alpha']

        return mu_sample, lambda_sample, alpha_sample

    @staticmethod
    @torch.no_grad()
    def _log_prob_noise_count_tensor(data: torch.Tensor,
                                     mu_est: torch.Tensor,
                                     lambda_est: torch.Tensor,
                                     alpha_est: Optional[torch.Tensor],
                                     n_counts_max: int = 100,
                                     debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate a MAP estimate of the true counts given a single sample
        estimate of mu, the mean of the true count matrix, lambda, the rate
        parameter of the Poisson background counts, and the data.

        NOTE: this is un-normalized log probability

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            mu_est: Dense tensor of Negative Binomial means for true counts.
            lambda_est: Dense tensor of Poisson rate params for noise counts.
            alpha_est: Dense tensor of Dirichlet concentration params that
                inform the overdispersion of the Negative Binomial.  None will
                use an all-Poisson model

        Returns:
            log_prob_tensor: Probability of each noise count value.
            poisson_values_low: The starting point for noise counts for each
                cell and gene, because they can be different.

        """

        # Estimate a reasonable low-end to begin the Poisson summation.
        n = min(n_counts_max, data.max().item())  # No need to exceed the max value
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
        # This results in NaNs.
        if alpha_est is None:
            # Poisson only model
            log_prob_tensor = (dist.Poisson(lambda_est.unsqueeze(-1), validate_args=False)
                               .log_prob(noise_count_tensor)
                               + dist.Poisson(mu_est.unsqueeze(-1), validate_args=False)
                               .log_prob(data.unsqueeze(-1) - noise_count_tensor))
            logger.debug('Using all poisson model (since alpha is not supplied to posterior)')
        else:
            logits = (mu_est.log() - alpha_est.log()).unsqueeze(-1)
            log_prob_tensor = (dist.Poisson(lambda_est.unsqueeze(-1), validate_args=False)
                               .log_prob(noise_count_tensor)
                               + dist.NegativeBinomial(total_count=alpha_est.unsqueeze(-1),
                                                       logits=logits,
                                                       validate_args=False)
                               .log_prob(data.unsqueeze(-1) - noise_count_tensor))

        # Set log_prob to -inf if noise > data.
        neg_inf_tensor = torch.ones_like(log_prob_tensor) * -np.inf
        log_prob_tensor = torch.where((noise_count_tensor <= data.unsqueeze(-1)),
                                      log_prob_tensor,
                                      neg_inf_tensor)

        # Set log_prob to -inf, -inf, -inf, 0 for entries where mu == 0, since they will be NaN
        # TODO: either this or require that mu > 0...
        # log_prob_tensor = torch.where(mu_est == 0,
        #                               data,
        #                               log_prob_tensor)

        logger.debug(f'Prob computation with tensor of shape {log_prob_tensor.shape}')

        if debug:
            assert not torch.isnan(log_prob_tensor).any(), \
                'log_prob_tensor contains a NaN'
            if torch.isinf(log_prob_tensor).all(dim=-1).any():
                print(torch.where(torch.isinf(log_prob_tensor).all(dim=-1)))
                raise AssertionError('There is at least one log_prob_tensor[n, g, :] that has all-zero probability')

        return log_prob_tensor, poisson_values_low

    @staticmethod
    @torch.no_grad()
    def _map_counts_from_params(data: torch.Tensor,
                                mu_est: torch.Tensor,
                                lambda_est: torch.Tensor,
                                alpha_est: Optional[torch.Tensor],
                                debug: bool = False) -> torch.Tensor:
        """Calculate a MAP estimate of the true counts given a single sample
        estimate of mu, the mean of the true count matrix, lambda, the rate
        parameter of the Poisson background counts, and the data.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            mu_est: Dense tensor of Negative Binomial means for true counts.
            lambda_est: Dense tensor of Poisson rate params for noise counts.
            alpha_est: Dense tensor of Dirichlet concentration params that
                inform the overdispersion of the Negative Binomial.  None will
                use an all-Poisson model

        Returns:
            dense_counts_torch: Dense matrix of true de-noised counts.

        """

        # The big tensor: log prob of each possible noise count value, for each cell and gene
        log_prob_tensor, poisson_values_low = ProbPosterior._log_prob_noise_count_tensor(
            data=data,
            mu_est=mu_est,
            lambda_est=lambda_est,
            alpha_est=alpha_est,
            debug=debug,
        )

        # Find the most probable number of noise counts per cell per gene.
        noise_count_map = torch.argmax(log_prob_tensor,
                                       dim=-1,
                                       keepdim=False).float()
        noise_count_map = noise_count_map + poisson_values_low  # add the offset back in

        # Handle the cases where y = 0 (no cell): all counts are noise.
        noise_count_map = torch.where(mu_est == 0,
                                      data,
                                      noise_count_map)

        # Compute the number of true counts.
        dense_counts_torch = torch.clamp(data - noise_count_map, min=0.)

        return dense_counts_torch

    @staticmethod
    @torch.no_grad()
    def _sample_counts_from_params(data: torch.Tensor,
                                   mu_est: torch.Tensor,
                                   lambda_est: torch.Tensor,
                                   alpha_est: Optional[torch.Tensor],
                                   debug: bool = False) -> torch.Tensor:
        """Take a single sample of the true counts given a single sample
        estimate of mu, the mean of the true count matrix, lambda, the rate
        parameter of the Poisson background counts, and the data.

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            mu_est: Dense tensor of Negative Binomial means for true counts.
            lambda_est: Dense tensor of Poisson rate params for noise counts.
            alpha_est: Dense tensor of Dirichlet concentration params that
                inform the overdispersion of the Negative Binomial.  None will
                use an all-Poisson model

        Returns:
            dense_counts_torch: Dense matrix of true de-noised counts.

        """

        # The big tensor: log prob of each possible noise count value, for each cell and gene
        log_prob_tensor, poisson_values_low = ProbPosterior._log_prob_noise_count_tensor(
            data=data,
            mu_est=mu_est,
            lambda_est=lambda_est,
            alpha_est=alpha_est,
            debug=debug,
        )

        # Find the most probable number of noise counts per cell per gene.
        noise_count_map = torch.distributions.Categorical(logits=log_prob_tensor).sample()
        noise_count_map = noise_count_map + poisson_values_low  # add the offset back in

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
                                         data_map: torch.Tensor,
                                         per_gene: bool) -> torch.Tensor:
        """(Done previously: given inferred latent variables and observed total counts,
        generate a MAP estimate for noise counts.) Here, use that MAP estimate to
        compute the expected false positive rate.

        Args:
            data: Dense tensor tiny minibatch of cell by gene count data.
            data_map: Dense tensor tiny minibatch of MAP output for that data.
            per_gene: True to return a value for each gene, False to return a scalar.

        Returns:
            fpr: Expected false positive rate.

        """

        empty_droplet_mean_counts = dist.LogNormal(loc=pyro.param('d_empty_loc'),
                                                   scale=pyro.param('d_empty_scale')).mean
        if self.vi_model.include_rho:
            swapping_fraction = dist.Beta(pyro.param('rho_alpha'), pyro.param('rho_beta')).mean
        else:
            swapping_fraction = 0.
        mean_cell_epsilon = (self.latents['epsilon'][self.latents['p'] > 0.5]).mean()

        if not per_gene:

            counts_per_cell = data.sum(dim=-1)
            ambient_fraction = empty_droplet_mean_counts / counts_per_cell
            target_fraction_counts_removed = (ambient_fraction + swapping_fraction) / mean_cell_epsilon
            map_counts_per_cell = data_map.sum(dim=-1)
            fraction_counts_removed_per_cell = (counts_per_cell - map_counts_per_cell) / counts_per_cell
            fpr_expectation = (fraction_counts_removed_per_cell - target_fraction_counts_removed).mean()
            fpr_expectation = torch.clamp(fpr_expectation, min=0.)
            return fpr_expectation.mean()

        else:

            empty_droplet_mean_counts_G = empty_droplet_mean_counts * pyro.param('chi_ambient')
            mean_cell_counts_G = data.mean(dim=0)
            ambient_fraction_G = empty_droplet_mean_counts_G / mean_cell_counts_G
            fraction_counts_removed_G = (data.sum(dim=0) - data_map.sum(dim=0)) / (data.sum(dim=0) + 1e-10)
            target_fraction_counts_removed_G = (ambient_fraction_G + swapping_fraction) / mean_cell_epsilon
            fpr_expectation_G = fraction_counts_removed_G - target_fraction_counts_removed_G
            fpr_expectation_G = torch.clamp(fpr_expectation_G, min=0.)
            print(fpr_expectation_G)
            return fpr_expectation_G

    @torch.no_grad()
    def _calculate_expected_fpr_given_lambda_mult(self,
                                                  data: torch.Tensor,
                                                  lambda_mult: Union[float, torch.Tensor],
                                                  mu_est: torch.Tensor,
                                                  alpha_est: torch.Tensor,
                                                  lambda_est: torch.Tensor,
                                                  per_gene: bool) -> torch.Tensor:
        """Given a float lambda_mult, calculate a MAP estimate of output counts,
        and use that estimate to calculate an expected false positive rate.

        Args:
            data: Dense tensor tiny minibatch of cell by gene count data.
            lambda_mult: Value of the lambda multiplier (can be gene-length vector).
            mu_est: Dense tensor of Negative Binomial means for true counts.
            lambda_est: Dense tensor of Poisson rate params for noise counts.
            alpha_est: Dense tensor of Dirichlet concentration params that
                inform the overdispersion of the Negative Binomial.
            per_gene: True to compute an expected FPR for each gene, otherwise
                computes an average over all genes

        Returns:
            fpr: Expected false positive rate.

        """

        # Compute MAP estimate of true de-noised counts.
        if type(lambda_mult) != float:
            lambda_mult = lambda_mult.unsqueeze(0)
        data_map = self._map_counts_from_params(data=data,
                                                mu_est=mu_est,
                                                lambda_est=lambda_est * lambda_mult,
                                                alpha_est=alpha_est)

        # Compute expected false positive rate.
        expected_fpr = self._calculate_expected_fpr_from_map(data=data,
                                                             data_map=data_map,
                                                             per_gene=per_gene)

        return expected_fpr

    @torch.no_grad()
    def _calculate_expected_fpr_given_q(self,
                                        data: torch.Tensor,
                                        q: Union[float, torch.Tensor],
                                        per_gene: bool) -> torch.Tensor:
        """Given a float lambda_mult, calculate a MAP estimate of output counts,
        and use that estimate to calculate an expected false positive rate.

        Args:
            data: Dense tensor tiny minibatch of cell by gene count data.
            q: Value of the noise CDF threshold (can be gene-length vector).
            per_gene: True to compute an expected FPR for each gene, otherwise
                computes an average over all genes

        Returns:
            fpr: Expected false positive rate.

        """

        # Compute MAP estimate of true de-noised counts.
        data_map = self.counts_from_cdf_threshold(
            data=data, q=q, n_samples=20, y_map=True,
        )

        # Compute expected false positive rate.
        expected_fpr = self._calculate_expected_fpr_from_map(data=data,
                                                             data_map=data_map,
                                                             per_gene=per_gene)

        return expected_fpr

    @torch.no_grad()
    def _beta_binary_search_given_fpr(self,
                                      cell_data: torch.Tensor,
                                      fpr: float,
                                      mu_est: torch.Tensor,
                                      lambda_est: torch.Tensor,
                                      alpha_est: torch.Tensor,
                                      lam_mult_init: float = 1.,
                                      fpr_tolerance: Optional[float] = None,
                                      per_gene: bool = False,
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
            per_gene: True to find one posterior regularization factor for each
                gene, instead of a single scalar.
            max_iterations: A cutoff to ensure termination. Even if a tolerable
                solution is not found, the algorithm will stop after this many
                iterations and return the best answer so far.

        Returns:
            lam_mult: Value of the lambda-multiplier.

        """

        if per_gene:
            raise NotImplementedError

        logger.debug('Binary search commencing')

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

            logger.debug(f'Binary search limits: {lam_mult_bracket}')

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
            logger.info(f'FPR control not achieved in {max_iterations} attempts. '
                        f'Output FPR is estimated to be {expected_fpr.item():.4f}')

        return lam_mult

    @torch.no_grad()
    def _q_binary_search_given_fpr(self,
                                    cell_data: torch.Tensor,
                                    fpr: float,
                                    mu_est: torch.Tensor,
                                    lambda_est: torch.Tensor,
                                    alpha_est: torch.Tensor,
                                    lam_mult_init: float = 1.,
                                    fpr_tolerance: Optional[float] = None,
                                    per_gene: bool = False,
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
            per_gene: True to find one posterior regularization factor for each
                gene, instead of a single scalar.
            max_iterations: A cutoff to ensure termination. Even if a tolerable
                solution is not found, the algorithm will stop after this many
                iterations and return the best answer so far.

        Returns:
            lam_mult: Value of the lambda-multiplier.

        """

        if per_gene:
            raise NotImplementedError

        logger.debug('Binary search commencing')

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

            logger.debug(f'Binary search limits: {lam_mult_bracket}')

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
            logger.info(f'FPR control not achieved in {max_iterations} attempts. '
                        f'Output FPR is estimated to be {expected_fpr.item():.4f}')

        return lam_mult

        # binary_search(
        #     evaluate_outcome_given_value: Callable[[torch.Tensor], torch.Tensor],
        # target_outcome: torch.Tensor,
        # init_value = torch.ones(),
        # )


@torch.no_grad()
def binary_search(
    evaluate_outcome_given_value: Callable[[torch.Tensor], torch.Tensor],
    target_outcome: torch.Tensor,
    init_range: torch.Tensor,
    target_tolerance: Optional[float] = 0.001,
    max_iterations: int = consts.POSTERIOR_REG_SEARCH_MAX_ITER,
    debug: bool = False,
) -> torch.Tensor:
    """Perform a binary search, given a target and an evaluation function.

    NOTE: evaluate_outcome_given_value(value) should increase monotonically
    with the input value. It is assumed that
    consts.POSTERIOR_REG_MIN < output_value < consts.POSTERIOR_REG_MAX.
    If this is not the case, the algorithm will produce an output close to one
    of those endpoints, and target_tolerance will not be achieved.
    Moreover, output_value must be positive (due to how we search for limits).

    Args:
        evaluate_outcome_given_value: Function that takes a value as its
            input and produces the outcome, which is the target we are
            trying to control. Should increase monotonically with value.
        target_outcome: Desired outcome value from evaluate_outcome_given_value(value).
        init_range: Search range, for each value.
        target_tolerance: Tolerated error in the target value.
        max_iterations: A cutoff to ensure termination. Even if a tolerable
            solution is not found, the algorithm will stop after this many
            iterations and return the best answer so far.
        debug: Print debugging messages.

    Returns:
        value: Result of binary search. Same shape as init_value.

    """

    logger.debug('Binary search commencing')

    assert (target_tolerance > 0), 'target_tolerance should be > 0.'
    assert len(init_range.shape) > 1, 'init_range must be at least two-dimensional ' \
                                      '(last dimension contains lower and upper bounds)'
    assert init_range.shape[-1] == 2, 'Last dimension of init_range should be 2: low and high'

    value_bracket = init_range.clone()

    # Binary search algorithm.
    for i in range(max_iterations):

        logger.debug(f'Binary search limits [batch_dim=0, :]: '
                     f'{value_bracket.reshape(-1, value_bracket.shape[-1])[0, :]}')

        # Current test value.
        value = value_bracket.mean(dim=-1)

        # Calculate an expected false positive rate for this lam_mult value.
        outcome = evaluate_outcome_given_value(value)
        residual = target_outcome - outcome

        # Check on residual and update our bracket values.
        stop_condition = (residual.abs() < target_tolerance).all()
        if stop_condition:
            break
        else:
            value_bracket[..., 0] = torch.where(outcome < target_outcome - target_tolerance,
                                                value,
                                                value_bracket[..., 0])
            value_bracket[..., 1] = torch.where(outcome > target_outcome + target_tolerance,
                                                value,
                                                value_bracket[..., 1])

    # If we stopped due to iteration limit, take the average value.
    if i == max_iterations:
        value = value_bracket.mean(dim=-1)
        logger.warning(f'Binary search target not achieved in {max_iterations} attempts. '
                       f'Output is estimated to be {outcome.mean().item():.4f}')

    # Warn if we railed out at the limits of the search
    if debug:
        if (value - target_tolerance <= init_range[..., 0]).sum() > 0:
            logger.debug(f'{(value - target_tolerance <= init_range[..., 0]).sum()} '
                         f'entries in the binary search hit the lower limit')
            logger.debug(value[value - target_tolerance <= init_range[..., 0]])
        if (value + target_tolerance >= init_range[..., 1]).sum() > 0:
            logger.debug(f'{(value + target_tolerance >= init_range[..., 1]).sum()} '
                         f'entries in the binary search hit the upper limit')
            logger.debug(value[value + target_tolerance >= init_range[..., 1]])

    return value


def flatten_non_deltas(log_pdf_NGC: torch.Tensor, fill_value=None) \
        -> Dict[str, Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]]:
    """Flatten tensor to include only PDFs that are not delta functions on zero"""

    not_delta_fcns_NG = (log_pdf_NGC.exp()[:, :, 0] != 1.)
    inds = torch.nonzero(not_delta_fcns_NG, as_tuple=True)

    def _flatten(x):
        return x[inds]

    def _unflatten(x, fill_value=fill_value):
        if fill_value is None:
            fill_value = torch.zeros(log_pdf_NGC.shape[-1]).to(x.device)
            fill_value[0] = 1.
            fill_value = fill_value.log().expand(log_pdf_NGC.shape)
        out = torch.ones(log_pdf_NGC.shape).to(x.device) * fill_value
        out[inds] = x
        return out

    flat_log_pdf = _flatten(log_pdf_NGC)

    return {'tensor': flat_log_pdf, 'flatten_fcn': _flatten, 'unflatten_fcn': _unflatten}


def log_pdf_log_mean_plus_alpha_std(noise_log_pdf_BC: torch.Tensor,
                                    offset_noise_counts_B: torch.Tensor,
                                    alpha: float) -> torch.Tensor:
    """Given the empirical log probability density of noise counts,
    compute the mean and the standard deviation, and return
    log(mean + alpha * std)

    NOTE: noise_log_pdf_BC should be normalized

    Args:
        noise_log_pdf_BC: The computed log probability distribution of
            noise counts. Last dimension is noise counts, and all
            other dimensions are batch dimensions.
        offset_noise_counts_B: Offsets computed for the noise count
            dimensions (i.e. if a noise count axis does not start
            at zero)
        alpha: The tuning parameter that determines

    Returns:
        log_mean_plus_alpha_std_B: One value per batch dimension,
            log(mean + alpha * std)

    """

    batch_shape = noise_log_pdf_BC.shape[:-1]
    noise_count_C = torch.arange(noise_log_pdf_BC.shape[-1]).to(noise_log_pdf_BC.device)
    noise_count_BC = (noise_count_C
                      .expand(batch_shape + noise_count_C.shape)
                      + offset_noise_counts_B.unsqueeze(-1))

    log_mean_B = torch.logsumexp(
        noise_log_pdf_BC + noise_count_BC.log(),
        dim=-1,
    )
    log_std_B = 0.5 * torch.logsumexp(
        noise_log_pdf_BC
        + 2 * (noise_count_BC - log_mean_B.exp().unsqueeze(-1)).abs().log(),
        dim=-1,
    )

    log_mean_plus_alpha_std_B = torch.logsumexp(
        torch.cat([log_mean_B.unsqueeze(-1), np.log(alpha) + log_std_B.unsqueeze(-1)], dim=-1),
        dim=-1,
    )

    return log_mean_plus_alpha_std_B


def get_alpha_log_constraint_violation_given_beta(
        beta_B: torch.Tensor,
        log_pdf_noise_counts_BC: torch.Tensor,
        noise_count_BC: torch.Tensor,
        log_mu_plus_alpha_sigma_B: torch.Tensor) -> torch.Tensor:
    """Returns log constraint violation for the regularized posterior of p(x), which
    here is p(\omega) = p(x) e^{\beta x}, and we want
    E[\omega] = E[x] + \alpha * Std[x] = log_mu_plus_alpha_sigma_B.exp()

    NOTE: Binary search to find the root of this function can yield a value for beta_B.

    Args:
        beta_B: The parameter of the regularized posterior, with batch dimension
        log_pdf_noise_counts_BC: The probability density of noise counts, with batch
            and count dimensions
        noise_count_BC: Noise counts, with batch and count dimensions
        log_mu_plus_alpha_sigma_B: The constraint value to be satisfied, with batch dimension

    Returns:
        The amount by which the desired equality with log_mu_plus_alpha_sigma_B is violated,
            with batch dimension

    """

    log_numerator_B = torch.logsumexp(
        noise_count_BC.log() + log_pdf_noise_counts_BC + beta_B.unsqueeze(-1) * noise_count_BC,
        dim=-1,
    )
    log_denominator_B = torch.logsumexp(
        log_pdf_noise_counts_BC + beta_B.unsqueeze(-1) * noise_count_BC,
        dim=-1,
    )
    return log_numerator_B - log_denominator_B - log_mu_plus_alpha_sigma_B
