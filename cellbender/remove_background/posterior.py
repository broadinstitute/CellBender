"""Posterior generation and regularization."""

import pyro
import pyro.distributions as dist
import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd

import cellbender.remove_background.consts as consts
from cellbender.remove_background.model import calculate_mu, calculate_lambda
from cellbender.monitor import get_hardware_usage
from cellbender.remove_background.data.dataprep import DataLoader
from cellbender.remove_background.estimation import EstimationMethod, \
    MultipleChoiceKnapsack, Mean, MAP, apply_function_dense_chunks
from cellbender.remove_background.sparse_utils import dense_to_sparse_op_torch, \
    log_prob_sparse_to_dense, zero_out_csr_rows
from cellbender.remove_background.checkpoint import load_from_checkpoint, \
    unpack_tarball, make_tarball

from typing import Tuple, List, Dict, Optional, Union, Callable
from abc import ABC, abstractmethod
import logging
import argparse
import tempfile
import os


logger = logging.getLogger('cellbender')


def load_or_compute_posterior_and_save(dataset_obj: 'SingleCellRNACountsDataset',
                                       inferred_model: 'RemoveBackgroundPyroModel',
                                       args: argparse.Namespace) -> 'Posterior':
    """After inference, compute the full posterior noise count log probability
    distribution. Save it and make it part of the checkpoint file.

    NOTE: Loads posterior from checkpoint file if available.
    NOTE: Saves posterior as args.output_file + '_posterior.npz' and adds this
        file to the checkpoint tarball as well.

    Args:
        dataset_obj: Input data in the form of a SingleCellRNACountsDataset
            object.
        inferred_model: Model after inference is complete.
        args: Input command line parsed arguments.

    Returns:
        posterior: Posterior object with noise count log prob computed, as well
            as regularization if called for.

    """

    assert os.path.exists(args.input_checkpoint_tarball), \
        f'Checkpoint file {args.input_checkpoint_tarball} does not exist, ' \
        f'presumably because saving of the checkpoint file has been manually ' \
        f'interrupted. load_or_compute_posterior_and_save() will not work ' \
        f'properly without an existing checkpoint file. Please re-run and ' \
        f'allow a checkpoint file to be saved.'

    def _do_posterior_regularization(posterior: Posterior):

        # Optional posterior regularization.
        if args.posterior_regularization is not None:
            if args.posterior_regularization == 'PRq':
                posterior.regularize_posterior(
                    regularization=PRq,
                    alpha=args.prq_alpha,
                    device='cuda',
                )
            elif args.posterior_regularization == 'PRmu':
                posterior.regularize_posterior(
                    regularization=PRmu,
                    raw_count_matrix=dataset_obj.data['matrix'],
                    fpr=args.fpr[0],
                    per_gene=False,
                    device='cuda',
                )
            elif args.posterior_regularization == 'PRmu_gene':
                posterior.regularize_posterior(
                    regularization=PRmu,
                    raw_count_matrix=dataset_obj.data['matrix'],
                    fpr=args.fpr[0],
                    per_gene=True,
                    device='cuda',
                )
            else:
                raise ValueError(f'Got a posterior regularization input of '
                                 f'"{args.posterior_regularization}", which is not '
                                 f'allowed. Use ["PRq", "PRmu", "PRmu_gene"]')

        else:
            # Delete a pre-existing posterior regularization in case an old one was saved.
            posterior.clear_regularized_posterior()

    posterior = Posterior(
        dataset_obj=dataset_obj,
        vi_model=inferred_model,
        posterior_batch_size=args.posterior_batch_size,
        debug=args.debug,
    )
    ckpt_posterior = load_from_checkpoint(tarball_name=args.input_checkpoint_tarball,
                                          filebase=args.checkpoint_filename,
                                          to_load='posterior')
    if os.path.exists(ckpt_posterior.get('posterior_file', 'does_not_exist')):
        # Load posterior if it was saved in the checkpoint.
        posterior.load(file=ckpt_posterior['posterior_file'])
        _do_posterior_regularization(posterior)
    else:
        # Compute posterior.
        logger.info('Posterior not currently included in checkpoint.')
        posterior.cell_noise_count_posterior_coo()
        _do_posterior_regularization(posterior)

        # Save posterior and add it to checkpoint tarball.
        saved = posterior.save(file=args.output_file[:-3] + '_posterior.npz')
        success = False
        if saved:
            with tempfile.TemporaryDirectory() as tmp_dir:
                unpacked = unpack_tarball(tarball_name=args.input_checkpoint_tarball,
                                          directory=tmp_dir)
                if unpacked:
                    posterior.save(file=os.path.join(tmp_dir, 'posterior.npz'), verbose=False)
                    all_ckpt_files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
                                      if os.path.isfile(os.path.join(tmp_dir, f))]
                    success = make_tarball(files=all_ckpt_files,
                                           tarball_name=args.input_checkpoint_tarball)
        if success:
            logger.info('Added posterior object to checkpoint file.')
        else:
            logger.warning('Failed to add posterior object to checkpoint file.')

    return posterior


class Posterior:
    """Posterior handles posteriors on latent variables and denoised counts.

    Args:
        dataset_obj: Dataset object.
        vi_model: Trained RemoveBackgroundPyroModel.
        posterior_batch_size: Number of barcodes in a minibatch, used to
            calculate posterior probabilities (memory hungry).
        counts_dtype: Data type of posterior count matrix.  Can be one of
            [np.uint32, np.float]
        float_threshold: For floating point count matrices, counts below
            this threshold will be set to zero, for the purposes of constructing
            a sparse matrix.  Unused if counts_dtype is np.uint32
        debug: True to print debugging messages (involves extra compute)

    Properties:
        full_noise_count_posterior_csr: The posterior noise log probability
            distribution, as a sparse matrix.
        latents_map: MAP estimate of latent variables

    Examples:

        posterior = Posterior()

    """

    def __init__(self,
                 dataset_obj: 'SingleCellRNACountsDataset',  # Dataset
                 vi_model: Optional['RemoveBackgroundPyroModel'],
                 posterior_batch_size: int = 128,
                 counts_dtype: np.dtype = np.uint32,
                 float_threshold: Optional[float] = 0.5,
                 debug: bool = False):
        self.dataset_obj = dataset_obj
        self.vi_model = vi_model
        self.vi_model.eval()
        self.use_cuda = (torch.cuda.is_available() if vi_model is None
                         else vi_model.use_cuda)
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.analyzed_gene_inds = (None if (dataset_obj is None)
                                   else dataset_obj.analyzed_gene_inds)
        self.count_matrix_shape = (None if (dataset_obj is None)
                                   else dataset_obj.data['matrix'].shape)
        self.barcode_inds = (None if (dataset_obj is None)
                             else np.arange(0, self.count_matrix_shape[0]))
        self.dtype = counts_dtype
        self.debug = debug
        self.float_threshold = float_threshold
        self.posterior_batch_size = posterior_batch_size
        self._noise_count_posterior_coo = None
        self._noise_count_posterior_kwargs = None
        self._noise_count_posterior_coo_offsets = None
        self._noise_count_regularized_posterior_coo = None
        self._noise_count_regularized_posterior_kwargs = None
        self._latents = None
        self.index_converter = IndexConverter(
            total_n_cells=dataset_obj.data['matrix'].shape[0],
            total_n_genes=dataset_obj.data['matrix'].shape[1],
        )

    def save(self, file: str, verbose: bool = True) -> bool:
        """Save the full posterior in compressed array .npz format."""

        if self._noise_count_posterior_coo is None:
            self.cell_noise_count_posterior_coo()

        d = {'posterior_noise_count_log_prob_coo': self._noise_count_posterior_coo,
             'posterior_noise_count_coo_offsets': self._noise_count_posterior_coo_offsets,
             'posterior_kwargs': self._noise_count_posterior_kwargs,
             'regularized_posterior_noise_count_log_prob_coo': self._noise_count_regularized_posterior_coo,
             'regularized_posterior_kwargs': self._noise_count_regularized_posterior_kwargs,
             'latents': self.latents_map}

        # TODO: this can choke (out of memory) on very large datasets... not sure why
        # TODO: is it the compression?  should I save uncompressed and then compress myself?
        try:
            np.savez_compressed(file=file, **d)
            if verbose:
                logger.info(f'Saved posterior as {file}')
            return True
        except MemoryError:
            logger.warning('Attempting to save the posterior as a compressed NPZ file '
                           'resulted in an out-of-memory error. This is a known issue '
                           'but please report it as a github issue.')
            return False

    def load(self, file: str) -> bool:
        """Load a saved posterior in compressed array .npz format."""

        d = np.load(file=file, allow_pickle=True)
        self._noise_count_posterior_coo = d['posterior_noise_count_log_prob_coo'].item()
        self._noise_count_posterior_coo_offsets = d['posterior_noise_count_coo_offsets'].item()
        self._noise_count_posterior_kwargs = d['posterior_kwargs'].item()
        self._noise_count_regularized_posterior_coo = d['regularized_posterior_noise_count_log_prob_coo'].item()
        self._noise_count_regularized_posterior_kwargs = d['regularized_posterior_kwargs'].item()
        self._latents = d['latents'].item()
        logger.info(f'Loaded pre-computed posterior from {file}')
        return True

    def compute_denoised_counts(self,
                                estimator_constructor: EstimationMethod,
                                **kwargs) -> sp.csc_matrix:
        """Probably the most important method: computation of the clean output count matrix.

        Args:
            estimator_constructor: A noise count estimator class derived from
                the EstimationMethod base class, and implementing the
                .estimate_noise() method, which creates a point estimate of
                noise. Pass in the constructor, not an object.
            **kwargs: Keyword arguments for estimator_constructor().estimate_noise()

        Returns:
            denoised_counts: Denoised output CSC sparse matrix (CSC for saving)

        """

        # Only compute using defaults if the cache is empty.
        if self._noise_count_regularized_posterior_coo is not None:
            # Priority is taken by a regularized posterior, since presumably
            # the user computed it for a reason.
            logger.debug('Using regularized posterior to compute denoised counts')
            logger.debug(self._noise_count_regularized_posterior_kwargs)
            posterior_coo = self._noise_count_regularized_posterior_coo
        else:
            # Use exact posterior if a regularized version is not computed.
            posterior_coo = (self._noise_count_posterior_coo
                             if (self._noise_count_posterior_coo is not None)
                             else self.cell_noise_count_posterior_coo())

        # Instantiate Estimator object.
        estimator = estimator_constructor(index_converter=self.index_converter)

        # Compute point estimate of noise in cells.
        noise_csr = estimator.estimate_noise(
            noise_log_prob_coo=posterior_coo,
            noise_offsets=self._noise_count_posterior_coo_offsets,
            **kwargs,
        )

        # Subtract cell noise from observed cell counts.
        count_matrix = self.dataset_obj.data['matrix']  # all barcodes
        cell_inds = self.dataset_obj.analyzed_barcode_inds[self.latents_map['p']
                                                           > consts.CELL_PROB_CUTOFF]
        non_cell_row_logic = np.array([i not in cell_inds for i in range(count_matrix.shape[0])])
        cell_counts = zero_out_csr_rows(csr=count_matrix, row_logic=non_cell_row_logic)
        denoised_counts = cell_counts - noise_csr

        return denoised_counts.tocsc()

    def regularize_posterior(self,
                             regularization: 'PosteriorRegularization',
                             **kwargs) -> sp.coo_matrix:
        """Do posterior regularization. This modifies self._noise_count_regularized_posterior_coo
        in place, and returns it.

        Args:
            regularization: A particular PosteriorRegularization ['PRmu', 'PRq']
            **kwargs: Arguments passed to the PosteriorRegularization's
                .regularize() method

        Returns:
            Returns the regularized posterior, which is also stored in
                self._noise_count_regularized_posterior_coo

        """

        # Check if this posterior regularization has already been computed.
        currently_cached = False if self._noise_count_regularized_posterior_kwargs is None else True
        if currently_cached:
            # Check if it's the right thing.
            for k, v in self._noise_count_regularized_posterior_kwargs.items():
                if k == 'method':
                    if v != regularization.name():
                        currently_cached = False
                        break
                elif k not in kwargs.keys():
                    currently_cached = False
                    break
                elif kwargs[k] != v:
                    currently_cached = False
                    break
        if currently_cached:
            # What's been requested is what's cached.
            logger.debug('Regularized posterior is already cached')
            return self._noise_count_regularized_posterior_coo

        # Compute the regularized posterior.
        self._noise_count_regularized_posterior_coo = regularization.regularize(
            noise_count_posterior_coo=self._noise_count_posterior_coo,
            noise_offsets=self._noise_count_posterior_coo_offsets,
            index_converter=self.index_converter,
            **kwargs,
        )
        kwargs.update({'method': regularization.name()})
        kwargs.pop('raw_count_matrix', None)  # do not store a copy here
        self._noise_count_regularized_posterior_kwargs = kwargs
        logger.debug('Updated posterior after performing regularization')
        return self._noise_count_regularized_posterior_coo

    def clear_regularized_posterior(self):
        """Remove the saved regularized posterior (so that compute_denoised_counts()
        will not default to using it).
        """
        self._noise_count_regularized_posterior_coo = None
        self._noise_count_regularized_posterior_kwargs = None

    def cell_noise_count_posterior_coo(self, **kwargs) -> sp.coo_matrix:
        """Compute the full-blown posterior on noise counts for all cells,
        and store it in COO sparse format on CPU, and cache in
        self._noise_count_posterior_csr

        NOTE: This is the main entrypoint for this class.

        Args:
            **kwargs: Passed to _get_cell_noise_count_posterior_coo()

        Returns:
            self._noise_count_posterior_coo: This sparse COO object contains all
                the information about the posterior noise count distribution,
                but it is a bit complicated. The data per entry (m, c) are
                stored in COO format. The rows "m" represent a combined
                cell-and-gene index, with a one-to-one mapping from m to
                (n, g). The columns "c" represent noise count values. Values
                are the log probabilities of a noise count value. A smaller
                matrix can be constructed by increasing the threshold
                smallest_log_probability.
        """

        if ((self._noise_count_posterior_coo is None)
                or (kwargs != self._noise_count_posterior_kwargs)):
            logger.debug('Running _get_cell_noise_count_posterior_coo() to compute posterior')
            self._get_cell_noise_count_posterior_coo(**kwargs)
            self._noise_count_posterior_kwargs = kwargs

        return self._noise_count_posterior_coo

    @property
    def latents_map(self) -> Dict[str, np.ndarray]:
        if self._latents is None:
            self._get_latents_map()
        return self._latents

    @torch.no_grad()
    def _get_cell_noise_count_posterior_coo(
            self,
            n_samples: int = 20,
            y_map: bool = True,
            n_counts_max: int = 20,
            smallest_log_probability: float = -10.) -> sp.coo_matrix:  # TODO: default -7 ?
        """Compute the full-blown posterior on noise counts for all cells,
        and store log probability in COO sparse format on CPU.

        Args:
            n_samples: Number of samples to use to compute the posterior log
                probability distribution. Samples have high variance, so it is
                important to use at least 20. However, they are expensive.
            y_map: Use the MAP value for y (cell / no cell) when sampling, to
                avoid samples with a cell and samples without a cell.
            n_counts_max: Maximum number of noise counts.
            smallest_log_probability: Do not store log prob values smaller than
                this -- they get set to zero (saves space)

        Returns:
            noise_count_posterior_coo: This sparse CSR object contains all
                the information about the posterior noise count distribution,
                but it is a bit complicated. The data per entry (m, c) are
                stored in COO format. The rows "m" represent a combined
                cell-and-gene index, and there is a one-to-one mapping from m to
                (n, g). The columns "c" represent noise count values. Values
                are the log probabilities of a noise count value. A smaller
                matrix can be constructed by increasing the threshold
                smallest_log_probability.

        """

        logger.debug('Computing full posterior noise counts')

        # Compute posterior in mini-batches.
        torch.cuda.empty_cache()

        # Dataloader for cells only.
        analyzed_bcs_only = True
        count_matrix = self.dataset_obj.get_count_matrix()  # analyzed barcodes
        cell_logic = (self.latents_map['p'] > consts.CELL_PROB_CUTOFF)
        dataloader_index_to_analyzed_bc_index = np.where(cell_logic)[0]
        cell_data_loader = DataLoader(
            count_matrix[cell_logic],
            empty_drop_dataset=None,
            batch_size=self.posterior_batch_size,
            fraction_empties=0.,
            shuffle=False,
            use_cuda=self.use_cuda,
        )

        bcs = []  # barcode index
        genes = []  # gene index
        c = []  # noise count value
        c_offset = []  # noise count offsets from zero
        log_probs = []
        ind = 0

        logger.info('Computing posterior noise count probabilities in mini-batches...')

        for data in cell_data_loader:

            if self.debug:
                logger.debug(f'Posterior minibatch starting with droplet {ind}')
                logger.debug('\n' + get_hardware_usage(use_cuda=self.use_cuda))

            # Compute noise count probabilities.
            noise_log_pdf_NGC, noise_count_offset_NG = self.noise_log_pdf(
                data=data,
                n_samples=n_samples,
                y_map=y_map,
                n_counts_max=n_counts_max,
            )

            # Compute a tensor to indicate sparsity.
            # First we want data = 0 to be all zeros
            # We also want anything below the threshold to be a zero
            tensor_for_nonzeros = noise_log_pdf_NGC.clone().exp()  # probability
            tensor_for_nonzeros.data[data == 0, :] = 0.  # remove data = 0
            tensor_for_nonzeros.data[noise_log_pdf_NGC < smallest_log_probability] = 0.

            # Convert to sparse format using "m" indices.
            bcs_i_chunk, genes_i_analyzed, c_i, log_prob_i = dense_to_sparse_op_torch(
                noise_log_pdf_NGC,
                tensor_for_nonzeros=tensor_for_nonzeros,
            )

            # Get the original gene index from gene index in the trimmed dataset.
            genes_i = self.analyzed_gene_inds[genes_i_analyzed]

            # Barcode index in the dataloader.
            bcs_i = bcs_i_chunk + ind

            # Obtain the real barcode index since we only use cells.
            bcs_i = dataloader_index_to_analyzed_bc_index[bcs_i]

            # Translate chunk barcode inds to overall inds.
            if analyzed_bcs_only:
                bcs_i = self.dataset_obj.analyzed_barcode_inds[bcs_i]
            else:
                bcs_i = self.barcode_inds[bcs_i]

            # Add sparse matrix values to lists.
            try:
                bcs.extend(bcs_i.tolist())
                genes.extend(genes_i.tolist())
                c.extend(c_i.tolist())
                log_probs.extend(log_prob_i.tolist())
                c_offset.extend(noise_count_offset_NG[bcs_i_chunk, genes_i_analyzed]
                                .detach().cpu().numpy())
            except TypeError as e:
                # edge case of a single value
                bcs.append(bcs_i)
                genes.append(genes_i)
                c.append(c_i)
                log_probs.append(log_prob_i)
                c_offset.append(noise_count_offset_NG[bcs_i_chunk, genes_i_analyzed]
                                .detach().cpu().numpy())

            # Increment barcode index counter.
            ind += data.shape[0]  # Same as data_loader.batch_size

        # Convert the lists to numpy arrays.
        log_probs = np.array(log_probs, dtype=np.float)
        c = np.array(c, dtype=np.uint32)
        barcodes = np.array(bcs, dtype=np.uint64)  # uint32 is too small!
        genes = np.array(genes, dtype=np.uint64)  # use same as above for IndexConverter
        noise_count_offsets = np.array(c_offset, dtype=np.uint32)

        # Translate (barcode, gene) inds to 'm' format index.
        m = self.index_converter.get_m_indices(cell_inds=barcodes, gene_inds=genes)

        # Put the counts into a sparse csr_matrix.
        self._noise_count_posterior_coo = sp.coo_matrix(
            (log_probs, (m, c)),
            shape=[np.prod(self.count_matrix_shape), n_counts_max],
        )
        self._noise_count_posterior_coo_offsets = dict(zip(m, noise_count_offsets))
        return self._noise_count_posterior_coo

    @torch.no_grad()
    def sample(self, data, lambda_multiplier=1., y_map: bool = False) -> torch.Tensor:
        """Draw a single posterior sample for the count matrix conditioned on data

        Args:
            data: Count matrix (slice: some droplets, all genes)
            lambda_multiplier: BasePosterior regularization multiplier
            y_map: True to enforce the use of the MAP estimate of y, cell or
                no cell. Useful in the case where many samples are collected,
                since typically in those cases it is confusing to have samples
                where a droplet is both cell-containing and empty.

        Returns:
            denoised_output_count_matrix: Single sample of the denoised output
                count matrix, sampling all stochastic latent variables in the model.

        """

        # Sample all the latent variables in the model and get mu, lambda, alpha.
        mu_sample, lambda_sample, alpha_sample = self.sample_mu_lambda_alpha(data, y_map=y_map)

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
    def map_denoised_counts_from_sampled_latents(self,
                                                 data,
                                                 n_samples: int,
                                                 lambda_multiplier: float = 1.,
                                                 y_map: bool = False) -> torch.Tensor:
        """Draw posterior samples for all stochastic latent variables in the model
         and use those values to compute a MAP estimate of the denoised count
         matrix conditioned on data.

        Args:
            data: Count matrix (slice: some droplets, all genes)
            lambda_multiplier: BasePosterior regularization multiplier
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
                      y_map: bool = True,
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
            lambda_multiplier: BasePosterior regularization multiplier
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
            mu_sample, lambda_sample, alpha_sample = self.sample_mu_lambda_alpha(data, y_map=y_map)

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
    def sample_mu_lambda_alpha(self,
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

        # If using MAP for y (so that you never get samples of cell and no cell),
        # then intervene and replace a sampled y with the MAP
        if y_map:
            guide_trace.nodes['y']['value'] = (
                    guide_trace.nodes['p_passback']['value'] > 0
            ).clone().detach()

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
        """Compute the log prob of noise counts [n, g, c] given mu, lambda, alpha, and the data.

        NOTE: this is un-normalized log probability

        Args:
            data: Dense tensor minibatch of cell by gene count data.
            mu_est: Dense tensor of Negative Binomial means for true counts.
            lambda_est: Dense tensor of Poisson rate params for noise counts.
            alpha_est: Dense tensor of Dirichlet concentration params that
                inform the overdispersion of the Negative Binomial.  None will
                use an all-Poisson model
            n_counts_max: Size of noise count dimension c
            debug: True will go slow and check for NaNs and zero-probability entries

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
                raise AssertionError('There is at least one log_prob_tensor[n, g, :] '
                                     'that has all-zero probability')

        return log_prob_tensor, poisson_values_low

    @torch.no_grad()
    def _get_latents_map(self):
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

            epsilon[ind:(ind + data.shape[0])] = \
                dist.Gamma(enc['epsilon'] * self.vi_model.epsilon_prior,
                           self.vi_model.epsilon_prior).mean.detach().cpu().numpy()

        self._latents = {'z': z,
                         'd': d,
                         'p': p,
                         'phi_loc_scale': [phi_loc.item(), phi_scale.item()],
                         'epsilon': epsilon}

    @torch.no_grad()
    def _get_mu_alpha_lambda_map(self,
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
        mu_map = calculate_mu(
            epsilon=epsilon,
            d_cell=d_cell,
            chi=chi_map,
            y=y,
            rho=rho,
        )
        lambda_map = calculate_lambda(
            epsilon=epsilon,
            chi_ambient=chi_ambient,
            d_empty=d_empty,
            y=y,
            d_cell=d_cell,
            rho=rho,
            chi_bar=self.vi_model.avg_gene_expression,
        )

        return {'mu': mu_map, 'lam': lambda_map, 'alpha': alpha_map}


class PosteriorRegularization(ABC):

    def __init__(self):
        super(PosteriorRegularization, self).__init__()

    @staticmethod
    @abstractmethod
    def name():
        """Short name of this regularization method"""
        pass

    @staticmethod
    @abstractmethod
    def regularize(noise_count_posterior_coo: sp.coo_matrix,
                   noise_offsets: Dict[int, int],
                   **kwargs) -> sp.coo_matrix:
        """Perform posterior regularization"""
        pass


class PRq(PosteriorRegularization):
    """Approximate noise CDF quantile targeting:

    E_reg[noise_counts] >= E[noise_counts] + \alpha * Std[noise_counts]

    """

    @staticmethod
    def name():
        return 'PRq'

    @staticmethod
    def _log_mean_plus_alpha_std(log_prob: torch.Tensor, alpha: float):
        c = torch.arange(log_prob.shape[1]).float().to(log_prob.device).unsqueeze(0)
        prob = log_prob.exp()
        mean = (c * prob).sum(dim=-1)
        std = (((c - mean.unsqueeze(-1)).pow(2) * prob).sum(dim=-1)).sqrt()
        return (mean + alpha * std).log()

    @staticmethod
    def _compute_log_target_dict(noise_count_posterior_coo: sp.coo_matrix,
                                 alpha: float) -> Dict[int, float]:
        """Given the noise count posterior, return log(mean + alpha * std)
        for each 'm' index

        NOTE: noise_log_pdf_BC should be normalized

        Args:
            noise_count_posterior_coo: The noise count posterior data structure
            alpha: The tunable parameter of mean-targeting posterior
                regularization. The output distribution has a mean which is
                input_mean + alpha * input_std (if possible)

        Returns:
            log_mean_plus_alpha_std: Dict keyed by 'm', where values are
                log(mean + alpha * std)

        """
        result = apply_function_dense_chunks(noise_log_prob_coo=noise_count_posterior_coo,
                                             fun=PRq._log_mean_plus_alpha_std,
                                             alpha=alpha)
        return dict(zip(result['m'], result['result']))

    @staticmethod
    def _get_alpha_log_constraint_violation_given_beta(
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

    @staticmethod
    def _chunked_compute_regularized_posterior(
            noise_count_posterior_coo: sp.coo_matrix,
            noise_offsets: Dict[int, int],
            log_constraint_violation_fcn: Callable[[torch.Tensor, torch.Tensor,
                                                    torch.Tensor, torch.Tensor], torch.Tensor],
            log_target_M: torch.Tensor,
            target_tolerance: float = 0.001,
            device: str = 'cpu',
            n_chunks: Optional[int] = None,
    ) -> sp.coo_matrix:
        """Go through posterior in chunks and compute regularized posterior,
        using the defined targets"""

        # Compute using dense chunks, chunked on m-index.
        if n_chunks is None:
            dense_size_gb = (len(np.unique(noise_count_posterior_coo.row))
                             * (noise_count_posterior_coo.shape[1]
                                + np.array(list(noise_offsets.values())).max())) * 4 / 1e9  # GB
            n_chunks = max(1, int(dense_size_gb // 1))  # approx 1 GB each

        # Make the sparse matrix compact in the sense that it should use contiguous row values.
        unique_rows, densifiable_coo_rows = np.unique(noise_count_posterior_coo.row, return_inverse=True)
        densifiable_csr = sp.csr_matrix((noise_count_posterior_coo.data,
                                         (densifiable_coo_rows, noise_count_posterior_coo.col)),
                                        shape=[len(unique_rows), noise_count_posterior_coo.shape[1]])
        chunk_size = int(np.ceil(densifiable_csr.shape[0] / n_chunks))

        m = []
        c = []
        log_prob_reg = []

        for i in range(n_chunks):
            # B index here represents a batch: the re-defined m-index
            log_pdf_noise_counts_BC = torch.tensor(
                log_prob_sparse_to_dense(densifiable_csr[(i * chunk_size):((i + 1) * chunk_size)])
            ).to(device)
            noise_count_BC = (torch.arange(log_pdf_noise_counts_BC.shape[1])
                              .to(log_pdf_noise_counts_BC.device)
                              .unsqueeze(0)
                              .expand(log_pdf_noise_counts_BC.shape))
            m_indices_for_chunk = unique_rows[(i * chunk_size):((i + 1) * chunk_size)]
            noise_count_BC = noise_count_BC + (torch.tensor([noise_offsets[m]
                                                             for m in m_indices_for_chunk],
                                                            dtype=torch.float)
                                               .unsqueeze(-1)
                                               .to(device))

            # Parallel binary search for beta for each entry of count matrix
            beta_B = torch_binary_search(
                evaluate_outcome_given_value=lambda x:
                log_constraint_violation_fcn(
                    beta_B=x,
                    log_pdf_noise_counts_BC=log_pdf_noise_counts_BC,
                    noise_count_BC=noise_count_BC,
                    log_mu_plus_alpha_sigma_B=log_target_M[(i * chunk_size):((i + 1) * chunk_size)],
                ),
                target_outcome=torch.zeros(noise_count_BC.shape[0]).to(device),
                init_range=(torch.tensor([-100., 100.])
                            .to(device)
                            .unsqueeze(0)
                            .expand((noise_count_BC.shape[0],) + (2,))),
                target_tolerance=target_tolerance,
                max_iterations=100,
            )

            # Generate regularized posteriors.
            log_pdf_reg_BC = log_pdf_noise_counts_BC + beta_B.unsqueeze(-1) * noise_count_BC
            log_pdf_reg_BC = log_pdf_reg_BC - torch.logsumexp(log_pdf_reg_BC, -1, keepdims=True)

            # Store sparse COO values in lists.
            tensor_for_nonzeros = log_pdf_reg_BC.clone().exp()  # probability
            m_i, c_i, log_prob_reg_i = dense_to_sparse_op_torch(
                log_pdf_reg_BC,
                tensor_for_nonzeros=tensor_for_nonzeros,
            )
            m_i = np.array([m_indices_for_chunk[j] for j in m_i])  # chunk m to actual m

            # Add sparse matrix values to lists.
            try:
                m.extend(m_i.tolist())
                c.extend(c_i.tolist())
                log_prob_reg.extend(log_prob_reg_i.tolist())
            except TypeError as e:
                # edge case of a single value
                m.append(m_i)
                c.append(c_i)
                log_prob_reg.append(log_prob_reg_i)

        reg_noise_count_posterior_coo = sp.coo_matrix((log_prob_reg, (m, c)),
                                                      shape=noise_count_posterior_coo.shape)
        return reg_noise_count_posterior_coo

    @staticmethod
    @torch.no_grad()
    def regularize(noise_count_posterior_coo: sp.coo_matrix,
                   noise_offsets: Dict[int, int],
                   alpha: float,
                   device: str = 'cuda',
                   target_tolerance: float = 0.001,
                   n_chunks: Optional[int] = None,
                   **kwargs) -> sp.coo_matrix:
        """Perform posterior regularization using approximate quantile-targeting.

        Args:
            noise_count_posterior_coo: Noise count posterior log prob COO
            noise_offsets: Offset noise counts per 'm' index
            alpha: The tunable parameter of quantile-targeting posterior
                regularization. The output distribution has a mean which is
                input_mean + alpha * input_std (if possible)
            device: Where to perform tensor operations: ['cuda', 'cpu']
            target_tolerance: Tolerance when searching using binary search
            n_chunks: For testing only - the number of chunks used to
                compute the result when iterating over the posterior

        Results:
            reg_noise_count_posterior_coo: The regularized noise count
                posterior data structure

        """
        logger.info(f'Regularizing noise count posterior using approximate quantile-targeting with alpha={alpha}')

        # Compute the expectation for the mean post-regularization.
        log_target_dict = PRq._compute_log_target_dict(
            noise_count_posterior_coo=noise_count_posterior_coo,
            alpha=alpha,
        )
        log_target_M = torch.tensor(list(log_target_dict.values())).to(device)

        reg_noise_count_posterior_coo = PRq._chunked_compute_regularized_posterior(
            noise_count_posterior_coo=noise_count_posterior_coo,
            noise_offsets=noise_offsets,
            log_target_M=log_target_M,
            log_constraint_violation_fcn=PRq._get_alpha_log_constraint_violation_given_beta,
            device=device,
            target_tolerance=target_tolerance,
            n_chunks=n_chunks,
        )

        return reg_noise_count_posterior_coo


class PRmu(PosteriorRegularization):
    """Approximate noise mean targeting:

    Overall (default):
        E_reg[\sum_{n} \sum_{g} noise_counts_{ng}] =
            E[\sum_{n} \sum_{g} noise_counts_{ng}] + nFPR * \sum_{n} \sum_{g} raw_counts_{ng}

    Per-gene:
        E_reg[\sum_{n} noise_counts_{ng}] =
            E[\sum_{n} noise_counts_{ng}] + nFPR * \sum_{n} raw_counts_{ng}

    """

    @staticmethod
    def name():
        return 'PRmu'

    @staticmethod
    def _binary_search_for_posterior_regularization_factor(
            noise_count_posterior_coo: sp.coo_matrix,
            noise_offsets: Dict[int, int],
            index_converter: 'IndexConverter',
            target_removal: torch.Tensor,
            shape: int,
            target_tolerance: float = 100,
            max_iterations: int = 20,
            device: str = 'cpu',
    ) -> torch.Tensor:
        """Go through posterior and compute regularization factor(s),
        using the defined targets"""

        def summarize_map_noise_counts(x: torch.Tensor,
                                       per_gene: bool) -> torch.Tensor:
            """Given a (subset of the) noise posterior, compute the MAP estimate
            and summarize it either as the overall sum or per-gene.
            """

            # Regularize posterior.
            regularized_noise_posterior_coo = PRmu._chunked_compute_regularized_posterior(
                noise_count_posterior_coo=noise_count_posterior_coo,
                noise_offsets=noise_offsets,
                index_converter=index_converter,
                beta=x,
                device=device,
            )

            # Compute MAP.
            estimator = MAP(index_converter=index_converter)
            map_noise_csr = estimator.estimate_noise(
                noise_log_prob_coo=regularized_noise_posterior_coo,
                noise_offsets=noise_offsets,
                device=device,
            )

            # Summarize removal.
            if per_gene:
                noise_counts = np.array(map_noise_csr.sum(axis=0)).squeeze()
            else:
                noise_counts = map_noise_csr.sum()

            return torch.tensor(noise_counts).to(device)

        # Perform binary search for beta.
        per_gene = False
        if target_removal.dim() > 0:
            if len(target_removal) > 1:
                per_gene = True

        beta = torch_binary_search(
            evaluate_outcome_given_value=lambda x:
            summarize_map_noise_counts(x=x, per_gene=per_gene),
            target_outcome=target_removal,
            init_range=(torch.tensor([-100., 200.])
                        .to(device)
                        .unsqueeze(0)
                        .expand((shape,) + (2,))),
            target_tolerance=target_tolerance,
            max_iterations=max_iterations,
            debug=True,
        )
        return beta

    @staticmethod
    def _chunked_compute_regularized_posterior(
            noise_count_posterior_coo: sp.coo_matrix,
            noise_offsets: Dict[int, int],
            index_converter: 'IndexConverter',
            beta: torch.Tensor,
            device: str = 'cpu',
            n_chunks: Optional[int] = None,
    ) -> sp.coo_matrix:
        """Go through posterior in chunks and compute regularized posterior,
        using the defined targets"""

        # Compute using dense chunks, chunked on m-index.
        if n_chunks is None:
            dense_size_gb = (len(np.unique(noise_count_posterior_coo.row))
                             * (noise_count_posterior_coo.shape[1]
                                + np.array(list(noise_offsets.values())).max())) * 4 / 1e9  # GB
            n_chunks = max(1, int(dense_size_gb // 1))  # approx 1 GB each

        # Make the sparse matrix compact in the sense that it should use contiguous row values.
        unique_rows, densifiable_coo_rows = np.unique(noise_count_posterior_coo.row, return_inverse=True)
        densifiable_csr = sp.csr_matrix((noise_count_posterior_coo.data,
                                         (densifiable_coo_rows, noise_count_posterior_coo.col)),
                                        shape=[len(unique_rows), noise_count_posterior_coo.shape[1]])
        chunk_size = int(np.ceil(densifiable_csr.shape[0] / n_chunks))

        m = []
        c = []
        log_prob_reg = []

        for i in range(n_chunks):
            # B index here represents a batch: the re-defined m-index
            log_pdf_noise_counts_BC = torch.tensor(
                log_prob_sparse_to_dense(densifiable_csr[(i * chunk_size):((i + 1) * chunk_size)])
            ).to(device)
            noise_count_BC = (torch.arange(log_pdf_noise_counts_BC.shape[1])
                              .to(log_pdf_noise_counts_BC.device)
                              .unsqueeze(0)
                              .expand(log_pdf_noise_counts_BC.shape))
            m_indices_for_chunk = unique_rows[(i * chunk_size):((i + 1) * chunk_size)]
            noise_count_BC = noise_count_BC + (torch.tensor([noise_offsets[m]
                                                             for m in m_indices_for_chunk],
                                                            dtype=torch.float)
                                               .unsqueeze(-1)
                                               .to(device))

            # Get beta for this chunk.
            if len(beta) == 1:
                # posterior regularization factor is a single scalar
                beta_B = beta
            else:
                # per-gene mode
                n, g = index_converter.get_ng_indices(m_inds=m_indices_for_chunk)
                beta_B = torch.tensor([beta[gene] for gene in g])

            # Generate regularized posteriors.
            log_pdf_reg_BC = log_pdf_noise_counts_BC + beta_B.unsqueeze(-1) * noise_count_BC
            log_pdf_reg_BC = log_pdf_reg_BC - torch.logsumexp(log_pdf_reg_BC, -1, keepdims=True)

            # Store sparse COO values in lists.
            tensor_for_nonzeros = log_pdf_reg_BC.clone().exp()  # probability
            # tensor_for_nonzeros.data[data == 0, :] = 0.  # remove data = 0
            m_i, c_i, log_prob_reg_i = dense_to_sparse_op_torch(
                log_pdf_reg_BC,
                tensor_for_nonzeros=tensor_for_nonzeros,
            )
            m_i = np.array([m_indices_for_chunk[j] for j in m_i])  # chunk m to actual m

            # Add sparse matrix values to lists.
            try:
                m.extend(m_i.tolist())
                c.extend(c_i.tolist())
                log_prob_reg.extend(log_prob_reg_i.tolist())
            except TypeError as e:
                # edge case of a single value
                m.append(m_i)
                c.append(c_i)
                log_prob_reg.append(log_prob_reg_i)

        reg_noise_count_posterior_coo = sp.coo_matrix((log_prob_reg, (m, c)),
                                                      shape=noise_count_posterior_coo.shape)
        return reg_noise_count_posterior_coo

    @staticmethod
    def _subset_posterior_by_cells(noise_count_posterior_coo: sp.coo_matrix,
                                   index_converter: 'IndexConverter',
                                   n_cells: int) -> sp.coo_matrix:
        """Return a random slice of the full posterior with a specified number
        of cells.

        NOTE: Assumes that all the entries in noise_count_posterior_coo are for
        cell-containing droplets, and not empty droplets.

        Args:
            noise_count_posterior_coo: The noise count posterior data structure
            n_cells: The number of cells in the output subset

        Returns:
            subset_coo: Posterior for a random subset of cells, in COO format
        """

        # Choose cells that will be included.
        m = noise_count_posterior_coo.row
        n, g = index_converter.get_ng_indices(m_inds=m)
        unique_cell_inds = np.unique(n)
        if n_cells > len(unique_cell_inds):
            logger.debug(f'Limiting n_cells during PRmu regularizer binary search to {unique_cell_inds}')
            n_cells = len(unique_cell_inds)
        chosen_n_values = set(np.random.choice(unique_cell_inds, size=n_cells, replace=False))
        element_logic = [val in chosen_n_values for val in n]

        # Subset the posterior.
        data_subset = noise_count_posterior_coo.data[element_logic]
        row_subset = noise_count_posterior_coo.row[element_logic]
        col_subset = noise_count_posterior_coo.col[element_logic]
        return sp.coo_matrix((data_subset, (row_subset, col_subset)),
                             shape=noise_count_posterior_coo.shape)

    @staticmethod
    @torch.no_grad()
    def regularize(noise_count_posterior_coo: sp.coo_matrix,
                   noise_offsets: Dict[int, int],
                   index_converter: 'IndexConverter',
                   raw_count_matrix: sp.csr_matrix,
                   fpr: float,
                   per_gene: bool = False,
                   device: str = 'cuda',
                   target_tolerance: float = 0.5,
                   n_cells: int = 1000,
                   n_chunks: Optional[int] = None,
                   **kwargs) -> sp.coo_matrix:
        """Perform posterior regularization using mean-targeting.

        Args:
            noise_count_posterior_coo: Noise count posterior log prob COO
            noise_offsets: Offset noise counts per 'm' index
            index_converter: IndexConverter object from 'm' to (n, g) and back
            raw_count_matrix: The raw count matrix
            fpr: The tunable parameter of mean-targeting posterior
                regularization. The output, summed over cells, has a removed
                gene count distribution similar to what would be expected from
                the noise model, plus this nominal false positive rate.
            per_gene: True to find one posterior regularization factor for each
                gene, False to find one overall scalar (behavior of v0.2.0)
            device: Where to perform tensor operations: ['cuda', 'cpu']
            target_tolerance: Tolerance when searching using binary search.
                In units of counts, so this really should not be less than 0.5
            n_cells: To save time, use only this many cells to estimate removal
            n_chunks: For testing only - the number of chunks used to
                compute the result when iterating over the posterior

        Results:
            reg_noise_count_posterior_coo: The regularized noise count
                posterior data structure

        """

        logger.info('Regularizing noise count posterior using mean-targeting')

        # Use a subset of the data to find regularization factors, to reduce time.
        logger.debug(f'Subsetting posterior to {n_cells} cells for this computation')
        posterior_subset_coo = PRmu._subset_posterior_by_cells(
            noise_count_posterior_coo=noise_count_posterior_coo,
            index_converter=index_converter,
            n_cells=n_cells,
        )

        # Compute target removal for MAP estimate using regularized posterior.
        n, g = index_converter.get_ng_indices(m_inds=posterior_subset_coo.row)
        included_cells = set(np.unique(n))
        zero_out_logic = np.array([i not in included_cells
                                   for i in range(raw_count_matrix.shape[0])])
        # print(zero_out_logic)
        raw_count_csr_for_cells = zero_out_csr_rows(csr=raw_count_matrix,
                                                    row_logic=zero_out_logic)
        # print(raw_count_csr_for_cells)
        logger.debug('Computing target removal')
        target_fun = compute_mean_target_removal_as_function(
            noise_count_posterior_coo=posterior_subset_coo,
            noise_offsets=noise_offsets,
            index_converter=index_converter,
            raw_count_csr_for_cells=raw_count_csr_for_cells,
            n_cells=len(included_cells),
            device=device,
            per_gene=per_gene,
        )
        target_removal = target_fun(fpr) * len(included_cells)
        logger.debug(f'Target removal is {target_removal}')

        # Find the posterior regularization factor(s).
        if per_gene:
            logger.debug('Computing optimal posterior regularization factors for each gene')
            shape = index_converter.total_n_genes
        else:
            logger.debug('Computing optimal posterior regularization factor')
            shape = 1
        beta = PRmu._binary_search_for_posterior_regularization_factor(
            noise_count_posterior_coo=posterior_subset_coo,
            noise_offsets=noise_offsets,
            index_converter=index_converter,
            target_removal=target_removal,
            device=device,
            target_tolerance=target_tolerance,
            shape=shape,
        )
        logger.debug(f'Optimal posterior regularization factor\n{beta}')

        # Compute the posterior using the regularization factor(s).
        logger.debug('Computing full regularized posterior')
        regularized_noise_posterior_coo = PRmu._chunked_compute_regularized_posterior(
            noise_count_posterior_coo=noise_count_posterior_coo,
            noise_offsets=noise_offsets,
            index_converter=index_converter,
            beta=beta,
            device=device,
        )

        return regularized_noise_posterior_coo


class IndexConverter:

    def __init__(self, total_n_cells: int, total_n_genes: int):
        """Convert between (n, g) indices and flattened 'm' indices

        Args:
            total_n_cells: Total rows in the full sparse matrix
            total_n_genes: Total columns in the full sparse matrix

        """
        self.total_n_cells = total_n_cells
        self.total_n_genes = total_n_genes
        self.matrix_shape = (total_n_cells, total_n_genes)

    def __repr__(self):
        return (f'IndexConverter with'
                f'\n\ttotal_n_cells: {self.total_n_cells}'
                f'\n\ttotal_n_genes: {self.total_n_genes}'
                f'\n\tmatrix_shape: {self.matrix_shape}')

    def get_m_indices(self, cell_inds: np.ndarray, gene_inds: np.ndarray) -> np.ndarray:
        """Given arrays of cell indices and gene indices, suitable for a sparse matrix,
        convert them to 'm' index values.
        """
        if not ((cell_inds >= 0) & (cell_inds < self.total_n_cells)).all():
            raise ValueError(f'Requested cell_inds out of range: '
                             f'{cell_inds[(cell_inds < 0) | (cell_inds >= self.total_n_cells)]}')
        if not ((gene_inds >= 0) & (gene_inds < self.total_n_genes)).all():
            raise ValueError(f'Requested gene_inds out of range: '
                             f'{gene_inds[(gene_inds < 0) | (gene_inds >= self.total_n_genes)]}')
        return cell_inds * self.total_n_genes + gene_inds

    def get_ng_indices(self, m_inds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Given a list of 'm' index values, return two arrays: cell index values
        and gene index values, suitable for a sparse matrix.
        """
        if not ((m_inds >= 0) & (m_inds < self.total_n_cells * self.total_n_genes)).all():
            raise ValueError(f'Requested m_inds out of range: '
                             f'{m_inds[(m_inds < 0) | (m_inds >= self.total_n_cells * self.total_n_genes)]}')
        return np.divmod(m_inds, self.total_n_genes)


# @torch.no_grad()
# def get_noise_budget_per_gene_as_function(
#         posterior: Posterior,
#         how: str = 'fpr') -> Callable[[float], np.ndarray]:
#     """Compute the noise budget on a per-gene basis, returned as a function
#     that takes a target value and returns counts per gene in one cell.
#
#     Args:
#         posterior: Posterior object
#         how: For now this can only be 'fpr'
#
#     Returns:
#         expected_noise_count_fcn_per_cell_G: Function that, when called with a
#             certain nominal FPR value, returns an array of per-gene noise counts
#             expected in each cell. Not just analyzed genes: all genes.
#
#     """
#
#     logger.debug('Computing per-gene noise targets')

    # if 'chi_ambient' in pyro.get_param_store().keys():
    #     chi_ambient_G = pyro.param('chi_ambient').detach()
    # else:
    #     chi_ambient_G = 0.
    #
    # chi_bar_G = posterior.vi_model.avg_gene_expression
    #
    # if how == 'fpr':
    #
    #     # Expectation for counts in empty droplets.
    #     empty_droplet_mean_counts = dist.LogNormal(loc=pyro.param('d_empty_loc'),
    #                                                scale=pyro.param('d_empty_scale')).mean
    #     if posterior.vi_model.include_rho:
    #         swapping_fraction = dist.Beta(pyro.param('rho_alpha'), pyro.param('rho_beta')).mean
    #     else:
    #         swapping_fraction = 0.
    #     empty_droplet_mean_counts_G = empty_droplet_mean_counts * chi_ambient_G
    #
    #     data_loader = posterior.dataset_obj.get_dataloader(
    #         use_cuda=posterior.use_cuda,
    #         analyzed_bcs_only=True,
    #         batch_size=512,
    #         shuffle=False,
    #     )
    #
    #     # Keep a running sum over expected noise counts as we minibatch.
    #     expected_noise_counts_without_fpr_G = torch.zeros(len(posterior.dataset_obj.analyzed_gene_inds)).to(posterior.device)
    #     expected_real_counts_G = torch.zeros(len(posterior.dataset_obj.analyzed_gene_inds)).to(posterior.device)
    #     expected_cells = 0
    #
    #     for i, data in enumerate(data_loader):
    #         enc = posterior.vi_model.encoder(x=data,
    #                                          chi_ambient=chi_ambient_G,
    #                                          cell_prior_log=posterior.vi_model.d_cell_loc_prior)
    #         p_batch = enc['p_y'].sigmoid().detach()
    #         epsilon_batch = dist.Gamma(enc['epsilon'] * posterior.vi_model.epsilon_prior,
    #                                    posterior.vi_model.epsilon_prior).mean.detach()
    #         expected_ambient_counts_in_cells_G = (empty_droplet_mean_counts_G
    #                                               * epsilon_batch[p_batch > 0.5].sum())
    #         expected_swapping_counts_in_cells_G = (swapping_fraction * chi_bar_G *
    #                                                (data * epsilon_batch.unsqueeze(-1))[p_batch > 0.5].sum())
    #         expected_noise_counts_without_fpr_G = (expected_noise_counts_without_fpr_G
    #                                                + expected_ambient_counts_in_cells_G
    #                                                + expected_swapping_counts_in_cells_G)
    #         expected_real_counts_G = (expected_real_counts_G
    #                                   + torch.clamp(data[p_batch > 0.5].sum(dim=0)
    #                                                 - expected_noise_counts_without_fpr_G, min=0.))
    #         expected_cells = expected_cells + (p_batch > 0.5).sum()
    #
    #     def expected_noise_count_fcn_per_cell_G(target_fpr: float) -> np.ndarray:
    #         """The function which gets returned as the output"""
    #         target_per_analyzed_gene = ((expected_noise_counts_without_fpr_G
    #                                      + expected_real_counts_G * target_fpr)  # fpr addition
    #                                     / expected_cells)
    #         target_per_analyzed_gene = target_per_analyzed_gene.cpu().numpy()
    #         target_per_gene = np.zeros(posterior.dataset_obj.data['matrix'].shape[1])
    #         target_per_gene[posterior.dataset_obj.analyzed_gene_inds] = target_per_analyzed_gene
    #         return target_per_gene
    #
    # elif how == 'cdf':
    #
    #     raise NotImplementedError('TODO')
    #
    # else:
    #     raise NotImplementedError(f'No method {how} for get_noise_budget_per_gene_as_function()')
    #
    # # TODO: note - floor() ruined the game (per cell) for the optimal calcs
    # return expected_noise_count_fcn_per_cell_G


def compute_mean_target_removal_as_function(noise_count_posterior_coo: sp.coo_matrix,
                                            noise_offsets: Dict[int, int],
                                            index_converter: IndexConverter,
                                            raw_count_csr_for_cells: sp.csr_matrix,
                                            n_cells: int,
                                            device: str,
                                            per_gene: bool) -> Callable[[float], torch.Tensor]:
    """Given the noise count posterior, return a function that computes target
    removal (either overall or per-gene) as a function of FPR.

    NOTE: computes the value "per cell", i.e. dividing
    by the number of cells, so that total removal can be computed by
    multiplying this by the number of cells in question.

    Args:
        noise_count_posterior_coo: Noise count posterior log prob COO
        noise_offsets: Offset noise counts per 'm' index
        index_converter: IndexConverter object from 'm' to (n, g) and back
        raw_count_csr_for_cells: The input count matrix for only the cells
            included in the posterior
        n_cells: Number of cells included in the posterior, same number as in
            raw_count_csr_for_cells
        device: 'cpu' or 'cuda'
        per_gene: True to come up with one target per gene

    Returns:
        target_removal_scaled_per_cell: Noise count removal target

    """

    # TODO: s1.h5 with FPR 0.99 only removes 50% of signal

    # Compute the expected noise using mean summarization.
    estimator = Mean(index_converter=index_converter)
    mean_noise_csr = estimator.estimate_noise(
        noise_log_prob_coo=noise_count_posterior_coo,
        noise_offsets=noise_offsets,
        device=device,
    )
    logger.debug(f'Total counts in raw matrix for cells = {raw_count_csr_for_cells.sum()}')
    logger.debug(f'Total noise counts from mean noise estimator = {mean_noise_csr.sum()}')

    # Compute the target removal.
    approx_signal_csr = raw_count_csr_for_cells - mean_noise_csr
    logger.debug(f'Approximate signal has total counts = {approx_signal_csr.sum()}')
    logger.debug(f'Number of cells = {n_cells}')

    def _target_fun(fpr: float) -> torch.Tensor:
        """The function which gets returned"""
        if per_gene:
            target = np.array(mean_noise_csr.sum(axis=0)).squeeze()
            target = target + fpr * np.array(approx_signal_csr.sum(axis=0)).squeeze()
        else:
            target = mean_noise_csr.sum()
            target = target + fpr * approx_signal_csr.sum()

        # Return target scaled to be per-cell.
        return torch.tensor(target / n_cells).to(device)

    return _target_fun


# @numba.njit(fastmath=True)
# def binary_search(
#     evaluate_outcome_given_value: Callable[[float], float],
#     target_outcome: float,
#     init_range: List[float],
#     target_tolerance: Optional[float] = 0.001,
#     max_iterations: int = consts.POSTERIOR_REG_SEARCH_MAX_ITER,
# ) -> float:
#     """Perform a binary search, given a target and an evaluation function.
#     No python, for jit.
#
#     NOTE: evaluate_outcome_given_value(value) should increase monotonically
#     with the input value. It is assumed that
#     consts.POSTERIOR_REG_MIN < output_value < consts.POSTERIOR_REG_MAX.
#     If this is not the case, the algorithm will produce an output close to one
#     of those endpoints, and target_tolerance will not be achieved.
#     Moreover, output_value must be positive (due to how we search for limits).
#
#     Args:
#         evaluate_outcome_given_value: Numba jitted function that takes a value
#             as its input and produces the outcome, which is the target we are
#             trying to control. Should increase monotonically with value.
#         target_outcome: Desired outcome value from evaluate_outcome_given_value(value).
#         init_range: Search range as [low_limit, high_limit]
#         target_tolerance: Tolerated error in the target value.
#         max_iterations: A cutoff to ensure termination. Even if a tolerable
#             solution is not found, the algorithm will stop after this many
#             iterations and return the best answer so far.
#
#     Returns:
#         value: Result of binary search.
#
#     """
#
#     # assert (target_tolerance > 0), 'target_tolerance should be > 0.'
#     # assert len(init_range.shape) > 1, 'init_range must be at least two-dimensional ' \
#     #                                   '(last dimension contains lower and upper bounds)'
#     # assert init_range.shape[-1] == 2, 'Last dimension of init_range should be 2: low and high'
#
#     value_bracket = init_range
#
#     # Binary search algorithm.
#     for i in range(max_iterations):
#
#         # Current test value.
#         value = np.mean(value_bracket)
#
#         # Calculate an expected false positive rate for this lam_mult value.
#         outcome = evaluate_outcome_given_value(value)
#         residual = target_outcome - outcome
#
#         # Check on residual and update our bracket values.
#         stop_condition = (residual.abs() < target_tolerance)
#         if stop_condition:
#             break
#         else:
#             if outcome < target_outcome - target_tolerance:
#                 value_bracket[0] =
#             else:
#
#             value_bracket[..., 0] = torch.where(outcome < target_outcome - target_tolerance,
#                                                 value,
#                                                 value_bracket[..., 0])
#             value_bracket[..., 1] = torch.where(outcome > target_outcome + target_tolerance,
#                                                 value,
#                                                 value_bracket[..., 1])
#
#     # If we stopped due to iteration limit, take the average value.
#     if i == max_iterations:
#         value = value_bracket.mean(dim=-1)
#         logger.warning(f'Binary search target not achieved in {max_iterations} attempts. '
#                        f'Output is estimated to be {outcome.mean().item():.4f}')
#
#     # Warn if we railed out at the limits of the search
#     if debug:
#         if (value - target_tolerance <= init_range[..., 0]).sum() > 0:
#             logger.debug(f'{(value - target_tolerance <= init_range[..., 0]).sum()} '
#                          f'entries in the binary search hit the lower limit')
#             logger.debug(value[value - target_tolerance <= init_range[..., 0]])
#         if (value + target_tolerance >= init_range[..., 1]).sum() > 0:
#             logger.debug(f'{(value + target_tolerance >= init_range[..., 1]).sum()} '
#                          f'entries in the binary search hit the upper limit')
#             logger.debug(value[value + target_tolerance >= init_range[..., 1]])
#
#     return value


@torch.no_grad()
def torch_binary_search(
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
