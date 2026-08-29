"""Posterior generation and regularization."""

import argparse
import gc
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, cast

import pyarrow.parquet as pq

if TYPE_CHECKING:
    from cellbender.remove_background.data.dataset import SingleCellRNACountsDataset
    from cellbender.remove_background.model import RemoveBackgroundPyroModel

import numpy as np
import pyro
import pyro.distributions as dist
import scipy.sparse as sp
import torch

import cellbender.remove_background.consts as consts
from cellbender.monitor import get_hardware_usage
from cellbender.remove_background.checkpoint import load_checkpoint, load_from_checkpoint, make_tarball, unpack_tarball
from cellbender.remove_background.data.dataprep import make_simple_dataloader
from cellbender.remove_background.data.dataset import get_dataset_obj
from cellbender.remove_background.data.io import (
    POSTERIOR_SCHEMA,
    load_posterior_global_latents_json,
    load_posterior_latents_csv,
    sort_posterior_parquet,
    write_posterior_batch_to_parquet,
    write_posterior_global_latents_json,
    write_posterior_latents_csv,
)
from cellbender.remove_background.estimation import estimate_mean_noise_per_gene
from cellbender.remove_background.model import calculate_lambda, calculate_mu
from cellbender.remove_background.sparse_utils import (
    dense_to_sparse_op_torch,
)

logger = logging.getLogger("cellbender")


def _posterior_latents_path(parquet_path: Path) -> Path:
    """Return the path to the per-barcode latents CSV sidecar."""
    name = parquet_path.name
    if name.endswith("_posterior.parquet"):
        new_name = name[: -len("_posterior.parquet")] + "_posterior_latents.csv.gz"
    else:
        new_name = parquet_path.stem + "_latents.csv.gz"
    return parquet_path.parent / new_name


def _posterior_global_latents_path(parquet_path: Path) -> Path:
    """Return the path to the global latents JSON sidecar (e.g. phi_loc_scale)."""
    name = parquet_path.name
    if name.endswith("_posterior.parquet"):
        new_name = name[: -len("_posterior.parquet")] + "_posterior_global_latents.json"
    else:
        new_name = parquet_path.stem + "_global_latents.json"
    return parquet_path.parent / new_name


def _checkpoint_assertion(tarball: str) -> None:
    assert os.path.exists(tarball), (
        f"Checkpoint file {tarball} does not exist, presumably because saving "
        f"of the checkpoint file has been manually interrupted. Please re-run "
        f"and allow a checkpoint file to be saved."
    )


def load_or_stream_posterior(
    dataset_obj: "SingleCellRNACountsDataset",
    inferred_model: Optional["RemoveBackgroundPyroModel"],
    args: argparse.Namespace,
) -> "Posterior":
    """Create a Posterior object and stream the posterior to parquet if needed.

    Loads from checkpoint when a pre-computed posterior exists there.  Otherwise
    streams the posterior computation to parquet (no sort yet).  Call
    sort_and_save_posterior() next — ideally after freeing the model from the
    caller's scope — to sort the parquet and persist it to the checkpoint.

    Args:
        dataset_obj: Input data.
        inferred_model: Trained model after inference is complete.
        args: Parsed command line arguments.

    Returns:
        Posterior object with parquet on disk (unsorted if freshly computed).
    """
    _checkpoint_assertion(args.input_checkpoint_tarball)

    posterior = Posterior(
        dataset_obj=dataset_obj,
        vi_model=inferred_model,
        posterior_batch_size=args.posterior_batch_size,
        debug=args.debug,
    )
    try:
        ckpt_posterior = load_from_checkpoint(
            tarball_name=args.input_checkpoint_tarball,
            filebase=args.checkpoint_filename,
            to_load=["posterior"],
            force_use_checkpoint=args.force_use_checkpoint,
        )
    except ValueError:
        # input checkpoint tarball was not a match for this workflow
        # but we still may have saved a new tarball
        ckpt_posterior = load_from_checkpoint(
            tarball_name=consts.CHECKPOINT_FILE_NAME,
            filebase=args.checkpoint_filename,
            to_load=["posterior"],
            force_use_checkpoint=args.force_use_checkpoint,
        )
    if os.path.exists(ckpt_posterior.get("posterior_file", "does_not_exist")):
        # Load pre-computed posterior from checkpoint; already sorted.
        posterior.load(file=ckpt_posterior["posterior_file"])
        # _sort_needed stays False — the checkpoint parquet is already sorted.
    else:
        # Stream the posterior to parquet (sort happens in sort_and_save_posterior).
        logger.info("Posterior not currently included in checkpoint.")
        posterior_parquet_file = args.output_file[:-3] + "_posterior.parquet"
        posterior.ensure_posterior_computed(path=Path(posterior_parquet_file))

    return posterior


def sort_and_save_posterior(
    posterior: "Posterior",
    args: argparse.Namespace,
) -> None:
    """Sort the posterior parquet, save it to the checkpoint, and regularize.

    Intended to be called after load_or_stream_posterior() and after the caller
    has freed the model from memory, so the sort runs with maximum headroom.

    Args:
        posterior: Posterior object returned by load_or_stream_posterior().
        args: Parsed command line arguments.
    """
    if posterior._sort_needed:
        assert posterior._posterior_parquet_path is not None
        logger.info("Starting posterior parquet sort...")
        sort_posterior_parquet(
            posterior._posterior_parquet_path,
            duckdb_memory_limit=args.duckdb_memory_limit,
        )
        logger.info("Posterior sort complete.")
        posterior._sort_needed = False

        # Save sorted parquet to the checkpoint tarball.
        posterior_parquet_file = str(posterior._posterior_parquet_path)
        saved = posterior.save(file=posterior_parquet_file)
        success = False
        if saved:
            with tempfile.TemporaryDirectory() as tmp_dir:
                unpacked = unpack_tarball(tarball_name=args.input_checkpoint_tarball, directory=tmp_dir)
                if unpacked:
                    shutil.copy(posterior_parquet_file, os.path.join(tmp_dir, "posterior.parquet"))
                    latents_src = _posterior_latents_path(Path(posterior_parquet_file))
                    if latents_src.exists():
                        shutil.copy(str(latents_src), os.path.join(tmp_dir, "posterior_latents.csv.gz"))
                    global_latents_src = _posterior_global_latents_path(Path(posterior_parquet_file))
                    if global_latents_src.exists():
                        shutil.copy(str(global_latents_src), os.path.join(tmp_dir, "posterior_global_latents.json"))
                    all_ckpt_files = [
                        os.path.join(tmp_dir, f)
                        for f in os.listdir(tmp_dir)
                        if os.path.isfile(os.path.join(tmp_dir, f))
                    ]
                    success = make_tarball(files=all_ckpt_files, tarball_name=args.input_checkpoint_tarball)
        if success:
            logger.info("Added posterior object to checkpoint file.")
        else:
            logger.warning("Failed to add posterior object to checkpoint file.")

    assert posterior.dataset_obj is not None


def load_or_compute_posterior_and_save(
    dataset_obj: "SingleCellRNACountsDataset",
    inferred_model: Optional["RemoveBackgroundPyroModel"],
    args: argparse.Namespace,
) -> "Posterior":
    """Stream, sort, save, and regularize the posterior in one call.

    Convenience wrapper around load_or_stream_posterior() +
    sort_and_save_posterior().  Prefer calling those two functions separately
    from run_remove_background() so that the model can be freed before the
    sort runs.
    """
    posterior = load_or_stream_posterior(
        dataset_obj=dataset_obj,
        inferred_model=inferred_model,
        args=args,
    )
    sort_and_save_posterior(posterior=posterior, args=args)
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

    def __init__(
        self,
        dataset_obj: Optional["SingleCellRNACountsDataset"],  # Dataset
        vi_model: Optional["RemoveBackgroundPyroModel"],
        posterior_batch_size: int = 128,
        counts_dtype: np.dtype = np.dtype(np.uint32),
        float_threshold: Optional[float] = 0.5,
        debug: bool = False,
    ):
        self.dataset_obj = dataset_obj
        self.vi_model = vi_model
        if vi_model is not None:
            vi_model.eval()
            import cellbender.remove_background.vae.encoder as encoder_module

            encoder = cast(encoder_module.CompositeEncoder, vi_model.encoder)
            encoder["z"].eval()
            encoder["other"].eval()
            vi_model.decoder.eval()
        self.use_cuda = torch.cuda.is_available() if vi_model is None else vi_model.use_cuda
        self.device = "cuda" if self.use_cuda else "cpu"
        self.analyzed_gene_inds = None if (dataset_obj is None) else dataset_obj.analyzed_gene_inds
        if dataset_obj is not None and dataset_obj.data is not None:
            self.count_matrix_shape: tuple | None = dataset_obj.data["matrix"].shape
        else:
            self.count_matrix_shape = None
        self.barcode_inds = None if (self.count_matrix_shape is None) else np.arange(0, self.count_matrix_shape[0])
        self.dtype = counts_dtype
        self.debug = debug
        self.float_threshold = float_threshold
        self.posterior_batch_size = posterior_batch_size
        self._posterior_parquet_path: Path | None = None
        self._sort_needed: bool = False
        self._noise_count_posterior_kwargs: dict | None = None
        self._latents: Dict[str, np.ndarray] | None = None
        self._model_loss: dict = {}
        if dataset_obj is not None and dataset_obj.data is not None:
            self.n_cells: int | None = dataset_obj.data["matrix"].shape[0]
            self.n_genes: int | None = dataset_obj.data["matrix"].shape[1]
        else:
            self.n_cells = None
            self.n_genes = None

    def save(self, file: str) -> bool:
        """Save the posterior parquet to *file* and copy the latents CSV sidecar."""
        self.ensure_posterior_computed()
        assert self._posterior_parquet_path is not None

        try:
            dst = Path(file)
            if self._posterior_parquet_path.resolve() != dst.resolve():
                logger.info(f"Copying posterior parquet to {file}")
                shutil.copy(str(self._posterior_parquet_path), str(dst))
                src_latents = _posterior_latents_path(self._posterior_parquet_path)
                if src_latents.exists():
                    shutil.copy(str(src_latents), str(_posterior_latents_path(dst)))
                self._posterior_parquet_path = dst
            else:
                logger.info(f"Posterior parquet already at {file}")
            return True
        except Exception as exc:
            logger.warning(f"Failed to save posterior: {exc}")
            return False

    def load(self, file: str) -> bool:
        """Load a previously computed posterior from a parquet file."""
        self._posterior_parquet_path = Path(file)
        latents_path = _posterior_latents_path(self._posterior_parquet_path)
        global_latents_path = _posterior_global_latents_path(self._posterior_parquet_path)
        if latents_path.exists():
            self._latents = load_posterior_latents_csv(latents_path)
            # Merge global latents (e.g. phi_loc_scale) back in.
            self._latents.update(load_posterior_global_latents_json(global_latents_path))
        logger.info(f"Loaded pre-computed posterior from {file}")
        return True

    def ensure_posterior_computed(self, path: Optional[Path] = None, **kwargs) -> None:
        """Compute the posterior if not already done; stream directly to parquet.

        Args:
            path: Destination parquet path.  If *None* a temporary file is used.
            **kwargs: Passed to _compute_and_stream_posterior().
        """
        if self._posterior_parquet_path is not None and (kwargs == {} or kwargs == self._noise_count_posterior_kwargs):
            return
        if path is None:
            import tempfile

            fd, tmp = tempfile.mkstemp(suffix="_posterior.parquet")
            import os as _os

            _os.close(fd)
            path = Path(tmp)
        self._compute_and_stream_posterior(path=path, **kwargs)
        self._noise_count_posterior_kwargs = kwargs

    def cell_noise_count_posterior_coo(self, **kwargs) -> None:
        """Trigger computation of the posterior (streaming parquet format).

        NOTE: Kept for backward compatibility.  Use ensure_posterior_computed() instead.
        """
        self.ensure_posterior_computed(**kwargs)

    @property
    def posterior_path(self) -> Optional[Path]:
        """Path to the posterior parquet file, or None if not yet computed."""
        return self._posterior_parquet_path

    @property
    def latents_map(self) -> Dict[str, np.ndarray]:
        if self._latents is None:
            self._get_latents_map()
        assert self._latents is not None
        return self._latents

    @property
    def model_loss(self) -> dict:
        """Training loss curve. Returns the live value when vi_model is loaded,
        or the cached copy after vi_model has been freed."""
        if self.vi_model is not None:
            return self.vi_model.loss
        return self._model_loss

    @torch.no_grad()
    def _compute_and_stream_posterior(
        self,
        path: Path,
        n_samples: int = 20,
        y_map: bool = True,
        n_counts_max: int = 20,
        smallest_log_probability: float = -10.0,
    ) -> None:
        """Compute posterior noise count probabilities and stream them to parquet.

        Does NOT sort the parquet — set self._sort_needed=True after this returns
        and call sort_posterior_parquet() separately (see sort_and_save_posterior).

        Args:
            path: Destination path for the posterior parquet file.
            n_samples: Number of samples for Monte-Carlo averaging.
            y_map: Use MAP estimate of y (cell/empty) instead of sampling.
            n_counts_max: Maximum noise count axis size.
            smallest_log_probability: Entries below this threshold are discarded.

        """

        logger.debug("Computing full posterior noise counts (streaming to parquet)")

        assert self.dataset_obj is not None
        torch.cuda.empty_cache()

        analyzed_bcs_only = True
        count_matrix = self.dataset_obj.get_count_matrix()  # analyzed barcodes
        cell_logic = self.latents_map["p"] > consts.CELL_PROB_CUTOFF

        # Raise an error if there are no cells found.
        if cell_logic.sum() == 0:
            logger.error(
                f"ERROR: Found zero droplets with posterior cell "
                f"probability > {consts.CELL_PROB_CUTOFF}. Please "
                f"check the log for estimated priors on expected cells, "
                f"total droplets included, UMI counts per cell, and "
                f"UMI counts in empty droplets, and see whether these "
                f"values make sense. Consider using additional input "
                f"arguments like --expected-cells, "
                f"--total-droplets-included, --force-cell-umi-prior, "
                f"and --force-empty-umi-prior, to make these values "
                f"accurate for your dataset."
            )
            raise RuntimeError("Zero cells found!")

        dataloader_index_to_analyzed_bc_index = torch.where(torch.tensor(cell_logic))[0]
        cell_data_loader = make_simple_dataloader(
            matrix=count_matrix[cell_logic],
            batch_size=self.posterior_batch_size,
            use_cuda=self.use_cuda,
            shuffle=False,
        )

        ind = 0
        n_minibatches = len(cell_data_loader)
        assert self.analyzed_gene_inds is not None
        analyzed_gene_inds = torch.tensor(self.analyzed_gene_inds.copy())
        if analyzed_bcs_only:
            barcode_inds = torch.tensor(self.dataset_obj.analyzed_barcode_inds.copy())
        else:
            assert self.barcode_inds is not None
            barcode_inds = torch.tensor(self.barcode_inds.copy())
        logger.info(f"Computing posterior noise count probabilities in {n_minibatches} chunk(s).")

        path.parent.mkdir(parents=True, exist_ok=True)
        with pq.ParquetWriter(str(path), schema=POSTERIOR_SCHEMA) as writer:
            for i, data in enumerate(cell_data_loader):
                data = data.to(cell_data_loader.device, non_blocking=True)
                if i == 0:
                    t = time.time()
                elif i == 1:
                    logger.info(f"    [{(time.time() - t) / 60:.2f} mins per chunk]")
                logger.info(f"Working on chunk ({i + 1}/{n_minibatches})")

                if self.debug:
                    logger.debug(f"Posterior minibatch starting with droplet {ind}")
                    logger.debug("\n" + get_hardware_usage(use_cuda=self.use_cuda))

                # Compute noise count probabilities.
                noise_log_pdf_NGC, noise_count_offset_NG = self.noise_log_pdf(
                    data=data,
                    n_samples=n_samples,
                    y_map=y_map,
                    n_counts_max=n_counts_max,
                )

                # Compute a tensor to indicate sparsity.
                tensor_for_nonzeros = noise_log_pdf_NGC.clone().exp()  # probability
                tensor_for_nonzeros.data[data == 0, :] = 0.0  # remove data = 0
                tensor_for_nonzeros.data[noise_log_pdf_NGC < smallest_log_probability] = 0.0

                # Convert to sparse format.
                bcs_i_chunk, genes_i_analyzed, c_i, log_prob_i = dense_to_sparse_op_torch(
                    noise_log_pdf_NGC,
                    tensor_for_nonzeros=tensor_for_nonzeros,
                )

                genes_i = analyzed_gene_inds[genes_i_analyzed.cpu()]
                bcs_i = (bcs_i_chunk + ind).cpu()
                bcs_i = dataloader_index_to_analyzed_bc_index[bcs_i]
                bcs_i = barcode_inds[bcs_i]

                # Per-entry noise offsets (add to compact c to get absolute noise count).
                offset_i = noise_count_offset_NG[bcs_i_chunk, genes_i_analyzed].detach().cpu()
                c_i_absolute = (c_i.detach().cpu() + offset_i).numpy().astype(np.int32)

                # Stream this batch to parquet with absolute c values.
                write_posterior_batch_to_parquet(
                    writer=writer,
                    cell_ids=bcs_i.numpy().astype(np.int32),
                    gene_ids=genes_i.numpy().astype(np.int32),
                    c_vals=c_i_absolute,
                    log_probs=log_prob_i.detach().cpu().numpy().astype(np.float32),
                )

                ind += data.shape[0]

                # Free large per-chunk tensors immediately to keep peak memory low.
                del noise_log_pdf_NGC, noise_count_offset_NG, tensor_for_nonzeros
                del bcs_i_chunk, genes_i_analyzed, c_i, log_prob_i
                del genes_i, bcs_i, offset_i, c_i_absolute, data
                gc.collect()

            logger.info("All chunks complete. Closing parquet writer...")

        logger.info("Parquet writer closed.")

        # Free the DataLoader, count matrix slice, and index tensors before the
        # sort. These are separate allocations from dataset_obj's full matrix.
        del cell_data_loader, count_matrix, dataloader_index_to_analyzed_bc_index
        del analyzed_gene_inds, barcode_inds
        gc.collect()

        # Cache the training loss curve before releasing the model reference,
        # so it remains accessible for writing to the output h5 file.
        if self.vi_model is not None:
            self._model_loss = self.vi_model.loss
        # Drop the Posterior's own reference to the model. The caller's reference
        # (inferred_model in run_remove_background) must also be dropped before
        # the sort runs — that is done by run_remove_background between calling
        # load_or_stream_posterior() and sort_and_save_posterior().
        self.vi_model = None
        torch.cuda.empty_cache()

        # Mark that the parquet needs sorting before DuckDB queries can use it.
        self._sort_needed = True

        # Write per-barcode latents CSV sidecar.
        write_posterior_latents_csv(_posterior_latents_path(path), self.latents_map)
        # Write global latents JSON sidecar (e.g. phi_loc_scale).
        write_posterior_global_latents_json(_posterior_global_latents_path(path), self.latents_map)

        self._posterior_parquet_path = path

    @torch.no_grad()
    def sample(self, data, lambda_multiplier=1.0, y_map: bool = False) -> torch.Tensor:
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
    def map_denoised_counts_from_sampled_latents(
        self, data, n_samples: int, lambda_multiplier: float = 1.0, y_map: bool = False
    ) -> torch.Tensor:
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
        denoised_output_count_matrix = torch.clamp(data - noise_counts, min=0.0)

        return denoised_output_count_matrix

    @torch.no_grad()
    def noise_log_pdf(
        self, data, n_samples: int = 1, lambda_multiplier=1.0, y_map: bool = True, n_counts_max: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            log_prob_noise_counts_NGC = log_prob_noise_counts_NGC - torch.logsumexp(
                log_prob_noise_counts_NGC, dim=-1, keepdim=True
            )

            # Add the probability from this sample to our running total.
            # Update rule is
            # log_prob_total_n = LAE [ log(1 - 1/n) + log_prob_total_{n-1}, log(1/n) + log_prob_sample ]
            if s == 1:
                noise_log_pdf_NGC = log_prob_noise_counts_NGC
            else:
                # This is a (normalized) running sum over samples in log-probability space.
                assert noise_log_pdf_NGC is not None
                noise_log_pdf_NGC = torch.logaddexp(
                    noise_log_pdf_NGC + torch.log(torch.tensor(1.0 - 1.0 / s).to(device=data.device)),
                    log_prob_noise_counts_NGC + torch.log(torch.tensor(1.0 / s).to(device=data.device)),
                )

        assert noise_log_pdf_NGC is not None
        assert noise_count_offset_NG is not None
        return noise_log_pdf_NGC, noise_count_offset_NG

    @torch.no_grad()
    def sample_mu_lambda_alpha(
        self, data: torch.Tensor, y_map: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # logger.debug("Replaying model with guide to sample mu, alpha, lambda")

        assert self.vi_model is not None
        # Use pyro poutine to trace the guide and sample parameter values.
        guide_trace = pyro.poutine.trace(self.vi_model.guide).get_trace(x=data)

        # If using MAP for y (so that you never get samples of cell and no cell),
        # then intervene and replace a sampled y with the MAP
        if y_map:
            guide_trace.nodes["y"]["value"] = (guide_trace.nodes["p_passback"]["value"] > 0).clone().detach()

        replayed_model = pyro.poutine.replay(self.vi_model.model, guide_trace)

        # Run the model using these sampled values.
        replayed_model_output = replayed_model(x=data)

        # The model returns mu, alpha, and lambda.
        mu_sample = replayed_model_output["mu"]
        lambda_sample = replayed_model_output["lam"]
        alpha_sample = replayed_model_output["alpha"]

        return mu_sample, lambda_sample, alpha_sample

    @staticmethod
    @torch.no_grad()
    def _log_prob_noise_count_tensor(
        data: torch.Tensor,
        mu_est: torch.Tensor,
        lambda_est: torch.Tensor,
        alpha_est: Optional[torch.Tensor],
        n_counts_max: int = 100,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        poisson_values_low = torch.clamp(torch.min(poisson_values_low, (data - n + 1).int()), min=0).float()

        # Construct a big tensor of possible noise counts per cell per gene,
        # shape (batch_cells, n_genes, max_noise_counts)
        noise_count_tensor = (
            torch.arange(start=0, end=n).expand([data.shape[0], data.shape[1], -1]).float().to(device=data.device)
        )
        noise_count_tensor = noise_count_tensor + poisson_values_low.unsqueeze(-1)

        # Compute probabilities of each number of noise counts.
        # NOTE: some values will be outside the support (negative values for NB).
        # This results in NaNs.
        if alpha_est is None:
            # Poisson only model
            log_prob_tensor = dist.Poisson(lambda_est.unsqueeze(-1), validate_args=False).log_prob(
                noise_count_tensor
            ) + dist.Poisson(mu_est.unsqueeze(-1), validate_args=False).log_prob(
                data.unsqueeze(-1) - noise_count_tensor
            )
            logger.debug("Using all poisson model (since alpha is not supplied to posterior)")
        else:
            logits = (mu_est.log() - alpha_est.log()).unsqueeze(-1)
            log_prob_tensor = dist.Poisson(lambda_est.unsqueeze(-1), validate_args=False).log_prob(
                noise_count_tensor
            ) + dist.NegativeBinomial(total_count=alpha_est.unsqueeze(-1), logits=logits, validate_args=False).log_prob(
                data.unsqueeze(-1) - noise_count_tensor
            )

        # Set log_prob to -inf if noise > data.
        neg_inf_tensor = torch.ones_like(log_prob_tensor) * -np.inf
        log_prob_tensor = torch.where((noise_count_tensor <= data.unsqueeze(-1)), log_prob_tensor, neg_inf_tensor)

        # logger.debug(f"Prob computation with tensor of shape {log_prob_tensor.shape}")

        if debug:
            assert not torch.isnan(log_prob_tensor).any(), "log_prob_tensor contains a NaN"
            if torch.isinf(log_prob_tensor).all(dim=-1).any():
                print(torch.where(torch.isinf(log_prob_tensor).all(dim=-1)))
                raise AssertionError("There is at least one log_prob_tensor[n, g, :] that has all-zero probability")

        return log_prob_tensor, poisson_values_low

    @torch.no_grad()
    def _get_latents_map(self):
        """Calculate the encoded latent variables."""

        logger.debug("Computing latent variables")

        if self.vi_model is None:
            self._latents = {"z": None, "d": None, "p": None, "phi_loc_scale": None, "epsilon": None}
            return None

        data_loader = self.dataset_obj.get_dataloader(
            use_cuda=self.use_cuda, analyzed_bcs_only=True, batch_size=500, shuffle=False
        )

        n_analyzed = data_loader.dataset.shape[0]

        z = np.zeros((n_analyzed, self.vi_model.encoder["z"].output_dim))
        d = np.zeros(n_analyzed)
        p = np.zeros(n_analyzed)
        epsilon = np.zeros(n_analyzed)

        phi_loc = pyro.param("phi_loc")
        phi_scale = pyro.param("phi_scale")
        if "chi_ambient" in pyro.get_param_store().keys():
            chi_ambient = pyro.param("chi_ambient").detach()
        else:
            chi_ambient = None

        start = 0
        for i, data in enumerate(data_loader):
            data = data.to(data_loader.device, non_blocking=True)
            end = start + data.shape[0]

            enc = self.vi_model.encoder(x=data, chi_ambient=chi_ambient, cell_prior_log=self.vi_model.d_cell_loc_prior)
            z[start:end, :] = enc["z"]["loc"].detach().cpu().numpy()

            d[start:end] = (
                dist.LogNormal(loc=enc["d_loc"], scale=pyro.param("d_cell_scale")).mean.detach().cpu().numpy()
            )

            p[start:end] = enc["p_y"].sigmoid().detach().cpu().numpy()

            epsilon[start:end] = (
                dist.Gamma(enc["epsilon"] * self.vi_model.epsilon_prior, self.vi_model.epsilon_prior)
                .mean.detach()
                .cpu()
                .numpy()
            )

            start = end

        self._latents = {
            "z": z,
            "d": d,
            "p": p,
            "phi_loc_scale": [phi_loc.item(), phi_scale.item()],
            "epsilon": epsilon,
        }

    @torch.no_grad()
    def _get_mu_alpha_lambda_map(self, data: torch.Tensor, chi_ambient: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        logger.debug("Computing MAP esitmate of mu, lambda, alpha")

        assert self.vi_model is not None
        # Encode latents.
        enc = self.vi_model.encoder(x=data, chi_ambient=chi_ambient, cell_prior_log=self.vi_model.d_cell_loc_prior)
        z_map = enc["z"]["loc"]

        chi_map = self.vi_model.decoder(z_map)
        phi_loc = pyro.param("phi_loc")
        phi_scale = pyro.param("phi_scale")
        phi_conc = phi_loc.pow(2) / phi_scale.pow(2)
        phi_rate = phi_loc / phi_scale.pow(2)
        alpha_map = 1.0 / dist.Gamma(phi_conc, phi_rate).mean

        y = (enc["p_y"] > 0).float()
        d_empty = dist.LogNormal(loc=pyro.param("d_empty_loc"), scale=pyro.param("d_empty_scale")).mean
        d_cell = dist.LogNormal(loc=enc["d_loc"], scale=pyro.param("d_cell_scale")).mean
        epsilon = dist.Gamma(enc["epsilon"] * self.vi_model.epsilon_prior, self.vi_model.epsilon_prior).mean

        if self.vi_model.include_rho:
            rho = pyro.param("rho_alpha") / (pyro.param("rho_alpha") + pyro.param("rho_beta"))
        else:
            rho = None

        # Calculate MAP estimates of mu and lambda.
        mu_map = calculate_mu(
            model_type=self.vi_model.model_type,
            epsilon=epsilon,
            d_cell=d_cell,
            chi=chi_map,
            y=y,
            rho=rho,
        )
        lambda_map = calculate_lambda(
            model_type=self.vi_model.model_type,
            epsilon=epsilon,
            chi_ambient=chi_ambient,
            d_empty=d_empty,
            y=y,
            d_cell=d_cell,
            rho=rho,
            chi_bar=self.vi_model.avg_gene_expression,
        )

        return {"mu": mu_map, "lam": lambda_map, "alpha": alpha_map}


def compute_mean_target_removal_as_function(
    noise_count_posterior_coo: Path,
    n_genes: int,
    raw_count_csr_for_cells: sp.csr_matrix,
    n_cells: int,
    device: str,
    per_gene: bool,
) -> Callable[[float], torch.Tensor]:
    """Given the posterior parquet, return a function that computes target
    removal (either overall or per-gene) as a function of FPR.

    NOTE: computes the value "per cell", i.e. dividing
    by the number of cells, so that total removal can be computed by
    multiplying this by the number of cells in question.

    Args:
        noise_count_posterior_coo: Path to the posterior parquet file.
        n_genes: Total number of genes; determines output array length.
        raw_count_csr_for_cells: The input count matrix for only the cells
            included in the posterior
        n_cells: Number of cells included in the posterior, same number as in
            raw_count_csr_for_cells
        device: 'cpu' or 'cuda' (retained for API compatibility)
        per_gene: True to come up with one target per gene

    Returns:
        target_removal_scaled_per_cell: Noise count removal target

    """

    # TODO: s1.h5 with FPR 0.99 only removes 50% of signal

    logger.debug("Computing per-gene mean noise totals...")
    mean_noise_per_gene = estimate_mean_noise_per_gene(
        source=noise_count_posterior_coo,
        n_genes=n_genes,
    )
    mean_noise_total = float(mean_noise_per_gene.sum())
    logger.debug(f"Total noise counts from mean noise estimator = {mean_noise_total}")

    raw_count_per_gene = np.array(raw_count_csr_for_cells.sum(axis=0)).squeeze()
    raw_count_total = float(raw_count_csr_for_cells.sum())
    logger.debug(f"Total counts in raw matrix for cells = {raw_count_total}")

    approx_signal_per_gene = raw_count_per_gene - mean_noise_per_gene
    approx_signal_total = raw_count_total - mean_noise_total
    logger.debug(f"Approximate signal has total counts = {approx_signal_total}")
    logger.debug(f"Number of cells = {n_cells}")

    def _target_fun(fpr: float) -> torch.Tensor:
        """The function which gets returned"""
        if per_gene:
            target = mean_noise_per_gene + fpr * approx_signal_per_gene
        else:
            target = mean_noise_total + fpr * approx_signal_total

        # Return target scaled to be per-cell.
        return torch.tensor(target / n_cells).to(device)

    return _target_fun


def restore_from_checkpoint(
    tarball_name: str, input_file: str
) -> Tuple["SingleCellRNACountsDataset", "RemoveBackgroundPyroModel", Posterior]:
    """Convenience function not used by the codebase"""

    d = load_checkpoint(filebase=None, tarball_name=tarball_name)
    d.update(load_from_checkpoint(filebase=None, tarball_name=tarball_name, to_load=["posterior"]))
    args = cast("argparse.Namespace", d["args"])
    args.input_file = input_file

    dataset_obj = get_dataset_obj(args=args)
    model = cast("RemoveBackgroundPyroModel", d["model"])
    posterior_file = cast(str, d["posterior_file"])

    posterior = Posterior(
        dataset_obj=dataset_obj,
        vi_model=model,
    )
    posterior.load(file=posterior_file)
    return dataset_obj, model, posterior
