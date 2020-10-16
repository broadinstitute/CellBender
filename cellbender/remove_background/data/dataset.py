"""Class and functions for working with a count matrix dataset."""

import tables
import numpy as np
import scipy.sparse as sp
import scipy.io as io
from scipy.stats import mode
from sklearn.decomposition import PCA
import torch

import cellbender.remove_background.model
import cellbender.remove_background.consts as consts
from cellbender.remove_background.infer import ProbPosterior
from cellbender.remove_background.data.dataprep import DataLoader

from typing import Dict, List, Union, Tuple, Optional
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # This needs to be after matplotlib.use('Agg')


class SingleCellRNACountsDataset:
    """Object for storing scRNA-seq count matrix data and basic manipulations
    and pre-processing (e.g. estimation of prior hyperparameters).

    Args:
        input_file: Input data file path.
        expected_cell_count: Expected number of real cells a priori.
        total_droplet_barcodes: Total number of droplets to include in the
            cell-calling analysis.
        model_name: Model to use.
        gene_blacklist: List of integer indices of genes to exclude entirely.
        low_count_threshold: Droplets with UMI counts below this number are
            excluded entirely from the analysis.
        lambda_multiplier: Float that multiplies background estimate before
            MAP estimation.

    Attributes:
        input_file: Name of data source file.
        analyzed_barcode_inds: numpy.ndarray of indices that denote which
            barcodes, as indexed in the original dataset, are being used in the
            inference procedure.
        analyzed_gene_inds: numpy.ndarray of indices that denote which genes, as
            indexed in the original dataset, are (nonzero and) being used in the
            inference procedure.
        data: Loaded data as a dict, with ['matrix', 'barcodes', 'gene_names'].
        is_trimmed: This gets set to True after running
            trim_dataset_for_analysis().
        model_name: Name of model being run.
        priors: Priors estimated from the data useful for modelling.
        posterior: Posterior estimated after inference.
        empty_UMI_threshold: This UMI count is the maximum UMI count in the
            user-defined surely empty droplets.

    Note: Count data is kept as the original, untransformed data.  Priors are
    in terms of the transformed count data.

    """

    def __init__(self,
                 input_file: str,
                 model_name: str,
                 exclude_antibodies: bool,
                 low_count_threshold: int,
                 fpr: List[float],
                 expected_cell_count: Optional[int] = None,
                 total_droplet_barcodes: int = consts.TOTAL_DROPLET_DEFAULT,
                 fraction_empties: Optional[float] = None,
                 gene_blacklist: List[int] = []):
        assert input_file is not None, "Attempting to load data, but no " \
                                       "input file was specified."
        self.input_file = input_file
        self.analyzed_barcode_inds = np.array([])  # Barcodes trained each epoch
        self.analyzed_gene_inds = np.array([])
        self.empty_barcode_inds = np.array([])  # Barcodes randomized each epoch
        self.data = None
        self.exclude_antibodies = exclude_antibodies
        self.model_name = model_name
        self.fraction_empties = fraction_empties
        self.is_trimmed = False
        self.low_count_threshold = low_count_threshold
        self.priors = {'n_cells': expected_cell_count}  # Expected cells could be None.
        self.posterior = None
        self.fpr = fpr
        self.random = np.random.RandomState(seed=1234)
        self.EMPIRICAL_LOW_UMI_TO_EMPTY_DROPLET_THRESHOLD = \
            consts.EMPIRICAL_LOW_UMI_TO_EMPTY_DROPLET_THRESHOLD
        self.SIMPLE_MODEL_D_STD_PRIOR = consts.SIMPLE_MODEL_D_STD_PRIOR
        self.CELL_PROB_CUTOFF = consts.CELL_PROB_CUTOFF

        # Load the dataset.
        self._load_data()

        # Trim the dataset.
        self._trim_dataset_for_analysis(total_droplet_barcodes=total_droplet_barcodes,
                                        low_UMI_count_cutoff=low_count_threshold,
                                        gene_blacklist=gene_blacklist)

        # Estimate priors.
        self._estimate_priors()

    def _detect_input_data_type(self) -> str:
        """Detect the type of input data."""

        # Error if no input data file has been specified.
        assert self.input_file is not None, \
            "Attempting to load data, but no input file was specified."

        # Detect type.
        if os.path.isdir(self.input_file):
            return 'cellranger_mtx'

        else:
            return 'cellranger_h5'

    def _load_data(self):
        """Load a dataset into the SingleCellRNACountsDataset object from
        the self.input_file"""

        # Detect input data type.
        data_type = self._detect_input_data_type()

        # Load the dataset.
        if data_type == 'cellranger_mtx':

            logging.info(f"Loading data from directory {self.input_file}")
            self.data = get_matrix_from_cellranger_mtx(self.input_file)

        elif data_type == 'cellranger_h5':

            logging.info(f"Loading data from file {self.input_file}")
            self.data = get_matrix_from_cellranger_h5(self.input_file)

        else:
            raise NotImplementedError

    def _trim_dataset_for_analysis(
            self,
            total_droplet_barcodes: int = consts.TOTAL_DROPLET_DEFAULT,
            low_UMI_count_cutoff: int = consts.LOW_UMI_CUTOFF,
            gene_blacklist: List[int] = []):
        """Trim the dataset for inference, choosing barcodes and genes to use.

        Sets the values of self.analyzed_barcode_inds, and
        self.empty_barcode_inds, which are used throughout training.

        Args:
            total_droplet_barcodes: Number of total droplets to include
                during inference each epoch, and for which to make cell calls.
            low_UMI_count_cutoff: Barcodes with total UMI counts below
                this number are excluded.
            gene_blacklist: List of gene indices to trim out and exclude.

        Note:
            self.priors['n_cells'] is only used to choose which barcodes to
            include in the analysis, as well as to estimate the sizes of cells
            and empty droplets (elsewhere).  It need only be a reasonable guess.
            The full analysis makes inferences about which barcodes contain
            cells, and ultimately will determine this number.
            However, if running only the 'simple' model, the expected_cell_count
            is taken to be the true number of cells, since empty droplets are
            not part of the 'simple' model.

        """

        logging.info("Trimming dataset for inference.")

        # Get data matrix and barcode order that sorts barcodes by UMI count.
        matrix = self.data['matrix']
        umi_counts = np.array(matrix.sum(axis=1)).squeeze()
        umi_count_order = np.argsort(umi_counts)[::-1]

        # Initially set the default to be the whole dataset.
        self.analyzed_barcode_inds = np.arange(start=0, stop=matrix.shape[0])
        self.analyzed_gene_inds = np.arange(start=0, stop=matrix.shape[1])

        # Expected cells must not exceed nonzero count barcodes.
        num_nonzero_barcodes = np.sum(umi_counts > 0).item()

        # Choose which genes to use based on their having nonzero counts.
        # (All barcodes must be included so that inference can generalize.)
        gene_counts_per_barcode = np.array(matrix.sum(axis=0)).squeeze()
        if self.exclude_antibodies:
            antibody_logic = (self.data['feature_types'] == b'Antibody Capture')
            # Exclude these by setting their counts to zero
            logging.info(f"Excluding {antibody_logic.sum()} features that "
                         f"correspond to antibody capture.")
            gene_counts_per_barcode[antibody_logic] = 0
        nonzero_gene_counts = (gene_counts_per_barcode > 0)
        self.analyzed_gene_inds = np.where(nonzero_gene_counts)[0].astype(dtype=int)

        if self.analyzed_gene_inds.size == 0:
            logging.warning("During data loading, found no genes with > 0 "
                            "counts.  Terminating analysis.")
            raise AssertionError("No genes with nonzero counts.  Check the "
                                 "input data file.")

        # Remove blacklisted genes.
        if len(gene_blacklist) > 0:

            # Make gene_blacklist a set for fast inclusion testing.
            gene_blacklist = set(gene_blacklist)

            # Ensure genes on the blacklist are excluded.
            self.analyzed_gene_inds = np.array([g for g in
                                                self.analyzed_gene_inds
                                                if g not in gene_blacklist])

        if self.analyzed_gene_inds.size == 0:
            logging.warning("All nonzero genes have been blacklisted.  "
                            "Terminating analysis.")
            raise AssertionError("All genes with nonzero counts have been "
                                 "blacklisted.  Examine the dataset and "
                                 "reduce the blacklist.")

        logging.info(f"Including {self.analyzed_gene_inds.size} genes that have"
                     f" nonzero counts.")

        # Estimate priors on cell size and 'empty' droplet size.
        self.priors['cell_counts'], self.priors['empty_counts'] = \
            get_d_priors_from_dataset(self)  # After gene trimming

        # Estimate the number of real cells if it was not specified.
        if self.priors['n_cells'] is None:
            self.priors['n_cells'] = estimate_cell_count_from_dataset(self)

        # n_cells is at most the number of nonzero barcodes.
        n_cells = min(self.priors['n_cells'], num_nonzero_barcodes)

        assert n_cells > 0, "No cells identified.  Try to use the option " \
                            "--expected-cells to specify a prior on cell " \
                            "count.  Also ensure that --low-count-threshold " \
                            "is not too large (less than empty droplets)."

        # If running the simple model, just use the expected cells, no more.
        if self.model_name == "simple":

            self.analyzed_barcode_inds = np.array(umi_count_order[:n_cells],
                                                  dtype=int)

            logging.info(f"Simple model: using "
                         f"{self.analyzed_barcode_inds.size} cell barcodes.")

        # If not using the simple model, include empty droplets.
        else:

            # Get the cell barcodes.
            cell_barcodes = umi_count_order[:n_cells]

            # Set the low UMI count cutoff to be the greater of either
            # the user input value, or an empirically-derived value.
            empirical_low_UMI = int(self.priors['empty_counts'] *
                                    self.EMPIRICAL_LOW_UMI_TO_EMPTY_DROPLET_THRESHOLD)
            low_UMI_count_cutoff = max(low_UMI_count_cutoff,
                                       empirical_low_UMI)
            logging.info(f"Excluding barcodes with counts below "
                         f"{low_UMI_count_cutoff}")

            # See how many barcodes there are to work with total.
            num_barcodes_above_umi_cutoff = \
                np.sum(umi_counts > low_UMI_count_cutoff).item()

            assert num_barcodes_above_umi_cutoff > 0, \
                f"There are no barcodes with UMI counts over the lower " \
                f"cutoff of {low_UMI_count_cutoff}"

            assert num_barcodes_above_umi_cutoff > n_cells, \
                f"There are no empty droplets with UMI counts over the lower " \
                f"cutoff of {low_UMI_count_cutoff}.  Some empty droplets are " \
                f"necessary for the analysis.  Reduce the " \
                f"--low-count-threshold parameter."

            # Get a number of transition-region barcodes.
            num_transition_barcodes = (total_droplet_barcodes
                                       - cell_barcodes.size)

            assert num_transition_barcodes > 0, \
                f"The number of cells is {cell_barcodes.size}, but the " \
                f"number of total droplets included is " \
                f"{total_droplet_barcodes}.  Increase " \
                f"--total_droplet_barcodes above {cell_barcodes.size}, or " \
                f"specify a different number of expected cells using " \
                f"--expected-cells"

            num = min(num_transition_barcodes,
                      num_barcodes_above_umi_cutoff - cell_barcodes.size)
            num = max(0, num)
            transition_barcodes = umi_count_order[n_cells:
                                                  (n_cells + num)]

            assert transition_barcodes.size > 0, \
                f"There are no barcodes identified from the transition " \
                f"region between cell and empty.  The intended number of " \
                f"transition barcodes was {num_transition_barcodes}.  " \
                f"This indicates that the low UMI count cutoff, " \
                f"{low_UMI_count_cutoff}, was likely too high.  Try to " \
                f"reduce --low-count-threshold"

            # Use the cell barcodes and transition barcodes for analysis.
            self.analyzed_barcode_inds = np.concatenate((
                cell_barcodes,
                transition_barcodes)).astype(dtype=int)

            # Identify probable empty droplet barcodes.
            if num < num_transition_barcodes:

                # This means we already used all the barcodes.
                empty_droplet_barcodes = np.array([])

            else:

                # Decide which empty barcodes to include.
                empty_droplet_sorted_barcode_inds = \
                    np.arange(n_cells + num, num_barcodes_above_umi_cutoff,
                              dtype=int)  # The entire range
                empty_droplet_barcodes = \
                    umi_count_order[empty_droplet_sorted_barcode_inds]

            self.empty_barcode_inds = empty_droplet_barcodes\
                .astype(dtype=int)

            # Find the UMI threshold for surely empty droplets.
            last_analyzed_bc = min(cell_barcodes.size + transition_barcodes.size,
                                   umi_count_order.size)
            self.empty_UMI_threshold = (umi_counts[umi_count_order]
                                        [last_analyzed_bc])

            # Find the max UMI count for any cell.
            self.max_UMI_count = umi_counts.max()

            logging.info(f"Using {cell_barcodes.size} probable cell "
                         f"barcodes, plus an additional "
                         f"{transition_barcodes.size} barcodes, "
                         f"and {empty_droplet_barcodes.size} empty "
                         f"droplets.")
            logging.info(f"Largest surely-empty droplet has "
                         f"{self.empty_UMI_threshold} UMI counts.")

            if ((low_UMI_count_cutoff == self.low_count_threshold)
                    and (empty_droplet_barcodes.size == 0)):
                logging.warning("Warning: few empty droplets identified. Low "
                                "UMI cutoff may be too high. Check the UMI "
                                "decay curve, and decrease the "
                                "--low-count-threshold parameter if necessary.")

        self.is_trimmed = True

    def _estimate_priors(self):
        """Estimate relevant priors, populating fields in the self.priors dict.
        """

        # Estimate the log UMI count turning point between cells and 'empties'.
        self.priors['log_counts_crossover'] = \
            np.mean(np.log1p([self.priors['cell_counts'],
                              self.priors['empty_counts']])).item()

        # TODO: overhaul estimation of d_std.  add estimate of d_empty_std

        # Estimate prior for the scale param of LogNormal for d.
        if self.model_name != "simple":
            self.priors['d_std'] = (np.log1p(self.priors['cell_counts'])
                                    - self.priors['log_counts_crossover']) / 5
        else:
            # Use a reasonable prior in log space.
            self.priors['d_std'] = self.SIMPLE_MODEL_D_STD_PRIOR

        # Priors for models that include empty droplets:
        if self.model_name != "simple":
            # Estimate fraction of trimmed dataset that contains cells.
            # cell_prob = self.priors['n_cells']
            # / self.analyzed_barcode_inds.size
            cell_prob = (1 - self.fraction_empties) \
                        * (self.priors['n_cells']
                           / self.analyzed_barcode_inds.size)
            self.priors['cell_prob'] = cell_prob

            assert cell_prob > 0, f"Fraction of trimmed dataset " \
                                  f"containing cells should be > 0, " \
                                  f"but is {cell_prob}."

            assert cell_prob <= 1, f"Fraction of trimmed dataset " \
                                   f"containing cells should be at most 1, " \
                                   f"but is {cell_prob}."

            # Turn cell probability into logit.
            self.priors['cell_logit'] = np.log(cell_prob
                                               / (1 - cell_prob)).item()

            # Estimate the ambient gene expression profile.
            self.priors['chi_ambient'], self.priors['chi_bar'] = \
                estimate_chi_ambient_from_dataset(self)

    def get_count_matrix(self) -> sp.csr.csr_matrix:
        """Get the count matrix, trimmed if trimming has occurred."""

        if self.is_trimmed:

            # Return the count matrix for selected barcodes and genes.
            trimmed_bc_matrix = self.data['matrix'][self.analyzed_barcode_inds,
                                                    :].tocsc()
            trimmed_matrix = trimmed_bc_matrix[:, self.analyzed_gene_inds].tocsr()
            return trimmed_matrix

        else:
            logging.warning("Using full count matrix, without any trimming.  "
                            "Could be slow.")
            return self.data['matrix']

    def get_count_matrix_empties(self) -> sp.csr.csr_matrix:
        """Get the count matrix for empty drops, trimmed if possible."""

        if self.is_trimmed:

            # Return the count matrix for selected barcodes and genes.
            trimmed_bc_matrix = self.data['matrix'][self.empty_barcode_inds,
                                                    :].tocsc()
            trimmed_matrix = trimmed_bc_matrix[:, self.analyzed_gene_inds].tocsr()
            return trimmed_matrix

        else:
            logging.error("Trying to get empty count matrix without "
                          "trimmed data.")
            return self.data['matrix']

    def get_count_matrix_all_barcodes(self) -> sp.csr.csr_matrix:
        """Get the count matrix, trimming only genes, not barcodes."""

        if self.is_trimmed:

            # Return the count matrix for selected barcodes and genes.
            trimmed_bc_matrix = self.data['matrix'].tocsc()
            trimmed_matrix = trimmed_bc_matrix[:, self.analyzed_gene_inds].tocsr()
            return trimmed_matrix

        else:
            logging.warning("Using full count matrix, without any trimming.  "
                            "Could be slow.")
            return self.data['matrix']

    def get_dataloader(self,
                       use_cuda: bool = True,
                       batch_size: int = 200,
                       shuffle: bool = False,
                       analyzed_bcs_only: bool = True) -> DataLoader:
        """Return a dataloader for the count matrix.

        Args:
            use_cuda: Whether to load data into GPU memory.
            batch_size: Size of minibatch of data yielded by dataloader.
            shuffle: Whether dataloader should shuffle the data.
            analyzed_bcs_only: Only include the barcodes that have been
                analyzed, not the surely empty droplets.

        Returns:
            data_loader: A dataloader that yields the entire dataset in batches.

        """

        if analyzed_bcs_only:
            count_matrix = self.get_count_matrix()
        else:
            count_matrix = self.get_count_matrix_all_barcodes()

        return DataLoader(count_matrix,
                          empty_drop_dataset=None,
                          batch_size=batch_size,
                          fraction_empties=0.,
                          shuffle=shuffle,
                          use_cuda=use_cuda)

    def save_to_output_file(
            self,
            output_file: str,
            inferred_model: 'RemoveBackgroundPyroModel',
            save_plots: bool = False) -> bool:
        """Write the results of an inference procedure to an output file.

        Output is an HDF5 file.  To be written:
        Inferred ambient-subtracted UMI count matrix.
        Inferred probabilities that each barcode contains a real cell.
        Inferred cell size scale factors.
        Inferred ambient gene expression count vector for droplets without
            cells.
        Inferred contamination fraction hyperparameters.
        Embeddings of gene expression of cells into a low-dimensional latent
            space.

        Args:
            inferred_model: RemoveBackgroundPyroModel which has
                already had the inference procedure run.
            output_file: Name of output .h5 file
            save_plots: Setting this to True will save plots of outputs.

        Returns:
            True if the output was written to file successfully.

        """

        logging.info("Preparing to write outputs to file...")

        # Create posterior.
        self.posterior = ProbPosterior(dataset_obj=self,
                                       vi_model=inferred_model,
                                       fpr=self.fpr[0])

        # Encoded values of latent variables.
        enc = self.posterior.latents
        z = enc['z']
        d = enc['d']
        p = enc['p']
        epsilon = enc['epsilon']
        phi_params = enc['phi_loc_scale']

        # Estimate the ambient-background-subtracted UMI count matrix.
        if self.model_name != "simple":

            inferred_count_matrix = self.posterior.mean

        else:

            # No need to generate a new count matrix for simple model.
            inferred_count_matrix = self.data['matrix'].tocsc()
            logging.info("Simple model: outputting un-altered count matrix.")

        # Inferred ambient gene expression vector.
        ambient_expression_trimmed = cellbender.remove_background.model.\
            get_ambient_expression()

        # Convert the indices from trimmed gene set to original gene indices.
        ambient_expression = np.zeros(self.data['matrix'].shape[1])
        ambient_expression[self.analyzed_gene_inds] = ambient_expression_trimmed

        # Figure out the indices of barcodes that have cells.
        if p is not None:
            p[np.isnan(p)] = 0.
            cell_barcode_inds = self.analyzed_barcode_inds
            if np.sum(p > 0.5) == 0:
                logging.warning("Warning: Found no cells!")
            filtered_inds_of_analyzed_barcodes = p > self.CELL_PROB_CUTOFF
        else:
            cell_barcode_inds = self.analyzed_barcode_inds
            filtered_inds_of_analyzed_barcodes = np.arange(0, d.size)

        # CellRanger version (format output like input).
        if self._detect_input_data_type() == 'cellranger_h5':
            cellranger_version = detect_cellranger_version_h5(self.input_file)
        else:
            cellranger_version = detect_cellranger_version_mtx(self.input_file)

        # Write to output file, for each lambda specified by user.
        if len(self.fpr) == 1:
            write_succeeded = write_matrix_to_cellranger_h5(
                cellranger_version=cellranger_version,
                output_file=output_file,
                gene_names=self.data['gene_names'],
                gene_ids=self.data['gene_ids'],
                feature_types=self.data['feature_types'],
                genomes=self.data['genomes'],
                barcodes=self.data['barcodes'],
                inferred_count_matrix=inferred_count_matrix,
                cell_barcode_inds=cell_barcode_inds,
                ambient_expression=ambient_expression,
                z=z, d=d, p=p, phi=phi_params, epsilon=epsilon,
                rho=cellbender.remove_background.model.get_rho(),
                fpr=self.fpr[0],
                lambda_multiplier=self.posterior.lambda_multiplier,
                loss=inferred_model.loss)

            # Generate filename for filtered matrix output.
            file_dir, file_base = os.path.split(output_file)
            file_name = os.path.splitext(os.path.basename(file_base))[0]
            filtered_output_file = os.path.join(file_dir, file_name + "_filtered.h5")

            # Write filtered matrix (cells only) to output file.
            if self.model_name != "simple":
                cell_barcode_inds = \
                    self.analyzed_barcode_inds[filtered_inds_of_analyzed_barcodes]

                cell_barcodes = self.data['barcodes'][cell_barcode_inds]

                write_matrix_to_cellranger_h5(
                    cellranger_version=cellranger_version,
                    output_file=filtered_output_file,
                    gene_names=self.data['gene_names'],
                    gene_ids=self.data['gene_ids'],
                    feature_types=self.data['feature_types'],
                    genomes=self.data['genomes'],
                    barcodes=cell_barcodes,
                    inferred_count_matrix=inferred_count_matrix[cell_barcode_inds, :],
                    cell_barcode_inds=None,
                    ambient_expression=ambient_expression,
                    z=z[filtered_inds_of_analyzed_barcodes, :],
                    d=d[filtered_inds_of_analyzed_barcodes],
                    p=p[filtered_inds_of_analyzed_barcodes],
                    phi=phi_params,
                    epsilon=epsilon[filtered_inds_of_analyzed_barcodes],
                    rho=cellbender.remove_background.model.get_rho(),
                    fpr=self.fpr[0],
                    lambda_multiplier=self.posterior.lambda_multiplier,
                    loss=inferred_model.loss)

        else:

            file_dir, file_base = os.path.split(output_file)
            file_name = os.path.splitext(os.path.basename(file_base))[0]

            for i, fpr in enumerate(self.fpr):

                if i > 0:  # first FPR has already been computed

                    # Re-compute posterior counts for each new lambda.
                    self.posterior.fpr = fpr  # reach in and change the FPR
                    self.posterior._get_mean()  # force re-computation of posterior
                    inferred_count_matrix = self.posterior.mean

                # Create an output filename for this lambda value.
                temp_output_filename = os.path.join(file_dir, file_name + f"_FPR_{fpr}.h5")

                write_succeeded = write_matrix_to_cellranger_h5(
                    cellranger_version=cellranger_version,
                    output_file=temp_output_filename,
                    gene_names=self.data['gene_names'],
                    gene_ids=self.data['gene_ids'],
                    feature_types=self.data['feature_types'],
                    genomes=self.data['genomes'],
                    barcodes=self.data['barcodes'],
                    inferred_count_matrix=inferred_count_matrix,
                    cell_barcode_inds=self.analyzed_barcode_inds,
                    ambient_expression=ambient_expression,
                    z=z, d=d, p=p, phi=phi_params, epsilon=epsilon,
                    rho=cellbender.remove_background.model.get_rho(),
                    fpr=fpr,
                    lambda_multiplier=self.posterior.lambda_multiplier,
                    loss=inferred_model.loss)

                # Generate filename for filtered matrix output.
                filtered_output_file = os.path.join(file_dir, file_name + f"_FPR_{fpr}_filtered.h5")

                # Write filtered matrix (cells only) to output file.
                if self.model_name != "simple":
                    cell_barcode_inds = \
                        self.analyzed_barcode_inds[filtered_inds_of_analyzed_barcodes]

                    cell_barcodes = self.data['barcodes'][cell_barcode_inds]

                    write_matrix_to_cellranger_h5(
                        cellranger_version=cellranger_version,
                        output_file=filtered_output_file,
                        gene_names=self.data['gene_names'],
                        gene_ids=self.data['gene_ids'],
                        feature_types=self.data['feature_types'],
                        genomes=self.data['genomes'],
                        barcodes=cell_barcodes,
                        inferred_count_matrix=inferred_count_matrix[cell_barcode_inds, :],
                        cell_barcode_inds=None,
                        ambient_expression=ambient_expression,
                        z=z[filtered_inds_of_analyzed_barcodes, :],
                        d=d[filtered_inds_of_analyzed_barcodes],
                        p=p[filtered_inds_of_analyzed_barcodes],
                        phi=phi_params,
                        epsilon=epsilon[filtered_inds_of_analyzed_barcodes],
                        rho=cellbender.remove_background.model.get_rho(),
                        fpr=fpr,
                        lambda_multiplier=self.posterior.lambda_multiplier,
                        loss=inferred_model.loss)

        # Save barcodes determined to contain cells as _cell_barcodes.csv
        try:
            barcode_names = np.array([str(cell_barcodes[i],
                                          encoding='UTF-8')
                                      for i in range(cell_barcodes.size)])
        except UnicodeDecodeError:
            # necessary if barcodes are ints
            barcode_names = cell_barcodes
        except TypeError:
            # necessary if barcodes are already decoded
            barcode_names = cell_barcodes
        bc_file_name = os.path.join(file_dir,
                                    file_name + "_cell_barcodes.csv")
        np.savetxt(bc_file_name, barcode_names, delimiter=',', fmt='%s')
        logging.info(f"Saved cell barcodes in {bc_file_name}")

        try:
            # Save plots, if called for.
            if save_plots:
                plt.figure(figsize=(6, 18))

                # Plot the train error.
                plt.subplot(3, 1, 1)
                plt.plot(inferred_model.loss['train']['elbo'], '.--',
                         label='Train')

                # Plot the test error, if there was held-out test data.
                if 'test' in inferred_model.loss.keys():
                    if len(inferred_model.loss['test']['epoch']) > 0:
                        plt.plot(inferred_model.loss['test']['epoch'],
                                 inferred_model.loss['test']['elbo'], 'o:',
                                 label='Test')
                        plt.legend()

                plt.gca().set_ylim(bottom=max(inferred_model.loss['train']
                                              ['elbo'][0],
                                              inferred_model.loss['train']
                                              ['elbo'][-1] - 2000))
                plt.xlabel('Epoch')
                plt.ylabel('ELBO')
                plt.title('Progress of the training procedure')

                # Plot the barcodes used, along with the inferred
                # cell probabilities.
                plt.subplot(3, 1, 2)
                count_mat = self.get_count_matrix()
                counts = np.array(count_mat.sum(axis=1)).squeeze()
                count_order = np.argsort(counts)[::-1]
                plt.semilogy(counts[count_order], color='black')
                plt.ylabel('UMI counts')
                plt.xlabel('Barcode index, sorted by UMI count')
                if p is not None:  # The case of a simple model.
                    plt.gca().twinx()
                    plt.plot(p[count_order], '.:', color='red', alpha=0.3)
                    plt.ylabel('Cell probability', color='red')
                    plt.ylim([-0.05, 1.05])
                    plt.title('Determination of which barcodes contain cells')
                else:
                    plt.title('The subset of barcodes used for training')

                # Plot the latent encoding via PCA.
                plt.subplot(3, 1, 3)
                pca = PCA(n_components=2)
                if p is None:
                    p = np.ones_like(d)
                z_pca = pca.fit_transform(z[p >= 0.5])
                plt.plot(z_pca[:, 0], z_pca[:, 1],
                         '.', ms=3, color='black', alpha=0.3)
                plt.ylabel('PC 1')
                plt.xlabel('PC 0')
                plt.title('PCA of latent encoding of cell gene expression')

                file_dir, file_base = os.path.split(output_file)
                file_name = os.path.splitext(os.path.basename(file_base))[0]
                fig_name = os.path.join(file_dir, file_name + ".pdf")
                plt.savefig(fig_name, bbox_inches='tight', format='pdf')
                logging.info(f"Saved summary plots as {fig_name}")

        except Exception:
            logging.warning("Unable to save plot.")

        return write_succeeded


def detect_cellranger_version_mtx(filedir: str) -> int:
    """Detect which version of CellRanger (2 or 3) created this mtx directory.

    Args:
        filedir: string path to .mtx file that contains the raw gene
            barcode matrix in a sparse coo text format.

    Returns:
        CellRanger version, either 2 or 3, as an integer.

    """

    assert os.path.isdir(filedir), f"The directory {filedir} is not accessible."

    if os.path.isfile(os.path.join(filedir, 'features.tsv.gz')):
        return 3

    else:
        return 2


def get_matrix_from_cellranger_mtx(filedir: str) \
        -> Dict[str, Union[sp.csr.csr_matrix, List[np.ndarray], np.ndarray]]:
    """Load a count matrix from an mtx directory from CellRanger's output.

    For CellRanger v2:
    The directory must contain three files:
        matrix.mtx
        barcodes.tsv
        genes.tsv

    For CellRanger v3:
    The directory must contain three files:
        matrix.mtx.gz
        barcodes.tsv.gz
        features.tsv.gz

    This function returns a dictionary that includes the count matrix, the gene
    names (which correspond to columns of the count matrix), and the barcodes
    (which correspond to rows of the count matrix).

    Args:
        filedir: string path to .mtx file that contains the raw gene
            barcode matrix in a sparse coo text format.

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string Ensembl ID of genes in the genome, which
             also correspond to the columns in the out['matrix'].
        out['feature_types']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string feature types of genes (or possibly
             antibody capture reads), which also correspond to the columns
             in the out['matrix'].

    """

    assert os.path.isdir(filedir), "The directory {filedir} is not accessible."

    # Decide whether data is CellRanger v2 or v3.
    cellranger_version = detect_cellranger_version_mtx(filedir=filedir)
    logging.info(f"CellRanger v{cellranger_version} format")

    # CellRanger version 3
    if cellranger_version == 3:

        matrix_file = os.path.join(filedir, 'matrix.mtx.gz')
        gene_file = os.path.join(filedir, 'features.tsv.gz')
        barcode_file = os.path.join(filedir, 'barcodes.tsv.gz')

        # Read in feature names.
        features = np.genfromtxt(fname=gene_file,
                                 delimiter="\t", skip_header=0,
                                 dtype='<U100')

        # Read in gene expression and feature data.
        gene_ids = features[:, 0].squeeze()  # first column
        gene_names = features[:, 1].squeeze()  # second column
        feature_types = features[:, 2].squeeze()  # third column

    # CellRanger version 2
    elif cellranger_version == 2:

        # Read in the count matrix using scipy.
        matrix_file = os.path.join(filedir, 'matrix.mtx')
        gene_file = os.path.join(filedir, "genes.tsv")
        barcode_file = os.path.join(filedir, "barcodes.tsv")

        # Read in gene names.
        gene_data = np.genfromtxt(fname=gene_file,
                                  delimiter="\t", skip_header=0,
                                  dtype='<U100')
        if len(gene_data.shape) == 1:  # custom file format with just gene names
            gene_names = gene_data.squeeze()
            gene_ids = None
        else:  # the 10x CellRanger v2 format with two columns
            gene_names = gene_data[:, 1].squeeze()  # second column
            gene_ids = gene_data[:, 0].squeeze()  # first column
        feature_types = None

    else:
        raise NotImplementedError('MTX format was not identifiable as CellRanger '
                                  'v2 or v3.  Please check 10x Genomics formatting.')

    # For both versions:

    # Read in sparse count matrix.
    count_matrix = io.mmread(matrix_file).tocsr().transpose()

    # Read in barcode names.
    barcodes = np.genfromtxt(fname=barcode_file,
                             delimiter="\t", skip_header=0, dtype='<U20')

    # Issue warnings if necessary, based on dimensions matching.
    if count_matrix.shape[1] != len(gene_names):
        logging.warning(f"Number of gene names in {filedir}/genes.tsv "
                        f"does not match the number expected from the "
                        f"count matrix.")
    if count_matrix.shape[0] != len(barcodes):
        logging.warning(f"Number of barcodes in {filedir}/barcodes.tsv "
                        f"does not match the number expected from the "
                        f"count matrix.")

    return {'matrix': count_matrix,
            'gene_names': gene_names,
            'feature_types': feature_types,
            'gene_ids': gene_ids,
            'genomes': None,  # TODO: check if this info is available in either version
            'barcodes': barcodes}


def detect_cellranger_version_h5(filename: str) -> int:
    """Detect which version of CellRanger (2 or 3) created this h5 file.

    Args:
        filename: string path to .mtx file that contains the raw gene
            barcode matrix in a sparse coo text format.

    Returns:
        version: CellRanger version, either 2 or 3, as an integer.

    """

    with tables.open_file(filename, 'r') as f:

        # For CellRanger v2, each group in the table (other than root)
        # contains a genome.
        # For CellRanger v3, there is a 'matrix' group that contains 'features'.

        version = 2

        try:

            # This works for version 3 but not for version 2.
            getattr(f.root.matrix, 'features')
            version = 3

        except tables.NoSuchNodeError:
            pass

    return version


def get_matrix_from_cellranger_h5(filename: str) \
        -> Dict[str, Union[sp.csr.csr_matrix, List[np.ndarray], np.ndarray]]:
    """Load a count matrix from an h5 file from CellRanger's output.

    The file needs to be a _raw_gene_bc_matrices_h5.h5 file.  This function
    returns a dictionary that includes the count matrix, the gene names (which
    correspond to columns of the count matrix), and the barcodes (which
    correspond to rows of the count matrix).

    This function works for CellRanger v2 and v3 HDF5 formats.

    Args:
        filename: string path to .h5 file that contains the raw gene
            barcode matrices

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string Ensembl ID of genes in the genome, which
             also correspond to the columns in the out['matrix'].
        out['feature_types']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string feature types of genes (or possibly
             antibody capture reads), which also correspond to the columns
             in the out['matrix'].

    """

    # Detect CellRanger version.
    cellranger_version = detect_cellranger_version_h5(filename=filename)
    logging.info(f"CellRanger v{cellranger_version} format")

    with tables.open_file(filename, 'r') as f:
        # Initialize empty lists.
        csc_list = []
        barcodes = None
        feature_ids = None
        feature_types = None
        genomes = None

        # CellRanger v2:
        # Each group in the table (other than root) contains a genome,
        # so walk through the groups to get data for each genome.
        if cellranger_version == 2:

            feature_names = []
            feature_ids = []

            for group in f.walk_groups():
                try:
                    # Read in data for this genome, and put it into a
                    # scipy.sparse.csc.csc_matrix
                    barcodes = getattr(group, 'barcodes').read()
                    data = getattr(group, 'data').read()
                    indices = getattr(group, 'indices').read()
                    indptr = getattr(group, 'indptr').read()
                    shape = getattr(group, 'shape').read()
                    csc_list.append(sp.csc_matrix((data, indices, indptr),
                                                  shape=shape))
                    feature_names.extend(getattr(group, 'gene_names').read())
                    feature_ids.extend(getattr(group, 'genes').read())

                except tables.NoSuchNodeError:
                    # This exists to bypass the root node, which has no data.
                    pass

            # Create numpy arrays.
            feature_names = np.array(feature_names)
            if len(feature_ids) > 0:
                feature_ids = np.array(feature_ids)
            else:
                feature_ids = None

        # CellRanger v3:
        # There is only the 'matrix' group.
        elif cellranger_version == 3:

            # Read in data for this genome, and put it into a
            # scipy.sparse.csc.csc_matrix
            barcodes = getattr(f.root.matrix, 'barcodes').read()
            data = getattr(f.root.matrix, 'data').read()
            indices = getattr(f.root.matrix, 'indices').read()
            indptr = getattr(f.root.matrix, 'indptr').read()
            shape = getattr(f.root.matrix, 'shape').read()
            csc_list.append(sp.csc_matrix((data, indices, indptr),
                                          shape=shape))

            # Read in 'feature' information
            feature_group = f.get_node(f.root.matrix, 'features')
            feature_names = getattr(feature_group, 'name').read()

            try:
                feature_types = getattr(feature_group, 'feature_type').read()
            except tables.NoSuchNodeError:
                # This exists in case someone produced a file without feature_type.
                pass
            try:
                feature_ids = getattr(feature_group, 'id').read()
            except tables.NoSuchNodeError:
                # This exists in case someone produced a file without feature id.
                pass
            try:
                genomes = getattr(feature_group, 'genome').read()
            except tables.NoSuchNodeError:
                # This exists in case someone produced a file without feature genome.
                pass

    # Put the data together (possibly from several genomes for v2 datasets).
    count_matrix = sp.vstack(csc_list, format='csc')
    count_matrix = count_matrix.transpose().tocsr()

    # Issue warnings if necessary, based on dimensions matching.
    if count_matrix.shape[1] != feature_names.size:
        logging.warning(f"Number of gene names ({feature_names.size}) in {filename} "
                        f"does not match the number expected from the count "
                        f"matrix ({count_matrix.shape[1]}).")
    if count_matrix.shape[0] != barcodes.size:
        logging.warning(f"Number of barcodes ({barcodes.size}) in {filename} "
                        f"does not match the number expected from the count "
                        f"matrix ({count_matrix.shape[0]}).")

    return {'matrix': count_matrix,
            'gene_names': feature_names,
            'gene_ids': feature_ids,
            'genomes': genomes,
            'feature_types': feature_types,
            'barcodes': barcodes}


def write_matrix_to_cellranger_h5(
        cellranger_version: int,
        output_file: str,
        gene_names: np.ndarray,
        barcodes: np.ndarray,
        inferred_count_matrix: sp.csc.csc_matrix,
        feature_types: Optional[np.ndarray] = None,
        gene_ids: Optional[np.ndarray] = None,
        genomes: Optional[np.ndarray] = None,
        cell_barcode_inds: Optional[np.ndarray] = None,
        ambient_expression: Optional[np.ndarray] = None,
        rho: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
        phi: Optional[np.ndarray] = None,
        epsilon: Optional[np.ndarray] = None,
        fpr: Optional[float] = None,
        lambda_multiplier: Optional[float] = None,
        loss: Optional[Dict] = None) -> bool:
    """Write count matrix data to output HDF5 file using CellRanger format.

    Args:
        cellranger_version: Either 2 or 3. Determines the format of the output
            h5 file.
        output_file: Path to output .h5 file (e.g., 'output.h5').
        gene_names: Name of each gene (column of count matrix).
        gene_ids: Ensembl ID of each gene (column of count matrix).
        genomes: Name of the genome that each gene comes from.
        feature_types: Type of each feature (column of count matrix).
        barcodes: Name of each barcode (row of count matrix).
        inferred_count_matrix: Count matrix to be written to file, in sparse
            format.  Rows are barcodes, columns are genes.
        cell_barcode_inds: Indices into the original cell barcode array that
            were found to contain cells.
        ambient_expression: Vector of gene expression of the ambient RNA
            background counts that contaminate cell counts.
        rho: Hyperparameters for the contamination fraction distribution.
        epsilon: Latent encoding of droplet RT efficiency.
        z: Latent encoding of gene expression.
        d: Latent cell size scale factor.
        p: Latent probability that a barcode contains a cell.
        phi: Latent global overdispersion mean and scale.
        fpr: Target false positive rate for the regularized posterior denoised
            counts, where false positives are true counts that are (erroneously)
            removed.
        lambda_multiplier: The lambda multiplier value used to achieve the
            targeted false positive rate.
        loss: Training and test error, as ELBO, for each epoch.

    Note:
        To match the CellRanger .h5 files, the matrix is stored as its
        transpose, with rows as genes and cell barcodes as columns.

    """

    assert isinstance(inferred_count_matrix,
                      sp.csc_matrix), "The count matrix must be csc_matrix " \
                                      "format in order to write to HDF5."

    assert gene_names.size == inferred_count_matrix.shape[1], \
        "The number of gene names must match the number of columns in the " \
        "count matrix."

    if gene_ids is not None:
        assert gene_names.size == gene_ids.size, \
            f"The number of gene_names {gene_names.shape} must match " \
            f"the number of gene_ids {gene_ids.shape}."

    if feature_types is not None:
        assert gene_names.size == feature_types.size, \
            f"The number of gene_names {gene_names.shape} must match " \
            f"the number of feature_types {feature_types.shape}."

    if genomes is not None:
        assert gene_names.size == genomes.size, \
            "The number of gene_names must match the number of genome designations."

    assert barcodes.size == inferred_count_matrix.shape[0], \
        "The number of barcodes must match the number of rows in the count" \
        "matrix."

    # This reverses the role of rows and columns, to match CellRanger format.
    inferred_count_matrix = inferred_count_matrix.transpose().tocsc()

    # Write to output file.
    try:
        with tables.open_file(output_file, "w",
                              title="CellBender remove-background output") as f:

            if cellranger_version == 2:

                # Create the group where data will be stored.
                group = f.create_group("/", "background_removed",
                                       "Counts after background correction")

                # Create arrays within that group for gene info.
                f.create_array(group, "gene_names", gene_names)
                if gene_ids is not None:
                    f.create_array(group, "genes", gene_ids)

            elif cellranger_version == 3:

                # Create the group where data will be stored: name is "matrix".
                group = f.create_group("/", "matrix",
                                       "Counts after background correction")

                # Create a sub-group called "features"
                feature_group = f.create_group(group, "features",
                                               "Genes and other features measured")

                # Create arrays within that group for feature info.
                f.create_array(feature_group, "name", gene_names)
                if gene_ids is not None:
                    f.create_array(feature_group, "id", gene_ids)
                if feature_types is not None:
                    f.create_array(feature_group, "feature_type", feature_types)
                if genomes is not None:
                    f.create_array(feature_group, "genome", genomes)

                # Copy the other extraneous information from the input file.
                # (Some user might need it for some reason.)
                # TODO

            else:
                raise NotImplementedError(f'Trying to save to CellRanger '
                                          f'v{cellranger_version} format, which '
                                          f'is not implemented.')

            # Code for both versions.
            f.create_array(group, "barcodes", barcodes)

            # Create arrays to store the count data.
            f.create_array(group, "data", inferred_count_matrix.data)
            f.create_array(group, "indices", inferred_count_matrix.indices)
            f.create_array(group, "indptr", inferred_count_matrix.indptr)
            f.create_array(group, "shape", inferred_count_matrix.shape)

            # Store background gene expression, barcode_inds, z, d, and p.
            if cell_barcode_inds is not None:
                f.create_array(group, "barcode_indices_for_latents",
                               cell_barcode_inds)
            if ambient_expression is not None:
                f.create_array(group, "ambient_expression", ambient_expression)
            if z is not None:
                f.create_array(group, "latent_gene_encoding", z)
            if d is not None:
                f.create_array(group, "latent_scale", d)
            if p is not None:
                f.create_array(group, "latent_cell_probability", p)
            if phi is not None:
                f.create_array(group, "overdispersion_mean_and_scale", phi)
            if rho is not None:
                f.create_array(group, "contamination_fraction_params", rho)
            if epsilon is not None:
                f.create_array(group, "latent_RT_efficiency", epsilon)
            if fpr is not None:
                f.create_array(group, "target_false_positive_rate", fpr)
            if lambda_multiplier is not None:
                f.create_array(group, "lambda_multiplier", lambda_multiplier)
            if loss is not None:
                f.create_array(group, "training_elbo_per_epoch",
                               np.array(loss['train']['elbo']))
                if 'test' in loss.keys():
                    f.create_array(group, "test_elbo",
                                   np.array(loss['test']['elbo']))
                    f.create_array(group, "test_epoch",
                                   np.array(loss['test']['epoch']))
                    f.create_array(group, "fraction_data_used_for_testing",
                                   1. - consts.TRAINING_FRACTION)

        logging.info(f"Succeeded in writing CellRanger v{cellranger_version} "
                     f"format output to file {output_file}")

        return True

    except Exception:
        logging.warning(f"Encountered an error writing output to file "
                        f"{output_file}.  "
                        f"Output may be incomplete.")

        return False


def get_d_priors_from_dataset(dataset: SingleCellRNACountsDataset) \
        -> Tuple[float, float]:
    """Compute an estimate of reasonable priors on cell size and ambient size.

    Given a dataset (scipy.sparse.csr matrix of counts where
    rows are barcodes and columns are genes), and an expected
    cell count, compute an estimate of reasonable priors on cell size
    and ambient count size.  This is done by a series of heuristics.

    Args:
        dataset: Dataset object containing a matrix of unique UMI counts,
            where rows are barcodes and columns are genes.

    Returns:
        cell_counts: Estimated mean number of UMI counts per real cell, in
            terms of transformed count data.
        empty_counts: Estimated mean number of UMI counts per 'empty'
            droplet, in terms of transformed count data.

    NOTE: Picks barcodes using cutoffs in untransformed count data.  The output
    is in terms of transformed counts.

    """

    # Count the total unique UMIs per barcode (summing after transforming).
    counts = np.array(dataset.data['matrix']
                      [:, dataset.analyzed_gene_inds].sum(axis=1)).squeeze()

    # If it's a model that does not model empty droplets, the dataset is cells.
    if dataset.model_name == 'simple':

        if dataset.priors['n_cells'] is None:
            # No prior on number of cells.  Assume all are cells.
            dataset.priors['n_cells'] = int(np.sum(counts > 0).item())

        # Sort order the cells by counts.
        sort_order = np.argsort(counts)[::-1]

        # Estimate cell count by median, taking 'cells' to be largest counts.
        cell_counts = int(np.median(counts[sort_order]
                                    [:dataset.priors['n_cells']]).item())

        empty_counts = 0

    # Models that include both cells and empty droplets.
    else:

        # Cutoff for original data.  Empirical.
        cut = dataset.low_count_threshold

        # Estimate the number of UMI counts in empty droplets.

        # Mode of (rounded) log counts (for counts > cut) is a robust
        # empty estimator.
        empty_log_counts = mode(np.round(np.log1p(counts[counts > cut]),
                                         decimals=1))[0]
        empty_counts = int(np.expm1(empty_log_counts).item())

        # Estimate the number of UMI counts in cells.

        # Use expected cells if it is available.
        if dataset.priors['n_cells'] is not None:

            # Sort order the cells by counts.
            sort_order = np.argsort(counts)[::-1]

            cell_counts = int(np.median(counts[sort_order]
                                        [:dataset.priors['n_cells']]).item())

        else:

            # Median of log counts above 5 * empty counts is a robust
            # cell estimator.
            cell_log_counts = np.median(np.log1p(counts[counts > 5 * empty_counts]))
            cell_counts = int(np.expm1(cell_log_counts).item())

        logging.info(f"Prior on counts in empty droplets is {empty_counts}")

    logging.info(f"Prior on counts for cells is {cell_counts}")

    return cell_counts, empty_counts


def estimate_cell_count_from_dataset(dataset: SingleCellRNACountsDataset) \
        -> int:
    """Compute an estimate of number of real cells in a dataset.

    Given a Dataset, compute an estimate of the number of real cells.

    Args:
        dataset: Dataset object containing a matrix of unique UMI counts,
            where rows are barcodes and columns are genes.

    Returns:
        cell_count_est: Estimated number of real cells.

    """

    # If it's a model that does not model empty droplets, the dataset is cells.
    # NOTE: this is overridden if --expected_cells is specified.
    if dataset.model_name == 'simple':
        return dataset.data['matrix'].shape[0]

    # Count number of UMIs in each barcode.
    counts = np.array(dataset.data['matrix'].sum(axis=1),
                      dtype=int).squeeze()

    # Find mid-way between cell_counts and empty_counts in log space.
    midway = np.mean([np.log1p(dataset.priors['cell_counts']),
                      np.log1p(dataset.priors['empty_counts'])])
    umi_cutoff = np.expm1(midway)

    # Count the number of barcodes with UMI counts above the cutoff.
    cell_count_est = int(np.sum(counts > umi_cutoff).item())

    return cell_count_est


def estimate_chi_ambient_from_dataset(dataset: SingleCellRNACountsDataset) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute an estimate of ambient RNA levels.

    Given a Dataset, compute an estimate of the ambient gene expression and
    compute the average gene expression.

    Args:
        dataset: Dataset object containing a matrix of unique UMI counts,
            where rows are barcodes and columns are genes.

    Returns:
        chi_ambient_init: Estimated number of real cells.
        chi_bar: Average gene expression over dataset.

    NOTE: This must be done on transformed data.

    """

    # Ensure that an estimate of the log count crossover point between cells
    # and empty droplets has already been calculated.
    try:
        log_crossover = dataset.priors['log_counts_crossover']
    except KeyError:
        raise AssertionError("Could not find dataset parameter "
                             "log_counts_crossover.")

    ep = np.finfo(np.float32).eps.item()  # Small value

    # Trimmed and appropriately transformed count matrix.
    count_matrix = dataset.get_count_matrix()

    # Empty droplets have log counts < log_crossover.
    empty_barcodes = (np.log1p(np.array(count_matrix.sum(axis=1)).squeeze())
                      < log_crossover)

    # Sum gene expression for the empty droplets.
    gene_expression = np.array(count_matrix[empty_barcodes, :].sum(axis=0))\
        .squeeze()

    # As a vector on a simplex.
    gene_expression = gene_expression + ep
    chi_ambient_init = \
        torch.Tensor(gene_expression / np.sum(gene_expression))

    # Full count matrix, appropriately transformed.
    full_count_matrix = dataset.get_count_matrix_all_barcodes()

    # Sum all gene expression.
    gene_expression_total = np.array(full_count_matrix.sum(axis=0)).squeeze()

    # As a vector on a simplex.
    gene_expression_total = gene_expression_total + ep
    chi_bar = \
        torch.Tensor(gene_expression_total / np.sum(gene_expression_total))

    return chi_ambient_init, chi_bar
