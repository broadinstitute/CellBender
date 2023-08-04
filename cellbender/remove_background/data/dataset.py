"""Class and functions for working with a count matrix dataset."""

import numpy as np
import scipy.sparse as sp
import torch

import cellbender.remove_background.consts as consts
from cellbender.remove_background.data.dataprep import DataLoader
from cellbender.remove_background.data.io import load_data
from cellbender.remove_background.data.priors import get_priors, \
    get_cell_count_given_expected_cells, \
    get_empty_count_given_expected_cells_and_total_droplets, \
    compute_crossover_surely_empty_and_stds
from cellbender.remove_background.sparse_utils import csr_set_rows_to_zero, \
    overwrite_matrix_with_columns_from_another

from typing import Dict, List, Optional, Iterable, Callable
import logging
import argparse


logger = logging.getLogger('cellbender')


class SingleCellRNACountsDataset:
    """Object for storing scRNA-seq count matrix data and basic manipulations
    and pre-processing (e.g. estimation of prior hyperparameters).

    Args:
        input_file: Input data file path.
        expected_cell_count: Expected number of real cells a priori.
        total_droplet_barcodes: Total number of droplets to include in the
            cell-calling analysis.
        force_cell_umi_prior: User wants to force cell UMI prior to be this
        force_empty_umi_prior: User wants to force empty UMI prior to be this
        ambient_counts_in_cells_low_limit: Limit to determine how many features
            are included in the analysis
        model_name: Model to use.
        gene_blacklist: List of integer indices of genes to exclude entirely.
        exclude_features: List of feature types to exclude from the analysis.
            Must be in ['Gene Expression', 'Antibody Capture',
            'CRISPR Guide Capture', 'Custom']
        low_count_threshold: Droplets with UMI counts below this number are
            excluded entirely from the analysis.
        fpr: Target expected false positive rate.

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
        posterior: BasePosterior estimated after inference.
        empty_UMI_threshold: This UMI count is the maximum UMI count in the
            user-defined surely empty droplets.

    Note: Count data is kept as the original, untransformed data.  Priors are
    in terms of the transformed count data.

    """

    def __init__(self,
                 input_file: str,
                 model_name: str,
                 exclude_features: List[str],
                 low_count_threshold: int,
                 fpr: List[float],
                 expected_cell_count: Optional[int] = None,
                 total_droplet_barcodes: Optional[int] = None,
                 force_cell_umi_prior: Optional[float] = None,
                 force_empty_umi_prior: Optional[float] = None,
                 fraction_empties: Optional[float] = None,
                 ambient_counts_in_cells_low_limit: float = consts.AMBIENT_COUNTS_IN_CELLS_LOW_LIMIT,
                 gene_blacklist: List[int] = []):
        assert input_file is not None, "Attempting to load data, but no " \
                                       "input file was specified."
        self.input_file = input_file
        self.analyzed_barcode_inds = np.array([])  # Barcodes trained each epoch
        self.empty_barcode_inds = np.array([])  # Barcodes sampled randomly each epoch
        self.low_count_cutoff = low_count_threshold
        self.data = None
        self.gmm = None
        self.exclude_features = exclude_features
        self.model_name = model_name
        self.fraction_empties = fraction_empties
        self.is_trimmed = False
        self.low_count_threshold = low_count_threshold
        self.ambient_counts_in_cells_low_limit = ambient_counts_in_cells_low_limit
        self.priors = {}
        self.posterior = None
        self.fpr = fpr

        # Are empty droplets included in this model?
        self.include_empties = False if (self.model_name in ['simple']) else True

        # Load the dataset.
        self.data = load_data(self.input_file)
        self.analyzed_gene_logic = True

        # Eliminate feature types not used in the analysis.
        self._exclude_feature_types()

        # Eliminate zero-count features and blacklisted features.
        self._clean_features(gene_blacklist=gene_blacklist)

        # Estimate priors.
        counts = np.array(self.data['matrix']
                          [:, self.analyzed_gene_logic].sum(axis=1)).squeeze()
        self.priors = get_priors(umi_counts=counts, low_count_threshold=low_count_threshold)

        # Overwrite heuristic priors with user inputs.
        if expected_cell_count is not None:
            logger.debug(f'Fixing expected_cells at {expected_cell_count}')
            self.priors['expected_cells'] = expected_cell_count
            self.priors.update(
                get_cell_count_given_expected_cells(
                    umi_counts=counts, expected_cells=expected_cell_count,
                )
            )
            if (expected_cell_count + consts.NUM_EMPTIES_INCREMENT) > self.priors['total_droplets']:
                # Bump this up to avoid an immediate error
                total_drops = expected_cell_count + consts.NUM_EMPTIES_INCREMENT
                self.priors['total_droplets'] = total_drops
                logger.debug(f'Incrementing total_droplets to be {total_drops}')
                if total_droplet_barcodes is None:
                    # If this isn't getting recomputed next, recompute now
                    self.priors.update(
                        get_empty_count_given_expected_cells_and_total_droplets(
                            umi_counts=counts, expected_cells=self.priors['expected_cells'],
                            total_droplets=total_drops,
                        )
                    )
        if total_droplet_barcodes is not None:
            logger.debug(f'Fixing total_droplets at {total_droplet_barcodes}')
            self.priors['total_droplets'] = total_droplet_barcodes
            self.priors.update(
                get_empty_count_given_expected_cells_and_total_droplets(
                    umi_counts=counts, expected_cells=self.priors['expected_cells'],
                    total_droplets=total_droplet_barcodes,
                )
            )

        # Force priors if user elects to do so.
        if force_cell_umi_prior is not None:
            logger.debug(f'Forcing cell UMI count prior to be {force_cell_umi_prior}')
            self.priors['cell_counts'] = force_cell_umi_prior
        if force_empty_umi_prior is not None:
            logger.debug(f'Forcing empty droplet UMI count prior to be {force_empty_umi_prior}')
            self.priors['empty_counts'] = force_empty_umi_prior
            middle = np.sqrt(self.priors['cell_counts'] * force_empty_umi_prior)
            self.priors['empty_count_upper_limit'] = min(middle, 2 * force_empty_umi_prior)

        # Recompute a few quantities if some things were replaced by user input.
        compute_crossover_surely_empty_and_stds(umi_counts=counts, priors=self.priors)
        logger.info(f"Prior on counts for cells is {int(self.priors['cell_counts'])}")
        logger.info(f"Prior on counts for empty droplets is {int(self.priors['empty_counts'])}")
        logger.debug('\n'.join(['Priors:'] + [f'{k}: {v}' for k, v in self.priors.items()]))

        # Do not analyze features which are not expected to contribute to noise.
        self._trim_noiseless_features()
        self.analyzed_gene_inds = np.where(self.analyzed_gene_logic)[0].astype(dtype=int)
        logger.info(f"Including {len(self.analyzed_gene_inds)} features in the analysis.")

        # Determine barcodes to be analyzed.
        self._trim_droplets()
        self.is_trimmed = True

        # Estimate gene expression priors.
        self.priors.update(self._estimate_chi_ambient())

        logger.debug('\n'.join(['Priors:'] + [f'{k}: {v}' for k, v in self.priors.items()]))

    def _exclude_feature_types(self):
        """Exclude feature types as specified by user.
        Sets the value of self.analyzed_gene_logic
        """

        # Skip if feature types are not specified.
        if self.data['feature_types'] is None:
            if len(self.exclude_features) > 0:
                logger.warning(f"WARNING: specified --exclude-feature-types "
                               f"{self.exclude_features} but no feature_type "
                               f"information was found in the input file. Proceeding "
                               f"without excluding any features. If this is not the "
                               f"intended outcome, please check your input file "
                               f"format.")
            else:
                logger.info('No feature type information specified in the input '
                            'file. Including all features in the analysis.')
            return

        # Feature types are specified.
        dtype = type(self.data['feature_types'][0])
        is_bytestring = (dtype != str) and (dtype != np.str_)

        def convert(s):
            if is_bytestring:
                return str(s, encoding='utf-8')
            return s

        feature_type_array = np.array([convert(f) for f in self.data['feature_types']], dtype=str)
        feature_types = np.unique(feature_type_array)
        feature_info = [f"{(feature_type_array == f).sum()} {f}" for f in feature_types]
        logger.info(f"Features in dataset: {', '.join(feature_info)}")

        # Keep track of a logical array for which features to include.
        inclusion_logic = np.ones(len(feature_type_array), dtype=bool)  # all True

        for feature in self.exclude_features:
            logger.info(f"Excluding {feature} features (output will equal input).")
            logic = np.array([f not in self.exclude_features
                              for f in feature_type_array], dtype=bool)
            inclusion_logic = np.logical_and(inclusion_logic, logic)
            logger.info(f"    - This results in the exclusion of "
                        f"{len(feature_type_array) - logic.sum()} "
                        f"features.")

        # Update self.analyzed_gene_logic
        self.analyzed_gene_logic = np.logical_and(
            self.analyzed_gene_logic,
            inclusion_logic,
        )

    def _clean_features(self, gene_blacklist: List[int] = []):
        """Trim the dataset by removing zero-count and blacklisted features.
        Sets the value of self.analyzed_gene_logic

        Args:
            gene_blacklist: List of gene indices to trim out and exclude.
        """

        logger.info("Trimming features for inference.")

        # Get data matrix and barcode order that sorts barcodes by UMI count.
        matrix = self.data['matrix']

        # Choose which genes to use based on their having nonzero counts.
        # (All barcodes must be included so that inference can generalize.)
        feature_counts_per_barcode = np.array(matrix.sum(axis=0)).squeeze()
        inclusion_logic = (feature_counts_per_barcode > 0)

        # Update self.analyzed_gene_inds
        self.analyzed_gene_logic = np.logical_and(self.analyzed_gene_logic, inclusion_logic)

        # Remove blacklisted genes.
        if len(gene_blacklist) > 0:
            gene_blacklist = set(gene_blacklist)
            inclusion_logic = np.array([g not in gene_blacklist
                                        for g in range(len(self.analyzed_gene_logic))])
            logger.info(f"Ignoring {np.logical_not(inclusion_logic).sum()} "
                        f"specified features.")
            self.analyzed_gene_logic = np.logical_and(
                self.analyzed_gene_logic,
                inclusion_logic,
            )

        if self.analyzed_gene_logic.sum() == 0:
            logger.warning("No features remain after eliminating zero-count "
                           "and ignored features. Terminating analysis.")
            raise AssertionError("No features with nonzero counts remain. Check "
                                 "the input data file and check "
                                 "--exclude-feature-types and --ignore-features")

        logger.info(f"{self.analyzed_gene_logic.sum()} features have nonzero counts.")

    def _trim_noiseless_features(self):
        """Trim the dataset for inference, choosing genes to use.

        To be run after trimming features first, then trimming barcodes.

        Sets the value of self.analyzed_gene_inds
        """

        assert len(self.priors.keys()) > 0, 'Run self.priors = get_priors() before ' \
                                            'self._trim_noiseless_features()'

        # Find average counts per gene in empty droplets.
        count_matrix = self.data['matrix'][:, self.analyzed_gene_logic]
        counts = np.array(count_matrix.sum(axis=1)).squeeze()
        cutoff = self.priors['empty_count_upper_limit']
        count_matrix_empties = count_matrix[(counts < cutoff)
                                            & (counts > self.low_count_threshold), :]
        mean_counts_per_empty_g = np.array(count_matrix_empties.mean(axis=0)).squeeze()

        # Estimate counts in cells, but use a minimum number of cells just as a floor.
        ambient_counts_in_cells_g = (max(self.priors['expected_cells'], 5000)
                                     * mean_counts_per_empty_g)

        # Include features that are projected to have noise above the low limit.
        inclusion_logic_subset = (ambient_counts_in_cells_g
                                  > self.ambient_counts_in_cells_low_limit)
        inclusion_logic = np.ones(len(self.analyzed_gene_logic), dtype=bool)
        inclusion_logic[self.analyzed_gene_logic] = inclusion_logic_subset

        # Record in self.analyzed_gene_logic
        self.analyzed_gene_logic = np.logical_and(
            self.analyzed_gene_logic,
            inclusion_logic,
        )
        logger.info(f"Excluding {np.logical_not(inclusion_logic).sum()} features that "
                    f"are estimated to have <= {self.ambient_counts_in_cells_low_limit} "
                    f"background counts in cells.")

    def _trim_droplets(self):
        """Trim the dataset for inference, choosing barcodes to use.

        Sets the values of self.analyzed_barcode_inds, and
        self.empty_barcode_inds, which are used throughout training.
        """

        logger.info("Trimming barcodes for inference.")

        # Get data matrix and barcode order that sorts barcodes by UMI count.
        matrix = self.get_count_matrix_all_barcodes()  # self.data['matrix']
        umi_counts = np.array(matrix.sum(axis=1)).squeeze()
        # umi_counts_features_trimmed = self.get_count_matrix_all_barcodes()
        umi_count_order = np.argsort(umi_counts)[::-1]
        n_cells = self.priors['expected_cells']
        total_droplet_barcodes = self.priors['total_droplets']

        # If running the simple model, just use the expected cells, no more.
        if not self.include_empties:

            self.analyzed_barcode_inds = np.array(umi_count_order[:n_cells],
                                                  dtype=int)

            logger.info(f"Excluding empty droplets due to '{self.model_name}' model: "
                        f"using {self.analyzed_barcode_inds.size} cell barcodes.")

        # If not using the simple model, include empty droplets.
        else:

            assert total_droplet_barcodes - n_cells > 0, \
                f"The number of cells is {n_cells}, but the " \
                f"number of total droplets included is " \
                f"{total_droplet_barcodes}.  Increase " \
                f"--total_droplet_barcodes above {n_cells}, or " \
                f"specify a different number of expected cells using " \
                f"--expected-cells"

            # Set the low UMI count cutoff to be the greater of either
            # the user input value, or an empirically-derived value.
            factor = consts.EMPIRICAL_LOW_UMI_TO_EMPTY_DROPLET_THRESHOLD
            empirical_low_count_cutoff = int(self.priors['empty_counts'] * factor)
            low_count_cutoff = max(self.low_count_threshold, empirical_low_count_cutoff)
            self.low_count_cutoff = low_count_cutoff
            logger.info(f"Excluding barcodes with counts below {low_count_cutoff}")

            # See how many barcodes there are to work with total.
            num_barcodes_above_umi_cutoff = np.sum(umi_counts > low_count_cutoff).item()

            assert num_barcodes_above_umi_cutoff > 0, \
                f"There are no barcodes with UMI counts over the lower " \
                f"cutoff of {low_count_cutoff}"

            assert num_barcodes_above_umi_cutoff > n_cells, \
                f"There are no empty droplets with UMI counts over the lower " \
                f"cutoff of {low_count_cutoff}.  Some empty droplets are " \
                f"necessary for the analysis.  Reduce the " \
                f"--low-count-threshold parameter."

            # Use the cell barcodes and transition barcodes for analysis.
            self.analyzed_barcode_inds = umi_count_order[:total_droplet_barcodes]

            # Decide which empty barcodes to include.
            empty_droplet_sorted_barcode_inds = \
                np.arange(total_droplet_barcodes, num_barcodes_above_umi_cutoff,
                          dtype=int)  # The entire range
            self.empty_barcode_inds = umi_count_order[empty_droplet_sorted_barcode_inds]

            # Find the UMI threshold for surely empty droplets.
            self.empty_UMI_threshold = umi_counts[umi_count_order][total_droplet_barcodes]

            # Find the max UMI count for any cell.
            self.max_UMI_count = umi_counts.max()

            # Estimate cell logit probability prior.
            cell_prob = (n_cells / total_droplet_barcodes) * (1. - self.fraction_empties)
            self.priors['cell_logit'] = np.log(cell_prob) - np.log(1. - cell_prob)

            logger.info(f"Using {n_cells} probable cell "
                        f"barcodes, plus an additional "
                        f"{total_droplet_barcodes - n_cells} barcodes, "
                        f"and {len(self.empty_barcode_inds)} empty "
                        f"droplets.")
            logger.info(f"Largest surely-empty droplet has "
                        f"{int(self.empty_UMI_threshold)} UMI counts.")

            if ((low_count_cutoff == self.low_count_threshold)
                    and (len(self.empty_barcode_inds) == 0)):
                logger.warning("Warning: few empty droplets identified. Low "
                               "UMI cutoff may be too high. Check the UMI "
                               "decay curve, and decrease the "
                               "--low-count-threshold parameter if necessary.")

    def _estimate_chi_ambient(self) -> Dict[str, torch.Tensor]:
        """Estimate chi_ambient and chi_bar"""

        matrix = self.data['matrix'].tocsc()
        count_matrix = matrix[:, self.analyzed_gene_inds].tocsr()
        umi_counts = np.array(count_matrix.sum(axis=1)).squeeze()

        # Estimate the ambient gene expression profile.
        ep = np.finfo(np.float32).eps.item()  # small value
        empty_droplet_logic = ((umi_counts < self.priors['surely_empty_counts'])
                               & (umi_counts > self.low_count_cutoff))
        chi_ambient = np.array(count_matrix[empty_droplet_logic, :].sum(axis=0)).squeeze() + ep
        chi_ambient = torch.tensor(chi_ambient / chi_ambient.sum()).float()
        chi_bar = np.array(count_matrix.sum(axis=0)).squeeze() + ep
        chi_bar = torch.tensor(chi_bar / chi_bar.sum()).float()

        return {'chi_ambient': chi_ambient, 'chi_bar': chi_bar}

    def get_count_matrix(self) -> sp.csr_matrix:
        """Get the count matrix, trimmed if trimming has occurred."""

        # Return the count matrix for selected barcodes and genes.
        trimmed_bc_matrix = self.data['matrix'][self.analyzed_barcode_inds,
                                                :].tocsc()
        trimmed_matrix = trimmed_bc_matrix[:, self.analyzed_gene_inds].tocsr()
        return trimmed_matrix

    def get_count_matrix_empties(self) -> sp.csr_matrix:
        """Get the count matrix for empty drops, trimmed if possible."""

        # Return the count matrix for selected barcodes and genes.
        trimmed_bc_matrix = self.data['matrix'][self.empty_barcode_inds,
                                                :].tocsc()
        trimmed_matrix = trimmed_bc_matrix[:, self.analyzed_gene_inds].tocsr()
        return trimmed_matrix

    def get_count_matrix_all_barcodes(self) -> sp.csr_matrix:
        """Get the count matrix, trimming only genes, not barcodes."""

        # Return the count matrix for selected barcodes and genes.
        trimmed_bc_matrix = self.data['matrix'].tocsc()
        trimmed_matrix = trimmed_bc_matrix[:, self.analyzed_gene_inds].tocsr()
        return trimmed_matrix

    def get_dataloader(self,
                       use_cuda: bool = True,
                       batch_size: int = 200,
                       shuffle: bool = False,
                       analyzed_bcs_only: bool = True,
                       sort_by: Optional[Callable[[sp.csr_matrix], float]] = None,
                       ) -> DataLoader:
        """Return a dataloader for the count matrix.

        Args:
            use_cuda: Whether to load data into GPU memory.
            batch_size: Size of minibatch of data yielded by dataloader.
            shuffle: Whether dataloader should shuffle the data.
            analyzed_bcs_only: Only include the barcodes that have been
                analyzed, not the surely empty droplets.
            sort_by: Lambda function which, when applied to the sparse matrix,
                will return values that can be sorted to give a sort order to
                the dataset. Dataloader will load data in order of increasing
                values.

        Returns:
            data_loader: A dataloader that yields the entire dataset in batches.

        """

        if analyzed_bcs_only:
            count_matrix = self.get_count_matrix()
        else:
            count_matrix = self.get_count_matrix_all_barcodes()

        data_loader = DataLoader(
            count_matrix,
            empty_drop_dataset=None,
            batch_size=batch_size,
            fraction_empties=0.,
            shuffle=shuffle,
            sort_by=sort_by,
            use_cuda=use_cuda,
        )
        return data_loader

    def restore_eliminated_features_in_cells(
            self,
            inferred_count_matrix: sp.csc_matrix,
            cell_probabilities_analyzed_bcs: np.ndarray
    ) -> sp.csc_matrix:
        """Add the data back in for any features not used during inference,
        (the used features being defined in self.analyzed_gene_inds),
        so that the output for these features exactly matches the input.

        Args:
            inferred_count_matrix: A sparse count matrix
            cell_probabilities_analyzed_bcs: Cell probabilities for the subset of
                barcodes that were analyzed.

        Returns:
            output: A sparse count matrix, with the unused feature values filled
                in with the raw data.

        """

        # Rescue the raw data for ignored features.
        out = overwrite_matrix_with_columns_from_another(
            mat1=inferred_count_matrix,
            mat2=self.data['matrix'],
            column_inds=self.analyzed_gene_inds)

        # But ensure that empty droplets are empty in the output.
        cell_probabilities_all_bcs = np.zeros(out.shape[0])
        cell_probabilities_all_bcs[self.analyzed_barcode_inds] = cell_probabilities_analyzed_bcs
        empty_inds = np.where(cell_probabilities_all_bcs <= consts.CELL_PROB_CUTOFF)[0]
        out = csr_set_rows_to_zero(csr=out.tocsr(), row_inds=empty_inds)

        return out.tocsc()


def get_dataset_obj(args: argparse.Namespace) -> SingleCellRNACountsDataset:
    """Helper function that uses the argparse namespace"""

    return SingleCellRNACountsDataset(
        input_file=args.input_file,
        expected_cell_count=args.expected_cell_count,
        total_droplet_barcodes=args.total_droplets,
        force_cell_umi_prior=args.force_cell_umi_prior,
        force_empty_umi_prior=args.force_empty_umi_prior,
        fraction_empties=args.fraction_empties,
        model_name=args.model,
        gene_blacklist=args.blacklisted_genes,
        exclude_features=args.exclude_features,
        low_count_threshold=args.low_count_threshold,
        ambient_counts_in_cells_low_limit=args.ambient_counts_in_cells_low_limit,
        fpr=args.fpr,
    )
