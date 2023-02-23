"""Class and functions for working with a count matrix dataset."""

import numpy as np
import scipy.sparse as sp
import pandas as pd
from scipy.stats import gaussian_kde
import torch

import cellbender.remove_background.model
import cellbender.remove_background.consts as consts
# TODO use the new posterior
from cellbender.remove_background.posterior import Posterior
from cellbender.remove_background.data.dataprep import DataLoader
from cellbender.remove_background.data.io import \
    load_data, write_matrix_to_cellranger_h5
from cellbender.remove_background.model import get_param_store_key
from cellbender.remove_background.gmm import GMM
from cellbender.remove_background.report import run_notebook_make_html

from typing import Dict, List, Union, Tuple, Optional, Iterable, Callable
import logging
import os
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # This needs to be after matplotlib.use('Agg')


logger = logging.getLogger('cellbender')


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
        exclude_features: List of feature types to exclude from the analysis.
            Must be in ['Gene Expression', 'Antibody Capture',
            'CRISPR Guide Capture', 'Custom']
        low_count_threshold: Droplets with UMI counts below this number are
            excluded entirely from the analysis.
        fpr: Target expected false positive rate.
        random_seed: Seed for random number generators.

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
                 fraction_empties: Optional[float] = None,
                 ambient_counts_in_cells_low_limit: float = consts.AMBIENT_COUNTS_IN_CELLS_LOW_LIMIT,
                 gene_blacklist: List[int] = []):
        assert input_file is not None, "Attempting to load data, but no " \
                                       "input file was specified."
        self.input_file = input_file
        self.analyzed_barcode_inds = np.array([])  # Barcodes trained each epoch
        self.analyzed_gene_inds = np.array([])
        self.empty_barcode_inds = np.array([])  # Barcodes randomized each epoch
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

        # Trim genes / features.
        self._trim_features(gene_blacklist=gene_blacklist)  # just eliminates blacklist and zeros

        # Estimate priors.
        self._estimate_priors()

        # Determine barcodes to be analyzed.
        if total_droplet_barcodes is None:
            total_droplet_barcodes = self.priors['total_droplet_barcodes']
        self._trim_droplets(total_droplet_barcodes=total_droplet_barcodes,
                            low_UMI_count_cutoff=low_count_threshold,
                            expected_cells=expected_cell_count)
        self.expected_cell_count = expected_cell_count

        # Trim features again, eliminating features not expressed in empties.
        self._trim_features_again()

        # Estimate priors.   # TODO clean this up and don't do everything twice
        self._estimate_priors()
        logger.debug('\n'.join(['Priors:'] + [f'{k}: {v}' for k, v in self.priors.items()]))

    def _trim_features(self, gene_blacklist: List[int] = []):
        """Trim the dataset for inference, choosing genes to use.

        Sets the value of self.analyzed_gene_inds

        Args:
            gene_blacklist: List of gene indices to trim out and exclude.

        """

        logger.info("Trimming dataset for inference.")

        # Get data matrix and barcode order that sorts barcodes by UMI count.
        matrix = self.data['matrix']

        # Initially set the default to be the whole dataset.
        self.analyzed_gene_inds = np.arange(start=0, stop=matrix.shape[1])

        # Exclude features as specified at the command line.
        if self.data['feature_types'] is not None:
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
            inclusion_logic = np.array([f not in self.exclude_features
                                        for f in feature_type_array], dtype=bool)
            if len(self.exclude_features) > 0:
                logger.info(f"Excluding feature type{'' if len(self.exclude_features) == 1 else 's'} "
                            + ", ".join(self.exclude_features) + ".")
        else:
            inclusion_logic = np.ones(matrix.shape[1], dtype=bool)
            logger.info('No feature type information specified in the input file. '
                        'Including all features in the analysis.')

        # Choose which genes to use based on their having nonzero counts.
        # (All barcodes must be included so that inference can generalize.)
        feature_counts_per_barcode = np.array(matrix.sum(axis=0)).squeeze()
        nonzero_feature_counts = (feature_counts_per_barcode > 0)
        self.analyzed_gene_inds = np.where(nonzero_feature_counts
                                           & inclusion_logic)[0].astype(dtype=int)

        if self.analyzed_gene_inds.size == 0:
            logger.warning("During data loading, found no features with > 0 "
                           "counts.  Terminating analysis.")
            raise AssertionError("No features with nonzero counts.  Check the "
                                 "input data file and check --exclude-features")

        # Remove blacklisted genes.
        if len(gene_blacklist) > 0:
            gene_blacklist = set(gene_blacklist)
            self.analyzed_gene_inds = np.array([g for g in
                                                self.analyzed_gene_inds
                                                if g not in gene_blacklist])

        if self.analyzed_gene_inds.size == 0:
            logger.warning("All nonzero genes have been blacklisted.  "
                           "Terminating analysis.")
            raise AssertionError("All genes with nonzero counts have been "
                                 "blacklisted.  Examine the dataset and "
                                 "reduce the blacklist.")

        logger.info(f"{self.analyzed_gene_inds.size} features have nonzero counts.")

    def _trim_features_again(self):
        """Trim the dataset for inference, choosing genes to use.

        To be run after trimming features first, then trimming barcodes.

        Sets the value of self.analyzed_gene_inds
        """

        assert self.is_trimmed, "Trim barcodes before running _trim_features_again()"

        logger.info("Further trimming features for inference.")

        # # Only include genes expressed at nonzero levels in purported empties.
        # count_matrix = self.get_count_matrix_all_barcodes()
        # counts = np.array(count_matrix.sum(axis=-1)).squeeze()
        # count_matrix_empties = \
        #     count_matrix[np.log(counts + consts.NBPC_MU_EPS_SAFEGAURD)
        #                  < self.priors['log_counts_crossover'], :]
        # counts_per_gene_in_empties = np.array(count_matrix_empties.sum(axis=0)).squeeze()
        # self.analyzed_gene_inds = self.analyzed_gene_inds[counts_per_gene_in_empties
        #                                                   > consts.COUNTS_IN_EMPTIES_LOW_LIMIT]  # TODO
        # logger.info(f"Including {self.analyzed_gene_inds.size} features that have"
        #              f" counts > {consts.COUNTS_IN_EMPTIES_LOW_LIMIT} in empty droplets.")

        # Only include genes we would expect to have at least X ambient counts in cells.
        count_matrix = self.get_count_matrix_all_barcodes()
        counts = np.array(count_matrix.sum(axis=1)).squeeze()
        # cutoff = np.exp(self.priors['log_counts_crossover'])
        cutoff = np.exp(np.mean([np.log1p(self.empty_UMI_threshold),
                                 self.priors['log_counts_crossover']]))
        count_matrix_empties = count_matrix[counts < cutoff, :]
        mean_counts_per_empty_g = np.array(count_matrix_empties.mean(axis=0)).squeeze()
        # Estimate counts in cells, but use a minimum number of cells just as a floor.
        ambient_counts_in_cells_g = max(self.priors['n_cells'], 5000) * mean_counts_per_empty_g
        self.analyzed_gene_inds = \
            self.analyzed_gene_inds[ambient_counts_in_cells_g
                                    > self.ambient_counts_in_cells_low_limit]
        logger.info(f"Including {self.analyzed_gene_inds.size} features that "
                    f"are estimated to have > {self.ambient_counts_in_cells_low_limit} "
                    f"background counts in cells.")

    def _trim_droplets(self,
                       total_droplet_barcodes: int,
                       low_UMI_count_cutoff: int,
                       expected_cells: Optional[int] = None):
        """Trim the dataset for inference, choosing barcodes to use.

        Sets the values of self.analyzed_barcode_inds, and
        self.empty_barcode_inds, which are used throughout training.

        Args:
            expected_cells: Number of expected cells input by user.
            total_droplet_barcodes: Number of total droplets to include
                during inference each epoch, and for which to make cell calls.
            low_UMI_count_cutoff: Barcodes with total UMI counts below
                this number are excluded.

        Note:
            User input expected_cells is only used to choose which barcodes to
            include in the analysis, as well as to estimate the sizes of cells
            and empty droplets (elsewhere).  It need only be a reasonable guess.
            The full analysis makes inferences about which barcodes contain
            cells, and ultimately will determine this number.
            However, if running only the 'simple' model, the expected_cell_count
            is taken to be the true number of cells, since empty droplets are
            not part of the 'simple' model.

        """

        logger.info("Trimming barcodes for inference.")

        # Get data matrix and barcode order that sorts barcodes by UMI count.
        matrix = self.data['matrix']
        umi_counts = np.array(matrix.sum(axis=1)).squeeze()
        umi_count_order = np.argsort(umi_counts)[::-1]

        # Initially set the default to be the whole dataset.
        self.analyzed_barcode_inds = np.arange(start=0, stop=matrix.shape[0])

        # Expected cells must not exceed nonzero count barcodes.
        num_nonzero_barcodes = np.sum(umi_counts > 0).item()

        # n_cells is at most the number of nonzero barcodes.
        n_cells = expected_cells if expected_cells is not None else self.priors['n_cells']
        n_cells = min(n_cells, num_nonzero_barcodes)

        assert n_cells > 0, "No cells identified.  Try to use the option " \
                            "--expected-cells to specify a prior on cell " \
                            "count.  Also ensure that --low-count-threshold " \
                            "is not too large (less than empty droplets)."

        # If running the simple model, just use the expected cells, no more.
        if not self.include_empties:

            self.analyzed_barcode_inds = np.array(umi_count_order[:n_cells],
                                                  dtype=int)

            logger.info(f"Excluding empty droplets due to '{self.model_name}' model: "
                        f"using {self.analyzed_barcode_inds.size} cell barcodes.")

        # If not using the simple model, include empty droplets.
        else:

            # Get the cell barcodes.
            cell_barcodes = umi_count_order[:n_cells]

            # Set the low UMI count cutoff to be the greater of either
            # the user input value, or an empirically-derived value.
            factor = consts.EMPIRICAL_LOW_UMI_TO_EMPTY_DROPLET_THRESHOLD
            empirical_low_UMI = int(self.priors['empty_counts'] * factor)
            low_UMI_count_cutoff = max(low_UMI_count_cutoff, empirical_low_UMI)
            logger.info(f"Excluding barcodes with counts below {low_UMI_count_cutoff}")

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
            num_transition_barcodes = (total_droplet_barcodes - cell_barcodes.size)

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
            transition_barcodes = umi_count_order[n_cells:(n_cells + num)]

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
            last_analyzed_bc = min(cell_barcodes.size + transition_barcodes.size - 1,
                                   umi_count_order.size - 1)
            self.empty_UMI_threshold = (umi_counts[umi_count_order][last_analyzed_bc])

            # Find the max UMI count for any cell.
            self.max_UMI_count = umi_counts.max()

            # Estimate cell logit probability prior.
            cell_prob = n_cells / total_droplet_barcodes
            self.priors['cell_logit'] = np.log(cell_prob) - np.log(1. - cell_prob)

            logger.info(f"Using {cell_barcodes.size} probable cell "
                        f"barcodes, plus an additional "
                        f"{transition_barcodes.size} barcodes, "
                        f"and {empty_droplet_barcodes.size} empty "
                        f"droplets.")
            logger.info(f"Largest surely-empty droplet has "
                        f"{self.empty_UMI_threshold} UMI counts.")

            if ((low_UMI_count_cutoff == self.low_count_threshold)
                    and (empty_droplet_barcodes.size == 0)):
                logger.warning("Warning: few empty droplets identified. Low "
                               "UMI cutoff may be too high. Check the UMI "
                               "decay curve, and decrease the "
                               "--low-count-threshold parameter if necessary.")

        self.is_trimmed = True

    def _estimate_priors(self):
        """Estimate relevant priors, populating fields in the self.priors dict.
        Fits a Gaussian mixture model to the total UMI count per droplet data.

        Populates self.priors with the keys:
            n_cells
            total_droplet_barcodes
            cell_prob
            cell_logit
            chi_ambient
            cell_counts
            empty_counts
            log_counts_crossover
            ebayes_cell_count_estimates

        """

        # TODO: handle case where expected_cells is provided

        # Count the total unique UMIs per barcode.
        counts = np.array(self.data['matrix']
                          [:, self.analyzed_gene_inds].sum(axis=1)).squeeze()
        order = np.argsort(counts)[::-1]

        # If it's a model that does not model empty droplets, the dataset is cells.
        if not self.include_empties:

            self.priors['n_cells'] = int(np.sum(counts > self.low_count_threshold).item())
            self.priors['cell_prob'] = 1.
            self.priors['cell_logit'] = 10.  # results in a cell prob near 1
            self.priors['cell_counts'] = np.median(counts[order][:self.priors['n_cells']]).item()
            self.priors['total_droplet_barcodes'] = self.priors['n_cells']

            # Fill in other priors with None.
            for key in ['chi_ambient', 'chi_bar', 'empty_counts', 'log_counts_crossover',
                        'ebayes_cell_count_estimates']:
                self.priors[key] = None

            logger.info(f"Prior on counts for cells is {self.priors['cell_counts']}")
            logger.info(f"Empty droplets ignored in the '{self.model_name}' model.")

        # Models that include both cells and empty droplets.
        else:

            # TODO might add a failsafe in case this fit doesn't go well...

            # Cut off low counts.
            logger.debug(f'Low count threshold is {self.low_count_threshold}')
            counts = counts[counts > self.low_count_threshold]

            # log transform
            log_counts = np.log(counts)
            # calculate range of data, rounding out to make sure we cover everything
            x = np.arange(
                np.floor(log_counts.min()), np.ceil(log_counts.max()) + 0.01, 0.1
            )

            # fit a KDE to estimate density
            k = gaussian_kde(log_counts)
            density = k.evaluate(x)

            # the density peak is almost surely the empty droplets
            log_peak_ind = np.argmax(density)
            log_peak = x[log_peak_ind]
            logger.debug(f"Estimated peak of log count distribution: {log_peak}")

            # log_peak is unlikely to be zero because we had a cutoff, but if the
            # max density is the first index that's probably bad
            if log_peak_ind == 0:
                logger.warning('The empty droplet plateau is being identified as '
                               'having approximately zero counts. Unless there really '
                               'is almost no noise in the dataset, this is likely '
                               'an error. Have you set --low-count-threshold too low?')

            # TODO: the below logic is all messed up
            # TODO: simplify: get rid of GMM, use Otsu's method instead
            # TODO: allow direct input

            # Gaussian PDF is about 60% peak height at 1 stdev away.
            std_above_ind = log_peak_ind + np.abs(
                density[log_peak_ind:] - density[log_peak_ind] * 0.6
            ).argmin()
            stdev = x[std_above_ind] - x[log_peak_ind]
            self.priors['surely_empty_count_estimate'] = np.exp(log_peak + 2 * stdev)
            self.priors['total_droplet_barcodes'] = (
                np.sum(counts > self.priors['surely_empty_count_estimate']).item() + 5000
            )
            logger.debug(f'surely_empty_count_estimate = {self.priors["surely_empty_count_estimate"]}')
            logger.debug(f'total_droplet_barcodes = {self.priors["total_droplet_barcodes"]}')
            cutoff = np.exp(log_peak - 3 * stdev)
            logger.debug(f'Cutting off counts below {cutoff} for the purposes '
                         f'of GMM fitting: assuming they represent barcode errors')
            counts = counts[counts > cutoff]

            # Fit a Gaussian mixture model to the counts.
            gmm = GMM(torch.tensor(counts).float()
                      .to('cuda' if torch.cuda.is_available() else 'cpu').log(),
                      n_components=consts.GMM_COMPONENTS,
                      alpha_prior=consts.GMM_ALPHA_PRIOR,
                      use_cuda=torch.cuda.is_available(),
                      verbose=False)
            gmm.train(epochs=consts.GMM_EPOCHS)
            self.gmm = gmm  # save this for later plotting

            map_est = gmm.map_estimate(sort_by='loc', ascending=True)

            # # TODO: temp plotting upfront
            # TODO: even if this is kept, this is not the place
            # TODO: tests will produce a plot, for example
            # TODO: tests will produce a plot, for example
            # # UMI count prior GMM plot.
            # fig = self.gmm.plot_summary()
            # fig.savefig('umi_hist.pdf', bbox_inches='tight', format='pdf')
            # logger.info("TEMP: Saved UMI count plot as umi_hist.pdf")
            #
            # # TODO ======

            # The first entry is empties since we sort by count.
            self.priors['empty_counts'] = np.exp(map_est['loc'][0]).item()
            logger.debug('MAP estimates from GMM fit to counts')
            logger.debug(map_est)
            logger.debug(f'{len(map_est["loc"])} peaks found')

            # Other modes are cells: currently the prior mixes them all.
            # Find a cutoff midway between empties and lowest cell mode.
            # TODO: ensure we have a cell mode at all! (edge case where we only fit empties...)
            if len(map_est['loc']) < 2:
                logger.debug('Attempting a workaround since only one peak was found by GMM')
                # We failed to find a cell peak, so make an estimate
                torch_counts = torch.tensor(counts, device='cpu').float().log()
                mode = torch_counts.mode()[0].item()
                cell_maximum = torch_counts.max().item()
                cell_minimum = (torch_counts[torch_counts > np.mean([mode, cell_maximum])]
                                .mode()[0].item())
                cell_mode = torch_counts[torch_counts > cell_minimum].mode()[0].item()
                cell_minimum = np.mean([mode, cell_mode]).item()
                cell_minimum = max(mode + 0.5, cell_minimum)
                self.priors['log_counts_crossover'] = cell_minimum
            else:
                self.priors['log_counts_crossover'] = np.mean(map_est['loc'][:2]).item()
            cell_logic = (counts > np.exp(self.priors['log_counts_crossover']))
            self.priors['n_cells'] = np.sum(cell_logic).item()
            self.priors['cell_counts'] = np.median(counts[cell_logic]).item()

            # TODO: playing with variance priors 2022/05/16
            # TODO: seems that a slim d_cell_scale prior is helpful in 0.2.0
            # variance_priors = self._estimate_prior_variances(
            #     empty_count_empirical_mean=np.exp(map_est['loc'][0]),
            #     empty_count_empirical_variance=map_est['scale'][0],
            #     log_cell_count_empirical_variance=np.log1p(counts[cell_logic]).var().item(),
            # )
            # self.priors['d_std'] = np.sqrt(variance_priors['d_cell_var'])
            # self.priors['d_empty_std'] = np.sqrt(variance_priors['d_empty_var'])

            # TODO ===============^^

            logger.debug(f'log counts is {np.log(counts)}')
            logger.debug(f'std of log counts > crossover is {np.std(np.log(counts)[np.log(counts) > self.priors["log_counts_crossover"]])}')

            self.priors['d_std'] = \
                np.std(np.log(counts)[np.log(counts) > self.priors['log_counts_crossover']]).item() / 5.
            self.priors['d_empty_std'] = \
                np.std(np.log(counts)[np.log(counts) < self.priors['log_counts_crossover']]).item() / 5.

            # Estimate the ambient gene expression profile.
            count_matrix = self.data['matrix'][:, self.analyzed_gene_inds]
            ep = np.finfo(np.float32).eps.item()  # small value
            chi_ambient = np.array(count_matrix[~cell_logic, :].sum(axis=0)).squeeze() + ep
            self.priors['chi_ambient'] = torch.tensor(chi_ambient / chi_ambient.sum()).float()
            chi_bar = np.array(count_matrix.sum(axis=0)).squeeze() + ep
            self.priors['chi_bar'] = torch.tensor(chi_bar / chi_bar.sum()).float()

            logger.info(f"Prior on counts for cells is {int(self.priors['cell_counts'])}")
            logger.info(f"Prior on counts for empty droplets is {int(self.priors['empty_counts'])}")

            logger.debug(f"Prior on cell count std is {self.priors['d_std']}")
            logger.debug(f"Prior on empty counts std is {self.priors['d_empty_std']}")

        # # Estimate the log UMI count turning point between cells and 'empties'.
        # self.priors['log_counts_crossover'] = \
        #     np.mean(np.log1p([self.priors['cell_counts'],
        #                       self.priors['empty_counts']])).item()
        #
        # # TODO: overhaul estimation of d_std.  add estimate of d_empty_std
        #
        # # Estimate prior for the scale param of LogNormal for d.
        # if self.model_name != "simple":
        #     self.priors['d_std'] = (np.log1p(self.priors['cell_counts'])
        #                             - self.priors['log_counts_crossover']) / 5
        # else:
        #     # Use a reasonable prior in log space.
        #     self.priors['d_std'] = self.SIMPLE_MODEL_D_STD_PRIOR
        #
        # # Priors for models that include empty droplets:
        # if self.model_name != "simple":
        #     # Estimate fraction of trimmed dataset that contains cells.
        #     # cell_prob = self.priors['n_cells']
        #     # / self.analyzed_barcode_inds.size
        #     cell_prob = (1 - self.fraction_empties) \
        #                 * (self.priors['n_cells']
        #                    / self.analyzed_barcode_inds.size)
        #     self.priors['cell_prob'] = cell_prob
        #
        #     assert cell_prob > 0, f"Fraction of trimmed dataset " \
        #                           f"containing cells should be > 0, " \
        #                           f"but is {cell_prob}."
        #
        #     assert cell_prob <= 1, f"Fraction of trimmed dataset " \
        #                            f"containing cells should be at most 1, " \
        #                            f"but is {cell_prob}."
        #
        #     # Turn cell probability into logit.
        #     self.priors['cell_logit'] = np.log(cell_prob
        #                                        / (1 - cell_prob)).item()
        #
        #     # Estimate the ambient gene expression profile.
        #     self.priors['chi_ambient'], self.priors['chi_bar'] = \
        #         estimate_chi_ambient_from_dataset(self)

    def _estimate_prior_variances(self,
                                  empty_count_empirical_mean: float,
                                  empty_count_empirical_variance: float,
                                  log_cell_count_empirical_variance: float,
                                  negbinom_overdispersion_phi: float = consts.PHI_LOC_PRIOR)\
            -> Dict[str, float]:
        """Estimate priors for variances

        Strategy:
            Assume the real variance of the number of molecules in the empties is
            due to being a Poisson draw: thus the variance is the mean.
            Overdispersion in the empirical variance beyond this is due to
            epsilon, and can be calculated using the law of total variance.
            Once the variance of epsilon is known, and given the overdispersion
            Phi of the negative binomial used to sample cell counts, the
            variance of d_cell can likewise be calculated.  This part is an
            empirical estimate rather than derived.

        """
        gamma_var = ((empty_count_empirical_variance - 2. * empty_count_empirical_mean)
                     / (empty_count_empirical_mean * (1. + empty_count_empirical_mean)))
        epsilon_c = 1. / gamma_var
        d_cell_var = log_cell_count_empirical_variance - negbinom_overdispersion_phi - 1. / epsilon_c
        return {'epsilon_c': torch.clamp(torch.tensor(epsilon_c), min=10., max=10000.).item(),
                'd_empty_var': empty_count_empirical_mean,
                'd_cell_var': torch.clamp(torch.tensor(d_cell_var), min=0.01, max=1.).item()}

    def get_count_matrix(self) -> sp.csr.csr_matrix:
        """Get the count matrix, trimmed if trimming has occurred."""

        if self.is_trimmed:

            # Return the count matrix for selected barcodes and genes.
            trimmed_bc_matrix = self.data['matrix'][self.analyzed_barcode_inds,
                                                    :].tocsc()
            trimmed_matrix = trimmed_bc_matrix[:, self.analyzed_gene_inds].tocsr()
            return trimmed_matrix

        else:
            logger.warning("Using full count matrix, without any trimming.  Could be slow.")
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
            logger.error("Trying to get empty count matrix without "
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
            logger.warning("Using full count matrix, without any trimming.  Could be slow.")
            return self.data['matrix']

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
        out = _overwrite_matrix_with_columns_from_another(
            mat1=inferred_count_matrix,
            mat2=self.data['matrix'],
            column_inds=self.analyzed_gene_inds)

        # But ensure that empty droplets are empty in the output.
        cell_probabilities_all_bcs = np.zeros(out.shape[0])
        cell_probabilities_all_bcs[self.analyzed_barcode_inds] = cell_probabilities_analyzed_bcs
        empty_inds = np.where(cell_probabilities_all_bcs <= consts.CELL_PROB_CUTOFF)[0]
        out = _csr_set_rows_to_zero(csr=out.tocsr(), row_inds=empty_inds)

        return out.tocsc()

    # def calculate_posterior(self,
    #                         inferred_model: Optional['RemoveBackgroundPyroModel'],
    #                         posterior_batch_size: int,
    #                         debug: bool = False):
    #     """Generate posterior for count matrix.
    #
    #     Args:
    #         inferred_model: RemoveBackgroundPyroModel which has
    #             already had the inference procedure run.
    #         posterior_batch_size: Max number of cells to be used in a batch
    #             for posterior regularization calculation, which requires a lot
    #             of memory.
    #
    #     NOTE: returns nothing, but stores the posterior in self.posterior
    #
    #     """
    #
    #     # Create posterior.
    #     if inferred_model is not None:
    #         self.posterior = Posterior(dataset_obj=self,
    #                                    vi_model=inferred_model,
    #                                    fpr=self.fpr[0],  # first FPR
    #                                    posterior_batch_size=posterior_batch_size,
    #                                    debug=debug)
    #
    #         # Encoded values of latent variables.
    #         enc = self.posterior.latents_map
    #         z = enc['z']
    #         d = enc['d']
    #         p = enc['p']
    #         epsilon = enc['epsilon']
    #
    #     else:
    #         self.posterior = NaivePosterior(dataset_obj=self)
    #         z = None
    #         d = None
    #         p = torch.ones(self.analyzed_barcode_inds.size)
    #         epsilon = None
    #         phi_params = None

    # def save_to_output_file(
    #         self,
    #         output_file: str,
    #         inferred_model: Optional['RemoveBackgroundPyroModel'],
    #         save_plots: bool = False,
    #         create_report: bool = False,
    #         truth_file: Optional[str] = None) -> bool:
    #     """Write the results of an inference procedure to an output file.
    #
    #     Output is an HDF5 file.  To be written:
    #     Inferred ambient-subtracted UMI count matrix.
    #     Inferred probabilities that each barcode contains a real cell.
    #     Inferred cell size scale factors.
    #     Inferred ambient gene expression count vector for droplets without
    #         cells.
    #     Inferred contamination fraction hyperparameters.
    #     Embeddings of gene expression of cells into a low-dimensional latent
    #         space.
    #
    #     Args:
    #         inferred_model: RemoveBackgroundPyroModel which has
    #             already had the inference procedure run.
    #         output_file: Name of output .h5 file
    #         save_plots: Setting this to True will save plots of outputs.
    #         create_report: Setting this to True will create an HTML report.
    #         truth_file: File containing truth data (for simulations, for report)
    #
    #     Returns:
    #         True if the output was written to file successfully.
    #
    #     """
    #
    #     logger.info("Preparing to write outputs to file...")
    #
    #     # Output file naming.
    #     file_dir, file_base = os.path.split(output_file)
    #     file_name = os.path.splitext(os.path.basename(file_base))[0]
    #
    #     # Obtain latents from posterior.
    #     if self.posterior.name != 'naive':  # TODO: handle 'naive' as a BasePosterior too, and eliminate if-else
    #         # Encoded values of latent variables.
    #         enc = self.posterior.latents_map
    #         z = enc['z']
    #         d = enc['d']
    #         p = enc['p']
    #         epsilon = enc['epsilon']
    #
    #     else:
    #         z = None
    #         d = None
    #         p = torch.ones(self.analyzed_barcode_inds.size)
    #         epsilon = None
    #         phi_params = None
    #
    #     # Figure out the indices of barcodes that have cells.
    #     if p is not None:
    #         p[np.isnan(p)] = 0.
    #         cell_barcode_inds = self.analyzed_barcode_inds
    #         if (p > consts.CELL_PROB_CUTOFF).sum() == 0:
    #             logger.warning("Warning: Found no cells!")
    #         analyzed_barcode_logic = (p > consts.CELL_PROB_CUTOFF)
    #     else:
    #         cell_barcode_inds = self.analyzed_barcode_inds
    #         analyzed_barcode_logic = np.arange(0, cell_barcode_inds.size)
    #
    #     # Save barcodes determined to contain cells as _cell_barcodes.csv
    #     cell_barcodes = self.data['barcodes'][self.analyzed_barcode_inds[analyzed_barcode_logic]]
    #     try:
    #         barcode_names = np.array([str(cell_barcodes[i], encoding='UTF-8')
    #                                   for i in range(cell_barcodes.size)])
    #     except UnicodeDecodeError:
    #         # necessary if barcodes are ints
    #         barcode_names = cell_barcodes
    #     except TypeError:
    #         # necessary if barcodes are already decoded
    #         barcode_names = cell_barcodes
    #     bc_file_name = os.path.join(file_dir, file_name + "_cell_barcodes.csv")
    #     np.savetxt(bc_file_name, barcode_names, delimiter=',', fmt='%s')
    #     logger.info(f"Saved cell barcodes in {bc_file_name}")
    #
    #     # Save plots, if called for.
    #     if save_plots:
    #         try:
    #             # File naming.
    #             gmm_fig_name = os.path.join(file_dir, file_name + "_umi_counts.pdf")
    #             summary_fig_name = os.path.join(file_dir, file_name + ".pdf")
    #
    #             # UMI count prior GMM plot.
    #             fig = self.gmm.plot_summary()
    #             fig.savefig(gmm_fig_name, bbox_inches='tight', format='pdf')
    #             logger.info(f"Saved UMI count plot as {gmm_fig_name}")
    #
    #             # Three-panel output summary plot.
    #             fig = self.plot_summary(inferred_model=inferred_model, p=p, z=z)
    #             fig.savefig(summary_fig_name, bbox_inches='tight', format='pdf')
    #             logger.info(f"Saved summary plots as {summary_fig_name}")
    #
    #         except Exception:
    #             logger.warning("Unable to save all plots.")
    #             logger.warning(traceback.format_exc())
    #
    #     # Estimate the ambient-background-subtracted UMI count matrix.
    #     if self.model_name == 'simple':
    #         # No need to generate a new count matrix for simple model.
    #         inferred_count_matrix = self.data['matrix'].tocsc()
    #         logger.info("Simple model: outputting un-altered count matrix.")
    #     else:
    #         inferred_count_matrix = self.posterior.denoised_counts
    #
    #     # TODO: there seems to be some kind of a huge gap between the above
    #     # TODO: computation and the below file writing.  HUGE gap in time!
    #
    #     # Inferred ambient gene expression vector.
    #     ambient_expression_trimmed = get_param_store_key('chi_ambient')
    #
    #     # Convert the indices from trimmed gene set to original gene indices.
    #     ambient_expression = np.zeros(self.data['matrix'].shape[1])
    #     ambient_expression[self.analyzed_gene_inds] = ambient_expression_trimmed
    #
    #     def _write_matrix(file: str,
    #                       inferred_count_matrix: sp.csc_matrix,
    #                       fpr: float,
    #                       analyzed_barcode_logic: np.ndarray = ...,
    #                       barcode_inds: np.ndarray = ...) -> bool:
    #         """Helper function for writing output h5 file.
    #         The ellipses as defaults serve to slice everything if the arguments
    #         are not supplied.
    #
    #         Uses variables from the outer scope of save_to_output_file()
    #         """
    #
    #         # CellRanger version (format output like input).
    #         if 'cellranger_version' in self.data.keys():
    #             cellranger_version = self.data['cellranger_version']
    #         else:
    #             cellranger_version = 3
    #
    #         # Some summary statistics:
    #         # Fraction of counts in each droplet that were removed.
    #         raw_count_matrix = self.data['matrix'][self.analyzed_barcode_inds, :]  # need all genes
    #         raw_counts_droplet = np.array(raw_count_matrix.sum(axis=1)).squeeze()
    #         out_counts_droplet = np.array(inferred_count_matrix[self.analyzed_barcode_inds, :]
    #                                       .sum(axis=1)).squeeze()
    #         background_fraction = ((raw_counts_droplet - out_counts_droplet) /
    #                                (raw_counts_droplet + 0.001))
    #
    #         # Subset latents.
    #         z_subset = None if z is None else z[analyzed_barcode_logic, :]
    #         d_subset = None if d is None else d[analyzed_barcode_logic]
    #         p_subset = None if p is None else p[analyzed_barcode_logic]
    #         epsilon_subset = None if epsilon is None else epsilon[analyzed_barcode_logic]
    #
    #         # Write h5.
    #         write_succeeded = write_matrix_to_cellranger_h5(
    #             cellranger_version=cellranger_version,
    #             output_file=file,
    #             gene_names=self.data['gene_names'],
    #             gene_ids=self.data['gene_ids'],
    #             feature_types=self.data['feature_types'],
    #             genomes=self.data['genomes'],
    #             barcodes=self.data['barcodes'][barcode_inds],
    #             count_matrix=inferred_count_matrix[barcode_inds, :],
    #             local_latents={'barcode_indices_for_latents': self.analyzed_barcode_inds,
    #                            'gene_expression_encoding': z_subset,
    #                            'cell_size': d_subset,
    #                            'cell_probability': p_subset,
    #                            'droplet_efficiency': epsilon_subset,
    #                            'background_fraction': background_fraction},
    #             global_latents={'ambient_expression': ambient_expression,
    #                             'empty_droplet_size_lognormal_loc': [get_param_store_key('d_empty_loc')],
    #                             'empty_droplet_size_lognormal_scale': [get_param_store_key('d_empty_scale')],
    #                             'cell_size_lognormal_std': [get_param_store_key('d_cell_scale')],
    #                             'swapping_fraction_dist_params':
    #                                 cellbender.remove_background.model.get_rho(),
    #                             'target_false_positive_rate': [fpr],
    #                             'posterior_type': [self.posterior.name]},
    #             metadata={'learning_curve': None if inferred_model is None else inferred_model.loss,
    #                       'barcodes_analyzed': self.data['barcodes'][self.analyzed_barcode_inds],
    #                       'barcodes_analyzed_inds': self.analyzed_barcode_inds,
    #                       'features_analyzed_inds': self.analyzed_gene_inds,
    #                       'fraction_data_used_for_testing': [1. - consts.TRAINING_FRACTION]},
    #         )
    #
    #         return write_succeeded
    #
    #     # Write to output file, for each lambda specified by user.
    #     logger.debug('Staring FPR loop')
    #
    #     for i, fpr in enumerate(self.fpr):
    #
    #         # TODO: what if this loop takes so long that we keep getting pre-empted?
    #         # TODO: think about checking for checkpoint and skipping FPRs for which
    #         # TODO: we (presumably??) have already written output files
    #         # TODO: well, this only works if output files become part of checkpoints...
    #
    #         logger.debug(f'Working on FPR {fpr}')
    #
    #         # Re-compute posterior counts for each new lambda.
    #         if i > 0:  # no need to re-compute for the first FPR: this is already done.
    #             self.posterior.fpr = fpr  # reach in and change the FPR
    #             self.posterior._get_denoised_counts()  # force re-computation of posterior
    #         inferred_count_matrix = self.posterior.denoised_counts
    #
    #         # TODO: put the trimmed features back in!  (make sure this works)
    #         import time
    #         t = time.time()
    #         logger.debug('Restoring eliminated features in cells')
    #         inferred_count_matrix = self._restore_eliminated_features_in_cells(
    #             inferred_count_matrix,
    #             self.posterior.latents_map['p'])
    #         logger.debug(f'Took {time.time() - t:.3f} sec')
    #
    #         # TODO: correct posterior cell probabilities so that no zero-count
    #         # TODO: droplets contain a cell.  maybe set p to 1e-10 or something
    #
    #         # Create an output file for this FPR.
    #         name_suffix = (f'_FPR_{fpr}' if len(self.fpr) > 1 else '')
    #         fpr_output_filename = os.path.join(file_dir, file_name + name_suffix + '.h5')
    #
    #         fpr = np.array('cohort', dtype=str) if (fpr == 'cohort') else fpr  # for pytables h5 writing
    #
    #         write_succeeded = _write_matrix(file=fpr_output_filename,
    #                                         inferred_count_matrix=inferred_count_matrix,
    #                                         fpr=fpr)
    #
    #         # Write filtered matrix (cells only) to output file.
    #         filtered_output_file = os.path.join(file_dir, file_name + name_suffix + '_filtered.h5')
    #         if self.include_empties:
    #             _write_matrix(file=filtered_output_file,
    #                           inferred_count_matrix=inferred_count_matrix,
    #                           fpr=fpr,
    #                           barcode_inds=self.analyzed_barcode_inds[analyzed_barcode_logic],
    #                           analyzed_barcode_logic=analyzed_barcode_logic)
    #
    #         # Create report, if called for.
    #         if create_report:
    #             # try:
    #             os.environ['INPUT_FILE'] = os.path.abspath(os.path.join(os.getcwd(), self.input_file))
    #             os.environ['OUTPUT_FILE'] = os.path.abspath(os.path.join(os.getcwd(), fpr_output_filename))
    #             if truth_file is not None:
    #                 os.environ['TRUTH_FILE'] = os.path.abspath(os.path.join(os.getcwd(), truth_file))
    #             html_report_file = os.path.join(file_dir, file_name + name_suffix + '_report.html')
    #             run_notebook_make_html(
    #                 file=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'report.ipynb')),
    #                 output=html_report_file,
    #             )
    #             logger.info(f'Succeeded in writing report to {html_report_file}')
    #
    #             # except Exception:
    #             #     logger.warning("Unable to create report.")
    #             #     logger.warning(traceback.format_exc())
    #
    #         # Write output metrics file.
    #         try:
    #             df = self.collect_output_metrics(inferred_count_matrix=inferred_count_matrix,
    #                                              fpr=fpr,
    #                                              cell_logic=(p >= consts.CELL_PROB_CUTOFF),
    #                                              loss=inferred_model.loss)
    #             metrics_file_name = os.path.join(file_dir, file_name + name_suffix + '_metrics.csv')
    #             df.to_csv(metrics_file_name, index=True, header=False, float_format='%.3f')
    #             logger.info(f'Saved output metrics as {metrics_file_name}')
    #         except Exception:
    #             logger.warning("Unable to collect output metrics.")
    #             logger.warning(traceback.format_exc())
    #
    #     return write_succeeded
    #
    # def plot_summary(self,
    #                  inferred_model: 'RemoveBackgroundPyroModel',
    #                  p: np.ndarray,
    #                  z: np.ndarray):
    #     """Output summary plot with three panels: training, cells, latent z."""
    #
    #     fig = plt.figure(figsize=(6, 18))
    #
    #     # Plot the train error.
    #     plt.subplot(3, 1, 1)
    #     if inferred_model is not None:
    #         plt.plot(inferred_model.loss['train']['elbo'], '.--', label='Train')
    #
    #         # Plot the test error, if there was held-out test data.
    #         if 'test' in inferred_model.loss.keys():
    #             if len(inferred_model.loss['test']['epoch']) > 0:
    #                 plt.plot(inferred_model.loss['test']['epoch'],
    #                          inferred_model.loss['test']['elbo'], 'o:', label='Test')
    #                 plt.legend()
    #
    #         plt.gca().set_ylim(bottom=max(inferred_model.loss['train']['elbo'][0],
    #                                       inferred_model.loss['train']['elbo'][-1] - 2000))
    #         plt.xlabel('Epoch')
    #         plt.ylabel('ELBO')
    #         plt.title('Progress of the training procedure')
    #     else:
    #         # Plot intentionally left blank, if there was no training (fast algorithm).
    #         plt.title('Plot intentionally left blank')
    #         plt.xticks([])
    #         plt.yticks([])
    #
    #     # Plot the barcodes used, along with the inferred
    #     # cell probabilities.
    #     plt.subplot(3, 1, 2)
    #     count_mat = self.get_count_matrix()
    #     counts = np.array(count_mat.sum(axis=1)).squeeze()
    #     count_order = np.argsort(counts)[::-1]
    #     plt.semilogy(counts[count_order], color='black')
    #     plt.ylabel('UMI counts')
    #     plt.xlabel('Barcode index, sorted by UMI count')
    #     if p is not None:  # The case of a simple model.
    #         plt.gca().twinx()
    #         plt.plot(p[count_order], '.:', color='red', alpha=0.3, rasterized=True)
    #         plt.ylabel('Cell probability', color='red')
    #         plt.ylim([-0.05, 1.05])
    #         plt.title('Determination of which barcodes contain cells')
    #     else:
    #         plt.title('The subset of barcodes used for training')
    #
    #     # Plot the latent encoding via PCA.
    #     plt.subplot(3, 1, 3)
    #     if p is None:
    #         p = np.ones(z.shape[0])
    #     # A = torch.tensor(z[p >= consts.CELL_PROB_CUTOFF])
    #     # U, S, V = torch.pca_lowrank(A)
    #     z_pca = pca_2d(z[p >= consts.CELL_PROB_CUTOFF])  # torch.matmul(A, V[:, :2])
    #     plt.plot(z_pca[:, 0], z_pca[:, 1],
    #              '.', ms=3, color='black', alpha=0.3, rasterized=True)
    #     plt.ylabel('PC 1')
    #     plt.xlabel('PC 0')
    #     plt.title('PCA of latent encoding of cell gene expression')
    #
    #     return fig
    #
    # def collect_output_metrics(self,
    #                            inferred_count_matrix: sp.csr_matrix,
    #                            fpr: Union[float, str],
    #                            cell_logic,
    #                            loss) -> pd.DataFrame:
    #     """Create a table with a few output metrics. The idea is for these to
    #     potentially be used by people creating automated pipelines."""
    #
    #     # Compute some metrics
    #     input_count_matrix = self.data['matrix'][self.analyzed_barcode_inds, :]
    #     total_raw_counts = self.data['matrix'].sum()
    #     total_output_counts = inferred_count_matrix.sum()
    #     total_counts_removed = total_raw_counts - total_output_counts
    #     fraction_counts_removed = total_counts_removed / total_raw_counts
    #     total_raw_counts_in_nonempty_droplets = input_count_matrix[cell_logic].sum()
    #     total_counts_removed_from_nonempty_droplets = \
    #         total_raw_counts_in_nonempty_droplets - inferred_count_matrix.sum()
    #     fraction_counts_removed_from_nonempty_droplets = \
    #         total_counts_removed_from_nonempty_droplets / total_raw_counts_in_nonempty_droplets
    #     average_counts_removed_per_nonempty_droplet = \
    #         total_counts_removed_from_nonempty_droplets / cell_logic.sum()
    #     expected_cells = self.expected_cell_count
    #     found_cells = cell_logic.sum()
    #     average_counts_per_cell = inferred_count_matrix.sum() / found_cells
    #     ratio_of_found_cells_to_expected_cells = \
    #         None if (expected_cells is None) else (found_cells / expected_cells)
    #     found_empties = len(self.analyzed_barcode_inds) - found_cells
    #     fraction_of_analyzed_droplets_that_are_nonempty = \
    #         found_cells / len(self.analyzed_barcode_inds)
    #     if len(loss['train']['elbo']) > 20:
    #         # compare mean ELBO increase over last 3 steps to the typical end(ish) fluctuations
    #         convergence_indicator = (np.mean(np.abs([(loss['train']['elbo'][i]
    #                                                   - loss['train']['elbo'][i - 1])
    #                                                  for i in range(-3, -1)]))
    #                                  / np.std(loss['train']['elbo'][-20:]))
    #     else:
    #         convergence_indicator = 'not enough training epochs to compute (requires more than 20)'
    #     overall_change_in_train_elbo = loss['train']['elbo'][-1] - loss['train']['elbo'][0]
    #
    #     all_metrics_dict = \
    #         {'total_raw_counts': total_raw_counts,
    #          'total_output_counts': total_output_counts,
    #          'total_counts_removed': total_counts_removed,
    #          'fraction_counts_removed': fraction_counts_removed,
    #          'total_raw_counts_in_cells':
    #              total_raw_counts_in_nonempty_droplets,
    #          'total_counts_removed_from_cells':
    #              total_counts_removed_from_nonempty_droplets,
    #          'fraction_counts_removed_from_cells':
    #              fraction_counts_removed_from_nonempty_droplets,
    #          'average_counts_removed_per_cell':
    #              average_counts_removed_per_nonempty_droplet,
    #          'target_fpr': fpr,
    #          'expected_cells': expected_cells,
    #          'found_cells': found_cells,
    #          'output_average_counts_per_cell': average_counts_per_cell,
    #          'ratio_of_found_cells_to_expected_cells':
    #              ratio_of_found_cells_to_expected_cells,
    #          'found_empties': found_empties,
    #          'fraction_of_analyzed_droplets_that_are_nonempty':
    #              fraction_of_analyzed_droplets_that_are_nonempty,
    #          'convergence_indicator': convergence_indicator,
    #          'overall_change_in_train_elbo': overall_change_in_train_elbo}
    #
    #     return pd.DataFrame(data=all_metrics_dict,
    #                         index=['metric']).transpose()


# def get_d_priors_from_dataset(dataset: SingleCellRNACountsDataset):
#     """Compute an estimate of reasonable priors on cell size and ambient size.
#
#     Given a SingleCellRNACountsDataset, compute an estimate of reasonable
#     priors on cell size and ambient count size.  This is done by fitting a
#     Gaussian mixture model.
#
#     Args:
#         dataset: Dataset object containing a matrix of unique UMI counts,
#             where rows are barcodes and columns are genes.
#         expected_cells: Prior on number of cells.
#
#     Returns:
#
#
#     """
#
#     # Count the total unique UMIs per barcode.
#     counts = np.array(dataset.data['matrix']
#                       [:, dataset.analyzed_gene_inds].sum(axis=1)).squeeze()
#
#     # If it's a model that does not model empty droplets, the dataset is cells.
#     if dataset.model_name == 'simple':
#
#         if dataset.priors['n_cells'] is None:
#             # No prior on number of cells.  Assume all are cells.
#             dataset.priors['n_cells'] = int(np.sum(counts > 0).item())
#
#         # Sort order the cells by counts.
#         sort_order = np.argsort(counts)[::-1]
#
#         # Estimate cell count by median, taking 'cells' to be largest counts.
#         cell_counts = int(np.median(counts[sort_order][:dataset.priors['n_cells']]).item())
#         empty_counts = 0
#
#     # Models that include both cells and empty droplets.
#     else:
#
#         # Cutoff for original data.  Empirical.
#         cut = dataset.low_count_threshold


# def get_d_priors_from_dataset(dataset: SingleCellRNACountsDataset) \
#         -> Tuple[float, float]:
#     """Compute an estimate of reasonable priors on cell size and ambient size.
#
#     Given a dataset (scipy.sparse.csr matrix of counts where
#     rows are barcodes and columns are genes), and an expected
#     cell count, compute an estimate of reasonable priors on cell size
#     and ambient count size.  This is done by a series of heuristics.
#
#     Args:
#         dataset: Dataset object containing a matrix of unique UMI counts,
#             where rows are barcodes and columns are genes.
#
#     Returns:
#         cell_counts: Estimated mean number of UMI counts per real cell, in
#             terms of transformed count data.
#         empty_counts: Estimated mean number of UMI counts per 'empty'
#             droplet, in terms of transformed count data.
#
#     NOTE: Picks barcodes using cutoffs in untransformed count data.  The output
#     is in terms of transformed counts.
#
#     """
#
#     # Count the total unique UMIs per barcode (summing after transforming).
#     counts = np.array(dataset.data['matrix']
#                       [:, dataset.analyzed_gene_inds].sum(axis=1)).squeeze()
#
#     # If it's a model that does not model empty droplets, the dataset is cells.
#     if dataset.model_name == 'simple':
#
#         if dataset.priors['n_cells'] is None:
#             # No prior on number of cells.  Assume all are cells.
#             dataset.priors['n_cells'] = int(np.sum(counts > 0).item())
#
#         # Sort order the cells by counts.
#         sort_order = np.argsort(counts)[::-1]
#
#         # Estimate cell count by median, taking 'cells' to be largest counts.
#         cell_counts = int(np.median(counts[sort_order]
#                                     [:dataset.priors['n_cells']]).item())
#
#         empty_counts = 0
#
#     # Models that include both cells and empty droplets.
#     else:
#
#         # Cutoff for original data.  Empirical.
#         cut = dataset.low_count_threshold
#
#         # Estimate the number of UMI counts in empty droplets.
#
#         # Mode of (rounded) log counts (for counts > cut) is a robust
#         # empty estimator.
#         empty_log_counts = mode(np.round(np.log1p(counts[counts > cut]),
#                                          decimals=1))[0]
#         empty_counts = int(np.expm1(empty_log_counts).item())
#
#         # Estimate the number of UMI counts in cells.
#
#         # Use expected cells if it is available.
#         if dataset.priors['n_cells'] is not None:
#
#             # Sort order the cells by counts.
#             sort_order = np.argsort(counts)[::-1]
#
#             cell_counts = int(np.median(counts[sort_order]
#                                         [:dataset.priors['n_cells']]).item())
#
#         else:
#
#             # Median of log counts above 5 * empty counts is a robust
#             # cell estimator.
#             cell_log_counts = np.median(np.log1p(counts[counts > 5 * empty_counts]))
#             cell_counts = int(np.expm1(cell_log_counts).item())
#
#         logger.info(f"Prior on counts in empty droplets is {empty_counts}")
#
#     logger.info(f"Prior on counts for cells is {cell_counts}")
#
#     return cell_counts, empty_counts


def estimate_cell_count_from_dataset(dataset: SingleCellRNACountsDataset) -> int:
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
    if not dataset.include_empties:
        return dataset.data['matrix'].shape[0]

    # Count the number of barcodes with UMI counts above the cutoff.
    cell_count_est = int(np.sum(estimate_cell_logic_from_dataset(dataset=dataset)).item())

    return cell_count_est


def estimate_cell_logic_from_dataset(dataset: SingleCellRNACountsDataset) -> int:
    """Compute an estimate of number of real cells in a dataset.

    Given a Dataset, compute an estimate of the number of real cells.

    Args:
        dataset: Dataset object containing a matrix of unique UMI counts,
            where rows are barcodes and columns are genes.

    Returns:
        cell_logic: Array where True indicates a probable cell.

    """

    # Count number of UMIs in each barcode.
    counts = np.array(dataset.data['matrix'].sum(axis=1),
                      dtype=int).squeeze()

    # Find mid-way between cell_counts and empty_counts in log space.
    midway = np.mean([np.log1p(dataset.priors['cell_counts']),
                      np.log1p(dataset.priors['empty_counts'])])
    umi_cutoff = np.expm1(midway)

    return counts > umi_cutoff


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
    gene_expression = np.array(count_matrix[empty_barcodes, :].sum(axis=0)).squeeze()

    # As a vector on a simplex.
    gene_expression = gene_expression + ep
    chi_ambient_init = torch.tensor(gene_expression / np.sum(gene_expression))

    # Full count matrix, appropriately transformed.
    full_count_matrix = dataset.get_count_matrix_all_barcodes()

    # Sum all gene expression.
    gene_expression_total = np.array(full_count_matrix.sum(axis=0)).squeeze()

    # As a vector on a simplex.
    gene_expression_total = gene_expression_total + ep
    chi_bar = torch.tensor(gene_expression_total / np.sum(gene_expression_total))

    return chi_ambient_init, chi_bar


def _overwrite_matrix_with_columns_from_another(mat1: sp.csc_matrix,
                                                mat2: sp.csc_matrix,
                                                column_inds: np.ndarray) -> sp.csc_matrix:
    """Given two sparse matrices of the same shape, replace columns that are not
    in `column_inds` in `mat1` with the entries from `mat2`.
    """
    column_inds = set(column_inds)

    mat1 = mat1.copy().tocsr()
    mat2 = mat2.copy().tocsr()  # failure to copy could overwrite actual count data

    # Zero out values in mat2 that are in the specified columns.
    inds = np.where([i in column_inds for i in mat2.indices])[0]
    mat2.data[inds] = 0
    mat2.eliminate_zeros()

    # Zero out values in mat1 that are not in the specified columns.
    inds = np.where([i not in column_inds for i in mat1.indices])[0]
    mat1.data[inds] = 0
    mat1.eliminate_zeros()

    # Put in the new values by addition.
    output = mat1 + mat2

    return output.tocsc()


def _csr_set_rows_to_zero(csr: sp.csr_matrix, row_inds: Iterable[int]):
    """Set all nonzero elements in rows "row_inds" to zero.
    Happens in-place, although output is returned as well.

    https://stackoverflow.com/questions/12129948/scipy-sparse-set-row-to-zeros
    """

    if not isinstance(csr, sp.csr_matrix):
        try:
            csr = csr.tocsr()
        except Exception:
            raise ValueError('Matrix given must be of CSR format.')
    for row in row_inds:
        csr.data[csr.indptr[row]:csr.indptr[row + 1]] = 0
    csr.eliminate_zeros()
    return csr


def pca_2d(mat: np.ndarray) -> torch.Tensor:
    """Perform PCA using pytorch and return top 2 PCs

    Args:
        mat: matrix where rows are observations and columns are features

    Returns:
        out: matrix where rows are observations and columns are top 2 PCs
    """

    A = torch.tensor(mat)
    U, S, V = torch.pca_lowrank(A)
    return torch.matmul(A, V[:, :2])
