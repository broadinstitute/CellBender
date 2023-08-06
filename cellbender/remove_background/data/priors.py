"""Functionality for estimating various priors from the data"""

import numpy as np
import torch
from scipy.stats import gaussian_kde

from cellbender.remove_background import consts

from typing import Dict, Tuple, Union
import logging


logger = logging.getLogger('cellbender')


def _threshold_otsu(umi_counts: np.ndarray, n_bins: int = 256) -> float:
    """Return threshold value based on fast implementation of Otsu's method.

    From skimage, with slight modifications:
    https://github.com/scikit-image/scikit-image/blob/
    a4e533ea2a1947f13b88219e5f2c5931ab092413/skimage/filters/thresholding.py#L312

    Args:
        umi_counts: Array of UMI counts
        n_bins: Number of bins used to calculate histogram

    Returns:
        threshold: Upper threshold value. All droplets with UMI counts greater
        than this value are assumed to contain cells.

    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    .. [2] https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html

    Notes
    -----
    The input image must be grayscale.
    """

    # create a UMI count histogram
    counts, bin_centers = _create_histogram(umi_counts=umi_counts, n_bins=n_bins)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold


def _create_histogram(umi_counts: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return a histogram.

    Args:
        umi_counts: Array of UMI counts
        n_bins: Number of bins used to calculate histogram

    Returns:
        counts: Each element is the number of droplets falling in each UMI
            count bin
        bin_centers: Each element is the value corresponding to the center of
            each UMI count bin
    """
    counts, bin_edges = np.histogram(umi_counts.reshape(-1), n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return counts.astype('float32', copy=False), bin_centers


def _peak_density_given_cutoff(umi_counts: np.ndarray,
                               cutoff: float,
                               cell_count_low_limit: float) -> Tuple[float, float]:
    """Run scipy.stats gaussian_kde on part of the UMI curve"""

    # get the UMI count values we are including
    noncell_counts = umi_counts[umi_counts <= cutoff]

    # resample them: the magic of looking at a log log plot
    n_putative_cells = (umi_counts > cell_count_low_limit).sum()
    n_putative_empties = len(noncell_counts)
    inds = np.logspace(np.log10(n_putative_cells),
                       np.log10(n_putative_cells + n_putative_empties),
                       num=1000,
                       base=10)
    inds = [max(0, min(int(ind - n_putative_cells), len(noncell_counts) - 1)) for ind in inds]

    noncell_counts = np.sort(noncell_counts)[::-1][inds]

    # find the peak density: that is the empty count prior

    # calculate range of data, rounding out to make sure we cover everything
    log_noncell_counts = np.log(noncell_counts)
    x = np.arange(
        np.floor(log_noncell_counts.min()) - 0.01,
        np.ceil(log_noncell_counts.max()) + 0.01,
        0.1
    )

    # fit a KDE to estimate density
    k = gaussian_kde(log_noncell_counts)
    density = k.evaluate(x)

    # the density peak is almost surely the empty droplets
    log_peak_ind = np.argmax(density)
    log_peak = x[log_peak_ind]
    empty_count_prior = np.exp(log_peak)

    # try to go about 1 stdev up from the peak
    peak_density = np.max(density)
    one_std_density = 0.6 * peak_density
    one_std_inds = np.where(density[log_peak_ind:] < one_std_density)[0]
    if len(one_std_inds) > 0:
        one_std_ind = one_std_inds[0]
    else:
        one_std_ind = len(density[log_peak_ind:]) - 1
    empty_count_upper_limit = np.exp(x[log_peak_ind:][one_std_ind])

    return empty_count_prior, empty_count_upper_limit


def get_cell_count_given_expected_cells(umi_counts: np.ndarray,
                                        expected_cells: int) -> Dict[str, float]:
    """In the case where a prior is passed in as input, use it

    Args:
        umi_counts: Array of UMI counts per droplet, in no particular order
        expected_cells: Input by user

    Returns:
        Dict with keys ['cell_counts']
    """
    order = np.argsort(umi_counts)[::-1]
    cell_counts = np.exp(np.mean(np.log(umi_counts[order][:expected_cells]))).item()
    return {'cell_counts': cell_counts}


def get_empty_count_given_expected_cells_and_total_droplets(
        umi_counts: np.ndarray,
        expected_cells: int,
        total_droplets: int,
) -> Dict[str, float]:
    """In the case where a prior is passed in as input, use it

    Args:
        umi_counts: Array of UMI counts per droplet, in no particular order
        expected_cells: Input by user, or prior estimate
        total_droplets: Input by user

    Returns:
        Dict with keys ['empty_counts', 'empty_count_upper_limit']
    """

    order = np.argsort(umi_counts)[::-1]
    starting_point = max(expected_cells, total_droplets - 500)
    empty_counts = np.median(umi_counts[order]
                             [int(starting_point):int(total_droplets)]).item()

    # need to estimate here
    cell_counts = np.exp(np.mean(np.log(umi_counts[order][:expected_cells]))).item()
    middle = np.sqrt(cell_counts * empty_counts)
    empty_count_upper_limit = min(middle, 1.5 * empty_counts)

    return {'empty_counts': empty_counts,
            'empty_count_upper_limit': empty_count_upper_limit}


def get_cell_count_empty_count(umi_counts: np.ndarray,
                               low_count_threshold: float = 15) -> Dict[str, float]:
    """Obtain priors on cell counts and empty droplet counts from a UMI curve
    using heuristics, and without applying any other prior information.

    Heuristics:
    0. Ignore droplets with counts below low_count_threshold
    1. Use Otsu's method to threshold the log UMI count data (ignoring droplets
       past 1/4 of the total droplets above low_count_threshold, as we go down
       the UMI curve). This is used as a lower limit on cell counts.
       It seems quite robust.
    2. Use the following iterative approach, until converged:
        a. Establish an upper cutoff on possible empty droplets, using the
           current estimate of empty counts and our cell count prior (the
           estimate is 3/4 of the geometric mean of the two).
        b. Use gaussian_kde from scipy.stats to create a smooth histogram of
           the log UMI counts, for droplets with counts below the cutoff.
           - A trick is used to resample the droplets before creating the
             histogram, so that it looks more like a log-log plot
        c. Identify the peak density of the histogram as the empty count
           estimate.
        - Convergence happens when our estimate of empty counts stops changing.

    Args:
        umi_counts: Array of UMI counts per droplet, in no particular order
        low_count_threshold: Ignore droplets with counts below this value

    Returns:
        Dict with keys ['cell_counts', 'empty_counts']
    """

    logger.debug('Beginning priors.get_cell_count_empty_count()')
    reverse_sorted_umi_counts = np.sort(umi_counts)[::-1]
    umi_counts_for_otsu = reverse_sorted_umi_counts[:(umi_counts > low_count_threshold).sum() // 4]

    log_cell_count_low_limit = _threshold_otsu(np.log(umi_counts_for_otsu))
    cell_count_low_limit = np.exp(log_cell_count_low_limit)

    logger.debug(f'cell_count_low_limit is {cell_count_low_limit}')
    cell_count_prior = np.mean(umi_counts[umi_counts > cell_count_low_limit])

    umi_counts_for_kde = reverse_sorted_umi_counts[reverse_sorted_umi_counts > low_count_threshold]

    # initial conditions for the loop
    # start low, but have a failsafe (especially for simulated data)
    cutoff = max(0.1 * cell_count_low_limit, umi_counts_for_kde[-100])
    empty_count_prior = -100
    empty_count_upper_limit = None
    delta = np.inf
    a = 0

    # iterate to convergence, at most 5 times
    while delta > 10:
        logger.debug(f'cutoff = {cutoff}')

        # use gaussian_kde to find the peak in the histogram
        new_empty_count_prior, empty_count_upper_limit = _peak_density_given_cutoff(
            umi_counts=umi_counts_for_kde,
            cutoff=cutoff,
            cell_count_low_limit=cell_count_low_limit,
        )
        logger.debug(f'new_empty_count_prior = {new_empty_count_prior}')

        # 3/4 of the geometric mean is our new upper cutoff
        cutoff = 0.75 * np.sqrt(cell_count_prior * new_empty_count_prior)
        delta = np.abs(new_empty_count_prior - empty_count_prior)
        logger.debug(f'delta = {delta}')
        empty_count_prior = new_empty_count_prior
        a += 1
        if a >= 5:
            logger.debug('Heuristics for determining empty counts exceeded 5 '
                         'iterations without converging')
            break

    # do a final estimation of cell counts:
    # go to the halfway point and then take the median of the droplets above
    count_crossover = np.sqrt(cell_count_prior * empty_count_prior)
    cell_count_prior = np.median(umi_counts[umi_counts > count_crossover])

    logger.debug(f'cell_count_prior is {cell_count_prior}')
    logger.debug(f'empty_count_prior is {empty_count_prior}')
    logger.debug('End of priors.get_cell_count_empty_count()')

    return {'cell_counts': cell_count_prior,
            'empty_counts': empty_count_prior,
            'empty_count_upper_limit': empty_count_upper_limit}


def get_expected_cells_and_total_droplets(umi_counts: np.ndarray,
                                          cell_counts: float,
                                          empty_counts: float,
                                          empty_count_upper_limit: float,
                                          max_empties: int = consts.MAX_EMPTIES_TO_INCLUDE) \
        -> Dict[str, int]:
    """Obtain priors on cell counts and empty droplet counts from a UMI curve
    using heuristics, and without applying any other prior information.

    NOTE: to be run using inputs from get_cell_count_empty_count()

    Args:
        umi_counts: Array of UMI counts per droplet, in no particular order
        cell_counts: Prior from get_cell_count_empty_count()
        empty_counts: Prior from get_cell_count_empty_count()
        empty_count_upper_limit: Prior from get_cell_count_empty_count()
        max_empties: Do not include more putative empty droplets than this

    Returns:
        Dict with keys ['expected_cells', 'total_droplets', 'transition_point']

    Example:
        >>> priors = get_cell_count_empty_count(umi_counts)
        >>> priors.update(get_expected_cells_and_total_droplets(umi_counts, **priors))
    """
    # expected cells does well when you give it a very conservative estimate
    expected_cells = (umi_counts >= cell_counts).sum()

    # total droplets will be between empty_count_prior and its upper limit
    total_droplets_count_value = np.sqrt(empty_counts * empty_count_upper_limit)
    total_droplets = (umi_counts >= total_droplets_count_value).sum()

    # find the transition point
    count_crossover = np.sqrt(cell_counts * empty_counts)
    transition_point = (umi_counts >= count_crossover).sum()

    logger.debug(f'In get_expected_cells_and_total_droplets(), found transition '
                 f'point at droplet {transition_point}')

    # ensure out heuristics don't go too far out datasets with many cells
    total_droplets = min(total_droplets, transition_point + max_empties)

    return {'expected_cells': expected_cells,
            'total_droplets': total_droplets,
            'transition_point': transition_point}


def get_priors(umi_counts: np.ndarray,
               low_count_threshold: float,
               max_total_droplets: int = consts.MAX_TOTAL_DROPLETS_GUESSED) \
        -> Dict[str, Union[int, float]]:
    """Get all priors using get_cell_count_empty_count() and
    get_expected_cells_and_total_droplets(), employing a failsafe if
    total_droplets is improbably large.

    Args:
        umi_counts: Array of UMI counts per droplet, in no particular order
        low_count_threshold: Ignore droplets with counts below this value
        max_total_droplets: If the initial heuristics come up with a
            total_droplets value greater than this, we re-run the heuristics
            with higher low_count_threshold

    Returns:
        Dict with keys ['cell_counts', 'empty_counts',
                        'empty_count_upper_limit', 'surely_empty_counts',
                        'expected_cells', 'total_droplets', 'log_counts_crossover']
    """

    logger.debug("Computing priors from the UMI curve")
    priors = get_cell_count_empty_count(
        umi_counts=umi_counts,
        low_count_threshold=low_count_threshold,
    )
    priors.update(get_expected_cells_and_total_droplets(umi_counts=umi_counts, **priors))
    logger.debug(f'Automatically computed priors: {priors}')

    a = 0
    while priors['total_droplets'] > max_total_droplets:
        logger.debug(f'Heuristics for estimating priors resulted in '
                     f'{priors["total_droplets"]} total_droplets, which is '
                     f'typically too large. Recomputing with '
                     f'low_count_threshold = {priors["empty_count_upper_limit"]:.0f}')
        priors = get_cell_count_empty_count(
            umi_counts=umi_counts,
            low_count_threshold=priors['empty_count_upper_limit'],
        )
        priors.update(get_expected_cells_and_total_droplets(umi_counts=umi_counts, **priors))
        logger.debug(f'Automatically computed priors: {priors}')
        a += 1
        if a > 5:
            break

    # compute a few last things
    compute_crossover_surely_empty_and_stds(umi_counts=umi_counts, priors=priors)

    return priors


def compute_crossover_surely_empty_and_stds(umi_counts, priors):
    """Given cell_counts and total_droplets, compute a few more quantities

    Args:
        umi_counts: Array of UMI counts per droplet, in no particular order
        priors: Dict of priors

    Returns:
        None.  Modifies priors dict in place.
    """

    assert 'total_droplets' in priors.keys(), \
        'Need total_droplets in priors to run compute_crossover_surely_empty_and_stds()'
    assert 'cell_counts' in priors.keys(), \
        'Need cell_counts in priors to run compute_crossover_surely_empty_and_stds()'

    # Compute a crossover point in log count space.
    reverse_sorted_counts = np.sort(umi_counts)[::-1]
    surely_empty_counts = reverse_sorted_counts[priors['total_droplets']]
    log_counts_crossover = (np.log(surely_empty_counts) + np.log(priors['cell_counts'])) / 2
    priors.update({'log_counts_crossover': log_counts_crossover,
                   'surely_empty_counts': surely_empty_counts})

    # Compute several other priors.
    log_nonzero_umi_counts = np.log(umi_counts[umi_counts > 0])
    d_std = np.std(log_nonzero_umi_counts[log_nonzero_umi_counts > log_counts_crossover]).item() / 5.
    d_empty_std = 0.01  # this is basically turned off in favor of epsilon
    priors.update({'d_std': d_std, 'd_empty_std': d_empty_std})
