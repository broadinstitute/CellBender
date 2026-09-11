"""Functionality for estimating various priors from the data"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.stats import gaussian_kde

from cellbender.remove_background import consts
from cellbender.remove_background.consts import GEX_FEATURE_TYPE

logger = logging.getLogger("cellbender")


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
    return counts.astype("float32", copy=False), bin_centers


def _peak_density_given_cutoff(
    umi_counts: np.ndarray, cutoff: float, cell_count_low_limit: float
) -> Tuple[float, float]:
    """Run scipy.stats gaussian_kde on part of the UMI curve"""

    # get the UMI count values we are including
    noncell_counts = umi_counts[umi_counts <= cutoff]

    # resample them: the magic of looking at a log log plot
    n_putative_cells = (umi_counts > cell_count_low_limit).sum()
    n_putative_empties = len(noncell_counts)
    inds_array = np.logspace(
        np.log10(n_putative_cells), np.log10(n_putative_cells + n_putative_empties), num=1000, base=10
    )
    inds: list[int] = [max(0, min(int(ind - n_putative_cells), len(noncell_counts) - 1)) for ind in inds_array]

    noncell_counts = np.sort(noncell_counts)[::-1][inds]

    # find the peak density: that is the empty count prior

    # calculate range of data, rounding out to make sure we cover everything
    log_noncell_counts = np.log(noncell_counts)
    x = np.arange(np.floor(log_noncell_counts.min()) - 0.01, np.ceil(log_noncell_counts.max()) + 0.01, 0.1)

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


def _get_cell_count_given_expected_cells(
    umi_counts: np.ndarray,
    expected_cells: int,
    sort_order: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute a cell UMI count prior given a known number of expected cells.

    Args:
        umi_counts: Array of UMI counts per droplet.
        expected_cells: Number of expected cells.
        sort_order: Optional pre-computed descending sort order of droplets.
            When provided (e.g. from GEX), this order is used instead of
            resorting by umi_counts.  Required for secondary modalities so that
            "cell" droplets are identified by their GEX rank, not their own rank.

    Returns:
        Dict with key 'cell_counts'.
    """
    if sort_order is None:
        sort_order = np.argsort(umi_counts)[::-1]
    top_k = umi_counts[sort_order[:expected_cells]]
    # np.maximum guard: secondary modalities can have zero counts in top cells
    cell_counts = np.exp(np.mean(np.log(np.maximum(top_k, 1)))).item()
    return {"cell_counts": cell_counts}


def _get_empty_count_given_expected_cells_and_total_droplets(
    umi_counts: np.ndarray,
    expected_cells: int,
    total_droplets: int,
    sort_order: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute an empty-droplet UMI count prior given known expected_cells and total_droplets.

    Args:
        umi_counts: Array of UMI counts per droplet.
        expected_cells: Number of expected cells.
        total_droplets: Total number of droplets to include.
        sort_order: Optional pre-computed descending sort order (see
            _get_cell_count_given_expected_cells for rationale).

    Returns:
        Dict with keys 'empty_counts' and 'empty_count_upper_limit'.
    """
    if sort_order is None:
        sort_order = np.argsort(umi_counts)[::-1]
    ordered = umi_counts[sort_order]
    starting_point = max(expected_cells, total_droplets - 500)
    empty_counts = np.median(ordered[int(starting_point) : int(total_droplets)]).item()

    # Upper limit: geometric mean of cell and empty priors, capped at 1.5x empty
    cell_counts_for_limit = np.exp(np.mean(np.log(np.maximum(ordered[:expected_cells], 1)))).item()
    middle = np.sqrt(cell_counts_for_limit * empty_counts)
    empty_count_upper_limit = min(middle, 1.5 * empty_counts)

    return {"empty_counts": empty_counts, "empty_count_upper_limit": empty_count_upper_limit}


def _get_cell_count_empty_count(umi_counts: np.ndarray, low_count_threshold: float = 15) -> Dict[str, float]:
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

    logger.debug("Beginning priors._get_cell_count_empty_count()")
    reverse_sorted_umi_counts = np.sort(umi_counts)[::-1]
    umi_counts_for_otsu = reverse_sorted_umi_counts[: (umi_counts > low_count_threshold).sum() // 4]

    log_cell_count_low_limit = _threshold_otsu(np.log(umi_counts_for_otsu))
    cell_count_low_limit = np.exp(log_cell_count_low_limit)

    logger.debug(f"cell_count_low_limit is {cell_count_low_limit}")
    cell_count_prior = np.mean(umi_counts[umi_counts > cell_count_low_limit])

    umi_counts_for_kde = reverse_sorted_umi_counts[reverse_sorted_umi_counts > low_count_threshold]

    # initial conditions for the loop
    # start low, but have a failsafe (especially for simulated data)
    cutoff = max(0.1 * cell_count_low_limit, umi_counts_for_kde[-100])
    empty_count_prior: float = -100.0
    empty_count_upper_limit: float = 0.0
    delta = np.inf
    a = 0

    # iterate to convergence, at most 5 times
    while delta > 10:
        logger.debug(f"cutoff = {cutoff}")

        # use gaussian_kde to find the peak in the histogram
        new_empty_count_prior, empty_count_upper_limit = _peak_density_given_cutoff(
            umi_counts=umi_counts_for_kde,
            cutoff=cutoff,
            cell_count_low_limit=cell_count_low_limit,
        )
        logger.debug(f"new_empty_count_prior = {new_empty_count_prior}")

        # 3/4 of the geometric mean is our new upper cutoff
        cutoff = 0.75 * np.sqrt(cell_count_prior * new_empty_count_prior)
        delta = np.abs(new_empty_count_prior - empty_count_prior)
        logger.debug(f"delta = {delta}")
        empty_count_prior = new_empty_count_prior
        a += 1
        if a >= 5:
            logger.debug("Heuristics for determining empty counts exceeded 5 iterations without converging")
            break

    # do a final estimation of cell counts:
    # go to the halfway point and then take the median of the droplets above
    count_crossover = np.sqrt(cell_count_prior * empty_count_prior)
    cell_count_prior = np.median(umi_counts[umi_counts > count_crossover])

    logger.debug(f"cell_count_prior is {cell_count_prior}")
    logger.debug(f"empty_count_prior is {empty_count_prior}")
    logger.debug("End of priors._get_cell_count_empty_count()")

    return {
        "cell_counts": cell_count_prior,
        "empty_counts": empty_count_prior,
        "empty_count_upper_limit": empty_count_upper_limit,
    }


def _get_expected_cells_and_total_droplets(
    umi_counts: np.ndarray,
    cell_counts: float,
    empty_counts: float,
    empty_count_upper_limit: float,
    max_empties: int | float = consts.MAX_EMPTIES_TO_INCLUDE,
) -> Dict[str, float]:
    """Obtain priors on cell counts and empty droplet counts from a UMI curve
    using heuristics, and without applying any other prior information.

    NOTE: to be run using inputs from _get_cell_count_empty_count()

    Args:
        umi_counts: Array of UMI counts per droplet, in no particular order
        cell_counts: Prior from _get_cell_count_empty_count()
        empty_counts: Prior from _get_cell_count_empty_count()
        empty_count_upper_limit: Prior from _get_cell_count_empty_count()
        max_empties: Do not include more putative empty droplets than this

    Returns:
        Dict with keys ['expected_cells', 'total_droplets', 'transition_point']

    Example:
        >>> priors = _get_cell_count_empty_count(umi_counts)
        >>> priors.update(_get_expected_cells_and_total_droplets(umi_counts, **priors))
    """
    # expected cells does well when you give it a very conservative estimate
    expected_cells = (umi_counts >= cell_counts).sum()

    # total droplets will be between empty_count_prior and its upper limit
    total_droplets_count_value = np.sqrt(empty_counts * empty_count_upper_limit)
    total_droplets = (umi_counts >= total_droplets_count_value).sum()

    # find the transition point
    count_crossover = np.sqrt(cell_counts * empty_counts)
    transition_point = (umi_counts >= count_crossover).sum()

    logger.debug(f"In _get_expected_cells_and_total_droplets(), found transition point at droplet {transition_point}")

    # ensure out heuristics don't go too far out datasets with many cells
    total_droplets = min(total_droplets, transition_point + max_empties)

    return {"expected_cells": expected_cells, "total_droplets": total_droplets, "transition_point": transition_point}


def _compute_crossover_d_std(
    umi_counts: np.ndarray,
    cell_counts: float,
    total_droplets: int,
) -> Dict[str, Any]:
    """Compute log-count crossover, surely_empty_counts, d_std, and d_empty_std.

    Args:
        umi_counts: Per-droplet UMI sums for this modality.
        cell_counts: Typical UMI count in a real cell (for this modality).
        total_droplets: Total droplets used in the analysis.

    Returns:
        Dict with keys 'surely_empty_counts', 'log_counts_crossover', 'd_std', 'd_empty_std'.
        d_std falls back to 0.1 if too few nonzero counts are available above the crossover.
    """
    reverse_sorted_counts = np.sort(umi_counts)[::-1]
    idx = min(int(total_droplets), len(reverse_sorted_counts) - 1)
    surely_empty_counts = float(reverse_sorted_counts[idx])
    log_counts_crossover = (np.log(max(surely_empty_counts, 1)) + np.log(max(cell_counts, 1))) / 2

    log_nonzero = np.log(umi_counts[umi_counts > 0])
    above_crossover = log_nonzero[log_nonzero > log_counts_crossover]
    d_std = float(np.std(above_crossover) / 5.0) if len(above_crossover) > 1 else 0.1

    return {
        "surely_empty_counts": surely_empty_counts,
        "log_counts_crossover": log_counts_crossover,
        "d_std": d_std,
        "d_empty_std": 0.01,
    }


def _compute_modality_umi_sums(
    matrix: sp.spmatrix,
    feature_types: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute per-droplet UMI sums for each modality.

    Args:
        matrix: (n_droplets, n_analyzed_features) sparse count matrix.
        feature_types: String array of feature types, shape (n_analyzed_features,).
            If None, all features are treated as Gene Expression.

    Returns:
        Dict mapping modality name to 1D array of per-droplet UMI sums.
    """
    if feature_types is None:
        return {GEX_FEATURE_TYPE: np.array(matrix.sum(axis=1)).squeeze()}

    unique_types = np.unique(feature_types)
    result: Dict[str, np.ndarray] = {}
    for mod in unique_types:
        mask = feature_types == mod
        result[mod] = np.array(matrix[:, mask].sum(axis=1)).squeeze()
    return result


def _get_priors_for_secondary_modality(
    umi_counts: np.ndarray,
    expected_cells: int,
    total_droplets: int,
    gex_sort_order: np.ndarray,
) -> Dict[str, Any]:
    """Compute priors for a non-GEX modality using GEX-derived cell/empty structure.

    Droplets are ranked by GEX UMI count (gex_sort_order), not by this modality's
    own counts.  This is because GEX is the most reliable indicator of cell presence,
    and secondary modalities (e.g. ATAC, Protein) can be too sparse to self-rank.

    Args:
        umi_counts: Per-droplet UMI sums for this modality.
        expected_cells: Number of cells (from GEX priors).
        total_droplets: Total droplets to analyze (from GEX priors).
        gex_sort_order: Descending sort order of droplets by GEX UMI count.

    Returns:
        Dict with keys: cell_counts, empty_counts, empty_count_upper_limit, d_std, d_empty_std.
    """
    cell_result = _get_cell_count_given_expected_cells(
        umi_counts=umi_counts,
        expected_cells=expected_cells,
        sort_order=gex_sort_order,
    )
    empty_result = _get_empty_count_given_expected_cells_and_total_droplets(
        umi_counts=umi_counts,
        expected_cells=expected_cells,
        total_droplets=total_droplets,
        sort_order=gex_sort_order,
    )

    # d_std from nonzero cell counts (using GEX-ordered top cells)
    top_cell_umi = umi_counts[gex_sort_order[:expected_cells]]
    nonzero_cell = top_cell_umi[top_cell_umi > 0]
    d_std = float(np.std(np.log(nonzero_cell)) / 5.0) if len(nonzero_cell) > 1 else 0.1

    return {
        "cell_counts": cell_result["cell_counts"],
        "empty_counts": empty_result["empty_counts"],
        "empty_count_upper_limit": empty_result["empty_count_upper_limit"],
        "d_std": d_std,
        "d_empty_std": 0.01,
    }


def get_all_priors(
    matrix: sp.spmatrix,
    analyzed_feature_types: Optional[np.ndarray],
    low_count_threshold: float,
    max_total_droplets: int = consts.MAX_TOTAL_DROPLETS_GUESSED,
    expected_cells_override: Optional[int] = None,
    total_droplets_override: Optional[int] = None,
    force_cell_umi_prior: Optional[float] = None,
    force_empty_umi_prior: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute priors for all modalities, anchoring cell/empty structure on GEX.

    Gene Expression UMI counts are used for all cell-calling heuristics (Otsu,
    KDE, crossover).  Secondary modalities inherit expected_cells and
    total_droplets from GEX, and use GEX droplet ordering to identify which
    droplets are cells.

    Args:
        matrix: (n_droplets, n_analyzed_features) sparse count matrix.
        analyzed_feature_types: Feature type string per analyzed feature, shape
            (n_analyzed_features,).  Pass None when all features are GEX.
        low_count_threshold: Droplets with fewer total GEX counts are ignored.
        max_total_droplets: Failsafe upper bound on total_droplets.
        expected_cells_override: User-supplied expected cell count.
        total_droplets_override: User-supplied total droplet count.
        force_cell_umi_prior: Force GEX cell UMI count prior to this value.
        force_empty_umi_prior: Force GEX empty UMI count prior to this value.

    Returns:
        Dict with:
          - 'expected_cells', 'total_droplets', 'transition_point'  (global, GEX-derived)
          - 'log_counts_crossover_gex', 'surely_empty_counts_gex'   (global, GEX-derived)
          - 'modalities': {feature_type_str: {cell_counts, empty_counts,
                           empty_count_upper_limit, d_std, d_empty_std}}
    """
    modality_sums = _compute_modality_umi_sums(matrix, analyzed_feature_types)
    gex_counts = modality_sums.get(GEX_FEATURE_TYPE)
    if gex_counts is None:
        # Data does not have a "Gene Expression" feature-type label (e.g., labeled "NA").
        # Fall back to using all features for prior estimation — same behaviour as the
        # legacy single-modality path.
        logger.debug(
            f"No '{GEX_FEATURE_TYPE}' feature type found in analyzed features; "
            "treating all features as Gene Expression for prior estimation."
        )
        gex_counts = np.array(matrix.sum(axis=1)).squeeze()
        modality_sums = {GEX_FEATURE_TYPE: gex_counts}

    # --- GEX heuristics: Otsu + iterative KDE ---
    logger.debug("Computing priors from the GEX UMI curve")
    gex_priors = _get_cell_count_empty_count(umi_counts=gex_counts, low_count_threshold=low_count_threshold)
    gex_priors.update(_get_expected_cells_and_total_droplets(umi_counts=gex_counts, **gex_priors))
    logger.debug(f"Automatically computed GEX priors: {gex_priors}")

    a = 0
    while gex_priors["total_droplets"] > max_total_droplets:
        logger.debug(
            f"Heuristics for estimating priors resulted in "
            f"{gex_priors['total_droplets']} total_droplets, which is "
            f"typically too large. Recomputing with "
            f"low_count_threshold = {gex_priors['empty_count_upper_limit']:.0f}"
        )
        gex_priors = _get_cell_count_empty_count(
            umi_counts=gex_counts,
            low_count_threshold=gex_priors["empty_count_upper_limit"],
        )
        gex_priors.update(_get_expected_cells_and_total_droplets(umi_counts=gex_counts, **gex_priors))
        logger.debug(f"Automatically computed GEX priors: {gex_priors}")
        a += 1
        if a > 5:
            break

    gex_sort_order = np.argsort(gex_counts)[::-1]

    # Assemble initial structured priors dict
    priors: Dict[str, Any] = {
        "expected_cells": gex_priors["expected_cells"],
        "total_droplets": gex_priors["total_droplets"],
        "transition_point": gex_priors["transition_point"],
        # crossover values are filled in at the end after all overrides
        "log_counts_crossover_gex": None,
        "surely_empty_counts_gex": None,
        # Descending sort of all droplets by GEX UMI count; used downstream
        # (e.g. guide transform computation) to identify high-confidence cells.
        "gex_sort_order": gex_sort_order,
        "modalities": {
            GEX_FEATURE_TYPE: {
                "cell_counts": gex_priors["cell_counts"],
                "empty_counts": gex_priors["empty_counts"],
                "empty_count_upper_limit": gex_priors["empty_count_upper_limit"],
                "d_std": None,
                "d_empty_std": None,
            }
        },
    }

    # --- Apply user overrides ---
    gex_mod = priors["modalities"][GEX_FEATURE_TYPE]

    if expected_cells_override is not None:
        logger.debug(f"Fixing expected_cells at {expected_cells_override}")
        priors["expected_cells"] = expected_cells_override
        cell_result = _get_cell_count_given_expected_cells(
            umi_counts=gex_counts,
            expected_cells=expected_cells_override,
            sort_order=gex_sort_order,
        )
        gex_mod["cell_counts"] = cell_result["cell_counts"]
        if (expected_cells_override + consts.NUM_EMPTIES_INCREMENT) > priors["total_droplets"]:
            total_drops = expected_cells_override + consts.NUM_EMPTIES_INCREMENT
            priors["total_droplets"] = total_drops
            logger.debug(f"Incrementing total_droplets to be {total_drops}")
            if total_droplets_override is None:
                empty_result = _get_empty_count_given_expected_cells_and_total_droplets(
                    umi_counts=gex_counts,
                    expected_cells=expected_cells_override,
                    total_droplets=total_drops,
                    sort_order=gex_sort_order,
                )
                gex_mod.update(empty_result)

    if total_droplets_override is not None:
        logger.debug(f"Fixing total_droplets at {total_droplets_override}")
        priors["total_droplets"] = total_droplets_override
        empty_result = _get_empty_count_given_expected_cells_and_total_droplets(
            umi_counts=gex_counts,
            expected_cells=int(priors["expected_cells"]),
            total_droplets=total_droplets_override,
            sort_order=gex_sort_order,
        )
        gex_mod.update(empty_result)

    if force_cell_umi_prior is not None:
        logger.debug(f"Forcing cell UMI count prior to be {force_cell_umi_prior}")
        gex_mod["cell_counts"] = force_cell_umi_prior

    if force_empty_umi_prior is not None:
        logger.debug(f"Forcing empty droplet UMI count prior to be {force_empty_umi_prior}")
        gex_mod["empty_counts"] = force_empty_umi_prior
        middle = np.sqrt(gex_mod["cell_counts"] * force_empty_umi_prior)
        gex_mod["empty_count_upper_limit"] = min(middle, 2 * force_empty_umi_prior)

    # --- Recompute crossover and d_std with final prior values ---
    gex_crossover = _compute_crossover_d_std(
        umi_counts=gex_counts,
        cell_counts=gex_mod["cell_counts"],
        total_droplets=int(priors["total_droplets"]),
    )
    priors["log_counts_crossover_gex"] = gex_crossover["log_counts_crossover"]
    priors["surely_empty_counts_gex"] = gex_crossover["surely_empty_counts"]
    gex_mod["d_std"] = gex_crossover["d_std"]
    gex_mod["d_empty_std"] = gex_crossover["d_empty_std"]

    logger.debug(f"Final GEX priors: {gex_mod}")

    # --- Secondary modalities ---
    for mod_type, mod_counts in modality_sums.items():
        if mod_type == GEX_FEATURE_TYPE:
            continue
        logger.debug(f"Computing priors for secondary modality: {mod_type}")
        priors["modalities"][mod_type] = _get_priors_for_secondary_modality(
            umi_counts=mod_counts,
            expected_cells=int(priors["expected_cells"]),
            total_droplets=int(priors["total_droplets"]),
            gex_sort_order=gex_sort_order,
        )

    return priors
