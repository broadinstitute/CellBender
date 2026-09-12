"""Compute the static GEX transform used as input to the GuidePerturbationModality encoder.

The transform produces HVG-filtered, z-scored, clamped log1p gene expression
features that carry perturbation-identity signal relative to negative control cells.
"""

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from cellbender.remove_background import consts

logger = logging.getLogger("cellbender")


@dataclass
class GuideTransformParams:
    """Parameters for the guide perturbation encoder input transform."""

    mean_ctrl: np.ndarray  # (n_hvg,) float32
    std_ctrl: np.ndarray  # (n_hvg,) float32
    hvg_indices: np.ndarray  # (n_hvg,) int64 - indices into GEX feature axis
    clamp_min: float
    clamp_max: float
    n_control_cells: int
    n_hvg: int


def _decode_names(names: np.ndarray) -> np.ndarray:
    """Decode byte-string feature names to str, no-op if already str."""
    if len(names) == 0:
        return names
    if isinstance(names[0], (bytes, np.bytes_)):
        return np.array([n.decode("utf-8") for n in names], dtype=str)
    return names


def _log1p_col_variance(mat: sp.spmatrix) -> np.ndarray:
    """Column-wise variance of log1p(mat) using sparse arithmetic.

    Zero entries contribute log1p(0) = 0 to both the mean and variance.
    """
    mat_lp = mat.astype(np.float64, copy=True).tocsr()
    mat_lp.data = np.log1p(mat_lp.data)
    n = mat.shape[0]
    sum_f = np.asarray(mat_lp.sum(axis=0)).squeeze()
    sum_sq_f = np.asarray(mat_lp.power(2).sum(axis=0)).squeeze()
    mean_f = sum_f / n
    var_f = (sum_sq_f / n - mean_f**2).clip(0)
    return var_f.astype(np.float32)


def compute_guide_transform_params(
    count_matrix: sp.csr_matrix,
    analyzed_feature_types: np.ndarray,
    raw_neg_ctrl_counts_n: np.ndarray,
    raw_total_guide_counts_n: np.ndarray,
    gex_sort_order: np.ndarray,
    expected_cells: int,
    num_hvgs: int = 2000,
    min_expression_frac: float = 0.05,
    min_ctrl_cells: int = 200,
    max_ctrl_cells: int = 2000,
    clamp_percentile: float = 1.0,
) -> GuideTransformParams:
    """Compute transform parameters for the guide perturbation encoder input.

    Identifies confident negative-control cells by scoring high-confidence
    cells on negative control guide count fraction and magnitude.  Uses those
    cells to compute per-gene mean and std of log1p(GEX) on HVG-selected genes,
    then computes data-derived z-score clamp bounds.

    The resulting transform applied to a cell is:
        1. Select HVG columns from GEX counts.
        2. Apply log1p.
        3. Z-score: (log1p(x) - mean_ctrl) / std_ctrl.
        4. Clamp to [clamp_min, clamp_max].

    Negative control feature matching and the corresponding ValueError are
    handled by the caller (run.py), which passes pre-computed per-droplet count
    vectors.  Using the raw (pre-filtering) matrix there ensures the negative
    control feature is available even when CellBender's ambient filter would
    exclude it from noise modelling.

    Args:
        count_matrix: (n_droplets, n_analyzed_features) sparse CSR matrix of
            raw counts, columns ordered by analyzed_feature_types.
        analyzed_feature_types: Feature type label per analyzed feature column.
        raw_neg_ctrl_counts_n: (n_droplets,) array of per-droplet total counts
            in the negative control guide feature(s), extracted from the
            unfiltered raw matrix.  Row order must match count_matrix.
        raw_total_guide_counts_n: (n_droplets,) array of per-droplet total
            counts across all CRISPR guide features, from the unfiltered raw
            matrix.  Used to compute the negative-control fraction.
        gex_sort_order: Descending sort of droplets by GEX total UMI count.
        expected_cells: Number of expected cells (from GEX priors).
        num_hvgs: Maximum number of highly variable genes to select.
        min_expression_frac: Exclude genes detected in fewer than this fraction
            of high-confidence cells.
        min_ctrl_cells: Issue a warning if fewer control cells are identified.
        max_ctrl_cells: Maximum number of control cells to use.
        clamp_percentile: Lower/upper percentile for z-score output clamping.
            1.0 → clamp at the 1st/99th percentile of the z-scored training cells.

    Returns:
        GuideTransformParams with mean_ctrl, std_ctrl, hvg_indices, clamp_min,
        clamp_max, n_control_cells, and n_hvg.

    Raises:
        ValueError: If no high-confidence cells pass the negative control filter.
    """
    # --- Extract GEX submatrix ---
    gex_mask = analyzed_feature_types == consts.GEX_FEATURE_TYPE
    gex_matrix: sp.csr_matrix = count_matrix[:, gex_mask].tocsr()

    # --- Identify high-confidence cells ---
    n_cells = min(expected_cells, gex_sort_order.shape[0])
    cell_inds = gex_sort_order[:n_cells]

    # --- Score high-confidence cells for negative control identity ---
    # score = neg_ctrl_frac * log1p(neg_ctrl_count): rewards both high fraction
    # and high absolute count jointly.
    neg_ctrl_counts = raw_neg_ctrl_counts_n[cell_inds]
    total_guide_counts = raw_total_guide_counts_n[cell_inds]
    neg_ctrl_frac = neg_ctrl_counts / (total_guide_counts + 1e-6)
    score = neg_ctrl_frac * np.log1p(neg_ctrl_counts)

    eligible = (neg_ctrl_frac >= 0.5) & (neg_ctrl_counts >= 5)
    if not eligible.any():
        raise ValueError(
            "No high-confidence cells have negative control guide fraction >= 0.5 "
            "with absolute count >= 5.  Check that --negative-control-guide-feature "
            "correctly identifies your negative control guides and that the "
            "experiment contains negative control cells."
        )

    eligible_inds = np.where(eligible)[0]
    sorted_eligible = eligible_inds[np.argsort(score[eligible_inds])[::-1]]
    ctrl_local_inds = sorted_eligible[:max_ctrl_cells]
    ctrl_cell_inds = cell_inds[ctrl_local_inds]
    n_ctrl = len(ctrl_cell_inds)

    if n_ctrl < min_ctrl_cells:
        logger.warning(
            f"Only {n_ctrl} confident negative control cells found "
            f"(recommended minimum: {min_ctrl_cells}).  "
            f"Transform statistics may be less reliable.  "
            f"Consider including more negative control cells in the experiment."
        )
    logger.info(f"Using {n_ctrl} confident negative control cells for guide transform computation")

    # --- HVG selection ---
    # Expression filter: exclude genes detected in < min_expression_frac of cells.
    gex_cell_sparse = gex_matrix[cell_inds, :]  # (n_cells, n_gex)
    gex_ctrl_sparse = gex_matrix[ctrl_cell_inds, :]  # (n_ctrl, n_gex)

    expr_frac = np.asarray((gex_cell_sparse > 0).mean(axis=0)).ravel()  # (n_gex,)
    expr_mask = expr_frac >= min_expression_frac
    expr_inds = np.where(expr_mask)[0]

    if len(expr_inds) == 0:
        raise ValueError(
            f"No genes pass the minimum expression filter "
            f"(min_expression_frac={min_expression_frac}).  "
            f"This is unexpected — check that the input matrix is not empty."
        )

    # Variance ratio: high var across all cells + low var in controls = perturbation signal.
    var_all_gex = _log1p_col_variance(gex_cell_sparse)  # (n_gex,)
    var_ctrl_gex = _log1p_col_variance(gex_ctrl_sparse)  # (n_gex,)

    var_all_expr = var_all_gex[expr_inds]
    var_ctrl_expr = var_ctrl_gex[expr_inds]
    reg = max(float(np.median(var_ctrl_expr)), 1e-6)
    hvg_score = var_all_expr / (var_ctrl_expr + reg)

    actual_num_hvgs = min(num_hvgs, len(expr_inds))
    top_within_expr = np.argsort(hvg_score)[::-1][:actual_num_hvgs]
    hvg_gex_indices = expr_inds[top_within_expr].astype(np.int64)  # (n_hvg,)

    logger.info(
        f"Selected {len(hvg_gex_indices)} HVGs (from {len(expr_inds)} expressed genes) for guide perturbation encoder"
    )

    # --- Compute mean and std on control cells at HVG genes ---
    ctrl_hvg_dense = np.asarray(gex_ctrl_sparse[:, hvg_gex_indices].todense(), dtype=np.float32)  # (n_ctrl, n_hvg)
    ctrl_log_hvg = np.log1p(ctrl_hvg_dense)
    mean_ctrl = ctrl_log_hvg.mean(axis=0).astype(np.float32)  # (n_hvg,)
    std_ctrl = ctrl_log_hvg.std(axis=0).astype(np.float32)  # (n_hvg,)

    # Floor std_ctrl to the median of positive std values.  Genes with
    # near-zero expression in control cells have std ≈ 0, which makes the
    # z-score denominator (≥ 1e-6) produce values in the millions and causes
    # the encoder's exp(scale_output) to overflow to inf → NaN during training.
    positive_std = std_ctrl[std_ctrl > 0]
    std_floor = float(np.median(positive_std)) if len(positive_std) > 0 else 1.0
    std_ctrl = np.maximum(std_ctrl, std_floor).astype(np.float32)

    # --- Compute clamp bounds from z-scored high-confidence cells ---
    cells_hvg_dense = np.asarray(gex_cell_sparse[:, hvg_gex_indices].todense(), dtype=np.float32)  # (n_cells, n_hvg)
    cells_log_hvg = np.log1p(cells_hvg_dense)
    std_safe = np.maximum(std_ctrl, 1e-6)
    z_scored = (cells_log_hvg - mean_ctrl) / std_safe  # (n_cells, n_hvg)
    # Cap clamp bounds at a hard limit: genes with residual near-zero std can
    # still produce very large z-scores; ±10 is a safe range for encoder input.
    _CLAMP_ABS_MAX = 10.0
    clamp_min = max(float(np.percentile(z_scored, clamp_percentile)), -_CLAMP_ABS_MAX)
    clamp_max = min(float(np.percentile(z_scored, 100.0 - clamp_percentile)), _CLAMP_ABS_MAX)

    logger.debug(f"Guide transform clamp bounds: [{clamp_min:.2f}, {clamp_max:.2f}]")

    return GuideTransformParams(
        mean_ctrl=mean_ctrl,
        std_ctrl=std_ctrl,
        hvg_indices=hvg_gex_indices,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        n_control_cells=n_ctrl,
        n_hvg=len(hvg_gex_indices),
    )
