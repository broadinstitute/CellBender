"""Tests for multi-modality support: ATACModality, ProteinModality, and _build_model wiring."""

import argparse

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

import cellbender.remove_background.consts as consts
from cellbender.remove_background.data.priors import get_all_priors
from cellbender.remove_background.modality import ATACModality, ModalityPriors, ProteinModality
from cellbender.remove_background.run import _build_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sparse_matrix(n_barcodes: int, n_features: int, seed: int = 0) -> sp.csr_matrix:
    """Make a sparse matrix with a bimodal UMI distribution (cells + empties).

    Cells (first quarter of barcodes) have ~5 counts per feature.
    Empties (remaining barcodes) have ~0.05 counts per feature, giving
    a clear bimodal UMI curve that prior estimation can handle.
    """
    rng = np.random.default_rng(seed)
    n_cells = max(n_barcodes // 4, 100)
    n_empties = n_barcodes - n_cells
    cell_data = rng.poisson(5.0, (n_cells, n_features)).astype(np.float32)
    empty_data = rng.poisson(0.05, (n_empties, n_features)).astype(np.float32)
    data = np.vstack([cell_data, empty_data])
    # Ensure no all-zero columns.
    data[0, :] = np.maximum(data[0, :], 1)
    return sp.csr_matrix(data)


def _make_args() -> argparse.Namespace:
    return argparse.Namespace(
        z_dim=8,
        z_hidden_dims=[32],
        model="full",
        use_cuda=False,
    )


class _MockDataset:
    """Minimal stand-in for SingleCellRNACountsDataset used by _build_model."""

    def __init__(self, matrix: sp.csr_matrix, feature_types: np.ndarray):
        n_barcodes, n_features = matrix.shape
        self.data = {"matrix": matrix}
        self.analyzed_barcode_inds = np.arange(n_barcodes, dtype=int)
        self.empty_UMI_threshold = 5
        self.analyzed_gene_inds = np.arange(n_features, dtype=int)
        self._feature_types = feature_types

        # Build priors using the same function the real dataset uses.
        self.priors = get_all_priors(
            matrix=matrix,
            analyzed_feature_types=feature_types,
            low_count_threshold=5,
        )
        # Add chi_ambient / chi_bar (normally added by _estimate_chi_ambient).
        chi_ambient_arr = np.array(matrix.sum(axis=0)).squeeze().astype(np.float32) + 1e-6
        self.priors["chi_ambient"] = torch.tensor(chi_ambient_arr / chi_ambient_arr.sum())
        self.priors["chi_bar"] = None
        self.priors["cell_logit"] = 0.0

    @property
    def analyzed_feature_types(self) -> np.ndarray:
        return self._feature_types


def _make_gex_only_dataset(n_barcodes: int = 1000, n_gex: int = 50) -> _MockDataset:
    matrix = _make_sparse_matrix(n_barcodes, n_gex)
    feature_types = np.array([consts.GEX_FEATURE_TYPE] * n_gex, dtype=str)
    return _MockDataset(matrix, feature_types)


def _make_gex_atac_dataset(n_barcodes: int = 1000, n_gex: int = 40, n_atac: int = 20) -> _MockDataset:
    matrix = _make_sparse_matrix(n_barcodes, n_gex + n_atac)
    feature_types = np.array([consts.GEX_FEATURE_TYPE] * n_gex + [consts.ATAC_FEATURE_TYPE] * n_atac, dtype=str)
    return _MockDataset(matrix, feature_types)


def _make_gex_protein_dataset(n_barcodes: int = 1000, n_gex: int = 40, n_protein: int = 10) -> _MockDataset:
    matrix = _make_sparse_matrix(n_barcodes, n_gex + n_protein)
    feature_types = np.array([consts.GEX_FEATURE_TYPE] * n_gex + [consts.PROTEIN_FEATURE_TYPE] * n_protein, dtype=str)
    return _MockDataset(matrix, feature_types)


def _make_secondary_priors(n_features: int) -> ModalityPriors:
    return ModalityPriors(
        d_cell_loc_prior=5.0,
        d_cell_scale_prior=0.1,
        d_empty_loc_prior=2.0,
        d_empty_scale_prior=0.01,
        chi_ambient_init=torch.ones(n_features) / n_features,
    )


class _FakeEncoder(nn.Module):
    """Minimal encoder that mimics EncodeZ output (dict with loc/scale)."""

    def __init__(self, input_dim: int, z_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, z_dim)
        self._z_dim = z_dim

    def forward(self, x: torch.Tensor):
        loc = self.linear(x.float())
        scale = torch.ones_like(loc)
        return {"loc": loc, "scale": scale}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_feature_type_constants_exist():
    assert consts.GEX_FEATURE_TYPE == "Gene Expression"
    assert consts.ATAC_FEATURE_TYPE == "Peaks"
    assert consts.PROTEIN_FEATURE_TYPE == "Antibody Capture"
    assert consts.CRISPR_FEATURE_TYPE == "CRISPR Guide Capture"


def test_atac_modality_instantiation():
    n_features, z_dim = 20, 8
    encoder = _FakeEncoder(n_features, z_dim)
    decoder = nn.Linear(2 * z_dim, n_features)
    decoder.input_dim = 2 * z_dim
    mod = ATACModality(
        priors=_make_secondary_priors(n_features),
        feature_indices_f=torch.arange(n_features),
        encoder=encoder,
        decoder=decoder,
        z_dim=z_dim,
    )
    assert mod.name == "atac"
    assert not mod.contributes_to_cell_probability
    assert not mod.contributes_to_epsilon_inference
    assert mod.z_dim == z_dim


def test_protein_modality_instantiation():
    n_features, z_dim = 15, 8
    encoder = _FakeEncoder(n_features, z_dim)
    decoder = nn.Linear(2 * z_dim, n_features)
    decoder.input_dim = 2 * z_dim
    mod = ProteinModality(
        priors=_make_secondary_priors(n_features),
        feature_indices_f=torch.arange(n_features),
        encoder=encoder,
        decoder=decoder,
        z_dim=z_dim,
    )
    assert mod.name == "protein"
    assert not mod.contributes_to_cell_probability
    assert not mod.contributes_to_epsilon_inference
    assert mod.z_dim == z_dim


def test_prepare_decoder_input_detaches_z_gex():
    """z_gex in prepare_decoder_input must be detached so ATAC loss cannot
    update the GEX encoder."""
    n_features, z_dim = 20, 8
    encoder = _FakeEncoder(n_features, z_dim)
    decoder = nn.Linear(2 * z_dim, n_features)
    decoder.input_dim = 2 * z_dim
    mod = ATACModality(
        priors=_make_secondary_priors(n_features),
        feature_indices_f=torch.arange(n_features),
        encoder=encoder,
        decoder=decoder,
        z_dim=z_dim,
    )
    z_gex = torch.randn(4, z_dim, requires_grad=True)
    z_atac = torch.randn(4, z_dim, requires_grad=True)
    dec_input = mod.prepare_decoder_input({"gene_expression": z_gex, "atac": z_atac})
    assert dec_input.shape == (4, 2 * z_dim)
    # Backprop through dec_input: z_atac should receive gradients,
    # z_gex should NOT (it was detached before concatenation).
    dec_input.sum().backward()
    assert z_gex.grad is None, "z_gex should be detached — no gradient should flow to GEX encoder"
    assert z_atac.grad is not None, "z_atac should receive gradients through the ATAC decoder"


def test_priors_secondary_modality_atac():
    """get_all_priors returns an entry for the ATAC modality."""
    n_barcodes, n_gex, n_atac = 200, 40, 20
    matrix = _make_sparse_matrix(n_barcodes, n_gex + n_atac)
    feature_types = np.array([consts.GEX_FEATURE_TYPE] * n_gex + [consts.ATAC_FEATURE_TYPE] * n_atac, dtype=str)
    priors = get_all_priors(matrix=matrix, analyzed_feature_types=feature_types, low_count_threshold=5)
    assert consts.GEX_FEATURE_TYPE in priors["modalities"]
    assert consts.ATAC_FEATURE_TYPE in priors["modalities"]
    atac_p = priors["modalities"][consts.ATAC_FEATURE_TYPE]
    for key in ("cell_counts", "empty_counts", "d_std", "d_empty_std"):
        assert key in atac_p, f"Missing key '{key}' in ATAC priors"


def test_priors_secondary_modality_protein():
    """get_all_priors returns an entry for the Protein modality."""
    n_barcodes, n_gex, n_protein = 200, 40, 10
    matrix = _make_sparse_matrix(n_barcodes, n_gex + n_protein)
    feature_types = np.array([consts.GEX_FEATURE_TYPE] * n_gex + [consts.PROTEIN_FEATURE_TYPE] * n_protein, dtype=str)
    priors = get_all_priors(matrix=matrix, analyzed_feature_types=feature_types, low_count_threshold=5)
    assert consts.PROTEIN_FEATURE_TYPE in priors["modalities"]


def test_build_model_gex_only():
    """_build_model with GEX-only data produces exactly one modality."""
    dataset = _make_gex_only_dataset()
    count_matrix = dataset.data["matrix"][dataset.analyzed_barcode_inds, :][:, dataset.analyzed_gene_inds]
    model = _build_model(sp.csr_matrix(count_matrix), _make_args(), dataset)
    assert set(model.modalities.keys()) == {"gene_expression"}


def test_build_model_gex_atac():
    """_build_model with GEX+ATAC data produces gene_expression and atac modalities."""
    dataset = _make_gex_atac_dataset(n_gex=40, n_atac=20)
    count_matrix = sp.csr_matrix(dataset.data["matrix"])
    model = _build_model(count_matrix, _make_args(), dataset)
    assert "gene_expression" in model.modalities
    assert "atac" in model.modalities
    assert model.modalities["atac"].feature_indices_f.shape[0] == 20
    # z_dim must be stored on secondary modality.
    assert model.modalities["atac"].z_dim == _make_args().z_dim


def test_build_model_gex_protein():
    """_build_model with GEX+Protein data produces gene_expression and protein modalities."""
    dataset = _make_gex_protein_dataset(n_gex=40, n_protein=10)
    count_matrix = sp.csr_matrix(dataset.data["matrix"])
    model = _build_model(count_matrix, _make_args(), dataset)
    assert "gene_expression" in model.modalities
    assert "protein" in model.modalities
    assert model.modalities["protein"].feature_indices_f.shape[0] == 10


def test_build_model_gex_feature_indices_correct():
    """GEX feature_indices_f must cover only GEX columns; ATAC indices must be disjoint."""
    dataset = _make_gex_atac_dataset(n_gex=40, n_atac=20)
    count_matrix = sp.csr_matrix(dataset.data["matrix"])
    model = _build_model(count_matrix, _make_args(), dataset)
    gex_idx = model.modalities["gene_expression"].feature_indices_f
    atac_idx = model.modalities["atac"].feature_indices_f
    assert gex_idx.shape[0] == 40
    assert len(set(gex_idx.tolist()) & set(atac_idx.tolist())) == 0


def test_build_model_secondary_decoder_input_dim():
    """Secondary modality decoder input must be 2*z_dim (z_gex ++ z_self)."""
    args = _make_args()
    dataset = _make_gex_atac_dataset(n_gex=40, n_atac=20)
    count_matrix = sp.csr_matrix(dataset.data["matrix"])
    model = _build_model(count_matrix, args, dataset)
    atac_decoder = model.modalities["atac"].decoder
    assert atac_decoder.input_dim == 2 * args.z_dim
