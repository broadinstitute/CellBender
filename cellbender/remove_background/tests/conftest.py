"""Test utility functions and session-scoped fixtures."""

import pytest
import scipy.sparse as sp
import numpy as np
import torch

from cellbender.remove_background.data.extras.simulate import \
    generate_sample_dirichlet_dataset
from cellbender.remove_background.data.io import write_matrix_to_cellranger_h5

import shutil


USE_CUDA = torch.cuda.is_available()


def sparse_matrix_equal(mat1, mat2):
    """Fast assertion that sparse matrices are equal"""
    if type(mat1) == sp.coo_matrix:
        return coo_equal(mat1, mat2)
    elif (type(mat1) == sp.csc_matrix) or (type(mat1) == sp.csr_matrix):
        return csc_or_csr_equal(mat1, mat2)
    else:
        raise ValueError(f"Error with test functions: sparse_matrix_equal() was called with a {type(mat1)}")


def csc_or_csr_equal(mat1: sp.csc_matrix,
                     mat2: sp.csc_matrix):
    """Fast assertion that CSC sparse matrices are equal"""
    return (mat1 != mat2).sum() == 0


def coo_equal(mat1: sp.csc_matrix,
              mat2: sp.csc_matrix):
    """Fast assertion that COO sparse matrices are equal"""
    return (
        ((mat1.data != mat2.data).sum() == 0)
        and ((mat1.row != mat2.row).sum() == 0)
        and ((mat1.col != mat2.col).sum() == 0)
    )


def tensors_equal(a: torch.Tensor,
                  b: torch.Tensor):
    """Assertion that torch tensors are equal for each element"""
    return (a == b).all()


def string_ndarray_equality(a: np.ndarray, b: np.ndarray) -> bool:
    return (a.astype(str) == b.astype(str)).all()


class SavedFileH5:
    def __init__(self, name, keys, version, shape, analyzed_shape=None,
                 local_keys=None, global_keys=None, meta_keys=None, barcodes_analyzed=None):
        self.name = name
        self.version = version
        self.keys = keys
        self.shape = shape
        self.analyzed_shape = analyzed_shape
        self.local_keys = local_keys
        self.global_keys = global_keys
        self.meta_keys = meta_keys
        self.barcodes_analyzed = barcodes_analyzed

    def __repr__(self):
        return f'Saved h5 file ({self.shape}) at {self.name} ' \
            f'(CellRanger v{self.version} with keys {self.keys})'


@pytest.fixture(scope='session')
def simulated_dataset():
    """Generate a small simulated dataset Dict once and make it visible to all tests"""
    d = generate_sample_dirichlet_dataset(
        n_droplets=100,
        n_genes=1000,
        chi_artificial_similarity=0.25,
        cells_of_each_type=[10, 10, 10],
        cell_mean_umi=[5000, 5000, 5000],
        cell_lognormal_sigma=0.01,
        empty_mean_umi=200,
        empty_lognormal_sigma=0.01,
        model_type='full',
        dirichlet_alpha=0.05,
        epsilon_param=20,
        rho_alpha=4,
        rho_beta=96,
        random_seed=0,
    )
    d['genomes'] = np.array(['simulated'] * d['gene_names'].size)
    d['gene_ids'] = np.array(['ENSEMBLSIM000' + s for s in d['gene_names']])
    d['feature_types'] = np.array(['Gene Expression'] * (d['gene_names'].size - 50)
                                  + ['Antibody Capture'] * 40
                                  + ['CRISPR Guide Capture'] * 5
                                  + ['Custom'] * 5)  # a mix of types
    d['matrix'] = (d['counts_true'] + d['counts_bkg']).tocsc()
    return d


@pytest.fixture(scope='session')
def h5_v3_file(tmpdir_factory, simulated_dataset) -> 'SavedFileH5':
    """Save an h5 file once and make it visible to all tests."""
    tmp_dir = tmpdir_factory.mktemp('data')
    filename = tmp_dir.join('tmp_v3.h5')
    cellranger_version = 3
    d = simulated_dataset
    write_matrix_to_cellranger_h5(
        output_file=filename,
        gene_names=d['gene_names'],
        gene_ids=d['gene_ids'],
        feature_types=d['feature_types'],
        genomes=d['genomes'],
        barcodes=d['barcodes'],
        count_matrix=d['matrix'],
        cellranger_version=cellranger_version,
    )
    yield SavedFileH5(name=filename,
                      keys=['gene_names', 'barcodes', 'genomes', 'gene_ids', 'feature_types'],
                      version=cellranger_version,
                      shape=d['matrix'].shape)
    shutil.rmtree(str(tmp_dir))


@pytest.fixture(scope='session')
def h5_v2_file(tmpdir_factory, simulated_dataset) -> 'SavedFileH5':
    """Save an h5 file once and make it visible to all tests."""
    tmp_dir = tmpdir_factory.mktemp('data')
    filename = tmp_dir.join('tmp_v2.h5')
    cellranger_version = 2
    d = simulated_dataset
    write_matrix_to_cellranger_h5(
        output_file=filename,
        gene_names=d['gene_names'],
        gene_ids=d['gene_ids'],
        # feature_types=d['feature_types'],
        # genomes=d['genomes'],
        barcodes=d['barcodes'],
        count_matrix=d['matrix'],
        cellranger_version=cellranger_version,
    )
    yield SavedFileH5(name=filename,
                      keys=['gene_names', 'barcodes', 'gene_ids'],
                      version=cellranger_version,
                      shape=d['matrix'].shape)
    shutil.rmtree(str(tmp_dir))


@pytest.fixture(scope='session')
def h5_v2_file_missing_ids(tmpdir_factory, simulated_dataset) -> 'SavedFileH5':
    """Save an h5 file once and make it visible to all tests."""
    tmp_dir = tmpdir_factory.mktemp('data')
    filename = tmp_dir.join('tmp_v2.h5')
    cellranger_version = 2
    d = simulated_dataset
    write_matrix_to_cellranger_h5(
        output_file=filename,
        gene_names=d['gene_names'],
        # gene_ids=d['gene_ids'],
        # feature_types=d['feature_types'],
        # genomes=d['genomes'],
        barcodes=d['barcodes'],
        count_matrix=d['matrix'],
        cellranger_version=cellranger_version,
    )
    yield SavedFileH5(name=filename,
                      keys=['gene_names', 'barcodes'],
                      version=cellranger_version,
                      shape=d['matrix'].shape)
    shutil.rmtree(str(tmp_dir))


@pytest.fixture(scope='session', params=[2, 3])
def h5_file(request, h5_v2_file, h5_v3_file):
    if request.param == 2:
        return h5_v2_file
    elif request.param == 3:
        return h5_v3_file
    else:
        raise ValueError(f'Test error: requested v{request.param} h5 file')


@pytest.fixture(scope='session')
def h5_v3_file_post_inference(tmpdir_factory, simulated_dataset) -> 'SavedFileH5':
    """Save an h5 file once and make it visible to all tests."""
    tmp_dir = tmpdir_factory.mktemp('data')
    filename = tmp_dir.join('tmp_v3_inferred.h5')
    cellranger_version = 3
    d = simulated_dataset

    barcodes_analyzed = 50

    write_matrix_to_cellranger_h5(
        output_file=filename,
        gene_names=d['gene_names'],
        gene_ids=d['gene_ids'],
        feature_types=d['feature_types'],
        genomes=d['genomes'],
        barcodes=d['barcodes'],
        count_matrix=d['matrix'],
        cellranger_version=cellranger_version,
        local_latents={'barcode_indices_for_latents': np.arange(0, barcodes_analyzed),
                       'gene_expression_encoding': np.random.rand(barcodes_analyzed, 10),
                       'cell_probability': np.random.rand(barcodes_analyzed),
                       'd': np.random.rand(barcodes_analyzed) + 5.},
        global_latents={'global_var1': np.array([1, 2, 3])},
        metadata={'metadata1': np.array(0.9),
                  'metadata2': {'key1': 'val1', 'key2': {'a': 'val2', 'b': 'val3'}},
                  'metadata3': None,
                  'metadata4': 0.5,
                  'metadata5': 'text'},
    )
    yield SavedFileH5(name=filename,
                      keys=['gene_names', 'barcodes', 'genomes', 'gene_ids', 'feature_types'],
                      version=cellranger_version,
                      shape=d['matrix'].shape,
                      analyzed_shape=(barcodes_analyzed, d['matrix'].shape[1]),
                      local_keys=['d', 'cell_probability', 'gene_expression_encoding'],
                      global_keys=['global_var1'],
                      meta_keys=['metadata1'],
                      barcodes_analyzed=barcodes_analyzed)
    shutil.rmtree(str(tmp_dir))
