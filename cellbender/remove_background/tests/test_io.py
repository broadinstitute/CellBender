"""Test input reading and output writing functionality."""

import pytest
import numpy as np
from scipy.io import mmwrite
import scipy.sparse as sp

from cellbender.remove_background.data.io import \
    load_data, get_matrix_from_cellranger_mtx, get_matrix_from_dropseq_dge, \
    get_matrix_from_bd_rhapsody, get_matrix_from_anndata, \
    detect_cellranger_version_h5, detect_cellranger_version_mtx, unravel_dict

from cellbender.remove_background.tests.conftest import \
    sparse_matrix_equal, string_ndarray_equality, h5_v2_file, \
    h5_v2_file_missing_ids, h5_v3_file, h5_file

from typing import List, Dict, Optional
import gzip
import shutil


def assert_loaded_matches_saved(d: Dict[str, np.ndarray],
                                loaded: Dict[str, np.ndarray],
                                keys: List[str],
                                cellranger_version: Optional[int] = None):
    """Check if a loaded file's data matches the data that was saved.

    Args:
        d: Dict of the data that was saved
        loaded: Dict of the data that was loaded from file
        keys: List of keys to the dicts that will be checked for equality
        cellranger_version: In [2, 3]
    """
    if 'cellranger_version' in loaded.keys():
        assert cellranger_version == loaded['cellranger_version']
    assert sparse_matrix_equal(loaded['matrix'], d['matrix'])
    for key in keys:
        if d[key] is None:
            continue
        assert loaded[key] is not None, \
            f'Loaded h5 key "{key}" was None, but data was saved: {d[key][:5]} ...'
        assert string_ndarray_equality(d[key], loaded[key]), \
            f'Loaded h5 key "{key}" did not match saved data'


@pytest.mark.parametrize('filetype', ['h5_v2_file', 'h5_v2_file_missing_ids', 'h5_v3_file'])
def test_simulate_save_load_h5(simulated_dataset,
                               filetype,
                               h5_v2_file,
                               h5_v2_file_missing_ids,
                               h5_v3_file):

    # get information from fixture, since you cannot pass fixtures to parametrize
    if filetype == 'h5_v2_file':
        saved_h5 = h5_v2_file
    elif filetype == 'h5_v2_file_missing_ids':
        saved_h5 = h5_v2_file_missing_ids
    elif filetype == 'h5_v3_file':
        saved_h5 = h5_v3_file

    # load data from file, using auto-loading, as it would be run
    loaded = load_data(input_file=saved_h5.name)

    # assert equality
    assert_loaded_matches_saved(
        d=simulated_dataset,
        loaded=loaded,
        keys=saved_h5.keys,
        cellranger_version=saved_h5.version,
    )


def test_detect_cellranger_version_h5(h5_file):
    v = detect_cellranger_version_h5(filename=h5_file.name)
    true_version = h5_file.version
    assert v == true_version


def gzip_file(file):
    """gzip a file"""
    with open(file, 'rb') as f_in, gzip.open(file + '.gz', 'wb') as f_out:
        f_out.writelines(f_in)


def save_mtx(tmpdir_factory, simulated_dataset, version: int) -> str:
    """Save data files in MTX format and return the directory path"""
    dirname = tmpdir_factory.mktemp(f'mtx_v{version}')

    # barcodes and sparse matrix... seems the MTX matrix is transposed
    mmwrite(dirname.join('matrix.mtx'), simulated_dataset['matrix'].transpose())
    np.savetxt(dirname.join('barcodes.tsv'), simulated_dataset['barcodes'], fmt='%s')

    # features and gzipping if v3
    features = np.concatenate((np.expand_dims(simulated_dataset['gene_ids'], axis=1),
                               np.expand_dims(simulated_dataset['gene_names'], axis=1),
                               np.expand_dims(simulated_dataset['feature_types'], axis=1)), axis=1)

    if version == 3:
        np.savetxt(dirname.join('features.tsv'), features, fmt='%s', delimiter='\t')
        gzip_file(dirname.join('matrix.mtx'))
        gzip_file(dirname.join('barcodes.tsv'))
        gzip_file(dirname.join('features.tsv'))
    elif version == 2:
        np.savetxt(dirname.join('genes.tsv'), features[:, :2], fmt='%s', delimiter='\t')
    else:
        raise ValueError(f'Test problem: version is {version}, but [2, 3] allowed')

    return dirname


@pytest.fixture(scope='session', params=[2, 3])
def mtx_directory(request, tmpdir_factory, simulated_dataset):
    dirname = save_mtx(tmpdir_factory=tmpdir_factory,
                       simulated_dataset=simulated_dataset,
                       version=request.param)
    yield dirname
    shutil.rmtree(str(dirname))


def test_detect_cellranger_version_mtx(mtx_directory):
    v = detect_cellranger_version_mtx(filedir=mtx_directory)
    true_version = 2 if ('_v2' in str(mtx_directory)) else 3
    assert v == true_version


def test_load_mtx(simulated_dataset, mtx_directory):

    # use the correct loader function
    loaded = get_matrix_from_cellranger_mtx(filedir=mtx_directory)
    version = 3 if ('_v3' in str(mtx_directory)) else 2
    assert_loaded_matches_saved(d=simulated_dataset,
                                loaded=loaded,
                                keys=(['gene_ids', 'gene_names', 'barcodes']
                                      + (['feature_types'] if (version == 3) else [])),
                                cellranger_version=version)

    # use auto-loading, as it would be run
    loaded = load_data(mtx_directory)
    assert_loaded_matches_saved(d=simulated_dataset,
                                loaded=loaded,
                                keys=(['gene_ids', 'gene_names', 'barcodes']
                                      + (['feature_types'] if (version == 3) else [])),
                                cellranger_version=version)


def save_dge(tmpdir_factory, simulated_dataset, do_gzip) -> str:
    """Save data files in DGE format and return the file path"""
    sep = '\t'
    name = 'dge.txt'
    if do_gzip:
        name = name + '.gz'
    tmp_dir = tmpdir_factory.mktemp('dge')
    filename = tmp_dir.join(name)
    load_fcn = gzip.open if do_gzip else open

    def row_generator(mat: sp.csc_matrix) -> List[str]:
        for i in range(mat.shape[1]):
            yield np.array(mat[:, i].todense()).squeeze().astype(int).astype(str).tolist()

    with load_fcn(filename, 'wb') as f:
        f.write(b'# some kind of header!\n')
        f.write(sep.join(['GENE'] + simulated_dataset['barcodes'].astype(str).tolist()).encode() + b'\n')
        for g, vals in zip(simulated_dataset['gene_names'], row_generator(simulated_dataset['matrix'])):
            f.write(sep.join([g] + vals).encode() + b'\n')

    return filename, tmp_dir


@pytest.fixture(scope='session', params=[True, False], ids=lambda x: 'gzipped' if x else 'not')
def dge_file(request, tmpdir_factory, simulated_dataset):
    filename, tmp_dir = save_dge(tmpdir_factory=tmpdir_factory,
                                 simulated_dataset=simulated_dataset,
                                 do_gzip=request.param)
    yield filename
    shutil.rmtree(str(tmp_dir))


def test_load_dge(simulated_dataset, dge_file):

    # use the correct loader function
    loaded = get_matrix_from_dropseq_dge(str(dge_file))
    assert_loaded_matches_saved(d=simulated_dataset,
                                loaded=loaded,
                                keys=['gene_names', 'barcodes'])

    # use auto-loading, as it would be run
    loaded = load_data(str(dge_file))
    assert_loaded_matches_saved(d=simulated_dataset,
                                loaded=loaded,
                                keys=['gene_names', 'barcodes'])


@pytest.mark.skip
def test_load_bd():
    pass


@pytest.mark.skip
def test_load_anndata():
    pass


@pytest.mark.skip
def test_load_loom():
    pass


@pytest.mark.skip
def test_write_matrix_to_cellranger_h5():
    pass


@pytest.mark.skip
def test_write_denoised_count_matrix():
    # from run.py, but should probably be refactored to io.py
    pass


def test_unravel_dict():
    key, value = 'pref', {'a': 1, 'b': {'c': 2, 'd': {'e': 3, 'f': 4}}}
    answer = {'pref_a': 1, 'pref_b_c': 2, 'pref_b_d_e': 3, 'pref_b_d_f': 4}
    d = unravel_dict(key, value)
    assert d == answer, 'unravel_dict failed to produce correct output'
