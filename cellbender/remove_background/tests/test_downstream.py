"""Test functions in downstream.py and a few more."""

import pytest

from cellbender.remove_background.downstream import anndata_from_h5, \
    load_anndata_from_input_and_output
from cellbender.remove_background.tests.conftest import SavedFileH5, \
    h5_file, h5_v3_file_post_inference


def convert(s):
    """The h5 gets saved with different key names than the h5_file class"""
    if s == 'gene_names':
        return 'gene_name'
    elif s == 'barcodes':
        return 'barcode'
    elif s == 'gene_ids':
        return 'gene_id'
    elif s == 'genomes':
        return 'genome'
    elif s == 'feature_types':
        return 'feature_type'
    return s


def test_anndata_from_h5(h5_file: SavedFileH5):

    # load AnnData and check its shape
    adata = anndata_from_h5(file=h5_file.name)
    assert h5_file.shape == adata.shape, 'Shape of loaded adata is not correct'

    # ensure everything that was supposed to be saved is loaded properly
    indices = [adata.obs.index.name, adata.var.index.name]
    all_columns = list(adata.obs.columns) + list(adata.var.columns) + indices
    for key in h5_file.keys:
        key = convert(key)
        assert key in all_columns, f'Saved information "{key}" is missing from loaded adata'


@pytest.mark.parametrize('analyzed_bcs_only', [True, False])
def test_anndata_from_inferred_h5(h5_v3_file_post_inference: SavedFileH5, analyzed_bcs_only):
    """Make sure the extra stuff we save gets loaded by downstream loader.
    TODO: tidy this up
    """

    adata = anndata_from_h5(file=h5_v3_file_post_inference.name, analyzed_barcodes_only=analyzed_bcs_only)
    print(adata)
    print(adata.obs.head())
    print(adata.var.head())

    # check the shape of the loaded AnnData
    if analyzed_bcs_only:
        expected_shape = (h5_v3_file_post_inference.barcodes_analyzed, h5_v3_file_post_inference.shape[1])
    else:
        expected_shape = h5_v3_file_post_inference.shape  # all barcodes
    assert adata.shape == expected_shape, 'Shape of loaded adata is not correct'

    # ensure everything that was supposed to be saved in .obs and .obsm and .var is loaded properly
    indices = [adata.obs.index.name, adata.var.index.name]
    all_columns = list(adata.obs.columns) + list(adata.obsm.keys()) + list(adata.var.columns) + indices

    relevant_keys = h5_v3_file_post_inference.keys
    if analyzed_bcs_only:
        relevant_keys = relevant_keys + h5_v3_file_post_inference.local_keys

    for key in relevant_keys:
        print(key)
        key = convert(key)
        assert key in all_columns, f'Saved information "{key}" is missing from loaded adata'

    # ensure other things are also loading
    extra_columns = list(adata.uns.keys())
    relevant_keys = h5_v3_file_post_inference.global_keys + h5_v3_file_post_inference.meta_keys
    if not analyzed_bcs_only:
        relevant_keys = relevant_keys + h5_v3_file_post_inference.local_keys

    for key in relevant_keys:
        print(key)
        assert key in extra_columns, \
            f'Saved .uns information "{key}" is missing from adata: {adata.uns.keys()}'


def test_load_anndata_from_input_and_output(h5_file, h5_v3_file_post_inference):
    adata = load_anndata_from_input_and_output(input_file=h5_file.name,
                                               output_file=h5_v3_file_post_inference.name,
                                               gene_expression_encoding_key='gene_expression_encoding',
                                               analyzed_barcodes_only=False)
    print(adata)
    assert h5_file.shape == adata.shape, \
        'Shape of loaded adata is not correct when loading all barcodes'
    for key in (h5_v3_file_post_inference.local_keys
                + h5_v3_file_post_inference.global_keys
                + h5_v3_file_post_inference.meta_keys):
        assert key in adata.uns.keys(), f'Key {key} missing from adata.uns'

    adata2 = load_anndata_from_input_and_output(input_file=h5_file.name,
                                                output_file=h5_v3_file_post_inference.name,
                                                gene_expression_encoding_key='gene_expression_encoding',
                                                analyzed_barcodes_only=True)
    print(adata2)
    assert h5_v3_file_post_inference.analyzed_shape == adata2.shape, \
        'Shape of loaded adata is not correct when loading only analyzed barcodes'
    for key in h5_v3_file_post_inference.local_keys:
        if key == 'gene_expression_encoding':
            assert key in adata2.obsm.keys(), f'Key {key} missing from adata.obsm'
        else:
            assert key in adata2.obs.keys(), f'Key {key} missing from adata.obs'
    for key in h5_v3_file_post_inference.global_keys + h5_v3_file_post_inference.meta_keys:
        assert key in adata2.uns.keys(), f'Key {key} missing from adata.uns'


@pytest.mark.skip(reason='TODO')
def test_scanpy_loading():
    pass


@pytest.mark.skip(reason='TODO')
def test_seurat_loading():
    pass
