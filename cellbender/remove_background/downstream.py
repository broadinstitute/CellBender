"""Functions for downstream work with outputs of remove-background."""

from cellbender.remove_background.data.io import load_data

import tables
import numpy as np
import scipy.sparse as sp
import anndata
from typing import Dict, Optional


def dict_from_h5(file: str) -> Dict[str, np.ndarray]:
    """Read in everything from an h5 file and put into a dictionary.

    Args:
        file: The h5 file

    Returns:
        Dictionary containing all the information from the h5 file
    """
    d = {}
    with tables.open_file(file) as f:
        # read in everything
        for array in f.walk_nodes("/", "Array"):
            d[array.name] = array.read()
    return d


def anndata_from_h5(file: str,
                    analyzed_barcodes_only: bool = True) -> anndata.AnnData:
    """Load an output h5 file into an AnnData object for downstream work.

    Args:
        file: The h5 file
        analyzed_barcodes_only: False to load all barcodes, so that the size of
            the AnnData object will match the size of the input raw count matrix.
            True to load a limited set of barcodes: only those analyzed by the
            algorithm. This allows relevant latent variables to be loaded
            properly into adata.obs and adata.obsm, rather than adata.uns.

    Returns:
        anndata.AnnData: The anndata object, populated with inferred latent variables
            and metadata.

    """

    d = dict_from_h5(file)
    X = sp.csc_matrix((d.pop('data'), d.pop('indices'), d.pop('indptr')),
                      shape=d.pop('shape')).transpose().tocsr()

    # check and see if we have barcode index annotations, and if the file is filtered
    barcode_key = [k for k in d.keys() if (('barcode' in k) and ('ind' in k))]
    if len(barcode_key) > 0:
        max_barcode_ind = d[barcode_key[0]].max()
        filtered_file = (max_barcode_ind >= X.shape[0])
    else:
        filtered_file = True

    if analyzed_barcodes_only:
        if filtered_file:
            # filtered file being read, so we don't need to subset
            print('Assuming we are loading a "filtered" file that contains only cells.')
            pass
        elif 'barcode_indices_for_latents' in d.keys():
            X = X[d['barcode_indices_for_latents'], :]
            d['barcodes'] = d['barcodes'][d['barcode_indices_for_latents']]
        elif 'barcodes_analyzed_inds' in d.keys():
            X = X[d['barcodes_analyzed_inds'], :]
            d['barcodes'] = d['barcodes'][d['barcodes_analyzed_inds']]
        else:
            print('Warning: analyzed_barcodes_only=True, but the key '
                  '"barcodes_analyzed_inds" or "barcode_indices_for_latents" '
                  'is missing from the h5 file. '
                  'Will output all barcodes, and proceed as if '
                  'analyzed_barcodes_only=False')

    # Construct the anndata object.
    adata = anndata.AnnData(X=X,
                            obs={'barcode': d.pop('barcodes').astype(str)},
                            var={'gene_name': (d.pop('gene_names') if 'gene_names' in d.keys()
                                               else d.pop('name')).astype(str)},
                            dtype=X.dtype)
    adata.obs.set_index('barcode', inplace=True)
    adata.var.set_index('gene_name', inplace=True)

    # For CellRanger v2 legacy format, "gene_ids" was called "genes"... rename this
    if 'genes' in d.keys():
        d['id'] = d.pop('genes')

    # For purely aesthetic purposes, rename "id" to "gene_id"
    if 'id' in d.keys():
        d['gene_id'] = d.pop('id')

    # If genomes are empty, try to guess them based on gene_id
    if 'genome' in d.keys():
        if np.array([s.decode() == '' for s in d['genome']]).all():
            if '_' in d['gene_id'][0].decode():
                print('Genome field blank, so attempting to guess genomes based on gene_id prefixes')
                d['genome'] = np.array([s.decode().split('_')[0] for s in d['gene_id']], dtype=str)

    # Add other information to the anndata object in the appropriate slot.
    _fill_adata_slots_automatically(adata, d)

    # Add a special additional field to .var if it exists.
    if 'features_analyzed_inds' in adata.uns.keys():
        adata.var['cellbender_analyzed'] = [True if (i in adata.uns['features_analyzed_inds'])
                                            else False for i in range(adata.shape[1])]
    elif 'features_analyzed_inds' in adata.var.keys():
        adata.var['cellbender_analyzed'] = [True if (i in adata.var['features_analyzed_inds'].values)
                                            else False for i in range(adata.shape[1])]

    if analyzed_barcodes_only:
        for col in adata.obs.columns[adata.obs.columns.str.startswith('barcodes_analyzed')
                                     | adata.obs.columns.str.startswith('barcode_indices')]:
            try:
                del adata.obs[col]
            except Exception:
                pass
    else:
        # Add a special additional field to .obs if all barcodes are included.
        if 'barcodes_analyzed_inds' in adata.uns.keys():
            adata.obs['cellbender_analyzed'] = [True if (i in adata.uns['barcodes_analyzed_inds'])
                                                else False for i in range(adata.shape[0])]
        elif 'barcodes_analyzed_inds' in adata.obs.keys():
            adata.obs['cellbender_analyzed'] = [True if (i in adata.obs['barcodes_analyzed_inds'].values)
                                                else False for i in range(adata.shape[0])]

    return adata


def _fill_adata_slots_automatically(adata, d):
    """Add other information to the adata object in the appropriate slot."""

    # TODO: what about "features_analyzed_inds"?  If not all features are analyzed, does this work?

    for key, value in d.items():
        try:
            if value is None:
                continue
            value = np.asarray(value)
            if len(value.shape) == 0:
                adata.uns[key] = value
            elif value.shape[0] == adata.shape[0]:
                if (len(value.shape) < 2) or (value.shape[1] < 2):
                    adata.obs[key] = value
                else:
                    adata.obsm[key] = value
            elif value.shape[0] == adata.shape[1]:
                if value.dtype.name.startswith('bytes'):
                    adata.var[key] = value.astype(str)
                else:
                    adata.var[key] = value
            else:
                adata.uns[key] = value
        except Exception:
            print('Unable to load data into AnnData: ', key, value, type(value))


def load_anndata_from_input(input_file: str) -> anndata.AnnData:
    """Load an input file into an AnnData object (used in report generation).
    Equivalent to something like scanpy.read(), but uses cellbender's io.

    Args:
        input_file: The raw data file

    Returns:
        adata.AnnData: The anndata object

    """

    # Load data as dict.
    d = load_data(input_file=input_file)

    # For purely aesthetic purposes, rename slots from the plural to singluar.
    for key in ['gene_id', 'barcode', 'genome', 'feature_type', 'gene_name']:
        if key + 's' in d.keys():
            d[key] = d.pop(key + 's')

    # Create anndata object from dict.
    adata = anndata.AnnData(X=d.pop('matrix'),
                            obs={'barcode': d.pop('barcode').astype(str)},
                            var={'gene_name': d.pop('gene_name').astype(str)},
                            dtype=int)
    adata.obs.set_index('barcode', inplace=True)
    adata.var.set_index('gene_name', inplace=True)

    # Add other information to the anndata object in the appropriate slot.
    _fill_adata_slots_automatically(adata, d)

    return adata


def load_anndata_from_input_and_output(input_file: str,
                                       output_file: str,
                                       analyzed_barcodes_only: bool = True,
                                       input_layer_key: str = 'cellranger',
                                       retain_input_metadata: bool = False,
                                       gene_expression_encoding_key: str = 'cellbender_embedding',
                                       truth_file: Optional[str] = None) -> anndata.AnnData:
    """Load remove-background output count matrix into an anndata object,
    together with remove-background metadata and the raw input counts.

    Args:
        input_file: Raw h5 file (or other compatible remove-background input)
            used as input for remove-background.
        output_file: Output h5 file created by remove-background (can be
            filtered or not).
        analyzed_barcodes_only: Argument passed to anndata_from_h5().
            False to load all barcodes, so that the size of
            the AnnData object will match the size of the input raw count matrix.
            True to load a limited set of barcodes: only those analyzed by the
            algorithm. This allows relevant latent variables to be loaded
            properly into adata.obs and adata.obsm, rather than adata.uns.
        input_layer_key: Key of the anndata.layer that is created for the raw
            input count matrix.
        retain_input_metadata: In addition to loading the CellBender metadata,
            which happens automatically, set this to True to retain all the
            metadata from the raw input file as well.
        gene_expression_encoding_key: The CellBender gene expression embedding
            will be loaded into adata.obsm[gene_expression_encoding_key]
        truth_file: File containing truth data if this is a simulation

    Return:
        anndata.AnnData: AnnData object with counts before and after remove-background,
            as well as inferred latent variables from remove-background.

    """

    # Load input data.
    adata_raw = load_anndata_from_input(input_file=input_file)

    # Load remove-background output data.
    adata_out = anndata_from_h5(output_file, analyzed_barcodes_only=analyzed_barcodes_only)

    # Subset the raw dataset to the relevant barcodes.
    adata_raw = adata_raw[adata_out.obs.index]

    # TODO: keep the stuff from the raw file too: from obs and var and uns
    # TODO: maybe use _fill_adata_slots_automatically()?  or just copy stuff

    # Put count matrices into 'layers' in anndata for clarity.
    adata_out.layers[input_layer_key] = adata_raw.X.copy()
    adata_out.layers['cellbender'] = adata_out.X.copy()

    # Pre-compute a bit of metadata.
    adata_out.var['n_' + input_layer_key] = \
        np.array(adata_out.layers[input_layer_key].sum(axis=0), dtype=int).squeeze()
    adata_out.var['n_cellbender'] = \
        np.array(adata_out.layers['cellbender'].sum(axis=0), dtype=int).squeeze()
    adata_out.obs['n_' + input_layer_key] = \
        np.array(adata_out.layers[input_layer_key].sum(axis=1), dtype=int).squeeze()
    adata_out.obs['n_cellbender'] = \
        np.array(adata_out.layers['cellbender'].sum(axis=1), dtype=int).squeeze()

    # Load truth data, if present.
    if truth_file is not None:
        adata_truth = anndata_from_h5(truth_file, analyzed_barcodes_only=False)
        adata_truth = adata_truth[adata_out.obs.index]
        adata_out.layers['truth'] = adata_truth.X.copy()
        adata_out.var['n_truth'] = np.array(adata_out.layers['truth'].sum(axis=0), dtype=int).squeeze()
        adata_out.obs['n_truth'] = np.array(adata_out.layers['truth'].sum(axis=1), dtype=int).squeeze()
        for key in adata_truth.obs.keys():
            if key.startswith('truth_'):
                adata_out.obs[key] = adata_truth.obs[key].copy()
        for key in adata_truth.uns.keys():
            if key.startswith('truth_'):
                adata_out.uns[key] = adata_truth.uns[key].copy()
        for key in adata_truth.var.keys():
            if key.startswith('truth_'):
                adata_out.var[key] = adata_truth.var[key].copy()

    # Rename the CellBender encoding of gene expression.
    if analyzed_barcodes_only:
        slot = adata_out.obsm
    else:
        slot = adata_out.uns
    embedding_key = None
    for key in ['gene_expression_encoding', 'latent_gene_encoding']:
        if key in slot.keys():
            embedding_key = key
            break
    if gene_expression_encoding_key != embedding_key:
        slot[gene_expression_encoding_key] = slot[embedding_key].copy()
        del slot[embedding_key]

    return adata_out


def _load_anndata_from_input_and_decontx(input_file: str,
                                         output: str,
                                         input_layer_key: str = 'cellranger',
                                         truth_file: Optional[str] = None) -> anndata.AnnData:
    """Load decontX output count matrix into an anndata object,
    together with remove-background metadata and the raw input counts.

    NOTE: this is used only for dev purposes and only in the report

    Args:
        input_file: Raw h5 file (or other compatible remove-background input)
            used as input for remove-background.
        output: Output h5 file, or a directory where decontX MTX and TSV files
            are stored
        input_layer_key: Key of the anndata.layer that is created for the raw
            input count matrix.
        truth_file: File containing truth data if this is a simulation

    Return:
        anndata.AnnData: AnnData object with counts before and after remove-background,
            as well as inferred latent variables from remove-background.

    """

    # Load decontX output data.
    print('UNSTABLE FEATURE: Trying to load decontX format MTX output')
    adata_out = load_anndata_from_input(input_file=output)
    adata_out.var_names_make_unique()

    # Load input data.
    adata_raw = load_anndata_from_input(input_file=input_file)
    adata_raw.var_names_make_unique()

    adata_raw = adata_raw[:, [g in adata_out.var.index for g in adata_raw.var.index]].copy()
    adata_out.var['genome'] = adata_raw.var['genome'].copy()
    adata_out.var['feature_type'] = adata_raw.var['feature_type'].copy()
    adata_out.var['gene_id'] = adata_raw.var['gene_id'].copy()

    # Subset the raw dataset to the relevant barcodes.
    empty_logic = np.array([b not in adata_out.obs.index for b in adata_raw.obs.index])
    empty_counts = np.array(adata_raw.X[empty_logic].sum(axis=1)).squeeze()
    approx_ambient = np.array(adata_raw.X[empty_logic][empty_counts > 5].sum(axis=0)).squeeze()
    approx_ambient = approx_ambient / (approx_ambient.sum() + 1e-10)
    print(f'Estimated that there are about {np.median(empty_counts[empty_counts > 5])} counts in empties')
    adata_raw = adata_raw[adata_out.obs.index].copy()
    adata_out.uns['empty_droplet_size_lognormal_loc'] = np.log(np.median(empty_counts[empty_counts > 5]))

    # Put count matrices into 'layers' in anndata for clarity.
    adata_out.layers[input_layer_key] = adata_raw.X.copy()
    adata_out.layers['decontx'] = adata_out.X.copy()

    # Pre-compute a bit of metadata.
    adata_out.var['n_' + input_layer_key] = np.array(adata_out.layers[input_layer_key].sum(axis=0)).squeeze()
    adata_out.var['n_decontx'] = np.array(adata_out.layers['decontx'].sum(axis=0)).squeeze()
    adata_out.obs['n_' + input_layer_key] = np.array(adata_out.layers[input_layer_key].sum(axis=1)).squeeze()
    adata_out.obs['n_decontx'] = np.array(adata_out.layers['decontx'].sum(axis=1)).squeeze()
    adata_out.obs['cell_probability'] = 1.  # because decontx data contains only cells
    adata_out.uns['target_false_positive_rate'] = 0.01  # TODO: placeholder
    adata_out.uns['approximate_ambient_profile'] = approx_ambient
    adata_out.var['ambient_expression'] = np.nan

    # Load truth data, if present.
    if truth_file is not None:
        adata_truth = anndata_from_h5(truth_file, analyzed_barcodes_only=False)
        adata_truth = adata_truth[adata_out.obs.index]

        # TODO; a check
        adata_truth = adata_truth[:, [g in adata_out.var.index for g in adata_truth.var.index]].copy()

        adata_out.layers['truth'] = adata_truth.X.copy()
        adata_out.var['n_truth'] = np.array(adata_out.layers['truth'].sum(axis=0)).squeeze()
        adata_out.obs['n_truth'] = np.array(adata_out.layers['truth'].sum(axis=1)).squeeze()
        for key in adata_truth.obs.keys():
            if key.startswith('truth_'):
                adata_out.obs[key] = adata_truth.obs[key].copy()
        for key in adata_truth.uns.keys():
            if key.startswith('truth_'):
                adata_out.uns[key] = adata_truth.uns[key].copy()
        for key in adata_truth.var.keys():
            if key.startswith('truth_'):
                adata_out.var[key] = adata_truth.var[key].copy()

    return adata_out


def load_anndata_from_input_and_outputs(input_file: str,
                                        output_files: Dict[str, str],
                                        analyzed_barcodes_only: bool = True,
                                        input_layer_key: str = 'cellranger',
                                        gene_expression_encoding_key: str = 'cellbender_embedding',
                                        truth_file: Optional[str] = None) -> anndata.AnnData:
    """Load remove-background output count matrices into an anndata object,
    together with remove-background metadata and the raw input counts.

    The use case would typically be cellbender runs with multiple output files
    at different FPRs, which we want to compare.

    Args:
        input_file: Raw h5 file (or other compatible remove-background input)
            used as input for remove-background.
        output_files: Output h5 files created by remove-background (can be
            filtered or not) or some other method.  Dict whose keys are layer keys
            and whose values are file names.
        analyzed_barcodes_only: Argument passed to anndata_from_h5().
            False to load all barcodes, so that the size of
            the AnnData object will match the size of the input raw count matrix.
            True to load a limited set of barcodes: only those analyzed by the
            algorithm. This allows relevant latent variables to be loaded
            properly into adata.obs and adata.obsm, rather than adata.uns.
        input_layer_key: Key of the anndata.layer that is created for the raw
            input count matrix.
        gene_expression_encoding_key: The CellBender gene expression embedding
            will be loaded into adata.obsm[gene_expression_encoding_key]
        truth_file: File containing truth data if this is a simulation

    Return:
        anndata.AnnData: AnnData object with counts before and after remove-background,
            as well as inferred latent variables from remove-background.

    """

    # Load input data.
    adata_raw = load_anndata_from_input(input_file=input_file)
    adata_raw.var_names_make_unique()

    # Load remove-background output data.
    assert type(output_files) == dict, 'output_files must be a dict whose keys are ' \
                                       'layer names and whose values are file paths.'
    outs = {}
    for key, output_file in output_files.items():
        outs[key] = anndata_from_h5(output_file, analyzed_barcodes_only=analyzed_barcodes_only)
        outs[key].var_names_make_unique()

    # Subset all datasets to the relevant barcodes and features.
    relevant_barcodes = set(adata_raw.obs_names)
    relevant_features = set(adata_raw.var_names)
    for key, ad in outs.items():
        relevant_barcodes = relevant_barcodes.intersection(set(ad.obs_names))
        relevant_features = relevant_features.intersection(set(ad.var_names))
    if len(relevant_barcodes) < len(adata_raw):
        print(f'Warning: subsetting to barcodes common to all datasets: there '
              f'are {len(relevant_barcodes)}')
    if len(relevant_features) < adata_raw.shape[1]:
        print(f'Warning: subsetting to features common to all datasets: there '
              f'are {len(relevant_features)}')
    adata_raw = adata_raw[list(relevant_barcodes)].copy()
    adata_raw = adata_raw[:, list(relevant_features)].copy()
    for i, (key, ad) in enumerate(outs.items()):
        outs[key] = ad[list(relevant_barcodes)].copy()
        outs[key] = outs[key][:, list(relevant_features)].copy()
        if i == 0:
            print(f'Loading latent variables from one output file: {key}')
            adata_out = outs[key].copy()

    # Put count matrices into 'layers' in anndata for clarity.
    adata_out.layers[input_layer_key] = adata_raw.X.copy()
    for key, ad in outs.items():
        adata_out.layers[key] = ad.X.copy()

    # Load truth data, if present.
    if truth_file is not None:
        adata_truth = anndata_from_h5(truth_file, analyzed_barcodes_only=False)
        adata_truth = adata_truth[adata_out.obs.index]
        adata_out.layers['truth'] = adata_truth.X.copy()
        adata_out.var['n_truth'] = np.array(adata_out.layers['truth'].sum(axis=0)).squeeze()
        adata_out.obs['n_truth'] = np.array(adata_out.layers['truth'].sum(axis=1)).squeeze()
        for key in adata_truth.obs.keys():
            if key.startswith('truth_'):
                adata_out.obs[key] = adata_truth.obs[key].copy()
        for key in adata_truth.uns.keys():
            if key.startswith('truth_'):
                adata_out.uns[key] = adata_truth.uns[key].copy()
        for key in adata_truth.var.keys():
            if key.startswith('truth_'):
                adata_out.var[key] = adata_truth.var[key].copy()

    # Rename the CellBender encoding of gene expression.
    if analyzed_barcodes_only:
        slot = adata_out.obsm
    else:
        slot = adata_out.uns
    embedding_key = None
    for key in ['gene_expression_encoding', 'latent_gene_encoding']:
        if key in slot.keys():
            embedding_key = key
            break
    if gene_expression_encoding_key != embedding_key:
        slot[gene_expression_encoding_key] = slot[embedding_key].copy()
        del slot[embedding_key]

    return adata_out
