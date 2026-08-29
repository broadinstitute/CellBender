"""Test input reading and output writing functionality."""

import gzip
import shutil
from typing import Dict, Generator, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp
import tables
from conftest import sparse_matrix_equal, string_ndarray_equality
from scipy.io import mmwrite

from cellbender.remove_background.data.io import (
    detect_cellranger_version_h5,
    detect_cellranger_version_mtx,
    get_matrix_from_anndata,
    get_matrix_from_cellranger_mtx,
    get_matrix_from_dropseq_dge,
    load_data,
    unravel_dict,
)


def assert_loaded_matches_saved(
    d: Dict[str, np.ndarray], loaded: Dict[str, np.ndarray], keys: List[str], cellranger_version: Optional[int] = None
):
    """Check if a loaded file's data matches the data that was saved.

    Args:
        d: Dict of the data that was saved
        loaded: Dict of the data that was loaded from file
        keys: List of keys to the dicts that will be checked for equality
        cellranger_version: In [2, 3]
    """
    if "cellranger_version" in loaded.keys():
        assert cellranger_version == loaded["cellranger_version"]
    assert sparse_matrix_equal(loaded["matrix"], d["matrix"])
    for key in keys:
        if d[key] is None:
            continue
        assert loaded[key] is not None, f'Loaded h5 key "{key}" was None, but data was saved: {d[key][:5]} ...'
        assert string_ndarray_equality(d[key], loaded[key]), f'Loaded h5 key "{key}" did not match saved data'


@pytest.mark.parametrize("filetype", ["h5_v2_file", "h5_v2_file_missing_ids", "h5_v3_file"])
def test_simulate_save_load_h5(simulated_dataset, filetype, h5_v2_file, h5_v2_file_missing_ids, h5_v3_file):

    # get information from fixture, since you cannot pass fixtures to parametrize
    if filetype == "h5_v2_file":
        saved_h5 = h5_v2_file
    elif filetype == "h5_v2_file_missing_ids":
        saved_h5 = h5_v2_file_missing_ids
    elif filetype == "h5_v3_file":
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
    with open(file, "rb") as f_in, gzip.open(file + ".gz", "wb") as f_out:
        f_out.writelines(f_in)


def save_mtx(tmpdir_factory, simulated_dataset, version: int) -> str:
    """Save data files in MTX format and return the directory path"""
    dirname = tmpdir_factory.mktemp(f"mtx_v{version}")

    # barcodes and sparse matrix... seems the MTX matrix is transposed
    mmwrite(dirname.join("matrix.mtx"), simulated_dataset["matrix"].transpose())
    np.savetxt(dirname.join("barcodes.tsv"), simulated_dataset["barcodes"], fmt="%s")

    # features and gzipping if v3
    features = np.concatenate(
        (
            np.expand_dims(simulated_dataset["gene_ids"], axis=1),
            np.expand_dims(simulated_dataset["gene_names"], axis=1),
            np.expand_dims(simulated_dataset["feature_types"], axis=1),
        ),
        axis=1,
    )

    if version == 3:
        np.savetxt(dirname.join("features.tsv"), features, fmt="%s", delimiter="\t")
        gzip_file(dirname.join("matrix.mtx"))
        gzip_file(dirname.join("barcodes.tsv"))
        gzip_file(dirname.join("features.tsv"))
    elif version == 2:
        np.savetxt(dirname.join("genes.tsv"), features[:, :2], fmt="%s", delimiter="\t")
    else:
        raise ValueError(f"Test problem: version is {version}, but [2, 3] allowed")

    return dirname


@pytest.fixture(scope="session", params=[2, 3])
def mtx_directory(request, tmpdir_factory, simulated_dataset):
    dirname = save_mtx(tmpdir_factory=tmpdir_factory, simulated_dataset=simulated_dataset, version=request.param)
    yield dirname
    shutil.rmtree(str(dirname))


def test_detect_cellranger_version_mtx(mtx_directory):
    v = detect_cellranger_version_mtx(filedir=mtx_directory)
    true_version = 2 if ("_v2" in str(mtx_directory)) else 3
    assert v == true_version


def test_load_mtx(simulated_dataset, mtx_directory):

    # use the correct loader function
    loaded = get_matrix_from_cellranger_mtx(filedir=mtx_directory)
    version = 3 if ("_v3" in str(mtx_directory)) else 2
    assert_loaded_matches_saved(
        d=simulated_dataset,
        loaded=loaded,
        keys=(["gene_ids", "gene_names", "barcodes"] + (["feature_types"] if (version == 3) else [])),
        cellranger_version=version,
    )

    # use auto-loading, as it would be run
    loaded = load_data(mtx_directory)
    assert_loaded_matches_saved(
        d=simulated_dataset,
        loaded=loaded,
        keys=(["gene_ids", "gene_names", "barcodes"] + (["feature_types"] if (version == 3) else [])),
        cellranger_version=version,
    )


def save_dge(tmpdir_factory, simulated_dataset, do_gzip) -> Tuple:
    """Save data files in DGE format and return the file path"""
    sep = "\t"
    name = "dge.txt"
    if do_gzip:
        name = name + ".gz"
    tmp_dir = tmpdir_factory.mktemp("dge")
    filename = tmp_dir.join(name)
    load_fcn = gzip.open if do_gzip else open

    def row_generator(mat: sp.csc_matrix) -> Generator[List[str], None, None]:
        for i in range(mat.shape[1]):
            yield np.array(mat[:, i].todense()).squeeze().astype(int).astype(str).tolist()

    with load_fcn(filename, "wb") as f:
        f.write(b"# some kind of header!\n")
        f.write(sep.join(["GENE"] + simulated_dataset["barcodes"].astype(str).tolist()).encode() + b"\n")
        for g, vals in zip(simulated_dataset["gene_names"], row_generator(simulated_dataset["matrix"])):
            f.write(sep.join([g] + vals).encode() + b"\n")

    return filename, tmp_dir


@pytest.fixture(scope="session", params=[True, False], ids=lambda x: "gzipped" if x else "not")
def dge_file(request, tmpdir_factory, simulated_dataset):
    filename, tmp_dir = save_dge(
        tmpdir_factory=tmpdir_factory, simulated_dataset=simulated_dataset, do_gzip=request.param
    )
    yield filename
    shutil.rmtree(str(tmp_dir))


def test_load_dge(simulated_dataset, dge_file):

    # use the correct loader function
    loaded = get_matrix_from_dropseq_dge(str(dge_file))
    assert_loaded_matches_saved(d=simulated_dataset, loaded=loaded, keys=["gene_names", "barcodes"])

    # use auto-loading, as it would be run
    loaded = load_data(str(dge_file))
    assert_loaded_matches_saved(d=simulated_dataset, loaded=loaded, keys=["gene_names", "barcodes"])


@pytest.mark.skip
def test_load_bd():
    pass


@pytest.fixture(scope="session")
def h5ad_file(tmpdir_factory, simulated_dataset):
    """Write simulated_dataset as an h5ad file and return its path."""
    tmp_dir = tmpdir_factory.mktemp("h5ad")
    filename = str(tmp_dir.join("tmp.h5ad"))
    d = simulated_dataset
    adata = anndata.AnnData(
        X=sp.csr_matrix(d["matrix"]),
        obs=pd.DataFrame(index=d["barcodes"]),
        var=pd.DataFrame(
            {"gene_ids": d["gene_ids"], "feature_types": d["feature_types"]},
            index=d["gene_names"],
        ),
    )
    adata.write_h5ad(filename)
    yield filename


def test_load_anndata(simulated_dataset, h5ad_file):
    d = simulated_dataset

    # Direct loader.
    loaded = get_matrix_from_anndata(h5ad_file)
    assert_loaded_matches_saved(
        d=d,
        loaded=loaded,
        keys=["barcodes", "gene_names", "gene_ids", "feature_types"],
    )

    # Auto-dispatch via load_data (exercises choose_data_loader for .h5ad extension).
    loaded = load_data(h5ad_file)
    assert_loaded_matches_saved(
        d=d,
        loaded=loaded,
        keys=["barcodes", "gene_names", "gene_ids", "feature_types"],
    )


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
    key, value = "pref", {"a": 1, "b": {"c": 2, "d": {"e": 3, "f": 4}}}
    answer = {"pref_a": 1, "pref_b_c": 2, "pref_b_d_e": 3, "pref_b_d_f": 4}
    d = unravel_dict(key, value)
    assert d == answer, "unravel_dict failed to produce correct output"


@pytest.mark.parametrize("barcode_batch_size", [1, 3], ids=["batch_1", "batch_3"])
def test_stream_denoised_to_cellranger_h5(tmp_path_factory, barcode_batch_size):
    """Vectorized batch H5 write produces correct denoised output."""
    import scipy.sparse as sp

    from cellbender.remove_background.data.io import stream_denoised_to_cellranger_h5

    # Layout:
    #   6 absolute barcodes × 8 genes
    #   analyzed barcodes: 0-4 (barcode 5 is outside the analysis window)
    #   analyzed genes:    0-5 (genes 6-7 are non-analyzed)
    #   cells:             barcodes 1, 3, 4  (local ids 1, 3, 4)
    n_barcodes = 6
    n_genes = 8
    analyzed_barcode_inds = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    cell_logic = np.array([False, True, False, True, True])

    raw_data = np.array(
        [
            [0, 0, 0, 0, 0, 0, 5, 3],  # bc 0: non-cell (not analyzed)
            [2, 5, 0, 3, 1, 0, 0, 0],  # bc 1: cell
            [0, 0, 1, 0, 0, 0, 0, 0],  # bc 2: analyzed empty
            [4, 0, 2, 0, 8, 1, 1, 0],  # bc 3: cell (gene 6 non-analyzed raw=1)
            [1, 1, 1, 1, 1, 1, 0, 1],  # bc 4: cell (gene 7 non-analyzed raw=1)
            [3, 0, 0, 0, 0, 0, 0, 0],  # bc 5: not analyzed
        ],
        dtype=np.int32,
    )
    raw_csr = sp.csr_matrix(raw_data)

    # Noise parquet (absolute gene ids, absolute cell ids — same as local here since
    # analyzed_barcode_inds and analyzed_gene_inds start from 0):
    #   bc1: gene 0 → noise 2,  gene 1 → noise 3
    #   bc3: gene 0 → noise 1,  gene 4 → noise 10 (> raw 8, must clamp)
    #   bc4: gene 2 → noise 1
    noise_cell_ids = np.array([1, 1, 3, 3, 4], dtype=np.int32)
    noise_gene_ids = np.array([0, 1, 0, 4, 2], dtype=np.int32)
    noise_counts = np.array([2, 3, 1, 10, 1], dtype=np.int32)

    tmp_dir = tmp_path_factory.mktemp("stream_h5")
    noise_path = tmp_dir / "noise.parquet"
    output_path = str(tmp_dir / "out.h5")

    # Write gene-sorted noise parquet (as MCKP would produce).
    sort_order = np.lexsort((noise_cell_ids, noise_gene_ids))
    pq.write_table(
        pa.table(
            {
                "cell_id": pa.array(noise_cell_ids[sort_order], type=pa.int32()),
                "gene_id": pa.array(noise_gene_ids[sort_order], type=pa.int32()),
                "noise_count": pa.array(noise_counts[sort_order], type=pa.int32()),
            }
        ),
        noise_path,
    )

    gene_names = np.array([f"Gene{i}" for i in range(n_genes)])
    barcodes_arr = np.array([f"BC{i}" for i in range(n_barcodes)])

    denoised_per_bc = stream_denoised_to_cellranger_h5(
        noise_parquet_path=noise_path,
        raw_count_matrix=raw_csr,
        output_file=output_path,
        cell_logic=cell_logic,
        analyzed_barcode_inds=analyzed_barcode_inds,
        gene_names=gene_names,
        barcodes=barcodes_arr,
        barcode_batch_size=barcode_batch_size,
    )

    # Read back: H5 stores shape [n_genes, n_barcodes] CSC transposed.
    with tables.open_file(output_path, "r") as f:
        data = f.root.matrix.data[:]
        indices = f.root.matrix.indices[:]
        indptr = f.root.matrix.indptr[:]
        shape = f.root.matrix.shape[:]
    import scipy.sparse as sp2

    written = sp2.csc_matrix((data, indices, indptr), shape=tuple(shape)).T.toarray()

    # bc 0: not a cell → all zeros
    assert np.all(written[0] == 0), "non-cell barcode must be all zeros"

    # bc 1: cell, noise: gene0-=2, gene1-=3; raw=[2,5,0,3,1,0,0,0]
    # expected: [0, 2, 0, 3, 1, 0, 0, 0]
    assert written[1, 0] == 0  # 2 - 2 = 0
    assert written[1, 1] == 2  # 5 - 3 = 2
    assert written[1, 3] == 3  # no noise → keeps raw
    assert written[1, 6] == 0  # non-analyzed but raw is 0

    # bc 2: analyzed empty → all zeros
    assert np.all(written[2] == 0), "analyzed empty must be all zeros"

    # bc 3: cell, noise: gene0-=1 (3), gene4-=10 (clamped); non-analyzed gene6 raw=1 kept
    assert written[3, 0] == 3  # 4 - 1
    assert written[3, 4] == 0  # 8 - 10 clamped
    assert written[3, 6] == 1  # non-analyzed gene keeps raw for cells

    # bc 4: cell, noise: gene2-=1; non-analyzed gene7 raw=1 kept
    assert written[4, 2] == 0  # 1 - 1 = 0
    assert written[4, 7] == 1  # non-analyzed gene keeps raw for cells

    # bc 5: not analyzed → all zeros
    assert np.all(written[5] == 0), "unanalyzed barcode must be all zeros"

    # No negatives anywhere.
    assert np.all(written >= 0), "denoised counts must be non-negative"

    # denoised_per_barcode sums must match.
    expected = np.array(
        [
            0,  # bc 0
            0 + 2 + 0 + 3 + 1 + 0 + 0 + 0,  # bc 1
            0,  # bc 2
            3 + 0 + 2 + 0 + 0 + 1 + 1 + 0,  # bc 3 (non-analyzed gene6=1)
            1 + 1 + 0 + 1 + 1 + 1 + 0 + 1,  # bc 4 (non-analyzed gene7=1)
            0,  # bc 5
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(denoised_per_bc, expected)
