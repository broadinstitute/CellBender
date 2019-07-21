#!/usr/bin/env python

import urllib.request
import sys
import tarfile
import os
import numpy as np
from scipy.io import mmread, mmwrite
import pandas as pd
import operator
import shutil

dataset_name = "pbmc4k (CellRanger 2.1.0, v2 Chemistry)"
dataset_url = "http://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc4k/pbmc4k_raw_gene_bc_matrices.tar.gz"
dataset_local_filename = "pbmc4k_raw_gene_bc_matrices.tar.gz"
expected_cells = 4000
num_cell_barcodes_to_keep = 500
num_empty_barcodes_to_keep = 50_000
num_genes_to_keep = 100
random_seed = 1984
rng = np.random.RandomState(random_seed)

print(f"Downloading {dataset_name}...")
try:
    urllib.request.urlretrieve(dataset_url, dataset_local_filename)
except IOError:
    print(f"Could not retrieve {dataset_name} -- cannot continue!")
    sys.exit()

print(f"Extracting {dataset_name}...")
tar = tarfile.open(dataset_local_filename, "r:gz")
tar.extractall()
tar.close()

extracted_dataset_local_path = os.path.join(os.curdir, "raw_gene_bc_matrices")
matrix_mtx_path = os.path.join(extracted_dataset_local_path, "GRCh38", "matrix.mtx")
genes_tsv_path = os.path.join(extracted_dataset_local_path, "GRCh38", "genes.tsv")
barcodes_tsv_path = os.path.join(extracted_dataset_local_path, "GRCh38", "barcodes.tsv")

print("Loading the gene expression matrix...")
matrix_mtx = mmread(matrix_mtx_path)
num_raw_genes = matrix_mtx.shape[0]
num_raw_barcodes = matrix_mtx.shape[1]

print("Loading genes and barcodes...")
genes_df = pd.read_csv(genes_tsv_path, delimiter="\t", header=None)
barcodes_df = pd.read_csv(barcodes_tsv_path, delimiter="\t", header=None)

print(f"Trimming {dataset_name}...")
# (naive) indices of expected cells
umi_per_barcode = np.asarray(np.sum(matrix_mtx, 0)).flatten()
total_gene_expression = np.asarray(np.sum(matrix_mtx, 1)).flatten()
umi_sorted_barcode_indices = list(
    map(operator.itemgetter(0),
        sorted(enumerate(umi_per_barcode), key=operator.itemgetter(1), reverse=True)))
cell_indices = umi_sorted_barcode_indices[:expected_cells]
empty_indices = umi_sorted_barcode_indices[expected_cells:]

# (naive) filter counts to non-empty droplets
cell_counts_csr = matrix_mtx.tocsc()[:, cell_indices].tocsr()

# select 'num_genes_to_keep' highly expressed genes
genes_to_keep_indices = sorted(range(num_raw_genes),
    key=lambda x: total_gene_expression[x], reverse=True)[:num_genes_to_keep]

# putative list of barcodes to keep
cell_barcodes_to_keep_indices = np.asarray(cell_indices)[
    rng.permutation(len(cell_indices))[:num_cell_barcodes_to_keep]].tolist()
empty_barcodes_to_keep_indices = np.asarray(empty_indices)[
    rng.permutation(len(empty_indices))[:num_empty_barcodes_to_keep]].tolist()
barcodes_to_keep_indices = cell_barcodes_to_keep_indices + empty_barcodes_to_keep_indices

# remove barcodes with zero expression on kept genes
trimmed_counts_matrix = matrix_mtx.tocsr()[genes_to_keep_indices, :].tocsc()[:, barcodes_to_keep_indices]
umi_per_putatively_kept_barcodes = np.asarray(np.sum(trimmed_counts_matrix, 0)).flatten()
barcodes_to_keep_indices = np.asarray(barcodes_to_keep_indices)[umi_per_putatively_kept_barcodes > 0]
barcodes_to_keep_indices = barcodes_to_keep_indices.tolist()

# slice the raw dataset
tiny_matrix_mtx = matrix_mtx.tocsr()[genes_to_keep_indices, :].tocsc()[:, barcodes_to_keep_indices].tocoo()
tiny_genes_df = genes_df.loc[genes_to_keep_indices]
tiny_barcodes_df = barcodes_df.loc[barcodes_to_keep_indices]

print(f"Number of genes in the trimmed dataset: {len(genes_to_keep_indices)}")
print(f"Number of barcodes in the trimmed dataset: {len(barcodes_to_keep_indices)}")
print(f"Expected number of cells in the trimmed dataset: {num_cell_barcodes_to_keep}")

# save
print("Saving the trimmed dataset...")
output_path = os.path.join(os.curdir, 'tiny_raw_gene_bc_matrices', 'GRCh38')
os.makedirs(output_path, exist_ok=True)
mmwrite(os.path.join(output_path, "matrix.mtx"), tiny_matrix_mtx)
tiny_genes_df.to_csv(os.path.join(output_path, "genes.tsv"), sep='\t', header=None, index=False)
tiny_barcodes_df.to_csv(os.path.join(output_path, "barcodes.tsv"), sep='\t', header=None, index=False)

print("Cleaning up...")
shutil.rmtree(extracted_dataset_local_path, ignore_errors=True)
os.remove(dataset_local_filename)

print("Done!")
