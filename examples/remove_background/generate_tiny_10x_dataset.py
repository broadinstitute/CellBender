#!/usr/bin/env python

import sys
import os
import numpy as np
from cellbender.remove_background.downstream import load_anndata_from_input

dataset_name = "heart10k (CellRanger 3.0.0, v3 Chemistry)"
dataset_url = "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/heart_10k_v3/heart_10k_v3_raw_feature_bc_matrix.h5"
dataset_local_filename = "heart10k_raw_feature_bc_matrix.h5"
expected_cells = 7000
start_of_empties = 10000
num_cell_barcodes_to_keep = 500
num_empty_barcodes_to_keep = 50_000
low_count_threshold = 30
num_genes_to_keep = 100
random_seed = 1984
rng = np.random.RandomState(random_seed)


if not os.path.exists(dataset_local_filename):

    print(f"Downloading {dataset_name}...")
    try:
        os.system(f'wget -O {dataset_local_filename} {dataset_url}')

    except Exception:
        print(f"10x Genomics website is preventing an automatic download. Please visit "
              f"https://www.10xgenomics.com/resources/datasets/10-k-heart-cells-from-an-e-18-mouse-v-3-chemistry-3-standard-3-0-0 "
              f"in a browser and manually download 'Gene / cell matrix HDF5 (raw)' into "
              f"this folder and re-run this script.")
        sys.exit()

print("Loading the dataset...")
adata = load_anndata_from_input(dataset_local_filename)
num_raw_genes = adata.shape[1]
print(f"Raw dataset size {adata.shape}")

print(f"Trimming {dataset_name}...")

# select 'num_genes_to_keep' highly expressed genes
total_gene_expression = np.array(adata.X.sum(axis=0)).squeeze()
genes_to_keep_indices = np.argsort(total_gene_expression)[::-1][:num_genes_to_keep]

# slice dataset on genes
adata = adata[:, genes_to_keep_indices].copy()

# find cells and empties
umi_per_barcode = np.array(adata.X.sum(axis=1)).squeeze()
umi_sorted_barcode_indices = np.argsort(umi_per_barcode)[::-1]
cell_indices = umi_sorted_barcode_indices[:expected_cells]
last_barcode = (umi_per_barcode > low_count_threshold).sum()
empty_indices = umi_sorted_barcode_indices[start_of_empties:last_barcode]

# putative list of barcodes to keep
cell_barcodes_to_keep_indices = np.asarray(cell_indices)[
    rng.permutation(len(cell_indices))[:num_cell_barcodes_to_keep]].tolist()
empty_barcodes_to_keep_indices = np.asarray(empty_indices)[
    rng.permutation(len(empty_indices))[:num_empty_barcodes_to_keep]].tolist()
barcodes_to_keep_indices = cell_barcodes_to_keep_indices + empty_barcodes_to_keep_indices

# slice dataset on barcodes
adata = adata[barcodes_to_keep_indices].copy()

# compensate for lost counts (due to the reduced number of genes)
adata.X = adata.X * int((num_raw_genes / num_genes_to_keep) ** 0.25)

print(f"Number of genes in the trimmed dataset: {len(genes_to_keep_indices)}")
print(f"Number of barcodes in the trimmed dataset: {len(barcodes_to_keep_indices)}")
print(f"Expected number of cells in the trimmed dataset: {num_cell_barcodes_to_keep}")

print(adata)

# save
output_file = 'tiny_raw_feature_bc_matrix.h5ad'
print(f"Saving the trimmed dataset as {output_file} ...")
adata.write(output_file)

print("Done!")
