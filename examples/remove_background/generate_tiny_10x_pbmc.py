#!/usr/bin/env python

import urllib.request
import sys
import os
import numpy as np
from cellbender.remove_background.downstream import load_anndata_from_input

dataset_name = "pbmc4k (CellRanger 2.1.0, v2 Chemistry)"
dataset_url = "https://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc4k/pbmc4k_raw_gene_bc_matrices_h5.h5"
dataset_local_filename = "pbmc4k_raw_gene_bc_matrices_h5.h5"
expected_cells = 4000
num_cell_barcodes_to_keep = 500
num_empty_barcodes_to_keep = 50_000
num_genes_to_keep = 100
random_seed = 1984
rng = np.random.RandomState(random_seed)

if not os.path.exists(dataset_local_filename):

    print(f"Downloading {dataset_name}...")
    try:
        urllib.request.urlretrieve(dataset_url, dataset_local_filename)

    except urllib.error.HTTPError:
        print(f"10x Genomics website is giving a 403 Forbidden error when an automatic "
              f"download was attempted. Please visit "
              f"https://www.10xgenomics.com/resources/datasets/4-k-pbm-cs-from-a-healthy-donor-2-standard-2-1-0 "
              f"in a browser and manually download 'Gene / cell matrix HDF5 (raw)' into "
              f"this folder and re-run this script.")
        sys.exit()

    except IOError:
        print(f"Could not retrieve {dataset_name} -- cannot continue!")
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
empty_indices = umi_sorted_barcode_indices[expected_cells:]

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
output_file = 'tiny_raw_gene_bc_matrices_h5.h5ad'
print(f"Saving the trimmed dataset as {output_file} ...")
adata.write(output_file)

print("Done!")
