#!/bin/bash

# Example run:
# $ ./run_usual_benchmarks.sh "v0.2.1"

GIT_HASH=$1

# from the paper ==========================

# 10x Genomics pbmc8k
python run_benchmark.py \
    --git $GIT_HASH \
    --input "gs://broad-dsde-methods-sfleming/cellbender_test/pbmc8k_raw_gene_bc_matrices.h5" \
    --sample "pbmc8k"

# 10x Genomics hgmm12k
python run_benchmark.py \
    --git $GIT_HASH \
    --input "gs://broad-dsde-methods-sfleming/cellbender_test/hgmm_12k_raw_gene_bc_matrices.h5" \
    --sample "hgmm12k"

# 10x Genomics pbmc5k CITE-seq
python run_benchmark.py \
    --git $GIT_HASH \
    --input "gs://broad-dsde-methods-sfleming/cellbender_test/5k_pbmc_protein_v3_nextgem_raw_feature_bc_matrix.h5" \
    --sample "pbmc5k" \
    --fpr "0.1"

# Broad PCL rat6k
python run_benchmark.py \
    --git $GIT_HASH \
    --input "gs://broad-dsde-methods-sfleming/cellbender_test/PCL_rat_A_LA6_raw_feature_bc_matrix.h5" \
    --sample "rat6k"

# Simulation s1
python run_benchmark.py \
    --git $GIT_HASH \
    --input "gs://broad-dsde-methods-sfleming/cellbender_test/s1.h5" \
    --truth "gs://broad-dsde-methods-sfleming/cellbender_test/s1_truth.h5" \
    --sample "s1"

# Simulation s4
python run_benchmark.py \
    --git $GIT_HASH \
    --input "gs://broad-dsde-methods-sfleming/cellbender_test/s4.h5" \
    --truth "gs://broad-dsde-methods-sfleming/cellbender_test/s4_truth.h5" \
    --sample "s4" \
    --fpr "0.2"

# Simulation s7, hgmm
python run_benchmark.py \
    --git $GIT_HASH \
    --input "gs://broad-dsde-methods-sfleming/cellbender_test/s7.h5" \
    --truth "gs://broad-dsde-methods-sfleming/cellbender_test/s7_truth.h5" \
    --sample "s7"

# additional datasets ========================

# Broad PCL human PV dataset, hard time calling high-count cells in v0.2.0, wobbly learning curve
python run_benchmark.py \
    --git $GIT_HASH \
    --input "gs://broad-dsde-methods-sfleming/cellbender_test/PCL_human_PV_1817_ls.h5" \
    --sample "pv20k"

python run_benchmark.py \
    --git $GIT_HASH \
    --input "gs://broad-dsde-methods-sfleming/cellbender_test/PCL_human_PV_1799_BPV.h5" \
    --sample "pv15k"

# public data from Caroline Porter, high UMI count cells called empty
python run_benchmark.py \
    --git $GIT_HASH \
    --input "gs://broad-dsde-methods-sfleming/cellbender_test/cporter_20230331_N18_Epi_A.h5" \
    --sample "epi2k"
