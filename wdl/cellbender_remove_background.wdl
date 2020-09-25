## Copyright Broad Institute, 2020
##
## LICENSING :
## This script is released under the WDL source code license (BSD-3)
## (see LICENSE in https://github.com/openwdl/wdl).

task run_cellbender_remove_background_gpu {

  # File-related inputs
  String sample_name
  File input_10x_h5_file_or_mtx_directory

  # Outputs
  String output_directory  # Google bucket path

  # Docker image for cellbender remove-background version
  String? docker_image = "us.gcr.io/broad-dsde-methods/cellbender:latest"

  # Method configuration inputs
  Int? expected_cells
  Int? total_droplets_included
  String? model
  Int? low_count_threshold
  String? fpr  # in quotes: floats separated by whitespace: the output false positive rate(s)
  Int? epochs
  Int? z_dim
  String? z_layers  # in quotes: integers separated by whitespace
  Float? empty_drop_training_fraction
  String? blacklist_genes  # in quotes: integers separated by whitespace
  Float? learning_rate

  # Hardware-related inputs
  String? hardware_zones = "us-east1-d us-east1-c us-central1-a us-central1-c us-west1-b"
  Int? hardware_disk_size_GB = 50
  Int? hardware_boot_disk_size_GB = 20
  Int? hardware_preemptible_tries = 2
  Int? hardware_cpu_count = 4
  Int? hardware_memory_GB = 15

  command {
    cellbender remove-background \
      --input "${input_10x_h5_file_or_mtx_directory}" \
      --output "${sample_name}_out.h5" \
      --cuda \
      ${"--expected-cells " + expected_cells} \
      ${"--total-droplets-included " + total_droplets_included} \
      ${"--fpr " + fpr} \
      ${"--model " + model} \
      ${"--low-count-threshold " + low_count_threshold} \
      ${"--epochs " + epochs} \
      ${"--z-dim " + z_dim} \
      ${"--z-layers " + z_layers} \
      ${"--empty-drop-training-fraction " + empty_drop_training_fraction} \
      ${"--blacklist-genes " + blacklist_genes} \
      ${"--learning-rate " + learning_rate}

    gsutil -m cp ${sample_name}_out* ${output_directory}/${sample_name}/
  }

  output {
    File log = "${sample_name}_out.log"
    File pdf = "${sample_name}_out.pdf"
    File csv = "${sample_name}_out_cell_barcodes.csv"
    Array[File] h5_array = glob("${sample_name}_out*.h5")  # v2 creates a number of outputs depending on "fpr"
    String output_dir = "${output_directory}/${sample_name}"
  }

  runtime {
    docker: "${docker_image}"
    bootDiskSizeGb: hardware_boot_disk_size_GB
    disks: "local-disk ${hardware_disk_size_GB} HDD"
    memory: "${hardware_memory_GB}G"
    cpu: hardware_cpu_count
    zones: "${hardware_zones}"
    gpuCount: 1
    gpuType: "nvidia-tesla-k80"
    preemptible: hardware_preemptible_tries
    maxRetries: 0
  }

}

workflow cellbender_remove_background {

  call run_cellbender_remove_background_gpu

  output {
    File log = run_cellbender_remove_background_gpu.log
    File pdf = run_cellbender_remove_background_gpu.pdf
    File csv = run_cellbender_remove_background_gpu.csv
    Array[File] h5_array = run_cellbender_remove_background_gpu.h5_array
    String output_directory = run_cellbender_remove_background_gpu.output_dir
  }

}
