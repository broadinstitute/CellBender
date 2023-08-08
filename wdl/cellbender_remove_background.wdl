version 1.0

## Copyright Broad Institute, 2022
##
## LICENSING :
## This script is released under the WDL source code license (BSD-3)
## (see LICENSE in https://github.com/openwdl/wdl).

task run_cellbender_remove_background_gpu {

    input {

        # File-related inputs
        String sample_name
        File input_file_unfiltered  # all barcodes, raw data
        File? barcodes_file  # for MTX and NPZ formats, the bacode information is in a separate file
        File? genes_file  # for MTX and NPZ formats, the gene information is in a separate file
        File? checkpoint_file  # start from a saved checkpoint
        File? truth_file  # only for developers using simulated data

        # Outputs
        String? output_bucket_base_directory  # Google bucket path

        # Docker image with CellBender
        String? docker_image = "us.gcr.io/broad-dsde-methods/cellbender:0.3.0"

        # Used by developers for testing non-dockerized versions of CellBender
        String? dev_git_hash__  # leave blank to run CellBender normally

        # Method configuration inputs
        Int? expected_cells
        Int? total_droplets_included
        Float? force_cell_umi_prior
        Float? force_empty_umi_prior
        String? model
        Int? low_count_threshold
        String? fpr  # in quotes: floats separated by whitespace: the output false positive rate(s)
        Int? epochs
        Int? z_dim
        String? z_layers  # in quotes: integers separated by whitespace
        Float? empty_drop_training_fraction
        Float? learning_rate
        String? exclude_feature_types  # in quotes: strings separated by whitespace
        String? ignore_features  # in quotes: integers separated by whitespace
        Float? projected_ambient_count_threshold
        Float? checkpoint_mins
        Float? final_elbo_fail_fraction
        Float? epoch_elbo_fail_fraction
        Int? num_training_tries
        Float? learning_rate_retry_mult
        Int? posterior_batch_size
        Boolean? estimator_multiple_cpu
        Boolean? constant_learning_rate
        Boolean? debug

        # Hardware-related inputs
        String? hardware_zones = "us-east1-d us-east1-c us-central1-a us-central1-c us-west1-b"
        Int? hardware_disk_size_GB = 50
        Int? hardware_boot_disk_size_GB = 20
        Int? hardware_preemptible_tries = 2
        Int? hardware_cpu_count = 4
        Int? hardware_memory_GB = 32
        String? hardware_gpu_type = "nvidia-tesla-t4"
        String? nvidia_driver_version = "470.82.01"  # need >=465.19.01 for CUDA 11.3

    }

    # For development only: install a non dockerized version of CellBender
    Boolean install_from_git = (if defined(dev_git_hash__) then true else false)

    # Compute the output bucket directory for this sample: output_bucket_base_directory/sample_name/
    String output_bucket_directory = (if defined(output_bucket_base_directory)
                                      then sub(select_first([output_bucket_base_directory]), "/+$", "") + "/${sample_name}/"
                                      else "")

    command {

        set -e  # fail the workflow if there is an error

        # install a specific commit of cellbender from github if called for (-- developers only)
        if [[ ~{install_from_git} == true ]]; then
            echo "Uninstalling pre-installed cellbender"
            yes | pip uninstall cellbender
            echo "Installing cellbender from github"
            # this more succinct version is broken in some older versions of cellbender
            echo "pip install --no-cache-dir -U git+https://github.com/broadinstitute/CellBender.git@~{dev_git_hash__}"
            # yes | pip install --no-cache-dir -U git+https://github.com/broadinstitute/CellBender.git@~{dev_git_hash__}
            # this should always work
            git clone -q https://github.com/broadinstitute/CellBender.git /cromwell_root/CellBender
            cd /cromwell_root/CellBender
            git checkout -q ~{dev_git_hash__}
            yes | pip install --no-cache-dir -U -e /cromwell_root/CellBender
            pip list
            cd /cromwell_root
        fi

        # put the barcodes_file in the right place, if it is provided
        if [[ ! -z "~{barcodes_file}" ]]; then
            dir=$(dirname ~{input_file_unfiltered})
            if [[ "~{input_file_unfiltered}" == *.npz ]]; then
                name="row_index.npy"
            elif [[ "~{barcodes_file}" == *.gz ]]; then
                name="barcodes.tsv.gz"
            else
                name="barcodes.tsv"
            fi
            echo "Moving barcodes file to "$dir"/"$name
            echo "mv ~{barcodes_file} "$dir"/"$name
            [ -f $dir/$name ] || mv ~{barcodes_file} $dir/$name
        fi

        # put the genes_file in the right place, if it is provided
        if [[ ! -z "~{genes_file}" ]]; then
            dir=$(dirname ~{input_file_unfiltered})
            if [[ "~{input_file_unfiltered}" == *.npz ]]; then
                name="col_index.npy"
            elif [[ "~{genes_file}" == *.gz ]]; then
                name="features.tsv.gz"
            else
                name="genes.tsv"
            fi
            echo "Moving genes file to "$dir"/"$name
            echo "mv ~{genes_file} "$dir"/"$name
            [ -f $dir/$name ] || mv ~{genes_file} $dir/$name
        fi

        cellbender remove-background \
            --input "~{input_file_unfiltered}" \
            --output "~{sample_name}_out.h5" \
            --cuda \
            ~{"--checkpoint " + checkpoint_file} \
            ~{"--expected-cells " + expected_cells} \
            ~{"--total-droplets-included " + total_droplets_included} \
            ~{"--fpr " + fpr} \
            ~{"--model " + model} \
            ~{"--low-count-threshold " + low_count_threshold} \
            ~{"--epochs " + epochs} \
            ~{"--force-cell-umi-prior " + force_cell_umi_prior} \
            ~{"--force-empty-umi-prior " + force_empty_umi_prior} \
            ~{"--z-dim " + z_dim} \
            ~{"--z-layers " + z_layers} \
            ~{"--empty-drop-training-fraction " + empty_drop_training_fraction} \
            ~{"--exclude-feature-types " + exclude_feature_types} \
            ~{"--ignore-features " + ignore_features} \
            ~{"--projected-ambient-count-threshold " + projected_ambient_count_threshold} \
            ~{"--learning-rate " + learning_rate} \
            ~{"--checkpoint-mins " + checkpoint_mins} \
            ~{"--final-elbo-fail-fraction " + final_elbo_fail_fraction} \
            ~{"--epoch-elbo-fail-fraction " + epoch_elbo_fail_fraction} \
            ~{"--num-training-tries " + num_training_tries} \
            ~{"--learning-rate-retry-mult " + learning_rate_retry_mult} \
            ~{"--posterior-batch-size " + posterior_batch_size} \
            ~{true="--estimator-multiple-cpu " false=" " estimator_multiple_cpu} \
            ~{true="--constant-learning-rate " false=" " constant_learning_rate} \
            ~{true="--debug " false=" " debug} \
            ~{"--truth " + truth_file}

        # copy outputs to google bucket if output_bucket_base_directory is supplied
        if [[ ! -z "~{output_bucket_directory}" ]]; then
            echo "Copying output data to ~{output_bucket_directory} using gsutil cp"
            gsutil -m cp ~{sample_name}_out* ~{output_bucket_directory}
        fi

    }

    output {
        File log = "${sample_name}_out.log"
        File pdf = "${sample_name}_out.pdf"
        File cell_csv = "${sample_name}_out_cell_barcodes.csv"
        Array[File] metrics_array = glob("${sample_name}_out*_metrics.csv")  # a number of outputs depending on "fpr"
        Array[File] report_array = glob("${sample_name}_out*_report.html")  # a number of outputs depending on "fpr"
        Array[File] h5_array = glob("${sample_name}_out*.h5")  # a number of outputs depending on "fpr"
        String output_dir = "${output_bucket_directory}"
        File ckpt_file = "ckpt.tar.gz"
    }

    runtime {
        docker: "${docker_image}"
        bootDiskSizeGb: hardware_boot_disk_size_GB
        disks: "local-disk ${hardware_disk_size_GB} HDD"
        memory: "${hardware_memory_GB}G"
        cpu: hardware_cpu_count
        zones: "${hardware_zones}"
        gpuCount: 1
        gpuType: "${hardware_gpu_type}"
        nvidiaDriverVersion: "${nvidia_driver_version}"
        preemptible: hardware_preemptible_tries
        checkpointFile: "ckpt.tar.gz"
        maxRetries: 0  # can be used in case of a PAPI error code 2 failure to install GPU drivers
    }
    meta {
        author: "Stephen Fleming"
        email: "sfleming@broadinstitute.org"
        description: "WDL that runs CellBender remove-background on a GPU on Google Cloud hardware. See the [CellBender GitHub repo](https://github.com/broadinstitute/CellBender) and [read the documentation](https://cellbender.readthedocs.io/en/v0.3.0/reference/index.html#command-line-options) for more information."
    }

    parameter_meta {
        sample_name :
            {help: "Name which will be prepended to output files and which will be used to construct the output google bucket file path, if output_bucket_base_directory is supplied, as output_bucket_base_directory/sample_name/"}
        input_file_unfiltered :
            {help: "Input file. This must be raw data that includes all barcodes. See http://cellbender.readthedocs.io for more information on file formats, but importantly, this must be one file, and cannot be a pointer to a directory that contains MTX and TSV files."}
        barcodes_file :
            {help: "Supply this only if your input_file_unfiltered is a sparse NPZ matrix from Optimus lacking metadata."}
        genes_file :
            {help: "Supply this only if your input_file_unfiltered is a sparse NPZ matrix from Optimus lacking metadata."}
        output_bucket_base_directory :
            {help: "Optional google bucket gsURL. If provided, the workflow will create a subfolder called sample_name in this directory and copy outputs there. (Note that the output data would then exist in two places.) Helpful for organization."}
        docker_image :
            {help: "CellBender docker image. Not all CellBender docker image tags will be compatible with this WDL.",
             suggestions: ["us.gcr.io/broad-dsde-methods/cellbender:0.3.0"]}
        checkpoint_file :
            {help: "Optional gsURL for a checkpoint file created using a previous run ('ckpt.tar.gz') on the same dataset (using the same CellBender docker image). This can be used to create a new output with a different --fpr without re-doing the training run."}
        truth_file :
            {help: "Optional file only used by CellBender developers or those trying to benchmark CellBender remove-background on simulated data. Normally, this input would not be supplied."}
        hardware_preemptible_tries :
            {help: "If nonzero, CellBender will be run on a preemptible instance, at a lower cost. If preempted, your run will not start from scratch, but will start from a checkpoint that is saved by CellBender and recovered by Cromwell.  For example, if hardware_preemptible_tries is 2, your run will attempt twice using preemptible instances, and if the job is preempted both times before completing, it will finish on a non-preemptible machine. The cost savings is significant. The potential drawback is that preemption wastes time."}
        checkpoint_mins :
            {help: "Time in minutes between creation of checkpoint files. Bear in mind that Cromwell copies checkpoints to a bucket every ten minutes."}
        hardware_gpu_type :
            {help: "Specify the type of GPU that should be used.  Ensure that the selected hardware_zones have the GPU available.",
             suggestions: ["nvidia-tesla-t4", "nvidia-tesla-k80"]}
    }

}

workflow cellbender_remove_background {

    call run_cellbender_remove_background_gpu

    output {
        File log = run_cellbender_remove_background_gpu.log
        File summary_pdf = run_cellbender_remove_background_gpu.pdf
        File cell_barcodes_csv = run_cellbender_remove_background_gpu.cell_csv
        Array[File] metrics_csv_array = run_cellbender_remove_background_gpu.metrics_array
        Array[File] html_report_array = run_cellbender_remove_background_gpu.report_array
        Array[File] h5_array = run_cellbender_remove_background_gpu.h5_array
        String output_directory = run_cellbender_remove_background_gpu.output_dir
        File checkpoint_file = run_cellbender_remove_background_gpu.ckpt_file
    }

}
