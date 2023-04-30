version 1.0

## Copyright Broad Institute, 2023
##
## LICENSING :
## This script is released under the WDL source code license (BSD-3)
## (see LICENSE in https://github.com/openwdl/wdl).

task run_check_pytorch_cuda_status {
    input {
        String docker_image
        Int? hardware_boot_disk_size_GB = 20
        String? hardware_zones = "us-east1-d us-east1-c us-central1-a us-central1-c us-west1-b"
        String? hardware_gpu_type = "nvidia-tesla-t4"
        String? nvidia_driver_version = "470.82.01"  # need >=465.19.01 for CUDA 11.3
    }
    command {
        set -e
        python <<CODE
        import torch
        assert torch.cuda.is_available()
        CODE

        cellbender remove-background -h
    }
    runtime {
        docker: "${docker_image}"
        bootDiskSizeGb: hardware_boot_disk_size_GB
        zones: "${hardware_zones}"
        gpuCount: 1
        gpuType: "${hardware_gpu_type}"
        nvidiaDriverVersion: "${nvidia_driver_version}"
        maxRetries: 0
    }
}


workflow check_pytorch_cuda_status {
    call run_check_pytorch_cuda_status
}
