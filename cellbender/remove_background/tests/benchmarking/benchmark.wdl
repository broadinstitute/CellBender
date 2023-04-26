version 1.0

import "cellbender_remove_background.wdl" as cellbender

## Copyright Broad Institute, 2023
##
## LICENSING :
## This script is released under the WDL source code license (BSD-3)
## (see LICENSE in https://github.com/openwdl/wdl).


workflow run_cellbender_benchmark {
    input {
        String? git_hash
    }

    call cellbender.run_cellbender_remove_background_gpu as cb {
        input:
            dev_git_hash__=git_hash

    }

    output {
        File log = cb.log
        File summary_pdf = cb.pdf
        Array[File] html_report_array = cb.report_array
        Array[File] h5_array = cb.h5_array
    }
}
