#!/bin/bash

set -euxo pipefail

# runs from the root directory of the repo

# find all WDL files
# WDL_FILES=$(find . -type f -name "*.wdl")
WDL_FILES=("./wdl/cellbender_remove_background.wdl")

for WDL in ${WDL_FILES}; do
  miniwdl check ${WDL}
done