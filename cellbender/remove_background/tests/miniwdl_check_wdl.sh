#!/bin/bash

set -euxo pipefail

# runs from the root directory of the repo

# find all WDL files
WDL_FILES=$(find . -type f -name "*.wdl")

for WDL in ${WDL_FILES}; do
  miniwdl check ${WDL}
done