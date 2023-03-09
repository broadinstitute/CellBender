# Benchmarking CellBender versions and changes

## Running benchmarking tests

There is a benchmarking WDL in /cellbender/remove_background/wdl called 
`benchmark.wdl`.  This can be run using a local installation of cromshell
(`conda install -c bioconda cromshell`) by running the python script in 
`run_benchmark.py` as in

```console
python run_benchmark.py \
    --git "v0.2.1" \
    --input "gs://broad-dsde-methods-sfleming/cellbender_test/pbmc8k_raw_gene_bc_matrices.h5" \
    --sample pbmc8k
```

The `--git` input can be a full commit sha or a tag or a branch.  Anything that 
can be pip installed via `pip install -y git+https://github.com/broadinstitute/CellBender.git@<SOMETHING>`

(The caveat here is that older versions of CellBender cannot be installed from 
github in this way, so this is a test that will work moving forward from 
commit `cb2d209d5aedffe71e28947bc5b7859600aef64d`)

### Datasets from the paper

To re-run the datasets that were analyzed in the CellBender paper, try running

```console
./run_usual_benchmarks.sh b9d2953c76c538d13549290bd986af4e6a1780d5
```

You can check their status via `cromshell` using

```console
cromshell list -u | tail -n 5
```

## Validation of outputs

TBW
