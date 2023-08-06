# Benchmarking CellBender versions and changes

## Running benchmarking tests

There is a benchmarking WDL in /cellbender/remove_background/tests/benchmarking called 
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

To re-run the datasets that were analyzed in the CellBender paper, plus 
several other interesting benchmarks, try running

```console
./run_usual_benchmarks.sh b9d2953c76c538d13549290bd986af4e6a1780d5
```

You can check their status via `cromshell` using

```console
cromshell list -u | tail -n 5
```

## Validation of outputs

Validation is comprised of a Terra notebook in the following workspace:
https://app.terra.bio/#workspaces/broad-firecloud-dsde-methods/sfleming_dev

To prepare a data table for the Terra workspace (once all jobs 
have completed):

```console
python run_benchmark_result_tabulation.py \
    --workflows b9d2953c76c538d13549290bd986af4e6a1780d5 \
    --output samples.tsv
```

where, for example, `b9d2953c76c538d13549290bd986af4e6a1780d5` is 
the git hash used to run `run_usual_benchmarks.sh`. (The script will `grep` 
the local cromshell database for all runs that match. More than one search 
term can be used.)

Then manually upload the `samples.tsv` to Terra as a data table.

Open the validation notebook and enter the git commit 
`b9d2953c76c538d13549290bd986af4e6a1780d5` at the top.  Then 
run the entire notebook to produce plots.
