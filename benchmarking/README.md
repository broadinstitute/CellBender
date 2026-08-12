# CellBender Benchmarking

End-to-end benchmarking runs CellBender on five real and simulated datasets on
Google Cloud GPU hardware, then executes a validation notebook that produces
diagnostic plots. Everything is orchestrated from GitHub Actions — no local
Cromwell, Terra, or cromshell installation required.

## Quick start: run benchmarks from GitHub

1. Go to **Actions → Benchmark CellBender → Run workflow** in the GitHub UI.
2. Fill in the inputs (all optional except the branch/SHA):

   | Input | Default | Notes |
   |---|---|---|
   | `git_ref` | `main` | Branch, tag, or full SHA to benchmark |
   | `run_note` | *(blank)* | Label appended to the artifact name, e.g. `v0.3.3-release` |
   | `machine_type` | `n1-highmem-8` | GCP machine type (8 vCPU / 52 GB RAM). Also try `n1-highmem-16`. |
   | `gpu_type` | `nvidia-tesla-t4` | GCP GPU. Also try `nvidia-tesla-k80`. |
   | `cpu_count` | *(blank)* | CPU hint — only used when `machine_type` is cleared |
   | `memory_gb` | *(blank)* | Memory hint — only used when `machine_type` is cleared |

3. The workflow takes 1–2 hours (jobs run in parallel on Batch). When it
   finishes, download the artifact from the workflow summary page. The artifact
   is a fully executed Jupyter notebook (`cellbender_benchmarking_executed.ipynb`)
   with all plots embedded. Open it in JupyterLab to review results.

## What the workflow does

```
workflow_dispatch
    │
    ├─ Checkout repo at git_ref, resolve full SHA
    ├─ Authenticate to GCP via Workload Identity Federation
    │
    ├─ submit_benchmarks.py
    │   ├─ Submits 5 Google Batch GPU jobs (one per dataset, all in parallel)
    │   ├─ Each job: installs CellBender from the exact commit, runs remove-background
    │   └─ Polls until all jobs SUCCEEDED; writes outputs.json with GCS paths
    │
    ├─ run_notebook.py (via Papermill)
    │   ├─ Injects git_commit and GCS output paths into the notebook
    │   ├─ Downloads output H5 files from GCS and runs all analysis cells
    │   └─ Saves executed notebook with embedded plots
    │
    └─ Upload executed notebook as a GitHub Actions artifact (90-day retention)
```

## Benchmark datasets

| Sample | Dataset | FPR | Ground truth |
|---|---|---|---|
| `pbmc8k` | 10x PBMC 8k | default | — |
| `hgmm12k` | 10x HGMM 12k (mixed species) | default | — |
| `pbmc5k` | 10x PBMC 5k CITE-seq | 0.1 | — |
| `rat6k` | Broad PCL rat heart 6k | default | — |
| `s7` | Simulated dataset s7 (hgmm-like) | default | `s7_truth.h5` |

All input files are in `gs://broad-dsde-methods-sfleming/cellbender_test/`.
Output files are written to
`gs://broad-dsde-methods-sfleming/cellbender_test/benchmarks/{git_sha}/{sample}/`.

## Running locally (without GitHub Actions)

You can run the submission script directly from a machine authenticated to GCP:

```bash
gcloud auth application-default login

python benchmarking/submit_benchmarks.py \
    --git-hash abc1234def \
    --run-id my-local-run \
    --outputs-file benchmarking/outputs.json
```

Then execute the notebook manually:

```bash
pip install papermill ipykernel scanpy colorcet cellbender

python benchmarking/run_notebook.py \
    --git-commit abc1234 \
    --outputs-file benchmarking/outputs.json \
    --input-notebook benchmarking/cellbender_benchmarking.ipynb \
    --output-notebook benchmarking/cellbender_benchmarking_executed.ipynb
```

Or open `cellbender_benchmarking.ipynb` directly in Jupyter and edit the
`git_commit` and `outputs_json` variables in the **parameters cell** (the first
code cell, tagged `parameters`) to point at your outputs before running.

## GCP setup (one-time, for maintainers only)

The workflow requires a GCP service account with Workload Identity Federation
configured for this repository. The setup commands and secret values are
intentionally not recorded here. Contact the repository maintainer
(`sfleming@broadinstitute.org`) to get access or to reconstruct the
infrastructure if needed.

The service account needs the following permissions:
- `roles/batch.jobsAdmin` on the GCP project
- `roles/storage.objectAdmin` on the benchmark GCS bucket
- `roles/iam.serviceAccountUser` on itself (so it can assign itself as the
  Batch job VM identity)
- Workload Identity binding for this GitHub repository
