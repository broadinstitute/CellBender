#!/usr/bin/env python3
"""Submit CellBender benchmarks as Google Batch GPU jobs and poll until completion.

Outputs a JSON file mapping {sample_name: gcs_output_dir} on success.
Exits with code 1 if any job fails.

Usage:
    python benchmarking/submit_benchmarks.py \
        --git-hash abc1234 \
        --run-id 10234567890-1 \
        --outputs-file benchmarking/outputs.json
"""

import argparse
import json
import re
import sys
import time

from google.cloud import batch_v1

PROJECT = "broad-dsde-methods"
REGION = "us-east1"
SERVICE_ACCOUNT = "cellbender-benchmarking@broad-dsde-methods.iam.gserviceaccount.com"
DOCKER_IMAGE = "us.gcr.io/broad-dsde-methods/cellbender:latest"

BENCHMARK_JOBS = [
    {
        "sample": "pbmc8k",
        "input": "gs://broad-dsde-methods-sfleming/cellbender_test/pbmc8k_raw_gene_bc_matrices.h5",
        "fpr": None,
        "truth": None,
    },
    {
        "sample": "hgmm12k",
        "input": "gs://broad-dsde-methods-sfleming/cellbender_test/hgmm_12k_raw_gene_bc_matrices.h5",
        "fpr": None,
        "truth": None,
    },
    {
        "sample": "pbmc5k",
        "input": "gs://broad-dsde-methods-sfleming/cellbender_test/5k_pbmc_protein_v3_nextgem_raw_feature_bc_matrix.h5",
        "fpr": "0.1",
        "truth": None,
    },
    {
        "sample": "rat6k",
        "input": "gs://broad-dsde-methods-sfleming/cellbender_test/PCL_rat_A_LA6_raw_feature_bc_matrix.h5",
        "fpr": None,
        "truth": None,
    },
    {
        "sample": "s7",
        "input": "gs://broad-dsde-methods-sfleming/cellbender_test/s7.h5",
        "fpr": None,
        "truth": "gs://broad-dsde-methods-sfleming/cellbender_test/s7_truth.h5",
    },
]


def build_job_script(
    git_hash: str,
    input_gcs: str,
    sample: str,
    output_gcs_dir: str,
    fpr: str | None,
    truth_gcs: str | None,
) -> str:
    lines = [
        "set -e",
        f"gsutil cp {input_gcs} /tmp/input.h5",
    ]

    if truth_gcs:
        lines.append(f"gsutil cp {truth_gcs} /tmp/truth.h5")

    # dev_git_hash__ mechanism: uninstall pre-installed CellBender, clone the
    # target commit, and reinstall from source so runtime code matches the SHA
    # being benchmarked.
    lines += [
        "pip uninstall -y cellbender",
        "git clone -q https://github.com/broadinstitute/CellBender.git /tmp/CellBender",
        "cd /tmp/CellBender",
        f"git checkout -q {git_hash}",
        "pip install -U pip setuptools",
        "pip install --no-cache-dir -U -e /tmp/CellBender",
        "pip list",
        "cd /tmp",
        "python3 -c \"import torch; assert torch.cuda.is_available(), 'CUDA unavailable — GPU driver not configured'\"",
    ]

    cmd_parts = [
        "cellbender remove-background",
        "    --input /tmp/input.h5",
        f"    --output /tmp/{sample}_out.h5",
        "    --cuda",
        "    --checkpoint-mins 200",
        "    --exclude-feature-types Peaks",
    ]
    if fpr:
        cmd_parts.append(f"    --fpr {fpr}")
    if truth_gcs:
        cmd_parts.append("    --truth /tmp/truth.h5")

    lines.append(" \\\n".join(cmd_parts))
    lines.append(f"gsutil -m cp /tmp/{sample}_out* {output_gcs_dir}/")

    return "\n".join(lines)


def make_job_id(run_id: str, sample: str) -> str:
    raw = f"bm-{run_id}-{sample}"
    sanitized = re.sub(r"[^a-z0-9\-]", "-", raw.lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return sanitized[:63]


def submit_job(
    client: batch_v1.BatchServiceClient,
    job_id: str,
    script: str,
    machine_type: str | None,
    gpu_type: str,
    cpu_count: int | None,
    memory_gb: int | None,
) -> batch_v1.Job:
    runnable = batch_v1.Runnable()
    runnable.container.image_uri = DOCKER_IMAGE
    runnable.container.commands = ["/bin/bash", "-c", script]

    task_spec = batch_v1.TaskSpec()
    task_spec.runnables = [runnable]

    # When no machine type is given, set CPU/memory hints and let Batch choose
    # a compatible N1 instance for the requested GPU.
    if not machine_type and (cpu_count or memory_gb):
        if cpu_count:
            task_spec.compute_resource.cpu_milli = cpu_count * 1000
        if memory_gb:
            task_spec.compute_resource.memory_mib = memory_gb * 1024

    task_group = batch_v1.TaskGroup()
    task_group.task_count = 1
    task_group.task_spec = task_spec

    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instances.install_gpu_drivers = True
    if machine_type:
        instances.policy.machine_type = machine_type
    instances.policy.accelerators = [batch_v1.AllocationPolicy.Accelerator(type_=gpu_type, count=1)]

    location = batch_v1.AllocationPolicy.LocationPolicy()
    location.allowed_locations = [
        "zones/us-east1-d",
        "zones/us-east1-c",
        "zones/us-central1-a",
        "zones/us-central1-c",
    ]

    # The Batch job VMs run as this SA so they can read/write the GCS bucket.
    # Prerequisite: run once to allow the SA to act as itself:
    #   gcloud iam service-accounts add-iam-policy-binding \
    #     cellbender-benchmarking@broad-dsde-methods.iam.gserviceaccount.com \
    #     --role=roles/iam.serviceAccountUser \
    #     --member="serviceAccount:cellbender-benchmarking@broad-dsde-methods.iam.gserviceaccount.com"
    allocation_policy = batch_v1.AllocationPolicy()
    allocation_policy.instances = [instances]
    allocation_policy.location = location
    allocation_policy.service_account.email = SERVICE_ACCOUNT

    job = batch_v1.Job()
    job.task_groups = [task_group]
    job.allocation_policy = allocation_policy
    job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

    return client.create_job(
        batch_v1.CreateJobRequest(
            parent=f"projects/{PROJECT}/locations/{REGION}",
            job_id=job_id,
            job=job,
        )
    )


def poll_until_done(
    client: batch_v1.BatchServiceClient,
    job_names: list[str],
    poll_interval: int,
) -> dict[str, str]:
    terminal = {"SUCCEEDED", "FAILED", "DELETION_IN_PROGRESS"}
    final: dict[str, str] = {}

    while len(final) < len(job_names):
        for name in job_names:
            if name in final:
                continue
            state = client.get_job(name=name).status.state.name
            if state in terminal:
                final[name] = state
                print(f"[done] {name.split('/')[-1]}: {state}", flush=True)

        if len(final) < len(job_names):
            remaining = len(job_names) - len(final)
            print(
                f"[poll] {remaining} job(s) still running, checking again in {poll_interval}s...",
                flush=True,
            )
            time.sleep(poll_interval)

    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-hash", required=True, help="Git SHA or branch to benchmark")
    parser.add_argument(
        "--run-id",
        default=time.strftime("%Y%m%d%H%M%S"),
        help="Unique suffix for Batch job IDs (default: timestamp)",
    )
    parser.add_argument(
        "--output-bucket",
        default="gs://broad-dsde-methods-sfleming/cellbender_test/benchmarks",
    )
    parser.add_argument("--outputs-file", default="outputs.json")
    parser.add_argument("--poll-interval", type=int, default=60)

    hw = parser.add_argument_group(
        "hardware",
        "Machine type takes precedence; cpu-count/memory-gb are used only when "
        "machine-type is omitted, as resource hints for Batch to auto-select a machine.",
    )
    hw.add_argument(
        "--machine-type",
        default="n1-highmem-8",
        help="GCP machine type (default: n1-highmem-8 = 8 vCPU / 52 GB). Examples: n1-highmem-16, n1-standard-8.",
    )
    hw.add_argument(
        "--gpu-type",
        default="nvidia-tesla-t4",
        help="GCP GPU accelerator type (default: nvidia-tesla-t4). Also valid: nvidia-tesla-k80.",
    )
    hw.add_argument(
        "--cpu-count",
        type=int,
        default=None,
        help="CPU count hint (used only when --machine-type is not set).",
    )
    hw.add_argument(
        "--memory-gb",
        type=int,
        default=None,
        help="Memory in GB hint (used only when --machine-type is not set).",
    )

    args = parser.parse_args()

    # A machine type of the sentinel default means the user didn't override it.
    # Treat an explicitly cleared machine type (empty string from workflow) the
    # same as absent.
    machine_type: str | None = args.machine_type or None

    client = batch_v1.BatchServiceClient()

    output_dirs: dict[str, str] = {}
    job_names: list[str] = []

    hw_desc = f"machine={machine_type or 'auto'}, gpu={args.gpu_type}"
    if not machine_type and (args.cpu_count or args.memory_gb):
        hw_desc += f", cpu={args.cpu_count}, memory={args.memory_gb}GB"
    print(f"Hardware: {hw_desc}", flush=True)

    for job_def in BENCHMARK_JOBS:
        sample = job_def["sample"]
        output_gcs_dir = f"{args.output_bucket.rstrip('/')}/{args.git_hash}/{sample}"
        output_dirs[sample] = output_gcs_dir

        job_id = make_job_id(args.run_id, sample)
        script = build_job_script(
            git_hash=args.git_hash,
            input_gcs=job_def["input"],
            sample=sample,
            output_gcs_dir=output_gcs_dir,
            fpr=job_def["fpr"],
            truth_gcs=job_def["truth"],
        )

        print(f"Submitting {job_id} ...", flush=True)
        job = submit_job(
            client,
            job_id,
            script,
            machine_type=machine_type,
            gpu_type=args.gpu_type,
            cpu_count=args.cpu_count,
            memory_gb=args.memory_gb,
        )
        job_names.append(job.name)
        print(f"  -> {job.name}", flush=True)

    print(f"\nPolling {len(job_names)} jobs every {args.poll_interval}s ...\n", flush=True)
    final_states = poll_until_done(client, job_names, args.poll_interval)

    failed = [n for n, s in final_states.items() if s != "SUCCEEDED"]
    if failed:
        print("\nFailed jobs:", file=sys.stderr)
        for name in failed:
            print(f"  {name}: {final_states[name]}", file=sys.stderr)
        sys.exit(1)

    with open(args.outputs_file, "w") as f:
        json.dump(output_dirs, f, indent=2)

    print(f"\nAll jobs succeeded. Output paths written to {args.outputs_file}.")


if __name__ == "__main__":
    main()
