#!/usr/bin/env python3
"""Submit a single CellBender Google Batch GPU job for ad-hoc dev testing.

Submits one job and exits immediately — no polling. Check Cloud Logging or the
GCP Batch console for progress; outputs land in the printed GCS directory.

Usage:
    python benchmarking/submit_single.py \
        --git-hash abc1234 \
        --input-file gs://bucket/path/to/input.h5 \
        --extra-args "--total-droplets-included 25000 --epochs 150"
"""

import argparse
import os
import re
import time

from google.cloud import batch_v1
from submit_benchmarks import (
    PROJECT,
    REGION,
    _build_preamble_lines,
    _machine_resources,
    make_job_id,
    submit_job,
)

DEFAULT_OUTPUT_BUCKET = "gs://broad-dsde-methods-sfleming/cellbender_test/github_dev_runs"


def build_dev_job_script(
    git_hash: str,
    input_gcs: str,
    sample: str,
    output_gcs_dir: str,
    extra_args: str,
) -> str:
    lines = _build_preamble_lines(git_hash, input_gcs)
    cmd_parts = [
        "cellbender remove-background",
        "    --input /tmp/input.h5",
        f"    --output /tmp/{sample}_out.h5",
        "    --cuda",
    ]
    if extra_args.strip():
        cmd_parts.append(f"    {extra_args.strip()}")
    lines.append(" \\\n".join(cmd_parts))
    lines.append(f"gsutil -m cp /tmp/{sample}_out* {output_gcs_dir}/")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-hash", required=True, help="Git SHA to run")
    parser.add_argument("--input-file", required=True, help="GCS path to input .h5 file")
    parser.add_argument(
        "--sample-name",
        default=None,
        help="Label used in the job ID and output path (default: input filename stem)",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra args appended verbatim to the cellbender command",
    )
    parser.add_argument(
        "--output-bucket",
        default=DEFAULT_OUTPUT_BUCKET,
        help=f"GCS output prefix (default: {DEFAULT_OUTPUT_BUCKET})",
    )
    parser.add_argument(
        "--run-id",
        default=time.strftime("%Y%m%d%H%M%S"),
        help="Unique suffix for the Batch job ID (default: timestamp)",
    )

    hw = parser.add_argument_group(
        "hardware",
        "CPU and memory are auto-derived from the machine type for N1/N2 families.",
    )
    hw.add_argument("--machine-type", default="n1-highmem-8")
    hw.add_argument("--gpu-type", default="nvidia-tesla-t4")
    hw.add_argument("--cpu-count", type=int, default=None)
    hw.add_argument("--memory-gb", type=int, default=None)

    args = parser.parse_args()

    raw_sample = args.sample_name or os.path.splitext(os.path.basename(args.input_file))[0]
    sample = re.sub(r"[^a-z0-9\-]", "-", raw_sample.lower()).strip("-")

    machine_type: str | None = args.machine_type or None
    cpu_count: int | None = args.cpu_count
    memory_gb: int | None = args.memory_gb

    if machine_type and (cpu_count is None or memory_gb is None):
        parsed = _machine_resources(machine_type)
        if parsed:
            derived_cpu, derived_mem = parsed
            cpu_count = cpu_count if cpu_count is not None else derived_cpu
            memory_gb = memory_gb if memory_gb is not None else derived_mem
        else:
            print(
                f"Warning: cannot auto-derive CPU/memory for {machine_type!r}. "
                "Pass --cpu-count and --memory-gb explicitly.",
                flush=True,
            )

    output_gcs_dir = f"{args.output_bucket.rstrip('/')}/{args.git_hash}/{sample}"
    job_id = make_job_id(args.run_id, sample)
    script = build_dev_job_script(
        git_hash=args.git_hash,
        input_gcs=args.input_file,
        sample=sample,
        output_gcs_dir=output_gcs_dir,
        extra_args=args.extra_args,
    )

    hw_desc = f"machine={machine_type or 'auto'}, gpu={args.gpu_type}, cpu={cpu_count}, memory={memory_gb}GB"
    print(f"Hardware: {hw_desc}", flush=True)

    client = batch_v1.BatchServiceClient()
    print(f"Submitting {job_id} ...", flush=True)
    job = submit_job(
        client,
        job_id,
        script,
        machine_type=machine_type,
        gpu_type=args.gpu_type,
        cpu_count=cpu_count,
        memory_gb=memory_gb,
    )

    print(f"\nJob submitted: {job.name}", flush=True)
    print(f"Output GCS dir: {output_gcs_dir}", flush=True)
    print(
        f"\nLogs: https://console.cloud.google.com/batch/jobs?project={PROJECT}&region={REGION}",
        flush=True,
    )


if __name__ == "__main__":
    main()
