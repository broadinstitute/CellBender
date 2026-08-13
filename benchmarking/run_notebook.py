#!/usr/bin/env python3
"""Execute the benchmarking notebook via Papermill, injecting GCS output paths."""

import argparse
import json

import papermill as pm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--outputs-file", required=True, help="JSON file from submit_benchmarks.py")
    parser.add_argument("--input-notebook", required=True)
    parser.add_argument("--output-notebook", required=True)
    parser.add_argument(
        "--figure-dir",
        default="/tmp/cellbender_benchmark_figures",
        help="Directory where savefig() calls write PDFs (default: /tmp/cellbender_benchmark_figures)",
    )
    args = parser.parse_args()

    with open(args.outputs_file) as f:
        outputs = json.load(f)

    pm.execute_notebook(
        args.input_notebook,
        args.output_notebook,
        parameters={
            "git_commit": args.git_commit,
            "outputs_json": json.dumps(outputs),
            "figure_dir": args.figure_dir,
        },
        kernel_name="python3",
    )


if __name__ == "__main__":
    main()
