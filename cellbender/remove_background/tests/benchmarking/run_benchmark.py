"""Run a github commit of CellBender on test data for benchmarking purposes
Example:
    $ conda activate cromshell
    $ python run_benchmark.py \
        --git 2927346f4c513a217ac8ad076e494dd1adbf70e1 \
        --input "gs://path" \
        --sample "name" \
        --truth "gs://path_to_truth"
"""

import tempfile
import json
import os
import sys
import subprocess
import shutil
import random
import argparse
from typing import Tuple, Dict, Optional, List


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a specific commit of CellBender on a dataset "
                    "using cromshell.",
    )

    parser.add_argument('-g', '--git',
                        type=str,
                        required=False,
                        default=None,
                        dest='git_hash',
                        help='Specific github commit from the CellBender repo: '
                             'can be a tag, a branch, or a commit sha. Not '
                             'including the flag will run using the base '
                             'docker image specified in benchmark_inputs.json')
    parser.add_argument('-i', '--input',
                        type=str,
                        required=True,
                        dest='input_file',
                        help='Input path (must be a gsURL, i.e. '
                             '"gs://bucket/path/file.h5")')
    parser.add_argument('-s', '--sample',
                        type=str,
                        required=True,
                        dest='sample',
                        help='Sample name: used to create output file names')
    parser.add_argument('-t', '--truth',
                        type=str,
                        required=False,
                        default=None,
                        dest='truth_file',
                        help='Truth data path for simulated data (must be a '
                             'gsURL, i.e. "gs://bucket/path/file.h5")')
    parser.add_argument('-f', '--fpr',
                        type=str,
                        required=False,
                        default="0.01",
                        dest='fpr',
                        help='FPR for CellBender')
    parser.add_argument('-q', '--quiet',
                        action='store_true',
                        default=False,
                        dest='quiet',
                        help='Include this flag to avoid printing to stdout')

    return parser


def update_input_json(template_file: str,
                      substitutions: Dict[str, str],
                      tmpdir: tempfile.TemporaryDirectory,
                      verbose: bool = True) -> Tuple[str, str]:
    """Create a new input json for a Cromwell run based on a template"""

    # read template into dict
    with open(template_file, mode='r') as f:
        template = json.load(f)

    # perform substitutions
    for key, value in template.items():
        replacement = substitutions.get(value, '__absent')
        template[key] = replacement if (replacement != '__absent') else value

    # write filled-in input json
    json_inputs_file = os.path.join(tmpdir, 'inputs.json')
    with open(json_inputs_file, mode='w') as f:
        json.dump(template, f, indent=2)

    # write options json that disables call-caching
    json_options_file = os.path.join(tmpdir, 'options.json')
    options_dict = {
        "write_to_cache": False,
        "read_from_cache": False,
    }
    with open(json_options_file, mode='w') as f:
        json.dump(options_dict, f, indent=2)

    if verbose:
        print('Contents of inputs.json:')
        subprocess.run(['cat', json_inputs_file])
        print('', end='\n')

    if verbose:
        print('Contents of options.json:')
        subprocess.run(['cat', json_options_file])
        print('', end='\n')

    return json_inputs_file, json_options_file


def cromshell_submit(wdl: str,
                     inputs: str,
                     options: str,
                     dependencies: List[str],
                     tmpdir: tempfile.TemporaryDirectory,
                     alias: Optional[str] = None,
                     verbose: bool = True) -> Tuple[str, str]:
    """Submit job via cromshell and return the workflow-id and alias

    NOTE: the whole dependency zipping thing is way more difficult than it has
    to be and took me hours.  it is super fragile.  but at least now I think I
    can put this benchmark.wdl anywhere I want to.
    """

    # zip up dependencies
    if verbose:
        print('Zipping dependencies')
    dependencies_zip = os.path.join(tmpdir, 'dependencies.zip')
    dependencies_dir = tmpdir
    for file in dependencies:
        shutil.copy(file, dependencies_dir)
    subprocess.run(['zip', '-j', dependencies_zip,
                    ' '.join([os.path.join(dependencies_dir, os.path.basename(f))
                              for f in dependencies])])

    # move WDL to tmpdir
    tmp_wdl = os.path.join(tmpdir, os.path.basename(wdl))
    shutil.copy(wdl, tmp_wdl)

    submit_cmd = ['cromshell', 'submit',
                  tmp_wdl,
                  inputs,
                  '--options-json',
                  options,
                  '--dependencies-zip',
                  dependencies_zip]

    # submit job
    if verbose:
        print(f'Submitting WDL {tmp_wdl}')
    current_path = os.getcwd()
    os.chdir(tmpdir)
    out = subprocess.run(submit_cmd)
    os.chdir(current_path)
    out.check_returncode()  # error if this failed

    # get workflow-id
    out = subprocess.run(['cromshell', 'status'], capture_output=True)
    d = json.loads(out.stdout)
    workflow_id = d.get('id')

    # alias job
    if alias is not None:
        # solve the issue where an alias cannot be made twice
        hash = random.getrandbits(128)
        alias = alias + '__runhash_' + str(hash)[:10]
        subprocess.run(['cromshell', 'alias', '-1', alias])

    return workflow_id, alias


if __name__ == '__main__':

    # handle input arguments
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    # args = validate_args(args)

    with tempfile.TemporaryDirectory() as tmpdir:

        # determine substitutions to be made in input json
        substitutions = {
            'GIT_HASH': args.git_hash,
            'INPUT_GSURL': args.input_file,
            'SAMPLE_NAME': args.sample,
            'TRUTH_GSURL': args.truth_file,
            'FPR': args.fpr,
        }

        # create the updated input json
        inputs_json, options_json = update_input_json(
            template_file='benchmark_inputs.json',
            substitutions=substitutions,
            tmpdir=tmpdir,
            verbose=not args.quiet,
        )

        # get the path to the cellbender WDL
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cellbender_wdl_path = os.path.abspath(os.path.join(
            this_dir, '..', '..', '..', '..', 'wdl',
            'cellbender_remove_background.wdl',
        ))

        # run cromshell submit
        if args.git_hash is not None:
            alias = '_'.join(['cellbender', args.sample, args.git_hash])
        else:
            alias = '_'.join(['cellbender', args.sample])
        workflow_id, alias = cromshell_submit(
            wdl=os.path.join(this_dir, 'benchmark.wdl'),
            inputs=inputs_json,
            options=options_json,
            dependencies=[cellbender_wdl_path],
            alias=alias,
            tmpdir=tmpdir,
        )

    # show workflow-id alias
    if not args.quiet:
        print('Sucessfully submitted job:')
        if alias is not None:
            print(alias)
        print(workflow_id)

    sys.exit(0)
