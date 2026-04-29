"""Tabulate outputs from CellBender benchmarking as a samples.tsv for Terra
Example:
    $ conda activate cromshell
    $ python run_benchmark_result_tabulation.py \
        --workflows 2927346f4c513a217ac8ad076e494dd1adbf70e1 \
        --output "samples.tsv"
"""

import pandas as pd
import os
import re
import sys
import subprocess
import argparse
from io import StringIO
from typing import Tuple, Dict, Optional, List, Union


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Terra-compatible data table TSV from cromshell "
                    "benchmarking outputs.",
    )
    parser.add_argument('-w', '--workflows',
                        type=str,
                        required=True,
                        dest='grep',
                        help='grep to choose workflows from the cromshell database')
    parser.add_argument('-n', '--note',
                        type=str,
                        required=True,
                        dest='note',
                        help='Annotation describing the git commit')
    parser.add_argument('-o', '--output',
                        type=str,
                        required=True,
                        dest='output_file',
                        help='Output TSV file')

    return parser


def find_cromshell_workflow_ids_dates(grep: str) -> Tuple[List[str], List[str]]:
    """Find workflows in the cromshell database matching some string"""

    date_col = 0
    workflow_id_col = 2

    # use cromshell list to get the database, then comb through it
    out = subprocess.run(['cromshell', 'list'], capture_output=True)
    s = out.stdout.decode()
    _RE_COMBINE_WHITESPACE = re.compile(r"(?a: +)")
    s = _RE_COMBINE_WHITESPACE.sub(" ", s).strip()
    greps = grep.split(' ')  # separately match all search elements (space delimited)
    selected_rows = [r for r in s.split('\n') if all([g in r for g in greps])]

    for r in selected_rows:
        if ('Succeeded' not in r):
            print(f'WARNING: Skipping\n{r}\nbecause the run has not completed')
    selected_rows = [r for r in selected_rows if ('Succeeded' in r)]

    workflow_ids = [r.split(' ')[workflow_id_col] for r in selected_rows]
    dates = [r.split(' ')[date_col] for r in selected_rows]

    return workflow_ids, dates


def get_cromshell_output_h5(workflow: str, grep: str = '_out.h5') -> Union[str, List[str]]:
    """Use cromshell list-outputs to get the relevant file gsURL"""

    output = grep_from_command(['cromshell', 'list-outputs', workflow], grep=grep)
    out = output.decode().lstrip('run_cellbender_benchmark.h5_array: ').rstrip('\n').split('\n')
    if len(out) > 1:
        return out
    else:
        return out[0]


def sample_name_from_h5(h5: str) -> str:
    """Get sample name by parsing h5 file name"""
    return os.path.basename(h5).replace('_out.h5', '')


def grep_from_command(command: List[str], grep: str) -> bytes:
    """Pipe a command to grep and give the output"""
    ps = subprocess.Popen(command, stdout=subprocess.PIPE)
    output = subprocess.check_output(['grep', grep], stdin=ps.stdout)
    ps.wait()
    return output


def metadata_from_workflow_id(workflow: str) -> Tuple[str, str, Optional[str]]:
    """Get other metadata for a workflow id: git hash, input file, truth file"""

    # git hash
    output = grep_from_command(['cromshell', 'metadata', workflow],
                               grep='"git_hash":')
    git_hash = output.decode().split('"git_hash": ')[-1].lstrip('"').split('"')[0]

    # input file
    output = grep_from_command(['cromshell', 'metadata', workflow],
                               grep='run_cellbender_benchmark.cb.input_file_unfiltered')
    input_file = 'gs://' + output.decode().split('gs://')[-1].split('"')[0]

    # truth file
    output = grep_from_command(['cromshell', 'metadata', workflow],
                               grep='run_cellbender_benchmark.cb.truth_file')
    if 'null' not in output.decode():
        truth_file = 'gs://' + output.decode().split('gs://')[-1].split('"')[0]
    else:
        truth_file = None

    return git_hash, input_file, truth_file


if __name__ == '__main__':

    # handle input arguments
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    # args = validate_args(args)

    workflow_ids, dates = find_cromshell_workflow_ids_dates(args.grep)
    output_h5s = [get_cromshell_output_h5(id) for id in workflow_ids]
    samples = [sample_name_from_h5(h5) for h5 in output_h5s]
    list_of_tuples = [metadata_from_workflow_id(id) for id in workflow_ids]
    if len(list_of_tuples) > 0:
        git_hashes, input_h5s, truth_h5s = zip(*list_of_tuples)

        run_ids = [sample + '_' + git for sample, git in zip(samples, git_hashes)]

        df = pd.DataFrame(data={'entity:sample_id': run_ids,
                                'git_commit': git_hashes,
                                'sample': samples,
                                'output_h5': output_h5s,
                                'input_h5': input_h5s,
                                'truth_h5': truth_h5s,
                                'cromwell_workflow_id': workflow_ids,
                                'date_time': dates,
                                'note': args.note})

        df.to_csv(args.output_file, sep='\t', index=False)
    else:
        print('No submissions selected')

    sys.exit(0)
