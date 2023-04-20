"""Full run through on small simulated data"""

from cellbender.remove_background.cli import CLI
from cellbender.base_cli import get_populated_argparser

import os


def test_full_run(tmpdir_factory, h5_v3_file):
    tmp_dir = tmpdir_factory.mktemp('data')
    filename = tmp_dir.join('out.h5')

    os.chdir(tmp_dir)  # so checkpoint file goes to temp dir and gets removed

    input_args = ['cellbender', 'remove-background',
                  '--input', str(h5_v3_file.name),
                  '--output', str(filename),
                  '--epochs', '5']

    args = get_populated_argparser().parse_args(input_args[1:])
    args = CLI.validate_args(args=args)
    CLI.run(args=args)
