"""Full run through on small simulated data"""

from cellbender.remove_background.cli import CLI
from cellbender.base_cli import get_populated_argparser
from cellbender.remove_background.downstream import anndata_from_h5
from cellbender.remove_background import consts
import numpy as np
import os
import pytest

from .conftest import USE_CUDA


@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
def test_full_run(tmpdir_factory, h5_v3_file, cuda):
    """Do a full run of the command line tool using a small simulated dataset"""

    tmp_dir = tmpdir_factory.mktemp('data')
    filename = tmp_dir.join('out.h5')

    os.chdir(tmp_dir)  # so checkpoint file goes to temp dir and gets removed

    # handle input arguments using the argparser
    input_args = ['cellbender', 'remove-background',
                  '--input', str(h5_v3_file.name),
                  '--output', str(filename),
                  '--epochs', '5']
    if cuda:
        input_args.append('--cuda')
    args = get_populated_argparser().parse_args(input_args[1:])
    args = CLI.validate_args(args=args)

    # do a full run through
    posterior = CLI.run(args=args)

    # do some checks

    # ensure the cell probabilities in the posterior object match the output file
    p_for_analyzed_barcodes = posterior.latents_map['p']
    adata = anndata_from_h5(str(filename), analyzed_barcodes_only=True)
    file_p_for_analyzed_barcodes = adata.obs['cell_probability'].values
    np.testing.assert_array_equal(p_for_analyzed_barcodes, file_p_for_analyzed_barcodes)

    # ensure the cell barcodes are the same both ways
    cell_barcodes = np.genfromtxt(str(filename)[:-3] + '_cell_barcodes.csv', dtype=str, delimiter='\n')
    adata_cell_barcodes = adata.obs_names[adata.obs['cell_probability'] > consts.CELL_PROB_CUTOFF]
    assert set(cell_barcodes) == set(adata_cell_barcodes), \
        'Cell barcodes in h5 are different from those in CSV file'
    
    # ensure there are no negative count matrix entries in the output
    assert np.all(adata.X.data >= 0), 'Negative count matrix entries in output'
