"""Functions for creation of an HTML report that plots and explains output."""

from cellbender.remove_background.downstream import \
    load_anndata_from_input, \
    load_anndata_from_input_and_output, \
    _load_anndata_from_input_and_decontx
from cellbender.base_cli import get_version
from cellbender.remove_background import consts
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.sparse as sp
import scipy.stats
from IPython.display import display, Markdown, HTML

import subprocess
import datetime
import os
import shutil
import logging
from typing import Dict, Optional


logger = logging.getLogger('cellbender')
warnings = []
TIMEOUT = 1200  # twenty minutes should always be way more than enough

# counteract an error when I run locally
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

run_notebook_str = lambda file: \
    f'jupyter nbconvert ' \
    f'--ExecutePreprocessor.timeout={TIMEOUT} ' \
    f'--to notebook ' \
    f'--allow-errors ' \
    f'--execute {file}'
to_html_str = lambda file, output: \
    f'jupyter nbconvert ' \
    f'--to html ' \
    f'--TemplateExporter.exclude_input=True ' \
    f'{file}'


def _run_notebook(file):
    shutil.copy(file, 'tmp.report.ipynb')
    subprocess.run(run_notebook_str(file='tmp.report.ipynb'), shell=True)
    os.remove('tmp.report.ipynb')
    return 'tmp.report.nbconvert.ipynb'


def _to_html(file, output) -> str:
    subprocess.run(to_html_str(file=file, output=output), shell=True)
    shutil.move(file.replace(".ipynb", ".html"), output)
    os.remove(file)
    return output


def _postprocess_html(file: str, title: str):
    try:
        with open(file, mode='r', encoding="utf8", errors="surrogateescape") as f:
            html = f.read()
        html = html.replace('<title>tmp.report.nbconvert</title>',
                            f'<title>{title}</title>')
        with open(file, mode='w', encoding="utf8", errors="surrogateescape") as f:
            f.write(html)
    except:
        logger.warning('Failed to overwrite default HTML report title. '
                       'This is purely aesthetic and does not affect output.')


def run_notebook_make_html(file, output) -> str:
    """Run Jupyter notebook to populate report and then convert to HTML.

    Args:
        file: Notebook file
        output: Output file.  Should end in ".html"

    Returns:
        output: Output file

    """
    assert output.endswith('.html'), 'Output HTML filename should end with .html'
    html_file = _to_html(file=_run_notebook(file), output=output)
    _postprocess_html(
        file=html_file,
        title=('CellBender: ' + os.path.basename(output).replace('_report.html', '')),
    )
    return html_file


def generate_summary_plots(input_file: str,
                           output_file: str,
                           truth_file: Optional[Dict] = None,
                           dev_mode: bool = consts.EXTENDED_REPORT):
    """Read in cellbender's output file and generate summary plots.

    Args:
        input_file: Raw CellRanger

    """

    global warnings
    warnings = []

    display(Markdown(f'### CellBender version {get_version()}'))
    display(Markdown(str(datetime.datetime.now()).split('.')[0]))
    display(Markdown(f'# {os.path.basename(output_file)}'))

    # load datasets, before and after CellBender
    input_layer_key = 'raw'
    if os.path.isdir(output_file):
        adata = _load_anndata_from_input_and_decontx(input_file=input_file,
                                                     output_mtx_directory=output_file,
                                                     input_layer_key=input_layer_key,
                                                     truth_file=truth_file)
        out_key = 'decontx'
    else:
        adata = load_anndata_from_input_and_output(input_file=input_file,
                                                   output_file=output_file,
                                                   analyzed_barcodes_only=True,
                                                   input_layer_key=input_layer_key,
                                                   truth_file=truth_file)
        out_key = 'cellbender'

    # need to make any duplicate var indices unique (for pandas manipulations)
    adata.var_names_make_unique()

    display(Markdown('## Loaded dataset'))
    print(adata)

    # bit of pre-compute
    cells = (adata.obs['cell_probability'] > consts.CELL_PROB_CUTOFF)
    adata.var['n_removed'] = adata.var[f'n_{input_layer_key}'] - adata.var[f'n_{out_key}']
    adata.var['fraction_removed'] = adata.var['n_removed'] / (adata.var[f'n_{input_layer_key}'] + 1e-5)
    adata.var['fraction_remaining'] = adata.var[f'n_{out_key}'] / (adata.var[f'n_{input_layer_key}'] + 1e-5)
    adata.var[f'n_{input_layer_key}_cells'] = np.array(adata.layers[input_layer_key][cells].sum(axis=0)).squeeze()
    adata.var[f'n_{out_key}_cells'] = np.array(adata.layers[out_key][cells].sum(axis=0)).squeeze()
    adata.var['n_removed_cells'] = (adata.var[f'n_{input_layer_key}_cells']
                                    - adata.var[f'n_{out_key}_cells'])
    adata.var['fraction_removed_cells'] = (adata.var['n_removed_cells']
                                           / (adata.var[f'n_{input_layer_key}_cells'] + 1e-5))
    adata.var['fraction_remaining_cells'] = (adata.var[f'n_{out_key}_cells']
                                             / (adata.var[f'n_{input_layer_key}_cells'] + 1e-5))

    # this inline command is necessary after cellbender imports
    plt.rcParams.update({'font.size': 12})

    # input UMI curve
    raw_full_adata = plot_input_umi_curve(input_file)

    # prove that remove-background is only subtracting counts, never adding
    if out_key == 'cellbender':
        assert (adata.layers[input_layer_key] < adata.layers[out_key]).sum() == 0, \
            "There is an entry in the output greater than the input"
    else:
        if (adata.layers[input_layer_key] < adata.layers[out_key]).sum() == 0:
            display(Markdown('WARNING: There is an entry in the output greater than the input'))

    display(Markdown('## Examine how many counts were removed in total'))
    try:
        assess_overall_count_removal(adata, raw_full_adata=raw_full_adata, out_key=out_key)
    except ValueError:
        display(Markdown('Skipping assessment over overall count removal. Presumably '
                         'this is due to including the whole dataset in '
                         '--total-droplets-included.'))

    # plot learning curve
    if out_key == 'cellbender':
        try:
            assess_learning_curve(adata)
        except Exception:
            pass
    else:
        display(Markdown('Skipping learning curve assessment.'))

    # look at per-gene count removal
    assess_count_removal_per_gene(adata, raw_full_adata=raw_full_adata, extended=dev_mode)

    if dev_mode:
        display(Markdown('## Histograms of counts per cell for several genes'))
        plot_gene_removal_histograms(adata, out_layer_key=out_key)
        display(Markdown('Typically we see that some of the low-count cells have '
                         'their counts removed, since they were background noise.'))

    # plot UMI curve and cell probabilities
    if out_key == 'cellbender':
        display(Markdown('## Cell probabilities'))
        display(Markdown('The inferred posterior probability '
                         'that each droplet is non-empty.'))
        display(Markdown('*<span style="color:gray">We sometimes write "non-empty" '
                         'instead of "cell" because dead cells and other cellular '
                         'debris can still lead to a "non-empty" droplet, which will '
                         'have a high posterior cell probability.  But these '
                         'kinds of low-quality droplets should be removed during '
                         'cell QC to retain only high-quality cells for downstream '
                         'analyses.</span>*'))
        plot_counts_and_probs_per_cell(adata)
    else:
        display(Markdown('Skipping cell probability assessment.'))

    # concordance of data before and after
    display(Markdown('## Concordance of data before and after `remove-background`'))
    plot_validation_plots(adata, output_layer_key=out_key, extended=dev_mode)

    # PCA of gene expression
    if out_key == 'cellbender':
        display(Markdown('## PCA of encoded gene expression'))
        plot_gene_expression_pca(adata, extended=dev_mode)
    else:
        display(Markdown('Skipping gene expression embedding assessment.'))

    # "mixed species" plots
    mixed_species_plots(adata, input_layer_key=input_layer_key, output_layer_key=out_key)

    if dev_mode and (truth_file is not None):

        # accuracy plots ==========================================

        display(Markdown('# Comparison with truth data'))

        display(Markdown('## Removal per gene'))

        display(Markdown('Counts per gene are summed over cell-containing droplets'))
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(adata.var['n_truth'].values, adata.var[f'n_{out_key}_cells'].values, '.', color='k')
        plt.plot([0, adata.var['n_truth'].max()], [0, adata.var['n_truth'].max()],
                 color='lightgray', alpha=0.5)
        plt.xlabel('True counts per gene')
        plt.ylabel(f'{out_key} counts per gene')
        plt.subplot(1, 2, 2)
        logic = (adata.var['n_truth'].values > 0)
        plt.plot(adata.var['n_truth'].values[logic],
                 ((adata.var[f'n_{out_key}_cells'].values[logic] - adata.var['n_truth'].values[logic])
                  / adata.var['n_truth'].values[logic]),
                 '.', color='k')
        plt.plot([0, adata.var['n_truth'].max()], [0, 0],
                 color='lightgray', alpha=0.5)
        plt.xlabel('True counts per gene')
        plt.ylabel(f'{out_key}: residual count ratio per gene\n({out_key} - truth) / truth')
        plt.tight_layout()
        plt.show()

        adata.var['fraction_remaining_cells_truth'] = (adata.var[f'n_truth']
                                                       / (adata.var[f'n_{input_layer_key}_cells'] + 1e-5))

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.semilogx(adata.var[f'n_{input_layer_key}_cells'],
                     adata.var['fraction_remaining_cells'], 'k.', ms=1)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Number of counts in raw data')
        plt.ylabel('Fraction of counts remaining')
        plt.title('Genes: removal of counts\nfrom (inferred) cell-containing droplets')
        plt.subplot(1, 2, 2)
        plt.semilogx(adata.var[f'n_{input_layer_key}_cells'],
                     adata.var['fraction_remaining_cells_truth'], 'k.', ms=1)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Number of counts in raw data')
        plt.ylabel('Truth: fraction of counts remaining')
        plt.title('Genes: truth')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.semilogx(adata.var[f'n_{input_layer_key}_cells'],
                     adata.var['fraction_remaining_cells'] - adata.var['fraction_remaining_cells_truth'],
                     'k.', ms=1)
        plt.ylim([-1.05, 1.05])
        plt.xlabel('Number of counts in raw data')
        plt.ylabel('Residual fraction of counts remaining')
        plt.title('Genes: residual')
        plt.subplot(1, 2, 2)
        plt.semilogx(adata.var[f'n_{input_layer_key}_cells'],
                     adata.var[f'n_{input_layer_key}_cells'] - adata.var['n_truth'],
                     'k.', ms=1, label='raw')
        plt.semilogx(adata.var[f'n_{input_layer_key}_cells'],
                     adata.var[f'n_{out_key}_cells'] - adata.var['n_truth'],
                     'r.', ms=1, label=f'{out_key}')
        plt.legend()
        plt.xlabel('Number of counts in raw data')
        plt.ylabel('Residual counts remaining')
        plt.title('Genes: residual')
        plt.tight_layout()
        plt.show()

        display(Markdown('## Revisiting histograms of counts per cell'))
        display(Markdown('Now showing the truth in addition to the `remove-background` '
                         'output.'))
        plot_gene_removal_histograms(adata, plot_truth=True, out_layer_key=out_key)

        # plot z, and comparisons of learned to true gene expression
        if out_key == 'cellbender':
            cluster_and_compare_expression_to_truth(adata=adata)
        else:
            display(Markdown('Skipping gene expression embedding assessment.'))

        # gene expression as images: visualize changes
        display(Markdown('## Visualization just for fun'))
        display(Markdown('This is a strange but somewhat fun way to visualize what '
                         'is going on with the data for each cell.  We look at one '
                         'cell at a time, and visualize gene expression as an '
                         'image, where pixels are ordered by their true expression '
                         'in the ambient RNA, where upper left is most expressed '
                         'in ambient.  We plot the output gene expression in '
                         'blue/yellow, and then we look at three residuals (in red):'
                         '\n\n1. (raw - truth): what was supposed to be removed'
                         '\n2. (raw - posterior): what was actually removed'
                         '\n3. (truth - posterior): the residual, '
                         'where red means too little was removed and blue means too '
                         'much was removed.'))
        try:
            show_gene_expression_before_and_after(adata=adata, num=5)
        except:
            display(Markdown('WARNING: skipped showing gene expression as images, due to an error'))

    if dev_mode:

        # inference of latents
        if out_key == 'cellbender':
            compare_latents(adata)
        else:
            display(Markdown('Skipping gene expression embedding assessment.'))

    if truth_file is not None:

        # ROC curves
        display(Markdown('## Quantification of performance'))
        display(Markdown('Here we take a look at removal of noise counts from cell '
                         'containing droplets only.'))
        true_fpr = cell_roc_count_roc(
            output_csr=adata.layers[out_key],
            input_csr=adata.layers[input_layer_key],
            truth_csr=adata.layers['truth'],
            cell_calls=(adata.obs['cell_probability'] > 0.5),
            truth_cell_labels=adata.obs['truth_cell_label'],
        )
        if type(adata.uns['target_false_positive_rate'][0]) == np.float64:
            if true_fpr > adata.uns['target_false_positive_rate'][0]:
                warnings.append('FPR exceeds target FPR.')
                display(Markdown(f'WARNING: FPR of {true_fpr:.4f} exceeds target FPR of '
                                 f'{adata.uns["target_false_positive_rate"]}. Keep '
                                 f'in mind however that the target FPR is meant to '
                                 f'target false positives over and above some '
                                 f'basal level (dataset dependent), so the '
                                 f'measured FPR should exceed the target by some '
                                 f'amount.'))

    display(Markdown('# Summary of warnings:'))
    if len(warnings) == 0:
        display(Markdown('None.'))
    else:
        for warning in warnings:
            display(Markdown(warning))


def plot_input_umi_curve(inputfile):
    adata = load_anndata_from_input(inputfile)
    plt.loglog(sorted(np.array(adata.X.sum(axis=1)).squeeze(), reverse=True))
    plt.xlabel('Ranked Barcode ID')
    plt.ylabel('UMI counts')
    plt.title(f'UMI curve\nRaw input data: {os.path.basename(inputfile)}')
    plt.show()
    return adata


def assess_overall_count_removal(adata, raw_full_adata, input_layer_key='raw', out_key='cellbender'):
    global warnings
    cells = (adata.obs['cell_probability'] > 0.5)
    initial_counts = adata.layers[input_layer_key][cells].sum()
    removed_counts = initial_counts - adata.layers[out_key][cells].sum()
    removed_percentage = removed_counts / initial_counts * 100
    print(f'removed {removed_counts:.0f} counts from non-empty droplets')
    print(f'removed {removed_percentage:.2f}% of the counts in non-empty droplets')

    from scipy.stats import norm

    log_counts = np.log10(np.array(adata.layers[input_layer_key].sum(axis=1)).squeeze())
    empty_log_counts = np.array(raw_full_adata.X[
                                    [bc not in adata.obs_names
                                     for bc in raw_full_adata.obs_names]
                                ].sum(axis=1)).squeeze()
    empty_log_counts = np.log10(empty_log_counts[empty_log_counts > consts.LOW_UMI_CUTOFF])
    bins = np.linspace(empty_log_counts.min(), log_counts.max(), 100)
    # binwidth = bins[1] - bins[0]

    # def plot_normal_fit(x, loc, scale, n, label):
    #     plt.plot(x, n * binwidth * norm.pdf(x=x, loc=loc, scale=scale), label=label)

    plt.hist(empty_log_counts.tolist() + log_counts[~cells].tolist(),
             histtype='step', label='empty droplets', bins=bins)
    plt.hist(log_counts[cells], histtype='step', label='non-empty droplets', bins=bins)
    xx = np.linspace(plt.gca().get_xlim()[0], plt.gca().get_xlim()[-1], 100)
    # if 'cell_size' in adata.obs.keys():
    #     plt.hist(np.log10(adata.obs['cell_size'][cells]),
    #              histtype='step', label='inferred cell sizes', bins=bins)
    #     plot_normal_fit(x=xx,
    #                     loc=np.log10(adata.obs['cell_size'][cells]).mean(),
    #                     scale=np.log10(adata.obs['cell_size'][cells]).std(),
    #                     n=cells.sum(),
    #                     label='inferred cell sizes')
    # if (('empty_droplet_size_lognormal_loc' in adata.uns.keys())
    #         and ('empty_droplet_size_lognormal_scale' in adata.uns.keys())):
    #     plot_normal_fit(x=xx,
    #                     loc=np.log10(np.exp(adata.uns['empty_droplet_size_lognormal_loc'])),
    #                     scale=np.log10(np.exp(adata.uns['empty_droplet_size_lognormal_scale'])),
    #                     n=(~cells).sum() + len(empty_log_counts),
    #                     label='inferred empty sizes')
    plt.ylabel('Number of droplets')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(bottom=1)
    plt.yscale('log')
    # x-axis log10 to regular number
    plt.xticks(plt.gca().get_xticks(),
               [f'{n:.0f}' for n in np.power(10, plt.gca().get_xticks())], rotation=90)
    plt.xlim(left=empty_log_counts.min())
    plt.xlabel('UMI counts per droplet')
    plt.show()

    estimated_ambient_per_droplet = np.exp(adata.uns['empty_droplet_size_lognormal_loc']).item()
    expected_fraction_removed_from_cells = estimated_ambient_per_droplet * cells.sum() / initial_counts

    fpr = adata.uns['target_false_positive_rate'].item()  # this is an np.ndarray with one element
    cohort_mode = False
    if type(fpr) != float:
        cohort_mode = True

    print('Rough estimate of expectations based on nothing but the plot above:')
    print(f'roughly {estimated_ambient_per_droplet * cells.sum():.0f} noise counts '
          f'should be in non-empty droplets')
    print(f'that is approximately {expected_fraction_removed_from_cells * 100:.2f}% of '
          f'the counts in non-empty droplets')

    if not cohort_mode:
        expected_percentage = (expected_fraction_removed_from_cells + fpr) * 100
        print(f'with a false positive rate [FPR] of {fpr * 100}%, we would expect to remove about '
              f'{expected_percentage:.2f}% of the counts in non-empty droplets')
    else:
        expected_percentage = expected_fraction_removed_from_cells * 100
        print(f'ran in cohort mode, so no false positive rate [FPR] target was set, '
              f'but we would still expect to remove about '
              f'{expected_percentage:.2f}% of the counts in non-empty droplets')

    display(Markdown('\n'))

    if np.abs(expected_percentage - removed_percentage) <= 0.5:
        display(Markdown('It looks like the algorithm did a great job meeting that expectation.'))
    elif np.abs(expected_percentage - removed_percentage) <= 1:
        display(Markdown('It looks like the algorithm did a decent job meeting that expectation.'))
    elif np.abs(expected_percentage - removed_percentage) <= 5:
        if removed_percentage < expected_percentage:
            display(Markdown('The algorithm removed a bit less than naive expectations '
                             'would indicate, but this is likely okay.  If removal '
                             'seems insufficient, the FPR can be increased.'))
        else:
            display(Markdown('The algorithm removed a bit more than naive expectations '
                             'would indicate, but this is likely okay. Spot-check '
                             'removal of a few top-removed genes as a QC measure. '
                             'If less removal is desired, decrease the FPR.'))
    elif removed_percentage - expected_percentage > 5:
        display(Markdown('The algorithm seems to have removed more overall counts '
                         'than would be naively expected.'))
        warnings.append('Algorithm removed more counts overall than naive expectations.', )
    elif expected_percentage - removed_percentage > 5:
        display(Markdown('The algorithm seems to have removed fewer overall counts '
                         'than would be naively expected.'))
        warnings.append('Algorithm removed fewer counts overall than naive expectations.', )


def assess_learning_curve(adata,
                          spike_size: float = 0.5,
                          deviation_size: float = 0.25,
                          monotonicity_cutoff: float = 0.1):
    global warnings
    display(Markdown('## Assessing convergence of the algorithm'))
    plot_learning_curve(adata)
    if 'learning_curve_train_elbo' not in adata.uns.keys():
        return
    display(Markdown(
        '*<span style="color:gray">The learning curve tells us about the progress of the algorithm in '
        'inferring all the latent variables in our model.  We want to see '
        'the ELBO increasing as training epochs increase.  Generally it is '
        'desirable for the ELBO to converge at some high plateau, and be fairly '
        'stable.</span>*'))
    display(Markdown(
        '*<span style="color:gray">What to watch out for:</span>*'
    ))
    display(Markdown(
        '*<span style="color:gray">1. large downward spikes in the ELBO (of value more than a few hundred)</span>*\n'
        '*<span style="color:gray">2. the test ELBO can be smaller than the train ELBO, but generally we '
        'want to see both curves increasing and reaching a stable plateau.  We '
        'do not want the test ELBO to dip way back down at the end.</span>*\n'
        '*<span style="color:gray">3. lack of convergence, where it looks like the ELBO would change '
        'quite a bit if training went on for more epochs.</span>*'
    ))

    if adata.uns['learning_curve_train_epoch'][-1] < 50:
        display(Markdown('Short run.  Will not analyze the learning curve.'))
        warnings.append(f'Short run of only {adata.uns["learning_curve_train_epoch"][-1]} epochs')
        return

    train_elbo_min_max = np.percentile(adata.uns['learning_curve_train_elbo'], q=[5, 95])
    train_elbo_range = train_elbo_min_max.max() - train_elbo_min_max.min()

    # look only from epoch 45 onward for spikes in train ELBO
    large_spikes_in_train = np.any((adata.uns['learning_curve_train_elbo'][46:]
                                    - adata.uns['learning_curve_train_elbo'][45:-1])
                                   < -train_elbo_range * spike_size)

    second_half_train_elbo = (adata.uns['learning_curve_train_elbo']
                              [(len(adata.uns['learning_curve_train_elbo']) // 2):])
    large_deviation_in_train = np.any(second_half_train_elbo
                                      < np.median(second_half_train_elbo)
                                      - train_elbo_range * deviation_size)

    half = len(adata.uns['learning_curve_train_elbo']) // 2
    threequarter = len(adata.uns['learning_curve_train_elbo']) * 3 // 4
    typical_end_variation = np.std(adata.uns['learning_curve_train_elbo'][half:threequarter])
    low_end_in_train = (adata.uns['learning_curve_train_elbo'][-1]
                        < adata.uns['learning_curve_train_elbo'].max() - 5 * typical_end_variation)

    # look only from epoch 45 onward for spikes in train ELBO
    non_monotonicity = ((adata.uns['learning_curve_train_elbo'][46:]
                         - adata.uns['learning_curve_train_elbo'][45:-1])
                        < -3 * typical_end_variation).sum() / len(adata.uns['learning_curve_train_elbo'])
    non_monotonic = (non_monotonicity > monotonicity_cutoff)

    def windowed_cumsum(x, n=20):
        return np.array([np.cumsum(x[i:(i + n)])[-1] for i in range(len(x) - n)])

    windowsize = 20
    tracking_trace = windowed_cumsum(adata.uns['learning_curve_train_elbo'][1:]
                                     - adata.uns['learning_curve_train_elbo'][:-1],
                                     n=windowsize)
    big_dip = -1 * (adata.uns['learning_curve_train_elbo'][-1]
                    - adata.uns['learning_curve_train_elbo'][5]) / 10
    backtracking = (tracking_trace.min() < big_dip)
    backtracking_ind = np.argmin(tracking_trace) + windowsize

    halftest = len(adata.uns['learning_curve_test_elbo']) // 2
    threequartertest = len(adata.uns['learning_curve_test_elbo']) * 3 // 4
    typical_end_variation_test = np.std(adata.uns['learning_curve_test_elbo'][halftest:threequartertest])
    runaway_test = (adata.uns['learning_curve_test_elbo'][-1]
                    < adata.uns['learning_curve_test_elbo'].max() - 4 * typical_end_variation_test)

    non_convergence = (np.mean([adata.uns['learning_curve_train_elbo'][-1]
                                - adata.uns['learning_curve_train_elbo'][-2],
                                adata.uns['learning_curve_train_elbo'][-2]
                                - adata.uns['learning_curve_train_elbo'][-3]])
                       > 2 * typical_end_variation)

    display(Markdown('**Automated assessment** --------'))
    if large_spikes_in_train:
        warnings.append('Large spikes in training ELBO.')
        display(Markdown('- *WARNING*: Large spikes detected in the training ELBO.'))
    if large_deviation_in_train:
        warnings.append('Large deviation in training ELBO from max value late in learning.')
        display(Markdown('- *WARNING*: The training ELBO deviates quite a bit from '
                         'the max value during the second half of training.'))
    if low_end_in_train:
        warnings.append('Large deviation in training ELBO from max value at end.')
        display(Markdown('- The training ELBO deviates quite a bit from '
                         'the max value at the last epoch.'))
    if non_monotonic:
        warnings.append('Non-monotonic training ELBO.')
        display(Markdown('- We typically expect to see the training ELBO increase almost '
                         'monotonically.  This curve seems to have a lot more downward '
                         'motion than we like to see.'))
    if backtracking:
        warnings.append('Back-tracking in training ELBO.')
        display(Markdown('- We typically expect to see the training ELBO increase almost '
                         'monotonically.  This curve seems to have a concerted '
                         f'period of motion in the wrong direction near epoch {backtracking_ind}. '
                         f'If this is early in training, this is probably okay.'))
    if runaway_test:
        warnings.append('Final test ELBO is much lower than the max test ELBO.')
        display(Markdown('- We hope to see the test ELBO follow the training ELBO, '
                         'increasing almost monotonically (though there will be '
                         'deviations, and that is expected).  There may be a large '
                         'gap, and that is okay.  However, this curve '
                         'ends with a low test ELBO compared to the max test ELBO '
                         'value during training. The output could be suboptimal.'))
    if non_convergence:
        warnings.append('Non-convergence of training ELBO.')
        display(Markdown('- We typically expect to see the training ELBO come to a '
                         'stable plateau value near the end of training.  Here '
                         'the training ELBO is still moving quite a bit.'))

    display(Markdown('**Summary**:'))
    if large_spikes_in_train or large_deviation_in_train:
        display(Markdown('This is unusual behavior, and a reduced --learning-rate '
                         'is indicated.  Re-run with half the current learning '
                         'rate and compare the results.'))
    elif low_end_in_train or non_monotonic or runaway_test:
        display(Markdown('This is slightly unusual behavior, and a reduced '
                         '--learning-rate might be indicated.  Consider re-running '
                         'with half the current learning rate to compare the results.'))
    elif non_convergence:
        display(Markdown('This is slightly unusual behavior, and more training '
                         '--epochs might be indicated.  Consider re-running '
                         'for more epochs to compare the results.'))
    else:
        display(Markdown('This learning curve looks normal.'))


def plot_learning_curve(adata):

    if 'learning_curve_train_elbo' not in adata.uns.keys():
        print('No learning curve recorded!')
        return

    def _mkplot():
        plt.plot(adata.uns['learning_curve_train_epoch'],
                 adata.uns['learning_curve_train_elbo'], label='train')
        try:
            plt.plot(adata.uns['learning_curve_test_epoch'],
                     adata.uns['learning_curve_test_elbo'], '.:', label='test')
            plt.legend()
        except Exception:
            pass
        plt.title('Learning curve')
        plt.ylabel('ELBO')
        plt.xlabel('Epoch')

    if len(adata.uns['learning_curve_train_elbo']) > 20:

        # two panels: zoom on the right-hand side
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        _mkplot()
        plt.subplot(1, 2, 2)
        _mkplot()
        plt.title('Learning curve (zoomed in)')
        low = np.percentile(adata.uns['learning_curve_train_elbo'], q=10)
        if (len(adata.uns['learning_curve_train_elbo']) > 0) \
                and (len(adata.uns['learning_curve_test_elbo']) > 0):
            high = max(adata.uns['learning_curve_train_elbo'].max(),
                       adata.uns['learning_curve_test_elbo'].max())
        else:
            high = adata.uns['learning_curve_train_elbo'].max()
        plt.ylim([low, high + (high - low) / 10])
        plt.tight_layout()
        plt.show()

    else:
        _mkplot()
        plt.show()


def assess_count_removal_per_gene(adata,
                                  raw_full_adata,
                                  input_layer_key='raw',
                                  r_squared_cutoff=0.5,
                                  extended=True):

    global warnings
    display(Markdown('## Examine count removal per gene'))

    # how well does it correlate with our expectation about the ambient RNA profile?
    cells = (adata.obs['cell_probability'] > 0.5)
    counts = np.array(raw_full_adata.X.sum(axis=1)).squeeze()
    clims = [adata.obs[f'n_{input_layer_key}'][~cells].mean() / 2,
             np.percentile(adata.obs[f'n_{input_layer_key}'][cells].values, q=2)]
    # if all are called "cells" then clims[0] will be a nan
    if np.isnan(clims[0]):
        clims[0] = counts.min()
    if 'approximate_ambient_profile' in adata.uns.keys():
        approximate_ambient_profile = adata.uns['approximate_ambient_profile']
    else:
        empty_count_matrix = raw_full_adata[(counts > clims[0]) & (counts < clims[1])].X
        if empty_count_matrix.shape[0] > 100:
            approximate_ambient_profile = np.array(raw_full_adata[(counts > clims[0])
                                                                  & (counts < clims[1])].X.mean(axis=0)).squeeze()
        else:
            # a very rare edge case I've seen once
            display(Markdown('Having some trouble finding the empty droplets via heuristics. '
                             'The "approximate background estimated from empty droplets" may be inaccurate.'))
            approximate_ambient_profile = np.array(raw_full_adata[counts < clims[1]].X.mean(axis=0)).squeeze()
        approximate_ambient_profile = approximate_ambient_profile / approximate_ambient_profile.sum()
    y = adata.var['n_removed'] / adata.var['n_removed'].sum()
    maxval = (approximate_ambient_profile / approximate_ambient_profile.sum()).max()

    def _plot_identity(maxval):
        plt.plot([0, maxval], [0, maxval], 'lightgray')

    if extended:

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(approximate_ambient_profile, adata.var['ambient_expression'], '.', ms=2)
        _plot_identity(maxval)
        plt.xlabel('Approximate background per gene\nestimated from empty droplets')
        plt.ylabel('Inferred ambient profile')
        plt.title('Genes: inferred ambient')

        plt.subplot(1, 2, 2)
        plt.plot(approximate_ambient_profile, y, '.', ms=2)
        _plot_identity(maxval)
        plt.xlabel('Approximate background per gene\nestimated from empty droplets')
        plt.ylabel('Removal per gene')
        plt.title('Genes: removal')
        plt.tight_layout()
        plt.show()

    else:

        plt.plot(approximate_ambient_profile, y, '.', ms=2)
        _plot_identity(maxval)
        plt.xlabel('Approximate background per gene\nestimated from empty droplets')
        plt.ylabel('Removal per gene')
        plt.title('Genes: removal')
        plt.tight_layout()
        plt.show()

    cutoff = 1e-6
    logic = np.logical_not((approximate_ambient_profile < cutoff) | (y < cutoff))
    r_squared_result = scipy.stats.pearsonr(np.log(approximate_ambient_profile[logic]),
                                            np.log(y[logic]))
    if hasattr(r_squared_result, 'statistic'):
        # scipy version 1.9.0+
        r_squared = r_squared_result.statistic
    else:
        r_squared = r_squared_result[0]
    display(Markdown(f'Pearson correlation coefficient for the above is {r_squared:.4f}'))
    if r_squared > r_squared_cutoff:
        display(Markdown('This meets expectations.'))
    else:
        warnings.append('Per-gene removal does not closely match a naive estimate '
                        'of ambient RNA from empty droplets.  Does it look like '
                        'CellBender correctly identified the empty droplets?')
        display(Markdown('WARNING: This deviates from expectations, and may '
                         'indicate that the run did not go well'))

    percentile = 90
    genecount_lowlim = int(np.percentile(adata.var[f'n_{input_layer_key}'], q=percentile))
    display(Markdown('### Table of top genes removed\n\nRanked by fraction removed, '
                     f'and excluding genes with fewer than {genecount_lowlim} '
                     f'total raw counts ({percentile}th percentile)'))
    df = adata.var[adata.var['cellbender_analyzed']]  # exclude omitted features
    df = df[[c for c in df.columns if (c != 'features_analyzed_inds')]]
    display(HTML(df[df[f'n_{input_layer_key}'] > genecount_lowlim]
                 .sort_values(by='fraction_removed', ascending=False).head(10).to_html()))

    for g in adata.var[(adata.var[f'n_{input_layer_key}_cells'] > genecount_lowlim)
                       & (adata.var['fraction_removed'] > 0.8)].index:
        warnings.append(f'Expression of gene {g} decreases quite a bit')
        display(Markdown(f'**WARNING**: The expression of the highly-expressed '
                         f'gene {g} decreases quite markedly after CellBender.  '
                         f'Check to ensure this makes sense!'))

    if extended:

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.semilogx(adata.var[f'n_{input_layer_key}'],
                     adata.var['fraction_remaining'], 'k.', ms=1)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Number of counts in raw data')
        plt.ylabel('Fraction of counts remaining')
        plt.title('Genes: removal of counts\nfrom the entire dataset')
        plt.subplot(1, 2, 2)
        plt.semilogx(adata.var[f'n_{input_layer_key}_cells'],
                     adata.var['fraction_remaining_cells'], 'k.', ms=1)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Number of counts in raw data')
        plt.ylabel('Fraction of counts remaining')
        plt.title('Genes: removal of counts\nfrom (inferred) cell-containing droplets')
        plt.show()


def plot_counts_and_probs_per_cell(adata, input_layer_key='raw'):

    limit_to_features_analyzed = True

    if limit_to_features_analyzed:
        var_logic = adata.var['cellbender_analyzed']
    else:
        var_logic = ...

    in_counts = np.array(adata.layers[input_layer_key][:, var_logic].sum(axis=1)).squeeze()
    # cellbender_counts = np.array(adata.layers['cellbender'][:, var_logic].sum(axis=1)).squeeze()
    order = np.argsort(in_counts)[::-1]
    # plt.semilogy(cellbender_counts[order], '.:', ms=3, color='lightgray', alpha=0.5, label='cellbender')
    plt.semilogy(in_counts[order], 'k-', lw=1, label=input_layer_key)
    plt.xlabel('Sorted barcode ID')
    plt.ylabel('Unique UMI counts' + ('\n(for features analyzed by CellBender)'
                                      if limit_to_features_analyzed else ''))
    plt.legend(loc='lower left', title='UMI counts')
    plt.gca().twinx()
    plt.plot(adata.obs['cell_probability'][order].values, '.', ms=2, alpha=0.2, color='red')
    plt.ylabel('Inferred cell probability', color='red')
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], color='red')
    plt.ylim([-0.05, 1.05])
    plt.show()


def plot_validation_plots(adata, input_layer_key='raw',
                          output_layer_key='cellbender',
                          extended=True):

    display(Markdown('*<span style="color:gray">The intent is to change the input '
                     'data as little as possible while achieving noise removal.  '
                     'These plots show general summary statistics about similarity '
                     'of the input and output data.  We expect to see the data '
                     'lying close to a straight line (gray).  There may be '
                     'outlier genes/features, which are often those highest-'
                     'expressed in the ambient RNA.</span>*'))
    display(Markdown('The plots here show data '
                     'for inferred cell-containing droplets, and exclude the '
                     'empty droplets.'))

    cells = (adata.obs['cell_probability'] > 0.5)

    # counts per barcode
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.loglog(adata.obs[f'n_{input_layer_key}'][cells].values,
               adata.obs[f'n_{output_layer_key}'][cells].values,
               '.', ms=3, alpha=0.8, rasterized=True)
    minmax = [adata.obs[f'n_{input_layer_key}'][cells].min() / 10,
              adata.obs[f'n_{input_layer_key}'][cells].max() * 10]
    plt.loglog(minmax, minmax, lw=1, color='gray', alpha=0.5)
    plt.xlabel('Input counts per barcode')
    plt.ylabel(f'{output_layer_key} counts per barcode')
    plt.title('Droplet count concordance\nin inferred cell-containing droplets')
    plt.axis('equal')

    # counts per gene
    plt.subplot(1, 2, 2)
    plt.loglog(adata.var[f'n_{input_layer_key}_cells'],
               adata.var[f'n_{output_layer_key}_cells'],
               '.', ms=3, alpha=0.8, rasterized=True)
    minmax = [adata.var[f'n_{input_layer_key}_cells'].min() / 10,
              adata.var[f'n_{input_layer_key}_cells'].max() * 10]
    plt.loglog(minmax, minmax, lw=1, color='gray', alpha=0.5)
    plt.xlabel('Input counts per gene')
    plt.ylabel(f'{output_layer_key} counts per gene')
    plt.title('Gene count concordance\nin inferred cell-containing droplets')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    cells = (adata.obs['cell_probability'] >= 0.5)

    if extended:

        # Fano factor per barcode
        plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 1)
        plt.loglog(fano(adata.layers[input_layer_key][cells], axis=1),
                   fano(adata.layers[output_layer_key][cells], axis=1), '.', ms=1)
        plt.loglog([1e0, 1e3], [1e0, 1e3], lw=1, color='gray', alpha=0.5)
        plt.xlabel('Input Fano factor per droplet')
        plt.ylabel(f'{output_layer_key} Fano factor per droplet')
        plt.title('Droplet count variance\nin inferred cell-containing droplets')
        plt.axis('equal')

        # Fano factor per gene
        plt.subplot(1, 2, 2)
        plt.loglog(fano(adata.layers[input_layer_key][cells], axis=0),
                   fano(adata.layers[output_layer_key][cells], axis=0), '.', ms=1)
        plt.loglog([1e0, 1e3], [1e0, 1e3], lw=1, color='gray', alpha=0.5)
        plt.xlabel('Input Fano factor per gene')
        plt.ylabel(f'{output_layer_key} Fano factor per gene')
        plt.title('Gene count variance\nin inferred cell-containing droplets')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        # histogram of per-cell cosine distances
        #     cells_in_data = latents['barcodes_analyzed_inds'][latents['cell_probability'] >= 0.5]
        cosine_dist = []
        for bc in np.random.permutation(np.where(cells)[0])[:300]:
            cosine_dist.append(cosine(np.array(adata.layers[input_layer_key][bc, :].todense()).squeeze(),
                                      np.array(adata.layers[output_layer_key][bc, :].todense()).squeeze()))
        cosine_dist = np.array(cosine_dist)
        plt.hist(cosine_dist, bins=100)
        plt.xlabel('Cosine distance between cell before and after')
        plt.ylabel('Number of cells')
        plt.title('Per-cell expression changes')
        plt.show()

        print(f'cosine_dist.mean = {cosine_dist.mean():.4f}')

        display(Markdown('We want this cosine distance to be as small as it can be '
                         '(though it cannot be zero, since we are removing counts).  '
                         'There is no specific threshold value above which we are '
                         'concerned, but typically we see values below 0.05.'))


def plot_gene_removal_histograms(adata, input_layer_key='raw', plot_truth=False, out_layer_key='cellbender'):
    order_gene = np.argsort(np.array(adata.var[f'n_{input_layer_key}']))[::-1]
    bins = np.arange(50) - 0.5

    plt.figure(figsize=(14, 6))
    for i, g_ind in enumerate([0, 5, 10, 20, 100, 1000]):
        if g_ind >= len(order_gene):
            break
        plt.subplot(2, 3, i + 1)
        plt.hist(np.array(adata.layers[input_layer_key][:, order_gene[g_ind]].todense()).squeeze(),
                 bins=bins, log=True, label=input_layer_key, histtype='step')
        plt.hist(np.array(adata.layers[out_layer_key][:, order_gene[g_ind]].todense()).squeeze(),
                 bins=bins, log=True, label=out_layer_key, alpha=0.75, histtype='step')
        if plot_truth:
            plt.hist(np.array(adata.layers['truth'][:, order_gene[g_ind]].todense()).squeeze(),
                     bins=bins, log=True, label='truth', alpha=0.5, histtype='step')
        plt.xlabel('counts per cell', fontsize=12)
        plt.title(f'{adata.var_names[order_gene[g_ind]]}: rank {g_ind} in ambient', fontsize=12)
        plt.ylabel('number of cells', fontsize=12)
        plt.legend()
    plt.tight_layout()
    plt.show()


def show_gene_expression_before_and_after(adata,
                                          input_layer_key: str = 'raw',
                                          num: int = 10):
    """Display gene expression as an image.
    Show what was removed.
    Show what should have been removed.
    Show residual.
    """

    inds = np.where(adata.obs['cell_probability'] > 0.5)[0][:num]

    # ambient sort order
    order = np.argsort(adata.var['truth_ambient_expression'])[::-1]

    for i in inds:

        raw = np.array(adata.layers[input_layer_key][i, :].todense(), dtype=float).squeeze()

        # size of images
        nz_inds = np.where(raw[order] > 0)[0]
        nrows = int(np.floor(np.sqrt(nz_inds.size)).item())
        imsize = (nrows, int(np.ceil(nz_inds.size / max(1, nrows)).item()))

        raw = np.resize(raw[order][nz_inds], imsize[0] * imsize[1]).reshape(imsize)
        overflow = (imsize[0] * imsize[1]) - nz_inds.size + 1
        raw[-1, -overflow:] = 0.
        post = np.array(adata.layers['cellbender'][i, :].todense(), dtype=float).squeeze()
        post = np.resize(post[order][nz_inds], imsize[0] * imsize[1]).reshape(imsize)
        post[-1, -overflow:] = 0.

        true_ambient = np.resize(adata.var['truth_ambient_expression'][order][nz_inds],
                                 imsize[0] * imsize[1]).reshape(imsize)
        true_ambient[-1, -overflow:] = 0.

        true = np.array(adata.layers['truth'][i, :].todense(), dtype=float).squeeze()
        true = np.resize(true[order][nz_inds], imsize[0] * imsize[1]).reshape(imsize)
        true[-1, -overflow:] = 0.
        lim = max(np.log1p(post).max(), np.log1p(true).max(), np.log1p(raw).max())

        plt.figure(figsize=(14, 3))

        #         plt.subplot(1, 4, 1)
        #         plt.imshow(np.log1p(raw), vmin=1, vmax=lim)
        #         plt.title(f'log raw [{np.expm1(lim):.1f}]')
        #         plt.xticks([])
        #         plt.yticks([])

        plt.subplot(1, 4, 1)
        plt.imshow(np.log1p(post), vmin=1, vmax=lim)
        plt.title(f'log posterior [{np.expm1(lim):.1f}]')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 4, 2)
        dat = (raw - true)
        minmax = max(-1 * dat.min(), dat.max())
        plt.imshow(dat, cmap='seismic', vmin=-1 * minmax, vmax=minmax)
        plt.title(f'raw - truth [{minmax:.1f}]')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 4, 3)
        dat = (raw - post)
        #             minmax = max(-1*dat.min(), dat.max())
        cmap = plt.get_cmap('seismic', 2 * minmax + 1)
        plt.imshow(dat, cmap=cmap, vmin=-1 * minmax, vmax=minmax)
        plt.title(f'raw - posterior [{minmax}]')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 4, 4)
        dat = (true - post)
        thisminmax = max(-1 * dat.min(), dat.max())
        minmax = max(thisminmax, minmax)
        plt.imshow(-dat, cmap='seismic', vmin=-1 * minmax, vmax=minmax)
        if minmax == dat.max():
            plt.title(f'truth - posterior [- {minmax:.1f}]')
        else:
            plt.title(f'truth - posterior [{minmax:.1f}]')
        plt.xticks([])
        plt.yticks([])

        plt.show()

        ambient_counts = raw - true
        removed_counts = raw - post
        removed_ambient = np.sum(np.minimum(removed_counts[ambient_counts > 0], ambient_counts[ambient_counts > 0]))

        print(f'Fraction of ambient counts removed: {removed_ambient / np.sum(ambient_counts):.2f}')
        print(f'Ambient counts removed: {int(removed_ambient)}')
        print(f'Remaining ambient counts: {- int(np.sum(dat[dat < 0]))}')
        print(f'Erroneously subtracted real counts: {int(np.sum(dat[dat > 0]))}')


def plot_gene_expression_pca(adata, key='cellbender_embedding',
                             input_layer_key='raw', extended=True):

    cells = (adata.obs['cell_probability'] > 0.5)
    adata.obsm['X_pca'] = pca_2d(adata.obsm[key]).detach().numpy()

    # plot z PCA colored by latent size
    sizeorder = np.argsort(adata.obs['cell_size'][cells])
    s = plt.scatter(x=adata.obsm['X_pca'][:, 0][cells][sizeorder],
                    y=adata.obsm['X_pca'][:, 1][cells][sizeorder],
                    c=np.log10(adata.obs['cell_size'][cells][sizeorder]),
                    s=2,
                    cmap='brg',
                    alpha=0.5)
    plt.title('Gene expression embedding\ncolored by log10 inferred cell size $d_n$')
    plt.xlabel('PCA 0')
    plt.ylabel('PCA 1')
    plt.xticks([])
    plt.yticks([])
    plt.gcf().colorbar(s, pad=0.1, label='log10 cell size')
    plt.show()

    display(Markdown('*<span style="color:gray">We are not looking for anything '
                     'specific in the PCA plot '
                     'of the gene expression embedding, but often we see clusters '
                     'that correspond to different cell types.  If you see only '
                     'a single large blob, then the dataset might contain only '
                     'one cell type, or perhaps there are few counts per '
                     'droplet.</span>*'))

    if extended:

        # plot z PCA colored by a few features
        def _pca_color(g: int, layer):
            outcounts = np.array(adata.layers['cellbender'][:, adata.var_names == g].todense()).squeeze()
            rawcounts = np.array(adata.layers[input_layer_key][:, adata.var_names == g].todense()).squeeze()
            # cmax = 2 * (rawcounts - outcounts)[rawcounts > 0].mean()
            cmax = np.percentile(rawcounts[rawcounts > 0], q=80)
            if layer == 'cellbender':
                counts = outcounts
            else:
                counts = rawcounts
            order = np.argsort(counts[cells])

            s = plt.scatter(x=adata.obsm['X_pca'][:, 0][cells][order],
                            y=adata.obsm['X_pca'][:, 1][cells][order],
                            c=counts[cells][order],
                            s=10,
                            vmin=0,
                            vmax=cmax,  # min(20, max(1, cmax)),
                            cmap='Oranges',
                            alpha=0.25)
            plt.title(f'{g}: {layer}')
            plt.xlabel('PCA 0')
            plt.ylabel('PCA 1')
            plt.gcf().colorbar(s, pad=0.05, label='Counts (truncated)')

        percentile = 90
        genecount_lowlim = int(np.percentile(adata.var[f'n_{input_layer_key}'], q=percentile))
        if 'feature_type' in adata.var.keys():
            feature_logic = (adata.var['feature_type'] == 'Gene Expression')
        else:
            feature_logic = True
        features = (adata.var[feature_logic
                              & (adata.var['n_cellbender'] > genecount_lowlim)]
                    .sort_values(by='fraction_removed', ascending=False)
                    .groupby('genome')
                    .head(2)
                    .index
                    .values
                    .tolist())
        if 'feature_type' in adata.var.keys():
            if (adata.var['feature_type'] != 'Gene Expression').sum() > 0:
                features.extend(adata.var[(adata.var['feature_type'] != 'Gene Expression')
                                          & (adata.var['n_cellbender'] > genecount_lowlim)]
                                .sort_values(by='fraction_removed', ascending=False)
                                .groupby('feature_type')
                                .head(2)
                                .index
                                .values
                                .tolist())

        display(Markdown('### Visualization of a few features'))
        display(Markdown('Focusing on a few top features which were removed the most.'))

        for g in features:
            plt.figure(figsize=(11, 4))
            plt.subplot(1, 2, 1)
            _pca_color(g, layer=input_layer_key)
            plt.subplot(1, 2, 2)
            _pca_color(g, layer='cellbender')
            plt.tight_layout()
            plt.show()

        display(Markdown('*<span style="color:gray">We typically see selective '
                         'removal of some genes from '
                         'particular cell types.  The genes above have been picked '
                         'randomly based on fraction_removed, and so might not be '
                         'the best genes for visualization.  These sorts of plots '
                         'can be an interesting way to visualize what '
                         '`remove-background` does.</span>*'))


def cluster_cells(adata, embedding_key='cellbender_embedding', n=2):
    """Run PCA (if not done) and use spectral clustering to label cells.
    Returns:
        cluster: np.ndarray of cluster labels as ints
    """
    from sklearn.cluster import SpectralClustering

    cells = (adata.obs['cell_probability'] > 0.5)

    if 'X_pca' not in adata.obsm.keys():
        from sklearn.decomposition import PCA
        z = adata.obsm[embedding_key]
        adata.obsm['X_pca'] = PCA(n_components=20).fit_transform(z)

    if 'truth_cell_probability' in adata.obs.keys():
        # if truth exists, use it
        n_cell_labels = np.array([k.startswith('truth_gene_expression_cell_label_')
                                  for k in adata.var.keys()]).sum()
    else:
        n_cell_labels = n
    spec = (SpectralClustering(n_clusters=n_cell_labels, random_state=0)
            .fit_predict(adata.obsm[embedding_key][cells, :]))
    cluster = np.zeros(adata.shape[0])
    cluster[cells] = spec + 1  # offset by 1 so that empty is 0
    return cluster.astype(int)


def cluster_and_compare_expression_to_truth(adata, embedding_key='cellbender_embedding'):

    cluster = cluster_cells(adata, embedding_key=embedding_key)
    adata.obs['cluster'] = cluster
    cells = (adata.obs['cell_probability'] > 0.5)

    display(Markdown('## Gene expression embedding, $z_n$'))
    display(Markdown('Here we find cluster labels *de novo* using spectral clustering, '
                     'in order to compare learned profiles with the truth.  There '
                     'is inherently an issue of non-identifiabiity, where we cannot '
                     'know the order of the true cluster labels'))

    # plot z PCA colored by spectral cluster
    plt.figure(figsize=(4, 4))
    for k in np.unique(adata.obs['cluster'][cells]):
        plt.plot(adata.obsm['X_pca'][adata.obs['cluster'] == k, 0],
                 adata.obsm['X_pca'][adata.obs['cluster'] == k, 1],
                 '.', ls='', ms=5, alpha=0.25)
    plt.title('Gene expression embedding,\ncolored by spectral clustering')
    plt.xlabel('PC 0')
    plt.ylabel('PC 1')
    plt.show()

    display(Markdown('## Agreement of per-celltype gene expression with truth'))
    display(Markdown('We can now compare the gene expression profile of cells '
                     'from the same cluster (found *de novo*) before and after '
                     'CellBender.  We expect to see the ambient profile match, '
                     'and we expect that each *de novo* cluster will match the '
                     'gene expression of one of the "truth" clusters, although '
                     'the label order may be swapped.  Here are three ways to '
                     'visualize this same data.'))

    # load truth data into a matrix
    true_chi = np.zeros((np.unique(cluster).size, adata.shape[1]))
    true_chi[0, :] = adata.var['truth_ambient_expression']
    for i in range(1, np.unique(cluster).size):
        true_chi[i, :] = adata.var[f'truth_gene_expression_cell_label_{i}']

    # figure out the learned expression profiles chi
    learned_chi = np.zeros((np.unique(cluster).size, true_chi.shape[1]))
    learned_chi[0, :] = adata.var['ambient_expression']
    for k in adata.obs['cluster'].unique():
        if k == 0:
            continue
        # get chi from mean cell expression in that cluster
        summmed_expression = np.array(adata.layers['cellbender'][adata.obs['cluster'] == k, :]
                                      .sum(axis=0)).squeeze()
        learned_chi[k, :] = summmed_expression / summmed_expression.sum()

    # compare learned expression to the underlying true expression
    def _gridplot(adata, upper_lim=1e-1, scale='linear'):
        plt.figure(figsize=(10, 9))
        n_clusters = adata.obs['cluster'].nunique()
        for k in adata.obs['cluster'].unique():
            for j in adata.obs['cluster'].unique():
                plt.subplot(n_clusters,
                            n_clusters,
                            n_clusters * k + j + 1)
                plt.plot([1e-7, 1e-1], [1e-7, 1e-1], color='black', lw=0.2)
                plt.plot(true_chi[j, :], learned_chi[k, :], '.', ls='', ms=1, alpha=0.5)
                plt.ylim([1e-6, upper_lim])
                plt.xlim([1e-6, upper_lim])
                plt.xscale(scale)
                plt.yscale(scale)
                if j == 0:
                    plt.ylabel(f'Learned {k if k > 0 else "ambient"}')
                else:
                    plt.yticks([])
                if k == n_clusters - 1:
                    plt.xlabel(f'True {j if j > 0 else "ambient"}')
                else:
                    plt.xticks([])
        plt.show()

    _gridplot(adata, scale='log')
    _gridplot(adata, upper_lim=1e-2,)

    # same in the format of cosine distance
    n_clusters = adata.obs['cluster'].nunique()
    dist = np.zeros((n_clusters, n_clusters))
    for k in adata.obs['cluster'].unique():
        for j in adata.obs['cluster'].unique():
            u, v = true_chi[j, :], learned_chi[k, :]
            # cosine distance
            dist[k, j] = cosine(u, v)

    fig = plt.figure(figsize=(5, 5))
    if dist.max() > 0.9:
        plt.imshow(dist, vmin=0, vmax=dist[dist < 0.9].max() + 0.1,
                   cmap=plt.cm.gray_r)
    else:
        plt.imshow(dist, vmin=0,
                   cmap=plt.cm.gray_r)

    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            plt.text(i - 0.3, j + 0.05, f'{dist[j, i]:.3f}', fontsize=14, color='gray')

    plt.xticks(range(n_clusters), ['BKG'] + [i + 1 for i in range(n_clusters - 1)])
    # plt.gca().xaxis.set_ticks_position('top')
    # plt.gca().xaxis.set_label_position('top')
    plt.yticks(range(n_clusters), ['BKG'] + [i + 1 for i in range(n_clusters - 1)])
    plt.ylim(plt.gca().get_xlim()[::-1])
    plt.xlabel('True')
    plt.ylabel('Inferred')
    plt.title('Cosine distance between\ntrue and inferred expression')

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax, format='%.2f')

    plt.tight_layout()
    plt.show()

    return learned_chi


def cosine(u: np.ndarray, v: np.ndarray):
    return 1 - np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)) + 1e-10)


def fano(count_matrix_csr, axis=0):
    """Calculate Fano factor for count matrix, either per gene or per cell,
    as specified by axis.
    """

    mean = np.array(count_matrix_csr.mean(axis=axis)).squeeze()
    square_mean = np.array(count_matrix_csr.power(2).mean(axis=axis)).squeeze()
    var = square_mean - np.power(mean, 2)
    return var / (mean + 1e-10)


def plot_counts_per_gene(adata):

    order_gene = np.argsort(adata.var['n_cellbender'])[::-1]
    plt.loglog(adata.var['n_cellbender'][order_gene])
    plt.loglog(adata.var['n_cellbender'][order_gene], ':', alpha=0.5)
    plt.xlabel('Sorted gene ID')
    plt.ylabel('Unique UMI counts')
    plt.show()


def compare_latents(adata, input_layer_key='raw'):
    """Compare inferred latents to truth"""

    truth_exists = ('truth_cell_probability' in adata.obs.keys())

    display(Markdown('## Inference of latent variables (for experts)'))

    if not truth_exists:
        display(Markdown('This section deals with particular details of the '
                         'latent variables in the `remove-background` model.  '
                         'This analysis is provided for interested experts, but '
                         'is not necessary for judging the success of a run.'))

    display(Markdown('### Ambient gene expression profile'))
    plt.figure(figsize=(12, 3))
    plt.plot(adata.var['ambient_expression'].values)
    plt.ylabel('Fractional expression')
    plt.xlabel('Gene')
    plt.title('Inferred ambient profile')
    plt.show()

    if truth_exists:
        plt.figure(figsize=(12, 3))
        plt.plot(adata.var['ambient_expression'].values
                 - adata.var['truth_ambient_expression'].values)
        plt.ylabel('Residual: inferred - true')
        plt.xlabel('Gene')
        plt.title('Inferred ambient profile')
        plt.show()

    display(Markdown('### Swapping fraction, rho'))

    rho = adata.uns.get('swapping_fraction_dist_params', None)
    if rho is None:
        display(Markdown('Swapping fraction not inferred.  `--model` may have '
                         'been "ambient" instead of "swapping" or "full"'))
    else:
        from scipy.stats import beta
        nbins = 100
        xx = np.linspace(0, 1, nbins)
        if truth_exists:
            plt.hist(adata.obs['truth_swapping_fraction'].values,
                     bins=xx, label='Truth', alpha=0.5)
        plt.plot(xx, (adata.shape[0] / nbins
                      * beta.pdf(xx, consts.RHO_ALPHA_PRIOR, consts.RHO_BETA_PRIOR)),
                 label='Prior', color='k')
        plt.plot(xx, (adata.shape[0] / nbins
                      * beta.pdf(xx, rho[0], rho[1])), label='Inferred', color='r')
        plt.title('Inferred distribution for swapping fraction')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Swapping fraction rho')
        plt.ylabel('Number of droplets')
        plt.show()

    if truth_exists:

        display(Markdown('### Droplet efficiency and cell size latents'))
        display(Markdown('Including empty droplets'))
        keys = ['droplet_efficiency', 'cell_size']

        plt.figure(figsize=(5 * len(keys), 4))
        for i, key in enumerate(keys):
            plt.subplot(1, len(keys), i + 1)
            otherkey = 'droplet_efficiency' if (key == 'cell_size') else 'cell_size'
            plt.scatter(adata.obs[f'truth_{key}'], adata.obs[key],
                        c=adata.obs[otherkey], s=5, cmap='coolwarm', alpha=0.5)
            cbar = plt.colorbar()
            cbar.set_label(otherkey.capitalize().replace('_', ' '))
            xx = [adata.obs[f'truth_{key}'].min(), adata.obs[f'truth_{key}'].max()]
            plt.plot(xx, xx, color='lightgray')
            plt.title(key.capitalize().replace('_', ' '))
            plt.xlabel('truth')
            plt.ylabel('cellbender')
            plt.axis('equal')
        plt.tight_layout()
        plt.show()

        display(Markdown('Cells only'))
        cells = (adata.obs['cell_probability'] > 0.5)
        plt.figure(figsize=(5 * len(keys), 4))
        for i, key in enumerate(keys):
            plt.subplot(1, len(keys), i + 1)
            for c in sorted(adata.obs['truth_cell_label'][cells].unique()):
                ctype = (adata.obs['truth_cell_label'] == c)
                plt.plot(adata.obs[f'truth_{key}'][cells & ctype],
                         adata.obs[key][cells & ctype], '.',
                         label=f'{c} ({ctype.sum()} cells)', alpha=0.5)
            xx = [adata.obs[f'truth_{key}'][cells].min(),
                  adata.obs[f'truth_{key}'][cells].max()]
            plt.plot(xx, xx, color='lightgray')
            plt.title(key.capitalize().replace('_', ' ') + ': cells')
            plt.xlabel('truth')
            plt.ylabel('cellbender')
            plt.axis('equal')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cell type')
        plt.tight_layout()
        plt.show()

    display(Markdown(r'### Droplet efficiency $\epsilon$'))
    if truth_exists:
        display(Markdown('Comparison with the truth.'))

    order = np.argsort(adata.obs[f'n_{input_layer_key}'].values)[::-1]
    plt.figure(figsize=(6, 4))
    plt.semilogy(adata.obs[f'n_{input_layer_key}'].values[order], 'k', lw=2)
    plt.ylabel('UMI counts')
    plt.xlabel('Droplet')

    plt.gca().twinx()
    if truth_exists:
        plt.plot(adata.obs['truth_droplet_efficiency'].values[order],
                 label='truth', alpha=0.5)
    plt.plot(adata.obs['droplet_efficiency'].values[order],
             'r', label='inferred', alpha=0.5)
    plt.ylabel(r'$\epsilon$', fontsize=18, color='r')
    plt.legend()
    plt.show()

    display(Markdown('Comparison with the prior.'))
    from numpy.random import gamma
    rh = consts.EPSILON_PRIOR
    delta = 0.01
    xx = np.arange(0, 2.1, step=delta)

    if truth_exists:
        cells = (adata.obs['truth_cell_probability'] == 1)
        yy, _ = np.histogram(
            gamma(adata.uns['truth_epsilon_param'],
                  scale=1. / adata.uns['truth_epsilon_param'], size=[50000]),
            bins=xx,
        )
        plt.step(xx[:-1], yy / np.sum(yy * delta), label='truth', color='r')
    else:
        cells = (adata.obs['cell_probability'] > 0.5)

    yy, _ = np.histogram(gamma(rh, scale=1. / rh, size=[50000]), bins=xx)
    plt.step(xx[:-1], yy / np.sum(yy * delta), label='prior', color='k')

    yy, _ = np.histogram(adata.obs['droplet_efficiency'][cells], bins=xx)
    plt.step(xx[:-1], yy / np.sum(yy * delta), alpha=0.5, label='posterior: cells')

    yy, _ = np.histogram(adata.obs['droplet_efficiency'][~cells], bins=xx)
    plt.step(xx[:-1], yy / np.sum(yy * delta), alpha=0.5, label='posterior: empties')

    plt.ylabel('Probability density')
    plt.xlabel(r'$\epsilon$', fontsize=18)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    display(Markdown('### Cell size $d$'))
    if truth_exists:
        display(Markdown('Comparison with the truth.'))
    cell_order = np.argsort(adata.obs[f'n_{input_layer_key}'].values[cells])[::-1]
    plt.figure(figsize=(6, 4))
    plt.semilogy(adata.obs[f'n_{input_layer_key}'].values[cells][cell_order], 'k', lw=2)
    plt.ylabel('UMI counts')
    plt.xlabel('Droplet')

    plt.gca().twinx()
    if truth_exists:
        plt.semilogy(adata.obs['truth_cell_size'].values[cells][cell_order],
                     label='truth', alpha=0.5)
    plt.semilogy(adata.obs['cell_size'].values[cells][cell_order],
                 'r', label='inferred', alpha=0.5)
    plt.ylabel('$d$', fontsize=18, color='r')
    plt.legend()
    plt.show()

    display(Markdown(r'### The combination $d \epsilon$'))
    display(Markdown('It is easier for the model to nail this down, since there '
                     'is less degeneracy.'))

    plt.figure(figsize=(6, 4))
    plt.semilogy(adata.obs[f'n_{input_layer_key}'].values[cells][cell_order], 'k', lw=2)
    plt.ylabel('UMI counts')
    plt.xlabel('Droplet')

    plt.gca().twinx()
    if truth_exists:
        plt.semilogy((adata.obs['truth_cell_size'].values[cells][cell_order]
                     * adata.obs['truth_droplet_efficiency'].values[cells][cell_order]),
                     label='truth', alpha=0.5)
    plt.semilogy((adata.obs['cell_size'].values[cells][cell_order]
                 * adata.obs['droplet_efficiency'].values[cells][cell_order]),
                 'r', label='inferred', alpha=0.5)
    plt.ylabel(r'$d \epsilon$', fontsize=18, color='r')
    plt.legend()
    plt.show()

    display(Markdown(r'### Joint distributions of $d$ and $\epsilon$'))
    if truth_exists:
        display(Markdown('In true cells'))
    else:
        display(Markdown('In inferred cells'))
    if truth_exists:
        plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 1)
        for c in sorted(adata.obs['truth_cell_label'][cells].unique()):
            ctype = (adata.obs['truth_cell_label'] == c)
            plt.semilogy(adata.obs['truth_droplet_efficiency'][ctype],
                         adata.obs['truth_cell_size'][ctype],
                         '.', alpha=0.5)
        plt.ylabel('$d$', fontsize=18)
        plt.xlabel(r'$\epsilon$', fontsize=18)
        plt.title('Truth')
        plt.subplot(1, 2, 2)
        for c in sorted(adata.obs['truth_cell_label'][cells].unique()):
            ctype = (adata.obs['truth_cell_label'] == c)
            plt.semilogy(adata.obs['droplet_efficiency'][ctype],
                         adata.obs['cell_size'][ctype],
                         '.', label=f'{c} ({ctype.sum()} cells)', alpha=0.5)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cell type')
    else:
        plt.figure(figsize=(4, 4))
        plt.semilogy(adata.obs['droplet_efficiency'][cells],
                     adata.obs['cell_size'][cells],
                     'k.', ms=1, alpha=0.5)
    plt.ylabel('$d$', fontsize=18)
    plt.xlabel(r'$\epsilon$', fontsize=18)
    plt.title('Inferred')
    plt.tight_layout()
    plt.show()


def mixed_species_plots(adata, input_layer_key='raw', output_layer_key='cellbender',
                        ngene_cutoff=5000, count_cutoff=500):
    """Make mixed species plots if it seems appropriate"""

    # find a list of genes with expression exclusive to a cell type
    if 'genome' not in adata.var.keys():
        return

    if ('feature_type' in adata.var.keys()) and ('Gene Expression' in adata.var['feature_type'].values):
        genomes = adata.var[adata.var['feature_type'] == 'Gene Expression']['genome'].unique()
    else:
        genomes = adata.var['genome'].unique()
    cells = (adata.obs['cell_probability'] > 0.5)

    if len(genomes) != 2:
        return

    display(Markdown('## "Mixed species" analysis'))
    display(Markdown('There are multiple genomes in the dataset, so we can '
                     'analyze these genes assuming they were from a "mixed '
                     'species" experiment, where we know that certain genes '
                     'come only from certain celltypes.'))

    for genome in genomes:
        if 'feature_type' in adata.var.keys():
            if 'Gene Expression' in adata.var["feature_type"].values:
                var_subset = adata.var[(adata.var["genome"] == genome)
                                       & (adata.var["feature_type"] == "Gene Expression")]
            else:
                var_subset = adata.var[(adata.var["genome"] == genome)]
        else:
            var_subset = adata.var[(adata.var["genome"] == genome)]
        print(f'Genome "{genome}" has {len(var_subset)} genes: '
              f'{", ".join(var_subset.index.values[:3])} ...')

    for i, genome1 in enumerate(genomes):
        for genome2 in genomes[(i + 1):]:
            plt.figure(figsize=(5, 5))
            for k in [input_layer_key, output_layer_key]:
                x = np.array(adata.layers[k]
                             [:, (adata.var['genome'] == genome1)].sum(axis=1)).squeeze()
                y = np.array(adata.layers[k]
                             [:, (adata.var['genome'] == genome2)].sum(axis=1)).squeeze()
                plt.plot(x + 1,
                         y + 1,
                         '.', ms=2, label=k, alpha=0.3,
                         color='k' if (k == input_layer_key) else 'r')
                x = x[cells]
                y = y[cells]
                contam_x = (y / (y + x + 1e-10))[x > (10 * y)]
                contam_y = (x / (x + y + 1e-10))[y > (10 * x)]
                contam = np.concatenate((contam_x, contam_y), axis=0)
                display(Markdown(f'Mean "contamination" per droplet in {k} data: '
                                 f'{contam.mean() * 100:.2f} %'))
                display(Markdown(f'Median "contamination" per droplet in {k} data: '
                                 f'{np.median(contam) * 100:.2f} %'))
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim(bottom=0.55)
            plt.xlim(left=0.55)
            plt.xlabel(f'{genome1} counts (+1)')
            plt.ylabel(f'{genome2} counts (+1)')
            plt.title(f'Counts per droplet: {output_layer_key}')
            lgnd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            _set_legend_dot_size(lgnd, size=10)
            plt.show()


def _set_legend_dot_size(lgnd, size: int):
    """Set dot size in a matplotlib legend.
    Different versions of matplotlib seem to need different things.
    Just give up and move on if it doesn't work.
    I know this is a bit ridiculous but so is matplotlib.
    """
    for h in lgnd.legendHandles:
        worked = False
        try:
            h._legmarker.set_markersize(size)
            worked = True
        except AttributeError:
            pass
        if worked:
            continue
        try:
            h.set_sizes([size])
            worked = True
        except AttributeError:
            pass
        if worked:
            continue
        try:
            h._sizes = [size]
            worked = True
        except AttributeError:
            pass
        if worked:
            continue
        try:
            h.set_markersize(size)
            worked = True
        except AttributeError:
            pass


def cell_roc_count_roc(output_csr: sp.csr_matrix,
                       input_csr: sp.csr_matrix,
                       truth_csr: sp.csr_matrix,
                       cell_calls: np.ndarray,
                       truth_cell_labels: np.ndarray = None) -> float:  # 0 = empty, nonzero = cell
    """Plot a ROC curve (point) for cell identification, and plot
    a second for noise count identification.

    Returns:
        fpr: Acutal false positive rate for counts in cells

    Note:
    True positive = a noise count that has been removed
    True negative = a real count that has not been removed
    False positive = a real count that has been removed
    False negative = a noise count that has not been removed

                 |  noise   |   real   |
    ------------------------------------
        removed  |    TP    |    FP    |
    -------------|----------o----------|
    not removed  |    FN    |    TN    |
    ------------------------------------

    For the cell identification case:

    True positive = a cell called a cell
    True negative = an empty called empty
    False positive = an empty called a cell
    False negative = a cell called empty

                   |   cell   |   empty  |
    --------------------------------------
    called a cell  |    TP    |    FP    |
    ---------------|----------o----------|
     called empty  |    FN    |    TN    |
    --------------------------------------

    """

    assert cell_calls.size == truth_csr.shape[0], "Cell calls must be number of barcodes."

    # find the real cells as things with counts in truth data
    cell_truth = (np.array(truth_csr.sum(axis=1)).squeeze() > 0)
    called_cell = (cell_calls != 0)

    display(Markdown('### Identification of non-empty droplets'))
    display(Markdown('TPR = true positive rate\n\nFPR = false positive rate'))

    # cell breakdown
    tpr = np.logical_and(cell_truth, called_cell).sum() / cell_truth.sum()
    fpr = np.logical_and(np.logical_not(cell_truth), called_cell).sum() / called_cell.sum()
    print(f'\nCell identification TPR = {tpr:.3f}')
    print(f'Cell identification FPR = {fpr:.3f}')

    fig = plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, 'o')
    plt.ylabel('fraction of true cells called')
    plt.xlabel('fraction of called cells that are really empty')
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.title('Cell identification')
    plt.show()

    # counts
    # just work with cells
    cell_inds = np.where(cell_truth)[0]
    real_counts = truth_csr[cell_inds]
    removed_counts = input_csr[cell_inds] - output_csr[cell_inds]
    noise_counts = input_csr[cell_inds] - truth_csr[cell_inds]

    # count breakdown
    tp = (np.array(noise_counts
                   .minimum(removed_counts)
                   .sum(axis=1)).squeeze())
    erroneously_removed_counts = removed_counts - noise_counts
    erroneously_removed_counts[erroneously_removed_counts < 0] = 0
    fp = np.array(erroneously_removed_counts.sum(axis=1)).squeeze()
    noise_remaining = noise_counts - removed_counts
    noise_remaining[noise_remaining < 0] = 0
    fn = np.array(noise_remaining.sum(axis=1)).squeeze()
    tn = np.array(real_counts.minimum(output_csr[cell_inds]).sum(axis=1)).squeeze()

    display(Markdown('Take a look at the statistics of count removal, with the '
                     'following definitions:\n\n- TP = true positive: a noise '
                     'count (correctly) removed\n- FP = false positive: a real '
                     'count (incorrectly) removed\n- TN = true negative: a real '
                     'count (correctly) not removed\n- FN = false negative: a '
                     'noise count (incorrectly) not removed'))

    print(f'max ambient counts in any matrix entry is {(input_csr - truth_csr).max()}')

    plt.figure(figsize=(8, 1))
    plt.hist(tp, bins=100)
    plt.title('TP: noise counts removed per cell')
    plt.show()

    plt.figure(figsize=(8, 1))
    plt.hist(fp, bins=100)
    plt.title('FP: real counts removed per cell')
    plt.show()

    plt.figure(figsize=(8, 1))
    plt.hist(tn, bins=100)
    plt.title('TN: real counts not removed per cell')
    plt.show()

    plt.figure(figsize=(8, 1))
    plt.hist(fn, bins=100)
    plt.title('FN: noise counts not removed per cell')
    plt.show()

    plot_roc_points(tp, fp, tn, fn, point_colors=np.array(truth_cell_labels)[cell_truth],
                    title='Background removal from cells', color='black', show=False)
    plt.show()

    display(Markdown('### Summary statistics for noise removal'))

    print(f'Cell background removal TPR = {(tp / (tp + fn)).mean():.3f}'
          f' +/- {(tp / (tp + fn)).std():.3f}')
    print(f'Cell background removal FPR = {1 - (tn / (tn + fp)).mean():.3f}'
          f' +/- {(tn / (tn + fp)).std():.3f}')

    display(Markdown('And finally one more way to quanitify the amount of noise remaining:'))

    contam_cell_raw = (np.array(noise_counts.sum(axis=1)).squeeze()
                       / np.array(real_counts.sum(axis=1)).squeeze())
    contam_cell_after = (np.array(noise_remaining.sum(axis=1)).squeeze()
                         / np.array(real_counts.sum(axis=1)).squeeze())
    print(f'Per cell contamination fraction raw = {contam_cell_raw.mean():.3f}'
          f' +/- {contam_cell_raw.std():.3f}')
    print(f'Per cell contamination fraction after = {contam_cell_after.mean():.3f}'
          f' +/- {contam_cell_after.std():.3f}')

    return 1 - (tn / (tn + fp)).mean()  # FPR


def plot_roc_points(tp: np.ndarray,
                    fp: np.ndarray,
                    tn: np.ndarray,
                    fn: np.ndarray,
                    point_colors: np.ndarray,
                    title: str = 'roc',
                    show: bool = True,
                    **kwargs):
    """Plot points (and one summary point with error bars) on a ROC curve,
    where the x-axis is FPR and the y-axis is TPR.

    Args:
        tp: True positives, multiple measurements of the same value.
        fp: False positives
        tn: True negatives
        fn: False negatives
        point_colors: Color for each point
        title: Plot title
        show: True to show plot
        kwargs: Passed to the plotter for the summary point with error bars

    """

    fig = plt.figure(figsize=(5, 5))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    sens_mean = sensitivity.mean()
    sens_stdv = sensitivity.std()
    spec_mean = specificity.mean()
    spec_stdv = specificity.std()
    # individual points
    if type(point_colors) == str:
        point_colors = np.array([point_colors] * len(sensitivity))
    else:
        assert len(point_colors) == len(sensitivity), 'point_colors is wrong length'
    unique_colors = np.unique(point_colors)
    for c in unique_colors:
        plt.plot((1 - specificity)[point_colors == c],
                 sensitivity[point_colors == c],
                 '.', ms=1, alpha=0.5, label=c)
    lgnd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    _set_legend_dot_size(lgnd, size=15)
    # midpoint
    plt.plot(1 - spec_mean, sens_mean, 'o', **kwargs)
    # errorbars
    plt.plot([1 - (spec_mean - spec_stdv),
              1 - (spec_mean + spec_stdv)],
             [sens_mean, sens_mean], **kwargs)
    plt.plot([1 - spec_mean, 1 - spec_mean],
             [sens_mean - sens_stdv,
              sens_mean + sens_stdv], **kwargs)
    plt.ylabel('sensitivity = TP / (TP + FN)\ntrue positive rate')
    plt.xlabel('1 - specificity = 1 - TN / (TN + FP)\nfalse positive rate')
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.title(title)

    if show:
        plt.show()
    else:
        return fig


def pca_2d(mat: np.ndarray) -> torch.Tensor:
    """Perform PCA using pytorch and return top 2 PCs

    Args:
        mat: matrix where rows are observations and columns are features

    Returns:
        out: matrix where rows are observations and columns are top 2 PCs
    """

    A = torch.as_tensor(mat).float()
    U, S, V = torch.pca_lowrank(A)
    return torch.matmul(A, V[:, :2])


def plot_summary(loss: Dict[str, Dict[str, np.ndarray]],
                 umi_counts: np.ndarray,
                 p: np.ndarray,
                 z: np.ndarray):
    """Output summary plot with three panels: training, cells, latent z."""

    fig = plt.figure(figsize=(6, 18))

    # Plot the train error.
    plt.subplot(3, 1, 1)
    try:
        plt.plot(loss['train']['elbo'], '.--', label='Train')

        # Plot the test error, if there was held-out test data.
        if 'test' in loss.keys():
            if len(loss['test']['epoch']) > 0:
                plt.plot(loss['test']['epoch'],
                         loss['test']['elbo'], 'o:', label='Test')
                plt.legend()

        ylim_low = max(loss['train']['elbo'][0], loss['train']['elbo'][-1] - 2000)
        try:
            ylim_high = max(max(loss['train']['elbo']), max(loss['test']['elbo']))
        except ValueError:
            ylim_high = max(loss['train']['elbo'])
        ylim_high = ylim_high + (ylim_high - ylim_low) / 20
        plt.gca().set_ylim([ylim_low, ylim_high])
    except:
        logger.warning('Error plotting the learning curve. Skipping.')
        pass

    plt.xlabel('Epoch')
    plt.ylabel('ELBO')
    plt.title('Progress of the training procedure')

    # Plot the barcodes used, along with the inferred
    # cell probabilities.
    plt.subplot(3, 1, 2)
    count_order = np.argsort(umi_counts)[::-1]
    plt.semilogy(umi_counts[count_order], color='black')
    plt.ylabel('UMI counts')
    plt.xlabel('Barcode index, sorted by UMI count')
    if p is not None:  # The case of a simple model.
        plt.gca().twinx()
        plt.plot(p[count_order], '.:', color='red', alpha=0.3, rasterized=True)
        plt.ylabel('Cell probability', color='red')
        plt.ylim([-0.05, 1.05])
        plt.title('Determination of which barcodes contain cells')
    else:
        plt.title('The subset of barcodes used for training')

    plt.subplot(3, 1, 3)
    if p is None:
        p = np.ones(z.shape[0])

    # Do PCA on the latent encoding z.
    z_pca = pca_2d(z[p >= consts.CELL_PROB_CUTOFF])

    # Plot the latent encoding via PCA.
    plt.plot(z_pca[:, 0], z_pca[:, 1],
             '.', ms=3, color='black', alpha=0.3, rasterized=True)
    plt.ylabel('PC 1')
    plt.xlabel('PC 0')
    plt.title('PCA of latent encoding of gene expression in cells')

    return fig
