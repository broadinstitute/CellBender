"""Test functions in posterior.py"""

import pytest
import scipy.sparse as sp
import numpy as np
import torch

from cellbender.remove_background.data.dataprep import DataLoader
from cellbender.remove_background.posterior import Posterior, torch_binary_search, \
    PRmu, PRq, IndexConverter, compute_mean_target_removal_as_function
from cellbender.remove_background.sparse_utils import dense_to_sparse_op_torch, \
    log_prob_sparse_to_dense, todense_fill
from cellbender.remove_background.estimation import Mean

from typing import Dict, Union

from .conftest import sparse_matrix_equal, simulated_dataset, tensors_equal


USE_CUDA = torch.cuda.is_available()


# NOTE: issues caught
# - have a test that actually creates a posterior
# - using uint32 for barcode index in a COO caused integer overflow


@pytest.fixture(scope='module')
def log_prob_coo_base() -> Dict[str, Union[sp.coo_matrix, np.ndarray, Dict[int, int]]]:
    n = -np.inf
    m = np.array(
        [[0, n, n, n, n, n, n, n],  # map 0, mean 0
         [n, 0, n, n, n, n, n, n],  # map 1, mean 1
         [-0.3, -1.5, np.log(1. - np.exp(np.array([-0.3, -1.5])).sum())] + [n] * 5,
         [-3, -1.21, -0.7, -2, -4, np.log(1. - np.exp(np.array([-3, -1.21, -0.7, -2, -4])).sum())] + [n] * 2,
         ]
    )
    # make m sparse, i.e. zero probability entries are absent
    rows, cols, vals = dense_to_sparse_op_torch(torch.tensor(m), tensor_for_nonzeros=torch.tensor(m).exp())
    # make it a bit more difficult by having an empty row at the beginning
    rows = rows + 1
    shape = list(m.shape)
    shape[0] = shape[0] + 1
    offset_dict = dict(zip(range(1, len(m) + 1), [0] * len(m) + [1]))  # noise count offsets (last is 1)
    return {'coo': sp.coo_matrix((vals, (rows, cols)), shape=shape),
            'offsets': offset_dict}


@pytest.fixture(scope='module', params=['sorted', 'unsorted'])
def log_prob_coo(request, log_prob_coo_base) \
        -> Dict[str, Union[sp.coo_matrix, np.ndarray, Dict[int, int]]]:
    """When used as an input argument, this offers up a series of dicts that
    can be used for tests"""
    if request.param == 'sorted':
        return log_prob_coo_base

    elif request.param == 'unsorted':
        coo = log_prob_coo_base['coo']
        order = np.random.permutation(np.arange(len(coo.data)))
        new_coo = sp.coo_matrix((coo.data[order], (coo.row[order], coo.col[order])),
                                shape=coo.shape)
        out = {'coo': new_coo}
        out.update({k: v for k, v in log_prob_coo_base.items() if (k != 'coo')})
        return out

    else:
        raise ValueError(f'Test writing error: requested "{request.param}" log_prob_coo')


@pytest.mark.parametrize('alpha', [0, 1, 2], ids=lambda a: f'alpha{a}')
@pytest.mark.parametrize('n_chunks', [1, 2], ids=lambda n: f'{n}chunks')
@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
def test_PRq(log_prob_coo, alpha, n_chunks, cuda):

    target_tolerance = 0.001

    print('input log_prob matrix, densified')
    dense_input_log_prob = torch.tensor(log_prob_sparse_to_dense(log_prob_coo['coo'])).float()
    print(dense_input_log_prob)
    print('probability sums per row')
    print(torch.logsumexp(dense_input_log_prob, dim=-1).exp())
    print('(row 0 is a missing row)')

    print('input mean noise counts per row')
    counts = torch.arange(dense_input_log_prob.shape[-1]).float().unsqueeze(dim=0)
    input_means = (dense_input_log_prob.exp() * counts).sum(dim=-1)
    print(input_means)

    print('input std per row')
    input_std = (dense_input_log_prob.exp()
                 * (counts - input_means.unsqueeze(dim=-1)).pow(2)).sum(dim=-1).sqrt()
    print(input_std)

    print('\ntruth: expected means after regularization')
    truth_means_after_regularization = input_means + alpha * input_std
    print(truth_means_after_regularization)
    print('and the log')
    print(np.log(truth_means_after_regularization))

    print('testing compute_log_target_dict()')
    target_dict = PRq._compute_log_target_dict(noise_count_posterior_coo=log_prob_coo['coo'],
                                               alpha=alpha)
    print(target_dict)
    for m in target_dict.keys():
        np.testing.assert_almost_equal(target_dict[m],
                                       np.log(truth_means_after_regularization[m]))

    print('targets are correct\n\n')

    print('means after regularization')
    regularized_coo = PRq.regularize(
        noise_count_posterior_coo=log_prob_coo['coo'],
        noise_offsets=log_prob_coo['offsets'],
        alpha=alpha,
        device='cuda' if cuda else 'cpu',
        target_tolerance=target_tolerance,
        n_chunks=n_chunks,
    )
    print('regularized posterior:')
    dense_regularized_log_prob = torch.tensor(log_prob_sparse_to_dense(regularized_coo)).float()
    print(dense_regularized_log_prob)
    means_after_regularization = (dense_regularized_log_prob.exp() * counts).sum(dim=-1)
    print('means after regularization:')
    print(means_after_regularization)

    torch.testing.assert_close(
        actual=truth_means_after_regularization,
        expected=means_after_regularization,
        rtol=target_tolerance,
        atol=target_tolerance,
    )


@pytest.mark.parametrize('fpr', [0., 0.1, 1], ids=lambda a: f'fpr{a}')
@pytest.mark.parametrize('n_chunks', [1, 2], ids=lambda n: f'{n}chunks')
# @pytest.mark.parametrize('per_gene', [False, True], ids=lambda n: 'per_gene' if n else 'overall')
@pytest.mark.parametrize('per_gene', [False], ids=lambda n: 'per_gene' if n else 'overall')
@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
def test_PRmu(log_prob_coo, fpr, per_gene, n_chunks, cuda):

    target_tolerance = 0.5

    index_converter = IndexConverter(total_n_cells=log_prob_coo['coo'].shape[0], total_n_genes=1)
    print(index_converter)

    print('raw count matrix')
    count_matrix = sp.csr_matrix(np.expand_dims(np.array([0, 0, 1, 2, 5]), axis=-1))  # reflecting filled in log_prob values
    print(count_matrix)

    print('input log_prob matrix, densified')
    dense_input_log_prob = torch.tensor(log_prob_sparse_to_dense(log_prob_coo['coo'])).float()
    print(dense_input_log_prob)
    print('probability sums per row')
    print(torch.logsumexp(dense_input_log_prob, dim=-1).exp())
    print('(row 0 is a missing row)')

    estimator = Mean(index_converter=index_converter)
    mean_noise_csr = estimator.estimate_noise(
        noise_log_prob_coo=log_prob_coo['coo'],
        noise_offsets=log_prob_coo['offsets'],
        device='cuda' if cuda else 'cpu',
    )
    print(f'Mean estimator removes {mean_noise_csr.sum()} counts total')

    print('testing compute_target_removal()')
    n_cells = 4  # hard coded from the log_prob_coo
    target_fun = compute_mean_target_removal_as_function(
        noise_count_posterior_coo=log_prob_coo['coo'],
        noise_offsets=log_prob_coo['offsets'],
        raw_count_csr_for_cells=count_matrix,
        n_cells=n_cells,
        index_converter=index_converter,
        device='cuda' if cuda else 'cpu',
        per_gene=per_gene,
    )
    targets = target_fun(fpr)
    print(f'aiming to remove {targets} overall counts per cell')
    print(f'so about {targets * n_cells} counts total')

    print('means after regularization')
    regularized_coo = PRmu.regularize(
        noise_count_posterior_coo=log_prob_coo['coo'],
        noise_offsets=log_prob_coo['offsets'],
        index_converter=index_converter,
        raw_count_matrix=count_matrix,
        fpr=fpr,
        per_gene=per_gene,
        device='cuda' if cuda else 'cpu',
        target_tolerance=target_tolerance,
        n_chunks=n_chunks,
    )
    print('regularized posterior:')
    dense_regularized_log_prob = torch.tensor(log_prob_sparse_to_dense(regularized_coo)).float()
    print(dense_regularized_log_prob)

    print('MAP noise:')
    map_noise = torch.argmax(dense_regularized_log_prob, dim=-1)
    print(map_noise)

    if fpr == 0.:
        torch.testing.assert_close(
            actual=map_noise.sum().float(),
            expected=torch.tensor(mean_noise_csr.sum()).float(),
            rtol=1,
            atol=1,
        )
    elif fpr == 1.:
        torch.testing.assert_close(
            actual=map_noise.sum().float(),
            expected=torch.tensor(count_matrix.sum()).float(),
            rtol=1,
            atol=1,
        )
    else:
        assert torch.tensor(mean_noise_csr.sum()).float() - 1 <= map_noise.sum().float(), \
            'Noise estimate is less than Mean estimator'
        assert torch.tensor(count_matrix.sum()).float() >= map_noise.sum().float(), \
            'Noise estimate is greater than sum of counts... this should never happen'

    # TODO: this test is very weak... it's just hard to test it exactly...
    # TODO: passing should mean the code will run, but not that it's quantitatively accurate


@pytest.mark.skip
def test_create_posterior():
    pass


def test_index_converter():
    index_converter = IndexConverter(total_n_cells=10, total_n_genes=5)
    print(index_converter)

    # check basic conversion
    n = np.array([0, 1, 2, 3])
    g = n.copy()
    m = index_converter.get_m_indices(cell_inds=n, gene_inds=g)
    print(f'm inds are {m}')
    truth = 5 * n + g
    print(f'expected {truth}')
    np.testing.assert_equal(m, truth)

    # back and forth
    n_star, g_star = index_converter.get_ng_indices(m_inds=m)
    np.testing.assert_equal(n, n_star)
    np.testing.assert_equal(g, g_star)

    # check on input validity checking
    with pytest.raises(ValueError):
        index_converter.get_m_indices(cell_inds=np.array([-1]), gene_inds=g)
    with pytest.raises(ValueError):
        index_converter.get_m_indices(cell_inds=np.array([10]), gene_inds=g)
    with pytest.raises(ValueError):
        index_converter.get_m_indices(cell_inds=n, gene_inds=np.array([-1]))
    with pytest.raises(ValueError):
        index_converter.get_m_indices(cell_inds=n, gene_inds=np.array([5]))
    with pytest.raises(ValueError):
        index_converter.get_ng_indices(m_inds=np.array([-1]))
    with pytest.raises(ValueError):
        index_converter.get_ng_indices(m_inds=np.array([10 * 5]))


@pytest.mark.skip
def test_estimation_array_to_csr():
    Posterior._estimation_array_to_csr()
    pass


def test_torch_binary_search():
    """Test the general binary search function."""

    tol = 0.001

    def fun1(x):
        return x - 1.

    out = torch_binary_search(
        evaluate_outcome_given_value=fun1,
        target_outcome=torch.tensor([0.]),
        init_range=torch.tensor([[0., 10.]]),
        target_tolerance=tol,
    )
    print('Single value binary search')
    print('Target value = [1.]')
    print(f'Output = {out}')
    assert ((out - torch.tensor([1.])).abs() <= tol).all(), \
        'Single input binary search failed'

    def fun2(x):
        x = x.clone()
        x[0] = x[0] - 1.
        x[1] = x[1] - 2.
        return x

    out = torch_binary_search(
        evaluate_outcome_given_value=fun2,
        target_outcome=torch.tensor([0., 0.]),
        init_range=torch.tensor([[-10., 5.], [0., 10.]]),
        target_tolerance=tol,
    )
    print('Two-value binary search')
    print('Target value = [1., 2.]')
    print(f'Output = {out}')
    assert ((out - torch.tensor([1., 2.])).abs() <= tol).all(), \
        'Two-argument input binary search failed'


@pytest.mark.parametrize('fpr', [0., 0.1, 0.5, 0.75, 1], ids=lambda a: f'fpr{a}')
@pytest.mark.parametrize('per_gene', [False], ids=lambda n: 'per_gene' if n else 'overall')
@pytest.mark.parametrize('cuda',
                         [False,
                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
                                       reason='requires CUDA'))],
                         ids=lambda b: 'cuda' if b else 'cpu')
def test_compute_mean_target_removal_as_function(log_prob_coo, fpr, per_gene, cuda):
    """The target removal computation, very important for the MCKP output"""

    noise_count_posterior_coo = log_prob_coo['coo']
    noise_offsets = log_prob_coo['offsets']
    device = 'cuda' if cuda else 'cpu'

    print('log prob posterior coo')
    print(noise_count_posterior_coo)

    index_converter = IndexConverter(total_n_cells=log_prob_coo['coo'].shape[0],
                                     total_n_genes=1)
    print(index_converter)

    print('raw count matrix')
    count_matrix = sp.csr_matrix(
        np.expand_dims(np.array([0, 0, 1, 2, 5]), axis=-1)
    )  # reflecting filled in log_prob values
    print(count_matrix)

    n_cells = log_prob_coo['coo'].shape[0]  # hard coded from the log_prob_coo

    target_fun = compute_mean_target_removal_as_function(
        noise_count_posterior_coo=noise_count_posterior_coo,
        noise_offsets=noise_offsets,
        index_converter=index_converter,
        raw_count_csr_for_cells=count_matrix,
        n_cells=n_cells,
        device=device,
        per_gene=per_gene,
    )

    target = (target_fun(fpr) * n_cells).item()
    print(f'\nwith fpr={fpr:.2f}, target is: {target:.1g}')

    assert target >= 1, 'There is one noise count guaranteed from this test posterior'
    if fpr == 1:
        torch.testing.assert_close(target, float(count_matrix.sum()))

    # assert False
    # TODO: this has not been tested out
