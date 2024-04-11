"""Test functions in estimation.py"""
import pandas as pd
import pytest
import scipy.sparse as sp
import numpy as np
import torch

from cellbender.remove_background.estimation import Mean, MAP, \
    SingleSample, ThresholdCDF, MultipleChoiceKnapsack, pandas_grouped_apply, _estimation_array_to_csr, COUNT_DATATYPE
from cellbender.remove_background.posterior import IndexConverter, \
    dense_to_sparse_op_torch, log_prob_sparse_to_dense
from cellbender.remove_background.tests.conftest import sparse_matrix_equal

from typing import Dict, Union


@pytest.fixture(scope='module')
def log_prob_coo_base() -> Dict[str, Union[sp.coo_matrix, np.ndarray, Dict[int, int]]]:
    n = -np.inf
    m = np.array(
        [[0, n, n, n, n, n, n, n, n, n],  # map 0, mean 0
         [n, 0, n, n, n, n, n, n, n, n],  # map 1, mean 1
         [n, n, 0, n, n, n, n, n, n, n],  # map 2, mean 2
         [-6, -2, np.log(1. - 2 * np.exp(-2) - np.exp(-6)), -2, n, n, n, n, n, n],
         [-2.5, -1.5, -0.5, -3, np.log(1. - np.exp([-2.5, -1.5, -0.5, -3]).sum()),
          n, n, n, n, n],
         [-0.74, -1, -2, -4, np.log(1. - np.exp([-0.74, -1, -2, -4]).sum()),
          n, n, n, n, n],
         [-1, -0.74, -2, -4, np.log(1. - np.exp([-0.74, -1, -2, -4]).sum()),
          n, n, n, n, n],
         [-2, -1, -0.74, -4, np.log(1. - np.exp([-0.74, -1, -2, -4]).sum()),
          n, n, n, n, n],
         ]
    )
    # make m sparse, i.e. zero probability entries are absent
    rows, cols, vals = dense_to_sparse_op_torch(torch.tensor(m), tensor_for_nonzeros=torch.tensor(m).exp())
    # make it a bit more difficult by having an empty row at the beginning
    rows = rows + 1
    shape = list(m.shape)
    shape[0] = shape[0] + 1
    offset_dict = dict(zip(range(1, 9), [0] * 7 + [1]))  # noise count offsets (last is 1)

    maps = np.argmax(m, axis=1)
    maps = maps + np.array([offset_dict[m] for m in offset_dict.keys()])
    cdf_logic = (torch.logcumsumexp(torch.tensor(m), dim=-1) > np.log(0.5))
    cdfs = [np.where(a)[0][0] for a in cdf_logic]
    cdfs = cdfs + np.array([offset_dict[m] for m in offset_dict.keys()])

    return {'coo': sp.coo_matrix((vals, (rows, cols)), shape=shape),
            'offsets': offset_dict,  # not all noise counts start at zero
            'maps': np.array([0] + maps.tolist()),
            'cdfs': np.array([0] + cdfs.tolist())}


@pytest.fixture(scope='module', params=['exact', 'filtered', 'unsorted'])
def log_prob_coo(request, log_prob_coo_base) \
        -> Dict[str, Union[sp.coo_matrix, np.ndarray, Dict[int, int]]]:
    """When used as an input argument, this offers up a series of dicts that
    can be used for tests"""
    if request.param == 'exact':
        return log_prob_coo_base

    elif request.param == 'filtered':
        coo = log_prob_coo_base['coo']
        logic = (coo.data >= -6)
        new_coo = sp.coo_matrix((coo.data[logic], (coo.row[logic], coo.col[logic])),
                                shape=coo.shape)
        out = {'coo': new_coo}
        out.update({k: v for k, v in log_prob_coo_base.items() if (k != 'coo')})
        return out

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


def test_mean_massive_m(log_prob_coo):
    """Sets up a posterior COO with massive m values that are > max(int32).
    Will trigger github issue #252 if a bug exists, no assertion necessary.
    """

    coo = log_prob_coo['coo']
    greater_than_max_int32 = 2200000000
    new_row = coo.row.astype(np.int64) + greater_than_max_int32
    new_shape = (coo.shape[0] + greater_than_max_int32, coo.shape[1])
    new_coo = sp.coo_matrix((coo.data, (new_row, coo.col)),
                            shape=new_shape)
    print(f'original COO shape: {coo.shape}')
    print(f'new COO shape: {new_coo.shape}')
    print(f'new row minimum value: {new_coo.row.min()}')
    print(f'new row maximum value: {new_coo.row.max()}')
    offset_dict = {k + greater_than_max_int32: v for k, v in log_prob_coo['offsets'].items()}

    # this is just a shim
    converter = IndexConverter(total_n_cells=new_coo.shape[0],
                               total_n_genes=new_coo.shape[1])

    # set up and estimate
    estimator = Mean(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=new_coo,
                                         noise_offsets=offset_dict)


@pytest.fixture(scope='module', params=['exact', 'filtered', 'unsorted'])
def mckp_log_prob_coo(request, log_prob_coo_base) \
        -> Dict[str, Union[sp.coo_matrix, np.ndarray, Dict[int, int]]]:
    """When used as an input argument, this offers up a series of dicts that
    can be used for tests.

    NOTE: separate for MCKP because we cannot include an empty 'm' because it
    throws everything off (which gene is what, etc.)
    """
    def _fix(v):
        if type(v) == dict:
            return {(k - 1): val for k, val in v.items()}
        elif type(v) == sp.coo_matrix:
            return _eliminate_row_zero(v)
        else:
            return v

    def _eliminate_row_zero(coo_: sp.coo_matrix) -> sp.coo_matrix:
        row = coo_.row - 1
        shape = list(coo_.shape)
        shape[0] = shape[0] - 1
        return sp.coo_matrix((coo_.data, (row, coo_.col)), shape=shape)

    if request.param == 'exact':
        out = log_prob_coo_base

    elif request.param == 'filtered':
        coo = log_prob_coo_base['coo']
        logic = (coo.data >= -6)
        new_coo = sp.coo_matrix((coo.data[logic], (coo.row[logic], coo.col[logic])),
                                shape=coo.shape)
        out = {'coo': new_coo}
        out.update({k: v for k, v in log_prob_coo_base.items() if (k != 'coo')})

    elif request.param == 'unsorted':
        coo = log_prob_coo_base['coo']
        order = np.random.permutation(np.arange(len(coo.data)))
        new_coo = sp.coo_matrix((coo.data[order], (coo.row[order], coo.col[order])),
                                shape=coo.shape)
        out = {'coo': new_coo}
        out.update({k: v for k, v in log_prob_coo_base.items() if (k != 'coo')})

    else:
        raise ValueError(f'Test writing error: requested "{request.param}" log_prob_coo')

    return {k: _fix(v) for k, v in out.items()}


def test_single_sample(log_prob_coo):
    """Test the single sample estimator"""

    # the input
    print(log_prob_coo)
    print('input log probs')
    dense = log_prob_sparse_to_dense(log_prob_coo['coo'])
    print(dense)

    # with this shape converter, we get one row, where each value is one m
    converter = IndexConverter(total_n_cells=1,
                               total_n_genes=log_prob_coo['coo'].shape[0])

    # set up and estimate
    estimator = SingleSample(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=log_prob_coo['coo'],
                                         noise_offsets=log_prob_coo['offsets'])

    # output
    print('dense noise count estimate, per m')
    out_per_m = np.array(noise_csr.todense()).squeeze()
    print(out_per_m)

    # test
    for i in log_prob_coo['offsets'].keys():
        print(f'testing "m" value {i}')
        allowed_vals = np.arange(dense.shape[1])[dense[i, :] > -np.inf] + log_prob_coo['offsets'][i]
        print('allowed values')
        print(allowed_vals)
        print('sample')
        print(out_per_m[i])
        assert out_per_m[i] in allowed_vals, \
            f'sample {out_per_m[i]} is not allowed for {dense[i, :]}'


def test_mean(log_prob_coo):
    """Test the mean estimator"""

    def _add_offsets_to_truth(truth: np.ndarray, offset_dict: Dict[int, int]):
        return truth + np.array([offset_dict.get(m, 0) for m in range(len(truth))])

    offset_dict = log_prob_coo['offsets']

    # the input
    print(log_prob_coo)
    print('input log probs')
    dense = log_prob_sparse_to_dense(log_prob_coo['coo'])
    print(dense)

    # with this shape converter, we get one row, where each value is one m
    converter = IndexConverter(total_n_cells=1,
                               total_n_genes=log_prob_coo['coo'].shape[0])

    # set up and estimate
    estimator = Mean(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=log_prob_coo['coo'],
                                         noise_offsets=offset_dict)

    # output
    print('dense noise count estimate, per m')
    out_per_m = np.array(noise_csr.todense()).squeeze()
    print(out_per_m)

    # truth
    brute_force = np.matmul(np.arange(dense.shape[1]), np.exp(dense).transpose())
    brute_force = _add_offsets_to_truth(truth=brute_force, offset_dict=offset_dict)
    print('truth')
    print(brute_force)

    # test
    np.testing.assert_allclose(out_per_m, brute_force)


def test_map(log_prob_coo):
    """Test the MAP estimator"""

    offset_dict = log_prob_coo['offsets']

    # the input
    print(log_prob_coo)
    print('input log probs')
    print(log_prob_sparse_to_dense(log_prob_coo['coo']))

    # with this shape converter, we get one row, where each value is one m
    converter = IndexConverter(total_n_cells=1,
                               total_n_genes=log_prob_coo['coo'].shape[0])

    # set up and estimate
    estimator = MAP(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=log_prob_coo['coo'],
                                         noise_offsets=offset_dict)

    # output
    print('dense noise count estimate, per m')
    out_per_m = np.array(noise_csr.todense()).squeeze()
    print(out_per_m)
    print('truth')
    print(log_prob_coo['maps'])

    # test
    np.testing.assert_array_equal(out_per_m, log_prob_coo['maps'])


def test_cdf(log_prob_coo):
    """Test the estimator based on CDF thresholding"""

    offset_dict = log_prob_coo['offsets']

    # the input
    print(log_prob_coo)
    print('input log probs')
    print(log_prob_sparse_to_dense(log_prob_coo['coo']))

    # with this shape converter, we get one row, where each value is one m
    converter = IndexConverter(total_n_cells=1,
                               total_n_genes=log_prob_coo['coo'].shape[0])

    # set up and estimate
    estimator = ThresholdCDF(index_converter=converter)
    noise_csr = estimator.estimate_noise(noise_log_prob_coo=log_prob_coo['coo'],
                                         noise_offsets=offset_dict,
                                         q=0.5)

    # output
    print('dense noise count estimate, per m')
    out_per_m = np.array(noise_csr.todense()).squeeze()
    print(out_per_m)
    print('truth')
    print(log_prob_coo['cdfs'])

    # test
    np.testing.assert_array_equal(out_per_m, log_prob_coo['cdfs'])


@pytest.mark.parametrize('n_chunks, parallel_compute',
                         ([1, False],
                          [2, False],
                          [2, True]), ids=['1chunk', '2chunks_1cpu', '2chunks_parallel'])
@pytest.mark.parametrize('n_cells, target, truth, truth_mat',
                         ([1, np.zeros(8), np.array([0, 1, 2, 0, 0, 0, 0, 1]), None],
                          [1, np.ones(8), np.array([0, 1, 2, 1, 1, 1, 1, 1]), None],
                          [1, np.ones(8) * 2, np.array([0, 1, 2, 2, 2, 2, 2, 2]), None],
                          [4, np.zeros(2), np.array([2, 2]), None],
                          [4, np.ones(2) * 4, np.array([4, 4]), np.array([[0, 1], [2, 2], [2, 0], [0, 1]])],
                          [4, np.ones(2) * 9, np.array([9, 9]), np.array([[0, 1], [2, 3], [4, 2], [3, 3]])]),
                         ids=['1_cell_target_0',
                              '1_cell_target_1',
                              '1_cell_target_2',
                              '4_cell_target_0',
                              '4_cell_target_4',
                              '4_cell_target_9'])
def test_mckp(mckp_log_prob_coo, n_cells, target, truth, truth_mat, n_chunks, parallel_compute):
    """Test the multiple choice knapsack problem estimator"""

    offset_dict = mckp_log_prob_coo['offsets']

    # the input
    print('input log probs ===============================================')
    print(log_prob_sparse_to_dense(mckp_log_prob_coo['coo']))

    # set up and estimate# with this shape converter, we have 1 cell with 8 genes
    converter = IndexConverter(total_n_cells=n_cells,
                               total_n_genes=mckp_log_prob_coo['coo'].shape[0] // n_cells)
    estimator = MultipleChoiceKnapsack(index_converter=converter)
    noise_csr = estimator.estimate_noise(
        noise_log_prob_coo=mckp_log_prob_coo['coo'],
        noise_offsets=offset_dict,
        noise_targets_per_gene=target,
        verbose=True,
        n_chunks=n_chunks,
        use_multiple_processes=parallel_compute,
    )

    assert noise_csr.shape == (converter.total_n_cells, converter.total_n_genes)

    # output
    print('dense noise count estimate')
    out_mat = np.array(noise_csr.todense())
    print(out_mat)
    print('noise counts per gene')
    out = out_mat.sum(axis=0)
    print(out)
    print('truth')
    print(truth)

    # test
    if truth_mat is not None:
        np.testing.assert_array_equal(out_mat, truth_mat)
    np.testing.assert_array_equal(out, truth)


def _firstval(df):
    return df['log_prob'].iat[0]


def _meanval(df):
    return df['log_prob'].mean()


@pytest.mark.parametrize('fun', (_firstval, _meanval), ids=['first_value', 'mean'])
def test_parallel_pandas_grouped_apply(fun):
    """Test that the parallel apply gives the same thing as non-parallel"""

    df = pd.DataFrame(data={'m': [0, 0, 0, 1, 1, 1, 2, 2, 2],
                            'c': [0, 1, 2] * 3,
                            'log_prob': [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    print('input data')
    print(df)

    reg = pandas_grouped_apply(
        coo=sp.coo_matrix((df['log_prob'], (df['m'], df['c'])), shape=[3, 3]),
        fun=fun,
        parallel=False,
    )
    print('normal application of groupby apply')
    print(reg)

    parallel = pandas_grouped_apply(
        coo=sp.coo_matrix((df['log_prob'], (df['m'], df['c'])), shape=[3, 3]),
        fun=fun,
        parallel=True,
    )
    print('parallel application of groupby apply')
    print(parallel)

    np.testing.assert_array_equal(reg['m'], parallel['m'])
    np.testing.assert_array_equal(reg['result'], parallel['result'])


def test_estimation_array_to_csr():

    larger_than_uint16 = 2**16 + 1

    converter = IndexConverter(total_n_cells=larger_than_uint16,
                               total_n_genes=larger_than_uint16)
    m = larger_than_uint16 + np.arange(-10, 10)
    data = np.random.rand(len(m)) * -10
    noise_offsets = None

    output_csr = _estimation_array_to_csr(index_converter=converter, data=data, m=m, noise_offsets=noise_offsets, dtype=COUNT_DATATYPE)
    
    # reimplementation here with totally permissive datatypes
    cell_and_gene_dtype = np.float64
    row, col = converter.get_ng_indices(m_inds=m)
    if noise_offsets is not None:
        data = data + np.array([noise_offsets.get(i, 0) for i in m])
    coo = sp.coo_matrix((data.astype(COUNT_DATATYPE), (row.astype(cell_and_gene_dtype), col.astype(cell_and_gene_dtype))),
                        shape=converter.matrix_shape, dtype=COUNT_DATATYPE)
    coo.sum_duplicates()
    truth_csr = coo.tocsr()

    assert sparse_matrix_equal(output_csr, truth_csr)
