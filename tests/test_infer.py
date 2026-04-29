# """Test functions in infer.py"""
#
# import pytest
# import scipy.sparse as sp
# import numpy as np
# import torch
#
# from cellbender.remove_background.data.dataprep import DataLoader
# from cellbender.remove_background.infer import BasePosterior, Posterior, \
#     binary_search, dense_to_sparse_op_torch, dense_to_sparse_op_numpy
#
# from .conftest import sparse_matrix_equal, simulated_dataset, tensors_equal
#
#
# USE_CUDA = torch.cuda.is_available()
#
#
# @pytest.mark.parametrize('cuda',
#                          [False,
#                           pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
#                                        reason='requires CUDA'))],
#                          ids=lambda b: 'cuda' if b else 'cpu')
# def test_dense_to_sparse_op_numpy(simulated_dataset, cuda):
#     """test infer.py BasePosterior.dense_to_sparse_op_numpy()"""
#
#     d = simulated_dataset
#     data_loader = DataLoader(
#         d['matrix'],
#         empty_drop_dataset=None,
#         batch_size=5,
#         fraction_empties=0.,
#         shuffle=False,
#         use_cuda=cuda,
#     )
#
#     barcodes = []
#     genes = []
#     counts = []
#     ind = 0
#
#     for data in data_loader:
#         dense_counts = data  # just make it the same!
#
#         # Convert to sparse.
#         bcs_i_chunk, genes_i, counts_i = \
#             dense_to_sparse_op_numpy(dense_counts.detach().cpu().numpy())
#
#         # Barcode index in the dataloader.
#         bcs_i = bcs_i_chunk + ind
#
#         # Obtain the real barcode index after unsorting the dataloader.
#         bcs_i = data_loader.unsort_inds(bcs_i)
#
#         # Add sparse matrix values to lists.
#         barcodes.append(bcs_i)
#         genes.append(genes_i)
#         counts.append(counts_i)
#
#         # Increment barcode index counter.
#         ind += data.shape[0]  # Same as data_loader.batch_size
#
#     # Convert the lists to numpy arrays.
#     counts = np.array(np.concatenate(tuple(counts)), dtype=np.uint32)
#     barcodes = np.array(np.concatenate(tuple(barcodes)), dtype=np.uint32)
#     genes = np.array(np.concatenate(tuple(genes)), dtype=np.uint32)  # uint16 is too small!
#
#     # Put the counts into a sparse csc_matrix.
#     out = sp.csc_matrix((counts, (barcodes, genes)),
#                         shape=d['matrix'].shape)
#
#     assert sparse_matrix_equal(out, d['matrix'])
#
#
# @pytest.mark.parametrize('cuda',
#                          [False,
#                           pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
#                                        reason='requires CUDA'))],
#                          ids=lambda b: 'cuda' if b else 'cpu')
# def test_dense_to_sparse_op_torch(simulated_dataset, cuda):
#     """test infer.py BasePosterior.dense_to_sparse_op_torch()"""
#
#     d = simulated_dataset
#     data_loader = DataLoader(
#         d['matrix'],
#         empty_drop_dataset=None,
#         batch_size=5,
#         fraction_empties=0.,
#         shuffle=False,
#         use_cuda=cuda,
#     )
#
#     barcodes = []
#     genes = []
#     counts = []
#     ind = 0
#
#     for data in data_loader:
#         dense_counts = data  # just make it the same!
#
#         # Convert to sparse.
#         bcs_i_chunk, genes_i, counts_i = \
#             dense_to_sparse_op_torch(dense_counts)
#
#         # Barcode index in the dataloader.
#         bcs_i = bcs_i_chunk + ind
#
#         # Obtain the real barcode index after unsorting the dataloader.
#         bcs_i = data_loader.unsort_inds(bcs_i)
#
#         # Add sparse matrix values to lists.
#         barcodes.append(bcs_i)
#         genes.append(genes_i)
#         counts.append(counts_i)
#
#         # Increment barcode index counter.
#         ind += data.shape[0]  # Same as data_loader.batch_size
#
#     # Convert the lists to numpy arrays.
#     counts = np.concatenate(counts).astype(np.uint32)
#     barcodes = np.concatenate(barcodes).astype(np.uint32)
#     genes = np.concatenate(genes).astype(np.uint32)  # uint16 is too small!
#
#     # Put the counts into a sparse csc_matrix.
#     out = sp.csc_matrix((counts, (barcodes, genes)),
#                         shape=d['matrix'].shape)
#
#     assert sparse_matrix_equal(out, d['matrix'])
#
#
# def test_binary_search():
#     """Test the general binary search function."""
#
#     tol = 0.001
#
#     def fun1(x):
#         return x - 1.
#
#     out = binary_search(evaluate_outcome_given_value=fun1,
#                         target_outcome=torch.tensor([0.]),
#                         init_range=torch.tensor([[0., 10.]]),
#                         target_tolerance=tol)
#     print('Single value binary search')
#     print('Target value = [1.]')
#     print(f'Output = {out}')
#     assert ((out - torch.tensor([1.])).abs() <= tol).all(), \
#         'Single input binary search failed'
#
#     def fun2(x):
#         x = x.clone()
#         x[0] = x[0] - 1.
#         x[1] = x[1] - 2.
#         return x
#
#     out = binary_search(evaluate_outcome_given_value=fun2,
#                         target_outcome=torch.tensor([0., 0.]),
#                         init_range=torch.tensor([[-10., 5.], [0., 10.]]),
#                         target_tolerance=tol)
#     print('Two-value binary search')
#     print('Target value = [1., 2.]')
#     print(f'Output = {out}')
#     assert ((out - torch.tensor([1., 2.])).abs() <= tol).all(), \
#         'Two-argument input binary search failed'
#
# @pytest.mark.parametrize('cuda',
#                          [False,
#                           pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
#                                        reason='requires CUDA'))],
#                          ids=lambda b: 'cuda' if b else 'cpu')
# @pytest.mark.parametrize('fun',
#                          [Posterior._mckp_noise_given_log_prob_tensor,
#                           Posterior._mckp_noise_given_log_prob_tensor_fast],
#                          ids=['inchworm', 'sort'])
# def test_mckp_noise_given_log_prob_tensor(cuda, fun):
#     """Test the bespoke inchworm algorithm for solving a discrete
#     convex constrained optimization problem"""
#
#     device = 'cuda' if cuda else 'cpu'
#
#     poisson_noise_means = torch.tensor([[1.9160, 0.5520],
#                                         [2.7160, 0.0840],
#                                         [3.9080, 2.5280]]).to(device)
#     log_p_ngc = (torch.distributions.Poisson(rate=poisson_noise_means.unsqueeze(-1))
#                  .log_prob(torch.arange(10).to(device)))
#
#     debug = True
#
#     print('\nA couple normal test cases')
#     out = fun(
#         log_prob_noise_counts_NGC=log_p_ngc,
#         offset_noise_counts_NG=torch.zeros_like(poisson_noise_means),
#         data_NG=torch.ones_like(poisson_noise_means) * 10.,
#         target_G=torch.tensor([1., 3.]).to(device),
#         debug=debug,
#     )
#     assert tensors_equal(out[0], torch.tensor([[0., 0.], [0., 0.], [1., 3.]]).to(device))
#     assert tensors_equal(out[1], torch.tensor([0., 0.]).to(device))
#
#     out = fun(
#         log_prob_noise_counts_NGC=log_p_ngc,
#         offset_noise_counts_NG=torch.zeros_like(poisson_noise_means),
#         data_NG=torch.ones_like(poisson_noise_means) * 10.,
#         target_G=torch.tensor([6., 6.]).to(device),
#         debug=debug,
#     )
#     assert tensors_equal(out[0], torch.tensor([[1., 1.], [2., 0.], [3., 5.]]).to(device))
#     assert tensors_equal(out[1], torch.tensor([0., 0.]).to(device))
#
#     print('\nEdge case of zero removal')
#     out = fun(
#         log_prob_noise_counts_NGC=log_p_ngc,
#         offset_noise_counts_NG=torch.zeros_like(poisson_noise_means),
#         data_NG=torch.ones_like(poisson_noise_means) * 10.,
#         target_G=torch.tensor([0., 0.]).to(device),
#         debug=debug,
#     )
#     assert tensors_equal(out[0], torch.tensor([[0., 0.], [0., 0.], [0., 0.]]).to(device))
#     assert tensors_equal(out[1], torch.tensor([0., 0.]).to(device))
#
#     print('\nEdge case of massive target')
#     out = fun(
#         log_prob_noise_counts_NGC=log_p_ngc,
#         offset_noise_counts_NG=torch.zeros_like(poisson_noise_means),
#         data_NG=torch.ones_like(poisson_noise_means) * 10.,
#         target_G=torch.tensor([100., 0.]).to(device),
#         debug=debug,
#     )
#     assert tensors_equal(out[0], torch.tensor([[9., 0.], [9., 0.], [9., 0.]]).to(device))
#     assert tensors_equal(out[1], torch.tensor([73., 0.]).to(device))
#
#     print('\nNonzero offset noise counts')
#     out = fun(
#         log_prob_noise_counts_NGC=log_p_ngc,
#         offset_noise_counts_NG=torch.ones_like(poisson_noise_means),
#         data_NG=torch.ones_like(poisson_noise_means) * 10.,
#         target_G=torch.tensor([3., 3.]).to(device),
#         debug=debug,
#     )
#     assert tensors_equal(out[0], torch.tensor([[1., 1.], [1., 1.], [1., 1.]]).to(device))
#     assert tensors_equal(out[1], torch.tensor([0., 0.]).to(device))
#
#     print('\nNonzero offset noise counts with zero target')
#     out = fun(
#         log_prob_noise_counts_NGC=log_p_ngc,
#         offset_noise_counts_NG=torch.tensor([[1., 0.], [0., 0.], [0., 0.]]).to(device),
#         data_NG=torch.ones_like(poisson_noise_means) * 10.,
#         target_G=torch.tensor([0., 0.]).to(device),
#         debug=debug,
#     )
#     assert tensors_equal(out[0], torch.tensor([[1., 0.], [0., 0.], [0., 0.]]).to(device))
#     assert tensors_equal(out[1], torch.tensor([-1., 0.]).to(device))
#
#     print('\nNonzero offset noise counts')
#     out = fun(
#         log_prob_noise_counts_NGC=log_p_ngc,
#         offset_noise_counts_NG=torch.ones_like(poisson_noise_means),
#         data_NG=torch.ones_like(poisson_noise_means) * 10.,
#         target_G=torch.tensor([6., 6.]).to(device),
#         debug=debug,
#     )
#     assert tensors_equal(out[0], torch.tensor([[1., 1.], [2., 1.], [3., 4.]]).to(device))
#     assert tensors_equal(out[1], torch.tensor([0., 0.]).to(device))
#
#     # # This cannot happen for a properly constructed log prob tensor:
#     # print('Argmax > data')
#     # out = fun(
#     #     log_prob_noise_counts_NGC=log_p_ngc,
#     #     offset_noise_counts_NG=torch.ones_like(poisson_noise_means),
#     #     data_NG=torch.ones_like(poisson_noise_means) * 1.,
#     #     target_G=torch.tensor([6., 6.]).to(device),
#     #     debug=debug,
#     # )
#     # assert tensors_equal(out[0], torch.tensor([[1., 1.], [1., 1.], [1., 1.]]).to(device))
#     # assert tensors_equal(out[1], torch.tensor([4., 4.]).to(device))
#
#     print('\nExtra genes that have no moves')
#     poisson_noise_means = torch.tensor([[1.9160, 0.5520, 0.1, 0.1],
#                                         [2.7160, 0.0840, 0.2, 0.3],
#                                         [3.9080, 2.5280, 0.3, 0.2]]).to(device)
#     data_NG = torch.tensor([[2, 0, 0, 0],
#                             [0, 0, 1, 0],
#                             [0, 0, 0, 0]]).float().to(device)
#     log_p_ngc = (torch.distributions.Poisson(rate=poisson_noise_means.unsqueeze(-1))
#                  .log_prob(torch.arange(10).to(device)))
#     log_p_ngc = torch.where(torch.arange(10).to(device).unsqueeze(0).unsqueeze(0) > data_NG.unsqueeze(-1),
#                             torch.ones_like(log_p_ngc) * -np.inf,
#                             log_p_ngc)
#     target_G = torch.tensor([2., 2., 2., 2.]).to(device)
#     print('data')
#     print(data_NG)
#     print('log_prob')
#     print(log_p_ngc)
#     out = fun(
#         log_prob_noise_counts_NGC=log_p_ngc,
#         offset_noise_counts_NG=torch.zeros_like(poisson_noise_means),
#         data_NG=data_NG,
#         target_G=target_G,
#         debug=debug,
#     )
#     assert tensors_equal(out[0], data_NG)
#     assert tensors_equal(out[1], target_G - data_NG.sum(dim=0))
#
