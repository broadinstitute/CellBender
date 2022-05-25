"""Gaussian mixture model for a prior on cell sizes."""

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import ClippedAdam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
import cellbender.remove_background.consts as consts
import scipy.stats
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # This needs to be after matplotlib.use('Agg')

from typing import Dict, Union


class GMM(nn.Module):
    """Bayesian mixture of 1D Gaussians, for a dataset that fits in memory."""

    def __init__(self,
                 data: torch.Tensor,
                 n_components: int = 5,
                 alpha_prior: float = 1e-3,
                 use_cuda: bool = False,
                 verbose: bool = False):
        super(GMM, self).__init__()

        self.verbose = verbose
        self.K = n_components
        self.alpha_prior = alpha_prior

        # Keep track of dataset
        self.data = data
        self.data_mean = data.mean()
        self.data_std = data.std()

        # Set up optimizer and loss function
        self.optim = ClippedAdam({'lr': 0.05, 'betas': [0.8, 0.99]})
        self.elbo = TraceEnum_ELBO(max_plate_nesting=1)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.device = 'cuda'
            self.cuda()
        else:
            self.device = 'cpu'
        self.seed = consts.RANDOM_SEED

        # Bit of pre-calculating
        self.mode = self.data.cpu().mode()[0].item()
        self.cell_maximum = self.data.max().cpu().item()
        cell_minimum = (self.data[self.data > np.mean([self.mode, self.cell_maximum])]
                        .cpu().mode()[0].item())
        self.cell_mode = self.data[self.data > cell_minimum].cpu().mode()[0].item()
        self.cell_minimum = np.mean([self.mode, self.cell_mode]).item()
        self.cell_minimum = max(self.mode + 0.5, self.cell_minimum)

        # Initialize
        self.guide = None
        self.svi = None
        self.clean_slate()
        if self.verbose:
            print('.')

    def init_loc_fn(self, site):
        if site["name"] == "weight":
            return torch.tensor([0.9] + [0.1 / (self.K - 1)] * (self.K - 1)).to(self.device)
        if site["name"] == "scale":
            return torch.ones(self.K).to(self.device)  # uniform
        if site["name"] == "loc":
            evenly_spaced_list = np.linspace(self.cell_minimum, self.cell_maximum,
                                             num=self.K - 1).tolist()
            locs = torch.tensor([self.mode] + evenly_spaced_list).to(self.device)
            return locs
        raise ValueError(site["name"])

    def initialize(self):
        pyro.clear_param_store()
        self.guide = AutoDelta(poutine.block(self.model, expose=['weight', 'loc', 'scale']),
                               init_loc_fn=self.init_loc_fn)
        svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        return self.guide, svi

    def clean_slate(self):
        pyro.set_rng_seed(self.seed)
        self.guide, self.svi = self.initialize()

    @config_enumerate
    def model(self, x):

        # Mixture weights
        alpha_k = torch.tensor([self.alpha_prior] * self.K).to(self.device)
        weights = pyro.sample('weight', dist.Dirichlet(alpha_k))

        # Mixture components
        with pyro.plate('components', self.K):
            locs = pyro.sample('loc', dist.Normal(self.data_mean, self.data_std))
            scale = pyro.sample('scale', dist.LogNormal(-torch.ones(1).to(self.device), 0.5))

        # Data
        with pyro.plate('data', len(self.data)):
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            pyro.sample('obs', dist.Normal(locs[assignment], scale[assignment]), obs=x)

    def train(self, epochs=500, start_from_scratch=True):

        if start_from_scratch:
            self.clean_slate()

        for i in range(epochs):
            loss = self.svi.step(self.data)
            if self.verbose:
                print('.' if ((i + 1) % 100) else '\n', end='')

    def map_estimate(self,
                     sort_by: str = 'weight',
                     ascending: bool = False,
                     exclude_weights_below: float = 1e-2) -> Dict[str, np.ndarray]:
        map_est = self.guide(self.data)
        weights_numpy = map_est['weight'].detach().cpu().numpy().copy()
        for key in ['loc', 'scale', 'weight']:
            map_est[key] = map_est[key].detach().cpu().numpy()
            map_est[key] = map_est[key][weights_numpy >= exclude_weights_below]
        order = np.argsort(map_est[sort_by])
        if not ascending:
            order = order[::-1]
        for key, value in map_est.items():
            map_est[key] = value[order]
        return self._filter_map_estimate(map_est)

    def _filter_map_estimate(self, map_est) -> Dict[str, np.ndarray]:
        """Ensure there is no small 'cell' mode that's really part of the empties"""

        # empty_loc = map_est['loc'][0]
        # empty_scale = map_est['scale'][0]
        empty_component_ind = np.argmin(map_est['loc'])
        empty_loc = map_est['loc'][empty_component_ind]
        empty_scale = map_est['scale'][empty_component_ind]

        three_stdev = empty_loc + 3 * empty_scale
        minimum_cutoff = max(three_stdev, self.cell_minimum)
        max_cell_loc_ind = np.argmax(map_est['loc'])

        # Figure out which components to keep based on being 3 stdev above empties.
        keep_inds = []
        for i, loc in enumerate(map_est['loc']):
            if (i == empty_component_ind) or (i == max_cell_loc_ind) or (loc > minimum_cutoff):
                keep_inds.append(i)

        # Filter out modes that overlap empties.
        for key, value in map_est.items():
            map_est[key] = value[keep_inds]

        return map_est

    def renormalize_weights(self, weights: Union[torch.Tensor, np.ndarray]):

        # Renormalize weights so that they are normalized over cells only.
        weights[weights < weights.max()] = \
            (weights[weights < weights.max()]
             / weights[weights < weights.max()].sum())

    def plot_summary(self, exclude_weights_below: float = 1e-2):

        # Get MAP estimates for parameters
        map_estimates = self.map_estimate(sort_by='loc',
                                          ascending=True,
                                          exclude_weights_below=exclude_weights_below)
        weights = map_estimates['weight']
        locs = map_estimates['loc']
        scales = map_estimates['scale']

        x = np.linspace(self.data.min().item(), self.data.max().item(), num=100)
        ys = []
        ws = []
        for loc, scale, w in zip(locs, scales, weights):
            if w.item() < exclude_weights_below:
                # print(f'skipping mixture component with weight {w:.3g}')
                continue
            pdf = scipy.stats.norm.pdf((x - loc.item()) / scale.item())
            normalization = pdf.sum() * (x[1] - x[0])
            ys.append(w.item() * pdf / normalization)
            ws.append(w)

        fig = plt.figure()
        hist, _ = np.histogram(self.data.data.cpu().numpy(), density=True, bins=x)
        plt.bar(x[:-1], hist, width=x[1] - x[0], color='lightgray')
        summed = np.zeros(ys[0].shape)

        # Renormalize cell weights for their legend labels.
        ws = np.array(ws)
        self.renormalize_weights(ws)
        for i, (y, w) in enumerate(zip(ys, ws)):
            plt.plot(x, y, '-', lw=2, label='Empties' if (i == 0) else f'Cell comp. {i}: {w:.2f}')
            summed = summed + y
        plt.plot(x, summed, 'k--', alpha=0.75, label='Sum')
        plt.legend()
        plt.title('Prior on counts: mixture model')
        plt.ylabel('Probability density')
        plt.xlabel('UMI counts per droplet')
        xlims = plt.gca().get_xlim()
        plt.xticks(ticks=np.linspace(xlims[0], xlims[1], num=10),
                   labels=[int(x) for x in np.exp(np.linspace(xlims[0], xlims[1], num=10))])
        return fig
