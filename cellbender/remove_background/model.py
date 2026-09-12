"""Multimodal VAE model and guide for CellBender remove-background.

RemoveBackgroundPyroModel is the orchestrator.  It owns a registry of
ModalityModule instances (one per assay modality) and delegates all
modality-specific Pyro calls to them.  Global quantities shared across
modalities (cell probability y_n, ambient scaling epsilon_n) are sampled here.
Per-modality PCR-chimera swapping fractions (rho_n_{name}) are sampled inside
each ModalityModule's model_per_droplet / guide_per_droplet methods.

Variable naming convention
--------------------------
  _n   : one value per droplet   (batch dimension N)
  _f   : one value per feature   (feature dimension F)
  _nf  : per-droplet per-feature (N x F)
  _nk  : per-droplet per-latent  (N x K)  — z tensors
  _mask_n : boolean mask, one entry per droplet
"""

from numbers import Number
from typing import Dict, List, Optional, Union, cast

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.infer import config_enumerate

import cellbender.remove_background.consts as consts
from cellbender.remove_background.distributions.NullDist import NullDist
from cellbender.remove_background.exceptions import NanException
from cellbender.remove_background.modality import GeneExpressionModality, ModalityModule, calculate_lambda, calculate_mu


class RemoveBackgroundPyroModel(nn.Module):
    """Multimodal VAE orchestrator for CellBender remove-background.

    Args:
        modalities: nn.ModuleDict mapping modality name -> ModalityModule.
            Must contain at least "gene_expression".
        n_droplets: Total number of droplets in the dataset.
        empty_UMI_threshold: UMI count below which a droplet is considered
            surely empty (used for the p_n prior).
        log_counts_crossover: log(UMI) at the cell / empty transition boundary.
        p_logit_prior: Naive logit prior for cell probability.
        use_cuda: Move model to GPU if True.
        epsilon_prior: Concentration/rate parameter for the Gamma prior on
            epsilon_n (ambient RT-efficiency scaling).
        rho_alpha_prior: Alpha for the Beta prior on RNA swapping fraction rho_n.
        rho_beta_prior: Beta for the Beta prior on rho_n.
        include_rho: Whether to model RNA swapping (rho_n).
        include_empties: Whether to model empty droplets.  Always True for
            multimodal datasets; set False only for the legacy 'simple' model.
        model_type: Legacy string kept for checkpoint compatibility.
    """

    def __init__(
        self,
        modalities: nn.ModuleDict,
        n_droplets: int,
        empty_UMI_threshold: int,
        log_counts_crossover: float,
        p_logit_prior: float,
        use_cuda: bool,
        epsilon_prior: float = consts.EPSILON_PRIOR,
        include_empties: bool = True,
        model_type: str = "full",
        z_hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        self.modalities: nn.ModuleDict = modalities
        self.n_droplets = n_droplets
        self.include_empties = include_empties
        self.model_type = model_type
        self.log_counts_crossover = log_counts_crossover
        self.counts_crossover = np.exp(log_counts_crossover)
        self.z_hidden_dims: List[int] = z_hidden_dims or []

        self.loss: Dict[str, Dict[str, list]] = {
            "train": {"epoch": [], "elbo": []},
            "test": {"epoch": [], "elbo": []},
            "learning_rate": {"epoch": [], "value": []},
        }

        if use_cuda:
            self.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.use_cuda = use_cuda

        # Global scalar priors (not modality-specific).
        self.epsilon_prior = torch.tensor(epsilon_prior).to(self.device)
        self.p_logit_prior = torch.tensor(p_logit_prior).float().to(self.device)
        self.empty_UMI_threshold = torch.tensor(empty_UMI_threshold).float().to(self.device)

        # Convenience reference to GE modality (always required).
        assert "gene_expression" in modalities, "modalities must include 'gene_expression'"
        self._gene_expression = cast(GeneExpressionModality, modalities["gene_expression"])

    @property
    def z_dim(self) -> int:
        return self._gene_expression.z_dim

    @property
    def n_genes(self) -> int:
        return int(self._gene_expression.feature_indices_f.shape[0])

    @property
    def include_rho(self) -> bool:
        """True if any modality models the PCR-chimera / swapping fraction."""
        return self._gene_expression.include_rho

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _iter_modalities(self):
        """Iterate modalities.items() with ModalityModule types (not bare nn.Module)."""
        for name, mod in self.modalities.items():
            assert isinstance(mod, ModalityModule)
            yield name, mod

    def split_by_modality(self, x_nf: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split the full count matrix into per-modality sub-matrices."""
        return {name: x_nf[:, mod.feature_indices_f] for name, mod in self._iter_modalities()}

    # ------------------------------------------------------------------
    # Pyro model
    # ------------------------------------------------------------------

    def model(self, x_nf: torch.Tensor):
        """Data likelihood model (generative process).

        Args:
            x_nf: Mini-batch count matrix, shape (N, total_features).
                  Rows are droplets, columns are all features across all modalities.
        """

        # Split into per-modality tensors.
        raw_data_dict = self.split_by_modality(x_nf)
        N = x_nf.shape[0]

        # Register all decoders with Pyro.
        for name, mod in self._iter_modalities():
            pyro.module(f"decoder_{name}", mod.decoder, update_module_params=True)

        # --- Outside plate: global parameters per modality (chi_ambient, phi) ---

        global_model_params: Dict[str, dict] = {}
        for name, mod in self._iter_modalities():
            global_model_params[name] = mod.model_global()

        # --- Inside plate: per-droplet sampling ---

        # Total UMI counts used for cell/empty classification (from
        # cell-probability-contributing modalities only).
        counts_n = sum(
            (
                raw_data_dict[name].sum(dim=-1)
                for name, mod in self._iter_modalities()
                if mod.contributes_to_cell_probability
            ),
            torch.zeros(N, device=self.device),
        )

        with pyro.plate("data", N, use_cuda=self.use_cuda, device=self.device):
            # Global per-droplet latents (not modality-specific).
            epsilon_n = pyro.sample(
                "epsilon_n",
                dist.Gamma(
                    concentration=self.epsilon_prior,
                    rate=self.epsilon_prior,
                ).expand([N]),
            )

            if self.include_empties:
                p_logit_prior_n = get_p_logit_prior(
                    log_counts=counts_n.log(),
                    log_cell_prior_counts=self._gene_expression.d_cell_loc_prior,
                    surely_empty_counts=self.empty_UMI_threshold,
                    naive_p_logit_prior=self.p_logit_prior,
                )
                y_n = pyro.sample("y_n", dist.Bernoulli(logits=p_logit_prior_n))
                empty_mask_n = cast(torch.BoolTensor, y_n.detach().bool().logical_not())
            else:
                y_n = torch.ones(N, device=self.device)
                empty_mask_n = cast(torch.BoolTensor, torch.zeros(N, dtype=torch.bool, device=self.device))

            # --- z prior sampling (one site per modality) ---

            latents_dict: Dict[str, Optional[torch.Tensor]] = {}
            for name, mod in self._iter_modalities():
                latents_dict[name] = mod.model_latent(N)

            # --- Per-modality: per-droplet params, decode, observe ---

            surely_cell_mask_n: Optional[torch.Tensor] = None
            probably_empty_mask_n: Optional[torch.Tensor] = None
            probably_cell_mask_n: Optional[torch.Tensor] = None

            # Per-modality outputs collected for the return dict (used by posterior inference).
            modality_outputs: Dict[str, Dict[str, torch.Tensor]] = {}

            for name, mod in self._iter_modalities():
                x_mod_nf = raw_data_dict[name]
                phi = global_model_params[name]["phi"]
                chi_ambient_f = global_model_params[name]["chi_ambient_f"]

                d_cell_n, d_empty_n, rho_n = mod.model_per_droplet(
                    x_nf=x_mod_nf,
                    empty_mask_n=empty_mask_n,
                    epsilon_n=epsilon_n,
                    y_n=y_n,
                    phi=phi,
                )

                # Decode to fractional expression.
                decoder_input = mod.prepare_decoder_input(latents_dict)
                chi_nf = pyro.deterministic(f"chi_{name}", mod.decode(decoder_input), event_dim=1)

                # Compute mean signal and ambient rates using per-modality rho.
                mu_nf = calculate_mu(
                    epsilon_n=epsilon_n,
                    d_cell_n=d_cell_n,
                    chi_nf=chi_nf,
                    y_n=y_n if self.include_empties else None,
                    rho_n=rho_n,
                )
                lam_nf = calculate_lambda(
                    epsilon_n=epsilon_n,
                    chi_ambient_f=chi_ambient_f,
                    d_empty_n=d_empty_n,
                    y_n=y_n if self.include_empties else None,
                    d_cell_n=d_cell_n,
                    rho_n=rho_n,
                    chi_bar_f=mod.chi_bar_f,
                )
                if not self.include_empties:
                    lam_nf = torch.zeros_like(mu_nf)

                pyro.sample(
                    f"obs_{name}",
                    mod.observation_dist(mu_nf, lam_nf, phi).to_event(1),
                    obs=x_mod_nf,
                )

                modality_outputs[name] = {
                    "mu": mu_nf,
                    "lam": lam_nf,
                    "alpha": (1.0 / phi).expand_as(mu_nf),
                }

            # Compute cell/empty masks once (from GE counts).
            if self.include_empties:
                assert y_n is not None
                ge_counts_n = raw_data_dict["gene_expression"].sum(dim=-1)
                surely_cell_mask_n = (ge_counts_n >= self._gene_expression.d_cell_loc_prior.exp()).bool()
                probably_empty_mask_n = (ge_counts_n < self.counts_crossover).bool()
                probably_cell_mask_n = (ge_counts_n >= self.counts_crossover).bool()

                # Regularise logit cell probability.
                pyro.sample(
                    "p_logit_reg",
                    dist.Normal(
                        loc=self.p_logit_prior,
                        scale=consts.P_LOGIT_SCALE * torch.ones([1], device=self.device),
                    ),
                )

                # Pass the p_n posterior back into the model for soft supervision.
                p_logit_n = pyro.sample(
                    "p_passback",
                    NullDist(torch.zeros(1, device=self.device)).expand_by([N]),
                )

                with poutine.mask(mask=cast(torch.BoolTensor, probably_empty_mask_n)):
                    with poutine.scale(scale=consts.REG_SCALE_SOFT_SUPERVISION):
                        pyro.sample(
                            "obs_probably_empty_y",
                            dist.Normal(
                                loc=-1 * torch.ones_like(y_n) * consts.REG_LOGIT_MEAN,
                                scale=consts.REG_LOGIT_SOFT_SCALE,
                            ),
                            obs=p_logit_n,
                        )

                with poutine.mask(mask=cast(torch.BoolTensor, probably_cell_mask_n)):
                    with poutine.scale(scale=consts.REG_SCALE_SOFT_SUPERVISION):
                        pyro.sample(
                            "obs_probably_cell_y",
                            dist.Normal(
                                loc=torch.ones_like(y_n) * consts.REG_LOGIT_MEAN,
                                scale=consts.REG_LOGIT_SOFT_SCALE,
                            ),
                            obs=p_logit_n,
                        )

        # --- Outside plate: regularise epsilon_n mean ---

        if self.include_empties and probably_cell_mask_n is not None and probably_empty_mask_n is not None:
            if surely_cell_mask_n is not None and surely_cell_mask_n.sum() >= 2 and probably_cell_mask_n.sum() >= 2:
                epsilon_median = epsilon_n[probably_cell_mask_n].median()
                pyro.sample(
                    "epsilon_mean",
                    dist.Normal(loc=epsilon_median, scale=0.01),
                    obs=torch.ones_like(epsilon_median),
                )

            if probably_empty_mask_n.sum() >= 1:
                epsilon_median_empty = epsilon_n[probably_empty_mask_n].median()
                pyro.sample(
                    "epsilon_empty_mean",
                    dist.Normal(loc=epsilon_median_empty, scale=0.01),
                    obs=torch.ones_like(epsilon_median_empty),
                )

        return {
            "z_gex": latents_dict.get("gene_expression"),
            "chi_ambient_gex": global_model_params.get("gene_expression", {}).get("chi_ambient_f"),
            "modalities": modality_outputs,
        }

    # ------------------------------------------------------------------
    # Pyro guide
    # ------------------------------------------------------------------

    @config_enumerate(default="parallel")
    def guide(self, x_nf: torch.Tensor):
        """Variational posterior.

        Args:
            x_nf: Mini-batch count matrix, same shape as in model().
        """

        nan_check = False
        if nan_check:
            for param in pyro.get_param_store().keys():
                if torch.isnan(pyro.param(param).sum()):
                    raise NanException(param)

        raw_data_dict = self.split_by_modality(x_nf)
        N = x_nf.shape[0]

        # Register all encoders with Pyro.
        for name, mod in self._iter_modalities():
            pyro.module(f"encoder_{name}", mod.encoder, update_module_params=True)

        # --- Outside plate: global variational params per modality (phi) ---

        global_guide_params: Dict[str, dict] = {}
        for name, mod in self._iter_modalities():
            global_guide_params[name] = mod.guide_global()

        # --- Run GE encoder before entering the plate ---
        # The GE encoder output is needed for the global p_n and epsilon_n sites,
        # which must be sampled inside the plate.  Running it here avoids a
        # second forward pass inside the plate.

        x_gex_nf = raw_data_dict["gene_expression"]
        enc_gex = self._gene_expression.run_encoder(x_gex_nf)

        # --- Inside plate ---

        with pyro.plate("data", N, use_cuda=self.use_cuda, device=self.device):
            if self.include_empties:
                # p regularisation and passback (GE-only global supervision sites).
                pyro.sample("p_logit_reg", dist.Normal(loc=enc_gex["p_y"], scale=consts.P_LOGIT_SCALE))
                pyro.sample("p_passback", NullDist(enc_gex["p_y"].detach()))

                y_n = pyro.sample("y_n", dist.Bernoulli(logits=enc_gex["p_y"]))
                prob_n = enc_gex["p_y"].sigmoid().detach()
            else:
                y_n = torch.ones(N, device=self.device)
                prob_n = torch.ones(N, device=self.device)

            # Sample global epsilon_n using the GE encoder's estimate.
            epsilon_n_dist = self._gene_expression.guide_epsilon_dist(enc_gex)
            if epsilon_n_dist is not None:
                epsilon_gated = prob_n * enc_gex["epsilon"] + (1.0 - prob_n) * 1.0
                pyro.sample(
                    "epsilon_n",
                    dist.Gamma(
                        concentration=epsilon_gated * consts.EPSILON_PRIOR,
                        rate=torch.tensor(consts.EPSILON_PRIOR, device=self.device),
                    ),
                )

            # --- z_nk sampling for each modality ---

            latents_dict: Dict[str, Optional[torch.Tensor]] = {}

            # GE z is gated by y_n (only meaningful for cell-containing droplets).
            with poutine.mask(mask=cast(torch.BoolTensor, y_n.bool().detach())):
                latents_dict["gene_expression"] = self._gene_expression.guide_latent(x_gex_nf)

            # All other modalities: also gate z by y_n so that empty droplets
            # do not contribute gradients to the secondary latent spaces.
            for name, mod in self._iter_modalities():
                if name == "gene_expression":
                    continue
                enc_input = mod.prepare_encoder_input(raw_data_dict)
                with poutine.mask(mask=cast(torch.BoolTensor, y_n.bool().detach())):
                    latents_dict[name] = mod.guide_latent(enc_input)

            # --- Per-droplet variational params for each modality ---

            for name, mod in self._iter_modalities():
                x_mod_nf = raw_data_dict[name]
                mod.guide_per_droplet(x_mod_nf, prob_n)

        # Clear encoder caches to avoid stale tensors between training steps.
        for _, mod in self._iter_modalities():
            if hasattr(mod, "_enc_cache"):
                mod._enc_cache = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_p_logit_prior(
    log_counts: torch.Tensor,
    log_cell_prior_counts: Union[torch.Tensor, float],
    surely_empty_counts: Union[torch.Tensor, int],
    naive_p_logit_prior: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Compute a per-droplet logit cell-probability prior from UMI counts.

    Droplets well below the empty/cell boundary get a strongly negative logit
    (surely empty); droplets at or above the expected cell size get a strongly
    positive logit (surely cell); all others get the naive prior.
    """
    ones = torch.ones_like(log_counts)
    p_logit_prior_n = ones * naive_p_logit_prior
    p_logit_prior_n = torch.where(
        log_counts <= (torch.as_tensor(surely_empty_counts, dtype=log_counts.dtype).log() + log_cell_prior_counts) / 2,
        ones * -100.0,
        p_logit_prior_n,
    )
    p_logit_prior_n = torch.where(
        log_counts >= log_cell_prior_counts,
        ones * consts.REG_LOGIT_MEAN,
        p_logit_prior_n,
    )
    return p_logit_prior_n


def get_modality_params(*prefixes: str) -> Dict[str, np.ndarray]:
    """Collect param store entries by prefix into {modality_name: array}.

    Single prefix (e.g. "chi_ambient_"): returns the full param array per name.
    Multiple prefixes (e.g. "rho_alpha_", "rho_beta_"): stacks scalar items
    from each prefix into a 1-D array; entries are included only when ALL
    prefixes are present.
    """
    param_store = pyro.get_param_store()
    keys = set(param_store.keys())
    result: Dict[str, np.ndarray] = {}
    first = prefixes[0]
    for key in keys:
        if not key.startswith(first):
            continue
        mod_name = key[len(first) :]
        if not all(f"{p}{mod_name}" in keys for p in prefixes[1:]):
            continue
        if len(prefixes) == 1:
            result[mod_name] = to_ndarray(pyro.param(key))
        else:
            result[mod_name] = np.array([to_ndarray(pyro.param(f"{p}{mod_name}")).item() for p in prefixes])
    return result


def get_rho() -> Optional[Dict[str, np.ndarray]]:
    """Return per-modality [alpha, beta] of the learned rho posteriors, or None."""
    result = get_modality_params("rho_alpha_", "rho_beta_")
    return result if result else None


def get_chi_ambient() -> Dict[str, np.ndarray]:
    """Return per-modality ambient expression profile arrays."""
    return get_modality_params("chi_ambient_")


def get_phi() -> Dict[str, np.ndarray]:
    """Return per-modality [phi_loc, phi_scale] overdispersion params."""
    return get_modality_params("phi_loc_", "phi_scale_")


def get_param_store_key(key: str) -> Union[np.ndarray, None]:
    val = None
    if key in pyro.get_param_store():
        val = to_ndarray(pyro.param(key)).squeeze()
    return val


def to_ndarray(x: Union[Number, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a numeric value or array to a numpy array on CPU."""
    if type(x) is np.ndarray:
        return x
    elif type(x) is torch.Tensor:
        return x.detach().cpu().numpy()
    elif isinstance(x, Number):
        return np.array(x)
    else:
        raise TypeError(f"to_ndarray() received input of type {type(x)}")
