"""Modality-specific modules for the CellBender multimodal VAE.

Each ModalityModule owns its encoder, decoder, and all Pyro parameters and
sample sites for one assay modality (gene expression, CRISPR guide capture,
ATAC, protein, etc.).  The orchestrator (RemoveBackgroundPyroModel) loops over
a registry of ModalityModule instances to build the full model and guide.

Adding a new modality:
  1. Subclass ModalityModule and decorate with @register_modality("my_name").
  2. Implement all abstract methods.
  3. Register at model init time by including its feature indices in modality_spec.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, cast

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from torch.distributions import constraints

import cellbender.remove_background.consts as consts
from cellbender.remove_background.distributions.NegativeBinomialPoissonConvApprox import (
    NegativeBinomialPoissonConvApprox as NBPCapprox,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODALITY_REGISTRY: Dict[str, type] = {}


def register_modality(name: str):
    """Class decorator that registers a ModalityModule subclass by name."""

    def decorator(cls):
        MODALITY_REGISTRY[name] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# ModalityPriors
# ---------------------------------------------------------------------------


@dataclass
class ModalityPriors:
    """All prior values for a single modality.

    The orchestrator constructs one ModalityPriors instance per modality by
    reading modality-specific keys from dataset_obj_priors, then passes it to
    the ModalityModule constructor.

    Naming convention for dataset_obj_priors keys:
        d_cell_gex_loc_prior, d_cell_gex_scale_prior
        d_empty_gex_loc_prior, d_empty_gex_scale_prior
        chi_ambient_gex, chi_bar_gex
        phi_gex_loc_prior, phi_gex_scale_prior
        (analogous suffixes for other modalities)
    """

    d_cell_loc_prior: float
    d_cell_scale_prior: float
    d_empty_loc_prior: float
    d_empty_scale_prior: float
    chi_ambient_init: torch.Tensor
    phi_loc_prior: float = consts.PHI_LOC_PRIOR
    phi_scale_prior: float = consts.PHI_SCALE_PRIOR
    # Mean cell expression profile used in the RNA-swapping model.
    chi_bar: Optional[torch.Tensor] = None
    # Per-modality RNA swapping / PCR-chimera fraction.
    include_rho: bool = False
    rho_alpha_prior: float = consts.RHO_ALPHA_PRIOR
    rho_beta_prior: float = consts.RHO_BETA_PRIOR


# ---------------------------------------------------------------------------
# ModalityModule abstract base class
# ---------------------------------------------------------------------------


class ModalityModule(nn.Module, ABC):
    """Abstract base class for one assay modality.

    Subclasses contain the encoder, decoder, and all Pyro sample/param calls
    for their modality.  The orchestrator calls the methods in a defined order:

    In model():
      Outside plate: model_global() -> dict with 'phi', 'chi_ambient_f'
      Inside  plate: model_latent(N) -> z_nk
                     model_per_droplet(...) -> (d_cell_n, d_empty_n, rho_n)

    In guide():
      Outside plate: guide_global() -> dict with 'phi'
      GE only first: run_encoder(x_nf) -> enc dict  (GE-specific convenience)
                     guide_epsilon_dist(enc) -> Distribution | None
      Inside  plate: guide_latent(encoder_input_nf) -> z_nk
                     guide_per_droplet(x_nf, prob_n) -> (d_cell_n, d_empty_n)
    """

    # Buffer / submodule type annotations (override nn.Module.__getattr__ return type)
    feature_indices_f: torch.Tensor
    phi_conc_prior: torch.Tensor
    phi_rate_prior: torch.Tensor
    rho_alpha_prior_buf: torch.Tensor
    rho_beta_prior_buf: torch.Tensor
    encoder: nn.Module
    decoder: nn.Module

    def __init__(
        self,
        priors: ModalityPriors,
        feature_indices_f: torch.Tensor,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.priors = priors
        self.encoder = encoder
        self.decoder = decoder
        self.include_rho: bool = priors.include_rho
        self.register_buffer("feature_indices_f", feature_indices_f)
        phi_conc = priors.phi_loc_prior**2 / priors.phi_scale_prior**2
        phi_rate = priors.phi_loc_prior / priors.phi_scale_prior**2
        self.register_buffer("phi_conc_prior", torch.tensor(phi_conc).float())
        self.register_buffer("phi_rate_prior", torch.tensor(phi_rate).float())
        self.register_buffer("rho_alpha_prior_buf", torch.tensor(priors.rho_alpha_prior).float())
        self.register_buffer("rho_beta_prior_buf", torch.tensor(priors.rho_beta_prior).float())

    @property
    def device(self) -> torch.device:
        """Current device, derived from the feature_indices buffer — moves with the module."""
        return self.feature_indices_f.device

    # --- Abstract identity properties ---

    @property
    @abstractmethod
    def name(self) -> str:
        """Modality name; used as a prefix for all Pyro site/param names."""
        ...

    @property
    @abstractmethod
    def contributes_to_cell_probability(self) -> bool:
        """Whether this modality's counts feed into the cell/empty classifier."""
        ...

    @property
    @abstractmethod
    def contributes_to_epsilon_inference(self) -> bool:
        """Whether this modality's encoder informs the global epsilon_n posterior."""
        ...

    # Convenience property: average expression profile for swapping model.
    # Subclasses that use it override to return their buffer.
    @property
    def chi_bar_f(self) -> Optional[torch.Tensor]:
        return None

    # --- Abstract routing methods ---

    @abstractmethod
    def prepare_encoder_input(
        self,
        raw_data_dict: Dict[str, torch.Tensor],
        transforms: Optional[dict] = None,
    ) -> torch.Tensor:
        """Select / transform raw features to produce this modality's encoder input."""
        ...

    @abstractmethod
    def prepare_decoder_input(self, latents_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select / concatenate latents to produce this modality's decoder input."""
        ...

    # --- Abstract Pyro model methods ---

    @abstractmethod
    def model_global(self) -> dict:
        """Declare global (scalar) parameters and sample global latents.

        Called outside pyro.plate.  Must declare chi_ambient as a pyro.param
        and sample phi.
        Returns dict containing at least 'phi' and 'chi_ambient_f'.
        """
        ...

    @abstractmethod
    def model_latent(self, n_droplets: int) -> Optional[torch.Tensor]:
        """Sample z_nk from the prior inside pyro.plate."""
        ...

    @abstractmethod
    def model_per_droplet(
        self,
        x_nf: torch.Tensor,
        empty_mask_n: torch.Tensor,
        epsilon_n: torch.Tensor,
        y_n: torch.Tensor,
        phi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Sample per-droplet parameters inside pyro.plate.

        Must sample d_cell_{name}, d_empty_{name}, emit obs_empty_{name}, and
        (when include_rho) sample rho_n_{name}.
        Returns (d_cell_n, d_empty_n, rho_n).  rho_n is None when not include_rho.
        """
        ...

    # --- Abstract Pyro guide methods ---

    @abstractmethod
    def guide_global(self) -> dict:
        """Declare variational parameters and sample global latents.

        Called outside pyro.plate.  Must sample phi_{name}.
        Returns dict containing at least 'phi'.
        """
        ...

    @abstractmethod
    def guide_latent(self, encoder_input_nf: torch.Tensor) -> Optional[torch.Tensor]:
        """Run encoder and sample z_nk from variational inside pyro.plate."""
        ...

    def guide_epsilon_dist(self, enc_output: dict) -> Optional[dist.Distribution]:
        """Return the variational distribution for global epsilon_n, or None."""
        return None

    @abstractmethod
    def guide_per_droplet(
        self,
        x_nf: torch.Tensor,
        prob_n: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample per-droplet variational parameters inside pyro.plate.

        Must sample d_cell_{name}, d_empty_{name}, and (when include_rho)
        rho_n_{name}.
        Returns (d_cell_n, d_empty_n).
        """
        ...

    # --- Abstract compute methods ---

    @abstractmethod
    def decode(self, decoder_input: torch.Tensor) -> torch.Tensor:
        """Decode latents to chi_nf (fractional expression over this modality's features)."""
        ...

    @abstractmethod
    def observation_dist(
        self,
        mu_nf: torch.Tensor,
        lam_nf: torch.Tensor,
        phi: torch.Tensor,
    ) -> dist.Distribution:
        """Return the observation likelihood distribution for this modality."""
        ...


# ---------------------------------------------------------------------------
# Module-level compute functions (used by model.py and simulate.py)
# ---------------------------------------------------------------------------


def calculate_mu(
    epsilon_n: torch.Tensor,
    d_cell_n: torch.Tensor,
    chi_nf: torch.Tensor,
    y_n: Optional[torch.Tensor] = None,
    rho_n: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute mean signal rate mu_nf = E[true counts per droplet per feature]."""
    mu_nf = epsilon_n.unsqueeze(-1) * d_cell_n.unsqueeze(-1) * chi_nf
    if y_n is not None:
        mu_nf = y_n.unsqueeze(-1) * mu_nf
    if rho_n is not None:
        mu_nf = (1.0 - rho_n.unsqueeze(-1)) * mu_nf
    return mu_nf


def calculate_lambda(
    epsilon_n: torch.Tensor,
    chi_ambient_f: torch.Tensor,
    d_empty_n: torch.Tensor,
    y_n: Optional[torch.Tensor] = None,
    d_cell_n: Optional[torch.Tensor] = None,
    rho_n: Optional[torch.Tensor] = None,
    chi_bar_f: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ambient noise rate lam_nf = E[ambient counts per droplet per feature]."""
    if rho_n is not None and chi_bar_f is not None and y_n is not None and d_cell_n is not None:
        return epsilon_n.unsqueeze(-1) * (
            (1.0 - rho_n.unsqueeze(-1)) * chi_ambient_f * d_empty_n.unsqueeze(-1)
            + rho_n.unsqueeze(-1) * chi_bar_f * (y_n.unsqueeze(-1) * d_cell_n.unsqueeze(-1) + d_empty_n.unsqueeze(-1))
        )
    return epsilon_n.unsqueeze(-1) * d_empty_n.unsqueeze(-1) * chi_ambient_f


# ---------------------------------------------------------------------------
# GeneExpressionModality
# ---------------------------------------------------------------------------


@register_modality("gene_expression")
class GeneExpressionModality(ModalityModule):
    """Gene expression modality — the primary / backbone modality.

    Encoder: CompositeEncoder (EncodeZ + EncodeNonZLatents).
    Decoder: standard softmax Decoder.
    Informs: cell probability (p_n), ambient scaling (epsilon_n), z.
    Owns:    chi_ambient_gene_expression (simplex param), d_empty_gene_expression,
             d_cell_gene_expression, phi_gene_expression.
             When include_rho: rho_n_gene_expression (per-modality swapping fraction).
    """

    name = "gene_expression"
    contributes_to_cell_probability = True
    contributes_to_epsilon_inference = True

    # Buffer type annotations
    d_cell_loc_prior: torch.Tensor
    d_cell_scale_prior: torch.Tensor
    d_empty_loc_prior: torch.Tensor
    d_empty_scale_prior: torch.Tensor
    chi_ambient_init: torch.Tensor
    z_loc_prior: torch.Tensor
    z_scale_prior: torch.Tensor
    _chi_bar_f: Optional[torch.Tensor]
    _enc_cache: Optional[dict]

    def __init__(
        self,
        priors: ModalityPriors,
        feature_indices_f: torch.Tensor,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__(priors, feature_indices_f, encoder, decoder)
        self.z_dim = cast(int, decoder.input_dim)

        self.register_buffer("d_cell_loc_prior", torch.tensor(priors.d_cell_loc_prior).float())
        self.register_buffer("d_cell_scale_prior", torch.tensor(priors.d_cell_scale_prior).float())
        self.register_buffer("d_empty_loc_prior", torch.tensor(priors.d_empty_loc_prior).float())
        self.register_buffer("d_empty_scale_prior", torch.tensor(priors.d_empty_scale_prior).float())
        self.register_buffer("chi_ambient_init", priors.chi_ambient_init.clone().float())
        self.register_buffer("z_loc_prior", torch.zeros(self.z_dim).float())
        self.register_buffer("z_scale_prior", torch.ones(self.z_dim).float())

        if priors.chi_bar is not None:
            self.register_buffer("_chi_bar_f", priors.chi_bar.clone().float())
        else:
            self._chi_bar_f = None

        self._enc_cache = None

    @property
    def chi_bar_f(self) -> Optional[torch.Tensor]:
        return self._chi_bar_f

    # --- Routing ---

    def prepare_encoder_input(self, raw_data_dict, transforms=None):
        return raw_data_dict["gene_expression"]

    def prepare_decoder_input(self, latents_dict):
        return latents_dict["gene_expression"]

    # --- GE-specific: run the full composite encoder once ---

    def run_encoder(self, x_gex_nf: torch.Tensor) -> dict:
        """Run composite encoder (EncodeZ + EncodeNonZLatents) and cache the result."""
        chi_ambient_f = pyro.param(f"chi_ambient_{self.name}")
        param_store = pyro.get_param_store()
        d_empty_loc_key = f"d_empty_loc_{self.name}"
        if d_empty_loc_key in param_store:
            d_empty_loc = param_store[d_empty_loc_key].detach()
        else:
            d_empty_loc = self.d_empty_loc_prior
        enc = self.encoder(
            x=x_gex_nf,
            chi_ambient=chi_ambient_f.detach(),
            cell_prior_log=self.d_cell_loc_prior,
            d_empty_loc=d_empty_loc,
        )
        self._enc_cache = enc
        return enc

    # --- Model ---

    def model_global(self) -> dict:
        chi_ambient_f = pyro.param(
            f"chi_ambient_{self.name}",
            self.chi_ambient_init,
            constraint=constraints.simplex,
        )
        phi = pyro.sample(
            f"phi_{self.name}",
            dist.Gamma(self.phi_conc_prior, self.phi_rate_prior),
        )
        return {"phi": phi, "chi_ambient_f": chi_ambient_f}

    def model_latent(self, n_droplets: int) -> torch.Tensor:
        return pyro.sample(
            f"z_{self.name}",
            dist.Normal(self.z_loc_prior, self.z_scale_prior).expand_by([n_droplets]).to_event(1),
        )

    def model_per_droplet(
        self, x_nf, empty_mask_n, epsilon_n, y_n, phi
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        chi_ambient_f = pyro.param(f"chi_ambient_{self.name}")

        d_empty_n = pyro.sample(
            f"d_empty_{self.name}",
            dist.LogNormal(self.d_empty_loc_prior, self.d_empty_scale_prior).expand([x_nf.shape[0]]),
        )
        d_cell_n = pyro.sample(
            f"d_cell_{self.name}",
            dist.LogNormal(self.d_cell_loc_prior, self.d_cell_scale_prior).expand([x_nf.shape[0]]),
        )

        # Sample per-modality PCR-chimera / swapping fraction.
        if self.include_rho:
            rho_n: Optional[torch.Tensor] = pyro.sample(
                f"rho_n_{self.name}",
                dist.Beta(self.rho_alpha_prior_buf, self.rho_beta_prior_buf).expand([x_nf.shape[0]]),
            )
        else:
            rho_n = None

        # Semi-supervise chi_ambient using known empty droplets.
        with poutine.mask(mask=empty_mask_n):
            with poutine.scale(scale=consts.REG_SCALE_AMBIENT_EXPRESSION):
                r = rho_n.detach() if rho_n is not None else None
                lam_empty_nf = calculate_lambda(
                    epsilon_n=torch.ones_like(epsilon_n),
                    chi_ambient_f=chi_ambient_f,
                    d_empty_n=d_empty_n,
                    y_n=torch.zeros_like(y_n),
                    d_cell_n=d_cell_n.detach(),
                    rho_n=r,
                    chi_bar_f=self._chi_bar_f,
                )
                pyro.sample(
                    f"obs_empty_{self.name}",
                    dist.Poisson(lam_empty_nf + consts.POISSON_EPS_SAFEGAURD).to_event(1),
                    obs=x_nf,
                )

        return d_cell_n, d_empty_n, rho_n

    # --- Guide ---

    def guide_global(self) -> dict:
        chi_ambient_f = pyro.param(
            f"chi_ambient_{self.name}",
            self.chi_ambient_init,
            constraint=constraints.simplex,
        )
        phi_loc = pyro.param(
            f"phi_loc_{self.name}",
            torch.tensor(self.priors.phi_loc_prior, device=self.device),
            constraint=constraints.positive,
        )
        phi_scale = pyro.param(
            f"phi_scale_{self.name}",
            torch.tensor(self.priors.phi_scale_prior, device=self.device),
            constraint=constraints.positive,
        )
        phi_conc = phi_loc.pow(2) / phi_scale.pow(2)
        phi_rate = phi_loc / phi_scale.pow(2)
        phi = pyro.sample(f"phi_{self.name}", dist.Gamma(phi_conc, phi_rate))
        # Declare per-modality rho variational params outside the plate.
        if self.include_rho:
            pyro.param(
                f"rho_alpha_{self.name}",
                self.rho_alpha_prior_buf.clone(),
                constraint=constraints.interval(consts.RHO_PARAM_MIN, consts.RHO_PARAM_MAX),
            )
            pyro.param(
                f"rho_beta_{self.name}",
                self.rho_beta_prior_buf.clone(),
                constraint=constraints.interval(consts.RHO_PARAM_MIN, consts.RHO_PARAM_MAX),
            )
        return {"phi": phi, "chi_ambient_f": chi_ambient_f}

    def guide_epsilon_dist(self, enc_output: dict) -> dist.Distribution:
        epsilon_gated = enc_output["epsilon"]
        return dist.Gamma(
            concentration=epsilon_gated * consts.EPSILON_PRIOR,
            rate=torch.tensor(consts.EPSILON_PRIOR, device=self.device),
        )

    def guide_latent(self, encoder_input_nf: torch.Tensor) -> torch.Tensor:
        if self._enc_cache is None:
            self.run_encoder(encoder_input_nf)
        assert self._enc_cache is not None
        z_loc = self._enc_cache["z"]["loc"]
        z_scale = self._enc_cache["z"]["scale"].clamp(min=1e-6, max=1e4)
        return pyro.sample(
            f"z_{self.name}",
            dist.Normal(loc=z_loc, scale=z_scale).to_event(1),
        )

    def guide_per_droplet(self, x_nf: torch.Tensor, prob_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d_empty_loc = pyro.param(
            f"d_empty_loc_{self.name}",
            self.d_empty_loc_prior.clone(),
            constraint=constraints.positive,
        )
        d_empty_scale = pyro.param(
            f"d_empty_scale_{self.name}",
            self.d_empty_scale_prior.clone(),
            constraint=constraints.positive,
        )
        d_empty_n = pyro.sample(
            f"d_empty_{self.name}",
            dist.LogNormal(d_empty_loc, d_empty_scale).expand([x_nf.shape[0]]),
        )

        d_cell_scale = pyro.param(
            f"d_cell_scale_{self.name}",
            torch.tensor([consts.D_CELL_SCALE_INIT], device=self.device),
            constraint=constraints.positive,
        )
        if self._enc_cache is not None:
            d_cell_loc_gated = prob_n * self._enc_cache["d_loc"] + (1.0 - prob_n) * self.d_cell_loc_prior
        else:
            d_cell_loc_gated = self.d_cell_loc_prior.expand(x_nf.shape[0])

        d_cell_n = pyro.sample(
            f"d_cell_{self.name}",
            dist.LogNormal(loc=d_cell_loc_gated, scale=d_cell_scale),
        )

        if self.include_rho:
            rho_alpha = pyro.param(f"rho_alpha_{self.name}")
            rho_beta = pyro.param(f"rho_beta_{self.name}")
            pyro.sample(
                f"rho_n_{self.name}",
                dist.Beta(rho_alpha, rho_beta).expand([x_nf.shape[0]]),
            )

        return d_cell_n, d_empty_n

    # --- Compute ---

    def decode(self, decoder_input: torch.Tensor) -> torch.Tensor:
        return self.decoder(decoder_input)

    def observation_dist(self, mu_nf, lam_nf, phi) -> dist.Distribution:
        alpha = phi.reciprocal()
        return NBPCapprox(
            mu=mu_nf + consts.NBPC_MU_EPS_SAFEGAURD,
            alpha=alpha + consts.NBPC_ALPHA_EPS_SAFEGAURD,
            lam=lam_nf + consts.NBPC_LAM_EPS_SAFEGAURD,
        )


# ---------------------------------------------------------------------------
# GuidePerturbationModality
# ---------------------------------------------------------------------------


@register_modality("guide_perturbation")
class GuidePerturbationModality(ModalityModule):
    """CRISPR guide RNA capture modality.

    The encoder sees a precomputed static transform of GE data (not raw guide
    counts) to capture perturbation identity.  The decoder reconstructs guide
    RNA counts from the perturbation latent z_guide_nk.

    Observation model: NBPCapprox (signal NB + noise Poisson convolution),
    consistent with the posterior noise-count estimator.

    cell_probability: does NOT contribute (guide capture is unreliable for
    distinguishing cells from empty droplets).
    epsilon: does NOT contribute (epsilon is inferred from GE only).

    When include_rho: models per-modality PCR-chimera fraction rho_n_guide_perturbation.
    chi_bar_guide_perturbation is the population-average guide count profile (law of
    mass action), used in the swapping model.
    """

    name = "guide_perturbation"
    contributes_to_cell_probability = False
    contributes_to_epsilon_inference = False

    # Buffer type annotations
    d_cell_loc_prior: torch.Tensor
    d_cell_scale_prior: torch.Tensor
    d_empty_loc_prior: torch.Tensor
    d_empty_scale_prior: torch.Tensor
    chi_ambient_init: torch.Tensor
    z_loc_prior: torch.Tensor
    z_scale_prior: torch.Tensor
    _chi_bar_f_buf: Optional[torch.Tensor]
    transform_mean: Optional[torch.Tensor]
    transform_std: Optional[torch.Tensor]
    transform_hvg_indices: Optional[torch.Tensor]
    transform_clamp_min: torch.Tensor
    transform_clamp_max: torch.Tensor

    def __init__(
        self,
        priors: ModalityPriors,
        feature_indices_f: torch.Tensor,
        encoder: nn.Module,
        decoder: nn.Module,
        transform_mean: Optional[torch.Tensor] = None,
        transform_std: Optional[torch.Tensor] = None,
        transform_hvg_indices: Optional[torch.Tensor] = None,
        transform_clamp_min: float = -5.0,
        transform_clamp_max: float = 5.0,
    ):
        super().__init__(priors, feature_indices_f, encoder, decoder)
        self.z_dim = cast(int, decoder.input_dim)

        self.register_buffer("d_cell_loc_prior", torch.tensor(priors.d_cell_loc_prior).float())
        self.register_buffer("d_cell_scale_prior", torch.tensor(priors.d_cell_scale_prior).float())
        self.register_buffer("d_empty_loc_prior", torch.tensor(priors.d_empty_loc_prior).float())
        self.register_buffer("d_empty_scale_prior", torch.tensor(priors.d_empty_scale_prior).float())
        self.register_buffer("chi_ambient_init", priors.chi_ambient_init.clone().float())
        self.register_buffer("z_loc_prior", torch.zeros(self.z_dim).float())
        self.register_buffer("z_scale_prior", torch.ones(self.z_dim).float())

        if priors.chi_bar is not None:
            self.register_buffer("_chi_bar_f_buf", priors.chi_bar.clone().float())
        else:
            self._chi_bar_f_buf = None

        self.register_buffer(
            "transform_mean",
            transform_mean.float() if transform_mean is not None else None,
        )
        self.register_buffer(
            "transform_std",
            transform_std.float() if transform_std is not None else None,
        )
        self.register_buffer(
            "transform_hvg_indices",
            transform_hvg_indices.long() if transform_hvg_indices is not None else None,
        )
        self.register_buffer("transform_clamp_min", torch.tensor(transform_clamp_min).float())
        self.register_buffer("transform_clamp_max", torch.tensor(transform_clamp_max).float())

    @property
    def chi_bar_f(self) -> Optional[torch.Tensor]:
        return self._chi_bar_f_buf

    def _apply_transform(self, x_gex_nf: torch.Tensor) -> torch.Tensor:
        """Select HVG columns from GEX counts and z-score using control cell statistics."""
        if self.transform_hvg_indices is None:
            raise RuntimeError(
                "GuidePerturbationModality transform parameters are not set.  "
                "Ensure compute_guide_transform_params() was called and its results "
                "were passed to the constructor via transform_mean / transform_std / "
                "transform_hvg_indices."
            )
        x = x_gex_nf[:, self.transform_hvg_indices].log1p()
        if self.transform_mean is not None and self.transform_std is not None:
            std = self.transform_std.clamp(min=1e-6)
            x = (x - self.transform_mean) / std
        return x.clamp(self.transform_clamp_min.item(), self.transform_clamp_max.item())

    # --- Routing ---

    def prepare_encoder_input(self, raw_data_dict, transforms=None):
        return self._apply_transform(raw_data_dict["gene_expression"])

    def prepare_decoder_input(self, latents_dict):
        return latents_dict["guide_perturbation"]

    # --- Model ---

    def model_global(self) -> dict:
        chi_ambient_f = pyro.param(
            f"chi_ambient_{self.name}",
            self.chi_ambient_init,
            constraint=constraints.simplex,
        )
        phi = pyro.sample(
            f"phi_{self.name}",
            dist.Gamma(self.phi_conc_prior, self.phi_rate_prior),
        )
        return {"phi": phi, "chi_ambient_f": chi_ambient_f}

    def model_latent(self, n_droplets: int) -> torch.Tensor:
        return pyro.sample(
            f"z_{self.name}",
            dist.Normal(self.z_loc_prior, self.z_scale_prior).expand_by([n_droplets]).to_event(1),
        )

    def model_per_droplet(
        self, x_nf, empty_mask_n, epsilon_n, y_n, phi
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        chi_ambient_f = pyro.param(f"chi_ambient_{self.name}")

        d_empty_n = pyro.sample(
            f"d_empty_{self.name}",
            dist.LogNormal(self.d_empty_loc_prior, self.d_empty_scale_prior).expand([x_nf.shape[0]]),
        )
        d_cell_n = pyro.sample(
            f"d_cell_{self.name}",
            dist.LogNormal(self.d_cell_loc_prior, self.d_cell_scale_prior).expand([x_nf.shape[0]]),
        )

        if self.include_rho:
            rho_n: Optional[torch.Tensor] = pyro.sample(
                f"rho_n_{self.name}",
                dist.Beta(self.rho_alpha_prior_buf, self.rho_beta_prior_buf).expand([x_nf.shape[0]]),
            )
        else:
            rho_n = None

        # Semi-supervise guide chi_ambient using known empty droplets.
        with poutine.mask(mask=empty_mask_n):
            with poutine.scale(scale=consts.REG_SCALE_AMBIENT_EXPRESSION):
                r = rho_n.detach() if rho_n is not None else None
                lam_empty_nf = calculate_lambda(
                    epsilon_n=torch.ones_like(epsilon_n),
                    chi_ambient_f=chi_ambient_f,
                    d_empty_n=d_empty_n,
                    y_n=torch.zeros_like(y_n),
                    d_cell_n=d_cell_n.detach(),
                    rho_n=r,
                    chi_bar_f=self._chi_bar_f_buf,
                )
                pyro.sample(
                    f"obs_empty_{self.name}",
                    dist.Poisson(lam_empty_nf + consts.POISSON_EPS_SAFEGAURD).to_event(1),
                    obs=x_nf,
                )

        return d_cell_n, d_empty_n, rho_n

    # --- Guide ---

    def guide_global(self) -> dict:
        # Declare chi_ambient in the guide so the Adam optimizer tracks it.
        chi_ambient_f = pyro.param(
            f"chi_ambient_{self.name}",
            self.chi_ambient_init,
            constraint=constraints.simplex,
        )
        phi_loc = pyro.param(
            f"phi_loc_{self.name}",
            torch.tensor(self.priors.phi_loc_prior, device=self.device),
            constraint=constraints.positive,
        )
        phi_scale = pyro.param(
            f"phi_scale_{self.name}",
            torch.tensor(self.priors.phi_scale_prior, device=self.device),
            constraint=constraints.positive,
        )
        phi_conc = phi_loc.pow(2) / phi_scale.pow(2)
        phi_rate = phi_loc / phi_scale.pow(2)
        phi = pyro.sample(f"phi_{self.name}", dist.Gamma(phi_conc, phi_rate))
        if self.include_rho:
            pyro.param(
                f"rho_alpha_{self.name}",
                self.rho_alpha_prior_buf.clone(),
                constraint=constraints.interval(consts.RHO_PARAM_MIN, consts.RHO_PARAM_MAX),
            )
            pyro.param(
                f"rho_beta_{self.name}",
                self.rho_beta_prior_buf.clone(),
                constraint=constraints.interval(consts.RHO_PARAM_MIN, consts.RHO_PARAM_MAX),
            )
        return {"phi": phi, "chi_ambient_f": chi_ambient_f}

    def guide_latent(self, encoder_input_nf: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(encoder_input_nf)
        return pyro.sample(
            f"z_{self.name}",
            dist.Normal(loc=enc["loc"], scale=enc["scale"]).to_event(1),
        )

    def guide_per_droplet(self, x_nf: torch.Tensor, prob_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d_empty_loc = pyro.param(
            f"d_empty_loc_{self.name}",
            self.d_empty_loc_prior.clone(),
            constraint=constraints.positive,
        )
        d_empty_scale = pyro.param(
            f"d_empty_scale_{self.name}",
            self.d_empty_scale_prior.clone(),
            constraint=constraints.positive,
        )
        d_empty_n = pyro.sample(
            f"d_empty_{self.name}",
            dist.LogNormal(d_empty_loc, d_empty_scale).expand([x_nf.shape[0]]),
        )

        d_cell_scale = pyro.param(
            f"d_cell_scale_{self.name}",
            torch.tensor([consts.D_CELL_SCALE_INIT], device=self.device),
            constraint=constraints.positive,
        )
        # Use log(max guide count + 1) as a data-derived point estimate for d_loc_guide.
        # The encoder sees GEX-derived features and cannot estimate guide depth directly;
        # the max count across guides is a reliable proxy for the dominant guide UMI depth.
        d_cell_loc_gated = prob_n * x_nf.max(dim=-1).values.log1p() + (1.0 - prob_n) * self.d_cell_loc_prior

        d_cell_n = pyro.sample(
            f"d_cell_{self.name}",
            dist.LogNormal(loc=d_cell_loc_gated, scale=d_cell_scale),
        )

        if self.include_rho:
            rho_alpha = pyro.param(f"rho_alpha_{self.name}")
            rho_beta = pyro.param(f"rho_beta_{self.name}")
            pyro.sample(
                f"rho_n_{self.name}",
                dist.Beta(rho_alpha, rho_beta).expand([x_nf.shape[0]]),
            )

        return d_cell_n, d_empty_n

    # --- Compute ---

    def decode(self, decoder_input: torch.Tensor) -> torch.Tensor:
        return self.decoder(decoder_input)

    def observation_dist(self, mu_nf, lam_nf, phi) -> dist.Distribution:
        """NBPCapprox observation model: consistent with posterior noise estimator."""
        alpha = phi.reciprocal()
        return NBPCapprox(
            mu=mu_nf + consts.NBPC_MU_EPS_SAFEGAURD,
            alpha=alpha + consts.NBPC_ALPHA_EPS_SAFEGAURD,
            lam=lam_nf + consts.NBPC_LAM_EPS_SAFEGAURD,
        )


# ---------------------------------------------------------------------------
# _SecondaryModalityBase — shared base for ATAC and Protein
# ---------------------------------------------------------------------------


class _SecondaryModalityBase(ModalityModule, ABC):
    """Base for secondary modalities (ATAC, Protein).

    Each secondary modality infers its own per-modality latent z_self from its
    own features.  The decoder reconstructs chi from concat(z_gex.detach(), z_self).

    These modalities do not contribute to cell probability or epsilon inference.
    When include_rho: models per-modality PCR-chimera fraction rho_n_{name}.
    chi_bar_{name} is the population-average profile for this modality (law of
    mass action), used in the swapping model alongside rho.

    d_cell_{name} uses log(total_counts + 1) for the posterior d_loc estimate,
    gated by cell probability (analogous to the GEX encoder's depth estimate).
    """

    contributes_to_cell_probability = False
    contributes_to_epsilon_inference = False

    # Buffer type annotations
    d_cell_loc_prior: torch.Tensor
    d_cell_scale_prior: torch.Tensor
    d_empty_loc_prior: torch.Tensor
    d_empty_scale_prior: torch.Tensor
    chi_ambient_init: torch.Tensor
    z_loc_prior: torch.Tensor
    z_scale_prior: torch.Tensor
    _chi_bar_f_buf: Optional[torch.Tensor]

    def __init__(
        self,
        priors: ModalityPriors,
        feature_indices_f: torch.Tensor,
        encoder: nn.Module,
        decoder: nn.Module,
        z_dim: int,
    ):
        super().__init__(priors, feature_indices_f, encoder, decoder)
        self.z_dim = z_dim
        self.register_buffer("d_cell_loc_prior", torch.tensor(priors.d_cell_loc_prior).float())
        self.register_buffer("d_cell_scale_prior", torch.tensor(priors.d_cell_scale_prior).float())
        self.register_buffer("d_empty_loc_prior", torch.tensor(priors.d_empty_loc_prior).float())
        self.register_buffer("d_empty_scale_prior", torch.tensor(priors.d_empty_scale_prior).float())
        self.register_buffer("chi_ambient_init", priors.chi_ambient_init.clone().float())
        self.register_buffer("z_loc_prior", torch.zeros(z_dim).float())
        self.register_buffer("z_scale_prior", torch.ones(z_dim).float())

        if priors.chi_bar is not None:
            self.register_buffer("_chi_bar_f_buf", priors.chi_bar.clone().float())
        else:
            self._chi_bar_f_buf = None

    @property
    def chi_bar_f(self) -> Optional[torch.Tensor]:
        return self._chi_bar_f_buf

    def prepare_encoder_input(self, raw_data_dict, transforms=None):
        return raw_data_dict[self.name]

    def prepare_decoder_input(self, latents_dict):
        z_gex = latents_dict["gene_expression"].detach()
        z_self = latents_dict[self.name]
        return torch.cat([z_gex, z_self], dim=-1)

    # --- Model ---

    def model_global(self) -> dict:
        chi_ambient_f = pyro.param(
            f"chi_ambient_{self.name}",
            self.chi_ambient_init,
            constraint=constraints.simplex,
        )
        phi = pyro.sample(
            f"phi_{self.name}",
            dist.Gamma(self.phi_conc_prior, self.phi_rate_prior),
        )
        return {"phi": phi, "chi_ambient_f": chi_ambient_f}

    def model_latent(self, n_droplets: int) -> torch.Tensor:
        return pyro.sample(
            f"z_{self.name}",
            dist.Normal(self.z_loc_prior, self.z_scale_prior).expand_by([n_droplets]).to_event(1),
        )

    def model_per_droplet(
        self, x_nf, empty_mask_n, epsilon_n, y_n, phi
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        chi_ambient_f = pyro.param(f"chi_ambient_{self.name}")
        d_empty_n = pyro.sample(
            f"d_empty_{self.name}",
            dist.LogNormal(self.d_empty_loc_prior, self.d_empty_scale_prior).expand([x_nf.shape[0]]),
        )
        d_cell_n = pyro.sample(
            f"d_cell_{self.name}",
            dist.LogNormal(self.d_cell_loc_prior, self.d_cell_scale_prior).expand([x_nf.shape[0]]),
        )

        if self.include_rho:
            rho_n: Optional[torch.Tensor] = pyro.sample(
                f"rho_n_{self.name}",
                dist.Beta(self.rho_alpha_prior_buf, self.rho_beta_prior_buf).expand([x_nf.shape[0]]),
            )
        else:
            rho_n = None

        # Semi-supervise chi_ambient using known empty droplets.
        with poutine.mask(mask=empty_mask_n):
            with poutine.scale(scale=consts.REG_SCALE_AMBIENT_EXPRESSION):
                r = rho_n.detach() if rho_n is not None else None
                lam_empty_nf = calculate_lambda(
                    epsilon_n=torch.ones_like(epsilon_n),
                    chi_ambient_f=chi_ambient_f,
                    d_empty_n=d_empty_n,
                    y_n=torch.zeros_like(y_n),
                    d_cell_n=d_cell_n.detach(),
                    rho_n=r,
                    chi_bar_f=self._chi_bar_f_buf,
                )
                pyro.sample(
                    f"obs_empty_{self.name}",
                    dist.Poisson(lam_empty_nf + consts.POISSON_EPS_SAFEGAURD).to_event(1),
                    obs=x_nf,
                )
        return d_cell_n, d_empty_n, rho_n

    # --- Guide ---

    def guide_global(self) -> dict:
        chi_ambient_f = pyro.param(
            f"chi_ambient_{self.name}",
            self.chi_ambient_init,
            constraint=constraints.simplex,
        )
        phi_loc = pyro.param(
            f"phi_loc_{self.name}",
            torch.tensor(self.priors.phi_loc_prior, device=self.device),
            constraint=constraints.positive,
        )
        phi_scale = pyro.param(
            f"phi_scale_{self.name}",
            torch.tensor(self.priors.phi_scale_prior, device=self.device),
            constraint=constraints.positive,
        )
        phi_conc = phi_loc.pow(2) / phi_scale.pow(2)
        phi_rate = phi_loc / phi_scale.pow(2)
        phi = pyro.sample(f"phi_{self.name}", dist.Gamma(phi_conc, phi_rate))
        if self.include_rho:
            pyro.param(
                f"rho_alpha_{self.name}",
                self.rho_alpha_prior_buf.clone(),
                constraint=constraints.interval(consts.RHO_PARAM_MIN, consts.RHO_PARAM_MAX),
            )
            pyro.param(
                f"rho_beta_{self.name}",
                self.rho_beta_prior_buf.clone(),
                constraint=constraints.interval(consts.RHO_PARAM_MIN, consts.RHO_PARAM_MAX),
            )
        return {"phi": phi, "chi_ambient_f": chi_ambient_f}

    def guide_latent(self, encoder_input_nf: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(encoder_input_nf)
        z_loc = enc["loc"]
        z_scale = enc["scale"].clamp(min=1e-6, max=1e4)
        return pyro.sample(
            f"z_{self.name}",
            dist.Normal(loc=z_loc, scale=z_scale).to_event(1),
        )

    def guide_per_droplet(self, x_nf: torch.Tensor, prob_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d_empty_loc = pyro.param(
            f"d_empty_loc_{self.name}",
            self.d_empty_loc_prior.clone(),
            constraint=constraints.positive,
        )
        d_empty_scale = pyro.param(
            f"d_empty_scale_{self.name}",
            self.d_empty_scale_prior.clone(),
            constraint=constraints.positive,
        )
        d_empty_n = pyro.sample(
            f"d_empty_{self.name}",
            dist.LogNormal(d_empty_loc, d_empty_scale).expand([x_nf.shape[0]]),
        )
        d_cell_scale = pyro.param(
            f"d_cell_scale_{self.name}",
            torch.tensor([consts.D_CELL_SCALE_INIT], device=self.device),
            constraint=constraints.positive,
        )
        # Use log(total modality counts + 1) as d_loc, gated by cell probability.
        # Total counts in this modality is a direct proxy for per-modality depth.
        d_cell_loc_gated = prob_n * x_nf.sum(dim=-1).log1p() + (1.0 - prob_n) * self.d_cell_loc_prior
        d_cell_n = pyro.sample(
            f"d_cell_{self.name}",
            dist.LogNormal(loc=d_cell_loc_gated, scale=d_cell_scale),
        )

        if self.include_rho:
            rho_alpha = pyro.param(f"rho_alpha_{self.name}")
            rho_beta = pyro.param(f"rho_beta_{self.name}")
            pyro.sample(
                f"rho_n_{self.name}",
                dist.Beta(rho_alpha, rho_beta).expand([x_nf.shape[0]]),
            )

        return d_cell_n, d_empty_n

    # --- Compute ---

    def decode(self, decoder_input: torch.Tensor) -> torch.Tensor:
        return self.decoder(decoder_input)

    def observation_dist(self, mu_nf, lam_nf, phi) -> dist.Distribution:
        """NBPCapprox observation model: consistent with posterior noise estimator."""
        alpha = phi.reciprocal()
        return NBPCapprox(
            mu=mu_nf + consts.NBPC_MU_EPS_SAFEGAURD,
            alpha=alpha + consts.NBPC_ALPHA_EPS_SAFEGAURD,
            lam=lam_nf + consts.NBPC_LAM_EPS_SAFEGAURD,
        )


# ---------------------------------------------------------------------------
# ATACModality
# ---------------------------------------------------------------------------


@register_modality("atac")
class ATACModality(_SecondaryModalityBase):
    """ATAC-seq (chromatin accessibility peaks) modality.

    Encoder: EncodeZ on log-normalised ATAC features → z_atac.
    Decoder: z_gex.detach() ++ z_atac → chi_atac (simplex over peaks).
    Observation: NBPCapprox.
    """

    name = "atac"


# ---------------------------------------------------------------------------
# ProteinModality
# ---------------------------------------------------------------------------


@register_modality("protein")
class ProteinModality(_SecondaryModalityBase):
    """Antibody capture / protein modality (CITE-seq).

    Encoder: EncodeZ on log-normalised protein features → z_protein.
    Decoder: z_gex.detach() ++ z_protein → chi_protein (simplex over proteins).
    Observation: NBPCapprox.
    """

    name = "protein"
