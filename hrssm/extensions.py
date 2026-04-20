"""
Extended model variants for next-step experiments.

  HyperbolicWorldModelSeparateBeta  — decoupled KL weights for tau and fibre terms
  HyperbolicWorldModelAwareDecoder  — geometry-aware decoder using Busemann features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from .distributions import HorosphericalGaussian
from .world_model import HyperbolicWorldModel, _mlp


# ---------------------------------------------------------------------------
# Experiment 4 — Separate KL weights (beta_tau vs beta_b)
# ---------------------------------------------------------------------------

class HyperbolicWorldModelSeparateBeta(HyperbolicWorldModel):
    """
    Extends HyperbolicWorldModel with decoupled KL weights:

      loss = recon + beta_tau * height_KL + beta_b * (fibre_KL + drift)

    The standard model uses beta_tau = beta_b = beta.
    Setting beta_b < 1 suppresses the (d-1)-weighted fibre penalty,
    potentially allowing b to encode branch identity at higher d.

    Usage — same as HyperbolicWorldModel, but call forward_separate_beta():

        recons, kls, info = model.forward_separate_beta(obs, beta_tau=1.0, beta_b=0.1)
    """

    def forward_separate_beta(
        self,
        obs_seq: torch.Tensor,          # (batch, T, obs_dim)
        beta_tau: float = 1.0,
        beta_b:   float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with decoupled KL weights.
        Returns (recons, weighted_kls, info) — same shape contract as forward().
        weighted_kls already has beta_tau / beta_b baked in.
        """
        batch, T, _ = obs_seq.shape
        device = obs_seq.device
        h = torch.zeros(batch, self.hidden_dim, device=device)

        recons, kls = [], []
        mu_taus, mu_bs, sigma_taus = [], [], []

        for t in range(T):
            enc = self.obs_encoder(obs_seq[:, t])

            prior_params = self.prior_net(h)
            prior        = self._parse_hg(prior_params)

            post_params = self.posterior_net(torch.cat([h, enc], dim=-1))
            posterior   = self._parse_hg(post_params)

            tau, b = posterior.rsample()
            z      = torch.cat([tau.unsqueeze(-1), b], dim=-1)

            recons.append(self.decoder(z))

            # --- Decompose KL into three terms ---
            height_kl = (
                torch.log(prior.sigma_tau / posterior.sigma_tau)
                + (posterior.sigma_tau**2 + (posterior.mu_tau - prior.mu_tau)**2)
                  / (2.0 * prior.sigma_tau**2)
                - 0.5
            )

            fibre_kl = posterior.d_fibre * (
                torch.log(prior.sigma_b / posterior.sigma_b)
                + posterior.sigma_b**2 / (2.0 * prior.sigma_b**2)
                - 0.5
            )

            mu_b_diff_sq = ((posterior.mu_b - prior.mu_b)**2).sum(dim=-1)
            log_mgf = torch.clamp(
                -2.0 * posterior.mu_tau + 2.0 * posterior.sigma_tau**2,
                max=HorosphericalGaussian.LOG_MGF_MAX
            )
            drift = mu_b_diff_sq / (2.0 * prior.sigma_b**2) * torch.exp(log_mgf)

            weighted_kl = beta_tau * height_kl + beta_b * (fibre_kl + drift)
            kls.append(weighted_kl)

            mu_taus.append(posterior.mu_tau.detach())
            mu_bs.append(posterior.mu_b.detach())
            sigma_taus.append(posterior.sigma_tau.detach())

            h = self.gru(z, h)

        recons = torch.stack(recons, dim=1)
        kls    = torch.stack(kls,    dim=1)
        info   = {
            "mu_tau":    torch.stack(mu_taus,    dim=1),
            "mu_b":      torch.stack(mu_bs,      dim=1),
            "sigma_tau": torch.stack(sigma_taus, dim=1),
        }
        return recons, kls, info


# ---------------------------------------------------------------------------
# Experiment 6 — Hyperbolic-aware decoder
# ---------------------------------------------------------------------------

class _AwareDecoder(nn.Module):
    """
    Decoder that receives both the raw (tau, b) and two geometric features:
      - exp(tau)                  : amplifies the depth-level signal
      - ||b||^2 * exp(-2*tau)    : approximate squared geodesic distance from origin

    These are O(1) derived features. The decoder can now use geometric
    distances directly rather than treating (tau, b) as flat Euclidean coordinates.
    """

    def __init__(self, hidden_dim: int, latent_dim: int, obs_dim: int):
        super().__init__()
        # latent_dim (tau, b) + 2 geometric features
        self.net = _mlp(latent_dim + 2, hidden_dim, obs_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, d) = concat(tau (1), b (d-1))"""
        tau = z[:, :1]                                     # (batch, 1)
        b   = z[:, 1:]                                     # (batch, d-1)
        exp_tau  = tau.exp()                               # (batch, 1)
        geo_dist = (b**2).sum(-1, keepdim=True) * (-2*tau).exp()  # (batch, 1)
        z_aug    = torch.cat([z, exp_tau, geo_dist], dim=-1)
        return self.net(z_aug)


class HyperbolicWorldModelAwareDecoder(HyperbolicWorldModel):
    """
    HyperbolicWorldModel with a geometry-aware decoder.

    Same as the standard model except the decoder receives two extra features:
    exp(tau) and ||b||^2 * exp(-2*tau), letting it exploit the geometric
    structure that the KL already enforces in the latent space.
    """

    def __init__(
        self,
        obs_dim:    int = 64,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        enc_dim:    int = 128,
    ):
        super().__init__(obs_dim, latent_dim, hidden_dim, enc_dim)
        # Replace the flat MLP decoder with the geometry-aware version
        self.decoder = _AwareDecoder(hidden_dim, latent_dim, obs_dim)
