"""
World models for H-RSSM.

Two variants:
  HyperbolicWorldModel: latent z = (tau, b) in H^d, prior/posterior are HG distributions.
  EuclideanWorldModel:  latent z in R^d,    prior/posterior are diagonal Gaussians.

Both follow the RSSM pattern:
  h_0 = 0
  for t = 0 .. T-1:
    prior(z_t)     = f_prior(h_t)
    posterior(z_t) = f_post(h_t, enc(o_t))
    z_t            ~ posterior   (reparameterised, teacher-forced)
    o_hat_t        = decoder(z_t)
    h_{t+1}        = GRU(z_t, h_t)

The two models have identical parameter counts at the same d:
  Hyperbolic prior/posterior output d+2 scalars (mu_tau, mu_b, log_sigma_tau, log_sigma_b).
  Euclidean  prior/posterior output 2d  scalars (mu, log_sigma).
  Difference: 2 extra parameters per step for hyperbolic (negligible).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any

from .distributions import HorosphericalGaussian, EuclideanGaussian


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Hyperbolic world model
# ---------------------------------------------------------------------------

class HyperbolicWorldModel(nn.Module):
    """
    RSSM with HorosphericalGaussian latent space.

    Latent dimension d_latent includes both tau (dim 1) and b (dim d_latent-1).
    The GRU operates on the concatenated vector (tau, b) ∈ R^d_latent.

    HG parameter head outputs d_latent + 2 values:
        [mu_tau (1), mu_b (d-1), log_sigma_tau (1), log_sigma_b (1)]
    """

    def __init__(
        self,
        obs_dim:    int = 64,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        enc_dim:    int = 128,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.latent_dim = latent_dim   # d  (= 1 + d_fibre)
        self.hidden_dim = hidden_dim

        # d_latent + 2 = mu_tau(1) + mu_b(d-1) + log_sigma_tau(1) + log_sigma_b(1) = d+2
        hg_params = latent_dim + 2

        self.obs_encoder   = _mlp(obs_dim,          enc_dim,    enc_dim)
        self.prior_net     = _mlp(hidden_dim,        hidden_dim, hg_params)
        self.posterior_net = _mlp(hidden_dim + enc_dim, hidden_dim, hg_params)
        self.decoder       = _mlp(latent_dim,        hidden_dim, obs_dim)
        self.gru           = nn.GRUCell(latent_dim, hidden_dim)

    # ------------------------------------------------------------------

    def _parse_hg(self, params: torch.Tensor) -> HorosphericalGaussian:
        """Split (batch, d+2) parameter vector into a HG distribution."""
        d = self.latent_dim
        mu_tau        = params[:, 0]
        mu_b          = params[:, 1:d]
        log_sigma_tau = params[:, d]
        log_sigma_b   = params[:, d + 1]
        return HorosphericalGaussian(mu_tau, mu_b, log_sigma_tau, log_sigma_b)

    # ------------------------------------------------------------------

    def forward(
        self,
        obs_seq: torch.Tensor,              # (batch, T, obs_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            recons:  (batch, T, obs_dim)  reconstructed observations
            kls:     (batch, T)           KL divergence per step
            info:    dict with diagnostic tensors
        """
        batch, T, _ = obs_seq.shape
        device = obs_seq.device
        h = torch.zeros(batch, self.hidden_dim, device=device)

        recons, kls = [], []
        mu_taus, mu_bs, sigma_taus = [], [], []

        for t in range(T):
            enc = self.obs_encoder(obs_seq[:, t])      # (batch, enc_dim)

            # Prior
            prior_params = self.prior_net(h)           # (batch, d+2)
            prior        = self._parse_hg(prior_params)

            # Posterior
            post_params  = self.posterior_net(torch.cat([h, enc], dim=-1))
            posterior    = self._parse_hg(post_params)

            # Sample z (reparameterised)
            tau, b = posterior.rsample()               # (batch,), (batch, d-1)
            z      = torch.cat([tau.unsqueeze(-1), b], dim=-1)  # (batch, d)

            # Reconstruct and compute KL
            recons.append(self.decoder(z))
            kls.append(posterior.kl_divergence(prior))

            # Diagnostics
            mu_taus.append(posterior.mu_tau.detach())
            mu_bs.append(posterior.mu_b.detach())
            sigma_taus.append(posterior.sigma_tau.detach())

            h = self.gru(z, h)

        recons = torch.stack(recons, dim=1)            # (batch, T, obs_dim)
        kls    = torch.stack(kls,    dim=1)            # (batch, T)

        info = {
            "mu_tau":    torch.stack(mu_taus,    dim=1),   # (batch, T)
            "mu_b":      torch.stack(mu_bs,      dim=1),   # (batch, T, d-1)
            "sigma_tau": torch.stack(sigma_taus, dim=1),   # (batch, T)
        }
        return recons, kls, info


# ---------------------------------------------------------------------------
# Euclidean world model
# ---------------------------------------------------------------------------

class EuclideanWorldModel(nn.Module):
    """
    RSSM with diagonal-Gaussian latent space (Euclidean baseline).
    Architecture is identical to HyperbolicWorldModel except:
      - prior/posterior output 2*d parameters (mu, log_sigma per dimension)
      - no HG-specific reparameterisation coupling
    """

    def __init__(
        self,
        obs_dim:    int = 64,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        enc_dim:    int = 128,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        gauss_params = 2 * latent_dim   # mu(d) + log_sigma(d)

        self.obs_encoder   = _mlp(obs_dim,          enc_dim,    enc_dim)
        self.prior_net     = _mlp(hidden_dim,        hidden_dim, gauss_params)
        self.posterior_net = _mlp(hidden_dim + enc_dim, hidden_dim, gauss_params)
        self.decoder       = _mlp(latent_dim,        hidden_dim, obs_dim)
        self.gru           = nn.GRUCell(latent_dim, hidden_dim)

    # ------------------------------------------------------------------

    def _parse_gauss(self, params: torch.Tensor) -> EuclideanGaussian:
        d    = self.latent_dim
        mu   = params[:, :d]
        lsig = params[:, d:]
        return EuclideanGaussian(mu, lsig)

    # ------------------------------------------------------------------

    def forward(
        self,
        obs_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        batch, T, _ = obs_seq.shape
        device = obs_seq.device
        h = torch.zeros(batch, self.hidden_dim, device=device)

        recons, kls, mus = [], [], []

        for t in range(T):
            enc = self.obs_encoder(obs_seq[:, t])

            prior     = self._parse_gauss(self.prior_net(h))
            posterior = self._parse_gauss(
                self.posterior_net(torch.cat([h, enc], dim=-1))
            )

            z = posterior.rsample()

            recons.append(self.decoder(z))
            kls.append(posterior.kl_divergence(prior))
            mus.append(posterior.mu.detach())

            h = self.gru(z, h)

        recons = torch.stack(recons, dim=1)
        kls    = torch.stack(kls,    dim=1)
        info   = {"mu": torch.stack(mus, dim=1)}   # (batch, T, d)
        return recons, kls, info


# ---------------------------------------------------------------------------
# Shared training step
# ---------------------------------------------------------------------------

def elbo_loss(
    recons:  torch.Tensor,   # (batch, T, obs_dim)
    obs_seq: torch.Tensor,   # (batch, T, obs_dim)
    kls:     torch.Tensor,   # (batch, T)
    beta:    float = 1.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    ELBO = reconstruction MSE + beta * KL.
    Returns (total_loss, recon_scalar, kl_scalar).
    """
    recon = F.mse_loss(recons, obs_seq)
    kl    = kls.mean()
    return recon + beta * kl, recon.item(), kl.item()
