"""
Probability distributions for H-RSSM.

HorosphericalGaussian: tractable distribution on upper half-space H^d.
EuclideanGaussian:     diagonal Gaussian on R^d (Euclidean baseline).

Upper half-space H^d: coordinates z = (tau, b), tau in R, b in R^{d-1}.
Metric:  ds^2 = dtau^2 + e^{-2*tau} ||db||^2
d-vol:   e^{-(d-1)*tau} dtau db

Key property: the fibre variance Var(b|tau) = e^{2*tau} * sigma_b^2 makes
the b-integral cancel exactly with the volume form factor, yielding a
closed-form normaliser and O(d) KL divergence.
"""

import torch
import math
from typing import Tuple


class HorosphericalGaussian:
    """
    HG(mu_tau, mu_b, sigma_tau, sigma_b) on H^d.

    Parameters (all batched along first dimension):
        mu_tau:       (batch,)      Busemann mean
        mu_b:         (batch, d-1)  fibre mean
        log_sigma_tau:(batch,)      log std of tau
        log_sigma_b:  (batch,)      log scale of b
    """

    TAU_CLAMP  = 8.0    # |tau| clamped during sampling
    SIGMA_MIN  = 0.05
    SIGMA_MAX  = 5.0
    LOG_MGF_MAX = 10.0  # clamp log E[e^{-2*tau}] to prevent KL overflow

    def __init__(
        self,
        mu_tau: torch.Tensor,
        mu_b: torch.Tensor,
        log_sigma_tau: torch.Tensor,
        log_sigma_b: torch.Tensor,
    ):
        self.mu_tau = mu_tau
        self.mu_b   = mu_b
        self.sigma_tau = torch.clamp(torch.exp(log_sigma_tau), self.SIGMA_MIN, self.SIGMA_MAX)
        self.sigma_b   = torch.clamp(torch.exp(log_sigma_b),   self.SIGMA_MIN, self.SIGMA_MAX)
        self.d_fibre   = mu_b.shape[-1]   # d - 1

    def rsample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterised sample.

        Returns:
            tau: (batch,)
            b:   (batch, d-1)

        Coupling: b depends on tau through exp(tau). This must be kept intact
        for correct gradient flow through the reparameterisation.
        """
        eps1 = torch.randn_like(self.mu_tau)
        tau  = self.mu_tau + self.sigma_tau * eps1
        tau  = torch.clamp(tau, -self.TAU_CLAMP, self.TAU_CLAMP)

        eps2 = torch.randn_like(self.mu_b)
        # Conditional: b | tau ~ N(mu_b, e^{2*tau} * sigma_b^2 * I)
        b = self.mu_b + torch.exp(tau).unsqueeze(-1) * self.sigma_b.unsqueeze(-1) * eps2
        return tau, b

    def kl_divergence(self, other: "HorosphericalGaussian") -> torch.Tensor:
        """
        Exact closed-form KL[self || other].

        KL = height_KL + (d-1) * fibre_KL + drift_term

        height_KL: KL between the 1-D tau marginals (standard Gaussian KL)
        fibre_KL:  KL between the isotropic fibre scales (std-only Gaussian KL)
        drift_term: ||mu_b_q - mu_b_p||^2 / (2*sigma_b_p^2) * E_q[e^{-2*tau}]
                    where E_q[e^{-2*tau}] = exp(-2*mu_tau_q + 2*sigma_tau_q^2)
                    is the MGF of N(mu_tau_q, sigma_tau_q^2) evaluated at s = -2.
        """
        # --- Height KL: KL[N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)] ---
        height_kl = (
            torch.log(other.sigma_tau / self.sigma_tau)
            + (self.sigma_tau**2 + (self.mu_tau - other.mu_tau)**2)
              / (2.0 * other.sigma_tau**2)
            - 0.5
        )  # (batch,)

        # --- Fibre KL: (d-1) * KL[N(0, sigma_b_q^2) || N(0, sigma_b_p^2)] ---
        fibre_kl = self.d_fibre * (
            torch.log(other.sigma_b / self.sigma_b)
            + self.sigma_b**2 / (2.0 * other.sigma_b**2)
            - 0.5
        )  # (batch,)

        # --- Drift: ||mu_b_q - mu_b_p||^2 / (2*sigma_b_p^2) * E[e^{-2*tau}] ---
        mu_b_diff_sq = ((self.mu_b - other.mu_b)**2).sum(dim=-1)  # (batch,)
        # log MGF of N(mu_tau_q, sigma_tau_q^2) at s=-2: -2*mu_tau + 2*sigma_tau^2
        log_mgf = -2.0 * self.mu_tau + 2.0 * self.sigma_tau**2
        log_mgf = torch.clamp(log_mgf, max=self.LOG_MGF_MAX)
        drift = mu_b_diff_sq / (2.0 * other.sigma_b**2) * torch.exp(log_mgf)  # (batch,)

        return height_kl + fibre_kl + drift

    def grad_attenuation(self) -> torch.Tensor:
        """
        E_q[e^tau] = exp(mu_tau + sigma_tau^2/2) — the gradient attenuation diagnostic.
        A spike in this value indicates the tau-instability failure regime.
        """
        return torch.exp(self.mu_tau + 0.5 * self.sigma_tau**2).mean()

    @staticmethod
    def kl_mc_estimate(q: "HorosphericalGaussian", p: "HorosphericalGaussian",
                       n_samples: int = 100_000) -> float:
        """
        Monte Carlo KL[q||p] estimate for validation.

        Since _log_prob_dvol returns the normalized log density w.r.t. d-vol,
        KL_MC = E_q[log q_vol(z) - log p_vol(z)].
        Sampling z from q is valid because q and q_vol represent the same law.
        """
        with torch.no_grad():
            tau, b = q.rsample_n(n_samples)
            log_q  = q._log_prob_dvol(tau, b)
            log_p  = p._log_prob_dvol(tau, b)
            return (log_q - log_p).mean().item()

    def rsample_n(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """n iid samples from a single (non-batched) HG distribution."""
        eps1 = torch.randn(n)
        tau  = self.mu_tau + self.sigma_tau * eps1
        eps2 = torch.randn(n, self.d_fibre)
        b    = self.mu_b + torch.exp(tau).unsqueeze(-1) * self.sigma_b * eps2
        return tau, b

    def _log_prob_dvol(self, tau: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Normalized log density w.r.t. the Riemannian volume form d-vol.

        log q_vol(tau, b) = log_unnorm(tau, b) - log Z_q

        where log Z_q = 0.5*log(2*pi) + log(sigma_tau) + (d-1)/2*(log(2*pi) + 2*log(sigma_b))
        is the closed-form normaliser that follows from the key cancellation.
        """
        log_tau     = -0.5 * ((tau - self.mu_tau) / self.sigma_tau)**2
        exp_neg2tau = torch.exp(-2.0 * tau)
        diff_b      = b - self.mu_b
        log_b       = -0.5 * (diff_b**2).sum(-1) * exp_neg2tau / self.sigma_b**2
        log_Z = (
            0.5 * math.log(2.0 * math.pi) + torch.log(self.sigma_tau)
            + self.d_fibre / 2.0 * (math.log(2.0 * math.pi) + 2.0 * torch.log(self.sigma_b))
        )
        return log_tau + log_b - log_Z


class EuclideanGaussian:
    """Diagonal Gaussian N(mu, diag(sigma^2)) on R^d."""

    SIGMA_MIN = 0.05
    SIGMA_MAX = 5.0

    def __init__(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        """
        Args:
            mu:        (batch, d)
            log_sigma: (batch, d)  per-dimension log std
        """
        self.mu    = mu
        self.sigma = torch.clamp(torch.exp(log_sigma), self.SIGMA_MIN, self.SIGMA_MAX)

    def rsample(self) -> torch.Tensor:
        """Returns (batch, d) reparameterised sample."""
        return self.mu + self.sigma * torch.randn_like(self.mu)

    def kl_divergence(self, other: "EuclideanGaussian") -> torch.Tensor:
        """
        KL[self || other] summed over dimensions.  Returns (batch,).
        Standard diagonal Gaussian KL.
        """
        kl = (
            torch.log(other.sigma / self.sigma)
            + (self.sigma**2 + (self.mu - other.mu)**2) / (2.0 * other.sigma**2)
            - 0.5
        )
        return kl.sum(dim=-1)
