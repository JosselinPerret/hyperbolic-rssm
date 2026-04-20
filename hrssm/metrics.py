"""
Evaluation metrics for H-RSSM experiments.

  compute_rho_tau       — Spearman correlation between posterior mu_tau and node depth.
  compute_pc1_rho       — Same for Euclidean model: best PCA component vs depth.
  compute_linear_probes — Depth probe (tau -> depth) and branch probe (b -> branch).
  compute_test_mse      — Reconstruction MSE on held-out trajectories.
  compute_grad_attenuation — E[e^tau] instability diagnostic.

All functions accept an optional `device` argument (default: cpu).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Optional


def _get_device(device):
    if device is None:
        return torch.device("cpu")
    return torch.device(device) if isinstance(device, str) else device


def _collect_hyperbolic(model, env, n_traj: int = 80, seq_len: int = 10,
                        seed: int = 0, device=None):
    """Run model in eval mode; return mu_tau, mu_b arrays and ground-truth labels."""
    dev = _get_device(device)
    model.eval()
    rng = np.random.RandomState(seed)
    mu_taus, mu_bs, depths, branches = [], [], [], []

    with torch.no_grad():
        for _ in range(n_traj):
            obs_b, dep_b, br_b = env.sample_batch(1, seq_len, rng)
            obs_t = torch.tensor(obs_b, dtype=torch.float32).to(dev)
            _, _, info = model(obs_t)
            mu_taus.append(info["mu_tau"][0].cpu().numpy())     # (T,)
            mu_bs.append(info["mu_b"][0].cpu().numpy())         # (T, d-1)
            depths.append(dep_b[0])                             # (T,)
            branches.append(br_b[0])                            # (T,)

    return (
        np.concatenate(mu_taus),             # (N,)
        np.vstack(mu_bs),                    # (N, d-1)
        np.concatenate(depths).astype(int),  # (N,)
        np.concatenate(branches).astype(int),
    )


def _collect_euclidean(model, env, n_traj: int = 80, seq_len: int = 10,
                       seed: int = 0, device=None):
    dev = _get_device(device)
    model.eval()
    rng = np.random.RandomState(seed)
    mus, depths, branches = [], [], []

    with torch.no_grad():
        for _ in range(n_traj):
            obs_b, dep_b, br_b = env.sample_batch(1, seq_len, rng)
            obs_t = torch.tensor(obs_b, dtype=torch.float32).to(dev)
            _, _, info = model(obs_t)
            mus.append(info["mu"][0].cpu().numpy())
            depths.append(dep_b[0])
            branches.append(br_b[0])

    return (
        np.vstack(mus),
        np.concatenate(depths).astype(int),
        np.concatenate(branches).astype(int),
    )


# ---------------------------------------------------------------------------

def compute_rho_tau(model, env, n_traj: int = 80, seq_len: int = 10,
                    seed: int = 0, device=None) -> float:
    """Spearman rho between E_q[tau] and ground-truth node depth (hyperbolic model)."""
    mu_taus, _, depths, _ = _collect_hyperbolic(model, env, n_traj, seq_len, seed, device)
    rho, _ = spearmanr(mu_taus, depths)
    return float(rho)


def compute_pc1_rho(model, env, n_traj: int = 80, seq_len: int = 10,
                    seed: int = 0, device=None) -> float:
    """
    Best single-PC vs depth correlation for the Euclidean model.
    Fits PCA on the posterior means and returns max |rho| over all components.
    """
    mus, depths, _ = _collect_euclidean(model, env, n_traj, seq_len, seed, device)
    pca = PCA()
    pcs = pca.fit_transform(mus)
    rhos = [abs(spearmanr(pcs[:, i], depths)[0]) for i in range(min(pcs.shape[1], 8))]
    return float(max(rhos))


def compute_linear_probes(
    model,
    env,
    n_traj: int = 100,
    seq_len: int = 10,
    seed: int = 0,
    device=None,
) -> Dict[str, float]:
    """
    Fit logistic regression probes and return accuracy.

    Hyperbolic returns: tau_to_depth, b_to_branch
    Euclidean returns:  z_to_depth, z_to_branch
    """
    is_hyp = hasattr(model, "_parse_hg")

    if is_hyp:
        mu_taus, mu_bs, depths, branches = _collect_hyperbolic(
            model, env, n_traj, seq_len, seed, device
        )

        # Depth probe: tau -> depth
        sc = StandardScaler()
        X_tau = sc.fit_transform(mu_taus.reshape(-1, 1))
        lr_depth = LogisticRegression(max_iter=500, C=1.0)
        lr_depth.fit(X_tau, depths)
        tau_acc = lr_depth.score(X_tau, depths)

        # Branch probe: b -> branch_id (only nodes at depth 1)
        mask = depths == 1
        if mask.sum() > 20:
            sc2 = StandardScaler()
            X_b  = sc2.fit_transform(mu_bs[mask])
            y_br = branches[mask]
            lr_branch = LogisticRegression(max_iter=500, C=1.0)
            lr_branch.fit(X_b, y_br)
            b_acc = lr_branch.score(X_b, y_br)
        else:
            b_acc = float("nan")

        return {"tau_to_depth": float(tau_acc), "b_to_branch": float(b_acc)}

    else:
        mus, depths, branches = _collect_euclidean(model, env, n_traj, seq_len, seed, device)

        sc = StandardScaler()
        X  = sc.fit_transform(mus)
        lr_depth = LogisticRegression(max_iter=500, C=1.0)
        lr_depth.fit(X, depths)
        z_acc = lr_depth.score(X, depths)

        mask = depths == 1
        if mask.sum() > 20:
            sc2 = StandardScaler()
            X_b  = sc2.fit_transform(mus[mask])
            y_br = branches[mask]
            lr_branch = LogisticRegression(max_iter=500, C=1.0)
            lr_branch.fit(X_b, y_br)
            b_acc = lr_branch.score(X_b, y_br)
        else:
            b_acc = float("nan")

        return {"z_to_depth": float(z_acc), "z_to_branch": float(b_acc)}


def compute_test_mse(
    model,
    env,
    n_traj: int = 100,
    seq_len: int = 10,
    seed: int = 99,
    device=None,
) -> float:
    """MSE on held-out trajectories (posterior z, same observations as input)."""
    dev = _get_device(device)
    model.eval()
    rng = np.random.RandomState(seed)
    total_mse, n = 0.0, 0

    with torch.no_grad():
        for _ in range(n_traj):
            obs_b, _, _ = env.sample_batch(1, seq_len, rng)
            obs_t = torch.tensor(obs_b, dtype=torch.float32).to(dev)
            recons, _, _ = model(obs_t)
            total_mse += F.mse_loss(recons, obs_t).item()
            n += 1

    return total_mse / n


def compute_grad_attenuation(model, env, n_traj: int = 20, seq_len: int = 8,
                             device=None) -> float:
    """
    E[e^{mu_tau + 0.5*sigma_tau^2}] — gradient attenuation diagnostic (hyperbolic only).
    A value >> 1 indicates the tau-instability regime.
    """
    dev = _get_device(device)
    model.eval()
    rng = np.random.RandomState(7)
    vals = []

    with torch.no_grad():
        for _ in range(n_traj):
            obs_b, _, _ = env.sample_batch(1, seq_len, rng)
            obs_t = torch.tensor(obs_b, dtype=torch.float32).to(dev)
            _, _, info = model(obs_t)
            mu_t  = info["mu_tau"][0].cpu().numpy()
            sig_t = info["sigma_tau"][0].cpu().numpy()
            vals.extend(np.exp(mu_t + 0.5 * sig_t**2).tolist())

    return float(np.mean(vals))
