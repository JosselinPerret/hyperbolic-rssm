"""
Experiment 3 — Geometric inductive bias across seeds.

Trains H^d and R^d models at d=16 with multiple seeds. Every EVAL_EVERY steps:
  - Hyperbolic: Spearman rho between mu_tau and ground-truth depth
  - Euclidean:  max |rho| between any PCA component and depth

Key question: does the geometric prior reduce variance across seeds even when
mean performance is comparable?

Writes results to results/structure_discovery.json.

Usage:
    python experiments/structure_discovery.py [--device cuda] [--quick]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from hrssm.tree_mdp    import BaryTreeMDP
from hrssm.world_model import HyperbolicWorldModel, EuclideanWorldModel, elbo_loss
from hrssm.metrics     import compute_rho_tau, compute_pc1_rho
from experiments.train_utils import get_device


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

LATENT_DIM  = 16
N_STEPS     = 2_000
BATCH_SIZE  = 32
SEQ_LEN     = 8
LR          = 3e-4
BETA_FINAL  = 1.0
BETA_WARMUP = 500
HIDDEN_DIM  = 256
EVAL_EVERY  = 100
N_SEEDS     = 5
SEED_BASE   = 0

ENV_B = 4
ENV_L = 5


# ---------------------------------------------------------------------------

def run_structure_discovery(n_steps=N_STEPS, n_seeds=N_SEEDS, hidden_dim=HIDDEN_DIM,
                            eval_every=EVAL_EVERY, device=None):
    dev = get_device(device)

    print("=" * 60)
    print("Experiment 3: Geometric inductive bias across seeds")
    print(f"d={LATENT_DIM}, {n_steps} steps, {n_seeds} seeds, device={dev}")
    print("=" * 60)

    env = BaryTreeMDP(B=ENV_B, L=ENV_L, obs_dim=64, seed=42)

    hyp_rho_all = []   # (n_seeds, n_checkpoints)
    euc_rho_all = []
    eval_steps  = list(range(eval_every, n_steps + 1, eval_every))

    for seed_i in range(n_seeds):
        seed = SEED_BASE + seed_i
        print(f"\n  Seed {seed_i + 1}/{n_seeds}")

        hyp_rhos, euc_rhos = [], []
        rng_train = np.random.RandomState(seed)

        # ---- Hyperbolic ----
        torch.manual_seed(seed)
        hyp = HyperbolicWorldModel(obs_dim=64, latent_dim=LATENT_DIM,
                                   hidden_dim=hidden_dim).to(dev)
        opt_h = torch.optim.Adam(hyp.parameters(), lr=LR)

        for step in tqdm(range(1, n_steps + 1), desc=f"  Hyp seed={seed_i}", leave=False):
            beta = min(1.0, step / max(1, BETA_WARMUP)) * BETA_FINAL
            obs_b, _, _ = env.sample_batch(BATCH_SIZE, SEQ_LEN, rng_train)
            obs_t = torch.tensor(obs_b, dtype=torch.float32).to(dev)
            hyp.train()
            opt_h.zero_grad()
            recons, kls, _ = hyp(obs_t)
            loss, _, _ = elbo_loss(recons, obs_t, kls, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(hyp.parameters(), 100.0)
            opt_h.step()

            if step % eval_every == 0:
                rho = compute_rho_tau(hyp, env, n_traj=40, seq_len=SEQ_LEN,
                                      seed=seed + 100, device=dev)
                hyp_rhos.append(rho)

        hyp_rho_all.append(hyp_rhos)

        # ---- Euclidean (same trajectories) ----
        torch.manual_seed(seed)
        euc = EuclideanWorldModel(obs_dim=64, latent_dim=LATENT_DIM,
                                  hidden_dim=hidden_dim).to(dev)
        opt_e = torch.optim.Adam(euc.parameters(), lr=LR)
        rng_train = np.random.RandomState(seed)   # reset to same trajectories

        for step in tqdm(range(1, n_steps + 1), desc=f"  Euc seed={seed_i}", leave=False):
            beta = min(1.0, step / max(1, BETA_WARMUP)) * BETA_FINAL
            obs_b, _, _ = env.sample_batch(BATCH_SIZE, SEQ_LEN, rng_train)
            obs_t = torch.tensor(obs_b, dtype=torch.float32).to(dev)
            euc.train()
            opt_e.zero_grad()
            recons, kls, _ = euc(obs_t)
            loss, _, _ = elbo_loss(recons, obs_t, kls, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(euc.parameters(), 100.0)
            opt_e.step()

            if step % eval_every == 0:
                rho = compute_pc1_rho(euc, env, n_traj=40, seq_len=SEQ_LEN,
                                      seed=seed + 100, device=dev)
                euc_rhos.append(rho)

        euc_rho_all.append(euc_rhos)

    # ---- Aggregate ----
    hyp_arr  = np.array(hyp_rho_all)
    euc_arr  = np.array(euc_rho_all)
    hyp_mean = np.mean(hyp_arr, axis=0).tolist()
    euc_mean = np.mean(euc_arr, axis=0).tolist()
    hyp_std  = np.std(hyp_arr,  axis=0).tolist()
    euc_std  = np.std(euc_arr,  axis=0).tolist()

    results = {
        "eval_steps":         eval_steps,
        "hyp_rho_mean":       [round(v, 4) for v in hyp_mean],
        "hyp_rho_std":        [round(v, 4) for v in hyp_std],
        "euc_rho_mean":       [round(v, 4) for v in euc_mean],
        "euc_rho_std":        [round(v, 4) for v in euc_std],
        "n_seeds":            n_seeds,
        "latent_dim":         LATENT_DIM,
        "env":                {"B": ENV_B, "L": ENV_L},
        "hyp_rho_per_seed":   [[round(v, 4) for v in r] for r in hyp_rho_all],
        "euc_rho_per_seed":   [[round(v, 4) for v in r] for r in euc_rho_all],
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/structure_discovery.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    var_ratio = np.mean(euc_std) / (np.mean(hyp_std) + 1e-8)
    print(f"\nFinal step ({n_steps}) summary:")
    print(f"  Hyperbolic: {hyp_mean[-1]:.3f} ± {hyp_std[-1]:.3f}")
    print(f"  Euclidean:  {euc_mean[-1]:.3f} ± {euc_std[-1]:.3f}")
    print(f"  Variance ratio (Euc/Hyp): {var_ratio:.1f}×")
    print(f"Results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 300 steps, 1 seed, eval every 50")
    args = parser.parse_args()

    n_steps    = 300 if args.quick else N_STEPS
    n_seeds    = 1   if args.quick else N_SEEDS
    eval_every = 50  if args.quick else EVAL_EVERY

    run_structure_discovery(n_steps=n_steps, n_seeds=n_seeds,
                            eval_every=eval_every, device=args.device)


if __name__ == "__main__":
    main()
