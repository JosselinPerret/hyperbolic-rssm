"""
Experiments 1 & 2 — Capacity sweep: H^d vs R^d under dimensional constraint.

Sweeps latent_dim in {2, 4, 8, 16, 32} and trains both models for 15 000 steps.
For each dimension records:
  - Test reconstruction MSE
  - Spearman rho_tau (hyperbolic) / best-PC rho (Euclidean)
  - Linear probe accuracies (tau/b -> depth/branch)
  - Gradient attenuation E[e^tau]

Writes results to results/capacity_test.json.

Usage:
    python experiments/capacity_test.py [--device cuda] [--quick]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import argparse
import numpy as np

from hrssm.tree_mdp    import BaryTreeMDP
from hrssm.world_model import HyperbolicWorldModel, EuclideanWorldModel
from hrssm.metrics     import (
    compute_rho_tau, compute_pc1_rho,
    compute_linear_probes, compute_test_mse,
    compute_grad_attenuation,
)
from experiments.train_utils import train_model, get_device


# ---------------------------------------------------------------------------
# Default hyperparameters (match article results)
# ---------------------------------------------------------------------------

DIMS        = [2, 4, 8, 16, 32]
N_STEPS     = 15_000
BATCH_SIZE  = 32
SEQ_LEN     = 8
LR          = 3e-4
BETA_FINAL  = 1.0
BETA_WARMUP = 2_000
HIDDEN_DIM  = 256
SEED        = 42

ENV_B = 4
ENV_L = 5


# ---------------------------------------------------------------------------

def run_capacity_test(dims=None, n_steps=N_STEPS, hidden_dim=HIDDEN_DIM,
                      device=None, seed=SEED):
    dims = dims or DIMS
    dev  = get_device(device)

    print("=" * 60)
    print("Capacity test: H^d vs R^d")
    print(f"Tree: B={ENV_B}, L={ENV_L} ({int((ENV_B**(ENV_L+1)-1)/(ENV_B-1))} nodes)")
    print(f"Steps: {n_steps}  |  device: {dev}  |  hidden_dim: {hidden_dim}")
    print("=" * 60)

    env = BaryTreeMDP(B=ENV_B, L=ENV_L, obs_dim=64, seed=seed)

    results = {"dims": dims, "hyperbolic": [], "euclidean": []}

    for d in dims:
        print(f"\n--- d = {d} ---")

        # ---- Hyperbolic ----
        t0  = time.time()
        hyp = HyperbolicWorldModel(obs_dim=64, latent_dim=d, hidden_dim=hidden_dim)
        train_model(hyp, env,
                    n_steps=n_steps, batch_size=BATCH_SIZE,
                    seq_len=SEQ_LEN, lr=LR,
                    beta_final=BETA_FINAL, beta_warmup=BETA_WARMUP,
                    seed=seed, device=dev, desc=f"Hyp d={d}")
        t_hyp = time.time() - t0

        mse_hyp    = compute_test_mse(hyp, env, device=dev)
        rho_hyp    = compute_rho_tau(hyp, env, device=dev)
        grad_att   = compute_grad_attenuation(hyp, env, device=dev)
        probes_hyp = compute_linear_probes(hyp, env, device=dev)

        hyp_entry = {
            "d":               d,
            "mse":             round(mse_hyp, 4),
            "rho_tau":         round(rho_hyp, 4),
            "grad_att":        round(grad_att, 4),
            "probe_tau_depth": round(probes_hyp.get("tau_to_depth", 0), 4),
            "probe_b_branch":  round(probes_hyp.get("b_to_branch", 0), 4),
            "train_time_s":    round(t_hyp, 1),
        }
        results["hyperbolic"].append(hyp_entry)
        print(f"  Hyp  MSE={mse_hyp:.4f}  rho_tau={rho_hyp:.3f}  "
              f"E[e^tau]={grad_att:.3f}  probe_depth={probes_hyp.get('tau_to_depth',0):.3f}")

        # ---- Euclidean ----
        t0  = time.time()
        euc = EuclideanWorldModel(obs_dim=64, latent_dim=d, hidden_dim=hidden_dim)
        train_model(euc, env,
                    n_steps=n_steps, batch_size=BATCH_SIZE,
                    seq_len=SEQ_LEN, lr=LR,
                    beta_final=BETA_FINAL, beta_warmup=BETA_WARMUP,
                    seed=seed, device=dev, desc=f"Euc d={d}")
        t_euc = time.time() - t0

        mse_euc    = compute_test_mse(euc, env, device=dev)
        rho_euc    = compute_pc1_rho(euc, env, device=dev)
        probes_euc = compute_linear_probes(euc, env, device=dev)

        euc_entry = {
            "d":              d,
            "mse":            round(mse_euc, 4),
            "rho_pc1":        round(rho_euc, 4),
            "probe_z_depth":  round(probes_euc.get("z_to_depth", 0), 4),
            "probe_z_branch": round(probes_euc.get("z_to_branch", 0), 4),
            "train_time_s":   round(t_euc, 1),
        }
        results["euclidean"].append(euc_entry)
        print(f"  Euc  MSE={mse_euc:.4f}  rho_pc1={rho_euc:.3f}  "
              f"probe_depth={probes_euc.get('z_to_depth',0):.3f}")

    os.makedirs("results", exist_ok=True)
    out_path = "results/capacity_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detect if omitted)")
    parser.add_argument("--quick", action="store_true", help="Smoke test: 500 steps, d=[4,16]")
    args = parser.parse_args()

    n_steps = 500  if args.quick else N_STEPS
    dims    = [4, 16] if args.quick else DIMS

    run_capacity_test(dims=dims, n_steps=n_steps, device=args.device)


if __name__ == "__main__":
    main()
