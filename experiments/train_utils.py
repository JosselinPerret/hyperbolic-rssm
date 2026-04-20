"""
Shared training loop utilities for H-RSSM experiments.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable
from tqdm.auto import tqdm

from hrssm.world_model import elbo_loss


def get_device(device=None) -> torch.device:
    """Auto-select device if not specified: CUDA > MPS > CPU."""
    if device is not None:
        return torch.device(device) if isinstance(device, str) else device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    model: nn.Module,
    env,
    n_steps:     int   = 15_000,
    batch_size:  int   = 32,
    seq_len:     int   = 8,
    lr:          float = 3e-4,
    beta_final:  float = 1.0,
    beta_warmup: int   = 2_000,
    seed:        int   = 0,
    eval_every:  int   = 500,
    eval_fn:     Optional[Callable] = None,
    device=None,
    verbose:     bool  = True,
    desc:        str   = "Training",
) -> dict:
    """
    Train a world model on the tree MDP.

    Args:
        model:       HyperbolicWorldModel or EuclideanWorldModel
        env:         BaryTreeMDP instance
        n_steps:     number of gradient steps
        batch_size:  trajectories per step
        seq_len:     length of each trajectory
        lr:          Adam learning rate
        beta_final:  final KL weight (linear warmup from 0)
        beta_warmup: number of warmup steps for beta
        seed:        RNG seed for trajectory sampling
        eval_every:  evaluation interval in steps (0 = never)
        eval_fn:     callable(model, step) -> dict (optional)
        device:      torch.device or string; None = auto-detect
        verbose:     show tqdm progress bar
        desc:        label for the progress bar

    Returns:
        history dict: steps, recon_loss, kl_loss, eval_steps, eval_results
    """
    dev = get_device(device)
    model = model.to(dev)
    model.train()

    rng       = np.random.RandomState(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "steps":        [],
        "recon_loss":   [],
        "kl_loss":      [],
        "eval_steps":   [],
        "eval_results": [],
    }

    pbar = tqdm(range(1, n_steps + 1), desc=desc, disable=not verbose,
                dynamic_ncols=True)

    for step in pbar:
        beta = min(1.0, step / max(1, beta_warmup)) * beta_final

        obs_b, _, _ = env.sample_batch(batch_size, seq_len, rng)
        obs_t       = torch.tensor(obs_b, dtype=torch.float32).to(dev)

        model.train()
        optimizer.zero_grad()
        recons, kls, _ = model(obs_t)
        loss, recon_s, kl_s = elbo_loss(recons, obs_t, kls, beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()

        history["steps"].append(step)
        history["recon_loss"].append(recon_s)
        history["kl_loss"].append(kl_s)

        if step % max(1, n_steps // 20) == 0:
            pbar.set_postfix(recon=f"{recon_s:.4f}", kl=f"{kl_s:.3f}")

        if eval_every > 0 and step % eval_every == 0 and eval_fn is not None:
            result = eval_fn(model, step)
            history["eval_steps"].append(step)
            history["eval_results"].append(result)

    return history
