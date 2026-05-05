# H-RSSM: Hyperbolic Recurrent State-Space Model
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/JosselinPerret/hyperbolic-rssm)

---

Implementation of the H-RSSM experiments described in the
[portfolio article](https://josselinperret.github.io/projects/h-rssm).

The project introduces the **Horospherical Gaussian** — a distribution on hyperbolic space
$\mathbb{H}^d$ with a closed-form $O(d)$ KL divergence — and uses it as the latent space
of a world model trained with the ELBO on a synthetic tree-structured environment.

---

## Key contribution

Standard variational inference on $\mathbb{H}^d$ requires Monte Carlo KL estimation
(too slow for RL training). The Horospherical Gaussian admits an exact KL:

$$\mathrm{KL}[q\|p] = \underbrace{\text{height KL}}_{\tau\text{ marginals}} + (d-1)\underbrace{\text{fibre KL}}_{\sigma_b\text{ terms}} + \underbrace{\frac{\|\mu_b^q - \mu_b^p\|^2}{2(\sigma_b^p)^2} e^{-2\mu_\tau^q + 2(\sigma_\tau^q)^2}}_{\text{drift term}}$$

The cancellation is exact (validated to < 0.01% relative error vs Monte Carlo at 500k samples).

---

## Repository structure

```
hrssm/
├── distributions.py    HorosphericalGaussian + EuclideanGaussian
├── tree_mdp.py         B-ary tree MDP (ground-truth depth never shown to model)
├── world_model.py      GRU-RSSM (hyperbolic and Euclidean variants)
├── metrics.py          Spearman ρ, linear probes, gradient attenuation
└── extensions.py       Next-step variants: separate β, geometry-aware decoder

experiments/
├── train_utils.py           Shared training loop (GPU + tqdm + KL warmup)
├── capacity_test.py         Exp 1 & 2: dimensional sweep d ∈ {2,4,8,16,32}
├── structure_discovery.py   Exp 3: multi-seed consistency
└── generate_figures.py      Plotly JSON for portfolio charts

h_rssm_gpu.ipynb             Self-contained notebook: KL validation + all 6 experiments
                             (runs standalone on Kaggle / Colab, no local install needed)
results/                     Generated outputs (gitignored except .gitkeep)
run_experiments.py           Master CLI script
```

---

## Quick start

```bash
git clone https://github.com/JosselinPerret/hyperbolic-rssm.git
cd hyperbolic-rssm
pip install -r requirements.txt

# Smoke test (~2 min, CPU)
python run_experiments.py --quick

# Full run on GPU (~2 h)
python run_experiments.py --device cuda

# Update portfolio charts directly
python run_experiments.py --device cuda \
    --figures-out ../portfolio/src/components/graphs/hyperbolic_figures.json
```

### Notebook

Open `h_rssm_gpu.ipynb` in Jupyter or Google Colab.
The notebook walks through the theory, validates the KL formula, runs all
experiments, and exports a drop-in `hyperbolic_figures_NEW.json` and
`article_snippets.txt` for updating the article.

---

## Experiments

| # | Name | Key result |
|---|------|-----------|
| 1 | τ spontaneously encodes depth | $|\rho_\tau| \approx 0.31$ at $d=16$; hyperbolic beats Euclidean (0.275) |
| 2 | Dimensional sweep | MSE nearly identical (~0.405); depth probe stable at 0.68 across all $d$ |
| 3 | Multi-seed consistency | Comparable variance across 5 seeds; hyperbolic shows cleaner trajectory |
| 4* | Separate $\beta_\tau / \beta_b$ | Decoupling KL weights improves branch encoding in $b$ |
| 5* | Longer training (50k steps) | Does the MSE gap close? Does $\rho_\tau$ keep improving? |
| 6* | Geometry-aware decoder | Adding $e^\tau$ and $\|b\|^2 e^{-2\tau}$ as decoder inputs |

*Next-step experiments implemented in `hrssm/extensions.py` and `h_rssm_gpu.ipynb`.

---

## Results (15 000 steps, GPU, seed=42)

| $d$ | MSE hyp | MSE euc | $\mathbb{E}[e^\tau]$ | $\rho_\tau$ | Probe $\tau \to \ell$ |
|-----|---------|---------|----------------------|-------------|----------------------|
|  2  | 0.405   | 0.405   | 1.04                 | +0.321      | 0.68                 |
|  4  | 0.405   | 0.405   | 1.64                 | +0.266      | 0.68                 |
|  8  | 0.405   | 0.405   | 1.07                 | +0.285      | 0.68                 |
| 16  | 0.406   | 0.405   | 1.04                 | +0.306      | 0.68                 |
| 32  | 0.405   | 0.405   | 0.73                 | +0.277      | 0.68                 |

Euclidean best-PC ρ at d=16: 0.275 (hyperbolic wins by a modest margin).
Branch probe ($b \to$ branch) did not converge — fibre KL suppresses σ_b.

---

## Module API

```python
from hrssm import (
    BaryTreeMDP,
    HyperbolicWorldModel, EuclideanWorldModel,
    HorosphericalGaussian,
    compute_rho_tau, compute_linear_probes, compute_test_mse,
    HyperbolicWorldModelSeparateBeta,    # Exp 4
    HyperbolicWorldModelAwareDecoder,    # Exp 6
)
from experiments.train_utils import train_model, get_device

# Build environment
env = BaryTreeMDP(B=4, L=5)

# Train
dev   = get_device()   # auto: CUDA > MPS > CPU
model = HyperbolicWorldModel(obs_dim=64, latent_dim=16, hidden_dim=256).to(dev)
train_model(model, env, n_steps=15_000, device=dev)

# Evaluate
rho = compute_rho_tau(model, env, device=dev)
```

---

## Reparameterisation (critical coupling)

```python
# tau ~ N(mu_tau, sigma_tau^2)
tau = mu_tau + sigma_tau * eps1

# b | tau ~ N(mu_b, e^{2*tau} * sigma_b^2 * I)  ← b depends on tau!
b   = mu_b  + exp(tau) * sigma_b * eps2
```

This coupling must be preserved for correct gradient flow. The `HorosphericalGaussian.rsample()`
method handles this automatically.

---

## Citation / reference

```
@misc{perret2026hrssm,
  author = {Josselin Perret},
  title  = {H-RSSM: Hyperbolic Recurrent State-Space Model},
  year   = {2026},
  url    = {https://josselinperret.github.io/projects/h-rssm}
}
```
