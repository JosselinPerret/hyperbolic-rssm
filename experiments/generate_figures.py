"""
Generate Plotly-compatible JSON from experiment results.

Reads results/capacity_test.json and results/structure_discovery.json,
then writes updated hyperbolic_figures.json (all existing charts plus the
new structure_discovery chart for Experiment 3).

Usage:
    python experiments/generate_figures.py \
        --cap  results/capacity_test.json \
        --disc results/structure_discovery.json \
        --out  ../path/to/src/components/graphs/hyperbolic_figures.json
"""

import json
import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Helper — load existing figures file to preserve charts we don't regenerate
# ---------------------------------------------------------------------------

def load_existing(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Rebuild capacity-test charts from real results
# ---------------------------------------------------------------------------

def build_capacity_curve(cap: dict) -> dict:
    dims  = cap["dims"]
    mse_h = [e["mse"] for e in cap["hyperbolic"]]
    mse_e = [e["mse"] for e in cap["euclidean"]]

    y_min = min(min(mse_h), min(mse_e)) - 0.3
    y_max = max(max(mse_h), max(mse_e)) + 0.3

    return {
        "data": [
            {
                "type": "scatter", "mode": "lines+markers",
                "name": "Hyperbolic H^d",
                "x": dims, "y": [round(v, 3) for v in mse_h],
                "line": {"color": "#4C72B0", "width": 2.5},
                "marker": {"size": 10, "symbol": "circle"},
                "hovertemplate": "d = %{x}<br>MSE = %{y:.3f}<extra>Hyperbolic</extra>",
            },
            {
                "type": "scatter", "mode": "lines+markers",
                "name": "Euclidean R^d",
                "x": dims, "y": [round(v, 3) for v in mse_e],
                "line": {"color": "#DD8452", "width": 2.5, "dash": "dash"},
                "marker": {"size": 10, "symbol": "square"},
                "hovertemplate": "d = %{x}<br>MSE = %{y:.3f}<extra>Euclidean</extra>",
            },
            {
                "type": "scatter", "mode": "lines",
                "name": "Sarkar threshold log\u2082(n) \u2248 10.4",
                "x": [10.4, 10.4], "y": [y_min, y_max],
                "line": {"color": "#888", "width": 1.5, "dash": "dot"},
                "hoverinfo": "skip",
            },
        ],
        "layout": {
            "title": {"text": "Capacity curve: H^d vs R^d", "font": {"size": 16}},
            "xaxis": {
                "title": "Latent dimension d",
                "type": "log",
                "tickvals": dims,
                "ticktext": [str(d) for d in dims],
            },
            "yaxis": {"title": "Reconstruction MSE (test set)", "range": [y_min, y_max]},
            "legend": {"x": 0.55, "y": 0.98},
            "hovermode": "x unified",
            "margin": {"t": 60, "b": 60, "l": 70, "r": 30},
        },
    }


def build_mse_gap(cap: dict) -> dict:
    dims  = cap["dims"]
    mse_h = [e["mse"] for e in cap["hyperbolic"]]
    mse_e = [e["mse"] for e in cap["euclidean"]]
    gaps  = [round(e - h, 4) for e, h in zip(mse_e, mse_h)]

    colors    = ["#4C72B0" if g > 0 else "#DD8452" for g in gaps]
    custom    = ["Hyperbolic wins" if g > 0 else "Euclidean wins" for g in gaps]
    x_labels  = [f"d={d}" for d in dims]

    return {
        "data": [
            {
                "type": "bar",
                "name": "MSE gap (Euclidean \u2212 Hyperbolic)",
                "x": x_labels,
                "y": gaps,
                "marker": {"color": colors},
                "hovertemplate": "%{x}<br>Gap = %{y:+.3f}<br>%{customdata}<extra></extra>",
                "customdata": custom,
                "text": [f"{g:+.3f}" for g in gaps],
                "textposition": "outside",
            }
        ],
        "layout": {
            "title": {"text": "Hyperbolic advantage by dimension", "font": {"size": 16}},
            "xaxis": {"title": "Latent dimension d"},
            "yaxis": {
                "title": "MSE gap (Euclidean \u2212 Hyperbolic)",
                "zeroline": True,
                "zerolinecolor": "#000",
                "zerolinewidth": 2,
            },
            "margin": {"t": 60, "b": 60, "l": 80, "r": 30},
        },
    }


def build_grad_attenuation(cap: dict) -> dict:
    dims     = cap["dims"]
    grad_att = [e["grad_att"] for e in cap["hyperbolic"]]
    mu_taus  = [None] * len(dims)  # not stored; use zeros as placeholder

    return {
        "data": [
            {
                "type": "scatter", "mode": "lines+markers",
                "name": "E[e^\u03c4] \u2014 gradient attenuation",
                "x": dims, "y": [round(v, 3) for v in grad_att],
                "yaxis": "y",
                "line": {"color": "#c0392b", "width": 2.5},
                "marker": {"size": 10, "symbol": "diamond", "color": "#c0392b"},
                "hovertemplate": "d = %{x}<br>E[e^\u03c4] = %{y:.3f}<extra></extra>",
            },
            {
                "type": "scatter", "mode": "lines",
                "name": "Ideal (E[e^\u03c4] = 1)",
                "x": [dims[0], dims[-1]], "y": [1, 1],
                "yaxis": "y",
                "line": {"color": "#c0392b", "width": 1, "dash": "dot"},
                "hoverinfo": "skip",
            },
        ],
        "layout": {
            "title": {"text": "\u03c4-collapse diagnosis: gradient attenuation", "font": {"size": 16}},
            "xaxis": {
                "title": "Latent dimension d",
                "type": "log",
                "tickvals": dims,
                "ticktext": [str(d) for d in dims],
            },
            "yaxis": {"title": "E[e^\u03c4] (effective b-gradient scale)"},
            "legend": {"x": 0.55, "y": 0.98},
            "margin": {"t": 60, "b": 60, "l": 70, "r": 30},
        },
    }


def build_linear_probes(cap: dict) -> dict:
    dims         = cap["dims"]
    tau_depth    = [e.get("probe_tau_depth", 0) for e in cap["hyperbolic"]]
    b_branch     = [e.get("probe_b_branch",  0) for e in cap["hyperbolic"]]
    z_depth      = [e.get("probe_z_depth",   0) for e in cap["euclidean"]]
    z_branch     = [e.get("probe_z_branch",  0) for e in cap["euclidean"]]

    return {
        "data": [
            {
                "type": "scatter", "mode": "lines+markers",
                "name": "Depth from \u03c4 (hyperbolic)",
                "x": dims, "y": [round(v, 3) for v in tau_depth],
                "line": {"color": "#4C72B0", "width": 2.5},
                "marker": {"size": 10},
            },
            {
                "type": "scatter", "mode": "lines+markers",
                "name": "Branch from b (hyperbolic, depth 1)",
                "x": dims, "y": [round(v, 3) for v in b_branch],
                "line": {"color": "#DD8452", "width": 2.5, "dash": "dash"},
                "marker": {"size": 10, "symbol": "diamond", "color": "#DD8452"},
            },
            {
                "type": "scatter", "mode": "lines+markers",
                "name": "Depth from z (Euclidean)",
                "x": dims, "y": [round(v, 3) for v in z_depth],
                "line": {"color": "#DD8452", "width": 2.5},
                "marker": {"size": 10, "symbol": "square"},
            },
            {
                "type": "scatter", "mode": "lines+markers",
                "name": "Branch from z (Euclidean, depth 1)",
                "x": dims, "y": [round(v, 3) for v in z_branch],
                "line": {"color": "#DD8452", "width": 2.5, "dash": "dash"},
                "marker": {"size": 10, "symbol": "cross"},
            },
        ],
        "layout": {
            "title": {"text": "Linear probe accuracies", "font": {"size": 16}},
            "xaxis": {"title": "Latent dimension d"},
            "yaxis": {"title": "Probe accuracy", "range": [0, 1.05]},
            "legend": {"x": 0.01, "y": 0.05},
            "margin": {"t": 60, "b": 60, "l": 70, "r": 30},
        },
    }


# ---------------------------------------------------------------------------
# New: Experiment 3 — Hierarchy Discovery Speed
# ---------------------------------------------------------------------------

def build_structure_discovery(disc: dict) -> dict:
    steps     = disc["eval_steps"]
    hyp_mean  = disc["hyp_rho_mean"]
    hyp_std   = disc["hyp_rho_std"]
    euc_mean  = disc["euc_rho_mean"]
    euc_std   = disc["euc_rho_std"]
    n_seeds   = disc["n_seeds"]

    # Upper/lower confidence bands
    hyp_upper = [round(m + s, 4) for m, s in zip(hyp_mean, hyp_std)]
    hyp_lower = [round(m - s, 4) for m, s in zip(hyp_mean, hyp_std)]
    euc_upper = [round(m + s, 4) for m, s in zip(euc_mean, euc_std)]
    euc_lower = [round(m - s, 4) for m, s in zip(euc_mean, euc_std)]

    return {
        "data": [
            # Hyperbolic confidence band
            {
                "type": "scatter", "mode": "lines",
                "name": "Hyperbolic \u00b1\u03c3",
                "x": steps + steps[::-1],
                "y": hyp_upper + hyp_lower[::-1],
                "fill": "toself",
                "fillcolor": "rgba(76,114,176,0.15)",
                "line": {"color": "transparent"},
                "showlegend": False,
                "hoverinfo": "skip",
            },
            # Euclidean confidence band
            {
                "type": "scatter", "mode": "lines",
                "name": "Euclidean \u00b1\u03c3",
                "x": steps + steps[::-1],
                "y": euc_upper + euc_lower[::-1],
                "fill": "toself",
                "fillcolor": "rgba(221,132,82,0.15)",
                "line": {"color": "transparent"},
                "showlegend": False,
                "hoverinfo": "skip",
            },
            # Hyperbolic mean
            {
                "type": "scatter", "mode": "lines",
                "name": f"Hyperbolic \u03c1\u03c4 (mean over {n_seeds} seeds)",
                "x": steps, "y": [round(v, 4) for v in hyp_mean],
                "line": {"color": "#4C72B0", "width": 2.5},
                "hovertemplate": "step=%{x}<br>\u03c1\u03c4=%{y:.3f}<extra>Hyperbolic</extra>",
            },
            # Euclidean mean
            {
                "type": "scatter", "mode": "lines",
                "name": f"Euclidean best-PC \u03c1 (mean over {n_seeds} seeds)",
                "x": steps, "y": [round(v, 4) for v in euc_mean],
                "line": {"color": "#DD8452", "width": 2.5, "dash": "dash"},
                "hovertemplate": "step=%{x}<br>\u03c1=%{y:.3f}<extra>Euclidean</extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Experiment 3: Hierarchy discovery speed",
                "font": {"size": 16},
            },
            "xaxis": {"title": "Training step"},
            "yaxis": {
                "title": "Depth\u2013representation Spearman \u03c1",
                "range": [0, 1.0],
            },
            "legend": {"x": 0.02, "y": 0.98},
            "hovermode": "x unified",
            "annotations": [
                {
                    "x": disc["eval_steps"][-1] * 0.6,
                    "y": 0.05,
                    "xref": "x", "yref": "y",
                    "text": (
                        f"d={disc['latent_dim']}, B={disc['env']['B']}, L={disc['env']['L']}<br>"
                        f"Shaded region: \u00b11\u03c3 over {n_seeds} seeds"
                    ),
                    "showarrow": False,
                    "font": {"size": 10, "color": "#555"},
                    "bgcolor": "rgba(255,255,255,0.7)",
                    "bordercolor": "#ccc",
                    "borderwidth": 1,
                }
            ],
            "margin": {"t": 60, "b": 60, "l": 70, "r": 30},
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap",  default="results/capacity_test.json")
    parser.add_argument("--disc", default="results/structure_discovery.json")
    parser.add_argument("--existing", default=None,
                        help="Path to existing hyperbolic_figures.json to update")
    parser.add_argument("--out",  default="results/hyperbolic_figures_new.json")
    args = parser.parse_args()

    with open(args.cap)  as f: cap  = json.load(f)
    with open(args.disc) as f: disc = json.load(f)

    # Start from existing file if provided (preserves charts we don't touch)
    figures = load_existing(args.existing) if args.existing else {}

    figures["capacity_curve"]      = build_capacity_curve(cap)
    figures["mse_gap"]             = build_mse_gap(cap)
    figures["grad_attenuation"]    = build_grad_attenuation(cap)
    figures["linear_probes"]       = build_linear_probes(cap)
    figures["structure_discovery"] = build_structure_discovery(disc)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(figures, f, indent=2)

    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
