"""
Master script — runs all H-RSSM experiments and generates article figures.

Usage:
    python run_experiments.py [--device cuda] [--quick] [--figures-out PATH] [--existing-figures PATH]

    --device        cuda / mps / cpu  (auto-detect if omitted)
    --quick         Smoke test: 500 steps, d=[4,16], 1 seed.
    --figures-out   Path to write updated hyperbolic_figures.json.
                    Defaults to results/hyperbolic_figures_new.json.
    --existing-figures  Path to existing figures.json to update in-place
                        (preserves charts that are not regenerated).

Example — update the portfolio article directly:
    python run_experiments.py --device cuda \\
        --figures-out ../portfolio/src/components/graphs/hyperbolic_figures.json
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from experiments.capacity_test       import run_capacity_test
from experiments.structure_discovery  import run_structure_discovery
from experiments.generate_figures    import (
    build_capacity_curve, build_mse_gap,
    build_grad_attenuation, build_linear_probes,
    build_structure_discovery, load_existing,
)
from experiments.train_utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",           default=None,
                        help="cuda / mps / cpu (auto-detect if omitted)")
    parser.add_argument("--quick",            action="store_true")
    parser.add_argument("--figures-out",      default="results/hyperbolic_figures_new.json")
    parser.add_argument("--existing-figures", default=None,
                        help="Existing hyperbolic_figures.json to update in-place")
    args = parser.parse_args()

    dev = get_device(args.device)
    print(f"Device: {dev}")

    n_steps_cap  = 500   if args.quick else 15_000
    n_steps_disc = 300   if args.quick else 2_000
    dims         = [4, 16] if args.quick else None
    n_seeds      = 1    if args.quick else 5
    eval_every   = 50   if args.quick else 100

    if args.quick:
        print("[quick mode] steps=500/300, dims=[4,16], 1 seed\n")

    os.makedirs("results", exist_ok=True)

    print("\n### Running capacity test ###")
    cap_results = run_capacity_test(
        dims=dims, n_steps=n_steps_cap, device=dev
    )

    print("\n### Running structure discovery ###")
    disc_results = run_structure_discovery(
        n_steps=n_steps_disc, n_seeds=n_seeds, eval_every=eval_every, device=dev
    )

    print("\n### Generating figures ###")
    figures = load_existing(args.existing_figures) if args.existing_figures else {}
    figures["capacity_curve"]      = build_capacity_curve(cap_results)
    figures["mse_gap"]             = build_mse_gap(cap_results)
    figures["grad_attenuation"]    = build_grad_attenuation(cap_results)
    figures["linear_probes"]       = build_linear_probes(cap_results)
    figures["structure_discovery"] = build_structure_discovery(disc_results)

    out_path = args.figures_out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(figures, f, indent=2)
    print(f"\nFigures written to {out_path}")


if __name__ == "__main__":
    main()
