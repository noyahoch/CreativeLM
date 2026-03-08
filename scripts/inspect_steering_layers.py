#!/usr/bin/env python3
"""
Print per-layer steering stats from a steering_vectors.pt so you can choose a layer.

The "best" layer (saved as steer_layer) was chosen as the one with highest
rel_signal among layers with frac_positive >= 1.0. This script prints
rel_signal and rel_signal_vs_best (relative to that chosen base) for every
probed layer.

Usage:
  cd DLP && python scripts/inspect_steering_layers.py --vectors results/bridge_steering/<setup>/steering_vectors.pt
  python scripts/inspect_steering_layers.py --vectors path/to/steering_vectors.pt [--csv out.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# DLP package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print per-layer steering stats (rel_signal, frac_positive, etc.) from steering_vectors.pt",
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        required=True,
        help="Path to steering_vectors.pt",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="If set, write the table to this CSV (with rel_signal_vs_best column)",
    )
    args = parser.parse_args()

    if not args.vectors.exists():
        print(f"Error: not found: {args.vectors}", file=sys.stderr)
        return 1

    state = torch.load(args.vectors, map_location="cpu", weights_only=False)
    steer_layer = int(state["steer_layer"])
    stats_df = state.get("stats_df")

    if stats_df is None:
        print("Error: checkpoint has no stats_df (old format?).", file=sys.stderr)
        return 1

    import pandas as pd
    if not isinstance(stats_df, pd.DataFrame):
        stats_df = pd.DataFrame(stats_df)

    rel_best = stats_df.loc[stats_df["layer"] == steer_layer, "rel_signal"].iloc[0]
    stats_df = stats_df.copy()
    stats_df["rel_signal_vs_best"] = (stats_df["rel_signal"] / (rel_best + 1e-8)).round(4)
    stats_df["chosen"] = (stats_df["layer"] == steer_layer).map({True: "*", False: ""})

    print(f"Vectors: {args.vectors}")
    print(f"Chosen layer (base): {steer_layer}  (rel_signal = {rel_best:.4f})")
    print()
    print(stats_df.to_string(index=False))
    print()
    print("rel_signal_vs_best: rel_signal at this layer / rel_signal at chosen layer (1.0 = base).")
    print("Use --layer N in run_aut_inference.py to steer at layer N.")

    if args.csv:
        stats_df.to_csv(args.csv, index=False)
        print(f"Wrote: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
