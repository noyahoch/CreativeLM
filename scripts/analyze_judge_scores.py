#!/usr/bin/env python3
"""
Post-judgment analysis — fair comparison of scored uses across methods.

Reads *_scores.csv files (output of run_vllm_judge.py) and produces:
  1. Raw summary     — mean scores per method (no adjustment)
  2. Top-K summary   — for each object, take top-K uses per method
                       (K = min uses across methods for that object),
                       ranked by --rank-by (default: judge_novelty)
  3. Fixed-K summary — take exactly --top-k best uses per object per method

This makes the comparison fair: "best 5 vs best 5" instead of "5 vs 300".

Usage:
  cd DLP

  # Read all score CSVs in a folder
  python scripts/analyze_judge_scores.py \
    --input-dir results/judge/all_inference \
    --output-dir results/judge/all_inference/analysis

  # Read specific score files
  python scripts/analyze_judge_scores.py \
    -i results/judge/compare/baseline_scores.csv \
    -i results/judge/compare/steered_a1.0_scores.csv \
    -o results/judge/compare/analysis

  # Use a fixed K (best 5 per object per method)
  python scripts/analyze_judge_scores.py \
    --input-dir results/judge/all_inference \
    -o results/judge/all_inference/analysis \
    --top-k 5

Output:
  raw_summary.csv     — plain mean/median per method (all uses)
  topk_summary.csv    — equalized comparison (top-K per object, K = min across methods)
  fixed_k_summary.csv — (only with --top-k) top-K per object per method, fixed K
  comparison.txt      — side-by-side printout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Loading ───────────────────────────────────────────────────────────────────


def load_score_csvs(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        for c in ["judge_novelty", "judge_usability", "judge_overall"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
        print(f"  {p.name}: {len(df)} rows, method={df['method'].iloc[0] if 'method' in df.columns else '?'}")
    return pd.concat(dfs, ignore_index=True)


# ── Aggregation helper ────────────────────────────────────────────────────────


def _agg_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    has_usability = "judge_usability" in df.columns and df["judge_usability"].notna().any()

    agg = dict(
        n_uses=("judge_overall", "count"),
        n_objects=("object", "nunique"),
        mean_overall=("judge_overall", "mean"),
        mean_novelty=("judge_novelty", "mean"),
    )
    if has_usability:
        agg.update(
            mean_usability=("judge_usability", "mean"),
            median_novelty=("judge_novelty", lambda s: float(np.nanpercentile(s.dropna(), 50)) if s.notna().any() else np.nan),
            median_usability=("judge_usability", lambda s: float(np.nanpercentile(s.dropna(), 50)) if s.notna().any() else np.nan),
        )

    summary = (
        df.groupby("method", dropna=False)
        .agg(**agg)
        .round(3)
        .reset_index()
    )

    baseline_nov = None
    if "baseline" in summary["method"].values:
        baseline_nov = summary.set_index("method").loc["baseline", "mean_novelty"]
    elif summary["method"].str.contains("baseline").any():
        baseline_rows = summary[summary["method"].str.contains("baseline")]
        if len(baseline_rows) == 1:
            baseline_nov = baseline_rows.iloc[0]["mean_novelty"]

    if baseline_nov is not None:
        summary["delta_novelty"] = (summary["mean_novelty"] - baseline_nov).round(3)

    summary = summary.sort_values("mean_novelty", ascending=False)
    summary.attrs["label"] = label
    return summary


# ── Equalization strategies ───────────────────────────────────────────────────


def equalize_top_k_per_object(
    scored_df: pd.DataFrame,
    rank_by: str = "judge_novelty",
) -> pd.DataFrame:
    """For each object, find K = min(uses across methods), then take top-K per method."""
    scored = scored_df.dropna(subset=[rank_by]).copy()

    counts = scored.groupby(["object", "method"]).size().reset_index(name="n")
    min_per_obj = counts.groupby("object")["n"].min().to_dict()

    parts = []
    for (obj, method), group in scored.groupby(["object", "method"]):
        k = min_per_obj.get(obj, len(group))
        top = group.nlargest(k, rank_by)
        parts.append(top)

    if not parts:
        return scored.iloc[:0]
    return pd.concat(parts, ignore_index=True)


def fixed_top_k(
    scored_df: pd.DataFrame,
    k: int,
    rank_by: str = "judge_novelty",
) -> pd.DataFrame:
    """For each (object, method), take the top-K uses."""
    scored = scored_df.dropna(subset=[rank_by]).copy()
    parts = []
    for (obj, method), group in scored.groupby(["object", "method"]):
        top = group.nlargest(min(k, len(group)), rank_by)
        parts.append(top)
    if not parts:
        return scored.iloc[:0]
    return pd.concat(parts, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-judgment analysis — fair comparison across methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", action="append", dest="inputs", type=Path,
        help="Path to *_scores.csv (repeatable)",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Load all *_scores.csv from this directory",
    )
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument(
        "--rank-by", default="judge_novelty",
        choices=["judge_novelty", "judge_usability", "judge_overall"],
        help="Column to rank uses by when selecting top-K (default: judge_novelty)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Fixed K: take best K uses per object per method (optional)",
    )

    args = parser.parse_args()

    # Resolve inputs
    if args.input_dir:
        input_paths = sorted(args.input_dir.glob("*_scores.csv"))
        if not input_paths:
            print(f"Error: no *_scores.csv in {args.input_dir}", file=sys.stderr)
            return 1
    elif args.inputs:
        input_paths = args.inputs
    else:
        print("Error: provide --input FILE(s) or --input-dir DIR", file=sys.stderr)
        return 1

    print(f"Loading {len(input_paths)} score file(s)...")
    all_scores = load_score_csvs(input_paths)
    methods = all_scores["method"].unique()
    objects = all_scores["object"].unique()
    print(f"  {len(all_scores)} total scored uses, {len(methods)} methods, {len(objects)} objects\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw summary
    raw = _agg_summary(all_scores, "Raw (all uses)")
    raw_path = args.output_dir / "raw_summary.csv"
    raw.to_csv(raw_path, index=False)

    # 2. Equalized top-K (K = min per object)
    eq_df = equalize_top_k_per_object(all_scores, rank_by=args.rank_by)
    eq_summary = _agg_summary(eq_df, f"Equalized top-K (K=min per object, ranked by {args.rank_by})")
    eq_path = args.output_dir / "topk_summary.csv"
    eq_summary.to_csv(eq_path, index=False)

    # 3. Fixed top-K (optional)
    fixed_summary = None
    if args.top_k:
        fk_df = fixed_top_k(all_scores, k=args.top_k, rank_by=args.rank_by)
        fixed_summary = _agg_summary(fk_df, f"Fixed top-{args.top_k} per object (ranked by {args.rank_by})")
        fk_path = args.output_dir / "fixed_k_summary.csv"
        fixed_summary.to_csv(fk_path, index=False)

    # Print comparison
    sep = "=" * 72
    lines = []
    for label, summary in [
        ("RAW (all uses)", raw),
        (f"EQUALIZED top-K (K = min per object, rank by {args.rank_by})", eq_summary),
    ]:
        lines.append(f"\n{sep}\n{label}\n{sep}")
        lines.append(summary.to_string(index=False))

    if fixed_summary is not None:
        lines.append(f"\n{sep}\nFIXED top-{args.top_k} per object (rank by {args.rank_by})\n{sep}")
        lines.append(fixed_summary.to_string(index=False))

    comparison_text = "\n".join(lines)
    print(comparison_text)

    comp_path = args.output_dir / "comparison.txt"
    comp_path.write_text(comparison_text + "\n")
    print(f"\nSaved: {raw_path}")
    print(f"       {eq_path}")
    if args.top_k:
        print(f"       {args.output_dir / 'fixed_k_summary.csv'}")
    print(f"       {comp_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
