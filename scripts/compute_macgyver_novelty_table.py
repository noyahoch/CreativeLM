#!/usr/bin/env python3
"""
Aggregate MacGyver quality + n-gram originality into the paper's novelty table.

The paper's formula:
  1. Quality: raw judge score 0–5, normalized to [0, 1] by dividing by 5.
  2. Originality: per-item n-gram originality (already in [0, 1]).
  3. Novelty = harmonic_mean(normalized_quality, originality)
             = 2 * q * o / (q + o)    when both > 0, else 0.
  4. Top-10% Novelty: mean novelty over the top 10% of items (by novelty).
  5. Delta to Baseline: method novelty minus baseline novelty.

Inputs:
  --quality-dir   : directory with *_macgyver_scores.csv files
                    (columns: quality_score, method, ...)
  --originality-dir : directory with *_originality.csv files
                    (columns: text_idx, originality_4, originality_5, originality_6)
  --baseline      : method name to use as baseline for delta computation
                    (default: auto-detect method containing "baseline")

Output:
  A CSV table + printed table matching the paper's format:
    Method | Quality | Orig n=4 | Orig n=5 | Orig n=6 |
           | Novelty n=4 (Δ) | Novelty n=5 (Δ) | Novelty n=6 (Δ) |
           | Top10% n=4 (Δ)  | Top10% n=5 (Δ)  | Top10% n=6 (Δ)  |

Examples:
  cd DLP

  python scripts/compute_macgyver_novelty_table.py \
    --quality-dir results/novelty/macgyver_quality_t07 \
    --originality-dir results/novelty/llama_steered_cluster8 \
    -o results/novelty/macgyver_novelty_table.csv

  # Explicit baseline
  python scripts/compute_macgyver_novelty_table.py \
    --quality-dir results/novelty/macgyver_quality_t07 \
    --originality-dir results/novelty/llama_steered_cluster8 \
    --baseline llama_steered_cluster8_baseline \
    -o results/novelty/macgyver_novelty_table.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MAX_QUALITY = 5.0
TOP_PERCENTILE = 0.10


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise harmonic mean; returns 0 where either input is 0."""
    denom = a + b
    out = np.where(denom > 0, 2.0 * a * b / denom, 0.0)
    return out


def load_quality_scores(quality_dir: Path) -> pd.DataFrame:
    """Load all *_macgyver_scores.csv, return DataFrame with (method, text_idx, quality_score)."""
    frames = []
    for csv_path in sorted(quality_dir.glob("*_macgyver_scores.csv")):
        df = pd.read_csv(csv_path)
        if "quality_score" not in df.columns:
            print(f"  Warning: skipping {csv_path.name} (no quality_score column)", file=sys.stderr)
            continue
        if "method" not in df.columns:
            method_name = re.sub(r"_macgyver_scores$", "", csv_path.stem)
            df["method"] = method_name
        df = df.reset_index(drop=True)
        df["text_idx"] = df.index
        frames.append(df[["method", "text_idx", "quality_score"]].copy())
    if not frames:
        raise FileNotFoundError(f"No *_macgyver_scores.csv found in {quality_dir}")
    return pd.concat(frames, ignore_index=True)


def load_originality_scores(originality_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all *_originality.csv, keyed by method slug."""
    result = {}
    for csv_path in sorted(originality_dir.glob("*_originality.csv")):
        method_slug = re.sub(r"_originality$", "", csv_path.stem)
        df = pd.read_csv(csv_path)
        result[method_slug] = df
    if not result:
        raise FileNotFoundError(f"No *_originality.csv found in {originality_dir}")
    return result


def detect_baseline(methods: list[str], explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for m in methods:
        if "baseline" in m.lower():
            return m
    return None


def compute_table(
    quality_df: pd.DataFrame,
    originality_by_method: dict[str, pd.DataFrame],
    baseline_method: str | None,
    quality_dir_prefix: str | None = None,
) -> pd.DataFrame:
    """Build the paper's novelty table."""
    rows = []

    quality_methods = quality_df["method"].unique().tolist()
    orig_methods = list(originality_by_method.keys())

    # Match quality methods to originality methods by slug suffix
    matched: list[tuple[str, str]] = []
    for q_method in quality_methods:
        for o_method in orig_methods:
            if o_method in q_method or q_method.endswith(o_method):
                matched.append((q_method, o_method))
                break
        else:
            q_suffix = q_method.rsplit("_", 1)[-1] if "_" in q_method else q_method
            for o_method in orig_methods:
                if o_method == q_suffix or o_method.endswith(q_suffix):
                    matched.append((q_method, o_method))
                    break

    if not matched:
        print(
            f"Warning: could not match any quality methods to originality methods.\n"
            f"  Quality methods:     {quality_methods}\n"
            f"  Originality methods: {orig_methods}",
            file=sys.stderr,
        )
        return pd.DataFrame()

    print(f"Matched {len(matched)} method(s):")
    for q, o in matched:
        print(f"  quality='{q}' <-> originality='{o}'")

    baseline_novelty: dict[str, float] = {}
    baseline_top10: dict[str, float] = {}

    for q_method, o_method in matched:
        q_rows = quality_df[quality_df["method"] == q_method].copy()
        o_df = originality_by_method[o_method].copy()

        n_items = min(len(q_rows), len(o_df))
        q_rows = q_rows.head(n_items).reset_index(drop=True)
        o_df = o_df.head(n_items).reset_index(drop=True)

        q_scores = pd.to_numeric(q_rows["quality_score"], errors="coerce").fillna(0).values
        norm_q = q_scores / MAX_QUALITY

        mean_quality = float(np.mean(norm_q))

        row: dict[str, object] = {
            "method": q_method,
            "n_items": n_items,
            "quality": round(mean_quality * MAX_QUALITY, 3),
            "quality_norm": round(mean_quality, 3),
        }

        for n in [4, 5, 6]:
            col = f"originality_{n}"
            if col not in o_df.columns:
                continue
            orig = o_df[col].values[:n_items]
            mean_orig = float(np.mean(orig))
            novelty = harmonic_mean(norm_q, orig)
            mean_novelty = float(np.mean(novelty))

            sorted_nov = np.sort(novelty)[::-1]
            top_k = max(1, int(np.ceil(len(sorted_nov) * TOP_PERCENTILE)))
            top10_novelty = float(np.mean(sorted_nov[:top_k]))

            row[f"orig_n{n}"] = round(mean_orig, 3)
            row[f"novelty_n{n}"] = round(mean_novelty, 3)
            row[f"top10_novelty_n{n}"] = round(top10_novelty, 3)

            is_baseline = (
                baseline_method is not None
                and (q_method == baseline_method or o_method == baseline_method
                     or baseline_method in q_method)
            )
            if is_baseline:
                baseline_novelty[f"n{n}"] = mean_novelty
                baseline_top10[f"n{n}"] = top10_novelty

        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute deltas
    for n in [4, 5, 6]:
        nk = f"n{n}"
        if nk in baseline_novelty:
            df[f"delta_novelty_n{n}"] = (df[f"novelty_{nk}"] - baseline_novelty[nk]).round(3)
            df[f"delta_top10_n{n}"] = (df[f"top10_novelty_{nk}"] - baseline_top10[nk]).round(3)
        else:
            df[f"delta_novelty_n{n}"] = np.nan
            df[f"delta_top10_n{n}"] = np.nan

    return df


def format_delta(val: float) -> str:
    if np.isnan(val):
        return "-"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.3f}"


def print_table(df: pd.DataFrame) -> None:
    """Print a human-readable version of the paper's table."""
    col_w = 14

    header_parts = [f"{'Method':<45}", f"{'Quality':>{col_w}}"]
    for prefix in ["Orig", "Novelty (Δ)", "Top10% (Δ)"]:
        for n in [4, 5, 6]:
            header_parts.append(f"{prefix} n={n}".rjust(col_w))
    header = " | ".join([header_parts[0], header_parts[1],
                          " ".join(header_parts[2:5]),
                          " ".join(header_parts[5:8]),
                          " ".join(header_parts[8:11])])
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        method = str(row["method"])
        qual = f"{row.get('quality', 0):.3f}"

        orig_cells = []
        for n in [4, 5, 6]:
            orig_cells.append(f"{row.get(f'orig_n{n}', 0):.3f}".rjust(col_w))

        nov_cells = []
        for n in [4, 5, 6]:
            nov = row.get(f"novelty_n{n}", 0)
            delta = row.get(f"delta_novelty_n{n}", float("nan"))
            if np.isnan(delta):
                cell = f"{nov:.3f}"
            else:
                cell = f"{nov:.3f} ({format_delta(delta)})"
            nov_cells.append(cell.rjust(col_w))

        top_cells = []
        for n in [4, 5, 6]:
            t10 = row.get(f"top10_novelty_n{n}", 0)
            delta = row.get(f"delta_top10_n{n}", float("nan"))
            if np.isnan(delta):
                cell = f"{t10:.3f}"
            else:
                cell = f"{t10:.3f} ({format_delta(delta)})"
            top_cells.append(cell.rjust(col_w))

        line = " | ".join([
            f"{method:<45}",
            qual.rjust(col_w),
            " ".join(orig_cells),
            " ".join(nov_cells),
            " ".join(top_cells),
        ])
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute paper-style MacGyver novelty table from quality + originality scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quality-dir", "-q", type=Path, required=True,
        help="Dir with *_macgyver_scores.csv (quality_score column, 0–5)",
    )
    parser.add_argument(
        "--originality-dir", "-r", type=Path, required=True,
        help="Dir with *_originality.csv (originality_4/5/6 columns, [0,1])",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output CSV path for the novelty table",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Method name to use as baseline for delta (default: auto-detect 'baseline')",
    )
    args = parser.parse_args()

    if not args.quality_dir.is_dir():
        print(f"Error: --quality-dir {args.quality_dir} not found", file=sys.stderr)
        return 1
    if not args.originality_dir.is_dir():
        print(f"Error: --originality-dir {args.originality_dir} not found", file=sys.stderr)
        return 1

    print(f"Loading quality scores from {args.quality_dir}")
    quality_df = load_quality_scores(args.quality_dir)
    print(f"  {len(quality_df)} rows, methods: {quality_df['method'].unique().tolist()}")

    print(f"Loading originality scores from {args.originality_dir}")
    orig_by_method = load_originality_scores(args.originality_dir)
    print(f"  Methods: {list(orig_by_method.keys())}")

    all_methods = list(quality_df["method"].unique()) + list(orig_by_method.keys())
    baseline = detect_baseline(all_methods, args.baseline)
    if baseline:
        print(f"Baseline method: {baseline}")
    else:
        print("Warning: no baseline detected — deltas will be NaN", file=sys.stderr)

    table = compute_table(quality_df, orig_by_method, baseline)

    if table.empty:
        print("Error: no methods could be matched", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    print()
    print_table(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
