#!/usr/bin/env python3
"""
Analyze AUT_OSCAI_RESULTS.xlsx: statistics across methods, per object, and by parameters.

Each sheet = one method. Columns: prompt (object), response, originality, mean, median, etc.

Usage:
  cd DLP
  python scripts/analyze_oscai_results.py results/judge/oscai/AUT_OSCAI_RESULTS.xlsx -o results/judge/oscai/analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def dataframe_to_png(df: pd.DataFrame, path: Path, max_rows: int = 50, font_size: int = 9) -> None:
    """Render DataFrame as a table PNG for easier reading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.size"] = font_size
    df_show = df.head(max_rows) if len(df) > max_rows else df
    # Round numeric columns for display
    df_show = df_show.copy()
    for c in df_show.select_dtypes(include=[np.number]).columns:
        df_show[c] = df_show[c].round(4)
    fig, ax = plt.subplots(figsize=(1.2 * len(df_show.columns), 0.35 * len(df_show) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="left",
        colColours=["#e0e0e0"] * len(df_show.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.2, 1.8)
    if len(df) > max_rows:
        ax.set_title(f"First {max_rows} of {len(df)} rows", fontsize=font_size + 1)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()


def load_all_sheets(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    dfs = []
    for name in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=name)
        df["method"] = name
        if "meadian" in df.columns and "median" not in df.columns:
            df = df.rename(columns={"meadian": "median"})
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["originality"] = pd.to_numeric(out["originality"], errors="coerce")
    if "mean" in out.columns:
        out["mean"] = pd.to_numeric(out["mean"], errors="coerce")
    if "median" in out.columns:
        out["median"] = pd.to_numeric(out["median"], errors="coerce")
    if "response" in out.columns:
        out["response_len_chars"] = out["response"].astype(str).str.len()
        out["response_len_words"] = out["response"].astype(str).str.split().str.len()
    return out


# Linear rescale [1, 5] -> [1, 10] for comparison with 1-10 judge scores
def _scale_1_5_to_1_10(x: float) -> float:
    return 1.0 + (float(x) - 1.0) * (9.0 / 4.0)


def rescale_originality_to_1_10(summary: pd.DataFrame) -> pd.DataFrame:
    """Add *_1_10 columns: originality metrics rescaled from [1,5] to [1,10]."""
    out = summary.copy()
    for col in ("mean_originality", "median_originality", "min_originality", "max_originality",
                "p25_originality", "p75_originality"):
        if col in out.columns:
            out[f"{col}_1_10"] = out[col].apply(lambda v: round(_scale_1_5_to_1_10(v), 4) if pd.notna(v) else np.nan)
    if "std_originality" in out.columns:
        out["std_originality_1_10"] = (out["std_originality"] * (9.0 / 4.0)).round(4)
    if "mean_of_mean" in out.columns:
        out["mean_of_mean_1_10"] = out["mean_of_mean"].apply(lambda v: round(_scale_1_5_to_1_10(v), 4) if pd.notna(v) else np.nan)
    if "mean_of_median" in out.columns:
        out["mean_of_median_1_10"] = out["mean_of_median"].apply(lambda v: round(_scale_1_5_to_1_10(v), 4) if pd.notna(v) else np.nan)
    return out


def analyze_by_method(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "n_answers": ("originality", "count"),
        "n_missing_orig": ("originality", lambda s: s.isna().sum()),
        "mean_originality": ("originality", "mean"),
        "median_originality": ("originality", lambda s: np.nanpercentile(s, 50)),
        "std_originality": ("originality", "std"),
        "min_originality": ("originality", "min"),
        "max_originality": ("originality", "max"),
        "p25_originality": ("originality", lambda s: np.nanpercentile(s, 25)),
        "p75_originality": ("originality", lambda s: np.nanpercentile(s, 75)),
    }
    if "mean" in df.columns and df["mean"].notna().any():
        agg["mean_of_mean"] = ("mean", "mean")
    if "median" in df.columns and df["median"].notna().any():
        agg["mean_of_median"] = ("median", "mean")

    summary = df.groupby("method", dropna=False).agg(**agg).round(4)
    summary["n_objects"] = df.groupby("method")["prompt"].nunique()
    summary = summary.reset_index()
    return summary.sort_values("mean_originality", ascending=False)


def analyze_per_object(df: pd.DataFrame) -> pd.DataFrame:
    """Per (method, prompt): count, mean originality, mean response length."""
    agg = {
        "n_answers": ("originality", "count"),
        "mean_originality": ("originality", "mean"),
        "median_originality": ("originality", lambda s: np.nanpercentile(s, 50)),
    }
    if "response_len_words" in df.columns:
        agg["mean_response_words"] = ("response_len_words", "mean")
    if "response_len_chars" in df.columns:
        agg["mean_response_chars"] = ("response_len_chars", "mean")

    per_obj = df.groupby(["method", "prompt"], dropna=False).agg(**agg).round(4).reset_index()
    return per_obj


def analyze_objects_across_methods(per_object_df: pd.DataFrame) -> pd.DataFrame:
    """Per object (prompt): spread of mean originality across methods."""
    by_prompt = (
        per_object_df.groupby("prompt")
        .agg(
            n_methods=("method", "nunique"),
            mean_orig_over_methods=("mean_originality", "mean"),
            std_orig_over_methods=("mean_originality", "std"),
            min_orig=("mean_originality", "min"),
            max_orig=("mean_originality", "max"),
        )
        .round(4)
        .reset_index()
    )
    return by_prompt


def analyze_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Per method: num answers (total and per object), response lengths."""
    n_answers_total = df.groupby("method").size().reset_index(name="n_answers_total")
    per_obj_counts = (
        df.groupby(["method", "prompt"]).size().reset_index(name="count").groupby("method")["count"]
    )
    n_answers_per_obj = per_obj_counts.agg(
        mean_answers_per_object="mean",
        min_answers_per_object="min",
        max_answers_per_object="max",
    ).reset_index()
    params = n_answers_total.merge(n_answers_per_obj, on="method")

    if "response_len_words" in df.columns:
        len_stats = (
            df.groupby("method")["response_len_words"]
            .agg(mean_words="mean", median_words=lambda s: np.nanpercentile(s, 50))
            .reset_index()
        )
        params = params.merge(len_stats, on="method")
    if "response_len_chars" in df.columns:
        char_stats = (
            df.groupby("method")["response_len_chars"]
            .agg(mean_chars="mean", median_chars=lambda s: np.nanpercentile(s, 50))
            .reset_index()
        )
        params = params.merge(char_stats, on="method")

    return params.round(4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze AUT_OSCAI_RESULTS.xlsx")
    parser.add_argument("xlsx", type=Path, help="Path to AUT_OSCAI_RESULTS.xlsx")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Write CSVs here (default: same dir as xlsx)")
    args = parser.parse_args()

    if not args.xlsx.exists():
        print(f"Error: {args.xlsx} not found", file=sys.stderr)
        return 1

    out_dir = args.output_dir or args.xlsx.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sheets...")
    df = load_all_sheets(args.xlsx)
    print(f"  Total rows: {len(df)}, methods: {df['method'].nunique()}, objects: {df['prompt'].nunique()}\n")

    # 1. Statistics across methods (1-5 scale)
    by_method = analyze_by_method(df)
    path_method = out_dir / "stats_by_method.csv"
    by_method.to_csv(path_method, index=False)
    dataframe_to_png(by_method, out_dir / "stats_by_method.png", max_rows=50)
    print("1. Statistics across methods (originality 1-5)")
    print(f"   Saved: {path_method}  +  {out_dir / 'stats_by_method.png'}")

    # 1b. Same stats with originality rescaled to 1-10 for comparison with judge
    by_method_1_10 = rescale_originality_to_1_10(by_method)
    path_1_10 = out_dir / "stats_by_method_1_10.csv"
    by_method_1_10.to_csv(path_1_10, index=False)
    cols_1_10 = ["method"] + [c for c in by_method_1_10.columns if c != "method" and c.endswith("_1_10")]
    dataframe_to_png(by_method_1_10[cols_1_10], out_dir / "stats_by_method_1_10.png", max_rows=50)
    print(f"   Saved: {path_1_10}  +  {out_dir / 'stats_by_method_1_10.png'}  (originality rescaled 1-5 -> 1-10)")
    print(by_method.to_string(index=False))
    print()

    # 2. Per object (and object-level summary across methods)
    per_object = analyze_per_object(df)
    path_per_obj = out_dir / "stats_per_object_per_method.csv"
    per_object.to_csv(path_per_obj, index=False)
    print("2. Per object (per method)")
    print(f"   Saved: {path_per_obj}  ({len(per_object)} rows)")

    by_object = analyze_objects_across_methods(per_object)
    path_by_obj = out_dir / "stats_per_object_across_methods.csv"
    by_object.to_csv(path_by_obj, index=False)
    dataframe_to_png(by_object, out_dir / "stats_per_object_across_methods.png", max_rows=40)
    print(f"   Per-object summary across methods: {path_by_obj}  +  {out_dir / 'stats_per_object_across_methods.png'}")
    print(by_object.head(15).to_string(index=False))
    print()

    # 3. Parameters: num answers, response lengths
    params = analyze_parameters(df)
    path_params = out_dir / "stats_parameters.csv"
    params.to_csv(path_params, index=False)
    dataframe_to_png(params, out_dir / "stats_parameters.png", max_rows=50)
    print("3. Parameters (num answers, response lengths)")
    print(f"   Saved: {path_params}  +  {out_dir / 'stats_parameters.png'}")
    print(params.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
