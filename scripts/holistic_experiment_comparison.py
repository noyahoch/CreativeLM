#!/usr/bin/env python3
"""
Holistic comparison of experiments across Qwen judge, diversity, and OSCAI metrics.

Given a list of experiment folder names (matching aut_inference subdirectories),
loads the per-method summaries from each evaluation source, filters to the
requested experiments, normalises method names, merges everything into a single
table, and writes CSV + Markdown outputs.

Usage:
  cd DLP

  # Compare two clustering experiments (judge + diversity only)
  python scripts/holistic_experiment_comparison.py \
    --folders llama_t07_dbanal_clustered_2 llama_t07_dbanal_clustered_8 \
    -o results/judge/holistic

  # Include OSCAI scores (method names auto-mapped via --method-names)
  python scripts/holistic_experiment_comparison.py \
    --folders llama_t07 llama_t07_dbanal_bcon llama_t07_dbanal_clustered \
    --oscai-stats results/judge/oscai/analysis/stats_by_method.csv \
    -o results/judge/holistic

  # Sort by diversity instead of novelty
  python scripts/holistic_experiment_comparison.py \
    --folders llama_t07 llama_t15 \
    --sort-by sem_diversity \
    -o results/judge/holistic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_real_folders(inference_dir: Path) -> set[str]:
    """Return the set of immediate subdirectory names under the inference dir."""
    if not inference_dir.is_dir():
        return set()
    return {p.name for p in inference_dir.iterdir() if p.is_dir()}


def _match_judge_method(
    method: str,
    folders: list[str],
    all_real_folders: set[str] | None = None,
) -> tuple[str, str] | None:
    """Split a judge method name into (folder, run) using the known folder list.

    Folders are tried longest-first so that e.g. ``llama_t07_dbanal_clustered_2``
    matches before ``llama_t07_dbanal_clustered``.

    When *all_real_folders* is provided, a candidate match is rejected if a
    longer real folder also matches — this prevents ``llama_t07`` from
    accidentally claiming methods that belong to ``llama_t07_dbanal_negd``.
    """
    for folder in folders:
        prefix = folder + "_"
        if method.startswith(prefix):
            if all_real_folders is not None:
                stolen = any(
                    f != folder and method.startswith(f + "_") and len(f) > len(folder)
                    for f in all_real_folders
                )
                if stolen:
                    continue
            return folder, method[len(prefix):]
        if method == folder:
            return folder, ""
    return None


def _canonical(folder: str, run: str) -> str:
    return f"{folder}/{run}" if run else folder


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_judge(
    path: Path,
    folders: list[str],
    all_real_folders: set[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        m = _match_judge_method(row["method"], folders, all_real_folders)
        if m is None:
            continue
        folder, run = m
        r = row.to_dict()
        r["folder"] = folder
        r["run"] = run
        r["canonical"] = _canonical(folder, run)
        rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.rename(columns={
            "mean_overall": "qwen_overall",
            "mean_novelty": "qwen_novelty",
            "mean_usability": "qwen_usability",
            "median_novelty": "qwen_med_novelty",
            "median_usability": "qwen_med_usability",
            "std_overall": "qwen_std",
            "n_uses": "qwen_n_uses",
            "n_missing": "qwen_n_missing",
        })
        out = out.drop(columns=["method"], errors="ignore")
    return out


def load_diversity(path: Path, folders: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        parts = row["method"].split("/", 1)
        folder = parts[0]
        run = parts[1] if len(parts) > 1 else ""
        if folder not in folders:
            continue
        r = row.to_dict()
        r["folder"] = folder
        r["run"] = run
        r["canonical"] = _canonical(folder, run)
        rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.rename(columns={
            "mean_sem_div": "sem_diversity",
            "std_sem_div": "sem_div_std",
            "mean_ng1": "ngram_div_1",
            "mean_ng2": "ngram_div_2",
            "mean_ng3": "ngram_div_3",
        })
        out = out.drop(columns=["method", "n_objects"], errors="ignore")
    return out


def load_oscai(
    path: Path,
    folders: list[str],
    method_map: dict[str, str] | None = None,
    all_real_folders: set[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if method_map:
        df["method"] = df["method"].map(lambda m: method_map.get(m, m))

    rows = []
    for _, row in df.iterrows():
        m = _match_judge_method(row["method"], folders, all_real_folders)
        if m is None:
            continue
        folder, run = m
        r = row.to_dict()
        r["folder"] = folder
        r["run"] = run
        r["canonical"] = _canonical(folder, run)
        rows.append(r)

    out = pd.DataFrame(rows)
    if not out.empty:
        keep = ["canonical", "folder", "run"]
        rename_map = {}
        for col in out.columns:
            if col in keep or col == "method":
                continue
            if "1_10" in col:
                rename_map[col] = f"oscai_{col.replace('_1_10', '')}"
            else:
                rename_map[col] = f"oscai_{col}"
        out = out.rename(columns=rename_map)
        out = out.drop(columns=["method"], errors="ignore")
    return out


def load_method_map(path: Path) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    """Load the mapping CSV.

    Returns (oscai_to_canonical, canonical_to_paper, canonical_to_method,
             canonical_to_steered_vector, canonical_to_factor,
             canonical_to_alpha, canonical_to_temperature,
             canonical_to_b_source, canonical_to_source_d, canonical_to_where_injected,
             canonical_to_steering_vector_dir).
    """
    df = pd.read_csv(path)
    oscai_to_canonical: dict[str, str] = {}
    canonical_to_paper: dict[str, str] = {}
    canonical_to_method: dict[str, str] = {}
    canonical_to_steered_vector: dict[str, str] = {}
    canonical_to_factor: dict[str, str] = {}
    canonical_to_alpha: dict[str, str] = {}
    canonical_to_temperature: dict[str, str] = {}
    canonical_to_b_source: dict[str, str] = {}
    canonical_to_source_d: dict[str, str] = {}
    canonical_to_where_injected: dict[str, str] = {}
    canonical_to_steering_vector_dir: dict[str, str] = {}

    def _str(v) -> str:
        if pd.isna(v) or v == "":
            return ""
        return str(v).strip()

    for _, row in df.iterrows():
        oscai = row.get("oscai_method")
        canon = row.get("canonical_method")
        if pd.notna(oscai) and pd.notna(canon):
            oscai_to_canonical[str(oscai)] = str(canon)
        if not pd.notna(canon):
            continue
        c = str(canon)
        if pd.notna(row.get("paper_name")):
            canonical_to_paper[c] = _str(row["paper_name"])
        if "method" in row:
            canonical_to_method[c] = _str(row["method"])
        if "steered_vector" in row:
            canonical_to_steered_vector[c] = _str(row["steered_vector"])
        if "factor" in row:
            canonical_to_factor[c] = _str(row["factor"])
        if "alpha" in row:
            canonical_to_alpha[c] = _str(row["alpha"])
        if "temperature" in row:
            canonical_to_temperature[c] = _str(row["temperature"])
        if "b_source" in row:
            canonical_to_b_source[c] = _str(row["b_source"])
        if "source_d" in row:
            canonical_to_source_d[c] = _str(row["source_d"])
        if "where_injected" in row:
            canonical_to_where_injected[c] = _str(row["where_injected"])
        if "steering_vector_dir" in row:
            canonical_to_steering_vector_dir[c] = _str(row["steering_vector_dir"])

    return (
        oscai_to_canonical,
        canonical_to_paper,
        canonical_to_method,
        canonical_to_steered_vector,
        canonical_to_factor,
        canonical_to_alpha,
        canonical_to_temperature,
        canonical_to_b_source,
        canonical_to_source_d,
        canonical_to_where_injected,
        canonical_to_steering_vector_dir,
    )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_sources(
    judge_df: pd.DataFrame | None,
    div_df: pd.DataFrame | None,
    oscai_df: pd.DataFrame | None,
) -> pd.DataFrame:
    shared_cols = ["canonical", "folder", "run"]
    frames = [f for f in [judge_df, div_df, oscai_df] if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=shared_cols, how="outer", suffixes=("", "_dup"))
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged = merged.drop(columns=dup_cols)

    return merged


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_KEY_COLS = ["folder", "run", "canonical"]

_DISPLAY_COLS = [
    "method", "steered_vector", "factor", "alpha", "temperature",
    "b_source", "source_d", "where_injected", "steering_vector_dir",
    "paper_name", "folder", "run",
    "qwen_novelty", "qwen_usability", "qwen_overall",
    "sem_diversity", "ngram_div_1", "ngram_div_2",
]

_OSCAI_DISPLAY = [
    "oscai_mean_originality", "oscai_median_originality",
]


def _auto_paper_name(canonical: str) -> str:
    """Generate a fallback paper name from a canonical method key."""
    return canonical.replace("/", " / ").replace("_", " ")


def _flat_method(row: pd.Series) -> str:
    folder = row.get("folder", "")
    run = row.get("run", "")
    return f"{folder}_{run}" if run else folder


def _lookup(row: pd.Series, d: dict[str, str], fallback: str = "") -> str:
    jm = row.get("_judge_method")
    if pd.notna(jm) and jm in d:
        return d[jm]
    flat = _flat_method(row)
    if flat in d:
        return d[flat]
    return fallback


def attach_paper_names(
    df: pd.DataFrame,
    canonical_to_paper: dict[str, str],
    canonical_to_method: dict[str, str] | None = None,
    canonical_to_steered_vector: dict[str, str] | None = None,
    canonical_to_factor: dict[str, str] | None = None,
    canonical_to_alpha: dict[str, str] | None = None,
    canonical_to_temperature: dict[str, str] | None = None,
    canonical_to_b_source: dict[str, str] | None = None,
    canonical_to_source_d: dict[str, str] | None = None,
    canonical_to_where_injected: dict[str, str] | None = None,
    canonical_to_steering_vector_dir: dict[str, str] | None = None,
    judge_method_col: str = "canonical",
) -> pd.DataFrame:
    """Add paper_name and optional structured columns by lookup."""
    if df.empty:
        return df

    def _lookup_paper(row: pd.Series) -> str:
        canon = row.get(judge_method_col, "")
        v = _lookup(row, canonical_to_paper)
        if v:
            return v
        return _auto_paper_name(str(canon)) if pd.notna(canon) else ""

    df = df.copy()
    df["paper_name"] = df.apply(_lookup_paper, axis=1)
    if canonical_to_method:
        df["method"] = df.apply(lambda r: _lookup(r, canonical_to_method), axis=1)
    if canonical_to_steered_vector:
        df["steered_vector"] = df.apply(lambda r: _lookup(r, canonical_to_steered_vector), axis=1)
    if canonical_to_factor:
        df["factor"] = df.apply(lambda r: _lookup(r, canonical_to_factor), axis=1)
    if canonical_to_alpha:
        df["alpha"] = df.apply(lambda r: _lookup(r, canonical_to_alpha), axis=1)
    if canonical_to_temperature:
        df["temperature"] = df.apply(lambda r: _lookup(r, canonical_to_temperature), axis=1)
    if canonical_to_b_source:
        df["b_source"] = df.apply(lambda r: _lookup(r, canonical_to_b_source), axis=1)
    if canonical_to_source_d:
        df["source_d"] = df.apply(lambda r: _lookup(r, canonical_to_source_d), axis=1)
    if canonical_to_where_injected:
        df["where_injected"] = df.apply(lambda r: _lookup(r, canonical_to_where_injected), axis=1)
    if canonical_to_steering_vector_dir:
        df["steering_vector_dir"] = df.apply(lambda r: _lookup(r, canonical_to_steering_vector_dir), axis=1)
    return df


def write_outputs(df: pd.DataFrame, out_dir: Path, sort_by: str, has_oscai: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    csv_path = out_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(df)} rows)")

    display = [c for c in _DISPLAY_COLS if c in df.columns]
    if has_oscai:
        display += [c for c in _OSCAI_DISPLAY if c in df.columns]
    show = df[display].copy()
    for c in show.select_dtypes(include="number").columns:
        show[c] = show[c].round(4)

    md_lines = [
        "# Holistic Experiment Comparison",
        "",
        f"Sorted by **{sort_by}** (descending). {len(df)} methods.",
        "",
        show.to_markdown(index=False),
        "",
    ]
    md_text = "\n".join(md_lines)

    md_path = out_dir / "comparison.md"
    md_path.write_text(md_text)
    print(f"Wrote {md_path}")
    print()
    print(md_text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Holistic comparison across Qwen judge, diversity, and OSCAI.",
    )
    ap.add_argument(
        "--folders", nargs="+", default=None,
        help="Experiment folder names to include (e.g. llama_t07_dbanal_clustered_2). Ignored if --all-folders.",
    )
    ap.add_argument(
        "--all-folders", action="store_true",
        help="Include all subdirectories of --inference-dir in the comparison",
    )
    ap.add_argument(
        "--judge-summary", type=Path,
        default=Path("results/judge/all_qwen32b/summary.csv"),
        help="Path to Qwen judge summary.csv",
    )
    ap.add_argument(
        "--diversity-summary", type=Path,
        default=Path("results/judge/diversity/diversity_summary.csv"),
        help="Path to diversity_summary.csv",
    )
    ap.add_argument(
        "--oscai-stats", type=Path, default=None,
        help="Path to OSCAI stats_by_method.csv (or _1_10.csv). Optional.",
    )
    ap.add_argument(
        "--method-names", type=Path,
        default=Path("results/judge/oscai/method_map.csv"),
        help="CSV with oscai_method, canonical_method, paper_name, method, steered_vector, factor, alpha, temperature",
    )
    ap.add_argument(
        "-o", "--output-dir", type=Path,
        default=Path("results/judge/holistic"),
        help="Output directory",
    )
    ap.add_argument(
        "--sort-by", type=str, default="qwen_novelty",
        help="Column to sort the output by (default: qwen_novelty)",
    )
    ap.add_argument(
        "--inference-dir", type=Path,
        default=Path("results/aut_inference"),
        help="Inference results dir (used to disambiguate folder prefixes)",
    )
    args = ap.parse_args()

    all_real_folders = _discover_real_folders(args.inference_dir)
    if args.all_folders:
        if not all_real_folders:
            print("Error: --all-folders but no subdirs in --inference-dir.", file=sys.stderr)
            return 1
        folders_sorted = sorted(all_real_folders, key=len, reverse=True)
        print(f"Using all {len(folders_sorted)} folders from {args.inference_dir}")
    elif args.folders:
        folders_sorted = sorted(args.folders, key=len, reverse=True)
        if all_real_folders:
            print(f"Discovered {len(all_real_folders)} inference folders in {args.inference_dir}")
    else:
        print("Error: provide --folders FOLDER ... or --all-folders.", file=sys.stderr)
        return 1

    oscai_to_canonical: dict[str, str] = {}
    canonical_to_paper: dict[str, str] = {}
    canonical_to_method: dict[str, str] = {}
    canonical_to_steered_vector: dict[str, str] = {}
    canonical_to_factor: dict[str, str] = {}
    canonical_to_alpha: dict[str, str] = {}
    canonical_to_temperature: dict[str, str] = {}
    canonical_to_b_source: dict[str, str] = {}
    canonical_to_source_d: dict[str, str] = {}
    canonical_to_where_injected: dict[str, str] = {}
    canonical_to_steering_vector_dir: dict[str, str] = {}
    if args.method_names and args.method_names.exists():
        (
            oscai_to_canonical,
            canonical_to_paper,
            canonical_to_method,
            canonical_to_steered_vector,
            canonical_to_factor,
            canonical_to_alpha,
            canonical_to_temperature,
            canonical_to_b_source,
            canonical_to_source_d,
            canonical_to_where_injected,
            canonical_to_steering_vector_dir,
        ) = load_method_map(args.method_names)
        print(f"Names:     {len(canonical_to_paper)} paper names from {args.method_names}")

    judge_df = None
    if args.judge_summary.exists():
        judge_df = load_judge(args.judge_summary, folders_sorted, all_real_folders)
        n = len(judge_df) if judge_df is not None else 0
        print(f"Judge:     {n} methods matched from {args.judge_summary}")
    else:
        print(f"Judge:     {args.judge_summary} not found, skipping")

    div_df = None
    if args.diversity_summary.exists():
        div_df = load_diversity(args.diversity_summary, folders_sorted)
        n = len(div_df) if div_df is not None else 0
        print(f"Diversity: {n} methods matched from {args.diversity_summary}")
    else:
        print(f"Diversity: {args.diversity_summary} not found, skipping")

    oscai_df = None
    has_oscai = False
    if args.oscai_stats is not None:
        if args.oscai_stats.exists():
            oscai_df = load_oscai(
                args.oscai_stats, folders_sorted,
                oscai_to_canonical or None, all_real_folders,
            )
            n = len(oscai_df) if oscai_df is not None else 0
            print(f"OSCAI:     {n} methods matched from {args.oscai_stats}")
            has_oscai = n > 0
        else:
            print(f"OSCAI:     {args.oscai_stats} not found, skipping")

    merged = merge_sources(judge_df, div_df, oscai_df)
    if merged.empty:
        print("\nNo methods matched the given folders. Check --folders values.", file=sys.stderr)
        return 1

    if canonical_to_paper:
        merged["_judge_method"] = merged.apply(
            lambda r: f"{r['folder']}_{r['run']}" if r.get("run") else r.get("folder", ""),
            axis=1,
        )
        merged = attach_paper_names(
            merged,
            canonical_to_paper,
            canonical_to_method=canonical_to_method or None,
            canonical_to_steered_vector=canonical_to_steered_vector or None,
            canonical_to_factor=canonical_to_factor or None,
            canonical_to_alpha=canonical_to_alpha or None,
            canonical_to_temperature=canonical_to_temperature or None,
            canonical_to_b_source=canonical_to_b_source or None,
            canonical_to_source_d=canonical_to_source_d or None,
            canonical_to_where_injected=canonical_to_where_injected or None,
            canonical_to_steering_vector_dir=canonical_to_steering_vector_dir or None,
        )
        merged = merged.drop(columns=["_judge_method"], errors="ignore")

    print(f"\nMerged:    {len(merged)} rows")
    write_outputs(merged, args.output_dir, args.sort_by, has_oscai)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
