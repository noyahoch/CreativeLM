#!/usr/bin/env python3
"""
Select the best reply per prompt from multi-inference AUT results.

Workflow:
  1. run_aut_inference.py --num-inferences N  →  {method}_results.csv + {method}_uses.txt
  2. User grades uses.txt externally            →  scored_uses.csv  (object, use, score)
  3. This script selects the best sample_idx per eval_idx based on highest mean score.

Input:
  --input   Original results CSV (has eval_idx, sample_idx, reply, problem_text, …)
  --scores  Scored uses CSV with columns: object, use, score
            Lines correspond 1-to-1 with uses.txt (same order).

Output (written next to --output path):
  {stem}_results.csv  — filtered CSV keeping only the best sample per prompt
  {stem}_uses.txt     — flat "Object, use" lines from the winning replies

Usage:
  python scripts/select_best_uses.py \\
    --input  results/aut_inference/baseline_sampled/baseline_results.csv \\
    --scores results/aut_inference/baseline_sampled/scored_uses.csv \\
    --output results/aut_inference/baseline_sampled/baseline_best

  # Also supports per-reply scores (eval_idx, sample_idx, score):
  python scripts/select_best_uses.py \\
    --input  results/aut_inference/baseline_sampled/baseline_results.csv \\
    --scores results/aut_inference/baseline_sampled/reply_scores.csv \\
    --output results/aut_inference/baseline_sampled/baseline_best
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


# ── Parsing helpers (same logic as run_aut_inference.py) ──────────────────────

_PREAMBLE_RE = re.compile(
    r"(?i)^(here are|the following|below are|sure|of course|certainly|"
    r"i('d| would| can| will)|let me|give \d+|(\d+ )?(unconventional|unusual|creative|alternative) uses|"
    r"uses for)",
)

MIN_USE_WORDS = 4


def _parse_uses_from_reply(reply: str) -> list[str]:
    uses: list[str] = []
    for line in reply.split("\n"):
        line = line.strip()
        line = re.sub(r"^[\-\*\•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = line.strip().rstrip(":")
        if not line or len(line.split()) < MIN_USE_WORDS:
            continue
        if _PREAMBLE_RE.search(line):
            continue
        uses.append(line)
    return uses


def _parse_object_from_A(a_text: str) -> str:
    m = re.search(
        r"uses for (?:a |an )?(?:piece of )?([\w\s\-]+?)(?:\s*\([^)]*\))?(?:\s+(?:in|focusing|that|as)\b|\.|$)",
        a_text, re.I,
    )
    return m.group(1).strip() if m else a_text


def _strip_inner_commas(text: str) -> str:
    text = re.sub(r",\s+(and|or|but|so|yet|nor)\b", r" \1", text)
    text = text.replace(", ", " ").replace(",", "")
    return text


# ── Core logic ────────────────────────────────────────────────────────────────

def _build_use_source_map(df: pd.DataFrame) -> list[dict]:
    """For each CSV row, parse uses and track their source (eval_idx, sample_idx).

    Returns an ordered list of dicts matching the uses.txt line order.
    """
    records: list[dict] = []
    for _, row in df.iterrows():
        obj = _parse_object_from_A(row["problem_text"])
        for use in _parse_uses_from_reply(str(row["reply"])):
            records.append({
                "eval_idx": int(row["eval_idx"]),
                "sample_idx": int(row["sample_idx"]),
                "object": obj,
                "use_clean": _strip_inner_commas(use),
                "use_raw": use,
            })
    return records


def _load_per_use_scores(scores_path: Path) -> pd.DataFrame:
    """Load scored uses CSV.  Expected columns: object, use, score."""
    df = pd.read_csv(scores_path)
    cols_lower = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols_lower)
    required = {"object", "use", "score"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(
            f"Scores CSV missing columns: {missing}. "
            f"Expected: object, use, score. Got: {list(df.columns)}"
        )
    return df


def _load_per_reply_scores(scores_path: Path) -> pd.DataFrame:
    """Load per-reply scores CSV.  Expected columns: eval_idx, sample_idx, score."""
    df = pd.read_csv(scores_path)
    cols_lower = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols_lower)
    return df


def _detect_score_format(scores_path: Path) -> str:
    """Detect whether scores file is per-use (object,use,score) or per-reply (eval_idx,sample_idx,score)."""
    df = pd.read_csv(scores_path, nrows=1)
    cols = {c.strip().lower() for c in df.columns}
    if {"eval_idx", "sample_idx", "score"}.issubset(cols):
        return "per_reply"
    if {"object", "use", "score"}.issubset(cols):
        return "per_use"
    raise ValueError(
        f"Cannot detect scores format. Columns: {list(df.columns)}. "
        "Expected either (object, use, score) or (eval_idx, sample_idx, score)."
    )


def _save_flat(df: pd.DataFrame, path: Path) -> int:
    """Write flat 'Object, use' lines from the filtered CSV. Returns use count."""
    lines: list[str] = []
    for _, row in df.iterrows():
        obj = _parse_object_from_A(row["problem_text"])
        for use in _parse_uses_from_reply(str(row["reply"])):
            lines.append(f"{obj}, {_strip_inner_commas(use)}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select the best sample per prompt from multi-inference AUT results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Original results CSV from run_aut_inference.py (has sample_idx)")
    parser.add_argument("--scores", type=Path, required=True,
                        help="Scored uses CSV: (object, use, score) or (eval_idx, sample_idx, score)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output stem — writes {stem}_results.csv and {stem}_uses.txt")
    parser.add_argument("--agg", choices=["mean", "median", "sum"], default="mean",
                        help="Aggregation for per-use scores → per-reply score (default: mean)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1
    if not args.scores.exists():
        print(f"Error: {args.scores} not found", file=sys.stderr)
        return 1

    # Load original results
    orig = pd.read_csv(args.input)
    if "sample_idx" not in orig.columns:
        print("Error: input CSV has no sample_idx column — was it generated with --num-inferences?",
              file=sys.stderr)
        return 1

    n_prompts = orig["eval_idx"].nunique()
    n_samples = orig["sample_idx"].nunique()
    print(f"Loaded {len(orig)} rows: {n_prompts} prompts × {n_samples} samples")

    # Detect score format and compute per-reply aggregated scores
    fmt = _detect_score_format(args.scores)
    print(f"Scores format detected: {fmt}")

    if fmt == "per_reply":
        reply_scores = _load_per_reply_scores(args.scores)
        reply_scores = reply_scores[["eval_idx", "sample_idx", "score"]].copy()
    else:
        scored_uses = _load_per_use_scores(args.scores)
        use_map = _build_use_source_map(orig)

        if len(scored_uses) != len(use_map):
            print(f"Warning: scored uses ({len(scored_uses)} lines) != "
                  f"expected uses ({len(use_map)} lines from CSV). "
                  "Matching by line order up to the shorter length.",
                  file=sys.stderr)

        n_match = min(len(scored_uses), len(use_map))
        records: list[dict] = []
        for i in range(n_match):
            records.append({
                "eval_idx": use_map[i]["eval_idx"],
                "sample_idx": use_map[i]["sample_idx"],
                "score": float(scored_uses.iloc[i]["score"]),
            })

        use_scores = pd.DataFrame(records)
        reply_scores = (
            use_scores
            .groupby(["eval_idx", "sample_idx"])["score"]
            .agg(args.agg)
            .reset_index()
        )

    # For each eval_idx, pick the sample_idx with the highest aggregated score
    best_idx = reply_scores.loc[reply_scores.groupby("eval_idx")["score"].idxmax()]
    winners = set(zip(best_idx["eval_idx"].astype(int), best_idx["sample_idx"].astype(int)))
    print(f"Selected best sample for {len(winners)} prompts")

    # Filter original CSV to winning rows
    mask = orig.apply(
        lambda r: (int(r["eval_idx"]), int(r["sample_idx"])) in winners,
        axis=1,
    )
    filtered = orig[mask].copy()
    filtered = filtered.sort_values(["eval_idx", "sample_idx"]).reset_index(drop=True)

    # Write outputs
    out_stem = args.output
    out_dir = out_stem.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(f"{out_stem}_results.csv")
    filtered.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}  ({len(filtered)} rows)")

    txt_path = Path(f"{out_stem}_uses.txt")
    n_uses = _save_flat(filtered, txt_path)
    print(f"  Flat: {txt_path}  ({n_uses} uses)")

    # Print summary statistics
    merged = filtered.merge(
        best_idx[["eval_idx", "sample_idx", "score"]],
        on=["eval_idx", "sample_idx"],
        how="left",
    )
    if "score" in merged.columns:
        print(f"\n  Score stats (winning replies):")
        print(f"    mean={merged['score'].mean():.3f}  "
              f"median={merged['score'].median():.3f}  "
              f"std={merged['score'].std():.3f}")
        all_scores = reply_scores["score"]
        print(f"  Score stats (all replies):")
        print(f"    mean={all_scores.mean():.3f}  "
              f"median={all_scores.median():.3f}  "
              f"std={all_scores.std():.3f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
