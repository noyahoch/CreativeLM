#!/usr/bin/env python3
"""
Compute n-gram originality scores for generated texts.

Two modes of operation:

1. **Directory mode** (like run_aut_inference): point at one or more
   inference dirs, auto-discover all *_results.csv files, and write
   per-method originality CSVs + a combined summary under --output-dir.

     cd DLP && python scripts/compute_originality.py \
         --index results/novelty/fineweb_10m.npz \
         --inference-dirs results/aut_inference/llama_t07 \
                          results/aut_inference/crpo_llama_nov_t07 \
         -o results/novelty

2. **Single-file mode**: score a single CSV / JSONL / JSON / text file.

     cd DLP && python scripts/compute_originality.py \
         --index results/novelty/fineweb_10m.npz \
         --input results/aut_inference/llama_t07/baseline_results.csv \
         --text-column reply \
         -o results/novelty
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dlp.novelty import (
    NgramCorpusIndex,
    compute_originality_batch,
    summarise_originality,
)

# ---------------------------------------------------------------------------
# Use parsing (shared with compute_response_diversity / run_aut_inference)
# ---------------------------------------------------------------------------

_PREAMBLE_RE = re.compile(
    r"(?i)^(here are|the following|below are|sure|of course|certainly|"
    r"i('d| would| can| will)|let me|give \d+|(\d+ )?(unconventional|unusual|creative|alternative) uses|"
    r"uses for)",
)
_MIN_USE_WORDS = 4


def _parse_uses(reply: str) -> list[str]:
    """Extract individual uses from a model reply (bullet / numbered lines)."""
    uses = []
    for line in str(reply).split("\n"):
        line = line.strip()
        line = re.sub(r"^[\-\*\•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^\*\*[^*]+\*\*:?\s*", "", line)
        line = line.strip().rstrip(":")
        if not line or len(line.split()) < _MIN_USE_WORDS:
            continue
        if _PREAMBLE_RE.search(line):
            continue
        uses.append(line)
    return uses


def _load_uses_from_flat_file(uses_path: Path) -> list[tuple[str, str]]:
    """Load uses from run_aut_inference *_uses.txt (one line per use: 'object, use').

    Returns list of (object, use) so we can keep object in the output CSV.
    """
    rows: list[tuple[str, str]] = []
    for line in uses_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        idx = line.find(",")
        if idx >= 0:
            obj, use = line[:idx].strip(), line[idx + 1 :].strip()
        else:
            obj, use = "", line
        if use:
            rows.append((obj, use))
    return rows


# ---------------------------------------------------------------------------
# Text loading helpers
# ---------------------------------------------------------------------------

def _load_texts_from_file(input_path: Path, text_column: str) -> list[str]:
    """Load texts from CSV, JSON, JSONL, or plain-text."""
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(input_path)
        if text_column not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Column '{text_column}' not found in {input_path}. "
                f"Available: {available}"
            )
        return df[text_column].dropna().astype(str).tolist()

    if suffix == ".jsonl":
        texts: list[str] = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    texts.append(obj)
                elif isinstance(obj, dict) and text_column in obj:
                    texts.append(str(obj[text_column]))
        return texts

    if suffix == ".json":
        data = json.loads(input_path.read_text())
        if isinstance(data, list):
            if data and isinstance(data[0], str):
                return data
            return [str(item.get(text_column, "")) for item in data if isinstance(item, dict)]
        raise ValueError(f"Unsupported JSON structure in {input_path}")

    return [
        line
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_file(
    input_path: Path,
    index: NgramCorpusIndex,
    text_column: str,
    n_values: list[int],
    per_use: bool = False,
) -> pd.DataFrame | None:
    """Score a single file. Returns the scores DataFrame or None on error."""
    try:
        texts = _load_texts_from_file(input_path, text_column)
    except ValueError as exc:
        print(f"  WARNING: skipping {input_path}: {exc}", file=sys.stderr)
        return None
    if not texts:
        print(f"  WARNING: no texts found in {input_path}", file=sys.stderr)
        return None

    if not per_use:
        return compute_originality_batch(texts, index, n_values=n_values)

    rows: list[dict] = []
    for reply_idx, reply in enumerate(texts):
        uses = _parse_uses(reply)
        if not uses:
            rows.append({"reply_idx": reply_idx, "use_idx": 0, "use": "", "num_tokens": 0,
                         **{f"originality_{n}": float("nan") for n in n_values}})
            continue
        use_scores = compute_originality_batch(uses, index, n_values=n_values)
        for ui, (_, row) in enumerate(use_scores.iterrows()):
            entry: dict = {"reply_idx": reply_idx, "use_idx": ui, "use": uses[ui]}
            entry["num_tokens"] = row["num_tokens"]
            for n in n_values:
                entry[f"originality_{n}"] = row[f"originality_{n}"]
            rows.append(entry)
    return pd.DataFrame(rows)


def _print_summary(summary: dict, n_values: list[int], label: str = "") -> None:
    header = "=== Originality Summary ==="
    if label:
        header = f"=== Originality: {label} ==="
    print(f"\n{header}")
    print(f"Texts evaluated: {summary['count']}")
    for n in n_values:
        prefix = f"originality_{n}"
        mean = summary.get(f"{prefix}_mean", float("nan"))
        median = summary.get(f"{prefix}_median", float("nan"))
        std = summary.get(f"{prefix}_std", float("nan"))
        lo = summary.get(f"{prefix}_min", float("nan"))
        hi = summary.get(f"{prefix}_max", float("nan"))
        print(
            f"  n={n}:  mean={mean:.4f}  median={median:.4f}  "
            f"std={std:.4f}  min={lo:.4f}  max={hi:.4f}"
        )


# ---------------------------------------------------------------------------
# Directory-mode: scan inference dirs for *_results.csv
# ---------------------------------------------------------------------------

def _expand_inference_dirs(raw_dirs: list[Path]) -> list[Path]:
    """Expand parent directories that contain subdirs with *_results.csv.

    If a given path has no *_results.csv itself but has child directories
    that do, expand it into those children.  This lets users pass e.g.
    ``results/aut_inference`` and have every sub-experiment picked up.
    """
    expanded: list[Path] = []
    for d in raw_dirs:
        if not d.is_dir():
            continue
        if list(d.glob("*_results.csv")):
            expanded.append(d)
        else:
            children = sorted(
                child for child in d.iterdir()
                if child.is_dir() and list(child.glob("*_results.csv"))
            )
            if children:
                print(
                    f"Expanding {d} → {len(children)} sub-directories "
                    f"with *_results.csv"
                )
                expanded.extend(children)
            else:
                print(f"WARNING: no *_results.csv found in {d} or its sub-directories",
                      file=sys.stderr)
    return expanded


def _run_directory_mode(
    inference_dirs: list[Path],
    index: NgramCorpusIndex,
    output_dir: Path,
    text_column: str,
    n_values: list[int],
    per_use: bool = False,
) -> int:
    inference_dirs = _expand_inference_dirs(inference_dirs)
    if not inference_dirs:
        print("Error: no inference directories with *_results.csv found.", file=sys.stderr)
        return 1

    all_summaries: list[dict] = []

    for inf_dir in inference_dirs:
        if not inf_dir.is_dir():
            print(f"WARNING: {inf_dir} is not a directory, skipping", file=sys.stderr)
            continue

        dir_label = inf_dir.name
        result_csvs = sorted(inf_dir.glob("*_results.csv"))
        if not result_csvs:
            print(f"WARNING: no *_results.csv found in {inf_dir}", file=sys.stderr)
            continue

        run_output = output_dir / dir_label
        run_output.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Inference dir: {inf_dir}  ({len(result_csvs)} result files)")
        print(f"Output dir:    {run_output}")
        print(f"{'=' * 60}")

        for csv_path in result_csvs:
            method_slug = csv_path.stem.replace("_results", "")
            print(f"\n  Method: {method_slug}")

            if per_use:
                uses_txt = inf_dir / f"{method_slug}_uses.txt"
                if uses_txt.exists():
                    object_use_pairs = _load_uses_from_flat_file(uses_txt)
                    use_texts = [u for _, u in object_use_pairs]
                    if not use_texts:
                        print(f"    WARNING: no uses in {uses_txt}", file=sys.stderr)
                        continue
                    use_scores = compute_originality_batch(use_texts, index, n_values=n_values)
                    rows = []
                    for i, ((obj, use), (_, row)) in enumerate(zip(object_use_pairs, use_scores.iterrows())):
                        rows.append({
                            "use_idx": i,
                            "object": obj,
                            "use": use,
                            "num_tokens": row["num_tokens"],
                            **{f"originality_{n}": row[f"originality_{n}"] for n in n_values},
                        })
                    scores_df = pd.DataFrame(rows)
                    print(f"    Using {uses_txt}  ({len(use_texts)} uses)")
                else:
                    scores_df = _score_file(csv_path, index, text_column, n_values, per_use=True)
            else:
                scores_df = _score_file(csv_path, index, text_column, n_values, per_use=False)

            if scores_df is None:
                continue

            suffix = "_originality_per_use.csv" if per_use else "_originality.csv"
            out_csv = run_output / f"{method_slug}{suffix}"
            scores_df.to_csv(out_csv, index=False)
            print(f"    Wrote {out_csv}  ({len(scores_df)} rows)")

            summary = summarise_originality(scores_df, n_values=n_values)
            summary["dir"] = dir_label
            summary["method"] = method_slug
            all_summaries.append(summary)

            for n in n_values:
                mean = summary.get(f"originality_{n}_mean", float("nan"))
                print(f"    n={n}: mean={mean:.4f}")

    if not all_summaries:
        print("\nNo results to summarise.", file=sys.stderr)
        return 1

    summary_df = pd.DataFrame(all_summaries)
    cols = ["dir", "method", "count"] + [
        f"originality_{n}_{stat}"
        for n in n_values
        for stat in ("mean", "median", "std", "min", "max")
    ]
    cols = [c for c in cols if c in summary_df.columns]
    summary_df = summary_df[cols]
    summary_path = output_dir / "originality_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Combined summary: {summary_path}")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False))
    return 0


# ---------------------------------------------------------------------------
# Single-file mode
# ---------------------------------------------------------------------------

def _run_single_file_mode(
    input_path: Path,
    index: NgramCorpusIndex,
    output_dir: Path | None,
    text_column: str,
    n_values: list[int],
    per_use: bool = False,
) -> int:
    texts = _load_texts_from_file(input_path, text_column)
    print(f"Loaded {len(texts)} texts from {input_path}")
    if not texts:
        print("Error: no texts found in input file.", file=sys.stderr)
        return 1

    if per_use:
        all_uses: list[str] = []
        for reply in texts:
            all_uses.extend(_parse_uses(reply))
        print(f"Parsed {len(all_uses)} individual uses from {len(texts)} replies")
        scores_df = compute_originality_batch(all_uses, index, n_values=n_values)
    else:
        scores_df = compute_originality_batch(texts, index, n_values=n_values)

    summary = summarise_originality(scores_df, n_values=n_values)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem.replace("_results", "")
        suffix = "_originality_per_use.csv" if per_use else "_originality.csv"
        out_csv = output_dir / f"{stem}{suffix}"
        scores_df.to_csv(out_csv, index=False)
        print(f"\nWrote per-text scores to {out_csv}")

    _print_summary(summary, n_values, label=input_path.name)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute n-gram originality scores for generated texts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--index",
        type=Path,
        required=True,
        help="Path to the .npz n-gram corpus index",
    )

    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--inference-dirs",
        type=Path,
        nargs="+",
        help="Inference result directories (each containing *_results.csv files)",
    )
    source.add_argument(
        "--input",
        type=Path,
        help="Single input file: CSV, JSONL, JSON, or plain-text",
    )

    ap.add_argument(
        "--text-column",
        type=str,
        default="reply",
        help="Column / field name containing generated text (default: reply)",
    )
    ap.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[4, 5, 6],
        help="N-gram sizes to evaluate (default: 4 5 6)",
    )
    ap.add_argument(
        "--per-use",
        action="store_true",
        help="Score each use separately. In directory mode uses "
             "{method}_uses.txt when present (one use per line, 'object, use'); "
             "otherwise parses replies from CSV. Without this flag the whole "
             "reply is scored as one text.",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("results/novelty"),
        help="Output directory (default: results/novelty)",
    )
    args = ap.parse_args()

    if not args.index.exists():
        print(f"Error: index file not found: {args.index}", file=sys.stderr)
        return 1

    index = NgramCorpusIndex.load(args.index)

    if args.inference_dirs:
        return _run_directory_mode(
            args.inference_dirs, index, args.output_dir,
            args.text_column, args.n_values, per_use=args.per_use,
        )
    else:
        return _run_single_file_mode(
            args.input, index, args.output_dir,
            args.text_column, args.n_values, per_use=args.per_use,
        )


if __name__ == "__main__":
    raise SystemExit(main())
