#!/usr/bin/env python3
"""
Compute within-response diversity for AUT inference results.

For each (eval_idx, method) response containing ~8 uses, compute:
  1. Semantic diversity: mean pairwise cosine distance of use embeddings.
  2. N-gram diversity: distinct/total n-gram ratio for n=1,2,3.

Usage:
  cd DLP && python scripts/compute_response_diversity.py \
    --inference-dirs results/aut_inference/llama_t07 \
                     results/aut_inference/llama_t07_bonly \
                     results/aut_inference/crpo_llama_nov_t07 \
                     results/aut_inference/llama_brainstorm \
                     results/aut_inference/llama_creative_t1.5_minp0.1 \
    --model all-MiniLM-L6-v2 \
    -o results/judge/diversity
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Use parsing (same logic as run_aut_inference / qualitative_alpha_sweep)
# ---------------------------------------------------------------------------

_PREAMBLE_RE = re.compile(
    r"(?i)^(here are|the following|below are|sure|of course|certainly|"
    r"i('d| would| can| will)|let me|give \d+|(\d+ )?(unconventional|unusual|creative|alternative) uses|"
    r"uses for)",
)
_MIN_USE_WORDS = 4


def parse_uses(reply: str) -> list[str]:
    """Extract individual uses from a model reply."""
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


def parse_object_from_A(a_text: str) -> str:
    m = re.search(
        r"uses for (?:a |an )?(?:piece of )?([\w\s\-]+?)"
        r"(?:\s*\([^)]*\))?(?:\s+(?:in|focusing|that|as)\b|\.|$)",
        a_text,
        re.I,
    )
    return m.group(1).strip() if m else "object"


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def semantic_diversity(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance among a set of embeddings.

    Returns a value in [0, 2]; higher means more diverse.
    Returns 0.0 if fewer than 2 embeddings.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms
    cos_sim = normed @ normed.T
    total = 0.0
    count = 0
    for i, j in combinations(range(n), 2):
        total += 1.0 - cos_sim[i, j]
        count += 1
    return total / count


def ngram_diversity(uses: list[str], n: int) -> float:
    """Distinct n-gram ratio across all uses. Higher = more diverse."""
    all_ngrams: list[tuple[str, ...]] = []
    for use in uses:
        tokens = use.lower().split()
        all_ngrams.extend(zip(*(tokens[i:] for i in range(n))))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_inference_results(
    inference_dirs: list[Path],
) -> list[dict]:
    """Load all *_results.csv from the given dirs, return flat list of records."""
    records = []
    for d in inference_dirs:
        dir_label = d.name
        for csv_path in sorted(d.glob("*_results.csv")):
            method_slug = csv_path.stem.replace("_results", "")
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                records.append({
                    "dir": dir_label,
                    "method": f"{dir_label}/{method_slug}",
                    "eval_idx": int(row["eval_idx"]),
                    "reply": str(row.get("reply", "")),
                    "problem_text": str(row.get("problem_text", "")),
                })
    return records


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute within-response diversity for AUT inference results.",
    )
    ap.add_argument(
        "--inference-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Directories containing *_results.csv files",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model for embeddings",
    )
    ap.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("results/judge/diversity"),
        help="Output directory for diversity CSVs",
    )
    args = ap.parse_args()

    for d in args.inference_dirs:
        if not d.exists():
            print(f"Warning: {d} does not exist, skipping")

    print(f"Loading sentence-transformer model: {args.model}")
    st_model = SentenceTransformer(args.model)

    records = load_inference_results(
        [d for d in args.inference_dirs if d.exists()]
    )
    print(f"Loaded {len(records)} responses from {len(args.inference_dirs)} dirs")

    if not records:
        print(
            "Error: no *_results.csv found in the given inference dirs, or dirs do not exist.",
            file=sys.stderr,
        )
        print("Each dir should contain files like steered_a0.5_results.csv with columns: eval_idx, reply, problem_text.", file=sys.stderr)
        return 1

    rows = []
    for rec in records:
        uses = parse_uses(rec["reply"])
        obj = parse_object_from_A(rec["problem_text"])

        if len(uses) < 2:
            rows.append({
                "method": rec["method"],
                "eval_idx": rec["eval_idx"],
                "object": obj,
                "n_uses": len(uses),
                "sem_diversity": float("nan"),
                "ngram_div_1": float("nan"),
                "ngram_div_2": float("nan"),
                "ngram_div_3": float("nan"),
            })
            continue

        embeddings = st_model.encode(uses, convert_to_numpy=True)
        sem_div = semantic_diversity(embeddings)
        ng1 = ngram_diversity(uses, 1)
        ng2 = ngram_diversity(uses, 2)
        ng3 = ngram_diversity(uses, 3)

        rows.append({
            "method": rec["method"],
            "eval_idx": rec["eval_idx"],
            "object": obj,
            "n_uses": len(uses),
            "sem_diversity": round(sem_div, 4),
            "ngram_div_1": round(ng1, 4),
            "ngram_div_2": round(ng2, 4),
            "ngram_div_3": round(ng3, 4),
        })

    scores_df = pd.DataFrame(rows)

    if scores_df.empty:
        print("Error: no rows with 2+ uses; cannot compute diversity.", file=sys.stderr)
        return 1

    summary = (
        scores_df
        .groupby("method")
        .agg(
            n_objects=("eval_idx", "count"),
            mean_sem_div=("sem_diversity", "mean"),
            std_sem_div=("sem_diversity", "std"),
            mean_ng1=("ngram_div_1", "mean"),
            mean_ng2=("ngram_div_2", "mean"),
            mean_ng3=("ngram_div_3", "mean"),
        )
        .round(4)
        .sort_values("mean_sem_div", ascending=False)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "diversity_scores.csv"
    summary_path = args.output_dir / "diversity_summary.csv"
    scores_df.to_csv(scores_path, index=False)
    summary.to_csv(summary_path)

    print(f"\nWrote per-response scores: {scores_path}")
    print(f"Wrote summary:             {summary_path}")
    print(f"\n{summary.to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
