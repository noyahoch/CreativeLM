#!/usr/bin/env python3
"""
Extract qualitative examples showing how steering strength α evolves across the alpha sweep.

Reads inference results from results/aut_inference/alpha_sweep/ and optionally judge scores
from results/judge/all_qwen32b/, then outputs markdown with example uses for the same
object at baseline and α ∈ {0.5, 1.0, 1.5, 1.75, 2.0}.

Usage:
  cd DLP && python scripts/qualitative_alpha_sweep.py -o results/judge/qualitative_alpha_evolution.md
  python scripts/qualitative_alpha_sweep.py --inference-dir results/aut_inference/alpha_sweep --judge-dir results/judge/all_qwen32b -o results/judge/qualitative_alpha_evolution.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Reuse parsing from run_aut_inference (minimal copy to avoid import path issues)
PREAMBLE_RE = re.compile(
    r"(?i)^(here are|the following|below are|sure|of course|certainly|"
    r"i('d| would| can| will)|let me|give \d+|(\d+ )?(unconventional|unusual|creative|alternative) uses|"
    r"uses for)",
)
MIN_USE_WORDS = 4


def parse_object_from_A(a_text: str) -> str:
    m = re.search(
        r"uses for (?:a |an )?(?:piece of )?([\w\s\-]+?)(?:\s*\([^)]*\))?(?:\s+(?:in|focusing|that|as)\b|\.|$)",
        a_text,
        re.I,
    )
    return m.group(1).strip() if m else "object"


def parse_uses_from_reply(reply: str) -> list[str]:
    uses = []
    for line in reply.split("\n"):
        line = line.strip()
        line = re.sub(r"^[\-\*\•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = line.strip().rstrip(":")
        if not line or len(line.split()) < MIN_USE_WORDS:
            continue
        if PREAMBLE_RE.search(line):
            continue
        uses.append(line)
    return uses


# Method slug -> (display name, sort order for alpha)
ALPHA_METHODS = [
    ("alpha_sweep_baseline", "Baseline", 0),
    ("alpha_sweep_steered_alpha_0.5", "Steered (α=0.5)", 0.5),
    ("alpha_sweep_steered_alpha_1.0", "Steered (α=1.0)", 1.0),
    ("alpha_sweep_steered_alpha_1.5", "Steered (α=1.5)", 1.5),
    ("alpha_sweep_steered_a1.75", "Steered (α=1.75)", 1.75),
    ("alpha_sweep_steered_alpha_2.0", "Steered (α=2.0)", 2.0),
]

# Inference CSV method names (from run_aut_inference output)
INFERENCE_METHOD_MAP = {
    "alpha_sweep_baseline": "baseline",
    "alpha_sweep_steered_alpha_0.5": "steered_alpha_0.5",
    "alpha_sweep_steered_alpha_1.0": "steered_alpha_1.0",
    "alpha_sweep_steered_alpha_1.5": "steered_alpha_1.5",
    "alpha_sweep_steered_a1.75": "steered_a1.75",
    "alpha_sweep_steered_alpha_2.0": "steered_alpha_2.0",
}


def load_inference_uses(
    inference_dir: Path,
    eval_indices: list[int],
) -> dict[tuple[int, str], list[str]]:
    """(eval_idx, method_slug) -> list of use strings."""
    out: dict[tuple[int, str], list[str]] = {}
    for paper_name, _, _ in ALPHA_METHODS:
        inf_slug = INFERENCE_METHOD_MAP.get(paper_name, paper_name)
        csv_path = inference_dir / f"{inf_slug}_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for eval_idx in eval_indices:
            row = df[df["eval_idx"] == eval_idx]
            if row.empty:
                continue
            reply = row.iloc[0]["reply"]
            problem_text = row.iloc[0]["problem_text"]
            obj = parse_object_from_A(problem_text)
            uses = parse_uses_from_reply(str(reply))
            out[(eval_idx, paper_name)] = uses
    return out


def load_judge_stats(judge_dir: Path) -> dict[str, tuple[float, float]]:
    """method -> (mean_novelty, mean_usability) from summary or from aggregating scores."""
    summary_path = judge_dir / "summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        return {
            row["method"]: (row["mean_novelty"], row["mean_usability"])
            for _, row in df.iterrows()
        }
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Qualitative alpha-sweep examples for the paper")
    ap.add_argument("--inference-dir", type=Path, default=Path("results/aut_inference/alpha_sweep"))
    ap.add_argument("--judge-dir", type=Path, default=Path("results/judge/all_qwen32b"))
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--max-uses", type=int, default=3, help="Max uses to show per method per object")
    ap.add_argument("--objects", type=int, default=3, help="Number of objects (eval_idx 0,1,...) to show")
    args = ap.parse_args()

    inference_dir = args.inference_dir
    if not inference_dir.exists():
        print(f"Error: inference dir not found: {inference_dir}")
        return 1

    eval_indices = list(range(args.objects))
    uses_by_key = load_inference_uses(inference_dir, eval_indices)

    judge_stats = {}
    if args.judge_dir.exists():
        judge_stats = load_judge_stats(args.judge_dir)

    lines = [
        "# Qualitative analysis: how α evolves",
        "",
        "Below we show the **first few alternative uses** generated for the same object (same prompt) at **baseline** and at increasing steering strength **α**. The progression illustrates α as a continuous knob from conventional → more novel → over-steered.",
        "",
    ]

    for eval_idx in eval_indices:
        # Get object name from any available method
        obj_name = None
        for paper_name, _, _ in ALPHA_METHODS:
            key = (eval_idx, paper_name)
            if key in uses_by_key and uses_by_key[key]:
                # We need problem_text to get object; get from first method's CSV
                for inf_slug in INFERENCE_METHOD_MAP.values():
                    csv_path = inference_dir / f"{inf_slug}_results.csv"
                    if csv_path.exists():
                        df = pd.read_csv(csv_path)
                        row = df[df["eval_idx"] == eval_idx]
                        if not row.empty:
                            obj_name = parse_object_from_A(row.iloc[0]["problem_text"])
                            break
            if obj_name:
                break
        if not obj_name:
            obj_name = f"Object (eval_idx={eval_idx})"

        lines.append(f"## {obj_name}")
        lines.append("")

        for paper_name, display_name, alpha in ALPHA_METHODS:
            key = (eval_idx, paper_name)
            uses = uses_by_key.get(key, [])
            # summary.csv uses internal method names (alpha_sweep_baseline, alpha_sweep_steered_alpha_0.5, ...)
            summary_key = INFERENCE_METHOD_MAP.get(paper_name, paper_name)
            if summary_key == "baseline":
                summary_key = "alpha_sweep_baseline"
            else:
                summary_key = "alpha_sweep_" + summary_key
            stats = judge_stats.get(summary_key)

            sub = []
            sub.append(f"**{display_name}**")
            if stats:
                sub.append(f" *(mean novelty {stats[0]:.2f}, usability {stats[1]:.2f})*")
            sub.append("\n\n")
            for i, u in enumerate(uses[: args.max_uses]):
                sub.append(f"{i+1}. {u}\n")
            if len(uses) > args.max_uses:
                sub.append(f"   *… and {len(uses) - args.max_uses} more uses.*\n")
            lines.append("".join(sub))
            lines.append("")

    md = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"Wrote: {args.output}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
