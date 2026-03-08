#!/usr/bin/env python3
"""
LLM-as-a-judge quality scoring for MacGyver creative problem-solving.

Reads inference results (CSV / JSONL with problem + model response), sends
each pair to a vLLM-served judge model using the MacGyver 5-point additive
rubric, and writes:

  - {method}_macgyver_scores.csv   per-item scores (one file per method)
  - macgyver_quality_summary.csv   per-method aggregated statistics

The rubric awards 1 point each for: (1) using only given resources,
(2) correct understanding of resource properties, (3) physical feasibility,
(4) practical effectiveness, (5) completeness and logical structure.

Examples:
  cd DLP

  # Score a single inference results file
  python scripts/score_macgyver_quality.py \
    --input results/macgyver_inference/baseline_results.csv \
    -o results/novelty/macgyver_quality

  # Score all *_results.csv under a directory
  python scripts/score_macgyver_quality.py \
    --input-dir results/macgyver_inference \
    -o results/novelty/macgyver_quality

  # Custom vLLM endpoint and model
  python scripts/score_macgyver_quality.py \
    --input results/macgyver_inference/baseline_results.csv \
    --vllm-url http://localhost:8000/v1 \
    --judge-model Qwen/Qwen3-32B \
    -o results/novelty/macgyver_quality

  # Skip methods already scored (incremental)
  python scripts/score_macgyver_quality.py \
    --input-dir results/macgyver_inference \
    -o results/novelty/macgyver_quality \
    --skip-existing

Progress: The script prints "[progress] N/Total (pct%)" every ~2%. A tqdm bar
is shown only when stderr is a real TTY (e.g. a normal terminal); in IDE
terminals the bar is disabled to avoid broken output. To see live request count
from the server side, run scripts/vllm_request_progress.py --total N in another
terminal. Use "python -u" for unbuffered stderr if progress lines don't appear.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import importlib.util

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import directly from the module files to avoid pulling heavy deps
# (e.g. transformers) via dlp.evaluation.__init__.
_DLP_ROOT = Path(__file__).resolve().parent.parent / "dlp" / "evaluation"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_prompts_mod = _load_module("dlp.evaluation.prompts", _DLP_ROOT / "prompts.py")
_api_mod = _load_module("dlp.evaluation.api", _DLP_ROOT / "api.py")
_judge_mod = _load_module("dlp.evaluation.judge", _DLP_ROOT / "judge.py")

MacGyverJudge = _judge_mod.MacGyverJudge

# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

_PROMPT_COL_CANDIDATES = ("user_prompt", "problem", "prompt", "Problem")
_REPLY_COL_CANDIDATES = ("reply", "model_response", "response", "Solution", "solution")


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find {label} column. Tried {candidates}; "
        f"available: {list(df.columns)}"
    )


def load_results_file(path: Path) -> pd.DataFrame:
    """Load a CSV or JSONL of inference results into a standardised DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".jsonl", ".json"):
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    prompt_col = _resolve_column(df, _PROMPT_COL_CANDIDATES, "prompt")
    reply_col = _resolve_column(df, _REPLY_COL_CANDIDATES, "reply")

    df = df.rename(columns={prompt_col: "user_prompt", reply_col: "model_response"})

    if "method" not in df.columns:
        df["method"] = re.sub(r"_results$", "", path.stem)
    if "id" not in df.columns:
        df["id"] = range(len(df))
    df["id"] = df["id"].astype(str)

    return df


# ---------------------------------------------------------------------------
# Async batch judging (delegates to MacGyverJudge.rate_one_async)
# ---------------------------------------------------------------------------

async def judge_all(
    df: pd.DataFrame,
    vllm_url: str,
    judge_model: str,
    temperature: float,
    max_concurrent: int,
    max_retries: int,
    timeout_s: float,
) -> pd.DataFrame:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=vllm_url, api_key="vllm")
    sema = asyncio.Semaphore(max_concurrent)

    tasks = []
    index_by_future: dict[asyncio.Task, int] = {}
    for pos, (_, row) in enumerate(df.iterrows()):
        t = asyncio.create_task(
            MacGyverJudge.rate_one_async(
                client,
                judge_model,
                row["user_prompt"],
                row["model_response"],
                sema,
                temperature=temperature,
                max_retries=max_retries,
                timeout_s=timeout_s,
            )
        )
        tasks.append(t)
        index_by_future[t] = pos

    results: list[dict | None] = [None] * len(tasks)
    pending = set(tasks)
    total = len(tasks)
    progress_interval = max(1, total // 50)  # print every ~2%
    completed = 0
    # In many IDE terminals stderr is not a TTY, so tqdm won't draw the bar
    # (it would spam newlines instead). Disable the bar when not a TTY; we
    # always print [progress] lines so you see progress either way.
    use_tqdm = sys.stderr.isatty()
    with tqdm(
        total=total,
        desc="Judging MacGyver",
        file=sys.stderr,
        mininterval=1.0,
        miniters=1,
        disable=not use_tqdm,
        dynamic_ncols=False,
        ncols=80,
    ) as pbar:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                pos = index_by_future[t]
                try:
                    results[pos] = t.result()
                except Exception as e:
                    results[pos] = {
                        "quality_score": None,
                        "judge_explanation": str(e),
                        "err": str(e),
                        "attempt": -1,
                    }
                completed += 1
                pbar.update(1)
                if completed % progress_interval == 0 or completed == total:
                    msg = f"  [progress] {completed}/{total} ({100 * completed / total:.1f}%)"
                    print(msg, file=sys.stderr, flush=True)

    out = df.copy()
    out["quality_score"] = None
    out["judge_explanation"] = ""

    for pos, res in enumerate(results):
        ix = out.index[pos]
        if res is None:
            continue
        out.at[ix, "quality_score"] = res.get("quality_score")
        out.at[ix, "judge_explanation"] = res.get("judge_explanation", "")

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    scored_df = scored_df.copy()
    scored_df["quality_score"] = pd.to_numeric(
        scored_df["quality_score"], errors="coerce"
    )

    summary = (
        scored_df.groupby("method", dropna=False)
        .agg(
            n_items=("quality_score", "count"),
            n_scored=("quality_score", lambda s: int(s.notna().sum())),
            mean_quality=("quality_score", "mean"),
            median_quality=(
                "quality_score",
                lambda s: float(np.nanpercentile(s.dropna(), 50)) if s.notna().any() else float("nan"),
            ),
            std_quality=("quality_score", "std"),
            min_quality=("quality_score", "min"),
            max_quality=("quality_score", "max"),
        )
        .round(3)
        .reset_index()
    )

    baseline_quality = None
    if "baseline" in summary["method"].values:
        baseline_quality = summary.set_index("method").loc["baseline", "mean_quality"]
    if baseline_quality is not None:
        summary["delta_vs_baseline"] = (
            summary["mean_quality"] - baseline_quality
        ).round(3)

    return summary.sort_values("mean_quality", ascending=False)


# ---------------------------------------------------------------------------
# Incremental helpers
# ---------------------------------------------------------------------------

def _load_existing_scores(output_dir: Path) -> dict[str, pd.DataFrame]:
    existing: dict[str, pd.DataFrame] = {}
    if not output_dir.exists():
        return existing
    for csv_path in sorted(output_dir.glob("*_macgyver_scores.csv")):
        method_name = re.sub(r"_macgyver_scores$", "", csv_path.stem)
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                existing[method_name] = df
        except Exception as e:
            print(f"  Warning: could not load {csv_path.name}: {e}", file=sys.stderr)
    return existing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge quality scoring for MacGyver inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", action="append", dest="inputs", type=Path,
        help="Path to inference results file (CSV/JSONL). Repeatable.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Discover all *_results.csv under this path (recursive).",
    )
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument(
        "--prompt-col", default=None,
        help="Override prompt column name (auto-detected by default).",
    )
    parser.add_argument(
        "--reply-col", default=None,
        help="Override reply column name (auto-detected by default).",
    )
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--judge-model", default="Qwen/Qwen2.5-32B-Instruct-AWQ",
        help="Judge model served by vLLM",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Judge sampling temperature (default: 0)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=256,
        help="Max concurrent requests to vLLM",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--timeout", type=float, default=600.0,
        help="Per-request timeout in seconds (default: 600; must be long enough for concurrent generation)",
    )
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Cap items per file/method (default: all)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip methods already scored in --output-dir.",
    )

    args = parser.parse_args()

    # ── Resolve inputs ────────────────────────────────────────────────────
    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            print(f"Error: --input-dir {args.input_dir} not found", file=sys.stderr)
            return 1
        input_paths = sorted(args.input_dir.rglob("*_results.csv"))
        if not input_paths:
            input_paths = sorted(args.input_dir.rglob("*.csv"))
        if not input_paths:
            print(f"Error: no CSV files under {args.input_dir}", file=sys.stderr)
            return 1
        print(f"Found {len(input_paths)} result files under {args.input_dir}")
    else:
        if not args.inputs:
            print("Error: provide --input FILE(s) or --input-dir DIR", file=sys.stderr)
            return 1
        input_paths = args.inputs

    # ── Load and normalise ────────────────────────────────────────────────
    dfs: list[pd.DataFrame] = []
    use_input_dir = args.input_dir is not None
    for p in input_paths:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            return 1
        try:
            df = load_results_file(p)
        except ValueError as e:
            print(f"  Warning: skipping {p}: {e}", file=sys.stderr)
            continue

        if args.prompt_col and args.prompt_col in df.columns:
            df = df.rename(columns={args.prompt_col: "user_prompt"})
        if args.reply_col and args.reply_col in df.columns:
            df = df.rename(columns={args.reply_col: "model_response"})

        if use_input_dir:
            base_method = df["method"].iloc[0]
            df = df.copy()
            df["method"] = f"{p.parent.name}_{base_method}"

        if args.max_items is not None:
            df = df.head(args.max_items)

        print(f"  {p.name}: {len(df)} items  method={df['method'].iloc[0]}")
        dfs.append(df)

    if not dfs:
        print("Error: no items loaded from any input file", file=sys.stderr)
        return 1

    # ── Skip existing ─────────────────────────────────────────────────────
    existing_scores: dict[str, pd.DataFrame] = {}
    if args.skip_existing:
        existing_scores = _load_existing_scores(args.output_dir)
        if existing_scores:
            print(
                f"\n--skip-existing: found {len(existing_scores)} already-scored "
                f"method(s) in {args.output_dir}"
            )
        new_dfs = [
            df for df in dfs
            if df["method"].iloc[0] not in existing_scores
        ]
        skipped = {df["method"].iloc[0] for df in dfs} - {
            df["method"].iloc[0] for df in new_dfs
        }
        if skipped:
            print(f"  Skipping {len(skipped)} method(s): {sorted(skipped)}")
        dfs = new_dfs

    if not dfs:
        if existing_scores:
            print("\nNo new methods to judge — rebuilding summary from existing scores.")
            all_scored = pd.concat(existing_scores.values(), ignore_index=True)
            summary = compute_summary(all_scored)
            summary_path = args.output_dir / "macgyver_quality_summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"Summary: {summary_path}\n")
            print(summary.to_string(index=False))
            return 0
        print("Error: no items to judge", file=sys.stderr)
        return 1

    combined_df = pd.concat(dfs, ignore_index=True)
    print(
        f"\nTotal: {len(combined_df)} items across "
        f"{combined_df['method'].nunique()} method(s)"
    )

    # ── Judge ─────────────────────────────────────────────────────────────
    scored_df = asyncio.run(
        judge_all(
            combined_df,
            vllm_url=args.vllm_url,
            judge_model=args.judge_model,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent,
            max_retries=args.max_retries,
            timeout_s=args.timeout,
        )
    )

    # ── Save per-method scores ────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_scored = 0
    total_failed = 0
    for method_name, group in scored_df.groupby("method", sort=False):
        safe_name = re.sub(r"[^\w\-.]", "_", str(method_name))
        scores_path = args.output_dir / f"{safe_name}_macgyver_scores.csv"
        group.to_csv(scores_path, index=False)
        n_ok = group["quality_score"].notna().sum()
        total_scored += n_ok
        total_failed += len(group) - n_ok
        print(f"  {scores_path.name}: {len(group)} rows ({n_ok} scored)")

    print(
        f"\nPer-method scores in {args.output_dir}  "
        f"(total scored: {total_scored}, failed: {total_failed})"
    )

    # ── Summary ───────────────────────────────────────────────────────────
    if existing_scores:
        all_scored = pd.concat(
            [*existing_scores.values(), scored_df], ignore_index=True
        )
    else:
        all_scored = scored_df

    summary = compute_summary(all_scored)
    summary_path = args.output_dir / "macgyver_quality_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}\n")
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
