#!/usr/bin/env python3
"""
LLM-as-a-judge via vLLM — score uses.txt files for novelty and usability.

Reads one or more *_uses.txt files (format: "Object, use description" per line),
sends each use to a vLLM-served judge model, and outputs:

  - {method}_scores.csv  per-use scores for each method (one file per method)
  - summary.csv          per-method aggregated statistics (one table)
  - Prints the summary comparison table to stdout

Requires a vLLM server running the judge model. For speed, use tensor parallelism
and enough workers so the server can handle --max-concurrent (default 256), e.g.:
  python -m vllm.entrypoints.openai.api_server \
    --model prometheus-eval/prometheus-7b-v2.0 --port 8000 --tensor-parallel-size 2

Usage:
  cd DLP

  # Score a single file
  python scripts/run_vllm_judge.py \
    --input results/aut_inference/llama_t07/baseline_uses.txt \
    --output-dir results/judge/llama_t07

  # Score multiple files and compare
  python scripts/run_vllm_judge.py \
    --input results/aut_inference/llama_t07/baseline_uses.txt \
    --input results/aut_inference/llama_t07/steered_a1.0_uses.txt \
    --input results/aut_inference/crpo_llama_nov_t07/baseline_uses.txt \
    --output-dir results/judge/compare_llama

  # Custom vLLM endpoint and model
  python scripts/run_vllm_judge.py \
    --input results/aut_inference/llama_t07/baseline_uses.txt \
    --vllm-url http://localhost:8000/v1 \
    --judge-model Qwen/Qwen3-32B \
    --output-dir results/judge/llama_qwen_judge

  # Debug: judge only first N uses per file
  python scripts/run_vllm_judge.py \
    -i results/aut_inference/llama_t07/baseline_uses.txt \
    -o results/judge/debug \
    --max-uses 20

  # Run on all *_uses.txt under a folder (e.g. all inference runs)
  python scripts/run_vllm_judge.py \
    --input-dir results/aut_inference \
    -o results/judge/all_inference

  # Same with max-uses per file (each method gets at most N uses)
  python scripts/run_vllm_judge.py \
    --input-dir results/aut_inference \
    -o results/judge/all_sampled \
    --max-uses 50

  # Incremental run: skip methods already judged, only judge new ones,
  # then rebuild summary from all scores (existing + new)
  python scripts/run_vllm_judge.py \
    --input-dir results/aut_inference \
    -o results/judge/all_qwen32b \
    --skip-existing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Parsing uses.txt ──────────────────────────────────────────────────────────


def parse_uses_file(path: Path) -> pd.DataFrame:
    """Parse a uses.txt file into a DataFrame with columns: method, object, use_text.

    Expected format per line: "Object, use description"
    The method is derived from the filename stem (e.g. "baseline_uses.txt" → "baseline").
    """
    stem = path.stem
    method = re.sub(r"_uses$", "", stem)

    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        comma_pos = line.find(",")
        if comma_pos == -1:
            continue
        obj = line[:comma_pos].strip().lower()
        use = line[comma_pos + 1:].strip()
        if not use:
            continue
        rows.append({"method": method, "object": obj, "use_text": use})

    return pd.DataFrame(rows)


def _content_key(df: pd.DataFrame) -> int:
    """Hash (object, use_text) rows so identical files get the same key."""
    return hash(tuple(zip(df["object"].tolist(), df["use_text"].tolist())))


# ── Judge prompt (matches vllm_judge.ipynb active version) ────────────────────


def build_judge_prompt(object_name: str, use_text: str) -> str:
    use_preview = (use_text or "")[:400]
    return f"""\
Output ONLY one valid JSON object. No extra text.

Object: {object_name}
Proposed use: {use_preview}

You are a judge for a dataset of intentionally unconventional uses.
This dataset contains many intentionally unconventional uses, so do not default to 1.
Choose the closest anchor below.

NOVELTY (1–10): how uncommon/non-obvious the underlying idea is for this object+setting.
- 1–2: truly default / first-thought / widely common
- 3–4: common pattern with a small twist
- 5–6: clearly non-default; plausible new angle
- 7–8: rare and surprising; object-specific in a meaningful way
- 9–10: extremely rare; would surprise most evaluators; still coherent

USABILITY (1–10): likelihood it works as stated with realistic constraints/resources.
- 1–2: basically won't work as stated
- 3–4: fragile / missing key details / major assumptions
- 5–6: plausible but needs care or minor additions
- 7–8: practical and straightforward
- 9–10: very reliable/robust

Output JSON exactly:
{{
  "analysis": {{
    "key_object_property": "string (<= 8 words)",
    "reasoning": "string (<= 20 words)"
  }},
  "scores": {{
    "novelty": integer,
    "usability": integer
  }}
}}"""


CREATIVITY_ONLY_PROMPT_TEMPLATE = """\
Output ONLY one valid JSON object. No extra text.
Rate the creativity and originality of the following alternative use for a {object_name} on a scale of 1 to 5:

Alternative use: {use_text}

Rating criteria:
1 - Common/obvious use, not creative
2 - Somewhat uncommon but still fairly obvious
3 - Moderately creative and original
4 - Very creative and original
5 - Exceptionally creative, highly original and unconventional

Respond with only the number (1-5):"""


# 1–5 scale with one decimal, for comparison to OSCAI originality scores
OSCAI_SCALE_PROMPT_TEMPLATE = """\
Rate the creativity and originality of the following alternative use for a {object_name} on a scale of 1.0 to 5.0.

Alternative use: {use_text}

Use one decimal (e.g. 3.6, 4.2). Criteria:
1.0–1.9 - Common/obvious use
2.0–2.9 - Somewhat uncommon but fairly obvious
3.0–3.9 - Moderately creative and original
4.0–4.9 - Very creative and original
5.0 - Exceptionally creative and unconventional

Output only the number (e.g. 3.6):"""


def build_oscai_scale_prompt(object_name: str, use_text: str) -> str:
    return OSCAI_SCALE_PROMPT_TEMPLATE.format(
        object_name=object_name,
        use_text=(use_text or "")[:400],
    )


def build_creativity_only_prompt(object_name: str, use_text: str) -> str:
    return CREATIVITY_ONLY_PROMPT_TEMPLATE.format(
        object_name=object_name,
        use_text=(use_text or "")[:400],
    )


# ── vLLM async client ────────────────────────────────────────────────────────


async def judge_one(
    client,
    model: str,
    object_name: str,
    use_text: str,
    sema: asyncio.Semaphore,
    prompt_style: str,
    temperature: float,
    max_retries: int,
    timeout_s: float,
) -> dict:
    if prompt_style == "creativity_and_usability":
        prompt = build_judge_prompt(object_name, use_text)
    elif prompt_style == "oscai_scale":
        prompt = build_oscai_scale_prompt(object_name, use_text)
    else:
        prompt = build_creativity_only_prompt(object_name, use_text)

    last_raw = ""
    for attempt in range(max_retries):
        async with sema:
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=400,
                    ),
                    timeout=timeout_s,
                )
                last_raw = (resp.choices[0].message.content or "").strip()

                # Extract JSON block if wrapped in extra text
                if not last_raw.startswith("{"):
                    start, end = last_raw.find("{"), last_raw.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        last_raw = last_raw[start:end + 1]

                if prompt_style == "creativity_only":
                    score = _extract_creativity_only_score(last_raw)
                    return {
                        "novelty": score, "usability": None, "overall": score,
                        "reasoning": "", "raw": last_raw,
                        "err": "", "attempt": attempt,
                    }
                if prompt_style == "oscai_scale":
                    score = _extract_oscai_scale_score(last_raw)
                    return {
                        "novelty": score, "usability": None, "overall": score,
                        "reasoning": "", "raw": last_raw,
                        "err": "", "attempt": attempt,
                    }

                d = json.loads(last_raw)
                n = int(d["scores"]["novelty"])
                u = int(d["scores"]["usability"])
                o = round((n + u) / 2)
                return {
                    "novelty": n, "usability": u, "overall": o,
                    "reasoning": d.get("analysis", {}).get("reasoning", ""),
                    "raw": last_raw, "err": "", "attempt": attempt,
                }

            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "novelty": None, "usability": None, "overall": None,
                        "reasoning": str(e), "raw": last_raw,
                        "err": f"{type(e).__name__}: {e}", "attempt": attempt,
                    }
    return {}


def _extract_creativity_only_score(raw: str) -> int | None:
    """Extract a single integer 1-5 from the creativity-only prompt response."""
    m = re.search(r"[1-5]", raw)
    return int(m.group()) if m else None


def _extract_oscai_scale_score(raw: str) -> float | None:
    """Extract a float 1.0–5.0 (one decimal) for OSCAI-scale comparison."""
    m = re.search(r"[1-5]\.\d", raw)
    if m:
        return round(float(m.group()), 1)
    m = re.search(r"[1-5]", raw)
    return float(m.group()) if m else None


async def judge_all(
    use_df: pd.DataFrame,
    vllm_url: str,
    judge_model: str,
    prompt_style: str,
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
    for pos, (_, r) in enumerate(use_df.iterrows()):
        t = asyncio.create_task(
            judge_one(
                client, judge_model,
                r["object"], r["use_text"],
                sema, prompt_style, temperature,
                max_retries, timeout_s,
            )
        )
        tasks.append(t)
        index_by_future[t] = pos

    results = [None] * len(tasks)
    pending = set(tasks)
    with tqdm(total=len(tasks), desc="Judging") as pbar:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                pos = index_by_future[t]
                try:
                    results[pos] = t.result()
                except Exception as e:
                    results[pos] = {"novelty": None, "usability": None, "overall": None,
                                    "reasoning": str(e), "raw": "", "err": str(e), "attempt": -1}
                pbar.update(1)

    out = use_df.copy()
    out["judge_novelty"] = None
    out["judge_usability"] = None
    out["judge_overall"] = None
    out["judge_reasoning"] = ""

    for pos, res in enumerate(results):
        row_ix = out.index[pos]
        if res is None or isinstance(res, Exception):
            out.at[row_ix, "judge_reasoning"] = str(res)
            continue
        out.at[row_ix, "judge_novelty"] = res.get("novelty")
        out.at[row_ix, "judge_usability"] = res.get("usability")
        out.at[row_ix, "judge_overall"] = res.get("overall")
        out.at[row_ix, "judge_reasoning"] = res.get("reasoning", "")

    return out


# ── Aggregation ───────────────────────────────────────────────────────────────


def compute_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    for c in ["judge_novelty", "judge_usability", "judge_overall"]:
        if c in scored_df.columns:
            scored_df[c] = pd.to_numeric(scored_df[c], errors="coerce")

    agg_dict = dict(
        n_uses=("judge_overall", "count"),
        n_missing=("judge_overall", lambda s: int(pd.isna(s).sum())),
        mean_overall=("judge_overall", "mean"),
        mean_novelty=("judge_novelty", "mean"),
    )
    if scored_df["judge_usability"].notna().any():
        agg_dict.update(
            mean_usability=("judge_usability", "mean"),
            median_novelty=("judge_novelty", lambda s: float(np.nanpercentile(s, 50))),
            median_usability=("judge_usability", lambda s: float(np.nanpercentile(s, 50))),
            std_overall=("judge_overall", "std"),
        )

    summary = (
        scored_df
        .groupby("method", dropna=False)
        .agg(**agg_dict)
        .round(3)
        .reset_index()
    )

    baseline_novelty = None
    if "baseline" in summary["method"].values:
        baseline_novelty = summary.set_index("method").loc["baseline", "mean_novelty"]
    if baseline_novelty is not None:
        summary["delta_novelty_vs_baseline"] = (summary["mean_novelty"] - baseline_novelty).round(3)

    return summary.sort_values("mean_novelty", ascending=False)


# ── Incremental helpers ───────────────────────────────────────────────────────


def _load_existing_scores(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all existing *_scores.csv files from the output directory.

    Returns a dict mapping method_name → scored DataFrame.
    """
    existing: dict[str, pd.DataFrame] = {}
    if not output_dir.exists():
        return existing
    for csv_path in sorted(output_dir.glob("*_scores.csv")):
        method_name = re.sub(r"_scores$", "", csv_path.stem)
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                existing[method_name] = df
        except Exception as e:
            print(f"  Warning: could not load {csv_path.name}: {e}", file=sys.stderr)
    return existing


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge via vLLM — score uses.txt files for novelty and usability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", action="append", dest="inputs", type=Path,
        help="Path to *_uses.txt file (repeatable). Ignored if --input-dir is set.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Discover all *_uses.txt under this path (recursive). Use this or -i.",
    )
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible API base URL")
    parser.add_argument("--judge-model", default="prometheus-eval/prometheus-7b-v2.0",
                        help="Judge model served by vLLM")
    parser.add_argument("--prompt-style",
                        choices=["creativity_only", "creativity_and_usability", "oscai_scale"],
                        default="creativity_and_usability",
                        help="creativity_only = 1-5 integer; "
                             "creativity_and_usability = novelty + usability 1-10 (default); "
                             "oscai_scale = 1-5 with one decimal for OSCAI comparison")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Judge sampling temperature (default: 0 = deterministic)")
    parser.add_argument("--max-concurrent", type=int, default=256,
                        help="Max concurrent requests to vLLM (default: 256; raise if judge is slow)")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Per-request timeout in seconds")
    parser.add_argument("--max-uses", type=int, default=None,
                        help="Cap uses per file/method: judge at most N uses from each input (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip methods that already have _scores.csv in --output-dir. "
                             "Existing scores are loaded and merged with new scores for the summary.")

    args = parser.parse_args()

    # Resolve input paths: --input-dir (recursive glob) or explicit --input list
    if args.input_dir is not None:
        if not args.input_dir.exists() or not args.input_dir.is_dir():
            print(f"Error: --input-dir {args.input_dir} not found or not a directory", file=sys.stderr)
            return 1
        input_paths = sorted(args.input_dir.rglob("*_uses.txt"))
        if not input_paths:
            print(f"Error: no *_uses.txt files under {args.input_dir}", file=sys.stderr)
            return 1
        print(f"Found {len(input_paths)} *_uses.txt files under {args.input_dir}")
    else:
        if not args.inputs:
            print("Error: provide --input FILE(s) or --input-dir DIR", file=sys.stderr)
            return 1
        input_paths = args.inputs

    # Parse all input files; apply --max-uses per file (per method)
    use_input_dir = args.input_dir is not None
    dfs = []
    for p in input_paths:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            return 1
        df = parse_uses_file(p)
        if df.empty:
            print(f"Warning: {p} yielded 0 uses, skipping", file=sys.stderr)
            continue
        # When discovering from a dir, keep methods distinct by folder (e.g. llama_t07_baseline)
        if use_input_dir:
            base_method = df["method"].iloc[0]
            df = df.copy()
            df["method"] = f"{p.parent.name}_{base_method}"
        if args.max_uses is not None:
            df = df.head(args.max_uses)
            print(f"  {p.name}: {len(df)} uses (--max-uses {args.max_uses})  method={df['method'].iloc[0]}")
        else:
            print(f"  {p.name}: {len(df)} uses  method={df['method'].iloc[0]}")
        dfs.append(df)

    if not dfs:
        print("Error: no uses loaded from any input file", file=sys.stderr)
        return 1

    # --skip-existing: detect methods already judged in the output dir
    existing_scores: dict[str, pd.DataFrame] = {}
    if args.skip_existing:
        existing_scores = _load_existing_scores(args.output_dir)
        if existing_scores:
            print(f"\n--skip-existing: found {len(existing_scores)} already-judged method(s) in {args.output_dir}")

        all_methods_discovered = {df["method"].iloc[0] for df in dfs}
        new_dfs = [df for df in dfs if df["method"].iloc[0] not in existing_scores]
        skipped = all_methods_discovered - {df["method"].iloc[0] for df in new_dfs}
        if skipped:
            print(f"  Skipping {len(skipped)} method(s): {sorted(skipped)}")
        dfs = new_dfs

    # If nothing new to judge, just rebuild the summary from existing scores
    if not dfs:
        if existing_scores:
            print("\nNo new methods to judge — rebuilding summary from existing scores.")
            all_scored = pd.concat(existing_scores.values(), ignore_index=True)
            summary = compute_summary(all_scored)
            summary_path = args.output_dir / "summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"Summary: {summary_path}\n")
            print(summary.to_string(index=False))
            return 0
        print("Error: no uses loaded from any input file", file=sys.stderr)
        return 1

    # Deduplicate by content: identical files (e.g. same baseline in two folders) judged once, scores reused
    by_content: dict[int, list[pd.DataFrame]] = {}
    for df in dfs:
        key = _content_key(df)
        by_content.setdefault(key, []).append(df)

    canonical_to_all_methods: dict[str, list[str]] = {}
    unique_dfs = []
    for key, group in by_content.items():
        first = group[0]
        methods = [g["method"].iloc[0] for g in group]
        canonical = methods[0]
        canonical_to_all_methods[canonical] = methods
        first = first.copy()
        first["method"] = canonical
        unique_dfs.append(first)
        if len(methods) > 1:
            print(f"  [dedup] {len(methods)} identical files → judge once, reuse for: {methods}")

    use_df = pd.concat(unique_dfs, ignore_index=True)
    print(f"\nTotal: {len(use_df)} uses across {use_df['method'].nunique()} new method(s) to judge (after content dedup)")

    # Run judging
    scored_df = asyncio.run(judge_all(
        use_df,
        vllm_url=args.vllm_url,
        judge_model=args.judge_model,
        prompt_style=args.prompt_style,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        timeout_s=args.timeout,
    ))

    # Expand: assign same scores to all method names that shared this content
    expanded = []
    for _, row in scored_df.iterrows():
        methods = canonical_to_all_methods.get(row["method"], [row["method"]])
        for m in methods:
            r = row.to_dict()
            r["method"] = m
            expanded.append(r)
    scored_df = pd.DataFrame(expanded)

    # Save per-use scores: one file per method
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_scored = 0
    total_failed = 0
    for method_name, group in scored_df.groupby("method", sort=False):
        safe_name = re.sub(r'[^\w\-.]', "_", str(method_name))
        scores_path = args.output_dir / f"{safe_name}_scores.csv"
        group.to_csv(scores_path, index=False)
        n_ok = group["judge_novelty"].notna().sum()
        total_scored += n_ok
        total_failed += len(group) - n_ok
        print(f"  {scores_path.name}: {len(group)} rows ({n_ok} scored)")
    print(f"\nPer-method scores in {args.output_dir}  (total scored: {total_scored}, failed: {total_failed})")

    # Merge existing + new scores for summary (when --skip-existing)
    if existing_scores:
        all_scored = pd.concat([*existing_scores.values(), scored_df], ignore_index=True)
        print(f"  Merged {len(existing_scores)} existing + {scored_df['method'].nunique()} new method(s) for summary")
    else:
        all_scored = scored_df

    # Compute and save summary
    summary = compute_summary(all_scored)
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}\n")
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
