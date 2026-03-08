#!/usr/bin/env python3
"""
Brainstorm-then-Select: LLM self-evaluation for AUT uses.

Implements the selection phase from Summers-Stay et al.,
"Brainstorm, then Select: a Generative Language Model Improves Its
Creativity Score" (2022).

Pipeline:
  Phase 1 (brainstorm) — already done by run_aut_inference.py:
      Generate many uses per object with sampling.

  Phase 2 (select) — THIS SCRIPT:
      For each use the *same LLM* evaluates:
        1. Advantages of the proposed use  (generated text)
        2. Drawbacks of the proposed use   (generated text)
        3. Utility:    "Is this a good idea? Yes/No"  → log P(Yes)-log P(No)
        4. Originality: "Would you be surprised?  Yes/No" → log P(Yes)-log P(No)
      Filter uses by confidence threshold on utility, originality, or both.

Usage:
  # Full chain (advantages → drawbacks → utility + originality)
  python scripts/run_brainstorm_select.py \\
    --input results/aut_inference/baseline_sampled/baseline_uses.txt \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --filter both \\
    --output results/aut_inference/baseline_sampled/selected

  # Skip chain — directly score utility/originality (faster, less accurate)
  python scripts/run_brainstorm_select.py \\
    --input results/aut_inference/baseline_sampled/baseline_uses.txt \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --skip-chain \\
    --filter originality \\
    --output results/aut_inference/baseline_sampled/selected_orig

Output (in directory of --output):
  {stem}_uses.txt     filtered "Object, use" lines
  {stem}_scores.csv   per-use scores and generated reasoning
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from dlp.models import HFLoader


# ── Prompts (adapted from Tables 2 & 4 of Summers-Stay et al.) ───────────────

def _advantages_messages(obj: str, use: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": (
        f"Q: Name one or more advantages to using a {obj} "
        f"for the following purpose: {use}\nA:"
    )}]


def _drawbacks_messages(obj: str, use: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": (
        f"Q: Name one or more drawbacks to using a {obj} "
        f"for the following purpose: {use}\nA:"
    )}]


def _utility_messages_chain(
    obj: str, use: str, advantages: str, drawbacks: str,
) -> list[dict[str, str]]:
    """Utility evaluation with advantages/drawbacks chain-of-thought."""
    return [{"role": "user", "content": (
        f"Advantages: {advantages}\n"
        f"Drawbacks: {drawbacks}\n"
        f"Q: Based on these advantages and drawbacks, do you think "
        f"using a {obj} for the purpose '{use}' is a good idea? "
        f"Answer Yes or No.\nA:"
    )}]


def _utility_messages_direct(obj: str, use: str) -> list[dict[str, str]]:
    """Simplified utility evaluation (no chain)."""
    return [{"role": "user", "content": (
        f"Q: Do you think using a {obj} for the purpose '{use}' "
        f"is a good and practical idea? Answer Yes or No.\nA:"
    )}]


def _originality_messages(obj: str, use: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": (
        f"Q: If someone suggested using a {obj} for the following purpose: "
        f"{use}, would you be surprised and think it was a novel idea? "
        f"Answer Yes or No.\nA:"
    )}]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_uses(path: Path) -> list[tuple[str, str]]:
    """Parse uses.txt → list of (object, use) tuples."""
    items: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        comma_idx = line.find(",")
        if comma_idx == -1:
            continue
        obj = line[:comma_idx].strip()
        use = line[comma_idx + 1:].strip()
        if obj and use:
            items.append((obj, use))
    return items


def _batched(seq: list, bs: int):
    """Yield successive bs-sized chunks from seq."""
    for i in range(0, len(seq), bs):
        yield seq[i : i + bs]


# ── Core evaluation ──────────────────────────────────────────────────────────

YES_NO_TOKENS = ["Yes", "No"]


def _run_full_chain(
    loader: HFLoader,
    uses: list[tuple[str, str]],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict]:
    """Run the full advantages → drawbacks → utility + originality chain."""
    n = len(uses)
    advantages = [""] * n
    drawbacks = [""] * n
    utility_conf = [0.0] * n
    originality_conf = [0.0] * n

    # --- Phase A1: generate advantages (batched) ---
    print("  Generating advantages...")
    for batch_start in tqdm(range(0, n, batch_size), desc="advantages"):
        batch_end = min(batch_start + batch_size, n)
        msgs_batch = [
            _advantages_messages(uses[i][0], uses[i][1])
            for i in range(batch_start, batch_end)
        ]
        replies = loader.generate_batch(
            msgs_batch, max_new_tokens=max_new_tokens, do_sample=False,
        )
        for j, reply in enumerate(replies):
            advantages[batch_start + j] = reply.strip()

    # --- Phase A2: generate drawbacks (batched) ---
    print("  Generating drawbacks...")
    for batch_start in tqdm(range(0, n, batch_size), desc="drawbacks"):
        batch_end = min(batch_start + batch_size, n)
        msgs_batch = [
            _drawbacks_messages(uses[i][0], uses[i][1])
            for i in range(batch_start, batch_end)
        ]
        replies = loader.generate_batch(
            msgs_batch, max_new_tokens=max_new_tokens, do_sample=False,
        )
        for j, reply in enumerate(replies):
            drawbacks[batch_start + j] = reply.strip()

    # --- Phase A3: score originality (batched) ---
    print("  Scoring originality...")
    for batch_start in tqdm(range(0, n, batch_size), desc="originality"):
        batch_end = min(batch_start + batch_size, n)
        msgs_batch = [
            _originality_messages(uses[i][0], uses[i][1])
            for i in range(batch_start, batch_end)
        ]
        scores = loader.score_next_token_batch(msgs_batch, YES_NO_TOKENS)
        for j, s in enumerate(scores):
            originality_conf[batch_start + j] = s["Yes"] - s["No"]

    # --- Phase B: score utility using advantages + drawbacks (batched) ---
    print("  Scoring utility (with chain-of-thought)...")
    for batch_start in tqdm(range(0, n, batch_size), desc="utility"):
        batch_end = min(batch_start + batch_size, n)
        msgs_batch = [
            _utility_messages_chain(
                uses[i][0], uses[i][1], advantages[i], drawbacks[i],
            )
            for i in range(batch_start, batch_end)
        ]
        scores = loader.score_next_token_batch(msgs_batch, YES_NO_TOKENS)
        for j, s in enumerate(scores):
            utility_conf[batch_start + j] = s["Yes"] - s["No"]

    # --- Assemble results ---
    records: list[dict] = []
    for i, (obj, use) in enumerate(uses):
        records.append({
            "object": obj,
            "use": use,
            "advantages": advantages[i],
            "drawbacks": drawbacks[i],
            "utility_conf": utility_conf[i],
            "originality_conf": originality_conf[i],
        })
    return records


def _run_skip_chain(
    loader: HFLoader,
    uses: list[tuple[str, str]],
    batch_size: int,
) -> list[dict]:
    """Score utility and originality directly (no advantages/drawbacks)."""
    n = len(uses)
    utility_conf = [0.0] * n
    originality_conf = [0.0] * n

    print("  Scoring utility (direct)...")
    for batch_start in tqdm(range(0, n, batch_size), desc="utility"):
        batch_end = min(batch_start + batch_size, n)
        msgs_batch = [
            _utility_messages_direct(uses[i][0], uses[i][1])
            for i in range(batch_start, batch_end)
        ]
        scores = loader.score_next_token_batch(msgs_batch, YES_NO_TOKENS)
        for j, s in enumerate(scores):
            utility_conf[batch_start + j] = s["Yes"] - s["No"]

    print("  Scoring originality...")
    for batch_start in tqdm(range(0, n, batch_size), desc="originality"):
        batch_end = min(batch_start + batch_size, n)
        msgs_batch = [
            _originality_messages(uses[i][0], uses[i][1])
            for i in range(batch_start, batch_end)
        ]
        scores = loader.score_next_token_batch(msgs_batch, YES_NO_TOKENS)
        for j, s in enumerate(scores):
            originality_conf[batch_start + j] = s["Yes"] - s["No"]

    records: list[dict] = []
    for i, (obj, use) in enumerate(uses):
        records.append({
            "object": obj,
            "use": use,
            "advantages": "",
            "drawbacks": "",
            "utility_conf": utility_conf[i],
            "originality_conf": originality_conf[i],
        })
    return records


# ── Filtering ────────────────────────────────────────────────────────────────

def _apply_filter(
    records: list[dict],
    filter_mode: str,
    threshold: float,
) -> list[dict]:
    """Keep records passing the confidence threshold."""
    kept: list[dict] = []
    for r in records:
        u_pass = r["utility_conf"] > threshold
        o_pass = r["originality_conf"] > threshold
        if filter_mode == "utility" and u_pass:
            kept.append(r)
        elif filter_mode == "originality" and o_pass:
            kept.append(r)
        elif filter_mode == "both" and u_pass and o_pass:
            kept.append(r)
    return kept


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Brainstorm-then-Select: LLM self-evaluation and filtering of AUT uses "
            "(Summers-Stay et al., 2022)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="uses.txt from run_aut_inference.py (lines of 'Object, use')",
    )
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output stem — writes {stem}_uses.txt and {stem}_scores.csv",
    )

    parser.add_argument(
        "--filter", choices=["utility", "originality", "both"], default="both",
        dest="filter_mode",
        help="Which confidence filter to apply (default: both)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.0,
        help="Minimum log P(Yes)-log P(No) to keep a use (default: 0.0, i.e. Yes > No)",
    )
    parser.add_argument(
        "--skip-chain", action="store_true",
        help="Skip advantages/drawbacks generation; score utility directly (faster)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Max tokens for advantages/drawbacks generation (default: 128)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1

    # Load uses
    uses = _load_uses(args.input)
    if not uses:
        print(f"Error: no valid 'Object, use' lines found in {args.input}", file=sys.stderr)
        return 1
    print(f"Loaded {len(uses)} uses from {args.input}")

    # Load model
    print(f"Loading model: {args.model_name}")
    loader = HFLoader(model_name_or_path=args.model_name)
    loader.load()

    # Evaluate
    if args.skip_chain:
        print("Running direct evaluation (no chain-of-thought)...")
        records = _run_skip_chain(loader, uses, args.batch_size)
    else:
        print("Running full chain evaluation (advantages → drawbacks → utility + originality)...")
        records = _run_full_chain(loader, uses, args.batch_size, args.max_new_tokens)

    # Print score distribution before filtering
    u_scores = [r["utility_conf"] for r in records]
    o_scores = [r["originality_conf"] for r in records]
    print(f"\nScore distribution (all {len(records)} uses):")
    print(f"  Utility conf:     mean={_mean(u_scores):.3f}  "
          f"median={_median(u_scores):.3f}  "
          f"min={min(u_scores):.3f}  max={max(u_scores):.3f}")
    print(f"  Originality conf: mean={_mean(o_scores):.3f}  "
          f"median={_median(o_scores):.3f}  "
          f"min={min(o_scores):.3f}  max={max(o_scores):.3f}")

    # Filter
    kept = _apply_filter(records, args.filter_mode, args.confidence_threshold)
    print(f"\nFilter={args.filter_mode}, threshold={args.confidence_threshold}")
    print(f"  Kept {len(kept)}/{len(records)} uses "
          f"({100 * len(kept) / len(records):.1f}%)")

    # Write outputs
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output

    # Filtered uses.txt
    txt_path = Path(f"{stem}_uses.txt")
    lines = [f"{r['object']}, {r['use']}" for r in kept]
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"  Filtered uses: {txt_path}  ({len(lines)} lines)")

    # Full scores CSV (all uses, not just kept)
    csv_path = Path(f"{stem}_scores.csv")
    fieldnames = [
        "object", "use", "utility_conf", "originality_conf",
        "kept", "advantages", "drawbacks",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        kept_set = {(r["object"], r["use"]) for r in kept}
        for r in records:
            writer.writerow({
                "object": r["object"],
                "use": r["use"],
                "utility_conf": f"{r['utility_conf']:.4f}",
                "originality_conf": f"{r['originality_conf']:.4f}",
                "kept": (r["object"], r["use"]) in kept_set,
                "advantages": r["advantages"],
                "drawbacks": r["drawbacks"],
            })
    print(f"  Scores CSV: {csv_path}  ({len(records)} rows)")

    # Print kept score stats
    if kept:
        ku = [r["utility_conf"] for r in kept]
        ko = [r["originality_conf"] for r in kept]
        print(f"\nKept uses score distribution:")
        print(f"  Utility conf:     mean={_mean(ku):.3f}  median={_median(ku):.3f}")
        print(f"  Originality conf: mean={_mean(ko):.3f}  median={_median(ko):.3f}")

    print("\nDone.")
    return 0


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


if __name__ == "__main__":
    sys.exit(main())
