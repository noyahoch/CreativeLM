#!/usr/bin/env python3
"""
AUT inference only — run any method on the ABCD held-out eval set.

Loads abcd_aut.json, applies the same eval/train split as bridge_steering.ipynb
(seed=7, N=40), builds identical prompt_msgs, runs each requested method via
methods.py, and saves results in the CSV format that vllm_judge.ipynb consumes
plus a flat "Object, use" text file.

No judging happens here — that stays in vllm_judge.ipynb or a separate script.

Usage:
  cd DLP

  # Baseline only
  python scripts/run_aut_inference.py \\
    --abcd-data dataset/abcd_aut.json \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --method baseline \\
    --output-dir results/aut_inference/llama_baseline

  # Baseline + steered (one alpha)
  python scripts/run_aut_inference.py \\
    --abcd-data dataset/abcd_aut.json \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --method baseline \\
    --method steered --vectors results/bridge_steering/.../steering_vectors.pt \\
    --output-dir results/aut_inference/llama_compare

  # Steered with multiple alphas (writes steered_a0.5_*, steered_a1.0_*, ...)
  python scripts/run_aut_inference.py \\
    --abcd-data dataset/abcd_aut.json \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --method steered --vectors results/bridge_steering/.../steering_vectors.pt \\
    --alpha 0.5 --alpha 1.0 --alpha 1.5 \\
    --output-dir results/aut_inference/llama_alpha_sweep

  # Sweep layers × alphas (writes steered_L12_a1.0_*, steered_L16_a1.0_*, ...)
  python scripts/run_aut_inference.py \\
    --abcd-data dataset/abcd_aut.json \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --method steered --vectors results/bridge_steering/.../steering_vectors.pt \\
    --layer 12 --layer 16 --layer 20 \\
    --alpha 0.5 --alpha 1.0 \\
    --output-dir results/aut_inference/llama_layer_sweep

  # All five methods
  python scripts/run_aut_inference.py \\
    --abcd-data dataset/abcd_aut.json \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --vectors results/bridge_steering/.../steering_vectors.pt \\
    --method baseline --method steered --method fewshot --method twohop --method abcd_framework \\
    --output-dir results/aut_inference/all_methods

  # Generate-and-select: 4 sampled replies per prompt (grade uses.txt, then select_best_uses.py)
  python scripts/run_aut_inference.py \\
    --abcd-data dataset/abcd_aut.json \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --method baseline --do-sample --temperature 0.7 \\
    --num-inferences 4 \\
    --output-dir results/aut_inference/baseline_sampled

Output per method (in output-dir):
  {method}_results.csv   — same schema as bridge_steering.ipynb CSVs (+ sample_idx column)
  {method}_uses.txt      — flat "Object, use" lines
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from dlp.evaluation.methods import build_method, available_methods, AUTInferenceMethod
from dlp.models import HFLoader


# ── Data loading & prompt building (matches bridge_steering.ipynb exactly) ────

EVAL_SEED = 7
N_EVAL_HOLDOUT = 40

AUT_USER_SUFFIX = (
    '\nReturn exactly 8 unconventional uses.'
    '\nFormat: one use per line, starting with "-".'
)


def _load_abcd_split(
    abcd_path: Path,
    eval_seed: int = EVAL_SEED,
    n_eval: int = N_EVAL_HOLDOUT,
) -> tuple[list[dict], list[dict], set[int]]:
    """Load abcd_aut.json and split into (eval_items, train_items, eval_idxs)."""
    with open(abcd_path) as f:
        data = json.load(f)
    n_eval = min(n_eval, len(data))
    eval_idxs = set(random.Random(eval_seed).sample(range(len(data)), n_eval))
    eval_items = [data[i] for i in sorted(eval_idxs)]
    train_items = [data[i] for i in range(len(data)) if i not in eval_idxs]
    return eval_items, train_items, eval_idxs


def _format_user_prompt(item: dict) -> str:
    """Exact same prompt as bridge_steering.ipynb _format_user_prompt_aut."""
    return item["A"] + AUT_USER_SUFFIX


def _build_prompt_msgs(item: dict) -> list[dict]:
    """Build [system, user] messages — empty system, same as notebook."""
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": _format_user_prompt(item)},
    ]


def _parse_object_from_A(a_text: str) -> str:
    """Extract object name from A field (e.g. 'rubber band' from 'Give 8 uses for a rubber band in ...')."""
    m = re.search(
        r"uses for (?:a |an )?(?:piece of )?([\w\s\-]+?)(?:\s*\([^)]*\))?(?:\s+(?:in|focusing|that|as)\b|\.|$)",
        a_text, re.I,
    )
    return m.group(1).strip() if m else a_text


# ── CSV output (matches vllm_judge.ipynb schema) ─────────────────────────────

CSV_COLUMNS = [
    "eval_idx", "sample_idx", "id", "method", "user_prompt", "reply",
    "reply_len_words", "has_mechanism_line",
    "ground_truth_B", "ground_truth_C", "problem_text",
]


def _make_csv_row(
    eval_idx: int,
    sample_idx: int,
    item: dict,
    method_slug: str,
    user_prompt: str,
    reply: str,
) -> dict:
    return {
        "eval_idx": eval_idx,
        "sample_idx": sample_idx,
        "id": item["id"],
        "method": method_slug,
        "user_prompt": user_prompt,
        "reply": reply,
        "reply_len_words": len(str(reply).split()),
        "has_mechanism_line": bool(re.search(r"(?i)mechanism\s*:", str(reply))),
        "ground_truth_B": item.get("B", {}).get("rule", ""),
        "ground_truth_C": item.get("C", ""),
        "problem_text": item["A"],
    }


# ── Flat output ───────────────────────────────────────────────────────────────

_PREAMBLE_RE = re.compile(
    r"(?i)^(here are|the following|below are|sure|of course|certainly|"
    r"i('d| would| can| will)|let me|give \d+|(\d+ )?(unconventional|unusual|creative|alternative) uses|"
    r"uses for)",
)

MIN_USE_WORDS = 4


def _parse_uses_from_reply(reply: str) -> list[str]:
    """Extract individual uses from model reply (bullet or numbered lines).

    Filters out preamble / header lines the model sometimes echoes and
    very short fragments that aren't real uses.
    """
    uses = []
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


def _strip_inner_commas(text: str) -> str:
    """Remove commas from a use so the only comma in the line is 'Object, use'."""
    text = re.sub(r",\s+(and|or|but|so|yet|nor)\b", r" \1", text)
    text = text.replace(", ", " ").replace(",", "")
    return text


def _save_flat(rows: list[dict], path: Path) -> None:
    """Write flat 'Object, use' lines — exactly one comma per line (after object)."""
    lines = []
    for r in rows:
        obj = _parse_object_from_A(r["problem_text"])
        for use in _parse_uses_from_reply(r["reply"]):
            lines.append(f"{obj}, {_strip_inner_commas(use)}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="AUT inference only — run methods on ABCD held-out, save CSV + flat output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--abcd-data", type=Path, required=True,
                        help="Path to abcd_aut.json")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", "-o", type=Path, required=True)

    parser.add_argument(
        "--method", action="append", dest="methods",
        help=f"Method to run (repeatable). Choices: {', '.join(available_methods())}",
    )

    # Method-specific args
    parser.add_argument("--vectors", type=Path, default=None,
                        help="steering_vectors.pt (for --method steered)")
    parser.add_argument("--alpha", action="append", dest="alphas", type=float,
                        metavar="FLOAT",
                        help="Steering strength; repeat for multiple (e.g. --alpha 0.5 --alpha 1.0). Default: 1.0")
    parser.add_argument("--layer", action="append", dest="layers", type=int,
                        metavar="INT",
                        help="Layer to steer at; repeat for multiple (e.g. --layer 12 --layer 16). "
                             "Default: auto-selected best layer from checkpoint")
    parser.add_argument("--n-shots", type=int, default=2,
                        help="Few-shot examples for fewshot/twohop (default: 2)")
    parser.add_argument("--steer-mode", type=str, default="all_new_tokens",
                        choices=["all_new_tokens", "first_k_assistant_tokens", "last_prompt_only", "all"],
                        help="Steering hook mode (default: all_new_tokens). "
                             "first_k_assistant_tokens: steer only the first K tokens (set K with --k-assist)")
    parser.add_argument("--k-assist", type=int, default=16,
                        help="Number of initial assistant tokens to steer when --steer-mode first_k_assistant_tokens (default: 16)")

    # Generation params
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy (default, matches notebook)")
    parser.add_argument("--do-sample", action="store_true",
                        help="Enable sampling (default: greedy like notebook)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling threshold (default: HF default = 1.0)")
    parser.add_argument("--min-p", type=float, default=None,
                        help="Min-p sampling threshold (creative decoding baseline). "
                             "E.g. --min-p 0.05 --temperature 1.5")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for parallel generation (default: 8)")
    parser.add_argument("--num-inferences", type=int, default=1,
                        help="Generate N replies per prompt with sampling (default: 1). "
                             "Values > 1 require --do-sample.")

    # Eval split
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED,
                        help=f"Eval split seed (default: {EVAL_SEED}, same as notebook)")
    parser.add_argument("--n-eval", type=int, default=N_EVAL_HOLDOUT,
                        help=f"Number of held-out eval items (default: {N_EVAL_HOLDOUT})")

    args = parser.parse_args()

    if not args.abcd_data.exists():
        print(f"Error: {args.abcd_data} not found", file=sys.stderr)
        return 1

    method_names = args.methods or ["baseline"]
    alphas = args.alphas if args.alphas and len(args.alphas) > 0 else [1.0]
    layers = args.layers if args.layers and len(args.layers) > 0 else [None]

    # Load data and split
    eval_items, train_items, eval_idxs = _load_abcd_split(
        args.abcd_data, eval_seed=args.eval_seed, n_eval=args.n_eval,
    )
    train_ids = {it["id"] for it in train_items}
    print(f"Loaded {len(eval_items) + len(train_items)} items: "
          f"{len(train_items)} train, {len(eval_items)} eval (seed={args.eval_seed})")

    # Build methods (one steered variant per alpha × layer combination)
    methods: list[AUTInferenceMethod] = []
    for name in method_names:
        if name == "steered":
            for layer in layers:
                for alpha in alphas:
                    try:
                        m = build_method(
                            name,
                            vectors_path=args.vectors,
                            alpha=alpha,
                            layer_idx=layer,
                            abcd_path=args.abcd_data,
                            n_shots=args.n_shots,
                            train_ids=train_ids,
                            steer_mode=args.steer_mode,
                            k_assist=args.k_assist,
                        )
                    except ValueError as e:
                        print(f"Error: {e}", file=sys.stderr)
                        return 1
                    parts = ["steered"]
                    if layer is not None:
                        parts.append(f"L{layer}")
                    parts.append(f"a{alpha}")
                    if args.steer_mode == "first_k_assistant_tokens":
                        parts.append(f"k{args.k_assist}")
                    m.slug = "_".join(parts)
                    methods.append(m)
        else:
            try:
                m = build_method(
                    name,
                    vectors_path=args.vectors,
                    alpha=alphas[0],
                    abcd_path=args.abcd_data,
                    n_shots=args.n_shots,
                    train_ids=train_ids,
                )
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
            methods.append(m)

    # Load model
    print(f"Loading model: {args.model_name}")
    loader = HFLoader(model_name_or_path=args.model_name)
    loader.load()

    # Validate steering vector size if steered method is used
    if args.vectors:
        import torch
        state = torch.load(args.vectors, map_location="cpu", weights_only=False)
        v = state["v_steer"]
        if loader.model.config.hidden_size != v.shape[0]:
            print(
                f"Error: steering vector dim {v.shape[0]} != model hidden size "
                f"{loader.model.config.hidden_size}",
                file=sys.stderr,
            )
            return 1

    num_inf = args.num_inferences
    if num_inf > 1 and not args.do_sample:
        print("Error: --num-inferences > 1 requires --do-sample", file=sys.stderr)
        return 1

    # Extra sampling kwargs forwarded to model.generate()
    gen_kwargs: dict[str, float] = {}
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.min_p is not None:
        gen_kwargs["min_p"] = args.min_p

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build flattened work list: (eval_idx, sample_idx, item, prompt_msgs, user_prompt)
    work: list[tuple[int, int, dict, list[dict], str]] = []
    for eval_idx, item in enumerate(eval_items):
        msgs = _build_prompt_msgs(item)
        prompt = _format_user_prompt(item)
        for s in range(num_inf):
            work.append((eval_idx, s, item, msgs, prompt))

    # Run each method
    bs = args.batch_size
    for method in methods:
        print(f"\n{'=' * 60}")
        desc = method.slug
        if num_inf > 1:
            desc += f"  (N={num_inf} inferences/prompt)"
        print(f"Running method: {desc}"
              f"  (batch={'yes, bs=' + str(bs) if method.supports_batch else 'no — sequential'})")
        print(f"{'=' * 60}")

        rows: list[dict] = []

        if method.supports_batch and bs > 1:
            for start in tqdm(range(0, len(work), bs),
                              desc=method.slug,
                              total=(len(work) + bs - 1) // bs):
                end = min(start + bs, len(work))
                batch_msgs = [w[3] for w in work[start:end]]
                replies = method.generate_batch(
                    loader,
                    batch_msgs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                    **gen_kwargs,
                )
                for i, reply in enumerate(replies):
                    ev_idx, s_idx, item, _, prompt = work[start + i]
                    rows.append(_make_csv_row(
                        ev_idx, s_idx, item, method.slug, prompt, reply,
                    ))
        else:
            for ev_idx, s_idx, item, msgs, prompt in tqdm(work, desc=method.slug):
                reply = method.generate(
                    loader,
                    msgs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                    **gen_kwargs,
                )
                rows.append(_make_csv_row(ev_idx, s_idx, item, method.slug, prompt, reply))

        # Save CSV
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        csv_path = output_dir / f"{method.slug}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}  ({len(df)} rows)")

        # Save flat
        txt_path = output_dir / f"{method.slug}_uses.txt"
        _save_flat(rows, txt_path)
        n_uses = sum(1 for line in txt_path.read_text().splitlines() if line.strip())
        print(f"  Flat: {txt_path}  ({n_uses} uses)")

    print(f"\nDone. All outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
