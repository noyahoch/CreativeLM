#!/usr/bin/env python3
"""
MacGyver inference — run methods on the MacGyver creative problem-solving dataset.

Loads problem_solution_pair.xlsx via MacGyverDataset, applies the requested
subset filter and train/eval split, runs each requested method (baseline,
steered, etc.), and saves results in the CSV format that
score_macgyver_quality.py consumes.

Usage:
  cd DLP

  # Baseline only (default: solvable_unconventional subset, all eval)
  python scripts/run_macgyver_inference.py \\
    --macgyver-data external_repos/MacGyver/data/MacGyver/problem_solution_pair.xlsx \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --method baseline \\
    --output-dir results/macgyver_inference/llama_baseline

  # CrPO model
  python scripts/run_macgyver_inference.py \\
    --macgyver-data external_repos/MacGyver/data/MacGyver/problem_solution_pair.xlsx \\
    --model-name CNCL-Penn-State/CrPO-llama-3.1-8b-instruct-nov \\
    --method baseline \\
    --output-dir results/macgyver_inference/crpo_nov

  # Steered with multiple alphas
  python scripts/run_macgyver_inference.py \\
    --macgyver-data external_repos/MacGyver/data/MacGyver/problem_solution_pair.xlsx \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --method steered --vectors results/bridge_steering/.../steering_vectors.pt \\
    --alpha 0.5 --alpha 1.0 --alpha 1.5 \\
    --output-dir results/macgyver_inference/llama_steered

  # Sampling with multiple inferences per prompt
  python scripts/run_macgyver_inference.py \\
    --macgyver-data external_repos/MacGyver/data/MacGyver/problem_solution_pair.xlsx \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --method baseline --do-sample --temperature 0.7 \\
    --num-inferences 4 \\
    --output-dir results/macgyver_inference/llama_sampled

  # Benchmark subset only (323 items from the paper)
  python scripts/run_macgyver_inference.py \\
    --macgyver-data external_repos/MacGyver/data/MacGyver/problem_solution_pair.xlsx \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --subset benchmark \\
    --output-dir results/macgyver_inference/llama_benchmark

  # Novelty-instruct prompt (paper E.2.3 — asks for creative/novel output)
  python scripts/run_macgyver_inference.py \\
    --macgyver-data external_repos/MacGyver/data/MacGyver/problem_solution_pair.xlsx \\
    --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --method baseline --do-sample --temperature 0.7 \\
    --prompt-style novelty_instruct \\
    --output-dir results/macgyver_inference/llama_novelty_instruct

Output per method (in output-dir):
  {method}_results.csv   — columns: eval_idx, sample_idx, id, method,
                           user_prompt, reply, reply_len_words,
                           solvable, unconventional, ref_solution, ref_label
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from dlp.data.macgyver import MacGyverDataset, MacGyverSubset
from dlp.evaluation.methods import (
    AUTInferenceMethod,
    BaselineMethod,
    SteeredMethod,
    available_methods,
    build_method,
)
from dlp.models import HFLoader

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

MACGYVER_SYSTEM = (
    "You are a creative problem solver. Given a scenario and a set of "
    "available tools, propose a practical step-by-step solution using only "
    "the provided items."
)

MACGYVER_USER_SUFFIX = (
    "\n\nGive a feasible solution very concisely. "
    "Use Step 1, Step 2, etc., and mention the tools used in each step. "
    "Use as few steps as possible and keep the answer under 150 words."
)

# From the paper (E.2.3): prompt that asks for novelty-aware generation.
MACGYVER_INSTRUCT_PROMPT = """\
MacGyver are real-world problems deliberately designed \
to trigger innovative usage of objects and necessitate \
out-of-the-box thinking.
Here are some tips for answering MacGyver questions:
1. Understand the Problem Context Thoroughly
* Carefully read the problem description, including the \
tools and constraints provided.
* Identify the objective and key limitations, focusing on \
how they constrain traditional solutions.
2.Leverage Divergent Thinking:
* Enumerate potential unconventional uses for each tool \
provided, exploring creative possibilities beyond typical \
applications.
* Consider combining tools in innovative ways to enhance \
functionality or bypass constraints.
3. Apply Convergent Thinking:
* Refine the solution to ensure it directly addresses the \
problem with minimal steps.
* Validate that the approach adheres to physical, logical, \
and contextual constraints described in the task.
4. Avoid Physically or Contextually Infeasible Proposals:
* Cross-check the proposed actions against basic physical laws \
(e.g., leverage, strength, materials).
* Ensure that all tools suggested in the solution are \
explicitly available and aligned with stated constraints.
5. Demonstrate High-Quality Creativity:
* Propose solutions that are novel and insightful, avoiding \
over-reliance on generic or training-data-replicative \
patterns.
* Structure responses to emphasize clarity and logical \
progression, ensuring they can be easily understood by \
the user.
Here is the MacGyver prompt I want you to answer:
{prompt}
Instruction:
- First, think about how to answer in a way that demonstrates \
high quality and creativity while avoiding over-reliance \
on n-grams from pretraining data by using the tips provided \
above.
- Return your response, ensuring it is enclosed with asterisks."""

PROMPT_STYLES = ["default", "novelty_instruct"]


def _format_user_prompt(item: dict, prompt_style: str = "default") -> str:
    if prompt_style == "novelty_instruct":
        return MACGYVER_INSTRUCT_PROMPT.format(prompt=item["problem"])
    return item["problem"] + MACGYVER_USER_SUFFIX


def _build_prompt_msgs(item: dict, prompt_style: str = "default") -> list[dict]:
    if prompt_style == "novelty_instruct":
        return [{"role": "user", "content": _format_user_prompt(item, prompt_style)}]
    return [
        {"role": "system", "content": MACGYVER_SYSTEM},
        {"role": "user", "content": _format_user_prompt(item, prompt_style)},
    ]


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "eval_idx", "sample_idx", "id", "method", "user_prompt", "reply",
    "reply_len_words", "solvable", "unconventional", "ref_solution", "ref_label",
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
        "solvable": item.get("solvable", True),
        "unconventional": item.get("unconventional", True),
        "ref_solution": item.get("solution", ""),
        "ref_label": item.get("label", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="MacGyver inference — run methods on the creative problem-solving dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--macgyver-data", type=Path, required=True,
        help="Path to problem_solution_pair.xlsx",
    )
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", "-o", type=Path, required=True)

    # Dataset filtering
    subset_choices = [s.value for s in MacGyverSubset]
    parser.add_argument(
        "--subset", type=str, default="solvable_unconventional",
        choices=subset_choices,
        help=f"Dataset subset (default: solvable_unconventional). Choices: {', '.join(subset_choices)}",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.0,
        help="Fraction for train split (default: 0.0 = all eval, no train hold-out)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/eval split")
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Cap number of eval items (for quick testing)",
    )

    # Methods
    parser.add_argument(
        "--method", action="append", dest="methods",
        help=f"Method to run (repeatable). Choices: {', '.join(available_methods())}",
    )

    # Steered method args
    parser.add_argument(
        "--vectors", type=Path, default=None,
        help="steering_vectors.pt (for --method steered)",
    )
    parser.add_argument(
        "--alpha", action="append", dest="alphas", type=float, metavar="FLOAT",
        help="Steering strength; repeat for multiple. Default: 1.0",
    )
    parser.add_argument(
        "--layer", action="append", dest="layers", type=int, metavar="INT",
        help="Layer to steer at; repeat for multiple. Default: auto from checkpoint",
    )
    parser.add_argument(
        "--steer-mode", type=str, default="all_new_tokens",
        choices=["all_new_tokens", "first_k_assistant_tokens", "last_prompt_only", "all"],
        help="Steering hook mode (default: all_new_tokens)",
    )
    parser.add_argument(
        "--k-assist", type=int, default=16,
        help="Tokens to steer with first_k_assistant_tokens mode (default: 16)",
    )

    # Generation params
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="0 = greedy (default)",
    )
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling threshold")
    parser.add_argument("--min-p", type=float, default=None, help="Min-p sampling threshold")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument(
        "--num-inferences", type=int, default=1,
        help="N replies per prompt with sampling (default: 1). >1 requires --do-sample.",
    )

    # Prompt style
    parser.add_argument(
        "--prompt-style", type=str, default="default",
        choices=PROMPT_STYLES,
        help="Prompt template: 'default' (concise step-by-step) or "
             "'novelty_instruct' (paper E.2.3 — asks for creative/novel output). "
             "Default: default",
    )

    args = parser.parse_args()

    if not args.macgyver_data.exists():
        print(f"Error: {args.macgyver_data} not found", file=sys.stderr)
        return 1

    # ── Load dataset ──────────────────────────────────────────────────────
    ds = MacGyverDataset(
        path=args.macgyver_data,
        subset=args.subset,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    eval_items = ds.eval_items()
    if args.max_items is not None:
        eval_items = eval_items[: args.max_items]

    print(f"MacGyver dataset: {ds}")
    print(f"Eval items: {len(eval_items)}")

    # ── Build methods ─────────────────────────────────────────────────────
    method_names = args.methods or ["baseline"]
    alphas = args.alphas if args.alphas else [1.0]
    layers = args.layers if args.layers else [None]

    methods: list[AUTInferenceMethod] = []
    for name in method_names:
        if name == "steered":
            for layer in layers:
                for alpha in alphas:
                    if not args.vectors:
                        print("Error: --method steered requires --vectors", file=sys.stderr)
                        return 1
                    m = SteeredMethod(
                        vectors_path=args.vectors,
                        alpha=alpha,
                        layer_idx=layer,
                        steer_mode=args.steer_mode,
                        k_assist=args.k_assist,
                    )
                    parts = ["steered"]
                    if layer is not None:
                        parts.append(f"L{layer}")
                    parts.append(f"a{alpha}")
                    if args.steer_mode == "first_k_assistant_tokens":
                        parts.append(f"k{args.k_assist}")
                    m.slug = "_".join(parts)
                    methods.append(m)
        elif name == "baseline":
            methods.append(BaselineMethod())
        else:
            try:
                m = build_method(name)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
            methods.append(m)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_name}")
    loader = HFLoader(model_name_or_path=args.model_name)
    loader.load()

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

    gen_kwargs: dict[str, float] = {}
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.min_p is not None:
        gen_kwargs["min_p"] = args.min_p

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build work list ───────────────────────────────────────────────────
    prompt_style = args.prompt_style
    if prompt_style != "default":
        print(f"Prompt style: {prompt_style}")

    work: list[tuple[int, int, dict, list[dict], str]] = []
    for eval_idx, item in enumerate(eval_items):
        msgs = _build_prompt_msgs(item, prompt_style)
        prompt = _format_user_prompt(item, prompt_style)
        for s in range(num_inf):
            work.append((eval_idx, s, item, msgs, prompt))

    # ── Run each method ───────────────────────────────────────────────────
    bs = args.batch_size
    for method in methods:
        print(f"\n{'=' * 60}")
        desc = method.slug
        if num_inf > 1:
            desc += f"  (N={num_inf} inferences/prompt)"
        print(
            f"Running method: {desc}"
            f"  (batch={'yes, bs=' + str(bs) if method.supports_batch else 'no — sequential'})"
        )
        print(f"{'=' * 60}")

        rows: list[dict] = []

        if method.supports_batch and bs > 1:
            for start in tqdm(
                range(0, len(work), bs),
                desc=method.slug,
                total=(len(work) + bs - 1) // bs,
            ):
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
                    rows.append(
                        _make_csv_row(ev_idx, s_idx, item, method.slug, prompt, reply)
                    )
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
                rows.append(
                    _make_csv_row(ev_idx, s_idx, item, method.slug, prompt, reply)
                )

        # Save CSV
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        csv_path = output_dir / f"{method.slug}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}  ({len(df)} rows)")

    print(f"\nDone. All outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
