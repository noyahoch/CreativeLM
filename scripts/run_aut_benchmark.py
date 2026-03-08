#!/usr/bin/env python3
"""
CreativityPrism AUT benchmark runner with pluggable methods.

Available methods (--method, repeatable):
  baseline        Plain generation (no steering).
  steered         Activation-steered generation (requires --vectors).
  fewshot         Prepend N mechanism examples from abcd_aut.json (requires --abcd-data).
  twohop          Hop 1: generate mechanism; hop 2: generate uses from it.
                  Optionally pass --abcd-data for few-shot hop-1 examples.
  abcd_framework  ABCD-framed system prompt (no extra data needed).

Output:
  Each method writes inference_output.json (CreativityPrism format) under
  output_dir/aut_<method>_<model_slug>/.
  With --output-format flat or both, also writes aut_outputs.txt with
  "Object, use" lines (one per line).

Usage examples:
  # Baseline only
  python scripts/run_aut_benchmark.py \\
    --aut-data dataset/aut_push_skipped.json \\
    --output-dir results/aut_bench --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --method baseline

  # All methods, flat output
  python scripts/run_aut_benchmark.py \\
    --aut-data dataset/aut_push_skipped.json \\
    --abcd-data dataset/abcd_aut.json \\
    --vectors results/bridge_steering/.../steering_vectors.pt \\
    --output-dir results/aut_bench --model-name meta-llama/Llama-3.1-8B-Instruct \\
    --method baseline --method steered --method fewshot --method twohop --method abcd_framework \\
    --output-format both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dlp.evaluation import AUTBenchmarkRunner
from dlp.evaluation.generate import save_flat
from dlp.evaluation.methods import build_method, available_methods
from dlp.utils.config import BenchmarkConfig


def _maybe_save_flat(out_dir: Path, fmt: str) -> None:
    """Write aut_outputs.txt (flat 'Object, use' format) if requested."""
    if fmt not in ("flat", "both"):
        return
    inf = out_dir / "inference_output.json"
    if not inf.exists():
        return
    raw = json.loads(inf.read_text())
    # inference_output.json: { dp_id: { iter_id: "- use\n- use2..." } }
    items = []
    for dp_id, iters in raw.items():
        uses_text = next(iter(iters.values()), "") if iters else ""
        uses = [
            line.lstrip("- ").strip()
            for line in uses_text.split("\n")
            if line.strip()
        ]
        items.append({"id": dp_id, "object": dp_id, "uses": uses})
    txt_path = out_dir / "aut_outputs.txt"
    save_flat(items, txt_path)
    print(f"  Flat output: {txt_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AUT benchmark: multiple inference methods (baseline, steered, fewshot, twohop, abcd_framework)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--aut-data", type=Path, required=True,
                        help="Path to aut_push_skipped.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Base output dir; writes aut_<method>_<model>/ per method")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--method", action="append", dest="method",
        help=f"Method to run (repeatable). Choices: {', '.join(available_methods())}. "
             "Omit to run baseline (+ steered if --vectors given).",
    )

    # Method-specific data
    parser.add_argument("--vectors", type=Path, default=None,
                        help="steering_vectors.pt (required for --method steered)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Steering strength for steered method (default: 1.0)")
    parser.add_argument("--abcd-data", type=Path, default=None,
                        help="abcd_aut.json for few-shot / twohop methods")
    parser.add_argument("--n-shots", type=int, default=2,
                        help="Few-shot examples for fewshot/twohop (default: 2)")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--test-size", type=float, default=1e10,
                        help="Max items per iteration (default: all)")

    # Output format
    parser.add_argument(
        "--output-format", choices=["json", "flat", "both"], default="json",
        help=(
            "json = inference_output.json only (default); "
            "flat = also write aut_outputs.txt with 'Object, use' lines; "
            "both = write both."
        ),
    )

    # Legacy flags (kept for backward compat)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-steered", action="store_true")

    args = parser.parse_args()

    if not args.aut_data.exists():
        print(f"Error: aut data not found: {args.aut_data}", file=sys.stderr)
        return 1
    if args.vectors is not None and not args.vectors.exists():
        print(f"Error: vectors not found: {args.vectors}", file=sys.stderr)
        return 1

    # Build method list
    method_names = args.method or []
    if not method_names:
        # Default: baseline + steered (if vectors given)
        method_names = []
        if not args.no_baseline:
            method_names.append("baseline")
        if not args.no_steered and args.vectors:
            method_names.append("steered")

    methods = []
    for name in method_names:
        try:
            m = build_method(
                name,
                vectors_path=args.vectors,
                alpha=args.alpha,
                abcd_path=args.abcd_data,
                n_shots=args.n_shots,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        methods.append(m)

    if not methods:
        print("No methods to run. Pass --method or provide --vectors for steered.", file=sys.stderr)
        return 1

    config = BenchmarkConfig(
        aut_data_path=args.aut_data,
        vectors_path=args.vectors,
        output_dir=args.output_dir,
        model_name=args.model_name,
        alpha=args.alpha,
        test_size=args.test_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    runner = AUTBenchmarkRunner(config)

    written = runner.run(methods=methods)
    for out_dir in written:
        _maybe_save_flat(out_dir, args.output_format)
        print(f"Wrote {out_dir}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
