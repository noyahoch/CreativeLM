#!/usr/bin/env python3
"""
CLI for AUT pipeline: generate outputs (OpenAI) and/or run LLM-as-a-judge.

For local HF model inference (Llama, Qwen, etc.), use run_aut_inference.py instead.

Examples:
  # Generate then judge (full pipeline)
  python scripts/run_judge.py --mode full --objects brick paperclip --output-dir results/judge_outputs/run1

  # Generate only
  python scripts/run_judge.py --mode generate --objects brick paperclip --output-dir results/judge_outputs/run1

  # Judge only (existing AUT outputs)
  python scripts/run_judge.py --mode judge --input path/to/aut_outputs.json --output-dir results/judge_outputs/run1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dlp.evaluation.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="AUT pipeline: generate outputs and/or LLM-as-a-judge")
    parser.add_argument("--mode", choices=["full", "generate", "judge"], default="full",
                        help="full = generate then judge; generate = only outputs; judge = only judge (requires --input)")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Directory for all outputs (aut_outputs.json, judge_output.json, judge_report.json)")
    parser.add_argument("--objects", nargs="*",
                        help="Object names for generation (e.g. brick paperclip)")
    parser.add_argument("--objects-file",
                        help="Path to JSON array of object names for generation")
    parser.add_argument("--input", "-i",
                        help="Path to existing AUT outputs JSON (required for --mode judge)")
    parser.add_argument("--num-uses", type=int, default=10,
                        help="Number of uses to generate per object (default: 10)")
    parser.add_argument("--generate-model", default="gpt-4o-mini",
                        help="Model for generation (default: gpt-4o-mini)")
    parser.add_argument("--generate-temperature", type=float, default=0.7,
                        help="Temperature for generation (default: 0.7)")
    parser.add_argument("--judge-model", "-m", default="gpt-4o-mini",
                        help="Model for judge (default: gpt-4o-mini)")
    parser.add_argument("--judge-temperature", "-t", type=float, default=0.0,
                        help="Judge temperature (default: 0)")
    parser.add_argument("--max-uses", type=int, default=15,
                        help="Max uses per item to rate (default: 15)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    if args.mode in ("full", "generate"):
        if args.objects_file:
            objects = args.objects_file
        elif args.objects:
            objects = args.objects
        else:
            print("For --mode %s pass --objects or --objects-file" % args.mode, file=sys.stderr)
            return 1
    else:
        if not args.input:
            print("For --mode judge pass --input path/to/aut_outputs.json", file=sys.stderr)
            return 1
        objects = None

    try:
        result = run_pipeline(
            output_dir=args.output_dir,
            objects=objects,
            input_path=args.input or None,
            mode=args.mode,
            num_uses=args.num_uses,
            generate_model=args.generate_model,
            generate_temperature=args.generate_temperature,
            judge_model=args.judge_model,
            judge_temperature=args.judge_temperature,
            max_uses_per_item=args.max_uses,
        )
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    print("Wrote results to %s" % args.output_dir)
    if args.mode != "generate" and "report" in result:
        r = result["report"]
        print("Report: n_items=%s mean_creativity=%s mean_fluency=%s" % (
            r.get("n_items"), r.get("mean_creativity"), r.get("mean_fluency"),
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
