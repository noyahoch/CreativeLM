#!/usr/bin/env python3
"""
Runner script: run the Qwen model on a single prompt or on the ABCD dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from creativelm import QwenModelLoader


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Qwen model: single prompt or dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt
  python run.py --prompt "List 5 unusual uses for a paperclip."

  # Dataset (all items)
  python run.py --dataset

  # Dataset, first 3 items, save outputs
  python run.py --dataset --limit 3 --output results.json

  # Use field B as system prompt for dataset items
  python run.py --dataset --use-b --output results.json
""",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "-p",
        "--prompt",
        metavar="TEXT",
        help="Run on this single prompt and print the reply.",
    )
    mode.add_argument(
        "-d",
        "--dataset",
        nargs="?",
        const="dataset/abcd_tuples.json",
        metavar="PATH",
        help="Run on dataset JSON (default: dataset/abcd_tuples.json).",
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Write results to this file (JSON).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="Qwen/Qwen2-7B-Instruct",
        help="Model name or path (default: Qwen/Qwen2-7B-Instruct).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens per reply (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding (no sampling).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Only run on first N dataset items (dataset mode only).",
    )
    parser.add_argument(
        "--use-b",
        action="store_true",
        help="Use field B as system prompt for each dataset item (dataset mode only).",
    )
    parser.add_argument(
        "--no-fast-download",
        action="store_true",
        help="Disable hf-transfer (use if you have proxy or download issues).",
    )
    args = parser.parse_args()

    loader = QwenModelLoader(
        args.model,
        use_fast_download=not args.no_fast_download,
    )

    if args.prompt is not None:
        # Single prompt mode
        reply = loader.prompt(
            args.prompt,
            max_new_tokens=args.max_tokens,
            do_sample=not args.no_sample,
            temperature=args.temperature,
        )
        print(reply)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(
                    {"prompt": args.prompt, "output": reply},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        return 0

    # Dataset mode
    path = Path(args.dataset)
    if not path.is_file():
        print(f"Error: dataset file not found: {path}", file=sys.stderr)
        return 1
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        items = [items]
    if args.limit is not None:
        items = items[: args.limit]

    results = []
    for i, item in enumerate(items):
        item_id = item.get("id", i)
        prompt_a = item.get("A", "")
        if not prompt_a:
            print(f"Warning: item {item_id} has no 'A' field, skipping.", file=sys.stderr)
            continue
        system = item.get("B") if args.use_b else None
        print(f"[{i+1}/{len(items)}] {item_id} ...", flush=True)
        output = loader.prompt(
            prompt_a,
            system_prompt=system,
            max_new_tokens=args.max_tokens,
            do_sample=not args.no_sample,
            temperature=args.temperature,
        )
        results.append(
            {
                "id": item_id,
                "task": item.get("task"),
                "A": prompt_a,
                "B": item.get("B"),
                "model_output": output,
                "C": item.get("C"),
                "D": item.get("D"),
            }
        )
        print(output[:200] + ("..." if len(output) > 200 else ""), flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(results)} results to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
