#!/usr/bin/env python3
"""
Poll vLLM /metrics and print completed request count (and optional progress vs total).

Useful when running score_macgyver_quality.py: the scoring script's progress bar
may not update in some terminals, but vLLM's request_success counter does.
Run this in a separate terminal while the judge is running.

Example:
  # Just show total completed requests (updates every 5s)
  python scripts/vllm_request_progress.py

  # Show progress toward 3824 (e.g. 4 methods × 956 items)
  python scripts/vllm_request_progress.py --total 3824

  # Custom URL and interval
  python scripts/vllm_request_progress.py --metrics-url http://localhost:8000/metrics --interval 2 --total 3824
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request


_warned_no_metric: list[bool] = []  # mutable so we only warn once


def get_request_success(metrics_url: str) -> int | None:
    try:
        req = urllib.request.Request(metrics_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode()
    except Exception as e:
        print(f"Error fetching metrics: {e}", file=sys.stderr, flush=True)
        return None
    # Prometheus: vllm:request_success 1234.0 or vllm:request_success{model_name="..."} 1234.0
    m = re.search(r"vllm:request_success(?:\{[^}]*\})?\s+([\d.]+)", text)
    if m is None:
        if not _warned_no_metric:
            _warned_no_metric.append(True)
            print(
                "Warning: no vllm:request_success in response (wrong vLLM version or metrics disabled?)",
                file=sys.stderr,
                flush=True,
            )
        return None
    return int(float(m.group(1)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Poll vLLM /metrics and print completed request count (for judge progress)."
    )
    parser.add_argument(
        "--metrics-url",
        default="http://localhost:8000/metrics",
        help="vLLM metrics endpoint (default: http://localhost:8000/metrics)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between polls (default: 5)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        metavar="N",
        help="If set, show progress as N_completed/N (percent)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print once and exit (no polling loop)",
    )
    args = parser.parse_args()

    if args.once:
        n = get_request_success(args.metrics_url)
        if n is None:
            return 1
        if args.total is not None:
            pct = 100 * n / args.total if args.total else 0
            print(f"{n}/{args.total} ({pct:.1f}%)")
        else:
            print(n)
        return 0

    print(
        f"Polling {args.metrics_url} every {args.interval}s (Ctrl+C to stop)...",
        file=sys.stderr,
        flush=True,
    )
    try:
        while True:
            n = get_request_success(args.metrics_url)
            if n is not None:
                if args.total is not None:
                    pct = 100 * n / args.total if args.total else 0
                    print(
                        f"\r  Completed requests: {n}/{args.total} ({pct:.1f}%)  ",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  Completed requests: {n}  ", end="", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
