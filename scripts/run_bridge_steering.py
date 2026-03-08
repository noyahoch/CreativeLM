#!/usr/bin/env python3
"""
Run bridge steering: collect activations from B vs D contrastive pairs,
compute v_bridge via a pluggable extraction method, pick layer, save
steering_vectors.pt.

Two composable strategy axes control the pipeline:

  --b-source  : how B completions are produced (fixed | generated)
  --method    : how activations become steering vectors (mean_diff | clustered | multi_pca)

Any source x any method is a valid configuration.  The default
(--b-source fixed --method mean_diff) reproduces the original pipeline.

Usage:
  cd DLP && python scripts/run_bridge_steering.py \\
    --model-name "meta-llama/Llama-3.1-8B-Instruct" \\
    --abcd-data dataset/abcd_aut.json \\
    --output-dir results/bridge_steering \\
    [--b-source fixed] [--method mean_diff] [--use-pca] [--window 48]

  Output is <output-dir>/<setup_slug>/steering_vectors.pt.
  Same config skips; different config uses a new subdir (no override).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# DLP package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dlp.data import ABCDDataset
from dlp.models import HFLoader
from dlp.steering import (
    ResidualStreamCache,
    select_best_layer,
)
from dlp.steering.completion_sources import SOURCE_REGISTRY
from dlp.steering.extractors import EXTRACTOR_REGISTRY
from dlp.training.data_prep import TaskType, build_contrastive_pair


WINDOW_DEFAULT = 48
EVAL_HOLDOUT_DEFAULT = 40
EVAL_SEED_DEFAULT = 7
STEERING_VECTORS_FILENAME = "steering_vectors.pt"


def _model_slug(model_name: str) -> str:
    """e.g. meta-llama/Llama-3.1-8B-Instruct -> llama_31_8b_instruct"""
    s = model_name.split("/")[-1].lower()
    s = re.sub(r"[^a-z0-9.]+", "_", s).strip("_")
    return s or "model"


def _setup_slug(
    model_name: str,
    task_type: str,
    abcd_data: Path,
    train_frac: float,
    seed: int,
    eval_seed: int,
    eval_holdout: int,
    window: int,
    method: str,
    b_source: str,
    d_source: str,
    extra_config: dict,
) -> str:
    """Unique slug for this config.  Includes method + b_source so different
    strategies always get their own output directory."""
    model_part = _model_slug(model_name)
    data_stem = abcd_data.stem.lower()
    config_str = (
        f"{train_frac}_{seed}_{eval_seed}_{eval_holdout}"
        f"_{method}_{b_source}_{d_source}"
    )
    for k in sorted(extra_config):
        config_str += f"_{k}={extra_config[k]}"
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    parts = [
        model_part,
        task_type,
        data_stem,
        f"w{window}",
        method,
        b_source,
    ]
    if d_source != "D":
        parts.append(d_source.lower())
    parts.append(config_hash)
    return "_".join(parts)


def tokenize_full_sequence(prompt_msgs, completion, tokenizer):
    """Tokenize prompt + teacher-forced completion.  Returns dict with
    full_ids, assistant_start, n_completion_tokens."""
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    assistant_start = int(prompt_ids.shape[1])

    full_msgs = prompt_msgs + [{"role": "assistant", "content": completion}]
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]

    return {
        "full_ids": full_ids,
        "assistant_start": assistant_start,
        "n_completion_tokens": full_ids.shape[1] - assistant_start,
    }


def _collect_single_completion(
    prompt_msgs, completion, tokenizer, model, device, cache,
    probe_layers, effective_W,
):
    """Run one teacher-forced forward pass and return per-layer mean
    activation over the first *effective_W* completion tokens, or the whole
    completion if effective_W <= 0."""
    tok = tokenize_full_sequence(prompt_msgs, completion, tokenizer)
    inp = tok["full_ids"].to(device)
    n_comp = tok["n_completion_tokens"]
    w = n_comp if effective_W <= 0 else min(effective_W, n_comp)

    model(inp, use_cache=False)
    start = tok["assistant_start"]
    positions = list(range(start, start + w))
    vecs = {}
    for L in probe_layers:
        vecs[L] = cache.get_positions(L, positions).mean(dim=0).cpu()
    cache.clear()
    return vecs


def _min_B_completion_tokens(train_pairs, completion_source, tokenizer, model):
    """Min completion length over all B completions (for B-only window when not tied to D)."""
    min_comp = float("inf")
    for p in train_pairs:
        for b_text in completion_source.get_b_completions(p, model, tokenizer):
            tok = tokenize_full_sequence(p["prompt_msgs"], b_text, tokenizer)
            min_comp = min(min_comp, tok["n_completion_tokens"])
    return int(min_comp) if min_comp != float("inf") else 0


def collect_activations(
    train_pairs,
    completion_source,
    model,
    tokenizer,
    device,
    probe_layers,
    effective_W_D,
    effective_W_B,
):
    """Collect B and D activations for all training pairs.

    Uses effective_W_D for D and effective_W_B for B so B can use more tokens
    when B completions are long (e.g. multi_b_concat) and D is not needed to cap B.
    """
    acts_D = {L: [] for L in probe_layers}
    acts_B = {L: [] for L in probe_layers}
    cache = ResidualStreamCache(model, probe_layers)
    cache.register()

    with torch.no_grad():
        for p in tqdm(train_pairs, desc="Caching activations"):
            # D condition (always single teacher-forced completion)
            vecs_D = _collect_single_completion(
                p["prompt_msgs"], p["d_completion"], tokenizer,
                model, device, cache, probe_layers, effective_W_D,
            )
            for L in probe_layers:
                acts_D[L].append(vecs_D[L])

            # B condition: source decides how many completions per pair.
            b_completions = completion_source.get_b_completions(
                p, model, tokenizer
            )
            for b_text in b_completions:
                vecs_B = _collect_single_completion(
                    p["prompt_msgs"], b_text, tokenizer,
                    model, device, cache, probe_layers, effective_W_B,
                )
                for L in probe_layers:
                    acts_B[L].append(vecs_B[L])

    cache.remove()
    n_D = len(acts_D[probe_layers[0]])
    n_B = len(acts_B[probe_layers[0]])
    print(f"Collected activations: {n_D} D vectors, {n_B} B vectors per layer.")
    return acts_B, acts_D


def _build_parser() -> argparse.ArgumentParser:
    """Build the two-pass argument parser.

    Pass 1 (parse_known_args): resolve --method and --b-source so we can
    call add_args on the chosen classes.
    Pass 2 (parse_args): pick up the method/source-specific flags.
    """
    parser = argparse.ArgumentParser(
        description="Compute bridge steering vectors (B vs D) and save steering_vectors.pt.",
    )

    # Strategy axes (choices come from registries; listed here for readability)
    # --method: mean_diff, b_only, clustered, multi_pca
    # --b-source: fixed, generated, multi_b_separate, multi_b_concat
    method_choices = sorted(EXTRACTOR_REGISTRY.keys())
    source_choices = sorted(SOURCE_REGISTRY.keys())
    parser.add_argument(
        "--method",
        type=str,
        choices=method_choices,
        default="mean_diff",
        help=f"Vector extraction method. Choices: {', '.join(method_choices)}",
    )
    parser.add_argument(
        "--b-source",
        type=str,
        choices=source_choices,
        default="fixed",
        help=f"B-completion source. Choices: {', '.join(source_choices)}",
    )
    parser.add_argument(
        "--d-source",
        type=str,
        choices=["D", "D_banal"],
        default="D",
        help="Field to use for D (default) completions. 'D' = standard uses, 'D_banal' = banal uses.",
    )

    # Shared flags
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--abcd-data",
        type=Path,
        default=Path("dataset/abcd_aut.json"),
        help="Path to abcd_aut.json or abcd_ps.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bridge_steering"),
        help="Base directory; a setup-named subdir is created automatically.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["aut", "ps"],
        default="aut",
        help="Task type (aut or ps)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of items for train (rest held out for eval)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/eval split",
    )
    parser.add_argument(
        "--eval-holdout",
        type=int,
        default=EVAL_HOLDOUT_DEFAULT,
        help="Number of items to hold out for eval",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=EVAL_SEED_DEFAULT,
        help="Seed for eval holdout indices (reproducibility)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=WINDOW_DEFAULT,
        help="Max completion tokens to average over; 0 = whole completion (no cap)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (saves VRAM)",
    )
    return parser


def main() -> int:
    # --- Pass 1: resolve --method and --b-source -------------------------
    parser = _build_parser()
    pre_args, remaining = parser.parse_known_args()

    extractor_cls = EXTRACTOR_REGISTRY[pre_args.method]
    source_cls = SOURCE_REGISTRY[pre_args.b_source]

    # Let the chosen classes add their own flags
    extractor_cls.add_args(parser)
    source_cls.add_args(parser)

    # --- Pass 2: full parse ----------------------------------------------
    args = parser.parse_args()

    if not args.abcd_data.exists():
        print(f"Error: abcd data not found: {args.abcd_data}", file=sys.stderr)
        return 1

    # Build config dicts for the strategy objects (all non-shared args)
    all_config = vars(args)
    extractor = extractor_cls(all_config)
    source = source_cls(all_config)

    # Extra config keys that affect the slug (method/source-specific args
    # that aren't part of the shared set)
    shared_keys = {
        "method", "b_source", "d_source", "model_name", "abcd_data", "output_dir",
        "task_type", "train_frac", "seed", "eval_holdout", "eval_seed",
        "window", "load_in_8bit",
    }
    extra_config = {k: v for k, v in all_config.items() if k not in shared_keys}

    setup_slug = _setup_slug(
        args.model_name,
        args.task_type,
        args.abcd_data,
        args.train_frac,
        args.seed,
        args.eval_seed,
        args.eval_holdout,
        args.window,
        args.method,
        args.b_source,
        args.d_source,
        extra_config,
    )
    run_dir = args.output_dir / setup_slug
    out_pt = run_dir / STEERING_VECTORS_FILENAME
    if out_pt.exists():
        print(f"Run already exists for this config: {out_pt}")
        print("Skipping. Use a different --output-dir or change a config option to run again.")
        return 0

    # --- Data -------------------------------------------------------------
    task_type = TaskType(args.task_type)
    dataset = ABCDDataset(
        args.abcd_data,
        task_type=task_type,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    items = dataset.items
    pairs = [build_contrastive_pair(item, task_type, d_key=args.d_source) for item in items]

    # Carry B_list through so multi-B completion sources can access it
    for item, pair in zip(items, pairs):
        if "B_list" in item:
            pair["b_list"] = item["B_list"]

    # --- Model ------------------------------------------------------------
    print(f"Loading model: {args.model_name}")
    loader = HFLoader(
        model_name_or_path=args.model_name,
        load_in_8bit=args.load_in_8bit,
    )
    model, tokenizer = loader.load()
    device = next(model.parameters()).device
    n_layers = len(model.model.layers)

    # Probe layers: mid-to-late, step 2
    probe_layers = list(range(n_layers // 4, n_layers, 2))
    probe_layers = [l for l in probe_layers if 0 <= l < n_layers]
    print(f"Probing layers: {probe_layers} (of {n_layers})")

    # Train/eval split
    rng = random.Random(args.eval_seed)
    n_holdout = min(args.eval_holdout, len(pairs))
    eval_idxs = set(rng.sample(range(len(pairs)), n_holdout))
    train_pairs = [p for i, p in enumerate(pairs) if i not in eval_idxs]
    print(f"Train pairs: {len(train_pairs)}, eval held out: {len(eval_idxs)}")

    # Pre-tokenize D completions (B tokenization happens inside collect
    # since the source may produce variable completions)
    for p in train_pairs:
        p["tok_D"] = tokenize_full_sequence(
            p["prompt_msgs"], p["d_completion"], tokenizer
        )

    min_comp_D = min(p["tok_D"]["n_completion_tokens"] for p in train_pairs)
    # window <= 0: use whole completion for each (no cap); else cap by min length
    if args.window <= 0:
        effective_W_D = 0
        effective_W_B = 0
        print(f"WINDOW=whole completion (no cap), effective W_D=all, effective W_B=all")
    else:
        effective_W_D = min(args.window, min_comp_D)
        min_comp_B = _min_B_completion_tokens(train_pairs, source, tokenizer, model)
        effective_W_B = min(args.window, min_comp_B)
        print(f"WINDOW={args.window}, effective W_D={effective_W_D}, effective W_B={effective_W_B}")

    # --- Collect activations (source controls B completions) ---------------
    print(f"B-source: {source_cls.name}, method: {extractor_cls.name}")
    acts_B, acts_D = collect_activations(
        train_pairs, source, model, tokenizer, device,
        probe_layers, effective_W_D, effective_W_B,
    )

    # --- Compute steering vectors (method controls aggregation) -----------
    result = extractor.compute_vectors(acts_B, acts_D, probe_layers)

    steer_layer = select_best_layer(result.stats_df)
    v_steer = result.v_bridge[steer_layer].clone()
    print(f"Best layer: {steer_layer}, v_steer norm = {v_steer.norm().item():.2f}")

    # --- Save -------------------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    use_pca = result.metadata.get("pca_info") is not None
    save_dict = {
        "steer_layer": steer_layer,
        "v_steer": v_steer,
        "v_bridge": result.v_bridge,
        "model_name": args.model_name,
        "task_type": args.task_type,
        "probe_layers": probe_layers,
        "stats_df": result.stats_df,
        "method": args.method,
        "b_source": args.b_source,
        "d_source": args.d_source,
        "use_pca": use_pca,
    }
    save_dict.update(result.metadata)
    torch.save(save_dict, out_pt)

    print(f"Saved: {out_pt}")
    print("Run AUT benchmark with:")
    print(f"  python scripts/run_aut_benchmark.py --vectors {out_pt} --model-name \"{args.model_name}\" --method steered ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
