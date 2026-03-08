# Commands

## LLM as a judge

```bash
cd DLP

# Score a single file
python scripts/run_vllm_judge.py \
  --input results/aut_inference/llama_t07/baseline_uses.txt \
  --output-dir results/judge/llama_t07

# Score multiple files and compare
python scripts/run_vllm_judge.py \
  --input results/aut_inference/llama_t07/baseline_uses.txt \
  --input results/aut_inference/llama_t07/steered_a1.0_uses.txt \
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

# Run on all *_uses.txt under a folder
python scripts/run_vllm_judge.py \
  --input-dir results/aut_inference \
  -o results/judge/all_inference

# Same with max-uses per file
python scripts/run_vllm_judge.py \
  --input-dir results/aut_inference \
  -o results/judge/all_sampled \
  --max-uses 50

# Incremental run: skip methods already judged
python scripts/run_vllm_judge.py \
  --input-dir results/aut_inference \
  -o results/judge/all_qwen32b \
  --skip-existing
```

## AUT inference

```bash
cd DLP

# Baseline only
python scripts/run_aut_inference.py \
  --abcd-data dataset/abcd_aut.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method baseline \
  --output-dir results/aut_inference/llama_baseline

# Baseline + steered (one alpha)
python scripts/run_aut_inference.py \
  --abcd-data dataset/abcd_aut.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method baseline \
  --method steered --vectors results/bridge_steering/.../steering_vectors.pt \
  --output-dir results/aut_inference/llama_compare

# Steered with multiple alphas
python scripts/run_aut_inference.py \
  --abcd-data dataset/abcd_aut.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method steered --vectors results/bridge_steering/.../steering_vectors.pt \
  --alpha 0.5 --alpha 1.0 --alpha 1.5 \
  --output-dir results/aut_inference/llama_alpha_sweep

# Sweep layers × alphas
python scripts/run_aut_inference.py \
  --abcd-data dataset/abcd_aut.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method steered --vectors results/bridge_steering/.../steering_vectors.pt \
  --layer 12 --layer 16 --layer 20 \
  --alpha 0.5 --alpha 1.0 \
  --output-dir results/aut_inference/llama_layer_sweep

# All five methods
python scripts/run_aut_inference.py \
  --abcd-data dataset/abcd_aut.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --vectors results/bridge_steering/.../steering_vectors.pt \
  --method baseline --method steered --method fewshot --method twohop --method abcd_framework \
  --output-dir results/aut_inference/all_methods

# Generate-and-select: N sampled replies per prompt
python scripts/run_aut_inference.py \
  --abcd-data dataset/abcd_aut.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method baseline --do-sample --temperature 0.7 \
  --num-inferences 4 \
  --output-dir results/aut_inference/baseline_sampled

# Brainstorm: use --method brainstorm (sets do-sample, temperature=0.7, num-inferences=4 automatically)
python scripts/run_aut_inference.py \
  --abcd-data dataset/abcd_aut.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method brainstorm \
  --output-dir results/aut_inference/llama_brainstorm
# Then run selection: run_brainstorm_select.py --input .../llama_brainstorm/baseline_uses.txt ...
```

## Bridge steering

```bash
cd DLP && python scripts/run_bridge_steering.py \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --abcd-data dataset/abcd_aut.json \
  --output-dir results/bridge_steering \
  [--b-source fixed] [--method mean_diff] [--use-pca] [--window 48]
```

Relevant steering vectors (raw = no --use-pca):

- **mean diff raw:** `results/bridge_steering/llama_3.1_8b_instruct_aut_abcd_aut_w48_raw_fd62c2fd/steering_vectors.pt`
- **b only raw:** `results/bridge_steering/llama_3.1_8b_instruct_aut_abcd_aut_w48_b_only_fixed_0d7cb86c/steering_vectors.pt`
- **multiple b concat**: `results/bridge_steering/llama_3.1_8b_instruct_aut_abcd_aut_multiple_b_w0_b_only_multi_b_concat_fffbddb5/steering_vectors.pt`
- **multiple b sep**: `results/bridge_steering/llama_3.1_8b_instruct_aut_abcd_aut_multiple_b_w48_b_only_multi_b_separate_6e00ab00/steering_vectors.pt`

Multiple B (concatenated vs separated) — use dataset with `B_list`; same model as above.

B and D use **separate effective windows**: `W_D = min(window, shortest_D_completion)` and `W_B = min(window, shortest_B_completion)`. Use **`--window 0`** to average over the **whole completion** (no cap) for each B and D.

```bash
cd DLP

# Multi-B concatenated (one completion per item, all mechanisms in one block)
python scripts/run_bridge_steering.py \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --abcd-data dataset/abcd_aut_multiple_b.json \
  --output-dir results/bridge_steering \
  --method mean_diff \
  --b-source multi_b_concat \
  --window 0

# Multi-B separated (one completion per mechanism, more B activations per item)
python scripts/run_bridge_steering.py \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --abcd-data dataset/abcd_aut_multiple_b.json \
  --output-dir results/bridge_steering \
  --method mean_diff \
  --b-source multi_b_separate \
  --window 48

# Clustered (k=8, best novelty config)
python scripts/run_bridge_steering.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --abcd-data dataset/abcd_aut_multiple_b.json \
  --method clustered --n-clusters 8 --cluster-weighting size \
  --b-source multi_b_concat --d-source D_banal --window 0 \
  --output-dir results/bridge_steering
```

Use `--method b_only` instead of `--method mean_diff` to get b-only vectors for the same two sources. Then run AUT inference with the new `.../steering_vectors.pt` path from each run's output dir.

## Diversity

```bash
cd DLP && python scripts/compute_response_diversity.py \
  --inference-dirs results/aut_inference/llama_t07 \
                   results/aut_inference/llama_brainstorm \
  --model all-MiniLM-L6-v2 \
  -o results/judge/diversity
```

## OSCAI analysis

```bash
cd DLP
python scripts/analyze_oscai_results.py results/judge/oscai/AUT_OSCAI_RESULTS.xlsx -o results/judge/oscai/analysis
```

## Holistic experiment comparison

Compare selected experiments across Qwen judge scores, diversity, and (optionally) OSCAI
in one table. **Method names and columns are defined in `results/judge/oscai/method_map.csv`**.

```bash
cd DLP

# Compare clustering experiments (judge + diversity + OSCAI)
python scripts/holistic_experiment_comparison.py \
  --folders llama_t07_dbanal_clustered_2 llama_t07_dbanal_clustered_8 \
            llama_t07_dbanal_clustered llama_t07_dbanal_bcon llama_t07 \
  --oscai-stats results/judge/oscai/analysis/stats_by_method.csv \
  -o results/judge/holistic

# Judge + diversity only (no OSCAI)
python scripts/holistic_experiment_comparison.py \
  --folders llama_t07 llama_t15 \
  -o results/judge/holistic

# Sort by diversity instead of novelty
python scripts/holistic_experiment_comparison.py \
  --folders llama_t07 llama_t07_dbanal_bcon \
  --sort-by sem_diversity \
  -o results/judge/holistic
```

## N-gram originality

```bash
cd DLP

# Build corpus index (one-time)
python scripts/build_ngram_index.py \
  --dataset HuggingFaceFW/fineweb \
  --config sample-10BT \
  --num-tokens 10000000 \
  --output results/novelty/fineweb_10m.npz

# Compute originality over all inference dirs
python scripts/compute_originality.py \
  --index results/novelty/fineweb_10b.checkpoint.npz \
  --inference-dirs results/aut_inference/llama_t07 \
                   results/aut_inference/llama_brainstorm_t07 \
  --text-column reply \
  -o results/novelty

# Per-use originality
python scripts/compute_originality.py \
  --index results/novelty/fineweb_10b.checkpoint.npz \
  --inference-dirs results/aut_inference/llama_t07 \
  --text-column reply \
  --per-use \
  -o results/novelty
```

## MacGyver

See `commands_macgyver.md` for MacGyver-specific inference, quality scoring, and novelty table commands.
