# MacGyver Commands

## Dataset

Source: `external_repos/MacGyver/data/MacGyver/problem_solution_pair.xlsx`
Paper: Tian et al., "MacGyver: Are Large Language Models Creative Problem Solvers?", NAACL 2024.

1,683 real-world verbal problems requiring creative use of everyday objects.

### Load with Python

```python
from dlp.data import MacGyverDataset

DATA = "DLP/external_repos/MacGyver/data/MacGyver/problem_solution_pair.xlsx"

# Default: solvable + unconventional (956 items — the core creative subset)
ds = MacGyverDataset(DATA)

# All subsets
ds = MacGyverDataset(DATA, subset="all")                      # 1683
ds = MacGyverDataset(DATA, subset="solvable")                 # 1306
ds = MacGyverDataset(DATA, subset="solvable_unconventional")  # 956
ds = MacGyverDataset(DATA, subset="solvable_conventional")    # 350
ds = MacGyverDataset(DATA, subset="unsolvable")               # 377
ds = MacGyverDataset(DATA, subset="benchmark")                # 323 (exact paper subset)

# With train/eval split
ds = MacGyverDataset(DATA, subset="solvable_unconventional", train_frac=0.8)
train = ds.train_items()  # 764
eval_ = ds.eval_items()   # 192

# Each item is a dict:
#   id, problem, solvable, unconventional, solution, label
```

### Subset breakdown

| Subset                  | Count | Description                                |
|-------------------------|-------|--------------------------------------------|
| `all`                   | 1683  | Full dataset                               |
| `solvable`              | 1306  | Problems that can be solved                |
| `solvable_unconventional` | 956 | Solvable + requires creative tool use      |
| `solvable_conventional` | 350   | Solvable with standard tool use            |
| `unsolvable`            | 377   | Problems that cannot be solved with tools  |
| `benchmark`             | 323   | Exact subset used in the paper's benchmark |

## Inference

Run generation on MacGyver problems with any model/method. Reuses the same method classes as AUT (baseline, steered, etc.). Outputs `{method}_results.csv` consumable by `score_macgyver_quality.py`.

### Baseline (greedy)

```bash
cd DLP

MACGYVER=external_repos/MacGyver/data/MacGyver/problem_solution_pair.xlsx

python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method baseline \
  --output-dir results/macgyver_inference/llama_baseline
```

### Steered with multiple alphas

```bash
cd DLP

python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method steered \
  --vectors results/bridge_steering/.../steering_vectors.pt \
  --alpha 0.5 --alpha 1.0 --alpha 1.5 \
  --output-dir results/macgyver_inference/llama_steered
```


### Steered with layer sweep

```bash
cd DLP

python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method steered \
  --vectors results/bridge_steering/.../steering_vectors.pt \
  --layer 12 --layer 16 --layer 20 \
  --alpha 0.5 --alpha 1.0 \
  --output-dir results/macgyver_inference/llama_layer_sweep
```

### Sampling (temperature > 0)

```bash
cd DLP

python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method baseline --do-sample --temperature 0.7 \
  --output-dir results/macgyver_inference/llama_t07
```

### Novelty-instruct prompt (paper E.2.3)

Uses the paper's prompt that asks the model for creative/novel output and to avoid n-gram patterns from pretraining data. This is the "asking for novelty" baseline from the paper.

```bash
cd DLP

python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method baseline --do-sample --temperature 0.7 \
  --prompt-style novelty_instruct \
  --output-dir results/macgyver_inference/llama_novelty_instruct
```

### Multiple inferences per prompt

```bash
cd DLP

python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method baseline --do-sample --temperature 0.7 \
  --num-inferences 4 \
  --output-dir results/macgyver_inference/llama_sampled
```

### Subset selection

```bash
cd DLP

# Benchmark subset (323 items from the paper)
python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --subset benchmark \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --output-dir results/macgyver_inference/llama_benchmark

# All solvable (1306)
python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --subset solvable \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --output-dir results/macgyver_inference/llama_solvable

# Quick test: first 20 items
python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --max-items 20 \
  --output-dir results/macgyver_inference/debug
```

### With train/eval split

```bash
cd DLP

# 80/20 split — only run inference on the 20% eval set
python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --train-frac 0.8 \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --output-dir results/macgyver_inference/llama_eval_split
```

### Baseline + steered comparison (single run)

```bash
cd DLP

python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method baseline \
  --method steered --vectors results/bridge_steering/.../steering_vectors.pt \
  --alpha 1.0 --alpha 1.5 \
  --do-sample --temperature 0.7 \
  --output-dir results/macgyver_inference/llama_compare
```

### Output structure

```
results/macgyver_inference/llama_baseline/
├── baseline_results.csv          # eval_idx, sample_idx, id, method, user_prompt,
                                  # reply, reply_len_words, solvable, unconventional,
                                  # ref_solution, ref_label
```

## Full pipeline: inference → quality scoring

```bash
cd DLP

MACGYVER=external_repos/MacGyver/data/MacGyver/problem_solution_pair.xlsx

# 1. Inference
python scripts/run_macgyver_inference.py \
  --macgyver-data $MACGYVER \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method baseline \
  --do-sample --temperature 0.7 \
  --output-dir results/macgyver_inference/llama_t07

# 2. Quality scoring (vLLM judge must be running)
python scripts/score_macgyver_quality.py \
  --input results/macgyver_inference/llama_t07/baseline_results.csv \
  -o results/novelty/macgyver_quality

# Or score all methods at once
python scripts/score_macgyver_quality.py \
  --input-dir results/macgyver_inference/llama_t07 \
  -o results/novelty/macgyver_quality
```

---

## Quality scoring (LLM-as-a-judge)

Uses the 5-point additive rubric via vLLM. Requires a vLLM server running the judge model.

### Start vLLM judge server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model prometheus-eval/prometheus-7b-v2.0 \
  --port 8000 \
  --tensor-parallel-size 2
```

The server logs "Running: X reqs, Waiting: Y reqs" and throughput; it does not show how many requests have *completed*. To see progress in a separate terminal, use the metrics helper below.

### Monitor judge progress (vLLM completed requests)

While `score_macgyver_quality.py` is running, open a **second terminal** and poll vLLM’s `/metrics` to see how many judge requests have finished:

```bash
cd DLP

# Show completed count only (updates every 5s)
python scripts/vllm_request_progress.py

# Show progress toward total (e.g. 4 methods × 956 = 3824)
python scripts/vllm_request_progress.py --total 3824
```

This reads the `vllm:request_success` counter from `http://localhost:8000/metrics`. Use `Ctrl+C` to stop the progress loop.

**Why doesn’t the progress bar show in the scoring terminal?**  
The scoring script only draws a tqdm bar when stderr is a real TTY. In Cursor’s (and many IDE) terminals, stderr is not a TTY, so the bar is disabled and you only see `[progress] N/Total (pct%)` lines every ~2%. Run with `python -u scripts/score_macgyver_quality.py ...` if those lines don’t appear (unbuffered stderr). For live completed count, use `vllm_request_progress.py` in a second terminal.

### Speed (if judging shows ~1 item in 10+ minutes)

The per-request `--timeout` (default: 600s) must be long enough for concurrent generation. With 256 concurrent requests on a 32B model, each request gets ~1 token/s. At `max_tokens=300`, each takes ~300s. If the timeout is too short (e.g. 60s), almost every request times out and retries, wasting GPU cycles and making tqdm appear stuck.

Options to speed up judging:

1. **Lower concurrency** — fewer concurrent requests means each gets more throughput and finishes faster:
   ```bash
   python scripts/score_macgyver_quality.py \
     --input-dir results/macgyver_inference \
     -o results/novelty/macgyver_quality \
     --max-concurrent 32 --timeout 120
   ```
   With 32 concurrent requests the model generates ~8 tokens/s each, finishing in ~40s.

2. **Faster judge model** — use a smaller model (e.g. 7B instead of 32B) for the vLLM server; quality may drop slightly but throughput is much higher.

### Score a single inference results file

```bash
cd DLP

python scripts/score_macgyver_quality.py \
  --input results/macgyver_inference/baseline_results.csv \
  -o results/novelty/macgyver_quality
```

### Score all results under a directory

```bash
cd DLP

python scripts/score_macgyver_quality.py \
  --input-dir results/macgyver_inference \
  -o results/novelty/macgyver_quality
```

### Custom judge model

```bash
cd DLP

python scripts/score_macgyver_quality.py \
  --input results/macgyver_inference/baseline_results.csv \
  --vllm-url http://localhost:8000/v1 \
  --judge-model Qwen/Qwen3-32B \
  -o results/novelty/macgyver_quality
```

### Incremental scoring (skip already-scored methods)

With `--skip-existing`, methods that already have a `{method}_macgyver_scores.csv` in the output dir are not re-judged. New methods are scored and the summary is rebuilt from existing + new scores (same behaviour as the AUT judge).

```bash
cd DLP

python scripts/score_macgyver_quality.py \
  --input-dir results/macgyver_inference \
  -o results/novelty/macgyver_quality \
  --skip-existing
```

### Cap items per method (debug / quick test)

```bash
cd DLP

python scripts/score_macgyver_quality.py \
  --input results/macgyver_inference/baseline_results.csv \
  -o results/novelty/macgyver_quality \
  --max-items 20
```

### Override column names

When input CSVs use non-standard column names:

```bash
cd DLP

python scripts/score_macgyver_quality.py \
  --input results/macgyver_inference/custom_results.csv \
  --prompt-col problem_text \
  --reply-col generated_solution \
  -o results/novelty/macgyver_quality
```

### Output structure

```
results/novelty/macgyver_quality/
├── baseline_macgyver_scores.csv          # per-item: id, problem, response, quality_score, judge_explanation
├── steered_a1.0_macgyver_scores.csv
├── ...
└── macgyver_quality_summary.csv          # per-method: mean, median, std, min, max quality
```

### Input format

The script auto-detects column names. Recognized prompt columns: `user_prompt`, `problem`, `prompt`, `Problem`. Recognized reply columns: `reply`, `model_response`, `response`, `Solution`, `solution`. If `method` column is missing, the filename stem is used (e.g. `baseline_results.csv` → method `baseline`).

### Rubric

5-point additive (0–5):

1. Uses only given resources (no external tools introduced)
2. Correct understanding of resource properties and limitations
3. Adheres to physical constraints (size, weight, strength)
4. Practical and effective within scenario constraints
5. Complete, logically structured, clear explanation

### Programmatic use

```python
from dlp.evaluation.judge import MacGyverJudge

# Sync (one-at-a-time, via OpenAI-compatible API)
judge = MacGyverJudge(model="gpt-4o-mini", temperature=0.0)
results = judge.rate([
    {"user_prompt": "You spilled wine...", "model_response": "Step1: ..."},
])
# results[0]["quality_score"] → int 0-5
# results[0]["judge_explanation"] → str

# Async (high-throughput vLLM) — used by the CLI script
from openai import AsyncOpenAI
import asyncio

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="vllm")
sema = asyncio.Semaphore(256)
result = await MacGyverJudge.rate_one_async(
    client, "prometheus-eval/prometheus-7b-v2.0",
    user_prompt="...", model_response="...",
    sema=sema,
)
```

## N-gram originality (novelty metric)

Fraction of n-grams (n=4,5,6) in generated text that do **not** appear in a reference corpus. Based on Padmakumar et al., "Measuring LLM Novelty As The Frontier Of Original And High-Quality Output", ICLR 2026. Useful alongside quality for a full novelty view (originality × quality).

### Build corpus index (one-time)

Stream 5–10M (or 10B) word tokens from FineWeb, build and save the n-gram index.

**Runtime:** 10B tokens often takes on the order of **several hours** (e.g. 6–10+) depending on machine, network, and HuggingFace load. 10M tokens is usually a few minutes.

**If the run stops (crash, kill, OOM):** use `--checkpoint-every` so that progress is written periodically. You can then **resume** from the checkpoint with `--resume` (see below), or use the partial index as-is.

```bash
cd DLP

# 10M tokens (faster, smaller index)
python scripts/build_ngram_index.py \
  --dataset HuggingFaceFW/fineweb \
  --config sample-10BT \
  --num-tokens 10000000 \
  --n-values 4 5 6 \
  -o results/novelty/fineweb_10m.npz

# 10B tokens (long run; use checkpoint so you have a safe backup)
python scripts/build_ngram_index.py \
  --dataset HuggingFaceFW/fineweb \
  --config sample-10BT \
  --num-tokens 10000000000 \
  --checkpoint-every 500000000 \
  -o results/novelty/fineweb_10b.npz
```

Checkpoints are written **uncompressed** so each save finishes much faster (the build can appear stuck for many minutes if it had to compress hundreds of millions of n-grams). The final index is still saved compressed.

### If the 10B build stops: use the checkpoint

Checkpoints are written to `{output_stem}.checkpoint.npz` (and `.checkpoint.meta.json`) every `--checkpoint-every` tokens. If the process is killed or crashes:

- **Resume the build** — add `--resume` to continue from the checkpoint:
  ```bash
  python scripts/build_ngram_index.py \
    --dataset HuggingFaceFW/fineweb \
    --config sample-10BT \
    --num-tokens 10000000000 \
    --checkpoint-every 500000000 \
    --resume \
    -o results/novelty/fineweb_10b.npz
  ```
  The script loads the checkpoint, skips already-processed documents in the
  HuggingFace stream, and continues building toward the target token count.
- **Use the checkpoint as the index** — you can pass it directly to `compute_originality.py`; no need to copy:
  ```bash
  python scripts/compute_originality.py \
    --index results/novelty/fineweb_10b.checkpoint.npz \
    --inference-dirs results/macgyver_inference \
    -o results/novelty
  ```
- Or copy to a clearer name (e.g. `fineweb_5b.npz`) and use that.

### Score whole reply (one originality score per inference row)

```bash
cd DLP

# All methods in one inference dir
python scripts/compute_originality.py \
  --index results/novelty/fineweb_10b.checkpoint.npz \
  --inference-dirs results/macgyver_inference/llama_steered_cluster8/ \
  -o results/novelty

# Single CSV
python scripts/compute_originality.py \
  --index results/novelty/fineweb_10b.checkpoint.npz \
  --input results/macgyver_inference/baseline_results.csv \
  --text-column reply \
  -o results/novelty
```

### Score per use (when each reply has multiple uses)

Uses `{method}_uses.txt` when present (same format as AUT: one line per use). Otherwise parses the reply column.

```bash
cd DLP

python scripts/compute_originality.py \
  --index results/novelty/fineweb_10m.npz \
  --inference-dirs results/macgyver_inference \
  --per-use \
  -o results/novelty
```

### Expand a parent directory

Pass the whole `results/aut_inference` (or similar) folder; subdirs that contain `*_results.csv` are auto-discovered:

```bash
cd DLP

python scripts/compute_originality.py \
  --index results/novelty/fineweb_10m.npz \
  --inference-dirs results/aut_inference \
  --per-use \
  -o results/novelty
```

### Originality output structure

```
results/novelty/
├── originality_summary.csv                    # per (dir, method): count, originality_4/5/6 mean, median, std, min, max
└── <dir_label>/                               # e.g. macgyver_inference or llama_t07
    ├── baseline_originality.csv               # whole-reply mode: text_idx, num_tokens, originality_4, originality_5, originality_6
    ├── baseline_originality_per_use.csv       # per-use mode: use_idx, object, use, num_tokens, originality_4/5/6
    └── ...
```

## Novelty table (paper format)

Combines quality scores (0–5 judge) + n-gram originality into the paper's novelty metric:

1. **Quality** is normalized to [0, 1] by dividing by 5.
2. **Novelty** = harmonic mean of normalized quality and originality (per item).
3. **Top-10% Novelty** = mean novelty over the top 10% of items.
4. **Delta to Baseline** = method novelty minus baseline novelty.

### Generate the table

Requires two inputs:
- `--quality-dir`: directory with `*_macgyver_scores.csv` (from `score_macgyver_quality.py`)
- `--originality-dir`: directory with `*_originality.csv` (from `compute_originality.py`)

```bash
cd DLP

python scripts/compute_macgyver_novelty_table.py \
  --quality-dir results/novelty/macgyver_quality_t07 \
  --originality-dir results/novelty/llama_steered_cluster8 \
  -o results/novelty/macgyver_novelty_table.csv
```

### With explicit baseline

```bash
python scripts/compute_macgyver_novelty_table.py \
  --quality-dir results/novelty/macgyver_quality_t07 \
  --originality-dir results/novelty/llama_steered_cluster8 \
  --baseline llama_steered_cluster8_baseline \
  -o results/novelty/macgyver_novelty_table.csv
```

### Output columns

| Column | Description |
|--------|-------------|
| `quality` | Mean raw quality score (0–5) |
| `quality_norm` | Mean normalized quality (0–1) |
| `orig_n4/5/6` | Mean n-gram originality |
| `novelty_n4/5/6` | Mean harmonic mean of norm_quality and originality |
| `top10_novelty_n4/5/6` | Mean novelty over top 10% items |
| `delta_novelty_n4/5/6` | Novelty minus baseline novelty |
| `delta_top10_n4/5/6` | Top-10% novelty minus baseline |
