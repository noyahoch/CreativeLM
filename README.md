# DLP – Deep Learning for Problem-solving

Research codebase for studying mechanism-driven creativity in LLMs
(Alternative Uses Task and Problem Solving).

## Project structure

```
DLP/
├── pyproject.toml           # Single uv project (Python ≥ 3.9)
├── dlp/                     # Core Python package
│   ├── data/                # ABCDDataset, MacGyverDataset
│   ├── models/              # HFLoader, OpenAILoader
│   ├── steering/            # Bridge-steering vectors and hooks
│   ├── training/            # LoRA SFT helpers + data preparation
│   ├── evaluation/          # AUT LLM-as-a-judge pipeline
│   ├── novelty/             # N-gram originality metrics
│   └── utils/               # Config dataclasses
├── scripts/                 # CLI entry points
│   ├── run_bridge_steering.py       # Compute steering vectors (B vs D)
│   ├── run_aut_inference.py         # AUT inference (baseline/steered/fewshot/...)
│   ├── run_macgyver_inference.py    # MacGyver inference
│   ├── run_vllm_judge.py           # LLM-as-judge (novelty/usability)
│   ├── score_macgyver_quality.py    # MacGyver quality scoring
│   ├── run_aut_benchmark.py         # CreativityPrism AUT benchmark
│   ├── run_judge.py                 # AUT generate+judge (OpenAI)
│   ├── run_inference.py             # Simple HF inference
│   ├── run_brainstorm_select.py     # Brainstorm-then-select
│   ├── select_best_uses.py          # Best reply selection
│   ├── build_ngram_index.py         # Build n-gram corpus index
│   ├── compute_originality.py       # N-gram originality scoring
│   ├── compute_macgyver_novelty_table.py  # Paper novelty table
│   ├── compute_response_diversity.py      # Semantic/n-gram diversity
│   ├── holistic_experiment_comparison.py  # Cross-experiment comparison
│   ├── analyze_judge_scores.py      # Judge score analysis
│   ├── analyze_oscai_results.py     # OSCAI analysis
│   ├── qualitative_alpha_sweep.py   # Qualitative examples
│   ├── inspect_steering_layers.py   # Inspect steering vectors
│   └── vllm_request_progress.py     # vLLM progress monitor
├── notebooks/               # Jupyter notebooks (import from dlp package)
│   ├── bridge_steering.ipynb
│   ├── sft_bridge_internalization.ipynb
│   ├── eval_bridge_internalization.ipynb
│   └── direct_prompting.ipynb
├── dataset/                 # ABCD JSON datasets
├── docs/                    # Experiment documentation
├── results/                 # Experiment outputs (gitignored)
│   ├── bridge_steering/
│   ├── aut_inference/
│   ├── macgyver_inference/
│   ├── judge/
│   └── novelty/
└── external_repos/          # Third-party repos (gitignored local copies)
    └── CreativityPrism/
```

## Setup

```bash
cd DLP
uv sync
```

This creates `.venv/` with all dependencies. Use `uv run python ...` or activate
`.venv/bin/activate` before running scripts/notebooks.

### Jupyter kernel

```bash
uv run python -m ipykernel install --user --name dlp --display-name "DLP"
```

Then select the **DLP** kernel in JupyterLab / Jupyter Notebook.

### API keys

Set in environment or a `.env` file (gitignored):

```bash
export OPENAI_API_KEY=sk-...
```

## Quick start

### Run HF inference

```bash
# Single prompt
uv run python scripts/run_inference.py --prompt "List 5 creative uses for a brick."

# Dataset
uv run python scripts/run_inference.py --dataset dataset/abcd_aut.json --output results/inference.json
```

### Bridge steering

```bash
# Compute steering vectors (B vs D contrastive pairs)
uv run python scripts/run_bridge_steering.py \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --abcd-data dataset/abcd_aut.json \
  --output-dir results/bridge_steering \
  [--b-source fixed] [--method mean_diff] [--use-pca] [--window 0]
```

### AUT inference (baseline + steered)

```bash
uv run python scripts/run_aut_inference.py \
  --abcd-data dataset/abcd_aut.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --method baseline \
  --method steered --vectors results/bridge_steering/<setup_slug>/steering_vectors.pt \
  --alpha 0.5 --alpha 1.0 --alpha 1.5 \
  --do-sample --temperature 0.7 \
  --output-dir results/aut_inference/llama_steered

# Steering vectors must be from the same model (same hidden size).
```

### AUT judge pipeline

```bash
# Generate uses + judge in one step
uv run python scripts/run_judge.py --mode full \
  --objects brick paperclip toothbrush \
  --output-dir results/judge/run1

# Judge existing outputs via vLLM
uv run python scripts/run_vllm_judge.py \
  --input-dir results/aut_inference \
  -o results/judge/all_qwen32b \
  --skip-existing
```

### Use the dlp package in a notebook

```python
import sys; sys.path.insert(0, "..")  # from notebooks/

from dlp.models import HFLoader
from dlp.steering import steered_generate, compute_bridge_vectors
from dlp.evaluation import run_pipeline
```

## External repos

`external_repos/CreativityPrism` is not tracked by git. Clone it locally:

```bash
git clone https://github.com/your-org/CreativityPrism external_repos/CreativityPrism
```
