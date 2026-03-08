# Appendix: Steering Experiment Details

This appendix specifies the experimental setup for bridge steering (activation steering) used in the paper: data, hyperparameters, vector extraction, layer selection, and inference.

---

## A.1 Data

| Setting | Value |
|--------|--------|
| **Dataset** | ABCD-AUT (`dataset/abcd_aut.json`) |
| **Items** | 150 (each with prompt A, default uses D, mechanism B, creative uses C) |
| **Train / eval split** | 110 train pairs, 40 held out for eval |
| **Split seed** | `eval_seed=7` (reproducible holdout indices) |
| **Task type** | AUT (Alternative Uses Task); same pipeline supports `task_type=ps` (problem-solving) with `abcd_ps.json` |

**Contrastive pair per item:** One pair per item. Same user prompt (A + “Return exactly 8 unconventional uses…” + format instructions). Two completions:

- **D (default):** The eight default uses (listing mode).
- **B (mechanism):** Mechanism-only text: “Mechanism type: …”, “Rule: …”, “This unlocks: …” (no uses, no C, no Justification).

Only **train** pairs are used to compute steering vectors; eval items are never seen during vector construction.

---

## A.2 Model and Probing

| Setting | Default | Description |
|--------|--------|-------------|
| **Model** | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace CausalLM; any compatible model is supported. |
| **Probed layers** | `range(n_layers // 4, n_layers, 2)` | Mid-to-late layers, step 2 (e.g. for 32 layers: 8, 10, 12, …, 30). |
| **Activation readout** | Residual stream | Per-layer activations at completion-token positions. |
| **Completion window** | **0** (whole completion) or **48** tokens | `--window 0` averages over the entire completion (no cap); `--window 48` averages over the first 48 tokens (or the whole completion if shorter). The majority of experiments (all `multi_b_concat` runs) used `--window 0`; earlier single-B runs used `--window 48`. See Table A.8 for details per configuration. |
| **Load in 8-bit** | Off | `--load-in-8bit` optional to reduce VRAM. |

---

## A.3 Vector Extraction

Two composable strategy axes control the steering pipeline. Any source can be combined with any extraction method.

### Axis 1: B-completion source (`--b-source`)

Controls how the B-condition (mechanism-reasoning) completions are produced for each contrastive pair.

| Source | Description | B activations per item |
|--------|------------|------------------------|
| **`fixed`** | Single teacher-forced B completion from the dataset (`b_completion` field). | 1 |
| **`multi_b_separate`** | Multiple B texts from `B_list` field in `abcd_aut_multiple_b.json`; each scored independently. | \|B_list\| (typically 5–8) |
| **`multi_b_concat`** | Multiple B texts from `B_list`, **concatenated** into a single long completion, then teacher-forced as one sequence. | 1 (but covers all B variants) |

### Axis 2: Extraction method (`--method`)

Controls how the collected per-layer activations are aggregated into a single steering vector per layer.

#### 1. Mean-difference (`mean_diff`)

The default method. Per layer \(L\):

\[
v_{\text{bridge}}^{(L)} = \mu_B^{(L)} - \mu_D^{(L)}
\]

where \(\mu_B, \mu_D\) are the centroids of B- and D-condition activations over all train items. Optionally refined with **PCA** (`--use-pca`): compute per-item differences \(d_i = act_{B,i} - act_{D,i}\), take PC1 of the centered difference matrix, sign-align to \(\mu_B - \mu_D\), and scale to \(\|\mu_B - \mu_D\|\) so that \(\alpha\) remains comparable.

#### 2. B-only (`b_only`)

Uses only the B-condition centroid as the steering direction — no subtraction of D:

\[
v_{\text{bridge}}^{(L)} = \hat{\mu}_B^{(L)} \cdot \|\mu_B^{(L)} - \mu_D^{(L)}\|
\]

The direction is \(\mu_B\) but it is **scaled to \(\|B - D\|\) norm** so that \(\alpha\) is comparable to `mean_diff`. Without this scaling, \(\mu_B\) has much larger norm and moderate alphas would overshoot. Optional `--use-pca` replaces the centroid with PC1 of the B activation cloud (re-scaled after PCA).

#### 3. Negative-D (`neg_d`)

Steers *away* from listing mode without any B signal:

\[
v_{\text{bridge}}^{(L)} = -\hat{\mu}_D^{(L)} \cdot \|\mu_B^{(L)} - \mu_D^{(L)}\|
\]

Tests whether simply moving away from D is sufficient. Scaled to \(\|B - D\|\) norm.

#### 4. B-perpendicular (`b_perp`)

Projects out the D component from B, keeping only the signal orthogonal to listing mode:

\[
v_{\text{bridge}}^{(L)} = \mu_B - \text{proj}_{\mu_D}(\mu_B) = \mu_B - \frac{\mu_B \cdot \mu_D}{\|\mu_D\|^2}\mu_D
\]

Scaled to \(\|B - D\|\) norm. Optional `--use-pca` runs PCA on per-item B-perp vectors.

#### 5. Clustered (`clustered`)

K-means on per-item difference vectors \(\{d_i = act_{B,i} - act_{D,i}\}\). The final vector is a weighted combination of cluster centroids:

\[
v_{\text{bridge}}^{(L)} = \sum_{c=1}^{k} w_c \cdot \mu_c
\]

where \(\mu_c\) is the centroid of cluster \(c\) and \(w_c\) is uniform (default) or proportional to cluster size. Experiments tested **k = 2, 4, 8**. Captures sub-strategies in the B-D direction that a single centroid might average out.

#### 6. Multi-PCA (`multi_pca`)

Retains the top-\(k\) principal components of the per-item differences (default \(k = 5\)), each sign-aligned to \(\mu_B - \mu_D\), combined with variance-based weighting:

\[
v_{\text{bridge}}^{(L)} = \sum_{j=1}^{k} \frac{\sigma_j^2}{\sum_i \sigma_i^2} \cdot \text{PC}_j
\]

Scaled to \(\|B - D\|\) norm.

### D-source variants

| D source | Description |
|----------|-------------|
| **D** (default) | Standard default uses from the dataset. |
| **D_banal** | Deliberately banal/obvious uses, sharpening the contrast with creative B. |

### Layer selection (single steer layer)

For each probed layer, the script computes diagnostics on the train items:

- **rel_signal** = \(\|v_{\text{bridge}}\| / \|\mu_D\|\). Normalizes by D-centroid norm so that layers are comparable (later layers have larger raw norms).
- **frac_positive** = fraction of train items whose per-item difference \(d_i = act_{B,i} - act_{D,i}\) projects positively onto \(\hat{v}_{\text{bridge}}\). A value of 1.0 means all items separate correctly.

**Selection rule:** Restrict to layers with **frac_positive \(\geq\) 1.0** (full separability). Among those, choose the layer with **highest rel_signal**. That layer index is saved as `steer_layer` in `steering_vectors.pt`.

### Norm scaling (alpha comparability)

All extraction methods (except raw `mean_diff`) rescale the final vector to match \(\|\mu_B - \mu_D\|\) at each layer. This ensures that the same \(\alpha\) value produces comparable steering strength across methods. For `mean_diff` the norm is already \(\|\mu_B - \mu_D\|\) by construction.

---

## A.4 Training-Time Hyperparameters (Script Defaults)

| Argument | Default | Description |
|----------|--------|-------------|
| `--train-frac` | 0.8 | Not used when holdout is fixed by `--eval-holdout`. |
| `--seed` | 42 | Random seed (e.g. for data ordering / splits that use it). |
| `--eval-holdout` | 40 | Number of items held out for eval. |
| `--eval-seed` | 7 | Seed for sampling holdout indices. |
| `--window` | 48 (script default), but **0 in most runs** | Max completion tokens to average over; 0 = whole completion (no cap). Most multi-B runs used `--window 0`. |

---

## A.5 Inference-Time Steering

| Setting | Default | Description |
|--------|--------|-------------|
| **Hook** | Add \(\alpha \cdot v_{\text{steer}}\) to the residual stream at `steer_layer`. | Applied at the selected layer during generation. |
| **Steer mode** | `all_new_tokens` | When to apply: all new (assistant) tokens. Alternatives: `first_k_assistant_tokens`, `last_prompt_only`, `all`. |
| **`--k-assist`** | 16 | When steer mode is `first_k_assistant_tokens`, number of initial assistant tokens to steer. |
| **Alpha (\(\alpha\))** | 1.0 | Steering strength; commonly swept (e.g. 0.5, 1.0, 1.5). Default single value 1.0. |
| **Vector** | From checkpoint | `steer_layer` and `v_steer` (or chosen layer from `v_bridge`) loaded from `steering_vectors.pt`. |

Steering vectors are **model-specific** (same architecture and hidden size); e.g. Llama-3.1-8B vectors cannot be used with Qwen2-7B.

---

## A.6 Outputs and Reproducibility

- **Checkpoint:** `steering_vectors.pt` contains: `steer_layer`, `v_steer`, `v_bridge` (all probed layers), `model_name`, `task_type`, `probe_layers`, `stats_df`, `method`, `b_source`, `d_source`, `use_pca`, and method-specific metadata.
- **Run directory:** Script writes to `<output-dir>/<setup_slug>/`. The setup slug is a hash of (model, task, data path, train/eval/window/method/b_source and their options). Same config → same slug → skip if file exists; different config → new subdir.
- **Inspection:** To see per-layer stats and the chosen layer:
  ```bash
  python scripts/inspect_steering_layers.py --vectors results/bridge_steering/<setup>/steering_vectors.pt [--csv out.csv]
  ```

---

## A.7 Summary Table (Quick Reference)

| Stage | Detail |
|-------|--------|
| **Data** | 150 ABCD-AUT items; 110 train, 40 eval; one (prompt, D completion) vs (prompt, B completion) per item. |
| **Activations** | Residual stream, train only; layers 8–30 (step 2) for 32-layer model. Window = whole completion (`--window 0`) for most runs; first 48 tokens for early single-B runs. |
| **Vector** | \(\mu_B - \mu_D\) (optionally PCA-refined: PC1 scaled to \(\|\mu_B - \mu_D\|\)). |
| **Layer** | Among layers with frac_positive = 1.0, layer with largest rel_signal. |
| **Inference** | Add \(\alpha \cdot v_{\text{steer}}\) at `steer_layer` for all new tokens (or first K); \(\alpha\) e.g. 0.5, 1.0, 1.5. |

Implementation: `scripts/run_bridge_steering.py`, `scripts/run_aut_inference.py`, `scripts/run_macgyver_inference.py`; `dlp.steering` (vectors, extractors, completion sources, hooks).

---

## A.8 Per-Configuration Run Table

The table below lists every steering vector configuration that was run. 13 distinct steering vector checkpoints were used; each was evaluated at multiple alphas during inference.

### Window = 0 (whole completion, no cap)

All `multi_b_concat` configurations used `--window 0` with `abcd_aut_multiple_b.json`, averaging over the **entire** B and D completions.

| Paper name | Method | B-source | D-source | Alphas tested | Notes |
|-----------|--------|----------|----------|---------------|-------|
| Mean-diff B-concat | `mean_diff` | `multi_b_concat` | D (diverse) | 0.5, 1.0, 1.25, 1.5 | Also tested with `first_k_assistant_tokens` (k=45) injection. |
| D-banal B-concat | `mean_diff` | `multi_b_concat` | D_banal | 0.5, 1.0, 1.25, 1.5 | Primary configuration. |
| D-banal B-concat PCA-orig | `mean_diff` + PCA | `multi_b_concat` | D_banal | 0.5, 1.0, 1.25, 1.5 | PCA-refined mean-diff. |
| B-only concat | `b_only` | `multi_b_concat` | — | 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25 | B centroid only, no D subtraction; scaled to \(\|B-D\|\). |
| Clustered k=2 | `clustered` (k=2) | `multi_b_concat` | D_banal | 0.5, 1.0, 1.25, 1.5 | 2 sub-strategy clusters. |
| Clustered k=4 | `clustered` (k=4) | `multi_b_concat` | D_banal | 0.5, 1.0, 1.25, 1.5 | 4 sub-strategy clusters. |
| Clustered k=8 | `clustered` (k=8) | `multi_b_concat` | D_banal | 0.5, 1.0, 1.25, 1.5 | **Best novelty overall** at \(\alpha\)=1.5. |
| D-banal B-concat PCA | `multi_pca` (5 components) | `multi_b_concat` | D_banal | 0.5, 1.0, 1.25, 1.5 | Multi-PCA, variance-weighted. |
| D-banal B-concat proj | `b_perp` | `multi_b_concat` | D_banal | 0.5, 1.0, 1.25, 1.5 | B orthogonal to D. |
| D-banal neg | `neg_d` | `multi_b_concat` | D_banal | 0.5, 1.0, 1.25, 1.5 | Steer away from D only (no B signal). |

### Window = 48 (first 48 completion tokens)

Earlier and single-B experiments used `--window 48` with `abcd_aut.json` (single B per item).

| Paper name | Method | B-source | D-source | Alphas tested | Notes |
|-----------|--------|----------|----------|---------------|-------|
| Steered (T=0.7) | `mean_diff` (raw) | `fixed` | D | 0.5, 1.0, 1.25, 1.5 | Original pipeline, no PCA. Also tested at T=0 and T=1.5. |
| D-banal | `mean_diff` (raw) | `fixed` | D_banal | 0.5, 1.0, 1.25, 1.5 | Same vector as above; D_banal used only at inference prompt. |
| Steered L20 | `mean_diff` (raw) | `fixed` | D | 1.0, 1.25, 1.5 | Explicit layer override (L=20) instead of auto-selected. |
| Alpha sweep | `mean_diff` (raw) | `fixed` | D | 0.5, 1.0, 1.5, 1.75, 2.0 | Greedy (T=0) sweep. |
| B-only fixed | `b_only` | `fixed` | — | 0.5, 1.0, 1.25, 1.5 | B centroid from single fixed B. |
| B-only separate | `b_only` | `multi_b_separate` | — | 0.5, 1.0, 1.25, 1.5 | Multi-B scored independently (not concatenated). |

### Non-steered baselines and comparisons

| Paper name | Method | Temperature | Notes |
|-----------|--------|-------------|-------|
| Baseline | Greedy | 0.0 | Llama 3.1 8B, no steering. |
| Baseline (T=0.7) | Sampling | 0.7 | Same model, temperature 0.7. |
| Baseline (T=1.5) | Sampling | 1.5 | Same model, high temperature. |
| CrPO (novelty) | CrPO-finetuned model | 0.0 | CNCL-Penn-State/CrPO-llama-3.1-8b-instruct-nov. |
| CrPO novelty (T=0.7) | CrPO-finetuned model | 0.7 | Same CrPO model with sampling. |
| CrPO (nov+div+sur) | CrPO-finetuned model | 0.0 | CrPO trained on novelty+diversity+surprise. |
| CrPO (nov+div+sur T=0.7) | CrPO-finetuned model | 0.7 | Same, with sampling. |
| Creative decoding (min-p=0.05) | min-p sampling | 1.0 | `min_p=0.05`, `temperature=1.0`. |
| Creative decoding (min-p=0.1) | min-p sampling | 1.5 | `min_p=0.1`, `temperature=1.5`. |
| Brainstorm (n=4 T=0.7) | Sample + select | 0.7 | 4 samples per prompt, best selected. |

### Inference-time parameters across all steered runs

| Parameter | Values used |
|-----------|-------------|
| **Alpha (\(\alpha\))** | 0.5, 1.0, 1.25, 1.5 (most configs); extended to 1.75, 2.0, 2.25 for B-only concat. |
| **Temperature** | 0.0 (greedy, alpha sweep), 0.7 (primary), 1.5 (high-temp ablation). |
| **Steer mode** | `all_new_tokens` (default for all runs); `first_k_assistant_tokens` (k=45) tested for Mean-diff B-concat. |
| **Model** | Llama 3.1 8B Instruct for all runs (one Qwen2-7B run exists but not in the comparison table). |

**Key takeaway:** The majority of experiments used `--window 0` (whole completion) with `multi_b_concat` B-source and D_banal as the contrastive condition. The best-performing configuration was **Clustered k=8** with \(\alpha\)=1.5, achieving the highest judge novelty score (6.905).
