# Bridge steering experiment (Experiment 2) — How it was done

This doc summarizes how **contrastive “bridge-mode” steering** works: prompt pairs, vector computation, and usage in AUT. Bridge steering is run via **`scripts/run_bridge_steering.py`** only.

---

## Goal

We isolate a **bridge-mode direction** in the model’s residual stream (listing vs mechanism-reasoning) and at inference add **α · v_steer** so the model is nudged toward mechanism reasoning and more creative/unconventional outputs.

---

## Flow (narrative)

The data comes from **`DLP/dataset/abcd_aut.json`**: each item has a task **A** (e.g. “Give 8 unconventional uses for a rubber band in an office setting”), eight default uses **D**, a mechanism **B** (type, rule, unlocks, justification), and creative uses **C** (not used for building the steering vector). For each item we build **one contrastive pair** with the **same** user prompt and **two different teacher-forced completions**. The shared prompt is **A** plus “Return exactly 8 unconventional uses” and “Format: one use per line, starting with ‘-’.” In condition **D** the completion is the bullet list of the eight default uses (listing mode). In condition **B** the completion is mechanism-only text: exactly three lines—“Mechanism type: …”, “Rule: …”, and “This unlocks: …”—with **no uses, no C, and no Justification** (the Justification line is commented out in `_format_B_completion_aut`). So even the minimal “type3” B completion (type, rule, unlocks only) does not include C. Only the PS formatter `_format_B_completion_ps` appends the full creative solution C after the mechanism. For AUT the contrast is deliberately **listing vs mechanism reasoning** on the same instruction.

For each train pair we form two full sequences (prompt + D completion and prompt + B completion), tokenize with the chat template, and run a forward pass for each. At selected layers we read the **residual-stream activation** at the positions of the **first 48 completion tokens** (or fewer if one completion is shorter) and average over those positions. We do this over **train pairs only** (e.g. 110 of 150; the rest are held out for eval). Per layer we get mean D-condition activations **μ_D** and mean B-condition activations **μ_B**, and a raw bridge direction **v_bridge_raw = μ_B − μ_D**. Because that centroid difference mixes the consistent mechanism-reasoning direction with item-specific text and format noise, we **refine it with PCA**. For each layer we take the per-item differences **d_i = act_B_i − act_D_i**, stack and center them, and run SVD. The first principal component **PC1** is the unit direction of maximum variance across these differences; we flip its sign if needed so it aligns with the mean difference, then scale it to have the same norm as **v_bridge_raw** so **α** remains comparable. The norm is the **L2 (Euclidean) norm** of the vector in the residual stream at that layer (i.e. in R^{d_model}); it is not relative to another vector—it is simply ‖v_bridge_raw‖. So we set v_pca = PC1 · ‖v_bridge_raw‖ so that ‖v_pca‖ = ‖v_bridge_raw‖ and α has the same scale whether we use the raw or PCA-refined direction. This PCA-refined vector replaces **v_bridge** at that layer. We then pick a **STEER_LAYER** (e.g. by best **rel_signal** or **frac_positive**). **rel_signal** is ‖v_bridge‖ / ‖μ_D‖—the bridge direction’s norm relative to the D-centroid norm—so it is comparable across layers (later layers have larger raw norms; rel_signal avoids always choosing the deepest layer). **frac_positive** is the fraction of train items whose (B−D) difference projects positively onto v_bridge. We set **v_steer = v_bridge[STEER_LAYER]**—the PCA-refined direction at that layer.

The script saves **`steer_layer`** and **`v_steer`** in **`steering_vectors.pt`**. At inference (e.g. in **`run_aut_benchmark.py`**) we load that file, register a forward hook on the chosen layer that adds **α · v_steer** to the residual stream during generation (e.g. for the first *k* assistant tokens or all new tokens), and sweep **α** (e.g. 0.25, 0.5, 1.0, 1.5). The deployment prompt (e.g. CreativityPrism’s “Create a list of creative alternative uses… 5 words long”) can differ from the bridge-training prompt; the vector still nudges the model toward the mechanism-reasoning direction learned from the contrastive pairs.

---

## Quick reference

| Step | What |
|------|------|
| **Pairs** | One per item: same user prompt (A + “Return exactly 8 unconventional uses…”); D completion = 8 default uses, B completion = mechanism-only text. |
| **Activations** | Residual stream at selected layers, averaged over first 48 completion tokens, train pairs only. |
| **Vector** | **v_steer** = PCA-refined bridge at chosen layer: PC1 of per-item (B−D) diffs, scaled to ‖μ_B − μ_D‖. |
| **Inference** | Add **α · v_steer** at that layer during generation; α set by `--alpha` in `run_aut_inference.py` or `run_aut_benchmark.py`. |

Implementation: **`scripts/run_bridge_steering.py`** (uses `dlp.data`, `dlp.steering`, `dlp.models`).

---

## Running bridge steering (Llama 3.1 8B or any HF CausalLM)

Use the CLI script only. From `DLP/` with the project venv active:

```bash
python scripts/run_bridge_steering.py \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --abcd-data dataset/abcd_aut.json \
  --output-dir results/bridge_steering \
  [--use-pca] [--load-in-8bit]
```

**Run directory and skip-if-exists:** `--output-dir` is a *base* directory. The script creates a **setup-named subdir** (e.g. `llama_3.1_8b_instruct_aut_abcd_aut_w48_pca_a1b2c3d4`) so that:
- **Same config** → same subdir → if `steering_vectors.pt` already exists there, the script skips and does not re-run.
- **Different config** → different subdir → previous runs are never overwritten.

Output path is therefore **`<output-dir>/<setup_slug>/steering_vectors.pt`** (e.g. `results/bridge_steering/llama_3.1_8b_instruct_aut_abcd_aut_w48_pca_<hash>/steering_vectors.pt`). Then run the AUT benchmark with the **same** model name and that vectors file. The benchmark supports multiple **methods** (e.g. `baseline`, `steered`); only pass `--vectors` when using the `steered` method. You can run baseline only, steered only, or both:

```bash
# Baseline + steered (default when --vectors is set)
python scripts/run_aut_benchmark.py \
  --aut-data /path/to/aut_push_skipped.json \
  --vectors results/bridge_steering/<setup_slug>/steering_vectors.pt \
  --output-dir /path/to/output \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --method baseline --method steered

# Steered only
python scripts/run_aut_benchmark.py \
  --aut-data /path/to/aut_push_skipped.json \
  --vectors results/bridge_steering/<setup_slug>/steering_vectors.pt \
  --output-dir /path/to/output \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --method steered
```

Other methods (e.g. few-shot) can be added in `dlp.evaluation.methods` and wired via `--method`; only methods that need vectors require `--vectors`.
