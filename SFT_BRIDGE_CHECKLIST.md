# SFT Bridge Internalization (Experiment 3) — Checklist

## Before you start

- [ ] **Environment:** Use the DLP venv (e.g. `DLP/.venv`). Activate or select its kernel in Jupyter.
- [ ] **CUDA:** No real `nvcc` needed. The notebook creates a stub and sets `CUDA_HOME` so DeepSpeed/PyTorch don’t fail.
- [ ] **Data:** `DLP/dataset/abcd_aut.json` exists (20 ABCD items).

---

## Running the notebook

- [ ] **1. Run Cell 1 (imports + nvcc stub)**  
  - You should see: `nvcc stub OK (-V and --version); CUDA_HOME= ...` then `Imports OK.`  
  - If you see `ValueError: 'release' is not in list` or `FileNotFoundError: .../nvcc`, the stub didn’t run first — run Cell 1 again (or restart kernel and run Cell 1).

- [ ] **2. Run Cell 2 (load model)**  
  - Loads Qwen2-7B-Instruct. Wait until it finishes.

- [ ] **3. Run dataset cells (build SFT data)**  
  - Load ABCD data, build train/eval split, tokenize. No special checks.

- [ ] **4. Run baseline eval cells (optional)**  
  - Pre-SFT generation on held-out items. Good to have for later comparison.

- [ ] **5. Run LoRA setup cell**  
  - Wraps model with PEFT/LoRA. Check “Trainable parameters” printout.

- [ ] **6. Run Trainer config cell**  
  - Creates stub + sets `CUDA_HOME` again and patches `torch.utils.cpp_extension.CUDA_HOME`.  
  - Then `Trainer(...)` is created.  
  - If you get `FileNotFoundError` or `'release' is not in list`: **restart kernel**, run **Cell 1**, then run this cell again.

- [ ] **7. Run Train! cell**  
  - Runs SFT. Wait for 5 epochs to finish.

- [ ] **8. Run “Plot training loss” cell**  
  - Saves `sft_bridge_outputs/training_loss.png`.

- [ ] **9. Run post-SFT evaluation cells**  
  - Generation on held-out items, side-by-side comparison, novel-object test.

- [ ] **10. Run analysis + save cells**  
  - Quantitative metrics, plots, and saving results to `sft_bridge_outputs/`.

---

## If you skip ahead (e.g. “Run from Trainer cell”)

- [ ] At least run **Cell 1** once in this session so the nvcc stub exists and `CUDA_HOME` is set before any `torch`/transformers imports.
- [ ] Or: restart kernel and run **Cell 1**, then run the **Trainer config** cell (it recreates the stub and patches `CUDA_HOME`).

---

## Quick “is it working?” checks

| Step              | What to look for                                      |
|-------------------|--------------------------------------------------------|
| After Cell 1      | `nvcc stub OK` and `Imports OK.`                      |
| After Trainer()    | No `FileNotFoundError` for nvcc, no `'release'` error |
| After Train!      | “Training complete” and a final loss value           |
| After post-SFT    | Higher “mechanism presence” rate than baseline       |

---

## Optional

- [ ] **SFT + steering (ablation):** Run the “Combined SFT + Steering” cell only if `results/bridge_steering/<setup_slug>/steering_vectors.pt` exists (from Experiment 2).
- [ ] **Reproducibility:** `sft_bridge_outputs/experiment_config.json` records hyperparameters and paths.
