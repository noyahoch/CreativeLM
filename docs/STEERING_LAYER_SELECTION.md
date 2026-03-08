# How the steering layer was chosen

The steering layer (e.g. layer 16 for a given run) is **not** fixed by hand. It is chosen automatically when building the steering vectors in `run_bridge_steering.py`, using a data-driven criterion so that the same procedure works for any model and dataset.

## Procedure

1. **Contrastive activations**  
   For each training pair we have two conditions: **D** (listing mode: model “completes” with the eight default uses) and **B** (mechanism mode: model “completes” with mechanism-only text). We run a forward pass for each condition and, at a set of **probed layers**, read the residual-stream activations over the first 48 completion tokens and average over those positions. Probed layers are `range(n_layers // 4, n_layers, 2)` (e.g. for a 32-layer model: 8, 10, 12, …, 30).

2. **Per-layer bridge direction**  
   At each layer \(L\) we compute:
   - \(\mu_D\) = mean of D-condition activations (over train pairs)
   - \(\mu_B\) = mean of B-condition activations
   - **v_bridge** = \(\mu_B - \mu_D\) (optionally refined with PCA: first PC of per-item differences, scaled to \(\|\mu_B - \mu_D\|\))

3. **Diagnostics per layer**  
   For each train item \(i\) we have \(\text{diff}_i = \text{act}_{B,i} - \text{act}_{D,i}\). We project these onto the unit bridge direction and compute:
   - **rel_signal** = \(\|\text{v\_bridge}\| / \|\mu_D\|\)  
     This normalizes by the D-centroid norm so that layers are comparable (later layers have larger raw norms; rel_signal avoids always choosing the deepest layer).
   - **frac_positive** = fraction of train items for which \(\text{diff}_i \cdot \hat{v}_{\text{bridge}} > 0\)  
     So we require that, for every train pair, the B activation lies in the “mechanism” direction relative to D.

4. **Layer selection**  
   We restrict to layers where **frac_positive ≥ 1.0** (all train items separate correctly in the B vs D direction). Among those, we pick the layer with **highest rel_signal**. That layer index is saved as `steer_layer` in `steering_vectors.pt`.

So **“layer 16”** (or 27, or whatever your checkpoint has) is simply the layer that satisfied full separability and had the strongest relative bridge signal for your model, data split, and probed set. Different runs (e.g. different model or train/eval split) can yield a different chosen layer.

## Inspecting the choice

To see the per-layer stats and which layer was chosen for a given checkpoint:

```bash
cd DLP && python scripts/inspect_steering_layers.py --vectors results/bridge_steering/<setup>/steering_vectors.pt
```

Optionally `--csv out.csv` writes the table to CSV. The script prints the chosen layer and, for every probed layer, `rel_signal`, `frac_positive`, and `rel_signal_vs_best`.

## Code references

- **Selection logic:** `dlp/steering/vectors.py` — `select_best_layer(stats_df, frac_positive_threshold=1.0)`
- **Stats computation:** same file — `compute_bridge_vectors()` (rel_signal, frac_positive)
- **Usage in pipeline:** `scripts/run_bridge_steering.py` — after computing `v_bridge` and `stats_df`, calls `steer_layer = select_best_layer(stats_df)` and saves it in the checkpoint
