"""
Steering vector computation: mean-difference baseline and optional PCA refinement.

Given activations collected from contrastive pairs (B = mechanism-mode,
D = listing-mode), compute per-layer bridge steering vectors (v_bridge)
and diagnostics.
"""

from __future__ import annotations

from typing import Any

import torch
import pandas as pd


class BridgeVectorComputer:
    """
    Holds activations from contrastive pairs and computes per-layer bridge
    vectors (centroids, v_bridge, stats) with optional PCA refinement.
    """

    def __init__(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        layers: list[int],
    ) -> None:
        self.acts_B = acts_B
        self.acts_D = acts_D
        self.layers = layers

    def compute(
        self,
    ) -> tuple[
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        pd.DataFrame,
    ]:
        """
        Compute per-layer centroids, v_bridge, and diagnostic stats.

        Returns:
            (centroids_B, centroids_D, v_bridge, stats_df)
        """
        centroids_B, centroids_D, stats_df = compute_bridge_vectors(
            self.acts_B, self.acts_D, self.layers
        )
        v_bridge = {L: centroids_B[L] - centroids_D[L] for L in self.layers}
        return centroids_B, centroids_D, v_bridge, stats_df

    def refine_pca(
        self,
        v_bridge: dict[int, torch.Tensor],
    ) -> tuple[dict[int, torch.Tensor], dict[int, dict]]:
        """Return PCA-refined v_bridge and pca_info. Does not mutate v_bridge."""
        return refine_with_pca(
            self.acts_B, self.acts_D, v_bridge, self.layers
        )

    @staticmethod
    def select_best_layer(
        stats_df: pd.DataFrame,
        frac_positive_threshold: float = 1.0,
    ) -> int:
        """Select best layer index by relative signal among fully-separating layers."""
        return select_best_layer(stats_df, frac_positive_threshold)


def compute_bridge_vectors(
    acts_B: dict[int, list[torch.Tensor]],
    acts_D: dict[int, list[torch.Tensor]],
    layers: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], pd.DataFrame]:
    """
    Compute per-layer steering vectors and diagnostics.

    For each layer L:
      centroids_B[L] = mean of acts_B[L]
      centroids_D[L] = mean of acts_D[L]
      v_bridge[L]    = centroids_B[L] - centroids_D[L]

    Also computes diagnostic stats: cosine similarity between centroids,
    norms, relative signal, and fraction of items where projection onto
    v_bridge is positive (separability).

    Args:
        acts_B: {layer: [tensor(d_model), ...]} for mechanism-mode condition.
        acts_D: {layer: [tensor(d_model), ...]} for listing-mode condition.
        layers: Layers to process.

    Returns:
        (centroids_B, centroids_D, stats_df)
        v_bridge is embedded in the centroid difference (centroids_B - centroids_D).
    """
    centroids_B: dict[int, torch.Tensor] = {}
    centroids_D: dict[int, torch.Tensor] = {}
    v_bridge: dict[int, torch.Tensor] = {}
    layer_stats: list[dict[str, Any]] = []

    for L in layers:
        stack_D = torch.stack(acts_D[L])  # (N, d_model)
        stack_B = torch.stack(acts_B[L])  # (N, d_model)

        mu_D = stack_D.mean(dim=0)
        mu_B = stack_B.mean(dim=0)
        v = mu_B - mu_D

        centroids_D[L] = mu_D
        centroids_B[L] = mu_B
        v_bridge[L] = v

        cos_sim = torch.nn.functional.cosine_similarity(
            mu_B.unsqueeze(0).float(), mu_D.unsqueeze(0).float()
        ).item()
        v_norm = v.float().norm().item()
        mu_D_norm = mu_D.float().norm().item()
        mu_B_norm = mu_B.float().norm().item()

        diffs = (stack_B - stack_D).float()
        v_unit = (v / (v.norm() + 1e-8)).float()
        projs = (diffs @ v_unit).numpy()

        rel_signal = v_norm / (mu_D_norm + 1e-8)
        cv = float(projs.std()) / (float(projs.mean()) + 1e-8)

        layer_stats.append({
            "layer": L,
            "cos_centroids": round(cos_sim, 4),
            "|v_bridge|": round(v_norm, 2),
            "|mu_D|": round(mu_D_norm, 2),
            "|mu_B|": round(mu_B_norm, 2),
            "rel_signal": round(rel_signal, 4),
            "CV": round(cv, 4),
            "proj_mean": round(float(projs.mean()), 2),
            "proj_std": round(float(projs.std()), 2),
            "proj_min": round(float(projs.min()), 2),
            "frac_positive": round(float((projs > 0).mean()), 3),
        })

    stats_df = pd.DataFrame(layer_stats)
    return centroids_B, centroids_D, stats_df


def refine_with_pca(
    acts_B: dict[int, list[torch.Tensor]],
    acts_D: dict[int, list[torch.Tensor]],
    v_bridge: dict[int, torch.Tensor],
    layers: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, dict]]:
    """
    Replace raw centroid-difference v_bridge with PCA-refined direction.

    PCA on the per-item differences {act_B_i - act_D_i} extracts the
    direction explaining the most variance. PC1 is sign-aligned to the
    mean difference and scaled to match the raw v_bridge norm.

    Args:
        acts_B, acts_D: Per-layer activation lists (same as in compute_bridge_vectors).
        v_bridge: Raw bridge vectors to be replaced (modified in-place copy returned).
        layers: Layers to process.

    Returns:
        (v_bridge_pca, pca_info)
        v_bridge_pca: {layer: tensor} with PCA-refined vectors.
        pca_info: {layer: {"var_explained_pc1", "cos_raw_vs_pca", "raw_norm"}}.
    """
    v_bridge_pca: dict[int, torch.Tensor] = {L: v_bridge[L].clone() for L in layers}
    pca_info: dict[int, dict] = {}

    for L in layers:
        stack_D = torch.stack(acts_D[L]).float()
        stack_B = torch.stack(acts_B[L]).float()
        diffs = stack_B - stack_D
        diffs_centered = diffs - diffs.mean(dim=0)

        U, S, Vt = torch.linalg.svd(diffs_centered, full_matrices=False)
        pc1 = Vt[0]
        mean_diff = diffs.mean(dim=0)
        if (pc1 @ mean_diff) < 0:
            pc1 = -pc1

        raw_norm = v_bridge[L].float().norm()
        v_pca = pc1 * raw_norm

        total_var = (S ** 2).sum().item()
        pc1_var = (S[0] ** 2).item()
        var_explained = pc1_var / total_var if total_var > 0 else 0.0

        cos_raw_pca = torch.nn.functional.cosine_similarity(
            mean_diff.unsqueeze(0), pc1.unsqueeze(0)
        ).item()

        pca_info[L] = {
            "var_explained_pc1": round(var_explained, 4),
            "cos_raw_vs_pca": round(cos_raw_pca, 4),
            "raw_norm": round(raw_norm.item(), 2),
        }
        v_bridge_pca[L] = v_pca.to(v_bridge[L].dtype)

    return v_bridge_pca, pca_info


def refine_with_pca_b_only(
    acts_B: dict[int, list[torch.Tensor]],
    v_bridge: dict[int, torch.Tensor],
    layers: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, dict]]:
    """
    PCA on the B activations only (no D). For use with b_only extractor.

    Centers the B cloud, takes PC1 as the main direction, sign-aligns to the
    centroid, and scales to the original centroid norm.
    """
    v_pca: dict[int, torch.Tensor] = {L: v_bridge[L].clone() for L in layers}
    pca_info: dict[int, dict] = {}

    for L in layers:
        stack_B = torch.stack(acts_B[L]).float()
        centroid = stack_B.mean(dim=0)
        centered = stack_B - centroid
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        pc1 = Vt[0]
        if (pc1 @ centroid) < 0:
            pc1 = -pc1
        raw_norm = centroid.float().norm()
        v_pca[L] = (pc1 * raw_norm).to(v_bridge[L].dtype)

        total_var = (S ** 2).sum().item()
        pc1_var = (S[0] ** 2).item() if S.numel() > 0 else 0.0
        var_explained = pc1_var / total_var if total_var > 0 else 0.0
        cos_raw = torch.nn.functional.cosine_similarity(
            centroid.unsqueeze(0), pc1.unsqueeze(0)
        ).item()
        pca_info[L] = {
            "var_explained_pc1": round(var_explained, 4),
            "cos_raw_vs_pca": round(cos_raw, 4),
            "raw_norm": round(raw_norm.item(), 2),
        }

    return v_pca, pca_info


def select_best_layer(
    stats_df: pd.DataFrame,
    frac_positive_threshold: float = 1.0,
) -> int:
    """
    Select the best layer for steering by relative signal among fully-separating layers.

    Picks the layer with highest |v_bridge| / |mu_D| (rel_signal) among those
    where frac_positive >= frac_positive_threshold. Falls back to all layers if none qualify.

    Returns:
        Best layer index (int).
    """
    candidates = stats_df[stats_df["frac_positive"] >= frac_positive_threshold]
    if candidates.empty:
        candidates = stats_df
    best_row = candidates.sort_values("rel_signal", ascending=False).iloc[0]
    return int(best_row["layer"])
