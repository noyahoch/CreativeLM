"""
Vector extractors: control how collected activations are turned into
per-layer steering vectors.

This is one of two composable strategy axes in the steering framework:
  --b-source  (CompletionSource)  x  --method  (VectorExtractor)

Any source can be combined with any extractor.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch

from .vectors import (
    BridgeVectorComputer,
    refine_with_pca,
    refine_with_pca_b_only,
)


EXTRACTOR_REGISTRY: dict[str, type[VectorExtractor]] = {}


def register_extractor(cls: type[VectorExtractor]) -> type[VectorExtractor]:
    """Class decorator that adds a VectorExtractor subclass to the registry."""
    EXTRACTOR_REGISTRY[cls.name] = cls
    return cls


@dataclass
class ExtractionResult:
    """Standardised output from any VectorExtractor."""

    v_bridge: dict[int, torch.Tensor]
    stats_df: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorExtractor(ABC):
    """
    Base class for steering-vector extraction methods.

    Subclasses implement ``compute_vectors`` which receives the already-collected
    per-layer activation lists (acts_B, acts_D) and returns an
    ``ExtractionResult``.

    Activation *collection* is handled by the script's collection loop together
    with a ``CompletionSource``; the extractor is only responsible for the
    aggregation / computation stage.
    """

    name: str  # registry key, set on subclass

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        """Compute per-layer steering vectors from collected activations.

        Args:
            acts_B: ``{layer: [tensor(d_model), ...]}`` B-condition activations.
            acts_D: ``{layer: [tensor(d_model), ...]}`` D-condition activations.
            probe_layers: Layer indices present in both dicts.

        Returns:
            An ``ExtractionResult`` with ``v_bridge``, ``stats_df``, and
            optional ``metadata``.
        """
        ...

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add method-specific CLI arguments.  Override in subclasses."""


# --------------------------------------------------------------------------- #
# Concrete extractors
# --------------------------------------------------------------------------- #


@register_extractor
class MeanDiffExtractor(VectorExtractor):
    """Centroid difference (B - D) with optional single-component PCA refinement.

    This is the original method from the bridge-steering pipeline.  It delegates
    to ``BridgeVectorComputer`` and ``refine_with_pca`` in ``vectors.py``.
    """

    name = "mean_diff"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--use-pca",
            action="store_true",
            help="Refine v_bridge with PCA (PC1 scaled to raw norm)",
        )

    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        computer = BridgeVectorComputer(acts_B, acts_D, probe_layers)
        centroids_B, centroids_D, v_bridge, stats_df = computer.compute()

        pca_info = None
        if self.config.get("use_pca"):
            v_bridge, pca_info = refine_with_pca(
                acts_B, acts_D, v_bridge, probe_layers
            )

        return ExtractionResult(
            v_bridge=v_bridge,
            stats_df=stats_df,
            metadata={
                "centroids_B": centroids_B,
                "centroids_D": centroids_D,
                "pca_info": pca_info,
            },
        )


@register_extractor
class BOnlyExtractor(VectorExtractor):
    """Use only the B centroid as the steering vector (no subtraction of D).

    v_bridge[L] = mean(acts_B[L]), optionally refined with PCA on the B cloud
    (--use-pca).  Avoids subtracting D, which may remove diversity-related
    signal.  Use with multi_b_separate to steer toward a diverse creative
    centroid.

    The vector is scaled to the same L2 norm as (B - D) so that alpha is
    comparable to mean_diff: same alpha gives similar steering strength.
    Without this, B_only would have much larger norm and alpha > 0.5 would
    overshoot (repetitive or degenerate outputs).
    """

    name = "b_only"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--use-pca",
            action="store_true",
            help="Refine B centroid with PCA on B activations (PC1, BOnlyExtractor)",
        )
        parser.add_argument(
            "--no-scale-to-diff",
            action="store_true",
            help="Do not scale B_only vector to (B-D) norm (alpha not comparable to mean_diff)",
        )

    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        v_bridge: dict[int, torch.Tensor] = {}
        layer_stats: list[dict[str, Any]] = []
        scale_to_diff = not self.config.get("no_scale_to_diff", False)
        target_norms: dict[int, float] = {}

        for L in probe_layers:
            stack_B = torch.stack(acts_B[L])
            stack_D = torch.stack(acts_D[L])
            centroid_B = stack_B.mean(dim=0).float()
            centroid_D = stack_D.mean(dim=0).float()
            norm_B = centroid_B.norm().item()
            diff = centroid_B - centroid_D
            norm_diff = diff.norm().item()
            if scale_to_diff and norm_diff > 1e-8:
                target_norms[L] = norm_diff

            if scale_to_diff and norm_B > 1e-8 and norm_diff > 1e-8:
                # Same direction as B, magnitude of (B-D) so alpha is comparable to mean_diff
                v_bridge[L] = (centroid_B * (norm_diff / norm_B)).to(acts_B[L][0].dtype)
                rel_signal = norm_diff / (centroid_D.norm().item() + 1e-8)
            else:
                v_bridge[L] = centroid_B.to(acts_B[L][0].dtype)
                rel_signal = norm_B

            layer_stats.append({
                "layer": L,
                "rel_signal": round(rel_signal, 4),
                "frac_positive": 1.0,
            })

        stats_df = pd.DataFrame(layer_stats)
        centroids_D = {L: torch.stack(acts_D[L]).mean(dim=0) for L in probe_layers}

        pca_info = None
        if self.config.get("use_pca"):
            v_bridge, pca_info = refine_with_pca_b_only(
                acts_B, v_bridge, probe_layers
            )
        # Re-apply scale-to-diff after PCA so final vectors have (B-D) norm
        if scale_to_diff and target_norms:
            for L in probe_layers:
                if L in target_norms:
                    n = v_bridge[L].float().norm().item()
                    if n > 1e-8:
                        v_bridge[L] = (v_bridge[L] * (target_norms[L] / n)).to(v_bridge[L].dtype)

        return ExtractionResult(
            v_bridge=v_bridge,
            stats_df=stats_df,
            metadata={
                "centroids_B": {L: v_bridge[L].clone() for L in probe_layers},
                "centroids_D": centroids_D,
                "b_only": True,
                "b_only_scaled_to_diff": scale_to_diff,
                "pca_info": pca_info,
            },
        )


@register_extractor
class NegDExtractor(VectorExtractor):
    """Subtract D only: steer *away* from listing mode without any B signal.

    v_bridge[L] = -centroid_D

    Scaled to ||B - D|| norm by default so alpha is comparable to mean_diff.
    This tests whether simply moving away from D is enough to produce
    creative outputs, without needing B at all.
    """

    name = "neg_d"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--no-scale-to-diff",
            action="store_true",
            help="Do not scale neg_d vector to (B-D) norm (alpha not comparable to mean_diff)",
        )

    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        v_bridge: dict[int, torch.Tensor] = {}
        layer_stats: list[dict[str, Any]] = []
        scale_to_diff = not self.config.get("no_scale_to_diff", False)

        for L in probe_layers:
            stack_B = torch.stack(acts_B[L]).float()
            stack_D = torch.stack(acts_D[L]).float()
            centroid_B = stack_B.mean(dim=0)
            centroid_D = stack_D.mean(dim=0)
            neg_d = -centroid_D

            norm_d = centroid_D.norm().item()
            diff = centroid_B - centroid_D
            norm_diff = diff.norm().item()

            if scale_to_diff and norm_d > 1e-8 and norm_diff > 1e-8:
                v_bridge[L] = (neg_d * (norm_diff / norm_d)).to(acts_D[L][0].dtype)
                rel_signal = norm_diff / (norm_d + 1e-8)
            else:
                v_bridge[L] = neg_d.to(acts_D[L][0].dtype)
                rel_signal = norm_d

            layer_stats.append({
                "layer": L,
                "rel_signal": round(rel_signal, 4),
                "frac_positive": 1.0,
            })

        stats_df = pd.DataFrame(layer_stats)

        return ExtractionResult(
            v_bridge=v_bridge,
            stats_df=stats_df,
            metadata={
                "centroids_B": {L: torch.stack(acts_B[L]).mean(dim=0) for L in probe_layers},
                "centroids_D": {L: torch.stack(acts_D[L]).mean(dim=0) for L in probe_layers},
                "neg_d": True,
            },
        )


@register_extractor
class BPerpExtractor(VectorExtractor):
    """B orthogonal to D: project out the D component from B.

    v_bridge[L] = B_perp = centroid_B - proj_D(centroid_B)
                = centroid_B - (centroid_B · d_hat) * d_hat

    where d_hat = centroid_D / ||centroid_D||.

    This keeps only the part of B that is orthogonal to D — the
    mechanism-reasoning signal that is NOT shared with listing mode.
    Unlike mean_diff (B - D), this removes the D direction entirely
    rather than subtracting the D magnitude along D.

    Scaled to ||B - D|| norm so alpha is comparable to mean_diff.
    """

    name = "b_perp"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--use-pca",
            action="store_true",
            help="Refine B_perp with PCA on per-item B_perp vectors",
        )
        parser.add_argument(
            "--no-scale-to-diff",
            action="store_true",
            help="Do not scale B_perp to (B-D) norm (alpha not comparable to mean_diff)",
        )

    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        v_bridge: dict[int, torch.Tensor] = {}
        layer_stats: list[dict[str, Any]] = []
        scale_to_diff = not self.config.get("no_scale_to_diff", False)

        for L in probe_layers:
            stack_B = torch.stack(acts_B[L]).float()
            stack_D = torch.stack(acts_D[L]).float()
            centroid_B = stack_B.mean(dim=0)
            centroid_D = stack_D.mean(dim=0)

            d_norm_sq = centroid_D.dot(centroid_D)
            if d_norm_sq > 1e-16:
                proj_D_B = (centroid_B.dot(centroid_D) / d_norm_sq) * centroid_D
            else:
                proj_D_B = torch.zeros_like(centroid_B)
            b_perp = centroid_B - proj_D_B

            norm_perp = b_perp.norm().item()
            diff = centroid_B - centroid_D
            norm_diff = diff.norm().item()

            if scale_to_diff and norm_perp > 1e-8 and norm_diff > 1e-8:
                v_bridge[L] = (b_perp * (norm_diff / norm_perp)).to(acts_B[L][0].dtype)
                rel_signal = norm_diff / (centroid_D.norm().item() + 1e-8)
            else:
                v_bridge[L] = b_perp.to(acts_B[L][0].dtype)
                rel_signal = norm_perp / (centroid_D.norm().item() + 1e-8)

            cos_perp_diff = torch.nn.functional.cosine_similarity(
                b_perp.unsqueeze(0), diff.unsqueeze(0)
            ).item()

            layer_stats.append({
                "layer": L,
                "rel_signal": round(rel_signal, 4),
                "frac_positive": 1.0,
                "|b_perp|": round(norm_perp, 2),
                "|b-d|": round(norm_diff, 2),
                "cos_perp_vs_diff": round(cos_perp_diff, 4),
            })

        stats_df = pd.DataFrame(layer_stats)

        pca_info = None
        if self.config.get("use_pca"):
            acts_B_perp: dict[int, list[torch.Tensor]] = {}
            for L in probe_layers:
                stack_B = torch.stack(acts_B[L]).float()
                stack_D = torch.stack(acts_D[L]).float()
                centroid_D = stack_D.mean(dim=0)
                d_norm_sq = centroid_D.dot(centroid_D)
                perps = []
                for b_vec in stack_B:
                    if d_norm_sq > 1e-16:
                        proj = (b_vec.dot(centroid_D) / d_norm_sq) * centroid_D
                    else:
                        proj = torch.zeros_like(b_vec)
                    perps.append(b_vec - proj)
                acts_B_perp[L] = perps
            v_bridge, pca_info = refine_with_pca_b_only(
                acts_B_perp, v_bridge, probe_layers
            )
            if scale_to_diff:
                for L in probe_layers:
                    stack_B = torch.stack(acts_B[L]).float()
                    stack_D = torch.stack(acts_D[L]).float()
                    diff = stack_B.mean(dim=0) - stack_D.mean(dim=0)
                    norm_diff = diff.norm().item()
                    n = v_bridge[L].float().norm().item()
                    if n > 1e-8 and norm_diff > 1e-8:
                        v_bridge[L] = (v_bridge[L] * (norm_diff / n)).to(v_bridge[L].dtype)

        return ExtractionResult(
            v_bridge=v_bridge,
            stats_df=stats_df,
            metadata={
                "centroids_B": {L: torch.stack(acts_B[L]).mean(dim=0) for L in probe_layers},
                "centroids_D": {L: torch.stack(acts_D[L]).mean(dim=0) for L in probe_layers},
                "b_perp": True,
                "pca_info": pca_info,
            },
        )


@register_extractor
class ClusteredExtractor(VectorExtractor):
    """Cluster per-item difference vectors, then combine cluster centroids.

    Idea: the B-D differences may contain multiple creative sub-strategies.
    Clustering reveals them; the final steering vector is a (weighted)
    combination of cluster centroids.

    The combined vector is scaled to ``||mean_diff||`` so alpha is comparable.
    """

    name = "clustered"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--n-clusters",
            type=int,
            default=3,
            help="Number of clusters for the diff vectors (ClusteredExtractor)",
        )
        parser.add_argument(
            "--cluster-weighting",
            choices=["uniform", "size", "variance"],
            default="uniform",
            help="How to weight cluster centroids when combining (ClusteredExtractor)",
        )

    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        from sklearn.cluster import KMeans

        n_clusters = self.config.get("n_clusters", 3)
        weighting = self.config.get("cluster_weighting", "uniform")

        computer = BridgeVectorComputer(acts_B, acts_D, probe_layers)
        centroids_B, centroids_D, v_bridge_raw, stats_df = computer.compute()

        v_bridge: dict[int, torch.Tensor] = {}
        cluster_info: dict[int, dict] = {}

        for L in probe_layers:
            stack_B = torch.stack(acts_B[L]).float()
            stack_D = torch.stack(acts_D[L]).float()
            diffs = stack_B - stack_D  # (N, d_model)

            k = min(n_clusters, diffs.shape[0])
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(diffs.numpy())

            cluster_centroids = []
            weights = []
            for c in range(k):
                mask = labels == c
                c_diffs = diffs[mask]
                centroid = c_diffs.mean(dim=0)
                cluster_centroids.append(centroid)

                if weighting == "uniform":
                    weights.append(1.0)
                elif weighting == "size":
                    weights.append(float(mask.sum()))
                elif weighting == "variance":
                    weights.append(float(c_diffs.var(dim=0).sum()))

            w = torch.tensor(weights)
            w = w / (w.sum() + 1e-8)

            combined = torch.zeros_like(cluster_centroids[0])
            for centroid, wi in zip(cluster_centroids, w):
                combined += wi * centroid

            raw_norm = v_bridge_raw[L].float().norm().item()
            combined_norm = combined.norm().item()
            if combined_norm > 1e-8 and raw_norm > 1e-8:
                combined = combined * (raw_norm / combined_norm)

            v_bridge[L] = combined.to(acts_B[L][0].dtype)

            cos_vs_mean = torch.nn.functional.cosine_similarity(
                combined.unsqueeze(0),
                v_bridge_raw[L].float().unsqueeze(0),
            ).item()

            cluster_info[L] = {
                "n_clusters": k,
                "weighting": weighting,
                "cluster_sizes": [int((labels == c).sum()) for c in range(k)],
                "weights": [round(float(wi), 4) for wi in w],
                "cos_clustered_vs_mean_diff": round(cos_vs_mean, 4),
            }

        return ExtractionResult(
            v_bridge=v_bridge,
            stats_df=stats_df,
            metadata={
                "centroids_B": centroids_B,
                "centroids_D": centroids_D,
                "cluster_info": cluster_info,
            },
        )


@register_extractor
class MultiPCAExtractor(VectorExtractor):
    """Use top-k PCA components instead of only PC1.

    The current PCA refinement takes only the first principal component of
    the per-item diffs.  This extractor keeps the top-k components and
    combines them into a single steering vector.

    Combination modes:
      - weighted_variance: variance-weighted sum of sign-aligned PCs → single vector.
      - separate: store each PC as a separate entry in v_bridge (keys become
        (layer, pc_idx) tuples).  Useful for analysis but the steering pipeline
        only uses the ``v_steer`` / ``steer_layer`` top-level keys, so the
        combined vector is still saved as v_steer.

    The final vector is scaled to ``||mean_diff||`` so alpha is comparable.
    """

    name = "multi_pca"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--n-components",
            type=int,
            default=3,
            help="Number of PCA components to keep (MultiPCAExtractor)",
        )
        parser.add_argument(
            "--pca-combination",
            choices=["weighted_variance", "separate"],
            default="weighted_variance",
            help=(
                "How to combine PCA components: "
                "'weighted_variance' = variance-weighted sum into single vector, "
                "'separate' = store per-PC vectors in metadata (v_bridge still gets the combined) "
                "(MultiPCAExtractor)"
            ),
        )

    def compute_vectors(
        self,
        acts_B: dict[int, list[torch.Tensor]],
        acts_D: dict[int, list[torch.Tensor]],
        probe_layers: list[int],
    ) -> ExtractionResult:
        n_components = self.config.get("n_components", 3)
        combination = self.config.get("pca_combination", "weighted_variance")

        computer = BridgeVectorComputer(acts_B, acts_D, probe_layers)
        centroids_B, centroids_D, v_bridge_raw, stats_df = computer.compute()

        v_bridge: dict[int, torch.Tensor] = {}
        pca_info: dict[int, dict] = {}

        for L in probe_layers:
            stack_B = torch.stack(acts_B[L]).float()
            stack_D = torch.stack(acts_D[L]).float()
            diffs = stack_B - stack_D  # (N, d_model)
            mean_diff = diffs.mean(dim=0)
            diffs_centered = diffs - mean_diff

            U, S, Vt = torch.linalg.svd(diffs_centered, full_matrices=False)
            k = min(n_components, Vt.shape[0])
            pcs = Vt[:k]        # (k, d_model)
            variances = S[:k] ** 2

            # Sign-align each PC to mean_diff
            for i in range(k):
                if pcs[i] @ mean_diff < 0:
                    pcs[i] = -pcs[i]

            total_var = (S ** 2).sum().item()
            var_explained = [round((variances[i] / total_var).item(), 4) for i in range(k)]
            cos_with_mean = [
                round(torch.nn.functional.cosine_similarity(
                    pcs[i].unsqueeze(0), mean_diff.unsqueeze(0)
                ).item(), 4)
                for i in range(k)
            ]

            # Combine: variance-weighted sum
            weights = variances / (variances.sum() + 1e-8)
            combined = (weights.unsqueeze(1) * pcs).sum(dim=0)  # (d_model,)

            # Scale to mean_diff norm
            raw_norm = v_bridge_raw[L].float().norm().item()
            combined_norm = combined.norm().item()
            if combined_norm > 1e-8 and raw_norm > 1e-8:
                combined = combined * (raw_norm / combined_norm)

            v_bridge[L] = combined.to(acts_B[L][0].dtype)

            layer_info: dict[str, Any] = {
                "n_components": k,
                "var_explained": var_explained,
                "cos_pc_vs_mean_diff": cos_with_mean,
                "weights": [round(w.item(), 4) for w in weights],
            }

            if combination == "separate":
                pc_vectors = {}
                for i in range(k):
                    pc_scaled = pcs[i] * raw_norm
                    pc_vectors[i] = pc_scaled.to(acts_B[L][0].dtype)
                layer_info["pc_vectors"] = pc_vectors

            pca_info[L] = layer_info

        return ExtractionResult(
            v_bridge=v_bridge,
            stats_df=stats_df,
            metadata={
                "centroids_B": centroids_B,
                "centroids_D": centroids_D,
                "pca_info": pca_info,
            },
        )
