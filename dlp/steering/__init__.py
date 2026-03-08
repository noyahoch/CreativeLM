"""Bridge activation-steering utilities."""

from pathlib import Path
from typing import Any

from .cache import ResidualStreamCache
from .completion_sources import (
    CompletionSource,
    FixedCompletion,
    GeneratedCompletions,
    MultiBConcatenated,
    MultiBSeparate,
    SOURCE_REGISTRY,
    register_source,
)
from .extractors import (
    BOnlyExtractor,
    ClusteredExtractor,
    ExtractionResult,
    EXTRACTOR_REGISTRY,
    MeanDiffExtractor,
    MultiPCAExtractor,
    VectorExtractor,
    register_extractor,
)
from .hooks import SteeringHook, steered_generate, steered_generate_batch
from .vectors import (
    BridgeVectorComputer,
    compute_bridge_vectors,
    refine_with_pca,
    select_best_layer,
)

__all__ = [
    # Completion sources (--b-source axis)
    "CompletionSource",
    "FixedCompletion",
    "GeneratedCompletions",
    "MultiBSeparate",
    "MultiBConcatenated",
    "SOURCE_REGISTRY",
    "register_source",
    # Vector extractors (--method axis)
    "VectorExtractor",
    "ExtractionResult",
    "MeanDiffExtractor",
    "BOnlyExtractor",
    "ClusteredExtractor",
    "MultiPCAExtractor",
    "EXTRACTOR_REGISTRY",
    "register_extractor",
    # Original exports
    "BridgeVectorComputer",
    "ResidualStreamCache",
    "SteeringHook",
    "steered_generate",
    "steered_generate_batch",
    "compute_bridge_vectors",
    "load_steering_vectors",
    "refine_with_pca",
    "select_best_layer",
]


def load_steering_vectors(path: str | Path) -> tuple[int, Any]:
    """
    Load steering vectors saved by the DLP training pipeline.

    The file is a torch checkpoint with keys ``steer_layer`` (int) and
    ``v_steer`` (tensor of shape ``(d_model,)``).

    To tell how the vector was computed, inspect the checkpoint:
      - ``state["method"]``: "mean_diff", "b_only", "clustered", "multi_pca"
      - ``state["use_pca"]``: True if PCA refinement was applied (mean_diff or b_only)
      - ``state["b_source"]``: "fixed", "multi_b_separate", "multi_b_concat", "generated"

    Returns:
        (layer_idx, v_steer) ready to pass to ``steered_generate``.
    """
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state["steer_layer"]), state["v_steer"]
