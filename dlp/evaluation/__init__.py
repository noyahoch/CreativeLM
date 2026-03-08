"""
LLM-as-a-judge pipeline for creativity evaluation.

- AUT: generate alternative uses, rate 1-5 creativity, aggregate fluency + mean creativity.
- MacGyver: score problem-solving solutions with the 5-point additive quality rubric.
- Benchmark: run AUT with pluggable methods (baseline, steered, few-shot, etc.).
"""

from .benchmark import AUTBenchmarkRunner
from .generate import generate_aut_outputs, generate_uses_for_object, save_flat
from .judge import AUTJudge, MacGyverJudge, parse_macgyver_score
from .methods import (
    AUTInferenceMethod,
    BaselineMethod,
    SteeredMethod,
    FewShotMethod,
    TwoHopMethod,
    AbcdFrameworkMethod,
    available_methods,
    build_method,
)
from .pipeline import AUTPipeline, run_generate, run_judge_only, run_pipeline

__all__ = [
    "AUTBenchmarkRunner",
    "AUTInferenceMethod",
    "AUTJudge",
    "MacGyverJudge",
    "AUTPipeline",
    "AbcdFrameworkMethod",
    "BaselineMethod",
    "FewShotMethod",
    "SteeredMethod",
    "TwoHopMethod",
    "available_methods",
    "build_method",
    "generate_aut_outputs",
    "generate_uses_for_object",
    "parse_macgyver_score",
    "save_flat",
    "run_generate",
    "run_judge_only",
    "run_pipeline",
]
