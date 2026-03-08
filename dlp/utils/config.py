"""
Config dataclasses for DLP experiments.

Scripts and notebooks can build one config object and pass it into
SteeringVectorComputer, SFTTrainer, AUTPipeline, AUTBenchmarkRunner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SteeringConfig:
    """Config for bridge steering vector computation and injection."""

    layers: list[int] = field(default_factory=lambda: list(range(0, 28, 2)))
    use_pca: bool = True
    frac_positive_threshold: float = 1.0
    output_dir: str | Path = "results/bridge_steering"


@dataclass
class SFTConfig:
    """Config for LoRA SFT training (bridge internalization)."""

    task_type: str = "aut"  # "aut" | "ps"
    output_dir: str | Path = "results/sft_runs"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    num_epochs: int = 5
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])


@dataclass
class JudgeConfig:
    """Config for AUT generate + judge pipeline."""

    generate_model: str = "gpt-4o-mini"
    generate_temperature: float = 0.7
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.0
    num_uses: int = 10
    max_uses_per_item: int = 15


@dataclass
class BenchmarkConfig:
    """Config for CreativityPrism-style AUT benchmark with pluggable methods."""

    aut_data_path: str | Path = ""
    output_dir: str | Path = "results/aut_benchmark"
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    test_size: float = 1e10
    max_new_tokens: int = 512
    temperature: float = 0.75
    # Optional: used when running the default method list (no explicit --method)
    vectors_path: str | Path | None = None
    alpha: float = 1.0
    no_baseline: bool = False
    no_steered: bool = False
