"""SFT training helpers for bridge internalization experiments."""

from .data_prep import (
    TaskType,
    build_contrastive_pair,
    build_sft_dataset,
    build_sft_example,
    format_user_prompt,
    format_user_prompt_aut,
    format_user_prompt_ps,
    format_target,
    format_target_aut,
    format_target_ps,
)
from .sft import SFTTrainer, default_lora_config, default_training_args, train

__all__ = [
    "TaskType",
    "SFTTrainer",
    "build_contrastive_pair",
    "build_sft_dataset",
    "build_sft_example",
    "format_user_prompt",
    "format_user_prompt_aut",
    "format_user_prompt_ps",
    "format_target",
    "format_target_aut",
    "format_target_ps",
    "default_lora_config",
    "default_training_args",
    "train",
]
