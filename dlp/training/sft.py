"""
LoRA SFT training helpers for DLP experiments.

Provides default LoRA config, training arguments, and a high-level train() function
that wraps HuggingFace Trainer. The notebook only needs to call train() with its
dataset, model, and tokenizer.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# NVCC stub (required when running inside some container environments where
# DeepSpeed tries to locate nvcc but CUDA_HOME is not set).
# ---------------------------------------------------------------------------

def _patch_nvcc_stub() -> None:
    """Create a fake nvcc in a temp dir and set CUDA_HOME if not already set."""
    if os.environ.get("CUDA_HOME"):
        return
    os.environ.setdefault("BNB_CUDA_VERSION", "121")
    fake_cuda = os.path.join(tempfile.gettempdir(), "fake_cuda_nvcc_stub")
    fake_bin = os.path.join(fake_cuda, "bin")
    os.makedirs(fake_bin, exist_ok=True)
    nvcc = os.path.join(fake_bin, "nvcc")
    if not os.path.exists(nvcc):
        with open(nvcc, "w") as f:
            f.write('#!/bin/sh\nV="Cuda compilation tools, release 12.1, V12.1.0"\n')
            f.write('case "$1" in -V) echo "$V" ;; --version) echo "$V" ;; esac\nexit 0\n')
        os.chmod(nvcc, 0o755)
    os.environ["CUDA_HOME"] = fake_cuda


def default_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """
    Return a LoraConfig with sensible defaults for Qwen2-7B SFT.

    Args:
        r: LoRA rank.
        lora_alpha: LoRA scaling (typically 2*r).
        lora_dropout: Dropout probability.
        target_modules: Which projection layers to adapt; defaults to q/k/v/o_proj.

    Returns:
        peft.LoraConfig instance.
    """
    from peft import LoraConfig
    from peft import TaskType as PeftTaskType

    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    return LoraConfig(
        task_type=PeftTaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )


def default_training_args(
    output_dir: str | Path,
    num_train_epochs: int = 5,
    learning_rate: float = 1e-5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
) -> Any:
    """
    Return TrainingArguments configured for small-dataset LoRA SFT.

    Args:
        output_dir: Where to save checkpoints.
        num_train_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        per_device_train_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Effective batch = batch_size * this.

    Returns:
        transformers.TrainingArguments instance.
    """
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=0,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
        dataloader_pin_memory=False,
    )


def tokenize_sft_dataset(
    examples: list[dict],
    tokenizer: Any,
    max_length: int = 1024,
) -> Any:
    """
    Tokenize SFT examples for causal LM training with loss masking.

    Each example must have "messages" (chat format list of dicts).
    Only assistant tokens are used for loss (prompt tokens are masked with -100).

    Returns:
        A datasets.Dataset with input_ids, attention_mask, labels.
    """
    import torch
    from datasets import Dataset

    def _encode(example: dict) -> dict:
        msgs = example["messages"]
        # Full sequence
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        # Prompt only (without assistant turn)
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )

        full_ids = tokenizer(
            full_text, truncation=True, max_length=max_length, return_tensors="pt"
        )["input_ids"][0]
        prompt_len = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

        labels = full_ids.clone()
        labels[:prompt_len] = -100  # mask prompt tokens from loss

        return {
            "input_ids": full_ids.tolist(),
            "attention_mask": [1] * len(full_ids),
            "labels": labels.tolist(),
        }

    records = [_encode(ex) for ex in examples]
    return Dataset.from_list(records)


class SFTTrainer:
    """
    LoRA SFT trainer: wraps model, tokenizer, examples, and runs HuggingFace Trainer.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_examples: list[dict],
        output_dir: str | Path,
        lora_config: Any | None = None,
        training_args: Any | None = None,
        max_length: int = 1024,
    ) -> None:
        _patch_nvcc_stub()
        self.model = model
        self.tokenizer = tokenizer
        self.train_examples = train_examples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.lora_config = lora_config or default_lora_config()
        self.training_args = training_args or default_training_args(
            self.output_dir / "checkpoints"
        )
        self._peft_model: Any = None
        self._train_result: Any = None

    @staticmethod
    def get_default_lora_config(
        r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> Any:
        """Return default LoraConfig."""
        return default_lora_config(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )

    @staticmethod
    def get_default_training_args(
        output_dir: str | Path,
        num_train_epochs: int = 5,
        learning_rate: float = 1e-5,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
    ) -> Any:
        """Return default TrainingArguments."""
        return default_training_args(
            output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    def prepare_dataset(self) -> Any:
        """Tokenize train_examples and return HF Dataset."""
        return tokenize_sft_dataset(
            self.train_examples, self.tokenizer, max_length=self.max_length
        )

    def train(self) -> dict[str, Any]:
        """Run LoRA training. Returns dict with train_result, peft_model, output_dir, etc."""
        from peft import get_peft_model
        from transformers import DataCollatorForSeq2Seq, Trainer

        self.model.enable_input_require_grads()
        self._peft_model = get_peft_model(self.model, self.lora_config)
        trainable, total = self._peft_model.get_nb_trainable_parameters()
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        train_dataset = self.prepare_dataset()
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, padding=True, return_tensors="pt"
        )
        trainer = Trainer(
            model=self._peft_model,
            args=self.training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        self._train_result = trainer.train()
        return {
            "train_result": self._train_result,
            "output_dir": str(self.output_dir),
            "lora_config": self.lora_config,
            "training_args": self.training_args,
            "peft_model": self._peft_model,
        }

    def save_config(self, config: dict[str, Any]) -> Path:
        """Write config dict as JSON to output_dir."""
        path = self.output_dir / "run_config.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        return path


def train(
    model: Any,
    tokenizer: Any,
    train_examples: list[dict],
    output_dir: str | Path,
    lora_config: Any | None = None,
    training_args: Any | None = None,
    max_length: int = 1024,
) -> dict[str, Any]:
    """
    High-level SFT training entry point (backward-compatible wrapper).

    Builds SFTTrainer and calls train().
    """
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_examples,
        output_dir=output_dir,
        lora_config=lora_config,
        training_args=training_args,
        max_length=max_length,
    )
    return trainer.train()
