"""Qwen model loader using Hugging Face Transformers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenModelLoader:
    """Load Qwen (Qwen2/Qwen2.5/Qwen3) models and tokenizer from Hugging Face."""

    def __init__(
        self,
        model_name_or_path: str | Path = "Qwen/Qwen2-7B-Instruct",
        *,
        torch_dtype: torch.dtype | str = "auto",
        device_map: str | dict[str, Any] = "auto",
        trust_remote_code: bool = True,
        use_fast_download: bool = True,
        **model_kwargs: Any,
    ) -> None:
        """
        Args:
            model_name_or_path: Hugging Face model id (e.g. "Qwen/Qwen2-7B-Instruct")
                or path to a local directory with saved model/tokenizer.
            torch_dtype: Model dtype. Use "auto" for automatic selection, or
                torch.float16 / torch.bfloat16 for lower memory.
            device_map: How to map model to devices. "auto" uses accelerate to
                split across GPU/CPU as needed.
            trust_remote_code: Allow running custom code from the model repo.
                Required for Qwen models.
            use_fast_download: If True (default), use hf-transfer for much faster
                downloads. Set to False if you use a proxy or see download issues.
            **model_kwargs: Extra arguments passed to from_pretrained (e.g.
                revision, use_flash_attention_2).
        """
        self.model_name_or_path = str(model_name_or_path)
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.use_fast_download = use_fast_download
        self.model_kwargs = model_kwargs

        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    def load(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer. Caches them after first call.

        Returns:
            (model, tokenizer)
        """
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        if self.use_fast_download:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        dtype: torch.dtype | str = (
            self.torch_dtype
            if isinstance(self.torch_dtype, torch.dtype) or self.torch_dtype == "auto"
            else getattr(torch, str(self.torch_dtype), torch.float32)
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            **self.model_kwargs,
        )

        return self._model, self._tokenizer

    @property
    def model(self) -> AutoModelForCausalLM:
        """Model; loads on first access if not already loaded."""
        if self._model is None:
            self.load()
        assert self._model is not None
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Tokenizer; loads on first access if not already loaded."""
        if self._tokenizer is None:
            self.load()
        assert self._tokenizer is not None
        return self._tokenizer

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **generate_kwargs: Any,
    ) -> str:
        """
        Generate a reply given chat messages.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}.
            max_new_tokens: Maximum new tokens to generate.
            do_sample: Whether to sample; if False, greedy decoding.
            temperature: Sampling temperature (used if do_sample=True).
            top_p: Nucleus sampling (used if do_sample=True).
            **generate_kwargs: Passed to model.generate().

        Returns:
            Decoded response text (assistant reply only).
        """
        model, tokenizer = self.load()

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )

        # Decode only the new part
        generated = out[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        **generate_kwargs: Any,
    ) -> str:
        """
        Single prompt → model output. Easiest way to get a response.

        Args:
            user_prompt: The user's question or instruction.
            system_prompt: Optional system message (sets behavior/role).
            max_new_tokens: Maximum length of the reply.
            do_sample: If True, use temperature/top_p; if False, greedy.
            temperature: Sampling temperature (when do_sample=True).
            **generate_kwargs: Passed to generate().

        Returns:
            The model's reply as a string.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.generate(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **generate_kwargs,
        )
