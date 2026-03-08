"""
Activation-steering hook: adds α × v_steer to the residual stream during generation.
"""

from __future__ import annotations

from typing import Any

import torch


class SteeringHook:
    """
    Hook that adds α * v_steer to the residual stream at layer L during generation.

    Modes:
      - "last_prompt_only": inject only at position last_prompt_pos on the first (prefill) pass.
      - "all_new_tokens": inject at every generated token + last prompt token on prefill.
      - "all": inject at every position on every forward pass.
      - "first_k_assistant_tokens": inject only for the first K decode steps,
         plus optionally at last prompt token on prefill.
    """

    def __init__(
        self,
        model: Any,
        layer_idx: int,
        v_steer: torch.Tensor,
        alpha: float,
        mode: str = "all_new_tokens",
        last_prompt_pos: int = -1,
        k_assist: int = 16,
        also_inject_last_prompt: bool = True,
    ) -> None:
        self.model = model
        self.layer_idx = layer_idx
        self.device = next(model.parameters()).device
        self.v_steer = v_steer.to(self.device).to(next(model.parameters()).dtype)
        self.alpha = float(alpha)
        self.mode = mode
        self.last_prompt_pos = int(last_prompt_pos)
        self.k_assist = int(k_assist)
        self.also_inject_last_prompt = bool(also_inject_last_prompt)

        self._hook: Any = None
        self._decode_step: int = 0

    def _hook_fn(self, module: Any, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        is_prefill = hidden.shape[1] > 1
        steering = self.alpha * self.v_steer  # (d_model,)

        if self.mode == "last_prompt_only":
            if is_prefill and self.last_prompt_pos >= 0:
                hidden = hidden.clone()
                hidden[:, self.last_prompt_pos, :] += steering

        elif self.mode == "all_new_tokens":
            hidden = hidden.clone()
            if is_prefill:
                if self.last_prompt_pos >= 0:
                    hidden[:, self.last_prompt_pos, :] += steering
            else:
                hidden[:, :, :] += steering

        elif self.mode == "all":
            hidden = hidden + steering

        elif self.mode == "first_k_assistant_tokens":
            hidden = hidden.clone()
            if is_prefill:
                if self.also_inject_last_prompt and self.last_prompt_pos >= 0:
                    hidden[:, self.last_prompt_pos, :] += steering
            else:
                self._decode_step += 1
                if self._decode_step <= self.k_assist:
                    hidden[:, :, :] += steering

        else:
            raise ValueError(f"Unknown SteeringHook mode: {self.mode!r}")

        if rest is not None:
            return (hidden,) + rest
        return hidden

    def register(self) -> "SteeringHook":
        """Register hook on the target layer."""
        self.remove()
        self._decode_step = 0
        self._hook = self.model.model.layers[self.layer_idx].register_forward_hook(
            self._hook_fn
        )
        return self

    def remove(self) -> None:
        """Remove the hook if registered."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self) -> "SteeringHook":
        return self.register()

    def __exit__(self, *_: Any) -> None:
        self.remove()


@torch.no_grad()
def steered_generate(
    loader: Any,
    prompt_msgs: list[dict],
    layer_idx: int,
    v_steer: torch.Tensor,
    alpha: float,
    mode: str = "all_new_tokens",
    max_new_tokens: int = 384,
    do_sample: bool = False,
    temperature: float = 1.0,
    k_assist: int = 16,
    also_inject_last_prompt: bool = True,
    **generate_kwargs: Any,
) -> str:
    """
    Run generation with activation steering applied via SteeringHook.

    Args:
        loader: HFLoader instance (or any object with .load() returning (model, tokenizer)).
        prompt_msgs: List of {"role": ..., "content": ...} messages.
        layer_idx: Layer at which to inject the steering vector.
        v_steer: Steering direction tensor.
        alpha: Steering strength multiplier.
        mode: One of "last_prompt_only", "all_new_tokens", "all", "first_k_assistant_tokens".
        max_new_tokens: Max tokens to generate.
        do_sample: Whether to use sampling.
        temperature: Sampling temperature.
        k_assist: Used when mode="first_k_assistant_tokens".
        also_inject_last_prompt: Also inject at last prompt token during prefill.

    Returns:
        Generated text string.
    """
    model, tokenizer = loader.load()

    text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    last_prompt_pos = inputs["input_ids"].shape[1] - 1

    hook = SteeringHook(
        model,
        layer_idx=layer_idx,
        v_steer=v_steer,
        alpha=alpha,
        mode=mode,
        last_prompt_pos=last_prompt_pos,
        k_assist=k_assist,
        also_inject_last_prompt=also_inject_last_prompt,
    )
    hook.register()
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )
    finally:
        hook.remove()

    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def steered_generate_batch(
    loader: Any,
    batch_msgs: list[list[dict]],
    layer_idx: int,
    v_steer: torch.Tensor,
    alpha: float,
    mode: str = "all_new_tokens",
    max_new_tokens: int = 384,
    do_sample: bool = False,
    temperature: float = 1.0,
    k_assist: int = 16,
    also_inject_last_prompt: bool = True,
    **generate_kwargs: Any,
) -> list[str]:
    """Batched steered generation using left-padding."""
    if len(batch_msgs) == 1:
        return [steered_generate(
            loader, batch_msgs[0], layer_idx, v_steer, alpha,
            mode=mode, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, k_assist=k_assist,
            also_inject_last_prompt=also_inject_last_prompt,
            **generate_kwargs,
        )]

    model, tokenizer = loader.load()

    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_msgs
    ]

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    finally:
        tokenizer.padding_side = orig_side

    prompt_len = inputs["input_ids"].shape[1]
    last_prompt_pos = prompt_len - 1

    hook = SteeringHook(
        model,
        layer_idx=layer_idx,
        v_steer=v_steer,
        alpha=alpha,
        mode=mode,
        last_prompt_pos=last_prompt_pos,
        k_assist=k_assist,
        also_inject_last_prompt=also_inject_last_prompt,
    )
    hook.register()
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )
    finally:
        hook.remove()

    results: list[str] = []
    for seq in out:
        generated = seq[prompt_len:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True))
    return results
