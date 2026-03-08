"""
AUT inference methods: pluggable ways to produce model outputs.

Each method has a ``slug`` (used for output dirs / column names) and a
``generate(loader, messages, ...)`` that returns raw text.

Available methods
-----------------
* ``BaselineMethod``   – plain generation, no steering or extra prompting.
* ``SteeredMethod``    – activation-steered generation (requires steering_vectors.pt).
* ``FewShotMethod``    – prepend N mechanism examples from abcd_aut.json to the prompt.
* ``TwoHopMethod``     – first generate a mechanism (hop 1), then uses (hop 2).
* ``AbcdFrameworkMethod`` – prompt with the full ABCD framework (A/B/C/D framing).

Adding a new method: implement the ``AUTInferenceMethod`` Protocol, give it a unique
``slug``, and register it in ``run_aut_benchmark.py``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Protocol

import torch

from dlp.steering import steered_generate, steered_generate_batch


# ── Protocol ──────────────────────────────────────────────────────────────────

class AUTInferenceMethod(Protocol):
    """Protocol for an AUT inference method: slug + generate(loader, messages) -> str."""

    slug: str
    supports_batch: bool

    def generate(
        self,
        loader: Any,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str: ...

    def generate_batch(
        self,
        loader: Any,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> list[str]: ...


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_abcd_aut(abcd_path: str | Path) -> list[dict]:
    with open(abcd_path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("data", [])


# ── Methods ───────────────────────────────────────────────────────────────────

class BaselineMethod:
    """Plain generation with no steering or extra prompting."""

    slug = "baseline"
    supports_batch = True

    def generate(
        self,
        loader: Any,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str:
        return loader.generate(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )

    def generate_batch(
        self,
        loader: Any,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        return loader.generate_batch(
            batch_messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )


class SteeredMethod:
    """Generation with activation steering from a saved steering_vectors.pt.

    Args:
        vectors_path: Path to the saved .pt checkpoint.
        alpha: Steering strength multiplier.
        layer_idx: Override which layer to steer at. If None, uses the
                   auto-selected ``steer_layer`` from the checkpoint.
                   The checkpoint's ``v_bridge`` dict holds vectors for all
                   probed layers, so you can pick any of them.
        steer_mode: Hook mode — "all_new_tokens" (default), "first_k_assistant_tokens",
                    "last_prompt_only", or "all".
        k_assist: Number of initial assistant tokens to steer when
                  steer_mode="first_k_assistant_tokens".
    """

    slug = "steered"
    supports_batch = True

    def __init__(
        self,
        vectors_path: str | Path,
        alpha: float = 1.0,
        layer_idx: int | None = None,
        steer_mode: str = "all_new_tokens",
        k_assist: int = 16,
    ) -> None:
        self.vectors_path = Path(vectors_path)
        self.alpha = alpha
        self._layer_override = layer_idx
        self._state: tuple[int, torch.Tensor] | None = None
        self.steer_mode = steer_mode
        self.k_assist = k_assist

    def _ensure_state(self) -> tuple[int, torch.Tensor]:
        if self._state is None:
            state = torch.load(
                self.vectors_path, map_location="cpu", weights_only=False
            )
            if self._layer_override is not None:
                v_bridge = state.get("v_bridge", {})
                if self._layer_override not in v_bridge:
                    available = sorted(v_bridge.keys()) if v_bridge else []
                    raise ValueError(
                        f"Layer {self._layer_override} not in checkpoint. "
                        f"Available layers: {available}"
                    )
                self._state = (self._layer_override, v_bridge[self._layer_override])
            else:
                self._state = (int(state["steer_layer"]), state["v_steer"])
        return self._state

    def generate(
        self,
        loader: Any,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str:
        layer_idx, v_steer = self._ensure_state()
        return steered_generate(
            loader,
            messages,
            layer_idx=layer_idx,
            v_steer=v_steer,
            alpha=self.alpha,
            mode=self.steer_mode,
            k_assist=self.k_assist,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )

    def generate_batch(
        self,
        loader: Any,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        layer_idx, v_steer = self._ensure_state()
        return steered_generate_batch(
            loader,
            batch_messages,
            layer_idx=layer_idx,
            v_steer=v_steer,
            alpha=self.alpha,
            mode=self.steer_mode,
            k_assist=self.k_assist,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )


class FewShotMethod:
    """
    Prepend N mechanism examples from abcd_aut.json to the user prompt.

    Each example shows: object task → mechanism type/rule → 8 uses.
    This teaches the model the "mechanism first" reasoning style.

    Args:
        abcd_path:  Path to abcd_aut.json (training examples).
        n_shots:    Number of examples to prepend (default 2).
        seed:       Random seed for example selection.
        train_ids:  Optional set of item IDs to draw examples from (avoids leaking
                    eval items). If None, uses all items.
    """

    slug = "fewshot"
    supports_batch = True

    def __init__(
        self,
        abcd_path: str | Path,
        n_shots: int = 2,
        seed: int = 42,
        train_ids: set[str] | None = None,
    ) -> None:
        self.n_shots = n_shots
        self.seed = seed
        all_items = _load_abcd_aut(abcd_path)
        self._pool = [it for it in all_items if train_ids is None or it["id"] in train_ids]

    def _build_system(self) -> str:
        rng = random.Random(self.seed)
        examples = rng.sample(self._pool, min(self.n_shots, len(self._pool)))
        parts = []
        for ex in examples:
            b = ex["B"]
            uses_text = "\n".join(f"- {u}" for u in ex.get("D", [])[:4])
            parts.append(
                f"Example:\n"
                f"Task: {ex['A']}\n"
                f"Mechanism: {b['type']} — {b['rule']}\n"
                f"Uses:\n{uses_text}"
            )
        header = (
            "You are doing the Alternative Uses Task (AUT). "
            "Before listing uses, identify a non-obvious mechanism or property of the object "
            "(e.g. its physical structure, material, or geometry) that unlocks creative uses. "
            "Then list uses that follow from that mechanism.\n\n"
        )
        return header + "\n\n".join(parts)

    def _augment(self, messages: list[dict]) -> list[dict]:
        system_content = self._build_system()
        return [{"role": "system", "content": system_content}] + [
            m for m in messages if m["role"] != "system"
        ]

    def generate(
        self,
        loader: Any,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str:
        return loader.generate(
            self._augment(messages),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )

    def generate_batch(
        self,
        loader: Any,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        augmented = [self._augment(msgs) for msgs in batch_messages]
        return loader.generate_batch(
            augmented,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )


class TwoHopMethod:
    """
    Two-hop generation: (1) generate a mechanism, then (2) generate uses conditioned on it.

    Hop 1: Ask the model to identify a non-obvious mechanism/property of the object.
           The mechanism is appended to the context.
    Hop 2: Ask for uses, with the mechanism already in the assistant turn.

    Args:
        abcd_path:         Optional path to abcd_aut.json for hop-1 few-shot examples.
        n_shots:           Examples for hop-1 prompt (0 = zero-shot).
        seed:              Seed for example selection.
        train_ids:         Optional set of IDs to draw examples from.
        hop1_max_tokens:   Max tokens for mechanism generation (default 128).
    """

    slug = "twohop"
    supports_batch = False

    _HOP1_SYSTEM = (
        "You are an expert at cross-domain analogical transfer. "
        "Identify a specific, non-obvious physical property, material structure, or functional "
        "mechanism of the given object that most people overlook. "
        "Do NOT list any uses. Output only the mechanism: type, rule, and what it unlocks."
    )

    def __init__(
        self,
        abcd_path: str | Path | None = None,
        n_shots: int = 2,
        seed: int = 42,
        train_ids: set[str] | None = None,
        hop1_max_tokens: int = 128,
    ) -> None:
        self.n_shots = n_shots
        self.seed = seed
        self.hop1_max_tokens = hop1_max_tokens
        self._pool: list[dict] = []
        if abcd_path is not None:
            all_items = _load_abcd_aut(abcd_path)
            self._pool = [it for it in all_items if train_ids is None or it["id"] in train_ids]

    def _hop1_examples(self) -> str:
        if not self._pool or self.n_shots == 0:
            return ""
        rng = random.Random(self.seed)
        examples = rng.sample(self._pool, min(self.n_shots, len(self._pool)))
        parts = []
        for ex in examples:
            b = ex["B"]
            parts.append(
                f"Object: {ex['A']}\n"
                f"Mechanism type: {b['type']}\n"
                f"Rule: {b['rule']}\n"
                f"Unlocks: {b.get('unlocks', '')}"
            )
        return "Examples:\n" + "\n\n".join(parts) + "\n\n"

    def generate(
        self,
        loader: Any,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str:
        user_msgs = [m for m in messages if m["role"] == "user"]
        user_content = user_msgs[-1]["content"] if user_msgs else ""

        hop1_user = self._hop1_examples() + f"Object: {user_content}\nMechanism:"
        hop1_messages = [
            {"role": "system", "content": self._HOP1_SYSTEM},
            {"role": "user", "content": hop1_user},
        ]
        mechanism = loader.generate(
            hop1_messages,
            max_new_tokens=self.hop1_max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        ).strip()

        hop2_messages = [
            m for m in messages if m["role"] == "system"
        ] + [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"Mechanism I'll use: {mechanism}"},
            {"role": "user", "content": "Now list the uses based on that mechanism."},
        ]
        return loader.generate(
            hop2_messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )

    def generate_batch(
        self,
        loader: Any,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        return [
            self.generate(loader, msgs, max_new_tokens, temperature, do_sample, **kwargs)
            for msgs in batch_messages
        ]


class AbcdFrameworkMethod:
    """
    Prompt the model with the full ABCD framework framing.

    Explains A (object), B (mechanism bridge), C (creative uses), D (default uses)
    and asks the model to reason through B before giving uses.
    """

    slug = "abcd_framework"
    supports_batch = True

    _SYSTEM = (
        "You are an advanced reasoning engine. "
        "To generate creative alternative uses (C), you must first identify a Bridge Mechanism (B): "
        "a non-obvious physical property, material structure, or cross-domain analogy that "
        "transforms how the object is seen. Avoid default/obvious uses (D). "
        "Format your reply as:\n"
        "B (mechanism): <one sentence>\n"
        "Uses:\n- <use 1>\n- <use 2>\n..."
    )

    def _augment(self, messages: list[dict]) -> list[dict]:
        return [{"role": "system", "content": self._SYSTEM}] + [
            m for m in messages if m["role"] != "system"
        ]

    def generate(
        self,
        loader: Any,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str:
        return loader.generate(
            self._augment(messages),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )

    def generate_batch(
        self,
        loader: Any,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        augmented = [self._augment(msgs) for msgs in batch_messages]
        return loader.generate_batch(
            augmented,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )


# ── Registry: name → factory ──────────────────────────────────────────────────

def available_methods() -> list[str]:
    """Return the list of method names that can be built via ``build_method``."""
    return ["baseline", "steered", "fewshot", "twohop", "abcd_framework"]


def build_method(
    name: str,
    vectors_path: str | Path | None = None,
    alpha: float = 1.0,
    layer_idx: int | None = None,
    abcd_path: str | Path | None = None,
    n_shots: int = 2,
    train_ids: set[str] | None = None,
    steer_mode: str = "all_new_tokens",
    k_assist: int = 16,
) -> AUTInferenceMethod:
    """
    Build a method by name.  Extra kwargs are only required for the relevant methods:

    * ``steered``:       ``vectors_path`` required; ``alpha``, ``layer_idx``, ``steer_mode``, ``k_assist`` optional.
    * ``fewshot``:       ``abcd_path`` required; ``n_shots``, ``train_ids`` optional.
    * ``twohop``:        ``abcd_path`` optional (zero-shot if absent); ``n_shots``, ``train_ids`` optional.
    * ``abcd_framework``: no extra args.
    """
    if name == "baseline":
        return BaselineMethod()
    if name == "steered":
        if not vectors_path:
            raise ValueError("--method steered requires --vectors")
        return SteeredMethod(
            vectors_path=vectors_path, alpha=alpha, layer_idx=layer_idx,
            steer_mode=steer_mode, k_assist=k_assist,
        )
    if name == "fewshot":
        if not abcd_path:
            raise ValueError("--method fewshot requires --abcd-data")
        return FewShotMethod(abcd_path=abcd_path, n_shots=n_shots, train_ids=train_ids)
    if name == "twohop":
        return TwoHopMethod(abcd_path=abcd_path, n_shots=n_shots, train_ids=train_ids)
    if name == "abcd_framework":
        return AbcdFrameworkMethod()
    raise ValueError(
        f"Unknown method {name!r}. Available: {available_methods()}"
    )
