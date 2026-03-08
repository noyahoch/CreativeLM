"""
Completion sources: control how B-condition completions are produced for each
contrastive pair.

This is one of two composable strategy axes in the steering framework:
  --b-source  (CompletionSource)  x  --method  (VectorExtractor)

Any source can be combined with any extractor.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any


SOURCE_REGISTRY: dict[str, type[CompletionSource]] = {}


def register_source(cls: type[CompletionSource]) -> type[CompletionSource]:
    """Class decorator that adds a CompletionSource subclass to the registry."""
    SOURCE_REGISTRY[cls.name] = cls
    return cls


class CompletionSource(ABC):
    """
    Base class for B-completion sources.

    Subclasses control how many (and what) B-condition completion strings are
    produced for each contrastive pair.  The activation collection loop calls
    ``get_b_completions`` once per pair and collects activations for *every*
    returned completion.  With a single-completion source (``FixedCompletion``)
    this reproduces the original behaviour; with a multi-completion source the
    B-activation set grows proportionally.
    """

    name: str  # registry key, set on subclass

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        """Return one or more B-condition completion strings for *pair*.

        Args:
            pair: A contrastive-pair dict (output of ``build_contrastive_pair``).
                  Always contains ``"b_completion"`` and ``"prompt_msgs"``.
            model: The loaded HF CausalLM (available for generation sources).
            tokenizer: The loaded HF tokenizer.

        Returns:
            List of completion strings.  Length >= 1.
        """
        ...

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add source-specific CLI arguments.  Override in subclasses."""


# --------------------------------------------------------------------------- #
# Concrete sources
# --------------------------------------------------------------------------- #


@register_source
class FixedCompletion(CompletionSource):
    """Use the single teacher-forced B completion from the dataset (current default)."""

    name = "fixed"

    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        return [pair["b_completion"]]


@register_source
class GeneratedCompletions(CompletionSource):
    """Generate N creative completions from the model for each item.

    This enriches the B-activation distribution with diverse creative outputs
    rather than relying on a single teacher-forced text.
    """

    name = "generated"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--num-b-completions",
            type=int,
            default=5,
            help="Number of B completions to generate per item (GeneratedCompletions source)",
        )
        parser.add_argument(
            "--b-temperature",
            type=float,
            default=0.7,
            help="Sampling temperature for generated B completions",
        )
        parser.add_argument(
            "--b-max-new-tokens",
            type=int,
            default=256,
            help="Max new tokens for each generated B completion",
        )

    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        raise NotImplementedError(
            "GeneratedCompletions: generate N creative completions from the "
            "model for each item.  Use pair['prompt_msgs'] to build the prompt, "
            "then sample self.config['num_b_completions'] completions with "
            "temperature=self.config['b_temperature']."
        )


# --------------------------------------------------------------------------- #
# Multi-B sources (for datasets with B_list)
# --------------------------------------------------------------------------- #


def format_single_b(b: dict[str, str]) -> str:
    """Format one B-mechanism dict into a teacher-forced completion string.

    Mirrors ``format_B_completion_aut`` in ``data_prep.py`` but operates on
    a single B dict rather than an item with a ``"B"`` key.
    """
    return (
        f"Mechanism type: {b['type']}\n"
        f"Rule: {b['rule']}\n"
        f"This unlocks: {b['unlocks']}\n"
    )


def _get_b_list(pair: dict[str, Any]) -> list[dict[str, str]]:
    """Extract the B_list from a pair, falling back to the single B."""
    b_list = pair.get("b_list")
    if b_list:
        return b_list
    b = pair.get("B")
    if b:
        return [b]
    raise ValueError(
        "Pair has neither 'b_list' nor 'B'.  Use --b-source fixed for "
        "datasets with only a pre-formatted 'b_completion' string."
    )


@register_source
class MultiBSeparate(CompletionSource):
    """One completion per B entry in ``B_list``.

    Each bridge mechanism becomes its own teacher-forced completion, producing
    one activation vector per mechanism.  With 4 bridges per item this
    quadruples the B-activation count relative to ``FixedCompletion``.

    Requires the pair dict to contain ``"b_list"`` (a list of B dicts).
    Falls back to ``pair["B"]`` (single dict) if ``b_list`` is absent.
    """

    name = "multi_b_separate"

    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        return [format_single_b(b) for b in _get_b_list(pair)]


@register_source
class MultiBConcatenated(CompletionSource):
    """All B entries concatenated into a single multi-bridge completion.

    Returns exactly 1 completion per item where every bridge mechanism is
    laid out sequentially.  This lets the model "see" all mechanisms in one
    forward pass rather than separately.

    Requires the pair dict to contain ``"b_list"`` (a list of B dicts).
    Falls back to ``pair["B"]`` (single dict) if ``b_list`` is absent.
    """

    name = "multi_b_concat"

    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        b_list = _get_b_list(pair)
        lines: list[str] = []
        for i, b in enumerate(b_list, 1):
            lines.append(f"Bridge {i} type: {b['type']}")
            lines.append(f"Bridge {i} rule: {b['rule']}")
            lines.append(f"Bridge {i} unlocks: {b['unlocks']}")
        return ["\n".join(lines)]
