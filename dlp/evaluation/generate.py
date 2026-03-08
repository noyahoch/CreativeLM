"""
Generate AUT outputs: given an object (e.g. brick), produce alternative uses via LLM.

Uses OpenAI chat models. For local HF model inference, use methods.py +
scripts/run_aut_inference.py instead.
"""

import re
from pathlib import Path
from typing import Any

from .api import call_openai

GENERATE_SYSTEM = """You are doing the Alternative Uses Task (AUT). For a given common object, list as many different, creative alternative uses as you can. Be concise: one short phrase per use. Do not explain or number them."""

GENERATE_USER_TEMPLATE = """
List {num_uses} alternative uses for a {object}. One use per line, starting each line with a dash and a space (e.g. "- Use as a doorstop"). No numbering or explanations."""


def _parse_uses_from_raw(raw: str) -> list[str]:
    """Extract list of uses from model output (bullets or numbered lines)."""
    uses = []
    for line in raw.split("\n"):
        line = line.strip()
        line = re.sub(r"^[\-\*\•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"\s*:\s*[\d.]+\s*$", "", line).strip()
        if line and len(line) > 1:
            uses.append(line)
    return uses


def generate_uses_for_object(
    object_name: str,
    num_uses: int = 10,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> list[str]:
    """
    Generate alternative uses for one object via OpenAI.

    Returns list of use strings (parsed from model output).
    """
    user_content = GENERATE_USER_TEMPLATE.format(
        object=object_name,
        num_uses=num_uses,
    )
    messages = [
        {"role": "system", "content": GENERATE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    config = {"model": model, "temperature": temperature, "max_tokens": max_tokens, **kwargs}
    raw = call_openai(messages, config)
    return _parse_uses_from_raw(raw)[:num_uses]


def generate_aut_outputs(
    objects: list[str] | list[dict],
    num_uses: int = 10,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    **kwargs: Any,
) -> list[dict]:
    """
    Generate AUT outputs for multiple objects.

    objects: list of object names (str) or list of dicts with "object" (and optional "id").
    Returns list of {"id"?, "object", "uses": list[str]}.
    """
    from tqdm import tqdm

    items = []
    for i, obj in enumerate(tqdm(objects, desc="Generate")):
        if isinstance(obj, dict):
            name = obj.get("object", "")
            id_ = obj.get("id", str(i + 1))
        else:
            name = str(obj).strip()
            id_ = str(i + 1)
        if not name:
            continue
        uses = generate_uses_for_object(
            name,
            num_uses=num_uses,
            model=model,
            temperature=temperature,
            **kwargs,
        )
        items.append({"id": id_, "object": name, "uses": uses})
    return items


def save_flat(
    items: list[dict],
    path: str | Path,
) -> None:
    """
    Save AUT outputs as flat ``Object, use`` lines (one use per line).

    Format::

        Pants, to wear them
        Pants, to tie things with
        Pants, makeshift flag
    """
    def _strip_commas(text: str) -> str:
        text = re.sub(r",\s+(and|or|but|so|yet|nor)\b", r" \1", text)
        return text.replace(", ", " ").replace(",", "")

    lines = []
    for it in items:
        obj = it.get("object", "")
        uses = it.get("uses", [])
        if isinstance(uses, str):
            uses = [u.strip() for u in uses.split("\n") if u.strip()]
        for use in uses:
            lines.append(f"{obj}, {_strip_commas(use)}")
    Path(path).write_text("\n".join(lines) + "\n")
