"""
SFT data preparation: prompt/target formatting and contrastive pair building.

Supports two task types:
  - TaskType.AUT  – Alternative Uses Task (abcd_aut.json)
  - TaskType.PS   – Problem Solving Task  (abcd_aiden.json)
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class TaskType(Enum):
    AUT = "aut"
    PS = "ps"


# ---------------------------------------------------------------------------
# User-prompt formatters
# ---------------------------------------------------------------------------

def format_user_prompt_ps(item: dict) -> str:
    """Build the user prompt for a PS item (mechanism-first format)."""
    problem = item["A"]
    return (
        "Problem Solving Task\n\n"
        f"Problem: {problem}\n\n"
        "Give one creative, non-obvious solution. "
        "State the mechanism in one line (Mechanism: <type> — <rule>), then your solution in a short paragraph."
    )


def format_user_prompt_baseline_ps(item: dict) -> str:
    """Baseline PS prompt: creative solution only, no mechanism format."""
    problem = item["A"]
    return (
        "Problem Solving Task\n\n"
        f"Problem: {problem}\n\n"
        "Give one creative, non-obvious solution in a short paragraph."
    )


def format_user_prompt_aut(item: dict) -> str:
    """Build the user prompt for an AUT item."""
    task = item["A"]
    return (
        "Alternative Uses Task\n\n"
        f"{task}\n\n"
        "Give 8 unconventional uses. List them clearly."
    )


def format_user_prompt_baseline_aut(item: dict) -> str:
    """Baseline AUT prompt (same as standard; mechanism is not requested)."""
    return format_user_prompt_aut(item)


def format_user_prompt(item: dict, task_type: TaskType) -> str:
    """Route to the correct user-prompt formatter."""
    if task_type == TaskType.PS:
        return format_user_prompt_ps(item)
    return format_user_prompt_aut(item)


# ---------------------------------------------------------------------------
# Target (assistant) formatters
# ---------------------------------------------------------------------------

def format_target_ps(item: dict) -> str:
    """Mechanism line + Solution block for PS."""
    b, c = item["B"], item["C"]
    return f"Mechanism: {b['type']} — {b['rule']}\n\nSolution:\n\n{c}"


def format_target_aut(item: dict) -> str:
    """Mechanism line + bulleted uses list for AUT."""
    b, c = item["B"], item["C"]
    uses = c if isinstance(c, list) else [c]
    uses_text = "\n".join(f"- {u}" for u in uses)
    return (
        f"Mechanism: {b['type']} — {b['rule']}\n\n"
        f"Uses:\n{uses_text}"
    )


def format_target(item: dict, task_type: TaskType) -> str:
    """Route to the correct target formatter."""
    if task_type == TaskType.PS:
        return format_target_ps(item)
    return format_target_aut(item)


# ---------------------------------------------------------------------------
# Teacher-forced completions for contrastive / steering pairs
# ---------------------------------------------------------------------------

def format_D_completion(item: dict, task_type: TaskType, d_key: str = "D") -> str:
    """
    Default-listing completion (teacher-forced, condition D).

    AUT: bullet list of default uses from item[d_key] (default: "D", alt: "D_banal").
    PS: conventional paragraph built from top-3 items.
    """
    if task_type == TaskType.PS:
        top = item[d_key][:3]
        return (
            f"A straightforward approach: {top[0].lower()}. "
            f"Additionally, {top[1].lower()}. "
            f"To reinforce this, {top[2].lower()}. "
            "These standard practices should help address the issue."
        )
    return "\n".join(f"- {use}" for use in item[d_key])


def format_B_completion_aut(item: dict) -> str:
    """
    Mechanism-only completion for AUT (condition B, teacher-forced).

    Pure analytical reasoning — type, rule, unlocks. No uses list.
    This is what we want to isolate from the "listing" direction.
    """
    b = item["B"]
    return (
        f"Mechanism type: {b['type']}\n"
        f"Rule: {b['rule']}\n"
        f"This unlocks: {b['unlocks']}\n"
    )


def format_B_completion_ps(item: dict) -> str:
    """
    Mechanism + creative-solution completion for PS (condition B, teacher-forced).

    Mechanism identification → creative application.
    """
    b, c = item["B"], item["C"]
    return (
        f"Mechanism type: {b['type']}\n"
        f"Rule: {b['rule']}\n\n"
        f"{c}"
    )


def format_B_completion(item: dict, task_type: TaskType) -> str:
    """Route to the correct B-completion formatter."""
    if task_type == TaskType.PS:
        return format_B_completion_ps(item)
    return format_B_completion_aut(item)


# ---------------------------------------------------------------------------
# Contrastive pair builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = ""


def build_contrastive_pair(item: dict, task_type: TaskType, d_key: str = "D") -> dict:
    """
    Build a contrastive pair: same prompt, different teacher-forced completions.

      D: prompt + conventional-solution completion  (default mode)
      B: prompt + creative/mechanism completion      (creative mode)

    v_bridge = mean(acts_B) - mean(acts_D) at early completion tokens
    captures the creative-thinking direction.

    d_key selects which field to use for the D completion ("D" or "D_banal").
    """
    user_prompt = format_user_prompt(item, task_type)
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return {
        "id": item.get("id", ""),
        "domain": item.get("domain", item.get("task", "")),
        "user_prompt": user_prompt,
        "prompt_msgs": prompt_msgs,
        "d_completion": format_D_completion(item, task_type, d_key),
        "b_completion": format_B_completion(item, task_type),
    }


def build_sft_example(item: dict, task_type: TaskType) -> dict[str, str]:
    """
    Build a single SFT training example as {"input_text": ..., "target_text": ...}.

    The model is trained to produce a mechanism line followed by the solution/uses.
    """
    return {
        "id": item.get("id", ""),
        "input_text": format_user_prompt(item, task_type),
        "target_text": format_target(item, task_type),
    }


def build_sft_dataset(
    items: list[dict],
    task_type: TaskType,
    system_prompt: str = "",
) -> list[dict[str, Any]]:
    """
    Build a list of SFT training examples from ABCD items.

    Returns:
        List of dicts with keys: id, messages (chat format), labels.
    """
    examples = []
    for item in items:
        user_prompt = format_user_prompt(item, task_type)
        target = format_target(item, task_type)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": target})
        examples.append({
            "id": item.get("id", ""),
            "messages": messages,
            "input_text": user_prompt,
            "target_text": target,
        })
    return examples
