"""
LLM-as-a-judge: AUT creativity ratings and MacGyver quality scoring.
"""

import re
from typing import Any

from .api import call_openai
from .prompts import (
    JUDGE_SYSTEM,
    MACGYVER_JUDGE_SYSTEM,
    build_judge_user_message,
    build_macgyver_judge_prompt,
)


def _parse_judge_output(raw: str, uses: list[str]) -> list[tuple[str, float]]:
    """
    Parse judge raw text into (use, score) pairs aligned to input uses.
    Extracts USE: ... SCORE: N; then aligns to input list by order or text match.
    """
    parsed: list[tuple[str, float]] = []
    pattern = re.compile(
        r"USE:\s*(.+?)\s*SCORE:\s*([\d.]+)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(raw):
        use_text = m.group(1).strip()
        try:
            score = float(m.group(2))
            score = max(1.0, min(5.0, score))
        except ValueError:
            continue
        parsed.append((use_text, score))

    if not uses:
        return parsed
    result: list[tuple[str, float]] = []
    use_norm = [re.sub(r"\s+", " ", u).strip().lower() for u in uses]
    parsed_by_key = {re.sub(r"\s+", " ", u).strip().lower(): (u, s) for u, s in parsed}
    for i, orig_use in enumerate(uses):
        key = use_norm[i]
        if key in parsed_by_key:
            _, sc = parsed_by_key[key]
            result.append((orig_use, sc))
        elif i < len(parsed):
            result.append((orig_use, parsed[i][1]))
        else:
            for k, (_, sc) in parsed_by_key.items():
                if key in k or k in key:
                    result.append((orig_use, sc))
                    break
            else:
                result.append((orig_use, float("nan")))
    return result[: len(uses)]


class AUTJudge:
    """Judge AUT outputs with an LLM: rate each use 1-5 and optionally aggregate."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_uses_per_item: int = 15,
        **kwargs: Any,
    ):
        self.config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        self.max_uses_per_item = max_uses_per_item

    def _rate_one(self, object_name: str, uses: list[str]) -> list[tuple[str, float]]:
        """Rate one item (object + list of uses). Returns [(use, score), ...]."""
        uses = [u.strip() for u in uses if u and str(u).strip()][: self.max_uses_per_item]
        if not uses:
            return []
        user_content = build_judge_user_message(object_name, uses)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        raw = call_openai(messages, self.config)
        return _parse_judge_output(raw, uses)

    def rate(self, items: list[dict]) -> list[dict]:
        """
        Rate multiple AUT items.

        Each item must have:
          - "object": str (e.g. "brick")
          - "uses": list[str] (alternative uses)

        Optional: "id" for tracking.

        Returns list of dicts with original keys plus:
          - "scores": list of (use, score) tuples
          - "avg_creativity": float
          - "fluency": int (number of uses)
        """
        results = []
        for it in items:
            obj = it.get("object", "")
            uses = it.get("uses", [])
            if isinstance(uses, str):
                uses = [u.strip() for u in uses.split("\n") if u.strip()]
            out = {k: v for k, v in it.items()}
            scores = self._rate_one(obj, uses)
            out["scores"] = scores
            out["fluency"] = len(uses)
            scored = [(u, s) for u, s in scores if s == s]
            out["avg_creativity"] = (
                sum(s for _, s in scored) / len(scored) if scored else float("nan")
            )
            results.append(out)
        return results


# ---------------------------------------------------------------------------
# MacGyver quality judge
# ---------------------------------------------------------------------------

_MACGYVER_SCORE_RE = re.compile(r"[Ss]core\s*:\s*(\d)")
_MACGYVER_FRAC_RE = re.compile(r"\b([0-5])\s*/\s*5\b")


def parse_macgyver_score(raw: str) -> int | None:
    """Extract a 0-5 integer quality score from the judge response."""
    m = _MACGYVER_SCORE_RE.search(raw)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 5:
            return val
    m = _MACGYVER_FRAC_RE.search(raw)
    if m:
        return int(m.group(1))
    return None


class MacGyverJudge:
    """Judge MacGyver problem-solving outputs with the 5-point additive rubric.

    Supports two backends:
      * **sync** (default): uses ``call_openai`` — works with any
        OpenAI-compatible endpoint (set ``base_url`` for vLLM / DeepSeek).
      * **async**: call :meth:`rate_async` with an ``AsyncOpenAI`` client
        for high-throughput vLLM batching (used by the CLI script).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 300,
        **kwargs: Any,
    ) -> None:
        self.config: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

    # -- sync (one-at-a-time, via call_openai) ------------------------------

    def _rate_one_sync(self, user_prompt: str, model_response: str) -> dict[str, Any]:
        """Score a single (problem, solution) pair synchronously."""
        content = build_macgyver_judge_prompt(user_prompt, model_response)
        messages = [
            {"role": "system", "content": MACGYVER_JUDGE_SYSTEM},
            {"role": "user", "content": content},
        ]
        raw = call_openai(messages, self.config)
        score = parse_macgyver_score(raw)
        return {"quality_score": score, "judge_explanation": raw}

    def rate(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rate a list of items synchronously.

        Each item must have ``user_prompt`` (or ``problem``) and
        ``model_response`` (or ``reply``).
        """
        results: list[dict[str, Any]] = []
        for it in items:
            prompt = it.get("user_prompt") or it.get("problem", "")
            response = it.get("model_response") or it.get("reply", "")
            verdict = self._rate_one_sync(prompt, response)
            results.append({**it, **verdict})
        return results

    # -- async (high-throughput, for vLLM) ----------------------------------

    @staticmethod
    async def rate_one_async(
        client: Any,
        model: str,
        user_prompt: str,
        model_response: str,
        sema: Any,
        temperature: float = 0.0,
        max_tokens: int = 300,
        max_retries: int = 3,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Score one pair asynchronously via an ``AsyncOpenAI`` client."""
        import asyncio

        content = build_macgyver_judge_prompt(user_prompt, model_response)
        messages = [
            {"role": "system", "content": MACGYVER_JUDGE_SYSTEM},
            {"role": "user", "content": content},
        ]
        last_raw = ""
        for attempt in range(max_retries):
            async with sema:
                try:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                        timeout=timeout_s,
                    )
                    last_raw = (resp.choices[0].message.content or "").strip()
                    score = parse_macgyver_score(last_raw)
                    if score is not None:
                        return {
                            "quality_score": score,
                            "judge_explanation": last_raw,
                            "err": "",
                            "attempt": attempt,
                        }
                    if attempt == max_retries - 1:
                        return {
                            "quality_score": None,
                            "judge_explanation": last_raw,
                            "err": "score_parse_failed",
                            "attempt": attempt,
                        }
                except Exception as e:
                    if attempt == max_retries - 1:
                        return {
                            "quality_score": None,
                            "judge_explanation": last_raw,
                            "err": f"{type(e).__name__}: {e}",
                            "attempt": attempt,
                        }
        return {}
