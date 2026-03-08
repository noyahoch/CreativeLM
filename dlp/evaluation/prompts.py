"""Prompt templates for LLM-as-a-judge (AUT + MacGyver)."""

JUDGE_SYSTEM = """You are an expert rater for creativity assessments. Your task is to rate how creative each "alternative use" is on a scale of 1 to 5.
- 1 = not at all creative (obvious, common, or not a real use)
- 2 = slightly creative
- 3 = moderately creative
- 4 = creative
- 5 = very creative (novel, surprising, or clever)

Respond only with a list of lines in this exact format, one per use:
USE: <exact use text>
SCORE: <number 1-5>

Do not add explanations. Keep the USE text exactly as given."""

JUDGE_USER_TEMPLATE = """Object: {object}

Alternative uses to rate (one per line):
{uses_text}

Provide your ratings in the format:
USE: <use>
SCORE: <1-5>"""


def build_judge_user_message(obj: str, uses: list[str]) -> str:
    """Build the user message for the judge given object and list of uses."""
    uses_text = "\n".join(f"- {u.strip()}" for u in uses if (u and str(u).strip()))
    return JUDGE_USER_TEMPLATE.format(object=obj, uses_text=uses_text)
# Judge prompt: rate novelty (1–10) and usability (1–10) of the proposed use for the object.
# Output must match JudgeOut (analysis.key_object_property, analysis.reasoning, scores.novelty/usability/overall).

# def build_per_use_prompt(object_name: str, use_text: str) -> str:
#     use_preview = (use_text or "")[:400]
#     return f"""
# Output ONLY one valid JSON object. No extra text.

# Object: {object_name}
# Proposed use: {use_preview}

# Score integers 1–10 and compute:
# overall = round((novelty + usability) / 2)

# NOVELTY (1–10): How uncommon and non-obvious the underlying idea is for this object in this setting.
# Ignore writing quality/length/format.

# When scoring novelty, use these natural checks:
# - If it’s basically a common “process template” (reminders, labeling, brainstorming, voting, checklists, generic feedback walls), novelty is usually low.
# - Higher novelty comes from ideas that rely on what’s distinctive about this object (a property or affordance you wouldn’t get similarly from many other objects).
# - Use 9–10 when the idea would genuinely surprise most evaluators while still being coherent.

# Anchors:
# 1–2 obvious/default
# 3–4 familiar or generic twist
# 5–6 clearly non-default
# 7–8 rare + clearly object-specific
# 9–10 extremely rare + object-specific + surprising

# USABILITY (1–10): Likelihood it works as stated with realistic constraints/resources.
# Penalize hidden tools, fragile assumptions, contradictions, safety issues.

# Output JSON exactly (use key_object_property and reasoning so the parser can read it):
# {{
#   "analysis": {{
#     "key_object_property": "one key physical/functional property used",
#     "reasoning": "brief reason, <= 20 words"
#   }},
#   "scores": {{
#     "novelty": integer,
#     "usability": integer,
#     "overall": integer
#   }}
# }}
# """.strip()

# def build_per_use_prompt(object_name: str, use_text: str) -> str:
#     use_preview = (use_text or "")[:400]
#     return f"""
# Output ONLY one valid JSON object. No extra text.

# Object: {object_name}
# Proposed use: {use_preview}

# Use the full 1–10 scale.
# - novelty=1 only if it's an obvious default use for this object
# - usability=1 only if it basically cannot work as stated

# NOVELTY (1–10): how uncommon/non-obvious this idea is for this object+setting.
# Anchors:
# 1–2 obvious/default; 3–4 common twist; 5–6 non-default; 7–8 rare or creative; 9–10 extremely rare

# USABILITY (1–10): how likely it works as stated with realistic resources.
# Anchors:
# 1–2 won't work; 3–4 barely plausible; 5–6 plausible but underspecified; 7–8 practical; 9–10 very reliable

# overall = round((novelty + usability)/2)

# Output JSON exactly:
# {{
#   "analysis": {{
#     "why_novelty": "string (<= 12 words)",
#     "why_usability": "string (<= 12 words)"
#   }},
#   "scores": {{
#     "novelty": integer,
#     "usability": integer,
#     "overall": integer
#   }}
# }}
# """.strip()
def build_per_use_prompt(object_name: str, use_text: str) -> str:
    use_preview = (use_text or "")[:400]
    return f"""
Output ONLY one valid JSON object. No extra text.

Object: {object_name}
Proposed use: {use_preview}

You are a judge for a dataset of intentionally unconventional uses.
This dataset contains many intentionally unconventional uses, so do not default to 1.
Choose the closest anchor below.

NOVELTY (1–10): how uncommon/non-obvious the underlying idea is for this object+setting.
- 1–2: truly default / first-thought / widely common
- 3–4: common pattern with a small twist
- 5–6: clearly non-default; plausible new angle
- 7–8: rare and surprising; object-specific in a meaningful way
- 9–10: extremely rare; would surprise most evaluators; still coherent

USABILITY (1–10): likelihood it works as stated with realistic constraints/resources.
- 1–2: basically won’t work as stated
- 3–4: fragile / missing key details / major assumptions
- 5–6: plausible but needs care or minor additions
- 7–8: practical and straightforward
- 9–10: very reliable/robust

overall = round((novelty + usability) / 2)

Output JSON exactly:
{{
  "analysis": {{
    "key_object_property": "string (<= 8 words)",
    "reasoning": "string (<= 20 words)"
  }},
  "scores": {{
    "novelty": integer,
    "usability": integer,
    "overall": integer
  }}
}}
""".strip()


# def build_per_use_prompt(object_name: str, use_text: str) -> str:
#     use_preview = (use_text or "")[:400]
#     return f"""
# Output ONLY one valid JSON object. No extra text.

# Object: {object_name}
# Proposed use: {use_preview}

# Return integer scores 1–10 and compute:
# overall = round(0.6*novelty + 0.4*usability)

# NOVELTY (1–10): rate how non-obvious/creative the underlying idea is for THIS object+setting.
# Score higher when the use depends on a specific property of this object (not easily interchangeable).
# Anchors:
# 1–2 default/obvious
# 3–4 small twist on common use
# 5–6 clearly non-default and creative, but plausible
# 7–8 very creative and object-specific
# 9–10 extremely rare and surprising, still coherent

# USABILITY (1–10): probability it works as stated with realistic constraints/resources.
# Anchors:
# 1–2 won’t work
# 3–4 barely plausible / missing key details
# 5–6 plausible but underspecified
# 7–8 practical
# 9–10 very reliable

# Output JSON exactly:
# {{
#   "analysis": {{
#     "why_novelty": "string (<= 12 words)",
#     "why_usability": "string (<= 12 words)"
#   }},
#   "scores": {{
#     "novelty": integer,
#     "usability": integer,
#     "overall": integer
#   }}
# }}
# """.strip()

# def build_per_use_prompt(object_name: str, use_text: str) -> str:
#     use_preview = (use_text or "")[:400]
#     return f"""
# Output ONLY one valid JSON object. No extra text.

# Object: {object_name}
# Proposed use: {use_preview}

# Score integers 1–10. Use the full scale (avoid defaulting to 7/8).
# Set overall = round((novelty + usability) / 2)

# NOVELTY (1–10) = creativity + non-obviousness for this object+setting.
# Calibration: start at 5. Move up/down only with clear evidence.
# - 9–10: genuinely original insight; surprising yet coherent.
# - 7–8: clearly creative; not a routine variant; feels “new”.
# - 5–6: mild twist; somewhat non-obvious but familiar.
# - 3–4: common pattern / small variation.
# - 1–2: obvious / default / first-thought.

# USABILITY (1–10) = likelihood it works as stated with realistic resources.
# Calibration: start at 7. Move up/down only with clear evidence.
# - 9–10: very reliable as stated; minimal missing details.
# - 7–8: practical; minor gaps acceptable.
# - 5–6: plausible but underspecified/fragile.
# - 3–4: doubtful without major extra assumptions.
# - 1–2: basically won’t work / contradictory.

# Output JSON exactly:
# {{
#   "analysis": {{
#     "why_novelty": "string (<= 12 words)",
#     "why_usability": "string (<= 12 words)"
#   }},
#   "scores": {{
#     "novelty": integer,
#     "usability": integer,
#     "overall": integer
#   }}
# }}
# """.strip()

# def build_per_use_prompt(object_name: str, use_text: str) -> str:
#     use_preview = (use_text or "")[:400]
#     return f"""
# Output ONLY one valid JSON object. No extra text.

# Object: {object_name}
# Proposed use: {use_preview}

# Score integers 1–10. Use the full scale.
# overall = round((novelty + usability) / 2)

# NOVELTY (1–10) = creativity + non-obviousness for THIS object+setting.
# Default novelty = 5. Adjust only with evidence.

# To give novelty >= 7, it must satisfy BOTH:
# (1) Not a generic template (organize/label/remind/vote/brainstorm/checklist/feedback wall).
# (2) The object’s specific properties are essential (not easily swapped).

# To give novelty >= 9, it must satisfy ALL:
# (1) Not a generic template,
# (2) Object-specific and hard to substitute,
# (3) Includes a surprising mechanism/insight that most people wouldn’t propose.

# Anchors:
# 1–2 obvious/default
# 3–4 common template or small twist
# 5–6 some creativity but still familiar
# 7–8 clearly inventive + object-specific
# 9–10 exceptional, surprising, coherent

# USABILITY (1–10) = likelihood it works as stated with realistic resources.
# Default usability = 7. Adjust only with evidence.
# Caps:
# - Needs major unspecified tools/materials => usability <= 5
# - Unsafe/damaging unless mitigation stated => usability <= 5
# - Physically contradictory/implausible => usability <= 2

# Soft calibration (don’t mention numbers in output):
# Across many items, most novelty should land 4–6; 7–8 is rarer; 9–10 is very rare.

# Output JSON exactly:
# {{
#   "analysis": {{
#     "why_novelty": "string (<= 12 words)",
#     "why_usability": "string (<= 12 words)"
#   }},
#   "scores": {{
#     "novelty": integer,
#     "usability": integer,
#     "overall": integer
#   }}
# }}
# """.strip()


# def build_per_use_prompt(object_name: str, use_text: str) -> str:
#     use_preview = (use_text or "")[:400]
#     return f"""
# Output ONLY one valid JSON object. No extra text.

# Object: {object_name}
# Proposed use: {use_preview}
# This dataset contains many intentionally unconventional uses, so do not default to 1.
# Choose the closest anchor below.

# NOVELTY (1–10): how uncommon/non-obvious the underlying idea is for this object+setting.
# - 1–2: truly default / first-thought / widely common
# - 3–4: common pattern with a small twist
# - 5–6: clearly non-default; plausible new angle
# - 7–8: creative and surprising; object-specific in a meaningful way
# - 9–10: extremely creative; would surprise most evaluators; still coherent

# Rule for 9–10: use 9–10 ONLY if the proposed use explicitly states
# a concrete object-specific mechanism (a property/affordance that is essential).

# USABILITY (1–10): likelihood it works as stated with realistic constraints/resources.
# - 1–2: basically won’t work as stated
# - 3–4: fragile / missing key details / major assumptions
# - 5–6: plausible but needs care or minor additions
# - 7–8: practical and straightforward
# - 9–10: very reliable/robust

# overall = round((novelty + usability) / 2)

# Output JSON exactly:
# {{
#   "analysis": {{
#     "key_object_property": "string (<= 8 words)",
#     "reasoning": "string (<= 20 words)"
#   }},
#   "scores": {{
#     "novelty": integer,
#     "usability": integer,
#     "overall": integer
#   }}
# }}
# """.strip()
def build_per_use_prompt(object_name: str, use_text: str) -> str:
    use_preview = (use_text or "")[:400]
    return f"""
Output ONLY one valid JSON object. No extra text.
Rate the creativity and originality of the following alternative use for a {object_name} on a scale of 1 to 5:

Alternative use: {use_preview}

Rating criteria:
1 - Common/obvious use, not creative
2 - Somewhat uncommon but still fairly obvious
3 - Moderately creative and original
4 - Very creative and original
5 - Exceptionally creative, highly original and unconventional

Respond with only the number (1-5):"""


# ---------------------------------------------------------------------------
# MacGyver quality judge — 5-point additive rubric
# ---------------------------------------------------------------------------

MACGYVER_JUDGE_SYSTEM = (
    "You are an expert evaluator for creative problem-solving tasks. "
    "Score solutions using the additive rubric provided."
)

MACGYVER_JUDGE_TEMPLATE = """\
Review the solution generated in response to a MacGyver-style problem and \
score it using the additive 5-point scoring system described below. Points \
are accumulated based on the satisfaction of each criterion:

Problem statement: {user_prompt}

Model's solution: {model_response}

- Add 1 point if the solution attempts to address the problem using only \
the given resources, without introducing external tools or elements not \
mentioned.
- Add another point if the solution demonstrates a reasonable understanding \
of the properties and limitations of the available resources, and applies \
them correctly.
- Award a third point if the solution adheres to the physical constraints \
of the problem (e.g., size, weight, strength) and does not propose an \
obviously unfeasible approach.
- Grant a fourth point if the solution is practical and likely to solve the \
problem effectively within the constraints of the scenario.
- Bestow a fifth point for a solution that is complete, logically \
structured, and provides a clear explanation of how it solves the problem.

After examining the problem, available resources, and the proposed solution:
- First explain your total score, up to 100 words.
- Then conclude with the score using the format:
Score: <total points>"""


def build_macgyver_judge_prompt(user_prompt: str, model_response: str) -> str:
    """Build the judge prompt for a single MacGyver (problem, solution) pair."""
    return MACGYVER_JUDGE_TEMPLATE.format(
        user_prompt=user_prompt.strip(),
        model_response=(model_response or "").strip()[:2000],
    )