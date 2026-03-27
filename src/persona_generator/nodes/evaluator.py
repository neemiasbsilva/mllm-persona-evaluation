"""Node D: Coherence & Guardrail Evaluator.

Pure rule-based validation — no additional LLM call, keeping cost and latency
low.  Detects AI self-identification, moral sermonizing, and voice-shift
artifacts that indicate the backstory synthesizer broke character.

On failure the node populates ``evaluation_feedback`` with a human-readable
explanation that is injected back into Node B's prompt on the next retry.
"""

import re

from persona_generator.config import settings
from persona_generator.logging_config import get_logger
from persona_generator.state import PersonaState

log = get_logger(__name__)

# ── Guardrail patterns ─────────────────────────────────────────────────────────
# Each tuple: (pattern, human-readable failure reason)
_GUARDRAIL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"as an ai (language model|assistant)", re.IGNORECASE),
        'Contains AI self-identification ("as an AI language model/assistant").',
    ),
    (
        re.compile(
            r"\b(i must note|it'?s important to (?:note|acknowledge|recognize)"
            r"|please be aware|i want to clarify|i should mention)\b",
            re.IGNORECASE,
        ),
        "Contains AI hedging / disclaimer language.",
    ),
    (
        re.compile(
            r"\b(in conclusion|to summarize|to sum up|overall,? i believe)\b",
            re.IGNORECASE,
        ),
        "Ends with an AI-style summary or moralizing conclusion.",
    ),
    (
        re.compile(
            r"\b(this individual|the subject|the persona|the character)\b",
            re.IGNORECASE,
        ),
        "Voice shift: refers to self in 3rd person (breaks first-person narration).",
    ),
    (
        re.compile(
            r"\b(i cannot|i will not|i am unable to|i'm not able to)\b",
            re.IGNORECASE,
        ),
        "Contains model refusal language.",
    ),
]

# Minimum word count — reject if the model returned a stub
_MIN_WORD_COUNT = 300


def coherence_evaluator(state: PersonaState) -> PersonaState:
    """Node D — validate the synthetic interview for character breaks.

    Args:
        state: Current pipeline state with ``synthetic_interview`` populated.

    Returns:
        Updated state with ``validation_status``, ``retry_count``, and
        ``evaluation_feedback`` set.
    """
    persona_id = state["persona_id"]
    interview = state["synthetic_interview"]
    retry_count = state["retry_count"]

    failures: list[str] = []

    # 1. Minimum length check
    word_count = len(interview.split())
    if word_count < _MIN_WORD_COUNT:
        failures.append(
            f"Backstory too short ({word_count} words; minimum {_MIN_WORD_COUNT})."
        )

    # 2. Guardrail pattern checks
    for pattern, reason in _GUARDRAIL_PATTERNS:
        if pattern.search(interview):
            failures.append(reason)

    passed = len(failures) == 0

    if passed:
        log.info(
            "node_d_evaluator_pass",
            persona_id=persona_id,
            retry_count=retry_count,
            word_count=word_count,
        )
        return {
            **state,
            "validation_status": True,
            "retry_count": retry_count,
            "evaluation_feedback": "",
        }
    else:
        feedback = " | ".join(failures)
        new_retry_count = retry_count + 1

        log.warning(
            "node_d_evaluator_fail",
            persona_id=persona_id,
            retry_count=new_retry_count,
            failures=failures,
            word_count=word_count,
            max_retries=settings.max_retries,
        )
        return {
            **state,
            "validation_status": False,
            "retry_count": new_retry_count,
            "evaluation_feedback": feedback,
        }
