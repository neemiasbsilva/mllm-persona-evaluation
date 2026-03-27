"""Node: vision_annotator.

Calls the configured vision model (default: qwen3-vl:32b) via Ollama with the
assembled system prompt and base64 image, then parses the structured JSON response.

Each call is bounded by ``LLM_TIMEOUT_S`` (default 180 s) via asyncio.wait_for
and capped at ``ANNOTATION_NUM_PREDICT`` tokens (default 512) to prevent hangs
on slow or resource-constrained hardware.

Retry logic:
  - On timeout: re-invoke up to ``max_parse_retries`` times with a compact prompt.
  - On malformed JSON: re-invoke with a correction instruction appended.
  - On invalid sentiment label: attempt to fuzzy-correct before retrying.
  - After exhausting retries: record the failure and surface the raw output.
"""

import asyncio
import json
import re
import time

from langchain_ollama import ChatOllama

from annotator.config import annotator_settings
from annotator.state import AnnotationState, SENTIMENT_LABELS
from persona_generator.logging_config import get_logger

log = get_logger(__name__)

# ── LLM ──────────────────────────────────────────────────────────────────────
# Separate instance from the text-gen LLM; uses the vision model.
_llm = ChatOllama(
    model=annotator_settings.vision_model,
    base_url=annotator_settings.ollama_base_url,
    temperature=0.1,    # low temperature for deterministic annotation
    num_predict=annotator_settings.annotation_num_predict,  # cap generation length
    think=True,       # disable thinking mode: output goes directly to response.content
    options={
        "seed":    annotator_settings.random_seed,
        "num_ctx": annotator_settings.num_ctx,  # smaller ctx → less VRAM → more parallel slots
    },
)

# ── Sentiment label normalisation map ────────────────────────────────────────
# Handles common variants the model might produce before retrying
_SENTIMENT_ALIASES: dict[str, str] = {
    "positive":          "Positive",
    "slightlypositive":  "SlightlyPositive",
    "slightly positive": "SlightlyPositive",
    "neutral":           "Neutral",
    "slightlynegative":  "SlightlyNegative",
    "slightly negative": "SlightlyNegative",
    "negative":          "Negative",
}


def _normalise_sentiment(raw: str) -> str | None:
    """Map a raw sentiment string to one of the five canonical labels.

    Returns None if no mapping is found.
    """
    key = raw.strip().lower().replace("_", " ")
    for label in SENTIMENT_LABELS:
        if key == label.lower():
            return label
    return _SENTIMENT_ALIASES.get(key)


def _extract_json(text: str) -> dict:
    """Extract and parse the first JSON object from model output text.

    The model sometimes wraps the JSON in markdown fences or adds a leading
    sentence.  This function strips those before parsing.

    Raises:
        json.JSONDecodeError: If no valid JSON object is found.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON object found in output", text, 0)
    return json.loads(match.group())


async def vision_annotator(state: AnnotationState) -> AnnotationState:
    """Node: call llava:34b and parse the structured annotation response.

    Args:
        state: Must have ``system_prompt`` and ``image_b64`` populated.

    Returns:
        Updated state with ``predicted_sentiment``, ``predicted_perceptions``,
        ``caption``, ``justification``, ``raw_response``, and ``parse_retries`` set.
    """
    persona_id  = state["persona_id"]
    image_id    = state["image_id"]
    condition   = state["condition"]
    system_prompt = state["system_prompt"]
    image_b64   = state["image_b64"]

    user_message = [
        {"type": "text",      "text": "What is the sentiment of this image?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]
    correction_suffix = ""
    raw_response = ""
    parse_retries = 0

    t0 = time.monotonic()

    for attempt in range(annotator_settings.max_parse_retries + 1):
        parse_retries = attempt

        messages = [
            ("system", system_prompt),
            ("human",  user_message if attempt == 0
                       else [{"type": "text",
                              "text": f"What is the sentiment of this image?{correction_suffix}"},
                             {"type": "image_url",
                              "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]),
        ]

        try:
            response = await asyncio.wait_for(
                _llm.ainvoke(messages),
                timeout=annotator_settings.llm_timeout_s,
            )
            raw_response = response.content.strip()
            parsed = _extract_json(raw_response)
        except asyncio.TimeoutError:
            log.warning(
                "annotator_timeout",
                persona_id=persona_id,
                image_id=image_id,
                condition=condition,
                attempt=attempt,
                timeout_s=annotator_settings.llm_timeout_s,
            )
            correction_suffix = (
                "\n\nPrevious attempt timed out. "
                "Return ONLY a compact JSON object with keys "
                "'sentiment', 'perceptions', 'caption', 'justification'."
            )
            continue
        except (json.JSONDecodeError, Exception) as exc:
            log.warning(
                "annotator_parse_error",
                persona_id=persona_id,
                image_id=image_id,
                condition=condition,
                attempt=attempt,
                error=str(exc),
                raw=raw_response[:200],
            )
            correction_suffix = (
                "\n\nYour previous response was not valid JSON. "
                "Return ONLY a JSON object with keys "
                "'sentiment', 'perceptions', 'caption', 'justification'."
            )
            continue

        # ── Validate sentiment label ──────────────────────────────────────────
        raw_sentiment = parsed.get("sentiment", "")
        sentiment = _normalise_sentiment(raw_sentiment)

        if sentiment is None:
            log.warning(
                "annotator_invalid_sentiment",
                persona_id=persona_id,
                image_id=image_id,
                condition=condition,
                attempt=attempt,
                raw_sentiment=raw_sentiment,
            )
            correction_suffix = (
                f"\n\nYour previous 'sentiment' value '{raw_sentiment}' is invalid. "
                f"It must be exactly one of: {list(SENTIMENT_LABELS)}."
            )
            continue

        # ── Success ───────────────────────────────────────────────────────────
        duration_ms = int((time.monotonic() - t0) * 1000)
        perceptions = parsed.get("perceptions", [])
        if isinstance(perceptions, str):
            perceptions = [perceptions]

        log.info(
            "annotator_success",
            persona_id=persona_id,
            image_id=image_id,
            condition=condition,
            sentiment=sentiment,
            parse_retries=parse_retries,
            duration_ms=duration_ms,
        )

        return {
            **state,
            "predicted_sentiment":    sentiment,
            "predicted_perceptions":  [str(p) for p in perceptions],
            "caption":                str(parsed.get("caption", "")),
            "justification":          str(parsed.get("justification", "")),
            "raw_response":           raw_response,
            "parse_retries":          parse_retries,
        }

    # ── All retries exhausted ─────────────────────────────────────────────────
    duration_ms = int((time.monotonic() - t0) * 1000)
    log.error(
        "annotator_failed",
        persona_id=persona_id,
        image_id=image_id,
        condition=condition,
        parse_retries=parse_retries,
        duration_ms=duration_ms,
        raw=raw_response[:300],
    )

    return {
        **state,
        "predicted_sentiment":   "",
        "predicted_perceptions": [],
        "caption":               "",
        "justification":         "",
        "raw_response":          raw_response,
        "parse_retries":         parse_retries,
    }
