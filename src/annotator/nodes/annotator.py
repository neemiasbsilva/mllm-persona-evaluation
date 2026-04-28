"""Node: vision_annotator.

Calls the configured vision model via Ollama with the assembled system prompt
and base64 image, then parses the structured JSON response.

Each call is bounded by ``LLM_TIMEOUT_S`` (default 180 s) via asyncio.wait_for
and capped at ``ANNOTATION_NUM_PREDICT`` tokens to prevent hangs on slow hardware.

Retry logic:
  - On timeout: re-invoke up to ``max_parse_retries`` times with a compact prompt.
  - On malformed JSON: re-invoke with a correction instruction appended.
  - On invalid sentiment label: attempt to fuzzy-correct before retrying.
  - After exhausting retries: record the failure and surface the raw output.

Public API
----------
  vision_annotator        — default async node (baseline, qwen3-vl, think=True)
  build_annotator_node()  — factory: returns a node with custom model/think args
"""

import asyncio
import json
import re
import time
from typing import Callable

from langchain_ollama import ChatOllama

from annotator.config import AnnotatorSettings, annotator_settings
from annotator.state import AnnotationState, SENTIMENT_LABELS
from persona_generator.logging_config import get_logger

log = get_logger(__name__)

# ── Default LLM (baseline condition) ─────────────────────────────────────────
_llm = ChatOllama(
    model=annotator_settings.vision_model,
    base_url=annotator_settings.ollama_base_url,
    temperature=0.1,
    num_predict=annotator_settings.annotation_num_predict,
    think=True,
    options={
        "seed":    annotator_settings.random_seed,
        "num_ctx": annotator_settings.num_ctx,
    },
)

# ── Sentiment label normalisation map ────────────────────────────────────────
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
    key = raw.strip().lower().replace("_", " ")
    for label in SENTIMENT_LABELS:
        if key == label.lower():
            return label
    return _SENTIMENT_ALIASES.get(key)


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON object found in output", text, 0)
    return json.loads(match.group())


# ── Core annotation logic (shared by default node and factory nodes) ──────────

async def _run_annotation(
    state: AnnotationState,
    llm: ChatOllama,
    cfg: AnnotatorSettings,
) -> AnnotationState:
    """Execute the annotation retry loop using the given *llm* instance."""
    persona_id    = state["persona_id"]
    image_id      = state["image_id"]
    condition     = state["condition"]
    system_prompt = state["system_prompt"]
    image_b64     = state["image_b64"]

    user_message = [
        {"type": "text",      "text": "What is the sentiment of this image?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]
    correction_suffix = ""
    raw_response      = ""
    parse_retries     = 0
    t0 = time.monotonic()

    for attempt in range(cfg.max_parse_retries + 1):
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
                llm.ainvoke(messages),
                timeout=cfg.llm_timeout_s,
            )
            raw_response = response.content.strip()
            parsed = _extract_json(raw_response)
        except asyncio.TimeoutError:
            log.warning(
                "annotator_timeout",
                persona_id=persona_id, image_id=image_id,
                condition=condition, attempt=attempt,
                timeout_s=cfg.llm_timeout_s,
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
                persona_id=persona_id, image_id=image_id,
                condition=condition, attempt=attempt,
                error=str(exc), raw=raw_response[:200],
            )
            correction_suffix = (
                "\n\nYour previous response was not valid JSON. "
                "Return ONLY a JSON object with keys "
                "'sentiment', 'perceptions', 'caption', 'justification'."
            )
            continue

        raw_sentiment = parsed.get("sentiment", "")
        sentiment = _normalise_sentiment(raw_sentiment)

        if sentiment is None:
            log.warning(
                "annotator_invalid_sentiment",
                persona_id=persona_id, image_id=image_id,
                condition=condition, attempt=attempt,
                raw_sentiment=raw_sentiment,
            )
            correction_suffix = (
                f"\n\nYour previous 'sentiment' value '{raw_sentiment}' is invalid. "
                f"It must be exactly one of: {list(SENTIMENT_LABELS)}."
            )
            continue

        duration_ms = int((time.monotonic() - t0) * 1000)
        perceptions = parsed.get("perceptions", [])
        if isinstance(perceptions, str):
            perceptions = [perceptions]

        log.info(
            "annotator_success",
            persona_id=persona_id, image_id=image_id, condition=condition,
            sentiment=sentiment, parse_retries=parse_retries,
            duration_ms=duration_ms,
        )

        return {
            **state,
            "predicted_sentiment":   sentiment,
            "predicted_perceptions": [str(p) for p in perceptions],
            "caption":               str(parsed.get("caption", "")),
            "justification":         str(parsed.get("justification", "")),
            "raw_response":          raw_response,
            "parse_retries":         parse_retries,
        }

    duration_ms = int((time.monotonic() - t0) * 1000)
    log.error(
        "annotator_failed",
        persona_id=persona_id, image_id=image_id, condition=condition,
        parse_retries=parse_retries, duration_ms=duration_ms,
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


# ── Default node (used by baseline graph) ────────────────────────────────────

async def vision_annotator(state: AnnotationState) -> AnnotationState:
    """Default annotation node — uses baseline LLM singleton (think=True)."""
    return await _run_annotation(state, _llm, annotator_settings)


# ── Factory for custom model / think setting ──────────────────────────────────

def build_annotator_node(
    model: str,
    think: bool,
    cfg: AnnotatorSettings | None = None,
) -> Callable[[AnnotationState], "Coroutine[AnnotationState]"]:
    """Return an async annotation node using *model* and the given *think* flag.

    Args:
        model:  Ollama model tag, e.g. ``'qwen3-vl:8b'``.
        think:  When True, chain-of-thought thinking tokens are generated before
                the final JSON response (Qwen3 extended thinking mode).
                When False, the model responds directly without a reasoning prefix.
        cfg:    Optional settings override; falls back to the module singleton.

    Returns:
        An async callable with the same signature as ``vision_annotator``.
    """
    resolved_cfg = cfg or annotator_settings

    llm = ChatOllama(
        model=model,
        base_url=resolved_cfg.ollama_base_url,
        temperature=0.1,
        num_predict=resolved_cfg.annotation_num_predict,
        think=think,
        options={
            "seed":    resolved_cfg.random_seed,
            "num_ctx": resolved_cfg.num_ctx,
        },
    )

    async def _node(state: AnnotationState) -> AnnotationState:
        return await _run_annotation(state, llm, resolved_cfg)

    _node.__name__ = f"vision_annotator_{'think' if think else 'no_think'}_{model.replace(':', '_')}"
    return _node
