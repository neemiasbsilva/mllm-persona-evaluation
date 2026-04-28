"""AnnotationState: the TypedDict flowing through the annotation graph."""

from typing import TypedDict


SENTIMENT_LABELS = (
    "Positive",
    "SlightlyPositive",
    "Neutral",
    "SlightlyNegative",
    "Negative",
)

CONDITIONS = ("baseline", "no_persona_think", "no_persona_no_think")


class AnnotationState(TypedDict):
    """State object for a single (persona × image × condition) annotation run.

    Fields are populated progressively:
      Pipeline input  → persona_id, image_id, condition, persona_record
      image_loader    → image_b64
      prompt_assembler→ system_prompt
      vision_annotator→ predicted_sentiment, predicted_perceptions,
                        caption, justification, raw_response, parse_retries
    """

    # ── Input fields (set by pipeline before graph.invoke) ────────────────────
    persona_id: str
    image_id: str
    """Filename stem (Google Drive ID) — corresponds to {IMAGE_DIR}/{image_id}.jpg."""

    condition: str
    """Experimental condition: 'baseline' (flat demographic tags)."""

    persona_record: dict
    """Full persona dict loaded from the appropriate JSONL (raw_demographics,
    synthetic_interview, expert_reflections, final_system_prompt, etc.)."""

    # ── Node: image_loader ────────────────────────────────────────────────────
    image_b64: str
    """Base64-encoded JPEG string (data URI ready for llava:34b)."""

    # ── Node: prompt_assembler ────────────────────────────────────────────────
    system_prompt: str
    """Fully assembled system block selected according to `condition`."""

    # ── Node: vision_annotator ────────────────────────────────────────────────
    predicted_sentiment: str
    """One of the five perceptsent labels (see SENTIMENT_LABELS)."""

    predicted_perceptions: list[str]
    """1-5 perception labels chosen from the closed 593-item vocabulary."""

    caption: str
    """Objective 1-2 sentence description of the image content (persona-independent)."""

    justification: str
    """1-sentence in-character explanation for the sentiment score."""

    raw_response: str
    """Full model output string — retained for debugging malformed outputs."""

    parse_retries: int
    """Number of JSON parse attempts made by the annotator node."""
