"""Vision annotation prompt template — baseline demographic tag condition.

The system prompt uses the 4 flat demographic tags from raw_demographics.
The annotation instruction requires four structured fields:
  sentiment     — one of five perceptsent labels
  perceptions   — 1-5 labels chosen from the 593-item closed vocabulary
  caption       — objective 1-2 sentence image description (persona-agnostic)
  justification — 1 sentence in the persona's own voice
"""

import json
from pathlib import Path

from annotator.state import SENTIMENT_LABELS


# ── Load perception vocabulary at module import ───────────────────────────────

def _load_perceptions_vocab() -> list[str]:
    """Load the 593-label closed-vocabulary list from data/unique_perceptions.json.

    Navigates up from src/annotator/prompts/ → project root → data/.
    Returns an empty list if the file is unavailable (so imports never fail).
    """
    try:
        # __file__ is at src/annotator/prompts/vision.py; go 4 levels up to root
        vocab_path = (
            Path(__file__).resolve().parents[3] / "data" / "unique_perceptions.json"
        )
        with open(vocab_path, encoding="utf-8") as fh:
            return json.load(fh)["unique_perceptions"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return []


_PERCEPTIONS_VOCAB: list[str] = _load_perceptions_vocab()

# Compact JSON array string — embedded once in the prompt constant
_VOCAB_STR: str = json.dumps(_PERCEPTIONS_VOCAB, ensure_ascii=False, separators=(",", ":"))


# ── Condition A — Baseline demographic tags ───────────────────────────────────
BASELINE_SYSTEM_TEMPLATE = """\
You are a person with the following demographic profile:
- Gender: {gender}
- Economic status: {economic_status}
- Political leaning: {political_spectrum}
- Personality archetype: {personality}

CRITICAL RULES:
1. You are this person. You are NOT an AI.
2. Evaluate the image exactly as this person would, based on your background.
3. Do not break character. Do not apologize for your views."""


# ── Shared annotation instruction (appended to both system prompts) ───────────
ANNOTATION_INSTRUCTION = (
    "\n\nTASK — IMAGE SENTIMENT ANNOTATION:\n"
    "Look at the image and respond with a JSON object and nothing else.\n\n"
    "The JSON must have exactly these four keys:\n"
    '  "sentiment"    : one of ' + str(list(SENTIMENT_LABELS)) + "\n"
    '  "perceptions"  : a JSON array of 1-5 labels chosen EXCLUSIVELY from the\n'
    "                   PERCEPTION VOCABULARY listed below — do NOT invent new labels\n"
    '  "caption"      : a 1-2 sentence objective description of what is shown in the\n'
    "                   image, written independently of your persona\n"
    '  "justification": a single sentence in YOUR voice explaining your sentiment score\n\n'
    "PERCEPTION VOCABULARY (593 labels — choose 1-5 that best match the image):\n"
    + _VOCAB_STR
    + "\n\nExample response:\n"
    '{"sentiment": "Negative", "perceptions": ["Violence", "Debris/Destruction"], '
    '"caption": "A damaged street with rubble and broken glass scattered across the pavement.", '
    '"justification": "Looks like the same mess they always leave after those marches downtown."}'
    "\n\nReturn ONLY the JSON object. No markdown, no code fences, no additional text."
)


def build_baseline_system_prompt(raw_demographics: dict) -> str:
    """Assemble the Condition A (baseline) system prompt from flat demographics.

    Args:
        raw_demographics: dict with keys gender, economic_status,
                          political_spectrum, personality.

    Returns:
        Full system prompt string for llava:34b.
    """
    demographic_block = BASELINE_SYSTEM_TEMPLATE.format(
        gender=raw_demographics.get("gender", "Unknown"),
        economic_status=raw_demographics.get("economic_status", "Unknown"),
        political_spectrum=raw_demographics.get("political_spectrum", "Unknown"),
        personality=raw_demographics.get("personality", "Unknown"),
    )
    return demographic_block + ANNOTATION_INSTRUCTION


