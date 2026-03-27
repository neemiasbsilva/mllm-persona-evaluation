"""Node C: Expert Reflection Generator.

Prompts the LLM to adopt a panel of domain experts (cognitive psychologist,
sociologist, political scientist) who analyse the synthetic_interview and
extract exactly 4 latent psychological/perceptual principles as a JSON list.
"""

import json
import time

from langchain_ollama import ChatOllama

from persona_generator.config import settings
from persona_generator.logging_config import get_logger
from persona_generator.prompts.reflection import expert_reflection_template
from persona_generator.state import PersonaState

log = get_logger(__name__)

_llm = ChatOllama(
    model=settings.text_model,
    base_url=settings.ollama_base_url,
    temperature=0.3,  # lower temperature for clinical, consistent extraction
    num_predict=400,
    options={"seed": settings.random_seed},
)

_chain = expert_reflection_template | _llm


def _parse_reflections(raw: str) -> list[str]:
    """Extract the JSON list of principles from the model's raw output.

    The model is instructed to return only a JSON list.  We strip surrounding
    whitespace/markdown fences before parsing.
    """
    text = raw.strip()

    # Strip optional markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            line
            for line in lines
            if not line.startswith("```")
        ).strip()

    principles: list[str] = json.loads(text)

    if not isinstance(principles, list) or len(principles) != 4:
        raise ValueError(
            f"Expected a JSON list of exactly 4 strings, got: {principles!r}"
        )

    return [str(p) for p in principles]


async def expert_reflector(state: PersonaState) -> PersonaState:
    """Node C — extract 4 latent psychological principles from the interview.

    Args:
        state: Current pipeline state with ``synthetic_interview`` populated.

    Returns:
        Updated state with ``expert_reflections`` populated.
    """
    persona_id = state["persona_id"]

    log.info("node_c_reflector_start", persona_id=persona_id)

    t0 = time.monotonic()
    response = await _chain.ainvoke(
        {"synthetic_interview": state["synthetic_interview"]}
    )
    duration_ms = int((time.monotonic() - t0) * 1000)

    try:
        principles = _parse_reflections(response.content)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning(
            "node_c_reflector_parse_error",
            persona_id=persona_id,
            error=str(exc),
            raw_output=response.content[:300],
        )
        # Fallback: wrap raw output in a single-element list so the pipeline
        # can continue; Node D will catch quality issues if any.
        principles = [response.content.strip()]

    log.info(
        "node_c_reflector_complete",
        persona_id=persona_id,
        duration_ms=duration_ms,
        num_principles=len(principles),
    )

    return {**state, "expert_reflections": principles}
