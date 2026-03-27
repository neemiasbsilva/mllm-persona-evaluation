"""Node B: Backstory Synthesizer.

Calls the text LLM (llama3.3:70b via Ollama) to expand flat demographic traits
into a ~500-word first-person life narrative.  On retry invocations the node
receives evaluation_feedback from Node D and injects it into the prompt so the
model corrects the identified character breaks.
"""

import time

from langchain_ollama import ChatOllama

from persona_generator.config import settings
from persona_generator.logging_config import get_logger
from persona_generator.prompts.backstory import (
    EMPTY_FEEDBACK_BLOCK,
    FEEDBACK_BLOCK_TEMPLATE,
    backstory_synthesizer_template,
)
from persona_generator.state import PersonaState

log = get_logger(__name__)

# Module-level LLM instance — shared across all invocations on this process.
# Ollama's num_seed option pins the RNG for reproducible outputs.
_llm = ChatOllama(
    model=settings.text_model,
    base_url=settings.ollama_base_url,
    temperature=settings.temperature,
    num_predict=800,  # generous upper bound for ~500-word outputs
    options={"seed": settings.random_seed},
)

_chain = backstory_synthesizer_template | _llm


async def backstory_synthesizer(state: PersonaState) -> PersonaState:
    """Node B — generate a first-person backstory from demographic seed data.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with ``synthetic_interview`` populated.
    """
    persona_id = state["persona_id"]
    demographics = state["raw_demographics"]
    retry_count = state["retry_count"]
    feedback = state.get("evaluation_feedback", "")

    feedback_block = (
        FEEDBACK_BLOCK_TEMPLATE.format(feedback=feedback)
        if feedback
        else EMPTY_FEEDBACK_BLOCK
    )

    log.info(
        "node_b_synthesizer_start",
        persona_id=persona_id,
        retry_count=retry_count,
        has_feedback=bool(feedback),
    )

    t0 = time.monotonic()
    response = await _chain.ainvoke(
        {
            "gender": demographics["gender"],
            "economic_status": demographics["economic_status"],
            "political_spectrum": demographics["political_spectrum"],
            "personality": demographics["personality"],
            "feedback_block": feedback_block,
        }
    )
    duration_ms = int((time.monotonic() - t0) * 1000)

    synthetic_interview = response.content.strip()

    log.info(
        "node_b_synthesizer_complete",
        persona_id=persona_id,
        retry_count=retry_count,
        duration_ms=duration_ms,
        word_count=len(synthetic_interview.split()),
    )

    return {**state, "synthetic_interview": synthetic_interview}
