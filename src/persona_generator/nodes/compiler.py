"""Node E: Prompt Compiler (terminal node).

Assembles the validated synthetic_interview and expert_reflections into the
immutable final_system_prompt string.  No LLM call is made here — this is
a pure string-formatting step.

The compiled prompt is the artefact written to personas.jsonl and consumed
by the downstream llava:34b vision annotation task.
"""

from persona_generator.logging_config import get_logger
from persona_generator.prompts.compiler import compile_final_system_prompt
from persona_generator.state import PersonaState

log = get_logger(__name__)


def prompt_compiler(state: PersonaState) -> PersonaState:
    """Node E — compile the final persona system prompt.

    Args:
        state: Current pipeline state with ``synthetic_interview`` and
               ``expert_reflections`` validated and populated.

    Returns:
        Updated state with ``final_system_prompt`` populated.
    """
    persona_id = state["persona_id"]

    final_prompt = compile_final_system_prompt(
        synthetic_interview=state["synthetic_interview"],
        expert_reflections=state["expert_reflections"],
    )

    log.info(
        "node_e_compiler_complete",
        persona_id=persona_id,
        prompt_length=len(final_prompt),
    )

    return {**state, "final_system_prompt": final_prompt}
