"""Node: prompt_assembler.

Constructs the system prompt from the persona's flat demographic tags
and stores it in ``state['system_prompt']``.
"""

from annotator.prompts.vision import build_baseline_system_prompt, build_no_persona_system_prompt
from annotator.state import AnnotationState, CONDITIONS
from persona_generator.logging_config import get_logger

log = get_logger(__name__)


def prompt_assembler(state: AnnotationState) -> AnnotationState:
    """Assemble the system prompt for this (persona, condition) pair.

    Args:
        state: Must have ``condition``, ``persona_record`` set.

    Returns:
        Updated state with ``system_prompt`` populated.

    Raises:
        ValueError: If ``condition`` is not one of the expected values.
    """
    condition = state["condition"]
    persona = state["persona_record"]

    if condition not in CONDITIONS:
        raise ValueError(
            f"Unknown condition '{condition}'. Expected one of {CONDITIONS}."
        )

    if condition == "baseline":
        system_prompt = build_baseline_system_prompt(persona["raw_demographics"])
    elif condition in ("no_persona_think", "no_persona_no_think"):
        system_prompt = build_no_persona_system_prompt()
    else:
        raise ValueError(
            f"Unknown condition '{condition}'. Expected one of {CONDITIONS}."
        )

    log.debug(
        "prompt_assembler_complete",
        persona_id=state["persona_id"],
        image_id=state["image_id"],
        condition=condition,
        prompt_chars=len(system_prompt),
    )

    return {**state, "system_prompt": system_prompt}
