"""Node A: Demographic Seeder.

Ingests a single demographics dictionary (parsed from a CSV row) and maps it
into the PersonaState, initialising all downstream fields to safe defaults so
the rest of the graph receives a fully-typed state from the first step.
"""

from persona_generator.logging_config import get_logger
from persona_generator.state import PersonaState

log = get_logger(__name__)

REQUIRED_DEMOGRAPHIC_KEYS = {
    "gender",
    "economic_status",
    "political_spectrum",
    "personality",
}


def demographic_seeder(state: PersonaState) -> PersonaState:
    """Node A — validate and register raw demographics in the state.

    This node is intentionally lightweight: it validates that the required
    keys are present and initialises every downstream field so that LangGraph
    never encounters a KeyError when reading the state in later nodes.

    Args:
        state: Incoming state, expected to have ``raw_demographics`` and
               ``persona_id`` already set by the pipeline orchestrator.

    Returns:
        Updated state with all fields initialised.

    Raises:
        ValueError: If any required demographic key is missing.
    """
    persona_id = state["persona_id"]
    demographics = state["raw_demographics"]

    missing = REQUIRED_DEMOGRAPHIC_KEYS - set(demographics.keys())
    if missing:
        raise ValueError(
            f"[Persona {persona_id}] Missing required demographic keys: {missing}"
        )

    log.info(
        "node_a_seeder_complete",
        persona_id=persona_id,
        gender=demographics.get("gender"),
        economic_status=demographics.get("economic_status"),
        political_spectrum=demographics.get("political_spectrum"),
        personality=demographics.get("personality"),
    )

    return {
        **state,
        # Downstream fields initialised to safe defaults
        "synthetic_interview": "",
        "expert_reflections": [],
        "validation_status": False,
        "retry_count": 0,
        "evaluation_feedback": "",
        "final_system_prompt": "",
    }
