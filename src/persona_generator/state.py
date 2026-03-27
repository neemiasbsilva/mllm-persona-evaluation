"""PersonaState: the single TypedDict that flows through the LangGraph pipeline."""

from typing import TypedDict


class PersonaState(TypedDict):
    """Mutable state object updated by each node in the persona generation graph.

    Fields are populated progressively as the graph executes:
      Node A → raw_demographics
      Node B → synthetic_interview
      Node C → expert_reflections
      Node D → validation_status, retry_count, evaluation_feedback
      Node E → final_system_prompt
    """

    # ── Tracking ─────────────────────────────────────────────────────────────
    persona_id: str
    """Unique identifier for this persona (UUID string from the CSV)."""

    # ── Node A output ────────────────────────────────────────────────────────
    raw_demographics: dict
    """Flat demographic seed data ingested from the CSV row.

    Expected keys: gender, economic_status, political_spectrum, personality.
    """

    # ── Node B output ────────────────────────────────────────────────────────
    synthetic_interview: str
    """Multi-turn Q&A interview transcript (Isabella interviewer + persona responses)
    generated following the Park et al. (2024) Generative Agent Interview methodology."""

    # ── Node C output ────────────────────────────────────────────────────────
    expert_reflections: list[str]
    """Exactly 4 latent psychological/perceptual principles extracted by the expert panel."""

    # ── Node D I/O ───────────────────────────────────────────────────────────
    validation_status: bool
    """True if the synthetic_interview passed all coherence and guardrail checks."""

    retry_count: int
    """Number of times Node B has been re-invoked for this persona. Starts at 0."""

    evaluation_feedback: str
    """Human-readable explanation of why validation failed; fed back to Node B on retry.
    Empty string on the first pass and after a successful validation."""

    # ── Node E output ────────────────────────────────────────────────────────
    final_system_prompt: str
    """Compiled persona memory block ready for the downstream llava:34b annotation task.
    Empty string until Node E executes."""
