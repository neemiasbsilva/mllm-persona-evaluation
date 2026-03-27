"""LangGraph StateGraph composition: build_persona_graph().

Graph topology
--------------

    [seeder] → [synthesizer] → [reflector] → [evaluator]
                    ↑                               |
                    |      (fail, retries left)     |
                    └───────────────────────────────┘
                                                    |
                                          (pass) → [compiler] → END
                                          (retries exhausted)  → END

The conditional router reads validation_status and retry_count from the state
and selects the next node accordingly.

Usage
-----
    from persona_generator.graph import build_persona_graph

    compiled = build_persona_graph(checkpointer=my_checkpointer)
    result = await compiled.ainvoke(initial_state, config={"configurable": {"thread_id": "42"}})
"""

from langgraph.graph import END, StateGraph

from persona_generator.config import settings
from persona_generator.nodes.compiler import prompt_compiler
from persona_generator.nodes.evaluator import coherence_evaluator
from persona_generator.nodes.reflector import expert_reflector
from persona_generator.nodes.seeder import demographic_seeder
from persona_generator.nodes.synthesizer import backstory_synthesizer
from persona_generator.state import PersonaState

# ── Node name constants (avoids magic strings scattered across the codebase) ──
NODE_SEEDER = "seeder"
NODE_SYNTHESIZER = "synthesizer"
NODE_REFLECTOR = "reflector"
NODE_EVALUATOR = "evaluator"
NODE_COMPILER = "compiler"


def _route_from_evaluator(state: PersonaState) -> str:
    """Conditional edge function called after Node D.

    Returns the name of the next node based on validation outcome:
      - Passed → compiler (Node E)
      - Failed, retries remaining → synthesizer (Node B, retry loop)
      - Failed, retries exhausted → END (discard; pipeline records as failed)
    """
    if state["validation_status"]:
        return NODE_COMPILER

    if state["retry_count"] < settings.max_retries:
        return NODE_SYNTHESIZER

    # Retry budget exhausted — surface failure to the pipeline orchestrator.
    return END


def build_persona_graph(checkpointer=None) -> "CompiledGraph":  # noqa: F821
    """Construct and compile the persona generation StateGraph.

    Args:
        checkpointer: An optional LangGraph checkpointer instance (e.g.
                      SqliteSaver).  When provided, every super-step is
                      persisted, enabling fault-tolerant resumption.

    Returns:
        A compiled LangGraph graph ready for ``ainvoke`` / ``invoke``.
    """
    graph = StateGraph(PersonaState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node(NODE_SEEDER, demographic_seeder)
    graph.add_node(NODE_SYNTHESIZER, backstory_synthesizer)
    graph.add_node(NODE_REFLECTOR, expert_reflector)
    graph.add_node(NODE_EVALUATOR, coherence_evaluator)
    graph.add_node(NODE_COMPILER, prompt_compiler)

    # ── Define edges ──────────────────────────────────────────────────────────
    graph.set_entry_point(NODE_SEEDER)

    graph.add_edge(NODE_SEEDER, NODE_SYNTHESIZER)
    graph.add_edge(NODE_SYNTHESIZER, NODE_REFLECTOR)
    graph.add_edge(NODE_REFLECTOR, NODE_EVALUATOR)

    # Conditional routing after the evaluator
    graph.add_conditional_edges(
        NODE_EVALUATOR,
        _route_from_evaluator,
        {
            NODE_COMPILER: NODE_COMPILER,
            NODE_SYNTHESIZER: NODE_SYNTHESIZER,
            END: END,
        },
    )

    graph.add_edge(NODE_COMPILER, END)

    # ── Compile ───────────────────────────────────────────────────────────────
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)
