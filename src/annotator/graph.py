"""Annotation StateGraph: build_annotation_graph().

Graph topology (linear — no conditional branching needed here; retry logic
is internal to the annotator node):

    [image_loader] → [prompt_assembler] → [vision_annotator] → END

Usage
-----
    from annotator.graph import build_annotation_graph

    graph = build_annotation_graph()
    result = await graph.ainvoke(initial_state)
"""

from langgraph.graph import END, StateGraph

from annotator.nodes.annotator import vision_annotator
from annotator.nodes.assembler import prompt_assembler
from annotator.nodes.image_loader import image_loader
from annotator.state import AnnotationState

NODE_IMAGE_LOADER    = "image_loader"
NODE_PROMPT_ASSEMBLER = "prompt_assembler"
NODE_VISION_ANNOTATOR = "vision_annotator"


def build_annotation_graph() -> "CompiledGraph":  # noqa: F821
    """Construct and compile the annotation StateGraph.

    No checkpointer is used here: each (persona × image × condition) triple
    is an atomic unit.  If it fails, the pipeline orchestrator writes it to
    annotation_failures.jsonl and continues.

    Returns:
        A compiled LangGraph graph ready for ``ainvoke``.
    """
    graph = StateGraph(AnnotationState)

    graph.add_node(NODE_IMAGE_LOADER,     image_loader)
    graph.add_node(NODE_PROMPT_ASSEMBLER, prompt_assembler)
    graph.add_node(NODE_VISION_ANNOTATOR, vision_annotator)

    graph.set_entry_point(NODE_IMAGE_LOADER)
    graph.add_edge(NODE_IMAGE_LOADER,     NODE_PROMPT_ASSEMBLER)
    graph.add_edge(NODE_PROMPT_ASSEMBLER, NODE_VISION_ANNOTATOR)
    graph.add_edge(NODE_VISION_ANNOTATOR, END)

    return graph.compile()
