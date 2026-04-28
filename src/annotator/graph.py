"""Annotation StateGraph: build_annotation_graph().

Graph topology (linear — no conditional branching needed here; retry logic
is internal to the annotator node):

    [image_loader] → [prompt_assembler] → [vision_annotator] → END

Usage
-----
    # Default (baseline condition, qwen3-vl from settings, think=True)
    graph = build_annotation_graph()
    result = await graph.ainvoke(initial_state)

    # No-persona with custom model and think setting
    graph = build_annotation_graph(model="qwen3-vl:8b", think=False)
    result = await graph.ainvoke(initial_state)
"""

from langgraph.graph import END, StateGraph

from annotator.nodes.annotator import vision_annotator, build_annotator_node
from annotator.nodes.assembler import prompt_assembler
from annotator.nodes.image_loader import image_loader
from annotator.state import AnnotationState

NODE_IMAGE_LOADER     = "image_loader"
NODE_PROMPT_ASSEMBLER = "prompt_assembler"
NODE_VISION_ANNOTATOR = "vision_annotator"


def build_annotation_graph(
    model: str | None = None,
    think: bool = True,
    cfg=None,
) -> "CompiledGraph":  # noqa: F821
    """Construct and compile the annotation StateGraph.

    Args:
        model:  When provided, creates a new annotator node using this Ollama
                model instead of the default singleton (e.g. ``'qwen3-vl:8b'``).
        think:  Passed to the annotator node — controls Qwen3 extended thinking.
                Ignored when *model* is None (default node always uses think=True).
        cfg:    Optional ``AnnotatorSettings`` override; used when *model* is set.

    Returns:
        A compiled LangGraph graph ready for ``ainvoke``.
    """
    if model is not None:
        annotator_node = build_annotator_node(model=model, think=think, cfg=cfg)
    else:
        annotator_node = vision_annotator

    graph = StateGraph(AnnotationState)
    graph.add_node(NODE_IMAGE_LOADER,     image_loader)
    graph.add_node(NODE_PROMPT_ASSEMBLER, prompt_assembler)
    graph.add_node(NODE_VISION_ANNOTATOR, annotator_node)

    graph.set_entry_point(NODE_IMAGE_LOADER)
    graph.add_edge(NODE_IMAGE_LOADER,     NODE_PROMPT_ASSEMBLER)
    graph.add_edge(NODE_PROMPT_ASSEMBLER, NODE_VISION_ANNOTATOR)
    graph.add_edge(NODE_VISION_ANNOTATOR, END)

    return graph.compile()
