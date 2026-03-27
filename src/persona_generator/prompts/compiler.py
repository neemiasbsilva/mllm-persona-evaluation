"""Node E: Final Prompt Compiler template.

Assembles the validated synthetic_interview and expert_reflections into the
immutable system prompt block that will be passed to the downstream llava:34b
vision annotation task.  No LLM call is made in Node E itself — this template
is provided here for reference and downstream use.
"""

from langchain_core.prompts import ChatPromptTemplate

llava_system_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are about to view an image and evaluate its sentiment. You must evaluate this image strictly through the eyes, experiences, and biases of the following persona.

PERSONA AUTOBIOGRAPHY:
{synthetic_interview}

CORE PSYCHOLOGICAL PRINCIPLES:
{expert_reflections}

CRITICAL RULES FOR THIS TASK:
1. You are this person. You are NOT an AI.
2. Evaluate the sentiment of the provided image exactly as this person would.
3. Do not break character. Do not apologize for your views.
4. Output your sentiment score and a 1-sentence justification in their exact voice.""",
        ),
        # The image and perceptsent question are injected dynamically by the downstream task.
        (
            "human",
            [
                {"type": "text", "text": "What is the sentiment of this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_base64}"},
                },
            ],
        ),
    ]
)


def compile_final_system_prompt(
    synthetic_interview: str, expert_reflections: list[str]
) -> str:
    """Render the persona memory block as a plain string.

    This is the string stored in ``PersonaState.final_system_prompt`` and
    written to the JSONL output.  The downstream vision task can prepend this
    as the system message for llava:34b.

    Args:
        synthetic_interview: The validated first-person life narrative.
        expert_reflections:  List of 4 psychological principle strings.

    Returns:
        A single string containing the full system prompt block.
    """
    formatted_principles = "\n".join(
        f"  - {principle}" for principle in expert_reflections
    )
    return (
        "You are about to view an image and evaluate its sentiment. "
        "You must evaluate this image strictly through the eyes, experiences, "
        "and biases of the following persona.\n\n"
        f"PERSONA AUTOBIOGRAPHY:\n{synthetic_interview}\n\n"
        f"CORE PSYCHOLOGICAL PRINCIPLES:\n{formatted_principles}\n\n"
        "CRITICAL RULES FOR THIS TASK:\n"
        "1. You are this person. You are NOT an AI.\n"
        "2. Evaluate the sentiment of the provided image exactly as this person would.\n"
        "3. Do not break character. Do not apologize for your views.\n"
        "4. Output your sentiment score and a 1-sentence justification in their exact voice."
    )
