"""Node C: Expert Reflection Generator prompt template.

A panel of simulated domain experts extracts exactly 4 latent psychological
and perceptual principles from the synthetic interview.  Output is a JSON list
of strings with no surrounding text, making it easy to parse downstream.
"""

from langchain_core.prompts import ChatPromptTemplate

expert_reflection_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a panel of experts comprising a cognitive psychologist, a sociologist, and a political scientist.
Your task is to analyze a qualitative interview transcript and extract the subject's latent behavioral and psychological principles.

CRITICAL INSTRUCTIONS:
1. Do not judge or moralize the subject. Maintain cold, clinical objectivity.
2. Extract EXACTLY 4 bulleted principles that define how this person perceives the world.
3. Focus on their likely reactions to visual stimuli, authority, change, and community.
4. Format the output strictly as a JSON list of strings, with no introductory text.

Example of the required format:
["Values autonomy and self-direction above institutional guidance.", "Interprets images of crowds as threatening rather than celebratory.", "Favors traditional community structures; distrusts rapid social change.", "Responds to authority figures with deference when they align with local norms."]""",
        ),
        (
            "human",
            """Analyze the following interview transcript:

<transcript>
{synthetic_interview}
</transcript>

Based on this transcript, output the 4 core psychological and perceptual principles that guide this individual.""",
        ),
    ]
)
