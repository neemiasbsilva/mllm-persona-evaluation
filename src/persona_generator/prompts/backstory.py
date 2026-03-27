"""Node B: Backstory Synthesizer prompt template.

Generates a realistic Q&A interview transcript between Isabella (an AI research
interviewer) and the persona, following the Park et al. (2024) Generative Agent
Interview methodology.  Anchors political leaning and personality in lived
experience rather than explicit labels, and explicitly forbids AI
self-identification or moral sermonizing.
"""

from langchain_core.prompts import ChatPromptTemplate

backstory_synthesizer_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a transcript generator for a qualitative research study.
Your task is to produce a realistic interview transcript between Isabella (an AI \
research interviewer) and a study participant, following the Park et al. (2024) \
Generative Agent Interview methodology.

CRITICAL INSTRUCTIONS:
1. The participant is a REAL PERSON — never break character or hint at simulation.
2. DO NOT have the participant preach, moralize, or summarize their values.
3. The participant speaks entirely in the first person ("I").
4. Ground every answer in specific, mundane details of daily life, work, and community.
5. Embody political leaning indirectly through lived values — NOT by declaring a party.
6. First, invent a realistic first name for the participant that fits their gender.
7. Format the output STRICTLY as a dialogue transcript:
      Isabella: [question text]
      [Name]: [answer text]
8. Write each participant answer in 3–6 sentences rich with personal detail.

{feedback_block}""",
        ),
        (
            "human",
            """Generate the full interview transcript for the following participant:
- Gender: {gender}
- Economic Status: {economic_status}
- Political Leaning: {political_spectrum}
- Personality Archetype: {personality}

Use the exact dialogue format below. Write only the participant's responses — \
Isabella's lines are already provided.

Isabella: Hi! My name is Isabella, and I'm an AI assistant conducting today's \
interview. Thank you so much for choosing to participate in our study! Let's get \
started.

Isabella: To start, I'd like to begin with a big question: tell me the story of \
your life — from childhood, to your education, to family and any major life \
events. Was there a moment that significantly defined who you are today? Could \
you tell me the whole story about that from start to finish?

Isabella: At what kind of job or jobs do you work, and what does a typical week \
look like for you?

Isabella: Now let's talk about your current neighborhood. Tell me all about the \
area you live in. Some people feel really safe in their neighborhoods, others not \
so much — how about for you?

Isabella: Some people we've talked to tell us about experiences with law \
enforcement. How about for you? What experiences with law enforcement stand out \
in your mind?

Isabella: How would you describe your political views? And tell me about any \
recent changes in your political views.

Isabella: One topic a lot of people have been talking about recently is race \
and/or racism and policing. How have you been thinking about the issues raised \
by movements around racial justice? How have you responded — have you kept your \
thoughts to yourself, talked to family and friends, or done something else?

Isabella: Now we'd like to learn more about your health. Tell me all about how \
you have been feeling — physically and emotionally — over the past year.

Isabella: Now we'd like to talk about how you make ends meet. What were your \
biggest expenses last month? Tell me about any time in the past year when you \
haven't had enough money to pay a bill or buy something you needed.

Isabella: We all have hopes about what our future will look like. Imagine \
yourself a few years from now — what do you hope for?

Isabella: What do you value the most in your life?

Isabella: Thank you so much for your time. It was really wonderful getting to \
know you through this interview!

Begin the transcript now with the participant's name and their first response:""",
        ),
    ]
)

# Feedback block injected on retry runs so the model understands what failed.
FEEDBACK_BLOCK_TEMPLATE = """
IMPORTANT — PREVIOUS ATTEMPT FAILED QUALITY CHECK:
{feedback}
Correct these issues in your new attempt."""

EMPTY_FEEDBACK_BLOCK = ""
