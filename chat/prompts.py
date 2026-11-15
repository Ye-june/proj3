"""Choose how Harbor answers based on the type of question (story 8).

Each question is classified at runtime into one intent. The intent decides:

* the system prompt sent to the language model (``build_system_prompt``),
* how a source passage is shaped when no model is configured (``shape_passage``),
* whether Harbor puts the emergency line first (``urgent``).

Classification is rule-based so it is fast, free, testable, and explainable:
the first intent whose phrases appear in the question wins, checked in the
order of ``INTENTS``. A question matching nothing is ``factual``.
"""

from __future__ import annotations

from chat.grounding import sentences
from chat.policy import SYSTEM_PROMPT, normalize_for_match

URGENT_LINE = "If you are in immediate danger, call 911 or local emergency services now."

# Order matters: urgent beats everything, "why should we..." is an explanation and
# "how much water should we store" is a fact, not steps. No match at all is also factual.
INTENTS = {
    "urgent": {
        "label": "Urgent",
        "phrases": (
            "trapped", "stuck in", "water is rising", "rising water", "injured", "bleeding",
            "cant breathe", "can not breathe", "help me now", "drowning",
            "under rubble", "building collapsed", "house is on fire", "fire is coming",
        ),
        "instructions": (
            "The user may be in immediate danger. Harbor already shows the emergency number first, "
            "so do not repeat it. Give at most three short actions from the documents, most important first. "
            "No background and no supply lists."
        ),
    },
    "creative": {
        "label": "Written for you",
        "phrases": (
            "write", "draft", "compose", "text message", "message to my", "message for my",
            "make a plan", "make us a plan", "create a plan", "help me plan", "poem",
        ),
        "instructions": (
            "The user wants something written for them, such as a household plan or a message to send. "
            "Write it in the form they asked for, in plain, calm words. Every fact must still come from the documents. "
            "Do not invent names, places, phone numbers, or times."
        ),
    },
    "explanation": {
        "label": "Explanation",
        "phrases": (
            "why", "explain", "difference between", "whats the difference", "compare", "how does",
            "how do floods", "how do earthquakes", "what causes", "what does",
        ),
        "instructions": (
            "The user wants to understand why or how something works. Answer in one short paragraph "
            "that explains the reasoning the documents give. Do not add reasons the documents do not state."
        ),
    },
    "factual": {
        "label": "Quick fact",
        "phrases": ("how much", "how many", "how long", "how far", "how often", "what time"),
        "instructions": (
            "The user wants a specific fact. Answer in one to three sentences. Quote numbers, amounts, "
            "and times exactly as the documents state them. Do not add background."
        ),
    },
    "steps": {
        "label": "Step-by-step",
        "phrases": (
            "how do i", "how do we", "how should", "how to", "what should i do", "what should we do",
            "what should go in", "what to pack", "steps", "checklist", "should we", "should i",
            "prepare", "get ready",
        ),
        "instructions": (
            "The user wants to know what to do. Answer as a numbered list of short steps, "
            "one action per step, in the order the documents give them."
        ),
    },
}


def classify_intent(message: str) -> str:
    padded = f" {normalize_for_match(message)} "
    for intent, spec in INTENTS.items():
        if any(f" {phrase} " in padded for phrase in spec["phrases"]):
            return intent
    return "factual"


def intent_label(intent: str) -> str:
    return INTENTS[intent]["label"]


def build_system_prompt(intent: str) -> str:
    """The shared safety rules plus the response style for this intent."""
    spec = INTENTS[intent]
    return f"{SYSTEM_PROMPT}\nResponse style for this question ({spec['label']}):\n- {spec['instructions']}\n"


def shape_passage(body: str, intent: str, question: str) -> str:
    """Shape a source passage for the intent without adding any words of our own.

    Used when no language model is configured, so the intent still changes the answer.
    Explanation and creative questions get the whole passage: rearranging it would not explain more.
    """
    if intent == "steps":
        return "\n".join(f"{n}. {s}" for n, s in enumerate(sentences(body), start=1))
    if intent in ("factual", "urgent"):
        return _best_paragraph(body, question)
    return body


def _best_paragraph(body: str, question: str) -> str:
    # ponytail: word-overlap scoring; swap for embeddings if paragraphs get long or numerous.
    wanted = {w for w in normalize_for_match(question).split() if len(w) > 3}
    paragraphs = [p for p in body.split("\n\n") if p.strip()] or [body]
    return max(paragraphs, key=lambda p: len(wanted & set(normalize_for_match(p).split())))
