from typing import TypedDict, Optional, Annotated
import operator


class AgentState(TypedDict):
    # input
    query: str
    voice_input: Optional[bool]

    # retrieval
    retrieval_results: list[dict]
    graph_context: list[dict]

    # generation
    answer: str
    citations: list[str]
    confidence_score: float

    # guardrails
    hallucination_flags: list[str]
    regeneration_count: int

    # conversation
    conversation_history: Annotated[list[dict], operator.add]

    # metadata
    error: Optional[str]