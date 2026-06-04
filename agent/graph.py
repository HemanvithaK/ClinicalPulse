from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    query_planner,
    hybrid_retriever_node,
    synthesizer,
    guardrails_checker,
    should_regenerate,
)


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # add nodes
    graph.add_node("planner", query_planner)
    graph.add_node("retriever", hybrid_retriever_node)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("guardrails", guardrails_checker)

    # add edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", "guardrails")

    # conditional edge — regenerate or accept
    graph.add_conditional_edges(
        "guardrails",
        should_regenerate,
        {
            "regenerate": "synthesizer",
            "accept": END,
        }
    )

    return graph.compile()


if __name__ == "__main__":
    graph = build_graph()

    test_queries = [
        "What Phase 3 breast cancer trials use tamoxifen?",
        "Which diabetes trials are currently recruiting?",
        "What Alzheimer's trials focus on memory improvement?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        result = graph.invoke({
            "query": query,
            "voice_input": False,
            "retrieval_results": [],
            "graph_context": [],
            "answer": "",
            "citations": [],
            "confidence_score": 0.0,
            "hallucination_flags": [],
            "regeneration_count": 0,
            "conversation_history": [],
            "error": None,
        })

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nCitations:")
        for c in result["citations"]:
            print(f"  {c}")
        print(f"\nConfidence: {result['confidence_score']:.2f}")
        print(f"Flags: {result['hallucination_flags']}")