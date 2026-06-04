from anthropic import Anthropic
from dotenv import load_dotenv
from agent.state import AgentState
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
import os
import json

load_dotenv()

claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
retriever = HybridRetriever()
reranker = Reranker()


def query_planner(state: AgentState) -> AgentState:
    """Classifies query intent and plans retrieval strategy."""
    print(f"\n[Planner] Query: {state['query']}")

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Analyze this clinical trial query and return JSON only:
Query: {state['query']}

Return:
{{
  "intent": "one of: find_trials, compare_drugs, check_eligibility, get_status",
  "keywords": ["key", "terms", "for", "search"],
  "filters": {{"phase": "Phase 3 or null", "status": "RECRUITING or null"}}
}}"""
        }]
    )

    try:
        raw = response.content[0].text
        clean = raw.replace("```json", "").replace("```", "").strip()
        plan = json.loads(clean)
        print(f"[Planner] Intent: {plan.get('intent')}, Keywords: {plan.get('keywords')}")
    except Exception:
        plan = {"intent": "find_trials", "keywords": [], "filters": {}}

    return {
        **state,
        "retrieval_results": [],
        "graph_context": [],
        "hallucination_flags": [],
        "regeneration_count": state.get("regeneration_count", 0),
        "conversation_history": [],
    }


def hybrid_retriever_node(state: AgentState) -> AgentState:
    """Runs hybrid retrieval and reranking."""
    print(f"\n[Retriever] Searching...")

    raw_results = retriever.search(state["query"], top_k=10)
    reranked = reranker.rerank(state["query"], raw_results, top_k=5)

    print(f"[Retriever] Top result: {reranked[0]['title'] if reranked else 'none'}")

    return {
        **state,
        "retrieval_results": reranked,
    }


def synthesizer(state: AgentState) -> AgentState:
    """Generates a cited answer using Claude."""
    print(f"\n[Synthesizer] Generating answer...")

    results = state["retrieval_results"]
    if not results:
        return {
            **state,
            "answer": "I could not find relevant clinical trials for your query.",
            "citations": [],
            "confidence_score": 0.0,
        }

    # build context block
    context_parts = []
    citations = []
    for i, r in enumerate(results):
        payload = r.get("payload", {})
        chunk = payload.get("chunk_text", r.get("title", ""))
        nct_id = r.get("nct_id", "")
        title = r.get("title", "")
        citations.append(f"[{i+1}] {nct_id}: {title}")
        context_parts.append(f"[{i+1}] {chunk}")

    context = "\n\n".join(context_parts)

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        system="""You are ClinicalPulse, a clinical trial intelligence assistant.
Answer questions using ONLY the provided trial context.
Always cite sources using [1], [2] etc.
Be concise, accurate, and clinically precise.
If the context doesn't fully answer the question, say so clearly.""",
        messages=[
            *[{"role": m["role"], "content": m["content"]}
              for m in state.get("conversation_history", [])],
            {
                "role": "user",
                "content": f"""Context:
{context}

Question: {state['query']}

Answer using only the context above and cite sources."""
            }
        ]
    )

    answer = response.content[0].text
    print(f"[Synthesizer] Answer generated ({len(answer)} chars)")

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "confidence_score": 0.85,
    }


def guardrails_checker(state: AgentState) -> AgentState:
    """Checks answer for hallucinations against retrieved context."""
    print(f"\n[Guardrails] Checking answer...")

    context = "\n".join([
        r.get("payload", {}).get("chunk_text", "")
        for r in state["retrieval_results"]
    ])

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""Check if this answer is fully supported by the context.
Return JSON only.

Context:
{context[:3000]}

Answer:
{state['answer']}

Return:
{{
  "is_grounded": true or false,
  "unsupported_claims": ["list any claims not in context"],
  "confidence": 0.0 to 1.0
}}"""
        }]
    )

    try:
        raw = response.content[0].text
        clean = raw.replace("```json", "").replace("```", "").strip()
        check = json.loads(clean)
        flags = check.get("unsupported_claims", [])
        confidence = check.get("confidence", 0.85)
        is_grounded = check.get("is_grounded", True)
        print(f"[Guardrails] Grounded: {is_grounded}, Flags: {len(flags)}")
    except Exception:
        flags = []
        confidence = 0.85
        is_grounded = True

    return {
        **state,
        "hallucination_flags": flags,
        "confidence_score": confidence,
    }


def should_regenerate(state: AgentState) -> str:
    """Decides whether to regenerate or return the answer."""
    flags = state.get("hallucination_flags", [])
    count = state.get("regeneration_count", 0)

    if flags and count < 2:
        print(f"[Router] Regenerating — {len(flags)} unsupported claims (attempt {count+1})")
        return "regenerate"

    print(f"[Router] Answer accepted")
    return "accept"