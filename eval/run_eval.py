import json
import pathlib
import sys
import os

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
from anthropic import Anthropic
from agent.graph import build_graph

load_dotenv()

EVAL_SET_FILE = pathlib.Path("eval/eval_set.json")
RESULTS_FILE = pathlib.Path("eval/eval_results.json")

claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
graph = build_graph()


def run_agent(query: str) -> dict:
    return graph.invoke({
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


def llm_judge(question: str, ground_truth: str, answer: str) -> dict:
    """Uses Claude to score the answer on 3 dimensions."""
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""Score this answer on 3 dimensions. Return JSON only.

Question: {question}
Ground truth: {ground_truth}
Answer to evaluate: {answer}

Return:
{{
  "faithfulness": 0.0-1.0,
  "relevance": 0.0-1.0,
  "completeness": 0.0-1.0,
  "reasoning": "one sentence explanation"
}}

Faithfulness: is the answer factually consistent with ground truth?
Relevance: does the answer address the question?
Completeness: does it cover the key points in ground truth?"""
        }]
    )

    try:
        raw = response.content[0].text
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {
            "faithfulness": 0.5,
            "relevance": 0.5,
            "completeness": 0.5,
            "reasoning": "Could not parse judge response"
        }


def run_evaluation():
    eval_set = json.loads(EVAL_SET_FILE.read_text())
    print(f"Running evaluation on {len(eval_set)} questions...\n")

    results = []
    total_faithfulness = 0
    total_relevance = 0
    total_completeness = 0

    for i, item in enumerate(eval_set):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i+1}/{len(eval_set)}] {question}")

        # run agent
        result = run_agent(question)
        answer = result["answer"]
        citations = result["citations"]
        confidence = result["confidence_score"]
        flags = result["hallucination_flags"]

        # judge
        scores = llm_judge(question, ground_truth, answer)

        total_faithfulness += scores["faithfulness"]
        total_relevance += scores["relevance"]
        total_completeness += scores["completeness"]

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "citations": citations,
            "confidence_score": confidence,
            "hallucination_flags": flags,
            "scores": scores,
        })

        print(f"  Faithfulness : {scores['faithfulness']:.2f}")
        print(f"  Relevance    : {scores['relevance']:.2f}")
        print(f"  Completeness : {scores['completeness']:.2f}")
        print(f"  Reasoning    : {scores['reasoning']}\n")

    n = len(eval_set)
    summary = {
        "total_questions": n,
        "avg_faithfulness": round(total_faithfulness / n, 3),
        "avg_relevance": round(total_relevance / n, 3),
        "avg_completeness": round(total_completeness / n, 3),
        "avg_composite": round((total_faithfulness + total_relevance + total_completeness) / (n * 3), 3),
        "results": results,
    }

    RESULTS_FILE.write_text(json.dumps(summary, indent=2))

    print("="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Faithfulness  : {summary['avg_faithfulness']:.3f}")
    print(f"Relevance     : {summary['avg_relevance']:.3f}")
    print(f"Completeness  : {summary['avg_completeness']:.3f}")
    print(f"Composite     : {summary['avg_composite']:.3f}")
    print(f"\nFull results saved to eval/eval_results.json")


if __name__ == "__main__":
    run_evaluation()