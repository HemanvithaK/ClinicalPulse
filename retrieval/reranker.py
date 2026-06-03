from sentence_transformers import CrossEncoder
from retrieval.hybrid_retriever import HybridRetriever
import torch

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        print("Loading cross-encoder reranker...")
        self.model = CrossEncoder(
            MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("Reranker loaded")

    def rerank(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
        if not results:
            return []

        # build (query, passage) pairs for the cross-encoder
        pairs = [
            (query, r["payload"].get("chunk_text", r.get("title", "")))
            for r in results
        ]

        # score each pair
        scores = self.model.predict(pairs)

        # attach scores and sort
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever()
    reranker = Reranker()

    queries = [
        "Phase 3 breast cancer trials with tamoxifen",
        "diabetes insulin treatment recruiting",
        "alzheimer memory loss drug trials",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        # get hybrid results
        results = retriever.search(query, top_k=10)

        # rerank
        reranked = reranker.rerank(query, results, top_k=3)

        print(f"\nTop 3 after reranking:")
        for i, r in enumerate(reranked):
            print(f"  {i+1}. {r['title']}")
            print(f"     Rerank score : {r['rerank_score']:.4f}")
            print(f"     Source       : {r['source']}")

    retriever.close()