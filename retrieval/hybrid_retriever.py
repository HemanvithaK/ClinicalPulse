from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import os
import json
import pathlib

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "clinical123")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "clinical_trials"
BM25_CORPUS_FILE = pathlib.Path("retrieval/bm25_corpus.json")


class HybridRetriever:
    def __init__(self):
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.neo4j = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.bm25 = None
        self.bm25_docs = []
        self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        if BM25_CORPUS_FILE.exists():
            data = json.loads(BM25_CORPUS_FILE.read_text())
            self.bm25_docs = data
            tokenized = [doc["text"].lower().split() for doc in data]
            self.bm25 = BM25Okapi(tokenized)
            print(f"BM25 loaded — {len(self.bm25_docs)} documents")
        else:
            print("No BM25 corpus found — run build_bm25_corpus() first")

    def build_bm25_corpus(self):
        """Build BM25 index from Qdrant payloads."""
        print("Building BM25 corpus from Qdrant...")
        results = self.qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        docs = []
        for point in results[0]:
            payload = point.payload
            text = payload.get("chunk_text", "")
            if text:
                docs.append({
                    "id": str(point.id),
                    "nct_id": payload.get("nct_id"),
                    "title": payload.get("title"),
                    "text": text,
                    "payload": payload,
                })
        self.bm25_docs = docs
        tokenized = [doc["text"].lower().split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized)
        BM25_CORPUS_FILE.write_text(json.dumps(docs, indent=2))
        print(f"BM25 corpus built — {len(docs)} documents")

    def _embed_query(self, query: str) -> list[float]:
        response = self.openai.embeddings.create(
            input=query,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding

    def vector_search(self, query: str, top_k: int = 10) -> list[dict]:
        vector = self._embed_query(query)
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True,)
    
        return [
            {
            "nct_id": r.payload.get("nct_id"),
            "title": r.payload.get("title"),
            "score": r.score,
            "source": "vector",
            "payload": r.payload,
        }
        for r in results.points
    ]

    def bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            {
                "nct_id": self.bm25_docs[i]["nct_id"],
                "title": self.bm25_docs[i]["title"],
                "score": float(scores[i]),
                "source": "bm25",
                "payload": self.bm25_docs[i]["payload"],
            }
            for i in top_indices
            if scores[i] > 0
        ]

    def graph_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Extract keywords from query and find related trials via Neo4j."""
        keywords = [w for w in query.lower().split()
                   if len(w) > 3 and w not in
                   {"what", "which", "where", "when", "that", "this", "with", "have", "from"}]

        if not keywords:
            return []

        results = []
        with self.neo4j.session() as session:
            for keyword in keywords[:3]:  # top 3 keywords
                records = session.run("""
                    MATCH (t:Trial)-[:TREATS]->(c:Condition)
                    WHERE toLower(c.name) CONTAINS $keyword
                       OR toLower(t.title) CONTAINS $keyword
                    RETURN t.nct_id as nct_id,
                           t.title as title,
                           t.status as status,
                           t.phase as phase,
                           collect(c.name) as conditions
                    LIMIT $limit
                """, keyword=keyword, limit=top_k)

                for record in records:
                    results.append({
                        "nct_id": record["nct_id"],
                        "title": record["title"],
                        "score": 1.0,
                        "source": "graph",
                        "payload": {
                            "nct_id": record["nct_id"],
                            "title": record["title"],
                            "status": record["status"],
                            "phase": record["phase"],
                            "conditions": record["conditions"],
                        }
                    })
        return results

    def reciprocal_rank_fusion(
        self,
        result_lists: list[list[dict]],
        k: int = 60,
    ) -> list[dict]:
        """Merge multiple ranked lists into one using RRF."""
        scores = {}
        doc_map = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                nct_id = doc["nct_id"]
                if not nct_id:
                    continue
                if nct_id not in scores:
                    scores[nct_id] = 0.0
                    doc_map[nct_id] = doc
                scores[nct_id] += 1.0 / (k + rank + 1)

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [
            {**doc_map[nct_id], "rrf_score": scores[nct_id]}
            for nct_id in sorted_ids
        ]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        print(f"\nSearching: '{query}'")

        vector_results = self.vector_search(query, top_k=10)
        print(f"  Vector results: {len(vector_results)}")

        bm25_results = self.bm25_search(query, top_k=10)
        print(f"  BM25 results: {len(bm25_results)}")

        graph_results = self.graph_search(query, top_k=5)
        print(f"  Graph results: {len(graph_results)}")

        merged = self.reciprocal_rank_fusion(
            [vector_results, bm25_results, graph_results]
        )

        print(f"  After RRF fusion: {len(merged)} unique trials")
        return merged[:top_k]

    def close(self):
        self.neo4j.close()


if __name__ == "__main__":
    retriever = HybridRetriever()

    # build BM25 corpus from Qdrant data
    retriever.build_bm25_corpus()

    # test queries
    queries = [
        "Phase 3 breast cancer trials with tamoxifen",
        "diabetes insulin treatment recruiting",
        "alzheimer memory loss drug trials",
    ]

    for query in queries:
        results = retriever.search(query, top_k=3)
        print(f"\nTop results for: '{query}'")
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['source']}] {r['title']} (score: {r['rrf_score']:.4f})")

    retriever.close()