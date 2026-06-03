from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from openai import OpenAI
from dotenv import load_dotenv
from ingestion.normalizer import TrialRecord
import os
import uuid

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "clinical_trials"
VECTOR_SIZE = 1536  # text-embedding-3-small dimension


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")


def build_chunk_text(trial: TrialRecord) -> str:
    """Build a rich text chunk from trial fields for embedding."""
    parts = [
        f"Trial: {trial.title}",
        f"NCT ID: {trial.nct_id}",
        f"Status: {trial.status}",
        f"Phase: {trial.phase or 'Not specified'}",
        f"Conditions: {', '.join(trial.conditions)}",
        f"Interventions: {', '.join(trial.interventions[:5])}",  # cap at 5
        f"Sponsor: {trial.sponsor or 'Unknown'}",
        f"Enrollment: {trial.enrollment or 'Not specified'}",
        f"Start Date: {trial.start_date or 'Unknown'}",
        f"Locations: {', '.join(trial.locations[:3])}",  # cap at 3
    ]
    return "\n".join(parts)


def embed_text(text: str, openai_client: OpenAI) -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


def write_trial_to_qdrant(
    trial: TrialRecord,
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
):
    chunk_text = build_chunk_text(trial)
    vector = embed_text(chunk_text, openai_client)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "nct_id": trial.nct_id,
            "title": trial.title,
            "status": trial.status,
            "phase": trial.phase,
            "conditions": trial.conditions,
            "interventions": trial.interventions[:5],
            "sponsor": trial.sponsor,
            "enrollment": trial.enrollment,
            "start_date": trial.start_date,
            "last_updated": trial.last_updated,
            "chunk_text": chunk_text,
        },
    )

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point],
    )


def write_all_to_qdrant(trials: list[TrialRecord]):
    qdrant_client = get_qdrant_client()
    openai_client = get_openai_client()

    ensure_collection(qdrant_client)

    success = 0
    failed = 0

    for i, trial in enumerate(trials):
        try:
            write_trial_to_qdrant(trial, qdrant_client, openai_client)
            success += 1
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(trials)} trials...")
        except Exception as e:
            print(f"  Failed to embed {trial.nct_id}: {e}")
            failed += 1

    print(f"Qdrant write complete — {success} embedded, {failed} failed")
    return success


if __name__ == "__main__":
    from ingestion.api_client import run_ingestion
    from ingestion.normalizer import normalize_all

    print("Fetching trials...")
    raw = run_ingestion(["cancer", "diabetes", "alzheimer"], max_per_condition=50)

    print("Normalizing...")
    records = normalize_all(raw)

    print("Writing to Qdrant...")
    write_all_to_qdrant(records)

    print("\nDone! Open http://localhost:6333/dashboard and check the clinical_trials collection.")