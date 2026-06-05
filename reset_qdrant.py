# run this once to reset
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
client.delete_collection("clinical_trials")
print("Collection deleted")