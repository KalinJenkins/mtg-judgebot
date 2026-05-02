# retrieval.py
# Searches the ChromaDB vector database for chunks relevant to a query.

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "mtg_rules"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_NUM_RESULTS = 5


def load_retriever():
    """Load the embedding model and connect to ChromaDB. Returns (model, collection)."""
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection


def retrieve(query: str, model, collection, n_results: int = DEFAULT_NUM_RESULTS) -> list[dict]:
    """
    Embed the query and return the top n_results matching chunks.
    Each result is a dict with 'text', 'source', and 'rule_number'.
    """
    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": metadata.get("source", "unknown"),
            "rule_number": metadata.get("rule_number", "unknown"),
            "distance": round(distance, 4),
        })

    return chunks


if __name__ == "__main__":
    # Quick test — run this directly to verify retrieval is working
    print("Loading retriever...")
    model, collection = load_retriever()

    test_queries = [
        "Can I activate a planeswalker ability the turn it enters the battlefield?",
        "How does commander damage work?",
        "What happens if my commander dies?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retrieve(query, model, collection)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] Source: {r['source']} | Rule: {r['rule_number']} | Distance: {r['distance']}")
            print(f"       {r['text'][:120]}...")
