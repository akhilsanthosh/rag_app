from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import os, time

# --- CONFIG ---
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
embedder = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
COLL = os.getenv("COLLECTION", "tenant_docs_v1")

# --- RETRIEVE FUNCTION WITH LATENCY TRACKING ---
def retrieve(query: str, tenant: str, top_k: int = 8):
    print("\n Retrieving context for query:", query)
    total_start = time.time()  # ⏱ Start total timer

    # --- Step 1: Embedding the query ---
    t0 = time.time()
    qv = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    t1 = time.time()
    print(f" Embedding Time: {t1 - t0:.3f} seconds")

    # --- Step 2: Setting up tenant filter ---
    filt = Filter(
        must=[FieldCondition(key="tenant", match=MatchValue(value=tenant))]
    )

    # --- Step 3: Vector Search in Qdrant ---
    t2 = time.time()
    hits = qdrant.search(
        collection_name=COLL,
        query_vector=qv,        #  unnamed vector — simple usage
        query_filter=filt,
        limit=top_k,
        with_payload=True
    )
    t3 = time.time()
    print(f"🔍 Qdrant Search Time: {t3 - t2:.3f} seconds")

    # --- Step 4: Combine retrieved context ---
    ctx = "\n\n".join(h.payload["text"] for h in hits)

    total_end = time.time()
    print(f" Total Retrieval Time: {total_end - total_start:.3f} seconds\n")

    return ctx

