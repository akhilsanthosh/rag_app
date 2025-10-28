import os, re, uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition

# --- CONFIG ---
COLL = os.getenv("COLLECTION", "tenant_docs_v1")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Qdrant client (skip version check for safety)
qdrant = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL)

# --- CREATE COLLECTION ---
def ensure_collection():
    collections = [c.name for c in qdrant.get_collections().collections]
    
    if COLL not in collections:
        dim = embedder.get_sentence_embedding_dimension()
        qdrant.create_collection(
            collection_name=COLL,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print(f"✅ Created collection: {COLL} (dim={dim})")
    else:
        print(f"ℹ️ Collection '{COLL}' already exists")



# --- CHUNKING ---
def chunk_text(text: str, target=400, overlap=60):
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, buf, toks = [], [], 0

    # ✅ Fixed tokenizer (safe and works across SentenceTransformer versions)
    def count_tokens(t):
        tokens = embedder.tokenizer(t, return_tensors="pt")["input_ids"]
        return tokens.shape[1]

    for s in sents:
        l = count_tokens(s)
        if toks + l > target and buf:
            chunks.append(" ".join(buf))
            buf = buf[-2:]  # overlap
            toks = sum(count_tokens(x) for x in buf)
        buf.append(s)
        toks += l

    if buf:
        chunks.append(" ".join(buf))

    return chunks


# --- INGEST PDF OR FALLBACK TEXT ---
def ingest_file(path: str = None, tenant_id="demo"):
    texts = []

    if path and os.path.exists(path):
        print(f"📄 Extracting from file: {path}")
        elements = partition(filename=path)
        texts = [e.text for e in elements if getattr(e, "text", None)]
        text = "\n".join(texts).strip()
        if not text:
            print("⚠️  No text could be extracted. Maybe scanned or image-based PDF.")
            return
    else:
        print("⚙️ No file found — inserting sample demo text.")
        texts = [
            """Customers can request a refund within 7 days of purchase if the product is unused and in its original packaging.""",
            """Premium plan users have 24/7 access to customer support via phone and chat.""",
            """when was the first email services available to the public?"""
        ]

    all_chunks = []
    for t in texts:
        all_chunks.extend(chunk_text(t))

    print(f"📚 Created {len(all_chunks)} chunks. Generating embeddings...")
    vectors = embedder.encode(all_chunks, convert_to_numpy=True)

    pts = []
    for i, (txt, vec) in enumerate(zip(all_chunks, vectors)):
        pts.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={"tenant": tenant_id, "chunk": i, "text": txt}
            )
        )

    qdrant.upsert(collection_name=COLL, points=pts, wait=True)
    print(f"✅ Inserted {len(pts)} chunks for tenant '{tenant_id}' into {COLL}")

# --- MAIN ---
if __name__ == "__main__":
    ensure_collection()
    ingest_file("data/FINAL-VERSION-CUSTOMER-SERVICE-EBOOK_copy.pdf", tenant_id="demo")
