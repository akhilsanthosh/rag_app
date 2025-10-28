import time
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from retriever import retrieve
from generator import stream_llama

app = FastAPI(title="Ultra-Low-Latency RAG")

@app.get("/ask")
def ask(q: str = Query(...), tenant: str = Query("demo")):
    try:
        start = time.time()  # ⏱ Start timer

        context = retrieve(q, tenant)
        prompt = f"Answer based only on CONTEXT below:\n\nCONTEXT:\n{context}\n\nQUESTION:\n{q}"

        response = StreamingResponse(stream_llama(prompt), media_type="text/plain")

        end = time.time()  # ⏱ End timer
        print(f"⚡ Total Latency: {end - start:.2f} seconds")

        return response
    except Exception as e:
        print("❌ ERROR:", e)
        return PlainTextResponse(content=f"Server error: {e}", status_code=500)
