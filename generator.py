import requests, json, os, time

# --- CONFIG ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# --- STREAM FUNCTION WITH LATENCY TRACKING ---
def stream_llama(prompt: str, model="llama3.2:1b"):
    print("\n Starting LLM generation...")
    start_time = time.time()  # ⏱ Start timer
    print(f" Using LLM model: {model}")

    payload = {"model": model, "prompt": prompt, "stream": True}

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue
                part = json.loads(line.decode())

                # Stream partial responses (token-by-token)
                if "response" in part:
                    yield part["response"]

                # When model signals it's done
                if part.get("done"):
                    end_time = time.time()
                    print(f" LLM Generation Completed in {end_time - start_time:.2f} seconds\n")
                    break

    except requests.exceptions.RequestException as e:
        print(" Error during LLM generation:", e)
        yield f"[Error] LLM generation failed: {e}"
