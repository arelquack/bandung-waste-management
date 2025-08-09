import time
import statistics
import os
import sys
sys.path.append(os.path.abspath("src"))
from retriever import DocumentRetriever
from generator import LLMGenerator

DATA_PATH = os.path.join("data", "perda_data.pkl")

# Cold start load time
start = time.perf_counter()
retriever = DocumentRetriever(DATA_PATH)
load_time = (time.perf_counter() - start) * 1000.0

# Retrieval p50 over 100 unique queries to avoid cache hits
queries = [f"Apa sanksi bagi pembakar sampah? #{i}" for i in range(100)]

latencies_ms = []
for q in queries:
    t0 = time.perf_counter()
    _ = retriever.retrieve_chunks(q, top_k=3)
    t1 = time.perf_counter()
    latencies_ms.append((t1 - t0) * 1000.0)

p50 = statistics.median(latencies_ms)
p95 = statistics.quantiles(latencies_ms, n=20)[-1]

# Generator TTFB proxy (will likely use extractive fallback here)
gen = LLMGenerator()
q = "Apa sanksi bagi pembakar sampah?"
chunks = retriever.retrieve_chunks(q, top_k=3)
t0 = time.perf_counter()
_ = gen.generate_answer(q, chunks)
ttfb_ms = (time.perf_counter() - t0) * 1000.0

print({
    "cold_start_ms": round(load_time, 1),
    "retrieval_p50_ms": round(p50, 1),
    "retrieval_p95_ms": round(p95, 1),
    "generator_ttfb_ms": round(ttfb_ms, 1),
})