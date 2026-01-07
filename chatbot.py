import faiss
import pickle
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -------- CONFIG --------
INDEX_DIR = "faiss_index"
MAX_CONTEXT_CHARS = 5000

# -------- MODELS --------
client = OpenAI()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------- LOAD INDEX --------
index = faiss.read_index(f"{INDEX_DIR}/index.faiss")

with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
    data = pickle.load(f)

documents = data["documents"]
sources = data["sources"]

# -------- RETRIEVAL --------
def retrieve(query, k=8):
    q_emb = embedder.encode([query]).astype("float32")
    _, I = index.search(q_emb, min(k, index.ntotal))

    chunks = []
    srcs = []

    for i in I[0]:
        chunks.append(documents[i])
        srcs.append(sources[i])

    return chunks[:5], srcs[:5]

# -------- ANSWERING --------
def answer(query):
    context_chunks, srcs = retrieve(query)

    if not context_chunks:
        return "Information not available", set()

    context_text = "\n\n".join(context_chunks)
    context_text = context_text[:MAX_CONTEXT_CHARS]

    prompt = f"""
You are a college information assistant.

The context may contain:
- tables
- scanned or OCR text
- broken formatting
- fee structures

Your job is to carefully extract facts, numbers, and rules
from the context and answer clearly.

If numbers or policies appear, you MUST use them.
Only say "Information not available" if the context is empty.

Context:
{context_text}

Question:
{query}

Answer clearly using bullet points or short paragraphs.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400
    )

    return response.choices[0].message.content, set(srcs)
