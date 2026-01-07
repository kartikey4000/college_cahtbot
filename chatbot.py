import faiss
import pickle
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")
# -------- CONFIG --------
INDEX_DIR = "faiss_index"
MAX_CONTEXT_CHARS = 5000
# Configure your API Key here
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# -------- MODELS --------
# Initializing the Gemini model (using flash for speed/cost similar to gpt-4o-mini)
model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------- LOAD INDEX --------
index = faiss.read_index(f"{INDEX_DIR}/index.faiss")

with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
    data = pickle.load(f)

documents = data["documents"]
sources = data["sources"]

# -------- RETRIEVAL --------
def retrieve(query, k=8):
    q_emb = embedder.encode([query], convert_to_tensor=True).cpu().numpy().astype("float32")
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

    # Gemini specific generation
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=400,
        ),
    )

    return response.text, set(srcs)
