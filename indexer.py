import os
import pickle
import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

DATA_DIR = "college_knowledge"
INDEX_DIR = "faiss_index"
BASE_URL = "https://www.yourcollege.edu"   # CHANGE THIS
MAX_PAGES = 25

os.makedirs(INDEX_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
sources = []

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def is_valid_chunk(chunk):
    if len(chunk) < 250:
        return False
    if chunk.count(" ") < 40:
        return False
    return True

# -------- PDF LOADER --------
def load_pdfs():
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(DATA_DIR, file))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            for chunk in chunk_text(text):
                if is_valid_chunk(chunk):
                    documents.append(chunk)
                    sources.append(file)

# -------- WEBSITE SCRAPER --------
def extract_clean_text(soup):
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", id="content")
    )

    if main:
        return main.get_text(separator=" ", strip=True)
    elif soup.body:
        return soup.body.get_text(separator=" ", strip=True)
    return ""

def scrape_website():
    visited = set()
    queue = [BASE_URL]

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue

        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            text = extract_clean_text(soup)
            for chunk in chunk_text(text):
                if is_valid_chunk(chunk):
                    documents.append(chunk)
                    sources.append(url)

            visited.add(url)

            for link in soup.find_all("a", href=True):
                full = urljoin(BASE_URL, link["href"])
                if BASE_URL in full:
                    queue.append(full)

        except Exception:
            continue

print("📄 Loading PDFs...")
load_pdfs()

print("🌐 Scraping website...")
scrape_website()

print(f"✅ Total chunks indexed: {len(documents)}")

print("🔢 Creating embeddings...")
embeddings = model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, f"{INDEX_DIR}/index.faiss")

with open(f"{INDEX_DIR}/metadata.pkl", "wb") as f:
    pickle.dump({"documents": documents, "sources": sources}, f)

print("🎉 Indexing complete")

print("\n🔍 SAMPLE INDEXED CHUNKS:\n")
for i in range(5):
    print(f"--- Chunk {i} ---")
    print(documents[i][:500])
    print("Source:", sources[i])
    print()
