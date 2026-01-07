import trafilatura
import requests
import os
from urllib.parse import urljoin

BASE_URL = "https://yourcollege.ac.in"   # 🔴 CHANGE THIS
SAVE_DIR = "college_knowledge/website"

os.makedirs(SAVE_DIR, exist_ok=True)

URLS = [
    BASE_URL,
    urljoin(BASE_URL, "/admissions"),
    urljoin(BASE_URL, "/academics"),
    urljoin(BASE_URL, "/fees"),
    urljoin(BASE_URL, "/hostel"),
    urljoin(BASE_URL, "/placements"),
]

for url in URLS:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                filename = url.replace("https://", "").replace("/", "_")
                with open(f"{SAVE_DIR}/{filename}.txt", "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"✅ Saved: {url}")
    except Exception as e:
        print(f"❌ Failed: {url} ({e})")

print("🎯 Website scraping complete")
