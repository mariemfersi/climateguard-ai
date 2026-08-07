from __future__ import annotations
import argparse, hashlib, json, logging
from pathlib import Path
from typing import List
import chromadb
from chromadb.config import Settings as ChromaSettings
from agents.llm_client import get_llm
from config.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
DOCS_DIR = Path(__file__).parent / "documents"


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)
        if end >= len(text):
            break
    return chunks


def _load_documents() -> List[dict]:
    docs = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".json"}:
            continue
        content = path.read_text(encoding="utf-8")
        docs.append({"source": str(path.relative_to(DOCS_DIR)), "title": path.stem, "text": content})
    return docs


def build_index(reset: bool = False) -> None:
    settings = get_settings()
    llm = get_llm()
    persist_dir = Path(settings.rag_persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False))
    if reset:
        try:
            client.delete_collection(settings.rag_collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=settings.rag_collection_name, metadata={"hnsw:space": "cosine"})
    raw_docs = _load_documents()
    if not raw_docs:
        raise RuntimeError(f"No documents under {DOCS_DIR}")

    ids, documents, metadatas, embeddings = [], [], [], []
    for doc in raw_docs:
        for i, chunk in enumerate(_chunk_text(doc["text"], settings.rag_chunk_size, settings.rag_chunk_overlap)):
            chunk_id = hashlib.sha1(f"{doc['source']}::{i}::{chunk[:64]}".encode()).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({"source": doc["source"], "title": doc["title"], "chunk_index": i})

    logger.info("Embedding %d chunks…", len(documents))
    batch_size = 16
    for start in range(0, len(documents), batch_size):
        embeddings.extend(llm.embed(documents[start:start + batch_size]))

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    logger.info("Index ready — %d chunks", collection.count())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    build_index(reset=args.reset)

if __name__ == "__main__":
    main()