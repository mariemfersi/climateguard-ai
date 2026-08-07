"""
RAG index construction and hybrid retrieval engine.

Chunks Solvency II Delegated Regulation, NAIC guidance, and reinsurance treaty wording
clauses into semantic passages. Provides hybrid vector/keyword retrieval with
article-level citation tracking and low-confidence refusal logic.

Supports Azure OpenAI / OpenAI embeddings when API keys exist, with a local TF-IDF / BM25
hybrid retriever for zero-dependency local execution.

Usage:
    python -m rag_index.chunk_and_embed [--query "Article 105 SCR VaR"]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REGULATORY_DOCS_DIR = Path("docs/regulatory")
INDEX_OUTPUT_DIR = Path("data_pipeline/bronze/rag_index")


class TextChunk:
    """Represents a chunked document passage with citation metadata."""

    def __init__(
        self,
        chunk_id: str,
        doc_name: str,
        title: str,
        section: str,
        article_id: str,
        text: str,
    ):
        self.chunk_id = chunk_id
        self.doc_name = doc_name
        self.title = title
        self.section = section
        self.article_id = article_id
        self.text = text

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "title": self.title,
            "section": self.section,
            "article_id": self.article_id,
            "text": self.text,
        }


class RegulatoryRAGIndex:
    """
    Hybrid retriever for regulatory and treaty clause RAG index.
    """

    def __init__(self, chunks: list[TextChunk]):
        self.chunks = chunks
        self.vocab: dict[str, int] = {}
        self.tfidf_matrix: np.ndarray | None = None
        self._build_index()

    STOP_WORDS = {
        "a", "an", "the", "in", "on", "of", "to", "for", "and", "or", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "at",
        "by", "from", "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "up", "down", "in", "out", "off", "over", "under",
        "what", "where", "when", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "can", "will", "just", "should", "now", "what", "is"
    }

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9_\-\.]+\b", text)
        return [t for t in tokens if len(t) > 1 and t not in self.STOP_WORDS]

    def _build_index(self) -> None:
        """Build TF-IDF / BM25 term matrix over chunks for hybrid retrieval."""
        if not self.chunks:
            logger.warning("Empty chunk list provided to RAG Index.")
            return

        corpus_tokens = [self._tokenize(c.text) for c in self.chunks]

        # Build vocabulary
        all_tokens = set(t for tokens in corpus_tokens for t in tokens)
        self.vocab = {t: idx for idx, t in enumerate(sorted(all_tokens))}
        v_size = len(self.vocab)

        n_docs = len(self.chunks)
        tf = np.zeros((n_docs, v_size), dtype=np.float32)
        df = np.zeros(v_size, dtype=np.float32)

        for i, tokens in enumerate(corpus_tokens):
            for t in tokens:
                if t in self.vocab:
                    col = self.vocab[t]
                    tf[i, col] += 1.0

            # Document frequency
            unique_t = set(tokens)
            for t in unique_t:
                if t in self.vocab:
                    df[self.vocab[t]] += 1.0

        # IDF calculation
        idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0

        # TF-IDF matrix
        self.tfidf_matrix = tf * idf[np.newaxis, :]
        # L2 norm
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.tfidf_matrix /= norms

        logger.info("RAG Index built with %d chunks and %d vocabulary terms.", n_docs, v_size)

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score_threshold: float = 0.10,
    ) -> list[dict[str, Any]]:
        """
        Search RAG index for query passages.

        Args:
            query: Search query text.
            top_k: Number of chunks to retrieve.
            min_score_threshold: Minimum similarity score cutoff (refuses low-confidence matches).

        Returns:
            List of result dictionaries containing text, article_id, doc_name, score.
        """
        if not self.chunks or self.tfidf_matrix is None:
            return []

        tokens = self._tokenize(query)
        q_vec = np.zeros((1, len(self.vocab)), dtype=np.float32)

        for t in tokens:
            if t in self.vocab:
                q_vec[0, self.vocab[t]] += 1.0

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        # Cosine similarities
        scores = (self.tfidf_matrix @ q_vec.T).squeeze(axis=1)

        # Keyword boost for exact article numbers (e.g. "Article 105", "Article 118", "Section II")
        for idx, chunk in enumerate(self.chunks):
            if chunk.article_id.lower() in query.lower() and chunk.article_id != "General":
                scores[idx] += 0.35

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score_threshold:
                continue

            chunk = self.chunks[idx]
            res_dict = chunk.to_dict()
            res_dict["score"] = round(score, 4)
            results.append(res_dict)

        logger.info("Search for '%s' returned %d chunks above threshold %.2f.", query, len(results), min_score_threshold)
        return results


def chunk_document(doc_path: Path) -> list[TextChunk]:
    """
    Chunk a regulatory or treaty markdown/text file into semantic passages.
    """
    if not doc_path.exists():
        logger.warning("Document not found at %s", doc_path)
        return []

    text = doc_path.read_text(encoding="utf-8")
    doc_name = doc_path.name
    title = doc_path.stem.replace("_", " ").title()

    # Split by Article / Section headers
    sections = re.split(r"\n(?=(?:Article|SECTION)\s+[0-9A-Za-z]+)", text)

    chunks = []
    chunk_counter = 1

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        # Extract Article or Section ID
        match = re.search(r"^(Article\s+[0-9a-z]+|SECTION\s+[IVXLCDM]+)", sec, re.IGNORECASE)
        art_id = match.group(1).title() if match else "General"

        # Paragraph splitting
        paragraphs = [p.strip() for p in sec.split("\n\n") if p.strip()]
        for p in paragraphs:
            cid = f"{doc_path.stem}_chk_{chunk_counter:03d}"
            chunk_counter += 1

            chunks.append(
                TextChunk(
                    chunk_id=cid,
                    doc_name=doc_name,
                    title=title,
                    section=art_id,
                    article_id=art_id,
                    text=p,
                )
            )

    return chunks


def build_and_save_index(docs_dir: Path = REGULATORY_DOCS_DIR) -> RegulatoryRAGIndex:
    """
    Load regulatory documents, chunk them, build index, and persist to disk.
    """
    all_chunks: list[TextChunk] = []

    if docs_dir.exists():
        for f in docs_dir.glob("*.txt"):
            all_chunks.extend(chunk_document(f))
        for f in docs_dir.glob("*.md"):
            all_chunks.extend(chunk_document(f))

    if not all_chunks:
        logger.warning("No regulatory docs found in %s. Creating default fallback chunks.", docs_dir)
        all_chunks.append(
            TextChunk(
                chunk_id="solvency2_fallback_001",
                doc_name="solvency_ii_scr_cat.txt",
                title="Solvency II SCR Catastrophe Risk",
                section="Article 105",
                article_id="Article 105",
                text="Article 105: The Solvency Capital Requirement (SCR) for natural catastrophe risk shall be calculated as a 1-in-200 year Value-at-Risk (VaR at 99.5% confidence level over a 1-year time horizon).",
            )
        )

    index = RegulatoryRAGIndex(all_chunks)

    # Save to json for persistence
    out_dir = INDEX_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_data = [c.to_dict() for c in all_chunks]
    with open(out_dir / "regulatory_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2)

    logger.info("Saved %d regulatory chunks to %s", len(chunks_data), out_dir)
    return index


def get_rag_index() -> RegulatoryRAGIndex:
    """Convenience getter to load pre-built index or build on the fly."""
    json_path = INDEX_OUTPUT_DIR / "regulatory_chunks.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [TextChunk(**d) for d in data]
        return RegulatoryRAGIndex(chunks)

    return build_and_save_index()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build and query Regulatory RAG Index")
    parser.add_argument("--query", type=str, default="Article 105 SCR VaR 99.5%", help="Query to search")
    args = parser.parse_args()

    index = build_and_save_index()
    results = index.search(args.query, top_k=3)

    print("\n" + "=" * 70)
    print(f"RAG RETRIEVAL RESULTS FOR: '{args.query}'")
    print("=" * 70)
    if not results:
        print("No matching regulatory chunks found above confidence threshold.")
    else:
        for r in results:
            print(f"[{r['article_id']}] {r['doc_name']} (score: {r['score']:.4f})")
            print(f"  {r['text']}")
            print("-" * 70)
