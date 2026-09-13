"""
A tiny vector store: TF-IDF embeddings + cosine similarity, persisted to disk.

Why TF-IDF instead of a neural embedding model? For a POC that anyone can
`pip install` and run offline in under a minute, downloading a 90MB
sentence-transformer (or paying for embedding API calls) is friction that
doesn't teach the LLMOps concepts this project is about. The vector-db
*interface* (upsert/query over embeddings) is identical either way — swapping
this for Chroma/Pinecone + a real embedding model is a drop-in upgrade later,
and the README calls that out as the natural next step.
"""
import glob
import os
import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings


@dataclass
class Chunk:
    doc_id: str
    text: str


class VectorStore:
    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.chunks: List[Chunk] = []

    # ---- ingestion ----
    def build_from_docs(self, docs_dir: str = None, chunk_size: int = 400, overlap: int = 60):
        docs_dir = docs_dir or settings.DOCS_DIR
        self.chunks = []
        for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
            doc_id = os.path.splitext(os.path.basename(path))[0]
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for i, chunk_text in enumerate(self._chunk(text, chunk_size, overlap)):
                self.chunks.append(Chunk(doc_id=f"{doc_id}#chunk{i}", text=chunk_text.strip()))

        corpus = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.matrix = self.vectorizer.fit_transform(corpus)
        return len(self.chunks)

    @staticmethod
    def _chunk(text: str, size: int, overlap: int):
        words = text.split()
        step = max(size - overlap, 1)
        for start in range(0, len(words), step):
            piece = words[start : start + size]
            if not piece:
                continue
            yield " ".join(piece)
            if start + size >= len(words):
                break

    # ---- persistence ----
    def save(self, path: str = None):
        path = path or settings.INDEX_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "chunks": self.chunks}, f)

    def load(self, path: str = None) -> bool:
        path = path or settings.INDEX_PATH
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        self.chunks = data["chunks"]
        return True

    # ---- query ----
    def query(self, text: str, top_k: int = 5):
        if self.vectorizer is None or self.matrix is None or not self.chunks:
            return []
        query_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx if sims[i] > 0]


vector_store = VectorStore()
