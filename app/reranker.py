"""
A deliberately cheap reranking stage.

Real systems often run a cross-encoder here. For this POC that's an extra
heavy model download for marginal teaching value, so instead this rescoring
step blends the vector similarity with exact keyword overlap between the
query and each chunk — enough to demonstrate *why* a two-stage
retrieve-then-rerank pipeline exists (the first pass optimizes for recall,
the second for precision) without adding a second model to the stack.
"""
import re
from typing import List, Tuple

from app.vectorstore import Chunk

_WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def _keywords(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text)}


def rerank(query: str, candidates: List[Tuple[Chunk, float]], top_k: int) -> List[Tuple[Chunk, float]]:
    q_words = _keywords(query)
    if not q_words:
        return candidates[:top_k]

    rescored = []
    for chunk, vec_score in candidates:
        chunk_words = _keywords(chunk.text)
        overlap = len(q_words & chunk_words) / max(len(q_words), 1)
        blended = 0.6 * vec_score + 0.4 * overlap
        rescored.append((chunk, blended))

    rescored.sort(key=lambda pair: pair[1], reverse=True)
    return rescored[:top_k]
