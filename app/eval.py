"""
A tiny retrieval evaluation harness.

This is the difference between "it looks like it works" and "I measured
whether it works": a small labeled set of (query -> which doc should come
back) pairs, scored with recall@k. Real projects grow this into a proper
regression suite that runs in CI on every pipeline change.
"""
import json
from typing import List

from app.config import settings
from app.vectorstore import vector_store


def load_eval_set(path: str = None) -> List[dict]:
    path = path or settings.EVAL_SET_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval(top_k: int = 5) -> dict:
    eval_set = load_eval_set()
    if not eval_set:
        return {"recall_at_k": 0.0, "cases": [], "n": 0}

    hits = 0
    cases = []
    for case in eval_set:
        query = case["query"]
        expected_doc = case["expected_doc_id"]  # matches a source .txt filename, or None for out-of-scope queries
        results = vector_store.query(query, top_k=top_k)
        results = [(c, s) for c, s in results if s >= settings.MIN_RETRIEVAL_SCORE]
        retrieved_docs = [c.doc_id.split("#")[0] for c, _ in results]

        if expected_doc is None:
            # negative case: correct behavior is finding nothing above threshold
            hit = len(retrieved_docs) == 0
        else:
            hit = expected_doc in retrieved_docs

        hits += int(hit)
        cases.append({"query": query, "expected_doc_id": expected_doc, "retrieved": retrieved_docs, "hit": hit})

    recall = hits / len(eval_set)
    return {"recall_at_k": round(recall, 3), "cases": cases, "n": len(eval_set), "k": top_k}
