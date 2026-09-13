"""Run manually to (re)build the vector index from data/docs/*.txt:

    python scripts/ingest.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vectorstore import vector_store  # noqa: E402

if __name__ == "__main__":
    n = vector_store.build_from_docs()
    vector_store.save()
    print(f"Indexed {n} chunks from data/docs/ -> saved to data/index.pkl")
