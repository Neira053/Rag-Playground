from app.eval import run_eval
from app.vectorstore import vector_store


def test_eval_runs_and_returns_recall():
    vector_store.load()  # uses the checked-in index built from data/docs
    result = run_eval(top_k=5)
    assert result["n"] > 0
    assert 0.0 <= result["recall_at_k"] <= 1.0
    assert len(result["cases"]) == result["n"]


def test_eval_includes_hard_and_negative_cases():
    """The eval set shouldn't only contain queries that echo the source
    text verbatim -- see data/eval_set.json for paraphrased and
    out-of-scope cases added after the initial self-graded version."""
    import json
    from app.config import settings

    with open(settings.EVAL_SET_PATH) as f:
        cases = json.load(f)

    negative_cases = [c for c in cases if c.get("expected_doc_id") is None]
    assert len(negative_cases) >= 1, "eval set should include at least one out-of-scope query"
