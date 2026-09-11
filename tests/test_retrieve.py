# tests/test_retrieve.py
import pandas as pd
import pytest
from src.agent import retrieve


@pytest.fixture
def corpus():
    return pd.DataFrame(dict(
        customer_tweet_id=[1, 2, 3],
        customer_text=["my flight was cancelled",
                       "my suitcase never arrived",
                       "how many Avios for an upgrade"],
        agent_text=["Sorry - we can rebook you free of charge.",
                    "Please file a report at the baggage desk.",
                    "Upgrades start at 10,000 Avios."],
        split=["corpus"] * 3,
    ))


def test_search_returns_the_topically_matching_case_first(corpus):
    r = retrieve.Retriever(corpus)
    top = r.search("they cancelled my flight!", k=1)[0]
    assert "rebook" in top.agent_text


def test_search_respects_k(corpus):
    r = retrieve.Retriever(corpus)
    assert len(r.search("baggage lost", k=2)) == 2


def test_similarity_is_bounded_and_descending(corpus):
    r = retrieve.Retriever(corpus)
    cases = r.search("lost luggage", k=3)
    sims = [c.similarity for c in cases]
    assert sims == sorted(sims, reverse=True)
    assert all(-1.01 <= s <= 1.01 for s in sims)


def test_retriever_only_ever_holds_corpus_split_rows():
    df = pd.DataFrame(dict(
        customer_tweet_id=[1, 2],
        customer_text=["corpus row", "eval row"],
        agent_text=["a", "b"],
        split=["corpus", "eval"],
    ))
    r = retrieve.Retriever(df)
    assert len(r.corpus) == 1
    assert r.corpus.iloc[0].customer_text == "corpus row"


def test_retriever_refuses_a_frame_without_a_split_column():
    # The anti-leakage guarantee must not silently degrade to "trust the
    # caller" when the split column is missing.
    df = pd.DataFrame(dict(
        customer_tweet_id=[1, 2],
        customer_text=["row one", "row two"],
        agent_text=["a", "b"],
    ))
    with pytest.raises(ValueError):
        retrieve.Retriever(df)
