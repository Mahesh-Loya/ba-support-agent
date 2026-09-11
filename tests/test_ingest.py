import pandas as pd
import pytest
from src import ingest


def test_strip_signature_extracts_caret_handle():
    body, sig = ingest.strip_signature("Sorry about that, we'll check. ^Jane")
    assert body == "Sorry about that, we'll check."
    assert sig == "Jane"


def test_strip_signature_extracts_asterisk_handle():
    body, sig = ingest.strip_signature("The earlier flight was unavailable. *TMT")
    assert body == "The earlier flight was unavailable."
    assert sig == "TMT"


def test_strip_signature_returns_none_when_unsigned():
    body, sig = ingest.strip_signature("No signature here.")
    assert body == "No signature here."
    assert sig is None


def test_strip_part_marker_removes_trailing_and_leading_markers():
    assert ingest.strip_part_marker("Hello there 1/2") == "Hello there"
    assert ingest.strip_part_marker("2/2 and the rest") == "and the rest"
    assert ingest.strip_part_marker("Nothing to strip") == "Nothing to strip"


def test_reassemble_joins_parts_in_order_with_single_space():
    assert ingest.reassemble(["We're sorry for the delay", "caused. ^Jane"]) == \
        "We're sorry for the delay caused. ^Jane"


def test_build_pairs_reassembles_a_split_reply():
    df = pd.DataFrame([
        # customer opening message (not a reply to anything)
        dict(tweet_id=1, author_id="115892", inbound=True,
             created_at="Wed Nov 01 10:00:00 +0000 2017",
             text="@British_Airways my flight was cancelled, what now?",
             response_tweet_id="2", in_response_to_tweet_id=None),
        dict(tweet_id=2, author_id="British_Airways", inbound=False,
             created_at="Wed Nov 01 10:05:00 +0000 2017",
             text="@115892 Sorry for the disruption 1/2",
             response_tweet_id="3", in_response_to_tweet_id=1.0),
        dict(tweet_id=3, author_id="British_Airways", inbound=False,
             created_at="Wed Nov 01 10:06:00 +0000 2017",
             text="@115892 we can rebook you free of charge. ^Jane 2/2",
             response_tweet_id=None, in_response_to_tweet_id=2.0),
    ])
    pairs = ingest.build_pairs(df, "British_Airways")
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row.n_parts == 2
    assert row.agent_text == "Sorry for the disruption we can rebook you free of charge."
    assert row.agent_signature == "Jane"
    assert "flight was cancelled" in row.customer_text


def test_keep_longest_reply_prefers_longer_agent_text_over_arbitrary_row():
    # Two rows claim the same customer_tweet_id (e.g. two separate reply
    # chains that resolved to the same root) with different reply lengths.
    # An arbitrary drop_duplicates pick could keep either one; we require
    # the longer, more informative reply to survive regardless of row order.
    out = pd.DataFrame([
        dict(customer_tweet_id=1, customer_text="hi",
             agent_text="Short reply.", agent_signature="Jane",
             created_at="t1", n_parts=1),
        dict(customer_tweet_id=1, customer_text="hi",
             agent_text="A much longer, more detailed and helpful reply.",
             agent_signature="Tom", created_at="t2", n_parts=1),
        dict(customer_tweet_id=2, customer_text="bye",
             agent_text="Only reply.", agent_signature=None,
             created_at="t3", n_parts=1),
    ])
    deduped = ingest._keep_longest_reply(out)
    assert len(deduped) == 2
    row = deduped.set_index("customer_tweet_id").loc[1]
    assert row.agent_text == "A much longer, more detailed and helpful reply."
    assert row.agent_signature == "Tom"


def test_build_pairs_merges_multiple_reply_chains_to_the_same_root():
    # Two independent BA reply chains both point back at the same customer
    # tweet. build_pairs groups everything sharing a root together, so this
    # yields a single row whose agent_text is the chronological
    # concatenation of both chains (not a pick between them).
    df = pd.DataFrame([
        dict(tweet_id=1, author_id="115892", inbound=True,
             created_at="Wed Nov 01 10:00:00 +0000 2017",
             text="@British_Airways my flight was cancelled, what now?",
             response_tweet_id="2,4", in_response_to_tweet_id=None),
        dict(tweet_id=2, author_id="British_Airways", inbound=False,
             created_at="Wed Nov 01 10:05:00 +0000 2017",
             text="@115892 Sorry to hear that. ^Jane",
             response_tweet_id=None, in_response_to_tweet_id=1.0),
        dict(tweet_id=4, author_id="British_Airways", inbound=False,
             created_at="Wed Nov 01 11:00:00 +0000 2017",
             text="@115892 We can rebook you on the next available flight 1/2",
             response_tweet_id="5", in_response_to_tweet_id=1.0),
        dict(tweet_id=5, author_id="British_Airways", inbound=False,
             created_at="Wed Nov 01 11:01:00 +0000 2017",
             text="@115892 free of charge, just DM your booking reference. ^Tom 2/2",
             response_tweet_id=None, in_response_to_tweet_id=4.0),
    ])
    pairs = ingest.build_pairs(df, "British_Airways")
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row.agent_text == (
        "Sorry to hear that. We can rebook you on the next available "
        "flight free of charge, just DM your booking reference.")
    assert row.agent_signature == "Tom"


def test_build_pairs_ignores_replies_to_other_brands():
    df = pd.DataFrame([
        dict(tweet_id=1, author_id="115892", inbound=True,
             created_at="Wed Nov 01 10:00:00 +0000 2017", text="@Delta hi",
             response_tweet_id="2", in_response_to_tweet_id=None),
        dict(tweet_id=2, author_id="Delta", inbound=False,
             created_at="Wed Nov 01 10:05:00 +0000 2017", text="@115892 hello",
             response_tweet_id=None, in_response_to_tweet_id=1.0),
    ])
    assert len(ingest.build_pairs(df, "British_Airways")) == 0


def test_assign_split_is_temporal_and_disjoint():
    pairs = pd.DataFrame(dict(
        customer_tweet_id=[1, 2],
        created_at=pd.to_datetime(["2017-10-15T00:00:00Z", "2017-11-15T00:00:00Z"]),
    ))
    out = ingest.assign_split(pairs, "2017-11-01")
    assert out.set_index("customer_tweet_id").loc[1, "split"] == "corpus"
    assert out.set_index("customer_tweet_id").loc[2, "split"] == "eval"
    corpus_ids = set(out[out.split == "corpus"].customer_tweet_id)
    eval_ids = set(out[out.split == "eval"].customer_tweet_id)
    assert corpus_ids & eval_ids == set()
