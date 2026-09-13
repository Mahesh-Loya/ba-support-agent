import pandas as pd

from src import failure_analysis as fa


def _row(tweet_id, intent, pred_intent, human_action, agent_action,
         customer_text="hi", reply="reply", suffixed=True):
    d = dict(customer_tweet_id=tweet_id, intent=intent, pred_intent=pred_intent,
             customer_text=customer_text, reply=reply)
    if suffixed:
        d["action_x"] = human_action
        d["action_y"] = agent_action
    else:
        # bare-`action` shape: only valid when human/agent action agree,
        # since there is only one column to hold both.
        assert human_action == agent_action
        d["action"] = human_action
    return d


def test_confusion_pairs_ranks_most_frequent_pair_first_and_respects_n():
    rows = (
        [_row(i, "baggage", "refund_compensation", "auto", "auto") for i in range(5)]
        + [_row(100 + i, "loyalty_account", "general_enquiry", "auto", "auto") for i in range(2)]
        + [_row(200, "complaint_feedback", "praise_or_social", "auto", "auto")]
    )
    df = pd.DataFrame(rows)

    pairs = fa.confusion_pairs(df, n=5)
    assert pairs.iloc[0]["intent"] == "baggage"
    assert pairs.iloc[0]["pred_intent"] == "refund_compensation"
    assert pairs.iloc[0]["count"] == 5

    top1 = fa.confusion_pairs(df, n=1)
    assert len(top1) == 1
    assert top1.iloc[0]["intent"] == "baggage"


def test_misclassifications_returns_only_rows_where_pred_differs_from_human():
    rows = [
        _row(1, "baggage", "baggage", "auto", "auto"),          # correct
        _row(2, "baggage", "refund_compensation", "auto", "auto"),  # wrong
        _row(3, "flight_disruption", "flight_disruption", "escalate", "escalate"),  # correct
    ]
    df = pd.DataFrame(rows)

    m = fa.misclassifications(df)
    assert set(m.customer_tweet_id) == {2}
    assert (m.pred_intent != m.intent).all()


def test_escalation_disagreements_direction_with_suffixed_action_columns():
    rows = [
        _row(1, "baggage", "baggage", "escalate", "auto", suffixed=True),   # dangerous: found
        _row(2, "baggage", "baggage", "auto", "escalate", suffixed=True),   # over-cautious, NOT this
        _row(3, "baggage", "baggage", "escalate", "escalate", suffixed=True),  # agreement
        _row(4, "baggage", "baggage", "auto", "auto", suffixed=True),       # agreement
    ]
    df = pd.DataFrame(rows)

    out = fa.escalation_disagreements(df)
    assert set(out.customer_tweet_id) == {1}
    assert (out.action_x == "escalate").all()
    assert (out.action_y == "auto").all()


def test_escalation_disagreements_direction_with_bare_action_column():
    # Bare `action` shape (no merge suffixing occurred): human and agent
    # collapse onto a single column, so a "disagreement" can only be found
    # when that single column reads "auto" and is being interpreted as both
    # human and agent simultaneously - i.e. never, since it can't equal both
    # "escalate" and "auto" at once. This pins that the fallback path does
    # not crash and correctly yields no false positives.
    rows = [
        _row(1, "baggage", "baggage", "escalate", "escalate", suffixed=False),
        _row(2, "baggage", "baggage", "auto", "auto", suffixed=False),
    ]
    df = pd.DataFrame(rows)

    out = fa.escalation_disagreements(df)
    assert len(out) == 0
    assert "action" in df.columns


def test_no_failures_returns_empty_without_raising():
    rows = [
        _row(1, "baggage", "baggage", "auto", "auto"),
        _row(2, "flight_disruption", "flight_disruption", "escalate", "escalate"),
    ]
    df = pd.DataFrame(rows)

    m = fa.misclassifications(df)
    assert m.empty

    pairs = fa.confusion_pairs(df)
    assert pairs.empty

    esc = fa.escalation_disagreements(df)
    assert esc.empty

    top = fa.top_failures(df=df)
    assert top.empty


def test_top_failures_tolerates_fewer_example_rows_than_requested():
    rows = [_row(1, "baggage", "refund_compensation", "auto", "auto")]
    df = pd.DataFrame(rows)

    top = fa.top_failures(n=5, df=df)
    assert len(top) == 1
    assert top.iloc[0]["true"] == "baggage"
    assert top.iloc[0]["pred"] == "refund_compensation"
    assert top.iloc[0]["count"] == 1


def test_top_failures_injected_df_avoids_disk_load():
    rows = [_row(i, "baggage", "refund_compensation", "auto", "auto") for i in range(3)]
    df = pd.DataFrame(rows)

    # Passing df= must short-circuit load() entirely - no predictions.jsonl
    # needs to exist on disk for this to work.
    top = fa.top_failures(n=1, df=df)
    assert len(top) == 2  # head(2) example cap within the one confusion pair
    assert (top["count"] == 3).all()
