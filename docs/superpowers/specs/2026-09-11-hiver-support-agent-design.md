# Design: AI Support Agent for British Airways (Hiver SDE Intern Take-Home)

Date: 2026-09-11
Status: Draft for review

## 1. Problem framing

Build an AI support agent for one brand from the Customer Support on Twitter
dataset that classifies intent, drafts a grounded reply, and decides whether to
auto-handle or escalate. The brief states the grading axis explicitly: *"the
proof is worth more than the system."*

**Strategic choice.** Accuracy is the wrong headline metric for this product.
A support tool's commercial question is *how much volume can we safely
automate*, and the costs are asymmetric: a wrong auto-sent reply to a real
customer is far more expensive than an unnecessary escalation to a human.

So the headline claim is a **deployment decision**, not a score:

> At routing threshold T, the agent auto-handles **X%** of British Airways'
> inbound support volume at a **Y%** harmful-reply rate (95% CI ...), escalating
> the remainder with a stated reason.

Everything below serves that claim and the evidence for it.

### What "good" means for British Airways

BA's own historical behaviour defines the target. From measurement (Section 2),
a good BA reply is: short, human-signed, specific to the issue, honest about
what BA cannot do, and does *not* deflect to DM unless account data is needed.
A good agent reply matches that register and never invents a commitment BA
would not make (compensation, rebooking, refunds).

### Non-goals (deliberate)

Multi-turn conversation state; tool-calling / live booking systems;
fine-tuning; a web UI; retrieval reranking; multilingual support. These are
recorded in the report's "one more week" section, which is itself a graded
deliverable.

## 2. Brand selection (decided, with evidence)

Selection criterion, stated before measuring: *a brand whose historical replies
actually resolve issues*, since grounding a draft in "please DM us" or a bare
URL is vacuous.

Measured over all 2.81M rows:

| Brand | first-contact pairs | non-latin | multipart | support-dense | deflect % | link % |
|---|---|---|---|---|---|---|
| **British_Airways** | 19,611 | 0.0% | 31.4% | 39.1% | 14.0% | 6.8% |
| Delta | 28,485 | 0.0% | 9.6% | 35.3% | 16.5% | 15.3% |
| AmericanAir | 24,506 | 0.0% | 0.0% | 31.5% | 18.9% | 6.5% |
| AmazonHelp | 84,637 | 8.1% | 14.0% | 22.5% | 0.7% | 41.3% |
| AppleSupport | n/a | n/a | n/a | n/a | 52.5% | 75.4% |

"support-dense" = share of customer messages containing a concrete problem term.

**Decision: British_Airways.** Highest support-density, zero non-English noise,
lowest deflection and lowest link-dumping. Cost: 31.4% of BA replies are split
across tweets ("1/2", "2/2") and must be reassembled.

Rejected: AppleSupport (52% deflection, 75% link-dumping - the intuitive pick,
killed by measurement); AmazonHelp (8% Japanese, 41% links, 22% support-dense);
AmericanAir (heavy non-support chitchat, no agent signatures).
Runner-up: Delta.

## 3. Architecture

```
data/raw/          twcs.csv                  (gitignored, 493MB)
data/interim/      ba_threads.parquet        (committed subsample)
data/golden/       golden.jsonl              (committed, hand-labelled)
cache/             llm_cache.sqlite          (committed - see 3.1)
src/
  ingest.py        filter BA, rebuild threads, reassemble split replies
  taxonomy.py      embed + cluster -> propose intents -> human names them
  label_tui.py     keyboard-driven labelling tool
  agent/
    classify.py    message -> intent
    retrieve.py    message -> k similar historical resolved cases
    draft.py       intent + cases -> reply
    decide.py      -> auto or escalate, plus reason and confidence
  baselines.py     trivial + simple
  eval/
    metrics.py     accuracy, macro-F1, bootstrap CIs
    judge.py       LLM-as-judge rubric
    agreement.py   judge vs human (Cohen's kappa)
    risk_coverage.py  coverage/harm curve, threshold selection, calibration
  llm.py           provider abstraction + disk cache
reports/           REPORT.md, DECISIONS.md, figures/
```

Single CLI: `python -m src.cli {ingest,taxonomy,label,run,eval,report}`.

### 3.1 Reproducibility: the committed LLM cache

Every LLM call routes through `llm.py`, which caches responses in a SQLite file
**committed to the repo**, keyed by `sha256(provider, model, prompt, params)`.

Consequence: a reviewer clones, runs one command, needs **no API key**, and
regenerates the exact headline numbers in about 2 minutes. `--no-cache` forces
live calls. This satisfies both the free-tier constraint and the 15-minute
reproduction requirement.

The entry point is `python -m src.cli reproduce` rather than a Makefile, since
`make` is not present on the development machine (Windows) and cannot be assumed
on the reviewer's either. A thin `Makefile` may wrap it for convenience, but it
is never the documented path.

### 3.2 Model choices (free tier)

- **Drafter**: Gemini 2.5 Flash (free tier).
- **Judge**: Llama 3.3 70B via Groq (free tier).
- **Embeddings**: all-MiniLM-L6-v2 run locally via sentence-transformers.
  No API, no rate limit, fully reproducible.

**Judge and drafter are deliberately different model families.** A model
grading its own output inflates the score; this is measured, not assumed.

## 4. Data pipeline

1. Load twcs.csv, filter author_id == British_Airways.
2. Reconstruct threads via in_response_to_tweet_id.
3. **Reassemble multi-part replies** (31.4% of BA replies) into a single
   response by joining consecutive same-author tweets in a thread and stripping
   "1/2", "2/2" markers.
4. Strip agent signatures (^Jane) into a separate field - they are style, not
   content, and leaking them into retrieval would teach the model to sign as a
   human that does not exist.
5. Keep **first-contact pairs** (customer's opening message -> BA's reply) as
   the unit of work. About 19.6k available.
6. Temporal split: retrieval corpus and golden set come from **disjoint time
   ranges**, so the agent cannot retrieve the very thread it is being evaluated
   on. This is the single most important anti-leakage measure.

## 5. Intent taxonomy

Embed customer messages, cluster (HDBSCAN, or k-means with silhouette-guided k),
inspect top terms and 10 representative messages per cluster, then **name the
intents by hand**. Target about 8 intents plus "other".

Expected shape from samples seen: flight delay/cancellation, baggage
(lost/delayed/damaged), booking changes and seats, refunds and compensation
claims, check-in and boarding passes, loyalty/Executive Club, complaints and
service feedback, general enquiry, other/not-support.

The taxonomy is a *design output*, not a given. The report documents what was
collapsed and why (for example, if "delay" and "cancellation" are not separable
in practice they merge, and that finding is reported).

## 6. Agent design

Four independently testable stages:

1. **Classify** - LLM few-shot classification into the taxonomy, returning
   intent plus self-reported confidence.
2. **Retrieve** - top-k historically similar customer messages (cosine over
   MiniLM embeddings) with BA's actual replies, restricted to the corpus split.
3. **Draft** - generate a reply grounded in the retrieved real replies, matching
   BA's register (short, specific, no invented commitments).
4. **Decide** - auto-handle or escalate, with a written reason.

### 6.1 Escalation policy: rules first, then confidence

Two layers, because they answer different questions.

**Hard rules (always escalate, regardless of confidence).** Categories a
business must never automate: compensation/refund claims, legal threats, safety
incidents, lost-baggage claims, anything needing account or PII lookup, and
detected severe distress. These are a stated policy, not a model output.

**Confidence gate (everything else).** Route on a score; abstain below threshold
T. Four candidate signals, and we measure which is actually calibrated rather
than assuming:

- LLM self-reported confidence
- max retrieval similarity to a historical case
- margin of the TF-IDF logistic-regression classifier
- agreement between the LLM and TF-IDF classifiers

Comparing their reliability diagrams is a cheap and genuinely interesting result.

## 7. Baselines

- **Trivial**: majority-class intent; reply = BA's single most common canned
  response. Establishes the floor and, for reply quality, is often
  uncomfortably competitive - which is worth reporting honestly.
- **Simple**: TF-IDF + logistic regression for intent; reply = BA's historical
  response to the nearest-neighbour past message, copied verbatim. No LLM.

If the LLM agent does not clearly beat the simple baseline, *that is the
report's finding.*

## 8. Golden evaluation set (200 examples)

**Sampling.** Stratified, not random - random sampling yields about 90 examples
of the head intent and 2 of the tail. Allocation: proportional-with-floor across
provisional intents (minimum 15 each), plus a deliberate **hard/ambiguous
stratum** (about 25) sampled from low-confidence clusters and multi-issue
messages. Sampled from the held-out time range only.

**Labelling protocol.**

- Labels are applied **before any model output is seen**, to prevent anchoring.
- Each example gets: intent, auto-vs-escalate, escalation reason, and (for a
  100-example subset) a 4-axis quality judgement of BA's *actual* reply.
- A 25-example slice is labelled twice, several days apart, to estimate
  **intra-annotator agreement** - an honest ceiling on any accuracy number.
- Of the 100 reply-quality-labelled examples, a fixed **50-example subset is
  reserved solely for validating the LLM judge** and is never used to tune the
  judge rubric. The other 50 are available for rubric iteration.

Labelling is done by the candidate (the brief requires it, and it will be
questioned live). label_tui.py exists to make 200 labels take about 45 minutes.

## 9. Evaluation harness

| Layer | Metric |
|---|---|
| Intent | accuracy, macro-F1, per-class F1, confusion matrix, bootstrap 95% CIs |
| Reply | LLM judge, 4 binary axes: grounded / helpful / on-brand tone / no overpromise |
| **Judge** | **Cohen's kappa vs. the human labels on the held-out 50** |
| Escalation | coverage vs. harm-rate curve; calibration (reliability diagram, ECE) |

**Bootstrap CIs are mandatory on every headline number.** With n=200, a 3-point
difference between two systems is noise, and saying so pre-empts the obvious
interview question.

### 9.1 Cost model and threshold selection

Define k = cost(bad auto-reply) / cost(unnecessary escalation). The true value is
unknown, so k is a **CLI parameter**, and the report shows how the chosen
threshold and resulting coverage move as k ranges over 5 to 50. Picking a single
number and hiding the assumption would be the dishonest move.

## 10. "What is misleading about my headline number?" (planned content)

Drafted up front because it drives design, not written as an afterthought:

- n=200 golden examples means wide CIs; small differences are not real.
- Judge-human kappa is imperfect; every judge-derived number inherits that error.
- Intra-annotator disagreement caps achievable accuracy below 100%.
- Twitter replies are not email replies - length, register and expectations
  differ, so BA-on-Twitter performance does not transfer to Hiver's inbox use
  case.
- Grounding on historical BA replies inherits BA's own deflections; "matches what
  BA did" is not the same as "resolved the customer's problem."
- The corpus is from 2017; policies and tone have since changed.
- Harm rate is measured on sampled traffic, not adversarial traffic.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Free-tier rate limits stall iteration | Cache-first design; batch overnight; local embeddings |
| Multi-part reply reassembly is wrong | Unit tests on hand-checked threads |
| Retrieval leakage inflates results | Disjoint temporal split, asserted in tests |
| Judge agrees poorly with human | Report it honestly; it is a finding, not a failure |
| Labelling takes longer than planned | TUI tool; 150 is an acceptable floor |

## 12. Schedule (5-7 days, part-time)

| Day | Work |
|---|---|
| 1 | Setup, ingest, thread/multipart reassembly, taxonomy draft |
| 2 | Golden set labelling |
| 3 | Baselines + agent stages |
| 4 | Judge + agreement study |
| 5 | Risk-coverage, calibration, threshold, failure analysis |
| 6 | Report, decision log, README, clean-clone reproduction test |
| 7 | Buffer |

## 13. Deliverable mapping

| Brief requirement | Where satisfied |
|---|---|
| Runnable pipeline, under 15 min repro | make reproduce + committed LLM cache (3.1) |
| 150-250 hand-labelled golden set | Section 8, data/golden/golden.jsonl |
| Eval harness + judge + human agreement | Section 9, src/eval/ |
| Report: framing / baselines / failures / misleading | reports/REPORT.md, Sections 1, 7, 9, 10 |
| Decision log, 10-15 non-obvious decisions | reports/DECISIONS.md |
| Cite what was borrowed | README "Borrowed" section |
