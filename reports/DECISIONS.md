# Decision log

Eighteen non-obvious decisions made while building this project, and why.
Evidence for each (measurements, commits, code) lives in
`.superpowers/sdd/2026-09-11-ba-support-agent/progress.md`; this file states
the decision and the reasoning, not the full derivation.

## 1. British Airways over the intuitive AppleSupport pick

Selection was decided by measurement, not taste. Over the full 2.81M-row
corpus: BA has the highest support-density (39.1%) and lowest
DM-deflection (14.0%) and link-dumping (6.8%) of the brands considered.
AppleSupport was the intuitive choice (large, English, well-known) but
measured 52.5% deflection and 75.4% link-dumping — meaning most of its
"replies" are "please DM us" or a bare URL, which makes "ground a reply in
BA's historical resolution" vacuous. AmazonHelp had 4x the volume but 8.1%
Japanese and only 22.5% support-dense. The runner-up, Delta, was close but
BA's numbers were cleanly better on every axis that matters for grounding.

## 2. Temporal split (Oct 2017 corpus / Nov-Dec 2017 eval), enforced in code

Retrieval corpus and the golden eval set are disjoint time ranges
(`config.CORPUS_END = "2017-11-01"`), not a random split. A random split
would let the retriever pull the very thread it is later evaluated on,
inflating every downstream number. This is enforced, not just documented:
`Retriever.__init__` raises `ValueError` if the input frame has no `split`
column at all, so the anti-leakage guard cannot silently no-op if a future
caller forgets to pass split-tagged data.

## 3. Multi-part reply reassembly walks each tweet to its root parent, not a linked list

The plan assumed continuation tweets ("2/2") reply to the previous part's
tweet ID, i.e. a linked list. Measurement showed this is false for most of
the dataset: continuation tweets mostly reply to the *same* parent (the
customer's tweet) as the opening part. A naive linked-list walk would have
missed the majority of multi-part replies, truncating roughly 20% of
ground-truth replies to their first sentence. The shipped `build_pairs`
instead walks each agent tweet's `in_response_to_tweet_id` up through any
intermediate agent tweets to find the ultimate root, then groups every
agent tweet sharing that root as one reply, ordered by `(timestamp,
tweet_id)` — the tweet_id tiebreak matters because Twitter timestamps have
1-second granularity and pandas' default sort is not stable, so without it
reassembly order (and therefore the committed ground truth) would be
run-dependent.

## 4. Episode-splitting threshold (10 minutes) came from a measured gap distribution

Two reply parts of the same message can either be one continuous reply
episode or two separate replies BA sent to a conversation that had already
moved on; conflating the latter into ground truth silently fuses two
unrelated replies into one (observed: a duplicated "Hi Tom" greeting).
Rather than guess a cutoff, the inter-part gap distribution was measured
across every multi-part group: median 22s, p95 1.88min, p99 5.42min, then a
genuine dead zone with almost no gaps between 10 and 15 minutes before the
tail resumes (up to ~20 hours). The 10-minute threshold sits in that
measured dead zone — comfortably past p99 for real continuations, short of
the separate-episode tail — instead of the originally-planned 30-minute
fallback that would have had no empirical basis.

## 5. Agent signatures (`^Jane`) are split into a separate field

BA agents sign replies with a first name or initials. That's style, not
support content, so it is stripped out of `agent_text` into its own
`agent_signature` column during ingest. Leaving it in would let both
retrieval and the drafting LLM learn to sign outgoing replies as a specific
named human who does not exist and never agreed to anything — an
impersonation risk with no offsetting benefit.

## 6. The LLM response cache is a SQLite file committed to git

Every call in `src/llm.py` is keyed by `sha256(provider, model, prompt,
params)` and looked up in `cache/llm_cache.sqlite`, which is deliberately
*not* gitignored (see the explicit carve-out in `.gitignore`). This is the
mechanism that is supposed to let a reviewer clone the repo and run
`python -m src.cli reproduce` with zero API key and get the same numbers
back — the free-tier constraint and the 15-minute reproduction requirement
both depend on it. This was verified for real, not just designed: a clean
`python -m src.cli reproduce` with every API key explicitly unset completed
in 2 minutes 55 seconds and printed the same numbers as the live run.

## 7. Judge and drafter are on two different companies' models, not just two different models

The judge (`openai/gpt-oss-120b`, via Groq) never grades output from the
drafter (`gpt-4o-mini`, via OpenAI) — a stronger guarantee against
self-grading than merely picking two different model sizes from one
vendor. This wasn't the original plan: both roles started on Groq's free
tier (`qwen/qwen3.8-27b` drafting, `gpt-oss-120b` judging — a fourth
candidate, `qwen/qwen3.6-27b`, was measured and rejected for emitting
`<think>` blocks that break JSON parsing). The drafter moved to OpenAI for
the reason in decision 16. The *stronger* model stayed the judge, not the
drafter, because judge reliability gates every reply-quality and harm-rate
number in the report, while a small model is ample for drafting 1-2
sentence support tweets — giving the agent the weaker model is the honest
direction to err in.

## 8. Embeddings run locally (all-MiniLM-L6-v2), never via API

`src/embed.py` loads `sentence-transformers/all-MiniLM-L6-v2` locally and
disk-caches the resulting vectors by content hash. This removes rate limits
from the retrieval path entirely and makes retrieval results exactly
reproducible byte-for-byte, independent of any provider's model version
drift — a property an API-based embedding call could not offer.

## 9. Escalation is two layers: hard rules first, then a confidence gate

`decide.py` checks a fixed set of regex hard rules (`crisis_wellbeing`,
`money_claim`, `legal_or_safety`, `lost_baggage`, `account_lookup`,
`severe_distress`) before any confidence score is consulted, and any
`refund_compensation`-intent message is always escalated regardless of
score. These answer "should this category ever be automated" (a policy
question), while the confidence gate below the rules answers "is this
specific instance safe enough to route" (a calibration question). The
`crisis_wellbeing` rule was added after review specifically flagged that
self-harm/breakdown/panic language had no hard-rule coverage — a case where
a no-confidence-gate-can-save-you situation existed and had to be caught
regardless of model score. It is deliberately over-inclusive (a false
escalation costs one extra human minute; a missed one on crisis language is
unacceptable) — including a residual, documented gap: bare "desperate" is
excluded because it could not be separated from "desperate for a window
seat" without suppressing genuine crisis uses. That gap is a live-interview
answer, not an oversight.

## 10. The harm/escalation cost ratio `k` is a CLI parameter with a sensitivity sweep

`k = cost(bad auto-reply) / cost(unnecessary escalation)` has no true known
value. `risk_coverage.pick_threshold(curve, k)` takes it as an argument
rather than hardcoding one, and `sensitivity()` sweeps it across a range
(2, 5, 10, 20, 50) so the report shows how the chosen threshold and
resulting coverage move as the assumption changes. Picking one number and
presenting it as if it were known would be the dishonest version of this
result.

## 11. Two places where missing or bad data are never allowed to read as a good number

First: `judge.parse_verdict` fails an axis to `False` — never `True` — both
when the judge's output is unparseable JSON *and*, separately, when a
judge-supplied value is a stringified boolean. The stringified-boolean case
was a real, caught bug: Python's `bool("false")` is `True`, so a judge that
responded with the string `"false"` would have silently flipped a failing
axis into a passing one, inflating the reported safety numbers. A strict
`_coerce_bool` now recognises only real booleans, 0/1, and
`true/false/yes/no` (case-insensitively); anything else defaults to
`False`. Second: `risk_coverage_curve` reports `harm_rate` as `NaN`, not
`0.0`, at any threshold where zero messages are auto-handled — a `0.0`
there would read as "0% harm rate" (perfect safety) when it actually means
"no data at this threshold," which is a materially different claim.
`pick_threshold` also refuses to select a threshold backed by fewer than
`min_auto=10` auto-handled examples, falling back to a threshold explicitly
flagged `degenerate: True` rather than presenting an unreliable estimate as
a confident pick. In both cases, judge flakiness or thin data can only make
the project's own numbers look worse, never better.

## 12. Golden set stratified with a per-intent floor plus a deliberate "hard" stratum

Random sampling over the eval split would give roughly 90 examples of the
head intent and 2 of the tail ones, making macro-F1 meaningless.
`sampling.stratified_sample` allocates a floor per intent-derived cluster
(15) and fills the remainder proportionally, then `build_golden_pool` sets
aside a further 25-example "hard" stratum — the bottom 10% by cosine
similarity to each example's own cluster centroid, i.e. genuinely ambiguous
messages — sampled separately so the golden set is not just easy, clearly-
clustered examples.

## 13. Labels are collected blind, twice over

`label_tui.py`'s intent/action mode shows the labeller only the customer's
message; BA's actual historical reply is deliberately never shown, so the
human's intent/action judgement cannot be anchored by what BA actually did.
The separate quality mode (which *does* need to show BA's actual reply,
since judging that reply is the task) writes to a different file
(`quality.jsonl`) so the two judgement tasks can never contaminate each
other. All labels are collected before any model output (classification,
draft, or judge verdict) is seen.

## 14. The taxonomy is a measured output, not the plan's assumption

The design spec's skeleton taxonomy included `checkin_boarding`. Clustering
the actual corpus (KMeans, k=12, MiniLM embeddings) showed no cluster with
distinct check-in/boarding-pass language — the nearest cluster was
Manage-My-Booking/app/website errors — so `checkin_boarding` was dropped
and folded into `booking_management` rather than kept for the sake of
matching the plan. Conversely, `praise_or_social` was added: three clusters
totalling ~25% of the sampled corpus were compliments, banter, or reactions
to marketing content — real, high-volume traffic an auto-responder must
acknowledge differently from a support ticket, and none of it was in the
plan's skeleton. Both changes are logged with the cluster evidence in
`data/interim/taxonomy.json`.

## 15. A `LogisticRegression(C=10.0)` change was reverted after review

An implementer had set `C=10.0` on the simple TF-IDF baseline specifically
because the plan's own 6-row toy test fixture let the class-imbalance
intercept dominate word evidence at the default `C=1.0`, failing that test
on sklearn 1.7.2. This was provisionally accepted, then reverted on review:
changing a production hyperparameter to satisfy a toy fixture — never
validated against the real 7,825-row corpus — moves the very bar the LLM
agent is measured against, which is backwards. The fixture was fixed
instead (a balanced, 12-row set with varied vocabulary), and the baseline
reverted to `C=1.0` (sklearn's default).

## 16. The drafter moved from Groq's free tier to a paid OpenAI key mid-project

Groq's free tier caps at 200,000 tokens/day, account-wide, for the drafter
model. That is smaller than the ~2,200 drafter-role calls this project
needs (2,000 for baseline training alone), so the free tier made the real
evaluation run a genuine multi-day, quota-limited process — confirmed
directly by running it and watching it exhaust mid-run twice. A separate,
independent bug compounded this: raising `MAX_WORKERS` to 8 to try to go
faster made things *worse*, because the real bottleneck was a 7,000-
tokens-per-minute cap, not connection latency — 8 workers retrying in
lockstep after a shared 429 collided again on their very next attempt,
risking a permanently failed item rather than any speedup (fixed by
setting `MAX_WORKERS=1`; see `src/pipeline.py`'s comment for the measured
numbers). Neither problem is solvable by writing better code against a
free tier with a hard account-wide ceiling. A paid OpenAI key removes it:
verified live at 200,000 tokens/*minute* with no observed daily cap,
costing an estimated $2-5 for the entire remaining workload at
`gpt-4o-mini` pricing ($0.15/$0.60 per million tokens). `gpt-4o-mini` was
chosen deliberately over newer models already available on the same
account (`gpt-5.x`, `gpt-6-astra`) because those postdate this project's
own model-assisted development and risking a ~2,200-call unattended run on
an unfamiliar model's behaviour is exactly the mistake that caused the
`gpt-oss-120b` empty-response bug (decision 11) in the first place — a
model should be live-verified before being trusted at volume, not assumed
safe because it is newer.

## 17. The first quality-label pass was rejected and partially redone

The human's first full pass over the 100-example quality set (judging BA's
actual historical replies on 4 axes) came back statistically implausible:
100/0, 100/0, 100/0, and 97/3 — near-unanimous "yes," inconsistent with
independently measured corpus facts (14% of BA replies deflect to DM,
6.8% are link-only). Rather than accept it, a random 25-example subset was
relabelled with explicit attention to three concrete patterns (a
DM-deflection with no real answer is not `helpful`; a specific promise is
an `overpromise`; an off-tone or robotic reply is not `on_brand`). The
redone 25 showed real variance (`helpful`: 18 yes / 7 no). The final
100-example file keeps the careful redo wherever it exists and the
original elsewhere (75 examples), producing a file that is honest about
being a mixed-confidence label set — see decision 18 for what this still
costs the judge-agreement result.

## 18. The judge-vs-human agreement came back weak, and that is reported as the finding, not hidden

Cohen's kappa between the LLM judge and the human on the same 100 BA
replies: -0.020 (`grounded`), 0.071 (`helpful`), 0.030 (`on_brand`), -0.019
(`no_overpromise`) — poor to slight on every axis. Two different things are
happening and the report distinguishes them rather than presenting one
number: on `no_overpromise`, raw agreement is high (89%) but kappa is
negative, the textbook signature of a near-constant human rater (99%
"yes," largely because BA staff are trained not to promise specifics on
Twitter) making kappa's chance-correction unstable regardless of the
judge's real quality. On `grounded`/`helpful`/`on_brand`, raw agreement
itself is only 52-61% — the judge is substantively disagreeing with the
human, not just suffering a skewed baseline. The decision here is
editorial, not technical: report both readings side by side rather than
collapsing to "kappa is low, judge is bad" or explaining it away as "just
a skewed sample" — the mandatory ask was evidence of agreement, and weak
evidence, correctly characterised, is still the honest answer.
