# Findings

The measured record behind `system1`: what was tested, what it cost, and which
guesses turned out to be wrong. `README.md` is the usable document; this is the
working-out, kept because several of these results argue against the obvious
choice.

Every number here was produced by the scripts in this repo, on CLINC150 unless
stated otherwise. Where a finding has been superseded, the later result is the
one stated. Forward-looking decisions live in `NEXT_STEPS.md`; how this engine
compares to other decision models is in `RELATED_WORK.md`.

---

# What the engine is made of

## Class representations: examples beat descriptions

The baseline embeds one handwritten description per class. `anchors.py` can
instead embed *k* real utterances from the train split and use their centroid —
SetFit's data without SetFit's training, which is worth measuring before
reaching for a trainer. Selection ran on validation only:

| representation | val accuracy | OOS AUROC |
|---|---|---|
| description only | 0.845 | 0.995 |
| desc + 4 examples | 0.945 | 0.999 |
| 8 examples (centroid) | 0.940 | 0.999 |
| **16 examples (centroid)** | **0.965** | **0.999** |
| 16 examples (max-sim) | 0.895 | 0.998 |
| 32 examples (centroid) | 0.955 | 0.999 |

Centroids beat max-similarity everywhere, and the handwritten description stops
helping once there are ~8 real examples — at k=16 it is actively slightly
worse. **A sentence describing a class is a worse anchor than a handful of
things people actually said.**

On test, against the zero-shot baseline, each with its own fitted calibration:

| | description only | 16 examples |
|---|---|---|
| in-scope accuracy | 0.880 | **0.933** |
| errors | 36/300 | **20/300** |
| ECE | 0.043 | 0.041 |
| OOS AUROC (max cosine) | 0.989 | **0.994** |
| OOS rejected / in-scope kept | 93.6% / 98.3% | **97.5% / 96.7%** |
| escalation needed for ≤5% error | 17.5% | **1.5%** |

A 44% reduction in errors for 160 labels and no training. It also mostly
dissolves the escalation problem: validation error at *zero* escalation fell
from 15.0% to 3.5%, so tier 2 is called for a handful of inputs rather than a
sixth of traffic.

Three caveats that matter:

* **Validation said 0.965, test delivered 0.933.** That 3-point gap is the cost
  of choosing k on validation — real, expected, and the reason the choice was
  not made on test.
* **k=8, 16 and 32 are statistically indistinguishable.** With 200 in-scope
  validation items the standard error near 0.95 is about 0.015. 16 is the
  argmax of a noisy sweep, not a demonstrated optimum.
* **It is no longer zero-shot.** The baseline needed one sentence per class;
  this needs 16 labelled examples per class. `cascade.py`'s support-ticket
  schema has none, so it still runs on descriptions — see *Where descriptions
  run out*, below.

## Which encoder

`build_encoder.py` compiles any BERT-architecture sentence encoder, so this is
a flag rather than a rewrite. Each row has its own fitted calibration:

| encoder | params | ms/query | accuracy | ECE | OOS AUROC | errors |
|---|---|---|---|---|---|---|
| MiniLM-L6 | 22.6M | 1.18 | 0.933 | 0.041 | 0.994 | 20/300 |
| **BGE-small** | 33.2M | 1.75 | **0.953** | 0.034 | 0.995 | 14/300 |
| BGE-base | 108.9M | 4.07 | 0.957 | 0.026 | 0.997 | 13/300 |

Returns fall off a cliff. Going 22.6M → 33.2M costs 0.57 ms and buys **+2.0
points**; going 33.2M → 108.9M costs 3.3× the parameters and buys **+0.4**.
BGE-small is the knee.

**But BGE is not a drop-in for an uncalibrated schema.** On the support-ticket
fixtures its similarities overlap:

```
                 MiniLM          BGE-small
clear         0.319 - 0.594    0.616 - 0.794
out-of-scope  0.070 - 0.169    0.488 - 0.662   <- overlaps clear
```

MiniLM separates cleanly; BGE compresses everything into a narrow high-cosine
band, which **ranks better and thresholds worse**. No single `min_sim` works
there. So `cascade.py` stays on MiniLM, and BGE is opt-in via `--encoder bge`
on the paths that have fitted thresholds. Better on the benchmark, not
automatically better in place — the same lesson as calibration not transferring
between schemas, arriving from a different direction.

## A second dataset: MASSIVE

CLINC is the dataset this project argues with, because it ships an explicit
out-of-scope class. It is also a dataset nobody else reports on, and the 10-way
banking router in `evalset.py` is a schema of our own carving — so the headline
number has had nothing to sit beside it. `massive.py` adds
`mteb/amazon_massive_intent`, which other decision models do publish on.

Same engine, same encoder, same k=16 centroids, no training, no descriptions:

| | CLINC (ours, 10-way) | MASSIVE (60-way) |
|---|---|---|
| in-scope accuracy | 0.933 | **0.710** |
| ECE | 0.041 | **0.032** |
| test items | 300 | 2,974 |
| validation → test | 0.965 → 0.933 | 0.725 → 0.710 |

**0.710 on 60 intents from a 22.6M-parameter encoder with nothing trained.** For
reference, Laya reports 0.783 on MASSIVE intent from 421M parameters with an RL
training stage — 7.3 points ahead at 19× the size. That is their number on their
harness and split, not a head-to-head; see `RELATED_WORK.md`. The
validation-to-test gap is 1.5 points against CLINC's 3.2, which is what a
ten-times-larger test set buys.

Two things do not transfer, and both are the dataset's doing rather than the
engine's:

* **Refusal cannot be measured here at all.** MASSIVE has no out-of-scope class,
  so OOS AUROC, the operating curve and `min_sim` are all unavailable.
  `calibrate.py` still emits a `min_sim` quantile and now prints a warning
  saying it is an in-scope quantile that has been shown to reject nothing, and
  records `oos_validated: false` in the JSON. The 0.994 in `README.md` is
  CLINC's and stays CLINC's.
* **Two classes cannot supply k=16.** MASSIVE's train split ranges from 810
  examples to 4, so `music_dislikeness` and `cooking_query` contribute what they
  have. 59 of the 60 intents appear in validation and test.

## What MASSIVE says about confident errors

The prediction going in was that a 60-way schema would be *more* crowded than a
10-way one and would therefore produce more undetectable errors. **It did not.**
The share is essentially identical:

| | CLINC | MASSIVE |
|---|---|---|
| wrong at p ≥ 0.90, share of in-scope | 2.3% (7/300) | **2.2% (64/2,974)** |
| of those, flagged by the gates | 0/7 | **5/64** |
| error rate *among* confident items | 0.026 | 0.059 |
| items reaching p ≥ 0.90 | 265/300 (88%) | 1,090/2,974 (37%) |

The matching top line is a coincidence of the bottom two moving in opposite
directions. On the harder task the engine is confident about far fewer inputs
(37% against 88%) and wrong more often when it is (0.059 against 0.026). Those
cancel. **What survives is that calibration is doing its job** — confidence
tracks difficulty across a 22-point accuracy gap — and that the gates still
catch almost nothing: 59 of 64 confident errors are silent, against 7 of 7.

The confusions are adjacency again, and more so among the confident ones. MASSIVE
intents carry a scenario prefix (`calendar_`, `email_`, `iot_`), which makes
adjacency checkable rather than a matter of opinion:

```
                            within-scenario
all 863 errors                 42.4%
the 64 confident errors        51.6%

   8  email_sendemail  -> email_addcontact
   6  general_quirky   -> general_greet
   5  transport_query  -> transport_ticket
   2  cooking_recipe   -> cooking_query
```

So the *rate* of confident error is not a function of how many classes there are.
It is a function of how many **neighbours** each class has, and MASSIVE's 60
intents are spread across 18 scenarios rather than packed into one domain the way
our ten banking routes are.

---

# Making confidence mean something

## Calibration, and how a toy test set got it backwards

The knobs are fitted on validation by NLL and written to `calibration.json`;
`Choice()` reads it. Fitting changed nothing about accuracy — temperature
cannot move an argmax — and everything about whether `prob` means anything:

| | fixture-tuned | fitted on validation |
|---|---|---|
| temperature | 0.10 | **0.0469** |
| `min_sim` | 0.25 | **0.3085** |
| `min_margin` | 0.30 | **0.40** |
| test ECE | 0.237 | **0.043** |
| mean confidence vs accuracy | 0.649 vs 0.880 | 0.860 vs 0.880 |

An earlier version of the README contained a temperature sweep over sixteen
hand-written inputs, concluding that 0.05 was "far too sharp and never
escalates" and that 0.10 separated cleanly. On real data 0.05 is almost exactly
right (ECE 0.033) and 0.10 is badly wrong (ECE 0.196) — the engine was left
*under*confident, 23 points below its own accuracy, which is the opposite of
the overconfidence the fixtures appeared to show.

So the sweep did not merely produce an imprecise number. **It argued
convincingly for moving away from a correct initial guess.** That is the cost of
tuning against a test set written by the same person who wrote the thing being
tested.

## Confident errors, and why no threshold catches them

ECE averages the whole reliability diagram. It is silent about the tail that
actually costs something: an answer that is wrong *and* sure. `evaluate.py` now
reports it directly, in-scope, on the 16-example test configuration:

| p(top) ≥ | items | wrong | error among them | share of in-scope |
|---|---|---|---|---|
| 0.90 | 265 | 7 | 0.026 | **2.3%** |
| 0.95 | 253 | 4 | 0.016 | 1.3% |
| 0.99 | 231 | 1 | 0.004 | 0.3% |

Seven items out of 300 — so **roughly a third of all 20 errors are confident
ones**. And of those seven, `min_sim` and `min_margin` flag **zero**:

```
min_sim = 0.4286                      min_margin = 0.05

   p   marg   sim   predicted      truth              text
0.99   0.99  0.65   credit_limit   transactions       is my visa bill over my limit this month
0.99   0.98  0.61   report_fraud   report_lost_card   my bank of america visa platinum was swiped from my belongings
0.98   0.97  0.49   credit_limit   pay_bill           pay $175 on my visa
0.97   0.94  0.68   report_fraud   report_lost_card   someone used my chase card without my authorization
0.95   0.92  0.58   credit_limit   transactions       what were my last 10 charges on my credit card
0.94   0.89  0.60   balance        transactions       how much have i spent on my debit card this month
0.91   0.82  0.66   bill_due       pay_bill           today my electric bill will be paid, or should be
```

Every one clears both gates by a wide margin. That is not a threshold that needs
tuning — **the signals are working correctly and the answer is still wrong.**

The reason is visible in the label pairs. All seven are *adjacent intents*:
`transactions` against `credit_limit` and `balance`, `pay_bill` against
`bill_due`, `report_lost_card` against `report_fraud`. These inputs sit
genuinely between two labels — *"someone used my chase card without my
authorization"* is a defensible `report_fraud` — so the embedding is close to
the wrong anchor **because it should be**. Confidence measures distance to an
anchor, and the distance is small. There is nothing left for a threshold to see.

This is the same mechanism as the masking result below, arriving from a third
direction: a schema whose classes are semantically adjacent has failures that no
post-hoc signal computed from that schema can detect.

**Calibration made this worse, not better, and that is correct.** Sweeping the
temperature:

| temperature | reach p ≥ 0.90 | wrong | share of in-scope |
|---|---|---|---|
| 0.1000 | 69/300 | 0 | 0.0% |
| 0.0300 | 258/300 | 6 | 2.0% |
| **0.0260** (fitted) | **265/300** | **7** | **2.3%** |
| 0.0200 | 281/300 | 14 | 4.7% |

The zero at T=0.10 is not safety. Only 69 of 300 items reach 0.90 at all — the
engine scores no confident errors by being too underconfident to be confident
about anything, which is exactly the state *Calibration, and how a toy test set
got it backwards* describes and exactly what fitting moved away from. **Trading
ECE 0.196 for ECE 0.041 bought seven confident errors, and was still the right
trade.** The metric to watch is not one or the other.

For external reference, kev reports the same metric — wrong answers at p ≥ 0.9 —
at 4.0% after fitting, against 8.7% before, on its own data. Our 2.3% is on a
different dataset with a different harness and is not a head-to-head. See
`RELATED_WORK.md`.

## The escalation curve

This is what a cascade is for, and the reason accuracy alone is the wrong
headline. Sweeping `min_margin` over in-scope validation items:

| margin | escalated | error among kept |
|---|---|---|
| 0.00 | 0.0% | 0.150 |
| 0.20 | 11.0% | 0.084 |
| **0.40** (fitted) | **17.5%** | **0.048** |
| 0.60 | 21.5% | 0.038 |
| 0.90 | 45.5% | 0.000 |

**Handing 17.5% of traffic to tier 2 cuts the error rate from 15.0% to 4.8%.**
The remaining errors are reachable too, at 45.5% escalation — the curve does go
to zero, which is the useful property. Pick the point your latency and cost
budget allows; `--target-error` sets it.

Two things to remember about this table. `min_margin` is computed after the
softmax, so it only means anything at the temperature it was fitted at — refit
it whenever `temp` changes. And this is the **description-only** configuration,
which is why it starts at 15.0% error; with 16-example centroids the same curve
starts at 3.5% and needs only 1.5% escalation to hit the 5% target. The shape is
the point; the absolute numbers moved once the anchors did.

## The operating curve

`min_sim` is fitted to retain 99% of in-scope traffic, since losing real traffic
is the expensive error and rejecting out-of-scope is the cheap win. On test:

| threshold | OOS rejected | in-scope kept | accuracy on kept |
|---|---|---|---|
| 0.20 | 80.9% | 99.3% | 0.886 |
| 0.25 | 90.2% | 99.0% | 0.889 |
| **0.3085** (fitted) | **93.6%** | **98.3%** | **0.892** |
| 0.35 | 96.3% | 93.3% | 0.914 |
| 0.40 | 98.2% | 87.0% | 0.927 |

Validation predicted 95.0% / 99%; test delivered 93.6% / 98.3%, so the threshold
generalises.

## What a question costs, once the input is embedded

`State` embeds an input once and answers every question from that one vector.
Sixteen support tickets, three questions each — a `Choice` over 5 routes, a
`Score` over a 4-level rubric, and a `Boolean`:

| | ms per ticket |
|---|---|
| three questions, three separate calls | 3.46 |
| three questions, one `State` | **1.13** |
| the three questions alone, vector already computed | **0.025** |

**The encode is 1.10 ms; the three typed decisions on top of it are 25 µs.** A
44:1 ratio, and a fourth question costs about 8 µs — a `(384, K)` matmul and a
softmax. Cost here is per *input*, not per *decision*, which is a different
scaling law from every engine in `RELATED_WORK.md`: kev caches a state
representation across questions to get 861 ms down to 242 ms, Laya batches ten
questions to reach 7.2 ms each from 32.8 ms. Both are engineering their way
toward what falls out of an embedding architecture for free.

The limit is the honest half. These questions are independent — each is a dot
product against its own anchors, and none of them can see another's answer.
jevfire is explicit about the same constraint with a 27B model behind it
("fields score independently, no cross-field conditioning"). A question whose
answer depends on another's is not expressible here, and no amount of sharing
the vector changes that.

## Serving it, and making the calibration finding structural

`serve.py` puts the engine behind `POST /v1/systemone`, the API kev serves from
a 0.8-9B model. Throughput on a laptop, stdlib `http.server`, one lock around
the encoder because Core ML's `predict` is not re-entrant:

| | req/s | server-side p50 | p95 |
|---|---|---|---|
| 1 client, one question | 225 | 2.25 ms | 4.07 ms |
| 16 clients, one question | **708** | 1.21 ms | 2.56 ms |
| 32 clients, one question | 666 | 1.27 ms | 2.36 ms |
| 16 clients, three questions | 660 (**1,980 decisions/s**) | 1.28 ms | — |

It saturates at ~700 req/s by 16 clients, which is the serialised encoder: 1/708
is 1.41 ms, the single-threaded encode time. The three-question row is the
`State` result arriving over HTTP — three typed decisions for the price of one
encode, so decisions/second is 3x requests/second at the same latency.

Two things about serving *this* engine that a prompted model does not have:

**An option is a vector, not a string in a prompt.** A question whose criteria
the server has not seen must be compiled first, at one encode per option: 33 ms
for a 3-option choice plus a 4-level score plus a noul, against 1.5 ms once
cached. Cold cost scales with the *schema*, warm cost does not scale at all.

**The calibration finding is now enforced rather than documented.** Compiled
schemas are keyed by a hash of their criteria, and a registered calibration is
keyed the same way. So:

```
register calibration for {"billing": "payments, charges and refunds", ...}
  -> answers carry "calibrated": true, "escalate": false

change one word to "payments, charges and returns"
  -> different hash, different vectors, different schema
  -> answers carry "calibrated": false, "escalate": null
```

`escalate` is `null` when uncalibrated, never `false`. This repo has found the
same bug from four directions -- CLINC's `min_sim` refusing in-scope tickets,
BGE's similarities overlapping in an uncalibrated schema, the toy temperature
sweep, MASSIVE having no out-of-scope rows to fit against -- and every time it
was silent. A server is the first place where it could be made *impossible*
rather than merely written down, because the server owns the boundary between a
schema and its thresholds.

---

# What it cannot do

## Two kinds of not-knowing

The engine can fail to know in two unrelated ways, and only one of them survives
a softmax:

| | signal | question it answers |
|---|---|---|
| ambiguous between labels | `margin` (post-softmax) | *which* of these? |
| outside the schema | `sim` (raw max cosine) | *any* of these? |

Raw cosine separates them cleanly:

```
clear          0.319 - 0.594
ambiguous      0.217 - 0.441
out-of-scope   0.070 - 0.169
```

The softmax normalises magnitude away, so margin alone cannot see the second
case: *"I'd like to speak to a manager"* scores a higher margin (0.34) than the
genuinely ambiguous *"it broke"* (0.16).

**On fixtures, `min_sim` appeared to add nothing** — margin and probability
already escalated 4/4 out-of-scope inputs there. On CLINC150 it is decisively
the best signal available (AUROC 0.989 against 0.894 for margin). Sixteen inputs
could not show it.

What `sim` *also* supplies is the **reason**, which determines the right action —
and here the action matters more than the detection, because tier 2 is bound to
the same enum and is therefore also forced to pick a wrong label:

```
what are your office hours   -> tier 2 says: feature
happy holidays everyone      -> tier 2 says: feature
```

So `cascade.py` short-circuits `out-of-scope` to "no category" instead of
escalating. Cheaper *and* more correct than asking a 950× more expensive model
the same impossible question.

## Masking is a correctness assumption, not a safety net

`route(text, allowed=[...])` decides among a subset, applying the mask before
the softmax so probabilities renormalise over the legal set. `Choice(...,
always_escalate={"feature"})` sends a label to tier 2 regardless of confidence,
with `reason="structural"` — some decisions need a *bigger* model rather than a
*more certain* one, and a confidence threshold cannot express that.

The masking result is the uncomfortable one. Removing each test item's true
label from the legal set, the engine flags only **30.7%** of the resulting
forced-wrong decisions, at a mean confidence of 0.843. The other 69% are
confidently mislabelled into whatever neighbour remains — these ten banking
intents are semantically adjacent, so `balance` masked out still leaves things
that look like it.

**If your mask is wrong, the engine will comply without complaint.** A mask
derived from structure (a checkbox cannot accept typed text) is safe; a
heuristic mask is exposed.

**But 30.7% is a property of this schema, not of the method.** The same test on
MASSIVE's 60 intents flags **89.3%** of forced-wrong decisions, at a mean
confidence of 0.532 against 0.843:

| | CLINC, 10 banking routes | MASSIVE, 60 intents |
|---|---|---|
| flagged rather than silently mislabelled | 30.7% | **89.3%** |
| mean p(top) on forced-wrong decisions | 0.843 | **0.532** |

The direction is the opposite of the obvious one: the schema with **six times as
many labels is far safer under a wrong mask.** Masking `balance` out of ten
adjacent banking intents leaves `transactions` and `credit_limit`, which look
close enough to be answered confidently. Masking `alarm_set` out of sixty
intents spread over eighteen scenarios leaves nothing nearby, the similarities
collapse, and the gates fire.

So the risk is not label count. It is **neighbour density** — how much of the
schema sits next to any given class. A large, well-spread schema tolerates a bad
mask; a small, tightly-clustered one does not, which is the reverse of the
intuition and the reverse of what we expected before running it.

A third point confirms it by construction. MASSIVE's labels are
`scenario_intent`, so restricting a decision to one scenario builds a small,
deliberately adjacent schema — exactly the dangerous shape:

| schema | flagged | mean p(top) when forced wrong |
|---|---|---|
| MASSIVE, all 60 intents | 89.3% | 0.532 |
| **MASSIVE, within one scenario** | **47.8%** | **0.756** |
| CLINC, 10 banking routes | 30.7% | 0.843 |

Narrowing a 60-way decision to its own scenario roughly halves the chance of
noticing a bad mask. This is the cost side of hierarchical classification, and
it is worth knowing before anyone reaches for it — see `NEXT_STEPS.md`, where
the accuracy side turned out not to pay either.

## Perfectly reproducible, and badly diluted

TypeSafe's self-consistency cookbook runs one rubric on one post fifteen times,
varying only a throwaway `uid` field, and reports Jev agreeing with itself
**90.8%** of the time — 99.2% if you refuse to act below p=0.60, which costs
25.8% of traffic to human review. kev ships a `/v1/systemone/permute` endpoint
for the same reason. `stability.py` runs the equivalent here.

**Determinism is exact.** Same text, fifteen runs, and a second `Encoder` over
the same package:

```
embeddings bit-identical:     14/14 repeat runs
max |delta| across runs:      0
fresh Encoder, same package:  bit-identical, max |delta| 0
```

Not "0.9999" — zero. There is no sampling anywhere in the path, so a restarted
server agrees with the one it replaced, and option order cannot matter because
a dot product has no order. An entire cookbook and an endpoint exist to manage
a problem this architecture does not have.

**Their actual experiment is about irrelevant text, though, and there we are
merely better rather than immune.** Repeating it — same input, a fresh random
`uid` in a JSON state on every call:

| | self-agreement |
|---|---|
| Jev (TypeSafe's report) | 90.8% |
| **system1** | **97.8%** |

4 of 40 inputs ever flip, and p(top) moves by 0.057 on average. The uid changes
the string, so it changes the embedding; this is sensitivity to irrelevant
text, not nondeterminism, and the two should not be conflated.

**And on that axis this is the worst result in the repo.** Mean pooling
averages every unmasked token, so irrelevant text is not a distractor competing
for attention — it is arithmetically mixed into the vector in proportion to its
length:

| filler | mean tokens | signal share | same label | mean sim |
|---|---|---|---|---|
| none | 12 | 100% | 100/100 | 0.662 |
| 1× | 54 | 21.8% | **63/100** | 0.466 |
| 2× | 96 | 12.3% | **52/100** | 0.459 |
| 3× | 138 (truncated) | 9.3% | 79/100 | 0.480 |
| 6× | 264 (truncated) | 9.3% | 79/100 | 0.480 |

**One paragraph of unrelated boilerplate changes 37% of decisions.** Jev 1.13's
jaggedness list has this as a known failure ("large irrelevant state… acts as a
distractor"), and mean pooling is a more direct way to suffer from it than
attention is. The last two rows are identical because `SEQ = 128` truncates
them into the same 128 tokens, which caps signal share at 9.3%.

The non-monotonicity is unexplained: 12.3% signal retains fewer labels than
9.3% does. Not chased, because the actionable result does not depend on it —
**filter state in code before sending it**, which is TypeSafe's own advice for
their model and applies with more force here.

## `Score` is weak, but its uncertainty is not

Embedding similarity captures topic, not intensity. On a 0–3 anger rubric at the
default temperature:

| truth | scored | confidence | message |
|---|---|---|---|
| 0 | 1.45 | **0.31** | "thanks for the help!" |
| 1 | 1.25 | 0.64 | "this is mildly annoying" |
| 2 | 1.74 | 0.72 | "I've been waiting three weeks and nobody has replied" |
| 3 | 1.68 | **0.32** | "ABSOLUTELY UNACCEPTABLE, I am done with this company" |

Everything collapses into 1.25–1.74: there is essentially no dynamic range, and
the two extremes are the two it gets most wrong. But `confidence` is 0.31 and
0.32 on exactly those, against 0.64–0.72 on the two it gets closest. **The
uncertainty signal survives where the estimate does not** — so treat `Score` as
a detector of "this needs a real model", not as a measurement.

## Where descriptions run out

`calibration.json` belongs to the CLINC banking routes and does not transfer:
its `min_sim` of 0.3085 refuses genuinely in-scope support tickets at 0.27–0.28.
`calibrate_tickets.py` therefore fits the ticket schema on its own data —
`tickets.VALIDATION`, 72 tickets disjoint from the fixtures `cascade.py` prints,
so the demo stays a held-out test rather than a restatement of the fit.

Temperature and `min_margin` fit cleanly. `min_sim` does not, and the reason is
the anchors:

```
in-scope  max-cosine range   0.051 - 0.572
oos       max-cosine range   0.061 - 0.279   <- contained inside in-scope
```

A concrete bug report sits *below* chatty out-of-scope text:

```
0.051  [bug]  dates display as 1970 on the dashboard
0.068  [bug]  the export button produces an empty csv every time
...
0.279  [oos]  I have attached the document you asked for
0.227  [oos]  unsubscribe
```

`bug` is anchored on the description *"the software crashes, errors, freezes or
behaves incorrectly"*. **A real bug report describes a symptom, and shares almost
no vocabulary with the abstract category.** This is *Class representations*
arriving from the failure side: on CLINC, 16 example centroids took OOS AUROC to
0.994; here, with one handwritten sentence per class, there is no threshold that
both keeps in-scope traffic and rejects out-of-scope.

At the 0.99 retention target `calibrate.py` uses, the fitted threshold collapses
to 0.0597 — under the entire out-of-scope range — and rejects **nothing**. The
script now warns when that happens, and defaults to 0.80 retention, the knee:

```
 thresh  in-scope kept  oos rejected
   0.05        100.0%         0.0%
   0.10         92.0%        22.7%
   0.15         84.0%        63.6%
   0.20         78.0%        81.8%  <- fitted (0.1937)
   0.25         66.0%        95.5%
   0.30         50.0%       100.0%
```

Example centroids were tried as a fix and did not rescue it: 6 hand-written
examples per route, scored on the 20 held out, gave OOS AUROC 0.764 — *worse*
than the description baseline. The examples are written by the same person as
the descriptions and inherit the same blind spots, which is precisely what
`evalset.py` exists to avoid. CLINC's examples are human-written and collected
independently; these are not.

**The honest reading:** the demo schema has no data behind it, so its thresholds
rest on 72 strings written by the author of the engine. That is enough to place a
threshold and not much more. The CLINC numbers in `README.md` are the ones that
carry evidential weight.

---

# Build notes

## fp16 costs one decision in three thousand

Apple ships face *recognition* in a non-ANE build while face detection, quality
and pose run on the ANE, which suggested fp16 might hurt threshold decisions on
embeddings. The first answer here was ten hand-written strings at K=5 — the same
fixtures-for-data mistake as the temperature sweep. `drift.py` runs the whole
evaluation set instead, anchors and queries both embedded by each path so anchor
drift is included, and at two label counts because the standing caveat was that
precision should matter more as K grows:

| fp16 ANE vs. fp32 torch | CLINC, K=10, 1,300 rows | MASSIVE, K=60, 2,974 rows |
|---|---|---|
| embedding cosine, mean | 0.9999816 | 0.9999774 |
| embedding cosine, min | 0.9999534 | 0.9998997 |
| identical decision | 1,295/1,300 (99.62%) | **2,971/2,974 (99.90%)** |
| — in-scope | **300/300 (100%)** | 2,971/2,974 |
| — out-of-scope | 995/1,000 (99.50%) | (none) |
| mean \|Δp(top)\| | 0.00178 | 0.00106 |
| max \|Δp(top)\| | 0.01219 | 0.01389 |
| flips that broke a correct answer | **0** | **1** |

**On CLINC every in-scope decision is identical.** All five disagreements are
out-of-scope items, which have no correct label to lose and are refused on `sim`
— a magnitude, which barely moves — rather than on the argmax.

At K=60 fp16 costs exactly one correct answer out of 2,974, and it is this one:

```
0.489 -> 0.489   iot_hue_lightoff -> iot_hue_lighton   "please turn lights off"
```

The two anchors are near-equidistant and the engine was reporting p=0.489 on
both paths — it was already saying it could not tell. **The caveat was right
about the direction and wrong about the size:** going from 10 labels to 60 does
move drift from zero to non-zero, and the magnitude is 0.03%, on a decision that
was a coin flip in fp32 as well.

## The CPU-only path silently drops the normalisation

Found while building `drift.py`, not while looking for it. `Encoder` takes a
`compute_units` argument, and on one setting it returns garbage:

| compute_units | ‖embedding‖ |
|---|---|
| CPU_AND_NE (default) | 1.000 |
| CPU_AND_GPU | 1.000 |
| ALL | 1.000 |
| **CPU_ONLY** | **19.596** |

The last op in `minilm.py` is `pooled * rsqrt(Σpooled² + 1e-12)`. On the CPU-only
backend Core ML does not apply it, and the raw pooled vector comes back instead.
Renormalising by hand recovers agreement with the ANE path to cosine 0.99997, so
the encoder is fine — it is the final op alone.

Everything downstream assumes unit vectors: cosine scoring, `min_sim`
thresholds, anchor centroids. With norms near 19.6 the "similarities" run to
17.5, every `min_sim` comparison passes, and the softmax saturates — **the
failure is silent, total, and looks like confident success.** It would have been
invisible in any accuracy-only test, because the argmax of a dot product is
mostly preserved; only the calibrated parts break.

No published number is affected. Every script uses `CPU_AND_NE`, and `where.py`
loads `CPU_ONLY` only to read the compute plan, never to run inference. But the
parameter is public API, so `Encoder.embed()` now normalises in Python rather
than trusting the graph. That is one line and it makes the class's contract true
by construction instead of by backend.

## Conversion

Three things cost time, all in `build.py` / `minilm.py`:

* **HuggingFace's `BertModel` will not convert.** Its attention-mask handling
  emits an `aten::Int` cast coremltools cannot fold (`TypeError: only
  0-dimensional arrays can be converted to Python scalars`). `minilm.py`
  reimplements the forward pass; `build.py` verifies it against HF to cosine
  0.99999994 and aborts if it drifts.
* **Any `x.shape[1]` fails the same way** under tracing. Every shape is a
  compile-time constant. This is the static-shape constraint showing up as a
  build error rather than a runtime one.
* **torch 2.14 is untested with coremltools 9.0** — pinned to 2.7.0.

## Where the ops land

`where.py` asks Core ML's own compute planner rather than inferring from timing:
**157 of 166 ops (94.6%) on the `MLNeuralEngineComputeDevice`**. The nine on CPU
are fp32↔fp16 casts plus the embedding `gather` — table lookup is not an ANE
operation. Every matmul, softmax and layernorm is.

The forward pass in `minilm.py` uses `nn.Linear` over `(B, T, H)` with
`permute`/`reshape` for the attention heads, not Apple's `(B, C, 1, L)`
four-dimensional convolution layout. It reaches 94.6% ANE residency anyway, so
the layout is not a correctness issue — whether it is a *performance* issue is
untested. See `NEXT_STEPS.md`.
