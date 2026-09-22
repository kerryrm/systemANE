# systemANE

On CLINC150 — 1,300 human-written utterances collected independently of this
project, 300 in-scope plus 1,000 genuinely out-of-scope — the engine gets
**93.3% accuracy, ECE 0.041, and 0.994 AUROC at spotting out-of-scope input**,
from a 22.6M-parameter model with no fine-tuning, at **1.4 ms on the Apple
Neural Engine**.

It is a System-1 decision engine: typed decisions over a sentence encoder,
paired with a second tier that only runs when the first is not sure.

```
tier 1   MiniLM-L6 (22.6M) on the ANE     1.4 ms    answers most inputs
tier 2   Apple Foundation Model (~3B)    1212 ms    answers what tier 1 escalates
```

Tier 1 is not smart. It is cheap enough to run on everything, and — once
calibrated — honest enough to say when it should not decide. That is the whole
idea; the accuracy lives in tier 2.

Proof of concept. Tier 2 is served by `fm serve`, the Apple Foundation Models
CLI built into macOS 27 at `/usr/bin/fm` — there is nothing external to install
for it. `fmserve.py` talks to it over stdlib HTTP only.

## Quick start

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python build.py       # convert MiniLM -> encoder.mlpackage (once)
.venv/bin/python where.py       # confirm it lands on the ANE

fm serve &                      # tier 2, in another shell (port 1976)
.venv/bin/python cascade.py     # the two-tier demo
```

`cascade.py --no-tier2` runs tier 1 alone, and needs no Foundation Model at
all. Python 3.12 — coremltools has no 3.14 wheels. macOS 27 on Apple silicon:
the ANE for tier 1, the built-in Foundation Model for tier 2.

## What it does

`cascade.py`, verbatim, with `fm serve` running:

```
ticket                                    tier1        p  marg   sim  why           tier2     ms
------------------------------------------------------------------------------------------------
I was charged twice for last month        billing   0.95  0.89  0.41
can't log in after the update, it says... auth      1.00  1.00  0.59
the app crashes every time I open the...  bug       1.00  1.00  0.54
would be great if you supported dark mode feature   1.00  0.99  0.32  structural    feature  2477
where is my package, it was supposed t... shipping  1.00  1.00  0.54
my subscription renewed but I cancelle... billing   0.99  0.98  0.31
2FA codes never arrive on my phone        auth      1.00  1.00  0.39
the checkout page throws a 500 when I ... billing   0.94  0.89  0.28
it broke                                  bug       0.75  0.52  0.27  ambiguous     bug       575
I need help with my account               auth      0.99  0.97  0.44
this isn't working                        auth      0.76  0.65  0.22  ambiguous     bug       585
something is wrong                        bug       0.96  0.94  0.31
can someone call me                       feature   0.65  0.47  0.14  out-of-scope  (none)
what are your office hours                bug       0.44  0.13  0.07  out-of-scope  (none)
I'd like to speak to a manager            feature   0.91  0.88  0.17  out-of-scope  (none)
happy holidays everyone                   shipping  0.46  0.21  0.08  out-of-scope  (none)

tier 1 answered 9/16, refused 4/16 (out of scope), escalated 3/16
tier 1 latency: mean 1.42 ms
tier 2 latency: mean 1212 ms (856x tier 1)
cascade total 3660 ms vs 19399 ms if every ticket went to the LLM (5.3x saving)
```

Three things in that table are the point:

* **The clear tickets are routed in ~1.4 ms each.** No LLM.
* **The four out-of-scope inputs are refused, not guessed at.** They are not
  escalated either: tier 2 is bound to the same five labels, so it would also
  be forced to pick a wrong one. Refusing is cheaper *and* more correct.
* **Escalation fixed a wrong answer.** *"this isn't working"* was heading for
  `auth` at p=0.76; the margin was under threshold, tier 2 was asked, and it
  came back `bug`.

Thresholds come from `calibration_tickets.json`, fitted by
`calibrate_tickets.py` on `tickets.VALIDATION` — 72 tickets disjoint from the
sixteen above, so this is a held-out test and not a restatement of the fit.
Those 72 are hand-written by the author, though, which is why the headline
number comes from CLINC150 instead.

## Typed decisions, not generated text

```python
enc   = Encoder()
route = Choice(enc, ROUTES)           # option descriptions embedded once
d     = route("I was charged twice")  # ~1.4 ms
# Decision(label='billing', prob=0.95, margin=0.89, escalate=False)
```

The output is a distribution over the schema you passed in, so an off-schema
label is not unlikely — it is unrepresentable.

## Two signals, because there are two ways to not know

| | signal | question it answers |
|---|---|---|
| ambiguous between labels | `margin` (post-softmax) | *which* of these? |
| outside the schema | `sim` (raw max cosine) | *any* of these? |

The softmax normalises magnitude away, so `margin` alone cannot see the second
case: *"I'd like to speak to a manager"* scores a higher margin than the
genuinely ambiguous *"it broke"*. On CLINC, `sim` separates out-of-scope at
AUROC 0.994 against 0.945 for margin. Keeping them apart is also what lets the
engine *refuse* rather than escalate, which is the right action when no label
is correct.

**This is where the embedding approach earns its keep.** Most decision models
score options through the language-model head — at option letters, at a
`[MASK]` per option, at a single-token label. That is better at picking *which*
one. But it softmaxes over the option set, and a softmax has no way to say
*none of these*: whatever the model thought of the input in absolute terms is
normalised away before anything downstream sees it. Answering the second
question then needs a separately trained head.

Here it falls out of the geometry. `sim` is the raw cosine, before any
normalisation — a scalar that is simply low when nothing in the schema
resembles the input, at no cost and with nothing trained. Of the five engines
in `RELATED_WORK.md`, none publishes an out-of-scope number.

## Reproducing the headline

```sh
.venv/bin/python calibrate.py --k 16 --no-description   # fit on validation
.venv/bin/python evaluate.py  --k 16                    # score on test
```

Against the zero-shot baseline, which embeds one handwritten sentence per class:

| | description only | 16 examples |
|---|---|---|
| in-scope accuracy | 0.880 | **0.933** |
| errors | 36/300 | **20/300** |
| OOS AUROC | 0.989 | **0.994** |
| escalation needed for ≤5% error | 17.5% | **1.5%** |

A 44% reduction in errors for 160 labels and no training — and it mostly
dissolves the escalation problem, so tier 2 is called for a handful of inputs
rather than a sixth of traffic.

`where.py` confirms the ANE claim by asking Core ML's own compute planner
rather than inferring from timing: 157 of 166 ops (94.6%) on the
`MLNeuralEngineComputeDevice`. The nine on CPU are fp32↔fp16 casts plus the
embedding `gather` — table lookup is not an ANE operation. Every matmul,
softmax and layernorm is.

## Limits

**Calibration does not transfer between schemas, and the failure is silent.**
`calibration.json` is fitted for the CLINC banking routes; applied to the
support tickets its `min_sim` of 0.3085 refuses genuinely in-scope tickets at
0.27–0.28. Every schema needs its own fit. `Choice()` takes `calibration=` for
exactly this reason.

**A confident wrong answer is invisible to a confidence threshold.** On the
CLINC test set, 7 of 300 in-scope items (2.3%) are answered wrongly at p ≥ 0.90
— about a third of all errors — and `min_sim` and `min_margin` flag **none** of
them. That is not a threshold needing tuning. All seven are adjacent-intent
confusions (`transactions` vs. `credit_limit`, `report_lost_card` vs.
`report_fraud`), so the embedding sits close to the wrong anchor because it
genuinely should, and every signal the engine computes says the answer is fine.

This is the floor on what a cascade can do: escalation fixes uncertainty, and
these decisions are not uncertain. `FINDINGS.md` has the seven and the
temperature sweep behind them.

## Layout

```
minilm.py             MiniLM-L6 forward pass, written out so the graph is ours
build.py              convert to encoder.mlpackage (verifies against HuggingFace)
build_encoder.py      compile any BERT-architecture encoder (--model, --pooling)
where.py              per-op device assignment from Core ML's compute planner
system1.py            Encoder + Choice / Boolean / Score primitives
fmserve.py            minimal stdlib client for `fm serve` (tier 2)
cascade.py            the two-tier demo

tickets.py            demo schema: 5 routes, fixtures, and a validation set
calibrate_tickets.py  fit the demo schema's thresholds

evalset.py            CLINC150 loading and splits
anchors.py            class representations: descriptions vs example centroids
calibrate.py          fit temp / min_sim / min_margin on validation
evaluate.py           test-set accuracy, ECE, OOS AUROC
sweep_anchors.py      the k sweep behind the anchors table

FINDINGS.md           what was measured, and which guesses were wrong
RELATED_WORK.md       five other decision engines, and what they change here
NEXT_STEPS.md         what is worth doing next, and what is deliberately not
```

Model packages and tokenizers are build artifacts and gitignored; run
`build.py` or `build_encoder.py` to regenerate. The fitted `calibration*.json`
files are results, not artifacts, and are tracked.
