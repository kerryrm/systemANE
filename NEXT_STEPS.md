# Next steps

Ranked by value per hour. Each item says what would settle it and what a
negative result looks like, because several of these are worth doing precisely
so they can come back negative.

Nothing here is committed work. `FINDINGS.md` is the measured record;
`RELATED_WORK.md` is where most of these came from.

---

## ~~1. Confident-error rate in `evaluate.py`~~ — done

`evaluate.py` now reports it. **2.3% of in-scope test items (7/300) are wrong at
p ≥ 0.90, about a third of all errors, and `min_sim`/`min_margin` flag zero of
them.** The result was sharper than expected in a way that closes the question
rather than opening it: all seven are adjacent-intent confusions, so the gates
are not mis-tuned — there is no signal for them to fire on. See *Confident
errors, and why no threshold catches them* in `FINDINGS.md`.

Two things fell out of it worth carrying forward. Calibration made this tail
*worse* (0 → 7 confident errors) while improving ECE from 0.196 to 0.041, and
that was still the right trade — so neither metric should be read alone. And it
puts a floor on the cascade: escalation fixes uncertainty, and these decisions
are not uncertain.

## 2. Chance-corrected confidence and normalized entropy

`margin` is `p1 − p2`. It has two documented problems: it saturates at our fitted
temperatures, and it is not comparable across a 5-label schema and a 160-label
one — which is half of why *"calibration does not transfer between schemas."*

Add two fields to `Decision`:

* `confidence = (p_max − 1/K) / (1 − 1/K)` — chance-corrected, comparable across
  K. Used by both kev and Laya.
* `certainty = 1 − H(p)/log K` — normalized entropy, uses the whole distribution
  rather than two points. decider-2b reports this *alongside* confidence rather
  than instead of it.

Then re-run the escalation curve on each and see which thresholds more cleanly.

* **Effort:** an afternoon, plus a re-fit.
* **Negative result:** all three signals rank identically and margin was fine.
  Likely, honestly — they are monotone in each other for K=2 and only diverge on
  flat tails. The comparability across K is the part that should survive.
* **Do not** remove `margin`. The fitted `min_margin` values in the tracked
  `calibration*.json` files depend on it.

## ~~3. MASSIVE, for an externally comparable number~~ — done

`massive.py` loads `mteb/amazon_massive_intent`; `--dataset massive` runs the
existing scripts against it. **0.710 accuracy on 2,974 test items, ECE 0.032**,
same encoder, k=16 centroids, nothing trained — against Laya's published 0.783
at 421M parameters with an RL stage. Reported as two measurements of one
dataset, not a head-to-head.

It also revised two earlier findings, both against the intuition:

* **Masking robustness is about neighbour density, not label count.** CLINC's 10
  adjacent banking routes flag 30.7% of forced-wrong decisions; MASSIVE's 60
  intents flag **89.3%**. The bigger schema is far safer under a wrong mask,
  because its classes are spread over 18 scenarios rather than packed into one
  domain.
* **The confident-error share barely moved** (2.3% → 2.2%) despite a 22-point
  accuracy drop, because the engine gets confident about far fewer inputs on the
  harder task (37% vs 88% reach p ≥ 0.90) and is wrong more often when it does.
  That is calibration working across datasets.

The prediction going in — that 60 classes would crowd more than 10 and produce
more undetectable errors — was wrong in both directions.

## 4. Decision drift over the whole eval set, not ten fixtures

`build.py` verifies *embedding* cosine to 0.99999994 — good. The fp16 finding,
though, spot-checks ten hand-written fixtures for `max |Δp| ≤ 0.0033`.

`laya-coreml` checks 189/189 decisions against upstream and reports max
probability drift of 0.002925. That is the right shape: the quantity you care
about is drift in the **decision**, measured over real data.

Run the CLINC test set through both the fp32 torch path and the fp16 ANE path,
and report top-1 agreement plus max and mean `|Δp|` over all 1,300 items.

* **Effort:** half a day. `evaluate.py` already runs both paths.
* **Why it matters beyond tidiness:** `FINDINGS.md` says *"expect precision to
  matter as the label count grows"* and then tests at K=5. CLINC is K=160. This
  is the test that sentence was asking for.
* **Negative result:** perfect agreement, and the fp16 caveat can be stated as
  settled rather than bounded.

## 5. A multi-question API

kev caches state representations across questions (772-token state: 861 ms →
242 ms). Laya batches 10 questions to reach ~7.2 ms each. decider-2b caches
schemas for 11,180 decisions/second.

All of that is free here — N typed questions about one input is one encode plus
N dot products — and we have never said so or exposed it:

```python
s = State(enc, "I was charged twice for last month")
s.ask(department=route, urgency=anger, churn_risk=at_risk)
# {"department": Decision(...), "urgency": ..., "churn_risk": ...}
```

* **Effort:** a day, mostly API design. No model work.
* **What it buys:** a structural advantage of the embedding approach, currently
  invisible. It is also the shape every other engine in `RELATED_WORK.md`
  presents, so it makes the comparison legible.
* **Risk:** none technically. The risk is scope — resist adding cross-field
  conditioning, which the architecture genuinely cannot do.

---

## Experiments, in decreasing confidence that they are worth it

## 6. Does an MLM head fix the description-anchor failure?

The sharpest known failure in `FINDINGS.md` is a vocabulary-overlap failure. A
real bug report — *"dates display as 1970 on the dashboard"* — scores 0.051
against the `bug` anchor, **below** out-of-scope text like *"unsubscribe"* at
0.227, because the anchor describes the category (*"the software crashes,
errors, freezes"*) and the report describes a symptom.

Four of the five engines in `RELATED_WORK.md` score options through the
language-model head instead of comparing pooled embeddings. A conditional
prediction at a `[MASK]` position should not have this failure mode — it is not
matching vocabulary, it is asking what word belongs there.

MiniLM-L6 has no usable MLM head, but `build_encoder.py` compiles arbitrary
BERT-architecture encoders and a base BERT does ship one.

* **Scope this as a measurement, not a rebuild.** Take the ticket schema's known
  failure cases, score them both ways, see whether the ordering inverts. Do not
  build a second engine to find out.
* **The reason for caution:** Laya's card reports base checkpoints at **0.362 on
  typed-decisions zero-shot — near chance.** Their entire 421M-plus-RL stack
  exists because the untrained version does not work. That may be about their
  decision-head architecture rather than plain MLM prompting, but it is the
  strongest available prior and it points the wrong way.
* **Negative result is the likely one**, and it is cheap and worth having: it
  would say the description-anchor failure is not fixable without training, which
  makes 16-example centroids the answer rather than a workaround.

## 7. Energy per decision

`laya-coreml` measures 0.1540 J on the ANE against 0.4288 J on a compiled MLX GPU
path for the same model — **2.78× energy for only 1.39× latency.**

`README.md` rests its case entirely on latency, which that comparison suggests is
the weaker half. If the ANE's real argument is power, we are not making it.

* **Effort:** unknown, and that is the problem. `powermetrics` needs sudo and
  attributing joules to a 1.4 ms operation is hard — their numbers are
  system-level, which is a much easier measurement and a much weaker one.
* **Do it only if** the measurement can be made honestly. A system-level delta
  over a long loop, clearly labelled as such, is acceptable. A per-decision joule
  figure we cannot defend is worse than no figure.

## 8. ANE layout: `nn.Linear` vs. Apple's 4D convolution form

`minilm.py` uses `nn.Linear` over `(B, T, H)` with `permute`/`reshape` for
attention heads. Apple's ANE-transformers recipe — which `laya-coreml` follows —
uses `(B, C, 1, L)` activations with 1×1 convolutions and per-head attention.

We already get 157/166 ops (94.6%) onto the ANE, so this is not a correctness
question. It is entirely a question of whether the current layout leaves
throughput on the table, and at 1.4 ms there may be very little table left.

* **Effort:** days. A second forward pass to write and verify against the first.
* **Prerequisite:** do not start this without a reason to think 1.4 ms is the
  bottleneck in something. Right now, nothing is bottlenecked on it.

---

## Deliberately not doing

**ModernBERT-large.** `FINDINGS.md` previously declined it on two grounds: 12×
the parameters of BGE-small for a plausible +0.5 points, and the cost of
reimplementing the forward pass (RoPE, GeGLU, no biases anywhere). `laya-coreml`
has now done that reimplementation, on the ANE, open source — **so the cost
objection is gone and the value objection is not.** Our own encoder table is the
argument: 22.6M → 33.2M buys +2.0 points; 33.2M → 108.9M buys +0.4. There is no
reading of that curve where 395M is worth it.

One thing to note if anyone revisits: `laya-coreml`'s ANE build drops to a
**96-token** limit against 512–1024 for their CPU+GPU build. We run `SEQ = 128`.
That ceiling is real and they hit it.

**mmBERT-base.** Worse value still. 197M of its parameters are a 256,000-entry
embedding table — over half the model — reached by a `gather` that stays on CPU.

**Fine-tuning anything.** kev measured the tax: a fine-tune from base scored 0.33
on their own evaluation set against 0.84 for the released adapter; only
`--init_from` kept both. Our adaptation is 16 sentences per class and one encode
pass, and it cannot degrade any other schema because nothing is shared. That is a
real property and it should be given up deliberately, not by drift.

**Cross-field conditioning.** jevfire is explicit that its fields score
independently with no cross-field conditioning, and it has a 27B model to work
with. We have 22.6M and a dot product. Do not imply the architecture can do
something it cannot.
