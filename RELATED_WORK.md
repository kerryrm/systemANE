# Related work

Five open decision engines, read on 2026-09-22, against this one.

Every number in this document is **their published claim**, taken from a model
card or README and not reproduced here. Numbers for `system1` are measured —
`FINDINGS.md` has the working-out. The two are not on the same footing and are
marked accordingly throughout.

| | [decider-2b](https://huggingface.co/Mapika/decider-2b) | [Laya](https://huggingface.co/convaiinnovations/laya) | [laya-coreml](https://github.com/mizorewww/laya-coreml) | [kev](https://github.com/jaredpalmer/kev) | [jevfire](https://github.com/kikoncuo/jevfire) | **system1** |
|---|---|---|---|---|---|---|
| base | Qwen3.5-2B | ModernBERT-large | (Laya, ported) | Qwen3.5 0.8/4/9B | Qwen3.8-27B | MiniLM-L6 |
| params | 1.9B | 421M | 421M | 0.8–9B | 27B | **22.6M** |
| trained? | SFT + PPO | RL (RLCD) | no (port) | LoRA r16 | **no** | **no** |
| scoring | LM-head option letters | `[MASK]`-per-option | same | pointer head | single-token labels | **cosine vs. anchors** |
| latency | 4.0 ms (GH200) | 32.8 ms (T4) | 4.98 ms (M3 Max ANE) | 329 ms–2 s (M5) | 110 ms (4 fields) | **1.4 ms (ANE)** |
| OOS / refusal | — | escalate head | — | — | none | **AUROC 0.994** |

---

## The shape is now standard

All five expose the same three primitives `system1.py` exposes — `choice` /
`score` / `noul`(boolean) — typed against an explicit option list, returning a
distribution rather than text. Five independent teams converged on that set.
Convergent evolution is weak evidence, but it is evidence: the primitives are
not arbitrary.

Where all five differ from us is the **scoring mechanism**. Four of the five use
the language-model head:

* **decider-2b** takes the hidden state at each answer slot and projects it
  through the option-letter rows of the LM head, then softmaxes over the valid
  letters.
* **Laya** scores every option at its own `[MASK]` token.
* **jevfire** maps each option to a verified single-token label and asks vLLM to
  score those labels at one output position — *"score labels; assemble the object
  in Python."*
* **kev** adds a rank-16 LoRA and a small pointer head over a shared state
  sequence, with an attention mask preventing questions from reading each other.

`system1` uses geometry instead: cosine of a pooled sentence embedding against a
pooled anchor embedding. That is the axis this document is about.

---

## What is worth taking

### 1. Confident-error rate, as a first-class metric

kev reports **wrong answers at p ≥ 0.9**: 8.7% before temperature fitting, 4.0%
after (they put Jev at 3.7%). That is exactly the failure `FINDINGS.md` names and
cannot currently see — a confident wrong answer is invisible to a confidence
threshold, and our fitted temperatures (0.026 on CLINC, 0.036 on tickets)
saturate the softmax enough to make it likely.

We report ECE 0.041, which averages that tail away. The ≥0.9-and-wrong rate is a
few lines in `evaluate.py` and it targets our one documented blind spot directly.
Highest value per hour of anything here.

### 2. An externally comparable benchmark

Laya publishes MASSIVE-intent **0.783** at 421M parameters. We publish CLINC150
**0.933** at 22.6M. Different datasets, different splits, not comparable — but
MASSIVE is the same task shape and `evalset.py` already does this kind of
loading. Right now we have no external reference point at all, which means the
headline number cannot be placed against anything.

### 3. Chance-corrected confidence, and entropy

kev and Laya both use `(p_max − 1/K) / (1 − 1/K)`. decider-2b reports
*"certainty = 1 − normalized entropy"* as a field **alongside** confidence.

Our `margin` is `p1 − p2`, which has two problems `FINDINGS.md` already
documents: it saturates at low temperature, and it is not comparable across a
5-label schema and a 160-label one. Chance-correction fixes the second directly;
normalized entropy uses the whole distribution rather than two points and
saturates less. Both are pure post-processing on a distribution we already
compute.

### 4. A learned escalate head

Laya's decision head includes an **act/escalate** output. We hard-code
`always_escalate={"feature"}` because a confidence threshold structurally cannot
express *"this needs a bigger model, not a more certain one."* Laya learns it.

We should not train one. But it confirms that the hack is pointing at something
real rather than being a demo convenience, and it is the clearest external
support for the `structural` escalation reason existing at all.

### 5. Energy is the ANE's actual argument

`laya-coreml` benchmarks the same model three ways on an M3 Max:

| implementation | P50 / P95 | energy per decision |
|---|---|---|
| compiled MLX FP16 (GPU) | 6.94 / 7.39 ms | 0.4288 J |
| Core ML ANE FP16 | 4.98 / 5.31 ms | 0.1540 J |
| Core ML ANE W8 | 4.88 / 5.23 ms | 0.1344 J |

The ANE is only **1.39× faster** than a compiled GPU path — but **2.78× more
energy-efficient**. And their W8-palettized variant is the tell: essentially the
same latency, lower energy again. On the ANE you buy power, not speed.

`README.md` rests its whole case on latency, which this suggests is the weaker
half. We have never measured joules, and we have never tried quantization.

---

## What it changes about our own conclusions

### The ModernBERT-large argument is half dead

`FINDINGS.md` previously declined ModernBERT-large partly on implementation cost
— *"needs a reimplemented forward pass (RoPE, GeGLU, no biases anywhere)."*
`laya-coreml` has now done exactly that, on the ANE, open source. **The cost
objection is gone.** The value objection — 12× the parameters of BGE-small for a
plausible +0.5 points, extrapolating from our own encoder table — still stands,
and is now the only reason. That is a better place for the argument to rest.

Two of their details bear on us if we ever revisit it:

* Their ANE variant drops to a **96-token** limit, against 512–1024 for their
  CPU+GPU build. We run `SEQ = 128`. That ceiling is real and they hit it.
* They export with *"BC1L activations, 1×1 projections and per-head attention"* —
  Apple's ANE-transformers layout. `minilm.py` uses `nn.Linear` over `(B, T, H)`
  with `permute`/`reshape`. We reach 94.6% ANE residency anyway, so it is not a
  correctness issue, but they evidently thought the layout was necessary and we
  have never tested whether it is faster.

### Their validation discipline is better than ours

`laya-coreml` checks **189/189 validation questions** match upstream on FP16, and
reports **max probability drift 0.002925** for the ANE checkpoint across 59
fitting questions.

We verify *embedding* cosine (0.99999994, in `build.py`) and separately spot-check
ten fixtures for `max |Δp| ≤ 0.0033`. Same idea, better executed: the quantity
you actually care about is decision drift, and it should be measured over the
whole evaluation set rather than ten hand-written strings. This is the same
mistake as the temperature sweep in *Calibration, and how a toy test set got it
backwards* — fixtures standing in for data.

### Ordinal scoring is hard for everyone

Laya's weakest primitive is `score`: **SST-5 0.372**, and the card lists ordinal
questions as a known weakness. Ours is weak in the same place — `FINDINGS.md`
measures no dynamic range at all on a 0–3 anger rubric.

Our write-up currently reads as though this is a limitation of *embedding
similarity* specifically: "captures topic, not intensity." A 421M trained model
with a completely different scoring mechanism fails the same way. The diagnosis
is too specific to our architecture; ordinal scoring is just hard.

### Fine-tuning has a tax we do not pay

kev measured it: a fine-tune from the base scored **0.33** on their own
evaluation set against **0.84** for the released adapter. The same data trained
with `--init_from` kept 0.83 there and reached 0.88 on the new domain.

Catastrophic forgetting is a live cost for every trained engine in this table,
and it makes domain adaptation a procedure with a footgun in it. Our adaptation
is 16 example sentences per class and one encode pass. It cannot degrade any
other schema, because there is nothing shared to degrade.

---

## Where this engine is actually ahead

### Nobody refuses

Not one of the five reports an out-of-scope metric. jevfire is explicit that its
scores are *"relative label probabilities… not probabilities that an answer is
correct"* and ships no calibration at all. Laya's card says it *"ships
over-confident"* and needs temperature scaling.

And it is not a feature they can bolt on cheaply. Softmaxing over option letters
or `[MASK]` positions destroys magnitude exactly the way our softmax does, so
there is no pre-softmax scalar left to threshold — the option scores themselves
carry no *none of these* signal at all. **The LM head is better at *which one*
and, by construction, silent about *any of them*.**

Laya is the exception that shows the price: it answers the second question with
a separately trained act/escalate head. That is the right fix for that
architecture, and it costs a head and the data to train it. Here the same signal
is the raw cosine — already computed, nothing trained, and calibrated to
AUROC 0.994 on CLINC by fitting a single threshold.

That is the sharpest result in this comparison, and it is the one thing
`README.md` should say that it currently does not.

### State caching is our default, not a feature

kev advertises caching the state representation across questions: a repeated
772-token state answered in **242 ms instead of 861 ms**. Laya batches 10
questions to reach ~7.2 ms each against 32.8 ms for one. decider-2b reports
11,180 decisions/second for schema-cached fixed question sets.

For us, N typed questions about one input is **one encode plus N dot products**.
The caching they engineer is free in an embedding architecture, and we have never
said so anywhere.

Their multi-question API is also nicer than ours:

```
"department":  {"choice": "billing", "confidence": 0.94}
"urgency":     {"score": 1.84 / 2.0}
"churn_risk":  {"noul": 0.892}
```

One state, a dict of typed questions, one call. `system1.py` answers one question
per call. Closing that gap costs us almost nothing.

### Scale

jevfire is the closest philosophical cousin — zero training, frozen weights,
rescore what the model already computes. It needs Qwen3.8-27B on an RTX PRO 6000
Blackwell to hit 109.9 ms for four fields.

Two zero-training approaches to the same problem. One of them needs a datacenter
GPU; the other is 1.4 ms on a laptop.

---

## The caveat on all of it

These are published claims, on different datasets, different splits and different
hardware, several of them self-reported on model cards with no third party having
run them. `JevBench` and `typed-decisions` are not datasets we have inspected.
Laya's card reports its own advantage over Jev (0.766 vs 0.727) using its own
harness.

Nothing here has been reproduced. The architectural observations — LM head versus
cosine, what a softmax destroys, where caching is free — do not depend on the
numbers being right. The performance comparisons do.
