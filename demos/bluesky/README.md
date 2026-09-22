# bluesky

Classify every link shared on Bluesky, live, on the ANE — and find out whether
the encoder earns its place or whether a two-line domain rule already did the
job. In `~/Downloads` the rule won and the model was decoration
(`../downloads`). Here it loses 82–18, which is what makes this worth building.

```sh
../../.venv/bin/python harvest.py cards.jsonl 1500     # 25 min of firehose
../../.venv/bin/python evaluate.py cards.jsonl.gz --cats video music news --balanced
../../.venv/bin/python demo.py                         # the live thing
../../.venv/bin/python demo.py --show-refused          # watch what it drops
```

`demo.py` defaults to `min_sim 0.25` — the 80%-retention point, which refuses
about 63% of the tail. The 90% point (0.195) lets borderline junk through:
`gofund.me` "Donate to Keep My Daughter's Dream" was admitted as *video* at
0.22. Showing fewer, better items is the right trade for a demo.

`cards.jsonl.gz` is a 25-minute capture from 2026-09-22, kept so the numbers
below reproduce without re-harvesting. Needs `aiohttp`, which the venv already
has via `datasets`.

**[▶ ANE-jetstreaming.mp4](ANE-jetstreaming.mp4)** — 24 seconds of it running.
`rule` marks a link whose domain is in the table below; `new` marks one the
encoder placed from the card text alone, which is most of them.

The errors are in there too, and they are the ones the numbers predict. A book
listing — *"Continental Drift by Mai-Linh Hong"* — goes to `music`, because
"title by person" is exactly how a song is credited. A French headline about
planetary boundaries goes to `video` at 0.27, barely over the threshold. At
0.82 accuracy roughly one row in five is wrong, and watching which ones is more
informative than the number.

## The stream

[Jetstream](https://jet.firehose.stream/) re-emits the ATProto relay as plain
JSON over a websocket. No auth, no key, `wss://jet.firehose.stream/tap`.
Measured over 100 seconds:

```
posts                    47/s
posts with a link      23.6%        11.6 links/s
  ... with an embed card  70%
  ... card has a title    69% of linked posts   median  58 chars
  ... has a description   64%                   median 118 chars
```

**The link card's title and description arrive inside the firehose event.** No
YouTube API, no oEmbed fetch, nothing on the critical path — roughly 176
characters of well-formed prose per link, which is far better input than the
post text it is attached to (`"定期的にみたくなる https://…"`).

Two properties make this a good fit rather than a convenient one. Posts are
short — median 68 characters — so the dilution failure in `FINDINGS.md`, where
one paragraph of irrelevant text flips 37% of decisions, never triggers. And at
47 posts/s against ~800/s of encoder throughput on one thread, a laptop
classifies the entire global firehose at about 6% utilisation.

## Free labels, and why they are worth having

The domain is a label nobody on this project wrote: `youtube.com` → video,
`bandcamp.com` → music, `arxiv.org` → research. `evalset.py` exists to avoid
grading ourselves on our own writing, and this has the same property for free.

The number that decides whether the experiment is interesting at all:

```
distinct domains in 25 min   2,997
seen exactly once            66%
placed by a domain rule      18.4%
left for a model             81.6%
```

**This is the inverse of `../downloads`**, where host rules placed 99% and both
the Foundation Model and example anchors measured worthless. A long tail is
what gives an encoder a job.

## The test: split by domain, never by post

Centroids are built from one set of sites and tested on *different* sites in
the same category. Splitting by post would let the encoder pass by memorising
one publication's title formatting.

```
video   train  youtube.com, twitch.tv, nicovideo.jp
        test   youtu.be, tiktok.com, rumble.com
music   train  spotify.com, soundcloud.com, discogs.com
        test   bandcamp.com, audiomack.com
news    train  Guardian, Globo, AP, Spiegel, Politico, Bloomberg … (21 outlets)
        test   NYT, BBC, Le Monde, Reuters, CNN, NPR, WaPo … (20 others)
```

| k | accuracy | macro-F1 |
|---|---|---|
| 8 | 0.710 | 0.604 |
| 32 | 0.757 | 0.643 |
| 64 | 0.772 | 0.659 |
| **128** | **0.821** | **0.704** |

823 held-out cards from domains never seen in training, against a
majority-class baseline of 0.599. Recall: music 1.00, news 0.79, video 0.75.
News trained on the Guardian and Spiegel transfers to the NYT and Reuters,
which is the generalisation that matters.

(`--balanced` reports 0.85 on an equal-sized sample, but that rests on 26 music
items and moves ±0.07 with the seed. The raw accuracy and macro-F1 are the
numbers to quote.)

**0.821 is a floor.** The largest confusion is video↔music, and a YouTube link
to a music video is labelled `video` by the domain rule while *being* music.
The model is right and the label is wrong. Noisy labels cap what is measurable.

## What the encoder does that the rule cannot

The most confident placements from the tail — domains no rule covers:

```
0.658 video  [goo.gl        ] New video · Tuesday, Sep 22 🎬. Tap to view!
0.583 music  [dlvr.it       ] Z108 | Top 40 / Dance. Today's hits. Nonstop energy.
0.581 music  [apple.com     ] Abundantemente Morte by Luiz Melodia on Apple Music
0.530 news   [google.com    ] First Thing: News outlets suspend coverage of Trump
0.505 news   [techxplore.com] Calls for global regulation of AI are growing
```

Two of those are link shorteners, where the domain is *by construction*
uninformative and only the card text exists. That is the job.

## Three things that went wrong

**"blog" was a category error.** A domain rule labels the **host**, and hosts
divide into two kinds. `nytimes.com` and `arxiv.org` are topic-bearing — the
host implies the content. `substack.com` and `medium.com` are infrastructure:
they host anything, and a Substack piece about politics *is* news. Labelling
them `blog` put 33 of 66 news cards into a category the text cannot predict,
because the distinction is not in the text. They are in `rules.EXCLUDE` now,
which dropped the label rate from 24.3% to 18.4% and fixed the worst confusion.

**Only three categories have the volume.** `fund`, `research`, `shop`, `code`,
`art`, `games` and `books` reached 1–4 domains and 10–12 test cards each in 25
minutes; `fund` scored 0.08 recall on 12 items. The honest schema is
video / music / news / refused, not eleven buckets. Longer harvests would fix
some of this and not others — `art` and `books` are genuinely rare here.

**TLD is the wrong axis.** Tried first, and it fails: `.com` is 598 of ~800,
the rest are country codes that encode language and geography rather than
topic, and `.be` at 44 is almost entirely `youtu.be` — Belgium's TLD swamped by
a URL shortener. Domain works; TLD does not.

## A latency surprise

The demo reported **4.3 ms** per encode against the 1.4 ms in the root README.
The encoder is not slower here — link cards arrive ~150 ms apart, so every call
is a cold one. That turned into `../../warmup.py` and a section in
`FINDINGS.md`: every Core ML backend pays a wake-up cost, and the ANE pays the
smallest, so its lead over CPU and GPU is *larger* for sporadic calls than in a
benchmark loop. The demo header now shows both figures.

## What this says about the engine

**k keeps paying here, well past where it stopped on CLINC.** `FINDINGS.md`
found k=8, 16 and 32 statistically indistinguishable; here accuracy climbs from
0.710 to 0.821 between k=8 and k=128. Noisier, more heterogeneous anchors keep
absorbing examples for longer, so the k=16 result does not transfer.

**Refusal is workable but unproven.** AUROC 0.821 separating in-schema from
tail; at 90% retention it refuses 42.9% of the tail. That is well short of
CLINC's 0.994 — but the tail is *not* ground-truth out-of-scope, since much of
it genuinely is video, music or news from an unruled domain. It is a noisy
target, not a weak signal, and this experiment cannot tell those apart without
hand-labelling.
