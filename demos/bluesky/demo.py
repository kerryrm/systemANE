#!/usr/bin/env python3
"""Every link shared on Bluesky, sorted live on the Neural Engine.

    ./demo.py                      # ctrl-c to stop
    ./demo.py --min-sim 0.25       # stricter: refuse more, show less
    ./demo.py --show-refused       # watch what it throws away

Anchors are built at startup from `cards.jsonl.gz` -- real link cards labelled
by their domain, never by hand. The domains in that file cover 18% of the
stream; everything you see classified from any other domain is the encoder
generalising past the rule that trained it.

The header is the point: posts arrive at ~47/s and one thread encodes at
~800/s, so the engine spends most of its life idle while keeping up with the
entire global firehose.
"""
import argparse, asyncio, collections, gzip, json, os, shutil, sys, time
from urllib.parse import urlparse
import numpy as np
import aiohttp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules import label                                    # noqa: E402
from system1 import Encoder                                # noqa: E402
from harvest import domain                                 # noqa: E402

URL = "wss://jet.firehose.stream/tap?wantedCollections=app.bsky.feed.post"
CATS = ["video", "music", "news"]
COLOR = {"video": "\033[95m", "music": "\033[96m", "news": "\033[93m"}
DIM, BOLD, OFF = "\033[2m", "\033[1m", "\033[0m"
GREY, GREEN = "\033[90m", "\033[92m"


def build_anchors(enc, path, k):
    op = gzip.open if path.endswith(".gz") else open
    pool = collections.defaultdict(list)
    for line in op(path, "rt"):
        r = json.loads(line)
        c = label(r["domain"])
        if c in CATS:
            t = (r["title"] + ". " + r["desc"]).strip(" .") if r["desc"] else r["title"]
            if len(t) > 12:
                pool[c].append(t)
    M, n = [], {}
    for c in CATS:
        v = enc.embed(pool[c][:k]).mean(0)
        M.append(v / np.linalg.norm(v))
        n[c] = min(len(pool[c]), k)
    return np.stack(M), n


class Screen:
    def __init__(self, keep=16):
        self.rows = collections.deque(maxlen=keep)
        self.last = 0.0

    def add(self, row):
        self.rows.append(row)

    def draw(self, stats, force=False):
        now = time.time()
        if not force and now - self.last < 0.1:
            return
        self.last = now
        w = max(shutil.get_terminal_size((100, 30)).columns, 60)
        out = ["\033[H\033[J"]
        el = stats["elapsed"]
        out.append(f"{BOLD}  systemANE · the Bluesky firehose, sorted on the "
                   f"Neural Engine{OFF}\n")
        out.append(f"{GREY}  {'─' * (w - 4)}{OFF}\n")
        out.append(
            f"  {int(el // 60):02d}:{int(el % 60):02d}   "
            f"posts {stats['posts']:<6d} {stats['posts'] / max(el, 1):>5.1f}/s   "
            f"cards {stats['cards']:<5d} {stats['cards'] / max(el, 1):>4.1f}/s   "
            f"{GREEN}shown {stats['shown']:<5d}{OFF} "
            f"{DIM}refused {stats['refused']:<5d} "
            f"{stats['refused'] / max(stats['cards'], 1):>4.0%}{OFF}\n")
        per, warm = stats["encode_ms"], stats["warm_ms"]
        # Cards arrive ~150ms apart, so every encode is a cold one. `warm` is
        # the same model in a tight loop, measured at startup -- see
        # ../../warmup.py for why they differ by ~3x on every backend.
        out.append(
            f"  encode {per:.2f} ms {DIM}(warm {warm:.2f}){OFF}   "
            f"capacity {1000 / max(warm, 1e-9):>5.0f} posts/s warm   "
            f"{BOLD}using {stats['cards'] / max(el, 1) * per / 10:>4.1f}%{OFF} of one thread"
            f"   {DIM}{'  '.join(f'{c}:{stats[c]}' for c in CATS)}{OFF}\n")
        out.append(f"{GREY}  {'─' * (w - 4)}{OFF}\n")
        for r in self.rows:
            out.append(r[:w + 24] + "\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "cards.jsonl.gz"))
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--min-sim", type=float, default=0.25,
                    help="0.25 keeps ~80%% of in-schema and refuses ~63%% of the\n                          tail; 0.195 is the 90%%-retention point and lets\n                          borderline junk through. See README.")
    ap.add_argument("--show-refused", action="store_true")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after N seconds (for capture); default runs until ctrl-c)")
    args = ap.parse_args()

    print("loading encoder ...")
    enc = Encoder(os.path.join(ROOT, "encoder.mlpackage"), os.path.join(ROOT, "tok"))
    M, n = build_anchors(enc, args.cards, args.k)
    print(f"anchors: {', '.join(f'{c} k={n[c]}' for c in CATS)}   "
          f"min_sim {args.min_sim}")
    time.sleep(0.8)

    scr = Screen()
    t = time.perf_counter()
    for _ in range(60):
        enc.embed(["a warm-up sentence for the saturated figure"])
    warm_ms = (time.perf_counter() - t) / 60 * 1000
    st = {"posts": 0, "cards": 0, "shown": 0, "refused": 0, "elapsed": 0.0,
          "encode_ms": warm_ms, "warm_ms": warm_ms, **{c: 0 for c in CATS}}
    ms = collections.deque(maxlen=200)
    t0 = time.time()

    async with aiohttp.ClientSession() as sess:
        while True:
            try:
                ws = await sess.ws_connect(URL, heartbeat=30)
            except Exception:
                await asyncio.sleep(2); continue
            try:
                async for msg in ws:
                    if msg.type is not aiohttp.WSMsgType.TEXT:
                        continue
                    e = json.loads(msg.data)
                    c = e.get("commit") or {}
                    if (c.get("collection") != "app.bsky.feed.post"
                            or c.get("operation") != "create"):
                        continue
                    st["posts"] += 1
                    st["elapsed"] = time.time() - t0
                    if args.seconds and st["elapsed"] > args.seconds:
                        scr.draw(st, force=True)
                        await ws.close()
                        return
                    r = c.get("record") or {}
                    ext = ((r.get("embed") or {}).get("external") or {})
                    uri, title = ext.get("uri"), (ext.get("title") or "").strip()
                    if not uri or len(title) < 13:
                        scr.draw(st)
                        continue
                    st["cards"] += 1
                    desc = (ext.get("description") or "").strip()
                    text = (title + ". " + desc).strip(" .") if desc else title
                    t1 = time.perf_counter()
                    v = enc.embed([text])[0]
                    ms.append((time.perf_counter() - t1) * 1000)
                    st["encode_ms"] = sum(ms) / len(ms)
                    sims = v @ M.T
                    i = int(sims.argmax())
                    d = domain(uri)
                    known = label(d) in CATS
                    if sims[i] < args.min_sim:
                        st["refused"] += 1
                        if args.show_refused:
                            scr.add(f"  {DIM}{'refused':>7s} {sims[i]:.2f} "
                                    f"[{d[:20]:20s}] {title[:58]}{OFF}")
                    else:
                        st["shown"] += 1
                        st[CATS[i]] += 1
                        tag = f"{GREY}rule{OFF}" if known else f"{GREEN} new{OFF}"
                        scr.add(f"  {COLOR[CATS[i]]}{CATS[i]:>7s}{OFF} "
                                f"{sims[i]:.2f} {tag} "
                                f"{DIM}[{d[:20]:20s}]{OFF} {title[:58]}")
                    scr.draw(st)
            except Exception:
                pass
            finally:
                await ws.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nbye")
