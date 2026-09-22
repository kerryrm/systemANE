"""Stream Jetstream, keep every post with an embed card, write JSONL."""
import asyncio, json, sys, time, aiohttp
from urllib.parse import urlparse

URL = "wss://jet.firehose.stream/tap?wantedCollections=app.bsky.feed.post"

MULTI = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.br", "com.au", "net.au",
         "co.jp", "ne.jp", "or.jp", "go.jp", "co.nz", "com.mx", "co.za",
         "com.tr", "co.kr", "com.ar", "com.es", "co.in", "com.sg"}

def domain(u):
    h = (urlparse(u).netloc or "").lower().removeprefix("www.")
    p = h.split(".")
    if len(p) >= 3 and ".".join(p[-2:]) in MULTI:
        return ".".join(p[-3:])
    return ".".join(p[-2:]) if len(p) >= 2 else h

async def main(OUT, SECS):
    n = 0
    t0 = time.time()
    with open(OUT, "w") as fh:
        async with aiohttp.ClientSession() as s:
            while time.time() - t0 < SECS:
                try:
                    ws = await s.ws_connect(URL, heartbeat=30)
                except Exception as e:
                    print("reconnect:", type(e).__name__, flush=True)
                    await asyncio.sleep(3); continue
                try:
                    async for msg in ws:
                        if msg.type is not aiohttp.WSMsgType.TEXT: continue
                        e = json.loads(msg.data); c = e.get("commit") or {}
                        if (c.get("collection") != "app.bsky.feed.post"
                                or c.get("operation") != "create"):
                            continue
                        r = c.get("record") or {}
                        ext = ((r.get("embed") or {}).get("external") or {})
                        uri, title = ext.get("uri"), (ext.get("title") or "").strip()
                        if not uri or not title:
                            continue
                        fh.write(json.dumps({
                            "domain": domain(uri),
                            "title": title,
                            "desc": (ext.get("description") or "").strip(),
                            "langs": r.get("langs") or [],
                        }, ensure_ascii=False) + "\n")
                        n += 1
                        if n % 500 == 0:
                            fh.flush()
                            print(f"{time.time()-t0:6.0f}s  {n} cards", flush=True)
                        if time.time() - t0 > SECS: break
                except Exception as e:
                    print("stream error:", type(e).__name__, flush=True)
                finally:
                    await ws.close()
    print(f"done: {n} cards in {time.time()-t0:.0f}s -> {OUT}", flush=True)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1],
                     int(sys.argv[2]) if len(sys.argv) > 2 else 1500))
