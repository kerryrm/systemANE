#!/usr/bin/env python3
"""What an idle gap between calls costs, on each compute unit.

The 1.4 ms in README.md is measured in a tight loop. Real traffic is not a
tight loop: demos/bluesky sees a link card every ~150 ms and measured 4.3 ms
per encode, three times the published figure. This is where that goes.

The control matters more than the headline. Every backend pays a wake-up cost,
so this is not an ANE quirk -- and the ANE pays the least of the three, which
means its advantage is larger for sporadic calls than for a benchmark loop.

  ./warmup.py                      # all three compute units
  ./warmup.py --units CPU_AND_NE   # just the deployed path
"""
import argparse
import time

from system1 import Encoder

TEXT = ("Trump openly gloating and knows he can cross any red line, says "
        "analyst today")


def sweep(enc, gaps, reps, warm):
    out = {}
    for _ in range(warm):
        enc.embed([TEXT])
    for gap in gaps:
        xs = []
        n = reps if gap < 0.1 else max(reps // 3, 20)
        for _ in range(n):
            if gap:
                time.sleep(gap)
            t = time.perf_counter()
            enc.embed([TEXT])
            xs.append((time.perf_counter() - t) * 1000)
        xs.sort()
        out[gap] = xs[len(xs) // 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="*",
                    default=["CPU_AND_NE", "CPU_ONLY", "CPU_AND_GPU"])
    ap.add_argument("--gaps", type=float, nargs="*",
                    default=[0, 0.001, 0.005, 0.02, 0.05, 0.2])
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--warm", type=int, default=40)
    args = ap.parse_args()

    res = {}
    for u in args.units:
        res[u] = sweep(Encoder(compute_units=u), args.gaps, args.reps, args.warm)

    print("median ms per encode, by idle gap before the call "
          "(the sleep is not timed)\n")
    print(f"{'gap':>8}  " + "  ".join(f"{u:>12s}" for u in args.units))
    for g in args.gaps:
        print(f"{g*1000:7.0f}ms  "
              + "  ".join(f"{res[u][g]:12.2f}" for u in args.units))
    print()
    for u in args.units:
        warm, cold = res[u][args.gaps[0]], res[u][args.gaps[-1]]
        print(f"  {u:12s} warm {warm:5.2f}  cold {cold:5.2f}  "
              f"{cold/warm:.1f}x  (+{cold-warm:.2f} ms)")
    if "CPU_AND_NE" in res and len(args.units) > 1:
        ne = res["CPU_AND_NE"]
        print("\n  ANE advantage over the others:")
        for u in args.units:
            if u == "CPU_AND_NE":
                continue
            print(f"    vs {u:12s} warm {res[u][args.gaps[0]]/ne[args.gaps[0]]:.1f}x"
                  f"   cold {res[u][args.gaps[-1]]/ne[args.gaps[-1]]:.1f}x")


if __name__ == "__main__":
    main()
