#!/usr/bin/env python3
"""Two-tier cascade: ANE reflex first, local LLM only when it is not sure.

Tier 1  MiniLM-L6 on the ANE          ~1.2 ms   answers most inputs
Tier 2  Apple Foundation Model        ~seconds  answers what tier 1 escalates
                                      (`fm serve` on port 1976, built into macOS)

The point is not that tier 1 is smart. It is that tier 1 is cheap enough to
run on everything, and honest enough to say when it should not decide.
"""
import argparse
import time

from fmserve import FMError, FMServer
from system1 import Choice, Encoder
from tickets import AMBIGUOUS, CLEAR, OUT_OF_SCOPE, ROUTES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--min-margin", type=float, default=None)
    ap.add_argument("--min-sim", type=float, default=None)
    ap.add_argument("--min-prob", type=float, default=0.55)
    ap.add_argument("--no-tier2", action="store_true", help="skip the LLM")
    ap.add_argument("--calibration", default="calibration_tickets.json",
                    help="thresholds fitted by calibrate_tickets.py")
    args = ap.parse_args()

    enc = Encoder()
    # This schema has its own fitted thresholds. calibration.json belongs to
    # the CLINC banking routes and does not transfer -- its min_sim refuses
    # genuinely in-scope tickets here -- so calibrate_tickets.py fits temp,
    # min_sim and min_margin on tickets.VALIDATION, which is disjoint from the
    # fixtures printed below.
    # A feature request needs a written reply, not a more certain label --
    # so it goes to tier 2 by construction rather than by confidence. This is
    # jev-ultrafast's rule that only TYPE_TEXT needs an LLM.
    route = Choice(enc, ROUTES, temp=args.temp, min_margin=args.min_margin,
                   min_prob=args.min_prob, min_sim=args.min_sim,
                   calibration=args.calibration, always_escalate={"feature"})
    fm = FMServer()
    tier2_up = (not args.no_tier2) and fm.up()
    if not args.no_tier2 and not tier2_up:
        print("note: fm serve unreachable on 127.0.0.1:1976 — tier 1 only\n")

    t1_ms, t2_ms, escalated, answered, refused = [], [], 0, 0, 0
    print(f"{'ticket':46s} {'tier1':9s} {'p':>5s} {'marg':>5s} {'sim':>5s} "
          f"{'why':13s} {'tier2':9s} {'ms':>6s}")
    print("-" * 108)
    for text in CLEAR + AMBIGUOUS + OUT_OF_SCOPE:
        t0 = time.perf_counter()
        d = route(text)
        t1 = (time.perf_counter() - t0) * 1000
        t1_ms.append(t1)

        final, t2 = "", ""
        if d.reason == "out-of-scope":
            # Tier 2 is bound to the same enum, so it would also be forced to
            # pick a wrong label. Refusing is both cheaper and more correct.
            refused += 1
            final = "(none)"
        elif d.escalate:
            escalated += 1
            if tier2_up:
                s = time.perf_counter()
                try:
                    final = fm.choose(text, ROUTES)
                except FMError as exc:
                    final = f"!{exc}"[:9]
                ms = (time.perf_counter() - s) * 1000
                t2_ms.append(ms)
                t2 = f"{ms:6.0f}"
            else:
                final = "(skipped)"
        else:
            answered += 1
        print(f"{text[:44]:46s} {d.label:9s} {d.prob:5.2f} {d.margin:5.2f} "
              f"{d.sim:5.2f} {d.reason:13s} {final:9s} {t2:>6s}")

    n = len(CLEAR) + len(AMBIGUOUS) + len(OUT_OF_SCOPE)
    print(f"\ntier 1 answered {answered}/{n}, refused {refused}/{n} "
          f"(out of scope), escalated {escalated}/{n}")
    print(f"tier 1 latency: mean {sum(t1_ms)/len(t1_ms):.2f} ms")
    print(f"thresholds: {args.calibration} "
          f"(temp {route.temp:.4f}, min_sim {route.min_sim:.3f}, "
          f"min_margin {route.min_margin:.2f}) — fitted on tickets.VALIDATION, "
          f"disjoint from these fixtures")
    if t2_ms:
        print(f"tier 2 latency: mean {sum(t2_ms)/len(t2_ms):.0f} ms "
              f"({sum(t2_ms)/len(t2_ms)/(sum(t1_ms)/len(t1_ms)):.0f}x tier 1)")
        total_cascade = sum(t1_ms) + sum(t2_ms)
        all_llm = (sum(t2_ms) / len(t2_ms)) * n
        print(f"cascade total {total_cascade:.0f} ms vs {all_llm:.0f} ms if every "
              f"ticket went to the LLM ({all_llm/total_cascade:.1f}x saving)")


if __name__ == "__main__":
    main()
