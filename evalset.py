"""A real evaluation set, from CLINC150 (`clinc_oos`, "plus" config).

Chosen over hand-written fixtures deliberately: if the author of the engine
also writes the test set, the test set inherits the engine's blind spots. These
are human-written utterances collected independently, and CLINC ships an
explicit out-of-scope class -- the exact axis this project argues about.

A 10-way banking router, plus OOS. Descriptions below were written from the
intent *names* only, without looking at any example utterance.
"""
from functools import lru_cache

OOS = "oos"

ROUTES = {
    "balance":           "checking how much money is in my account",
    "transfer":          "moving money between my own accounts",
    "transactions":      "reviewing recent purchases or account activity",
    "pay_bill":          "making a payment on a bill",
    "bill_due":          "when a bill payment is due",
    "report_lost_card":  "a lost or stolen card that needs reporting",
    "card_declined":     "a card that was refused or declined at payment",
    "pin_change":        "changing the PIN number on a card",
    "report_fraud":      "reporting fraudulent or unauthorised charges",
    "credit_limit":      "the spending limit on a credit card",
}


@lru_cache(maxsize=4)
def load(split="test", include_oos=True, max_oos=None):
    """-> list of (text, label); label is a ROUTES key or "oos"."""
    from datasets import load_dataset
    ds = load_dataset("clinc_oos", "plus")[split]
    names = ds.features["intent"].names
    keep = set(ROUTES)
    rows, oos = [], []
    for text, idx in zip(ds["text"], ds["intent"]):
        name = names[idx]
        if name in keep:
            rows.append((text, name))
        elif name == OOS and include_oos:
            oos.append((text, OOS))
    if max_oos is not None:
        oos = oos[:max_oos]
    return rows + oos


def summary(rows):
    from collections import Counter
    c = Counter(l for _, l in rows)
    return f"{len(rows)} rows, {len(c)-(OOS in c)} in-scope classes, {c.get(OOS,0)} out-of-scope"
