"""A second evaluation set, from MASSIVE (`mteb/amazon_massive_intent`, en).

CLINC150 is the set this project argues with: it ships an explicit
out-of-scope class, which is the axis `system1` cares most about. MASSIVE
exists here for a different reason — it is what other decision models publish
on, so it is the only way to put this engine's number next to theirs.

Two differences from `evalset.py` matter when reading any result:

  * **60 intents, not 10.** The CLINC router in `evalset.py` is a 10-way
    banking schema carved out of CLINC150. This is the whole MASSIVE label
    set, so it is a far harder decision and a much more crowded one.
  * **No out-of-scope class.** MASSIVE has none, so `min_sim`, OOS AUROC and
    the operating curve are all unavailable here. Refusal cannot be measured
    on this dataset at all.

Class counts are also very uneven — 810 train examples for the largest class
and 4 for the smallest — so two classes cannot supply the k=16 examples the
headline configuration asks for, and contribute what they have.

Descriptions are derived mechanically from the intent name (`alarm_set` ->
"alarm set"). They are deliberately *not* handwritten: the CLINC descriptions
in `evalset.py` took care to write, and inventing 60 more by hand would make
the description baseline a measure of the author's effort rather than of the
method. Run this dataset with `--no-description`.
"""
from functools import lru_cache

OOS = "oos"

_HF = ("mteb/amazon_massive_intent", "en")

_INTENTS = [
    "alarm_query", "alarm_remove", "alarm_set", "audio_volume_down",
    "audio_volume_mute", "audio_volume_other", "audio_volume_up",
    "calendar_query", "calendar_remove", "calendar_set", "cooking_query",
    "cooking_recipe", "datetime_convert", "datetime_query", "email_addcontact",
    "email_query", "email_querycontact", "email_sendemail", "general_greet",
    "general_joke", "general_quirky", "iot_cleaning", "iot_coffee",
    "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff",
    "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on",
    "lists_createoradd", "lists_query", "lists_remove", "music_dislikeness",
    "music_likeness", "music_query", "music_settings", "news_query",
    "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio",
    "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock",
    "recommendation_events", "recommendation_locations",
    "recommendation_movies", "social_post", "social_query", "takeaway_order",
    "takeaway_query", "transport_query", "transport_taxi", "transport_ticket",
    "transport_traffic", "weather_query",
]

# Mechanical, not handwritten -- see the module docstring.
ROUTES = {name: name.replace("_", " ") for name in _INTENTS}


@lru_cache(maxsize=4)
def load(split="test", include_oos=True, max_oos=None):
    """-> list of (text, label). MASSIVE has no out-of-scope rows to include."""
    from datasets import load_dataset
    ds = load_dataset(*_HF)[split]
    keep = set(ROUTES)
    return [(t, l) for t, l in zip(ds["text"], ds["label_text"]) if l in keep]


def summary(rows):
    from collections import Counter
    c = Counter(l for _, l in rows)
    return (f"{len(rows)} rows, {len(c) - (OOS in c)} in-scope classes, "
            f"{c.get(OOS, 0)} out-of-scope")
