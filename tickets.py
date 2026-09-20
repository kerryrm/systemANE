"""Shared fixtures: a routing schema and a set of tickets, five of which are
deliberately ambiguous and should escalate rather than be answered."""

ROUTES = {
    "billing":  "a problem with payment, charges, invoices, refunds or subscription cost",
    "auth":     "cannot log in, password reset, two factor authentication, locked account",
    "bug":      "the software crashes, errors, freezes or behaves incorrectly",
    "feature":  "a request for new functionality or an enhancement suggestion",
    "shipping": "questions about delivery, tracking a package, or a late order",
}

CLEAR = [
    "I was charged twice for last month",
    "can't log in after the update, it says my password is wrong",
    "the app crashes every time I open the settings page",
    "would be great if you supported dark mode",
    "where is my package, it was supposed to arrive tuesday",
    "my subscription renewed but I cancelled it in march",
    "2FA codes never arrive on my phone",
    "the checkout page throws a 500 when I apply a discount code",
]

# Ambiguous *between* categories: margin is the signal.
AMBIGUOUS = [
    "it broke",
    "I need help with my account",
    "this isn't working",
    "something is wrong",
]

# Outside the schema entirely: no category is correct, so margin cannot help.
# Absolute cosine similarity is the signal -- see README ("Two signals").
OUT_OF_SCOPE = [
    "can someone call me",
    "what are your office hours",
    "I'd like to speak to a manager",
    "happy holidays everyone",
]


# -- a validation set, for fitting this schema's thresholds -----------------
# Fitting on CLEAR/AMBIGUOUS/OUT_OF_SCOPE above would repeat a mistake this
# project already made once: tuning against the same sixteen strings the demo
# then reports on, which produced a temperature that was wrong in both
# magnitude and direction. These are separate
# tickets, disjoint from the fixtures, used only by calibrate_tickets.py.
#
# Honest caveat: unlike CLINC150 these are hand-written by the author of the
# engine, so they can inherit its blind spots. They are good enough to place a
# threshold -- which is all they are used for -- and weaker evidence than
# evalset.py for anything else.
OOS = "oos"

VALIDATION = [
    ("you took the money twice this month and I only have one plan", "billing"),
    ("why has my invoice gone up by twelve pounds", "billing"),
    ("I want a refund for the annual plan I never used", "billing"),
    ("my card was charged after I downgraded to the free tier", "billing"),
    ("can you explain the pro-rata amount on invoice 4471", "billing"),
    ("the receipt shows tax but my company is tax exempt", "billing"),
    ("I need the billing address on my invoices changed", "billing"),
    ("cancelled in january and still being billed", "billing"),
    ("how much is the team plan per seat per year", "billing"),
    ("the discount code from your email did not reduce the price", "billing"),

    ("password reset email never arrives no matter how often I click", "auth"),
    ("account locked after too many attempts, how long until it opens", "auth"),
    ("the authenticator app shows codes but the site rejects them", "auth"),
    ("I lost my phone and my backup codes with it", "auth"),
    ("sso through okta stopped working for the whole team this morning", "auth"),
    ("it keeps signing me out every few minutes", "auth"),
    ("cannot get past the two factor step on a new laptop", "auth"),
    ("my email changed so I can no longer receive the login link", "auth"),
    ("says username or password incorrect but I just reset it", "auth"),
    ("how do I turn on two factor for everyone in the org", "auth"),

    ("the export button produces an empty csv every time", "bug"),
    ("app freezes for thirty seconds when I switch tabs", "bug"),
    ("getting a 500 error on the reports page since yesterday", "bug"),
    ("dates display as 1970 on the dashboard", "bug"),
    ("the mobile app closes itself when I upload a photo", "bug"),
    ("search returns results that do not contain the search term", "bug"),
    ("changes I save revert after a refresh", "bug"),
    ("the page renders on top of itself at small window sizes", "bug"),
    ("notifications fire twice for every event", "bug"),
    ("sync stopped halfway and now shows a spinner forever", "bug"),

    ("any chance of an api for bulk import", "feature"),
    ("please add keyboard shortcuts for the common actions", "feature"),
    ("we would really like single sign on with azure ad", "feature"),
    ("it would help to be able to schedule reports weekly", "feature"),
    ("can you support csv as well as xlsx for export", "feature"),
    ("a dark theme would be much easier on the eyes", "feature"),
    ("consider adding role based permissions for larger teams", "feature"),
    ("would love an undo for bulk deletes", "feature"),
    ("could the mobile app work offline", "feature"),
    ("suggestion: let us tag items and filter by tag", "feature"),

    ("order placed ten days ago and tracking has not updated", "shipping"),
    ("the courier says delivered but nothing arrived", "shipping"),
    ("can I change the delivery address before it ships", "shipping"),
    ("parcel is stuck at customs, what do I do", "shipping"),
    ("when will my order actually be dispatched", "shipping"),
    ("the tracking number you sent me is not recognised", "shipping"),
    ("delivery was supposed to be next day and it has been four", "shipping"),
    ("can you ship to a different country", "shipping"),
    ("the package arrived crushed and one item was missing", "shipping"),
    ("is there a way to get it delivered on saturday", "shipping"),

    ("what are your opening hours on bank holidays", OOS),
    ("can I speak to a human please", OOS),
    ("who is the ceo of your company", OOS),
    ("thanks for sorting that out so quickly", OOS),
    ("do you have any job openings in engineering", OOS),
    ("please remove me from your mailing list", OOS),
    ("happy new year to the whole team", OOS),
    ("can you sponsor our local football team", OOS),
    ("could you send me your company registration number", OOS),
    ("my colleague recommended I get in touch", OOS),
    ("testing testing is anyone there", OOS),
    ("please call me back on this number", OOS),
    ("what is the address of your head office", OOS),
    ("unsubscribe", OOS),
    ("good morning", OOS),
    ("are you a real person or a bot", OOS),
    ("I have attached the document you asked for", OOS),
    ("sorry ignore my last message", OOS),
    ("just checking this email address works", OOS),
    ("can I get a copy of your latest press release", OOS),
    ("what music do you play in the office", OOS),
    ("forwarding this from my colleague for your records", OOS),
]
