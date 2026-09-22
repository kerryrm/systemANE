"""Domain -> category. Free labels: written from domain knowledge, never from
looking at the harvested titles. Link shorteners and CDNs are excluded rather
than guessed at."""

RULES = {
 "video": ["youtube.com", "youtu.be", "vimeo.com", "twitch.tv", "tiktok.com",
           "dailymotion.com", "rumble.com", "nicovideo.jp"],
 "music": ["spotify.com", "bandcamp.com", "soundcloud.com", "last.fm",
           "genius.com", "discogs.com", "mixcloud.com", "audiomack.com"],
 "news":  ["nytimes.com", "theguardian.com", "bbc.co.uk", "bbc.com",
           "washingtonpost.com", "reuters.com", "apnews.com", "cnn.com",
           "npr.org", "globo.com", "elpais.com", "lemonde.fr", "spiegel.de",
           "orf.at", "cleveland.com", "politico.com", "axios.com",
           "bloomberg.com", "ft.com", "aljazeera.com", "independent.co.uk",
           "telegraph.co.uk", "thehill.com", "newsweek.com", "cbsnews.com",
           "nbcnews.com", "usatoday.com", "latimes.com", "sfgate.com",
           "theatlantic.com", "vox.com", "salon.com", "huffpost.com",
           "dailykos.com", "rollingstone.com", "mediaite.com", "thedailybeast.com",
           "folha.uol.com.br", "zeit.de", "faz.net", "heise.de", "nzz.ch",
           "abc.es", "lavanguardia.com", "corriere.it", "repubblica.it"],
 "code":  ["github.com", "gitlab.com", "stackoverflow.com", "codeberg.org",
           "npmjs.com", "pypi.org", "huggingface.co", "readthedocs.io"],
 "research": ["arxiv.org", "doi.org", "nature.com", "science.org",
              "biorxiv.org", "medrxiv.org", "plos.org", "springer.com",
              "sciencedirect.com", "researchgate.net", "jstor.org"],
 "art":   ["deviantart.com", "artstation.com", "pixiv.net", "behance.net",
           "cara.app", "newgrounds.com"],
 "shop":  ["etsy.com", "amazon.com", "ebay.com", "redbubble.com",
           "teepublic.com", "bigcartel.com", "society6.com", "threadless.com"],
 "games": ["steampowered.com", "itch.io", "gog.com", "epicgames.com",
           "nintendo.com", "playstation.com", "xbox.com", "roblox.com"],
 "fund":  ["patreon.com", "ko-fi.com", "kickstarter.com", "gofundme.com",
           "indiegogo.com", "buymeacoffee.com", "liberapay.com"],
 "books": ["goodreads.com", "bookshop.org", "archiveofourown.org",
           "wattpad.com", "storygraph.com", "royalroad.com"],
}

# Redirectors, CDNs and aggregators: the domain says nothing about the topic.
#
# The blog platforms are here for a subtler reason, found by running this the
# first time: a domain rule labels the HOST, and hosts divide into two kinds.
# nytimes.com and arxiv.org are topic-bearing -- the host implies the content.
# substack.com and medium.com are infrastructure -- they host anything, and a
# Substack piece about politics *is* news. Labelling them "blog" put 33 of 66
# news cards into a category the text cannot possibly predict, because the
# distinction is not in the text. Only topic-bearing hosts make usable labels.
EXCLUDE = {"dlvr.it", "bit.ly", "t.co", "ift.tt", "buff.ly", "tinyurl.com",
           "klipy.com", "bsky.app", "linktr.ee", "google.com", "flip.it",
           "trib.al", "zpr.io", "wp.me", "shorturl.at", "rb.gy", "ow.ly",
           "substack.com", "medium.com", "wordpress.com", "blogspot.com",
           "ghost.io", "dev.to", "tumblr.com", "bearblog.dev", "hashnode.dev"}

DOMAIN2CAT = {d: c for c, ds in RULES.items() for d in ds}

def label(domain):
    if domain in EXCLUDE:
        return None
    return DOMAIN2CAT.get(domain)
