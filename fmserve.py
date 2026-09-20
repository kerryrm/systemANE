"""Minimal stdlib client for the local Apple Foundation Models server.

`fm serve` is the Apple Foundation Models CLI shipped in macOS 27
(/usr/bin/fm); start it with `fm serve`, which listens on 127.0.0.1:1976 by
default. Stdlib only, so this project stays standalone.

Uses `response_format` json_schema, which that server honours as constrained
decoding, so tier 2 is held to the same label set as tier 1.
"""
import json
import re
import urllib.error
import urllib.request

# `fm serve` leaks raw chat-template tokens into message content. Any client
# that skips this occasionally surfaces <start_of_turn>model or <ctrl46> in
# what it thinks is the answer.
TEMPLATE_TOKENS = re.compile(r"<(?:start|end)_of_turn>(?:model|user)?\s*|<ctrl\d+>\s*")


def strip_template_tokens(text):
    return TEMPLATE_TOKENS.sub("", text).strip()


class FMError(RuntimeError):
    pass


class FMServer:
    def __init__(self, base_url="http://127.0.0.1:1976", model="system", timeout=120):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    def up(self):
        try:
            urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=3).read()
            return True
        except OSError:
            return False

    def _post(self, body):
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise FMError(f"HTTP {e.code}: {e.read().decode()[:160]}") from None
        except OSError as e:
            raise FMError(str(e)) from None

    def ask(self, prompt, system=None, temperature=0.0):
        """Free-form answer. Returns (text, usage)."""
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        d = self._post({"model": self.model, "messages": msgs,
                        "temperature": temperature, "stream": False})
        text = strip_template_tokens(d["choices"][0]["message"]["content"])
        return text, d.get("usage", {})

    def choose(self, text, options, system=None):
        """Route `text` into one of `options` (label -> description)."""
        labels = list(options)
        menu = "\n".join(f"- {l}: {d}" for l, d in options.items())
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or
                 "You classify support tickets. Answer only with the schema."},
                {"role": "user", "content":
                 f"Categories:\n{menu}\n\nTicket: {text}\n\nPick the single best category."}],
            "temperature": 0.0,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "route", "schema": {
                    "type": "object",
                    "properties": {"route": {"type": "string", "enum": labels}},
                    "required": ["route"]}}},
        }
        payload = self._post(body)
        text_out = strip_template_tokens(payload["choices"][0]["message"]["content"])
        try:
            label = json.loads(text_out).get("route")
        except json.JSONDecodeError:
            label = next((l for l in labels if l.lower() in text_out.lower()), None)
        if label not in labels:
            raise FMError(f"off-schema answer: {text_out[:120]!r}")
        return label
