#!/usr/bin/env python3
"""system1 as an HTTP server, speaking the System One API.

    ./serve.py                      # 127.0.0.1:1977
    ./serve.py --preload-tickets    # with the demo schema's fitted calibration

`POST /v1/systemone` takes one state and a dict of typed questions and answers
all of them from a single embedding -- the shape `State` already had. kev serves
the same endpoint from a 0.8-9B model; this serves it from 22.6M on the ANE.

Two things are specific to an embedding engine and are worth understanding
before pointing a client at it.

**Schemas are compiled, and compiling costs encodes.** An option is a *vector*
here, not a string in a prompt, so a question whose options this server has not
seen costs one encode per option before it can answer. A 5-option choice is
~6.6 ms cold and ~1.1 ms warm; a 60-option one is ~67 ms cold. Compiled schemas
are cached by a hash of their criteria, so this is a first-request cost. That
hash is also what makes the cache safe: change one description and it is a
different schema, because it is a different set of vectors.

**Probabilities are uncalibrated unless you have fitted them.** This project's
most repeated finding is that thresholds do not transfer between schemas and
the failure is silent -- the CLINC banking min_sim of 0.3085 refuses genuinely
in-scope support tickets at 0.27-0.28. A server that accepted any schema and
returned confident-looking numbers would ship that bug to everyone. So:

  * probabilities are always returned, and always carry `"calibrated": false`
    unless a calibration has been registered for that exact question;
  * `escalate` and `reason` are `null` when uncalibrated, never `false`. The
    server will not imply a threshold it has not fitted.

Register one with `POST /v1/schemas` (see `--preload-tickets` for the shape).
Fit it with calibrate.py, on validation data, for that schema.
"""
import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from system1 import FALLBACK, Boolean, Choice, Score, _softmax

MODEL = "system1-minilm-l6"
MAX_OPTIONS = 255
MAX_QUESTIONS = 64


class Registry:
    """Compiled questions, plus the calibrations that make them answerable.

    Keyed by a hash of the question's declared criteria, so a calibration binds
    to the exact option set it was fitted against and silently stops applying
    if that set changes -- which is the correct behaviour, loudly reported as
    `"calibrated": false` rather than quietly reused.
    """

    def __init__(self, encoder, max_compiled=256):
        self.enc = encoder
        self.lock = threading.Lock()      # Core ML predict is not re-entrant
        self.compiled = {}
        self.calibrations = {}
        self.max_compiled = max_compiled
        self.encodes = 0

    @staticmethod
    def key(spec):
        canon = json.dumps({"type": spec["type"], "criteria": spec.get("criteria")},
                           sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()[:32]

    def register_calibration(self, spec, cal):
        missing = {"temp", "min_sim", "min_margin"} - set(cal)
        if missing:
            raise Bad(f"calibration is missing {sorted(missing)}")
        self.calibrations[self.key(spec)] = dict(cal)
        return self.key(spec)

    def get(self, spec):
        """-> (question, calibration or None, key, encodes spent compiling)."""
        k = self.key(spec)
        cal = self.calibrations.get(k)
        with self.lock:
            hit = self.compiled.get(k)
            if hit is not None and hit[1] == cal:
                return hit[0], cal, k, 0
            n0 = self.encodes
            q = self._compile(spec, cal)
            if len(self.compiled) >= self.max_compiled:
                self.compiled.pop(next(iter(self.compiled)))
            self.compiled[k] = (q, cal)
            return q, cal, k, self.encodes - n0

    def _compile(self, spec, cal):
        t, crit = spec["type"], spec.get("criteria")
        # An uncalibrated question still needs *a* temperature to turn cosines
        # into probabilities. FALLBACK's is the one calibrate.py fitted on
        # CLINC -- a better-informed guess than a round number, and still a
        # guess, which is exactly what "calibrated": false is saying.
        kw = dict(cal or FALLBACK)
        kw.pop("k", None), kw.pop("mode", None), kw.pop("use_description", None)
        kw.pop("fitted_on", None), kw.pop("oos_validated", None)
        kw.pop("keep_in_scope", None)
        if t == "choice":
            opts = _choice_options(crit)
            self.encodes += sum(1 if isinstance(v, str) else len(v)
                                for v in opts.values())
            return Choice(self.enc, opts, calibration=None, **kw)
        if t == "noul":
            yes, no = _noul_texts(crit)
            self.encodes += 2
            return Boolean(self.enc, yes=yes, no=no, calibration=None, **kw)
        if t == "score":
            levels = _score_levels(crit)
            self.encodes += len(levels)
            return Score(self.enc, levels, temp=kw.get("temp", FALLBACK["temp"]))
        raise Bad(f"unknown question type {t!r}; expected choice, noul or score")


class Bad(Exception):
    """A client error -> 400."""


def _choice_options(crit):
    """criteria -> {label: anchor text or list of texts}.

    The API says options carry "description or null". A null description leaves
    the label name itself as the only anchor, which is the weakest thing this
    engine can be given -- FINDINGS.md measures descriptions losing to examples,
    and a bare identifier is worse than a description. Accepted, not advised.

    `examples` is an extension: a list per option, centroided into the anchor.
    It is the configuration the headline numbers use.
    """
    if isinstance(crit, list):
        crit = {c: None for c in crit}
    if not isinstance(crit, dict) or not crit:
        raise Bad("choice criteria must be a non-empty object or list")
    if len(crit) > MAX_OPTIONS:
        raise Bad(f"choice has {len(crit)} options; the limit is {MAX_OPTIONS}")
    out = {}
    for label, v in crit.items():
        if v is None:
            out[label] = str(label).replace("_", " ")
        elif isinstance(v, str):
            out[label] = v
        elif isinstance(v, dict):
            texts = ([v["description"]] if v.get("description") else []) \
                + list(v.get("examples") or [])
            if not texts:
                texts = [str(label).replace("_", " ")]
            out[label] = texts
        elif isinstance(v, list):
            out[label] = list(v)
        else:
            raise Bad(f"option {label!r} must be a string, list, object or null")
    return out


def _noul_texts(crit):
    if crit is None:
        return "yes, this is true", "no, this is false"
    if not isinstance(crit, dict):
        raise Bad("noul criteria must be an object or null")
    y = crit.get("true") or crit.get("yes") or "yes, this is true"
    n = crit.get("false") or crit.get("no") or "no, this is false"
    return y, n


def _score_levels(crit):
    if not isinstance(crit, list) or len(crit) < 2:
        raise Bad("score criteria must be a list of at least 2 descriptions, "
                  "ordered lowest to highest")
    if len(crit) > MAX_OPTIONS:
        raise Bad(f"score has {len(crit)} levels; the limit is {MAX_OPTIONS}")
    return list(crit)


def _confidence(p_max, k):
    """kev's and Laya's definition: chance-corrected, comparable across K.

    Our own `margin` (p1 - p2) is not: it saturates at a fitted temperature and
    means different things for a 5-label schema and a 60-label one. Both are
    returned -- this one for API compatibility, margin because the fitted
    min_margin is expressed in it.
    """
    return 0.0 if k < 2 else float(max(0.0, (p_max - 1 / k) / (1 - 1 / k)))


def answer(q, cal, vec, spec):
    t = spec["type"]
    if t == "score":
        r = q.decide(vec)
        p = np.array(r["dist"])
        out = {"type": "score", "score": round(r["score"], 4),
               "confidence": round(_confidence(float(p.max()), len(p)), 4),
               "probabilities": {str(i): round(float(x), 4) for i, x in enumerate(p)},
               "legend": {str(i): s for i, s in enumerate(_score_levels(spec["criteria"]))},
               "prob": round(r["confidence"], 4)}
        # Score has no sim gate and no fitted thresholds anywhere in this repo.
        # It is weak enough that FINDINGS.md calls it a detector, not a measure.
        out["calibrated"] = False
        out["escalate"] = None
        out["reason"] = None
        return out
    d = q.decide(vec)
    k = len(d.all)
    out = {"type": t, "confidence": round(_confidence(d.prob, k), 4),
           "probabilities": {str(a): b for a, b in d.all.items()},
           "prob": round(d.prob, 4), "margin": round(d.margin, 4),
           "sim": round(d.sim, 4)}
    if t == "noul":
        out["noul"] = round(d.prob if d.label else 1 - d.prob, 4)
        out["probabilities"] = {"true": out["noul"], "false": round(1 - out["noul"], 4)}
    else:
        out["choice"] = d.label
    if cal:
        out["calibrated"] = True
        out["escalate"] = bool(d.escalate)
        out["reason"] = d.reason or None
    else:
        # No fitted thresholds for this exact schema, so there is no honest
        # value for escalate. null is not false.
        out["calibrated"] = False
        out["escalate"] = None
        out["reason"] = None
    return out


class Server(ThreadingHTTPServer):
    # The default backlog of 5 drops connections past ~8 concurrent clients,
    # which shows up as a broken pipe on the client rather than an error here.
    # Threads are cheap; the ANE is the bottleneck and it is behind a lock.
    request_queue_size = 128
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "system1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        if self.server.verbose:
            super().log_message(fmt, *a)

    def _send(self, code, body):
        raw = json.dumps(body, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            raise Bad("empty body")
        try:
            return json.loads(self.rfile.read(n))
        except json.JSONDecodeError as e:
            raise Bad(f"invalid JSON: {e}") from None

    def do_GET(self):
        reg = self.server.registry
        if self.path in ("/health", "/"):
            return self._send(200, {"status": "ok", "model": MODEL})
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [{
                "id": MODEL, "object": "model", "params": 22_600_000,
                "encoder": "all-MiniLM-L6-v2", "dim": 384, "max_seq": 128,
                "compute": "CPU_AND_NE",
                "types": ["choice", "noul", "score"]}]})
        if self.path == "/v1/schemas":
            return self._send(200, {"compiled": len(reg.compiled),
                                    "calibrated": sorted(reg.calibrations)})
        self._send(404, {"error": {"message": f"no route {self.path}",
                                   "type": "not_found"}})

    def do_POST(self):
        try:
            if self.path == "/v1/systemone":
                return self._send(200, self._systemone(self._read()))
            if self.path == "/v1/schemas":
                return self._send(200, self._schemas(self._read()))
            raise Bad(f"no route {self.path}")
        except Bad as e:
            self._send(400, {"error": {"message": str(e), "type": "invalid_request"}})
        except Exception as e:                                  # noqa: BLE001
            self._send(500, {"error": {"message": f"{type(e).__name__}: {e}",
                                       "type": "internal"}})

    def _schemas(self, body):
        reg = self.server.registry
        spec, cal = body.get("question"), body.get("calibration")
        if not isinstance(spec, dict) or "type" not in spec:
            raise Bad("send {\"question\": {\"type\": ..., \"criteria\": ...}, "
                      "\"calibration\": {\"temp\": ..., \"min_sim\": ..., "
                      "\"min_margin\": ...}}")
        if not isinstance(cal, dict):
            raise Bad("calibration must be an object")
        return {"registered": reg.register_calibration(spec, cal),
                "name": body.get("name"),
                "note": "binds to this exact criteria set; change it and the "
                        "hash changes and answers report calibrated: false"}

    def _systemone(self, body):
        reg = self.server.registry
        t0 = time.perf_counter()
        state, qs = body.get("state"), body.get("questions")
        if state is None:
            raise Bad("'state' is required")
        if not isinstance(qs, dict) or not qs:
            raise Bad("'questions' must be a non-empty object")
        if len(qs) > MAX_QUESTIONS:
            raise Bad(f"{len(qs)} questions; the limit is {MAX_QUESTIONS}")
        text = state if isinstance(state, str) else json.dumps(state, sort_keys=True)

        prepared, cold = [], 0
        for name, spec in qs.items():
            if not isinstance(spec, dict) or "type" not in spec:
                raise Bad(f"question {name!r} needs a 'type'")
            q, cal, key, spent = reg.get(spec)
            cold += spent
            prepared.append((name, spec, q, cal, key))

        # The point of the whole thing: one encode, then a matmul per question.
        with reg.lock:
            t1 = time.perf_counter()
            vec = reg.enc.embed([text])[0]
            reg.encodes += 1
            encode_ms = (time.perf_counter() - t1) * 1000

        answers = {name: answer(q, cal, vec, spec)
                   for name, spec, q, cal, _ in prepared}
        uncal = [n for n, a in answers.items() if not a["calibrated"]]
        out = {"model": MODEL, "answers": answers,
               "usage": {"encodes": 1 + cold, "questions": len(qs),
                         "schema_encodes": cold, "chars": len(text)},
               "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
               "encode_ms": round(encode_ms, 3)}
        if uncal:
            out["warning"] = (
                f"no fitted calibration for {sorted(uncal)}; probabilities are "
                f"uncalibrated and escalate is null. Fit with calibrate.py and "
                f"register via POST /v1/schemas.")
        return out


def preload_tickets(reg):
    import tickets
    cal = json.load(open("calibration_tickets.json"))
    spec = {"type": "choice", "criteria": dict(tickets.ROUTES)}
    reg.register_calibration(spec, cal)
    reg.get(spec)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1977)
    ap.add_argument("--encoder", default="encoder.mlpackage")
    ap.add_argument("--tok", default="tok")
    ap.add_argument("--preload-tickets", action="store_true",
                    help="compile tickets.ROUTES with its fitted calibration")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    from system1 import Encoder
    print(f"loading {args.encoder} ...")
    reg = Registry(Encoder(args.encoder, args.tok))
    reg.enc.embed(["warm up the compute planner"])      # first call is slow
    if args.preload_tickets:
        spec = preload_tickets(reg)
        print(f"preloaded tickets schema ({len(spec['criteria'])} routes, calibrated)")

    srv = Server((args.host, args.port), Handler)
    srv.registry, srv.verbose = reg, args.verbose
    print(f"system1 on http://{args.host}:{args.port}  POST /v1/systemone")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
