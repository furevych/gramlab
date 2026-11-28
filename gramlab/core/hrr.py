"""run 22 · encode · calculate · decode, done properly: Holographic Reduced Representations over the project's own records.

The architecture question left open since Nov 22: can a record be *calculated over* and still give back the specific
tokens? The pilot used outer-product memories on synthetic worlds; this is Plate's HRR on real sentences:

  codebook   every filler (lemma, name, class, sense) and every role (ACTOR, ACTION, PATIENT, STATUS) is a fixed random
             unit vector of dimension d (layer 0 = the codebook; deterministic from the token, so nothing is stored per item)
  encode     record R = ACTOR ⊛ actor + ACTION ⊛ action + PATIENT ⊛ patient + STATUS ⊛ status     (⊛ = circular convolution)
  superpose  memory M = Σ R_i over k records (a document, a bucket, a whole collection slice)
  calculate  unbind with a role: M ⊛ ACTOR⁻¹ ≈ actor_1 + actor_2 + … + noise; chain: who did V to X? → unbind, cleanup
  decode     cleanup = nearest codebook vector (cosine) — the *specific token* comes back, or "no answer" below a margin

Measured, all on AMALGUM records produced by the gold linker (silver relations) and, for live conditions, by the learned
linker on the live front end:
  1. capacity: recovery accuracy of each slot vs. number of superposed records k and dimension d
  2. reversal: for a record and its role-swapped twin, is the twin's memory distinguishable? (cosine gap, and whether
     unbinding ACTOR from the twin returns the wrong filler — it should, by construction)
  3. one-step questions on a superposed memory: "who V-ed X?" = unbind ACTOR from (M ⊛ (ACTION ⊛ V + PATIENT ⊛ X)⁻¹)…
     done the practical way: probe p = ACTION ⊛ V + PATIENT ⊛ X; score each stored record's similarity to p; take the best
     record's ACTOR by unbinding — compared with the exact key index on the same questions
  4. cost: bytes and microseconds per record and per query, against the key index's postings

    from gramlab.core import hrr
    board = hrr.run(AMALGUM, out_dir=..., dims=(256, 512, 1024, 2048), ks=(1, 4, 16, 64, 256))
"""
from __future__ import annotations

import json
import random
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from .linker import GoldLinker
from .records import read_conllu_glob

ROLES = ("ACTOR", "ACTION", "PATIENT", "STATUS")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------------------------------------ algebra
def _vec(key: str, d: int) -> np.ndarray:
    """Deterministic random unit vector for a symbol: the codebook needs no storage."""
    rng = np.random.default_rng(zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF)
    v = rng.standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


def bind(a, b):
    """Circular convolution via FFT (Plate 1995)."""
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=a.shape[-1]).astype(np.float32)


def inverse(a):
    """Approximate inverse = involution a[-i mod d]; exact for unitary vectors, good enough for random unit ones."""
    return np.concatenate([a[:1], a[1:][::-1]]).astype(np.float32)


def unbind(m, role):
    return bind(m, inverse(role))


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


class Codebook:
    """Fillers seen so far (for cleanup) + roles. Vectors are recomputed from the symbol, so the book is just a list of names."""

    def __init__(self, d):
        self.d = d
        self.roles = {r: _vec("ROLE::" + r, d) for r in ROLES}
        self.names: list[str] = []
        self.index: dict[str, int] = {}
        self.M = np.zeros((0, d), dtype=np.float32)

    def add(self, names):
        new = [n for n in names if n not in self.index]
        if not new:
            return
        for n in new:
            self.index[n] = len(self.names); self.names.append(n)
        self.M = np.vstack([self.M, np.stack([_vec("SYM::" + n, self.d) for n in new])])

    def vec(self, name):
        return _vec("SYM::" + name, self.d)

    def cleanup(self, v, top=1):
        s = self.M @ unit(v)
        idx = np.argsort(-s)[:max(top, 1)]
        out = [(self.names[i], float(s[i])) for i in idx]
        return out[0] if top == 1 else out


# ------------------------------------------------------------------------------------------------ records
def records_from(sents, linker, limit=None, seed=0):
    """(actor, action, patient, status, sid) tuples from events with both participants; fillers are head lemmas lower-cased."""
    out = []
    for s in sents:
        for e in linker.link(s):
            if e.actor is None or e.patient is None:
                continue
            a = s.lemmas[e.actor.head].lower(); p = s.lemmas[e.patient.head].lower()
            if not a.isalpha() or not p.isalpha() or a == p:
                continue
            out.append((a, e.action.lower(), p, e.status, s.sid))
    if limit and len(out) > limit:
        random.Random(seed).shuffle(out); out = out[:limit]
    return out


def encode(rec, cb: Codebook):
    a, v, p, st, _ = rec
    R = cb.roles
    return bind(R["ACTOR"], cb.vec(a)) + bind(R["ACTION"], cb.vec(v)) + bind(R["PATIENT"], cb.vec(p)) + bind(R["STATUS"], cb.vec(st))


# ------------------------------------------------------------------------------------------------ experiments
def capacity(recs, d, ks, trials=40, seed=0, log=log):
    """Superpose k records into one memory M. Unbinding a role from M returns the *sum* of that role's k fillers, so the
    right test is: does each record's filler rank within the top k of the cleanup (recall@k over the whole codebook)?
    Also the question primitive on stored record vectors: probe ACTION⊛v + PATIENT⊛p → best record → unbind ACTOR."""
    rng = random.Random(seed)
    cb = Codebook(d)
    cb.add({x for r in recs for x in (r[0], r[1], r[2], r[3])})
    out = {}
    for k in ks:
        hits = Counter(); n = 0; q_hit = 0
        for _ in range(trials):
            batch = rng.sample(recs, k)
            E = np.stack([encode(r, cb) for r in batch])
            M = E.sum(0)
            En = unit(E)
            tops = {role: set(nm for nm, _ in cb.cleanup(unbind(M, cb.roles[role]), top=max(k, 2))) for role in ROLES[:3]}
            for i, r in enumerate(batch):
                for role, filler in zip(ROLES[:3], r[:3]):
                    hits[role] += filler in tops[role]
                probe = unit(bind(cb.roles["ACTION"], cb.vec(r[1])) + bind(cb.roles["PATIENT"], cb.vec(r[2])))
                best = int(np.argmax(En @ probe))
                name, _ = cb.cleanup(unbind(E[best], cb.roles["ACTOR"]))
                q_hit += name == r[0]
                n += 1
        out[k] = {role + "_recall_at_k": round(hits[role] / n, 4) for role in ROLES[:3]} | {"actor_via_question": round(q_hit / n, 4), "n": n}
        log(f"  d={d:5d} k={k:4d}  " + "  ".join(f"{r}@k:{out[k][r + '_recall_at_k']:.3f}" for r in ROLES[:3]) + f"  Q→actor:{out[k]['actor_via_question']:.3f}")
    return out


def reversal(recs, d, n=2000, seed=0):
    """Record vs its role-swapped twin: cosine between the two encodings, and does unbinding ACTOR from the twin give the
    twin's actor (= the original patient)? By construction it should, every time; the number says how clean."""
    rng = random.Random(seed)
    cb = Codebook(d)
    cb.add({x for r in recs for x in (r[0], r[1], r[2], r[3])})
    sample = rng.sample(recs, min(n, len(recs)))
    cos, ok_twin, ok_orig, margin = [], 0, 0, []
    for a, v, p, st, sid in sample:
        R = encode((a, v, p, st, sid), cb); T = encode((p, v, a, st, sid), cb)
        cos.append(float(unit(R) @ unit(T)))
        na, sa = cb.cleanup(unbind(R, cb.roles["ACTOR"])); nt, stt = cb.cleanup(unbind(T, cb.roles["ACTOR"]))
        ok_orig += na == a; ok_twin += nt == p
        # how far is the right filler from the wrong one when reading ACTOR out of the twin?
        s_right = float(cb.vec(p) @ unit(unbind(T, cb.roles["ACTOR"]))); s_wrong = float(cb.vec(a) @ unit(unbind(T, cb.roles["ACTOR"])))
        margin.append(s_right - s_wrong)
    return {"n": len(sample), "cosine_record_vs_twin_mean": round(float(np.mean(cos)), 4), "actor_decoded_right_original": round(ok_orig / len(sample), 4),
            "actor_decoded_right_twin": round(ok_twin / len(sample), 4), "margin_right_minus_wrong_mean": round(float(np.mean(margin)), 4),
            "margin_min": round(float(np.min(margin)), 4)}


def questions(recs, d, n_q=1000, pool=20000, seed=0, log=log):
    """'Who V-ed X?' over a stored collection: HRR scoring (probe vs stored record vectors) vs exact key lookup.
    Collection = `pool` records; questions = n_q of them with the actor hidden. Hit = decoded actor equals the stored one.
    Distractors are natural: other records sharing the verb or the patient."""
    rng = random.Random(seed)
    cb = Codebook(d)
    cb.add({x for r in recs for x in (r[0], r[1], r[2], r[3])})
    coll = rng.sample(recs, min(pool, len(recs)))
    E = unit(np.stack([encode(r, cb) for r in coll]))
    qs = rng.sample(range(len(coll)), min(n_q, len(coll)))
    exact = {}
    for i, r in enumerate(coll):
        exact.setdefault((r[1], r[2]), []).append(i)
    hrr_hit = key_hit = amb = 0
    t = time.time()
    for qi in qs:
        a, v, p, st, sid = coll[qi]
        probe = unit(bind(cb.roles["ACTION"], cb.vec(v)) + bind(cb.roles["PATIENT"], cb.vec(p)))
        best = int(np.argmax(E @ probe))
        name, _ = cb.cleanup(unbind(E[best], cb.roles["ACTOR"]))
        hrr_hit += name == a
        cands = exact[(v, p)]
        key_hit += coll[cands[0]][0] == a            # first posting, as a plain key lookup would return
        amb += len(cands) > 1
    us = (time.time() - t) / len(qs) * 1e6
    return {"pool": len(coll), "questions": len(qs), "hrr_top1_actor": round(hrr_hit / len(qs), 4), "key_first_posting_actor": round(key_hit / len(qs), 4),
            "questions_with_several_exact_matches": amb, "hrr_us_per_question": round(us, 1), "bytes_per_record_hrr_f32": 4 * d, "bytes_per_record_f16": 2 * d}


def run(amalgum_glob, out_dir="results", dims=(256, 512, 1024, 2048), ks=(1, 4, 16, 64, 256), limit=60000, seed=0, frontend=None, linker=None):
    t0 = time.time()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sents = read_conllu_glob(amalgum_glob)
    gold = GoldLinker()
    recs = records_from(sents, gold, limit=limit, seed=seed)
    fillers = {x for r in recs for x in (r[0], r[1], r[2])}
    log(f"{len(sents)} sentences → {len(recs)} records (gold roles), {len(fillers)} distinct fillers")
    board = {"records": len(recs), "fillers": len(fillers), "capacity": {}, "reversal": {}, "questions": {}, "live": None}
    for d in dims:
        log(f"=== d={d}")
        board["capacity"][d] = capacity(recs, d, ks, seed=seed, log=log)
        board["reversal"][d] = reversal(recs, d, seed=seed)
        log(f"  reversal: cos(record,twin)={board['reversal'][d]['cosine_record_vs_twin_mean']}  actor-from-twin right={board['reversal'][d]['actor_decoded_right_twin']}  margin min={board['reversal'][d]['margin_min']}")
        board["questions"][d] = questions(recs, d, seed=seed, log=log)
        log(f"  questions: hrr {board['questions'][d]['hrr_top1_actor']} vs key-first-posting {board['questions'][d]['key_first_posting_actor']} ({board['questions'][d]['questions_with_several_exact_matches']} ambiguous) · {board['questions'][d]['hrr_us_per_question']} µs/q")
    if frontend and linker:
        # live conditions: records from the learned linker on front-end tags, questions answered against gold records
        from .linker import BackfillLinker, LearnedLinker
        from .tagger import LiveFrontEnd
        fe = LiveFrontEnd.load(frontend); lk = BackfillLinker(LearnedLinker.load(linker))
        sub = sents[:20000]
        live = [fe.sentence(s.forms, s.sid, s.doc) for s in sub]
        live_recs = records_from(live, lk)
        gold_recs = {(r[4], r[1], r[2]): r[0] for r in records_from(sub, gold)}
        agree = sum(1 for r in live_recs if gold_recs.get((r[4], r[1], r[2])) == r[0])
        board["live"] = {"sentences": len(sub), "live_records": len(live_recs), "gold_records": len(gold_recs), "live_records_agreeing_with_gold": agree,
                         "precision_vs_gold": round(agree / max(len(live_recs), 1), 4)}
        log(f"live: {board['live']}")
    board["seconds"] = round(time.time() - t0, 1)
    (Path(out_dir) / "hrr.json").write_text(json.dumps(board, indent=2))
    log(f"done in {board['seconds']}s")
    return board
