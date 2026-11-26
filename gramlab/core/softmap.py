"""core-v2 · the soft map: layer-0 sense *distributions* + role slots + learned codes, scored by weights, not equality.

Why: core-v1 decided everything at index time (one sense, one key) and matched by equality, so one wrong slot made an
event invisible (RESULTS runs 9c, 16). This module keeps the agreed layers and makes each of them soft:

  stage 0  indexation   every content token -> a distribution over WordNet synsets (SemCor prior x sense order); names,
                        numbers and words outside WordNet -> a hashed word id. Nothing is decided here.
  stage 1  structure    role slots ACTOR / ACTION / PATIENT from the linker (gold relations on the parsed corpus, the
                        learned linker on live text); every other content token -> REST. A sentence is four weighted bags.
  stage 2  codes        one d-dim code per synset / word id, *initialised from the WordNet hierarchy* (a synset's code is
                        a decayed sum of random vectors of its hypernym chain, so siblings start close); a token's code is
                        its sense-weighted mix; a slot's vector is the mean over its tokens; the sentence vector is the
                        gated concatenation of the slot vectors. Codes and gates are trained contrastively (in-batch
                        negatives) on self-paraphrases of AMALGUM sentences: synonym substitution from *any* sense plus
                        word dropout -> the original sentence. No teacher model, numpy only (runs on the Kaggle CPU tier).
  scoring               cosine; hybrid = BM25/max + cosine. Reversal (actor <-> patient) moves words between slots, so it
                        costs two slots' worth of similarity instead of nothing.

Benchmark (same construction as bench_retrieval, held out by document): collection = every AMALGUM sentence + a
role-swapped decoy per query; queries = full-sentence paraphrases (verb synonym from the first sense + voice flip),
tagged and linked *live*; systems = BM25, each map variant alone, BM25 + map hybrid. Variants: roles/synsets (the
design), roles/words (same slots, no layer 0), bow/synsets (layer 0, no structure), roles/synsets untrained (stage 0 + 2
init only). Compare with the key index of run 15 (recall@1 0.902, decoy separation 95.8%).

    python -m gramlab.core.softmap --amalgum 'data/amalgum/amalgum/*/dep/*.conllu' --semcor data/semcor \
        --frontend results/frontend.pkl --linker results/linker.pkl --out results
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .bench_retrieval import BM25, main_event, participle_table, passivize
from .linker import GoldLinker
from .records import Event, read_conllu_glob
from .wordnet import WordNet

SLOTS = ("ACTOR", "ACTION", "PATIENT", "REST")
_UD_TO_WN = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
DROP = {"DET", "ADP", "PUNCT", "CCONJ", "SCONJ", "AUX", "PRON", "SYM", "X", "INTJ", "PART"}
KEEP_PART = {"not", "n't", "never"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------------------------------------ stage 0
class SoftIndexer:
    """token (lemma, upos) -> [(row, weight)] over synset rows (0..n_syn) and hashed word rows (n_syn..)."""

    def __init__(self, wn: WordNet, prior: Counter | None = None, top_k=4, n_hash=32768, decay=0.7):
        self.wn = wn
        self.keys = sorted(wn.data.keys())                     # (pos, off)
        self.row = {k: i for i, k in enumerate(self.keys)}
        self.n_syn = len(self.keys)
        self.n_hash = n_hash
        self.prior = prior or Counter()
        self.top_k, self.decay = top_k, decay
        self._cache = {}

    def __len__(self):
        return self.n_syn + self.n_hash

    def word_row(self, lemma):
        return self.n_syn + zlib.crc32(lemma.lower().encode("utf-8")) % self.n_hash

    def ids(self, lemma, upos):
        key = (lemma.lower(), upos)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        pos = _UD_TO_WN.get(upos)
        out = None
        if pos is not None:
            offs = self.wn.synsets(lemma, pos)
            if offs:
                w = np.array([(self.prior.get((pos, o), 0) + 1.0) * self.decay ** k for k, o in enumerate(offs)], dtype=np.float32)
                order = np.argsort(-w)[: self.top_k]
                w = w[order] / w[order].sum()
                out = [(self.row[pos, offs[i]], float(w[j])) for j, i in enumerate(order)]
        if out is None:
            out = [(self.word_row(lemma), 1.0)]
        self._cache[key] = out
        return out

    def init_codes(self, dim=64, depth=6, decay=0.6, seed=0):
        """Hierarchy-initialised code table: synset code = decayed sum of hashed Gaussian vectors along its hypernym chain."""
        R = np.zeros((len(self), dim), dtype=np.float32)
        base = {}

        def g(key):
            v = base.get(key)
            if v is None:
                v = np.random.default_rng(zlib.crc32(f"{key[0]}{key[1]}".encode()) + seed).standard_normal(dim).astype(np.float32)
                base[key] = v
            return v

        for i, (pos, off) in enumerate(self.keys):
            v, o, w = g((pos, off)).copy(), off, 1.0
            for _ in range(depth):
                h = self.wn.hypernyms(pos, o)
                if not h:
                    break
                o, w = h[0], w * decay
                v += w * g((pos, o))
            R[i] = v / (np.linalg.norm(v) + 1e-6)
        rng = np.random.default_rng(seed + 1)
        H = rng.standard_normal((self.n_hash, dim)).astype(np.float32)
        R[self.n_syn:] = H / np.linalg.norm(H, axis=1, keepdims=True)
        return R

    def sense_prior_from_semcor(self, semcor_dir):
        from .sense import read_semcor
        c = Counter()
        for sent in read_semcor(semcor_dir, self.wn):
            for t in sent:
                if t["gold"] is not None and t["pos"]:
                    c[(t["pos"], t["gold"])] += 1
        self.prior = c
        self._cache.clear()
        return c


# ------------------------------------------------------------------------------------------------ stage 1
def slot_labels(s, events, use_slots=True):
    """slot id per token: ACTOR/ACTION/PATIENT spans of the sentence's events, REST elsewhere (0 everywhere for bow)."""
    n = len(s.forms)
    lab = [3] * n
    if not use_slots:
        return [0] * n
    for e in events:
        if e.patient:
            for i in range(e.patient.start, e.patient.end):
                if lab[i] == 3:
                    lab[i] = 2
        if e.actor:
            for i in range(e.actor.start, e.actor.end):
                if lab[i] == 3:
                    lab[i] = 0
        if lab[e.verb] == 3:
            lab[e.verb] = 1
    return lab


def featurize(s, labels, indexer: SoftIndexer, swap=None):
    """-> (rows, weights, slots) int32/f32/int8 arrays. `swap`=(actor_span, patient_span) exchanges the two spans' slots
    (the role-reversed decoy: same words, opposite roles)."""
    rows, ws, sl = [], [], []
    lab = list(labels)
    if swap is not None:
        (a0, a1), (p0, p1) = swap
        for i in range(a0, a1):
            lab[i] = 2
        for i in range(p0, p1):
            lab[i] = 0
    for i, (lemma, upos) in enumerate(zip(s.lemmas, s.upos)):
        if upos in DROP and not (upos == "PART" and lemma.lower() in KEEP_PART):
            continue
        if not any(ch.isalnum() for ch in lemma):
            continue
        for r, w in indexer.ids(lemma, upos):
            rows.append(r); ws.append(w); sl.append(lab[i])
    return np.asarray(rows, dtype=np.int32), np.asarray(ws, dtype=np.float32), np.asarray(sl, dtype=np.int8)


class Corpus:
    """Flat storage of featurized texts: one (rows, weights, slots) triple per text, plus offsets."""

    def __init__(self):
        self.rows, self.ws, self.sl, self.doc = [], [], [], []
        self.off = [0]

    def add(self, feats, doc=0):
        r, w, s = feats
        self.rows.append(r); self.ws.append(w); self.sl.append(s); self.doc.append(np.full(len(r), doc, dtype=np.int32))
        self.off.append(self.off[-1] + len(r))

    def finish(self):
        self.rows = np.concatenate(self.rows) if self.rows else np.zeros(0, np.int32)
        self.ws = np.concatenate(self.ws) if self.ws else np.zeros(0, np.float32)
        self.sl = np.concatenate(self.sl) if self.sl else np.zeros(0, np.int8)
        self.doc = np.concatenate(self.doc) if self.doc else np.zeros(0, np.int32)
        self.off = np.asarray(self.off, dtype=np.int64)
        return self

    def __len__(self):
        return len(self.off) - 1

    def batch(self, ids):
        """Gather entries of the given texts -> (rows, weights, slots, item index within batch)."""
        parts = [np.arange(self.off[i], self.off[i + 1]) for i in ids]
        idx = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        item = np.repeat(np.arange(len(ids)), [len(p) for p in parts])
        return self.rows[idx], self.ws[idx], self.sl[idx], item


# ------------------------------------------------------------------------------------------------ stage 2
class SoftMap:
    """Code table R [n_ids, dim] + slot gates g [n_slots, dim]; encode = normalise(concat_s g_s * mean_s(codes))."""

    def __init__(self, R, n_slots=4, gates=None):
        self.R = R
        self.n_slots = n_slots
        self.dim = R.shape[1]
        self.g = np.ones((n_slots, self.dim), dtype=np.float32) if gates is None else gates

    def _slot_means(self, rows, ws, sl, item, B):
        D, S = self.dim, self.n_slots
        cell = item * S + sl.astype(np.int64)
        V = np.zeros((B * S, D), dtype=np.float32)
        np.add.at(V, cell, self.R[rows] * ws[:, None])
        W = np.bincount(cell, weights=ws, minlength=B * S).astype(np.float32)
        V /= np.maximum(W, 1e-6)[:, None]
        return V.reshape(B, S, D), W.reshape(B, S)

    def forward(self, rows, ws, sl, item, B):
        V, W = self._slot_means(rows, ws, sl, item, B)
        U = (V * self.g[None]).reshape(B, -1)
        norm = np.linalg.norm(U, axis=1, keepdims=True) + 1e-6
        Z = U / norm
        return Z, (V, W, U, norm)

    def encode(self, corpus: Corpus, ids=None, batch=4096):
        ids = np.arange(len(corpus)) if ids is None else np.asarray(ids)
        out = np.zeros((len(ids), self.n_slots * self.dim), dtype=np.float32)
        for s in range(0, len(ids), batch):
            chunk = ids[s:s + batch]
            rows, ws, sl, item = corpus.batch(chunk)
            out[s:s + len(chunk)] = self.forward(rows, ws, sl, item, len(chunk))[0]
        return out

    def backward(self, dZ, cache, rows, ws, sl, item, B):
        """Gradient of a loss wrt Z back to (dR rows, dg)."""
        V, W, U, norm = cache
        Z = U / norm
        dU = (dZ - Z * (Z * dZ).sum(1, keepdims=True)) / norm
        dU = dU.reshape(B, self.n_slots, self.dim)
        dg = (dU * V).sum(0)
        dV = dU * self.g[None]
        cell = item * self.n_slots + sl.astype(np.int64)
        dVflat = dV.reshape(-1, self.dim) / np.maximum(W.reshape(-1), 1e-6)[:, None]
        dRrows = dVflat[cell] * ws[:, None]
        return dRrows, dg

    def save(self, path):
        np.savez_compressed(path, R=self.R, g=self.g)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["R"], d["g"].shape[0], d["g"])


def train(model: SoftMap, qc: Corpus, pc: Corpus, epochs=4, batch=256, lr=3e-3, tau=0.05, seed=0, log=log, shuffle=True):
    """InfoNCE with in-batch negatives (query -> its source), Adam on the code table and gates."""
    rng = np.random.default_rng(seed)
    n = len(qc)
    mR, vR = np.zeros_like(model.R), np.zeros_like(model.R)
    mg, vg = np.zeros_like(model.g), np.zeros_like(model.g)
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    for ep in range(epochs):
        perm = rng.permutation(n) if shuffle else np.arange(n)
        tot, cnt, acc = 0.0, 0, 0
        for s in range(0, n, batch):
            ids = perm[s:s + batch]
            B = len(ids)
            if B < 8:
                continue
            qr, qw, qs, qi = qc.batch(ids)
            pr, pw, ps, pi = pc.batch(ids)
            Q, qcache = model.forward(qr, qw, qs, qi, B)
            P, pcache = model.forward(pr, pw, ps, pi, B)
            S = Q @ P.T / tau
            S -= S.max(1, keepdims=True)
            E = np.exp(S)
            prob = E / E.sum(1, keepdims=True)
            loss = -np.log(prob[np.arange(B), np.arange(B)] + 1e-9).mean()
            acc += int((S.argmax(1) == np.arange(B)).sum())
            dS = prob.copy(); dS[np.arange(B), np.arange(B)] -= 1.0; dS /= B
            dQ = dS @ P / tau
            dP = dS.T @ Q / tau
            dRq, dgq = model.backward(dQ, qcache, qr, qw, qs, qi, B)
            dRp, dgp = model.backward(dP, pcache, pr, pw, ps, pi, B)
            # sparse Adam on touched rows
            rows = np.concatenate([qr, pr]); grads = np.concatenate([dRq, dRp])
            uniq, inv = np.unique(rows, return_inverse=True)
            G = np.zeros((len(uniq), model.dim), dtype=np.float32)
            np.add.at(G, inv, grads)
            t += 1
            mR[uniq] = b1 * mR[uniq] + (1 - b1) * G
            vR[uniq] = b2 * vR[uniq] + (1 - b2) * G * G
            model.R[uniq] -= lr * (mR[uniq] / (1 - b1 ** t)) / (np.sqrt(vR[uniq] / (1 - b2 ** t)) + eps)
            dg = dgq + dgp
            mg = b1 * mg + (1 - b1) * dg
            vg = b2 * vg + (1 - b2) * dg * dg
            model.g -= lr * (mg / (1 - b1 ** t)) / (np.sqrt(vg / (1 - b2 ** t)) + eps)
            tot += loss * B; cnt += B
        log(f"[softmap] epoch {ep + 1}: loss {tot / max(cnt, 1):.4f}  in-batch top-1 {acc / max(cnt, 1):.3f}")
    return model


# ------------------------------------------------------------------------------------------------ paraphrases
def paraphrase(s, events, indexer: SoftIndexer, rng, p_sub=0.5, p_drop=0.15):
    """Self-paraphrase for training: substitute content words with a synonym from a *random* sense (so the map must
    learn which senses actually co-occur), drop words at random; roles stay where the source had them."""
    lemmas, upos, keep = [], [], []
    wn = indexer.wn
    for i, (lemma, u) in enumerate(zip(s.lemmas, s.upos)):
        if rng.random() < p_drop and u in ("NOUN", "VERB", "ADJ", "ADV", "PROPN"):
            continue
        new = lemma
        pos = _UD_TO_WN.get(u)
        if pos and rng.random() < p_sub:
            offs = wn.synsets(lemma, pos)
            if offs:
                cands = [w for w in wn.lemmas(pos, rng.choice(offs)) if " " not in w and w.lower() != lemma.lower()]
                if cands:
                    new = rng.choice(cands)
        lemmas.append(new); upos.append(u); keep.append(i)
    return lemmas, upos, keep


class _Lite:
    __slots__ = ("lemmas", "upos", "forms")

    def __init__(self, lemmas, upos):
        self.lemmas, self.upos, self.forms = lemmas, upos, lemmas


# ------------------------------------------------------------------------------------------------ sense sanity
def sense_report(model: SoftMap, indexer: SoftIndexer, probes=("call", "run", "bank", "fire", "strike"), rng=None):
    """Are codes of synonyms (same synset, different lemma rows) closer than random? Nearest synsets for a few probes."""
    rng = rng or random.Random(0)
    wn = indexer.wn
    R = model.R[: indexer.n_syn]
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-6)
    same, rand = [], []
    verbs = [k for k in indexer.keys if k[0] == "v"]
    for _ in range(2000):
        pos, off = rng.choice(verbs)
        hs = wn.hypernyms(pos, off)
        if not hs:
            continue
        sib = [o for h in hs for o in _hyponyms(wn, pos, h) if o != off]
        if not sib:
            continue
        same.append(float(Rn[indexer.row[pos, off]] @ Rn[indexer.row[pos, rng.choice(sib)]]))
        rand.append(float(Rn[indexer.row[pos, off]] @ Rn[indexer.row[rng.choice(verbs)]]))
    out = {"sibling_cosine_mean": round(float(np.mean(same)), 4), "random_cosine_mean": round(float(np.mean(rand)), 4), "pairs": len(same)}
    ex = {}
    for p in probes:
        for pos in ("v", "n"):
            offs = wn.synsets(p, pos)[:2]
            for off in offs:
                v = Rn[indexer.row[pos, off]]
                sims = Rn @ v
                top = np.argsort(-sims)[1:6]
                ex[wn.name(pos, off)] = [wn.name(*indexer.keys[i]) for i in top]
    out["neighbours"] = ex
    return out


def _hyponyms(wn, pos, off):
    return [o for sym, o, p in wn.data[pos, off][2] if sym in ("~", "~i") and p == pos][:20]


# ------------------------------------------------------------------------------------------------ benchmark
def run(amalgum_glob, semcor_dir, frontend_path, linker_path, out_dir="results", n_queries=2000, seed=0, dim=64, epochs=4,
        train_pairs=120000, holdout_mod=5, variants=("roles/synsets", "roles/words", "bow/synsets", "roles/synsets/untrained"), wordnet_path=None,
        key_bonus=0.0):
    """key_bonus > 0 adds the core-v1 key index as a third signal: BM25/max + cosine + key_bonus·[sentence in the query's
    family ∪ identity postings] (the run-14 lookup, decoys indexed under their swapped roles)."""
    t0 = time.time()
    rng = random.Random(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    wn = WordNet(wordnet_path)
    indexer = SoftIndexer(wn)
    prior = indexer.sense_prior_from_semcor(semcor_dir) if semcor_dir else Counter()
    log(f"WordNet {wn.stats()} · SemCor prior over {len(prior)} synsets")
    word_indexer = SoftIndexer(wn, prior)           # same table layout; `ids` overridden below for the words-only variant
    word_indexer.ids = lambda lemma, upos, _w=word_indexer: [(_w.word_row(lemma), 1.0)]
    hard_indexer = SoftIndexer(wn, prior, top_k=1)             # one sense, the most frequent (a hard layer 0)
    first_indexer = SoftIndexer(wn, Counter(), top_k=1)        # one sense, WordNet order (core-v1's choice)

    sents = read_conllu_glob(amalgum_glob)
    gold = GoldLinker()
    docs = sorted({s.doc for s in sents})
    doc_id = {d: i for i, d in enumerate(docs)}
    held = {d for d in docs if zlib.crc32(d.encode()) % holdout_mod == 0}
    log(f"{len(sents)} sentences, {len(docs)} docs, {len(held)} held-out docs")
    events = [gold.link(s) for s in sents]
    vbn_of = participle_table(sents)

    # queries: held-out sources eligible for the live paraphrase (same rule as bench_retrieval live mode)
    from .registry import GeneralIndex
    general = GeneralIndex(wordnet_path=wordnet_path)
    cands = []
    for si, s in enumerate(sents):
        if s.doc not in held:
            continue
        e = main_event(s, gold)
        if e is None or e.voice != "active" or s.xpos[e.verb] != "VBD" or e.actor.end != e.verb or e.patient.start != e.verb + 1:
            continue
        ss = general.synsets(e.action, "VERB")
        if not ss:
            continue
        first = set(wn.lemmas(*ss[0]))
        syn = next((w for w in general.synonyms(e.action, "VERB", 50) if " " not in w and w != e.action and w in first), None)
        if syn is None or syn not in vbn_of:
            continue
        cands.append((si, e, syn))
    rng.shuffle(cands)
    queries = cands[:n_queries]
    log(f"{len(cands)} eligible held-out sources, using {len(queries)}")

    # live front end + linker for the queries
    from .linker import LearnedLinker
    from .tagger import LiveFrontEnd
    fe = LiveFrontEnd.load(frontend_path)
    live_linker = LearnedLinker.load(linker_path)
    q_texts, q_sents, q_events, live_stats = [], [], [], Counter()
    for si, e, syn in queries:
        s = sents[si]
        q_text = passivize(s, e, syn, vbn_of[syn])
        q_sent = fe.sentence(q_text.split(), sid="q", doc="q")
        evs = [x for x in live_linker.link(q_sent) if x.actor and x.patient]
        q_event = next((x for x in evs if x.action == syn), None) or next((x for x in evs if x.voice == "passive"), None) or (evs[0] if evs else None)
        if q_event is None:
            live_stats["no_event"] += 1
            q_evs = []
        else:
            q_evs = [q_event]
            inside = lambda m, w: w.lower() in [x.lower() for x in m.text.split()]
            live_stats["roles_right"] += inside(q_event.actor, s.forms[e.actor.head]) and inside(q_event.patient, s.forms[e.patient.head])
        q_texts.append(q_text); q_sents.append(q_sent); q_events.append(q_evs)
    log(f"live queries: {dict(live_stats)} (of {len(queries)})")

    texts = [s.text() for s in sents]
    decoy_of = {}
    for si, e, syn in queries:
        decoy_of[si] = len(texts)
        from .bench_retrieval import swap_text
        texts.append(swap_text(sents[si], e))
    bm = BM25(texts)
    log(f"collection {len(texts)} incl. {len(decoy_of)} decoys; BM25 vocab {len(bm.vocab)}")
    bm_scores = [bm.scores(q) for q in q_texts]

    key_sets = None
    if key_bonus > 0:
        from .pipeline import Pipeline
        from .records import Occurrence
        from .registry import SpecificIndex
        t = time.time()
        pipe = Pipeline(linker=gold, general=general, specific=SpecificIndex())
        pipe.ingest(sents)
        sid_to_text = {s.sid: i for i, s in enumerate(sents)}
        for si, e, syn in queries:
            s = sents[si]
            swapped = Event(e.verb, e.action, e.patient, e.actor, e.voice, e.status)
            rs = pipe.keyer.resolve_event(swapped, s)
            keys = pipe.keyer.keys_from(rs)
            d_id = decoy_of[si]
            pipe.index.records.append(Occurrence(swapped, rs, texts[d_id], s.doc, s.sid + "#decoy", keys["specific"][0], keys["family"][0]))
            sid_to_text[s.sid + "#decoy"] = d_id
            for level in ("specific", "family", "identity", "proposal"):
                for k in keys[level]:
                    pipe.index.postings[(level, k, rs.status)].append(d_id)
        key_sets = []
        for qi, (si, e, syn) in enumerate(queries):
            q_event = q_events[qi][0] if q_events[qi] else Event(0, syn, None, None, "passive", "asserted")
            qkeys = pipe.keyer.keys_from(pipe.keyer.resolve_event(q_event), query=True)
            cset = set()
            for level, ks in (("family", qkeys["family"]), ("identity", qkeys["identity"]), ("proposal", qkeys["proposal"])):
                for k in ks:
                    for st in ("asserted", "negated", "possible"):
                        for i in pipe.index.postings.get((level, k, st), ()):
                            cset.add(sid_to_text[pipe.index.records[i].sid])
            key_sets.append(np.fromiter(cset, dtype=np.int64) if cset else np.zeros(0, dtype=np.int64))
        log(f"key index built in {time.time() - t:.0f}s; empty candidate sets {sum(len(c) == 0 for c in key_sets)}/{len(queries)}")

    # training pairs from non-held-out sentences
    train_ids = [i for i, s in enumerate(sents) if s.doc not in held and len(s.forms) >= 5]
    rng.shuffle(train_ids)
    train_ids = train_ids[:train_pairs]
    prng = random.Random(seed + 1)
    paras = [paraphrase(sents[i], events[i], indexer, prng) for i in train_ids]

    board = {"collection": len(texts), "decoys": len(decoy_of), "queries": len(queries), "held_out_docs": len(held), "train_pairs": len(train_ids),
             "dim": dim, "epochs": epochs, "live_queries": dict(live_stats), "systems": {}, "sense": {}}
    n = len(queries)

    counters = defaultdict(Counter)

    def tally(name, qi, sc):
        si = queries[qi][0]
        target, decoy = si, decoy_of[si]
        c = counters[name]
        rank = int((sc > sc[target]).sum()) + int((sc[:target] == sc[target]).sum())   # = stable argsort rank, as in bench_retrieval
        c["r1"] += int(rank == 0); c["r10"] += int(rank < 10)
        c["decoy_above"] += int(sc[decoy] > sc[target]); c["tie"] += int(sc[decoy] == sc[target])

    def close(name):
        c = counters[name]
        board["systems"][name] = {"recall@1": round(c["r1"] / n, 4), "recall@10": round(c["r10"] / n, 4),
                                  "decoy_above_source": round(float(c["decoy_above"]) / n, 4), "decoy_tied": round(float(c["tie"]) / n, 4),
                                  "source_above_decoy": round(1 - float(c["decoy_above"] + c["tie"]) / n, 4)}
        log(f"  {name:32} {board['systems'][name]}")

    for qi in range(n):
        tally("bm25", qi, bm_scores[qi])
    close("bm25")
    bm_max = [max(float(sc.max()), 1e-6) for sc in bm_scores]
    if key_sets is not None:
        for qi in range(n):
            sc = bm_scores[qi] / bm_max[qi]; sc[key_sets[qi]] += key_bonus
            tally("hybrid bm25+key", qi, sc)
        close("hybrid bm25+key")

    for variant in variants:
        parts = variant.split("/")
        use_slots, use_syn, trained = parts[0] == "roles", parts[1] == "synsets", "untrained" not in parts
        ix = indexer if use_syn else word_indexer
        if "hard" in parts:
            ix = hard_indexer
        elif "first" in parts:
            ix = first_indexer
        log(f"=== variant {variant}")
        corp = Corpus()
        for s, ev in zip(sents, events):
            corp.add(featurize(s, slot_labels(s, ev, use_slots), ix), 0)
        for si, e, syn in queries:                       # decoys: same words, roles exchanged
            s = sents[si]
            corp.add(featurize(s, slot_labels(s, events[si], use_slots), ix,
                               swap=((e.actor.start, e.actor.end), (e.patient.start, e.patient.end)) if use_slots else None), 0)
        corp.finish()
        qcorp = Corpus()
        for qs, qev in zip(q_sents, q_events):
            qcorp.add(featurize(qs, slot_labels(qs, qev, use_slots), ix), 0)
        qcorp.finish()
        model = SoftMap(ix.init_codes(dim=dim, seed=seed), n_slots=4 if use_slots else 1)
        if trained:
            qc, pc = Corpus(), Corpus()
            for i, (lem, up, keep) in zip(train_ids, paras):
                lab = slot_labels(sents[i], events[i], use_slots)
                qc.add(featurize(_Lite(lem, up), [lab[k] for k in keep], ix), 0)
                pc.add(featurize(sents[i], lab, ix), 0)
            qc.finish(); pc.finish()
            t = time.time()
            train(model, qc, pc, epochs=epochs, seed=seed)
            board.setdefault("train_seconds", {})[variant] = round(time.time() - t, 1)
        if use_syn:
            board["sense"][variant] = sense_report(model, ix)
            log(f"  sense: siblings {board['sense'][variant]['sibling_cosine_mean']} vs random {board['sense'][variant]['random_cosine_mean']}")
        P = model.encode(corp)
        Q = model.encode(qcorp)
        ex = []
        for s0 in range(0, n, 100):
            sims = Q[s0:s0 + 100] @ P.T
            for j in range(sims.shape[0]):
                qi = s0 + j
                tally(f"map {variant}", qi, sims[j])
                tally(f"hybrid bm25+map {variant}", qi, bm_scores[qi] / bm_max[qi] + sims[j])
                if key_sets is not None:
                    sc3 = bm_scores[qi] / bm_max[qi] + sims[j]; sc3[key_sets[qi]] += key_bonus
                    tally(f"three-way bm25+map+key {variant}", qi, sc3)
                if qi < 5:
                    si, top = queries[qi][0], int(np.argmax(sims[j]))
                    ex.append({"query": q_texts[qi][:120], "source": texts[si][:120], "map_top1": texts[top][:120], "hit": top == si})
        close(f"map {variant}"); close(f"hybrid bm25+map {variant}")
        if key_sets is not None:
            close(f"three-way bm25+map+key {variant}")
        board.setdefault("examples", {})[variant] = ex
        if variant == variants[0]:
            model.save(str(Path(out_dir) / "softmap.npz"))
    board["seconds"] = round(time.time() - t0, 1)
    (Path(out_dir) / "softmap_retrieval.json").write_text(json.dumps(board, indent=2))
    log(f"done in {board['seconds']}s -> {out_dir}/softmap_retrieval.json")
    return board


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--amalgum", required=True)
    ap.add_argument("--semcor", default=None)
    ap.add_argument("--frontend", required=True)
    ap.add_argument("--linker", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--train_pairs", type=int, default=120000)
    ap.add_argument("--variants", default="roles/synsets,roles/words,bow/synsets,roles/synsets/untrained")
    ap.add_argument("--wordnet", default=None)
    ap.add_argument("--key_bonus", type=float, default=0.0, help="add the core-v1 key index as a third signal (bonus on BM25/max scale)")
    a = ap.parse_args(argv)
    run(a.amalgum, a.semcor, a.frontend, a.linker, a.out, a.n, dim=a.dim, epochs=a.epochs, train_pairs=a.train_pairs,
        variants=tuple(a.variants.split(",")), wordnet_path=a.wordnet, key_bonus=a.key_bonus)


if __name__ == "__main__":
    main()
