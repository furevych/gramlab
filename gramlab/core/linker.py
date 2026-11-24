"""core-v1 · layer 1 — the role linker: Sentence -> [Event].

Contract: `link(sentence) -> list[Event]`.  Three occupants:

* GoldLinker    — reads UD relations (nsubj/obj, nsubj:pass/obl:agent|obl+by). Used for training
                  data and as the benchmark's gold; it is not a model.
* RuleLinker    — word order, the baseline (nearest nominal before/after; passive by 'be'+participle).
* LearnedLinker — hashed-feature logistic regression over (predicate, candidate) pairs, trained on
                  any UD-parsed text (AMALGUM silver by default). Scored on GUM test by bench/links.

Status (asserted / negated / possible) is read from the predicate's own dependents, the same way for
every occupant, so the benchmark isolates role recovery.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier

from .records import Event, Sentence, NOMINAL

WINDOW = 12
N_FEATS = 1 << 20
NEG = {"not", "never", "n't", "no"}
MODAL = {"may", "might", "could"}


def predicates(s: Sentence):
    """Verbal predicates with a subject or object, with voice and the gold role indices."""
    out = []
    for v in range(len(s)):
        if s.upos[v] not in ("VERB", "AUX"):
            continue
        kids = s.children.get(v, ())
        rels = {}
        for c in kids:
            rels.setdefault(s.deprels[c], c)
        if "nsubj:pass" in rels:
            actor = rels.get("obl:agent")
            if actor is None:
                for c in kids:
                    if s.deprels[c] == "obl" and "by" in s.case_lemmas(c):
                        actor = c
                        break
            out.append((v, "passive", actor, rels["nsubj:pass"]))
        elif "nsubj" in rels or "obj" in rels:
            out.append((v, "active", rels.get("nsubj"), rels.get("obj")))
    return out


def status_of(s: Sentence, v: int) -> str:
    for c in s.children.get(v, ()):
        if s.deprels[c] in ("advmod", "neg") and s.lemmas[c].lower() in NEG:
            return "negated"
    for c in s.children.get(v, ()):
        if s.deprels[c] == "aux" and s.lemmas[c].lower() in MODAL:
            return "possible"
    return "asserted"


def make_event(s: Sentence, v: int, voice: str, actor, patient) -> Event:
    return Event(verb=v, action=s.lemmas[v].lower(),
                 actor=s.mention(actor) if actor is not None else None,
                 patient=s.mention(patient) if patient is not None else None,
                 voice=voice, status=status_of(s, v))


class GoldLinker:
    name = "gold (UD relations)"

    def link(self, s: Sentence):
        return [make_event(s, v, voice, a, p) for v, voice, a, p in predicates(s)]


# ----------------------------------------------------------------------------- rules
STOP = set("the a an of to in on at by for with and or but is are was were be been being not no this that these "
           "those it its as from than then so if when while which who whom whose what where how very also just".split())


class RuleLinker:
    """Nearest nominal before = actor, after = patient; passive ('be' within 3 tokens + VBN) flips
    and takes the nominal after 'by' as actor. words_only=True uses no tags at all."""
    name = "word-order rules"
    BE = {"is", "are", "was", "were", "be", "been", "being", "am", "'s", "'re"}

    def __init__(self, words_only=False):
        self.words_only = words_only

    def roles(self, s: Sentence, v: int):
        f = [w.lower() for w in s.forms]
        pre = f[max(0, v - 3):v]
        if self.words_only:
            passive = any(w in self.BE for w in pre) and (f[v].endswith("ed") or f[v].endswith("en"))
            nominal = lambda i: s.forms[i][:1].isalnum() and f[i] not in STOP
        else:
            passive = any(w in self.BE for w in pre) and s.xpos[v] == "VBN"
            nominal = lambda i: s.upos[i] in NOMINAL
        before = next((i for i in range(v - 1, max(-1, v - WINDOW - 1), -1) if nominal(i)), None)
        after = next((i for i in range(v + 1, min(len(f), v + WINDOW + 1)) if nominal(i)), None)
        if passive:
            agent = None
            for i in range(v + 1, min(len(f), v + WINDOW + 1)):
                if f[i] == "by":
                    agent = next((j for j in range(i + 1, min(len(f), i + 5)) if nominal(j)), None)
                    break
            return agent, before, "passive"
        return before, after, "active"

    def link(self, s: Sentence):
        out = []
        for v, _, _, _ in predicates(s):          # which tokens are predicates comes from the parse
            a, p, voice = self.roles(s, v)
            out.append(make_event(s, v, voice, a, p))
        return out


# ----------------------------------------------------------------------------- learned
def _shape(w):
    if w.isdigit():
        return "d"
    if w[:1].isupper():
        return "Cap" if w[1:].islower() else "CAP"
    return "low" if w.islower() else "mix"


def candidates(s: Sentence, v: int, words_only: bool):
    lo, hi = max(0, v - WINDOW), min(len(s), v + WINDOW + 1)
    if words_only:
        return [i for i in range(lo, hi) if i != v and s.forms[i][:1].isalnum()]
    return [i for i in range(lo, hi) if i != v and s.upos[i] in NOMINAL]


def pair_features(s: Sentence, v: int, c: int, words_only: bool):
    f = s.forms
    d = c - v
    side = "L" if d < 0 else "R"
    dist = min(abs(d), 6)
    between = f[min(v, c) + 1:max(v, c)]
    bl = [w.lower() for w in between]
    feats = [f"side={side}", f"dist={side}{dist}", f"cand={f[c].lower()}", f"verb={s.lemmas[v].lower()}",
             f"cshape={_shape(f[c])}", f"vform={f[v].lower()[-3:]}", f"nb={len(between) if len(between) < 6 else 6}",
             f"prev={f[c-1].lower() if c else '<s>'}", f"next={f[c+1].lower() if c + 1 < len(f) else '</s>'}",
             f"vprev={f[v-1].lower() if v else '<s>'}", f"vprev2={f[v-2].lower() if v > 1 else '<s>'}",
             f"by_between={'by' in bl}", f"side_by={side}:{'by' in bl}",
             f"verb_side={s.lemmas[v].lower()}:{side}", f"cand_side={f[c].lower()}:{side}",
             f"vsuf={s.lemmas[v].lower()[-2:]}:{side}", f"csuf={f[c].lower()[-2:]}"]
    for w in set(bl):
        feats.append(f"btw={w}")
    if not words_only:
        feats += [f"cpos={s.upos[c]}", f"cxpos={s.xpos[c]}", f"vxpos={s.xpos[v]}",
                  f"cpos_side={s.upos[c]}:{side}", f"vxpos_side={s.xpos[v]}:{side}",
                  f"vxpos_cpos_side={s.xpos[v]}:{s.upos[c]}:{side}",
                  f"vxpos_by={s.xpos[v]}:{'by' in bl}:{side}",
                  f"prevpos={s.upos[c-1] if c else '<s>'}",
                  f"vprevpos={s.upos[v-1] if v else '<s>'}:{s.xpos[v]}"]
        for w in set(s.upos[min(v, c) + 1:max(v, c)]):
            feats.append(f"btwpos={w}")
        feats.append(f"nominal_between={any(p in NOMINAL for p in s.upos[min(v, c) + 1:max(v, c)])}:{side}")
    return feats


_hcache = {}


def _hid(feat):
    x = _hcache.get(feat)
    if x is None:
        import hashlib
        x = int(hashlib.blake2b(feat.encode(), digest_size=4).hexdigest(), 16) % N_FEATS
        if len(_hcache) < 3_000_000:
            _hcache[feat] = x
    return x


def build_pairs(sents, words_only):
    rows, cols, y, idx = [], [], [], []
    r = 0
    for si, s in enumerate(sents):
        for v, voice, a, p in predicates(s):
            for c in candidates(s, v, words_only):
                lab = 1 if c == a else 2 if c == p else 0
                for ft in pair_features(s, v, c, words_only):
                    rows.append(r)
                    cols.append(_hid(ft))
                y.append(lab)
                idx.append((si, v, c))
                r += 1
    X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(r, N_FEATS))
    X.sum_duplicates()
    return X, np.array(y), idx


class LearnedLinker:
    name = "learned linker (hashed logistic)"

    def __init__(self, words_only=False, epochs=6, alpha=2e-6, threshold=0.3, seed=0):
        self.words_only = words_only
        self.threshold = threshold
        self.clf = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=epochs, tol=None, random_state=seed)
        self.n_pairs = 0
        self.nnz = 0

    def fit(self, sents):
        X, y, _ = build_pairs(sents, self.words_only)
        self.n_pairs = X.shape[0]
        self.clf.fit(X, y)
        self.nnz = int((np.abs(self.clf.coef_) > 1e-9).sum())
        return self

    def roles_batch(self, sents, threshold=None, return_scores=False):
        """(si, v) -> (actor idx, patient idx) for every predicate in sents.
        With return_scores: (si, v) -> ((actor, p_actor), (patient, p_patient)) — the best candidate per
        role regardless of threshold, so a composite linker can apply its own policy."""
        th = self.threshold if threshold is None else threshold
        X, _, idx = build_pairs(sents, self.words_only)
        P = self.clf.predict_proba(X) if X.shape[0] else np.zeros((0, 3))
        best = {}
        floor = 0.0 if return_scores else th
        for (si, v, c), p in zip(idx, P):
            a, pa = best.get((si, v), ((None, floor), (None, floor)))
            if p[1] > a[1]:
                a = (c, p[1])
            if p[2] > pa[1]:
                pa = (c, p[2])
            best[(si, v)] = (a, pa)
        if return_scores:
            return best
        out = {}
        for k, (a, pa) in best.items():
            if a[0] is not None and a[0] == pa[0]:
                out[k] = (a[0], None) if a[1] >= pa[1] else (None, pa[0])
            else:
                out[k] = (a[0], pa[0])
        return out

    def link_batch(self, sents):
        roles = self.roles_batch(sents)
        out = []
        for si, s in enumerate(sents):
            evs = []
            for v, voice, _, _ in predicates(s):
                a, p = roles.get((si, v), (None, None))
                evs.append(make_event(s, v, voice, a, p))
            out.append(evs)
        return out

    def link(self, s: Sentence):
        return self.link_batch([s])[0]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"words_only": self.words_only, "threshold": self.threshold, "clf": self.clf,
                         "n_pairs": self.n_pairs, "nnz": self.nnz}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls(d["words_only"], threshold=d["threshold"])
        m.clf, m.n_pairs, m.nnz = d["clf"], d["n_pairs"], d["nnz"]
        return m


class BackfillLinker:
    """Composite occupant: the learned linker's roles at its threshold; when exactly one role is
    asserted with confidence ≥ `confident`, the other role is filled from the model's own best
    candidate if its probability ≥ `second` (a lower bar, justified by the first role: a verb with a
    confident patient usually has an actor). Never assigns one token to both roles."""
    name = "learned + soft backfill"

    def __init__(self, learned: LearnedLinker, confident=0.6, second=0.1):
        self.learned = learned
        self.confident = confident
        self.second = second

    def link_batch(self, sents):
        th = self.learned.threshold
        scores = self.learned.roles_batch(sents, return_scores=True)
        out = []
        for si, s in enumerate(sents):
            evs = []
            for v, voice, _, _ in predicates(s):
                (a, pa), (p, pp) = scores.get((si, v), ((None, 0), (None, 0)))
                A = a if pa >= th else None
                Pt = p if pp >= th else None
                if A is not None and Pt is None and pa >= self.confident and pp >= self.second and p != A:
                    Pt = p
                elif Pt is not None and A is None and pp >= self.confident and pa >= self.second and a != Pt:
                    A = a
                if A is not None and A == Pt:
                    A, Pt = (A, None) if pa >= pp else (None, Pt)
                evs.append(make_event(s, v, voice, A, Pt))
            out.append(evs)
        return out

    def link(self, s: Sentence):
        return self.link_batch([s])[0]
