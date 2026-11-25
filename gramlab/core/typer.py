"""core-v1 · layer 1 — entity typing: mention head -> entity type (GUM's ten) and kind (general / specific).

On parsed GUM/AMALGUM the type comes from the annotation; on live text nothing supplies it, so names were class
`unknown` (run 11). This component predicts it from the surface: hashed logistic regression over the head word,
its shape and suffixes, the mention's span words, the neighbouring words and tags, trained on GUM train+dev
(gold types), scored on GUM test event arguments — the row `bench.PriorTyper` set (59.3% general / 58.1% specific).
Kind is decided by the head's tag (PROPN → specific) and the type by the model.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier

from .records import ENTITY_TYPES, NOMINAL, Sentence

N_FEATS = 1 << 20
TYPES = list(ENTITY_TYPES)
T2I = {t: i for i, t in enumerate(TYPES)}


def _shape(w):
    if w.isdigit():
        return "d"
    if w.isupper() and len(w) > 1:
        return "CAP"
    if w[:1].isupper():
        return "Cap"
    return "low" if w.islower() else "mix"


class EntityTyper:
    name = "hashed logistic entity typer (GUM train)"

    def __init__(self, alpha=1e-5, epochs=8, seed=0, general=None):
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed
        self.clf = None
        self._cache = {}
        self.general = general          # layer-0 GeneralIndex: WordNet lexicographer classes as features
        self._lex = {}

    def lexnames(self, lemma):
        """Lexicographer file of the lemma's first three noun senses (noun.person, noun.artifact, …), cached."""
        if self.general is None or self.general._wn is None:
            return []
        k = lemma.lower()
        if k not in self._lex:
            ss = self.general.synsets(k, "NOUN")[:3]
            names = []
            for rank, x in enumerate(ss):
                names.append(f"lex{rank}={self.general.lexname(x)}")
                if rank == 0:
                    pos, off = x
                    for h in self.general._wn.hypernyms(pos, off)[:1]:
                        names.append(f"hyperlex={self.general._wn.lexname(pos, h)}")
            if not ss:
                names.append("lex=none")
            self._lex[k] = names
        return self._lex[k]

    def _hid(self, f):
        x = self._cache.get(f)
        if x is None:
            import hashlib
            x = int(hashlib.blake2b(f.encode(), digest_size=4).hexdigest(), 16) % N_FEATS
            if len(self._cache) < 3_000_000:
                self._cache[f] = x
        return x

    def features(self, s: Sentence, i: int):
        w = s.forms[i]
        lw = w.lower()
        lo, hi = s.span_of(i)
        f = [f"w={lw}", f"lemma={s.lemmas[i].lower()}", f"shape={_shape(w)}", f"suf3={lw[-3:]}", f"suf2={lw[-2:]}",
             f"pre3={lw[:3]}", f"upos={s.upos[i]}", f"xpos={s.xpos[i]}", f"len={min(hi - lo, 5)}",
             f"prev={s.forms[i-1].lower() if i else '<s>'}", f"next={s.forms[i+1].lower() if i + 1 < len(s) else '</s>'}",
             f"prevpos={s.upos[i-1] if i else '<s>'}", f"nextpos={s.upos[i+1] if i + 1 < len(s) else '</s>'}",
             f"first={lo == 0}", f"det={s.forms[lo].lower() if lo < i else '<none>'}"]
        f += self.lexnames(s.lemmas[i])
        for j in range(lo, hi):
            if j != i:
                f.append(f"span={s.forms[j].lower()}")
                f.append(f"spanshape={_shape(s.forms[j])}")
                if s.upos[j] in ("NOUN", "PROPN"):
                    f += ["span" + x for x in self.lexnames(s.lemmas[j])]
        if i >= 2:
            f.append(f"prev2={s.forms[i-2].lower()}")
        if i + 2 < len(s):
            f.append(f"next2={s.forms[i+2].lower()}")
        # a preposition governing the mention is informative (in <place>, by <person>, at <time>)
        for j in range(max(0, lo - 2), lo):
            if s.upos[j] == "ADP":
                f.append(f"adp={s.forms[j].lower()}")
        return f

    def _items(self, sents, labelled):
        items, y = [], []
        for si, s in enumerate(sents):
            for i in range(len(s)):
                if s.upos[i] not in ("NOUN", "PROPN"):
                    continue
                if labelled:
                    t = s.mention(i).etype
                    if t not in T2I:
                        continue
                    y.append(T2I[t])
                items.append((si, i))
        return items, y

    def _matrix(self, sents, items):
        rows, cols = [], []
        for r, (si, i) in enumerate(items):
            for f in self.features(sents[si], i):
                rows.append(r)
                cols.append(self._hid(f))
        X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(items), N_FEATS))
        X.sum_duplicates()
        return X

    def fit(self, sents):
        items, y = self._items(sents, True)
        X = self._matrix(sents, items)
        self.clf = SGDClassifier(loss="log_loss", alpha=self.alpha, max_iter=self.epochs, tol=None, random_state=self.seed).fit(X, np.array(y))
        self.n_train = len(items)
        return self

    def predict_batch(self, sents, items):
        if not items:
            return []
        X = self._matrix(sents, items)
        return [TYPES[k] for k in self.clf.predict(X)]

    def predict(self, s: Sentence, i: int):
        return self.predict_batch([s], [(0, i)])[0]

    def predict_kind(self, s: Sentence, i: int):
        return "specific" if s.upos[i] == "PROPN" else "general"

    def annotate(self, s: Sentence):
        """Fill s.ents for every nominal head from the model (live text)."""
        items = [(0, i) for i in range(len(s)) if s.upos[i] in ("NOUN", "PROPN")]
        for (_, i), t in zip(items, self.predict_batch([s], items)):
            s.ents[i] = t
        return s

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "alpha": self.alpha, "uses_wordnet": self.general is not None}, f)

    @classmethod
    def load(cls, path, general=None):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls(alpha=d["alpha"], general=general if d.get("uses_wordnet") else None)
        m.clf = d["clf"]
        return m
