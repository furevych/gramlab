"""core-v1 · layer 1 — the live-text front end: tokens -> UPOS, XPOS, lemma, without a dependency parser.

GUM and AMALGUM arrive parsed; live text does not. The minimal occupant of the parser slot is a tagger plus a
lemma table, trained on AMALGUM's silver tags, feeding the role linker (which never used dependency labels as
features) with predicates = tagged verbs and voice from a surface rule. Benchmark (`bench.bench_live`): GUM test
re-tagged by this component, link F1 against gold vs the gold-parse pipeline.

* PerceptronTagger — averaged perceptron, joint label "UPOS|XPOS" (≈ 50 labels), standard features
  (word, lowercase, suffixes, shape, previous two labels, neighbouring words).
* Lemmatizer      — (lowercase form, XPOS) -> most frequent lemma from the training data, then suffix rules.
* tag_sentence    — builds a core Sentence (heads/deprels empty) from a token list.
"""
from __future__ import annotations

import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .records import Sentence

_num = re.compile(r"^\d[\d,.]*$")


def _shape(w):
    if _num.match(w):
        return "NUM"
    if w.isupper() and len(w) > 1:
        return "UPPER"
    if w[:1].isupper():
        return "Cap"
    if w.isalpha():
        return "low"
    if "-" in w:
        return "hyph"
    return "other"


class PerceptronTagger:
    name = "averaged perceptron (AMALGUM silver)"

    def __init__(self):
        self.weights = {}         # feature -> {label: weight}
        self.labels = set()
        self._totals = defaultdict(float)
        self._stamps = defaultdict(int)
        self._i = 0
        self.tagdict = {}         # unambiguous frequent words

    @staticmethod
    def features(words, i, prev, prev2):
        w = words[i]
        lw = w.lower()
        f = ["bias", "w=" + lw, "suf3=" + lw[-3:], "suf2=" + lw[-2:], "suf1=" + lw[-1:], "pre1=" + lw[:1],
             "shape=" + _shape(w), "p1=" + prev, "p2=" + prev2, "p1w=" + prev + ":" + lw,
             "w-1=" + (words[i - 1].lower() if i else "<s>"), "w+1=" + (words[i + 1].lower() if i + 1 < len(words) else "</s>"),
             "w-1suf=" + (words[i - 1].lower()[-3:] if i else "<s>"), "w+1suf=" + (words[i + 1].lower()[-3:] if i + 1 < len(words) else "</s>"),
             "w-2=" + (words[i - 2].lower() if i > 1 else "<s>"), "w+2=" + (words[i + 2].lower() if i + 2 < len(words) else "</s>"),
             "suf3p1=" + lw[-3:] + ":" + prev]
        if i == 0:
            f.append("first")
        return f

    def _predict(self, feats):
        scores = defaultdict(float)
        for ft in feats:
            ws = self.weights.get(ft)
            if ws:
                for lab, w in ws.items():
                    scores[lab] += w
        return max(self.labels, key=lambda l: (scores[l], l)) if scores else next(iter(self.labels))

    def _update(self, truth, guess, feats):
        self._i += 1
        if truth == guess:
            return
        for ft in feats:
            ws = self.weights.setdefault(ft, {})
            for lab, d in ((truth, 1.0), (guess, -1.0)):
                key = (ft, lab)
                self._totals[key] += (self._i - self._stamps[key]) * ws.get(lab, 0.0)
                self._stamps[key] = self._i
                ws[lab] = ws.get(lab, 0.0) + d

    def fit(self, sents, epochs=5, seed=0, tagdict_min=20, tagdict_purity=0.97):
        data = [(s.forms, [f"{u}|{x}" for u, x in zip(s.upos, s.xpos)]) for s in sents]
        counts = defaultdict(Counter)
        for words, labs in data:
            for w, l in zip(words, labs):
                counts[w.lower()][l] += 1
                self.labels.add(l)
        for w, c in counts.items():
            lab, n = c.most_common(1)[0]
            if sum(c.values()) >= tagdict_min and n / sum(c.values()) >= tagdict_purity:
                self.tagdict[w] = lab
        rng = random.Random(seed)
        for ep in range(epochs):
            rng.shuffle(data)
            for words, labs in data:
                prev, prev2 = "<s>", "<s>"
                for i, (w, truth) in enumerate(zip(words, labs)):
                    guess = self.tagdict.get(w.lower())
                    if guess is None:
                        feats = self.features(words, i, prev, prev2)
                        guess = self._predict(feats)
                        self._update(truth, guess, feats)
                    prev2, prev = prev, guess
        # average
        for ft, ws in self.weights.items():
            for lab in list(ws):
                key = (ft, lab)
                total = self._totals[key] + (self._i - self._stamps[key]) * ws[lab]
                ws[lab] = round(total / self._i, 4)
        self._totals, self._stamps = None, None
        return self

    def tag(self, words):
        out = []
        prev, prev2 = "<s>", "<s>"
        for i, w in enumerate(words):
            lab = self.tagdict.get(w.lower()) or self._predict(self.features(words, i, prev, prev2))
            out.append(lab)
            prev2, prev = prev, lab
        return [tuple(l.split("|", 1)) for l in out]


class Lemmatizer:
    name = "lemma table + suffix rules"

    def __init__(self):
        self.table = {}
        self.known = set()          # every lemma seen in training: the suffix rules pick a candidate that exists

    def fit(self, sents, min_count=1):
        c = defaultdict(Counter)
        for s in sents:
            for f, l, x in zip(s.forms, s.lemmas, s.xpos):
                c[(f.lower(), x)][l.lower()] += 1
                self.known.add(l.lower())
        self.table = {k: v.most_common(1)[0][0] for k, v in c.items() if sum(v.values()) >= min_count}
        return self

    def _pick(self, cands, default):
        for cnd in cands:
            if cnd in self.known:
                return cnd
        return default

    def lemma(self, form, xpos):
        lw = form.lower()
        hit = self.table.get((lw, xpos))
        if hit:
            return hit
        if xpos in ("NNS", "NNPS", "VBZ") and lw.endswith("s") and len(lw) > 3:
            if lw.endswith("ies"):
                return self._pick([lw[:-3] + "y", lw[:-1]], lw[:-3] + "y")
            if lw.endswith(("sses", "shes", "ches", "xes", "zes")):
                return self._pick([lw[:-2], lw[:-1]], lw[:-2])
            return self._pick([lw[:-1], lw[:-2]], lw[:-1])
        if xpos in ("VBD", "VBN") and lw.endswith("ed") and len(lw) > 4:
            if lw.endswith("ied"):
                return self._pick([lw[:-3] + "y", lw[:-1]], lw[:-3] + "y")
            stem = lw[:-2]
            cands = [stem, stem + "e"]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                cands.append(stem[:-1])                    # stopped → stop
            return self._pick(cands, stem)
        if xpos == "VBG" and lw.endswith("ing") and len(lw) > 5:
            stem = lw[:-3]
            cands = [stem, stem + "e"]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                cands.append(stem[:-1])
            return self._pick(cands, stem)
        return lw


class LiveFrontEnd:
    """tokens -> core Sentence with predicted UPOS/XPOS/lemma; no heads, no deprels, no entities."""
    name = "perceptron tagger + lemma table (no parser)"

    def __init__(self, tagger: PerceptronTagger, lemmatizer: Lemmatizer, typer=None):
        self.tagger = tagger
        self.lemmatizer = lemmatizer
        self.typer = typer                # core.typer.EntityTyper: fills entity types on live text

    def sentence(self, words, sid="", doc=""):
        tags = self.tagger.tag(words)
        upos = [u for u, x in tags]
        xpos = [x for u, x in tags]
        lemmas = [self.lemmatizer.lemma(w, x) for w, x in zip(words, xpos)]
        n = len(words)
        s = Sentence(sid, list(words), lemmas, upos, xpos, [-1] * n, ["_"] * n, [None] * n, doc,
                     ents_raw=[""] * n, feats=["_"] * n)
        if self.typer is not None:
            self.typer.annotate(s)
        return s

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"tagger": self.tagger, "lemmatizer": self.lemmatizer}, f)

    @classmethod
    def load(cls, path, typer=None):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["tagger"], d["lemmatizer"], typer)


_TOK = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d[\d,.]*|[^\sA-Za-z\d]")


def tokenize(text):
    """Plain regex tokenizer for live text (GUM-style tokens: punctuation split, contractions kept together
    except n't/'s which the tagger has seen as separate tokens)."""
    out = []
    for m in _TOK.finditer(text):
        w = m.group(0)
        if w.lower().endswith("n't") and len(w) > 3:
            out += [w[:-3], w[-3:]]
        elif w.lower().endswith("'s") and len(w) > 2:
            out += [w[:-2], w[-2:]]
        else:
            out.append(w)
    return out
