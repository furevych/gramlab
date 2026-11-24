"""core-v1 · layer 1 — sense picker: (lemma, pos, context) -> synset.

Layer 0 looks words up; it cannot tell which sense is meant, and run 9b showed that first-sense families
merge the wrong verbs. This is the component that chooses. Occupants:

* FirstSense   — WordNet's sense order (the fixed baseline; ~60% on verbs in the literature)
* LeskPicker   — overlap between the sentence's content lemmas and each candidate's gloss + synonyms +
                 hypernym names; ties go to the first sense. No training.
* NaiveBayesPicker — per-lemma Naive Bayes over context lemmas (±3) and neighbour POS, trained on
                 sense-tagged text (SemCor brown1 + brownv); unseen lemmas fall back to FirstSense.

Benchmark: SemCor brown2 (held out), accuracy on verbs and nouns, only tokens whose gold sense key
resolves in WordNet 3.0 and whose lemma has ≥2 senses (the rest is trivial).
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from .wordnet import WordNet

STOP = set("the a an of to in on at by for with and or but is are was were be been being not no this that these those "
           "it its as from than then so if when while which who whom whose what where how very also just have has had do "
           "does did will would can could may might shall should must i you he she we they me him her us them my your his "
           "our their there here out up down over into about after before between through during under again more most "
           "other some such only own same too s t".split())
_tok = re.compile(r"[a-z]+")

SEMCOR_POS = {"NN": "n", "NNS": "n", "NNP": "n", "NNPS": "n", "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v", "VBP": "v",
              "VBZ": "v", "JJ": "a", "JJR": "a", "JJS": "a", "RB": "r", "RBR": "r", "RBS": "r"}


def read_semcor(semcor_dir, wn: WordNet, parts=("brown1", "brownv")):
    """Sentences as lists of dicts(word, lemma, pos(wn), gold offset or None)."""
    sents = []
    for part in parts:
        for f in sorted(Path(semcor_dir, part, "tagfiles").glob("*.xml")):
            txt = f.read_text(encoding="utf-8", errors="ignore")
            txt = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)", "&amp;", txt)
            try:
                root = ET.fromstring(txt)
            except ET.ParseError:
                continue
            for s in root.iter("s"):
                toks = []
                for el in s:
                    if el.tag not in ("wf", "punc"):
                        continue
                    word = (el.text or "").strip()
                    lemma = el.get("lemma", word.lower())
                    pos = SEMCOR_POS.get(el.get("pos", ""), None)
                    gold = None
                    lexsn = el.get("lexsn")
                    if lexsn and lemma and pos and ";" not in lexsn:
                        gold = wn.offset_of_sensekey(f"{lemma}%{lexsn}")
                    toks.append({"word": word, "lemma": lemma.lower(), "pos": pos, "gold": gold, "tag": el.get("pos", "")})
                if toks:
                    sents.append(toks)
    return sents


def content_lemmas(sent, skip=None):
    out = []
    for i, t in enumerate(sent):
        if i == skip:
            continue
        l = t["lemma"]
        if l and l not in STOP and _tok.fullmatch(l.replace("_", "")) and len(l) > 2:
            out.append(l)
    return out


class FirstSense:
    name = "first sense"

    def __init__(self, wn: WordNet):
        self.wn = wn

    def pick(self, lemma, pos, sent=None, i=None):
        ss = self.wn.synsets(lemma, pos)
        return ss[0] if ss else None


class LeskPicker:
    name = "extended Lesk"

    def __init__(self, wn: WordNet):
        self.wn = wn
        self._sig = {}

    def signature(self, pos, off):
        k = (pos, off)
        if k not in self._sig:
            words = set(_tok.findall(self.wn.gloss(pos, off).lower()))
            for w in self.wn.lemmas(pos, off):
                words.update(_tok.findall(w.lower()))
            for h in self.wn.hypernyms(pos, off):
                for w in self.wn.lemmas(pos, h):
                    words.update(_tok.findall(w.lower()))
                words.update(_tok.findall(self.wn.gloss(pos, h).lower()))
            self._sig[k] = {w for w in words if w not in STOP and len(w) > 2}
        return self._sig[k]

    def pick(self, lemma, pos, sent, i):
        ss = self.wn.synsets(lemma, pos)
        if not ss:
            return None
        ctx = set(content_lemmas(sent, skip=i))
        best, best_score = ss[0], -1
        for rank, off in enumerate(ss):
            score = len(ctx & self.signature(pos, off)) - rank * 1e-3
            if score > best_score:
                best, best_score = off, score
        return best


class NaiveBayesPicker:
    name = "per-lemma naive Bayes (SemCor)"

    def __init__(self, wn: WordNet, window=3, alpha=0.5):
        self.wn = wn
        self.window = window
        self.alpha = alpha
        self.first = FirstSense(wn)
        self.prior = defaultdict(Counter)        # (lemma,pos) -> Counter(sense)
        self.feat = defaultdict(Counter)         # (lemma,pos,sense) -> Counter(feature)
        self.n_feat = defaultdict(int)
        self.vocab = defaultdict(set)

    def features(self, sent, i):
        f = []
        for d in range(-self.window, self.window + 1):
            j = i + d
            if d == 0 or j < 0 or j >= len(sent):
                continue
            l = sent[j]["lemma"]
            if l and l not in STOP:
                f.append("w=" + l)
            if abs(d) == 1:
                f.append(f"t{d}=" + sent[j]["tag"][:2])
                f.append(f"l{d}=" + l)
        return f

    def fit(self, sents):
        for s in sents:
            for i, t in enumerate(s):
                if t["gold"] is None or t["pos"] is None:
                    continue
                k = (t["lemma"], t["pos"])
                self.prior[k][t["gold"]] += 1
                fs = self.features(s, i)
                self.feat[k + (t["gold"],)].update(fs)
                self.n_feat[k + (t["gold"],)] += len(fs)
                self.vocab[k].update(fs)
        return self

    def pick(self, lemma, pos, sent, i):
        k = (lemma, pos)
        ss = self.wn.synsets(lemma, pos)
        if not ss:
            return None
        if k not in self.prior:
            return ss[0]
        fs = self.features(sent, i)
        V = len(self.vocab[k]) + 1
        total = sum(self.prior[k].values())
        best, best_lp = None, -math.inf
        for sense in ss:
            c = self.prior[k].get(sense, 0)
            lp = math.log((c + self.alpha) / (total + self.alpha * len(ss)))
            fc = self.feat.get(k + (sense,), Counter())
            n = self.n_feat.get(k + (sense,), 0)
            for f in fs:
                lp += math.log((fc.get(f, 0) + self.alpha) / (n + self.alpha * V))
            if lp > best_lp:
                best, best_lp = sense, lp
        return best

    def predominant(self, lemma, pos):
        """Most frequent trained sense for a lemma (a corpus-adapted first sense), else first sense."""
        k = (lemma, pos)
        if k in self.prior:
            return self.prior[k].most_common(1)[0][0]
        ss = self.wn.synsets(lemma, pos)
        return ss[0] if ss else None


def bench_sense(pickers: dict, sents, wn: WordNet, poses=("v", "n")):
    out = {}
    for name, pk in pickers.items():
        res = {}
        for pos in poses:
            n = ok = 0
            for s in sents:
                for i, t in enumerate(s):
                    if t["gold"] is None or t["pos"] != pos:
                        continue
                    ss = wn.synsets(t["lemma"], pos)
                    if len(ss) < 2 or t["gold"] not in ss:
                        continue
                    n += 1
                    ok += pk.pick(t["lemma"], pos, s, i) == t["gold"]
            res[pos] = {"n": n, "accuracy": round(ok / max(n, 1), 4)}
        out[name] = res
    return out
