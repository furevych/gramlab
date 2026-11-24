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
            items, golds = [], []
            for s in sents:
                for i, t in enumerate(s):
                    if t["gold"] is None or t["pos"] != pos:
                        continue
                    ss = wn.synsets(t["lemma"], pos)
                    if len(ss) < 2 or t["gold"] not in ss:
                        continue
                    items.append((t["lemma"], pos, s, i))
                    golds.append(t["gold"])
            if hasattr(pk, "pick_batch"):
                preds = pk.pick_batch(items)
            else:
                preds = [pk.pick(*it) for it in items]
            ok = sum(p == g for p, g in zip(preds, golds))
            res[pos] = {"n": len(items), "accuracy": round(ok / max(len(items), 1), 4)}
        out[name] = res
    return out


# ----------------------------------------------------------------------------- shared rank classifier
class RankPicker:
    """One model per POS, shared across lemmas: features conjoined with the lemma predict the *rank* of the
    sense in WordNet order (0 = first sense), so the model starts from the first-sense prior and corrects it
    from context. Hashed logistic regression (SGD). Lemmas unseen in training fall back to rank 0.
    alpha is chosen on a dev split (brownv), never on the test split."""
    name = "shared rank classifier (SemCor)"
    N = 1 << 21
    MAXR = 8

    def __init__(self, wn: WordNet, poses=("v",), alphas=(2e-5, 5e-5, 1e-4, 3e-4), epochs=10, seed=0):
        self.wn = wn
        self.poses = poses
        self.alphas = alphas
        self.epochs = epochs
        self.seed = seed
        self.clf = {}
        self.alpha = {}
        self.seen = {}
        self._cache = {}

    def _hid(self, f):
        x = self._cache.get(f)
        if x is None:
            import hashlib
            x = int(hashlib.blake2b(f.encode(), digest_size=4).hexdigest(), 16) % self.N
            if len(self._cache) < 4_000_000:
                self._cache[f] = x
        return x

    def features(self, sent, i, lemma, pos):
        L = lemma + "/" + pos
        out = [f"{L}|bias"]
        for d in range(-5, 6):
            j = i + d
            if d == 0 or j < 0 or j >= len(sent):
                continue
            l = sent[j]["lemma"]
            t = sent[j].get("tag", "")[:2]
            if l and l not in STOP:
                out.append(f"{L}|w={l}")
                out.append(f"{pos}|w={l}")
            if abs(d) <= 2:
                out.append(f"{L}|w{d}={l}")
                out.append(f"{L}|t{d}={t}")
        return out

    def _build(self, sents, pos, train_mode):
        import numpy as np
        from scipy import sparse
        rows, cols, y, meta = [], [], [], []
        r = 0
        for s in sents:
            for i, t in enumerate(s):
                if t["gold"] is None or t["pos"] != pos:
                    continue
                ss = self.wn.synsets(t["lemma"], pos)
                if len(ss) < 2 or t["gold"] not in ss:
                    continue
                rank = ss.index(t["gold"])
                if train_mode and rank >= self.MAXR:
                    continue
                for f in self.features(s, i, t["lemma"], pos):
                    rows.append(r)
                    cols.append(self._hid(f))
                y.append(min(rank, self.MAXR - 1))
                meta.append((t["lemma"], len(ss), rank))
                r += 1
        X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(r, self.N))
        X.sum_duplicates()
        return X, np.array(y), meta

    def _acc(self, clf, X, meta, seen):
        import numpy as np
        P = clf.predict_proba(X)
        ok = 0
        for k, (lemma, nss, gold) in enumerate(meta):
            pred = 0 if seen.get(lemma, 0) < 1 else int(np.argmax(P[k][:min(nss, self.MAXR)]))
            ok += pred == gold
        return ok / max(len(meta), 1)

    def fit(self, train, dev=None):
        from sklearn.linear_model import SGDClassifier
        for pos in self.poses:
            self.seen[pos] = Counter(t["lemma"] for s in train for t in s if t["gold"] and t["pos"] == pos)
            Xtr, ytr, _ = self._build(train, pos, True)
            best = None
            dev_pos = dev if dev is not None and any(t["gold"] and t["pos"] == pos for s_ in dev for t in s_) else None
            if dev_pos is None and len(self.alphas) > 1:
                cut = int(len(train) * 0.9)
                train, dev_pos = train[:cut], train[cut:]          # held-out slice of train for this POS
            if dev_pos is not None and len(self.alphas) > 1:
                dev = dev_pos
                Xtr, ytr, _ = self._build(train, pos, True)
                self.seen[pos] = Counter(t["lemma"] for s_ in train for t in s_ if t["gold"] and t["pos"] == pos)
                Xdv, _, mdv = self._build(dev, pos, False)
                for a in self.alphas:
                    clf = SGDClassifier(loss="log_loss", alpha=a, max_iter=self.epochs, tol=None, random_state=self.seed).fit(Xtr, ytr)
                    acc = self._acc(clf, Xdv, mdv, self.seen[pos])
                    if best is None or acc > best[0]:
                        best = (acc, a)
                a = best[1]
                # refit on train + dev with the chosen alpha
                Xall, yall, _ = self._build(train + dev, pos, True)
                self.seen[pos] = Counter(t["lemma"] for s in train + dev for t in s if t["gold"] and t["pos"] == pos)
                self.clf[pos] = SGDClassifier(loss="log_loss", alpha=a, max_iter=self.epochs, tol=None, random_state=self.seed).fit(Xall, yall)
            else:
                a = self.alphas[0]
                self.clf[pos] = SGDClassifier(loss="log_loss", alpha=a, max_iter=self.epochs, tol=None, random_state=self.seed).fit(Xtr, ytr)
            self.alpha[pos] = a
        return self

    def pick_batch(self, items):
        """items: list of (lemma, pos, sent, i) -> list of synsets; one matrix per POS."""
        import numpy as np
        from scipy import sparse
        out = [None] * len(items)
        by_pos = defaultdict(list)
        for k, (lemma, pos, sent, i) in enumerate(items):
            ss = self.wn.synsets(lemma, pos)
            if not ss:
                continue
            if pos not in self.clf or self.seen[pos].get(lemma, 0) < 1 or len(ss) < 2:
                out[k] = ss[0]
            else:
                by_pos[pos].append((k, ss))
        for pos, rows_ in by_pos.items():
            rows, cols = [], []
            for r, (k, ss) in enumerate(rows_):
                lemma, _, sent, i = items[k]
                for f in self.features(sent, i, lemma, pos):
                    rows.append(r)
                    cols.append(self._hid(f))
            X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(rows_), self.N))
            P = self.clf[pos].predict_proba(X)
            for r, (k, ss) in enumerate(rows_):
                out[k] = ss[int(np.argmax(P[r][:min(len(ss), self.MAXR)]))]
        return out

    def pick(self, lemma, pos, sent, i):
        import numpy as np
        ss = self.wn.synsets(lemma, pos)
        if not ss:
            return None
        if pos not in self.clf or self.seen[pos].get(lemma, 0) < 1 or len(ss) < 2:
            return ss[0]
        from scipy import sparse
        cols = [self._hid(f) for f in self.features(sent, i, lemma, pos)]
        X = sparse.csr_matrix((np.ones(len(cols), dtype=np.float32), ([0] * len(cols), cols)), shape=(1, self.N))
        p = self.clf[pos].predict_proba(X)[0]
        return ss[int(np.argmax(p[:min(len(ss), self.MAXR)]))]
