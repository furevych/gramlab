"""core-v1 · layer 1 — coreference for key slots: which earlier mention does this pronoun refer to?

39% of GUM-test key arguments are pronouns. On live text they get class "pronoun" (and no id) until this
component resolves them. GUM's Entity= annotation carries gold chains (`(12-person-...` opens entity 12,
`12)` closes it), so the benchmark is fixed: for each pronoun that is an event argument, does the predicted
antecedent belong to the same gold chain?

Occupants:
* NearestNominal — the nearest preceding nominal head (sentence-internal, then previous sentences)
* TypedNearest   — nearest preceding nominal whose entity type and number agree with the pronoun's
                   (person for he/she/who, non-person for it; plural for they); types from annotation
                   now, from the entity typer later.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .records import NOMINAL, Sentence

_ITEM = re.compile(r"\((\d+)-[^()]*(\))?|(\d+)\)")   # "(12-person-...)" single token, "(12-person-..." open, "12)" close

PERSON_PRON = {"he", "she", "him", "her", "his", "hers", "himself", "herself", "who", "whom", "whose", "i", "me", "my",
               "you", "your", "we", "us", "our"}
THING_PRON = {"it", "its", "itself", "this", "that", "which", "what"}
FIRST_SECOND = {"i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "we", "us", "our", "ours", "ourselves"}
PLURAL_PRON = {"they", "them", "their", "theirs", "themselves", "these", "those", "we", "us", "our"}


def chains_of(sents: list[Sentence]):
    """For each sentence: dict token index -> gold chain id of the innermost mention *opened* at that token
    (the mention's first token), plus a dict head index -> chain id by walking spans to their head.
    Chain ids are document-scoped, so sentences are grouped by doc."""
    out = []
    for s in sents:
        open_at = {}
        spans = []          # (chain, start, end)
        stack = []
        for i, c in enumerate(s.ents_raw):
            if not c:
                continue
            for m in _ITEM.finditer(c):
                if m.group(1) is not None:
                    cid = int(m.group(1))
                    if m.group(2):                       # opened and closed on this token
                        spans.append((cid, i, i + 1))
                    else:
                        stack.append((cid, i))
                else:
                    cid = int(m.group(3))
                    for k in range(len(stack) - 1, -1, -1):
                        if stack[k][0] == cid:
                            _, st = stack.pop(k)
                            spans.append((cid, st, i + 1))
                            break
        # head of a span: the token inside it whose head is outside (or the root)
        head_chain = {}
        for cid, st, en in sorted(spans, key=lambda x: x[2] - x[1], reverse=True):     # outer first, inner overrides
            for j in range(st, en):
                if s.heads[j] < st or s.heads[j] >= en:
                    head_chain[j] = cid
                    break
            else:
                head_chain[st] = cid
        out.append(head_chain)
    return out


class NearestNominal:
    name = "nearest preceding nominal"

    def antecedent(self, sents, si, i, typed=False):
        s = sents[si]
        for j in range(i - 1, -1, -1):
            if s.upos[j] in ("NOUN", "PROPN"):
                return (si, j)
        for ps in range(si - 1, max(-1, si - 3), -1):
            if sents[ps].doc != s.doc:
                break
            p = sents[ps]
            for j in range(len(p) - 1, -1, -1):
                if p.upos[j] in ("NOUN", "PROPN"):
                    return (ps, j)
        return None


class TypedNearest(NearestNominal):
    name = "nearest nominal with agreeing type/number"

    def agrees(self, sents, pron_s, pron_i, cand_s, cand_j):
        w = pron_s.forms[pron_i].lower()
        m = cand_s.mention(cand_j)
        if w in PERSON_PRON and m.etype not in ("person", "unknown"):
            return False
        if w in THING_PRON and m.etype == "person":
            return False
        plural_c = "Plur" in (cand_s.feats[cand_j] if hasattr(cand_s, "feats") else "")
        if w in PLURAL_PRON and not plural_c and m.etype != "organization" and cand_s.xpos[cand_j] not in ("NNS", "NNPS"):
            return False
        if w not in PLURAL_PRON and cand_s.xpos[cand_j] in ("NNS", "NNPS"):
            return False
        return True

    def antecedent(self, sents, si, i, typed=True):
        s = sents[si]
        for j in range(i - 1, -1, -1):
            if s.upos[j] in ("NOUN", "PROPN") and self.agrees(sents, s, i, s, j):
                return (si, j)
        for ps in range(si - 1, max(-1, si - 3), -1):
            if sents[ps].doc != s.doc:
                break
            p = sents[ps]
            for j in range(len(p) - 1, -1, -1):
                if p.upos[j] in ("NOUN", "PROPN") and self.agrees(sents, s, i, p, j):
                    return (ps, j)
        return None


def bench_coref(resolver, sents, predicates_fn):
    """Pronoun event arguments in GUM test: antecedent in the same gold chain?"""
    chains = chains_of(sents)
    n = ok = none = 0
    first_person = 0
    for si, s in enumerate(sents):
        for v, voice, a, p in predicates_fn(s):
            for g in (a, p):
                if g is None or s.upos[g] != "PRON":
                    continue
                w = s.forms[g].lower()
                if w in ("i", "me", "my", "you", "your", "we", "us", "our"):
                    first_person += 1          # speaker/addressee: not resolvable from text alone
                    continue
                gold = chains[si].get(g)
                if gold is None:
                    continue
                n += 1
                ant = resolver.antecedent(sents, si, g)
                if ant is None:
                    none += 1
                    continue
                ok += chains[ant[0]].get(ant[1]) == gold
    return {"pronoun_args": n, "accuracy": round(ok / max(n, 1), 4), "no_antecedent": none,
            "first_second_person_skipped": first_person}


class SubjectFirst(TypedNearest):
    """Agreeing candidates, but subjects of the current and previous sentences are tried first (salience),
    then any agreeing nominal, nearest first."""
    name = "agreeing subject first, then nearest"

    def antecedent(self, sents, si, i, typed=True):
        s = sents[si]
        windows = [(si, s)] + [(ps, sents[ps]) for ps in range(si - 1, max(-1, si - 3), -1) if sents[ps].doc == s.doc]
        form = s.forms[i].lower()
        for ws, w in windows:                               # the same pronoun just before: same referent
            limit = i if ws == si else len(w)
            for j in range(limit - 1, -1, -1):
                if w.upos[j] == "PRON" and w.forms[j].lower() == form:
                    return (ws, j)
        for ws, w in windows:
            limit = i if ws == si else len(w)
            for j in range(limit - 1, -1, -1):
                if w.upos[j] in ("NOUN", "PROPN") and w.deprels[j] in ("nsubj", "nsubj:pass") and self.agrees(sents, s, i, w, j):
                    return (ws, j)
        return super().antecedent(sents, si, i)


# ----------------------------------------------------------------------------- learned ranker
class RankedResolver:
    """Mention-pair ranker: for a pronoun, every preceding nominal head in the current and two previous
    sentences is a candidate; hashed logistic regression scores (pronoun, candidate) pairs, trained on
    GUM train's gold chains (positive = same chain); the top-scoring candidate is the antecedent.
    Features: pronoun word, distance (tokens, sentences, intervening nominals), candidate's dependency
    role, POS, number, entity type (annotation now, typer later), agreement flags, and conjunctions."""
    name = "learned mention-pair ranker (GUM train)"
    N = 1 << 20

    def __init__(self, window_sents=3, max_cands=40, alpha=1e-5, epochs=8, seed=0):
        self.window_sents = window_sents
        self.max_cands = max_cands
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed
        self.clf = None
        self._cache = {}

    def _hid(self, f):
        x = self._cache.get(f)
        if x is None:
            import hashlib
            x = int(hashlib.blake2b(f.encode(), digest_size=4).hexdigest(), 16) % self.N
            self._cache[f] = x
        return x

    def candidates(self, sents, si, i):
        s = sents[si]
        out = []
        ok = lambda t, j: t.upos[j] in ("NOUN", "PROPN") or (t.upos[j] == "PRON" and t.forms[j].lower() not in FIRST_SECOND)
        for j in range(i - 1, -1, -1):
            if ok(s, j):
                out.append((si, j))
        for ps in range(si - 1, max(-1, si - 1 - self.window_sents), -1):
            if sents[ps].doc != s.doc:
                break
            p = sents[ps]
            for j in range(len(p) - 1, -1, -1):
                if ok(p, j):
                    out.append((ps, j))
        return out[: self.max_cands]

    def features(self, sents, si, i, cs, cj, rank):
        s, c = sents[si], sents[cs]
        w = s.forms[i].lower()
        m = c.mention(cj)
        sd = si - cs
        td = (i - cj) if sd == 0 else (i + len(c) - cj)
        num = "pl" if c.xpos[cj] in ("NNS", "NNPS") else "sg"
        pnum = "pl" if w in PLURAL_PRON else "sg"
        pkind = "person" if w in PERSON_PRON else "thing" if w in THING_PRON else "other"
        cw = c.forms[cj].lower()
        f = [f"w={w}", f"sd={sd}", f"rank={min(rank, 9)}", f"td={min(td // 3, 10)}", f"role={c.deprels[cj]}",
             f"cpron={c.upos[cj] == 'PRON'}", f"same_form={cw == w}", f"w_cw={w}:{cw if c.upos[cj] == 'PRON' else '<nom>'}",
             f"pos={c.upos[cj]}", f"num={num}", f"agree_num={num == pnum}", f"etype={m.etype}",
             f"w_etype={w}:{m.etype}", f"pkind_etype={pkind}:{m.etype}", f"w_role={w}:{c.deprels[cj]}",
             f"w_sd={w}:{sd}", f"role_sd={c.deprels[cj]}:{sd}", f"w_num={w}:{num}", f"rank_role={min(rank, 9)}:{c.deprels[cj]}",
             f"cap={c.forms[cj][:1].isupper()}", f"subj={c.deprels[cj] in ('nsubj', 'nsubj:pass')}:{sd}",
             f"prole={s.deprels[i]}", f"prole_role={s.deprels[i]}:{c.deprels[cj]}"]
        return f

    def _pairs(self, sents, items, chains=None):
        import numpy as np
        from scipy import sparse
        rows, cols, y, idx = [], [], [], []
        r = 0
        for (si, i) in items:
            cands = self.candidates(sents, si, i)
            closest = None
            if chains is not None:
                gold = chains[si].get(i)
                closest = next((k for k, (cs, cj) in enumerate(cands) if chains[cs].get(cj) == gold), None)
            for rank, (cs, cj) in enumerate(cands):
                if chains is not None:
                    same = chains[cs].get(cj) == chains[si].get(i)
                    if same and rank != closest:
                        continue                      # train on the closest same-chain mention only
                    y.append(int(same))
                for f in self.features(sents, si, i, cs, cj, rank):
                    rows.append(r)
                    cols.append(self._hid(f))
                idx.append((si, i, cs, cj))
                r += 1
        X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(r, self.N))
        X.sum_duplicates()
        return X, (np.array(y) if chains is not None else None), idx

    @staticmethod
    def pronoun_items(sents, chains, predicates_fn=None):
        items = []
        for si, s in enumerate(sents):
            for i in range(len(s)):
                if s.upos[i] == "PRON" and s.forms[i].lower() not in ("i", "me", "my", "you", "your", "we", "us", "our") \
                        and chains[si].get(i) is not None:
                    items.append((si, i))
        return items

    def fit(self, sents):
        from sklearn.linear_model import SGDClassifier
        chains = chains_of(sents)
        items = self.pronoun_items(sents, chains)
        X, y, _ = self._pairs(sents, items, chains)
        self.clf = SGDClassifier(loss="log_loss", alpha=self.alpha, max_iter=self.epochs, tol=None, random_state=self.seed).fit(X, y)
        self.n_pairs = X.shape[0]
        return self

    def antecedents(self, sents, items):
        import numpy as np
        X, _, idx = self._pairs(sents, items)
        if X.shape[0] == 0:
            return {}
        P = self.clf.predict_proba(X)[:, 1]
        best = {}
        for (si, i, cs, cj), p in zip(idx, P):
            if (si, i) not in best or p > best[(si, i)][1]:
                best[(si, i)] = ((cs, cj), p)
        return {k: v[0] for k, v in best.items()}

    def antecedent(self, sents, si, i, typed=True):
        return self.antecedents(sents, [(si, i)]).get((si, i))
