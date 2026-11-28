"""run 23 · a structural summarizer: what is the text about, and which events carry it — computed from dependent units,
not counted from words.

Units are *entities* (coreference chains of names, nouns and pronouns), not nouns. Dependencies are the records the
linker produces (actor → action → patient, with status). Aboutness is salience on that structure:

  freq   baseline: how often an entity's head word occurs                       (counting)
  roles  Σ over mentions of role weight (actor > patient > other) × status weight × position weight   (centering theory)
  graph  roles, then propagated over the event graph (an entity acted on by a salient entity inherits salience):
         power iteration = PageRank with the roles score as the personalisation vector                 (dependent units)

A summary is then selection, not generation: the sentences whose records join the most salient entities, in original
order, plus the record skeleton (the plot as who-did-what-to-whom) and the entity registry (who is who). A writer can
rephrase this later; nothing here needs a language model, and everything in it can be checked against the text.

Evaluation that needs no human: AMALGUM documents open with their title as sentence 1. Hide it, predict the top entity
from the body, and check whether its name appears in the title — for the three scorers, by genre.

    from gramlab.core import summarize
    board = summarize.title_eval(AMALGUM, genres=('bio', 'news', 'voyage', 'fiction'))
    S = summarize.Summarizer.live(frontend, linker, resolver)
    S.summarize_text(open('persuasion.txt').read(), n=12)
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .linker import BackfillLinker, GoldLinker, LearnedLinker
from .records import read_conllu_glob
from .registry import normalize_name

PRON = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "it", "its", "himself", "herself", "themselves", "itself"}
ROLE_W = {"actor": 1.0, "patient": 0.6, "other": 0.25, "source": 0.15}
SPEECH = {"say", "tell", "report", "add", "claim", "state", "announce", "write", "note", "explain", "describe", "call", "ask", "accord", "confirm", "deny", "argue", "suggest", "believe", "think"}
STATUS_W = {"asserted": 1.0, "possible": 0.5, "negated": 0.4}
STOP_HEADS = {"thing", "one", "time", "way", "year", "day", "part", "lot", "number", "kind", "people", "man", "woman", "something", "nothing", "anything", "everything"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------------------------------------ entities
class Registry:
    """Mentions → entity ids. Names cluster by token containment ('Anne Elliot' ⊃ 'Anne', 'Miss Elliot');
    common nouns cluster by head lemma; pronouns follow the resolver (or the most salient recent entity)."""

    def __init__(self):
        self.names: dict[str, int] = {}
        self.name_tokens: dict[int, set] = {}
        self.label: dict[int, str] = {}
        self.kind: dict[int, str] = {}
        self.last_seen: dict[int, int] = {}
        self.n = 0

    def _new(self, label, kind, tokens=()):
        self.n += 1
        self.label[self.n] = label; self.kind[self.n] = kind; self.name_tokens[self.n] = set(tokens)
        return self.n

    def name_entity(self, text, si=0):
        """Multi-token names are entities; a single token (a surname or first name) attaches to the most recently
        mentioned fuller name that contains it, so 'Elliot' goes to whichever Elliot was last on stage and the shared
        surname never swallows the family."""
        key = normalize_name(text)
        if not key:
            return None
        toks = [t for t in key.split() if len(t) > 1 and t not in ("mr", "mrs", "miss", "ms", "dr", "sir", "lady", "lord", "captain", "admiral", "mr.", "mrs.")]
        if not toks:
            toks = key.split()
        tset = set(toks)
        if key in self.names:
            eid = self.names[key]; self.last_seen[eid] = si
            return eid
        if len(toks) == 1:
            cands = [eid for eid, nt in self.name_tokens.items() if self.kind[eid] == "name" and len(nt) > 1 and toks[0] in nt]
            if cands:
                eid = max(cands, key=lambda e: self.last_seen.get(e, -1))
                self.last_seen[eid] = si
                return eid                                   # not cached: next time it may be someone else
            eid = self._new(text, "name", toks); self.names[key] = eid; self.last_seen[eid] = si
            return eid
        # multi-token: same token set → same entity; otherwise new (a fuller name absorbs an earlier single-token entity)
        for eid, nt in self.name_tokens.items():
            if self.kind[eid] == "name" and nt == tset:
                self.names[key] = eid; self.last_seen[eid] = si
                return eid
        eid = self._new(text, "name", toks); self.names[key] = eid; self.last_seen[eid] = si
        for k2, e2 in list(self.names.items()):            # an earlier bare first name / surname with no fuller owner yet joins this one
            if e2 != eid and self.kind.get(e2) == "name" and len(self.name_tokens[e2]) == 1 and self.name_tokens[e2] < tset and self.last_seen.get(e2, -1) >= si - 3:
                self.names[k2] = eid
        return eid

    def noun_entity(self, lemma):
        key = "noun::" + lemma.lower()
        if key in self.names:
            return self.names[key]
        eid = self._new(lemma.lower(), "noun")
        self.names[key] = eid
        return eid


def mention_text(s, m):
    return " ".join(s.forms[m.start:m.end])


def is_name(s, m):
    return any(s.upos[i] == "PROPN" for i in range(m.start, m.end))


# ------------------------------------------------------------------------------------------------ analysis
class Analysis:
    def __init__(self, sents, reg: Registry, records, mentions):
        self.sents, self.reg, self.records, self.mentions = sents, reg, records, mentions

    def salience(self, method="graph", damping=0.6):
        reg = self.reg
        if method == "freq":
            c = Counter()
            for s in self.sents:
                for i, u in enumerate(s.upos):
                    if u in ("NOUN", "PROPN"):
                        key = reg.names.get(normalize_name(s.forms[i])) if u == "PROPN" else reg.names.get("noun::" + s.lemmas[i].lower())
                        if key:
                            c[key] += 1
            return c
        n = len(self.sents)
        base = Counter()
        for eid, role, status, si in self.mentions:
            pos = 1.0 + 0.5 * (si < max(2, n // 20)) + 0.25 * (si > n - max(2, n // 20))
            base[eid] += ROLE_W[role] * STATUS_W.get(status, 1.0) * pos
        if method == "roles" or not self.records:
            return base
        ids = sorted(set(base) | {a for a, _, p, _, _ in self.records} | {p for a, _, p, _, _ in self.records})
        idx = {e: i for i, e in enumerate(ids)}
        A = np.zeros((len(ids), len(ids)), dtype=np.float64)
        for a, v, p, st, si in self.records:
            w = STATUS_W.get(st, 1.0)
            A[idx[a], idx[p]] += w; A[idx[p], idx[a]] += 0.5 * w        # salience flows both ways, more from actor to patient
        out = A.sum(1, keepdims=True); A = np.divide(A, out, out=np.zeros_like(A), where=out > 0)
        pers = np.array([base.get(e, 0.0) for e in ids]); pers = pers / max(pers.sum(), 1e-9)
        r = pers.copy()
        for _ in range(50):
            r = (1 - damping) * pers + damping * (A.T @ r)
        tot = sum(base.values()) or 1.0
        return Counter({e: float(r[idx[e]]) * tot for e in ids})

    def features(self):
        """Per-entity feature dict for a learned scorer: mention counts by role, position, kind."""
        n = len(self.sents)
        f = defaultdict(lambda: Counter())
        for eid, role, status, si in self.mentions:
            f[eid][role] += 1
            f[eid]["first"] += si == 0
            f[eid]["early"] += si < max(2, n // 10)
            f[eid]["late"] += si > n - max(2, n // 10)
            f[eid]["sents"] += 0
        seen = defaultdict(set)
        for eid, role, status, si in self.mentions:
            seen[eid].add(si)
        out = {}
        tot = max(len(self.mentions), 1)
        for eid, c in f.items():
            out[eid] = {"actor": c["actor"], "patient": c["patient"], "other": c["other"], "source": c["source"], "first": c["first"],
                        "early": c["early"], "late": c["late"], "spread": len(seen[eid]) / max(n, 1), "share": sum(c[r] for r in ROLE_W) / tot,
                        "is_name": float(self.reg.kind[eid] == "name"), "log_total": float(np.log1p(sum(c[r] for r in ROLE_W)))}
        return out

    def about(self, method="graph", k=5, model=None):
        if method == "learned" and model is not None:
            feats = self.features()
            keys = model["keys"]
            X = np.array([[feats[e][kk] for kk in keys] for e in feats]) if feats else np.zeros((0, len(keys)))
            sal = Counter(dict(zip(feats, model["clf"].decision_function(X)))) if len(feats) else Counter()
        else:
            sal = self.salience(method)
        out = []
        for eid, sc in sal.most_common(len(sal)):
            if self.reg.kind[eid] == "noun" and self.reg.label[eid] in STOP_HEADS:
                continue
            out.append((self.reg.label[eid], self.reg.kind[eid], round(sc, 2)))
            if len(out) >= k:
                break
        return out

    def skeleton(self, method="graph", k=12):
        sal = self.salience(method)
        scored = [(sal.get(a, 0) + sal.get(p, 0), si, a, v, p, st) for a, v, p, st, si in self.records]
        top = sorted(sorted(scored, reverse=True)[:k], key=lambda x: x[1])
        return [f"{self.reg.label[a]} → {v} → {self.reg.label[p]}" + ("" if st == "asserted" else f" [{st}]") for _, si, a, v, p, st in top]

    def summary(self, n=8, method="graph"):
        sal = self.salience(method)
        score = defaultdict(float)
        for a, v, p, st, si in self.records:
            score[si] += (sal.get(a, 0) + sal.get(p, 0)) * STATUS_W.get(st, 1.0)
        for eid, role, status, si in self.mentions:
            score[si] += 0.15 * sal.get(eid, 0) * ROLE_W[role]
        for si in list(score):
            score[si] *= 1.0 / (1.0 + 0.02 * max(0, len(self.sents[si]) - 40))      # mild penalty for very long sentences
        picked = sorted(sorted(score, key=lambda s: -score[s])[:n])
        return [self.sents[si].text() for si in picked]


class Summarizer:
    def __init__(self, linker, front_end=None, resolver=None):
        self.linker, self.fe, self.resolver = linker, front_end, resolver

    @classmethod
    def live(cls, frontend_path, linker_path, resolver=None):
        from .tagger import LiveFrontEnd
        return cls(BackfillLinker(LearnedLinker.load(linker_path)), LiveFrontEnd.load(frontend_path), resolver)

    def analyse(self, sents) -> Analysis:
        reg = Registry()
        records, mentions = [], []
        recent: list[int] = []                                  # recently mentioned entities, most recent last
        pron_items = []
        ev_by_sent = [self.linker.link(s) for s in sents]
        # first pass: names and nouns
        ent_of = {}                                             # (si, head) -> eid
        for si, (s, evs) in enumerate(zip(sents, ev_by_sent)):
            spans = []
            for e in evs:
                if e.actor: spans.append((e.actor, "source" if e.action in SPEECH else "actor", e.status))
                if e.patient: spans.append((e.patient, "patient", e.status))
            seen = {m.head for m, _, _ in spans}
            for i, u in enumerate(s.upos):                      # other nominal heads, lightly weighted
                if u in ("NOUN", "PROPN") and i not in seen:          # heads and modifiers alike: a modifier is a dependent too
                    spans.append((type("M", (), {"start": i, "end": i + 1, "head": i})(), "other", "asserted"))
            for m, role, status in spans:
                head = s.forms[m.head]
                if s.upos[m.head] == "PRON" or head.lower() in PRON:
                    pron_items.append((si, m.head, role, status))
                    continue
                if is_name(s, m):
                    txt = " ".join(s.forms[i] for i in range(m.start, m.end) if s.upos[i] == "PROPN") or mention_text(s, m)
                    eid = reg.name_entity(txt, si)
                else:
                    eid = reg.noun_entity(s.lemmas[m.head])
                if eid is None:
                    continue
                ent_of[(si, m.head)] = eid
                mentions.append((eid, role, status, si))
        # pronouns: resolver if available, else most recent name entity
        ante = {}
        if self.resolver is not None and pron_items:
            try:
                ante = self.resolver.antecedents(sents, [(si, i) for si, i, _, _ in pron_items])
            except Exception:
                ante = {}
        last_name_before = {}
        cur = None
        for si, s in enumerate(sents):
            for i in range(len(s)):
                if (si, i) in ent_of and reg.kind[ent_of[(si, i)]] == "name":
                    cur = ent_of[(si, i)]
                last_name_before[(si, i)] = cur
        for si, i, role, status in pron_items:
            eid = None
            a = ante.get((si, i))
            if a and a in ent_of:
                eid = ent_of[a]
            if eid is None:
                eid = last_name_before.get((si, i))
            if eid is None:
                continue
            ent_of[(si, i)] = eid
            mentions.append((eid, role, status, si))
        # records
        for si, (s, evs) in enumerate(zip(sents, ev_by_sent)):
            for e in evs:
                if not (e.actor and e.patient):
                    continue
                a = ent_of.get((si, e.actor.head)); p = ent_of.get((si, e.patient.head))
                if a is None or p is None or a == p:
                    continue
                records.append((a, e.action, p, e.status, si))
        return Analysis(sents, reg, records, mentions)

    def analyse_text(self, text, doc="doc"):
        from .pipeline import Pipeline
        from .registry import GeneralIndex
        pipe = Pipeline(linker=self.linker, general=GeneralIndex(), front_end=self.fe)
        sents = pipe.read_text(text, doc)
        return self.analyse(sents)

    def summarize_text(self, text, n=8, method="graph"):
        an = self.analyse_text(text)
        return {"about": an.about(method), "skeleton": an.skeleton(method), "summary": an.summary(n, method), "sentences": len(an.sents), "records": len(an.records)}


# ------------------------------------------------------------------------------------------------ evaluation
def title_eval(amalgum_glob, genres=("bio", "news", "voyage", "fiction", "academic", "whow", "interview", "reddit"), max_docs=400, resolver=None, out_dir=None):
    """Hide sentence 1 (the title), predict the top entity from the body with each scorer, score a hit if a content
    token of the entity's label (len>2) occurs in the title. Gold parses and gold roles: this isolates the scorer."""
    S = Summarizer(GoldLinker(), None, resolver)
    board = {}
    for g in genres:
        sents = read_conllu_glob(amalgum_glob.replace("*/dep", f"{g}/dep"))
        docs = defaultdict(list)
        for s in sents:
            docs[s.doc].append(s)
        items = []
        for doc, ss in list(docs.items())[:max_docs]:
            if len(ss) < 6:
                continue
            title_toks = {w.lower() for w in ss[0].forms if w.isalpha() and len(w) > 2}
            an = S.analyse(ss[1:])
            if an.mentions:
                items.append((title_toks, an))
        def hit(lbl, title_toks):
            return any(t.lower() in title_toks for t in re.findall(r"[A-Za-z]+", lbl) if len(t) > 2)
        # learned scorer: train on even docs, test on odd docs (and vice versa) — every doc scored out of sample
        from sklearn.linear_model import LogisticRegression
        keys = ["actor", "patient", "other", "source", "first", "early", "late", "spread", "share", "is_name", "log_total"]
        def fit(train):
            X, y = [], []
            for tt, an in train:
                for eid, fv in an.features().items():
                    X.append([fv[kk] for kk in keys]); y.append(int(hit(an.reg.label[eid], tt)))
            if len(set(y)) < 2:
                return None
            return {"keys": keys, "clf": LogisticRegression(max_iter=1000, C=1.0).fit(np.array(X), np.array(y))}
        halves = [items[0::2], items[1::2]]
        models = [fit(halves[1]), fit(halves[0])]
        hits = Counter(); top3 = Counter(); n = len(items)
        for hi, half in enumerate(halves):
            for tt, an in half:
                for method in ("freq", "roles", "graph", "learned"):
                    ab = an.about(method, k=3, model=models[hi]) if method == "learned" else an.about(method, k=3)
                    if ab and hit(ab[0][0], tt):
                        hits[method] += 1
                    if any(hit(l, tt) for l, _, _ in ab):
                        top3[method] += 1
        methods = ("freq", "roles", "graph", "learned")
        board[g] = {"docs": n, **{m: round(hits[m] / max(n, 1), 3) for m in methods}, **{m + "@3": round(top3[m] / max(n, 1), 3) for m in methods}}
        if models[0]:
            board[g]["learned_weights"] = {kk: round(float(w), 3) for kk, w in zip(keys, models[0]["clf"].coef_[0])}
        log(f"{g:10} n={n:4d}  top-1  freq {board[g]['freq']:.3f}  roles {board[g]['roles']:.3f}  graph {board[g]['graph']:.3f}  learned {board[g]['learned']:.3f}   | top-3  {board[g]['freq@3']:.3f} {board[g]['roles@3']:.3f} {board[g]['graph@3']:.3f} {board[g]['learned@3']:.3f}")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "title_eval.json").write_text(json.dumps(board, indent=2))
    return board
