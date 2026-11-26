"""core-v1 · end-to-end: does the ordered meaning index help *finding*, not just representing?

    python -m gramlab.core.bench_retrieval --amalgum 'data/amalgum/amalgum/*/dep/*.conllu' [--n 2000] [--out results]

Collection: every AMALGUM sentence (191k, parsed) ingested through the pipeline (roles from the parse, keys from the
registries, family postings under the verb's top-3 senses). Two query types built from held-out source sentences
whose main event has a nominal actor and patient:

  synonym query   — "<actor words> <verb synonym> <patient words>", the synonym being another lemma of the verb's
                    chosen synset (so the verb word itself never appears in the source). BM25 must find the source
                    on the argument words alone; the key index can also match the sense.
  reversal decoy  — a copy of the source with actor and patient swapped is added to the collection. Same words,
                    opposite meaning: BM25 cannot separate them; the ordered key can.

Systems: BM25 over sentences · key index (family postings of the query event, BM25 to order within) · hybrid
(BM25 score + bonus for a shared family key). Metrics: recall@1/@10 of the source, decoy-above-source rate,
candidate-set size and µs per query. Synonyms come from WordNet, which the key index also uses, so the synonym
query is a test of the mechanism under favourable conditions and is reported as such; the decoy test is not.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

from .key import Keyer
from .linker import GoldLinker
from .pipeline import Pipeline
from .records import Event, Mention, read_conllu_glob
from .registry import GeneralIndex, SpecificIndex

TOKEN = None


def toks(text):
    return [w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(w) > 1]


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.vocab = {}
        rows, cols, vals = [], [], []
        self.lens = np.zeros(len(docs), dtype=np.float32)
        for i, d in enumerate(docs):
            c = Counter(toks(d))
            self.lens[i] = sum(c.values())
            for w, n in c.items():
                j = self.vocab.setdefault(w, len(self.vocab))
                rows.append(i); cols.append(j); vals.append(n)
        self.tf = sparse.csc_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(docs), len(self.vocab)))
        df = np.diff(self.tf.indptr)
        N = len(docs)
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.avg = float(self.lens.mean())
        self.k1, self.b = k1, b
        self.N = N

    def scores(self, query):
        s = np.zeros(self.N, dtype=np.float32)
        for w in set(toks(query)):
            j = self.vocab.get(w)
            if j is None:
                continue
            col = self.tf.getcol(j)
            idx, tf = col.indices, col.data
            denom = tf + self.k1 * (1 - self.b + self.b * self.lens[idx] / self.avg)
            s[idx] += self.idf[j] * tf * (self.k1 + 1) / denom
        return s


def main_event(s, linker):
    """The first event with a nominal (non-pronoun) actor and patient, or None."""
    for e in linker.link(s):
        if e.actor and e.patient and e.actor.kind != "pronoun" and e.patient.kind != "pronoun" \
                and e.actor.text.lower() != e.patient.text.lower():
            return e
    return None


def swap_text(s, e):
    """Source text with the actor and patient mention strings exchanged."""
    a, p = e.actor, e.patient
    forms = list(s.forms)
    if a.start < p.start:
        first, second = a, p
    else:
        first, second = p, a
    out = forms[:first.start] + forms[second.start:second.end] + forms[first.end:second.start] + forms[first.start:first.end] + forms[second.end:]
    return " ".join(out)


def participle_table(sents):
    """lemma -> most frequent past-participle form, from the training corpus."""
    c = defaultdict(Counter)
    for s in sents:
        for f, l, x in zip(s.forms, s.lemmas, s.xpos):
            if x == "VBN":
                c[l.lower()][f.lower()] += 1
    return {l: v.most_common(1)[0][0] for l, v in c.items()}


def passivize(s, e, syn, vbn):
    """Full-sentence paraphrase of an active 'prefix ACTOR V PATIENT suffix' sentence: 'prefix PATIENT was/were
    VBN(syn) by ACTOR suffix'. Only for sources where actor span, verb and patient span are contiguous and the verb
    is simple past; returns None otherwise."""
    a, p = e.actor, e.patient
    if e.voice != "active" or s.xpos[e.verb] != "VBD" or a.end != e.verb or p.start != e.verb + 1:
        return None
    f = s.forms
    be = "were" if s.xpos[p.head] in ("NNS", "NNPS") else "was"
    out = f[p.start:p.end] + [be, vbn, "by"] + f[a.start:a.end] + ["."]       # the clause only, not the rest of the sentence
    out[0] = out[0][:1].upper() + out[0][1:]
    return " ".join(out)


def run(amalgum_glob, n_queries=2000, out_dir="results", seed=0, bonus=3.0, query_senses=99, front_end=None, live_linker=None):
    """front_end + live_linker: queries become full-sentence paraphrases (synonym verb, voice flipped), tagged and
    linked live — the query's roles are parsed, not constructed."""
    rng = random.Random(seed)
    t0 = time.time()
    sents = read_conllu_glob(amalgum_glob)
    gold = GoldLinker()
    general, specific = GeneralIndex(), SpecificIndex()
    pipe = Pipeline(linker=gold, general=general, specific=specific)
    n_occ = pipe.ingest(sents)
    print(f"ingested {len(sents)} sentences, {n_occ} occurrences in {time.time()-t0:.0f}s; {pipe.index.stats()}")
    live = front_end is not None and live_linker is not None
    vbn_of = participle_table(sents) if live else {}

    # queries from sentences whose main event has nominal roles and a verb with a synonym in its chosen synset
    cands = []
    for si, s in enumerate(sents):
        e = main_event(s, gold)
        if e is None:
            continue
        if live and (e.voice != "active" or s.xpos[e.verb] != "VBD" or e.actor.end != e.verb or e.patient.start != e.verb + 1):
            continue
        syns = [w for w in general.synonyms(e.action, "VERB", 50) if " " not in w and w != e.action]
        ss = general.synsets(e.action, "VERB")
        if not ss:
            continue
        first = set(general._wn.lemmas(*ss[0])) if general._wn is not None else set()
        syn = next((w for w in syns if w in first), None)        # a synonym from the first (posted) sense
        if syn is None:
            continue
        if live and syn not in vbn_of:
            continue
        cands.append((si, e, syn))
    rng.shuffle(cands)
    queries = cands[:n_queries]
    print(f"{len(cands)} eligible sources, using {len(queries)}")

    # decoys: swapped copies appended to the collection, with swapped roles indexed
    texts = [s.text() for s in sents]
    decoy_of = {}
    for qi, (si, e, syn) in enumerate(queries):
        s = sents[si]
        d_text = swap_text(s, e)
        d_id = len(texts)
        texts.append(d_text)
        decoy_of[si] = d_id
        swapped = Event(e.verb, e.action, e.patient, e.actor, e.voice, e.status)
        # index the decoy under the swapped event, through the same resolution path as real occurrences
        rs = pipe.keyer.resolve_event(swapped, s)
        keys = pipe.keyer.keys_from(rs)
        from .records import Occurrence
        occ = Occurrence(swapped, rs, d_text, s.doc, s.sid + "#decoy", keys["specific"][0], keys["family"][0])
        pipe.index.records.append(occ)
        for level in ("specific", "family", "identity", "proposal"):
            for k in keys[level]:
                pipe.index.postings[(level, k, rs.status)].append(d_id)
    # occurrence ids == sentence ids only if every sentence yielded exactly one occurrence; build sentence->occurrence map
    occ_sent = {}
    for i, o in enumerate(pipe.index.records):
        occ_sent.setdefault(o.sid, []).append(i)
    print(f"collection {len(texts)} texts incl. {len(decoy_of)} decoys")

    t = time.time()
    bm = BM25(texts)
    print(f"BM25 built in {time.time()-t:.0f}s, vocab {len(bm.vocab)}")

    sid_to_text = {}
    for i, s in enumerate(sents):
        sid_to_text[s.sid] = i
    for si, d in decoy_of.items():
        sid_to_text[sents[si].sid + "#decoy"] = d

    res = {k: Counter() for k in ("bm25", "key", "hybrid")}
    times = defaultdict(list)
    cand_sizes = []
    live_stats = Counter()
    examples_live = []
    for si, e, syn in queries:
        s = sents[si]
        if live:
            q_text = passivize(s, e, syn, vbn_of[syn])
            t = time.time()
            q_sent = front_end.sentence(q_text.split(), sid="q", doc="q")
            evs = [x for x in live_linker.link(q_sent) if x.actor and x.patient]
            live_stats["parse_us"] += (time.time() - t) * 1e6
            q_event = next((x for x in evs if x.action == syn), None) or next((x for x in evs if x.voice == "passive"), None) or (evs[0] if evs else None)
            if q_event is None:
                live_stats["no_event"] += 1
                q_event = Event(0, syn, None, None, "passive", "asserted")
            else:
                # parsed roles right? the source's head word must fall inside the predicted mention span (and not the other way round)
                def inside(mention, head_word):
                    return head_word.lower() in [w.lower() for w in mention.text.split()]
                a_ok, p_ok = inside(q_event.actor, s.forms[e.actor.head]), inside(q_event.patient, s.forms[e.patient.head])
                live_stats["roles_right"] += a_ok and p_ok
                live_stats["roles_swapped"] += inside(q_event.actor, s.forms[e.patient.head]) and inside(q_event.patient, s.forms[e.actor.head]) and not (a_ok and p_ok)
                if len(examples_live) < 6:
                    examples_live.append({"source": s.text()[:120], "query": q_text[:120], "parsed": (q_event.actor.text, q_event.action, q_event.patient.text, q_event.voice)})
        else:
            q_text = f"{e.actor.text} {syn} {e.patient.text}"
            q_event = Event(e.verb, syn, e.actor, e.patient, e.voice, e.status)
        target, decoy = si, decoy_of[si]

        t = time.time()
        sc = bm.scores(q_text)
        order = np.argsort(-sc, kind="stable")
        times["bm25"].append(time.time() - t)
        rank = int(np.where(order == target)[0][0])
        res["bm25"]["r1"] += rank == 0
        res["bm25"]["r10"] += rank < 10
        res["bm25"]["decoy_above"] += sc[decoy] > sc[target]
        res["bm25"]["tie"] += sc[decoy] == sc[target]

        t = time.time()
        qkeys = pipe.keyer.keys_from(pipe.keyer.resolve_event(q_event), query=(query_senses > pipe.keyer.senses))
        cset, iset = set(), set()
        for fk in qkeys["family"]:
            for st in ("asserted", "negated", "possible"):
                for i in pipe.index.postings.get(("family", fk, st), ()):
                    cset.add(sid_to_text[pipe.index.records[i].sid])
        for level in ("identity", "proposal"):
            for ik in qkeys[level]:
                for st in ("asserted", "negated", "possible"):
                    for i in pipe.index.postings.get((level, ik, st), ()):
                        iset.add(sid_to_text[pipe.index.records[i].sid])
        cset |= iset                       # key lookup = family postings ∪ identity postings
        cand_sizes.append(len(cset))
        if live:
            live_stats["target_in_family_set"] += target in cset
            live_stats["target_in_identity_set"] += target in iset
        if cset:
            cl = np.fromiter(cset, dtype=np.int64)
            korder = cl[np.argsort(-sc[cl], kind="stable")]
            krank = int(np.where(korder == target)[0][0]) if target in cset else 10**6
            k_decoy_above = (decoy in cset) and (sc[decoy] > sc[target] if target in cset else True)
            k_tie = (decoy in cset) and (target in cset) and sc[decoy] == sc[target]
        else:
            krank, k_decoy_above, k_tie = rank, sc[decoy] > sc[target], sc[decoy] == sc[target]
        times["key"].append(time.time() - t)
        res["key"]["r1"] += krank == 0
        res["key"]["r10"] += krank < 10
        res["key"]["decoy_above"] += k_decoy_above
        res["key"]["tie"] += k_tie
        res["key"]["empty_candidates"] += not cset
        res["key"]["source_above_decoy"] += (target in cset) and ((decoy not in cset) or sc[target] > sc[decoy] or (target in iset and decoy not in iset))
        res["bm25"]["source_above_decoy"] += sc[target] > sc[decoy]

        t = time.time()
        hs = sc.copy()
        if cset:
            hs[np.fromiter(cset, dtype=np.int64)] += bonus
        if iset:
            hs[np.fromiter(iset, dtype=np.int64)] += bonus
        res["hybrid"]["source_above_decoy"] += hs[target] > hs[decoy]
        horder = np.argsort(-hs, kind="stable")
        times["hybrid"].append(time.time() - t)
        hrank = int(np.where(horder == target)[0][0])
        res["hybrid"]["r1"] += hrank == 0
        res["hybrid"]["r10"] += hrank < 10
        res["hybrid"]["decoy_above"] += hs[decoy] > hs[target]
        res["hybrid"]["tie"] += hs[decoy] == hs[target]

    n = len(queries)
    board = {"collection": len(texts), "decoys": len(decoy_of), "queries": n, "bonus": bonus, "query_senses": query_senses,
             "query_mode": "live full-sentence paraphrase (synonym + voice flip), tagged and linked live" if live else "constructed roles",
             "candidate_set_p50_p90_max": [int(np.percentile(cand_sizes, 50)), int(np.percentile(cand_sizes, 90)), int(max(cand_sizes))] if cand_sizes else [],
             "systems": {}}
    for k, c in res.items():
        board["systems"][k] = {"recall@1": round(c["r1"] / n, 4), "recall@10": round(c["r10"] / n, 4),
                               "source_above_decoy": round(float(c["source_above_decoy"]) / n, 4),
                               "decoy_above_source": round(float(c["decoy_above"]) / n, 4), "decoy_tied": round(float(c["tie"]) / n, 4),
                               "us_per_query": round(1e6 * float(np.mean(times[k])), 0), **({"empty_candidates": c["empty_candidates"]} if k == "key" else {})}
        print(k, board["systems"][k])
    print("candidate set p50/p90/max", board["candidate_set_p50_p90_max"])
    ex = []
    for si, e, syn in queries[:8]:
        ex.append({"source": sents[si].text()[:140], "query": f"{e.actor.text} {syn} {e.patient.text}", "decoy": texts[decoy_of[si]][:140]})
    board["examples"] = ex
    if live:
        board["live_queries"] = {"no_event": live_stats["no_event"], "roles_right": round(live_stats["roles_right"] / n, 4),
                                 "target_in_family_set": round(live_stats["target_in_family_set"] / n, 4), "target_in_identity_set": round(live_stats["target_in_identity_set"] / n, 4),
                                 "roles_swapped": round(live_stats["roles_swapped"] / n, 4), "parse_us_per_query": round(live_stats["parse_us"] / n, 0),
                                 "examples": examples_live}
        print("live queries", {k: v for k, v in board["live_queries"].items() if k != "examples"})
        for x in examples_live[:4]:
            print("   ", x)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "event_retrieval.json").write_text(json.dumps(board, indent=2))
    return board


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--amalgum", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--bonus", type=float, default=3.0)
    ap.add_argument("--query_senses", type=int, default=99)
    ap.add_argument("--frontend", default=None, help="frontend.pkl: switch to live full-sentence queries")
    ap.add_argument("--linker", default=None, help="linker.pkl (with --frontend)")
    ap.add_argument("--typer", default=None, help="typer.pkl (optional, with --frontend)")
    a = ap.parse_args(argv)
    fe = lk = None
    if a.frontend and a.linker:
        from .tagger import LiveFrontEnd
        from .linker import BackfillLinker, LearnedLinker
        typer = None
        if a.typer:
            from .typer import EntityTyper
            typer = EntityTyper.load(a.typer, general=GeneralIndex())
        fe = LiveFrontEnd.load(a.frontend, typer=typer)
        lk = BackfillLinker(LearnedLinker.load(a.linker))
    run(a.amalgum, a.n, a.out, bonus=a.bonus, query_senses=a.query_senses, front_end=fe, live_linker=lk)


if __name__ == "__main__":
    main()
