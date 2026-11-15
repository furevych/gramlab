"""Train and score the grammar meaning map (CPU).

  python scripts/map_stage.py --teacher work/teacher --index work/bm25 --queries work/bm25/eval_queries.jsonl \
      --stsb data/stsb.jsonl --out work/map --results results/map.json [--max_passages 0] [--epochs 8]

1. Trains two maps on the parsed AMALGUM sentences against the teacher embeddings (work/teacher/amalgum_sent.npy):
   `slots` (7 grammar slots) and `bow` (same model, one slot = bag of words) — the controlled comparison.
2. STS-B: Spearman of map cosine vs gold similarity, next to the teacher's own number (results/stsb_teacher.json).
3. Retrieval scoreboard: the same title / body-sentence queries as BM25 and the dense teacher, with each map encoding
   the BM25 passages (tagged with spaCy's POS tagger; no parser), plus a BM25 + map hybrid (BM25 top-50 re-ranked).
Writes results/map.json and prints one table.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gramlab.map import train_map, MapEncoder, SLOTS
from gramlab.retrieval import BM25Index, recall_at_k
from gramlab.prepare import UPOS_TAGS


def log(*a):
    print(*a, flush=True)


class Tagger:
    """Raw text -> (forms, upos). spaCy POS tagger when available; else every word is a NOUN (= bag of words)."""
    def __init__(self):
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
            log("[tagger] spaCy en_core_web_sm (POS only)")
        except Exception as e:
            log(f"[tagger] spaCy unavailable ({e!r}); falling back to all-NOUN tagging")

    def __call__(self, texts, n_process=1):
        if self.nlp is None:
            return [(t.split(), ["NOUN"] * len(t.split())) for t in texts]
        out = []
        for doc in self.nlp.pipe(texts, batch_size=256, n_process=n_process):
            out.append(([t.text for t in doc], [t.pos_ for t in doc]))
        return out


def load_amalgum(teacher_dir, limit=0):
    sents, rows = [], []
    with open(os.path.join(teacher_dir, "amalgum_sentences.jsonl"), encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            r = json.loads(line)
            upos = [UPOS_TAGS[u] if isinstance(u, int) else u for u in r["upos"]]
            sents.append((r["forms"], upos)); rows.append(i)
    Y = np.load(os.path.join(teacher_dir, "amalgum_sent.npy")).astype(np.float32)[rows]
    return sents, Y


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="work/teacher")
    ap.add_argument("--index", default="work/bm25")
    ap.add_argument("--queries", default="work/bm25/eval_queries.jsonl")
    ap.add_argument("--stsb", default="data/stsb.jsonl")
    ap.add_argument("--out", default="work/map")
    ap.add_argument("--results", default="results/map.json")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="training sentences (0 = all)")
    ap.add_argument("--max_passages", type=int, default=0, help="passages to encode for the retrieval scoreboard (0 = all)")
    ap.add_argument("--n_process", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    res = {"variants": {}}

    sents, Y = load_amalgum(args.teacher, args.limit)
    log(f"[map] {len(sents):,} AMALGUM sentences with teacher targets {Y.shape}")
    encoders = {}
    for name, n_slots in (("slots", len(SLOTS)), ("bow", 1)):
        enc = train_map(sents, Y, os.path.join(args.out, name), n_slots=n_slots, dim=args.dim, epochs=args.epochs, log=log)
        encoders[name] = enc
        res["variants"][name] = {"val_cosine_to_teacher": enc.meta["val_cosine"], "n_slots": n_slots, "dim": args.dim}
    log(f"[map] trained in {(time.time() - t0) / 60:.1f} min")

    tagger = Tagger()
    res["tagger"] = "spacy" if tagger.nlp is not None else "all-NOUN fallback"

    # ---- STS-B
    if args.stsb and os.path.exists(args.stsb):
        rows = [json.loads(l) for l in open(args.stsb, encoding="utf-8")]
        s1 = tagger([r["s1"] for r in rows], args.n_process); s2 = tagger([r["s2"] for r in rows], args.n_process)
        gold = [r["score"] for r in rows]
        for name, enc in encoders.items():
            a, b = enc.encode(s1), enc.encode(s2)
            rho = spearman((a * b).sum(1), gold)
            res["variants"][name]["stsb_spearman"] = round(rho, 4)
            log(f"[map] STS-B {name}: Spearman {rho:.4f}")
        tpath = os.path.join(os.path.dirname(args.results) or ".", "stsb_teacher.json")
        if os.path.exists(tpath):
            res["teacher_stsb_spearman"] = json.load(open(tpath))["teacher_spearman"]

    # ---- retrieval scoreboard
    if os.path.exists(os.path.join(args.index, "passages.jsonl")) and os.path.exists(args.queries):
        idx = BM25Index.load(args.index)
        n = len(idx) if not args.max_passages else min(args.max_passages, len(idx))
        log(f"[map] tagging + encoding {n:,} passages ...")
        texts = []
        for i in range(n):
            p = idx.passage(i)
            texts.append(f"{p['title']}. " + " ".join(p["text"].split()[:60]))
        tagged = []
        for s in range(0, n, 20000):
            tagged.extend(tagger(texts[s:s + 20000], args.n_process))
            log(f"[map]   tagged {len(tagged):,} ({(time.time() - t0) / 60:.1f} min)")
        queries = [json.loads(l) for l in open(args.queries, encoding="utf-8")]
        q_tagged = tagger([q["query"] for q in queries], args.n_process)
        ks = (1, 5, 10)
        for name, enc in encoders.items():
            P = enc.encode(tagged)                              # n, out_dim (unit vectors)
            Q = enc.encode(q_tagged)
            ranked = {}
            for s in range(0, len(queries), 200):               # 200 x n similarities at a time (memory)
                sims = Q[s:s + 200] @ P.T
                top = np.argpartition(-sims, 10, axis=1)[:, :10]
                for j, q in enumerate(queries[s:s + 200]):
                    order = top[j][np.argsort(-sims[j, top[j]])]
                    ranked[q["query"]] = [idx.titles[i] for i in order]
            r = recall_at_k(lambda q, k: ranked[q][:k], queries, ks)
            res["variants"][name]["retrieval"] = r
            log(f"[map] retrieval {name}: " + json.dumps(r["all"]) + "  body " + json.dumps(r["body"]))
            # hybrid: BM25 top-50 re-ranked by normalised BM25 + map cosine
            ranked_h = {}
            for qi, q in enumerate(queries):
                hits = idx.search(q["query"], 50)
                if not hits:
                    ranked_h[q["query"]] = []; continue
                mx = max(h[0] for h in hits)
                sc = [(h[0] / mx + (float(Q[qi] @ P[h[1]]) if h[1] < n else 0.0), h[1]) for h in hits]
                ranked_h[q["query"]] = [idx.titles[i] for _, i in sorted(sc, reverse=True)]
            rh = recall_at_k(lambda q, k: ranked_h[q][:k], queries, ks)
            res["variants"][name]["hybrid_bm25_retrieval"] = rh
            log(f"[map] hybrid bm25+{name}: " + json.dumps(rh["all"]) + "  body " + json.dumps(rh["body"]))
        for f, key in (("retrieval_bm25.json", "bm25_retrieval"), ("retrieval_dense.json", "teacher_retrieval")):
            p = os.path.join(os.path.dirname(args.results) or ".", f)
            if os.path.exists(p):
                res[key] = {k: v for k, v in json.load(open(p)).items() if k in ("title", "body", "all")}

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    json.dump(res, open(args.results, "w"), indent=1)
    log("\nsystem            STS-B   recall@1 all / title / body")
    for name, v in res["variants"].items():
        r = v.get("retrieval", {}); h = v.get("hybrid_bm25_retrieval", {})
        log(f"map {name:11} {v.get('stsb_spearman', float('nan')):.4f}   " + (f"{r['all']['recall@1']:.3f} / {r['title']['recall@1']:.3f} / {r['body']['recall@1']:.3f}" if r else "-")
            + (f"   hybrid {h['all']['recall@1']:.3f} / {h['title']['recall@1']:.3f} / {h['body']['recall@1']:.3f}" if h else ""))
    for key, label in (("bm25_retrieval", "bm25"), ("teacher_retrieval", "teacher (dense)")):
        if key in res:
            r = res[key]; log(f"{label:15} {res.get('teacher_stsb_spearman', float('nan')) if 'teacher' in key else float('nan'):.4f}   {r['all']['recall@1']:.3f} / {r['title']['recall@1']:.3f} / {r['body']['recall@1']:.3f}")
    log(f"[map] done in {(time.time() - t0) / 60:.1f} min -> {args.results}")


if __name__ == "__main__":
    main()
