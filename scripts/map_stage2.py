"""Meaning map v2: trained for retrieval (CPU).

  python scripts/map_stage2.py --index work/bm25 --wiki data/simplewiki.jsonl --wiki data/enwiki.jsonl --wiki data/enwiki2.jsonl \
      --queries work/bm25/eval_queries.jsonl --stsb data/stsb.jsonl --out work/map2 --results results/map2.json

v1 (scripts/map_stage.py) regressed single sentences onto the teacher and was useless as a retriever: passage
vectors were dominated by length and template words. v2 changes the training signal, not the idea:
  * mean-pooled codes (a 60-word passage and a 5-word question live on the same scale);
  * trained on BM25 passages ("title: text") against the teacher's *passage* embeddings (computed here on CPU
    for a training subset), plus question -> gold-passage pairs ("Tell me about <title>." / a sentence from a
    later paragraph of the article -> that article's passage embedding), articles in the scoreboard excluded;
  * both variants again: 7 grammar slots vs one slot (bag of words).
Scored on STS-B (for reference), the retrieval scoreboard (map alone, BM25 + map hybrid), and saves the best
variant's passage matrix (work/map2/<variant>/passages.npy) so chat can use the hybrid retriever.
"""
import argparse, json, os, random, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gramlab.map import train_map, MapEncoder, SLOTS, Vocab
from gramlab.retrieval import BM25Index, recall_at_k, clean_title, cut_words, EVAL_TEMPLATES
from map_stage import Tagger, spearman, log

_SENT = re.compile(r"(?<=[.!?])\s+")


def make_pairs(wiki_paths, title_to_idx, exclude, n_articles, rng, min_words=20):
    """(query_text, passage_index) pairs from articles that are in the index and not in the scoreboard."""
    pairs, seen = [], set()
    for path in wiki_paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                t = clean_title(r["title"]).lower()
                i = title_to_idx.get(r["title"])
                if i is None or t in exclude or t in seen:
                    continue
                paras = [p.strip() for p in r["text"].split("\n") if len(p.split()) >= min_words]
                if len(paras) < 2:
                    continue
                seen.add(t)
                pairs.append((rng.choice(EVAL_TEMPLATES).format(t=clean_title(r["title"])), i))
                sents = [x for x in _SENT.split(paras[1]) if 8 <= len(x.split()) <= 30]
                if sents:
                    pairs.append((sents[0], i))
                if len(seen) >= n_articles:
                    return pairs
    return pairs


def teacher_embed(texts, model_name, batch=128):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device="cpu")
    return m.encode(texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="work/bm25")
    ap.add_argument("--wiki", action="append", required=True)
    ap.add_argument("--queries", default="work/bm25/eval_queries.jsonl")
    ap.add_argument("--stsb", default="data/stsb.jsonl")
    ap.add_argument("--teacher_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default="work/map2")
    ap.add_argument("--results", default="results/map2.json")
    ap.add_argument("--train_passages", type=int, default=150000, help="random passages used as teacher-regression targets")
    ap.add_argument("--train_articles", type=int, default=150000, help="articles that give question/body-sentence -> passage pairs")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--n_process", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time(); rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    idx = BM25Index.load(args.index)
    n = len(idx)
    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8")]
    exclude = {q["gold"].lower() for q in queries}
    title_to_idx = {}
    for i, t in enumerate(idx.titles):
        title_to_idx.setdefault(t, i)

    # ---- training inputs: passages (teacher-regression) + question/body-sentence pairs
    p_ids = sorted(rng.sample(range(n), min(args.train_passages, n)))
    pairs = make_pairs(args.wiki, title_to_idx, exclude, args.train_articles, rng)
    pair_targets = sorted({i for _, i in pairs} - set(p_ids))
    target_ids = sorted(set(p_ids) | set(pair_targets))
    log(f"[map2] {len(p_ids):,} training passages, {len(pairs):,} question/sentence->passage pairs, "
        f"{len(target_ids):,} passages to embed with the teacher on CPU ...")
    ptext = lambda i: (lambda p: f"{p['title']}: {cut_words(p['text'], 60)}")(idx.passage(i))
    T = teacher_embed([ptext(i) for i in target_ids], args.teacher_model)
    tpos = {i: k for k, i in enumerate(target_ids)}
    log(f"[map2] teacher embeddings {T.shape} in {(time.time() - t0) / 60:.1f} min")

    tagger = Tagger()
    train_texts = [ptext(i) for i in p_ids] + [q for q, _ in pairs]
    Y = np.concatenate([T[[tpos[i] for i in p_ids]], T[[tpos[i] for _, i in pairs]]])
    log(f"[map2] tagging {len(train_texts):,} training texts ...")
    train_tagged = tagger(train_texts, args.n_process)
    log(f"[map2] tagging all {n:,} passages for the scoreboard ...")
    all_tagged = []
    for s in range(0, n, 50000):
        all_tagged.extend(tagger([ptext(i) for i in range(s, min(n, s + 50000))], args.n_process))
        log(f"[map2]   {len(all_tagged):,} ({(time.time() - t0) / 60:.1f} min)")
    q_tagged = tagger([q["query"] for q in queries], args.n_process)
    stsb_rows = [json.loads(l) for l in open(args.stsb, encoding="utf-8")] if args.stsb and os.path.exists(args.stsb) else []
    s1 = tagger([r["s1"] for r in stsb_rows], args.n_process); s2 = tagger([r["s2"] for r in stsb_rows], args.n_process)

    vocab = Vocab.build(train_tagged + all_tagged[:200000], max_words=60000)
    res = {"variants": {}, "train_passages": len(p_ids), "pairs": len(pairs)}
    best, best_score = None, -1
    for name, n_slots in (("slots", len(SLOTS)), ("bow", 1)):
        enc = train_map(train_tagged, Y, os.path.join(args.out, name), n_slots=n_slots, dim=args.dim, epochs=args.epochs,
                        pool="mean", vocab=vocab, log=log)
        v = {"val_cosine_to_teacher": enc.meta["val_cosine"], "n_slots": n_slots, "pool": "mean"}
        if stsb_rows:
            a, b = enc.encode(s1), enc.encode(s2)
            v["stsb_spearman"] = round(spearman((a * b).sum(1), [r["score"] for r in stsb_rows]), 4)
        P = enc.encode(all_tagged); Q = enc.encode(q_tagged)
        ranked, ranked_h = {}, {}
        for s in range(0, len(queries), 200):
            sims = Q[s:s + 200] @ P.T
            top = np.argpartition(-sims, 10, axis=1)[:, :10]
            for j, q in enumerate(queries[s:s + 200]):
                order = top[j][np.argsort(-sims[j, top[j]])]
                ranked[q["query"]] = [idx.titles[i] for i in order]
        for qi, q in enumerate(queries):
            hits = idx.search(q["query"], 50)
            if not hits:
                ranked_h[q["query"]] = []; continue
            mx = max(h[0] for h in hits)
            sc = sorted(((h[0] / mx + float(Q[qi] @ P[h[1]]), h[1]) for h in hits), reverse=True)
            ranked_h[q["query"]] = [idx.titles[i] for _, i in sc]
        v["retrieval"] = recall_at_k(lambda q, k: ranked[q][:k], queries)
        v["hybrid_bm25_retrieval"] = recall_at_k(lambda q, k: ranked_h[q][:k], queries)
        res["variants"][name] = v
        log(f"[map2] {name}: STS-B {v.get('stsb_spearman')}  retrieval {json.dumps(v['retrieval']['all'])} body {json.dumps(v['retrieval']['body'])}")
        log(f"[map2] {name}: hybrid {json.dumps(v['hybrid_bm25_retrieval']['all'])} body {json.dumps(v['hybrid_bm25_retrieval']['body'])}")
        score = v["hybrid_bm25_retrieval"]["all"]["recall@1"]
        np.save(os.path.join(args.out, name, "passages.npy"), P.astype(np.float16))
        if score > best_score:
            best, best_score = name, score
    res["best_for_hybrid"] = best
    for f, key in (("retrieval_bm25.json", "bm25_retrieval"), ("retrieval_dense.json", "teacher_retrieval")):
        p = os.path.join(os.path.dirname(args.results) or ".", f)
        if os.path.exists(p):
            res[key] = {k: v for k, v in json.load(open(p)).items() if k in ("title", "body", "all")}
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    json.dump(res, open(args.results, "w"), indent=1)
    log("\nsystem           STS-B   recall@1 all / title / body        hybrid all / title / body")
    for name, v in res["variants"].items():
        r, h = v["retrieval"], v["hybrid_bm25_retrieval"]
        log(f"map2 {name:10} {v.get('stsb_spearman', float('nan')):.4f}   {r['all']['recall@1']:.3f} / {r['title']['recall@1']:.3f} / {r['body']['recall@1']:.3f}"
            f"            {h['all']['recall@1']:.3f} / {h['title']['recall@1']:.3f} / {h['body']['recall@1']:.3f}")
    for key, label in (("bm25_retrieval", "bm25"), ("teacher_retrieval", "teacher")):
        if key in res:
            r = res[key]; log(f"{label:15}    -     {r['all']['recall@1']:.3f} / {r['title']['recall@1']:.3f} / {r['body']['recall@1']:.3f}")
    log(f"[map2] best variant for the hybrid: {best}; done in {(time.time() - t0) / 60:.1f} min -> {args.results}")


if __name__ == "__main__":
    main()
