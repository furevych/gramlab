"""Tune the BM25 + map hybrid on half the scoreboard, report on the other half, save the rule for chat (CPU, minutes).

  python scripts/hybrid_tune.py --index work/bm25 --map work/map2/slots --queries work/bm25/eval_queries.jsonl --results results/hybrid.json

Rules tried: BM25 alone; weight w in {0.5, 1, 2, 3} on the normalised BM25 score added to the map cosine; each with and
without the guard (keep BM25's answer when its top hit is an exact title). Picked by recall@1 on the dev half,
reported on the test half. Writes <map>/hybrid.json = the chosen rule, which HybridRetriever.load() and the chat
notebook pick up.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gramlab.map import MapEncoder, HybridRetriever
from gramlab.retrieval import BM25Index, recall_at_k
from map_stage import Tagger, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="work/bm25")
    ap.add_argument("--map", default="work/map2/slots")
    ap.add_argument("--queries", default="work/bm25/eval_queries.jsonl")
    ap.add_argument("--results", default="results/hybrid.json")
    args = ap.parse_args()
    t0 = time.time()
    idx = BM25Index.load(args.index)
    tagger = Tagger()
    tag = lambda q: tagger([q])[0]
    hyb = HybridRetriever.load(idx, args.map, tag)
    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8")]
    qvecs = hyb.enc.encode(tagger([q["query"] for q in queries]))
    dev = [q for i, q in enumerate(queries) if (i // 2) % 2 == 0]      # queries alternate title/body: split by pairs
    test = [q for i, q in enumerate(queries) if (i // 2) % 2 == 1]
    qpos = {q["query"]: i for i, q in enumerate(queries)}
    rules = [("bm25", None, None)] + [(f"w={w}{' guard' if g else ''}", w, g) for w in (0.5, 1.0, 2.0, 3.0) for g in (False, True)]
    table = {}
    for name, w, g in rules:
        if w is None:
            fn = lambda q, k: [idx.titles[i] for _, i in idx.search(q, k)]
        else:
            hyb.weight, hyb.guard = w, g
            fn = lambda q, k: [idx.titles[i] for _, i in hyb.search(q, k, qvec=qvecs[qpos[q]])]
        table[name] = {"dev": recall_at_k(fn, dev), "test": recall_at_k(fn, test)}
        d, t = table[name]["dev"], table[name]["test"]
        log(f"{name:12} dev all/title/body {d['all']['recall@1']:.3f}/{d['title']['recall@1']:.3f}/{d['body']['recall@1']:.3f}   "
            f"test {t['all']['recall@1']:.3f}/{t['title']['recall@1']:.3f}/{t['body']['recall@1']:.3f}")
    best = max((n for n in table if n != "bm25"), key=lambda n: table[n]["dev"]["all"]["recall@1"])
    w, g = next((w, g) for n, w, g in rules if n == best)
    chosen = {"rule": best, "weight": w, "guard": g, "test": table[best]["test"], "bm25_test": table["bm25"]["test"]}
    json.dump({"weight": w, "guard": g, "rule": best}, open(os.path.join(args.map, "hybrid.json"), "w"), indent=1)
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    json.dump({"chosen": chosen, "table": table}, open(args.results, "w"), indent=1)
    log(f"\n[hybrid] chosen on dev: {best}; on the held-out half: all {chosen['test']['all']['recall@1']:.3f} vs BM25 {chosen['bm25_test']['all']['recall@1']:.3f}, "
        f"body {chosen['test']['body']['recall@1']:.3f} vs {chosen['bm25_test']['body']['recall@1']:.3f}, "
        f"title {chosen['test']['title']['recall@1']:.3f} vs {chosen['bm25_test']['title']['recall@1']:.3f}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
