"""Teacher sentence embeddings: the regression targets for the grammar meaning map, plus a dense-retrieval baseline.

  python scripts/teacher_embed.py --index work/bm25 --amalgum external/amalgum/amalgum --out work/teacher \
      --queries work/bm25/eval_queries.jsonl --scoreboard results/retrieval_dense.json --stsb data/stsb.jsonl

Writes (all optional, each stage is skipped if its input is missing or its library fails to import):
  work/teacher/passages.npy          float16 [n_passages x dim]  teacher embedding of every BM25 passage ("title: text")
  work/teacher/amalgum_sentences.jsonl  one parsed AMALGUM sentence per line: forms, upos, chunk tags, doc/sent ids
  work/teacher/amalgum_sent.npy      float16 [n_sentences x dim]  teacher embedding of each of those sentences
  results/retrieval_dense.json       recall@k of the teacher as a retriever on the same queries as the BM25 scoreboard
  data/stsb.jsonl                    STS-B sentence pairs with gold similarity (the map's similarity test), and the
                                     teacher's own Spearman on them in results/stsb_teacher.json
The map (next stage, CPU) learns 32-dim slot codes from the parsed sentences so that summed codes regress onto these
targets; its score is then compared with the teacher's on STS-B and on the retrieval scoreboard.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gramlab.retrieval import recall_at_k, clean_title
from gramlab.tokenizer import detokenize


def log(*a):
    print("[teacher]", *a, flush=True)


def load_model(name):
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(name, device=dev)
    log(f"{name} on {dev}, dim {m.get_sentence_embedding_dimension()}")
    return m


def encode(model, texts, batch=256):
    return model.encode(texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


def embed_passages(model, index_dir, out_dir):
    path = os.path.join(index_dir, "passages.jsonl")
    titles, texts = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            titles.append(r["title"]); texts.append(f"{r['title']}: {r['text']}")
    t0 = time.time()
    emb = encode(model, texts).astype(np.float16)
    np.save(os.path.join(out_dir, "passages.npy"), emb)
    log(f"{len(texts):,} passages embedded in {time.time()-t0:.0f}s -> {out_dir}/passages.npy {emb.shape}")
    return titles, emb


def dense_scoreboard(model, titles, emb, queries_path, out_path, ks=(1, 5, 10)):
    import torch
    queries = [json.loads(l) for l in open(queries_path, encoding="utf-8")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = torch.from_numpy(emb.astype(np.float32)).to(dev)
    Q = torch.from_numpy(encode(model, [q["query"] for q in queries]).astype(np.float32)).to(dev)
    kmax = max(ks)
    top = []
    for s in range(0, Q.shape[0], 512):
        top.append(torch.topk(Q[s:s + 512] @ P.T, kmax, dim=1).indices.cpu().numpy())
    top = np.concatenate(top)
    ranked = {q["query"]: [titles[i] for i in row] for q, row in zip(queries, top)}
    res = recall_at_k(lambda q, k: ranked[q][:k], queries, ks)
    res["system"] = "dense teacher"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)
    log("dense retrieval scoreboard:", json.dumps({k: v for k, v in res.items() if k != "system"}))


def embed_amalgum(model, amalgum_dir, out_dir):
    from gramlab.data.registry import iter_docs
    registry = {"amalgum": {"type": "ud_conllu_dir", "path": amalgum_dir}}
    texts = []
    t0 = time.time()
    with open(os.path.join(out_dir, "amalgum_sentences.jsonl"), "w", encoding="utf-8") as fh:
        for di, doc in enumerate(iter_docs("amalgum", registry)):
            for si, s in enumerate(doc):
                fh.write(json.dumps({"doc": di, "sent": si, "forms": s["forms"], "upos": s["upos"], "chunk": s["chunk"]},
                                    ensure_ascii=False) + "\n")
                texts.append(detokenize(s["forms"]))
    log(f"{len(texts):,} parsed AMALGUM sentences written ({time.time()-t0:.0f}s); embedding ...")
    emb = encode(model, texts).astype(np.float16)
    np.save(os.path.join(out_dir, "amalgum_sent.npy"), emb)
    log(f"-> {out_dir}/amalgum_sent.npy {emb.shape} ({time.time()-t0:.0f}s)")


def stsb(model, out_path, result_path):
    from datasets import load_dataset
    try:
        ds = load_dataset("sentence-transformers/stsb")
        rows = [(r["sentence1"], r["sentence2"], float(r["score"]) * 5) for split in ds for r in ds[split]]
    except Exception:
        ds = load_dataset("mteb/stsbenchmark-sts")
        rows = [(r["sentence1"], r["sentence2"], float(r["score"])) for split in ds for r in ds[split]]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for s1, s2, sc in rows:
            fh.write(json.dumps({"s1": s1, "s2": s2, "score": sc}, ensure_ascii=False) + "\n")
    from scipy.stats import spearmanr
    a = encode(model, [r[0] for r in rows]); b = encode(model, [r[1] for r in rows])
    sims = (a * b).sum(axis=1)
    rho = float(spearmanr(sims, [r[2] for r in rows]).correlation)
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    json.dump({"pairs": len(rows), "teacher_spearman": round(rho, 4)}, open(result_path, "w"), indent=1)
    log(f"STS-B: {len(rows):,} pairs -> {out_path}; teacher Spearman {rho:.4f} (the map's number to approach)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--index", default=None, help="BM25 index dir with passages.jsonl")
    ap.add_argument("--queries", default=None, help="eval_queries.jsonl from `gramlab.retrieval eval`")
    ap.add_argument("--scoreboard", default="results/retrieval_dense.json")
    ap.add_argument("--amalgum", default=None, help="AMALGUM conllu root (external/amalgum/amalgum)")
    ap.add_argument("--stsb", default=None, help="write STS-B pairs here (needs the `datasets` library + internet)")
    ap.add_argument("--out", default="work/teacher")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.model)
    json.dump({"model": args.model, "dim": model.get_sentence_embedding_dimension(), "normalized": True},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    if args.index and os.path.exists(os.path.join(args.index, "passages.jsonl")):
        titles, emb = embed_passages(model, args.index, args.out)
        if args.queries and os.path.exists(args.queries):
            try:
                dense_scoreboard(model, titles, emb, args.queries, args.scoreboard)
            except Exception as e:
                log("dense scoreboard failed:", repr(e))
    if args.amalgum and os.path.isdir(args.amalgum):
        try:
            embed_amalgum(model, args.amalgum, args.out)
        except Exception as e:
            log("AMALGUM embedding failed:", repr(e))
    if args.stsb:
        try:
            stsb(model, args.stsb, os.path.join(os.path.dirname(args.scoreboard) or ".", "stsb_teacher.json"))
        except Exception as e:
            log("STS-B failed:", repr(e))


if __name__ == "__main__":
    main()
