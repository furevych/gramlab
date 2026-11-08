"""BM25 passage retrieval over the Wikipedia dumps, so a small model can answer from a retrieved
passage instead of from (non-existent) memorised facts.

  python -m gramlab.retrieval build  --wiki data/simplewiki.jsonl --wiki data/enwiki.jsonl --out work/bm25
  python -m gramlab.retrieval search --index work/bm25 --q "Tell me about Belgrade." -k 3

Index = one passage per article (its lead paragraph, cut to --max_words words at a sentence boundary),
scored with BM25 (k1=1.5, b=0.75) and the title counted twice. Stored as a scipy CSC matrix of
precomputed BM25 term weights (passages x terms), so a query is a sum of a few columns: ~1 ms for
300k passages. Pure numpy/scipy, nothing to install.

The retrieval-conditioned chat format (shared by scripts/make_rag_qa.py, finetune.py and chat.py):
    Context: <title>: <passage>
    Question: <user question>
"""
import argparse, array, json, math, os, re, sys, time, unicodedata
import numpy as np
import scipy.sparse as sp

STOPWORDS = set("""a an the of in on at to for from by with and or as is are was were be been being am this that these
those it its into than then there their they them he she his her him we our us you your i me my mine not no nor but if
about over under after before while during which who whom whose what when where why how also can could may might will
would shall should do does did done has have had having such more most very so too just only any some all each both
either neither own same other another much many s t re ve ll d tell know describe explain say give please something
anything want like mean need find show help""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def analyze(text: str):
    """lowercase, strip accents, split on non-alphanumerics, drop stopwords, crude plural stemming."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return [_stem(w) for w in _TOKEN.findall(text) if w not in STOPWORDS]


def cut_words(text: str, max_words: int) -> str:
    """First max_words words, preferring to stop at the last sentence end inside the limit."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    head = words[:max_words]
    for j in range(len(head) - 1, max(len(head) // 2, 0), -1):
        if head[j].endswith((".", "!", "?")) and not head[j].endswith(".."):
            return " ".join(head[:j + 1])
    return " ".join(head) + " ..."


def iter_wiki_passages(path, max_words=100, per_doc=1, min_words=20):
    """{"title","text"} per line (scripts/fetch_wikipedia.py output) -> (title, passage) pairs."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            n = 0
            for para in r["text"].split("\n"):
                para = para.strip()
                if len(para.split()) < min_words:
                    continue
                yield r["title"], cut_words(para, max_words)
                n += 1
                if n >= per_doc:
                    break


def format_rag_user(passages, question: str) -> str:
    """passages: list of {"title","text"}. The user turn the fine-tuned model is trained on."""
    lines = [f"Context: {p['title']}: {p['text']}" for p in passages]
    return "\n".join(lines + [f"Question: {question}"])


class BM25Index:
    def __init__(self, W, vocab, titles, offsets, passages_path, meta):
        self.W, self.vocab, self.titles, self.offsets, self.meta = W, vocab, titles, offsets, meta
        self.passages_path = passages_path
        self._fh = None

    def __len__(self):
        return len(self.titles)

    # ---------------------------------------------------------------- build / save / load
    @classmethod
    def build(cls, pairs, out_dir, k1=1.5, b=0.75, title_weight=2, log_every=50000):
        os.makedirs(out_dir, exist_ok=True)
        vocab, titles, offsets = {}, [], array.array("q")
        rows, cols, tfs = array.array("i"), array.array("i"), array.array("f")
        dl = array.array("i")
        t0 = time.time()
        with open(os.path.join(out_dir, "passages.jsonl"), "w", encoding="utf-8") as fp:
            for i, (title, text) in enumerate(pairs):
                offsets.append(fp.tell())
                fp.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")
                titles.append(title)
                toks = analyze(title) * title_weight + analyze(text)
                dl.append(len(toks))
                counts = {}
                for t in toks:
                    counts[t] = counts.get(t, 0) + 1
                for t, c in counts.items():
                    tid = vocab.get(t)
                    if tid is None:
                        tid = vocab[t] = len(vocab)
                    rows.append(i); cols.append(tid); tfs.append(c)
                if log_every and (i + 1) % log_every == 0:
                    print(f"[bm25] {i+1:,} passages, {len(vocab):,} terms, {len(rows):,} postings, {time.time()-t0:.0f}s", flush=True)
        N, V = len(titles), len(vocab)
        rows, cols, tfs, dl = (np.frombuffer(a, dtype=np.int32) if a.typecode == "i" else np.frombuffer(a, dtype=np.float32)
                               for a in (rows, cols, tfs, dl))
        df = np.bincount(cols, minlength=V).astype(np.float64)
        idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
        avgdl = float(dl.mean()) if N else 1.0
        norm = k1 * (1.0 - b + b * dl[rows] / avgdl)
        w = (idf[cols] * tfs * (k1 + 1.0) / (tfs + norm)).astype(np.float32)
        W = sp.csc_matrix((w, (rows, cols)), shape=(N, V), dtype=np.float32)
        meta = {"n_passages": N, "n_terms": V, "n_postings": int(len(w)), "k1": k1, "b": b, "title_weight": title_weight,
                "avgdl": avgdl, "seconds": round(time.time() - t0, 1)}
        idx = cls(W, vocab, titles, np.frombuffer(offsets, dtype=np.int64), os.path.join(out_dir, "passages.jsonl"), meta)
        idx.save(out_dir)
        print(f"[bm25] built {N:,} passages x {V:,} terms ({meta['n_postings']:,} postings) in {meta['seconds']}s -> {out_dir}")
        return idx

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        sp.save_npz(os.path.join(out_dir, "weights.npz"), self.W, compressed=False)
        terms = [None] * len(self.vocab)
        for t, i in self.vocab.items():
            terms[i] = t
        json.dump(terms, open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(self.titles, open(os.path.join(out_dir, "titles.json"), "w", encoding="utf-8"), ensure_ascii=False)
        np.save(os.path.join(out_dir, "offsets.npy"), np.asarray(self.offsets, dtype=np.int64))
        json.dump(self.meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)

    @classmethod
    def load(cls, d):
        W = sp.load_npz(os.path.join(d, "weights.npz")).tocsc()
        terms = json.load(open(os.path.join(d, "vocab.json"), encoding="utf-8"))
        titles = json.load(open(os.path.join(d, "titles.json"), encoding="utf-8"))
        offsets = np.load(os.path.join(d, "offsets.npy"))
        meta = json.load(open(os.path.join(d, "meta.json")))
        return cls(W, {t: i for i, t in enumerate(terms)}, titles, offsets, os.path.join(d, "passages.jsonl"), meta)

    # ---------------------------------------------------------------- query
    def search(self, query: str, k=5, exclude=()):
        """-> list of (score, passage_index), best first; empty if no query term is in the index."""
        ids = sorted({self.vocab[t] for t in analyze(query) if t in self.vocab})
        if not ids:
            return []
        scores = np.asarray(self.W[:, ids].sum(axis=1)).ravel()
        for e in exclude:
            scores[e] = 0.0
        kk = min(k, len(scores))
        top = np.argpartition(-scores, kk - 1)[:kk]
        top = top[np.argsort(-scores[top])]
        return [(float(scores[i]), int(i)) for i in top if scores[i] > 0]

    def passage(self, i: int):
        if self._fh is None:
            self._fh = open(self.passages_path, encoding="utf-8")
        self._fh.seek(int(self.offsets[i]))
        return json.loads(self._fh.readline())

    def retrieve(self, question: str, k=1, max_words=None):
        """Passages for a chat turn: top-k, optionally re-cut shorter so they fit the model's context."""
        out = []
        for score, i in self.search(question, k):
            p = self.passage(i)
            if max_words:
                p["text"] = cut_words(p["text"], max_words)
            p["score"] = round(score, 3); p["index"] = i
            out.append(p)
        return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--wiki", action="append", required=True, help="jsonl with title/text per line (repeatable)")
    b.add_argument("--out", required=True)
    b.add_argument("--max_words", type=int, default=100)
    b.add_argument("--per_doc", type=int, default=1, help="passages (paragraphs) indexed per article")
    b.add_argument("--min_words", type=int, default=20)
    b.add_argument("--max_passages", type=int, default=0, help="0 = all")
    s = sub.add_parser("search")
    s.add_argument("--index", required=True)
    s.add_argument("--q", required=True)
    s.add_argument("-k", type=int, default=3)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        def pairs():
            n = 0
            for path in args.wiki:
                for pair in iter_wiki_passages(path, args.max_words, args.per_doc, args.min_words):
                    yield pair
                    n += 1
                    if args.max_passages and n >= args.max_passages:
                        return
        BM25Index.build(pairs(), args.out)
    else:
        idx = BM25Index.load(args.index)
        t0 = time.time()
        hits = idx.retrieve(args.q, k=args.k)
        print(f"[bm25] {len(idx):,} passages, query terms {analyze(args.q)}, {1000*(time.time()-t0):.1f} ms")
        for h in hits:
            print(f"{h['score']:7.3f}  {h['title']}: {h['text'][:160]}{'...' if len(h['text']) > 160 else ''}")


if __name__ == "__main__":
    main()
