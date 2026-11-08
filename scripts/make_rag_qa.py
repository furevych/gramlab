"""Build a retrieval-conditioned Q&A set: the user turn carries a passage, the answer is read off it.

  python scripts/make_rag_qa.py --wiki data/simplewiki.jsonl --index work/bm25 --out data/rag_qa.jsonl --n 20000

A 25M-parameter model cannot memorise Wikipedia, but it can learn to *read*: given
    Context: <title>: <lead paragraph>
    Question: Tell me about <title>.
it answers with the first one or two sentences of the passage. Three kinds of example, so the
model learns to use the passage rather than guess:
  gold        (55%)  the article's own lead paragraph                          -> its opening sentence(s)
  gold+noise  (25%)  the gold passage and a BM25 near-miss, in random order    -> the gold answer
  miss        (20%)  only a near-miss / unrelated passage                      -> "I don't have information about <title>."
Questions use neutral templates (no person/place guessing). Passages are cut at sentence boundaries so
everything fits the model's 256-token context; finetune.py trims the context further if needed.
At chat time, `python -m gramlab.chat --retrieve work/bm25` reproduces exactly this format.
"""
import argparse, json, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gramlab.retrieval import BM25Index, cut_words, format_rag_user

TEMPLATES = ["Tell me about {t}.", "What is {t}?", "What do you know about {t}?", "Describe {t}.",
             "What can you tell me about {t}?", "Can you tell me something about {t}?", "{t}?", "Explain {t}."]
MISS_ANSWER = "I don't have information about {t}."
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def lead_paragraph(text, min_words):
    for para in text.split("\n"):
        para = para.strip()
        if len(para.split()) >= min_words:
            return para
    return None


def answer_from(passage, max_sents=2, lo=6, hi=45):
    """Opening sentence(s) of the passage, if they look like clean prose; else None."""
    sents = _SENT.split(passage)
    out = []
    for s in sents[:max_sents]:
        w = s.split()
        if not (lo <= len(w) <= hi) or not s[:1].isupper() or s[-1] not in ".!?":
            break
        out.append(s)
    return " ".join(out) if out else None


def clean_title(title):
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", action="append", required=True, help="title/text jsonl (repeatable); articles are sampled from all of them")
    ap.add_argument("--index", default=None, help="BM25 index dir for near-miss passages (optional; random passages otherwise)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000, help="articles to sample")
    ap.add_argument("--ctx_words", type=int, default=90, help="passage length in words (halved when two passages are shown)")
    ap.add_argument("--min_words", type=int, default=25)
    ap.add_argument("--mix", default="0.55,0.25,0.20", help="gold, gold+noise, miss proportions")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    p_gold, p_noise, p_miss = (float(x) for x in args.mix.split(","))

    articles = []
    for path in args.wiki:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                lead = lead_paragraph(r["text"], args.min_words)
                if lead and "may refer to" not in lead[:200]:
                    articles.append((r["title"], lead))
    print(f"[rag_qa] {len(articles):,} usable articles in {args.wiki}")
    rng.shuffle(articles)
    index = BM25Index.load(args.index) if args.index else None

    def near_miss(question, title, k=4):
        """A passage that BM25 would plausibly return for this question but that is not the article itself."""
        if index is not None:
            for _, i in index.search(question, k=k):
                p = index.passage(i)
                if clean_title(p["title"]).lower() != clean_title(title).lower():
                    return p
        t2, lead2 = rng.choice(articles)
        return {"title": t2, "text": lead2}

    counts = {"gold": 0, "gold+noise": 0, "miss": 0, "skipped": 0}
    with open(args.out, "w", encoding="utf-8") as out:
        for title, lead in articles[:args.n]:
            t = clean_title(title)
            if not t:
                counts["skipped"] += 1; continue
            q = rng.choice(TEMPLATES).format(t=t)
            r = rng.random()
            variant = "gold" if r < p_gold else "gold+noise" if r < p_gold + p_noise else "miss"
            words = args.ctx_words if variant == "gold" else args.ctx_words // 2
            gold = cut_words(lead, words)
            answer = answer_from(gold)
            if variant != "miss" and answer is None:
                answer = answer_from(gold, max_sents=1)
                if answer is None:
                    counts["skipped"] += 1; continue
            if variant == "gold":
                ctxs = [{"title": title, "text": gold}]
            elif variant == "gold+noise":
                other = near_miss(q, title)
                other = {"title": other["title"], "text": cut_words(other["text"], words)}
                ctxs = [{"title": title, "text": gold}, other]
                rng.shuffle(ctxs)
            else:
                other = near_miss(q, title)
                ctxs = [{"title": other["title"], "text": cut_words(other["text"], args.ctx_words)}]
                answer = MISS_ANSWER.format(t=t)
            out.write(json.dumps({"messages": [{"role": "user", "content": format_rag_user(ctxs, q)},
                                               {"role": "assistant", "content": answer}],
                                  "source": title, "variant": variant}, ensure_ascii=False) + "\n")
            counts[variant] += 1
    print(f"[rag_qa] wrote {sum(v for k, v in counts.items() if k != 'skipped'):,} dialogues -> {args.out}  {counts}")


if __name__ == "__main__":
    main()
