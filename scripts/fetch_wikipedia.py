"""Fetch clean encyclopedic text from the Wikimedia Wikipedia dump on the Hugging Face hub.

  python scripts/fetch_wikipedia.py --lang simple --out data/simplewiki.jsonl              # all of Simple English Wikipedia
  python scripts/fetch_wikipedia.py --lang en --max_articles 120000 --out data/enwiki.jsonl  # first 120k English articles (streamed)

Writes one JSON object per article: {"title": ..., "text": ...}; `corpora.json` reads it as plain_text.
Filters out stubs, disambiguation pages and list pages. Needs internet and the `datasets` library.
"""
import argparse, json, re, sys, time


def clean(text: str) -> str:
    text = re.sub(r"\n==+[^=\n]+==+\n", "\n\n", text)          # section headers that survived
    text = re.sub(r"\[\d+\]", "", text)                        # citation markers
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def keep(title: str, text: str, min_words: int) -> bool:
    if title.startswith(("List of", "Lists of", "Index of", "Outline of", "Timeline of")):
        return False
    if "may refer to" in text[:300] or "disambiguation" in title.lower():
        return False
    return len(text.split()) >= min_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="simple", help="simple | en | de | ...")
    ap.add_argument("--snapshot", default="20231101")
    ap.add_argument("--max_articles", type=int, default=0, help="0 = all")
    ap.add_argument("--min_words", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from datasets import load_dataset
    name = f"{args.snapshot}.{args.lang}"
    stream = bool(args.max_articles)
    print(f"[fetch] wikimedia/wikipedia {name} (streaming={stream}) ...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", name, split="train", streaming=stream)
    t0 = time.time(); n = kept = words = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in ds:
            n += 1
            text = clean(r["text"])
            if keep(r["title"], text, args.min_words):
                fh.write(json.dumps({"title": r["title"], "text": text}, ensure_ascii=False) + "\n")
                kept += 1; words += len(text.split())
            if kept and kept % 20000 == 0:
                print(f"[fetch] {kept:,} articles kept / {n:,} seen, {words/1e6:.1f}M words, {time.time()-t0:.0f}s", flush=True)
            if args.max_articles and kept >= args.max_articles:
                break
    print(f"[fetch] done: {kept:,} articles, {words/1e6:.1f}M words -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
