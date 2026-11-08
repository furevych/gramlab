"""Build a small synthetic dialogue set from AMALGUM documents, so the fine-tune → chat loop can be
exercised without any external dataset.

  python scripts/make_wiki_qa.py --amalgum external/amalgum/amalgum --out data/wiki_qa.jsonl

bio   : "Tell me about <title>."            -> first 1-2 declarative sentences of the article
voyage: "What should I know about <title>?" -> same
whow  : "How do I <title without 'How to'>?"-> first 2 sentences of the guide
The model learns the *format* of answering from this; the facts come from pretraining.
"""
import argparse, glob, json, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gramlab.data.conllu import read_conllu
from gramlab.tokenizer import detokenize

TEMPLATES = {
    "bio": ["Tell me about {t}.", "Who was {t}?", "What do you know about {t}?"],
    "voyage": ["What should I know about {t}?", "Tell me about {t}.", "I'm travelling to {t}. What is it like?"],
    "whow": ["How do I {t}?", "Any advice on how to {t}?", "{T}?"],
}


def doc_sentences(path):
    docs, cur, cur_id = [], [], None
    for s in read_conllu(path):
        if cur_id is not None and s.doc_id != cur_id:
            docs.append(cur); cur = []
        cur_id = s.doc_id
        cur.append(s)
    if cur:
        docs.append(cur)
    return docs


def clean(text):
    text = re.sub(r"\s*\[\s*\d+\s*\]", "", text)          # citation markers [ 1 ]
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amalgum", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_per_genre", type=int, default=700)
    args = ap.parse_args()
    rng = random.Random(3)
    out, n = open(args.out, "w", encoding="utf-8"), 0
    for genre, temps in TEMPLATES.items():
        count = 0
        for f in sorted(glob.glob(os.path.join(args.amalgum, genre, "dep", "*.conllu"))):
            for sents in doc_sentences(f):
                if len(sents) < 3:
                    continue
                title = clean(detokenize(sents[0].forms))
                if len(sents[0]) > 10 or not title:
                    continue
                body = [clean(detokenize(s.forms)) for s in sents[1:6]
                        if 6 <= len(s) <= 45 and s.forms[-1] in (".", "!", "?") and s.forms[0][:1].isupper()]
                if not body:
                    continue
                answer = " ".join(body[:2])
                t = title.rstrip(".")
                if genre == "whow":
                    t = re.sub(r"^how to\s+", "", t, flags=re.I)
                    t = t[0].lower() + t[1:] if t else t
                q = rng.choice(temps).format(t=t, T=t[0].upper() + t[1:] if t else t)
                out.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": answer}],
                                      "source": sents[0].doc_id}, ensure_ascii=False) + "\n")
                n += 1; count += 1
                if count >= args.max_per_genre:
                    break
            if count >= args.max_per_genre:
                break
        print(f"{genre}: {count} dialogues")
    out.close()
    print(f"wrote {n} dialogues -> {args.out}")


if __name__ == "__main__":
    main()
