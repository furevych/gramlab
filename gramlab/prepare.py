"""Prepare token shards for training.

  python -m gramlab.prepare --config configs/tiny.json             # all corpora in the config (+ eval corpus)
  python -m gramlab.prepare --config configs/tiny.json --corpus my_text   # add one new corpus later
  python -m gramlab.prepare --config configs/tiny.json --force     # redo

The tokenizer is trained once (from corpora with "tokenizer_source": true, else the first train
corpus) and saved under work/<tok>/tokenizer. Each corpus becomes work/<tok>/shards/<corpus>/:
  tokens.bin (uint16/uint32), upos.bin, chunk.bin (uint8, 255 = no tag), meta.json
Shards are incremental: an already-prepared corpus is skipped unless --force.
"""
import argparse, json, os, sys, time
import numpy as np
from .config import load_config, tokenizer_dir, shard_dir
from .data.registry import load_registry, iter_docs
from .tokenizer import build_tokenizer, load_tokenizer, EOS

UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
             "PUNCT", "SCONJ", "SYM", "VERB", "X", "EOS"]
CHUNK_TYPES = ["NP", "VP", "PP", "ADJP", "ADVP", "SBAR", "PRT", "INTJ"]
CHUNK_TAGS = ["O"] + [f"{p}-{t}" for t in CHUNK_TYPES for p in ("B", "I")] + ["EOS"]
UPOS_ID = {t: i for i, t in enumerate(UPOS_TAGS)}
CHUNK_ID = {t: i for i, t in enumerate(CHUNK_TAGS)}
NO_TAG = 255


def prepare_corpus(name, cfg, registry, tok, force=False):
    out = shard_dir(cfg, name)
    meta_path = os.path.join(out, "meta.json")
    if os.path.exists(meta_path) and not force:
        print(f"[prepare] {name}: already prepared, skipping (use --force to redo)")
        return json.load(open(meta_path))
    os.makedirs(out, exist_ok=True)
    dtype = np.uint16 if len(tok) < 65535 else np.uint32
    t0 = time.time()
    n_tok = n_words = n_bytes = n_docs = 0
    with open(os.path.join(out, "tokens.bin"), "wb") as ft, open(os.path.join(out, "upos.bin"), "wb") as fu, \
            open(os.path.join(out, "chunk.bin"), "wb") as fc:
        buf_t, buf_u, buf_c = [], [], []
        for doc in iter_docs(name, registry):
            n_docs += 1
            untagged = doc and all(not s.get("upos") for s in doc)
            if untagged and hasattr(tok, "encode_batch"):
                # fast path for plain text: one Rust batch call per document, no tag alignment needed
                texts = [" ".join(s["forms"]) for s in doc]
                for words, ids in zip((s["forms"] for s in doc), tok.encode_batch(texts)):
                    buf_t.extend(ids); buf_t.append(EOS)
                    buf_u.extend([NO_TAG] * (len(ids) + 1)); buf_c.extend([NO_TAG] * (len(ids) + 1))
                    n_tok += len(ids) + 1; n_words += len(words)
                    n_bytes += len(" ".join(words).encode("utf-8")) + 1
                if len(buf_t) > 1_000_000:
                    np.asarray(buf_t, dtype=dtype).tofile(ft); np.asarray(buf_u, dtype=np.uint8).tofile(fu)
                    np.asarray(buf_c, dtype=np.uint8).tofile(fc); buf_t, buf_u, buf_c = [], [], []
                continue
            for s in doc:
                words = s["forms"]
                ids, widx = tok.encode_words(words)
                up = s.get("upos"); ch = s.get("chunk")
                buf_t.extend(ids); buf_t.append(EOS)
                if up:
                    buf_u.extend(UPOS_ID.get(up[j], UPOS_ID["X"]) for j in widx); buf_u.append(UPOS_ID["EOS"])
                else:
                    buf_u.extend([NO_TAG] * (len(ids) + 1))
                if ch:
                    buf_c.extend(CHUNK_ID.get(ch[j], 0) for j in widx); buf_c.append(CHUNK_ID["EOS"])
                else:
                    buf_c.extend([NO_TAG] * (len(ids) + 1))
                n_tok += len(ids) + 1; n_words += len(words)
                n_bytes += len(" ".join(words).encode("utf-8")) + 1
            if len(buf_t) > 1_000_000:
                np.asarray(buf_t, dtype=dtype).tofile(ft); np.asarray(buf_u, dtype=np.uint8).tofile(fu)
                np.asarray(buf_c, dtype=np.uint8).tofile(fc); buf_t, buf_u, buf_c = [], [], []
        np.asarray(buf_t, dtype=dtype).tofile(ft); np.asarray(buf_u, dtype=np.uint8).tofile(fu)
        np.asarray(buf_c, dtype=np.uint8).tofile(fc)
    meta = {"corpus": name, "tokenizer": cfg["tokenizer"]["name"], "dtype": np.dtype(dtype).name, "n_tokens": n_tok,
            "n_words": n_words, "n_bytes": n_bytes, "n_docs": n_docs, "tokens_per_word": n_tok / max(n_words, 1),
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"[prepare] {name}: {n_docs} docs, {n_words:,} words -> {n_tok:,} tokens ({meta['tokens_per_word']:.2f}/word) in {meta['seconds']}s")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", action="append", help="prepare only these corpora (repeatable)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="override config, e.g. --set tokenizer.type=word")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    registry = load_registry(cfg["registry"])
    tdir = tokenizer_dir(cfg)
    if os.path.exists(os.path.join(tdir, "tokenizer.json")) and not args.force:
        tok = load_tokenizer(tdir)
        print(f"[prepare] tokenizer {cfg['tokenizer']['name']}: loaded ({len(tok)} types)")
    else:
        sources = [n for n, e in registry.items() if e.get("tokenizer_source")] or [cfg["train_corpora"][0]["name"]]
        print(f"[prepare] training tokenizer {cfg['tokenizer']['name']} on {sources} ...")
        tok = build_tokenizer(cfg["tokenizer"], lambda: (d for n in sources for d in iter_docs(n, registry)))
        tok.save(tdir)
        print(f"[prepare] tokenizer saved: {len(tok)} types -> {tdir}")
    names = args.corpus or ([c["name"] for c in cfg["train_corpora"]] + [cfg["eval_corpus"]])
    for n in names:
        if registry[n]["type"] == "dialogue_jsonl":
            print(f"[prepare] {n}: dialogue data is tokenized by finetune.py, skipping")
            continue
        prepare_corpus(n, cfg, registry, tok, force=args.force)


if __name__ == "__main__":
    main()
