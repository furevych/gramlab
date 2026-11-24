"""core-v1 · layer 0 · a dependency-free WordNet 3.0 reader.

Reads the `dict/` files (index.{noun,verb,adj,adv}, data.*, lexnames) straight from the WordNet
distribution or the nltk `wordnet.zip`, so layer 0 needs no nltk.  Gives what the GeneralIndex needs:

    synsets(lemma, pos) -> [offset, ...]   in WordNet's sense order (first = most frequent)
    lemmas(pos, offset)  -> [names]
    hypernyms(pos, offset) -> [offsets]    ('@' and '@i' pointers)
    lexname(pos, offset) -> 'verb.contact' ...  (the 45 lexicographer files: a coarse, fixed family)
    name(pos, offset)    -> 'wash.v.01'-style name

Point it at a directory with the files (WORDNET_DIR env var or `WordNet(path)`), or at the zip.
"""
from __future__ import annotations

import os
import zipfile
from functools import lru_cache
from pathlib import Path

POS_FILES = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}


def find_wordnet_dir():
    for cand in (os.environ.get("WORDNET_DIR"), "data/wordnet", "wordnet", os.path.expanduser("~/nltk_data/corpora/wordnet"),
                 "/usr/share/nltk_data/corpora/wordnet", "/kaggle/working/data/wordnet", "/usr/share/wordnet"):
        if cand and Path(cand, "index.verb").exists():
            return Path(cand)
    return None


class WordNet:
    def __init__(self, path=None):
        path = Path(path) if path else find_wordnet_dir()
        if path is None:
            raise FileNotFoundError("WordNet dict directory not found (set WORDNET_DIR or pass a path)")
        if path.suffix == ".zip":
            out = path.with_suffix("")
            if not (out / "index.verb").exists() and not (out / "wordnet" / "index.verb").exists():
                zipfile.ZipFile(path).extractall(out)
            path = out / "wordnet" if (out / "wordnet" / "index.verb").exists() else out
        self.path = path
        self.index = {}          # (pos, lemma) -> [offsets]
        self.data = {}           # (pos, offset) -> (lexfile, [lemmas], [(ptr, offset, pos)])
        self.lexnames = {}
        with open(path / "lexnames", encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    self.lexnames[int(p[0])] = p[1]
        for pos, fn in POS_FILES.items():
            with open(path / f"index.{fn}", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(" "):
                        continue
                    p = line.split()
                    lemma, n_ptr = p[0], int(p[3])
                    n_syn = int(p[2])
                    offsets = p[4 + n_ptr + 2: 4 + n_ptr + 2 + n_syn]
                    self.index[pos, lemma] = [int(o) for o in offsets]
            with open(path / f"data.{fn}", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(" "):
                        continue
                    body, _, gloss = line.partition("|")
                    head = body.split()
                    off, lexf = int(head[0]), int(head[1])
                    n_words = int(head[3], 16)
                    words = [head[4 + 2 * i] for i in range(n_words)]
                    i = 4 + 2 * n_words
                    n_ptr = int(head[i])
                    ptrs = []
                    for k in range(n_ptr):
                        sym, poff, ppos = head[i + 1 + 4 * k], head[i + 2 + 4 * k], head[i + 3 + 4 * k]
                        ptrs.append((sym, int(poff), ppos))
                    self.data[pos, off] = (lexf, words, ptrs, gloss.strip())
        self.sensekeys = {}
        sk = path / "index.sense"
        if sk.exists():
            with open(sk, encoding="utf-8") as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 2:
                        self.sensekeys[p[0]] = int(p[1])

    def synsets(self, lemma, pos):
        return self.index.get((pos, lemma.lower().replace(" ", "_")), [])

    def lemmas(self, pos, off):
        return [w.replace("_", " ") for w in self.data[pos, off][1]]

    def hypernyms(self, pos, off):
        return [o for sym, o, p in self.data[pos, off][2] if sym in ("@", "@i") and p == pos]

    def lexname(self, pos, off):
        return self.lexnames.get(self.data[pos, off][0], str(self.data[pos, off][0]))

    def name(self, pos, off):
        """'lemma.pos.NN' like nltk: first lemma + sense number among that lemma's synsets."""
        lemma = self.data[pos, off][1][0]
        try:
            k = self.synsets(lemma, pos).index(off) + 1
        except ValueError:
            k = 0
        return f"{lemma}.{pos}.{k:02d}"

    def family(self, pos, off, depth):
        """Hypernym `depth` steps up (first path), or the top of the chain."""
        o = off
        for _ in range(depth):
            h = self.hypernyms(pos, o)
            if not h:
                break
            o = h[0]
        return o

    def gloss(self, pos, off):
        return self.data[pos, off][3]

    def offset_of_sensekey(self, key):
        return self.sensekeys.get(key)

    def stats(self):
        return {"lemmas": len(self.index), "synsets": len(self.data), "path": str(self.path)}
