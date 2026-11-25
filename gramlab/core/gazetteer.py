"""core-v1 · layer 0 · gazetteer — entity types for names, derived from Wikipedia lead sentences.

No hand list: for an article "Belgrade", the lead "Belgrade is the capital and largest city of Serbia" gives the
head noun after *is a / is the* → "capital"/"city" → WordNet lexicographer class noun.location → type `place`.
The mapping from the 26 noun lexicographer classes to the ten entity types is the only authored table (a dozen
lines). Built once from the full-Wikipedia `passages.jsonl` (4.96M leads, Kaggle), saved as a tab file
`name<TAB>type`, loaded as a dict. The typer uses it as a feature (`gaz=<type>`), never as an override, so a wrong
lead reading can be outvoted by context.

    python -m gramlab.core.gazetteer build --passages work/bm25full/passages.jsonl --out data/gazetteer.tsv
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from collections import Counter
from pathlib import Path

LEX_TO_TYPE = {
    "noun.person": "person", "noun.location": "place", "noun.group": "organization",
    "noun.artifact": "object", "noun.object": "place", "noun.body": "object", "noun.possession": "object",
    "noun.event": "event", "noun.act": "event", "noun.process": "event", "noun.phenomenon": "event",
    "noun.time": "time", "noun.substance": "substance", "noun.food": "substance",
    "noun.animal": "animal", "noun.plant": "plant",
    "noun.cognition": "abstract", "noun.communication": "abstract", "noun.attribute": "abstract",
    "noun.state": "abstract", "noun.feeling": "abstract", "noun.relation": "abstract", "noun.quantity": "abstract",
    "noun.shape": "abstract", "noun.motive": "abstract", "noun.Tops": "abstract",
}
# lead-sentence pattern: "<title> (is|was|were|are) (a|an|the|one of the) [modifiers] <head noun> ..."
_LEAD = re.compile(r"^(?P<title>.{1,80}?)\s+(?:is|was|were|are)\s+(?:a|an|the|one of the|the name of a|a type of|a kind of)\s+(?P<np>[^.,;:(]{1,80})")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]*")
_norm = re.compile(r"[^a-z0-9 ]+")


def normalize(text):
    t = _norm.sub(" ", text.lower().replace("’", "'"))
    return " ".join(t.split())


def head_noun(np_text, wn):
    """The last word of the noun phrase that WordNet knows as a noun (skips adjectives like 'American',
    'largest'; stops at a preposition)."""
    words = []
    for w in np_text.split():
        lw = w.lower().strip("'\"")
        if lw in ("of", "in", "for", "from", "by", "on", "at", "with", "that", "which", "who", "located", "based", "known", "founded", "born"):
            break
        words.append(lw)
    for w in reversed(words):
        if _WORD.fullmatch(w) and wn.synsets(w, "n"):
            return w
    return None


def type_of_head(head, wn, k=3):
    """Type from the lexicographer classes of the head noun's first k senses: a geographic reading (noun.location,
    noun.object) anywhere in the top k wins (countries, villages and rivers are places even where WordNet's first
    sense is the political group or the natural object), then person, then the majority."""
    types = [LEX_TO_TYPE.get(wn.lexname("n", off)) for off in wn.synsets(head, "n")[:k]]
    types = [t for t in types if t]
    if not types:
        return None
    if "place" in types:
        return "place"
    if "person" in types:
        return "person"
    return Counter(types).most_common(1)[0][0]


def build(passages_path, out_path, wn, limit=None, log_every=500000):
    t0 = time.time()
    opener = gzip.open if str(passages_path).endswith(".gz") else open
    n = hit = 0
    counts = Counter()
    with opener(passages_path, "rt", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            n += 1
            if limit and n > limit:
                break
            try:
                r = json.loads(line)
            except ValueError:
                continue
            title = r.get("title", "")
            text = r.get("text", "")
            if not title or not text:
                continue
            bare = re.sub(r"\s*\([^)]*\)\s*$", "", title)
            m = _LEAD.match(text[:400])
            if not m:
                continue
            head = head_noun(m.group("np"), wn)
            if not head:
                continue
            t = type_of_head(head, wn)
            if not t:
                continue
            hit += 1
            counts[t] += 1
            out.write(f"{normalize(bare)}\t{t}\n")
            if n % log_every == 0:
                print(f"[gazetteer] {n:,} passages, {hit:,} typed, {time.time()-t0:.0f}s", flush=True)
    print(f"[gazetteer] done: {n:,} passages, {hit:,} typed ({100*hit/max(n,1):.1f}%), types {dict(counts)}, {time.time()-t0:.0f}s")
    return hit


class Gazetteer:
    def __init__(self, path=None):
        self.table = {}
        if path and Path(path).exists():
            self.load(path)

    def load(self, path):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 2 and p[0] not in self.table:        # first (plain-title) entry wins
                    self.table[p[0]] = p[1]
        return self

    def lookup(self, text):
        return self.table.get(normalize(text))

    def __len__(self):
        return len(self.table)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--passages", required=True)
    ap.add_argument("--out", default="data/gazetteer.tsv")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    from .wordnet import WordNet
    build(a.passages, a.out, WordNet(), a.limit)


if __name__ == "__main__":
    main()
