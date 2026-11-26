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
_LEAD = re.compile(r"^(?P<title>.{1,160}?)\s+(?:is|was|were|are)\s+(?:(?:a|an|the|any|one of the|the name of a|a type of|a kind of|a form of|a species of|a genus of)\s+)?(?P<np>[A-Za-z][^.,;:(]{1,80})")
_PAREN = re.compile(r"\s*\([^()]*\)")      # "Russia (Russian: Россия, romanized: Rossiya), or the Russian Federation, is a country"
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]*")
NAME_HEADS = {"name", "surname", "forename", "given", "nickname", "patronymic"}   # "Jamie is a unisex given name" → person
_norm = re.compile(r"[^a-z0-9 ]+")


def normalize(text):
    t = _norm.sub(" ", text.lower().replace("’", "'"))
    t = re.sub(r"^(?:the|a|an)\s+", "", " ".join(t.split()))      # "the United States" → "united states"
    return t


STOP_NP = {"of", "in", "for", "from", "by", "on", "at", "with", "to", "that", "which", "who", "whose", "and", "or", "but",
           "located", "based", "known", "founded", "born", "named", "used", "made", "played", "situated", "headquartered"}


def head_noun(np_text, wn):
    """Head of the first noun phrase: the last WordNet noun in the initial run of words before a preposition,
    conjunction, relative pronoun or participle ('American multinational technology company headquartered in'
    → company; 'country spanning Eastern Europe and North Asia' → country)."""
    words = []
    for w in np_text.split():
        lw = w.lower().strip("'\"")
        if lw in STOP_NP:
            break
        is_noun = bool(_WORD.fullmatch(lw)) and bool(wn.synsets(lw, "n"))
        if not is_noun and (lw.endswith("ing") or lw.endswith("ed")) and words:
            break                                   # a participle after at least one word closes the phrase
        words.append(lw)
    for w in reversed(words):
        if _WORD.fullmatch(w) and wn.synsets(w, "n"):
            return w
    return None


# hypernym anchors: a head noun is typed by the first anchor reached walking up from its top-k senses
ANCHORS = [("person", "person.n.01"), ("place", "location.n.01"), ("place", "geological_formation.n.01"), ("place", "body_of_water.n.01"),
           ("organization", "organization.n.01"), ("organization", "group.n.01"), ("animal", "animal.n.01"), ("plant", "plant.n.02"),
           ("substance", "substance.n.01"), ("substance", "food.n.01"), ("time", "time_period.n.01"), ("time", "time_unit.n.01"),
           ("event", "event.n.01"), ("event", "act.n.02"), ("object", "artifact.n.01"), ("object", "natural_object.n.01"),
           ("abstract", "abstraction.n.06"), ("abstract", "psychological_feature.n.01")]
_anchor_cache = {}


def _anchor_offsets(wn):
    key = id(wn)
    if key not in _anchor_cache:
        table = {}
        for t, name in ANCHORS:
            lemma, _, num = name.rsplit(".", 2)[0], None, int(name.rsplit(".", 1)[1])
            ss = wn.synsets(lemma, "n")
            if len(ss) >= num:
                table[ss[num - 1]] = t
        _anchor_cache[key] = table
    return _anchor_cache[key]


def type_of_head(head, wn, k=3, depth=12):
    """Type by hypernym closure: from each of the head noun's first k senses walk up the hypernym chain and take the
    first anchor met (person.n.01 → person, location.n.01 → place, organization/group → organization, artifact →
    object, event/act → event …). Senses vote; a geographic reading wins ties (countries are places even where the
    first sense is the political group)."""
    anchors = _anchor_offsets(wn)
    votes = Counter()
    first = None
    for rank, off in enumerate(wn.synsets(head, "n")[:k]):
        cur, seen = [off], set()
        found = None
        for _ in range(depth):
            nxt = []
            for o in cur:
                if o in anchors:
                    found = anchors[o]
                    break
                for h in wn.hypernyms("n", o):
                    if h not in seen:
                        seen.add(h)
                        nxt.append(h)
            if found or not nxt:
                break
            cur = nxt
        if found:
            votes[found] += k - rank                  # first sense weighs most
            first = first or found
    if not votes:
        return None
    top = votes.most_common(1)[0][0]
    if top == "organization" and "place" in votes:   # country/state/city: the political group vs the territory → place
        return "place"
    return top


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
            plain = 1 if bare == title else 0                 # "Russia" vs "Russia (song)": plain titles win at load time
            lead = _PAREN.sub("", text[:600])
            lead = _PAREN.sub("", lead)                        # nested once more
            m = _LEAD.match(lead)
            if not m:
                continue
            head = head_noun(m.group("np"), wn)
            if not head:
                continue
            t = "person" if head in NAME_HEADS else type_of_head(head, wn)
            if not t:
                continue
            hit += 1
            counts[t] += 1
            clean = 1 if plain and re.fullmatch(r"[A-Za-z0-9 \-']+", bare) else 0     # "Russia" outranks "Russia!"
            out.write(f"{normalize(bare)}\t{t}\t{plain + clean}\t{title}\n")
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
        """Several articles normalize to one name ("Russia", "Russia (song)", "Russia (horse)"): the plain title's
        type wins; without a plain title, the majority over the disambiguated ones."""
        opener = gzip.open if str(path).endswith(".gz") else open
        best, votes = {}, {}            # best: name -> (rank, type, title); rank 2 = clean plain title, 1 = plain, 0 = disambiguated
        self.source = {}
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 2:
                    continue
                name, t = p[0], p[1]
                rank = int(p[2]) if len(p) > 2 and p[2].isdigit() else 0
                title = p[3] if len(p) > 3 else ""
                if rank > 0:
                    if name not in best or rank > best[name][0]:
                        best[name] = (rank, t, title)
                else:
                    votes.setdefault(name, Counter())[(t, title)] += 1
        self.table = {}
        for n, c in votes.items():
            (t, title), _ = c.most_common(1)[0]
            self.table[n] = t
            self.source[n] = title
        for n, (rank, t, title) in best.items():
            self.table[n] = t
            self.source[n] = title
        return self

    def why(self, text):
        return self.source.get(normalize(text))

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
