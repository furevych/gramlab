"""core-v1 · records — the data contracts every component reads and writes.

Nothing here decides anything: a Sentence is what a reader or parser produced, an Event is what the
linker produced, an Occurrence is an Event stored with its source. Changing a field here is a core
version bump.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

CORE_VERSION = "core-v1"

# GUM / AMALGUM entity types. "pronoun" and "unknown" are ours (no annotation / no coreference).
ENTITY_TYPES = ("person", "place", "organization", "object", "event", "abstract", "time",
                "substance", "animal", "plant")
NOMINAL = {"NOUN", "PROPN", "PRON"}
_ENTITY_RE = re.compile(r"\(([0-9]+-)?([a-z]+)")


@dataclass
class Sentence:
    """One parsed sentence. Indices are 0-based token positions; heads use -1 for the root."""
    sid: str
    forms: list
    lemmas: list
    upos: list
    xpos: list
    heads: list
    deprels: list
    ents: list                      # entity type opened at this token (annotation), or None
    doc: str = ""
    children: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        ch = defaultdict(list)
        for i, h in enumerate(self.heads):
            if h >= 0:
                ch[h].append(i)
        self.children = ch

    def __len__(self):
        return len(self.forms)

    def text(self):
        return " ".join(self.forms)

    def dep_children(self, i, rel):
        return [c for c in self.children.get(i, ()) if self.deprels[c] == rel]

    def case_lemmas(self, i):
        return {self.lemmas[c].lower() for c in self.dep_children(i, "case")}

    def span_of(self, i):
        """Left edge of the mention headed at i (determiners, modifiers, compounds, flat names)."""
        lo = i
        hi = i
        for c in self.children.get(i, ()):
            if c < i and self.deprels[c] in ("det", "amod", "compound", "nummod", "nmod:poss", "flat", "case"):
                lo = min(lo, c)
            if c > i and self.deprels[c] in ("flat", "flat:name"):
                hi = max(hi, c)
        return lo, hi + 1

    def mention(self, i) -> "Mention":
        lo, hi = self.span_of(i)
        etype = self.ents[i]
        if etype is None:
            for j in range(lo, i):
                if self.ents[j]:
                    etype = self.ents[j]
                    break
        if etype is None:
            etype = "pronoun" if self.upos[i] == "PRON" else "unknown"
        kind = "specific" if (self.upos[i] == "PROPN" or any(self.upos[j] == "PROPN" for j in range(lo, hi))) else "general"
        if self.upos[i] == "PRON":
            kind = "pronoun"
        return Mention(head=i, start=lo, end=hi, text=" ".join(self.forms[lo:hi]),
                       lemma=self.lemmas[i].lower(), upos=self.upos[i], etype=etype, kind=kind)


@dataclass(frozen=True)
class Mention:
    """A nominal as it appears in one sentence. `kind`: general (common concept), specific (named
    individual), pronoun (needs coreference). `etype`: entity type from annotation or layer-1 typing."""
    head: int
    start: int
    end: int
    text: str
    lemma: str
    upos: str
    etype: str
    kind: str


@dataclass(frozen=True)
class Event:
    """Who did what to whom, in one sentence. Roles are Mentions or None; status is
    asserted / negated / possible; voice is active / passive."""
    verb: int
    action: str                     # verb lemma
    actor: Optional[Mention]
    patient: Optional[Mention]
    voice: str = "active"
    status: str = "asserted"


@dataclass(frozen=True)
class Occurrence:
    """An Event stored with its source: what the index keeps and the writer reads."""
    event: Event
    text: str
    doc: str
    sid: str
    specific_key: int
    family_key: int


# ----------------------------------------------------------------------------- readers
def read_conllu(path, limit=None):
    """Minimal CoNLL-U reader (GUM, AMALGUM, any UD file). Skips multiword and empty tokens."""
    sents, rows, sid, doc = [], [], "", ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id"):
                sid = line.split("=", 1)[1].strip()
            elif line.startswith("# newdoc id"):
                doc = line.split("=", 1)[1].strip()
            elif line.startswith("#"):
                continue
            elif not line:
                if rows:
                    sents.append(_make(sid, doc, rows))
                    rows = []
                    if limit and len(sents) >= limit:
                        return sents
            else:
                c = line.split("\t")
                if "-" in c[0] or "." in c[0]:
                    continue
                rows.append(c)
    if rows:
        sents.append(_make(sid, doc, rows))
    return sents


def _make(sid, doc, rows):
    ents = []
    for c in rows:
        e = None
        if len(c) > 9 and "Entity=" in c[9]:
            m = _ENTITY_RE.search(c[9].split("Entity=", 1)[1].split("|")[0])
            if m:
                e = m.group(2)
        ents.append(e)
    return Sentence(sid, [c[1] for c in rows], [c[2] for c in rows], [c[3] for c in rows],
                    [c[4] for c in rows], [int(c[6]) - 1 for c in rows], [c[7] for c in rows], ents, doc)


def read_conllu_glob(pattern, limit_sents=None, seed=0):
    import glob
    import random
    files = sorted(glob.glob(pattern, recursive=True))
    random.Random(seed).shuffle(files)
    out = []
    for p in files:
        out.extend(read_conllu(p))
        if limit_sents and len(out) >= limit_sents:
            break
    return out[:limit_sents] if limit_sents else out
