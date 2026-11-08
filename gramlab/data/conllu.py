"""Minimal CoNLL-U reader (no external deps).

Yields Sentence objects with parallel token lists. Multiword-token ranges
(e.g. "1-2") and empty nodes ("8.1") are skipped, so ids are 1..n contiguous.
"""
from dataclasses import dataclass, field
from typing import List, Iterator, Optional
import re

_RANGE = re.compile(r"^\d+-\d+$")
_EMPTY = re.compile(r"^\d+\.\d+$")


@dataclass
class Sentence:
    sent_id: str = ""
    doc_id: str = ""
    genre: str = ""
    text: str = ""
    forms: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    upos: List[str] = field(default_factory=list)
    xpos: List[str] = field(default_factory=list)
    heads: List[int] = field(default_factory=list)   # 0 = root, 1-based otherwise
    deprels: List[str] = field(default_factory=list)  # full label incl. subtype

    def __len__(self):
        return len(self.forms)


def _genre_from_doc(doc_id: str) -> str:
    # GUM_bio_xxx / AMALGUM_bio_xxx -> bio
    parts = doc_id.split("_")
    return parts[1] if len(parts) >= 3 else "unknown"


def read_conllu(path: str, genre: Optional[str] = None) -> Iterator[Sentence]:
    cur = Sentence()
    doc_id = ""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if cur.forms:
                    cur.doc_id = doc_id
                    cur.genre = genre or _genre_from_doc(doc_id)
                    yield cur
                cur = Sentence()
                continue
            if line.startswith("#"):
                if line.startswith("# newdoc id = "):
                    doc_id = line[len("# newdoc id = "):].strip()
                elif line.startswith("# sent_id = "):
                    cur.sent_id = line[len("# sent_id = "):].strip()
                elif line.startswith("# text = "):
                    cur.text = line[len("# text = "):]
                continue
            cols = line.split("\t")
            if len(cols) < 8:
                continue
            tid = cols[0]
            if _RANGE.match(tid) or _EMPTY.match(tid):
                continue
            cur.forms.append(cols[1])
            cur.lemmas.append(cols[2])
            cur.upos.append(cols[3])
            cur.xpos.append(cols[4])
            try:
                cur.heads.append(int(cols[6]))
            except ValueError:
                cur.heads.append(0)
            cur.deprels.append(cols[7])
    if cur.forms:
        cur.doc_id = doc_id
        cur.genre = genre or _genre_from_doc(doc_id)
        yield cur
