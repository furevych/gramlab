"""core-v1 · layer 2 — the index: occurrences, postings by key, text search.

add(sentence, events)  -> occurrence ids       (keys computed by the Keyer; nothing guessed)
lookup(key, level, status) -> occurrences      (level: 'specific' | 'family')
search(text, k)       -> passages              (BM25 / BM25+slot-map hybrid from Level 1 when attached)
related(event, level) -> occurrences sharing the event's key

The posting lists are the ordered meaning map from the Nov-2026 notebooks, now behind one interface and
keyed by the registry-backed Keyer instead of a hand table.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .key import Keyer
from .records import Event, Occurrence, Sentence


class MeaningIndex:
    def __init__(self, keyer: Keyer, retriever=None):
        self.keyer = keyer
        self.retriever = retriever          # object with .search(query, k) -> [(title, text, score)] or None
        self.records: list[Occurrence] = []
        self.postings = defaultdict(list)   # (level, key, status) -> [occurrence ids]

    def add(self, s: Sentence, events: list[Event]):
        ids = []
        for e in events:
            sk, fk = self.keyer.specific_key(e), self.keyer.family_key(e)
            occ = Occurrence(e, s.text(), s.doc, s.sid, sk, fk)
            i = len(self.records)
            self.records.append(occ)
            self.postings[("specific", sk, e.status)].append(i)
            self.postings[("family", fk, e.status)].append(i)
            ids.append(i)
        return ids

    def lookup(self, key: int, level="specific", status="asserted"):
        return [self.records[i] for i in self.postings.get((level, key, status), ())]

    def related(self, e: Event, level="family", status=None):
        k = self.keyer.family_key(e) if level == "family" else self.keyer.specific_key(e)
        statuses = (status,) if status else ("asserted", "negated", "possible")
        return [self.records[i] for st in statuses for i in self.postings.get((level, k, st), ())]

    def search(self, text: str, k=5):
        if self.retriever is None:
            return []
        return self.retriever.search(text, k=k)

    def stats(self):
        return {"occurrences": len(self.records),
                "specific_keys": len({k for (lvl, k, _) in self.postings if lvl == "specific"}),
                "family_keys": len({k for (lvl, k, _) in self.postings if lvl == "family"})}

    def save(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "occurrences.jsonl", "w", encoding="utf-8") as f:
            for o in self.records:
                e = o.event
                f.write(json.dumps({"doc": o.doc, "sid": o.sid, "text": o.text, "specific_key": o.specific_key,
                                    "family_key": o.family_key, "action": e.action, "voice": e.voice,
                                    "status": e.status,
                                    "actor": e.actor.text if e.actor else None,
                                    "patient": e.patient.text if e.patient else None}) + "\n")
        (p / "keys.json").write_text(json.dumps({str(k): v for k, v in self.keyer.table.items()}, indent=0))
