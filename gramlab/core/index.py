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
from .records import Binding, Event, Occurrence, ResolvedEvent, Sentence
from .registry import normalize_name


class MeaningIndex:
    def __init__(self, keyer: Keyer, retriever=None):
        self.keyer = keyer
        self.retriever = retriever          # object with .search(query, k) -> [(title, text, score)] or None
        self.records: list[Occurrence] = []
        self.postings = defaultdict(list)   # (level, key, status) -> [occurrence ids]

    def add(self, s: Sentence, events: list[Event], context=None):
        """context = (sents, si) lets the keyer resolve pronouns to earlier mentions. The event is resolved once;
        every posting comes from that record (specific, family, strict identity, permissive proposals)."""
        ids = []
        for e in events:
            r = self.keyer.resolve_event(e, s, context)
            keys = self.keyer.keys_from(r)
            occ = Occurrence(e, r, s.text(), s.doc, s.sid, keys["specific"][0], keys["family"][0])
            i = len(self.records)
            self.records.append(occ)
            for level in ("specific", "family", "identity", "proposal"):
                for k in keys[level]:
                    self.postings[(level, k, r.status)].append(i)
            ids.append(i)
        return ids


    # ------------------------------------------------------------------ questions
    @staticmethod
    def _same(b: Binding, given: str, given_id=None) -> bool:
        """Does the stored binding denote the given participant? Identity id when known, else the written name;
        a bare surname only matches when it is the last token of the stored name (Morata ~ Álvaro Morata)."""
        if b.state != "known":
            return False
        if given_id is not None and given_id == b.id:
            return True
        g, t = normalize_name(given), normalize_name(b.text)
        if not g or not t:
            return False
        return g == t or t.endswith(" " + g) or g.endswith(" " + t)

    def verify_in(self, sids, action: str, actor=None, patient=None):
        """BM25 proposes, the key verifies: among occurrences from the given sentences (sids, in proposal order),
        return those whose stored event states the asked one (same sense family, given participant bound)."""
        from .records import Mention
        def mention(text):
            if text in (None, "?"):
                return None
            words = text.split()
            kind = "specific" if any(w[:1].isupper() for w in words) else "general"
            return Mention(0, 0, len(words), text, words[-1].lower(), "PROPN" if kind == "specific" else "NOUN", "unknown", kind)
        q = Event(0, action, mention(actor), mention(patient), "active", "asserted")
        r = self.keyer.resolve_event(q)
        q_senses = set([r.sense, *r.alt_senses] + [self.keyer.general._name(x) for x in self.keyer.general.synsets(action, "VERB")])
        by_sid = {}
        for i, o in enumerate(self.records):
            by_sid.setdefault(o.sid, []).append(i)
        out = []
        for sid in sids:
            for i in by_sid.get(sid, ()):
                rr = self.records[i].resolved
                if not ({rr.sense, *rr.alt_senses} & q_senses):
                    continue
                if actor not in (None, "?") and not self._same(rr.actor, actor, r.actor.id if r.actor.state == "known" else None):
                    continue
                if patient not in (None, "?") and not self._same(rr.patient, patient, r.patient.id if r.patient.state == "known" else None):
                    continue
                req = rr.actor if actor == "?" else rr.patient if patient == "?" else None
                out.append({"source": self.records[i].text, "sid": sid, "status": rr.status, "occurrence": i,
                            "answer": (req.text if req is not None and req.state == "known" else "unknown") if req is not None else None,
                            "state": req.state if req is not None else None})
        return out

    def ask(self, action: str, actor=None, patient=None, status=None, limit=10, rank=None):
        """Structured question. actor/patient: "?" = requested, a string = given, None = unconstrained.
        Candidates come from the identity / proposal postings of the given parts (family postings when nothing is
        given); each candidate's stored bindings are then verified against the given participants; the requested
        role is read off the record. Returns dicts: answer (text or 'unknown'), state, source; asserted first."""
        from .records import Mention
        def mention(text):
            if text in (None, "?"):
                return None
            words = text.split()
            kind = "specific" if any(w[:1].isupper() for w in words) else "general"
            return Mention(0, 0, len(words), text, words[-1].lower(), "PROPN" if kind == "specific" else "NOUN", "unknown", kind)
        q = Event(0, action, mention(actor), mention(patient), "active", "asserted")
        r = self.keyer.resolve_event(q)
        keys = self.keyer.keys_from(r, query=True)
        given = actor not in (None, "?") or patient not in (None, "?")
        statuses = (status,) if status else ("asserted", "negated", "possible")
        cands = set()
        for level in (("identity", "proposal") if given else ("family",)):
            for k in keys[level]:
                for st in statuses:
                    cands.update(self.postings.get((level, k, st), ()))
        if given and not cands:                        # the name was read differently: fall back to class-level candidates
            for k in keys["family"]:
                for st in statuses:
                    cands.update(self.postings.get(("family", k, st), ()))
        q_senses = set([r.sense, *r.alt_senses] + [self.keyer.general._name(x) for x in self.keyer.general.synsets(action, "VERB")])
        out = []
        for i in sorted(cands):
            o = self.records[i]
            rr = o.resolved
            if actor not in (None, "?") and not self._same(rr.actor, actor, r.actor.id if r.actor.state == "known" else None):
                continue
            if patient not in (None, "?") and not self._same(rr.patient, patient, r.patient.id if r.patient.state == "known" else None):
                continue
            if not ({rr.sense, *rr.alt_senses} & q_senses):
                continue
            req = rr.actor if actor == "?" else rr.patient if patient == "?" else None
            ans = {"source": o.text, "sid": o.sid, "status": rr.status, "voice": rr.voice, "occurrence": i,
                   "actor": rr.actor.text or None, "patient": rr.patient.text or None}
            if req is not None:
                ans["answer"] = req.text if req.state == "known" else "unknown"
                ans["state"] = req.state
            out.append(ans)
        # order: asserted and bound first; among those, a caller-supplied score of the source text (e.g. BM25 of the
        # whole question against the sentence) decides, so the rest of the question's words are not thrown away
        for a in out:
            a["score"] = float(rank(a["source"], a["sid"])) if rank is not None else 0.0
        out.sort(key=lambda a: (a["status"] != "asserted", a.get("state") != "known", -a["score"], a["occurrence"]))
        return out[:limit]

    def lookup(self, key: int, level="specific", status="asserted"):
        return [self.records[i] for i in self.postings.get((level, key, status), ())]

    def related(self, e: Event, level="family", status=None):
        keys = self.keyer.keys_from(self.keyer.resolve_event(e), query=True)
        ks = keys["family"] if level == "family" else keys["specific"] if level == "specific" else keys["identity"] + keys["proposal"]
        statuses = (status,) if status else ("asserted", "negated", "possible")
        seen, out = set(), []
        for k in ks:
            for st in statuses:
                for i in self.postings.get((level, k, st), ()):
                    if i not in seen:
                        seen.add(i)
                        out.append(self.records[i])
        return out

    def search(self, text: str, k=5):
        if self.retriever is None:
            return []
        return self.retriever.search(text, k=k)

    def stats(self):
        return {"occurrences": len(self.records),
                "specific_keys": len({k for (lvl, k, _) in self.postings if lvl == "specific"}),
                "family_keys": len({k for (lvl, k, _) in self.postings if lvl == "family"}),
                "identity_keys": len({k for (lvl, k, _) in self.postings if lvl == "identity"}),
                "proposal_keys": len({k for (lvl, k, _) in self.postings if lvl == "proposal"})}

    def save(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "occurrences.jsonl", "w", encoding="utf-8") as f:
            for o in self.records:
                e = o.event
                r = o.resolved
                f.write(json.dumps({"doc": o.doc, "sid": o.sid, "text": o.text, "specific_key": o.specific_key,
                                    "family_key": o.family_key, "action": r.action, "sense": r.sense, "alt_senses": list(r.alt_senses),
                                    "voice": r.voice, "status": r.status,
                                    "actor": r.actor.__dict__, "patient": r.patient.__dict__}) + "\n")
        (p / "keys.json").write_text(json.dumps({str(k): v for k, v in self.keyer.table.items()}, indent=0))
