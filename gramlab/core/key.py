"""core-v1 · the ordered meaning key.

specific_key(event) = id( role-slot(actor), action concept, role-slot(patient), status )
family_key(event)   = id( class(actor), action family, class(patient) )

A role slot is (kind, class, id): kind ∈ general/specific/pronoun/missing, class = hypernym family
for general mentions or entity type for specific ones, id = synset or registry id.  Order is meaning:
(cat, wash, tom) ≠ (tom, wash, cat).  The key is a pure function of the Event and the two registries,
so it is reproducible; the index keeps the reverse table.
"""
from __future__ import annotations

from dataclasses import dataclass

from .records import Event, Mention
from .registry import GeneralIndex, SpecificIndex, stable_id

KEY_VERSION = "key-v1"


@dataclass(frozen=True)
class Slot:
    kind: str       # general / specific / pronoun / missing
    cls: str        # entity type (specific, pronoun) or concept family name (general)
    id: int         # synset id / registry id / 0

    def family_class(self):
        return self.cls


class Keyer:
    def __init__(self, general: GeneralIndex, specific: SpecificIndex, senses=3):
        self.general = general
        self.specific = specific
        self.senses = senses          # family postings: top-k senses of the action (run 9c)
        self.table = {}     # key id -> readable tuple

    def slot(self, m: Mention | None) -> Slot:
        if m is None:
            return Slot("missing", "missing", 0)
        if m.kind == "pronoun":
            # type from annotation (GUM's coreference layer) when present; live text gets "pronoun"
            # until a coreference component fills it in
            return Slot("pronoun", m.etype if m.etype not in ("unknown", "pronoun") else "pronoun", 0)
        if m.kind == "specific":
            e = self.specific.resolve(m.text, m.etype)
            return Slot("specific", m.etype, e["id"])
        c = self.general.concept(m.lemma, m.upos)
        # the class of a general mention is its entity type when annotated, else its concept family
        cls = m.etype if m.etype not in ("unknown", "pronoun") else c["family_name"]
        return Slot("general", cls, c["id"])

    def action(self, event: Event):
        return self.general.concept(event.action, "VERB")

    def specific_key(self, event: Event) -> int:
        a, p = self.slot(event.actor), self.slot(event.patient)
        act = self.action(event)
        readable = ((a.kind, a.cls, a.id), act["name"], (p.kind, p.cls, p.id), event.status)
        k = stable_id("key", *[str(x) for x in readable])
        self.table[k] = readable
        return k

    def family_key(self, event: Event) -> int:
        a, p = self.slot(event.actor), self.slot(event.patient)
        act = self.action(event)
        readable = (a.cls, act["family_name"], p.cls)
        k = stable_id("fam", *readable)
        self.table[k] = readable
        return k

    def family_keys(self, event: Event) -> list:
        """One family key per top-k sense of the action: (class(actor), sense, class(patient)).
        Without a sense picker, posting under several senses links synonyms (wash.v.03 ~ rinse) without
        collapsing verbs into one wrong hypernym family (run 9b/9c)."""
        a, p = self.slot(event.actor), self.slot(event.patient)
        ss = self.general.synsets(event.action, "VERB")[: self.senses]
        names = [self.general._name(x) for x in ss] or [self.action(event)["family_name"]]
        keys = []
        for name in names:
            readable = (a.cls, name, p.cls)
            k = stable_id("fam", *readable)
            self.table[k] = readable
            keys.append(k)
        return keys

    def class_triple(self, event: Event):
        """(class(actor), action lemma, class(patient)) — the benchmark's comparable form."""
        a, p = self.slot(event.actor), self.slot(event.patient)
        return (a.cls, event.action, p.cls)

    def decode(self, k):
        return self.table.get(k)
