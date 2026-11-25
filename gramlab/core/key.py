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
    def __init__(self, general: GeneralIndex, specific: SpecificIndex, senses=3, sense_picker=None, resolver=None):
        self.general = general
        self.specific = specific
        self.senses = senses          # family postings: top-k senses of the action (run 9c)
        self.sense_picker = sense_picker   # layer-1 picker (core.sense.RankPicker); chosen sense leads the postings
        self.resolver = resolver           # layer-1 coreference (core.coref.*): pronoun slots take their antecedent
        self.table = {}     # key id -> readable tuple
        self.resolved = 0
        self.unresolved = 0
        self._rcache = {}

    def resolve(self, m: Mention, context):
        """Follow a pronoun to a nominal antecedent (≤ 4 hops) when a resolver and (sents, si) are given."""
        if self.resolver is None or context is None or m.kind != "pronoun":
            return m
        from .coref import FIRST_SECOND
        if m.text.lower() in FIRST_SECOND:          # speaker / addressee: not resolvable from text alone
            return m
        sents, si = context
        cache_key = (id(sents), si, m.head)
        if cache_key in self._rcache:
            return self._rcache[cache_key]
        cur = (si, m.head)
        for _ in range(4):
            ant = self.resolver.antecedent(sents, cur[0], cur[1])
            if ant is None:
                break
            am = sents[ant[0]].mention(ant[1])
            if am.kind != "pronoun":
                self.resolved += 1
                self._rcache[cache_key] = am
                return am
            cur = ant
        self.unresolved += 1
        self._rcache[cache_key] = m
        return m

    @staticmethod
    def as_semcor(sent):
        """Sentence -> the token dicts the sense picker's features read."""
        return [{"lemma": l.lower(), "tag": x, "pos": None, "gold": None} for l, x in zip(sent.lemmas, sent.xpos)]

    def slot(self, m: Mention | None, context=None) -> Slot:
        if m is None:
            return Slot("missing", "missing", 0)
        m = self.resolve(m, context)
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

    def specific_key(self, event: Event, context=None) -> int:
        a, p = self.slot(event.actor, context), self.slot(event.patient, context)
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

    def family_keys(self, event: Event, sent=None, context=None, senses=None) -> list:
        """One family key per posted sense of the action: (class(actor), sense, class(patient)).
        Posting under several senses links synonyms (wash.v.03 ~ rinse) without collapsing verbs into one
        wrong hypernym family (run 9b/9c). With a sense picker and the sentence, the chosen sense comes first
        and is the occurrence's family_key; the top-k by WordNet order fill the rest."""
        a, p = self.slot(event.actor, context), self.slot(event.patient, context)
        all_ss = self.general.synsets(event.action, "VERB")
        ss = list(all_ss[: (senses or self.senses)])
        if self.sense_picker is not None and sent is not None and len(all_ss) > 1:
            chosen = self.sense_picker.pick(event.action, "v", self.as_semcor(sent), event.verb)
            if chosen is not None:
                chosen = ("v", chosen) if not isinstance(chosen, tuple) else chosen
                ss = [chosen] + [x for x in ss if x != chosen][: max(self.senses - 1, 0)]
        names = [self.general._name(x) for x in ss] or [self.action(event)["family_name"]]
        keys = []
        for name in names:
            readable = (a.cls, name, p.cls)
            k = stable_id("fam", *readable)
            self.table[k] = readable
            keys.append(k)
        return keys

    def identity_keys(self, event: Event, sent=None, context=None, senses=None) -> list:
        """Ordered identity postings: (id(actor), sense, id(patient)) for the same posted senses. Separates
        'Tom washed the cat' from 'the cat washed Tom' even when both slots have the same class.
        `senses` widens the sense list for lookups (postings stay at self.senses)."""
        a, p = self.slot(event.actor, context), self.slot(event.patient, context)
        all_ss = self.general.synsets(event.action, "VERB")
        ss = list(all_ss[: (senses or self.senses)])
        names = [self.general._name(x) for x in ss] or [self.action(event)["family_name"]]
        keys = []
        for name in names:
            readable = (a.id, name, p.id)
            k = stable_id("ident", *[str(x) for x in readable])
            self.table[k] = readable
            keys.append(k)
        return keys

    def class_triple(self, event: Event):
        """(class(actor), action lemma, class(patient)) — the benchmark's comparable form."""
        a, p = self.slot(event.actor), self.slot(event.patient)
        return (a.cls, event.action, p.cls)

    def decode(self, k):
        return self.table.get(k)
