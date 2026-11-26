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

from .records import Binding, Event, Mention, ResolvedEvent
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

    # ------------------------------------------------------------------ one resolved record per occurrence
    def binding(self, m: Mention | None, context=None) -> Binding:
        if m is None:
            return Binding("omitted", "missing", "missing", 0, "", "")
        r = self.resolve(m, context)
        sl = self.slot(m, context)
        state = "unresolved" if sl.kind == "pronoun" else "known"
        return Binding(state, sl.kind, sl.cls, sl.id, r.text, r.lemma)

    def senses_of(self, event: Event, sent=None, n=None) -> list:
        """Posted senses for the action, chosen sense first (sense picker when a sentence is given)."""
        all_ss = self.general.synsets(event.action, "VERB")
        ss = list(all_ss[: (n or self.senses)])
        if self.sense_picker is not None and sent is not None and len(all_ss) > 1:
            chosen = self.sense_picker.pick(event.action, "v", self.as_semcor(sent), event.verb)
            if chosen is not None:
                chosen = ("v", chosen) if not isinstance(chosen, tuple) else chosen
                ss = [chosen] + [x for x in ss if x != chosen][: max((n or self.senses) - 1, 0)]
        names = [self.general._name(x) for x in ss]
        return names or [self.action(event)["family_name"]]

    def resolve_event(self, event: Event, sent=None, context=None) -> ResolvedEvent:
        senses = self.senses_of(event, sent)
        return ResolvedEvent(self.binding(event.actor, context), event.action, senses[0], tuple(senses[1:]),
                             self.binding(event.patient, context), event.voice, event.status)

    def _ids_of(self, b: Binding, permissive: bool) -> list:
        """Strict: the binding's own id, only when known. Permissive (proposals): name tokens and the bridge to
        the other id space, so another parser's reading of the same mention still meets this one."""
        if b.state != "known":
            return [] if not permissive else []
        ids = [b.id]
        if not permissive:
            return ids
        if b.kind == "specific":
            toks = [w for w in b.text.split() if w[:1].isupper() and len(w) > 1]
            for w in reversed(toks[-3:]):
                hid = self.specific.resolve(w, b.cls)["id"]
                if hid not in ids:
                    ids.append(hid)
            c = self.general.concept(b.lemma, "NOUN")
            if c["covered"] and c["id"] not in ids:
                ids.append(c["id"])
        elif b.kind == "general" and b.text[:1].isupper():
            hid = self.specific.resolve(b.text, b.cls)["id"]
            if hid not in ids:
                ids.append(hid)
        return ids

    def keys_from(self, r: ResolvedEvent, query=False) -> dict:
        """All keys of a resolved event. specific: one id for the exact record. family: (class, sense, class) per
        posted sense. identity: (id, sense, id), strict — both bindings known. proposal: permissive identity keys
        (name tokens, bridged ids) that may match a different entity and must be verified before being believed.
        query=True widens the senses to all of the verb's senses (lookups), postings keep the top ones."""
        senses = [r.sense] + list(r.alt_senses)
        all_names = [self.general._name(x) for x in self.general.synsets(r.action, "VERB")]
        id_senses = senses + [n for n in all_names if n not in senses]     # identity/proposal keys: all senses (ids keep them specific)
        if query:
            senses = id_senses
        a, p = r.actor, r.patient
        out = {"specific": [], "family": [], "identity": [], "proposal": []}
        readable = ((a.kind, a.cls, a.id), r.sense, (p.kind, p.cls, p.id), r.status)
        k = stable_id("key", *[str(x) for x in readable]); self.table[k] = readable; out["specific"].append(k)
        a_strict, p_strict = self._ids_of(a, False), self._ids_of(p, False)
        a_perm, p_perm = self._ids_of(a, True), self._ids_of(p, True)
        for name in senses:
            fr = (a.cls, name, p.cls)
            k = stable_id("fam", *fr); self.table[k] = fr; out["family"].append(k)
        for name in id_senses:
            for ai in a_strict:
                for pi in p_strict:
                    ir = (ai, name, pi); k = stable_id("ident", *[str(x) for x in ir]); self.table[k] = ir; out["identity"].append(k)
            for ai in a_perm:
                for pi in p_perm:
                    ir = (ai, name, pi); k = stable_id("ident", *[str(x) for x in ir])
                    if k not in out["identity"]:
                        self.table[k] = ir; out["proposal"].append(k)
            # one-sided proposals: the given role known, the other requested/omitted (question lookups)
            for ai in a_perm:
                ir = (ai, name, "*"); k = stable_id("ident", *[str(x) for x in ir]); self.table[k] = ir; out["proposal"].append(k)
            for pi in p_perm:
                ir = ("*", name, pi); k = stable_id("ident", *[str(x) for x in ir]); self.table[k] = ir; out["proposal"].append(k)
        return out

    def specific_key(self, event: Event, context=None) -> int:
        return self.keys_from(self.resolve_event(event, None, context))["specific"][0]

    def family_key(self, event: Event) -> int:
        return self.keys_from(self.resolve_event(event))["family"][0]

    def family_keys(self, event: Event, sent=None, context=None, senses=None) -> list:
        return self.keys_from(self.resolve_event(event, sent, context), query=bool(senses and senses > self.senses))["family"]

    def identity_keys(self, event: Event, sent=None, context=None, senses=None) -> list:
        k = self.keys_from(self.resolve_event(event, sent, context), query=bool(senses and senses > self.senses))
        return k["identity"] + k["proposal"]

    def class_triple(self, event: Event, context=None):
        """(class(actor), action lemma, class(patient)) — the benchmark's comparable form.
        With context=(sents, si) and a resolver, pronoun slots take their antecedent's class."""
        a, p = self.slot(event.actor, context), self.slot(event.patient, context)
        return (a.cls, event.action, p.cls)

    def decode(self, k):
        return self.table.get(k)
