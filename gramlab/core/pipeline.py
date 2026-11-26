"""core-v1 · pipeline — wires the layers.

    text / CoNLL-U  --parser-->  Sentence  --linker-->  [Event]  --keyer-->  keys  --index-->  occurrences
    question  --index.search / index.related-->  occurrences  --writer-->  answer

Each arrow is one component with one contract; swap any of them, the rest does not move.
`Parser` is the only piece without a trained occupant in this repo: GUM/AMALGUM come parsed
(ConlluParser).  A live-text parser (spaCy/Stanza/Trankit → Sentence) is a layer-1 component to add,
scored by the same benchmarks through this pipeline.
"""
from __future__ import annotations

from .affect import Affect
from .index import MeaningIndex
from .key import Keyer
from .linker import GoldLinker, LearnedLinker, RuleLinker
from .records import CORE_VERSION, Sentence, read_conllu
from .registry import GeneralIndex, SpecificIndex


class ConlluParser:
    """Sentences already parsed (UD). The reference parser for benchmarks."""
    name = "conllu"

    def parse_file(self, path, limit=None):
        return read_conllu(path, limit)


_SPLIT = __import__("re").compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


class Pipeline:
    def __init__(self, linker=None, general=None, specific=None, affect=None, retriever=None, writer=None, sense_picker=None, resolver=None,
                 front_end=None):
        self.front_end = front_end        # core.tagger.LiveFrontEnd for raw text (no parser)
        self.general = general or GeneralIndex()
        self.specific = specific or SpecificIndex()
        self.keyer = Keyer(self.general, self.specific, sense_picker=sense_picker, resolver=resolver)
        self.linker = linker or RuleLinker()
        self.affect = affect or Affect()
        self.index = MeaningIndex(self.keyer, retriever)
        self.writer = writer          # object with .answer(question, passages) -> str, or None
        self.version = CORE_VERSION

    def ingest(self, sents: list[Sentence]):
        if hasattr(self.linker, "link_batch"):
            batches = self.linker.link_batch(sents)
        else:
            batches = [self.linker.link(s) for s in sents]
        n = 0
        for si, (s, evs) in enumerate(zip(sents, batches)):
            n += len(self.index.add(s, evs, context=(sents, si)))
        return n

    def events(self, s: Sentence):
        return self.linker.link(s)

    def read_text(self, text: str, doc: str = "") -> list[Sentence]:
        """Raw text -> Sentences through the live front end (tokenizer + tagger + lemma table)."""
        if self.front_end is None:
            raise RuntimeError("no front end: pass front_end=LiveFrontEnd.load(...) for raw text")
        from .tagger import tokenize
        out = []
        for k, chunk in enumerate(p for p in _SPLIT.split(text.strip()) if p.strip()):
            words = tokenize(chunk)
            if words:
                out.append(self.front_end.sentence(words, sid=f"{doc}#{k}", doc=doc))
        return out

    def ingest_text(self, text: str, doc: str = "") -> int:
        return self.ingest(self.read_text(text, doc))

    def describe(self, e, sent=None, context=None):
        """The resolved record as it would be stored (same resolution path as indexing), plus force."""
        r = self.keyer.resolve_event(e, sent, context)
        return {"actor": r.actor.__dict__, "action": r.action, "sense": r.sense, "alt_senses": list(r.alt_senses),
                "patient": r.patient.__dict__, "voice": r.voice, "status": r.status, "force": self.affect.force(e)}

    WH_PRON = {"who", "whom", "whose", "what", "which"}
    NOT_EVENT_WH = {"where", "when", "why", "how", "whence"}

    def ask(self, question: str, limit=10, rank=None):
        """A question in words, through the same front end and linker as the text: 'Who called John?',
        'Who did Tom call?', 'Who was called by Tom?', 'What did Tom wash?', 'Which network broadcast the game?'.
        The wh-word (or the noun phrase it determines) marks the requested role; the other role is the given
        participant, verified against the stored binding. Where/when/why/how questions ask for something the event
        record does not hold and are not attempted. Returns index.ask()'s list, or {"error": ...}."""
        if self.front_end is None:
            raise RuntimeError("no front end: pass front_end=LiveFrontEnd.load(...) to read questions")
        from .tagger import tokenize
        words = tokenize(question.rstrip("?").strip() + " .")
        low = [w.lower() for w in words]
        if not words or low[0] in self.NOT_EVENT_WH or not any(w in self.WH_PRON for w in low):
            return {"error": "not an event question", "tokens": words}
        # inversion: "Who did X call ?" → "X did call Who ."   /  "Which team did X beat ?" → "X did beat Which team ."
        wh_end = 1
        if low[0] in ("which", "what", "whose") and len(low) > 2 and low[1] not in ("did", "does", "do", "is", "was", "were", "are", "has", "have", "had"):
            while wh_end < len(low) - 1 and low[wh_end] not in ("did", "does", "do", "is", "was", "were", "are", "has", "have", "had") and \
                    not (self.front_end.tagger.tag([words[wh_end]])[0][0] == "VERB"):
                wh_end += 1
        wh_phrase = words[:wh_end]
        if len(wh_phrase) > 1 and self.front_end.typer is not None:
            # "what city / which year / what venue": the requested thing is a place or a time, which the event
            # record does not hold — decline rather than answer with a participant
            probe = self.front_end.sentence(wh_phrase[1:] + ["."], sid="q", doc="q")
            heads = [i for i in range(len(probe)) if probe.upos[i] in ("NOUN", "PROPN")]
            if heads and (probe.ents[heads[-1]] in ("place", "time") or probe.lemmas[heads[-1]] in ("city", "country", "state", "year", "venue", "town", "place", "location", "region", "date", "decade", "century")):
                return {"error": "asks for a place or time", "tokens": words}
        rest = words[wh_end:]
        if rest and rest[0].lower() in ("did", "does", "do") and len(rest) > 2:
            aux, body = rest[0], rest[1:-1]
            words2 = body[:-1] + [aux, body[-1]] + wh_phrase + ["."]
        else:
            words2 = words
        s = self.front_end.sentence(words2, sid="q", doc="q")
        events = [e for e in self.linker.link(s) if e.actor or e.patient]
        if not events:
            return {"error": "no event read from the question", "tokens": words2}
        e = max(events, key=lambda x: (x.actor is not None) + (x.patient is not None))

        def role(m):
            if m is None:
                return None
            toks = [w.lower() for w in m.text.split()]
            return "?" if (toks and (toks[0] in self.WH_PRON or toks[-1] in self.WH_PRON)) else m.text
        actor, patient = role(e.actor), role(e.patient)
        if actor is None and patient != "?":
            actor = "?"
        if patient is None and actor != "?":
            patient = "?"
        if actor != "?" and patient != "?":
            # neither mention carries the wh-word (it was outside the linker's spans): the requested role is the
            # one that is not the given participant named in the question
            actor = "?"
        if actor == "?" and patient == "?":
            return {"error": "no given participant", "tokens": words2}
        return {"question": {"actor": actor, "action": e.action, "patient": patient, "voice": e.voice},
                "answers": self.index.ask(e.action, actor=actor, patient=patient, limit=limit, rank=rank)}

    def answer(self, question: str, k=5):
        passages = self.index.search(question, k)
        if self.writer is None:
            return {"passages": passages, "answer": None}
        return {"passages": passages, "answer": self.writer.answer(question, passages)}

    def components(self):
        return {"core": self.version, "linker": getattr(self.linker, "name", type(self.linker).__name__),
                "general_index": self.general.backend, "specific_seed": len(self.specific.seed),
                "affect": self.affect.source, "sense_picker": getattr(self.keyer.sense_picker, "name", None), "resolver": getattr(self.keyer.resolver, "name", None), "retriever": type(self.index.retriever).__name__ if self.index.retriever else None,
                "writer": type(self.writer).__name__ if self.writer else None}
