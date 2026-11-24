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


class Pipeline:
    def __init__(self, linker=None, general=None, specific=None, affect=None, retriever=None, writer=None):
        self.general = general or GeneralIndex()
        self.specific = specific or SpecificIndex()
        self.keyer = Keyer(self.general, self.specific)
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
        for s, evs in zip(sents, batches):
            n += len(self.index.add(s, evs))
        return n

    def events(self, s: Sentence):
        return self.linker.link(s)

    def describe(self, e):
        """Readable record: key tuple, family tuple and force."""
        return {"actor": e.actor.text if e.actor else None, "action": e.action,
                "patient": e.patient.text if e.patient else None, "voice": e.voice, "status": e.status,
                "specific": self.keyer.decode(self.keyer.specific_key(e)),
                "family": self.keyer.decode(self.keyer.family_key(e)),
                "force": self.affect.force(e)}

    def answer(self, question: str, k=5):
        passages = self.index.search(question, k)
        if self.writer is None:
            return {"passages": passages, "answer": None}
        return {"passages": passages, "answer": self.writer.answer(question, passages)}

    def components(self):
        return {"core": self.version, "linker": getattr(self.linker, "name", type(self.linker).__name__),
                "general_index": self.general.backend, "specific_seed": len(self.specific.seed),
                "affect": self.affect.source, "retriever": type(self.index.retriever).__name__ if self.index.retriever else None,
                "writer": type(self.writer).__name__ if self.writer else None}
