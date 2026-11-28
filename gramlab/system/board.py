from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path


# ------------------------------------------------------------------------------------------------ document
@dataclass
class Document:
    text: str
    doc_id: str = "doc"
    layers: dict = field(default_factory=dict)      # layer name -> value
    trace: list = field(default_factory=list)       # (component, version, seconds)

    def has(self, *names):
        return all(n in self.layers for n in names)


# ------------------------------------------------------------------------------------------------ component
class Component:
    name = "component"
    version = "0"
    reads: tuple = ()
    writes: str = ""
    failure_value = None                              # what the layer holds when the component cannot decide
    benchmark_name = ""
    benchmark_value = None                            # last measured number
    benchmark_date = ""

    def apply(self, doc: Document):
        raise NotImplementedError

    def benchmark(self, **kw):
        """Re-run this component's own benchmark; return (name, value). Default: not re-runnable here."""
        return self.benchmark_name, self.benchmark_value

    def __call__(self, doc: Document):
        missing = [r for r in self.reads if r not in doc.layers]
        if missing:
            raise ValueError(f"{self.name} needs layers {missing}")
        t = time.time()
        out = self.apply(doc)
        doc.layers[self.writes] = out
        doc.trace.append((self.name, self.version, round(time.time() - t, 3)))
        return doc


# ------------------------------------------------------------------------------------------------ components
class FrontEnd(Component):
    """text -> sentences (tokens, UPOS/XPOS, lemmas). Failure: an untagged token keeps 'X'."""
    name, version, reads, writes = "front_end", "perceptron-amalgum40k", (), "sentences"
    benchmark_name, benchmark_value, benchmark_date = "UPOS accuracy, GUM test, live", 0.905, "2026-11-26"

    def __init__(self, path):
        from ..core.tagger import LiveFrontEnd
        self.fe = LiveFrontEnd.load(path)

    def apply(self, doc):
        from ..core.pipeline import Pipeline
        from ..core.registry import GeneralIndex
        pipe = Pipeline(linker=None, general=GeneralIndex(), front_end=self.fe)
        return pipe.read_text(doc.text, doc.doc_id)


class Linker(Component):
    """sentences -> events per sentence (verb, actor span, patient span, voice, status). Failure: a slot is None."""
    name, version, reads, writes = "linker", "hashed-logistic-backfill", ("sentences",), "events"
    benchmark_name, benchmark_value, benchmark_date = "link F1, GUM test gold tags / live", "0.887 / 0.854", "2026-11-26"

    def __init__(self, path):
        from ..core.linker import BackfillLinker, LearnedLinker
        self.lk = BackfillLinker(LearnedLinker.load(path))

    def apply(self, doc):
        return [self.lk.link(s) for s in doc.layers["sentences"]]


class Entities(Component):
    """sentences + events -> registry (who is who) + records (actor_id, action, patient_id, status, sentence) + mentions.
    Failure: a pronoun with no antecedent is dropped, never guessed into a record."""
    name, version, reads, writes = "entities", "names-by-recency+nearest-name", ("sentences", "events"), "analysis"
    benchmark_name, benchmark_value, benchmark_date = "coreference, GUM test (ranker)", 0.549, "2026-11-26"

    def __init__(self, linker: Linker, resolver=None):
        from ..core.summarize import Summarizer
        self.S = Summarizer(linker.lk, None, resolver)

    def apply(self, doc):
        return self.S.analyse(doc.layers["sentences"])


class ContextEstimator(Component):
    """analysis -> what the text is about: ranked entities. Learned scorer where a model is given, else roles."""
    name, version, reads, writes = "context", "centering-learned", ("analysis",), "about"
    benchmark_name, benchmark_value, benchmark_date = "title test top-1 (bio/news/voyage/how-to), vs counting", "0.74/0.54/0.74/0.54 vs 0.64/0.51/0.61/0.49", "2026-11-28"

    def __init__(self, model_path=None, k=10):
        self.model = pickle.load(open(model_path, "rb")) if model_path and Path(model_path).exists() else None
        self.k = k

    def apply(self, doc):
        an = doc.layers["analysis"]
        return an.about("learned" if self.model else "roles", self.k, self.model)

    def benchmark(self, amalgum_glob=None, **kw):
        if not amalgum_glob:
            return self.benchmark_name, self.benchmark_value
        from ..core.summarize import title_eval
        b = title_eval(amalgum_glob, genres=("bio", "news", "voyage", "whow"), max_docs=300)
        return self.benchmark_name, {g: (b[g]["learned"], b[g]["freq"]) for g in b}


class Skeleton(Component):
    """analysis -> the records that carry the text (top by participant salience, in order) + sentence selection."""
    name, version, reads, writes = "skeleton", "salience-select", ("analysis",), "skeleton"
    benchmark_name, benchmark_value, benchmark_date = "qualitative (Persuasion cast + spine)", "see run 23", "2026-11-28"

    def __init__(self, k=12, n_sentences=8):
        self.k, self.n = k, n_sentences

    def apply(self, doc):
        an = doc.layers["analysis"]
        return {"records": an.skeleton("roles", self.k), "sentences": an.summary(self.n, "roles")}


class DirectionVerifier(Component):
    """analysis -> for every pair of records with the same participants and action in opposite order (within the
    document, or against a reference set of records), a flagged reversal; negation conflicts flagged too.
    Failure: a record with an unresolved slot is not compared."""
    name, version, reads, writes = "direction", "ordered-key", ("analysis",), "direction"
    benchmark_name, benchmark_value, benchmark_date = "reversal separation, 698 held-out queries (BM25 0%, MiniLM 44%)", 0.96, "2026-11-27"

    def __init__(self, reference=None):
        self.reference = reference or []              # (actor_label, action, patient_label, status)

    def apply(self, doc):
        an = doc.layers["analysis"]
        lab = an.reg.label
        recs = [(lab[a].lower(), v, lab[p].lower(), st, si) for a, v, p, st, si in an.records]
        seen = {}
        reversals, negations = [], []
        for a, v, p, st, si in recs + [(x[0].lower(), x[1], x[2].lower(), x[3], -1) for x in self.reference]:
            if (p, v, a) in seen:
                reversals.append({"record": f"{a} → {v} → {p}", "sentence": si, "conflicts_with": seen[(p, v, a)]})
            if (a, v, p) in seen and seen[(a, v, p)]["status"] != st and "negated" in (st, seen[(a, v, p)]["status"]):
                negations.append({"record": f"{a} → {v} → {p}", "sentence": si, "status": st, "conflicts_with": seen[(a, v, p)]})
            seen.setdefault((a, v, p), {"sentence": si, "status": st})
        return {"reversals": reversals, "negation_conflicts": negations, "records_checked": len(recs)}


class Writer(Component):
    """skeleton + sentences -> prose. Optional; needs a generative model. Failure: returns the skeleton as lines."""
    name, version, reads, writes = "writer", "none", ("skeleton",), "prose"
    benchmark_name, benchmark_value, benchmark_date = "end-to-end answer accuracy", None, ""

    def __init__(self, generate=None):
        self.generate = generate                       # callable(prompt) -> str, or None

    def apply(self, doc):
        sk = doc.layers["skeleton"]
        if self.generate is None:
            return "\n".join(sk["records"])
        about = ", ".join(a[0] for a in doc.layers.get("about", [])[:5])
        prompt = ("Write a faithful summary of the text below. Use only facts in the FACTS list; do not add events.\n"
                  f"MAIN ENTITIES: {about}\nFACTS:\n" + "\n".join(sk["records"]) + "\nKEY SENTENCES:\n" + "\n".join(sk["sentences"]) + "\nSUMMARY:")
        return self.generate(prompt)


# ------------------------------------------------------------------------------------------------ system
class System:
    def __init__(self, components):
        self.components = components

    @classmethod
    def default(cls, models_dir="results", resolver=None, generate=None):
        m = Path(models_dir)
        fe = FrontEnd(str(m / "frontend.pkl")); lk = Linker(str(m / "linker.pkl"))
        comps = [fe, lk, Entities(lk, resolver), ContextEstimator(str(m / "about.pkl")), Skeleton(), DirectionVerifier(), Writer(generate)]
        return cls(comps)

    def run(self, text, doc_id="doc", upto=None):
        doc = Document(text, doc_id)
        for c in self.components:
            c(doc)
            if upto and c.name == upto:
                break
        return doc

    def status(self):
        rows = [("component", "version", "reads → writes", "benchmark", "value", "date")]
        for c in self.components:
            rows.append((c.name, c.version, f"{','.join(c.reads) or '-'} → {c.writes}", c.benchmark_name, str(c.benchmark_value), c.benchmark_date))
        w = [max(len(r[i]) for r in rows) for i in range(6)]
        return "\n".join("  ".join(x.ljust(w[i]) for i, x in enumerate(r)) for r in rows)

    def report(self, doc: Document):
        """A compact, readable account of one document's analysis."""
        an = doc.layers.get("analysis")
        out = {"doc": doc.doc_id, "sentences": len(doc.layers.get("sentences", [])),
               "records": len(an.records) if an else 0, "entities": len(an.reg.label) if an else 0,
               "about": [a[0] for a in doc.layers.get("about", [])],
               "skeleton": doc.layers.get("skeleton", {}).get("records"),
               "direction": doc.layers.get("direction"),
               "trace": doc.trace}
        return json.dumps(out, indent=1, ensure_ascii=False)
