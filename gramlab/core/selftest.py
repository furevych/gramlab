"""core-v1 · milestone self-test: text in, verified answer out, through the normal raw-text path.

    WORDNET_DIR=... python -m gramlab.core.selftest --frontend results/frontend.pkl --linker results/linker.pkl [--typer results/typer.pkl]

"Tom called John. John called Tom. John was called." → "Who called John?" must answer Tom with the first sentence;
with only "John was called." available, the actor must come back unknown (requested ≠ missing). The reversed pair
shares the family PERSON → CALL → PERSON, so the test is about verified bindings, not classes.
"""
from __future__ import annotations

import argparse
import json
import sys

from .linker import BackfillLinker, LearnedLinker
from .pipeline import Pipeline
from .registry import GeneralIndex
from .tagger import LiveFrontEnd


def run(frontend, linker, typer=None, verbose=True):
    general = GeneralIndex()
    typ = None
    if typer:
        from .typer import EntityTyper
        typ = EntityTyper.load(typer, general=general)
    fe = LiveFrontEnd.load(frontend, typer=typ)
    lk = BackfillLinker(LearnedLinker.load(linker))
    results = {}

    def check(name, pipe, question, expect_answer, expect_source=None):
        res = pipe.ask(question)
        answers = res.get("answers", [])
        top = answers[0] if answers else None
        got = top["answer"] if top else None
        ok = (got == expect_answer) and (expect_source is None or (top and top["source"].startswith(expect_source)))
        results[name] = {"question": question, "parsed": res.get("question"), "expected": expect_answer,
                         "got": got, "source": top["source"] if top else None, "n_answers": len(answers), "pass": ok}
        if verbose:
            print(("PASS" if ok else "FAIL"), name, "|", question, "->", got, "|", top["source"] if top else None, "| parsed", res.get("question"))

    p1 = Pipeline(linker=lk, general=general, front_end=fe)
    p1.ingest_text("Tom called John. John called Tom. John was called.", "doc1")
    check("who_called_john", p1, "Who called John?", "Tom", "Tom called John")
    check("who_called_tom", p1, "Who called Tom?", "John", "John called Tom")
    check("whom_did_tom_call", p1, "Who did Tom call?", "John", "Tom called John")
    check("passive_by", p1, "Who was called by Tom?", "John", "Tom called John")

    p2 = Pipeline(linker=lk, general=general, front_end=fe)
    p2.ingest_text("John was called.", "doc2")
    check("actor_unknown", p2, "Who called John?", "unknown")

    p3 = Pipeline(linker=lk, general=general, front_end=fe)
    p3.ingest_text("Tom washed the cat. The cat was washed by Tom. The cat washed Tom. Maya rinsed the dog.", "doc3")
    check("who_washed_cat", p3, "Who washed the cat?", "Tom", "Tom washed the cat")
    check("what_did_cat_wash", p3, "What did the cat wash?", "Tom", "The cat washed Tom")
    # synonym at the question side: rinse ~ wash share a sense; the verified participant keeps it specific
    check("who_rinsed_dog", p3, "Who washed the dog?", "Maya", "Maya rinsed the dog")
    n_pass = sum(r["pass"] for r in results.values())
    print(f"{n_pass}/{len(results)} passed")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", required=True)
    ap.add_argument("--linker", required=True)
    ap.add_argument("--typer", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = run(a.frontend, a.linker, a.typer)
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(json.dumps(r, indent=2))
    sys.exit(0 if all(x["pass"] for x in r.values()) else 1)


if __name__ == "__main__":
    main()
