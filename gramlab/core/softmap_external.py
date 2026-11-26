"""core-v2 · the soft map on the external test (SQuAD v1.1 dev), as a re-ranker beside BM25.

    python -m gramlab.core.softmap_external --squad dev-v1.1.json --frontend results/frontend.pkl --linker results/linker.pkl \
        --map results/softmap.npz --semcor data/semcor [--out results]

Same corpus and scoring as bench_external (run 16): paragraphs → sentences through the live front end; a question is
answered by the top sentence; a hit = the sentence contains a gold answer string. Systems: BM25; soft map alone; hybrid
BM25/max + cosine (weight fixed at 1, as in run 17; 0.5 reported as a sensitivity, not chosen). Roles on both sides come
from the learned linker on live text; when it finds no event the sentence/question is one REST bag, so the map always
answers (the run-16 event path attempted 7% of questions). Nothing here was fitted on SQuAD.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .bench_external import contains
from .bench_retrieval import BM25
from .linker import BackfillLinker, LearnedLinker
from .pipeline import Pipeline
from .registry import GeneralIndex
from .softmap import Corpus, SoftIndexer, SoftMap, featurize, log, slot_labels
from .tagger import LiveFrontEnd
from .wordnet import WordNet


def run(squad_path, frontend, linker, map_path, semcor_dir=None, out_dir="results", wordnet_path=None, limit_paragraphs=None,
        weights=(1.0, 0.5)):
    t0 = time.time()
    wn = WordNet(wordnet_path)
    ix = SoftIndexer(wn)
    if semcor_dir:
        ix.sense_prior_from_semcor(semcor_dir)
    model = SoftMap.load(map_path)
    use_slots = model.n_slots > 1
    general = GeneralIndex(wordnet_path=wordnet_path)
    fe = LiveFrontEnd.load(frontend)
    lk = BackfillLinker(LearnedLinker.load(linker))
    pipe = Pipeline(linker=lk, general=general, front_end=fe)

    data = json.loads(Path(squad_path).read_text(encoding="utf-8"))["data"]
    sent_texts, sents, questions = [], [], []
    n_par = 0
    for art in data:
        for pi, par in enumerate(art["paragraphs"]):
            if limit_paragraphs and n_par >= limit_paragraphs:
                break
            doc = f"{art['title']}#{pi}"
            ss = pipe.read_text(par["context"], doc)
            sents.extend(ss)
            sent_texts.extend(s.text() for s in ss)
            for q in par["qas"]:
                questions.append((q["question"], [a["text"] for a in q["answers"]]))
            n_par += 1
    log(f"{n_par} paragraphs → {len(sents)} sentences, {len(questions)} questions ({time.time() - t0:.0f}s)")

    corp = Corpus()
    for s in sents:
        corp.add(featurize(s, slot_labels(s, lk.link(s), use_slots), ix), 0)
    corp.finish()
    P = model.encode(corp)
    q_sents = [fe.sentence(fe_tokens(q), sid="q", doc="q") for q, _ in questions]
    qcorp = Corpus()
    n_event = 0
    for qs in q_sents:
        ev = lk.link(qs)
        n_event += bool(ev)
        qcorp.add(featurize(qs, slot_labels(qs, ev, use_slots), ix), 0)
    qcorp.finish()
    Q = model.encode(qcorp)
    log(f"encoded; questions with a parsed event {n_event}/{len(questions)} ({time.time() - t0:.0f}s)")

    bm = BM25(sent_texts)
    res = Counter()
    who = Counter()
    n = len(questions)
    for qi, (qtext, answers) in enumerate(questions):
        sc = bm.scores(qtext)
        mx = max(float(sc.max()), 1e-6)
        sim = P @ Q[qi]
        is_who = qtext.lower().startswith(("who", "whom", "whose"))
        tops = {"bm25": int(np.argmax(sc)), "map": int(np.argmax(sim))}
        for w in weights:
            tops[f"hybrid_w{w}"] = int(np.argmax(sc / mx + w * sim))
        for name, top in tops.items():
            ok = contains(sent_texts[top], answers)
            res[name] += ok
            if is_who:
                who[name] += ok
        res["n"] += 1
        who["n"] += is_who
    board = {"corpus": {"paragraphs": n_par, "sentences": len(sents), "questions": n, "questions_with_parsed_event": n_event},
             "top1_has_answer": {k: round(v / n, 4) for k, v in res.items() if k != "n"},
             "who_questions": {"n": who["n"], **{k: round(v / max(who["n"], 1), 4) for k, v in who.items() if k != "n"}},
             "hybrid_weight_note": "weight 1.0 is the run-17 setting (fixed before this test); other weights are sensitivity only",
             "seconds": round(time.time() - t0, 1)}
    print(json.dumps(board, indent=1))
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "softmap_external.json").write_text(json.dumps(board, indent=2))
    return board


def fe_tokens(text):
    from .tagger import tokenize
    return tokenize(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", required=True)
    ap.add_argument("--frontend", required=True)
    ap.add_argument("--linker", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--semcor", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--wordnet", default=None)
    ap.add_argument("--limit_paragraphs", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.squad, a.frontend, a.linker, a.map, a.semcor, a.out, a.wordnet, a.limit_paragraphs)


if __name__ == "__main__":
    main()
