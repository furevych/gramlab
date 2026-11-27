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
        weights=(1.0, 0.5), dense=None, key_bonus=0.0):
    """dense: sentence-transformers model as a finder; key_bonus > 0: core-v1 key verification (run-16 ask path) added to
    the dense and BM25 rankings: +bonus for sentences whose stored key matches the question's resolved event."""
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

    Pd = Qd = None
    if dense:
        from .softmap import DenseEncoder
        t = time.time()
        enc = DenseEncoder(dense)
        Pd = enc.encode(sent_texts); Qd = enc.encode([q for q, _ in questions])
        log(f"dense encoder {dense}: {len(sent_texts) + len(questions)} texts in {time.time() - t:.0f}s")
    key_sets = None
    if key_bonus > 0:
        t = time.time()
        for s in sents:
            pipe.ingest([s])
        sid_row = {s.sid: i for i, s in enumerate(sents)}
        key_sets = []
        for qtext, _ in questions:
            rows = set()
            try:
                r = pipe.ask(qtext, limit=50)
                hits = r.get("answers", []) if isinstance(r, dict) else []
            except Exception:
                hits = []
            for h in hits:
                sid = h.get("sid") if isinstance(h, dict) else None
                if sid in sid_row:
                    rows.add(sid_row[sid])
            key_sets.append(np.fromiter(rows, dtype=np.int64) if rows else np.zeros(0, dtype=np.int64))
        log(f"key index over {len(sents)} sentences; questions with ≥1 key candidate {sum(len(k) > 0 for k in key_sets)}/{len(questions)} ({time.time() - t:.0f}s)")

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
        if Pd is not None:
            d = Pd @ Qd[qi]
            tops["dense"] = int(np.argmax(d))
            tops["bm25+dense"] = int(np.argmax(sc / mx + d))
        if key_sets is not None:
            kb = np.zeros(len(sent_texts), dtype=np.float32); kb[key_sets[qi]] = key_bonus
            tops["bm25+key"] = int(np.argmax(sc / mx + kb))
            if Pd is not None:
                tops["dense+key"] = int(np.argmax(d + kb))
                tops["bm25+dense+key"] = int(np.argmax(sc / mx + d + kb))
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
    ap.add_argument("--train", default=None, help="SQuAD train json: train the map on question→sentence pairs first, save to --map")
    ap.add_argument("--init_map", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit_articles", type=int, default=None)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dense", default=None)
    ap.add_argument("--key_bonus", type=float, default=0.0)
    a = ap.parse_args(argv)
    if a.train:
        train_squad(a.train, a.frontend, a.linker, a.map, a.semcor, a.wordnet, dim=a.dim, epochs=a.epochs, init_map=a.init_map, limit_articles=a.limit_articles)
    run(a.squad, a.frontend, a.linker, a.map, a.semcor, a.out, a.wordnet, a.limit_paragraphs, dense=a.dense, key_bonus=a.key_bonus)



# ------------------------------------------------------------------------------------------------ external training signal
def train_squad(train_path, frontend, linker, out_path, semcor_dir=None, wordnet_path=None, dim=64, epochs=3, batch=256, seed=0,
                init_map=None, limit_articles=None):
    """Train the soft map on real question → answer-sentence pairs (SQuAD *train*; dev stays the test). Contrastive,
    in-batch negatives; batches are drawn article by article so the negatives include same-paragraph sentences.
    Codes start from the WordNet-hierarchy initialisation (or from `init_map`, e.g. the AMALGUM-trained map)."""
    import random
    from .softmap import train as _train
    t0 = time.time()
    wn = WordNet(wordnet_path)
    ix = SoftIndexer(wn)
    if semcor_dir:
        ix.sense_prior_from_semcor(semcor_dir)
    general = GeneralIndex(wordnet_path=wordnet_path)
    fe = LiveFrontEnd.load(frontend)
    lk = BackfillLinker(LearnedLinker.load(linker))
    pipe = Pipeline(linker=lk, general=general, front_end=fe)
    data = json.loads(Path(train_path).read_text(encoding="utf-8"))["data"]
    if limit_articles:
        data = data[:limit_articles]
    pairs = []                     # (article index, question Sentence, answer Sentence)
    for ai, art in enumerate(data):
        for par in art["paragraphs"]:
            ss = pipe.read_text(par["context"], "t")
            texts = [s.text() for s in ss]
            for q in par["qas"]:
                ans = q["answers"][0]["text"]
                tgt = next((s for s, t in zip(ss, texts) if contains(t, [ans])), None)
                if tgt is None:
                    continue
                qs = fe.sentence(fe_tokens(q["question"]), sid="q", doc="q")
                pairs.append((ai, qs, tgt))
        if ai % 50 == 0:
            log(f"  read {ai + 1}/{len(data)} articles, {len(pairs)} pairs ({time.time() - t0:.0f}s)")
    log(f"{len(pairs)} question→sentence pairs from {len(data)} articles ({time.time() - t0:.0f}s)")
    rng = random.Random(seed)
    arts = sorted({a for a, _, _ in pairs}); rng.shuffle(arts)
    order = {a: i for i, a in enumerate(arts)}
    pairs.sort(key=lambda x: (order[x[0]], rng.random()))
    qc, pc = Corpus(), Corpus()
    for _, qs, ts in pairs:
        qc.add(featurize(qs, slot_labels(qs, lk.link(qs), True), ix), 0)
        pc.add(featurize(ts, slot_labels(ts, lk.link(ts), True), ix), 0)
    qc.finish(); pc.finish()
    model = SoftMap.load(init_map) if init_map else SoftMap(ix.init_codes(dim=dim, seed=seed), n_slots=4)

    # article-blocked batches: permute whole blocks each epoch, keep within-block order
    n = len(pairs)
    blocks = [list(range(s, min(n, s + batch))) for s in range(0, n, batch)]
    import numpy as _np
    prng = _np.random.default_rng(seed)
    for ep in range(epochs):
        prng.shuffle(blocks)
        perm = _np.concatenate([_np.asarray(b) for b in blocks])
        qsub, psub = _reorder(qc, perm), _reorder(pc, perm)
        _train(model, qsub, psub, epochs=1, batch=batch, seed=seed + ep, shuffle=False, log=lambda m: log(m.replace("epoch 1", f"epoch {ep + 1}")))
    model.save(out_path)
    log(f"saved {out_path} ({time.time() - t0:.0f}s)")
    return model


def _reorder(c, perm):
    """A Corpus with texts in `perm` order."""
    out = Corpus()
    for i in perm:
        a, b = c.off[i], c.off[i + 1]
        out.add((c.rows[a:b], c.ws[a:b], c.sl[a:b]), 0)
    return out.finish()


if __name__ == "__main__":
    main()
