"""core-v1 · external benchmark: independently written questions over a real corpus (SQuAD v1.1 dev).

    python -m gramlab.core.bench_external --squad dev-v1.1.json --frontend results/frontend.pkl --linker results/linker.pkl [--typer ...]

Nothing here was built from this data. The 2,067 paragraphs are ingested through the live front end (sentence
split → tagger → typer → linker → resolved events → index); each of the 10,570 questions is answered two ways:

  BM25     — the question text against all sentences; top sentence.
  ask      — Pipeline.ask(question): the question read by the same front end, the wh-role requested, candidates
             from the given participant's postings, verified against stored bindings; top answer's source sentence.
  hybrid   — ask when it returns an answer, BM25 otherwise.

Scoring is sentence-level: does the returned sentence contain a gold answer string? (SQuAD answers are spans in the
paragraph, so this is the retrieval half of QA; the writer would read the sentence.) Reported overall, on the
questions ask attempts, and on who-questions, which is what the event path is built for. Also: the share of
attempted questions whose *bound role string* itself matches a gold answer (exact or containment either way).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .bench_retrieval import BM25
from .linker import BackfillLinker, LearnedLinker
from .pipeline import Pipeline
from .registry import GeneralIndex, normalize_name
from .tagger import LiveFrontEnd

_ws = re.compile(r"\s+")


def norm(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return _ws.sub(" ", s).strip()


def contains(sentence, answers):
    ns = norm(sentence)
    return any(norm(a) and norm(a) in ns for a in answers)


def answer_match(pred, answers):
    p = norm(pred or "")
    if not p:
        return False
    for a in answers:
        na = norm(a)
        if na and (p == na or p in na or na in p):
            return True
    return False


def run(squad_path, frontend, linker, typer=None, out_dir="results", limit_questions=None, limit_paragraphs=None):
    general = GeneralIndex()
    typ = None
    if typer:
        from .typer import EntityTyper
        typ = EntityTyper.load(typer, general=general)
    fe = LiveFrontEnd.load(frontend, typer=typ)
    lk = BackfillLinker(LearnedLinker.load(linker))
    pipe = Pipeline(linker=lk, general=general, front_end=fe)

    data = json.loads(Path(squad_path).read_text(encoding="utf-8"))["data"]
    t0 = time.time()
    sent_texts, sent_doc = [], []
    questions = []
    n_par = 0
    for art in data:
        for pi, par in enumerate(art["paragraphs"]):
            if limit_paragraphs and n_par >= limit_paragraphs:
                break
            doc = f"{art['title']}#{pi}"
            sents = pipe.read_text(par["context"], doc)
            pipe.ingest(sents)
            for s_ in sents:
                sent_texts.append(s_.text())
                sent_doc.append(doc)
            for q in par["qas"]:
                questions.append((q["question"], [a["text"] for a in q["answers"]], doc))
            n_par += 1
    ingest_s = time.time() - t0
    if limit_questions:
        questions = questions[:limit_questions]
    print(f"ingested {n_par} paragraphs → {len(sent_texts)} sentences, {len(pipe.index.records)} events in {ingest_s:.0f}s; {len(questions)} questions")

    bm = BM25(sent_texts)
    sid_index = {}                                    # occurrence sid -> sentence index (read_text numbers sentences per doc)
    k = 0
    for art in data:
        for pi, par in enumerate(art["paragraphs"]):
            pass
    for i, (txt, d) in enumerate(zip(sent_texts, sent_doc)):
        sid_index.setdefault(d, []).append(i)

    res = Counter()
    times = Counter()
    who = Counter()
    examples = {"ask_right": [], "ask_wrong": [], "bm25_only": []}
    for qtext, answers, doc in questions:
        is_who = qtext.lower().startswith(("who", "whom", "whose"))
        # BM25
        t = time.time()
        sc = bm.scores(qtext)
        top = int(np.argmax(sc))
        times["bm25"] += time.time() - t
        bm_ok = contains(sent_texts[top], answers)
        res["bm25_ok"] += bm_ok
        if is_who:
            who["n"] += 1
            who["bm25_ok"] += bm_ok
        # ask
        t = time.time()
        sc_q = sc
        def rank(source, sid, _sc=sc_q):
            d, _, kk = sid.rpartition("#")
            lst = sid_index.get(d, [])
            try:
                return _sc[lst[int(kk)]]
            except (ValueError, IndexError):
                return 0.0
        try:
            r = pipe.ask(qtext, limit=3, rank=rank)
        except Exception as ex:                       # a question the parser cannot read
            r = {"error": str(ex)}
        times["ask"] += time.time() - t
        ans = r.get("answers") or []
        top_a = next((a for a in ans if a.get("answer") and a.get("answer") != "unknown"), None)
        attempted = top_a is not None
        # variant: BM25 proposes its top-10 sentences, the key verifies which states the asked event
        ver_ok = bm_ok
        if r.get("question"):
            qd = r["question"]
            top10 = [int(i) for i in np.argsort(-sc)[:10]]
            sids = [f"{sent_doc[i]}#{sid_index[sent_doc[i]].index(i)}" for i in top10]
            vs = pipe.index.verify_in(sids, qd["action"], actor=qd["actor"], patient=qd["patient"])
            vs = [v for v in vs if v["answer"] and v["answer"] != "unknown"]
            if vs:
                res["verified_attempted"] += 1
                ver_ok = contains(vs[0]["source"], answers)
                res["verified_ok_on_attempted"] += ver_ok
                res["verified_bm25_ok_on_attempted"] += bm_ok
                res["verified_role_match"] += answer_match(vs[0]["answer"], answers)
                if is_who:
                    who["verified_attempted"] += 1
                    who["verified_ok_on_attempted"] += ver_ok
                    who["verified_bm25_ok_on_attempted"] += bm_ok
        res["verified_hybrid_ok"] += ver_ok
        if is_who:
            who["verified_hybrid_ok"] += ver_ok
        if attempted:
            res["attempted"] += 1
            ask_ok = contains(top_a["source"], answers)
            res["ask_ok"] += ask_ok
            res["ask_role_match"] += answer_match(top_a["answer"], answers)
            res["bm25_ok_on_attempted"] += bm_ok
            res["hybrid_ok"] += ask_ok
            if is_who:
                who["attempted"] += 1
                who["ask_ok"] += ask_ok
                who["role_match"] += answer_match(top_a["answer"], answers)
                who["bm25_ok_on_attempted"] += bm_ok
                who["hybrid_ok"] += ask_ok
            bucket = "ask_right" if ask_ok else "ask_wrong"
            if len(examples[bucket]) < 8:
                examples[bucket].append({"q": qtext, "gold": answers[:2], "parsed": r.get("question"), "answer": top_a["answer"], "source": top_a["source"][:160]})
        else:
            res["hybrid_ok"] += bm_ok
            if is_who:
                who["hybrid_ok"] += bm_ok
            if len(examples["bm25_only"]) < 4:
                examples["bm25_only"].append({"q": qtext, "parsed": r.get("question"), "reason": r.get("error") or ("no verified answer" if r.get("question") else "no event")})
    n = len(questions)
    att = max(res["attempted"], 1)
    board = {
        "corpus": {"paragraphs": n_par, "sentences": len(sent_texts), "events": len(pipe.index.records), "ingest_seconds": round(ingest_s, 1),
                   "ms_per_sentence": round(1000 * ingest_s / max(len(sent_texts), 1), 1)},
        "questions": n,
        "all": {"bm25_top1_has_answer": round(res["bm25_ok"] / n, 4), "hybrid_top1_has_answer": round(res["hybrid_ok"] / n, 4),
                "ask_attempted_share": round(res["attempted"] / n, 4)},
        "attempted": {"n": res["attempted"], "ask_top1_has_answer": round(res["ask_ok"] / att, 4), "bm25_top1_has_answer": round(res["bm25_ok_on_attempted"] / att, 4),
                      "ask_role_string_matches_gold": round(res["ask_role_match"] / att, 4)},
        "who_questions": {"n": who["n"], "attempted": who["attempted"], "bm25_top1_has_answer": round(who["bm25_ok"] / max(who["n"], 1), 4),
                          "hybrid_top1_has_answer": round(who["hybrid_ok"] / max(who["n"], 1), 4),
                          "ask_top1_has_answer_on_attempted": round(who["ask_ok"] / max(who["attempted"], 1), 4),
                          "bm25_top1_has_answer_on_attempted": round(who["bm25_ok_on_attempted"] / max(who["attempted"], 1), 4),
                          "ask_role_string_matches_gold_on_attempted": round(who["role_match"] / max(who["attempted"], 1), 4)},
        "bm25_proposes_key_verifies": {"attempted": res["verified_attempted"], "hybrid_top1_has_answer_all": round(res["verified_hybrid_ok"] / n, 4),
                                       "verified_top1_has_answer_on_attempted": round(res["verified_ok_on_attempted"] / max(res["verified_attempted"], 1), 4),
                                       "bm25_top1_has_answer_on_attempted": round(res["verified_bm25_ok_on_attempted"] / max(res["verified_attempted"], 1), 4),
                                       "role_string_matches_gold_on_attempted": round(res["verified_role_match"] / max(res["verified_attempted"], 1), 4),
                                       "who": {"attempted": who["verified_attempted"], "hybrid_top1_has_answer_all": round(who["verified_hybrid_ok"] / max(who["n"], 1), 4),
                                               "verified_on_attempted": round(who["verified_ok_on_attempted"] / max(who["verified_attempted"], 1), 4),
                                               "bm25_on_attempted": round(who["verified_bm25_ok_on_attempted"] / max(who["verified_attempted"], 1), 4)}},
        "ms_per_question": {"bm25": round(1000 * times["bm25"] / n, 2), "ask": round(1000 * times["ask"] / n, 2)},
        "examples": examples,
    }
    for k in ("corpus", "all", "attempted", "who_questions", "bm25_proposes_key_verifies", "ms_per_question"):
        print(k, json.dumps(board[k]))
    for b in ("ask_right", "ask_wrong", "bm25_only"):
        for x in board["examples"][b][:3]:
            print("  ", b, x)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "external_squad.json").write_text(json.dumps(board, indent=2))
    return board


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", required=True)
    ap.add_argument("--frontend", required=True)
    ap.add_argument("--linker", required=True)
    ap.add_argument("--typer", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit_questions", type=int, default=None)
    ap.add_argument("--limit_paragraphs", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.squad, a.frontend, a.linker, a.typer, a.out, a.limit_questions, a.limit_paragraphs)


if __name__ == "__main__":
    main()
