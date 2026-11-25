"""core-v1 · benchmarks — one fixed scoreboard, run on every change.

    python -m gramlab.core.bench --gum data/gum --amalgum 'data/amalgum/amalgum/*/dep/*.conllu' [--vad file] [--out results]

Rows (all on gold GUM test, never trained on):
  links     linker vs UD gold: link F1 (all / active / passive), role swaps, µs per sentence
  keys      ordered class key (class(actor), action, class(patient)) exact vs gold; position-only control
  typing    layer-1 entity typing of event arguments vs annotation (first occupant: lemma-majority prior)
  general   layer-0 coverage: share of action lemmas / general argument lemmas with a concept entry
  affect    lexicon coverage of actions and share of 'power' events (skipped without a lexicon file)
  index     ingest GUM test through the pipeline: occurrences, keys, occurrences per family key
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from .affect import Affect
from .linker import BackfillLinker, GoldLinker, LearnedLinker, RuleLinker, predicates
from .pipeline import Pipeline
from .records import NOMINAL, read_conllu, read_conllu_glob
from .registry import GeneralIndex, SpecificIndex


# ----------------------------------------------------------------------------- links
def bench_links(linker, sents, pred_batches=None, input_sents=None):
    """linker must expose link_batch or link. Compares argument heads to UD gold per predicate.
    input_sents: the sentences the linker reads (e.g. live-tagged copies); scoring always uses `sents` (gold)."""
    t = time.time()
    src = input_sents if input_sents is not None else sents
    if pred_batches is None:
        pred_batches = linker.link_batch(src) if hasattr(linker, "link_batch") else [linker.link(s) for s in src]
    us = (time.time() - t) / max(len(sents), 1) * 1e6
    tp, fp, fn, swaps, n = Counter(), Counter(), Counter(), Counter(), Counter()
    for s, evs in zip(sents, pred_batches):
        by_verb = {e.verb: e for e in evs}
        for v, voice, a, p in predicates(s):
            e = by_verb.get(v)
            pa = e.actor.head if e and e.actor else None
            pp = e.patient.head if e and e.patient else None
            n[voice] += 1
            for gold, pr in ((a, pa), (p, pp)):
                if gold is not None and pr == gold:
                    tp[voice] += 1
                else:
                    if gold is not None:
                        fn[voice] += 1
                    if pr is not None:
                        fp[voice] += 1
            if (pa is not None and pa == p) or (pp is not None and pp == a):
                swaps[voice] += 1
    out = {"us_per_sentence": round(us, 1)}
    for voice in ("all", "active", "passive"):
        ks = ("active", "passive") if voice == "all" else (voice,)
        TP, FP, FN = sum(tp[k] for k in ks), sum(fp[k] for k in ks), sum(fn[k] for k in ks)
        pr = TP / (TP + FP) if TP + FP else 0.0
        rc = TP / (TP + FN) if TP + FN else 0.0
        out[voice] = {"f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0, "precision": round(pr, 4),
                      "recall": round(rc, 4), "predicates": sum(n[k] for k in ks), "role_swaps": sum(swaps[k] for k in ks)}
    return out, pred_batches


# ----------------------------------------------------------------------------- keys
def bench_keys(pipe: Pipeline, sents, pred_batches):
    gold_linker = GoldLinker()
    res = defaultdict(Counter)
    for s, evs in zip(sents, pred_batches):
        by_verb = {e.verb: e for e in evs}
        for g in gold_linker.link(s):
            if g.actor is None or g.patient is None:
                continue
            gold = pipe.keyer.class_triple(g)
            e = by_verb.get(g.verb)
            pred = pipe.keyer.class_triple(e) if e else ("missing", g.action, "missing")
            v = g.verb
            b = next((i for i in range(v - 1, -1, -1) if s.upos[i] in NOMINAL), None)
            f = next((i for i in range(v + 1, len(s)) if s.upos[i] in NOMINAL), None)
            pos = (pipe.keyer.slot(s.mention(b)).cls if b is not None else "missing", g.action,
                   pipe.keyer.slot(s.mention(f)).cls if f is not None else "missing")
            c = res[g.voice]
            c["events"] += 1
            c["key_exact"] += pred == gold
            c["key_reversed"] += (pred[0] == gold[2] and pred[2] == gold[0] and gold[0] != gold[2])
            c["position_only_exact"] += pos == gold
            c["classes_known"] += ("unknown" not in gold and "pronoun" not in gold)
            c["pronoun_args"] += sum(m.kind == "pronoun" for m in (g.actor, g.patient))
    tot = Counter()
    for c in res.values():
        tot.update(c)
    res["all"] = tot
    out = {}
    for voice, c in res.items():
        n = c["events"] or 1
        out[voice] = {"events": c["events"], "key_exact": round(c["key_exact"] / n, 4),
                      "position_only_control": round(c["position_only_exact"] / n, 4),
                      "reversed": c["key_reversed"], "classes_known": round(c["classes_known"] / n, 4),
                      "pronoun_arg_share": round(c["pronoun_args"] / (2 * n), 4)}
    return out


# ----------------------------------------------------------------------------- typing
class PriorTyper:
    """First layer-1 occupant for entity typing: majority type per lemma on GUM train; unseen PROPN →
    person, unseen NOUN → abstract. Exists so the benchmark has a row to beat."""
    name = "lemma-majority prior"

    def fit(self, sents):
        c = defaultdict(Counter)
        for s in sents:
            for i in range(len(s)):
                if s.upos[i] in NOMINAL and s.ents[i]:
                    c[s.lemmas[i].lower()][s.ents[i]] += 1
        self.table = {l: cnt.most_common(1)[0][0] for l, cnt in c.items()}
        return self

    def predict(self, s, i):
        return self.table.get(s.lemmas[i].lower(), "person" if s.upos[i] == "PROPN" else "abstract")


def bench_typing(typer, sents):
    n = Counter()
    for s in sents:
        for v, voice, a, p in predicates(s):
            for g in (a, p):
                if g is None or s.upos[g] == "PRON":
                    continue
                gold = s.mention(g).etype
                if gold == "unknown":
                    continue
                kind = "specific" if s.upos[g] == "PROPN" else "general"
                n[kind, "n"] += 1
                n[kind, "ok"] += typer.predict(s, g) == gold
    return {k: {"n": n[k, "n"], "accuracy": round(n[k, "ok"] / max(n[k, "n"], 1), 4)} for k in ("general", "specific")}


# ----------------------------------------------------------------------------- general coverage
def bench_general(general: GeneralIndex, sents):
    """Layer-0 coverage, and the family question: at which hypernym depth do action families merge the verbs
    that WordNet itself calls synonyms (share any synset) without merging unrelated ones?
    synonym_pairs_merged = recall on synonym lemma pairs; unrelated_share = share of merged pairs with no shared synset."""
    acts, gens = Counter(), Counter()
    verbs = Counter()
    for s in sents:
        for v, voice, a, p in predicates(s):
            c = general.concept(s.lemmas[v], "VERB")
            acts[c["covered"]] += 1
            if c["covered"]:
                verbs[s.lemmas[v].lower()] += 1
            for g in (a, p):
                if g is not None and s.upos[g] == "NOUN":
                    gens[general.concept(s.lemmas[g], "NOUN")["covered"]] += 1
    out = {"backend": general.backend, "family_depth": general.family_depth,
           "actions_covered": round(acts[True] / max(sum(acts.values()), 1), 4),
           "general_args_covered": round(gens[True] / max(sum(gens.values()), 1), 4),
           "distinct_action_lemmas_covered": len(verbs), "families": {}}
    if general.backend == "lemma" or not verbs:
        return out
    lemmas = sorted(verbs)
    syn = {l: set(map(str, general.synsets(l, "VERB"))) for l in lemmas}
    synonym_pairs = {(a, b) for i, a in enumerate(lemmas) for b in lemmas[i + 1:] if syn[a] & syn[b]}
    for depth in (0, 1, 2, 3, 4, "lexname"):
        fam = {}
        for l in lemmas:
            c = general.concept(l, "VERB", depth=depth if depth != "lexname" else 0)
            fam[l] = c["lexname"] if depth == "lexname" else c["family_name"]
        groups = defaultdict(list)
        for l, f in fam.items():
            groups[f].append(l)
        merged = {(a, b) for g in groups.values() for i, a in enumerate(sorted(g)) for b in sorted(g)[i + 1:]}
        unrelated = sum(1 for a, b in merged if not (syn[a] & syn[b]))
        big = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:3]
        out["families"][str(depth)] = {
            "families": len(groups), "verbs_per_family": round(len(lemmas) / len(groups), 2),
            "synonym_pairs_merged": round(len(merged & synonym_pairs) / max(len(synonym_pairs), 1), 3),
            "merged_pairs": len(merged), "unrelated_share": round(unrelated / max(len(merged), 1), 3),
            "largest": [(f, len(g), g[:6]) for f, g in big]}
    out["synonym_pairs_in_test"] = len(synonym_pairs)
    # multi-sense posting: an event is posted under each of its verb's top-k synsets; two lemmas are
    # "linked" when they share a posted synset. Recall on synonym pairs vs fan-out (hub effect).
    out["multi_sense_postings"] = {}
    for k in (1, 2, 3, 5, 99):
        top = {l: set(map(str, general.synsets(l, "VERB")[:k])) for l in lemmas}
        linked = {(a, b) for i, a in enumerate(lemmas) for b in lemmas[i + 1:] if top[a] & top[b]}
        fan = Counter()
        for a, b in linked:
            fan[a] += 1
            fan[b] += 1
        out["multi_sense_postings"][str(k)] = {
            "synonym_pairs_linked": round(len(linked & synonym_pairs) / max(len(synonym_pairs), 1), 3),
            "linked_pairs": len(linked), "mean_links_per_verb": round(sum(fan.values()) / len(lemmas), 2),
            "max_links": max(fan.values()) if fan else 0, "hub": fan.most_common(1)[0][0] if fan else None}
    return out


# ----------------------------------------------------------------------------- affect
def bench_affect(affect: Affect, sents):
    if not affect.lex:
        return {"status": "skipped (no lexicon file; pass --vad)"}
    gold = GoldLinker()
    cov, power, total = 0, 0, 0
    examples = []
    for s in sents:
        for e in gold.link(s):
            total += 1
            f = affect.force(e)
            if f is not None:
                cov += 1
                if Affect.power(f):
                    power += 1
                    if len(examples) < 5:
                        examples.append((e.actor.text if e.actor else None, e.action, e.patient.text if e.patient else None, f))
    return {"source": affect.source, "lexicon_words": len(affect.lex), "actions_covered": round(cov / max(total, 1), 4),
            "power_event_share": round(power / max(cov, 1), 4), "examples": examples}


# ----------------------------------------------------------------------------- index
def bench_index(pipe: Pipeline, sents, sense_picker=None, resolver=None):
    gold_pipe = Pipeline(linker=GoldLinker(), general=pipe.general, specific=pipe.specific, sense_picker=sense_picker, resolver=resolver)
    n = gold_pipe.ingest(sents)
    st = gold_pipe.index.stats()
    per_family = Counter(k for (lvl, k, _) in gold_pipe.index.postings if lvl == "family")
    sizes = sorted((len(v) for (lvl, k, _), v in gold_pipe.index.postings.items() if lvl == "family"), reverse=True)
    biggest = []
    for (lvl, k, status), v in sorted(gold_pipe.index.postings.items(), key=lambda kv: -len(kv[1]))[:5]:
        if lvl == "family":
            biggest.append((gold_pipe.keyer.decode(k), status, len(v)))
    return {"occurrences": n, **st, "family_postings_top": biggest,
            "pronoun_slots_resolved_to_nominal": gold_pipe.keyer.resolved, "pronoun_slots_unresolved": gold_pipe.keyer.unresolved,
            "family_posting_sizes_p50_p90_max": [sizes[len(sizes) // 2], sizes[len(sizes) // 10], sizes[0]] if sizes else [],
            "specific_registry": gold_pipe.specific.stats()}


# ----------------------------------------------------------------------------- coreference
def bench_coreference(sents, train=None, dev=None):
    """Returns the row and, under "_resolver", the trained ranker (alpha chosen on dev, refit on train+dev)."""
    from .coref import NearestNominal, RankedResolver, SubjectFirst, TypedNearest, bench_coref
    resolvers = [NearestNominal(), TypedNearest(), SubjectFirst()]
    rr = None
    if train is not None:
        t = time.time()
        alpha = 1e-5
        if dev is not None:
            best = max(((a, bench_coref(RankedResolver(alpha=a).fit(train), dev, predicates)["accuracy"]) for a in (3e-6, 1e-5, 3e-5, 1e-4)),
                       key=lambda x: x[1])
            alpha = best[0]
        rr = RankedResolver(alpha=alpha).fit(train + (dev or []))
        resolvers.append(rr)
    out = {r.name: bench_coref(r, sents, predicates) for r in resolvers}
    if rr is not None:
        out[rr.name].update({"fit_seconds": round(time.time() - t, 1), "train_pairs": rr.n_pairs, "alpha": rr.alpha})
        out["_resolver"] = rr
    return out


# ----------------------------------------------------------------------------- live front end
def bench_live(linker, gum_train, train_sents, gum_test, n_train=40000, epochs=4, out_dir=None):
    """Tagger + lemmatizer trained on AMALGUM silver tags; GUM test re-tagged on gold tokens (no parse, no entities);
    the linker runs on the re-tagged sentences and is scored against gold roles. Cost of losing the parse."""
    from .tagger import Lemmatizer, LiveFrontEnd, PerceptronTagger
    t = time.time()
    sub = train_sents[:n_train] if train_sents else gum_train
    tagger = PerceptronTagger().fit(sub, epochs=epochs)
    lem = Lemmatizer().fit(train_sents or gum_train)
    fe = LiveFrontEnd(tagger, lem)
    fit_s = time.time() - t
    t = time.time()
    live = [fe.sentence(s.forms, s.sid, s.doc) for s in gum_test]
    tag_us = (time.time() - t) / len(gum_test) * 1e6
    n = u_ok = x_ok = l_ok = 0
    for g, l in zip(gum_test, live):
        for i in range(len(g)):
            n += 1
            u_ok += g.upos[i] == l.upos[i]
            x_ok += g.xpos[i] == l.xpos[i]
            l_ok += g.lemmas[i].lower() == l.lemmas[i]
    # predicates found: gold predicate verbs recovered by the live rule, and voice agreement
    gold_pred = {(si, v): voice for si, s in enumerate(gum_test) for v, voice, _, _ in predicates(s)}
    live_pred = {(si, v): voice for si, s in enumerate(live) for v, voice, _, _ in predicates(s)}
    found = sum(1 for k in gold_pred if k in live_pred)
    voice_ok = sum(1 for k, vc in gold_pred.items() if live_pred.get(k) == vc)
    links, _ = bench_links(linker, gum_test, input_sents=live)
    out = {"front_end": fe.name, "train_sentences": len(sub), "fit_seconds": round(fit_s, 1), "tag_us_per_sentence": round(tag_us, 0),
           "upos_accuracy": round(u_ok / n, 4), "xpos_accuracy": round(x_ok / n, 4), "lemma_accuracy": round(l_ok / n, 4),
           "gold_predicates": len(gold_pred), "live_predicates": len(live_pred), "gold_predicates_found": round(found / len(gold_pred), 4),
           "voice_agreement_on_found": round(voice_ok / max(found, 1), 4), "links_live": links}
    if out_dir:
        fe.save(Path(out_dir) / "frontend.pkl")
    return out


# ----------------------------------------------------------------------------- run
def bench_senses(general: GeneralIndex, semcor_dir):
    if general._wn is None or not semcor_dir or not Path(semcor_dir, "brown2").exists():
        return {"status": "skipped (needs WordNet files and --semcor dir)"}
    from .sense import FirstSense, LeskPicker, NaiveBayesPicker, RankPicker, bench_sense, read_semcor
    wn = general._wn
    train = read_semcor(semcor_dir, wn, ("brown1",))
    dev = read_semcor(semcor_dir, wn, ("brownv",))
    test = read_semcor(semcor_dir, wn, ("brown2",))
    nb = NaiveBayesPicker(wn, alpha=3.0).fit(train + dev)
    t = time.time()
    rank = RankPicker(wn, poses=("v", "n")).fit(train, dev)
    fit_s = time.time() - t
    r = bench_sense({"first sense (WordNet order)": FirstSense(wn), "extended Lesk": LeskPicker(wn),
                     "naive Bayes, SemCor-trained": nb, "shared rank classifier (SemCor)": rank}, test, wn)
    r["shared rank classifier (SemCor)"]["alpha"] = rank.alpha
    r["shared rank classifier (SemCor)"]["fit_seconds"] = round(fit_s, 1)
    r["note"] = "WordNet's sense order was derived from SemCor counts, so 'first sense' has seen the test set; a picker must beat it anyway"
    rank.poses = ("v",)                      # the occupant picks verbs only; nouns stay first-sense in the keyer
    rank.clf.pop("n", None)
    r["_picker"] = rank
    return r


def run(gum_dir, amalgum_glob=None, vad=None, out_dir="results", train_limit=None, linker_path=None, seed=0, semcor=None):
    gum_dir = Path(gum_dir)
    gum_train = read_conllu(gum_dir / "en_gum-ud-train.conllu")
    gum_dev = read_conllu(gum_dir / "en_gum-ud-dev.conllu")
    gum_test = read_conllu(gum_dir / "en_gum-ud-test.conllu")
    train = read_conllu_glob(amalgum_glob, train_limit, seed) if amalgum_glob and glob.glob(amalgum_glob) else gum_train
    print(f"GUM train/dev/test {len(gum_train)}/{len(gum_dev)}/{len(gum_test)}; linker training set {len(train)} sentences")

    board = {"core": "core-v1", "train_sentences": len(train), "linkers": {}}
    general = GeneralIndex()
    specific = SpecificIndex()
    affect = Affect(vad)

    linkers = {"rules": RuleLinker()}
    if linker_path and Path(linker_path).exists():
        linkers["learned"] = LearnedLinker.load(linker_path)
    else:
        t = time.time()
        m = LearnedLinker(words_only=False).fit(train)
        # threshold on dev, frozen before test
        best = max(((th, bench_links(_WithThreshold(m, th), gum_dev)[0]["all"]["f1"]) for th in (0.3, 0.4, 0.5)), key=lambda x: x[1])
        m.threshold = best[0]
        print(f"learned linker: fit {time.time()-t:.0f}s on {m.n_pairs} pairs, dev-chosen threshold {m.threshold} (dev F1 {best[1]:.3f})")
        linkers["learned"] = m
        if out_dir:
            m.save(Path(out_dir) / "linker.pkl")
    linkers["backfill"] = BackfillLinker(linkers["learned"])
    linkers["gold"] = GoldLinker()

    for name, lk in linkers.items():
        pipe = Pipeline(linker=lk, general=general, specific=specific, affect=affect)
        links, pred = bench_links(lk, gum_test)
        keys = bench_keys(pipe, gum_test, pred)
        board["linkers"][name] = {"links": links, "keys": keys}
        print(f"{name:8s} F1 all {links['all']['f1']:.3f} active {links['active']['f1']:.3f} passive {links['passive']['f1']:.3f} "
              f"swaps {links['all']['role_swaps']:3d} {links['us_per_sentence']:7.0f} µs/sent | key exact active {keys['active']['key_exact']:.3f} "
              f"passive {keys['passive']['key_exact']:.3f} (position-only {keys['active']['position_only_control']:.3f}/{keys['passive']['position_only_control']:.3f})")

    typer = PriorTyper().fit(gum_train)
    board["typing"] = bench_typing(typer, gum_test)
    board["general"] = bench_general(general, gum_test)
    board["affect"] = bench_affect(affect, gum_test)
    board["senses"] = bench_senses(general, semcor)
    board["coreference"] = bench_coreference(gum_test, gum_train, gum_dev)
    resolver = board["coreference"].pop("_resolver", None)
    board["live"] = bench_live(linkers["backfill"], gum_train, train if train is not gum_train else None, gum_test, out_dir=out_dir)
    picker = board["senses"].pop("_picker", None)
    board["index"] = bench_index(Pipeline(general=general, specific=specific), gum_test, picker, resolver)
    board["index"]["sense_picker"] = getattr(picker, "name", None)
    board["index"]["resolver"] = getattr(resolver, "name", None)
    for k in ("typing", "general", "affect", "senses", "coreference", "live", "index"):
        print(k, json.dumps(board[k], default=str))
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str))
    return board


class _WithThreshold:
    def __init__(self, m, th):
        self.m, self.th = m, th

    def link_batch(self, sents):
        roles = self.m.roles_batch(sents, self.th)
        from .linker import make_event
        return [[make_event(s, v, voice, *roles.get((si, v), (None, None))) for v, voice, _, _ in predicates(s)]
                for si, s in enumerate(sents)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gum", required=True)
    ap.add_argument("--amalgum", default=None)
    ap.add_argument("--vad", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--linker", default=None)
    ap.add_argument("--semcor", default=None)
    a = ap.parse_args(argv)
    run(a.gum, a.amalgum, a.vad, a.out, a.train_limit, a.linker, semcor=a.semcor)


if __name__ == "__main__":
    main()
