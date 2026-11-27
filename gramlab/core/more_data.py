"""run 21 · more and broader training data for the one function that works: the role linker (+ its live front end).

Sources beyond AMALGUM (which already spans 8 GUM genres, Wikipedia-family text):
  gold   UD English treebanks with gold relations, other registers: ParTUT (legal, talks, wiki), LinES (literature,
         manuals, Europarl), PUD (news/wiki), GENTLE (dictionary, esports, legal, medical, poetry, proofs, syllabi,
         threat letters), Atis (flight-booking dialogue), Pronouns (constructed). EWT stays out (project decision).
  silver non-Wikipedia text parsed by Stanza (UD-native tags and relations): Brown (15 genres, 1961), Gutenberg (18
         classic books), Reuters (newswire), Webtext (forums, ads, scripts), NPS chat.

Experiments, all scored on the fixed benchmark (GUM test, gold tags) and live (front-end tags), plus each gold
treebank's own test split as an out-of-domain check:
  A  AMALGUM only (run 9 recipe)          B  A + extra gold          C  A + silver          D  A + gold + silver
Thresholds are chosen on GUM dev as in run 9; nothing is tuned on any test split.

    from gramlab.core import more_data
    board = more_data.run(GUM, AMALGUM, extra={'partut': 'data/ud/UD_English-ParTUT', ...}, silver_sents=..., out_dir=...)
"""
from __future__ import annotations

import glob
import json
import random
import time
from collections import Counter
from pathlib import Path

from .bench import _WithThreshold, bench_links
from .linker import LearnedLinker, predicates
from .records import Sentence, read_conllu, read_conllu_glob
from .tagger import Lemmatizer, LiveFrontEnd, PerceptronTagger


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------------------------------------ sources
def read_treebank(dirpath):
    """train/dev/test Sentence lists of a UD treebank directory (missing splits -> [])."""
    out = {}
    for split in ("train", "dev", "test"):
        files = glob.glob(f"{dirpath}/*-ud-{split}.conllu")
        out[split] = read_conllu(files[0]) if files else []
    if not out["train"] and not out["dev"] and out["test"]:       # PUD / Pronouns: test only -> use as held-out only
        pass
    return out


def stanza_silver(texts, lang="en", batch=2000, max_sents=None, doc="silver", log=log):
    """Parse raw texts with Stanza into Sentence records (UD tags and relations). Needs the stanza package + model."""
    import stanza
    nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse", use_gpu=False, verbose=False,
                          tokenize_batch_size=64, pos_batch_size=3000, depparse_batch_size=3000)
    sents, t = [], time.time()
    for bi in range(0, len(texts), batch):
        chunk = texts[bi:bi + batch]
        for d in nlp.bulk_process([stanza.Document([], text=x) for x in chunk]):
            for si, s in enumerate(d.sentences):
                words = [w for w in s.words]
                if not (3 <= len(words) <= 60):
                    continue
                sents.append(Sentence(
                    sid=f"{doc}-{len(sents)}", forms=[w.text for w in words], lemmas=[w.lemma or w.text for w in words],
                    upos=[w.upos or "X" for w in words], xpos=[w.xpos or "XX" for w in words],
                    heads=[(w.head - 1) if w.head else -1 for w in words], deprels=[w.deprel or "dep" for w in words],
                    ents=[None] * len(words), doc=doc))
        log(f"  stanza: {min(bi + batch, len(texts))}/{len(texts)} texts -> {len(sents)} sentences ({time.time() - t:.0f}s)")
        if max_sents and len(sents) >= max_sents:
            break
    return sents[:max_sents] if max_sents else sents


def nltk_texts(root, max_chars_per_file=20000):
    """Plain texts from unzipped nltk corpora directories (brown is tagged: strip the /TAG suffixes)."""
    out = {}
    for name in ("brown", "gutenberg", "reuters", "webtext", "nps_chat"):
        d = Path(root) / name
        if not d.exists():
            continue
        texts = []
        if name == "brown":
            for f in sorted(d.glob("c[a-r][0-9][0-9]")):
                toks = [t.rsplit("/", 1)[0] for t in f.read_text(errors="ignore").split()]
                texts.append(" ".join(toks)[:max_chars_per_file])
        elif name == "nps_chat":
            import re
            for f in sorted(d.glob("*.xml")):
                posts = re.findall(r'<Post[^>]*>([^<]{3,})</Post>|<Post[^>]*>([^<]{3,})<terminals', f.read_text(errors="ignore"))
                texts.append(" ".join(a or b for a, b in posts)[:max_chars_per_file])
        elif name == "reuters":
            for f in sorted((d / "training").glob("*"))[:1500]:
                texts.append(f.read_text(errors="ignore")[:max_chars_per_file])
        else:
            for f in sorted(d.glob("*.txt")):
                txt = f.read_text(errors="ignore")
                for i in range(0, min(len(txt), 400000), max_chars_per_file):
                    texts.append(txt[i:i + max_chars_per_file])
        out[name] = texts
    return out


# ------------------------------------------------------------------------------------------------ experiment
def role_stats(sents):
    c = Counter()
    for s in sents:
        for v, voice, a, p in predicates(s):
            c["events"] += 1; c[voice] += 1
    return dict(c)


def fit_arm(name, train, gum_dev, n_live=40000, seed=0, log=log):
    t = time.time()
    m = LearnedLinker(words_only=False).fit(train)
    best = max(((th, bench_links(_WithThreshold(m, th), gum_dev)[0]["all"]["f1"]) for th in (0.3, 0.4, 0.5)), key=lambda x: x[1])
    m.threshold = best[0]
    rng = random.Random(seed)
    sub = list(train); rng.shuffle(sub); sub = sub[:n_live]
    tagger = PerceptronTagger().fit(sub, epochs=4)
    lem = Lemmatizer().fit(train)
    fe = LiveFrontEnd(tagger, lem, None)
    log(f"[{name}] linker on {len(train)} sentences / {m.n_pairs} pairs, dev threshold {m.threshold} (dev F1 {best[1]:.3f}); front end on {len(sub)}; {time.time() - t:.0f}s")
    return m, fe


def score(name, m, fe, evals, log=log):
    """evals: {label: sentences with gold tags and roles}. Returns F1 gold-tags and live for each."""
    out = {}
    for label, sents in evals.items():
        if not sents:
            continue
        gold_f1 = bench_links(m, sents)[0]["all"]["f1"]
        live = [fe.sentence(s.forms, s.sid, s.doc) for s in sents]
        live_f1 = bench_links(m, sents, input_sents=live)[0]["all"]["f1"]
        n = sum(len(s) for s in sents)
        tag_acc = sum(g.upos[i] == l.upos[i] for g, l in zip(sents, live) for i in range(len(g))) / max(n, 1)
        out[label] = {"f1": round(gold_f1, 4), "f1_live": round(live_f1, 4), "upos_live": round(tag_acc, 4), "sentences": len(sents)}
        log(f"  [{name}] {label:12} F1 {gold_f1:.3f}  live {live_f1:.3f}  (upos {tag_acc:.3f}, n={len(sents)})")
    return out


def run(gum_dir, amalgum_glob, extra=None, silver_sents=None, out_dir="results", seed=0, arms=("A", "B", "C", "D"), save_arm="D", train_limit=None):
    t0 = time.time()
    gum_dir = Path(gum_dir)
    gum_dev = read_conllu(gum_dir / "en_gum-ud-dev.conllu")
    gum_test = read_conllu(gum_dir / "en_gum-ud-test.conllu")
    amalgum = read_conllu_glob(amalgum_glob, train_limit, seed)
    extra = extra or {}
    tb = {k: read_treebank(v) for k, v in extra.items()}
    gold_extra = [s for k in tb for s in tb[k]["train"] + tb[k]["dev"]]
    evals = {"gum_test": gum_test}
    for k in tb:
        if tb[k]["test"]:
            evals[k + "_test"] = tb[k]["test"]
    silver_sents = silver_sents or []
    board = {"sources": {"amalgum": {"sentences": len(amalgum), **role_stats(amalgum)},
                         "gold_extra": {k: {"train+dev": len(tb[k]["train"]) + len(tb[k]["dev"]), "test": len(tb[k]["test"]), **role_stats(tb[k]["train"] + tb[k]["dev"])} for k in tb},
                         "silver": {"sentences": len(silver_sents), **role_stats(silver_sents)}},
             "arms": {}}
    log(f"AMALGUM {len(amalgum)} · gold extra {len(gold_extra)} · silver {len(silver_sents)} · evals {{{', '.join(f'{k}:{len(v)}' for k, v in evals.items())}}}")
    sets = {"A": amalgum, "B": amalgum + gold_extra, "C": amalgum + silver_sents, "D": amalgum + gold_extra + silver_sents}
    for arm in arms:
        if arm in ("B", "D") and not gold_extra:
            continue
        if arm in ("C", "D") and not silver_sents:
            continue
        m, fe = fit_arm(arm, sets[arm], gum_dev, seed=seed)
        board["arms"][arm] = {"train_sentences": len(sets[arm]), "threshold": m.threshold, "scores": score(arm, m, fe, evals)}
        if out_dir and arm == save_arm:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            m.save(Path(out_dir) / "linker.pkl"); fe.save(Path(out_dir) / "frontend.pkl")
    board["seconds"] = round(time.time() - t0, 1)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "more_data.json").write_text(json.dumps(board, indent=2))
    log(f"done in {board['seconds']}s")
    return board
