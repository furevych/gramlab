"""UD dependency tree -> CoNLL-2000-style base chunks (IOB2 tags).

Chunk inventory (as in CoNLL-2000): NP, VP, PP, ADJP, ADVP, SBAR, PRT, INTJ, O.
Chunks are non-overlapping, non-recursive and contiguous ("base" chunks):
  [NP He] [VP reckons] [NP the current account deficit] [VP will narrow]
  [PP to] [NP only # 1.8 billion] [PP in] [NP September] .

The mapping is rule-based over UD v2 relations.  It is deliberately simple and
transparent; its accuracy can be measured against gold chunk data (CoNLL-2000)
once a UD parser is available for that text.  Here it is applied to gold (GUM)
and silver (AMALGUM) trees so the two can be compared.
"""
from typing import List, Dict, Set
from .conllu import Sentence

NOMINAL_UPOS = {"NOUN", "PROPN", "PRON", "NUM", "SYM"}
# dependents absorbed into their nominal head's NP chunk
NP_INNER = {"det", "det:predet", "amod", "nummod", "compound", "flat", "flat:foreign",
            "fixed", "goeswith", "nmod:npmod", "nmod:tmod", "nummod:gov",
            "nmod:desc", "nmod:unmarked"}
HYPHENS = {"-", "‐", "–", "/"}
# relations under which a DET/ADJ token acts as a nominal itself ("this is", "the poor")
NOMINAL_FUNCS = {"nsubj", "nsubj:pass", "nsubj:outer", "obj", "iobj", "obl", "obl:npmod",
                 "obl:tmod", "obl:agent", "nmod", "nmod:poss", "root", "conj", "appos",
                 "dislocated", "vocative", "expl", "xcomp", "ccomp", "list", "parataxis"}
VERB_INNER = {"aux", "aux:pass"}


def _children(sent: Sentence) -> List[List[int]]:
    ch = [[] for _ in range(len(sent) + 1)]
    for i, h in enumerate(sent.heads, start=1):
        if 0 <= h <= len(sent):
            ch[h].append(i)
    return ch


def _contiguous_span(members: Set[int], center: int) -> List[int]:
    lo = center
    while (lo - 1) in members:
        lo -= 1
    hi = center
    while (hi + 1) in members:
        hi += 1
    return list(range(lo, hi + 1))


def chunk_sentence(sent: Sentence) -> List[str]:
    """Return IOB2 chunk tags, one per token (1-based internally)."""
    n = len(sent)
    ch = _children(sent)
    upos = [None] + sent.upos
    rel = [None] + sent.deprels
    head = [None] + sent.heads
    form = [None] + sent.forms
    label: Dict[int, str] = {}          # token index -> chunk type
    chunk_of: Dict[int, int] = {}       # token index -> chunk id
    chunks: List[List[int]] = []        # chunk id -> token indices

    def assign(span: List[int], typ: str):
        span = [i for i in span if i not in label]
        if not span:
            return None
        cid = len(chunks)
        chunks.append(span)
        for i in span:
            label[i] = typ
            chunk_of[i] = cid
        return cid

    def is_nominal_head(i: int) -> bool:
        if i in label:
            return False
        u = upos[i]
        r = rel[i]
        if r in NP_INNER and head[i] and (upos[head[i]] in NOMINAL_UPOS or upos[head[i]] in {"DET", "ADJ"}):
            return False  # absorbed by its nominal head
        if r == "nmod:poss" and head[i] and u in {"PRON", "DET"} and not any(rel[c] == "case" for c in ch[i]):
            return False  # pronominal possessive ("his") is absorbed
        if u in NOMINAL_UPOS:
            return True
        if u == "DET" and r in NOMINAL_FUNCS:
            return True
        if u == "ADJ" and any(rel[c] == "det" for c in ch[i]):   # "the poor", "the following"
            return True
        return False

    def absorb_np(i: int, acc: Set[int]):
        for c in ch[i]:
            r = rel[c]
            u = upos[c]
            if u == "PUNCT":
                if form[c] in HYPHENS:      # "Romantic - era composer" (hyphen split by tokenizer)
                    acc.add(c)
                continue
            if r in NP_INNER:
                acc.add(c)
                absorb_inner(c, acc)
            elif r == "nmod:poss" and u in {"PRON", "DET"} and not any(rel[g] == "case" for g in ch[c]):
                acc.add(c)
                absorb_inner(c, acc)
            elif r == "case" and u == "PART":          # possessive 's
                acc.add(c)
            elif r == "advmod" and u in {"ADV", "PART"} and c < i:
                # "only 1.8 billion", "even the", "not everyone" -- CoNLL keeps these in the NP
                acc.add(c)

    def absorb_inner(i: int, acc: Set[int]):
        for c in ch[i]:
            r = rel[c]
            u = upos[c]
            if u == "PUNCT":
                if form[c] in HYPHENS:
                    acc.add(c)
                continue
            if r in NP_INNER or (r == "advmod" and u in {"ADV", "PART"}) or (r == "case" and u == "PART"):
                acc.add(c)
                absorb_inner(c, acc)
            elif r in {"conj", "cc"} and upos[i] == "ADJ" and u in {"ADJ", "CCONJ"}:
                acc.add(c)                       # "red and blue cars"
                absorb_inner(c, acc)

    # ---- 1. NP chunks ----
    for i in range(1, n + 1):
        if not is_nominal_head(i):
            continue
        acc = {i}
        absorb_np(i, acc)
        span = _contiguous_span(acc, i)
        assign(span, "NP")

    # ---- 2. VP chunks (verbs, copulas) ----
    vp_span_of: Dict[int, int] = {}
    for i in range(1, n + 1):
        if i in label:
            continue
        u = upos[i]
        if u == "VERB" or (u == "AUX" and rel[i] not in VERB_INNER and rel[i] != "cop"):
            acc = {i}
            for c in ch[i]:
                r = rel[c]
                if r in VERB_INNER:
                    acc.add(c)
                elif r == "mark" and form[c].lower() == "to" and upos[c] == "PART":
                    acc.add(c)
                elif r == "advmod" and upos[c] in {"ADV", "PART"} and c < i:
                    acc.add(c)
            # adverbs are kept only if they sit inside the aux..verb cluster
            core = {c for c in acc if rel[c] in VERB_INNER or c == i or (rel[c] == "mark")}
            lo = min(core)
            acc = {c for c in acc if c >= lo or rel[c] != "advmod"}
            span = _contiguous_span(acc, i)
            cid = assign(span, "VP")
            if cid is not None:
                vp_span_of[i] = cid
    # copular predicates: "is excellent" -> VP(is) ADJP(excellent)
    for i in range(1, n + 1):
        cops = [c for c in ch[i] if rel[c] == "cop" and c not in label]
        if not cops:
            continue
        acc = set(cops)
        for c in ch[i]:
            if rel[c] in VERB_INNER or (rel[c] == "advmod" and upos[c] in {"ADV", "PART"} and c < i and c > min(cops) - 1):
                acc.add(c)
        for c in cops:
            span = _contiguous_span(acc, c)
            assign(span, "VP")
    # merge "want to sell"-style xcomp chains that are adjacent
    for v, cid in list(vp_span_of.items()):
        h = head[v]
        if h and rel[v] == "xcomp" and h in vp_span_of:
            a, b = chunks[vp_span_of[h]], chunks[cid]
            if a and b and max(a) + 1 == min(b):
                merged = a + b
                for t in b:
                    chunk_of[t] = vp_span_of[h]
                chunks[vp_span_of[h]] = merged
                chunks[cid] = []

    # ---- 3. PP / PRT / SBAR / ADJP / ADVP / INTJ ----
    for i in range(1, n + 1):
        if i in label:
            continue
        u, r = upos[i], rel[i]
        if u == "ADP":
            if r == "compound:prt":
                assign([i], "PRT")
            elif r == "case" or r == "mark":
                acc = {i} | {c for c in ch[i] if rel[c] == "fixed"}
                assign(_contiguous_span(acc, i), "PP")
            else:
                assign([i], "ADVP")
        elif u == "SCONJ":
            assign([i], "SBAR")
        elif u == "ADJ":
            acc = {i} | {c for c in ch[i] if rel[c] == "advmod" and upos[c] in {"ADV", "PART"} and c < i}
            assign(_contiguous_span(acc, i), "ADJP")
        elif u == "ADV":
            acc = {i} | {c for c in ch[i] if rel[c] == "advmod" and upos[c] in {"ADV", "PART"} and c < i}
            assign(_contiguous_span(acc, i), "ADVP")
        elif u == "INTJ":
            assign([i], "INTJ")
        elif u == "PART" and r == "compound:prt":
            assign([i], "PRT")
        elif u in {"DET", "NUM", "PRON", "NOUN", "PROPN"}:
            assign([i], "NP")

    # ---- IOB2 output ----
    tags = ["O"] * (n + 1)
    for cid, span in enumerate(chunks):
        if not span:
            continue
        span = sorted(span)
        typ = label[span[0]]
        prev = None
        for t in span:
            tags[t] = ("B-" if prev is None or t != prev + 1 else "I-") + typ
            prev = t
    return tags[1:]


def chunks_from_tags(tags: List[str]):
    """Yield (type, [token indices 0-based]) from IOB2 tags."""
    cur_t, cur = None, []
    for i, tg in enumerate(tags):
        if tg == "O" or tg.startswith("B-") or (tg.startswith("I-") and cur_t != tg[2:]):
            if cur:
                yield cur_t, cur
            cur_t, cur = (None, []) if tg == "O" else (tg[2:], [i])
        else:
            cur.append(i)
    if cur:
        yield cur_t, cur
