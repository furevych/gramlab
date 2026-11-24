"""core-v1 · layer 0 — indexation, not recognition.

Two ID spaces, frozen in shape:

* GeneralIndex  — concepts that exist in the language (common nouns, verbs, adjectives).
                  Preset: WordNet synset IDs and their hypernym paths (via nltk when installed).
                  Fallback without WordNet: lemma+POS IDs with a one-level family (the lemma itself),
                  so the pipeline runs anywhere and the coverage benchmark reports what is missing.
* SpecificIndex — individuals that exist in the world (persons, places, organisations, addresses…).
                  A seed registry (optional JSON, e.g. Wikidata-derived) plus a local registry that
                  mints an ID for every unresolved name, keeps its surface forms and never guesses
                  identity: linking two mentions to one ID is layer-1 work.

IDs are deterministic (blake2b of the canonical string), so two machines index the same thing to the
same number without a central counter; the registries also keep the reverse table for decoding.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def stable_id(*parts, bits=48):
    """Deterministic integer ID for a canonical tuple of strings."""
    s = "\x1f".join(str(p) for p in parts)
    return int(hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest(), 16) >> (64 - bits)


_UD_TO_WN = {"NOUN": "n", "VERB": "v", "AUX": "v", "ADJ": "a", "ADV": "r"}


class GeneralIndex:
    """lemma + UPOS -> (synset id, family id at depth d, sense name).  First-sense policy: layer 0
    does not disambiguate; a layer-1 sense picker may pass `sense` explicitly."""

    def __init__(self, family_depth=2):
        self.family_depth = family_depth
        self.backend = "lemma"
        self._wn = None
        try:
            from nltk.corpus import wordnet as wn  # optional preset
            wn.synsets("wash")                      # raises LookupError when the data is not installed
            self._wn = wn
            self.backend = "wordnet"
        except Exception:
            pass
        self.table = {}                              # id -> canonical name (for decoding)

    # -- lookup -----------------------------------------------------------------------------
    def synsets(self, lemma, upos):
        if self._wn is None:
            return []
        pos = _UD_TO_WN.get(upos)
        if pos is None:
            return []
        return self._wn.synsets(lemma.lower().replace(" ", "_"), pos=pos)

    def concept(self, lemma, upos, sense=None):
        """Returns dict(id, family, name, covered). `sense`: a synset name chosen by layer 1."""
        lemma = lemma.lower()
        ss = self.synsets(lemma, upos)
        if ss:
            s = ss[0]
            if sense is not None:
                for cand in ss:
                    if cand.name() == sense:
                        s = cand
                        break
            name = s.name()
            fam = self._family(s)
            cid = stable_id("wn", name)
            fid = stable_id("wnfam", fam)
            self.table[cid] = name
            self.table[fid] = fam
            return {"id": cid, "family": fid, "name": name, "family_name": fam, "covered": True}
        key = f"{lemma}/{upos}"
        cid = stable_id("lemma", key)
        self.table[cid] = key
        return {"id": cid, "family": cid, "name": key, "family_name": key, "covered": False}

    def _family(self, synset):
        """Hypernym `family_depth` steps up (or the top of the chain), first path."""
        s = synset
        for _ in range(self.family_depth):
            hyper = s.hypernyms() or s.instance_hypernyms()
            if not hyper:
                break
            s = hyper[0]
        return s.name()

    def synonyms(self, lemma, upos, max_n=20):
        out = []
        for s in self.synsets(lemma, upos):
            for l in s.lemma_names():
                l = l.replace("_", " ")
                if l != lemma and l not in out:
                    out.append(l)
        return out[:max_n]

    def decode(self, cid):
        return self.table.get(cid)


_norm_re = re.compile(r"[^a-z0-9 ]+")


def normalize_name(text):
    t = text.lower().replace("’", "'")
    t = _norm_re.sub(" ", t)
    t = re.sub(r"\b(the|a|an)\b", " ", t)
    return " ".join(t.split())


class SpecificIndex:
    """Named individuals. `resolve(mention_text, etype)` returns an entry; identity is only ever
    claimed from the seed registry (exact alias match) — everything else is a local ID per
    (normalised surface form, type) with status 'unresolved'."""

    def __init__(self, seed_path=None):
        self.seed = {}                      # normalised alias -> {"id":..., "name":..., "etype":...}
        self.local = {}                     # id -> entry
        self.aliases = defaultdict(set)     # id -> surface forms seen
        self.mentions = defaultdict(int)    # id -> count
        if seed_path and Path(seed_path).exists():
            self.load_seed(seed_path)

    def load_seed(self, path):
        """JSON list of {"id": "Q3711", "name": "Belgrade", "etype": "place", "aliases": [...]}."""
        for row in json.loads(Path(path).read_text(encoding="utf-8")):
            entry = {"id": stable_id("seed", row["id"]), "source_id": row["id"], "name": row["name"],
                     "etype": row.get("etype", "unknown"), "status": "seed"}
            for alias in [row["name"]] + list(row.get("aliases", ())):
                self.seed[normalize_name(alias)] = entry

    def resolve(self, text, etype="unknown"):
        key = normalize_name(text)
        if not key:
            key = text.lower()
        hit = self.seed.get(key)
        if hit is not None and (etype in ("unknown", hit["etype"])):
            self.aliases[hit["id"]].add(text)
            self.mentions[hit["id"]] += 1
            return hit
        lid = stable_id("local", key, etype)
        entry = self.local.get(lid)
        if entry is None:
            entry = {"id": lid, "source_id": None, "name": text, "etype": etype, "status": "unresolved"}
            self.local[lid] = entry
        self.aliases[lid].add(text)
        self.mentions[lid] += 1
        return entry

    def decode(self, sid):
        return self.local.get(sid) or next((e for e in self.seed.values() if e["id"] == sid), None)

    def save(self, path):
        Path(path).write_text(json.dumps({"local": list(self.local.values()),
                                          "aliases": {str(k): sorted(v) for k, v in self.aliases.items()},
                                          "mentions": {str(k): v for k, v in self.mentions.items()}}, indent=1))

    def stats(self):
        return {"seed_aliases": len(self.seed), "local_entities": len(self.local),
                "mentions": sum(self.mentions.values())}
