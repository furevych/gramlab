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

    def __init__(self, family_depth=0, wordnet_path=None):
        self.family_depth = family_depth
        self.backend = "lemma"
        self._wn = None            # dependency-free reader (core.wordnet)
        self._nltk = None          # nltk wordnet, if someone prefers it
        try:
            from .wordnet import WordNet
            self._wn = WordNet(wordnet_path)
            self.backend = "wordnet-file"
        except Exception:
            try:
                from nltk.corpus import wordnet as nwn
                nwn.synsets("wash")
                self._nltk = nwn
                self.backend = "wordnet-nltk"
            except Exception:
                pass
        self.table = {}                              # id -> canonical name (for decoding)

    # -- lookup -----------------------------------------------------------------------------
    def synsets(self, lemma, upos):
        """List of synset handles in sense order: (pos, offset) for the file reader, nltk Synsets otherwise."""
        pos = _UD_TO_WN.get(upos)
        if pos is None:
            return []
        if self._wn is not None:
            return [(pos, o) for o in self._wn.synsets(lemma, pos)]
        if self._nltk is not None:
            return self._nltk.synsets(lemma.lower().replace(" ", "_"), pos=pos)
        return []

    def _name(self, s):
        return self._wn.name(*s) if self._wn is not None else s.name()

    def _family_name(self, s, depth=None):
        depth = self.family_depth if depth is None else depth
        if self._wn is not None:
            pos, off = s
            return self._wn.name(pos, self._wn.family(pos, off, depth))
        x = s
        for _ in range(depth):
            hyper = x.hypernyms() or x.instance_hypernyms()
            if not hyper:
                break
            x = hyper[0]
        return x.name()

    def lexname(self, s):
        return self._wn.lexname(*s) if self._wn is not None else s.lexname()

    def concept(self, lemma, upos, sense=None, depth=None):
        """Returns dict(id, family, name, family_name, lexname, covered). `sense`: a synset name chosen by layer 1.
        `depth`: hypernym steps for the family (default self.family_depth)."""
        lemma = lemma.lower()
        ss = self.synsets(lemma, upos)
        if ss:
            s = ss[0]
            if sense is not None:
                for cand in ss:
                    if self._name(cand) == sense:
                        s = cand
                        break
            name = self._name(s)
            fam = self._family_name(s, depth)
            cid = stable_id("wn", name)
            fid = stable_id("wnfam", fam)
            self.table[cid] = name
            self.table[fid] = fam
            return {"id": cid, "family": fid, "name": name, "family_name": fam, "lexname": self.lexname(s), "covered": True}
        key = f"{lemma}/{upos}"
        cid = stable_id("lemma", key)
        self.table[cid] = key
        return {"id": cid, "family": cid, "name": key, "family_name": key, "lexname": None, "covered": False}

    def synonyms(self, lemma, upos, max_n=20):
        out = []
        for s in self.synsets(lemma, upos):
            names = self._wn.lemmas(*s) if self._wn is not None else [l.replace("_", " ") for l in s.lemma_names()]
            for l in names:
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
