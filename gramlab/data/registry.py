"""Corpus registry: every data source is an entry in corpora.json and a loader here.

A loader yields *documents*; a document is a list of sentences; a sentence is a dict
    {"forms": [...], "upos": [...] or None, "chunk": [...] or None}
Tags are optional: plain text has none and the grammar heads simply skip it.

Adding a new corpus = add an entry to corpora.json (and, for a new *format*, a
loader function + a line in LOADERS). Nothing else in the project changes.

Entry types:
  ud_conllu_dir   {"path": dir with *.conllu (recursively)}            UD trees -> tags via the chunker
  conllu_file     {"path": one .conllu file, "genres": [optional filter]}
  jsonl_sentences {"path": file.jsonl}   one sentence per line with forms/upos/chunk (grammstat format)
  plain_text      {"path": .txt (blank-line separated docs) or .jsonl with a "text" field}
  dialogue_jsonl  {"path": .jsonl, each line {"messages":[{"role","content"},...]}}  (fine-tuning only)
"""
import glob, json, os, re
from typing import Iterator, List, Dict, Optional
from .conllu import read_conllu
from .chunker import chunk_sentence

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    """Whitespace+punctuation split that resembles UD tokenization for plain text."""
    return _WORD.findall(text)


def _sent_from_conllu(s):
    return {"forms": s.forms, "upos": s.upos, "chunk": chunk_sentence(s)}


def load_ud_conllu_dir(entry) -> Iterator[List[Dict]]:
    files = sorted(glob.glob(os.path.join(entry["path"], "**", "*.conllu"), recursive=True))
    for f in files:
        doc, cur = [], None
        for s in read_conllu(f):
            if cur is not None and s.doc_id != cur and doc:
                yield doc
                doc = []
            cur = s.doc_id
            doc.append(_sent_from_conllu(s))
        if doc:
            yield doc


def load_conllu_file(entry) -> Iterator[List[Dict]]:
    genres = set(entry.get("genres") or [])
    doc, cur = [], None
    for s in read_conllu(entry["path"]):
        if genres and s.genre not in genres:
            continue
        if cur is not None and s.doc_id != cur and doc:
            yield doc
            doc = []
        cur = s.doc_id
        doc.append(_sent_from_conllu(s))
    if doc:
        yield doc


def load_jsonl_sentences(entry) -> Iterator[List[Dict]]:
    doc, cur = [], None
    with open(entry["path"], encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if cur is not None and r.get("doc") != cur and doc:
                yield doc
                doc = []
            cur = r.get("doc")
            doc.append({"forms": r["forms"], "upos": r.get("upos"), "chunk": r.get("chunk")})
    if doc:
        yield doc


def load_plain_text(entry) -> Iterator[List[Dict]]:
    path = entry["path"]

    def to_doc(text):
        sents = []
        for para in re.split(r"\n\s*\n|\n", text):
            para = para.strip()
            if not para:
                continue
            # crude sentence split; good enough for a pretraining stream
            for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", para):
                toks = simple_tokenize(sent)
                if toks:
                    sents.append({"forms": toks, "upos": None, "chunk": None})
        return sents

    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                d = to_doc(json.loads(line)["text"])
                if d:
                    yield d
    else:
        with open(path, encoding="utf-8") as fh:
            blob = fh.read()
        for chunk in re.split(r"\n\s*\n", blob):
            d = to_doc(chunk)
            if d:
                yield d


def load_dialogue_jsonl(entry) -> Iterator[Dict]:
    """Yields {"messages": [...]} records (used by finetune.py, not by prepare.py)."""
    with open(entry["path"], encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


LOADERS = {
    "ud_conllu_dir": load_ud_conllu_dir,
    "conllu_file": load_conllu_file,
    "jsonl_sentences": load_jsonl_sentences,
    "plain_text": load_plain_text,
    "dialogue_jsonl": load_dialogue_jsonl,
}


def load_registry(path="corpora.json") -> Dict[str, Dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def iter_docs(name: str, registry: Dict[str, Dict], limit_docs: Optional[int] = None):
    entry = registry[name]
    loader = LOADERS[entry["type"]]
    for i, doc in enumerate(loader(entry)):
        if limit_docs and i >= limit_docs:
            break
        yield doc
