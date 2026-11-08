"""Tokenizers with one interface: encode_words(words) -> (ids, word_index per id), encode(text), decode(ids).

  WordTokenizer  pure Python, one id per word, capped vocab + <unk>. Always available.
  BPETokenizer   byte-level BPE via the `tokenizers` library (on Kaggle). No <unk>, better for chat.
                 chunk_constrained=True trains merges only inside grammar chunks (SuperBPE-style
                 multi-word tokens whose boundaries come from the chunker) -- an experimental option.

Special ids are identical across tokenizers so checkpoints and chat templates agree:
  0 <pad>  1 <unk>  2 <eos>  3 <|user|>  4 <|assistant|>  5 <|end|>
"""
import json, os, re, collections
from typing import List, Tuple, Iterable

SPECIALS = ["<pad>", "<unk>", "<eos>", "<|user|>", "<|assistant|>", "<|end|>"]
PAD, UNK, EOS, USER, ASSISTANT, END = range(6)
_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    return _WORD.findall(text)


def detokenize(words: List[str]) -> str:
    text = " ".join(words)
    text = re.sub(r"\s+([.,;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{$])\s+", r"\1", text)
    text = re.sub(r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b", r"\1", text)
    return text


class WordTokenizer:
    kind = "word"

    def __init__(self, itos: List[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def train(cls, word_iter: Iterable[str], vocab_size: int):
        c = collections.Counter(word_iter)
        itos = list(SPECIALS) + [w for w, _ in c.most_common(vocab_size - len(SPECIALS))]
        return cls(itos)

    def __len__(self):
        return len(self.itos)

    def encode_words(self, words: List[str]) -> Tuple[List[int], List[int]]:
        ids = [self.stoi.get(w, UNK) for w in words]
        return ids, list(range(len(words)))

    def encode(self, text: str) -> List[int]:
        return self.encode_words(simple_tokenize(text))[0]

    def decode(self, ids: List[int]) -> str:
        words = [self.itos[i] for i in ids if i < len(self.itos) and i not in (PAD, EOS, USER, ASSISTANT, END)]
        return detokenize(words)

    def save(self, d: str):
        os.makedirs(d, exist_ok=True)
        json.dump({"kind": "word", "itos": self.itos}, open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, d: str):
        return cls(json.load(open(os.path.join(d, "tokenizer.json"), encoding="utf-8"))["itos"])


class BPETokenizer:
    kind = "bpe"

    def __init__(self, tok, chunk_constrained: bool):
        self.tok = tok
        self.chunk_constrained = chunk_constrained

    @classmethod
    def train(cls, text_iter: Iterable[str], vocab_size: int, chunk_constrained: bool = False):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        # chunk_constrained: no regex pre-split, so merges may cross spaces -- but training texts are
        # single chunks, so a merge can never cross a chunk boundary. Standard: usual word-level pre-split.
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=not chunk_constrained)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator(text_iter, trainer)
        return cls(tok, chunk_constrained)

    def __len__(self):
        return self.tok.get_vocab_size()

    def encode_words(self, words: List[str]) -> Tuple[List[int], List[int]]:
        text = " ".join(words)
        starts, pos = [], 0
        for w in words:
            starts.append(pos); pos += len(w) + 1
        enc = self.tok.encode(text)
        ids, widx, j = [], [], 0
        for tid, (a, b) in zip(enc.ids, enc.offsets):
            if a < len(text) and text[a] == " ":       # byte-level tokens carry their leading space
                a += 1
            # advance to the word containing the first character of this token
            while j + 1 < len(words) and a >= starts[j + 1]:
                j += 1
            ids.append(tid); widx.append(j)
        return ids, widx

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode([i for i in ids if i not in (PAD, EOS, USER, ASSISTANT, END)]).strip()

    def save(self, d: str):
        os.makedirs(d, exist_ok=True)
        self.tok.save(os.path.join(d, "bpe.json"))
        json.dump({"kind": "bpe", "chunk_constrained": self.chunk_constrained}, open(os.path.join(d, "tokenizer.json"), "w"))

    @classmethod
    def load(cls, d: str):
        from tokenizers import Tokenizer
        meta = json.load(open(os.path.join(d, "tokenizer.json")))
        return cls(Tokenizer.from_file(os.path.join(d, "bpe.json")), meta.get("chunk_constrained", False))


def have_hf_tokenizers() -> bool:
    try:
        import tokenizers  # noqa
        return True
    except Exception:
        return False


def load_tokenizer(d: str):
    meta = json.load(open(os.path.join(d, "tokenizer.json"), encoding="utf-8"))
    return BPETokenizer.load(d) if meta["kind"] == "bpe" else WordTokenizer.load(d)


def chunk_strings(sent) -> List[str]:
    """Surface strings of the chunks of one tagged sentence (used by chunk-constrained BPE)."""
    from .data.chunker import chunks_from_tags
    out = []
    for typ, idx in chunks_from_tags(sent["chunk"]):
        out.append(" ".join(sent["forms"][i] for i in idx))
    return out


def build_tokenizer(cfg: dict, docs_iter_factory):
    """cfg = config["tokenizer"]; docs_iter_factory() returns a fresh iterator of documents."""
    kind = cfg.get("type", "auto")
    if kind == "auto":
        kind = "bpe" if have_hf_tokenizers() else "word"
    vocab_size = int(cfg.get("vocab_size", 16000))
    if kind == "word":
        return WordTokenizer.train((w for doc in docs_iter_factory() for s in doc for w in s["forms"]), vocab_size)
    constrained = bool(cfg.get("chunk_constrained", False))

    def texts():
        for doc in docs_iter_factory():
            for s in doc:
                if constrained and s.get("chunk"):
                    for c in chunk_strings(s):
                        yield c
                else:
                    yield " ".join(s["forms"])
    return BPETokenizer.train(texts(), vocab_size, constrained)
