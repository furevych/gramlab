"""The grammar meaning map: one small code per word, summed per grammar slot, regressed onto a teacher.

A sentence is split into grammar slots by a POS-based chunker (no parser at inference, only a POS tagger):
    SUBJ   noun phrases before the first verb group        VERB   verb groups (AUX* VERB+ PART)
    OBJ    noun phrases after it, not inside a PP          PP     preposition + the noun phrase it governs
    ADJP   adjectives outside noun phrases                 ADVP   adverbs                  OTHER  the rest
Each word has a 32-dim code; a sentence's representation is the sum of its word codes per slot (7 x 32 = 224
numbers), and a linear readout maps that onto the teacher's sentence embedding (384-dim MiniLM). Training =
cosine regression against the teacher on the parsed AMALGUM sentences. The whole map is ~1.3M numbers for a
40k-word vocabulary, and encoding a sentence is a handful of table lookups and additions.

The experiment this answers: does the grammar structure buy anything over a plain bag of words at the same
size? `slots=1` collapses every word into one slot — identical model, no grammar — and is trained alongside.
"""
import json, os, re, zlib
import numpy as np

SLOTS = ["SUBJ", "VERB", "OBJ", "PP", "ADJP", "ADVP", "OTHER"]
SLOT_ID = {s: i for i, s in enumerate(SLOTS)}
NP_TAGS = {"DET", "NUM", "ADJ", "NOUN", "PROPN", "PRON"}
VP_TAGS = {"AUX", "VERB"}
DROP_TAGS = {"PUNCT", "SYM", "X", "EOS"}


def chunk_slots(upos):
    """UPOS tags -> slot name per token (None = dropped). Pure POS patterns, CoNLL-2000 style."""
    n = len(upos)
    slots = [None] * n
    seen_verb = False
    i = 0
    while i < n:
        t = upos[i]
        if t in VP_TAGS:
            j = i
            while j < n and (upos[j] in VP_TAGS or (upos[j] == "PART" and j > i) or (upos[j] == "ADV" and j > i and j + 1 < n and upos[j + 1] in VP_TAGS)):
                j += 1
            for k in range(i, j):
                slots[k] = "VERB"
            seen_verb = True
            i = j
        elif t == "ADP":
            j = i + 1
            while j < n and upos[j] in NP_TAGS:
                j += 1
            for k in range(i, j):
                slots[k] = "PP"
            i = j
        elif t in NP_TAGS:
            j = i
            while j < n and upos[j] in NP_TAGS:
                j += 1
            for k in range(i, j):
                slots[k] = "OBJ" if seen_verb else "SUBJ"
            i = j
        elif t == "ADJ":
            slots[i] = "ADJP"; i += 1
        elif t == "ADV":
            slots[i] = "ADVP"; i += 1
        elif t in DROP_TAGS:
            i += 1
        else:
            slots[i] = "OTHER"; i += 1
    return slots


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def norm_word(form):
    m = _WORD.findall(form.lower())
    return m[0] if m else None


class Vocab:
    """Top words get their own code; everything else is hashed into `n_hash` shared buckets (rare names still count)."""
    def __init__(self, words, n_hash=8192):
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}
        self.n_hash = n_hash

    @classmethod
    def build(cls, sentences, max_words=40000, min_count=2, n_hash=8192):
        counts = {}
        for forms, _ in sentences:
            for f in forms:
                w = norm_word(f)
                if w:
                    counts[w] = counts.get(w, 0) + 1
        words = [w for w, c in sorted(counts.items(), key=lambda x: -x[1]) if c >= min_count][:max_words]
        return cls(words, n_hash)

    def __len__(self):
        return len(self.words) + self.n_hash

    def id(self, word):
        i = self.index.get(word)
        if i is not None:
            return i
        return len(self.words) + zlib.crc32(word.encode("utf-8")) % self.n_hash

    def save(self, path):
        json.dump({"words": self.words, "n_hash": self.n_hash}, open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding="utf-8"))
        return cls(d["words"], d["n_hash"])


def featurize(forms, upos, vocab, use_slots=True, max_len=64):
    """-> (word_ids, slot_ids) lists of equal length; words outside any slot are dropped."""
    slots = chunk_slots(upos)
    w_ids, s_ids = [], []
    for f, s in zip(forms, slots):
        if s is None:
            continue
        w = norm_word(f)
        if not w:
            continue
        w_ids.append(vocab.id(w)); s_ids.append(SLOT_ID[s] if use_slots else 0)
        if len(w_ids) >= max_len:
            break
    return w_ids, s_ids


def pad_batch(feats, pad_word):
    L = max(1, max(len(w) for w, _ in feats))
    W = np.full((len(feats), L), pad_word, dtype=np.int64)
    S = np.zeros((len(feats), L), dtype=np.int64)
    M = np.zeros((len(feats), L), dtype=np.float32)
    for b, (w, s) in enumerate(feats):
        W[b, :len(w)] = w; S[b, :len(s)] = s; M[b, :len(w)] = 1.0
    return W, S, M


# ------------------------------------------------------------------------------------------------ torch model
def build_model(vocab_size, n_slots, dim=32, out_dim=384, pool="sum"):
    import torch, torch.nn as nn, torch.nn.functional as F

    class MeaningMap(nn.Module):
        def __init__(self):
            super().__init__()
            self.code = nn.Embedding(vocab_size + 1, dim, padding_idx=vocab_size)   # last row = padding
            self.slot_gate = nn.Parameter(torch.ones(n_slots, dim))                  # per-slot reweighting of a code
            self.out = nn.Linear(n_slots * dim, out_dim)
            self.n_slots, self.dim, self.pool = n_slots, dim, pool
            nn.init.normal_(self.code.weight, std=0.1)

        def slot_sums(self, W, S, M):
            c = self.code(W) * self.slot_gate[S] * M.unsqueeze(-1)                  # B,L,dim
            onehot = F.one_hot(S, self.n_slots).float() * M.unsqueeze(-1)            # B,L,slots
            h = torch.einsum("bls,bld->bsd", onehot, c)                              # B,slots,dim
            if self.pool == "mean":                                                  # length-invariant: a 60-word passage
                h = h / onehot.sum(1).clamp(min=1).unsqueeze(-1)                     # and a 5-word question live on the same scale
            return h

        def forward(self, W, S, M):
            h = self.slot_sums(W, S, M).flatten(1)
            return F.normalize(self.out(h), dim=-1)

    return MeaningMap()


class MapEncoder:
    """Loads a trained map and encodes (forms, upos) sentences to unit vectors in the teacher's space."""
    def __init__(self, d):
        import torch
        meta = json.load(open(os.path.join(d, "meta.json")))
        self.vocab = Vocab.load(os.path.join(d, "vocab.json"))
        self.use_slots = meta["n_slots"] > 1
        self.model = build_model(len(self.vocab), meta["n_slots"], meta["dim"], meta["out_dim"], meta.get("pool", "sum"))
        self.model.load_state_dict(torch.load(os.path.join(d, "map.pt"), map_location="cpu"))
        self.model.eval()
        self.meta = meta

    def encode(self, sentences, batch=512):
        import torch
        out = []
        with torch.no_grad():
            for s in range(0, len(sentences), batch):
                feats = [featurize(f, u, self.vocab, self.use_slots) for f, u in sentences[s:s + batch]]
                W, S, M = pad_batch(feats, len(self.vocab))
                out.append(self.model(torch.from_numpy(W), torch.from_numpy(S), torch.from_numpy(M)).numpy())
        return np.concatenate(out) if out else np.zeros((0, self.meta["out_dim"]), dtype=np.float32)


def train_map(sentences, targets, out_dir, n_slots=len(SLOTS), dim=32, epochs=8, batch=256, lr=3e-3, val_frac=0.05, seed=1, pool="sum",
              vocab=None, log=print):
    """sentences: list of (forms, upos); targets: float array [n, out_dim] (teacher embeddings, any scale).
    pool="mean" averages codes per slot instead of summing (use it when inputs differ in length)."""
    import torch, torch.nn.functional as F
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    vocab = vocab or Vocab.build(sentences)
    vocab.save(os.path.join(out_dir, "vocab.json"))
    use_slots = n_slots > 1
    feats = [featurize(f, u, vocab, use_slots) for f, u in sentences]
    keep = [i for i, (w, _) in enumerate(feats) if w]
    feats = [feats[i] for i in keep]
    Y = torch.from_numpy(np.asarray(targets, dtype=np.float32)[keep])
    Y = F.normalize(Y, dim=-1)
    idx = rng.permutation(len(feats)); n_val = max(1, int(len(feats) * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    model = build_model(len(vocab), n_slots, dim, Y.shape[1], pool)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_total = epochs * ((len(tr_idx) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / max(1, steps_total))
    pad = len(vocab)

    def run(ids, train):
        tot, n = 0.0, 0
        model.train(train)
        for s in range(0, len(ids), batch):
            chunk = ids[s:s + batch]
            W, S, M = pad_batch([feats[i] for i in chunk], pad)
            pred = model(torch.from_numpy(W), torch.from_numpy(S), torch.from_numpy(M))
            loss = 1 - (pred * Y[chunk]).sum(-1).mean()
            if train:
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(chunk); n += len(chunk)
        return tot / max(n, 1)

    log(f"[map] slots={n_slots} dim={dim} vocab={len(vocab):,} ({len(vocab.words):,} words + {vocab.n_hash} hash buckets), "
        f"{len(tr_idx):,} train / {n_val:,} val sentences, params {sum(p.numel() for p in model.parameters()):,}")
    for ep in range(epochs):
        rng.shuffle(tr_idx)
        tr = run(tr_idx, True)
        with torch.no_grad():
            va = run(val_idx, False)
        log(f"[map] epoch {ep + 1}: train cos-loss {tr:.4f}  val cos-loss {va:.4f}  (val cosine to teacher {1 - va:.4f})")
    torch.save(model.state_dict(), os.path.join(out_dir, "map.pt"))
    json.dump({"n_slots": n_slots, "dim": dim, "out_dim": int(Y.shape[1]), "epochs": epochs, "val_cosine": round(1 - va, 4), "pool": pool,
               "slots": SLOTS[:n_slots] if use_slots else ["ALL"]}, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return MapEncoder(out_dir)


class HybridRetriever:
    """BM25 candidates re-ranked by the map: score = weight * bm25/max + cosine(map(query), map(passage)).
    guard=True keeps BM25's answer whenever its top hit is an exact title match (the map only helps where words differ)."""
    def __init__(self, index, encoder, passages_vec, tag_fn, weight=1.0, guard=False, top_n=50):
        self.index, self.enc, self.P, self.tag, self.weight, self.guard, self.top_n = index, encoder, passages_vec, tag_fn, weight, guard, top_n

    @classmethod
    def load(cls, index, map_dir, tag_fn):
        enc = MapEncoder(map_dir)
        P = np.load(os.path.join(map_dir, "passages.npy")).astype(np.float32)
        cfg = {}
        hp = os.path.join(map_dir, "hybrid.json")
        if os.path.exists(hp):
            cfg = json.load(open(hp))
        return cls(index, enc, P, tag_fn, weight=cfg.get("weight", 1.0), guard=cfg.get("guard", False))

    def search(self, query, k=5, qvec=None):
        hits = self.index.search(query, self.top_n)
        if not hits:
            return []
        if self.guard:
            from .retrieval import analyze
            terms = [self.index.correct(t) for t in analyze(query)]
            covered = self.index.title_coverage([t for t in terms if t in self.index.vocab])
            if covered.get(hits[0][1], 0) >= 1.0:
                return hits[:k]
        if qvec is None:
            qvec = self.enc.encode([self.tag(query)])[0]
        mx = max(h[0] for h in hits)
        sc = sorted(((self.weight * h[0] / mx + float(qvec @ self.P[h[1]]), h[1]) for h in hits), reverse=True)
        return sc[:k]

    def retrieve(self, query, k=1, max_words=None):
        from .retrieval import cut_words
        out = []
        for score, i in self.search(query, k):
            p = self.index.passage(i)
            if max_words:
                p["text"] = cut_words(p["text"], max_words)
            p["score"] = round(float(score), 3); p["index"] = int(i)
            out.append(p)
        return out
