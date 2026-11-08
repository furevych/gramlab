"""JSON configs with defaults and dotted CLI overrides (--set train.steps=8000)."""
import copy, json, os

DEFAULTS = {
    "run_name": "run",
    "work_dir": "work",
    "registry": "corpora.json",
    "tokenizer": {"type": "auto", "vocab_size": 16000, "chunk_constrained": False, "name": None},
    "train_corpora": [{"name": "amalgum", "weight": 1.0}],
    "eval_corpus": "gum_test",
    "model": {"d_model": 256, "n_layer": 4, "n_head": 4, "ctx": 128, "dropout": 0.0,
              "aux_heads": [], "aux_weight": 0.3},
    "train": {"batch": 64, "accum": 1, "steps": 4000, "lr": 6e-4, "min_lr_ratio": 0.1, "warmup": 200,
              "weight_decay": 0.1, "grad_clip": 1.0, "eval_every": 250, "eval_tokens": 200000, "seed": 1,
              "distill": {"teacher": None, "weight": 0.5, "temperature": 2.0}},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _coerce(val: str):
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    if val.lower() in ("true", "false", "null", "none"):
        return {"true": True, "false": False}.get(val.lower())
    if val.startswith("[") or val.startswith("{"):
        return json.loads(val)
    return val


def load_config(path: str, overrides=()):
    cfg = _merge(DEFAULTS, json.load(open(path)))
    for item in overrides or []:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    tk = cfg["tokenizer"]
    if not tk.get("name"):
        kind = tk["type"]
        if kind == "auto":
            from .tokenizer import have_hf_tokenizers
            kind = "bpe" if have_hf_tokenizers() else "word"
        tk["name"] = f"{kind}{tk['vocab_size']}" + ("_chunk" if tk.get("chunk_constrained") else "")
    return cfg


def tokenizer_dir(cfg):
    return os.path.join(cfg["work_dir"], cfg["tokenizer"]["name"], "tokenizer")


def shard_dir(cfg, corpus):
    return os.path.join(cfg["work_dir"], cfg["tokenizer"]["name"], "shards", corpus)


def run_dir(cfg):
    return os.path.join(cfg["work_dir"], "runs", cfg["run_name"])
