"""Evaluate a checkpoint: perplexity / bits-per-byte on a prepared corpus, optional BLiMP, optional per-POS loss.

  python -m gramlab.evaluate --ckpt work/runs/tiny/ckpt.pt --corpus gum_test
  python -m gramlab.evaluate --ckpt work/runs/tiny/ckpt.pt --blimp /path/to/blimp/data   # *.jsonl with sentence_good/sentence_bad
  python -m gramlab.evaluate --ckpt work/runs/tiny/ckpt.pt --corpus gum_test --per_pos --out results/eval_tiny.json

Bits-per-byte is comparable across tokenizers; perplexity is not.
--per_pos breaks the next-token loss down by the UPOS tag and the chunk tag of the *target* token (the
corpus must be tagged, e.g. gum_test), so you can see where a model is weak (content words: NOUN/PROPN/NUM
are the facts it cannot know; function words: DET/ADP/AUX are grammar) and whether grammar heads help at
chunk boundaries (B-*) more than inside chunks (I-*). Plot with scripts/plot_pos_loss.py.
"""
import argparse, glob, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from .ckpt import load_checkpoint
from .train import Shards, evaluate as eval_shards
from .tokenizer import EOS
from .prepare import UPOS_TAGS, CHUNK_TAGS, NO_TAG


@torch.no_grad()
def sentence_logprob(model, tok, text, device):
    ids = [EOS] + tok.encode(text) + [EOS]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x[:, :-1][:, -model.ctx:])
    lp = F.log_softmax(logits.float(), dim=-1)
    tgt = x[:, 1:][:, -logits.size(1):]
    return lp.gather(2, tgt.unsqueeze(-1)).sum().item()


def blimp(model, tok, path, device, limit=None):
    results = {}
    for f in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        good = n = 0
        with open(f, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                if sentence_logprob(model, tok, r["sentence_good"], device) > sentence_logprob(model, tok, r["sentence_bad"], device):
                    good += 1
                n += 1
        results[os.path.basename(f)[:-6]] = good / max(n, 1)
    results["_mean"] = sum(v for k, v in results.items() if not k.startswith("_")) / max(1, len(results))
    return results


@torch.no_grad()
def per_tag_loss(model, shards, T, B, device, max_tokens):
    """Mean next-token NLL (nats) grouped by the target token's UPOS tag and chunk tag."""
    model.eval()
    it = shards.items[0]
    toks, upos, chunk = it["tokens"], it["upos"], it["chunk"]
    n = min(len(toks) - 1, max_tokens)
    n_win = n // T
    nU, nC = len(UPOS_TAGS), len(CHUNK_TAGS)
    sum_u, cnt_u = np.zeros(nU), np.zeros(nU)
    sum_c, cnt_c = np.zeros(nC), np.zeros(nC)
    for s in range(0, n_win, B):
        js = range(s, min(s + B, n_win))
        a = torch.from_numpy(np.stack([np.asarray(toks[j * T:(j + 1) * T + 1], dtype=np.int64) for j in js])).to(device)
        u = np.concatenate([np.asarray(upos[j * T + 1:(j + 1) * T + 1], dtype=np.int64) for j in js])
        c = np.concatenate([np.asarray(chunk[j * T + 1:(j + 1) * T + 1], dtype=np.int64) for j in js])
        logits, _ = model(a[:, :-1])
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), a[:, 1:].reshape(-1), reduction="none").cpu().numpy()
        m = u != NO_TAG
        sum_u += np.bincount(u[m], weights=nll[m], minlength=nU)[:nU]; cnt_u += np.bincount(u[m], minlength=nU)[:nU]
        m = c != NO_TAG
        sum_c += np.bincount(c[m], weights=nll[m], minlength=nC)[:nC]; cnt_c += np.bincount(c[m], minlength=nC)[:nC]
    model.train()
    tot_u = max(cnt_u.sum(), 1)
    per_upos = {UPOS_TAGS[i]: {"loss": round(float(sum_u[i] / cnt_u[i]), 4), "count": int(cnt_u[i]), "share": round(float(cnt_u[i] / tot_u), 4)}
                for i in range(nU) if cnt_u[i] > 0}
    per_chunk = {CHUNK_TAGS[i]: {"loss": round(float(sum_c[i] / cnt_c[i]), 4), "count": int(cnt_c[i])} for i in range(nC) if cnt_c[i] > 0}
    pos = {}
    for key, pred in (("B", lambda t: t.startswith("B-")), ("I", lambda t: t.startswith("I-")), ("O", lambda t: t == "O")):
        idx = [i for i, t in enumerate(CHUNK_TAGS) if pred(t)]
        c = cnt_c[idx].sum()
        if c > 0:
            pos[key] = {"loss": round(float(sum_c[idx].sum() / c), 4), "count": int(c)}
    tagged = int(cnt_u.sum())
    return {"per_upos": per_upos, "per_chunk": per_chunk, "per_chunk_position": pos, "tagged_tokens": tagged,
            "tagged_loss": round(float(sum_u.sum() / tot_u), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--blimp", default=None)
    ap.add_argument("--blimp_limit", type=int, default=None)
    ap.add_argument("--per_pos", action="store_true", help="loss by UPOS / chunk tag of the target token (needs a tagged --corpus)")
    ap.add_argument("--max_tokens", type=int, default=2_000_000)
    ap.add_argument("--out", default=None, help="also write the JSON here")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, state = load_checkpoint(args.ckpt, device)
    cfg = state["config"]
    out = {"ckpt": args.ckpt, "run_name": cfg.get("run_name"), "step": state["step"], "non_embedding_params": model.n_params(),
           "aux_heads": cfg["model"].get("aux_heads", [])}
    if args.corpus:
        sh = Shards(cfg, [{"name": args.corpus}])
        out.update(eval_shards(model, sh, cfg["model"]["ctx"], 32, device, args.max_tokens))
        if args.per_pos:
            out.update(per_tag_loss(model, sh, cfg["model"]["ctx"], 32, device, args.max_tokens))
    if args.blimp:
        out["blimp"] = blimp(model, tok, args.blimp, device, args.blimp_limit)
    print(json.dumps(out, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
