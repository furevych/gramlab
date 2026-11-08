"""Evaluate a checkpoint: perplexity / bits-per-byte on a prepared corpus, optional BLiMP.

  python -m gramlab.evaluate --ckpt work/runs/tiny/ckpt.pt --corpus gum_test
  python -m gramlab.evaluate --ckpt work/runs/tiny/ckpt.pt --blimp /path/to/blimp/data   # *.jsonl with sentence_good/sentence_bad

Bits-per-byte is comparable across tokenizers; perplexity is not.
"""
import argparse, glob, json, math, os
import torch
import torch.nn.functional as F
from .ckpt import load_checkpoint
from .train import Shards, evaluate as eval_shards
from .tokenizer import EOS


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--blimp", default=None)
    ap.add_argument("--blimp_limit", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=2_000_000)
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, state = load_checkpoint(args.ckpt, device)
    cfg = state["config"]
    out = {"ckpt": args.ckpt, "step": state["step"], "non_embedding_params": model.n_params()}
    if args.corpus:
        sh = Shards(cfg, [{"name": args.corpus}])
        out.update(eval_shards(model, sh, cfg["model"]["ctx"], 32, device, args.max_tokens))
    if args.blimp:
        out["blimp"] = blimp(model, tok, args.blimp, device, args.blimp_limit)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
