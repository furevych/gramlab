"""Supervised fine-tuning on dialogue data so the base model answers in turns.

  python -m gramlab.finetune --init_from work/runs/tiny/ckpt.pt --corpus wiki_qa --out tiny_sft --epochs 3
  python -m gramlab.finetune --init_from work/runs/small_wiki/ckpt.pt --corpus rag_qa --corpus wiki_qa --out small_wiki_rag --epochs 2

Dialogue corpus entries (corpora.json, type dialogue_jsonl): each line
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
Loss is computed on assistant tokens only. The checkpoint is saved with sft=True so chat.py
switches to turn-based mode automatically; if the data was retrieval-conditioned (user turns
carry a "Context:" block, see scripts/make_rag_qa.py) it is also saved with rag=True so chat.py
knows the model expects a retrieved passage in front of each question.
Mixed precision follows train.py (bf16 on Ampere+, fp16 + loss scaling on T4).
"""
import argparse, json, math, os, random, time
import torch
import torch.nn.functional as F
from .config import run_dir
from .data.registry import load_registry, iter_docs
from .ckpt import load_checkpoint, save_checkpoint
from .tokenizer import USER, ASSISTANT, END, PAD
from .amp import pick_precision

TAIL_KEEP = 24   # when a user turn must be cut to fit ctx, keep its last tokens (the question) and cut before them


def encode_dialogue(tok, messages, ctx):
    """-> (ids, labels) of length <= ctx+1. Over-long dialogues are shortened by cutting the longest
    user turn (the retrieved context), never the answer: the answer tokens are the training signal."""
    segs = []
    for m in messages:
        role_id = USER if m["role"] == "user" else ASSISTANT
        segs.append([role_id, tok.encode(m["content"]) + [END]])
    total = sum(1 + len(b) for _, b in segs)
    if total > ctx + 1:
        over = total - (ctx + 1)
        user_segs = sorted((i for i, (r, _) in enumerate(segs) if r == USER), key=lambda i: -len(segs[i][1]))
        for i in user_segs:
            if over <= 0:
                break
            body = segs[i][1]
            cut = min(over, max(0, len(body) - TAIL_KEEP - 1))
            if cut > 0:
                keep_tail = TAIL_KEEP + 1                        # question + <|end|>
                segs[i][1] = body[:len(body) - keep_tail - cut] + body[len(body) - keep_tail:]
                over -= cut
    ids, labels = [], []
    for role_id, body in segs:
        ids += [role_id] + body
        labels += [-100] + (body if role_id == ASSISTANT else [-100] * len(body))
    return ids[:ctx + 1], labels[:ctx + 1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", required=True)
    ap.add_argument("--corpus", required=True, action="append", help="dialogue corpus name in corpora.json (repeatable: corpora are concatenated)")
    ap.add_argument("--registry", default="corpora.json")
    ap.add_argument("--out", required=True, help="run name for the fine-tuned checkpoint")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype, use_amp, use_fp16, scaler, precision = pick_precision(device)
    model, tok, state = load_checkpoint(args.init_from, device)
    cfg = state["config"]
    ctx = cfg["model"]["ctx"]
    registry = load_registry(args.registry)
    rows = [r for c in args.corpus for r in iter_docs(c, registry)]
    is_rag = any(m["role"] == "user" and m["content"].startswith("Context:") for r in rows[:200] for m in r["messages"])
    data = [encode_dialogue(tok, r["messages"], ctx) for r in rows]
    n_cut = sum(1 for d in data if len(d[0]) == ctx + 1)
    data = [d for d in data if any(l != -100 for l in d[1])]
    random.shuffle(data)
    n_val = max(1, int(len(data) * args.val_frac))
    val, train = data[:n_val], data[n_val:]
    print(f"[finetune] {len(train)} train / {len(val)} val dialogues from {args.corpus}, ctx {ctx} ({n_cut} at the limit), "
          f"device {device}, precision {precision}, rag={is_rag}")

    def batches(rows, shuffle):
        idx = list(range(len(rows)))
        if shuffle:
            random.shuffle(idx)
        for s in range(0, len(idx), args.batch):
            chunk = [rows[i] for i in idx[s:s + args.batch]]
            L = max(len(ids) for ids, _ in chunk)
            x = torch.full((len(chunk), L - 1), PAD, dtype=torch.long)
            y = torch.full((len(chunk), L - 1), -100, dtype=torch.long)
            for b, (ids, labels) in enumerate(chunk):
                x[b, :len(ids) - 1] = torch.tensor(ids[:-1])
                y[b, :len(ids) - 1] = torch.tensor(labels[1:])
            yield x.to(device), y.to(device)

    @torch.no_grad()
    def val_loss():
        model.eval(); tot, cnt = 0.0, 0
        for x, y in batches(val, False):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(x)
            tot += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum").item()
            cnt += (y != -100).sum().item()
        model.train()
        return tot / max(cnt, 1)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_total = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / max(1, steps_total))
    cfg_out = json.loads(json.dumps(cfg)); cfg_out["run_name"] = args.out
    out_path = os.path.join(run_dir(cfg_out), "ckpt.pt")
    print(f"[finetune] initial val loss {val_loss():.3f}")
    t0 = time.time(); step = 0
    model.train()
    for ep in range(args.epochs):
        for x, y in batches(train, True):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
        print(f"[finetune] epoch {ep+1}: train loss {loss.item():.3f} val loss {val_loss():.3f} ({time.time()-t0:.0f}s)")
        save_checkpoint(out_path, model, None, state["step"] + step, cfg_out, state["tokenizer_dir"],
                        extra={"sft": True, "rag": is_rag, "sft_corpora": args.corpus})
    print(f"[finetune] saved -> {out_path}")


if __name__ == "__main__":
    main()
