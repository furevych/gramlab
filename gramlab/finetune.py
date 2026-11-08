"""Supervised fine-tuning on dialogue data so the base model answers in turns.

  python -m gramlab.finetune --init_from work/runs/tiny/ckpt.pt --corpus wiki_qa --out tiny_sft --epochs 3

Dialogue corpus entries (corpora.json, type dialogue_jsonl): each line
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
Loss is computed on assistant tokens only. The checkpoint is saved with sft=True so chat.py
switches to turn-based mode automatically.
"""
import argparse, json, math, os, random, time
import torch
import torch.nn.functional as F
from .config import load_config, run_dir
from .data.registry import load_registry, iter_docs
from .ckpt import load_checkpoint, save_checkpoint
from .tokenizer import USER, ASSISTANT, END, PAD


def encode_dialogue(tok, messages, ctx):
    ids, labels = [], []
    for m in messages:
        role_id = USER if m["role"] == "user" else ASSISTANT
        body = tok.encode(m["content"]) + [END]
        ids += [role_id] + body
        labels += [-100] + (body if role_id == ASSISTANT else [-100] * len(body))
    return ids[:ctx + 1], labels[:ctx + 1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", required=True)
    ap.add_argument("--corpus", required=True, help="dialogue corpus name in corpora.json")
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
    model, tok, state = load_checkpoint(args.init_from, device)
    cfg = state["config"]
    ctx = cfg["model"]["ctx"]
    registry = load_registry(args.registry)
    data = [encode_dialogue(tok, r["messages"], ctx) for r in iter_docs(args.corpus, registry)]
    data = [d for d in data if any(l != -100 for l in d[1])]
    random.shuffle(data)
    n_val = max(1, int(len(data) * args.val_frac))
    val, train = data[:n_val], data[n_val:]
    print(f"[finetune] {len(train)} train / {len(val)} val dialogues, ctx {ctx}, device {device}")

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
            logits, _ = model(x)
            tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum").item()
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
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
        print(f"[finetune] epoch {ep+1}: train loss {loss.item():.3f} val loss {val_loss():.3f} ({time.time()-t0:.0f}s)")
        save_checkpoint(out_path, model, None, state["step"] + step, cfg_out, state["tokenizer_dir"], extra={"sft": True})
    print(f"[finetune] saved -> {out_path}")


if __name__ == "__main__":
    main()
