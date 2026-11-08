"""Pretraining with resume, continued training on new data, grammar aux heads, optional distillation.

  python -m gramlab.train --config configs/tiny.json
  python -m gramlab.train --config configs/tiny.json --resume                 # continue the same run
  python -m gramlab.train --config configs/tiny_plus.json --init_from work/runs/tiny/ckpt.pt   # new data / new corpora mix
  python -m gramlab.train --config configs/tiny_upos.json                     # aux_heads ["upos"]
  python -m gramlab.train --config configs/student.json --set train.distill.teacher=work/runs/teacher/ckpt.pt
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from .config import load_config, tokenizer_dir, shard_dir, run_dir
from .model import GPT
from .tokenizer import load_tokenizer
from .ckpt import save_checkpoint, load_checkpoint
from .prepare import NO_TAG


class Shards:
    """Memory-mapped token/tag streams for a list of corpora with sampling weights."""

    def __init__(self, cfg, corpora):
        self.items = []
        for c in corpora:
            d = shard_dir(cfg, c["name"])
            meta = json.load(open(os.path.join(d, "meta.json")))
            toks = np.memmap(os.path.join(d, "tokens.bin"), dtype=np.dtype(meta["dtype"]), mode="r")
            up = np.memmap(os.path.join(d, "upos.bin"), dtype=np.uint8, mode="r")
            ch = np.memmap(os.path.join(d, "chunk.bin"), dtype=np.uint8, mode="r")
            self.items.append({"name": c["name"], "w": c.get("weight", 1.0) * len(toks), "tokens": toks, "upos": up, "chunk": ch, "meta": meta})
        w = np.array([it["w"] for it in self.items], dtype=np.float64)
        self.p = w / w.sum()
        self.n_tokens = int(sum(len(it["tokens"]) for it in self.items))

    def batch(self, B, T, rng, device):
        xs, ys, us, cs = [], [], [], []
        for _ in range(B):
            it = self.items[rng.choice(len(self.items), p=self.p)]
            i = rng.integers(0, len(it["tokens"]) - T - 1)
            xs.append(np.asarray(it["tokens"][i:i + T], dtype=np.int64))
            ys.append(np.asarray(it["tokens"][i + 1:i + 1 + T], dtype=np.int64))
            us.append(np.asarray(it["upos"][i + 1:i + 1 + T], dtype=np.int64))
            cs.append(np.asarray(it["chunk"][i + 1:i + 1 + T], dtype=np.int64))
        to = lambda a: torch.from_numpy(np.stack(a)).to(device, non_blocking=True)
        u, c = to(us), to(cs)
        u[u == NO_TAG] = -100; c[c == NO_TAG] = -100          # untagged text contributes no aux loss
        return to(xs), to(ys), {"upos": u, "chunk": c}


@torch.no_grad()
def evaluate(model, shards, T, B, device, max_tokens):
    model.eval()
    it = shards.items[0]
    toks = it["tokens"]
    n = min(len(toks) - 1, max_tokens)
    n_win = n // T
    tot, cnt = 0.0, 0
    for s in range(0, n_win, B):
        idx = [np.asarray(toks[j * T:(j + 1) * T + 1], dtype=np.int64) for j in range(s, min(s + B, n_win))]
        a = torch.from_numpy(np.stack(idx)).to(device)
        logits, _ = model(a[:, :-1])
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), a[:, 1:].reshape(-1), reduction="sum").item()
        cnt += a[:, 1:].numel()
    model.train()
    nll = tot / cnt
    bpb = tot / math.log(2) / (it["meta"]["n_bytes"] * cnt / it["meta"]["n_tokens"])   # bits per byte of underlying text
    return {"eval_loss": nll, "eval_ppl": math.exp(nll), "eval_bits_per_byte": bpb}


def lr_at(step, tr):
    if step < tr["warmup"]:
        return tr["lr"] * (step + 1) / tr["warmup"]
    prog = min(1.0, (step - tr["warmup"]) / max(1, tr["steps"] - tr["warmup"]))
    return tr["lr"] * (tr["min_lr_ratio"] + (1 - tr["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * prog)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init_from", default=None, help="checkpoint to initialise weights from (continued pretraining)")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    tr, mc = cfg["train"], cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:                                                    # real bf16 hardware only (Ampere+); T4 emulates bf16 slowly
        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and not use_bf16            # T4/V100: fp16 tensor cores + loss scaling
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    torch.manual_seed(tr["seed"]); rng = np.random.default_rng(tr["seed"])
    tok_dir = tokenizer_dir(cfg)
    tok = load_tokenizer(tok_dir)
    train_sh = Shards(cfg, cfg["train_corpora"])
    eval_sh = Shards(cfg, [{"name": cfg["eval_corpus"]}])
    rd = run_dir(cfg); os.makedirs(rd, exist_ok=True)
    ckpt_path = os.path.join(rd, "ckpt.pt")

    model = GPT(len(tok), mc).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tr["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=tr["lr"], betas=(0.9, 0.95), fused=(device == "cuda"))
    step = 0
    if args.resume and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); step = st["step"]
        print(f"[train] resumed from step {step}")
    elif args.init_from:
        st = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(st["model"], strict=False)   # allows adding/removing aux heads
        print(f"[train] initialised from {args.init_from} (missing={len(missing)}, unexpected={len(unexpected)})")

    teacher = None
    if tr["distill"]["teacher"]:
        teacher, ttok, tstate = load_checkpoint(tr["distill"]["teacher"], device)
        assert tstate["tokenizer_dir"] == tok_dir, "teacher and student must share the tokenizer for logit distillation"
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[train] distilling from {tr['distill']['teacher']} ({teacher.n_params()/1e6:.1f}M non-emb params)")

    print(f"[train] {cfg['run_name']}: {model.n_params()/1e6:.2f}M non-embedding params (+{model.tok.weight.numel()/1e6:.2f}M embedding), "
          f"vocab {len(tok)}, train tokens {train_sh.n_tokens:,}, device {device}, precision "
          f"{'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    log = open(os.path.join(rd, "log.jsonl"), "a")
    json.dump(cfg, open(os.path.join(rd, "config.json"), "w"), indent=1)
    model.train()
    t0 = time.time()
    B, T = tr["batch"], mc["ctx"]
    max_hours = float(tr.get("max_hours", 0) or 0)
    while step < tr["steps"]:
        if max_hours and (time.time() - t0) > max_hours * 3600:
            print(f"[train] time guard: {max_hours}h reached at step {step}; stopping early", flush=True)
            tr["steps"] = step                                  # final eval + checkpoint below
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tr)
        opt.zero_grad(set_to_none=True)
        tot_lm = 0.0
        for _ in range(tr["accum"]):
            x, y, tags = train_sh.batch(B, T, rng, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(use_bf16 or use_fp16)):
                logits, h = model(x)
                loss_lm = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss_lm
                for name, head in model.aux.items():
                    loss = loss + mc["aux_weight"] * F.cross_entropy(head(h).reshape(-1, head.out_features), tags[name].reshape(-1), ignore_index=-100)
                if teacher is not None:
                    Tt = tr["distill"]["temperature"]
                    with torch.no_grad():
                        t_logits, _ = teacher(x)
                    kd = F.kl_div(F.log_softmax(logits.float() / Tt, dim=-1), F.softmax(t_logits.float() / Tt, dim=-1),
                                  reduction="batchmean") * (Tt * Tt) / T
                    w = tr["distill"]["weight"]
                    loss = (1 - w) * loss + w * kd
            scaler.scale(loss / tr["accum"]).backward()
            tot_lm += loss_lm.item() / tr["accum"]
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % 50 == 0:
            print(f"step {step} lm_loss {tot_lm:.3f} lr {lr_at(step, tr):.2e} {(time.time()-t0)/step:.2f}s/step", flush=True)
        if step % tr["eval_every"] == 0 or step == tr["steps"]:
            ev = evaluate(model, eval_sh, T, B, device, tr["eval_tokens"])
            rec = {"step": step, "train_lm_loss": round(tot_lm, 4), **{k: round(v, 4) for k, v in ev.items()},
                   "tokens_seen": step * B * T * tr["accum"], "sec": round(time.time() - t0)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            save_checkpoint(ckpt_path, model, opt, step, cfg, tok_dir)
    print(f"[train] done -> {ckpt_path}")


if __name__ == "__main__":
    main()
