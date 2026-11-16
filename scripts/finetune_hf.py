"""Swap the writer: LoRA fine-tune a pretrained open model on the retrieval-conditioned dialogues (T4, ~45 min).

  python scripts/finetune_hf.py --base Qwen/Qwen2.5-0.5B-Instruct --corpus data/rag_qa.jsonl --out work/runs/writer_hf

The 25M writer reads passages correctly but misspells rare names; a pretrained 0.5B model already knows English and
only has to learn the job: answer from the passage, admit when the passage does not cover the question. Same data as
the 25M fine-tune (scripts/make_rag_qa.py), same Context/Question format, so chat.py / the chat notebook keep working.
Saves the merged model (fp16, ~1 GB) + tokenizer; chat loads it with transformers. Evaluates on a held-out slice:
faithfulness (share of answer words found in the passage), token F1 against the reference, and the miss-detection rate.
"""
import argparse, json, os, random, re, time
import torch


SYSTEM = ("Answer using only the passage given. If the passage does not cover the question, say "
          "\"I don't have information about\" the thing asked.")


def encode(tok, user, answer, max_len):
    chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(chat + [{"role": "assistant", "content": answer}], tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    f_ids = tok(full, add_special_tokens=False).input_ids
    if f_ids[:len(p_ids)] != p_ids or len(f_ids) > max_len:
        return None
    return f_ids, [-100] * len(p_ids) + f_ids[len(p_ids):]


def token_f1(a, b):
    ta, tb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    if not ta or not tb:
        return 0.0
    common = sum(min(ta.count(w), tb.count(w)) for w in set(ta))
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def faithfulness(answer, user):
    ctx = set(re.findall(r"\w+", user.lower()))
    words = re.findall(r"\w+", answer.lower())
    return sum(w in ctx for w in words) / max(1, len(words))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--corpus", default="data/rag_qa.jsonl")
    ap.add_argument("--out", default="work/runs/writer_hf")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--max_hours", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    random.shuffle(rows)
    val_rows, train_rows = rows[:args.val], rows[args.val:]
    data = []
    for r in train_rows:
        u, a = r["messages"][0]["content"], r["messages"][1]["content"]
        e = encode(tok, u, a, args.max_len)
        if e:
            data.append(e)
    print(f"[hf] {len(data):,} train dialogues (of {len(train_rows):,}; longer than {args.max_len} tokens dropped), {len(val_rows)} val, device {device}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_total = args.epochs * (len(data) // (args.batch * args.accum))
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps_total), steps_total)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    model.train(); step = 0; stop = False
    for ep in range(args.epochs):
        random.shuffle(data)
        for s in range(0, len(data) - args.batch + 1, args.batch):
            chunk = data[s:s + args.batch]
            L = max(len(i) for i, _ in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id); lab = torch.full((len(chunk), L), -100); att = torch.zeros((len(chunk), L), dtype=torch.long)
            for b, (i, l) in enumerate(chunk):
                ids[b, :len(i)] = torch.tensor(i); lab[b, :len(l)] = torch.tensor(l); att[b, :len(i)] = 1
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss / args.accum
            scaler.scale(loss).backward()
            if ((s // args.batch) + 1) % args.accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if step % 50 == 0:
                    print(f"[hf] epoch {ep + 1} step {step}/{steps_total} loss {loss.item() * args.accum:.3f} {(time.time() - t0) / 60:.1f} min", flush=True)
                if (time.time() - t0) > args.max_hours * 3600:
                    print("[hf] time guard reached", flush=True); stop = True; break
        if stop:
            break

    # ---- held-out evaluation + save
    model.eval()
    def answer(user, max_new=80):
        chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        ids = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    stats = {"gold": [], "miss": [], "f1": []}
    for r in val_rows:
        u, a = r["messages"][0]["content"], r["messages"][1]["content"]
        pred = answer(u)
        if r.get("variant") == "miss":
            stats["miss"].append("don't have information" in pred.lower())
        else:
            stats["gold"].append(faithfulness(pred, u)); stats["f1"].append(token_f1(pred, a))
    ev = {"base": args.base, "steps": step, "faithfulness": round(sum(stats["gold"]) / max(1, len(stats["gold"])), 4),
          "token_f1_vs_reference": round(sum(stats["f1"]) / max(1, len(stats["f1"])), 4),
          "miss_detected": round(sum(stats["miss"]) / max(1, len(stats["miss"])), 4), "n_val": len(val_rows), "minutes": round((time.time() - t0) / 60, 1)}
    print("[hf] held-out:", json.dumps(ev), flush=True)
    os.makedirs(args.out, exist_ok=True)
    merged = model.merge_and_unload()
    merged.half().save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    json.dump({"system": SYSTEM, "eval": ev, "sft": True, "rag": True, "writer": "hf"}, open(os.path.join(args.out, "gramlab.json"), "w"), indent=1)
    print(f"[hf] saved merged model -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
