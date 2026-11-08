"""Talk to a checkpoint.

  python -m gramlab.chat --ckpt work/runs/tiny/ckpt.pt                 # base model: you type a prefix, it continues
  python -m gramlab.chat --ckpt work/runs/tiny_sft/ckpt.pt             # fine-tuned: user/assistant turns
  python -m gramlab.chat --ckpt ... --prompt "The city of Aachen" --n 3  # non-interactive

Sampling knobs: --temperature --top_k --top_p --rep (repetition penalty) --max_new
"""
import argparse
import torch
from .ckpt import load_checkpoint
from .tokenizer import EOS, USER, ASSISTANT, END


def build_chat_prompt(tok, history):
    ids = []
    for role, text in history:
        ids += [USER if role == "user" else ASSISTANT] + tok.encode(text) + [END]
    return ids + [ASSISTANT]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--mode", choices=["auto", "base", "chat"], default="auto")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--rep", type=float, default=1.2)
    ap.add_argument("--max_new", type=int, default=80)
    ap.add_argument("--history", type=int, default=4, help="chat turns kept as context")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, state = load_checkpoint(args.ckpt, device)
    mode = args.mode if args.mode != "auto" else ("chat" if state.get("sft") else "base")
    print(f"[chat] {args.ckpt}: {model.n_params()/1e6:.2f}M non-emb params, step {state['step']}, mode={mode}. Ctrl-D / 'quit' to exit.")

    def respond(history_or_text):
        if mode == "chat":
            ids = build_chat_prompt(tok, history_or_text)
            stop = (END, EOS)
        else:
            ids = tok.encode(history_or_text)
            stop = ()
        x = torch.tensor([ids[-(model.ctx - 1):]], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=args.max_new, temperature=args.temperature, top_k=args.top_k,
                             top_p=args.top_p, repetition_penalty=args.rep, stop_ids=stop)
        new = out[0, x.size(1):].tolist()
        return tok.decode(new)

    if args.prompt is not None:
        for _ in range(args.n):
            print(respond([("user", args.prompt)] if mode == "chat" else args.prompt))
        return
    history = []
    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            break
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        if mode == "chat":
            history.append(("user", text))
            reply = respond(history[-2 * args.history:])
            history.append(("assistant", reply))
        else:
            reply = text + " " + respond(text)
        print("model>", reply)


if __name__ == "__main__":
    main()
