"""Talk to a checkpoint.

  python -m gramlab.chat --ckpt work/runs/tiny/ckpt.pt                 # base model: you type a prefix, it continues
  python -m gramlab.chat --ckpt work/runs/tiny_sft/ckpt.pt             # fine-tuned: user/assistant turns
  python -m gramlab.chat --ckpt work/runs/small_wiki_rag/ckpt.pt --retrieve work/bm25   # retrieval-conditioned chat
  python -m gramlab.chat --ckpt ... --prompt "The city of Aachen" --n 3  # non-interactive

With --retrieve, every question is first sent to the BM25 index (gramlab.retrieval) and the top
passage(s) are put in front of it in the format the model was fine-tuned on (scripts/make_rag_qa.py).
Sampling knobs: --temperature --top_k --top_p --rep (repetition penalty) --no_repeat (n-gram ban) --max_new
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
    ap.add_argument("--retrieve", default=None, help="BM25 index dir (python -m gramlab.retrieval build ...)")
    ap.add_argument("--k", type=int, default=1, help="passages per question with --retrieve")
    ap.add_argument("--ctx_words", type=int, default=90, help="words per retrieved passage (keep ctx room for the answer)")
    ap.add_argument("--show_context", action="store_true", help="print the retrieved passage(s) too")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--rep", type=float, default=1.2)
    ap.add_argument("--no_repeat", type=int, default=3, help="ban repeating any n-gram of this size already in the text (0 = off)")
    ap.add_argument("--max_new", type=int, default=80)
    ap.add_argument("--history", type=int, default=4, help="chat turns kept as context (0 with retrieval: the passage needs the room)")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, state = load_checkpoint(args.ckpt, device)
    mode = args.mode if args.mode != "auto" else ("chat" if state.get("sft") else "base")
    index = None
    if args.retrieve:
        from .retrieval import BM25Index, format_rag_user
        index = BM25Index.load(args.retrieve)
        mode = "chat"
        if not state.get("rag"):
            print("[chat] note: this checkpoint was not fine-tuned on retrieval-conditioned data (rag=False); answers may ignore the passage")
    elif state.get("rag"):
        print("[chat] note: this checkpoint expects a retrieved passage before each question; pass --retrieve work/bm25")
    n_hist = 0 if index is not None else args.history
    print(f"[chat] {args.ckpt}: {model.n_params()/1e6:.2f}M non-emb params, step {state['step']}, mode={mode}"
          f"{', retrieval over %s passages' % format(len(index), ',') if index is not None else ''}. Ctrl-D / 'quit' to exit.")

    def with_context(text):
        """user text -> (what the model sees, retrieved passages)"""
        if index is None:
            return text, []
        passages = index.retrieve(text, k=args.k, max_words=args.ctx_words)
        if not passages:
            return format_rag_user([{"title": "(no match)", "text": "No passage found."}], text), []
        return format_rag_user(passages, text), passages

    def respond(history_or_text):
        if mode == "chat":
            ids = build_chat_prompt(tok, history_or_text)
            stop = (END, EOS)
        else:
            ids = tok.encode(history_or_text)
            stop = ()
        x = torch.tensor([ids[-(model.ctx - 1):]], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=args.max_new, temperature=args.temperature, top_k=args.top_k,
                             top_p=args.top_p, repetition_penalty=args.rep, no_repeat_ngram=args.no_repeat, stop_ids=stop)
        new = out[0, x.size(1):].tolist()
        return tok.decode(new)

    def show_passages(passages):
        for p in passages:
            print(f"  [retrieved {p['score']:.1f}] {p['title']}: {p['text'][:200]}{'...' if len(p['text']) > 200 else ''}")

    if args.prompt is not None:
        user_text, passages = with_context(args.prompt)
        if args.show_context:
            show_passages(passages)
        for _ in range(args.n):
            print(respond([("user", user_text)] if mode == "chat" else args.prompt))
        return
    history = []       # plain turns, without retrieved contexts
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
            user_text, passages = with_context(text)
            if args.show_context:
                show_passages(passages)
            turns = history[-2 * n_hist:] if n_hist else []
            reply = respond(turns + [("user", user_text)])
            history += [("user", text), ("assistant", reply)]
        else:
            reply = text + " " + respond(text)
        print("model>", reply)


if __name__ == "__main__":
    main()
