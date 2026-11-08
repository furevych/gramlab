# gramlab — a compact LM you can train, talk to, retrain and extend

One small codebase, one loop:

```
prepare  →  train  →  chat  →  finetune (dialogue)  →  chat
   ↑                                                     │
   └──── add data / add an approach / scale the config ◄─┘
```

Everything is config-driven (`configs/*.json`), every data source is one entry in `corpora.json`,
and every checkpoint carries its config + tokenizer path, so any checkpoint can be resumed,
continued on new data, fine-tuned, evaluated or chatted with.

## 0. Status of this code
Built and tested in a sandbox **without PyTorch**:
- tested on real data here: `gramlab/data/*` (readers, chunker, registry), `tokenizer.py` (word tokenizer),
  `prepare.py` (tokenizer training + shards + incremental corpora), `scripts/make_wiki_qa.py`,
  `retrieval.py` (BM25 build/search), `scripts/make_rag_qa.py`, `scripts/plot_pos_loss.py`, the context
  truncation in `finetune.py`
- syntax-checked only (needs a GPU box): `model.py`, `train.py`, `finetune.py`, `evaluate.py`, `chat.py`,
  and the BPE path of `tokenizer.py` (uses the `tokenizers` library, present on Kaggle)
Expect small bugs on first GPU run; they will be shallow (shapes, dtypes), not structural.

## 1. Kaggle quickstart (GPU notebook)
```bash
git clone https://github.com/furevych/gramlab.git && cd gramlab
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/gucorpling/amalgum external/amalgum
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/UniversalDependencies/UD_English-GUM external/UD_English-GUM

python -m gramlab.prepare  --config configs/tiny.json            # BPE 16k tokenizer + shards (≈1 min)
python -m gramlab.train    --config configs/tiny.json            # ≈3M non-embedding params, 4000 steps ≈ 15 min on a T4
python -m gramlab.chat     --ckpt work/runs/tiny/ckpt.pt         # base model: type a prefix, it continues
python scripts/make_wiki_qa.py --amalgum external/amalgum/amalgum --out data/wiki_qa.jsonl
python -m gramlab.finetune --init_from work/runs/tiny/ckpt.pt --corpus wiki_qa --out tiny_sft --epochs 3
python -m gramlab.chat     --ckpt work/runs/tiny_sft/ckpt.pt     # turn-based: you> ... model> ...
python -m gramlab.evaluate --ckpt work/runs/tiny/ckpt.pt --corpus gum_test [--blimp path/to/blimp/data]

# retrieval: the model reads a passage instead of guessing facts
python -m gramlab.retrieval build --wiki data/simplewiki.jsonl --wiki data/enwiki.jsonl --out work/bm25
python scripts/make_rag_qa.py --wiki data/simplewiki.jsonl --index work/bm25 --out data/rag_qa.jsonl --n 30000
python -m gramlab.finetune --init_from work/runs/small_wiki/ckpt.pt --corpus rag_qa --out small_wiki_rag --epochs 2
python -m gramlab.chat --ckpt work/runs/small_wiki_rag/ckpt.pt --retrieve work/bm25 --show_context

# where is the model weak? loss by part of speech / chunk position of the predicted token
python -m gramlab.evaluate --ckpt work/runs/tiny/ckpt.pt --corpus gum_test --per_pos --out results/eval_tiny.json
python -m gramlab.evaluate --ckpt work/runs/tiny_upos/ckpt.pt --corpus gum_test --per_pos --out results/eval_tiny_upos.json
python scripts/plot_pos_loss.py --results results/eval_tiny.json results/eval_tiny_upos.json --out results/per_pos_loss.png
```

## 2. What to expect from the conversation
A 3M-parameter model trained on 3.5M tokens produces locally grammatical, topically drifting text;
after `finetune` on `wiki_qa` it answers in the right *format* ("X is a city in ...") with invented facts.
That is the correct result at this scale — use it to see the mechanics work, not to get answers.
Coherence improves with (in order of payoff): more and cleaner data (tens of millions of tokens of
parsed Wikipedia via Stanza), `configs/small.json` (≈25M params, ctx 256), then real dialogue data.
Judge progress with `eval_bits_per_byte` (comparable across tokenizers) and BLiMP, not by chatting.

**Facts come from retrieval, not from parameters.** A 25M model will never memorise Wikipedia, so `gramlab.retrieval`
keeps a BM25 index of ~330k Wikipedia lead paragraphs (pure numpy/scipy, ~1 ms a query) and the `rag_qa` fine-tune
teaches the model to answer *from* the passage put in front of the question:
```
Context: Belgrade: Belgrade is the capital and largest city of Serbia. It is located at the confluence of ...
Question: Tell me about Belgrade.
```
20% of the training examples show a wrong passage and the answer "I don't have information about X.", so a retrieval
miss produces an admission instead of an invention. `chat.py --retrieve work/bm25` reproduces the format at chat time.

## 3. Scaling knobs
- `configs/tiny.json` → `configs/small.json`: d_model / n_layer / ctx / batch×accum. Shards are memory-mapped,
  so data size is bounded by disk, not RAM; bf16 autocast is on automatically on CUDA.
- More data: add a corpus (below) and list it in `train_corpora` with a `weight`; sampling is proportional
  to weight × size. To parse new raw text into tagged data, run Stanza/spaCy on Kaggle and point a
  `ud_conllu_dir` entry at the output — the chunker and the grammar heads then work on it too.
- Continue a model on new data without starting over:
  `python -m gramlab.train --config configs/small.json --init_from work/runs/tiny/ckpt.pt`
  (strict=False loading, so you can add or drop aux heads when continuing). `--resume` continues the
  *same* run with its optimizer state (and fp16 loss scale); add `--more_steps 8000 --set train.lr=3e-4` to
  extend a finished or time-guarded run: the cosine schedule is re-spanned to the new total and re-warmed over
  `train.warmup` steps so the learning rate does not jump. With `train.max_hours` set, the trainer measures s/step
  after 200 steps and shortens the schedule to what fits, so the lr anneals before the guard instead of being cut off.
- Decoding: `generate()` applies a repetition penalty and bans any already-seen n-gram (`no_repeat_ngram=3`,
  `chat.py --no_repeat`), which ends the "Tokyo, Japan, Japan, Japan" loops.
- Precision is picked automatically (`gramlab/amp.py`): bf16 on Ampere+ GPUs, fp16 + dynamic loss scaling on
  T4/V100/P100 (bf16 is only emulated there), fp32 on CPU. Same in `finetune.py`.

## 4. Adding data (no code)
`corpora.json` entry types: `ud_conllu_dir`, `conllu_file`, `jsonl_sentences`, `plain_text`
(`.txt` with blank lines between documents, or `.jsonl` with a `text` field), `dialogue_jsonl`
(`{"messages":[{"role":"user","content":..},{"role":"assistant","content":..}]}`).
Then: `python -m gramlab.prepare --config configs/tiny.json --corpus <name>`. Existing shards are untouched.

## 5. Adding an approach (where the hooks are)
- **Structural signal (train-time only, free at inference)**: add a tag stream in `prepare.py`
  (like `upos.bin`), register its size in `model.AUX_HEADS`, add the name to `model.aux_heads` in a config.
  `configs/tiny_upos.json` is the ready-made grammar arm (`["upos", "chunk"]`).
- **Distillation**: `--set train.distill.teacher=work/runs/<teacher>/ckpt.pt` (same tokenizer required);
  `weight` and `temperature` under `train.distill`.
- **Tokenizer variants**: `tokenizer.type` = `bpe` | `word`; `tokenizer.chunk_constrained: true` trains
  SuperBPE-style multi-word tokens whose merges never cross a grammar-chunk boundary (experimental).
  Each tokenizer gets its own `work/<name>/` tree so variants coexist.
- **Output layer / attention / positional scheme**: `model.py` is 120 lines; swap `self.head` or `Block`.
- **Evaluation**: `evaluate.py` — add a metric function next to `blimp()` / `per_tag_loss()`. `--per_pos` splits the
  next-token loss by the UPOS tag and the chunk tag (B-*/I-*/O) of the *target* token, so you can see whether an
  approach helps function words (grammar) or content words (facts), and chunk starts more than chunk insides.
- **Retrieval**: `retrieval.py` — `analyze()` (stopwords, crude stemming) and `BM25Index`; swap in a dense index
  by keeping `retrieve(question, k, max_words) -> [{"title","text","score"}]`. The chat format lives in
  `format_rag_user()` and is shared by `scripts/make_rag_qa.py`, `finetune.py` and `chat.py`.

## 6. Unattended runs (Kaggle "Save & Run All")
Run 1 (`kaggle_overnight.ipynb`, now only in git history; its output is Kaggle Version 1) trained the 3M baseline vs
grammar heads, fetched the Wikipedia corpus and trained the 25M `small_wiki` for 0.7 epochs. Results and the reading
are in **RESULTS.md**.

`kaggle_continue.ipynb` is stage 2. Attach Version 1 as input; it copies only the tokenizer/shards and checkpoints,
finishes the 25M writer in fp16 (`MORE_STEPS`, schedule fitted to `TRAIN_HOURS`), builds the BM25 index and the
retrieval scoreboard, fine-tunes `small_wiki_rag` (retrieval-conditioned) and `small_wiki_sft` (plain, for comparison),
computes teacher embeddings for the map stage, draws the per-POS loss chart and prints answers with and without
retrieval. Set `QUICK = True` once first (~45 min). Full job ≈6 GPU-hours (≈5.3 without the sweep) of the weekly ~30; output ≈3 GB, kept as
the base for later runs (raw text included). Launch with *Save & Run All*, not an interactive session (idle GPU
sessions count against the quota).

## 6b. Roadmap (from the design discussion; GPU = Kaggle T4 hours)
| stage | what | GPU | status |
|---|---|---|---|
| grammar-heads sample-efficiency test | 3 arms × 3 seeds at 250k/1M/4M tokens, bits/byte + BLiMP (`scripts/sweep_arms.py`) | 0.75 h | stage 2 |
| writer | 25M model, ~3 epochs of 217M clean tokens, fp16 | ~5 h | stage 2 |
| retrieval baseline | BM25 over Wikipedia leads; retrieval-conditioned Q&A fine-tune; scoreboard (recall@k) | 0.5 h | stage 2 |
| map + index | teacher embeddings (stage 2) → regress 32-dim grammar-slot codes from chunked sentences (CPU) → index; STS-B test; beat BM25 on the scoreboard | 0 | next |
| intent scorer | power-word lexicon + POS → logistic regression | 0 | next |
| single-hop reasoning | ~10k step-by-step traces from an open teacher, fine-tune the writer | 10–20 h or API | later |
| two-hop | swap the writer for a ~1B open model (LoRA, 4-bit) in an iterative retrieve-think loop | 6–10 h | later |

Knowledge scales with disk (add text, re-index), fluency with GPU-hours (25M floor; 100–350M comfortable), the map by
regression against a stronger teacher (32 → 128 dims when collisions matter). Grammar heads are **parked**
(`configs/tiny_upos.json` stays as the record); a stronger variant, if any, gets 3 seeds.

## 7. Experiment protocol that this is built for
Same data, same tokenizer, same size; 3 seeds; report `eval_bits_per_byte` and BLiMP at 250k / 1M / 3.5M tokens:
1. `tiny` (baseline)  2. `tiny_upos` (grammar heads)  3. student distilled from each.
If 2 beats 1 at ≤1M tokens by more than the seed spread, grammar earns its keep at zero inference cost.

## Layout
```
corpora.json            data registry (edit to add data)
configs/                tiny.json, tiny_upos.json (parked), small_wiki.json
gramlab/
  data/conllu.py        CoNLL-U reader          data/chunker.py   UD tree → NP/VP/PP/... chunks
  data/registry.py      loaders per corpus type
  tokenizer.py          word / BPE / chunk-constrained BPE, shared special ids
  prepare.py            tokenizer + memory-mapped shards (tokens, upos, chunk)
  model.py              compact GPT, aux heads, sampling
  amp.py                bf16 / fp16+GradScaler / fp32 choice shared by train and finetune
  train.py              resumable trainer (--resume --more_steps), corpus mixing, aux losses, distillation, bits/byte eval
  finetune.py           dialogue SFT (loss on assistant tokens; over-long contexts trimmed, never the answer)
  retrieval.py          BM25 index over Wikipedia passages + the retrieval chat format
  chat.py               REPL; --retrieve work/bm25 puts the top passage in front of each question
  evaluate.py           perplexity / bits-per-byte / BLiMP / --per_pos    ckpt.py   save/load
scripts/make_wiki_qa.py synthetic Q&A from AMALGUM for the first fine-tune
scripts/make_rag_qa.py  retrieval-conditioned Q&A (gold / gold+near-miss / miss) from the Wikipedia dumps
scripts/plot_pos_loss.py per-POS loss chart from evaluate --per_pos results
scripts/sweep_arms.py    baseline vs grammar heads at small data, several seeds (sample-efficiency curve)
scripts/teacher_embed.py teacher embeddings (map targets), dense retrieval baseline, STS-B pairs
kaggle_continue.ipynb   stage-2 unattended run (continues from Kaggle Version 1)
RESULTS.md              findings per run and the decisions they caused
data/wiki_qa.jsonl      1,624 generated dialogues (included)
```
