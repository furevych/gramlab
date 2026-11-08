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
  `prepare.py` (tokenizer training + shards + incremental corpora), `scripts/make_wiki_qa.py`
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
```

## 2. What to expect from the conversation
A 3M-parameter model trained on 3.5M tokens produces locally grammatical, topically drifting text;
after `finetune` on `wiki_qa` it answers in the right *format* ("X is a city in ...") with invented facts.
That is the correct result at this scale — use it to see the mechanics work, not to get answers.
Coherence improves with (in order of payoff): more and cleaner data (tens of millions of tokens of
parsed Wikipedia via Stanza), `configs/small.json` (≈25M params, ctx 256), then real dialogue data.
Judge progress with `eval_bits_per_byte` (comparable across tokenizers) and BLiMP, not by chatting.

## 3. Scaling knobs
- `configs/tiny.json` → `configs/small.json`: d_model / n_layer / ctx / batch×accum. Shards are memory-mapped,
  so data size is bounded by disk, not RAM; bf16 autocast is on automatically on CUDA.
- More data: add a corpus (below) and list it in `train_corpora` with a `weight`; sampling is proportional
  to weight × size. To parse new raw text into tagged data, run Stanza/spaCy on Kaggle and point a
  `ud_conllu_dir` entry at the output — the chunker and the grammar heads then work on it too.
- Continue a model on new data without starting over:
  `python -m gramlab.train --config configs/small.json --init_from work/runs/tiny/ckpt.pt`
  (strict=False loading, so you can add or drop aux heads when continuing). `--resume` continues the
  *same* run with its optimizer state.

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
- **Evaluation**: `evaluate.py` — add a metric function next to `blimp()`.

## 6. Unattended run (Kaggle "Save & Run All")
`kaggle_overnight.ipynb` does everything in one background job (≈4–5 h on a T4): baseline vs grammar-heads
(bits/byte + BLiMP), fetches Simple English Wikipedia + 120k English Wikipedia articles with
`scripts/fetch_wikipedia.py`, trains `configs/small_wiki.json` (≈25M params, time-guarded by `train.max_hours`),
fine-tunes on the Q&A set, prints sample answers, and trims the output. T4s have no bf16, so training uses fp16 autocast.

## 7. Experiment protocol that this is built for
Same data, same tokenizer, same size; 3 seeds; report `eval_bits_per_byte` and BLiMP at 250k / 1M / 3.5M tokens:
1. `tiny` (baseline)  2. `tiny_upos` (grammar heads)  3. student distilled from each.
If 2 beats 1 at ≤1M tokens by more than the seed spread, grammar earns its keep at zero inference cost.

## Layout
```
corpora.json            data registry (edit to add data)
configs/                tiny.json, tiny_upos.json, small.json
gramlab/
  data/conllu.py        CoNLL-U reader          data/chunker.py   UD tree → NP/VP/PP/... chunks
  data/registry.py      loaders per corpus type
  tokenizer.py          word / BPE / chunk-constrained BPE, shared special ids
  prepare.py            tokenizer + memory-mapped shards (tokens, upos, chunk)
  model.py              compact GPT, aux heads, sampling
  train.py              resumable trainer, corpus mixing, aux losses, distillation, bits/byte eval
  finetune.py           dialogue SFT (loss on assistant tokens)      chat.py   REPL
  evaluate.py           perplexity / bits-per-byte / BLiMP            ckpt.py   save/load
scripts/make_wiki_qa.py synthetic Q&A from AMALGUM for the first fine-tune
data/wiki_qa.jsonl      1,624 generated dialogues (included)
```
