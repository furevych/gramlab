# Results log

## Run 1 — overnight, 2026-11-08 (Kaggle T4, 4.72 h, `kaggle_overnight.ipynb`, saved as Version 1)

Data: Simple English Wikipedia 149,547 articles (44M words) + English Wikipedia 120,000 articles (114M words)
+ AMALGUM (3.5M tokens, parsed) = **217M clean tokens**. Eval on gold GUM test, never trained on.

### The unexpected positive finding: the writer already works at 25M parameters, and the facts problem is exactly separable

After only **0.7 epochs** (4,813 steps, stopped by the time guard at 3 s/step in emulated bf16) the 25M model:

- writes fluent, grammatical English in the right register and answers in the right *format* after a fine-tune on
  just 1,624 synthetic dialogues ("Paris is a city in the central part of France …");
- has already attached the **frequent** associations — Paris → France, Belgrade → Serbia / Balkans — which co-occur
  hundreds of times in the data;
- gets every **rare** fact wrong (populations, dates, roles), each of which appears once.

That is the frequency theory confirmed on the model's own output, and it is good news for the architecture:
the small writer needs no more parameters for fluency and format — it only lacks facts, and facts that appear
once will never be stored in 25M parameters at any number of epochs. So the division of labour is right:
**facts come from retrieval (BM25 now, the grammar meaning map later); the 25M model only has to read them and
write.** Perplexity on gold GUM: 135.8 (3M model) → 68.5 (25M, step 4000), bits/byte 1.68 → 1.45, still falling
fast when stopped.

A second, quieter positive: the grammar-heads null result (below) says the LM objective already learns part of
speech for free. The writer therefore does not need parsed training text at all — plain clean Wikipedia is
enough — which removes the dependency on silver parses for training the LM and frees the grammar work for
the place it is actually meant for: the meaning map (finding and intent), not the language model.

### Experiment A vs B — grammar heads (3M model, 4,000 steps = 33M tokens ≈ 9 passes over AMALGUM, same data, one seed)

|                                | baseline | + grammar heads (UPOS + chunk) |
|--------------------------------|---------:|-------------------------------:|
| bits/byte on gold GUM          | 1.6845   | 1.6939                         |
| BLiMP mean (300 pairs/phenomenon) | 0.6039 | 0.6050                        |

Null. Per-phenomenon swings (+0.24 wh-islands, −0.14 irregular participles) are noise at this sample size and one
seed. Predicting the next token's tag from the same hidden state that predicts the next token adds almost nothing
the LM objective does not already carry. Caveat: the comparison was made after 9 epochs; the hypothesis was *data
efficiency*, so the honest check is bits/byte at steps 250–1,000 in `work/runs/tiny*/log.jsonl` (2–8M tokens).
**Decision: parked.** If grammar supervision is tried again it will be a stronger form (supervising attention
toward chunk structure, or distillation from a syntactic model), with 3 seeds.

### Experiment C — 25M model on the Wikipedia mix

| model | perplexity on gold GUM | bits/byte | tokens seen |
|-------|-----------------------:|----------:|------------:|
| 3M (`tiny`)        | 135.8 | 1.68 | 33M (9 epochs of AMALGUM) |
| 25M (`small_wiki`) | 68.5 (step 4000) | 1.45 | 158M (0.7 epochs of the mix) |

Sample answers (fine-tuned on `wiki_qa`, no retrieval): Paris → "a city in the central part of France …
population of around 2,000"; Napoleon → "(died 1630) is a French composer of the Renaissance";
Japan → "Tokyo, Tokyo, Japan, Japan, Japan …" (decoding loop; fixed by the no-repeat-n-gram ban).

### What this changed in the project

- **Cut**: grammar-head retraining and BLiMP from the standing pipeline (parked; `configs/tiny_upos.json` kept as
  the record); the fresh-start fallback and the chat widget from the unattended notebook; `configs/small.json`
  (superseded by `small_wiki`); re-fetching the Wikipedia text on every run (the dumps now stay in the output).
- **Kept / finished**: the 25M writer (continued in fp16 with a time-fitted schedule), BM25 retrieval with the
  retrieval-conditioned fine-tune, per-POS loss diagnostics, the no-repeat decoding fix.
- **Added for the next stage** (the grammar meaning map): a retrieval scoreboard (BM25 and a dense teacher
  baseline, recall@k on title and body-sentence queries) that the map has to beat, teacher sentence embeddings for
  the BM25 passages and for every parsed AMALGUM sentence (regression targets for the 32-dim slot codes), and the
  STS-B pairs for the map's similarity test.
