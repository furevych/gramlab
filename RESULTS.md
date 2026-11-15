# Results log

Earlier status doc (Nov 7): https://claude.ai/artifact/GuUe9ayUsbRyyvq26DhrrP

## Run 0 — CPU, grammstat (before gramlab): what the n-gram experiments established

AMALGUM (3.5M tokens, silver UD parses) for training, GUM gold for eval; Kneser-Ney trigrams.

1. **The chunk extractor is sound**: gold vs silver chunk inventories agree within half a point (NP 50.4% vs 50.9% of
   tokens, VP 16.2% vs 16.2%, PP 9.7% vs 9.7%). Parsing Wikipedia with Stanza and computing grammar statistics is safe.
2. **A grammar class as one extra integer per word helps, most at small data**: word trigram + POS-class trigram
   gives −13.6% perplexity at 100k training tokens and −8.0% at 3.5M (the class model at 100k matches a word model
   trained on 1.7× the text; at 3.5M the multiplier is 1.36×). Class tables add 3% to model size.
3. **Grammar chunks as vocabulary units lose**, even with oracle segmentation: 1M NP occurrences spread over 330k
   distinct strings; 10k types cover 53% of NP tokens vs 91% for words. Bits/byte: words 1.91, frequency-merged
   superwords 2.07, grammar chunks 3.18. Chunk segmentation generalises to new text (55.6 units/100 words on train
   and test) while the frequency merge table overfits (55.5 → 64.2).
4. **Chunk tags add nothing beyond POS** as prediction classes (chunk×UPOS ≈ UPOS). Chunks are for segmentation
   and interpretability.
5. **Power words are ambiguous without grammar, and POS fixes most of it**: in the corpus *strike* is a noun 69% of
   the time, *hate* 52% (hate speech), *love* 54%, *attack* 86%, *fire* 96%. Lexicon + POS + chunk type is still cheap;
   negation and sarcasm remain for a small classifier.

Design rules taken from the literature review in the same session: perplexity does not show what grammar buys
(Hu et al. 2020), so judge grammar work on BLiMP/SyntaxGym and on **sample efficiency**; the reliable way to make
syntax pay at zero inference cost is **distillation from a syntax-aware teacher** (Kuncoro et al. 2019) or a
train-time-only structural loss (tree-planted attention); never put a parser in the inference path; synonym
**indexes are lookup keys, sums of indexes impose a fake ordering** — the meaning map's slot codes are therefore
small learned vectors summed per slot, not integers; clean data is the largest lever (FineWeb-Edu: 10× fewer tokens).

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
This is consistent with run 0's curve, not a contradiction of it: the class gain was already down to −8% at 3.5M
tokens, so after 33M it could only be null. The planned test — 3 arms × 3 seeds at 250k / 1M / 4M tokens — was never
run; stage 2 runs it (`scripts/sweep_arms.py`). **Decision: parked unless the sweep shows a gain at ≤1M tokens.**
If grammar supervision is tried again beyond that, it will be a stronger form (supervising attention
toward chunk structure, or distillation from a syntactic model), with 3 seeds.

### Experiment C — 25M model on the Wikipedia mix

| model | perplexity on gold GUM | bits/byte | tokens seen |
|-------|-----------------------:|----------:|------------:|
| CPU n-gram model (run 0, grammstat) | — | 1.91 | AMALGUM |
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

## Run 2 — stage 2 full run, 2026-11-08/09 (Kaggle Version 2 of `gramlab`, 4.5 h, fp16 at 0.61 s/step)

**Writer.** `small_wiki` continued from step 4,000 to 26,686 (874M tokens ≈ 4 passes over the 217M-token mix):
perplexity on gold GUM 68.5 → **46.1**, bits/byte 1.450 → **1.314**. The curve is flat at the end (46.46 at step
24k, 46.1 at 26.7k): more epochs of this data buy nothing more; the next lever is more data or a bigger model.

**Retrieval scoreboard** (3,000 queries, 264k passages): BM25 with exact-title ranking — title queries recall@1
**0.982** (0.89 before the fix), body-sentence queries **0.266** / @5 0.423 / @10 0.491. Dense MiniLM teacher: title
0.849, body **0.331** / 0.466 / 0.516. The body case (different words, same meaning) is the map's target; BM25 and
the teacher are within 7 points of each other there and both are low.

**Grammar heads, sample-efficiency sweep** (3 seeds, bits/byte on gold GUM, min–max in brackets):

| tokens | none | upos | chunk |
|---|---|---|---|
| 250k | 2.597 (2.591–2.603) | 2.596 (2.594–2.598) | 2.600 (2.598–2.602) |
| 1M | 2.280 (2.279–2.280) | **2.260** (2.259–2.263) | 2.264 (2.263–2.266) |
| 4M | 2.043 (2.042–2.044) | 2.041 (2.038–2.044) | 2.042 (2.040–2.043) |

A real but small effect: at 1M tokens the UPOS head is 0.9% better than the baseline, outside the seed spread; at
250k nothing yet, at 4M the arms have converged. BLiMP at 100 pairs/phenomenon is noise (0.51–0.54). This is the
neural copy of run 0's curve (−13.6% → −8% perplexity for the n-gram class model) at a much smaller size: the
Transformer already extracts most of what a tag head could add. Decision stands: parked; not worth building on.

**Per-POS loss** (`results/per_pos_loss.png`): 25M model's tagged-token loss 3.83 vs 4.91 for the 3M model; loss by
chunk position 4.07 at chunk starts, **4.54 inside chunks** (content words), 2.27 outside (punctuation, function words).

**Answers.** The retrieval-conditioned model's val loss was 0.047 (copying is learned), yet its answers were garbled
("The capital of Japan is Stockholm", "third-longest river in Africa"). Cause: the repetition penalty and the n-gram
ban were applied to the whole window including the retrieved passage, so every word of the passage was penalised at
generation time — exactly the words a reader must copy. Fixed in `generate()` after this run (penalties only on
generated tokens; retrieval chat decodes at temperature 0.3). Two retrieval fixes also went in: exact-title ranking
prefers plain titles over disambiguated ones ("Paris" over "Paris (Paris Hilton album)"), and query words not in
the index are spell-corrected within one edit ("churchhill" → "churchill").

**Stage 3** (`kaggle_stage3.ipynb`): +400k English Wikipedia articles, one pass over the ~620M-token mix, fresh
BM25/Q&A, answers with the fixed decoding.
