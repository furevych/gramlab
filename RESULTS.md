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

## Run 3 — stage 3, 2026-11-14/15 (Kaggle Version 3 of `gramlab`, 4.5 h)

+400,000 English Wikipedia articles (411M tokens; mix now 628M tokens), one pass in the 4 h budget (step 26,686 →
48,742, 1.57B tokens seen in total). Writer: perplexity 46.1 → **44.8**, bits/byte 1.314 → **1.304**. As expected,
fresh data still moves it but slowly: the 25M writer is at its capacity. **Frozen here.**

**The answers are now correct.** With the decoding fix, the retrieval-conditioned model reads the passage faithfully:
"The capital of Japan is Tokyo. In the course of history, the national capital has been in many locations other than
Tokyo."; Napoleon, Einstein, Danube, photosynthesis, volcano likewise; "Zxqv Blorptown" → "I don't have information
about Zxqv Blorptown." The no-retrieval control stays fluent and wrong (Einstein "a German artist of the early
Baroque period"). This is the architecture working as designed: facts from the index, words from the writer.

Retrieval over 658,694 passages: title recall@1 0.981, body-sentence 0.233 (down from 0.266 with 2.5× more
passages), all 0.607. One residual: "who is churchhill" (misspelt, no first name) retrieved "The Churchills (Israeli
band)"; fixed after the run by ranking titles that *contain* the query words by how much of the title they cover
("Winston Churchill" = 0.5 beats a three-word disambiguated title) — applies at chat time, no retraining.

**Next**: the meaning map (CPU): regress slot codes from the chunked AMALGUM sentences onto `work/teacher/
amalgum_sent.npy`, score on STS-B (teacher 0.824) and on the body-sentence queries (BM25 0.233, teacher ≈0.33).

## Run 4 — map stage v1, 2026-11-15 (`gramlab-map` Version 1, CPU, 46 min; `kaggle_map.ipynb`)

Map: 32-dim code per word (40k words + 8k hash buckets), summed per POS-chunk slot, linear readout onto MiniLM's
384-dim space, cosine regression on 191k parsed AMALGUM sentences. 1.63M parameters. Control: same model, one slot.

| system | fit to teacher (val cosine) | STS-B Spearman | retrieval recall@1 all / title / body | BM25 + map hybrid |
|---|---|---|---|---|
| map, 7 grammar slots | **0.570** | 0.662 | 0.061 / 0.101 / 0.021 | 0.582 / 0.957 / 0.208 |
| map, bag of words | 0.537 | **0.679** | 0.053 / 0.091 / 0.015 | 0.578 / 0.956 / 0.200 |
| BM25 | — | — | **0.607 / 0.981 / 0.233** | — |
| teacher (MiniLM, 22M params) | 1 | 0.824 | 0.590 / 0.849 / **0.331** | — |

What it says:
1. **The word codes learned synonymy** — the "synonyms as numbers" idea works at 32 numbers per word: city →
   urban, cities, metropolis, town; kill → killed, murdered, assassinate, stabbed; hate → dislike, despise; quickly →
   fast, rapidly, swiftly; france → french, european, strasbourg. (Nearest neighbours of *born* are names: the
   regression cannot separate a word from what it co-occurs with.) This is directly usable for the intent scorer
   (lexicon expansion: hate → dislike/despise).
2. **Grammar slots help the map fit the teacher (+0.033 cosine) but not the tasks**: on STS-B the bag of words is
   slightly better (0.679 vs 0.662); the POS-pattern slots add errors on paraphrase pairs whose structure differs.
   Same pattern as the LM experiments: the grammar signal is real but small, and it is not where the gains are.
3. **As a retriever v1 is useless on its own and slightly hurts BM25 in the hybrid.** Cause is clear: it was
   trained on single sentences, but passages are ~60 words plus a title, so the slot *sums* are dominated by
   length and frequent words, and a query like "Tell me about Belgrade." is mostly template words. 0.66 STS-B for
   1.6M parameters is respectable (the teacher is 22M), but regression onto a sentence embedding is the wrong
   training signal for retrieval.

**Decision**: keep the codes, drop the slots for the map, and train v2 for the actual job: mean-pooled codes,
trained on *passages* (title + text) against teacher passage embeddings plus query→passage pairs (the title
questions and body sentences the scoreboard uses), scored on the same scoreboard. CPU, no GPU quota.

## Run 5 — map stage v2, 2026-11-15 (`gramlab-map` Version 2, CPU, 2.4 h)

Same map (32-dim word codes, 7 POS-chunk slots, linear readout), trained for retrieval: mean pooling; inputs = 150k
passages ("title: text") + 291k question→passage and body-sentence→passage pairs (articles outside the scoreboard);
targets = the teacher's passage embeddings (266k, computed on CPU). Vocabulary 60k + 8k hash buckets; 2.27M parameters.

| system | STS-B | map alone recall@1 all / title / body | BM25 + map hybrid all / title / body |
|---|---|---|---|
| map v2, 7 grammar slots | 0.598 | **0.179 / 0.289 / 0.069** | 0.604 / 0.957 / **0.251** |
| map v2, bag of words | 0.662 | 0.117 / 0.194 / 0.040 | 0.595 / 0.957 / 0.232 |
| BM25 | — | **0.607 / 0.981 / 0.233** | — |
| teacher (MiniLM) | 0.824 | 0.590 / 0.849 / 0.331 | — |

1. **First clear win for the grammar slots.** Trained for retrieval, the slot map beats the identical bag-of-words
   map by a wide margin on its own (recall@1 0.179 vs 0.117 over 3,000 queries; body 0.069 vs 0.040) — far outside
   sampling noise. Which words are subject, verb, object and prepositional object matters for matching a question
   to a passage, in a way it did not matter for sentence similarity (STS-B, where the bag of words still wins).
2. **The BM25 + slot-map hybrid is the best body-sentence retriever we have without the teacher**: 0.251 vs 0.233
   for BM25 (+1.8 points on 1,500 queries — about the noise level, so "at least as good", not yet "better"), while
   it costs 2.4 points on title queries (0.957 vs 0.981) because the map's re-ranking disturbs exact-title hits.
   The map alone is still far below BM25 (0.179 vs 0.607) and below the teacher on body queries (0.069 vs 0.331).
3. **Map v2 trained 3× more data than v1 for a worse fit** (val cosine 0.46 vs 0.57): passage targets are harder,
   and 2.3M parameters cannot do what MiniLM's 22M do. It is the cheap complement to BM25, not a replacement.
4. **The word codes split by use**: v1 (sentence-level, AMALGUM) gives clean synonymy (hate → dislike, despise;
   quickly → fast, rapidly) and is the one to use for the intent lexicon; v2 (passage-level Wikipedia) gives
   topical neighbours (river → tributaries, dam, basin; born → raised, graduated) and junk for abstract words
   (hate, rage, quickly) — the right codes for retrieval, the wrong ones for intent.

**Next (cheap, CPU)**: tune the hybrid so it keeps BM25's exact-title answers (apply the map only when BM25 has no
exact-title hit, or weight BM25 2:1), on half the scoreboard and report on the other half; then the chat
notebook's hybrid retriever uses it. The map's passage matrix is saved (`work/map2/slots/passages.npy`).

## Run 6 — hybrid tuning, 2026-11-15 (`gramlab-hybrid` Version 1, CPU, 2 min)

BM25 top-50 re-ranked by the slot map: score = w · BM25/max + cosine. Rule chosen on half the scoreboard, reported
on the other half (750 title + 750 body queries):

| rule (held-out half) | all | title | body |
|---|---|---|---|
| BM25 alone | 0.593 | 0.949 | 0.237 |
| hybrid, w = 2 (chosen on dev) | **0.599** | 0.947 | **0.252** |
| hybrid, w = 1 + exact-title guard (best on test, not chosen) | 0.607 | 0.953 | 0.261 |

The hybrid is at parity with BM25 on title queries and +1.5 points on body-sentence queries on the held-out half —
a consistent small edge across all weights 1–3, never a loss. Saved as the chat retriever (`hybrid.json`: w = 2).
A regression noticed on the way: the partial-title ranking added after run 3 cost ~2.5 points on title queries
(0.981 → 0.956); fixed by applying it only when no exact-title article exists.

Where the map stands: a 2.3M-parameter grammar-slot encoder that makes BM25 slightly better on paraphrased queries
and gives compact synonym codes; not a replacement for a 22M dense encoder (teacher body recall 0.331).

## Run 7 — full English Wikipedia index, 2026-11-16 (`gramlab-fullwiki` Version 1, CPU, 71 min)

4,955,232 article leads (258M words), BM25 index 3.0 GB (2.24M terms, 140M postings), built in 8.5 min after a
57-min stream of the dump. Scoreboard on the same 3,000 queries: title recall@1 **0.913**, body **0.167**, all 0.540
(vs 0.981 / 0.233 / 0.607 over 659k passages): 7.5× more passages means 7.5× more look-alikes, mostly on body
queries. Single-surname questions ("who is churchhill") now land on whichever "Churchill" article uses the word most
("Churchill Show", a Kenyan comedy) — a first name fixes it; a popularity prior would be the principled fix.
The chat notebook prefers this index when the `gramlab-fullwiki` output is attached.

## Run 8 — writer swap, 2026-11-16 (`gramlab-writer` Version 3, T4, 99 min)

Qwen2.5-0.5B-Instruct + LoRA (8.8M trainable of 503M) on the 27k retrieval-conditioned dialogues, 2 epochs (3,382
steps, 87 min). Held-out 400 dialogues: **faithfulness 0.999** (share of answer words present in the passage),
**token F1 vs reference 0.987**, miss detection 0.875. Merged fp16 model 954 MB.

Sample answers are exact, clean reads of the passage (Belgrade, Napoleon, Tokyo, Einstein, photosynthesis — no
spelling slips, unlike the 25M writer) and "I don't have information about …" whenever retrieval misses. Which now
makes retrieval the only visible weakness: "When was Albert Einstein born?" fetched "Albert Einstein Square", "Which
river flows through Vienna?" fetched "River Cole, West Midlands", because the partial-title rule required *all* query
words in the title. Replaced after the run by a coverage score — (shared words / title words) × (shared words / query
words) — so "Albert Einstein" (0.67) beats "Albert Einstein Square" (0.44) and "Vienna" gets boosted for the river
question; exact titles still score 1 and plain titles still beat disambiguated ones. Applies at chat time.

Two earlier attempts failed for environment reasons (Kaggle's old `torchao` next to current `transformers`; then
`apply_chat_template` returning a dict), costing ~1.5 GPU-hours; the model is now saved before evaluation.

**Level 1 is done**: 0.5B writer + BM25 over all of English Wikipedia (+ the slot-map hybrid on the 659k index).

## Run 9 — core-v1 and the role linker, 2026-11-23/24 (local CPU, 4 min; reproduced by `kaggle_core.ipynb`)

**What changed.** The project got a stable core (`gramlab/core/`, see its README): layer 0 = indexation
(WordNet concept ids + a specific-entity registry that never guesses identity, VAD lexicon as the fixed affect
origin), layer 1 = learned components (role linker; entity typing and a live-text parser are open slots), layer 2 =
the ordered meaning key and index, layer 3 = the writer. One scoreboard (`python -m gramlab.core.bench`), always on
gold GUM test. The Nov 15–23 notebooks were triaged against it (README, "Triage"): the ordered map, event status
and the Level-1 retriever/writer were kept as design or code; the regex constructions, hand lexicons, 12–36-sentence
test sets, EWT baselines and MATRES/clinical pilots were not carried forward.

**Role linker — who did what to whom, learned from real text.** Hashed-feature logistic regression over
(predicate, candidate) pairs in a ±12-token window, trained on AMALGUM's 191,546 silver-parsed sentences
(1.49M pairs, 43 s on CPU, 0.9M nonzero weights ≈ 3.6 MB), threshold chosen on GUM dev (0.3), scored once on GUM test
(1,464 sentences, 2,308 predicates, 234 passive, 24 passive with an explicit agent).

| linker | link F1 all | active | passive | role swaps / 2,308 | µs per sentence |
|---|---:|---:|---:|---:|---:|
| word-order rules (tags) | 0.649 | 0.644 | 0.712 | 69 | 20 |
| word-order rules (words only) | 0.388 | 0.395 | 0.300 | 62 | — |
| learned, words only (no tagger anywhere) | 0.854 | 0.867 | 0.710 | 29 | 247 |
| learned, with tags | 0.887 | 0.897 | 0.773 | 12 | 170–210 |
| **learned + soft backfill** (current occupant) | 0.884 | 0.893 | 0.781 | 12 | 180 |
| gold UD (ceiling) | 1 | 1 | 1 | 0 | 11 |

Learning curve (with tags, AMALGUM, nested samples; F1 all / passive): 125 sentences 0.648 / 0.304 · 1k 0.776 / 0.448 ·
10k 0.840 / 0.671 · 100k 0.878 / 0.815 · 191k 0.885 / 0.797. Words only: 0.538 → 0.846. Gold GUM train (11k) gives
0.848 with tags — 100k silver sentences beat 11k gold ones. For comparison, the parallel EWT thread reached 0.743 with
gold tags and 0.388 on passives. Role swaps — actor and patient exchanged, the one error the meaning map cannot
tolerate — fall from 89 (125 sentences) to 12.

**The ordered key on real text** (GUM-test events with both roles, n = 681 active / 24 passive; key =
(class(actor), verb lemma, class(patient)), classes from the annotation):

| roles from | key exact, active | key exact, passive | reversed keys |
|---|---:|---:|---:|
| position only (the toy map's rule: first nominal before = actor, after = patient) | 0.755 | 0.083 | — |
| word-order rules with passive handling | 0.753 | 0.792 | 0 |
| learned linker | 0.775 | 0.583 | 0 |
| **learned + soft backfill** | **0.830** | **0.708** | 0 |

What it says: (1) on active sentences word order already gets three quarters of keys right — grammar buys the rest,
and buys passives entirely (0.08 → 0.71); (2) no linker ever produced a mirrored key, so the ordered map's failure
mode is *missing* roles, not swapped ones; (3) the learned model alone under-fills keys because it abstains
(precision 0.91, recall 0.86); letting it fill the second role from its own next-best candidate when the first is
confident (p ≥ 0.6, second ≥ 0.1 — first values tried, not tuned) raises key accuracy by 5–12 points at no cost to
link F1 — the first component swap decided by the scoreboard. (4) 39% of arguments are pronouns; their class comes
from GUM's coreference annotation here and will be "pronoun" on live text until a coreference component exists.

**Passive agents** (24): 16 correct, 6 agent in window but nothing chosen ("is considered *by many*", "were murdered
… *by U.S. citizens*"), 1 other token chosen ("praised by the *Berlin* music critic"), 1 outside the window. The fix
is recall on *by*-phrases, not window size.

**Other rows.** Entity typing prior (lemma majority from GUM train): 59.3% general / 58.1% specific — the row an
entity typer has to beat (delegated). WordNet coverage: not measured here (nltk absent locally; the Kaggle notebook
installs it). Affect: skipped (no lexicon file attached). Index over GUM test: 2,308 occurrences, 1,970 specific
keys, 1,504 family keys; the largest family postings are (person, say/think/go, missing) — intransitives dominate,
and with families = verb lemma the index is still mostly one occurrence per key, which is why layer 0's hypernym
families are the next lever.

**Next, in order:** run `kaggle_core.ipynb` for the WordNet rows; entity typer and spaCy parser adapter (delegated,
scored on the same rows); families from WordNet depth 1–3 vs the word codes, scored by key collisions vs paraphrase
matches; coreference for pronoun slots (GUM has gold); the VAD lexicon file for the affect row.
