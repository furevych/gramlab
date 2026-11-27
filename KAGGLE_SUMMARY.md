# gramlab — what it does, what it is for, how it was built (status 2026‑11‑27)

Repo: github.com/furevych/gramlab · full log of 20 runs: `RESULTS.md` · notebook index: `KAGGLE_INDEX.md`

## What it does

gramlab reads an English sentence and turns it into an **ordered event record**:

> *who* (actor) → *did what* (action + dictionary sense) → *to whom* (patient) · status: asserted / negated / possible

Records are stored under keys at several levels — exact participants, their classes (person, place, organisation …),
their identities (names), and the action's sense family — so the same fact can be found by name, by category or by
paraphrase, and the stored bindings are checked before an answer is given ("known / unresolved / omitted").

Everything runs on CPU with no GPU: a perceptron tagger + lemmatizer (live front end), a hashed logistic role linker,
an entity typer, a coreference ranker, a sense picker, WordNet 3.0 as the frozen "layer 0" of concept ids, and the
in‑memory index. A 0.5B instruction model (Qwen2.5 + LoRA) phrases the final answer from the retrieved sentence.

## What it is good at — measured

| Capability | Number | Benchmark |
|---|---|---|
| Role assignment (who/whom), gold tags | **0.887 F1** | GUM test, gold UD |
| Role assignment on raw text, no parser | 0.85 F1 | GUM test re‑tagged live |
| Ordered key exact (active / passive) | 0.83 / 0.71 | GUM test |
| Telling "X called Y" from "Y called X" | **96 %** | 698 held‑out paraphrase queries + role‑swapped decoys |
| Same, for BM25 / for MiniLM dense encoder | 0 % / 44 % | same bench |
| Retrieval, BM25 alone → BM25 + key → BM25 + MiniLM + key | 0.69 → 0.89 → **0.89** recall@1 | same bench |
| Layer‑0 sense *distribution* vs first sense (no training) | +17 points | run 17 |

## What it is not good at — measured

| | Number | Benchmark |
|---|---|---|
| As a retriever for open questions (any structural variant) | 0.14–0.19 vs BM25 0.62 | SQuAD v1.1 dev |
| As a re‑ranking bonus on open questions | −3.6 points | SQuAD, BM25 + MiniLM 0.672 → 0.636 |
| Entity typing / coreference / verb senses | 73 ± 2 / 55 / 62 | GUM, SemCor |

Reason: on unrestricted prose the extraction chain is right about half the time, and a wrong triple looks like a right
one. The structural representation is **asymmetric by construction** (actor and patient are different slots), which is
exactly what similarity‑based retrievers lack — and it is the only property on which it beats them.

## Potential uses

- **Reversal guard beside any retriever** (BM25, dense, RAG): demote candidates that share the question's participants
  in the wrong order. Ready now, in internal or reviewed pipelines.
- **Contradiction / claim checking**: "A sued B" vs "B sued A"; "drug reduced mortality" vs the reverse.
- **Structured memory for an assistant**: who asked whom for what, who owes whom — answered by role, exactly.
- **Event extraction into a database** (who acquired / appointed / defeated whom), with asserted/negated/possible kept
  apart. Needs a confidence gate and a real parser first (see below).
- **Hard‑negative generator** for training order‑aware dense encoders: the reversal bench produces (query, source,
  swapped decoy) triples automatically from any parsed corpus.

Not for: open‑domain search, where/when/how‑many questions, or any setting that trusts a single extracted triple
without a stronger finder in front of it.

## Methodology, in short

1. **Stable core first** (Nov 23): fixed data contracts (`records.py`), fixed benchmark (`bench.run` scoreboard on
   gold GUM), every component replaceable and scored in isolation; the 21 earlier toy‑world notebooks were retired.
2. **Layer 0 = indexation, not recognition**: WordNet ids for general concepts, a separate registry for names; keep
   every candidate sense with a weight, decide later. Measured worth +17 points over choosing the first sense.
3. **Learned where there is a signal, rules where there is not**: linker, typer, coref, sense picker are small linear
   models with hashed features, hyper‑parameters chosen on GUM dev, GUM test never touched for tuning.
4. **One ordered record per event**, keys at several levels, verification against stored bindings before answering.
5. **Benchmarks before belief**: constructed paraphrase/reversal bench (corpus‑derived, flagged as such), live‑query
   mode (tagger + linker on raw text), SQuAD dev as the single external test never used for tuning, a fresh‑clone audit
   that reproduced every number within ±1 and corrected two.
6. **Negative results recorded as carefully as positive ones**: hypernym families (87 % wrong), gazetteer (±1 = noise),
   key as retriever (run 16), soft map in three forms (runs 17–19), key as SQuAD re‑ranker (run 20).

## Training data and recipe

| Component | Trained on | Evaluated on |
|---|---|---|
| Role linker | AMALGUM — 191,546 silver‑parsed sentences, 8 genres (academic, bio, fiction, interview, news, voyage, how‑to, reddit); gold roles derived from UD relations | GUM test (gold), 6 extra UD treebanks (run 21) |
| Live front end (tagger, lemmatizer) | 40k AMALGUM sentences, averaged perceptron, 4 epochs | GUM test re‑tagged |
| Entity typer | GUM train entity annotation (+ WordNet lexname features) | GUM test |
| Coreference ranker | GUM train coreference chains | GUM test |
| Sense picker | SemCor (brown1/brown2), shared rank classifier | SemCor held‑out |
| Writer | Qwen2.5‑0.5B‑Instruct + LoRA on retrieval dialogues (T4, 2 epochs) | manual + perplexity |
| 25M "small_wiki" writer (earlier) | Simple + English Wikipedia leads, ~1.7 GB, fp16, 4 passes | GUM gold, 44.8 ppl (frozen at capacity) |
| Not used | EWT (web text, rejected), hand‑written toy sentences | — |

Run 21 (`kaggle_more_data.ipynb`) widens the linker's training data beyond Wikipedia‑family text: gold UD treebanks of
other registers (legal, talks, literature, manuals, dialogue, GENTLE's eight odd genres) and Stanza‑parsed Brown,
Gutenberg, Reuters, web forums and chat; scored on GUM test and on each treebank's own test split.
