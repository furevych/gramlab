# The Direction Problem — order‑aware retrieval on a CPU

*gramlab · Taras Furevych · November 2026 · code and every number: github.com/furevych/gramlab (RESULTS.md, runs 0–21)*

Search engines and neural retrievers score **similarity**, and similarity has no direction: *Apple sued Samsung* and
*Samsung sued Apple* look the same to them. We measured how badly, and built a 7 MB, CPU‑only verifier that fixes it.

---

## 1 · The problem: same words, opposite facts

| | |
|---|---|
| Apple sued Samsung over the patents. | Samsung sued Apple over the patents. |
| The lender must notify the borrower. | The borrower must notify the lender. |
| The mass compresses the vessel. | The vessel compresses the mass. |

Both halves of each pair are plausible. Same bag of words, same topic, nearly the same vector. In law, finance and
clinical notes the direction *is* the fact — and a search system that returns the wrong half fails silently.

**How often each method ranks the correct sentence above its role‑reversed twin** (698 real sentences from an 8‑genre
corpus, paraphrased as queries, each paired with its swapped twin inside a 192,244‑sentence collection — run 20):

| BM25 word search | MiniLM dense encoder | gramlab ordered key |
|:---:|:---:|:---:|
| **0 %** — a tie every time | **44 %** — prefers the reversed one more often than not | **96 %** |

---

## 2 · What we built

`raw text → tagger + lemmatizer (perceptron, no parser) → role linker (hashed logistic, 0.887 F1) → ordered record
(actor → action → patient) → keys at several levels (name · class · identity · sense) → verified answer`

> **The regulator was not notified by the bank.** ⇒ actor = *bank* · action = *notify* · patient = *regulator* ·
> voice = passive · status = **negated**

- **Compact, CPU‑only.** ~7 MB of hashed linear weights and plain postings; < 1 ms per sentence; 192k sentences indexed
  in about a minute. No GPU, no download, no API.
- **Order‑aware by construction.** Actor and patient are different slots, so a reversed sentence is a *different key*,
  not a nearby point. Negation and modality are part of the record.
- **Inspectable and honest.** Every match carries its record — an auditable reason, not a score. No verified event → no
  answer; a missing actor is marked *omitted*, never guessed.
- **Retrainable.** The linker retrains in ~10 CPU‑minutes from any UD‑style parsed text. Layer 0 (WordNet 3.0) is frozen
  and keeps *every* sense of a word with a weight — deciding early cost 17 points (run 17).

---

## 3 · Results

**Reversal bench** — 698 real sentences, each with a role‑swapped twin, in a 192k collection (run 20):

| System | recall@1 | right sentence above its twin |
|---|:---:|:---:|
| BM25 (word search) | 0.69 | 0 % (tied) |
| MiniLM dense encoder (22 M params) | 0.29 | 44 % |
| BM25 + MiniLM — the standard hybrid | 0.32 | 44 % |
| BM25 + gramlab key | 0.89 | 94 % |
| MiniLM + gramlab key | 0.85 | 96 % |
| **BM25 + MiniLM + gramlab key** | **0.89** (0.99 in top 10) | **96 %** |

The dense encoder alone drops to 0.29 because it ranks the decoy first; the key repairs it to 0.85 without retraining anything.

**External test — SQuAD v1.1 dev, 10,570 real questions**, top sentence contains the answer (runs 16–20):

| | |
|---|:---:|
| BM25 | 0.615 |
| **BM25 + MiniLM** | **0.672** |
| any gramlab structural variant as *the* retriever | 0.14 – 0.19 |
| BM25 + MiniLM + key as a blanket re‑ranking bonus | 0.636 — worse |

Why: most real questions ask *when, where, how many, what is* — not who acted on whom. On unrestricted prose the
extraction chain is right about half the time, and a wrong record looks exactly like a right one, so as a blanket bonus
it promotes wrong sentences as often as right ones.

**So the component is a direction verifier** — applied where candidates share the participants — not a retriever and not
a general re‑ranker. Five attempts to make grammar *the* retriever (meaning map v1/v2, hypernym families, key alone,
soft map in three forms) are all recorded as negatives.

---

## 4 · Why it works: similarity is symmetric, facts are not

*Vector and word retrievers* compare two texts and return one number, and sim(A,B) = sim(B,A) by definition. Two
sentences that share participants, action and topic land at almost the same point; which ranks first is noise (44/56).
The encoder *could* see order — it reads positions — but it was trained to answer "same topic?", where reversal was never
penalised.

*The ordered record* makes position the role: (Apple, sue, Samsung) ≠ (Samsung, sue, Apple) by construction, with no
training. They are filed under different keys and a question about one never meets the other. The only misses are
extraction errors (linker 0.887 F1 over two slots), which is why separation is 96 %, not 70 %. This is the same split
production systems make: vectors to **find**, triples (knowledge graphs, fact checkers) to **assert**.

---

## 5 · Method and data

- **Stable core first.** Fixed data contracts, one permanent scoreboard on a gold treebank (GUM), every component
  replaceable and scored alone. 21 earlier toy‑world notebooks retired and labelled (see *gramlab INDEX*).
- **Learned where there is a signal.** Linker, typer, coreference and sense picker are small linear models with hashed
  features; hyper‑parameters chosen on GUM dev; GUM test never used for tuning; SQuAD dev the single external test,
  never tuned on.
- **Against real alternatives.** BM25 and a pretrained MiniLM encoder, on a reversal bench built from real sentences and
  in live‑text mode (tagger + linker, no parser).
- **Audit and negatives.** Fresh clone, every model retrained, every number reproduced within ±1, two corrections
  published; hypernym families (87 % wrong), gazetteer (±1 = noise), key‑as‑retriever, soft map ×3, key‑as‑re‑ranker
  all logged.

| Component | Trained on | Evaluated on |
|---|---|---|
| Role linker | AMALGUM — 191,546 silver‑parsed sentences, 8 genres (academic, biography, fiction, interview, news, travel, how‑to, forum); roles derived from UD relations | GUM test (gold); 6 further UD treebanks (run 21) |
| Live front end | 40k AMALGUM sentences, averaged perceptron, 4 epochs | GUM test re‑tagged |
| Entity typer · coreference | GUM train annotation (+ WordNet features) | GUM test |
| Sense picker | SemCor | SemCor held‑out |
| Layer 0 | nothing — WordNet 3.0 frozen, SemCor frequency priors | — |
| Answer writer | Qwen2.5‑0.5B‑Instruct + LoRA on retrieval dialogues (~2 GPU‑hours, one T4) | manual, perplexity |
| Not used | web‑text treebank (EWT), hand‑written sentences, anything from a test split | |

Run 21 (*gramlab-more-data*) widens the linker's data beyond Wikipedia‑family text: gold treebanks of legal,
parliamentary, literary, dialogue and eight further registers, plus Stanza‑parsed Brown, Gutenberg, Reuters, forums and chat.

---

## 6 · Who should care, and what is next

- **AI over clinical notes.** "The mass compresses the vessel" vs the reverse; "patient refused the procedure" vs
  "physician deferred it"; drug A potentiates drug B. Direction errors change care. A verifier that marks negation,
  modality and who‑acted‑on‑whom raises extraction accuracy and gives a reason a clinician can check — the most
  direct application of this work.
- **Search and RAG teams** (contracts, filings, news): demote candidates that share the question's
  parties in the wrong order. A few MB beside the index, no retraining of the big model, a reason you can show an auditor.
- **Fact‑checking**: claim vs evidence with swapped roles — the silent failure of similarity models; negation and modality
  come with the record.
- **On‑device assistants**: exact memory of who asked whom for what, who owes whom — answered by role, offline.
- **Model builders**: the reversal bench is an automatic hard‑negative generator from any parsed corpus — the data needed
  to teach dense encoders direction.

**Not for:** open‑domain search, when/where/how‑many questions, or any pipeline that would trust a single extracted
record without a stronger finder in front of it.

**Next, in order:** run 21 (broader data) · gate the key to fire only on shared‑participant reversals, so it never costs
anything on general questions · a real dependency parser in the front end (coordination, long gaps, ditransitives) · a
confidence threshold — precision over coverage, the right trade for a verifier.

---

### Notebooks on Kaggle

| Notebook | What it holds |
|---|---|
| gramlab-core | core‑v1 scoreboard: linker, keys, typing, senses, coreference, live front end, retrieval, SQuAD (runs 9–16) |
| gramlab-dense | run 20: MiniLM finder + key verifier on the reversal bench and SQuAD |
| gramlab-more-data | run 21: broader training data for the linker |
| gramlab, gramlab-map, gramlab-hybrid, gramlab-fullwiki, gramlab-writer, gramlab-chat | the earlier retrieval + writer pipeline (runs 1–8) |
| gramlab INDEX · notebook verdicts | every notebook, including the 21 retired ones, with a one‑line verdict |
