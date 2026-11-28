# gramlab — the component board

`gramlab/system/` runs a text through components that each read some layers of a Document and write one. Each has a
contract, a failure value (never a guess), a benchmark with a number and a date. `System.status()` prints this table
from the code; this file is the human copy and the rules.

| component | reads → writes | contract | failure value | benchmark | value | date |
|---|---|---|---|---|---|---|
| front_end | text → sentences | tokens, UPOS/XPOS, lemmas (perceptron, 40k AMALGUM) | tag `X` | UPOS acc., GUM test, live | 0.905 | 11‑26 |
| linker | sentences → events | per verb: actor span, patient span, voice, status | slot = None | link F1 GUM test, gold / live | 0.887 / 0.854 | 11‑26 |
| entities | sentences, events → analysis | registry (who is who), records (actor, action, patient, status, sentence), mentions | pronoun without antecedent dropped | coreference GUM test | 0.549 | 11‑26 |
| context | analysis → about | ranked entities the text is about (learned centering scorer) | — | title test top‑1, 4 genres vs counting | 0.74/0.54/0.74/0.54 vs 0.64/0.51/0.61/0.49 | 11‑28 |
| skeleton | analysis → skeleton | records that carry the text, in order; selected sentences | — | qualitative (Persuasion) | run 23 | 11‑28 |
| direction | analysis → direction | reversals (same participants, opposite order) and negation conflicts, within the document or against a reference | unresolved slot not compared | reversal separation, 698 held‑out queries | 0.96 (BM25 0 %, MiniLM 44 %) | 11‑27 |
| writer | skeleton, about → prose | faithful prose from the skeleton only (generative model, optional) | skeleton as lines | end‑to‑end accuracy | not measured | — |
| layer0 (inside entities/index) | — | WordNet sense distributions, SemCor priors | all senses kept | vs first sense | +17 | 11‑26 |

## Rules

1. One contract per component; a component states what it cannot decide instead of guessing.
2. One benchmark per component, re‑runnable by one command (`component.benchmark(...)`), logged in RESULTS.md with a date.
3. A component may replace another only if it scores at least as well on its own benchmark **and** does not lower the
   end‑to‑end numbers of the systems that use it (title test, reversal bench, SQuAD, Persuasion checks).
4. Every run records the versions it used (`doc.trace`).
5. Negative results stay in RESULTS.md next to the positive ones.

## What the system does today (one call)

`System.default(models_dir).run(text)` → sentences, events, registry + records, what the text is about, the record
skeleton and selected sentences, reversal and negation conflicts. On a six‑sentence merger note it ranks Harbor Trust and
Northfield Bank as the subject, lists the seven events with status, flags *Harbor Trust → acquire → Northfield Bank*
(sentence 5) as a reversal of sentence 2, and *bank → notify → regulator* (asserted, sentence 6) as conflicting with the
negated sentence 1. Latency ≈ 1.3 s for the first call (model load), milliseconds after.

## Open components (next)

- **writer** on Kaggle: Qwen2.5‑1.5B reading chapter text + skeleton + registry; output verified against records.
- **entities**: nominal → name linking ("the bank" → Northfield Bank), status on refused/failed events, retrain the
  coreference ranker with broader data (run 21).
- **context** for long texts: retrain the scorer without short‑document position features.
- **memory**: HRR scratchpad per context for partial probes (run 22), behind the key index.
