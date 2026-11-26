# gramlab core — `core-v1`

The stable part of the project. Five contracts, one pipeline, one scoreboard. A new experiment is either
**a replacement for one component**, scored on that component's fixed benchmark against the current
occupant, or **an end-to-end measurement** through the unchanged pipeline. Nothing else ships.

```
text / CoNLL-U --parser--> Sentence --linker--> [Event] --keyer--> keys --index--> occurrences
question --index.search / index.related--> occurrences --writer--> answer
```

## Layers

| layer | module | contract | occupant in core-v1 | benchmark (gold GUM test) |
|---|---|---|---|---|
| 0 · indexation | `registry.GeneralIndex` | lemma+POS → concept id, family id (WordNet synset / hypernym at depth *d*) | WordNet 3.0 read directly from its files (`core/wordnet.py`, no nltk); lemma ids if the files are absent | coverage of action and argument lemmas; family depth vs synonym merging |
| 0 · indexation | `registry.SpecificIndex` | name → entity id; seed registry (exact alias) + local ids for unresolved names, identity never guessed | empty seed; local registry | — (size reported) |
| 0 · origin | `affect.Affect` | action → (valence, arousal, dominance); `force(event)`; power-word corner | file-based lexicon (NRC VAD / Warriner), skipped if absent | coverage, power-event share |
| 1 · learned | `linker` | `Sentence → [Event]` | `BackfillLinker` = `LearnedLinker` (hashed logistic over predicate–candidate pairs, trained on AMALGUM silver) + its own second-best candidate when one role is confident; `RuleLinker` baseline; `GoldLinker` reads UD | link F1 all/active/passive, role swaps, µs/sentence |
| 1 · learned | entity typing | mention → type, general/specific | `typer.EntityTyper`: hashed logistic over surface + WordNet lexicographer classes, GUM-trained (74.5% general / 73.7% specific; prior 59 / 58); fills `ents` on live text | accuracy per kind on GUM-test event arguments |
| 1 · learned | sense picker | (lemma, pos, context) → synset | `sense.RankPicker` for verbs (shared hashed logistic model predicting the sense rank, SemCor-trained, 62.4% vs first sense 59.7%); first sense for nouns (picker 72.9% < 74.4%) | SemCor brown2 accuracy, verbs / nouns |
| 1 · learned | coreference | pronoun mention → antecedent mention | `coref.RankedResolver` (mention-pair ranker on GUM train chains, 54.9%); rules: subject-first 47.2%, nearest nominal 14.3% | GUM test: pronoun event arguments in the gold chain |
| 1 · learned | live front end (parser slot) | text → `Sentence` | `tagger.LiveFrontEnd`: perceptron tagger + lemma table on AMALGUM silver, predicates = tagged verbs, voice/status by surface rule, **no dependency parser** (link F1 0.851 vs 0.884 with gold parse) | GUM test re-tagged: tag/lemma accuracy, predicates found, link F1 |
| 2 · index | `key.Keyer`, `index.MeaningIndex` | one `ResolvedEvent` per occurrence (bindings with state known/unresolved/omitted, chosen sense + alternatives); keys derived from it only: specific, family `(class, sense, class)`, strict identity `(id, sense, id)`, and permissive *proposals* (name tokens, bridged ids, one-sided) that must be verified; `add / lookup / related / ask / search` |
| 2 · index (old row) | | `(slot(actor), action, slot(patient), status)` → id; family postings `(class, sense_k, class)` and identity postings `(id, sense_k, id)` for the action's top-k senses (k=3 when posting, 10 when looking up); `add / lookup / related / search` | postings + occurrence records; `search` delegates to Level-1 `BM25Index` / `HybridRetriever` when attached | key exact vs gold, position-only control; occurrences per key |
| 3 · writer | `Pipeline.writer` | `(question, passages) → text` | Level 1's Qwen-0.5B + LoRA (`scripts/finetune_hf.py`), attached at chat time | faithfulness / token F1 (RESULTS run 8) |

A **slot** is `(kind, class, id)`: kind ∈ general / specific / pronoun / missing; class = entity type for
specific mentions, entity type or concept family for general ones; id = synset or registry id. Order is
meaning. IDs are deterministic (`stable_id`), so two machines agree without a counter.

## Questions

`Pipeline.ask("Who called John?")` reads the question through the same front end and linker as the text, marks the
wh-role as *requested*, uses the given participant's identity/proposal postings to find candidates, **verifies** each
candidate's stored binding against the given participant, and returns the bound role with its source sentence —
`unknown` when the stored role is unresolved or omitted (requested ≠ missing). `python -m gramlab.core.selftest`
runs the milestone: *Tom called John. John called Tom. John was called.* → *Who called John?* = Tom + sentence;
with only *John was called.* → unknown; plus did-questions, passives and a synonym verb (8 checks).

## End-to-end benchmark

`python -m gramlab.core.bench_retrieval --amalgum '<glob>'` indexes all of AMALGUM through the pipeline and measures
event retrieval against BM25 with synonym-substituted role queries and role-reversal decoys (run 10 in `RESULTS.md`).

## Run the scoreboard

```
python -m gramlab.core.bench --gum data/gum --amalgum 'data/amalgum/amalgum/*/dep/*.conllu' [--vad nrc_vad.txt] --out results
```

`kaggle_core.ipynb` does the same on Kaggle CPU (fetches GUM, sparse-clones AMALGUM's `dep/` folders,
installs WordNet). Baseline numbers are in `RESULTS.md`, run 9.

## Triage of the Nov 15–23 notebooks (what was kept, what was not)

Kept, as core design or code:
- **ordered meaning map** (specific / family postings, occurrence records, lookups by status) → `key.py`, `index.py`, with the hand table replaced by the registries;
- **event status** (asserted / negated / possible read from the predicate's dependents) → `linker.status_of`;
- **BM25 + slot-map hybrid, full-Wikipedia index, 0.5B writer** (runs 2–8) → `index.search`, writer slot, unchanged;
- **entity classes from annotation** (route index, place linking) → `records.Mention.etype`, and the typing benchmark;
- the EWT thread's one transferable idea — a learning curve for a role linker — redone on AMALGUM/GUM (run 9).

Not carried forward, and why:
- regex construction parsers and the 17-noun / 4-verb worlds (frozen_construction_transfer, voice_composition, compositional_meaning_map, predicate_roles, clause_scope, shallow_roles, dependency_fallback): the behaviour they tested is now measured on 2,308 real predicates by `bench.links` / `bench.keys`; their hand-authored lexicons and families are replaced by layer 0;
- per-notebook test sets of 12–36 sentences: no longer admissible as evidence; GUM test is the fixed set;
- EWT-based baselines and coverage audits (project_progress, dependency_fallback): EWT is out of the project;
- MATRES temporal attachment / narrative time (6–28 pairs): parked until the event record has a time slot and a benchmark with hundreds of pairs;
- clinical (Freed) symptom rules: domain tuning before the general model, parked.

## What is deliberately still open

- pronouns: 39% of key arguments. The coreference ranker (run 9e) now gives a pronoun slot its antecedent's kind/class/id (651 of 701 third-person pronoun slots on GUM test reach a nominal); first/second person stay unresolved by design; accuracy 55% against a 75% candidate ceiling;
- action families: **first-sense hypernym families do not work** (run 9b: depth 1 merges 87% unrelated verb pairs, catches 11% of synonym pairs), so the default family is the synset itself (depth 0). Instead of one family, an event posts under its verb's top-3 senses (run 9c: links 26% of synonym pairs, no hubs); the verb sense picker (run 9d) now leads each event's family postings with its chosen sense;
- specific-entity seed: Wikidata-derived JSON (`SpecificIndex.load_seed`) not yet built; the full-Wikipedia index already keys articles by title;
- live keys: roles survive the loss of the parse (0.62 vs 0.68 exact on non-pronoun events) but the class key falls from 0.74 to 0.37 because the typer must be right on both slots — **entity typing is the live bottleneck now**, not linking; a real parser (spaCy adapter) is the other open comparison.
