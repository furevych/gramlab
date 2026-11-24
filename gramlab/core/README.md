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
| 1 · learned | entity typing | mention → type, general/specific | `bench.PriorTyper` (lemma majority) — a row to beat | accuracy per kind |
| 1 · learned | sense picker | (lemma, pos, context) → synset | `sense.FirstSense` (WordNet order); Lesk and SemCor naive Bayes do not beat it | SemCor brown2 accuracy, verbs / nouns |
| 1 · learned | parser for live text | text → `Sentence` | **none yet** (`ConlluParser` reads parsed files) | link F1 through `GoldLinker` on predicted vs gold parses |
| 2 · index | `key.Keyer`, `index.MeaningIndex` | `(slot(actor), action, slot(patient), status)` → id; family postings `(class, sense_k, class)` for the action's top-k senses (k=3); `add / lookup / related / search` | postings + occurrence records; `search` delegates to Level-1 `BM25Index` / `HybridRetriever` when attached | key exact vs gold, position-only control; occurrences per key |
| 3 · writer | `Pipeline.writer` | `(question, passages) → text` | Level 1's Qwen-0.5B + LoRA (`scripts/finetune_hf.py`), attached at chat time | faithfulness / token F1 (RESULTS run 8) |

A **slot** is `(kind, class, id)`: kind ∈ general / specific / pronoun / missing; class = entity type for
specific mentions, entity type or concept family for general ones; id = synset or registry id. Order is
meaning. IDs are deterministic (`stable_id`), so two machines agree without a counter.

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

- pronouns: 8% of GUM-test arguments, class `pronoun` in the key — needs a coreference component (GUM has gold);
- action families: **first-sense hypernym families do not work** (run 9b: depth 1 merges 87% unrelated verb pairs, catches 11% of synonym pairs), so the default family is the synset itself (depth 0). Instead of one family, an event posts under its verb's top-3 senses (run 9c: links 26% of synonym pairs, no hubs); a sense picker that beats first sense on SemCor (slot open, benchmark in place) would raise that;
- specific-entity seed: Wikidata-derived JSON (`SpecificIndex.load_seed`) not yet built; the full-Wikipedia index already keys articles by title;
- live-text parser (task delegated).
