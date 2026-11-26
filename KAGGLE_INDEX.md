# gramlab on Kaggle — notebook index and verdicts

Account: kaggle.com/tarasfur · Repo: github.com/furevych/gramlab · Results log: RESULTS.md (runs 0–17)

Titles on Kaggle carry the verdict as a prefix so the list is readable at a glance:

| Prefix | Meaning |
|---|---|
| ✗ TOY-WORLD | hand-written lexicon / regex constructions / synthetic sentences; nothing learned, nothing transferable |
| ✗ TINY-DATA | real external data, but a few dozen scored items — no conclusion possible |
| ✗ PRIMITIVE | the idea is fine, the implementation too crude to test it (dictionary + cue rules, sums of outer products) |
| ⟲ SUPERSEDED | sound at the time, replaced by a later run; kept for provenance |
| *(no prefix)* | live: part of the current pipeline or its benchmark |

## Live

| Notebook | Role | Where it is recorded |
|---|---|---|
| gramlab | stage-2/3 training of the 25M writer; its output holds shards, checkpoints, the Wikipedia index | RESULTS runs 1–3 |
| gramlab-map | meaning map v1 (negative: slots ≯ bag of words on STS-B) and v2 (positive: slot map beats bag of words for retrieval, recall@1 0.179 vs 0.117) | runs 4–5 |
| gramlab-hybrid | BM25 + map weight tuning on a dev/held-out split | run 6 |
| gramlab-fullwiki | BM25 over the lead paragraphs of all of English Wikipedia (4.96 M passages) | run 7 |
| gramlab-writer | Qwen2.5-0.5B + LoRA on retrieval dialogues | run 8 |
| gramlab-chat | ask() over the full index with either writer | — |
| gramlab-core | core-v1 scoreboard: linker, keys, typing, senses, coreference, live front end, retrieval, SQuAD | runs 9–16, audit |

## Failed or superseded (21 notebooks, Nov 15–23)

| Notebook | Verdict | One line |
|---|---|---|
| gramlab_shallow_roles | ✗ TOY-WORLD | 15 names, 4 verbs, regex constructions; 19 EWT sentences as "external" test |
| gramlab_voice_composition | ✗ TOY-WORLD | "Sam gave the bowl a rinse" variants over the same hand lexicon |
| gramlab_frozen_construction_transfer | ✗ TOY-WORLD | construction rules reused across 4 verbs; synthetic sentences only |
| gramlab_ordered_meaning_map | ✗ TOY-WORLD | first ordered key (PERSON→WASH→ANIMAL = id 10); 4×4×4 controlled combinations |
| gramlab_meaning_holdout | ✗ TOY-WORLD | 40 hand-written "holdout" sentences by the same author as the rules |
| gramlab_dependency_fallback | ✗ TOY-WORLD | dependency fallback vs guarded parser on a manual challenge set |
| gramlab_event_status | ✗ TOY-WORLD | n-gram classifiers on synthetic A/B order sentences |
| gramlab_predicate_roles | ✗ TOY-WORLD | same material, three feature sets |
| gramlab_temporal_retrieval | ✗ TOY-WORLD | bAbI task 14 (synthetic stories); 100% is the ceiling there and means nothing |
| gramlab_clause_scope | ✗ TOY-WORLD | which event a qualifier modifies, on two-event synthetic passages |
| gramlab_compositional_meaning_map | ✗ PRIMITIVE | sums of outer products of random codes — the "sum of indexes" idea that core-v1 replaced with the ordered key |
| gramlab_entity_accuracy_audit | ✗ PRIMITIVE | ATIS departure/arrival cities: dictionary + cue rules; 0.988 F1 on a closed vocabulary says nothing about open text |
| gramlab_learned_route_index | ✗ PRIMITIVE | same ATIS setup with a learned role classifier |
| gramlab_temporal_attachment | ✗ TINY-DATA | MATRES Platinum: 6–9 ordered pairs scored |
| gramlab_narrative_time | ✗ TINY-DATA | MATRES: 837 pairs, rule orders 3 of them |
| gramlab_external_place_identity | ✗ TINY-DATA | LGL gold-mention place linking with an independent gazetteer; a pilot, not continued |
| gramlab_project_progress | ⟲ SUPERSEDED | Nov 22 checkpoint of the toy prototype; superseded by core/README.md and RESULTS.md |
| gramlab-slot-diagnostics | ⟲ SUPERSEDED | timing/diagnostics of map v2 checkpoints; superseded by bench.run |
| gramlab-clean-slot-pilot | ⟲ SUPERSEDED | article-disjoint reranking pilot for map v2; superseded by gramlab-hybrid |
| gramlab-paired-validation | ⟲ SUPERSEDED | per-query validation of saved map/hybrid models; superseded by the audit of 2026-11-26 |
| gramlab_nextstage_kaggle | ⟲ SUPERSEDED | source audit scaffolding; nothing ran |
| gramlab-overnight | ⟲ SUPERSEDED | run 1 of the 25M writer; its output is still the input of gramlab |

Renaming changed the URL slugs (Kaggle derives them from the title): a renamed notebook is now at
`/code/tarasfur/<prefix>-<old-slug>`, e.g. `toy-world-gramlab-clause-scope`. Notebook inputs are attached by id and survive.

## What the failed group has in common

Everything in ✗ TOY-WORLD was built and scored on sentences written by the same hand that wrote the rules, so a 100% there
was never evidence. The ✗ PRIMITIVE group tested real questions (specific-entity identity, order-preserving composition)
with implementations too small to carry them; the questions remain open and are the subject of core-v2 (soft layer 0 +
role slots, RESULTS run 17). The ✗ TINY-DATA group picked external corpora whose scored slice was a few dozen items.
