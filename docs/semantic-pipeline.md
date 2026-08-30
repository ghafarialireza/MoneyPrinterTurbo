# The semantic visual pipeline

This document describes what this fork actually does, as built. It is the reference
for the part of the system that upstream MoneyPrinterTurbo does not have: choosing
footage that matches what the narration says, at the moment it says it, and proving
that it matches before spending the frame on it.

Last reviewed against the working tree on 2026-08-30, branch
`feat/visual-quality-pipeline`. Names in backticks are greppable — deliberately no
line numbers, because line numbers rot faster than the code they point at and a
stale pointer is worse than no pointer.

## What this fork changes

Upstream picks footage by handing a few search terms to a stock provider and cutting
whatever comes back from second zero, for a fixed number of seconds per clip. That
is fast and it is cheap, and it produces videos where the narration talks about one
thing while the picture shows another.

This fork replaces the middle of that process. The narration is broken into timed
units from the TTS provider's own word boundaries, those units are grouped into
semantic spans, each span becomes one or more visual beats with an explicit written
requirement, candidate clips are searched per beat, and a vision model is asked
whether a specific candidate actually shows what the requirement demands. Only a
clip that passes on evidence is used, and the exact in-clip window that passes is
what the renderer cuts. Everything else in the project — script generation, TTS,
subtitles, music, upload — is upstream's and is untouched by this work.

The legacy path is still present and still supported. It runs whenever smart
matching is off, keys are missing, or the source is not a searchable provider.

## The data chain

Four types carry a run from words to frames. They live in `app/models/schema.py`
and every one of them is a `pydantic.dataclasses.dataclass`, which matters: the
`Literal` type aliases are enforced at runtime, and `dataclasses.replace` revalidates.
Adding a new value for one of those fields means adding it to the `Literal` first,
or the pipeline raises `ValidationError` at the point of use rather than at import.

`TimedNarrationUnit` is one timing unit from the speech provider, aligned to an exact
span of narration text. It carries `timing_source` (`edge_tts_boundary`,
`azure_tts_boundary`, `whisper`, `estimated`) and `timing_quality` (`boundary`,
`speech_recognition`, `estimated`) so that everything downstream can tell measured
time from guessed time. This is the only place real timing enters the system; every
later stamp is derived from it.

`SemanticVisualSpan` is a contiguous run of narration that describes one visual idea.
Spans come from `llm.generate_semantic_visual_span_specs`. Its `grouping_source`
records how it was built: `llm` when grouping succeeded, `narration_slot_repaired`
when grouping failed but a second call rewrote each spoken line into a
camera-visible requirement, and `narration_slot_fallback` when both calls failed and
the "requirement" is the spoken narration itself. That last value is a warning label,
not a description of footage — text in that state must not be searched or verified as
though it described a picture.

`VisualBeat` is one shot. A span becomes one beat when it is short enough to hold a
single image, or several beats when it is long enough that a single static clip would
read as a stall. `_desired_visual_beat_shot_count` in `app/services/task.py` makes
that call against `_VISUAL_BEAT_PREFERRED_MIN_SECONDS` (2.0),
`_VISUAL_BEAT_PREFERRED_MAX_SECONDS` (5.0) and
`_VISUAL_BEAT_LONG_SPLIT_TARGET_SECONDS` (4.0). A beat shorter than
`VISUAL_BEAT_RAPID_CUT_SECONDS` (1.5) is flagged `rapid_cut`, because it reads as
punctuation rather than as a scene. Each beat carries its own
`visual_requirement` — capped at `_SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS` (240) — and
its own `search_queries`, and a `duration_policy` recording why its window is the
length it is.

`RenderSegment` is what the renderer consumes: a target window on the output timeline
paired with the source window inside the chosen asset. When render segments are
present they are authoritative — `app/services/video.py` sets `segments_authoritative`
from their presence and stops guessing clip lengths.

One consequence of the current grouping is worth stating plainly, because it drives
several design choices further down: `semantic_group_id` is assigned as
`group_position + 1`, so one span is one group. Combined with single-event
requirements and the 5-second preference, most beats end up being the only shot of
their own group. Any rescue that assumes a beat has same-group siblings to fall back
on is, in practice, dead code.

## A run, end to end

The order below is the order in `task._run_pipeline`, and it is the order in which
things can fail.

Script generation and terms come first, then `save_script_data`, then
`generate_audio`, then `generate_subtitle`. Only after audio exists is there any real
timing to work with, which is why every semantic stage sits behind it.

`build_narration_slots` turns the subtitle into narration lines with times.
`build_visual_slots` produces the legacy per-slot view that the old path needs.
`generate_semantic_visual_spans` groups the narration into spans, and
`build_visual_beats` cuts spans into shots. `persist_narration_timeline` writes the
timeline to disk at this point, before anything is searched, so that a run that dies
later can still be inspected.

`refine_split_span_shot_requirements` then gives each shot of a split span its own
requirement, rather than letting three shots share one sentence and compete for the
same footage. `generate_visual_beat_search_queries` writes the provider-facing search
strings, and `generate_visual_requirement_checklist` produces the
`VisualRequirementSpec` objects — the written list of facts a candidate must show.
The timeline is persisted a second time, now carrying queries and checklists.

`get_video_materials` runs selection. `generate_final_videos` renders.

## Grounding: search for what exists

The single most expensive mistake this pipeline can make is to ask a stock catalogue
for something nobody has filmed. A beat that searches for an unfindable concept walks
its entire escalation ladder, spends its whole analysis budget, and then fails — and
one failed beat used to lose the whole video.

So the requirement and the queries must name the closest concrete, filmable, commonly
shot scene that faithfully represents the narration's meaning. Not a transcription of
the words, and not something poetic. If the narration says a seed cracks open in the
dark, the requirement is a sprouting seedling, because that clip exists in quantity
and the other does not. Faithful to the meaning *and* filmable *and* common — all
three, or the beat is being set up to fail.

This principle is enforced in the prompts of `llm.generate_semantic_visual_span_specs`,
`llm.generate_visual_slot_queries`, `llm.generate_narration_visual_requirements` and
`llm.generate_alternative_visual_requirements`, where the multi-query fallback list is
ordered closest-findable-first.

It is deliberately **not** applied to `llm.generate_visual_requirement_specs`, the
decomposer that writes the checklist, nor to adjudication. Findability belongs
upstream, where the requirement is written. Verification stays literal — the moment a
verifier is taught to be generous, clips start passing on evidence they do not
contain, and the whole point of the pipeline is lost.

## Selection: a ladder, cheapest rung first

`material._download_videos_by_script_order_smart` fills one beat at a time. Each rung
below is tried only when everything above it has failed, and the ordering is by cost,
because the cheap rungs succeed often enough to make the expensive ones rare.

The first rung is the alternative phrasings the script stage already generated, tried
against the current provider. A beat that finds nothing usable is far more often
phrased badly than genuinely absent from the catalogue, and re-searching costs one
search plus one round of analysis. `max_query_variants_per_provider` caps this at
`_DEFAULT_MAX_QUERY_VARIANTS` (3) by default; the cap is what keeps the worst case
bounded.

The second rung is the next provider in the cascade. A different catalogue is
different evidence, so this rung is worth taking even when the first provider looked
confidently empty.

The third rung is `_rewrite_requirement_and_reselect`, which asks the LLM for a
different requirement — a different everyday category and a different visible moment,
with synonym swaps explicitly forbidden. This is the most expensive rung, because a
fresh requirement means a fresh search and a fresh page of analyses for an item that
has already proved difficult. It is also the rung the free rungs can substitute for,
so a video that has spent its per-video budget skips it deliberately and lets the
merge carry the window.

The fourth and last rung is the merge rescue, described below. It is free.

Cutting across all of this is the unrelated-footage abort in
`twelvelabs._footage_is_unrelated`. Between analysis batches, if nothing has been
accepted or even judged eligible, the best score so far is below
`_UNRELATED_FOOTAGE_SCORE_RATIO` (0.6) of the acceptance gate, and at least
`_MIN_UNRELATED_FOOTAGE_EVIDENCE` (3) clips have been analysed, the remaining batches
for that phrasing are abandoned. Before this existed, one beat of one render burned
roughly ninety analyses over thirteen minutes searching a catalogue that did not
contain its subject.

Two constraints on that abort are load-bearing. An API, quota, auth or malformed
response failure is never read as "unrelated" — that is a provider outage wearing the
costume of an empty catalogue, and confusing the two produces exactly the wrong
retry. And the abort skips only the remaining *phrasings* of the current provider; it
does not skip the provider cascade, and it does not skip the requirement rewrite,
because those two are the only escalations that change the evidence or the concept.

## Spend ceilings

Three ceilings bound cost, and they are checked in different places on purpose.

`analysis_budget_per_selection_round` bounds one beat's round, derived from the
candidate limit and `_DEFAULT_ANALYSIS_BUDGET_MULTIPLIER` (5).
`analysis_budget_per_video` bounds the whole render. `max_merged_beats_per_video`
bounds how many beats a single video may rescue by merging, so that a badly grounded
script degrades into a shorter cut list rather than into a slideshow.
`_effective_round_budget` narrows a round by whatever is left of the video's ceiling,
but never below one, so every beat still gets its first look on the first provider
even in a run that has already overspent. What gets cut is the cascade behind that
first look, not the look itself.

## Verification

`llm.adjudicate_visual_candidates` is where a candidate is accepted or refused. The
verdict rests on `decision` (`ACCEPT`, `REJECT`, `UNCERTAIN`) together with
`mandatory_fact_results`, cross-checked against the observed `source_statuses`: an
`ACCEPT` requires every mandatory fact to be `OBSERVED`. `CriticalFactStatus` allows
`OBSERVED`, `NOT_OBSERVED`, `CONTRADICTED`, `UNCERTAIN`, and
`ObservedActionDirectness` distinguishes `DIRECTLY_OBSERVED` from
`PARTIALLY_OBSERVED` and `INFERRED`.

`missing_or_contradictory_facts` is explanatory only. It used to be fatally
validated, which meant one stray, duplicated or non-string fact ID discarded a verdict
that had already passed every evidence check — enough to fail a beat, and through it
the video. It is now filtered to known IDs, order preserved, duplicates dropped, with
the ignored count logged at debug level. When filtering that list, filter to strings
*before* deduplicating, or an unhashable nested value re-raises the same
`TypeError` the filter exists to prevent.

## The merge rescue

When no provider, no phrasing and no rewritten requirement can fill a beat, the last
move before failure is to let a bordering shot cover the beat's window.
`_merge_unfillable_beats` does this. It costs nothing, because it reuses a clip that
has already been approved and downloaded, and it needs no new ceiling.

`_has_mergeable_neighbour` and `_adjacent_merge_survivors` find the candidates.
`_adjacent_merge_survivors` partitions them into same-group and adjacent-group and
returns `same_group or cross_group` — that partition is the whole safety of the
feature. Without it, a long clip from the next group would outbid a shorter clip that
was written for this exact moment, purely on the grounds of having more footage
spare.

Which of two qualifying neighbours absorbs is decided by `_pick_merge_survivor`,
which ranks by the room left in each asset and prefers the previous side unless a
challenger is roomier by more than `_MERGE_TIME_TOLERANCE_SECONDS` (1e-6). The
tolerance is not decoration. Each side's merged window has the same length, but one
is measured as this beat's end minus the previous beat's start and the other as the
next beat's end minus this beat's start, so the two headrooms routinely differ by
about 1e-15 seconds. A raw comparison let that rounding noise choose the survivor,
and the later shot won every genuinely tied contest. Preferring the previous side is
what keeps the rewritten timeline in index order, because a survivor keeps its own
index and absorbing forwards is what leaves the indexes ascending.

`_merged_visual_beat` derives the resulting `duration_policy`, and that label is
sticky: a survivor can absorb on both sides, so the label names the weakest claim any
absorption made. Without stickiness, a later same-group merge would silently upgrade
the label of a beat that is still covering a neighbour's window.
`validate_merged_beat_timeline` then proves the rewritten timeline is still
monotonic, gap-free and overlap-free.

`smart_material_cross_group_merge` gates the cross-group half of this, defaulting to
on. It is not exposed in the WebUI, and neither is any other `smart_material_*` knob;
they are config-file settings by design.

## Providers

Two searchable stock providers remain: Pexels, then Pixabay.
`_STOCK_VIDEO_PROVIDER_API_KEYS` maps each to its config key and
`_SMART_PROVIDER_CASCADE_ORDER` fixes the order. These two constants in
`app/services/material.py` are the single source of truth — a future provider change
happens there and nowhere else, and `smart_provider_chain` and
`provider_has_api_key` read from them.

Coverr was removed in full because it is a paid API. Two guards remain on purpose:
the CLI rejects `--video-source coverr`, and `smart_provider_chain("coverr")` returns
an empty list. Both are regression tests against quietly reintroducing it. Local and
generated sources bypass smart matching entirely, which
`supports_smart_visual_matching` decides.

TwelveLabs supplies the vision analysis: Marengo for retrieval and reranking
(`DEFAULT_MARENGO_MODEL`, `marengo3.0`) and Pegasus for candidate evaluation and
temporal segmentation (`DEFAULT_PEGASUS_MODEL`, `pegasus1.5`), requiring SDK
`MINIMUM_SDK_VERSION` (1.3.0) or newer. It is optional, opt-in, and fails closed:
with `twelvelabs_clip_qa_fail_closed` on, a beat whose footage cannot be verified
fails rather than passing unchecked. Segmentation runs on the winner only — candidates
are never segmented, because that would multiply the analysis cost by the candidate
count for information used once.

## Configuration

Every key below is read by the code and documented in `config.example.toml`.

| Key | Meaning |
| --- | --- |
| `pexels_api_keys`, `pixabay_api_keys` | stock provider credentials |
| `twelvelabs_api_keys` | vision analysis credentials |
| `twelvelabs_clip_qa` | master switch for candidate verification |
| `twelvelabs_clip_qa_fail_closed` | unverifiable footage fails instead of passing |
| `twelvelabs_clip_qa_min_score` | acceptance gate, default 0.70 |
| `twelvelabs_strong_early_stop_score` | stop searching on a clearly strong clip, default 0.90 |
| `twelvelabs_rerank_terms` | Marengo reranking of search terms |
| `twelvelabs_max_candidates_per_slot` | candidates considered per beat, default 15 |
| `twelvelabs_candidate_batch_size` | analyses per batch, default 5 |
| `twelvelabs_candidate_concurrency` | parallel analyses, default 5 |
| `twelvelabs_preferred_max_source_duration` | preferred asset length ceiling, default 30.0s |
| `twelvelabs_marengo_model` | retrieval model override |
| `smart_material_provider_cascade` | allow falling through to the next provider |
| `smart_material_max_query_variants` | phrasings tried per provider, default 3 |
| `smart_material_requirement_rewrite` | allow the expensive rewrite rung |
| `smart_material_cross_group_merge` | allow an adjacent-group neighbour to absorb |
| `smart_material_max_merged_beats` | merge ceiling per video |
| `smart_material_max_analyzed_candidates_per_round` | per-beat analysis ceiling |
| `smart_material_max_analyzed_candidates_per_video` | per-render analysis ceiling |

Do not lower `twelvelabs_max_candidates_per_slot` or
`twelvelabs_strong_early_stop_score` to make a run cheaper. They are quality knobs,
and lowering them trades away the thing the pipeline exists to provide.

## Provenance

Every decision a run makes is recorded, so a bad video can be explained after the
fact without re-running it. The codes written by selection are
`NO_CANDIDATES`, `DECOMPOSITION_FAILED`, `ANALYSIS_BUDGET_EXHAUSTED`,
`VIDEO_ANALYSIS_BUDGET_EXHAUSTED`, `REQUIREMENT_REWRITTEN`,
`REQUIREMENT_REWRITE_UNAVAILABLE`, `WINNER_DOWNLOAD_SUBSTITUTED`,
`FAIL_OPEN_FALLBACK`, `UNFILLABLE_BEAT_MERGED`, `MERGE_CEILING_REACHED`,
`MERGE_NEIGHBOUR_UNAVAILABLE`, `MERGE_ANALYSIS_BUDGET_EXHAUSTED`,
`MERGE_RESELECTION_FAILED` and `MERGE_SEGMENTATION_FAILED`.

Merge records additionally carry `merge_scope` (`same_semantic_group` or
`adjacent_semantic_group`) and `merge_fill` (`neighbour_window_extended` for the free
path, `fresh_selection_round` for the paid one). API keys and sensitive payloads are
never written to provenance or logs; `_redact_secret` and `_redact_request_error`
exist for that.

## Invariants

These are the rules that later work must not break. Each of them was learned by
breaking it.

Exact beat timing survives selection. A beat's target window and the chosen
`source_start`/`source_end` are preserved end to end, and the timeline stays
monotonic, gap-free and overlap-free. After a merge, `validate_merged_beat_timeline`
proves it rather than assuming it.

Never silently substitute `source_start = 0` after a smart-selection failure. That is
the legacy behaviour, and applying it to a beat the pipeline analysed means paying for
semantic precision and then discarding it — the frames the viewer sees become the
frames the pipeline explicitly rejected.

Use the user-selected LLM provider. Never hardcode a specific vendor anywhere in the
call path, and never hardcode an output language into a reusable prompt. The English
search strings sent to stock providers are a provider mechanism, not the output
language of the video.

Keep TwelveLabs optional and fail-closed, keep the legacy renderer path working, and
keep provider-aware deduplication, caches and provenance intact.

In this codebase, "sibling" means *same semantic group*. Use "neighbour" when either
group may be the absorber. The distinction is load-bearing in the merge code and
mixing the two words is how the rescue became dead code the first time.

## Testing

The suite lives under `test/` and runs with `pytest test -q`. Ruff runs on its default
rule set (`E4`, `E7`, `E9`, `F`) — `pyproject.toml` configures nothing but a per-file
`E402` ignore for `webui/Main.py` — so line length is not enforced; run
`ruff check app test cli.py webui docs/skill`.

Three patching rules exist because breaking them bills a live account. Any test that
reaches `task.start` or `_run_pipeline` must patch both the checklist gate and
`generate_visual_beat_search_queries`. Any test that lets a beat fail must neutralise
`llm.generate_alternative_visual_requirements`, or the rewrite rung fires for real.
Any test that reaches `llm.adjudicate_visual_candidates` must point
`_semantic_adjudication_cache_dir` at a temporary directory.

## Deliberately not done

There is no image search anywhere — only video. There is no cross-provider
competition within a single beat: providers are tried in sequence until one produces
an acceptable clip, so the best clip for a beat may sit in a catalogue the run never
reached. Candidates are not segmented, only winners. And the decomposer's checklist
stays literal, which means a requirement that was written badly cannot be rescued by
a lenient verifier — it has to be rewritten instead.

`docs/semantic-stage-s5-plan.md` is a dated design record from 2026-08-23, kept for
history. Where it disagrees with this document, this document is current.
