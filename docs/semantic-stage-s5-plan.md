# Semantic Stage S5 — Variable-duration `RenderSegment` + renderer integration

> **Historical record. Do not read this as current documentation.**
>
> This is the review and plan as written on 2026-08-23, kept unedited because it
> records why the decisions below were taken. Everything it plans has since been
> built, and several of its statements are now false: variable-duration
> `RenderSegment` rendering is wired and authoritative, and Coverr — which this
> document still inventories as a live provider — was removed in full as a paid API,
> leaving `pexels -> pixabay`. The status line and file counts below describe a
> working tree that no longer exists.
>
> For the pipeline as it is now, see [`semantic-pipeline.md`](semantic-pipeline.md).
> Where the two disagree, that document is current and this one is history.

Status *as of 2026-08-23*: review complete, implementation not started.
Branch: `feat/visual-quality-pipeline` (uncommitted, 15 modified files, 0 untracked).
Reviewed against working tree on 2026-08-23.

---

## Part 1 — Review of the current state

### 1.1 What was verified in this session, and what was not

Verified directly:

- Branch is `feat/visual-quality-pipeline`; `git status --porcelain` shows exactly 15 modified files and **zero** untracked files; diffstat matches the handoff (`15 files changed, 7512 insertions(+), 502 deletions(-)`); `git diff --check` passes.
- `python -m compileall app test` passes.
- Every code claim below was re-read from the working tree, with line numbers.

Not verified: the pytest suite (`325 passed, 6 skipped`) and Ruff. The Linux sandbox has no project dependencies installed and PyPI is unreachable from it (proxy returns 403), so `pytest` and `ruff` cannot run there, and the repo `.venv` is a Windows venv. **Those two numbers remain the handoff's claim, not a re-verified fact.** They must be re-run on Windows before S5 is considered validated.

### 1.2 The headline finding: S5 is blocked by wiring, not only by the renderer

The handoff frames S5 as "the renderer is still legacy fixed-duration." That is true but it is the second half of the problem. The first half is that `VisualBeat` never reaches material selection either.

The beat-aware selector already exists and is complete:

- `app/services/material.py:1461` — `_download_videos_by_script_order_smart(..., visual_slots=None, visual_beats=None, ...)` accepts either, rejects both (`:1476`), picks beats when present (`:1478`), and labels logs/errors accordingly (`:1479`).
- `app/services/material.py:1757` — when `visual_beats is not None` it writes `visual_beat_index` and `semantic_group_id` into the persisted source record.

But nothing in production ever passes beats:

- `app/services/material.py:1303` — `download_videos(...)` has no `visual_beats` parameter at all.
- `app/services/material.py:1810` — `_download_videos_by_script_order(...)` has no `visual_beats` parameter; its dispatch at `:1837` forwards only `visual_slots=visual_slots`.
- `app/services/task.py:2776` — `get_video_materials(..., visual_slots=visual_slots)`; beats are not passed.

So beats are built (`task.py:2706`), timeline-validated (`task.py:1246`), query-generated (`task.py:2726`) and persisted (`task.py:2738`) — and then dropped. The in-code note is honest about it: *"S4 query metadata is additive until the variable renderer path is approved"* (`task.py:2728-2729`). Net effect today: `VisualBeat` is a persisted debug artifact. Every frame the user sees still comes from fixed 5-second `VisualSlot`s.

This matters for sequencing: converting `combine_videos()` first would produce a renderer capable of variable segments that is fed nothing but uniform slots. The wiring and the renderer have to land together.

### 1.3 What the renderer already gets right

`combine_videos()` is not as far from S5 as the handoff implies. It already has a semantic branch:

- `video.py:583-584` — `source_ranges` length must equal `video_paths` length.
- `video.py:598-630` — when `source_ranges` is supplied, the per-file chopping loop is skipped entirely and exactly one `SubClippedVideoClip` per input file is produced from the selected `[start, end)`.
- `video.py:682-683` — playback speed is applied *before* transitions, so a 1-second fade stays 1 second at 0.5x or 2x. The comment explaining why is correct.
- `app/services/utils/video_effects.py` — every transition (`fadein`, `fadeout`, `slidein`, `slideout`, `zoomin`, `zoomout`) preserves clip duration. No transition silently shortens a segment.

So S5 is a **contract upgrade** — replace an anonymous `list[tuple[float, float]]` with typed segments that also carry authoritative *target* timing — not a renderer rewrite.

### 1.4 Renderer failure modes that variable-duration beats will hit

These are the concrete reasons the current renderer cannot be handed beats as-is.

**(a) Hard truncation at `max_clip_duration` — `video.py:740-741`**

```python
if clip.duration > max_clip_duration:
    clip = clip.subclipped(0, max_clip_duration)
```

This runs unconditionally, including in the `source_ranges` branch. `max_clip_duration` is `params.video_clip_duration` (`task.py:2099`), default 5, minimum 1. Any beat longer than the user's setting is silently cut, and because the concat is sequential, every downstream beat shifts earlier — narration and visuals desynchronise progressively for the rest of the video. With `video_clip_duration = 3` and 4-second beats, roughly a quarter of the timeline evaporates.

**(b) Early exit drops trailing segments — `video.py:664-666`**

```python
for i, subclipped_item in enumerate(subclipped_items):
    if video_duration >= required_video_duration:
        break
```

`required_video_duration = audio_duration + 0.1` (`video.py:96-104`, `_VIDEO_DURATION_SAFETY_MARGIN = 0.1` at `:77`). With authoritative segments the loop must process *all* segments regardless of accumulated duration; any `break` means the last beat is never rendered. Today this is masked because slot durations sum to less than `audio + 0.1`; with beats that sum to exactly `audio_duration`, it is one rounding error away from firing.

**(c) Duplicate-clip padding — `video.py:772-787`**

```python
if video_duration < required_video_duration:
    for clip in itertools.cycle(base_clips):
        ...
        processed_clips.append(clip)
```

Authoritative beats sum to exactly `audio_duration`, which is always less than `audio_duration + 0.1`, so **this loop always fires** and appends a duplicate of an earlier segment. The damage is bounded — `concat_video_clips_with_ffmpeg(..., max_duration=audio_duration)` emits `-t audio_duration` (`video.py:365-366`), so the output is trimmed at the audio end — but any frame-quantization shortfall in the written temp clips becomes a visible flash of an unrelated shot in the final frames. Small, real, and trivially avoidable.

**(d) Reordering — `video.py:656-659`**

`_prioritize_unique_source_clips()` is called unconditionally. It is a no-op unless `concat_mode == random` (`video.py:136-137`), and `task.py:2077-2078` forces `VideoConcatMode.sequential` whenever `match_materials_to_script` is on — so this is currently harmless. It is still a landmine: any future caller passing segments with `random` mode would shuffle a timeline whose order is semantically load-bearing. Make it structurally impossible, not accidentally safe.

**(e) Frame-quantization drift at `fps = 30` — `video.py:744-751`**

Each segment is written to its own temp file at module-level `fps`. A 3.267-second segment cannot be represented exactly in whole frames; each written duration lands on a frame boundary. Per segment the error is under 1/30 s, but errors accumulate across the concatenation, and unlike the current fixed-slot path there is nothing to absorb them. Twenty beats with consistent one-directional rounding is over half a second of accumulated visual/narration offset by the end — which defeats the entire point of S1–S4.

The fix is cheap and exact: derive each segment's frame count from the *cumulative* target boundaries rather than from its own duration.

```python
frames = round(target_end * fps) - round(target_start * fps)
```

Boundary error then stays at zero by construction: the start of segment *n+1* is exactly the end of segment *n*, and the final boundary is exactly `round(audio_duration * fps)`.

**(f) Fixed 1-second transitions — `video.py:714-738`**

The duration is the literal `1` in all seven branches. `VisualBeat.rapid_cut` beats are as short as 1.5 s (`task.py:74`, `_VISUAL_BEAT_RAPID_CUT_SECONDS = 1.5`); a 1-second fade consumes two-thirds of such a shot. `zoomin`/`zoomout` are worse: they ramp across the whole clip to `_ZOOM_MAX_SCALE = 1.2` regardless.

### 1.5 A live bug found during the review, independent of S5

**Correction to an earlier draft of this review.** I first reported this as a *missing API key* asymmetry — that `twelvelabs_clip_qa = true` with empty `twelvelabs_api_keys` would crash at the video stage. That is wrong. The preflight at `task.py:2508-2520` calls `validate_smart_visual_matching_configuration()` (`twelvelabs.py:333-353`), which fails the task early and clearly on a missing key or an outdated SDK. The key case is covered.

The real asymmetry is by **provider**, and it is worse than the one I originally described:

- Smart material selection is provider-agnostic. `download_videos()` maps `source` to Pexels, Pixabay, or Coverr (`material.py:1315-1322`), and the smart dispatch at `material.py:1837` gates only on `visual_slots` plus `is_smart_visual_matching_enabled()` — the provider is never checked. `ordered_timeline_enabled` covers every non-local source (`task.py:2579-2581`).
- The renderer's source-range load is Pexels-only: `task.py:2799-2803` requires `params.video_source == "pexels"`.
- The preflight is Pexels-only too: `task.py:2510`.

So with `video_source = "pixabay"` (or `"coverr"`), `match_materials_to_script` on, `twelvelabs_clip_qa` on and keys present: candidate evaluation and temporal segmentation both run, TwelveLabs is billed, the exact source window is chosen and persisted to `script.json` — and then the renderer never asks for it. `source_ranges` stays `None`, `combine_videos()` takes the legacy chopping loop (`video.py:632-654`), and every clip is cut **from second 0** for `max_clip_duration`. Paid-for semantic precision is silently discarded and the frames the user sees are the ones the pipeline explicitly rejected.

That is a direct violation of the spirit of "never silently substitute `source_start = 0`" — the substitution just happens one layer lower than where the rule is enforced. On the same configuration the preflight is also skipped, so a missing SDK surfaces as a mid-run material failure instead of an upfront message.

Two ways to fix, and this needs a decision (see Step 3): either extend the renderer/preflight gates to every search provider (Pixabay and Coverr get real support), or refuse smart selection outside Pexels so nothing is billed for output that will be thrown away. Extending is the better product answer; refusing is the smaller diff.

### 1.6 Secondary findings

- **Search filtering already scales with item duration.** `material.py:1561` uses `minimum_duration=math.ceil(max(4.0, required_source_duration))` per item, derived from `visual_item.duration` — not from `max_clip_duration`. Beats need no change here. Consequence to note: the search cache key includes `minimum_duration` (`material.py:1222-1227`), so sibling beats that share a query but differ in duration will miss each other's cache entry and issue separate provider searches.
- **`max_clip_duration` is effectively unused inside the smart selector** (`material.py:1469` is a passed-through parameter). Confirms the smart path is already duration-agnostic.
- **`VisualBeat` already satisfies the selector's structural contract.** `OrderedVisualItem` (`material.py:26-32`) requires `index`, `duration`, `visual_requirement`, `search_queries` — `VisualBeat` (`schema.py:255-275`) has all four. No adapter needed.
- **Beat count exceeds slot count**, so TwelveLabs spend and latency rise roughly in proportion. Segmentation stays winner-only (`material.py:1723`), so the "no extra candidate-analysis calls" constraint holds, but per-beat candidate evaluation is per-item by construction. Cost scales with the number of shots, and that is inherent to the feature, not a regression.
- **`segment_winner()` returns `None` when the winner's source is shorter than 4.0 s** (`twelvelabs.py:1722-1788`), and search already filters below 4.0 s, so short beats are safe. But a 1.5 s beat asking for a 1.5 s window inside a 4 s clip gives TwelveLabs very little to segment; expect lower-confidence picks on rapid-cut beats.
- **Unsplittable long spans.** `_choose_balanced_visual_beat_cuts()` (`task.py:1016`) only cuts at real timing-unit starts and returns `[]` when no valid cut exists, so a coarse span with no interior unit boundaries yields a single beat well over 5 s. The renderer must render it (that is the point of variable duration), but a 12-second single shot in a Short is a quality problem worth its own follow-up.
- **Source-range rounding.** `_material_source_record()` rounds `source_start_time`/`source_end_time` to 3 decimals, and `_normalize_selected_source_range()` (`material.py:684-715`) trims to exactly `required_source_duration`. So `(end - start) / speed` is within ~0.0005 s of the beat duration — close, but not identical. **Target timing must come from the beat, never from source-range arithmetic.** The source range answers *where in the clip*; the beat answers *how long on the timeline*.
- **Confirmed non-issues.** Transitions preserve duration. `exact_audio_duration` (`voice.py:2385-2413`) and `combine_videos`'s own read (`video.py:557`) both come from `AudioFileClip().duration` on the same file, so they agree exactly. `save_video()` filenames are `vid-{md5(url_without_query)}.mp4`, unique per rendition URL, and cross-item dedupe (`used_asset_identities`, `used_urls`) guarantees one file per beat today.

### 1.7 Decisions taken (2026-08-23)

1. **Beats replace slots in the smart path.** When `match_materials_to_script` and TwelveLabs smart matching are both active, material selection *and* rendering run on `VisualBeat`. No parallel opt-in flag. `VisualSlot` remains the only path when smart matching is off.
2. **Duration follows content need.** Each segment renders for exactly as long as its beat requires — longer or shorter. `params.video_clip_duration` no longer caps the beat path; it continues to govern the legacy path unchanged.
3. **Transition duration scales with segment length.** `min(1.0, 0.25 × segment_duration)`, floored at 0.1 s. Note this is a no-op for any clip of 4 s or more, so legacy behaviour at the default 5 s setting is bit-identical; it only changes clips shorter than 4 s, which is exactly the broken case.

---

## Part 2 — Plan

### Goal

Make the rendered timeline reproduce the S1–S4 semantic timeline exactly: when smart visual matching is active, every `VisualBeat` becomes one rendered segment of its own variable length, at its own timeline position, cut from its own verified source window — with zero cumulative drift against the narration audio.

### Done when

1. A typed `RenderSegment` model exists in `app/models/schema.py` and is the only contract `combine_videos()` accepts for semantic rendering.
2. With `match_materials_to_script` + TwelveLabs smart matching enabled, `get_video_materials()` selects materials from `visual_beats`, and `material_sources` records in `script.json` carry `visual_beat_index` for every beat.
3. For a run of N beats, the combined video contains exactly N segments in beat order, and for every segment *n*, the rendered boundary equals `round(beat[n].end_time × fps) / fps` — exact, not approximate.
4. No segment is truncated by `max_clip_duration`; a beat longer than `params.video_clip_duration` renders at full length.
5. The duplicate-clip padding loop and the early-exit `break` never execute on the segment path; the combined video's final frame is at the audio end with no repeated shot.
6. Transition duration on a 1.5 s segment is 0.375 s, and on a 5 s segment is 1.0 s.
7. `params.video_clip_duration`, `video_concat_mode`, and the whole legacy path behave exactly as before when smart matching is off — proven by the existing legacy tests passing unmodified.
8. Windows validation is green: full relevant suite, Ruff, `compileall`, `git diff --check`, secret scan.

### Constraints

Carried forward from the handoff and treated as non-negotiable:

- Preserve every current uncommitted change. No reset, checkout, clean, stash-drop, or overwrite.
- Do not redesign candidate selection, semantic verification, or provider fallback.
- Use the user-selected LLM provider. Never hardcode Gemini, OpenAI, or DeepSeek.
- TwelveLabs stays optional, opt-in, and fail-closed.
- No additional candidate-analysis calls. Temporal segmentation stays winner-only.
- Preserve exact `VisualBeat` target timing and the selected `source_start`/`source_end`.
- Preserve playback-speed source-duration math and short-beat behaviour.
- Timeline stays monotonic, gap-free, overlap-free, through the audio end.
- Preserve provider-aware dedupe, sibling-beat behaviour, caches, provenance.
- Keep the legacy renderer path intact when semantic/ordered smart matching is off.
- Do not couple internal timing to visible subtitle rendering.
- Never silently substitute `source_start = 0` after a smart-selection failure. (The existing explicit fail-open at `material.py:1739-1751` is opt-in and logged — that stays as-is.)
- No API keys or sensitive payloads in persisted provenance or logs.

### Steps

**Step 1 — Add `RenderSegment` to `app/models/schema.py`.**

Fields: `index`, `file_path`, `source_start`, `source_end`, `target_start`, `target_end`, `target_duration`, `playback_speed`, `visual_beat_index`, `semantic_group_id`. Pydantic dataclass, same `_Config` as the neighbouring models.

Result: importable model with validation.
Confirm: a unit test constructs a valid segment and asserts that a negative `source_start`, a non-positive `target_duration`, and `source_end <= source_start` are all rejected.

**Step 2 — Build and validate segments in `task.py`, replacing the `source_ranges` load.**

Add `material.load_render_segments(task_id, video_paths, visual_beats, clip_speed)`. Join persisted `material_sources` to beats on `visual_beat_index` — **not** on `local_file` basename, which is only unambiguous while cross-beat dedupe holds. Take `target_start`/`target_end`/`target_duration` from the beat (authoritative), `source_start`/`source_end` from the record. Raise a `ValueError` with a specific message on: missing record for a beat, duplicate `visual_beat_index`, count mismatch against `video_paths`, non-finite values, or `abs((source_end - source_start) / speed - target_duration) > 0.02`.

Then validate the assembled list independently of the beat validator: `segments[0].target_start == 0.0`, contiguity within 1e-6, `segments[-1].target_end == audio_duration`, indexes sequential.

Result: a validated `list[RenderSegment]` or a clean task failure at the `"video"` stage.
Confirm: unit tests for each raise path, plus one happy path asserting the segment list matches a known beat/record fixture exactly.

**Step 3 — Resolve the provider asymmetry (§1.5) before wiring beats. Needs a decision from ALIREZA.**

Pick one:

- **(a) Extend** — make the preflight (`task.py:2510`) and the segment load (`task.py:2800`) accept any search provider (`pexels`, `pixabay`, `coverr`) instead of Pexels only. Pixabay and Coverr then get real semantic rendering.
- **(b) Refuse** — make smart selection require `video_source == "pexels"`, so non-Pexels runs never bill TwelveLabs for output the renderer discards.

Either way, all three gates (preflight, selection, render) must derive from one shared predicate rather than three hand-written conditions, because S5 adds a fourth caller.

Result: no configuration exists in which TwelveLabs is billed and its chosen window is then ignored.
Confirm: a test per provider asserting that either segments are loaded and used, or smart selection never runs — never the middle state. For (a), also assert that a Pixabay run persists `material_sources` and that `combine_videos` receives the segments.

This step is deliberately early: it is the cheapest way to find out whether any existing test depends on the current asymmetry.

**Step 4 — Thread `visual_beats` through material selection.**

Add `visual_beats` to `download_videos()` (`material.py:1303`) and `_download_videos_by_script_order()` (`material.py:1810`), forward to the existing `_download_videos_by_script_order_smart(visual_beats=...)`. Pass beats — not slots — from `get_video_materials()` and `task.py:2776` when smart matching is active. Keep `visual_slots` as the parameter used when smart matching is off. The mutual-exclusion guard at `material.py:1476` stays as the safety net.

Result: `material_sources` records contain `visual_beat_index` and `semantic_group_id` on a smart run; one downloaded file per beat.
Confirm: extend `test/services/test_twelvelabs_smart.py` — assert the selector received `visual_beats`, that per-item `required_target_duration` equals the beat duration (not `video_clip_duration`), and that every persisted record has a `visual_beat_index`.

**Step 5 — Add the segments-authoritative branch to `combine_videos()`.**

Add `segments: list[RenderSegment] | None = None`, mutually exclusive with `source_ranges` (keep `source_ranges` working so the API contract and existing tests do not break). When `segments` is supplied:

- Skip `_prioritize_unique_source_clips()` entirely (§1.4d).
- Process every segment; no `required_video_duration` `break` (§1.4b).
- Compute `frames_n = round(target_end × fps) - round(target_start × fps)` and write each temp clip to exactly that frame count, so cumulative boundary error is zero (§1.4e).
- Do not apply the `max_clip_duration` truncation (§1.4a).
- Scale transition duration to `max(0.1, min(1.0, 0.25 × segment_duration))` (§1.4f).
- Skip the `itertools.cycle` padding. Absorb the 0.1 s safety margin by extending **only the final segment**: take it from unused source headroom if physically available, otherwise hold the last frame. Never repeat an earlier shot (§1.4c).

Result: N segments in, N temp clips out, boundaries frame-exact, total ≥ audio duration.
Confirm: extend the existing `_capture_source_ranges_for_clip_speed` harness in `test/services/test_video.py` (its fakes and patch set already cover `AudioFileClip`, `_open_video_clip_quietly`, `_write_videofile_with_codec_fallback`, `_prioritize_unique_source_clips`, `concat_video_clips_with_ffmpeg`, `delete_files`). Assert: written durations equal frame-quantized beat durations; cumulative boundaries match `round(t × 30) / 30` exactly for a beat list with awkward fractions; a 7-second beat with `video_clip_duration = 3` is written at 7 s; `_prioritize_unique_source_clips` is never called; the concat list has exactly N entries.

**Step 6 — Human decision point: aesthetic review of a real render.**

Everything above is mechanically verifiable. Whether beat-paced cutting actually *looks* better on a real Short is not. Produce one render on the beat path and one on the slot path from the same script and compare before this is considered done.

Result: an explicit accept/reject on the visual outcome.
Confirm: ALIREZA watches both.

**Step 7 — Validate on Windows and report honestly.**

Run the relevant pytest suite, Ruff, `compileall`, `git diff --check`, and the secret scan in the Windows venv. Report pass/fail per check with real numbers. Do not restate the previous session's numbers as if re-verified.

### Risks

- **Frame quantization is subtler than one line of arithmetic.** MoviePy's writer may not honour an exact frame count for every codec path. Mitigation: assert on the *written* duration in the test harness rather than trusting the requested value; if the writer rounds independently, switch to requesting durations already snapped to `1/fps` and verify the sum.
- **Final-segment extension can run out of source.** If the winner's clip has no headroom past `source_end`, holding the last frame is the fallback — visible as a ~0.1 s freeze at the very end. Acceptable; the alternative (repeating a shot) is worse. If it looks bad in the Step 6 review, the next option is trimming the audio-side margin instead.
- **Existing tests may encode the old truncation or padding behaviour.** Mitigation: Step 3 and Step 5 both run the full suite immediately; any legacy test that fails must be read carefully before being changed — a failing legacy test may be reporting a real regression in the untouched path, which is a stop condition, not a test to update.
- **TwelveLabs cost rises with beat count**, and sibling beats with differing durations miss the search cache (§1.6). Mitigation: log per-run segmentation call count and total analyzed source seconds (the counters already exist at `material.py:1507-1509`) so the increase is measurable rather than a surprise on the invoice.
- **Rapid-cut beats get weak segmentation.** A 1.5 s window inside a 4 s clip gives the segmenter little to work with. Mitigation: none in S5 — record it and revisit if Step 6 shows visibly poor picks on short beats.
- **Unsplittable long spans render as very long single shots** (§1.6). Out of scope here; flagged for follow-up.

### Abort conditions

- If any currently-passing test in the legacy path fails and the cause is a real behaviour change rather than an assertion about removed behaviour — stop, report, do not adjust the test.
- If frame-exact boundaries cannot be achieved through the MoviePy writer, stop before wiring beats into production. Drifting variable-duration segments are worse than correct fixed slots.
- If the diff starts requiring changes inside candidate selection, semantic verification, or provider fallback, stop — that is a constraint violation and means the design is wrong.

### Out of scope

- Changing how beats are built or how long they are (`task.py:978-1078` stays as-is). Content-driven duration is already implemented there; S5 only stops the renderer from overriding it.
- Splitting unsplittable long spans (§1.6).
- Subtitle timing and rendering — explicitly decoupled from internal timing.
- Any change to candidate selection, scoring, gates, adjudication, caches, or provenance.
- Commit, push, merge, or deploy.
- Making `VisualBeat` the default when smart matching is off.

---

## Part 3 — Provider and API inventory, as actually built

### 3.1 Where video can come from — five sources, three of them searchable APIs

`params.video_source` accepts:

| Source | What it is | Search API | Semantic path today |
|---|---|---|---|
| `pexels` (default) | stock video search, `https://api.pexels.com/v1/videos/search` (`material.py:782`) | yes | full — selection + source ranges + render |
| `pixabay` | stock video search, `https://pixabay.com/api/videos/` (`material.py:870`) | yes | selection only; ranges discarded (§1.5) |
| `coverr` | stock video search, `https://api.coverr.co/videos` (`material.py:1028`) | yes | selection only; ranges discarded (§1.5) |
| `local` | files the user supplies | n/a | none — ordered matching is skipped entirely (`task.py:2580`) |
| `loomloom` | AI video *generation* marketplace, `https://loomloom.shengsuanyun.com/loom/v1` (`loomloom.py:23`) | n/a | none — clips arrive pre-made (`task.py:1917-1967`) |

So: three searchable stock-video providers, one local source, one generative source. TwelveLabs is **not** a source — it is the judge.

### 3.2 There is no image search anywhere in the codebase

Worth stating plainly because it was assumed to exist: no `search_images`, no Pexels `/v1/search` photo call, no Pixabay image endpoint. Every provider call hits a **video** endpoint only. `video.py` can render stills (`_open_image_clip_with_fallback`, `_sanitize_image_file`), but images can only enter through the local material directory — never from an API. If still-image beats are wanted (a fair idea for a beat where no stock footage matches), that is new work, not a switch to flip.

### 3.3 What actually happens for one beat, end to end

1. The user-selected LLM writes one search query per beat from that beat's `visual_requirement` (`task.py:1478-1526`). Sibling beats in the same semantic group share the group's query.
2. The provider is searched with a per-item duration floor: `minimum_duration = ceil(max(4.0, beat_duration × clip_speed))` (`material.py:1561`). Results are cached; the cache key includes that floor (`material.py:1222-1227`).
3. Candidates are filtered before any paid analysis: HTTPS only, no credentials in the URL, resolvable `(provider, provider_asset_id)` identity, duration at or above the floor, and not already used by an earlier beat (`material.py:630-660`).
4. **TwelveLabs Pegasus** (`pegasus1.5`, `twelvelabs.py:1118`) watches each surviving candidate — by direct URL where the provider allows it, otherwise via a temporary uploaded asset that is deleted afterwards (`_wait_for_asset_ready`, `_delete_temporary_asset`, `twelvelabs.py:1074-1100`) — and returns structured observations against a typed requirement spec, not a free-text opinion (`_candidate_prompt`, `_parse_candidate_response`).
5. Deterministic mandatory gates run first on those observations; only gate-passers reach bounded LLM adjudication; a verified `ACCEPT` allows early stop (`select_best_candidate`, `twelvelabs.py:1258`). Evaluations are cached per `(candidate, requirement, model)`.
6. **Winner only**, a second Pegasus call asks *where inside this clip* the required action happens, and returns `source_start`/`source_end` sized to `beat_duration × clip_speed` (`segment_winner`, `twelvelabs.py:1722`). Losers are never segmented — that is the cost control.
7. The window is centred and trimmed to exactly the required length (`_normalize_selected_source_range`, `material.py:684-715`) and persisted with provenance to `script.json`.
8. **The renderer then ignores step 6 unless the source is Pexels, and ignores the beat's timing in all cases** — that is S5.

A second, older TwelveLabs surface also exists: `analyze_clip` / `evaluate_clip_match` (`twelvelabs.py:1791-1900`), the per-clip QA path. And **Marengo** (`marengo3.0`) is used only for text embeddings to rerank search terms by subject relevance (`embed_text`, `rerank_terms_by_subject`, `twelvelabs.py:475-538`) — no video embedding search is wired.

### 3.4 Gaps between this and the stated goal

The goal — "for this exact slice of speech, find the most relevant clip and the exact second-to-second window that covers it" — is roughly 80% built. What is missing:

1. **The renderer** ignores both the beat's timeline position and its length (S5, Part 2).
2. **Two of the three providers** get analysed and then discarded (§1.5).
3. **A 4-second floor** on candidate duration (`material.py:1561`, and `segment_winner` returns `None` below 4 s) means short, tightly-cropped stock clips — often the most on-point ones — are filtered out before analysis.
4. **Only one query per semantic group.** Sibling beats inherit it, so two consecutive shots of the same idea search identically and are distinguished only by the dedupe rule, not by intent.
5. **No cross-provider competition.** One provider is searched per run. The best clip for a beat may be on Coverr while the run is on Pexels.
6. **No still-image fallback** for beats where no footage qualifies (§3.2).

Items 3 and 5 are the ones most likely to be limiting output quality right now, and neither is part of S5.
