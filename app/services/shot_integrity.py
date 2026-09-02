"""Pixel-level checks on a clip that has already been downloaded.

Every gate a candidate passes before this one is a question put to a model and
paid for: the candidate analysis, the semantic adjudication, the temporal
segmentation. That makes all of them expensive, and it makes all of them
answerable by a model that has an incentive to agree with the question. This
module exists for the two defects that survived that whole ladder in production
and that nothing upstream can be trusted to catch:

**A window that spans a cut.** Temporal segmentation returns one interval and the
renderer plays exactly that interval. Nothing checked whether the interval stays
inside one continuous shot of the source, so a Pinterest reel that cuts every two
seconds could hand back an interval that plays two unrelated shots back to back —
inside a single beat, which is the one place the video must not cut. The measured
case: a 15 s reel whose pottery shot ends 2.33 s in, asked for 3.39 s starting at
0.006 s, so the last second of that beat was a completely different scene.

**Footage that is itself an advertisement.** One accepted pin carried a website
URL burned across the top of every frame and a "POTTERY TUTORIALS / on YOUTUBE"
call-to-action block occupying the bottom third, and it shipped. The candidate
analysis has a ``dominant_text_or_logo`` quality flag and the model returned
``false`` for that clip, which is the whole argument for measuring this locally
instead: a flag a model fills in is a claim, and pixels are evidence.

Three properties shape every choice below.

**It must not cost anything.** Both checks read the file that was downloaded
anyway, using only FFmpeg. The project deliberately does not depend on FFprobe
(``imageio-ffmpeg`` ships FFmpeg alone), so duration and frames both come from
FFmpeg invocations, and nothing here opens a socket.

**It must fail open.** A missing binary, an unreadable file or a timeout means
the check could not be performed, never that the candidate is bad. Every probe
returns ``None`` on failure and the caller keeps the window it already had. The
alternative — treating "could not measure" as "reject" — would turn a broken
FFmpeg install into a video with no footage in it.

**Its thresholds are measured, not guessed.** The overlay thresholds were
calibrated against the 44 clips in this project's own download cache: the two
clips carrying burned-in text score 0.15 and 0.19 on the edge-zone density below,
while the highest-scoring clean clip reaches 0.048 and the rest stay under 0.027.
The one near-frozen macro shot that also scores high is separated by the
middle-zone ratio rather than by density. The scene threshold was picked the same
way: at 0.20 every single-shot stock clip in that cache reports zero cuts, and
both multi-shot reels report theirs.
"""

import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.utils import utils

# Scene score above which FFmpeg's ``select`` filter calls a frame a new shot.
# Measured on this project's download cache: 0.20 reports zero cuts for all 30
# single-shot clips and finds every real cut in the edited reels. 0.30 misses a
# cut that a shipped window actually spanned, and 0.12 starts reporting the same
# cut twice.
_SCENE_THRESHOLD = 0.20

# Two detections closer together than this are one cut reported twice, which
# happens when a hard cut carries a one-frame dissolve.
_CUT_MERGE_SECONDS = 0.25

# A detection this close to the first or last frame is a decode artifact, not a
# shot change: there is no previous shot at t=0.
_CUT_EDGE_SECONDS = 0.15

# Kept clear of an internal shot boundary so a transition frame cannot leak into
# the window through rounding.
_SHOT_GUARD_SECONDS = 0.06

# Frames sampled for the overlay probe, and the size they are decoded to. The
# downscale is not only about speed: it averages away the compression noise that
# otherwise makes a frozen overlay look like it changes. At 320 px wide the noise
# wins and even a solid burned-in block measures zero.
_OVERLAY_FRAMES = 8
_OVERLAY_WIDTH = 160
_OVERLAY_HEIGHT = 288

# A pixel counts as part of an overlay when it barely changes across the sampled
# frames and sits on a hard edge in the median frame.
_OVERLAY_STATIC_MAX = 8
_OVERLAY_EDGE_MIN = 25

# Rows searched for burned-in text, as a fraction of frame height. Branding lives
# against the top or bottom edge; the band between them is the reference.
_OVERLAY_TOP_ZONE = 0.22
_OVERLAY_BOTTOM_ZONE = 0.68

# Decision thresholds for the edge zones. ``_OVERLAY_MIDDLE_RATIO`` is what keeps
# a locked-off shot — where the whole frame is static and edgy — from being read
# as an overlay: the near-frozen macro shot in the cache scores 0.220 at its
# bottom edge but 0.205 in the middle, a ratio of 1.07.
#
# Measured on the windows this pipeline actually shipped: the branded pin scores
# 0.149 against 0.025 in the middle, while the highest-scoring clean clip reaches
# 0.048 (a static bucket rim) and the other 41 stay under 0.027. The transition
# count is reported as evidence but deliberately not part of the decision — a
# logo with three large glyphs produces few transitions, and density with the
# middle-zone ratio already separates every case in the cache.
_OVERLAY_MIN_DENSITY = 0.09
_OVERLAY_MIDDLE_RATIO = 3.0

# Files below this size are test fixtures and placeholder writes, not video. The
# probes skip them instead of paying for a subprocess that is going to fail.
_MIN_PROBE_BYTES = 32 * 1024

_CUT_SCAN_TIMEOUT_SECONDS = 90
_FRAME_SAMPLE_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class ClipInspection:
    """What the local probes concluded about one downloaded clip.

    ``start_time``/``end_time`` are always a usable window: on rejection they are
    the window that was asked for, because the caller drops the candidate rather
    than rendering them.
    """

    start_time: float
    end_time: float
    rejected: bool = False
    rejection_reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


def _probe_ready(video_path: str) -> bool:
    try:
        return os.path.getsize(video_path) >= _MIN_PROBE_BYTES
    except OSError:
        return False


def _run_ffmpeg(arguments: list[str], *, timeout: int) -> subprocess.CompletedProcess | None:
    command = [utils.get_ffmpeg_binary(), "-hide_banner", "-nostdin", *arguments]
    try:
        return subprocess.run(command, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"local clip probe timed out after {timeout}s")
    except OSError as exc:
        logger.warning(f"local clip probe could not run FFmpeg: {str(exc)}")
    return None


def detect_shot_cuts(video_path: str) -> list[float] | None:
    """Timestamps where the source changes shot, or ``None`` if unmeasurable.

    An empty list is a real answer — it means one continuous shot — and is not
    the same as ``None``, which means the probe failed and the caller must not
    conclude anything.
    """
    if not _probe_ready(video_path):
        return None
    completed = _run_ffmpeg(
        [
            "-i",
            video_path,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            # The comma inside gt() has to be escaped or FFmpeg reads it as the
            # separator between two filters.
            "-filter:v",
            f"scale=224:-2,select=gt(scene\\,{_SCENE_THRESHOLD}),showinfo",
            "-f",
            "null",
            "-",
        ],
        timeout=_CUT_SCAN_TIMEOUT_SECONDS,
    )
    if completed is None or completed.returncode != 0:
        return None
    report = (completed.stderr or b"").decode("utf-8", "replace")
    detections: list[float] = []
    for raw in re.findall(r"pts_time:([0-9.]+)", report):
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value) and value > 0:
            detections.append(value)
    return _merge_cuts(sorted(detections))


def _merge_cuts(detections: list[float]) -> list[float]:
    merged: list[float] = []
    for value in detections:
        if value <= _CUT_EDGE_SECONDS:
            continue
        if merged and value - merged[-1] < _CUT_MERGE_SECONDS:
            continue
        merged.append(round(value, 3))
    return merged


def shot_bounds(cuts: list[float], source_duration: float) -> list[tuple[float, float]]:
    """The source split into continuous shots at ``cuts``."""
    duration = max(0.0, float(source_duration))
    edges = [0.0]
    for cut in cuts:
        if _CUT_EDGE_SECONDS < cut < duration - _CUT_EDGE_SECONDS:
            edges.append(float(cut))
    edges.append(duration)
    return [
        (edges[index], edges[index + 1])
        for index in range(len(edges) - 1)
        if edges[index + 1] > edges[index]
    ]


def window_inside_one_shot(
    cuts: list[float],
    *,
    source_duration: float,
    verified_start: float,
    verified_end: float,
    required_duration: float,
) -> tuple[float, float] | None:
    """Fit the required window into one shot the segmenter actually looked at.

    Only shots overlapping the verified interval are candidates. Reaching for an
    unrelated long shot elsewhere in the file would ship footage no gate ever
    examined, which is the defect the unfillable-merge path already had to fix.
    """
    required = float(required_duration)
    if not math.isfinite(required) or required <= 0:
        return None
    shots = shot_bounds(cuts, source_duration)
    if not shots:
        return None

    verified_start = max(0.0, float(verified_start))
    verified_end = min(float(source_duration), float(verified_end))
    if verified_end <= verified_start:
        return None
    anchor = (verified_start + verified_end) / 2.0

    overlapping = [
        shot
        for shot in shots
        if min(shot[1], verified_end) - max(shot[0], verified_start) > 0
    ]
    if not overlapping:
        return None
    # The shot holding the midpoint of the verified interval is the one the
    # description belongs to; the widest overlapping shot is the fallback when
    # the interval straddles a boundary.
    ordered = sorted(
        overlapping,
        key=lambda shot: (
            not (shot[0] <= anchor < shot[1]),
            -(min(shot[1], verified_end) - max(shot[0], verified_start)),
        ),
    )
    for shot_start, shot_end in ordered:
        # There is no transition frame to avoid at the very start or end of the
        # file, so the guard applies to internal boundaries only.
        usable_start = shot_start + (_SHOT_GUARD_SECONDS if shot_start > 0 else 0.0)
        usable_end = shot_end - (
            _SHOT_GUARD_SECONDS if shot_end < float(source_duration) - 1e-6 else 0.0
        )
        if usable_end - usable_start < required - 1e-6:
            continue
        start = (verified_start + verified_end) / 2.0 - required / 2.0
        start = min(max(start, usable_start), usable_end - required)
        return round(start, 3), round(start + required, 3)
    return None


def _sample_gray_frames(video_path: str, start: float, duration: float):
    """Evenly spaced grayscale frames from one window, as a numpy array."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with MoviePy
        return None
    span = max(0.2, float(duration))
    fps = max(1.0, (_OVERLAY_FRAMES - 1) / span)
    completed = _run_ffmpeg(
        [
            "-ss",
            f"{max(0.0, float(start)):.3f}",
            "-t",
            f"{span:.3f}",
            "-i",
            video_path,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"fps={fps:.4f},scale={_OVERLAY_WIDTH}:{_OVERLAY_HEIGHT},format=gray",
            "-frames:v",
            str(_OVERLAY_FRAMES),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        timeout=_FRAME_SAMPLE_TIMEOUT_SECONDS,
    )
    if completed is None or completed.returncode != 0:
        return None
    raw = completed.stdout or b""
    pixels = _OVERLAY_WIDTH * _OVERLAY_HEIGHT
    count = len(raw) // pixels
    if count < 3:
        return None
    return np.frombuffer(raw[: count * pixels], dtype=np.uint8).reshape(
        count, _OVERLAY_HEIGHT, _OVERLAY_WIDTH
    )


def measure_burned_in_overlay(
    video_path: str,
    *,
    start: float,
    duration: float,
) -> dict[str, Any] | None:
    """Measure how much of the window's top or bottom edge is frozen text.

    The signal is the combination of two things that natural footage does not
    produce together: pixels that do not change at all across the window, and
    hard edges in those same pixels. Grass in wind changes; a logo does not. A
    wall does not change but has no edges. Burned-in text is both, and it is
    packed against an edge of the frame.
    """
    if not _probe_ready(video_path):
        return None
    frames = _sample_gray_frames(video_path, start, duration)
    if frames is None:
        return None
    import numpy as np

    samples = frames.astype(np.int16)
    variation = samples.max(axis=0) - samples.min(axis=0)
    static = variation <= _OVERLAY_STATIC_MAX

    median = np.median(samples, axis=0)
    horizontal = np.zeros_like(median)
    vertical = np.zeros_like(median)
    horizontal[:, 1:] = np.abs(np.diff(median, axis=1))
    vertical[1:, :] = np.abs(np.diff(median, axis=0))
    edged = (horizontal + vertical) >= _OVERLAY_EDGE_MIN

    overlay = static & edged
    height = overlay.shape[0]
    top_rows = max(1, int(round(_OVERLAY_TOP_ZONE * height)))
    bottom_row = int(round(_OVERLAY_BOTTOM_ZONE * height))
    zones = {
        "top": float(overlay[:top_rows].mean()),
        "bottom": float(overlay[bottom_row:].mean()),
    }
    zone = "top" if zones["top"] >= zones["bottom"] else "bottom"
    middle = overlay[top_rows:bottom_row]
    middle_density = float(middle.mean()) if middle.size else 0.0

    strong = horizontal >= _OVERLAY_EDGE_MIN
    zone_strong = strong[:top_rows] if zone == "top" else strong[bottom_row:]
    transitions = float(zone_strong.sum(axis=1).mean())

    return {
        "zone": zone,
        "density": round(zones[zone], 4),
        "middle_density": round(middle_density, 4),
        "transitions": round(transitions, 1),
        "frames_sampled": int(frames.shape[0]),
    }


def overlay_is_burned_in(measurement: dict[str, Any] | None) -> bool:
    """Whether a measurement is strong enough to call the clip branded."""
    if not isinstance(measurement, dict):
        return False
    try:
        density = float(measurement["density"])
        middle_density = float(measurement["middle_density"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        density >= _OVERLAY_MIN_DENSITY
        and density >= _OVERLAY_MIDDLE_RATIO * middle_density
    )


def inspect_downloaded_clip(
    video_path: str,
    *,
    source_duration: float,
    start_time: float,
    end_time: float,
    verified_start: float | None = None,
    verified_end: float | None = None,
) -> ClipInspection:
    """Run both local checks and report the window that should be rendered.

    ``verified_start``/``verified_end`` are the interval the segmenter actually
    described, before it was padded to the slot length. When they are known the
    containment search anchors on them, so a window that was only padded across
    a cut is pulled back into the shot the description belongs to instead of
    being dropped.
    """
    required = max(0.0, float(end_time) - float(start_time))
    evidence: dict[str, Any] = {}
    window_start = float(start_time)
    window_end = float(end_time)

    cuts = detect_shot_cuts(video_path)
    if cuts is None:
        evidence["shot_cuts"] = "unavailable"
    else:
        evidence["shot_cuts"] = cuts
        anchor_start = window_start if verified_start is None else float(verified_start)
        anchor_end = window_end if verified_end is None else float(verified_end)
        contained = window_inside_one_shot(
            cuts,
            source_duration=source_duration,
            verified_start=anchor_start,
            verified_end=anchor_end,
            required_duration=required,
        )
        if contained is None:
            evidence["shot_containment"] = "no_shot_long_enough"
            return ClipInspection(
                start_time=window_start,
                end_time=window_end,
                rejected=True,
                rejection_reason=(
                    "no continuous shot of the source is long enough to cover "
                    f"the required {required:.3f}s around the verified interval"
                ),
                evidence=evidence,
            )
        shifted = abs(contained[0] - window_start)
        window_start, window_end = contained
        evidence["shot_containment"] = "shifted" if shifted > 1e-3 else "already_inside"
        if shifted > 1e-3:
            evidence["shifted_seconds"] = round(shifted, 3)

    overlay = measure_burned_in_overlay(
        video_path,
        start=window_start,
        duration=window_end - window_start,
    )
    if overlay is None:
        evidence["burned_in_overlay"] = "unavailable"
    else:
        evidence["burned_in_overlay"] = overlay
        if overlay_is_burned_in(overlay):
            return ClipInspection(
                start_time=window_start,
                end_time=window_end,
                rejected=True,
                rejection_reason=(
                    "the footage carries burned-in text or branding across its "
                    f"{overlay['zone']} edge (density {overlay['density']:.3f} "
                    f"against {overlay['middle_density']:.3f} elsewhere)"
                ),
                evidence=evidence,
            )

    return ClipInspection(
        start_time=window_start,
        end_time=window_end,
        evidence=evidence,
    )
