import math
import os
import random
import re
import threading
# ``field`` is aliased because this module's older record builders use ``field``
# as a loop variable; importing it under its own name would shadow the import.
from dataclasses import dataclass, field as dataclass_field, replace as dataclass_replace
from pathlib import Path
from typing import Any, Callable, List, Protocol, Sequence
from urllib.parse import quote_plus, urlencode, urlsplit, urlunsplit

import requests
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip

from app.config import config
from app.models.schema import (
    MaterialInfo,
    RenderSegment,
    VideoAspect,
    VideoConcatMode,
    VisualBeat,
    VisualRequirementSpec,
    VisualSlot,
    VISUAL_BEAT_RAPID_CUT_SECONDS,
)
from app.services import (
    llm,
    material_cache,
    pinterest,
    shot_integrity,
    task_artifacts,
)

from app.utils import utils

# Thread-safe counter for API key rotation
_api_key_counter = 0
_api_key_lock = threading.Lock()
_MIN_STOCK_RENDITION_SHORT_EDGE = 720
_MIN_SEMANTIC_QA_DURATION = 4

# Searchable stock-video providers, in cascade order, with their config keys.
# Smart matching walks this chain so a weak result from the preferred provider
# can still be replaced by a stronger one instead of failing the whole task.
#
# ``None`` marks a provider that needs no credential. Pinterest reads a public
# web search resource, so there is no key to configure and nothing to rotate.
# That is why membership and key lookup have to be two separate questions here:
# an unknown provider and a keyless provider both produce no config key, and
# collapsing them would drop Pinterest out of every chain it belongs in.
_STOCK_VIDEO_PROVIDER_API_KEYS: dict[str, str | None] = {
    "pinterest": None,
    "pexels": "pexels_api_keys",
    "pixabay": "pixabay_api_keys",
}
_SMART_PROVIDER_CASCADE_ORDER: tuple[str, ...] = ("pinterest", "pexels", "pixabay")
# Pinterest pages 25 pins at a time, and only some of them carry a progressive
# MP4 at a usable size, so three pages are needed to reach a candidate count
# comparable to one Pexels or Pixabay request. Each page is one HTTP call and
# paging stops as soon as the limit is reached or the results run out.
_PINTEREST_SEARCH_RESULT_LIMIT = 50
_PINTEREST_SEARCH_MAX_PAGES = 3
# A beat that finds nothing usable is far more often phrased badly than absent
# from the catalog, so the alternative phrasings the script stage already
# generated are tried on the current provider before the next provider is asked.
# The cap is what keeps the worst case bounded: each extra phrasing is one more
# search plus one more round of candidate analysis.
_DEFAULT_MAX_QUERY_VARIANTS = 3
# A rewritten requirement has no script-stage phrasings of its own. When the
# provider cannot produce fresh queries either, the requirement text is only
# usable as a query while it is still short enough to behave like one; stock
# search treats a long sentence as an over-constrained query and returns nothing.
_MAX_REQUIREMENT_QUERY_WORDS = 6
# Every phrasing, every provider in the cascade and the alternative-wording retry
# bill the same metered account, and only a *successful* round stops early — a
# round nothing can satisfy spends the full candidate cap on each of them, so the
# cost of one unfillable item is multiplicative where a healthy one is not.
# The budget is expressed as a multiple of the configured per-search cap so that
# lowering that cap lowers this too. The multiple is deliberately loose: phrasings
# of one item on one provider return heavily overlapping catalogs and the
# per-item exclusion set already stops the second phrasing re-buying the first
# one's verdicts, so the realistic designed cascade lands well under the ceiling
# and only the pathological case — every phrasing returning a fresh full page on
# every provider — is cut off.
_DEFAULT_ANALYSIS_BUDGET_MULTIPLIER = 5

# The per-round ceiling bounds one item; nothing bounds one video. Since the
# failure ladder lets a run reach the end instead of dying at its first unfillable
# item, a video can now pay for several hopeless items in a row, so the per-video
# ceiling is what turns "expensive" into "bounded". It is expressed as how many
# full candidate pages the video may average per item, because those are the two
# numbers the code actually knows in advance — the average clip length, which is
# what TwelveLabs really bills, is not knowable before the searches happen. Two
# pages per item is deliberately loose: a healthy item settles in its first batch,
# so a whole healthy video spends well under one page per item and never notices
# this, and a video with one or two genuinely hard items still gets its complete
# ladder. Only the runaway case — several hopeless items each burning their full
# two rounds — is cut, and it is cut into the free rungs rather than into failure.
_DEFAULT_VIDEO_ANALYSIS_PAGES_PER_ITEM = 2

# Beat boundaries come from the same in-memory objects the S4 validator already
# proved contiguous, so this only guards against float re-association.
_RENDER_SEGMENT_TIME_TOLERANCE = 1e-6
# Persisted source windows are rounded to milliseconds, and the renderer
# recomputes the exact source length from the frame-aligned target anyway. This
# tolerance only has to be wide enough to absorb that rounding.
_RENDER_SEGMENT_DURATION_TOLERANCE = 0.02


class OrderedVisualItem(Protocol):
    """Small shared surface consumed by ordered smart material selection."""

    index: int
    duration: float
    visual_requirement: str
    search_queries: list[str]


class SmartMaterialSelectionError(RuntimeError):
    """A safe user-facing failure from opt-in smart material selection."""


def _safe_public_url(value: Any) -> str | None:
    """
    只保留可公开展示的 HTTP(S) 页面地址，并移除查询参数和凭据。

    素材下载地址可能携带 API Key、签名 JWT 或临时 token。任务清单只需要
    帮助用户回到供应商的公开素材页，不应保存鉴权参数；用户信息形式的 URL
    同样拒绝，避免 ``https://user:pass@example.com`` 一类内容落盘。
    """
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _normalized_media_url(value: Any) -> str:
    """Return a conservative identity form for an exact media URL.

    Scheme/host casing, default ports, an empty root path, and fragments do not
    change the fetched media resource. Path and query bytes are deliberately
    preserved because provider CDN URLs can be signed and case-sensitive.
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    raw_url = value.strip()
    try:
        parsed = urlsplit(raw_url)
        port = parsed.port
    except ValueError:
        return raw_url
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return raw_url

    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"
    return urlunsplit((scheme, hostname, parsed.path or "/", parsed.query, ""))


def _provider_asset_identity(item: MaterialInfo) -> tuple[str, str] | None:
    """Resolve provider-aware identity while accepting legacy source metadata."""
    source = item.source_info if isinstance(item.source_info, dict) else {}
    provider = str(source.get("provider") or item.provider or "pexels").strip().lower()
    provider_asset_id = str(
        item.provider_asset_id
        or source.get("provider_asset_id")
        or source.get("asset_id")
        or ""
    ).strip()
    if not provider or not provider_asset_id:
        return None
    return provider, provider_asset_id


def _is_duplicate_material(
    item: MaterialInfo,
    seen_asset_identities: set[tuple[str, str]],
    seen_urls: set[str],
) -> bool:
    """Match duplicates by provider identity or exact normalized media URL."""
    identity = _provider_asset_identity(item)
    normalized_url = _normalized_media_url(item.url)
    return bool(
        (identity is not None and identity in seen_asset_identities)
        or (normalized_url and normalized_url in seen_urls)
    )


def _remember_material_identity(
    item: MaterialInfo,
    seen_asset_identities: set[tuple[str, str]],
    seen_urls: set[str],
) -> None:
    identity = _provider_asset_identity(item)
    normalized_url = _normalized_media_url(item.url)
    if identity is not None:
        seen_asset_identities.add(identity)
    if normalized_url:
        seen_urls.add(normalized_url)


def _creator_info(value: Any) -> dict[str, str] | None:
    """从不同供应商的作者结构中提取统一的公开字段。"""
    if isinstance(value, str) and value.strip():
        return {"name": value.strip()}
    if not isinstance(value, dict):
        return None

    creator: dict[str, str] = {}
    creator_id = value.get("id")
    creator_name = value.get("name") or value.get("username")
    creator_page = _safe_public_url(
        value.get("url") or value.get("profile_url") or value.get("profile_page")
    )
    if creator_id is not None:
        creator["id"] = str(creator_id)
    if creator_name:
        creator["name"] = str(creator_name)
    if creator_page:
        creator["profile_page"] = creator_page
    return creator or None


def _plain_text(value: Any, limit: int) -> str:
    """Strip control characters and clamp length for manifest storage."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "").strip())[:limit]


def _temporal_segment_record(value: Any) -> dict[str, Any]:
    """The part of a temporal segmentation result worth keeping in the manifest.

    ``source_start_time``/``source_end_time`` are already recorded at the top
    level, and they are the *padded* window. What is missing from the manifest —
    and what every audit of "why did this beat look wrong" needs — is how well
    the model said the clip matched, which slice of it the model actually
    described, and how much of the shipped window nothing described at all.
    """
    if not isinstance(value, dict):
        return {}
    segment: dict[str, Any] = {}
    for field in ("verified_start_time", "verified_end_time", "padded_seconds"):
        try:
            number = float(value[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 0:
            segment[field] = round(number, 3)
    try:
        segment["match_quality"] = round(
            min(1.0, max(0.0, float(value["match_quality"]))), 4
        )
    except (KeyError, TypeError, ValueError):
        pass
    for field in ("action_visible", "subject_visible"):
        if isinstance(value.get(field), bool):
            segment[field] = value[field]
    description = _plain_text(value.get("description"), 240)
    if description:
        segment["description"] = description
    return segment


def _local_clip_check_record(value: Any) -> dict[str, Any]:
    """The local pixel checks, flattened for the manifest.

    Kept short on purpose: an edited reel can have dozens of cuts, and the
    manifest only needs enough of them to show why the window moved.
    """
    if not isinstance(value, dict):
        return {}
    check: dict[str, Any] = {}
    cuts = value.get("shot_cuts")
    if isinstance(cuts, list):
        check["shot_cut_count"] = len(cuts)
        trimmed = []
        for cut in cuts[:24]:
            try:
                trimmed.append(round(float(cut), 3))
            except (TypeError, ValueError):
                continue
        if trimmed:
            check["shot_cuts"] = trimmed
    elif isinstance(cuts, str):
        check["shot_cuts"] = _plain_text(cuts, 40)
    containment = value.get("shot_containment")
    if containment:
        check["shot_containment"] = _plain_text(containment, 40)
    try:
        check["shifted_seconds"] = round(float(value["shifted_seconds"]), 3)
    except (KeyError, TypeError, ValueError):
        pass
    overlay = value.get("burned_in_overlay")
    if isinstance(overlay, dict):
        kept: dict[str, Any] = {}
        zone = _plain_text(overlay.get("zone"), 10)
        if zone:
            kept["zone"] = zone
        for field in ("density", "middle_density", "transitions"):
            try:
                kept[field] = round(float(overlay[field]), 4)
            except (KeyError, TypeError, ValueError):
                continue
        try:
            kept["frames_sampled"] = int(overlay["frames_sampled"])
        except (KeyError, TypeError, ValueError):
            pass
        if kept:
            check["burned_in_overlay"] = kept
    elif isinstance(overlay, str):
        check["burned_in_overlay"] = _plain_text(overlay, 40)
    return check


def _material_source_record(item: MaterialInfo, local_path: str) -> dict[str, Any]:
    """
    为成功下载的素材生成轻量来源记录。

    ``source_info`` 可能来自缓存，甚至来自外部构造的 ``MaterialInfo``，因此
    不能原样写入。这里按白名单重新构造，只保留公开页面、业务标识和尺寸，
    并只记录本地文件名，避免用户目录或 Docker 挂载路径进入任务文件。
    """
    source = item.source_info if isinstance(item.source_info, dict) else {}
    identity = _provider_asset_identity(item)
    record: dict[str, Any] = {
        "provider": identity[0]
        if identity is not None
        else str(item.provider or source.get("provider") or ""),
        "local_file": Path(local_path).name,
        "duration": int(item.duration),
    }

    search_term = item.search_query or source.get("search_query") or source.get(
        "search_term"
    )
    asset_id = identity[1] if identity is not None else None
    source_page = _safe_public_url(
        item.source_page_url
        or source.get("source_page_url")
        or source.get("source_page")
    )
    if isinstance(search_term, str) and search_term.strip():
        record["search_term"] = search_term.strip()
    if asset_id not in (None, ""):
        # Keep the original key for existing script.json readers while adding
        # the provider-neutral name used by new code.
        record["asset_id"] = str(asset_id)
        record["provider_asset_id"] = str(asset_id)
    if source_page:
        record["source_page"] = source_page

    creator = _creator_info(source.get("creator"))
    if creator:
        record["creator"] = creator

    raw_rendition = source.get("rendition")
    raw_rendition = raw_rendition if isinstance(raw_rendition, dict) else {}
    rendition = {
        field: value
        for field, value in {
            "id": item.rendition_id,
            "width": item.width,
            "height": item.height,
        }.items()
        if value not in (None, "")
    }
    for field in ("id", "width", "height"):
        value = raw_rendition.get(field)
        if value not in (None, "") and field not in rendition:
            rendition[field] = str(value) if field == "id" else value
    if rendition:
        if "id" in rendition:
            rendition["id"] = str(rendition["id"])
        record["rendition"] = rendition

    raw_semantic_qa = source.get("semantic_qa")
    if isinstance(raw_semantic_qa, dict):
        provider = str(raw_semantic_qa.get("provider") or "").strip()
        accepted = raw_semantic_qa.get("accepted")
        score = raw_semantic_qa.get("score")
        reason = re.sub(
            r"[\x00-\x1f\x7f]+",
            " ",
            str(raw_semantic_qa.get("reason") or "").strip(),
        )
        semantic_qa: dict[str, Any] = {}
        if provider:
            semantic_qa["provider"] = provider[:40]
        if isinstance(accepted, bool):
            semantic_qa["accepted"] = accepted
        try:
            semantic_qa["score"] = round(min(1.0, max(0.0, float(score))), 4)
        except (TypeError, ValueError):
            pass
        if reason:
            semantic_qa["reason"] = reason[:240]
        if semantic_qa:
            record["semantic_qa"] = semantic_qa

    slot_index = source.get("slot_index")
    if isinstance(slot_index, int) and slot_index > 0:
        record["slot_index"] = slot_index
    visual_beat_index = source.get("visual_beat_index")
    if isinstance(visual_beat_index, int) and visual_beat_index > 0:
        record["visual_beat_index"] = visual_beat_index
    semantic_group_id = source.get("semantic_group_id")
    if isinstance(semantic_group_id, int) and semantic_group_id > 0:
        record["semantic_group_id"] = semantic_group_id
    for field in ("required_target_duration", "required_source_duration"):
        try:
            value = float(source.get(field))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            record[field] = round(value, 3)

    if item.source_start_time is not None and item.source_end_time is not None:
        try:
            source_start = max(0.0, float(item.source_start_time))
            source_end = min(float(item.duration), float(item.source_end_time))
            if math.isfinite(source_start) and math.isfinite(source_end):
                if source_end > source_start:
                    record["source_start_time"] = round(source_start, 3)
                    record["source_end_time"] = round(source_end, 3)
        except (TypeError, ValueError):
            pass

    temporal_segment = _temporal_segment_record(source.get("temporal_segment"))
    if temporal_segment:
        record["temporal_segment"] = temporal_segment
    local_clip_check = _local_clip_check_record(source.get("local_clip_check"))
    if local_clip_check:
        record["local_clip_check"] = local_clip_check

    if isinstance(item.semantic_evaluation, dict):
        evaluation = {}
        for field in (
            "provider",
            "model",
            "schema_version",
            "accepted",
            "eligible_for_adjudication",
            "visual_requirement_spec",
            "observed_facts",
            "critical_gate",
            "semantic_adjudication",
            "scores",
            "overall_score",
            "quality_flags",
            "reason",
            "analysis_input",
        ):
            if field in item.semantic_evaluation:
                evaluation[field] = item.semantic_evaluation[field]
        if evaluation:
            record["semantic_evaluation"] = evaluation
    if item.overall_score is not None:
        try:
            record["overall_score"] = round(
                min(1.0, max(0.0, float(item.overall_score))), 4
            )
        except (TypeError, ValueError):
            pass
    return record


def _persist_material_sources(
    task_id: str,
    material_sources: list[dict[str, Any]],
    semantic_verifier_runs: list[dict[str, Any]] | None = None,
) -> None:
    """
    将当前实际下载成功的素材来源补充到任务清单。

    任务记录是辅助能力，不能改变视频下载函数的返回值，也不能因为写盘失败
    中断成片主流程。``patch_script_data`` 会负责原子替换和异常日志；这里仅在
    成功后记录数量，便于确认任务追溯信息是否已经落盘。
    """
    try:
        fields: dict[str, Any] = {"material_sources": material_sources}
        if semantic_verifier_runs is not None:
            fields["semantic_verifier_runs"] = semantic_verifier_runs
        saved = task_artifacts.patch_script_data(task_id, **fields)
        if saved:
            logger.info(
                f"saved material source records: "
                f"task_id={task_id}, count={len(material_sources)}"
            )
    except Exception as exc:
        # task_artifacts 自身已经按失败降级设计，这里仍保留最后一道隔离，
        # 防止未来实现调整或目录解析异常意外影响素材下载返回值。
        logger.warning(
            "failed to persist material source records: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}"
        )


def load_selected_source_ranges(
    task_id: str,
    video_paths: List[str],
) -> list[tuple[float, float]]:
    """Load the exact TwelveLabs-selected range for every downloaded winner."""
    payload = task_artifacts.read_script_data(task_id)
    records = payload.get("material_sources")
    if not isinstance(records, list):
        raise ValueError("smart material source ranges are missing from script.json")

    records_by_file: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        local_file = str(record.get("local_file") or "").strip()
        if not local_file or local_file in records_by_file:
            raise ValueError("smart material source records are ambiguous")
        records_by_file[local_file] = record

    ranges: list[tuple[float, float]] = []
    for video_path in video_paths:
        record = records_by_file.get(Path(video_path).name)
        if record is None:
            raise ValueError("smart material source range is missing for a winner")
        try:
            start_time = float(record["source_start_time"])
            end_time = float(record["source_end_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("smart material source range is invalid") from exc
        if (
            not math.isfinite(start_time)
            or not math.isfinite(end_time)
            or start_time < 0
            or end_time <= start_time
        ):
            raise ValueError("smart material source range is invalid")
        ranges.append((start_time, end_time))

    if len(ranges) != len(video_paths):
        raise ValueError("smart material source range count does not match winners")
    return ranges


def load_render_segments(
    task_id: str,
    video_paths: List[str],
    visual_beats: Sequence[VisualBeat],
    clip_speed: float = 1.0,
    audio_duration: float | None = None,
) -> list[RenderSegment]:
    """Bind every approved visual beat to the exact source window that was selected for it.

    ``load_selected_source_ranges`` joins records to files by basename, which is
    ambiguous as soon as two beats reuse the same asset. Beats carry an explicit
    ``visual_beat_index``, so this joins on that instead and treats the beat as
    the authority for target timing and the record as the authority for the
    source window. Any inconsistency raises: rendering a half-trusted timeline
    would desync the narration rather than degrade it.
    """
    if not visual_beats:
        raise ValueError("visual beats are required to build render segments")
    if len(video_paths) != len(visual_beats):
        raise ValueError(
            "render segment inputs disagree: "
            f"winners={len(video_paths)}, beats={len(visual_beats)}"
        )

    normalized_speed = utils.normalize_clip_speed(clip_speed)
    payload = task_artifacts.read_script_data(task_id)
    records = payload.get("material_sources")
    if not isinstance(records, list):
        raise ValueError("smart material source records are missing from script.json")

    records_by_beat: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        beat_index = record.get("visual_beat_index")
        if not isinstance(beat_index, int):
            raise ValueError(
                "smart material source records predate the visual beat timeline"
            )
        if beat_index in records_by_beat:
            raise ValueError(
                f"smart material source records are ambiguous for beat {beat_index}"
            )
        records_by_beat[beat_index] = record

    segments: list[RenderSegment] = []
    previous_end = 0.0
    for position, (video_path, beat) in enumerate(
        zip(video_paths, visual_beats), start=1
    ):
        record = records_by_beat.get(beat.index)
        if record is None:
            raise ValueError(
                f"smart material source record is missing for beat {beat.index}"
            )
        local_file = str(record.get("local_file") or "").strip()
        if local_file and local_file != Path(video_path).name:
            # A mismatch means the winner order and the persisted records have
            # diverged; rendering would attach a beat to somebody else's clip.
            raise ValueError(
                f"smart material source record for beat {beat.index} points at "
                f"{local_file!r} but the winner is {Path(video_path).name!r}"
            )

        try:
            source_start = float(record["source_start_time"])
            source_end = float(record["source_end_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"smart material source range is invalid for beat {beat.index}"
            ) from exc

        target_start = float(beat.start_time)
        target_end = float(beat.end_time)
        target_duration = target_end - target_start
        if (
            not math.isfinite(source_start)
            or not math.isfinite(source_end)
            or not math.isfinite(target_start)
            or not math.isfinite(target_end)
            or source_start < 0
            or source_end <= source_start
            or target_start < 0
            or target_duration <= 0
        ):
            raise ValueError(f"beat {beat.index} has an unusable render window")

        if position == 1 and target_start > _RENDER_SEGMENT_TIME_TOLERANCE:
            raise ValueError("the visual beat timeline must start at 0.0s")
        if abs(target_start - previous_end) > _RENDER_SEGMENT_TIME_TOLERANCE:
            raise ValueError(
                f"beat {beat.index} leaves a gap or overlap in the timeline"
            )

        rendered_length = (source_end - source_start) / normalized_speed
        if abs(rendered_length - target_duration) > _RENDER_SEGMENT_DURATION_TOLERANCE:
            # Persisted source times are rounded to milliseconds, so a small
            # delta is expected. A larger one means the selection was made for a
            # different slot length than the beat now claims.
            raise ValueError(
                f"beat {beat.index} source window renders {rendered_length:.3f}s "
                f"at {normalized_speed:.2f}x but its timeline slot is "
                f"{target_duration:.3f}s"
            )

        semantic_group_id = record.get("semantic_group_id")
        segments.append(
            RenderSegment(
                index=position,
                file_path=video_path,
                source_start=source_start,
                source_end=source_end,
                target_start=target_start,
                target_end=target_end,
                target_duration=target_duration,
                playback_speed=normalized_speed,
                visual_beat_index=beat.index,
                semantic_group_id=(
                    semantic_group_id
                    if isinstance(semantic_group_id, int)
                    else int(beat.semantic_group_id)
                ),
                provider=str(record.get("provider") or ""),
            )
        )
        previous_end = target_end

    if audio_duration is not None:
        coverage = float(audio_duration) - previous_end
        if coverage > _RENDER_SEGMENT_TIME_TOLERANCE:
            raise ValueError(
                "the visual beat timeline does not cover the narration audio: "
                f"timeline={previous_end:.3f}s, audio={float(audio_duration):.3f}s"
            )

    logger.info(
        f"built render segments: task_id={task_id}, segments={len(segments)}, "
        f"timeline={previous_end:.3f}s, speed={normalized_speed:.2f}x"
    )
    return segments


def _get_tls_verify() -> bool:
    # 默认开启 TLS 证书校验，防止素材搜索和下载过程被中间人篡改。
    # 仅在企业代理、自签证书等明确需要的场景下，允许用户通过
    # `config.toml` 显式设置 `tls_verify = false` 临时关闭。
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")

    if not tls_verify:
        logger.warning(
            "TLS certificate verification is disabled by config.app.tls_verify=false. "
            "Only use this in trusted proxy environments."
        )

    return bool(tls_verify)


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )

    # if only one key is provided, return it
    if isinstance(api_keys, str):
        return api_keys

    global _api_key_counter
    with _api_key_lock:
        _api_key_counter += 1
        return api_keys[_api_key_counter % len(api_keys)]


def _redact_secret(message: str, secret: str) -> str:
    """
    对即将写入日志的异常文本做最小范围脱敏。

    requests 的连接异常可能包含完整请求 URL，而 Pixabay API Key 通过查询
    参数传递。这里同时替换原始值和 URL 编码值，既保留网络错误信息用于排查，
    又避免密钥进入日志文件。
    """
    safe_message = str(message)
    if not secret:
        return safe_message

    safe_message = safe_message.replace(secret, "***")
    encoded_secret = quote_plus(secret)
    if encoded_secret != secret:
        safe_message = safe_message.replace(encoded_secret, "***")
    return safe_message


def _redact_request_error(error: Exception, *secrets: str) -> str:
    """
    保留网络异常的可排查信息，同时移除 API Key 和代理凭据。

    直接只记录异常类型会丢失 DNS、证书、超时等关键上下文；直接记录原始异常
    又可能回显完整请求 URL。统一入口可以让三个素材供应商使用相同脱敏规则。
    """
    safe_message = str(error)
    for secret in secrets:
        safe_message = _redact_secret(safe_message, str(secret or ""))
    for proxy_url in config.proxy.values():
        safe_message = _redact_secret(safe_message, str(proxy_url))
    return safe_message


def _is_cloudflare_challenge(response: requests.Response) -> bool:
    """
    识别 Cloudflare 返回的 HTML Challenge，而不是把它当成 Pixabay JSON。

    Cloudflare 通常会设置 `cf-mitigated: challenge`；部分部署只返回带有
    "Just a moment" 或 challenge-platform 的 HTML，因此保留内容特征兜底。
    响应正文仅在内存中判断，不写入日志，避免记录无价值的大段 HTML。
    """
    headers = getattr(response, "headers", {}) or {}
    if str(headers.get("cf-mitigated", "")).lower() == "challenge":
        return True

    content_type = str(headers.get("content-type", "")).lower()
    if "text/html" not in content_type:
        return False

    body = str(getattr(response, "text", "")).lower()
    return "just a moment" in body or "/cdn-cgi/challenge-platform/" in body


def _matches_video_aspect(
    width: Any,
    height: Any,
    video_aspect: VideoAspect,
    *,
    is_vertical: Any = None,
) -> bool:
    """
    判断远端素材是否与目标画面方向一致。

    Pexels 和 Pixabay 的响应字段并不统一，因此先使用宽高做可靠判断；
    部分历史响应缺少尺寸时，再使用明确的 ``is_vertical`` 布尔值兜底。
    无法确认方向的素材直接跳过，避免竖屏任务混入横屏素材并在成片中产生黑边。
    """
    aspect = VideoAspect(video_aspect)
    try:
        normalized_width = int(float(width))
        normalized_height = int(float(height))
    except (TypeError, ValueError):
        normalized_width = 0
        normalized_height = 0

    if normalized_width > 0 and normalized_height > 0:
        if aspect == VideoAspect.portrait:
            return normalized_height > normalized_width
        if aspect == VideoAspect.landscape:
            return normalized_width > normalized_height
        return normalized_width == normalized_height

    if isinstance(is_vertical, bool) and aspect != VideoAspect.square:
        return is_vertical == (aspect == VideoAspect.portrait)
    return False


def _orientation_from_dimensions(width: Any, height: Any) -> str | None:
    try:
        normalized_width = int(float(width))
        normalized_height = int(float(height))
    except (TypeError, ValueError):
        return None
    if normalized_width <= 0 or normalized_height <= 0:
        return None
    if normalized_width == normalized_height:
        return "square"
    return "landscape" if normalized_width > normalized_height else "portrait"


def _hydrate_material_metadata(
    item: MaterialInfo,
    *,
    search_query: str | None = None,
    query_attempt: int | None = None,
) -> MaterialInfo:
    """Populate typed metadata from legacy ``source_info`` in-place."""
    source = dict(item.source_info) if isinstance(item.source_info, dict) else {}
    provider = str(source.get("provider") or item.provider or "pexels").strip().lower()
    provider_asset_id = str(
        item.provider_asset_id
        or source.get("provider_asset_id")
        or source.get("asset_id")
        or ""
    ).strip()
    rendition = source.get("rendition")
    rendition = rendition if isinstance(rendition, dict) else {}

    item.provider = provider
    item.provider_asset_id = provider_asset_id or None
    item.preview_url = _safe_public_url(
        item.preview_url or source.get("preview_url")
    )
    try:
        item.width = int(item.width or rendition.get("width"))
        item.height = int(item.height or rendition.get("height"))
    except (TypeError, ValueError):
        item.width = None
        item.height = None
    item.orientation = item.orientation or _orientation_from_dimensions(
        item.width,
        item.height,
    )
    rendition_id = item.rendition_id or rendition.get("id")
    item.rendition_id = (
        str(rendition_id) if rendition_id not in (None, "") else None
    )
    item.search_query = str(
        search_query
        or item.search_query
        or source.get("search_query")
        or source.get("search_term")
        or ""
    ).strip() or None
    if query_attempt is not None:
        item.query_attempt = int(query_attempt)
    elif item.query_attempt is None and item.search_query:
        item.query_attempt = 1
    item.source_page_url = _safe_public_url(
        item.source_page_url
        or source.get("source_page_url")
        or source.get("source_page")
    )

    source["provider"] = provider
    if provider_asset_id:
        source["asset_id"] = provider_asset_id
        source["provider_asset_id"] = provider_asset_id
    if item.preview_url:
        source["preview_url"] = item.preview_url
    if item.search_query:
        source["search_term"] = item.search_query
    if item.source_page_url:
        source["source_page"] = item.source_page_url
    item.source_info = source or None
    return item


def _filter_materials_by_aspect(
    items: List[MaterialInfo],
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    对缓存结果再次校验方向。

    素材搜索缓存最长保留 24 小时，升级前写入的缓存可能包含方向不匹配的素材。
    在统一缓存入口过滤可以让修复立即生效，也能防御第三方 Provider 或旧缓存
    遗漏远端筛选。无法读取 rendition 尺寸的旧条目按未验证处理并跳过。
    """
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.square:
        # Pixabay 很少提供原生方形素材。方形输出沿用既有行为，
        # 接受可用候选并交给视频合成阶段裁剪，避免升级后 1:1 任务无素材。
        return list(items)

    filtered_items = []
    for item in items:
        source_info = item.source_info if isinstance(item.source_info, dict) else {}
        rendition = source_info.get("rendition")
        rendition = rendition if isinstance(rendition, dict) else {}
        if _matches_video_aspect(
            rendition.get("width"),
            rendition.get("height"),
            aspect,
        ):
            filtered_items.append(item)
    return filtered_items


def _prepare_twelvelabs_candidates(
    items: List[MaterialInfo],
    *,
    video_aspect: VideoAspect,
    required_source_duration: float,
    preferred_max_source_duration: float,
) -> List[MaterialInfo]:
    """Apply cheap metadata gates before any TwelveLabs call.

    Short stock clips stay in the original Pexels order. Longer clips are not
    discarded; they form a second bucket used only after the preferred bucket.
    """
    short_candidates: list[MaterialInfo] = []
    long_candidates: list[MaterialInfo] = []
    seen_asset_identities: set[tuple[str, str]] = set()
    seen_urls: set[str] = set()
    aspect = VideoAspect(video_aspect)
    minimum_duration = max(4.0, float(required_source_duration))

    for item in items:
        source = item.source_info if isinstance(item.source_info, dict) else {}
        rendition = source.get("rendition")
        rendition = rendition if isinstance(rendition, dict) else {}
        asset_identity = _provider_asset_identity(item)
        url = str(item.url or "").strip()
        try:
            duration = float(item.duration)
            width = int(rendition.get("width") or 0)
            height = int(rendition.get("height") or 0)
            parsed = urlsplit(url)
        except (TypeError, ValueError):
            continue
        if (
            asset_identity is None
            or not math.isfinite(duration)
            or duration < minimum_duration
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or not parsed.path.lower().endswith(".mp4")
            or not _matches_video_aspect(width, height, aspect)
            or min(width, height) < _MIN_STOCK_RENDITION_SHORT_EDGE
            or _is_duplicate_material(item, seen_asset_identities, seen_urls)
        ):
            continue
        _remember_material_identity(item, seen_asset_identities, seen_urls)
        if duration <= preferred_max_source_duration:
            short_candidates.append(item)
        else:
            long_candidates.append(item)
    return short_candidates + long_candidates


def required_source_duration_for_timeline(
    target_timeline_duration: float,
    clip_speed: float,
) -> float:
    """Mirror MoviePy speed semantics: source seconds = timeline seconds × speed."""
    try:
        target_duration = float(target_timeline_duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("target timeline duration must be positive") from exc
    if not math.isfinite(target_duration) or target_duration <= 0:
        raise ValueError("target timeline duration must be positive")
    return target_duration * utils.normalize_clip_speed(clip_speed)


# The segmentation service reports its interval rounded to three decimals, so an
# interval that is exactly long enough can measure up to 0.001 s short here purely
# from that rounding. Comparing it against a 1e-6 tolerance rejected correct
# windows roughly half the time, discarding candidates that had already passed the
# semantic gate. The accepted interval is re-derived from the unrounded requirement
# below, so this slack never reaches the timeline.
_SOURCE_RANGE_ROUNDING_SLACK_SECONDS = 2e-3


def _normalize_selected_source_range(
    segment: dict[str, Any],
    *,
    source_duration: float,
    required_source_duration: float,
) -> dict[str, Any] | None:
    """Validate and trim a semantic interval to the exact required source time."""
    try:
        start_time = max(0.0, float(segment["source_start_time"]))
        end_time = min(float(source_duration), float(segment["source_end_time"]))
        required = float(required_source_duration)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(value) for value in (start_time, end_time, required))
        or required <= 0
        or end_time <= start_time
        or end_time - start_time
        < required - _SOURCE_RANGE_ROUNDING_SLACK_SECONDS
    ):
        return None

    center = (start_time + end_time) / 2.0
    normalized_start = max(0.0, center - required / 2.0)
    normalized_end = normalized_start + required
    if normalized_end > source_duration:
        normalized_end = float(source_duration)
        normalized_start = max(0.0, normalized_end - required)
    if normalized_end - normalized_start < required - 1e-6:
        return None
    normalized = dict(segment)
    normalized["source_start_time"] = normalized_start
    normalized["source_end_time"] = normalized_end
    # ``padded_seconds`` was computed by the segmentation call against its own
    # requested length. This path can re-derive the window against a different
    # required duration, so the padding has to be restated here or the manifest
    # would under-report how much of the shipped window nothing ever described.
    try:
        verified_span = float(segment["verified_end_time"]) - float(
            segment["verified_start_time"]
        )
    except (KeyError, TypeError, ValueError):
        verified_span = None
    if verified_span is not None and math.isfinite(verified_span):
        normalized["padded_seconds"] = round(max(0.0, required - verified_span), 3)
    return normalized


def _select_best_video_rendition(
    renditions: Any,
    video_aspect: VideoAspect,
) -> dict[str, Any] | None:
    """Choose a usable rendition without requiring one exact resolution.

    Stock providers do not guarantee that every asset has a 1080x1920 or
    1920x1080 rendition. Requiring that exact pair discarded otherwise useful
    HD candidates. Prefer the closest rendition at or above the output size;
    if none exists, use the highest-quality 720p-or-better fallback with the
    correct orientation.
    """
    if not isinstance(renditions, list):
        return None

    aspect = VideoAspect(video_aspect)
    target_width, target_height = aspect.to_resolution()
    target_pixels = target_width * target_height
    ranked: list[tuple[tuple[int, int], dict[str, Any]]] = []

    for rendition in renditions:
        if not isinstance(rendition, dict):
            continue
        try:
            width = int(rendition.get("width") or 0)
            height = int(rendition.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if (
            not _matches_video_aspect(width, height, aspect)
            or min(width, height) < _MIN_STOCK_RENDITION_SHORT_EDGE
            or not rendition.get("link")
        ):
            continue

        pixels = width * height
        meets_target = width >= target_width and height >= target_height
        if meets_target:
            # Prefer the smallest rendition that already meets the output size
            # to avoid downloading a 4K file when Full HD is available.
            score = (2, -(pixels - target_pixels))
        else:
            score = (1, pixels)
        ranked.append((score, rendition))

    if not ranked:
        return None
    return max(ranked, key=lambda candidate: candidate[0])[1]


def search_videos_pinterest(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """Search Pinterest video pins, ranked by the same gates as Pexels.

    Pinterest needs no API key, so this is the one provider that is always
    reachable. Everything that makes it usable as stock footage is enforced
    here: ``_select_best_video_rendition`` applies the identical orientation and
    short-edge gates the licensed providers get, and a pin whose duration cannot
    be established is dropped rather than guessed at, because the render timeline
    sizes a beat's window from that number.

    Returning an empty list for a failure matches the interface every provider in
    this module already has — which is exactly why the failure is logged at error
    level with its actionable cause. The search cache refuses to store empty
    results for the same reason, so a rate limit costs one query rather than a
    day of falsely remembered emptiness.
    """
    aspect = VideoAspect(video_aspect)
    logger.info(
        f"searching videos on pinterest: term={search_term!r}, "
        f"proxy_enabled={bool(config.proxy)}"
    )

    try:
        records = pinterest.search_video_pins(
            search_term,
            limit=_PINTEREST_SEARCH_RESULT_LIMIT,
            max_pages=_PINTEREST_SEARCH_MAX_PAGES,
            proxies=config.proxy,
            verify=_get_tls_verify(),
        )
    except pinterest.PinterestSearchError as exc:
        # Pinterest is an undocumented public endpoint, so a block or a rate
        # limit is a normal operating condition rather than an exception worth
        # a traceback. It must still never be mistaken for an absent concept.
        logger.error(
            f"pinterest video search could not be completed: detail={exc}. "
            "Smart matching continues with the next provider in the cascade."
        )
        return []
    except Exception as e:
        logger.error(
            "pinterest video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e)}"
        )
        return []

    video_items: List[MaterialInfo] = []
    skipped_short = 0
    skipped_unknown_duration = 0
    skipped_rendition = 0

    for record in records:
        duration = record.get("duration")
        if duration is None:
            skipped_unknown_duration += 1
            continue
        # Floor rather than round: a source may never be reported as longer than
        # it is, or a beat can win a clip that cannot cover its own window.
        duration_seconds = int(math.floor(float(duration)))
        if duration_seconds < minimum_duration:
            skipped_short += 1
            continue

        video = _select_best_video_rendition(record.get("renditions"), aspect)
        if video is None:
            skipped_rendition += 1
            continue

        w = int(video["width"])
        h = int(video["height"])
        provider_asset_id = record.get("pin_id")
        source_page_url = _safe_public_url(record.get("pin_url"))
        item = MaterialInfo()
        item.provider = "pinterest"
        item.url = video["link"]
        item.duration = duration_seconds
        item.provider_asset_id = provider_asset_id
        item.preview_url = _safe_public_url(record.get("poster"))
        item.width = w
        item.height = h
        item.orientation = _orientation_from_dimensions(w, h)
        item.rendition_id = (
            str(video.get("id")) if video.get("id") is not None else None
        )
        item.search_query = search_term
        item.query_attempt = 1
        item.source_page_url = source_page_url
        item.source_info = {
            "provider": "pinterest",
            "search_term": search_term,
            "asset_id": provider_asset_id,
            "provider_asset_id": provider_asset_id,
            "source_page": source_page_url,
            "creator": _creator_info(record.get("creator")),
            "rendition": {
                "id": item.rendition_id,
                "width": w,
                "height": h,
            },
        }
        video_items.append(item)

    if records and not video_items:
        # Worth its own line: an empty result after a successful search means the
        # pins existed but none survived the quality gates, which is a different
        # problem from Pinterest having nothing for the concept.
        logger.info(
            "pinterest returned pins but none passed the material gates: "
            f"term={search_term!r}, pins={len(records)}, "
            f"too_short={skipped_short}, unknown_duration={skipped_unknown_duration}, "
            f"wrong_shape_or_resolution={skipped_rendition}"
        )
    return video_items


def search_videos_pexels(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    video_orientation = aspect.name
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    # Build URL
    params = {"query": search_term, "per_page": 80, "orientation": video_orientation}
    query_url = f"https://api.pexels.com/v1/videos/search?{urlencode(params)}"
    logger.info(f"searching videos on pexels: term={search_term!r}")

    try:
        r = requests.get(
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        response = r.json()
        video_items = []
        if "videos" not in response:
            logger.error("pexels video search returned an unsupported response")
            return video_items
        videos = response["videos"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            video = _select_best_video_rendition(v.get("video_files"), aspect)
            if video is None:
                continue
            w = int(video["width"])
            h = int(video["height"])
            provider_asset_id = (
                str(v.get("id")) if v.get("id") is not None else None
            )
            source_page_url = _safe_public_url(v.get("url"))
            item = MaterialInfo()
            item.provider = "pexels"
            item.url = video["link"]
            item.duration = duration
            item.provider_asset_id = provider_asset_id
            item.preview_url = _safe_public_url(v.get("image"))
            item.width = w
            item.height = h
            item.orientation = _orientation_from_dimensions(w, h)
            item.rendition_id = (
                str(video.get("id")) if video.get("id") is not None else None
            )
            item.search_query = search_term
            item.query_attempt = 1
            item.source_page_url = source_page_url
            item.source_info = {
                "provider": "pexels",
                "search_term": search_term,
                "asset_id": provider_asset_id,
                "provider_asset_id": provider_asset_id,
                "source_page": source_page_url,
                "creator": _creator_info(v.get("user")),
                "rendition": {
                    "id": item.rendition_id,
                    "width": w,
                    "height": h,
                },
            }
            video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "pexels video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


def search_videos_pixabay(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)

    video_width, video_height = aspect.to_resolution()

    api_key = get_api_key("pixabay_api_keys")
    # Build URL
    params = {
        "q": search_term,
        "video_type": "all",  # Accepted values: "all", "film", "animation"
        "per_page": 50,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/videos/?{urlencode(params)}"
    logger.info(
        f"searching videos on pixabay: term={search_term!r}, "
        f"proxy_enabled={bool(config.proxy)}"
    )

    try:
        r = requests.get(
            query_url, proxies=config.proxy, verify=_get_tls_verify(), timeout=(30, 60)
        )
        status_code = int(getattr(r, "status_code", 200))
        headers = getattr(r, "headers", {}) or {}
        content_type = str(headers.get("content-type", ""))
        retry_after = headers.get("retry-after")
        cf_ray = headers.get("cf-ray")

        if _is_cloudflare_challenge(r):
            logger.error(
                "pixabay search was blocked by a Cloudflare challenge: "
                f"status={status_code}, cf_ray={cf_ray or 'unknown'}. "
                "Check the server network or proxy, or use Pexels instead."
            )
            return []

        if status_code == 429:
            logger.error(
                "pixabay API rate limit exceeded: "
                f"status=429, retry_after={retry_after or 'unknown'}"
            )
            return []

        if status_code >= 400:
            logger.error(
                "pixabay search request failed: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        try:
            response = r.json()
        except ValueError:
            logger.error(
                "pixabay returned an unexpected non-JSON response: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        video_items = []
        if "hits" not in response:
            logger.error("pixabay video search returned an unsupported response")
            return video_items
        videos = response["hits"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            video_files = v["videos"]
            # loop through each url to determine the best quality
            for video_type in video_files:
                video = video_files[video_type]
                try:
                    w = int(video["width"])
                    h = int(video["height"])
                except (KeyError, TypeError, ValueError):
                    continue
                # Pixabay 很少返回原生方形视频；1:1 输出继续接受满足分辨率的
                # 候选并由合成阶段裁剪。横竖屏则必须严格匹配目标方向。
                orientation_matches = aspect == VideoAspect.square or (
                    _matches_video_aspect(w, h, aspect)
                )
                if orientation_matches and w >= video_width:
                    provider_asset_id = (
                        str(v.get("id")) if v.get("id") is not None else None
                    )
                    source_page_url = _safe_public_url(v.get("pageURL"))
                    item = MaterialInfo()
                    item.provider = "pixabay"
                    item.url = video["url"]
                    item.duration = duration
                    item.provider_asset_id = provider_asset_id
                    item.width = w
                    item.height = h
                    item.orientation = _orientation_from_dimensions(w, h)
                    item.rendition_id = str(video_type)
                    item.search_query = search_term
                    item.query_attempt = 1
                    item.source_page_url = source_page_url
                    item.source_info = {
                        "provider": "pixabay",
                        "search_term": search_term,
                        "asset_id": provider_asset_id,
                        "provider_asset_id": provider_asset_id,
                        "source_page": source_page_url,
                        "creator": _creator_info(
                            {
                                "id": v.get("user_id"),
                                "name": v.get("user"),
                            }
                        ),
                        "rendition": {
                            "id": video_type,
                            "width": w,
                            "height": h,
                        },
                    }
                    video_items.append(item)
                    break
        return video_items
    except Exception as e:
        error_message = _redact_request_error(e, api_key)
        logger.error(
            "pixabay search request failed: "
            f"error={type(e).__name__}, detail={error_message}"
        )

    return []


def _validate_saved_video(
    video_path: str,
    video_aspect: VideoAspect | None = None,
) -> bool:
    """Verify a downloaded/cached clip can be decoded and has the right shape."""
    clip = None
    try:
        clip = VideoFileClip(video_path)
        if clip.duration <= 0 or clip.fps <= 0:
            return False

        if video_aspect is not None and VideoAspect(video_aspect) != VideoAspect.square:
            width = getattr(clip, "w", None)
            height = getattr(clip, "h", None)
            if not _matches_video_aspect(width, height, video_aspect):
                logger.warning(
                    "video orientation does not match output: "
                    f"path={video_path}, width={width}, height={height}, "
                    f"expected={VideoAspect(video_aspect).value}"
                )
                return False
        return True
    except Exception as e:
        logger.warning(f"invalid video file: {video_path} => {str(e)}")
        return False
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception as close_error:
                logger.warning(
                    f"failed to close video clip: {video_path}, error: {str(close_error)}"
                )


def _remove_invalid_video(video_path: str) -> None:
    try:
        os.remove(video_path)
    except FileNotFoundError:
        pass
    except Exception as remove_error:
        logger.warning(
            f"failed to remove invalid video file: {video_path}, error: {str(remove_error)}"
        )


def save_video(
    video_url: str,
    save_dir: str = "",
    video_aspect: VideoAspect | None = None,
) -> str:
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    url_without_query = video_url.split("?")[0]
    url_hash = utils.md5(url_without_query)
    video_id = f"vid-{url_hash}"
    video_path = f"{save_dir}/{video_id}.mp4"

    # Cached files were previously returned without being decoded again. A bad
    # partial download could therefore poison every later task using the URL.
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        if _validate_saved_video(video_path, video_aspect):
            logger.info(f"video already exists: {video_path}")
            return video_path
        _remove_invalid_video(video_path)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    # if video does not exist, download it
    with open(video_path, "wb") as f:
        f.write(
            requests.get(
                video_url,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(60, 240),
            ).content
        )

    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        if _validate_saved_video(video_path, video_aspect):
            return video_path
        _remove_invalid_video(video_path)
    return ""


def _search_videos_with_cache(
    provider: str,
    search_videos: Callable[..., List[MaterialInfo]],
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    统一处理三个在线素材源的 24 小时搜索缓存。

    缓存只包裹搜索 API，不改变后续视频下载与去重逻辑。远端返回空列表时不写
    缓存，因为现有 provider 接口使用空列表同时表示“没有结果”和“请求失败”；
    在两者尚未拆分为明确结果类型前，宁可下次重试，也不能把临时故障缓存一天。
    """
    cache_args = {
        "provider": provider,
        "search_term": search_term,
        "minimum_duration": minimum_duration,
        "video_aspect": video_aspect,
    }

    def load_cache_safely() -> List[MaterialInfo] | None:
        try:
            return material_cache.load_material_search_cache(**cache_args)
        except Exception as exc:
            # 缓存是可选优化，任何缓存实现异常都必须按未命中处理，不能阻断
            # Pexels 或 Pixabay 的正常远端搜索。
            logger.warning(
                "material search cache read failed, continue with remote search: "
                f"provider={provider}, error={type(exc).__name__}, detail={exc}"
            )
            return None

    def load_matching_cache() -> tuple[List[MaterialInfo] | None, int]:
        cached_items = load_cache_safely()
        if cached_items is None:
            return None, 0

        filtered_cached_items = _filter_materials_by_aspect(
            cached_items,
            video_aspect,
        )
        ignored_count = len(cached_items) - len(filtered_cached_items)
        if ignored_count:
            # 旧版本缓存可能混入其它方向的素材。即使仍有少量可用条目，也要刷新
            # 完整候选集，否则在缓存有效期内会反复使用同一批少量视频。
            return None, ignored_count
        return filtered_cached_items, 0

    cached_items, ignored_count = load_matching_cache()
    if cached_items is not None:
        return cached_items
    if ignored_count:
        logger.info(
            "material search cache contains mismatched orientations, "
            f"refresh from provider: provider={provider}, term={search_term!r}, "
            f"ignored={ignored_count}"
        )

    cache_lock = material_cache.get_material_search_cache_lock(**cache_args)
    with cache_lock:
        # 等待相同搜索条件的线程完成后再次读取，避免多个 API 任务在首次缓存
        # 未命中时同时请求远端，降低第三方接口限流和风控触发概率。
        cached_items, _ = load_matching_cache()
        if cached_items is not None:
            return cached_items

        items = search_videos(
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
        # Provider 正常会写入当前关键词，但测试替身、第三方扩展或旧实现可能
        # 遗漏或携带错误值。缓存读取会根据缓存键恢复该字段，因此远端结果也在
        # 同一入口校正，保证首次搜索与缓存命中的任务来源记录保持一致。
        for item in items:
            _hydrate_material_metadata(
                item,
                search_query=search_term,
                query_attempt=1,
            )
        if items:
            try:
                material_cache.save_material_search_cache(
                    **cache_args,
                    items=items,
                )
            except Exception as exc:
                logger.warning(
                    "material search cache write failed, use remote results: "
                    f"provider={provider}, error={type(exc).__name__}, detail={exc}"
                )
        return items


def provider_has_api_key(provider: str) -> bool:
    """Report whether a searchable stock provider is ready to be queried.

    A provider is ready when it needs no credential or when at least one key is
    configured for it. Those two cases have to be answered by one predicate,
    because the cascade uses this to decide what to skip: treating a keyless
    provider as unconfigured would silently remove Pinterest from every chain,
    and treating an unknown provider as keyless would send searches to a provider
    this module cannot query at all.
    """
    normalized = str(provider or "").strip().lower()
    if normalized not in _STOCK_VIDEO_PROVIDER_API_KEYS:
        return False
    cfg_key = _STOCK_VIDEO_PROVIDER_API_KEYS[normalized]
    if cfg_key is None:
        return True
    api_keys = config.app.get(cfg_key)
    if isinstance(api_keys, str):
        return bool(api_keys.strip())
    if isinstance(api_keys, (list, tuple, set)):
        return any(str(value).strip() for value in api_keys)
    return bool(api_keys)


def is_provider_cascade_enabled() -> bool:
    """Allow operators to pin smart matching to the single selected provider."""
    value = config.app.get("smart_material_provider_cascade", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def max_query_variants_per_provider() -> int:
    """How many phrasings of one visual item may be tried on a single provider.

    The script stage already produced several phrasings per item, and trying the
    next phrasing on the provider that is already configured and cached is both
    cheaper and more faithful to the script than jumping to a different catalog
    whose library is thinner. ``1`` restores the previous behavior of spending
    exactly one query across the whole cascade. Values below 1 are read as 1,
    because "search with no query at all" is not a mode.
    """
    value = config.app.get(
        "smart_material_max_query_variants",
        _DEFAULT_MAX_QUERY_VARIANTS,
    )
    try:
        variants = int(str(value).strip())
    except (TypeError, ValueError):
        logger.warning(
            "smart_material_max_query_variants is not a number; "
            f"using {_DEFAULT_MAX_QUERY_VARIANTS}: value={value!r}"
        )
        return _DEFAULT_MAX_QUERY_VARIANTS
    return max(1, variants)


def is_requirement_rewrite_enabled() -> bool:
    """Allow operators to turn off the one alternative-wording retry per item.

    A beat that no provider and no phrasing could satisfy is usually asking for a
    scene that stock catalogs do not hold, so the last cheap move before failing
    the video is to describe the same narration a different way. Turning this off
    restores the previous behavior of failing as soon as an item is unfillable.
    """
    value = config.app.get("smart_material_requirement_rewrite", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def is_opening_shot_rewrite_enabled() -> bool:
    """Whether the opening shot may have its visual requirement re-described.

    The rewrite is allowed to change *what is depicted*, not merely how findable
    it is, so the shot that comes back can answer a different question than the
    narration asked. Every other beat survives that: a viewer who is already
    watching forgives one loose shot in the middle. The opening shot does not
    survive it, because in a vertical feed the frame at t=0 is the thumbnail and
    the whole video is judged on it before the first word is audible. A run where
    the opening shot fails is visible and fixable; a run where it quietly depicts
    something else is neither.

    So the default here is the opposite of the global toggle: the opening shot
    keeps its literal requirement and, if no provider and no phrasing can satisfy
    it, fails as unfillable and is handled by the same recovery the rest of the
    timeline already uses. Operators who would rather have any opening shot than
    none can set this to true.
    """
    value = config.app.get("smart_material_rewrite_opening_shot", False)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def is_cross_group_merge_enabled() -> bool:
    """Allow the merge rescue to cross a semantic group boundary.

    The merge rung was written when a span reliably split into several shots, so
    an unfillable beat almost always had a sibling of its own group beside it.
    Since the span grouper began emitting one single-event requirement per span,
    most spans are short enough to be a single beat — so most beats are the only
    shot of their group, no same-group neighbour exists, and the rescue that was
    meant to be the last rung before failure could not fire at all. A measured
    render lost its whole video to exactly that: the beat that failed was solo.

    Crossing the boundary is sound because the boundary is not a subject change.
    Spans are consecutive slices of one narration, so the neighbouring span
    describes the adjacent moment of the same story, and the absorbed window is
    covered by a clip that was verified against a requirement one moment away
    rather than the same moment. That is a visibly weaker rescue than a same-group
    merge, which is why a same-group neighbour is always preferred, the merged
    beat records a distinct duration policy, and this is switchable: turning it off
    restores failing the video whenever the unfillable beat is solo.
    """
    value = config.app.get("smart_material_cross_group_merge", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def analysis_budget_per_selection_round(candidate_limit: int) -> int:
    """How many stock candidates one selection round may pay to analyze.

    This is a spend ceiling, not a quality knob: it caps *analyses*, because
    searching a catalog is free and looking at a clip is not. An item that can be
    satisfied never approaches it — the selector stops at its first strong
    candidate — so the budget only ever binds on an item that was going to fail
    anyway, and it decides how much that failure is allowed to cost before it is
    reported instead of pursued across every remaining phrasing and provider.

    The ceiling is per round rather than per item so that the alternative-wording
    retry, which is the recovery most likely to actually work, is not starved by
    the failed round that preceded it. An item runs at most two rounds, so its
    worst case is twice this number.

    An absent setting means the multiplier-derived default, an explicit ``0``
    removes the ceiling entirely, and a positive value is an absolute cap. A
    negative value falls back to the default, because "cap the spend at less than
    nothing" has no reading that leaves a run able to select anything.
    """
    default = max(1, int(candidate_limit)) * _DEFAULT_ANALYSIS_BUDGET_MULTIPLIER
    value = config.app.get("smart_material_max_analyzed_candidates_per_round")
    if value is None or not str(value).strip():
        return default
    try:
        budget = int(str(value).strip())
    except (TypeError, ValueError):
        logger.warning(
            "smart_material_max_analyzed_candidates_per_round is not a number; "
            f"using {default}: value={value!r}"
        )
        return default
    if budget < 0:
        logger.warning(
            "smart_material_max_analyzed_candidates_per_round is negative; "
            f"using {default}: value={value!r}"
        )
        return default
    return budget


def analysis_budget_per_video(item_count: int, candidate_limit: int) -> int:
    """How many stock candidates one whole video may pay to analyze.

    The per-round ceiling answers "how much may one failure cost"; this answers
    "how much may one video cost". Both are needed because the failure ladder
    deliberately keeps a damaged run alive: before it, a video died at its first
    unfillable item and the per-round number was also the per-video number, and
    now several items can each spend two full rounds before the video is finished
    or abandoned.

    This ceiling never denies an item its first selection round. It is checked only
    where a run is about to buy *recovery* — another phrasing, another provider,
    the rewritten requirement, or a fresh search for a merged window — so a video
    that has spent its budget still looks once for every remaining item and then
    falls to the free rungs of the ladder instead of paying for the expensive ones.
    That ordering is the whole design: cutting recovery costs cuts money, while
    cutting first looks would silently cut quality.

    An absent setting means the derived default, an explicit ``0`` removes the
    ceiling entirely, and a positive value is an absolute cap. A negative value
    falls back to the default, for the same reason as the per-round budget.
    """
    default = (
        max(1, int(item_count))
        * max(1, int(candidate_limit))
        * _DEFAULT_VIDEO_ANALYSIS_PAGES_PER_ITEM
    )
    value = config.app.get("smart_material_max_analyzed_candidates_per_video")
    if value is None or not str(value).strip():
        return default
    try:
        budget = int(str(value).strip())
    except (TypeError, ValueError):
        logger.warning(
            "smart_material_max_analyzed_candidates_per_video is not a number; "
            f"using {default}: value={value!r}"
        )
        return default
    if budget < 0:
        logger.warning(
            "smart_material_max_analyzed_candidates_per_video is negative; "
            f"using {default}: value={value!r}"
        )
        return default
    return budget


def _effective_round_budget(
    round_budget: int,
    video_budget: int,
    video_spent: int,
) -> int:
    """The per-round ceiling, narrowed by what is left of the video's ceiling.

    Returning ``0`` would mean "no ceiling", so an exhausted video budget must not
    be passed through as zero. It becomes ``1`` instead, which is exactly the
    intended behavior at the selector's seam: the first phrasing on the first
    provider still runs, and every phrasing and provider after it is abandoned
    rather than bought.
    """
    if not video_budget:
        return round_budget
    remaining = max(0, video_budget - max(0, video_spent))
    if not round_budget:
        return max(1, remaining)
    return max(1, min(round_budget, remaining))


def max_merged_beats_per_video(beat_count: int) -> int:
    """How many unfillable beats a neighbour may absorb before the video fails.

    Merging is the last rung of the per-beat failure ladder: when no provider,
    phrasing, or rewritten requirement can fill a beat, a neighbouring shot takes
    over its window — a sibling of the same semantic group where one exists, and
    otherwise the shot of the adjacent group beside it. That costs nothing at the
    video model and keeps the narration timeline intact, but every merge removes a
    cut, so a video that merged most of its beats is no longer the edit that was
    planned. The default therefore allows a third of the beats — enough to survive a
    handful of unlucky requirements, not enough to quietly collapse a 14-shot edit
    into three.

    An absent setting means that default, an explicit ``0`` disables merging and
    restores the previous behavior of failing the video on the first unfillable
    beat, and a positive value is an absolute cap. Raising it past a third is a
    deliberate trade of visual variety for completion, so it is honored as written
    rather than clamped. A negative value falls back to the default, because
    "absorb fewer than no beats" has no reading.
    """
    try:
        total = int(beat_count)
    except (TypeError, ValueError):
        total = 0
    default = max(0, total // 3)
    value = config.app.get("smart_material_max_merged_beats")
    if value is None or not str(value).strip():
        return default
    try:
        ceiling = int(str(value).strip())
    except (TypeError, ValueError):
        logger.warning(
            "smart_material_max_merged_beats is not a number; "
            f"using {default}: value={value!r}"
        )
        return default
    if ceiling < 0:
        logger.warning(
            f"smart_material_max_merged_beats is negative; using {default}: "
            f"value={value!r}"
        )
        return default
    return ceiling


def _item_narration_text(visual_item: OrderedVisualItem) -> str:
    """The words actually spoken over one visual item, or an empty string.

    Beats carry ``spoken_text`` and slots carry narration text under two older
    names. None of them is part of the shared protocol, so this stays tolerant:
    without spoken text there is nothing to re-describe, and the caller treats
    that as "no rewrite available" rather than failing differently.
    """
    for attribute in ("spoken_text", "primary_narration_text", "narration_text"):
        text = " ".join(str(getattr(visual_item, attribute, "") or "").split()).strip()
        if text:
            return text
    return ""


def supports_smart_visual_matching(source: str) -> bool:
    """Report whether a video source can take part in smart visual matching.

    Smart matching needs a searchable stock provider so it can rank candidates
    and pull an exact sub-range. Local files and generated video have neither a
    catalog to search nor an alternative to fall back to. This predicate is the
    single place that decides, so the preflight check, the material download and
    the renderer hand-off can never disagree about which path a task is on.
    """
    return str(source or "").strip().lower() in _STOCK_VIDEO_PROVIDER_API_KEYS


def smart_provider_chain(source: str) -> list[str]:
    """Ordered providers smart matching may try for one visual beat.

    The provider the user picked always goes first, so a working configuration
    keeps its current behavior. The remaining searchable providers follow in the
    fixed cascade order and are only reached when the earlier ones produce no
    candidate that passes the semantic gates. Providers that are not ready to be
    queried are skipped instead of raising mid-timeline.

    Because Pinterest needs no credential it is always ready, so an unconfigured
    task now falls through to it rather than stopping. The fallback below is
    still load-bearing for the one case it cannot cover: with the cascade turned
    off there is nothing to fall through to, and the user has to see the
    actionable key error for the provider they actually selected.
    """
    primary = str(source or "").strip().lower()
    if primary not in _STOCK_VIDEO_PROVIDER_API_KEYS:
        return []
    ordered = [primary]
    if is_provider_cascade_enabled():
        ordered.extend(
            provider
            for provider in _SMART_PROVIDER_CASCADE_ORDER
            if provider != primary
        )
    usable = [provider for provider in ordered if provider_has_api_key(provider)]
    return usable or [primary]


def _remote_search_function(provider: str) -> Callable[..., List[MaterialInfo]]:
    # Resolved through the module globals on every call so a test double patched
    # onto one of these names is the function the cascade actually uses.
    if provider == "pinterest":
        return search_videos_pinterest
    if provider == "pixabay":
        return search_videos_pixabay
    return search_videos_pexels


def _cached_provider_search(provider: str) -> Callable[..., List[MaterialInfo]]:
    """Bind one provider to the shared search cache entry point."""
    remote_search_videos = _remote_search_function(provider)

    def search_videos(
        search_term: str,
        minimum_duration: int,
        video_aspect: VideoAspect,
    ) -> List[MaterialInfo]:
        return _search_videos_with_cache(
            provider=provider,
            search_videos=remote_search_videos,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )

    return search_videos


def download_videos(
    task_id: str,
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    match_script_order: bool = False,
    visual_slots: list[VisualSlot] | None = None,
    visual_beats: list[VisualBeat] | None = None,
    clip_speed: float = 1.0,
    requirement_specs: dict[str, VisualRequirementSpec] | None = None,
    merged_beats_out: list[VisualBeat] | None = None,
) -> List[str]:
    provider = str(source or "").strip().lower()
    if provider not in _STOCK_VIDEO_PROVIDER_API_KEYS:
        provider = "pexels"
    search_videos = _cached_provider_search(provider)
    provider_searches = [
        (chain_provider, _cached_provider_search(chain_provider))
        for chain_provider in smart_provider_chain(provider)
    ]

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""

    if match_script_order:
        return _download_videos_by_script_order(
            task_id=task_id,
            search_terms=search_terms,
            search_videos=search_videos,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            visual_slots=visual_slots,
            visual_beats=visual_beats,
            clip_speed=clip_speed,
            provider_searches=provider_searches,
            requirement_specs=requirement_specs,
            merged_beats_out=merged_beats_out,
        )

    valid_video_items = []
    valid_video_urls = []
    found_duration = 0.0
    for search_term in search_terms:
        video_items = search_videos(
            search_term=search_term,
            minimum_duration=max_clip_duration,
            video_aspect=video_aspect,
        )
        logger.info(f"found {len(video_items)} videos for '{search_term}'")

        for item in video_items:
            if item.url not in valid_video_urls:
                valid_video_items.append(item)
                valid_video_urls.append(item.url)
                found_duration += item.duration

    logger.info(
        f"found total videos: {len(valid_video_items)}, required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    video_paths = []
    material_sources: list[dict[str, Any]] = []

    concat_mode_value = getattr(video_concat_mode, "value", video_concat_mode)
    if concat_mode_value == VideoConcatMode.random.value:
        random.shuffle(valid_video_items)

    total_duration = 0.0
    for item in valid_video_items:
        try:
            source_info = item.source_info if isinstance(item.source_info, dict) else {}
            logger.info(
                f"downloading {item.provider} video: "
                f"asset_id={source_info.get('asset_id') or 'unknown'}"
            )
            saved_video_path = save_video(
                video_url=item.url,
                save_dir=material_directory,
                video_aspect=video_aspect,
            )
            if saved_video_path:
                logger.info(f"video saved: {saved_video_path}")
                video_paths.append(saved_video_path)
                try:
                    material_sources.append(
                        _material_source_record(item, saved_video_path)
                    )
                except Exception as source_error:
                    # 来源记录异常不能把已经成功下载的素材视为下载失败，更不能
                    # 阻断视频生成；保留供应商和异常类型用于后续定位。
                    logger.warning(
                        "failed to prepare material source record: "
                        f"provider={item.provider}, "
                        f"error={type(source_error).__name__}, detail={source_error}"
                    )
                seconds = min(max_clip_duration, item.duration)
                total_duration += seconds
                if total_duration > audio_duration:
                    logger.info(
                        f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                    )
                    break
        except Exception as e:
            logger.error(
                "failed to download material video: "
                f"provider={item.provider}, error={type(e).__name__}, "
                f"detail={_redact_request_error(e, item.url)}"
            )
    logger.success(f"downloaded {len(video_paths)} videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _build_script_order_term_plan(
    term_count: int,
    required_clip_count: int,
) -> List[int]:
    """Map timeline clip slots to search-term indexes without rewinding.

    When a narration needs more clips than there are search terms, the old
    round-robin downloader restarted at the opening term after reaching the
    final term.  That made the second half of a video visually jump back to
    the beginning of the script.  This plan spreads repeated terms across one
    monotonic pass instead.

    If there are fewer clip slots than terms, sample the full script range and
    keep both the opening and final term represented.
    """
    if term_count <= 0 or required_clip_count <= 0:
        return []
    if required_clip_count == 1:
        return [0]

    if required_clip_count >= term_count:
        return [
            min(term_count - 1, (slot * term_count) // required_clip_count)
            for slot in range(required_clip_count)
        ]

    return [
        round(slot * (term_count - 1) / (required_clip_count - 1))
        for slot in range(required_clip_count)
    ]


@dataclass
class _SmartProviderAttempt:
    """What one stock provider produced for one ordered visual item."""

    winner: MaterialInfo | None = None
    segment: dict[str, Any] | None = None
    verifier_runs: list[dict[str, Any]] = dataclass_field(default_factory=list)
    failure_reason: str = ""
    candidates_analyzed: int = 0
    source_seconds_analyzed: float = 0.0
    segmentation_calls: int = 0
    had_candidates: bool = False
    # The provider's own relevance ranking put nothing near the gate. Rewording
    # the same concept re-searches the same catalog and buys the same pool, so
    # the caller stops spending phrasings here and escalates instead.
    unrelated_footage: bool = False
    # Assets this attempt already paid a verdict for. The next phrasing of the
    # same item must not buy the same verdict twice.
    evaluated_identities: list[tuple[str, str]] = dataclass_field(default_factory=list)
    # Candidates verification approved but did not rank first, best first. Their
    # verdicts are already paid for, so a winner whose media transfer fails can
    # promote one of them without buying a second analysis.
    approved_alternates: list[MaterialInfo] = dataclass_field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return (
            not self.failure_reason
            and self.winner is not None
            and self.segment is not None
        )


def _approved_alternate_candidates(
    candidates: Sequence[MaterialInfo],
    candidate_evaluations: Sequence[Any],
    winner: MaterialInfo | None,
) -> list[MaterialInfo]:
    """Verified runner-ups for one attempt, best ranked first.

    Only candidates verification actually approved qualify. A rejected candidate
    is not a fallback: promoting one would put footage into the video that
    verification refused, which is the opposite of a per-beat fail-closed policy.
    """
    winner_identity = _provider_asset_identity(winner) if winner is not None else None
    candidates_by_identity: dict[tuple[str, str], MaterialInfo] = {}
    for candidate in candidates:
        identity = _provider_asset_identity(candidate)
        if identity is None or identity in candidates_by_identity:
            continue
        candidates_by_identity[identity] = candidate

    ranked: list[tuple[int, MaterialInfo]] = []
    seen: set[tuple[str, str]] = set()
    for position, evaluation in enumerate(candidate_evaluations):
        if not isinstance(evaluation, dict) or evaluation.get("accepted") is not True:
            continue
        identity = (
            str(evaluation.get("provider") or "").strip(),
            str(evaluation.get("provider_asset_id") or "").strip(),
        )
        if not identity[0] or not identity[1]:
            continue
        if identity == winner_identity or identity in seen:
            continue
        candidate = candidates_by_identity.get(identity)
        if candidate is None:
            continue
        seen.add(identity)
        try:
            rank = int(evaluation["ranking_position"])
        except (KeyError, TypeError, ValueError):
            # Ranking metadata is provenance, not a contract. Without it the
            # evaluation order is still a usable order.
            rank = len(candidate_evaluations) + position + 1
        ranked.append((rank, candidate))

    ranked.sort(key=lambda entry: entry[0])
    return [candidate for _, candidate in ranked]


def _segment_smart_candidate(
    *,
    candidate: MaterialInfo,
    visual_item: OrderedVisualItem,
    requirement: str,
    required_source_duration: float,
    normalized_speed: float,
    settings: dict[str, Any],
    twelvelabs_service,
    item_name: str,
    item_log_name: str,
) -> tuple[dict[str, Any] | None, str]:
    """Return the source window to render for one candidate, or why there is none.

    Segmentation stays winner-only: this runs for the candidate that is about to
    join the timeline, never for the field of candidates being judged. The
    requirement is passed in rather than read from the item, because a recovered
    item is verified and segmented against its rewritten requirement.
    """
    asset_identity = _provider_asset_identity(candidate)
    asset_id = asset_identity[1] if asset_identity is not None else ""
    try:
        segment = twelvelabs_service.segment_winner(
            video_url=candidate.url,
            narration_text=requirement,
            slot_duration=visual_item.duration,
            source_duration=float(candidate.duration),
            clip_speed=normalized_speed,
            requested_source_duration=required_source_duration,
        )
    except twelvelabs_service.TemporalSegmentationError as exc:
        if settings["fail_closed"]:
            if exc.category == "auth_quota":
                return None, (
                    f"{item_name} {visual_item.index} temporal segmentation "
                    f"failed: {exc}"
                )
            return None, (
                f"{item_name} {visual_item.index} temporal segmentation is "
                "temporarily "
                f"unavailable: {exc}"
            )
        logger.warning(
            "winner segmentation service failed; explicit fail-open is enabled: "
            f"{item_log_name}={visual_item.index}, reason={exc}"
        )
        segment = None
    if segment is not None:
        segment = _normalize_selected_source_range(
            segment,
            source_duration=float(candidate.duration),
            required_source_duration=required_source_duration,
        )
    if segment is None and settings["fail_closed"]:
        return None, (
            "No valid TwelveLabs temporal segment matched narration for "
            f"{item_log_name} {visual_item.index}"
        )
    if segment is None:
        segment = {
            "source_start_time": 0.0,
            "source_end_time": min(float(candidate.duration), required_source_duration),
            "description": "explicit fail-open zero-start fallback",
        }
        logger.warning(
            "winner segmentation unavailable; using explicit fail-open source "
            f"start for {item_log_name}={visual_item.index}, "
            f"asset_id={asset_id or 'unknown'}"
        )
    return segment, ""


def _attempt_smart_provider_selection(
    *,
    provider: str,
    search_videos: Callable[..., List[MaterialInfo]],
    visual_item: OrderedVisualItem,
    requirement: str,
    search_query: str,
    requirement_spec,
    required_source_duration: float,
    video_aspect: VideoAspect,
    settings: dict[str, Any],
    candidate_limit: int,
    normalized_speed: float,
    twelvelabs_service,
    used_asset_identities: set[tuple[str, str]],
    used_urls: set[str],
    item_name: str,
    item_log_name: str,
) -> _SmartProviderAttempt:
    """Search one stock provider and return its winner, or why it produced none.

    Every gate that can reject a provider is reported through
    ``failure_reason`` instead of raising, so the caller can try the next
    provider in the cascade before giving up on the whole timeline.

    ``requirement`` is the wording this attempt is judged against. It is normally
    the item's own requirement and differs from it only for a recovery round, so
    the persisted run records which wording actually bought the analysis.
    """
    attempt = _SmartProviderAttempt()
    run: dict[str, Any] = {
        "visual_item_type": item_log_name.replace(" ", "_"),
        "visual_item_index": visual_item.index,
        "visual_requirement": requirement,
        "search_query": search_query,
        "stock_provider": provider,
        "visual_requirement_spec": (
            llm.visual_requirement_spec_to_dict(requirement_spec)
            if requirement_spec is not None
            else None
        ),
        "candidate_evaluations": [],
    }
    semantic_group_id = getattr(visual_item, "semantic_group_id", None)
    if isinstance(semantic_group_id, int) and semantic_group_id > 0:
        run["semantic_group_id"] = semantic_group_id

    video_items = search_videos(
        search_term=search_query,
        minimum_duration=math.ceil(max(4.0, required_source_duration)),
        video_aspect=video_aspect,
    )
    prepared = _prepare_twelvelabs_candidates(
        video_items,
        video_aspect=video_aspect,
        required_source_duration=required_source_duration,
        preferred_max_source_duration=settings["preferred_max_source_duration"],
    )
    candidates = []
    for item in prepared:
        if _is_duplicate_material(item, used_asset_identities, used_urls):
            continue
        candidates.append(item)

    narration_preview = re.sub(r"\s+", " ", requirement).strip()[:160]
    logger.info(
        f"smart {item_log_name} candidates: "
        f"index={visual_item.index}, provider={provider}, "
        f"narration={narration_preview!r}, "
        f"query={search_query!r}, after_cheap_filters={len(candidates)}"
    )
    if not candidates:
        run["final_decision"] = "NO_CANDIDATES"
        attempt.verifier_runs.append(run)
        attempt.failure_reason = (
            f"No valid {provider} candidate remained for {item_log_name} "
            f"{visual_item.index} "
            "after metadata quality filters"
        )
        return attempt

    attempt.had_candidates = True
    if requirement_spec is None:
        winner = None
        stats: dict[str, Any] = {
            "api_candidates_analyzed": 0,
            "source_seconds_analyzed": 0.0,
            "candidate_evaluations": [],
            "api_failure_reason": "visual requirement decomposition was unavailable",
        }
    else:
        winner, stats = twelvelabs_service.select_best_candidate(
            candidates=candidates,
            slot_index=visual_item.index,
            slot_duration=visual_item.duration,
            narration_text=requirement,
            search_query=search_query,
            requirement_spec=requirement_spec,
            batch_size=settings["batch_size"],
            max_candidates=candidate_limit,
            minimum_score=settings["minimum_score"],
            strong_early_stop_score=settings["strong_early_stop_score"],
            concurrency=settings["concurrency"],
        )
    attempt.candidates_analyzed = int(stats["api_candidates_analyzed"])
    attempt.source_seconds_analyzed = float(stats["source_seconds_analyzed"])
    attempt.unrelated_footage = bool(stats.get("unrelated_footage"))
    run["candidate_evaluations"] = stats.get("candidate_evaluations", [])
    # Only the candidates that actually reached a verdict are reported back. The
    # tail this attempt never looked at stays available to the next phrasing,
    # which is the whole reason another phrasing is worth trying.
    for evaluation in run["candidate_evaluations"]:
        if not isinstance(evaluation, dict):
            continue
        evaluated_provider = str(evaluation.get("provider") or "").strip()
        evaluated_asset_id = str(evaluation.get("provider_asset_id") or "").strip()
        if evaluated_provider and evaluated_asset_id:
            attempt.evaluated_identities.append(
                (evaluated_provider, evaluated_asset_id)
            )
    run["final_decision"] = "ACCEPT" if winner is not None else "REJECT"

    if winner is None and not settings["fail_closed"]:
        # Explicit fail-open is honored on the first provider that returned any
        # candidate at all: the operator asked for output over quality, so
        # spending further provider and analysis budget would be pointless.
        winner = candidates[0]
        run["final_decision"] = "FAIL_OPEN_FALLBACK"
        logger.warning(
            "no TwelveLabs candidate passed; using explicit fail-open legacy "
            f"fallback for {item_log_name}={visual_item.index}"
        )
    if winner is None:
        attempt.verifier_runs.append(run)
        api_failure_reason = stats.get("api_failure_reason")
        if api_failure_reason:
            attempt.failure_reason = (
                f"{item_name} {visual_item.index} could not be analyzed: "
                f"{api_failure_reason}"
            )
        else:
            attempt.failure_reason = (
                f"No TwelveLabs candidate satisfied narration for {item_log_name} "
                f"{visual_item.index}; cross-segment fallback was not used"
            )
        return attempt

    winner_identity = _provider_asset_identity(winner)
    if winner_identity is not None:
        run["winner"] = {
            "provider": winner_identity[0],
            "provider_asset_id": winner_identity[1],
            "overall_score": round(float(winner.overall_score or 0.0), 4),
        }
    attempt.verifier_runs.append(run)
    attempt.winner = winner
    attempt.approved_alternates = _approved_alternate_candidates(
        candidates,
        run["candidate_evaluations"],
        winner,
    )

    attempt.segmentation_calls = 1
    segment, segment_failure = _segment_smart_candidate(
        candidate=winner,
        visual_item=visual_item,
        requirement=requirement,
        required_source_duration=required_source_duration,
        normalized_speed=normalized_speed,
        settings=settings,
        twelvelabs_service=twelvelabs_service,
        item_name=item_name,
        item_log_name=item_log_name,
    )
    if segment is None:
        attempt.failure_reason = segment_failure
        return attempt
    attempt.segment = segment
    return attempt


def _ordered_item_search_queries(
    visual_item: OrderedVisualItem,
    primary_query: str,
    limit: int,
) -> list[str]:
    """Phrasings to try for one visual item, in planned order.

    The planned query stays first, so a configuration that already produced a
    good clip keeps producing the same clip. The remaining phrasings the script
    stage generated for this item follow in their own order; duplicates and blank
    entries are dropped, because repeating a phrasing buys a second bill and no
    new candidates.
    """
    ordered = [primary_query]
    seen = {primary_query}
    for query in visual_item.search_queries or []:
        variant = str(query or "").strip()
        if not variant or variant in seen:
            continue
        seen.add(variant)
        ordered.append(variant)
    return ordered[: max(1, limit)]


@dataclass
class _SmartItemSelection:
    """What one full round of provider and phrasing attempts produced for an item.

    A round is the unit that can be repeated: the same item may be selected for
    twice, once against its planned requirement and once against a rewritten one.
    Usage counters and provenance are kept per round so the caller can accumulate
    them without knowing how many rounds ran.
    """

    attempt: _SmartProviderAttempt | None = None
    provider: str = ""
    query: str = ""
    requirement: str = ""
    failures: list[str] = dataclass_field(default_factory=list)
    verifier_runs: list[dict[str, Any]] = dataclass_field(default_factory=list)
    candidates_analyzed: int = 0
    source_seconds_analyzed: float = 0.0
    segmentation_calls: int = 0


def _select_smart_item_winner(
    *,
    visual_item: OrderedVisualItem,
    requirement: str,
    requirement_spec,
    item_queries: Sequence[str],
    provider_chain: Sequence[tuple[str, Callable[..., List[MaterialInfo]]]],
    required_source_duration: float,
    video_aspect: VideoAspect,
    settings: dict[str, Any],
    candidate_limit: int,
    normalized_speed: float,
    twelvelabs_service,
    item_asset_identities: set[tuple[str, str]],
    item_urls: set[str],
    item_name: str,
    item_log_name: str,
    analysis_budget: int = 0,
) -> _SmartItemSelection:
    """Try every phrasing on every provider for one item, best effort, no raising.

    ``item_asset_identities`` is updated in place, so a second round for the same
    item never re-buys a verdict the first round already paid for.

    ``analysis_budget`` is a spend ceiling in analyzed candidates. ``0`` means no
    ceiling. It is checked before opening each new phrasing or provider, never
    mid-attempt, so a search already dispatched still finishes and an item that
    settles inside its first batch never notices the budget at all.
    """
    selection = _SmartItemSelection(
        query=item_queries[0] if item_queries else "",
        requirement=requirement,
    )
    budget_exhausted = False
    for provider_name, provider_search in provider_chain:
        provider_settled = False
        for query_position, item_query in enumerate(item_queries):
            if analysis_budget and selection.candidates_analyzed >= analysis_budget:
                budget_exhausted = True
                break
            attempt = _attempt_smart_provider_selection(
                provider=provider_name,
                search_videos=provider_search,
                visual_item=visual_item,
                requirement=requirement,
                search_query=item_query,
                requirement_spec=requirement_spec,
                required_source_duration=required_source_duration,
                video_aspect=video_aspect,
                settings=settings,
                candidate_limit=candidate_limit,
                normalized_speed=normalized_speed,
                twelvelabs_service=twelvelabs_service,
                used_asset_identities=item_asset_identities,
                used_urls=item_urls,
                item_name=item_name,
                item_log_name=item_log_name,
            )
            selection.candidates_analyzed += attempt.candidates_analyzed
            selection.source_seconds_analyzed += attempt.source_seconds_analyzed
            selection.segmentation_calls += attempt.segmentation_calls
            selection.verifier_runs.extend(attempt.verifier_runs)
            item_asset_identities.update(attempt.evaluated_identities)
            if attempt.succeeded:
                selection.attempt = attempt
                selection.provider = provider_name
                selection.query = item_query
                break
            failure_context = []
            if len(provider_chain) > 1:
                failure_context.append(f"provider={provider_name}")
            if len(item_queries) > 1:
                failure_context.append(f"query={item_query!r}")
            selection.failures.append(
                f"{', '.join(failure_context)}: {attempt.failure_reason}"
                if failure_context
                else attempt.failure_reason
            )
            if attempt.had_candidates and not settings["fail_closed"]:
                # Fail-open already accepted this provider's best available
                # candidate, so neither another phrasing nor another provider
                # can improve the result.
                provider_settled = True
                break
            if attempt.unrelated_footage:
                # This catalog's own ranking says it holds nothing close. Another
                # phrasing of the same concept re-searches the same catalog and
                # buys the same pool, so the remaining phrasings are skipped and
                # the next provider -- a genuinely different catalog -- gets its
                # turn instead. The requirement rewrite above still runs, and it
                # is the rung that can actually change the concept.
                logger.warning(
                    "stock provider footage was uniformly unrelated; skipping the "
                    "remaining phrasings on this provider: "
                    f"{item_log_name}={visual_item.index}, "
                    f"provider={provider_name}, query={item_query!r}, "
                    f"skipped_phrasings={len(item_queries) - query_position - 1}"
                )
                break
            if query_position + 1 < len(item_queries):
                logger.warning(
                    "search query produced no usable winner; trying the next "
                    f"phrasing on the same provider: "
                    f"{item_log_name}={visual_item.index}, "
                    f"provider={provider_name}, query={item_query!r}, "
                    f"reason={attempt.failure_reason}"
                )
        if selection.attempt is not None or provider_settled or budget_exhausted:
            break
        if len(provider_chain) > 1:
            logger.warning(
                "stock provider produced no usable winner for any phrasing; "
                "trying the next provider: "
                f"{item_log_name}={visual_item.index}, "
                f"provider={provider_name}, queries={len(item_queries)}"
            )
    if budget_exhausted and selection.attempt is None:
        # The round was not going to be satisfied and had already cost its whole
        # budget; the remaining phrasings and providers are abandoned, not bought.
        logger.warning(
            "analysis budget reached before a winner; abandoning the remaining "
            f"phrasings and providers: {item_log_name}={visual_item.index}, "
            f"analyzed={selection.candidates_analyzed}, budget={analysis_budget}"
        )
        selection.failures.append(
            f"analysis budget of {analysis_budget} analyzed candidates reached for "
            f"{item_log_name} {visual_item.index}; remaining phrasings and "
            "providers were not requested"
        )
        # Persisted, not only raised: a round the ceiling cut short may still be
        # rescued by the rewrite, and a run that silently stopped looking would
        # otherwise be indistinguishable from one that searched everything.
        selection.verifier_runs.append(
            {
                "visual_item_type": item_log_name.replace(" ", "_"),
                "visual_item_index": visual_item.index,
                "visual_requirement": requirement,
                "final_decision": "ANALYSIS_BUDGET_EXHAUSTED",
                "candidates_analyzed": selection.candidates_analyzed,
                "analysis_budget": analysis_budget,
            }
        )
    return selection


def _alternative_item_search_queries(
    visual_item: OrderedVisualItem,
    requirement: str,
    narration_text: str,
    limit: int,
) -> list[str]:
    """Stock queries for a rewritten requirement.

    The phrasings the script stage planned belong to the requirement that just
    failed, so reusing them would search for the wording being abandoned. Fresh
    ones are requested for the new wording instead; if that request fails, a short
    requirement is itself a usable query and a long one is not.
    """
    start_time = float(getattr(visual_item, "start_time", 0.0) or 0.0)
    end_time = float(
        getattr(visual_item, "end_time", 0.0) or start_time + float(visual_item.duration)
    )
    try:
        generated = llm.generate_visual_slot_queries(
            video_subject=narration_text,
            visual_slots=[
                {
                    "slot_index": visual_item.index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "visual_requirement": requirement,
                }
            ],
            queries_per_slot=max(1, limit),
        )
    except Exception as exc:
        logger.warning(
            "could not generate queries for a rewritten visual requirement: "
            f"index={visual_item.index}, error={type(exc).__name__}: {exc}"
        )
        generated = {}

    queries: list[str] = []
    for query in (generated or {}).get(visual_item.index) or []:
        text = str(query or "").strip()
        if text and text not in queries:
            queries.append(text)
    if queries:
        return queries[: max(1, limit)]
    words = requirement.split()
    if 1 <= len(words) <= _MAX_REQUIREMENT_QUERY_WORDS:
        return [" ".join(words)]
    return []


def _rewrite_requirement_and_reselect(
    *,
    visual_item: OrderedVisualItem,
    failed_requirement: str,
    failure_summary: str,
    query_limit: int,
    provider_chain: Sequence[tuple[str, Callable[..., List[MaterialInfo]]]],
    required_source_duration: float,
    video_aspect: VideoAspect,
    settings: dict[str, Any],
    candidate_limit: int,
    normalized_speed: float,
    twelvelabs_service,
    item_asset_identities: set[tuple[str, str]],
    item_urls: set[str],
    item_name: str,
    item_log_name: str,
    analysis_budget: int = 0,
) -> _SmartItemSelection:
    """Describe one unfillable item a second way and select for it once more.

    The narration is what the video promised; the requirement is only one reading
    of it. So the recovery keeps the narration fixed and replaces the reading,
    which is why the alternative must quote the item's own spoken text to be
    accepted. Exactly one alternative is tried, and the returned selection carries
    the wording that was used so the caller segments and logs the same reading.
    """
    item_type = item_log_name.replace(" ", "_")

    def _unavailable(reason: str) -> _SmartItemSelection:
        return _SmartItemSelection(
            requirement=failed_requirement,
            failures=[
                f"No alternative visual requirement was usable for "
                f"{item_log_name} {visual_item.index}: {reason}"
            ],
            verifier_runs=[
                {
                    "visual_item_type": item_type,
                    "visual_item_index": visual_item.index,
                    "visual_requirement": failed_requirement,
                    "final_decision": "REQUIREMENT_REWRITE_UNAVAILABLE",
                    "reason": reason,
                }
            ],
        )

    narration_text = _item_narration_text(visual_item)
    if not narration_text:
        return _unavailable("the item carries no spoken text to re-describe")

    with llm.record_provider_availability() as rewrite_failure:
        alternatives = llm.generate_alternative_visual_requirements(
            [
                {
                    "item_index": visual_item.index,
                    "narration_text": narration_text,
                    "failed_requirement": failed_requirement,
                    "problem": failure_summary,
                }
            ]
        )
    alternative = alternatives.get(visual_item.index) or {}
    alternative_requirement = str(alternative.get("visual_requirement") or "").strip()
    if not alternative_requirement:
        if rewrite_failure.get("provider_unavailable"):
            return _unavailable(
                "the LLM provider was unavailable while re-describing it; "
                "check the provider quota or key"
            )
        return _unavailable("no grounded alternative wording was returned")

    with llm.record_provider_availability() as decompose_failure:
        requirement_spec = llm.generate_visual_requirement_specs(
            [alternative_requirement]
        ).get(llm.normalize_visual_requirement(alternative_requirement))
    if requirement_spec is None and settings["fail_closed"]:
        if decompose_failure.get("provider_unavailable"):
            return _unavailable(
                "the LLM provider was unavailable while decomposing it, so the "
                "wording was never judged; check the provider quota or key"
            )
        return _unavailable("the alternative wording could not be decomposed either")

    item_queries = _alternative_item_search_queries(
        visual_item,
        alternative_requirement,
        narration_text,
        query_limit,
    )
    if not item_queries:
        return _unavailable("no searchable query could be derived from it")

    logger.warning(
        "re-describing an unfillable visual requirement and searching once more: "
        f"{item_log_name}={visual_item.index}, "
        f"rejected={failed_requirement!r}, alternative={alternative_requirement!r}, "
        f"queries={len(item_queries)}"
    )
    selection = _select_smart_item_winner(
        visual_item=visual_item,
        requirement=alternative_requirement,
        requirement_spec=requirement_spec,
        item_queries=item_queries,
        provider_chain=provider_chain,
        required_source_duration=required_source_duration,
        video_aspect=video_aspect,
        settings=settings,
        candidate_limit=candidate_limit,
        normalized_speed=normalized_speed,
        twelvelabs_service=twelvelabs_service,
        item_asset_identities=item_asset_identities,
        item_urls=item_urls,
        item_name=item_name,
        item_log_name=item_log_name,
        analysis_budget=analysis_budget,
    )
    selection.verifier_runs.insert(
        0,
        {
            "visual_item_type": item_type,
            "visual_item_index": visual_item.index,
            "visual_requirement": failed_requirement,
            "final_decision": "REQUIREMENT_REWRITTEN",
            "alternative_visual_requirement": alternative_requirement,
            "narration_basis": str(alternative.get("narration_basis") or ""),
        },
    )
    return selection


@dataclass
class _SmartDownloadResult:
    """What transferring one approved selection to disk produced."""

    video_path: str = ""
    winner: MaterialInfo | None = None
    verifier_runs: list[dict[str, Any]] = dataclass_field(default_factory=list)
    failures: list[str] = dataclass_field(default_factory=list)
    segmentation_calls: int = 0


def _download_smart_winner(
    *,
    attempt: _SmartProviderAttempt,
    visual_item: OrderedVisualItem,
    requirement: str,
    provider: str,
    query: str,
    required_target_duration: float,
    required_source_duration: float,
    normalized_speed: float,
    settings: dict[str, Any],
    twelvelabs_service,
    material_directory: str,
    video_aspect: VideoAspect,
    is_visual_beat: bool,
    item_name: str,
    item_log_name: str,
) -> _SmartDownloadResult:
    """Transfer the approved selection for one visual item, or report why none arrived.

    A verified winner whose media transfer fails is a transport problem, not a
    verification problem, and it must not cost the whole video. Candidates this
    item already approved are promoted in ranking order; each one only needs its
    own source window, so segmentation stays winner-only.

    ``visual_item`` is the item the window is being cut for, which is not always
    the item the analysis was bought for: a merged beat reuses a neighbour's already
    approved asset and asks only for a longer window out of it.
    """
    result = _SmartDownloadResult()
    winner = attempt.winner
    segment = attempt.segment
    if winner is None or segment is None:
        # Unreachable through _SmartProviderAttempt.succeeded; kept so a future
        # refactor cannot silently render an unbound source window.
        raise SmartMaterialSelectionError(
            f"{item_name} {visual_item.index} produced an incomplete "
            "smart selection result"
        )

    download_plan: list[tuple[MaterialInfo, dict[str, Any] | None]] = [
        (winner, segment)
    ]
    download_plan.extend(
        (alternate, None) for alternate in attempt.approved_alternates
    )
    for plan_position, (candidate, candidate_segment) in enumerate(download_plan):
        if candidate_segment is None:
            result.segmentation_calls += 1
            candidate_segment, segment_failure = _segment_smart_candidate(
                candidate=candidate,
                visual_item=visual_item,
                requirement=requirement,
                required_source_duration=required_source_duration,
                normalized_speed=normalized_speed,
                settings=settings,
                twelvelabs_service=twelvelabs_service,
                item_name=item_name,
                item_log_name=item_log_name,
            )
            if candidate_segment is None:
                result.failures.append(segment_failure)
                continue
        candidate_identity = _provider_asset_identity(candidate)
        candidate_asset_id = (
            candidate_identity[1] if candidate_identity is not None else ""
        )
        source = (
            candidate.source_info if isinstance(candidate.source_info, dict) else {}
        )

        candidate.source_start_time = float(candidate_segment["source_start_time"])
        candidate.source_end_time = float(candidate_segment["source_end_time"])
        source = dict(source)
        source["slot_index"] = visual_item.index
        if is_visual_beat:
            source["visual_beat_index"] = visual_item.index
            source["semantic_group_id"] = int(getattr(visual_item, "semantic_group_id"))
        source["required_target_duration"] = required_target_duration
        source["required_source_duration"] = required_source_duration
        source["temporal_segment"] = dict(candidate_segment)
        candidate.source_info = source
        candidate.search_query = query
        logger.info(
            "smart visual winner: "
            f"{item_log_name}={visual_item.index}, provider={provider}, "
            f"query={query!r}, "
            f"asset_id={candidate_asset_id or 'unknown'}, "
            f"score={float(candidate.overall_score or 0):.4f}, "
            f"source_start={candidate.source_start_time:.3f}, "
            f"source_end={candidate.source_end_time:.3f}"
        )

        saved_video_path = save_video(
            video_url=candidate.url,
            save_dir=material_directory,
            video_aspect=video_aspect,
        )
        if not saved_video_path:
            result.failures.append(
                f"The selected {provider} winner for {item_log_name} "
                f"{visual_item.index} could not "
                "be downloaded"
            )
            continue

        # Everything above this line is a model's opinion about a URL. The file
        # now exists locally, so the two defects a model cannot be trusted on —
        # a window that spans a cut, and footage that is itself an advertisement
        # — are measured from its pixels before it is allowed onto the timeline.
        inspection = shot_integrity.inspect_downloaded_clip(
            saved_video_path,
            source_duration=max(
                float(candidate.duration or 0.0), candidate.source_end_time
            ),
            start_time=candidate.source_start_time,
            end_time=candidate.source_end_time,
            verified_start=candidate_segment.get("verified_start_time"),
            verified_end=candidate_segment.get("verified_end_time"),
        )
        source = dict(candidate.source_info or {})
        source["local_clip_check"] = dict(inspection.evidence)
        candidate.source_info = source
        if inspection.rejected:
            logger.warning(
                "local inspection rejected an approved clip: "
                f"{item_log_name}={visual_item.index}, provider={provider}, "
                f"asset_id={candidate_asset_id or 'unknown'}, "
                f"reason={inspection.rejection_reason}"
            )
            result.verifier_runs.append(
                {
                    "visual_item_type": item_log_name.replace(" ", "_"),
                    "visual_item_index": visual_item.index,
                    "visual_requirement": requirement,
                    "stock_provider": provider,
                    "search_query": query,
                    "provider_asset_id": candidate_asset_id,
                    "final_decision": "LOCAL_CLIP_REJECTED",
                    "rejection_reason": inspection.rejection_reason,
                    "local_clip_check": dict(inspection.evidence),
                }
            )
            result.failures.append(
                f"The selected {provider} clip for {item_log_name} "
                f"{visual_item.index} was rejected locally: "
                f"{inspection.rejection_reason}"
            )
            continue
        if abs(inspection.start_time - candidate.source_start_time) > 1e-3:
            logger.info(
                "pulled the rendered window back inside one shot: "
                f"{item_log_name}={visual_item.index}, "
                f"{candidate.source_start_time:.3f}..{candidate.source_end_time:.3f}"
                f" -> {inspection.start_time:.3f}..{inspection.end_time:.3f}"
            )
            candidate.source_start_time = inspection.start_time
            candidate.source_end_time = inspection.end_time
            segment_record = dict(source.get("temporal_segment") or {})
            segment_record["source_start_time"] = inspection.start_time
            segment_record["source_end_time"] = inspection.end_time
            source["temporal_segment"] = segment_record
            candidate.source_info = source

        if plan_position:
            logger.warning(
                "the ranked winner did not survive download and local "
                "inspection; promoted an already approved candidate for this "
                f"{item_log_name}: index={visual_item.index}, "
                f"provider={provider}, "
                f"asset_id={candidate_asset_id or 'unknown'}, "
                f"plan_position={plan_position + 1}"
            )
            result.verifier_runs.append(
                {
                    "visual_item_type": item_log_name.replace(" ", "_"),
                    "visual_item_index": visual_item.index,
                    "visual_requirement": requirement,
                    "stock_provider": provider,
                    "search_query": query,
                    "final_decision": "WINNER_DOWNLOAD_SUBSTITUTED",
                    "promoted_plan_position": plan_position + 1,
                }
            )
        result.video_path = saved_video_path
        result.winner = candidate
        return result
    return result


_MERGE_TIME_TOLERANCE_SECONDS = 1e-6


@dataclass
class _SmartItemOutcome:
    """What one visual item ended up with, before merges are resolved.

    Winners used to be appended to the timeline the moment they downloaded, which
    meant the first unfillable item had to fail the whole video: there was nowhere
    to record "this one is still open" and nothing left to reconsider. Keeping one
    outcome per item instead lets an unfillable item stay open long enough for a
    neighbouring shot to absorb its window.
    """

    visual_item: OrderedVisualItem
    requirement: str
    beat: VisualBeat | None = None
    provider: str = ""
    query: str = ""
    video_path: str = ""
    winner: MaterialInfo | None = None
    failures: list[str] = dataclass_field(default_factory=list)
    # The index of the beat that absorbed this one, or 0 while it is still its own
    # beat. An absorbed item contributes no clip and no record.
    absorbed_by: int = 0
    absorbed_indexes: list[int] = dataclass_field(default_factory=list)

    @property
    def filled(self) -> bool:
        return bool(self.video_path) and self.winner is not None


def _records_from_outcomes(
    outcomes: Sequence[_SmartItemOutcome],
) -> list[dict[str, Any]]:
    """Persisted source records for the items that actually reached the timeline.

    An item that is still open, or that a neighbour absorbed, contributes no clip and
    therefore no record — so ``material_sources`` stays one record per rendered
    beat even when the timeline no longer matches the plan.
    """
    return [
        _material_source_record(outcome.winner, outcome.video_path)
        for outcome in outcomes
        if outcome.filled and outcome.winner is not None
    ]


@dataclass
class _MergeResolution:
    """What resolving the open items by merging produced."""

    merged: int = 0
    verifier_runs: list[dict[str, Any]] = dataclass_field(default_factory=list)
    failures: list[str] = dataclass_field(default_factory=list)
    candidates_analyzed: int = 0
    source_seconds_analyzed: float = 0.0
    segmentation_calls: int = 0


def _beats_are_adjacent(earlier: VisualBeat, later: VisualBeat) -> bool:
    """Whether one beat ends exactly where the other begins."""
    return math.isclose(
        earlier.end_time,
        later.start_time,
        rel_tol=0.0,
        abs_tol=_MERGE_TIME_TOLERANCE_SECONDS,
    )


def _has_mergeable_neighbour(beats: Sequence[VisualBeat], position: int) -> bool:
    """Whether the beat at ``position`` could ever be absorbed by a neighbour.

    This reads the planned timeline only, so it costs nothing and is allowed to be
    optimistic: the neighbour it finds may itself turn out to be unfillable. Its job
    is to stop a run from paying for every remaining beat when the beat that just
    failed could never have been rescued.

    A same-group neighbour always qualifies. A neighbour from the adjacent group
    qualifies only while cross-group merging is enabled, and that is what keeps a
    beat which is the only shot of its own group from failing the whole video —
    the common shape now that most spans produce a single beat. The two are
    deliberately checked with the same adjacency rule, so this stays a pure
    timeline read that agrees with what ``_adjacent_merge_survivors`` will later
    accept.
    """
    beat = beats[position]
    cross_group = is_cross_group_merge_enabled()
    for other_position in (position - 1, position + 1):
        if not 0 <= other_position < len(beats):
            continue
        other = beats[other_position]
        if other.semantic_group_id != beat.semantic_group_id and not cross_group:
            continue
        if _beats_are_adjacent(other, beat) or _beats_are_adjacent(beat, other):
            return True
    return False


def _merged_visual_beat(survivor: VisualBeat, absorbed: VisualBeat) -> VisualBeat:
    """One beat covering both windows, keeping the survivor's identity.

    The survivor keeps its index because its clip, its persisted source record and
    its verifier runs already refer to it. The absorbed index is retired rather
    than reused, so a run record saying beat 5 was unfillable still points at a
    beat that no longer exists in the timeline — which is exactly what happened to
    it.

    The survivor's ``visual_requirement`` is kept, and it is not necessarily the
    absorbed beat's: shots of one split span are given their own requirements at
    the script stage precisely so they do not describe the same moment. So the
    merged beat is an approved clip for the survivor's requirement covering the
    absorbed beat's window as well, which is the honest description of a rescue
    and is why the merge record notes the requirement that went unfilled. The
    alternative to inheriting is another paid selection round for a beat whose own
    requirement has already been established as unfillable, or failing the video
    over one shot.

    The duration policy distinguishes the two strengths of that rescue. Absorbing
    a shot of the survivor's own semantic group means the covering clip was
    verified against a requirement written for the same span; absorbing a beat
    from a neighbouring group means it was verified against the adjacent moment of
    the narration instead. Both are honest, one is weaker, and a report reading
    the timeline afterwards can tell them apart without consulting the merge
    records.
    """
    first, second = (
        (survivor, absorbed)
        if survivor.start_time <= absorbed.start_time
        else (absorbed, survivor)
    )
    start_units = [
        unit for unit in (first.start_unit, second.start_unit) if unit is not None
    ]
    end_units = [
        unit
        for unit in (first.end_unit_exclusive, second.end_unit_exclusive)
        if unit is not None
    ]
    duration = second.end_time - first.start_time
    return dataclass_replace(
        survivor,
        start_time=first.start_time,
        end_time=second.end_time,
        duration=duration,
        spoken_text=" ".join(f"{first.spoken_text} {second.spoken_text}".split()),
        source_narration_slot_indexes=sorted(
            {
                *first.source_narration_slot_indexes,
                *second.source_narration_slot_indexes,
            }
        ),
        start_unit=min(start_units) if start_units else None,
        end_unit_exclusive=max(end_units) if end_units else None,
        duration_policy=(
            "unfillable_beat_cross_group_merged"
            if (
                survivor.semantic_group_id != absorbed.semantic_group_id
                # A survivor can absorb on both sides, so the policy names the
                # weakest claim any of those absorptions made. Without this, a
                # same-group merge landing after a cross-group one would upgrade a
                # beat that is still covering a neighbouring group's window.
                or survivor.duration_policy == "unfillable_beat_cross_group_merged"
            )
            else "unfillable_beat_merged"
        ),
        rapid_cut=(
            duration < VISUAL_BEAT_RAPID_CUT_SECONDS - _MERGE_TIME_TOLERANCE_SECONDS
        ),
    )


def validate_merged_beat_timeline(beats: Sequence[VisualBeat]) -> None:
    """Check a merged timeline still covers the narration without a seam.

    The timeline the script stage validated required beat indexes to run 1..N,
    which a merge deliberately breaks: absorbing a beat retires its index so the
    surviving clip keeps the identity its persisted record was written under. So
    this re-checks everything the renderer actually depends on — a start at zero,
    no gap or overlap, and a duration that matches its own endpoints — while
    requiring indexes only to be unique and increasing.
    """
    if not beats:
        raise ValueError("merged visual beat timeline is empty")
    if abs(beats[0].start_time) > _MERGE_TIME_TOLERANCE_SECONDS:
        raise ValueError("merged visual beat timeline must start at zero")
    previous_index = 0
    previous_end: float | None = None
    for beat in beats:
        if beat.index <= previous_index:
            raise ValueError("merged visual beat indexes must increase")
        previous_index = beat.index
        if not math.isfinite(beat.start_time) or not math.isfinite(beat.end_time):
            raise ValueError("merged visual beat timing must be finite")
        if beat.duration <= _MERGE_TIME_TOLERANCE_SECONDS:
            raise ValueError("merged visual beat duration must be positive")
        if not math.isclose(
            beat.duration,
            beat.end_time - beat.start_time,
            rel_tol=0.0,
            abs_tol=_MERGE_TIME_TOLERANCE_SECONDS,
        ):
            raise ValueError("merged visual beat duration is inconsistent")
        if previous_end is not None and not math.isclose(
            previous_end,
            beat.start_time,
            rel_tol=0.0,
            abs_tol=_MERGE_TIME_TOLERANCE_SECONDS,
        ):
            raise ValueError("merged visual beat timeline contains a gap or overlap")
        previous_end = beat.end_time


def _adjacent_merge_survivors(
    outcomes: Sequence[_SmartItemOutcome],
    position: int,
) -> list[_SmartItemOutcome]:
    """Filled neighbours that touch the open beat, previous side first.

    Only the nearest neighbour on each side is considered, and an already absorbed
    outcome is looked straight through, so a run of consecutive open beats can
    still collapse into one survivor: the first merge extends that survivor's
    window until it reaches the next open beat.

    The order is load-bearing rather than incidental. The caller ranks the two
    sides by the room left in their assets and only leaves this order when one
    side is roomier by a margin that could change a cut, so a previous-side
    neighbour absorbs whenever the two sides are effectively equal.

    Same-group neighbours are returned alone whenever any exists, and neighbours
    from an adjacent group are offered only when none does. That ordering is the
    whole safety of cross-group merging: the caller picks the survivor with the
    most room left in its asset, so without this separation a cross-group clip with
    a long asset could outbid a same-group clip that describes the very same
    moment. Cross-group neighbours are skipped entirely while the feature is off,
    which restores the previous behaviour exactly.
    """
    beat = outcomes[position].beat
    if beat is None:
        return []
    same_group: list[_SmartItemOutcome] = []
    cross_group: list[_SmartItemOutcome] = []
    allow_cross_group = is_cross_group_merge_enabled()
    for step in (-1, 1):
        other_position = position + step
        while 0 <= other_position < len(outcomes):
            other = outcomes[other_position]
            if other.absorbed_by:
                other_position += step
                continue
            other_beat = other.beat
            if (
                other.filled
                and other_beat is not None
                and (
                    _beats_are_adjacent(other_beat, beat)
                    if step < 0
                    else _beats_are_adjacent(beat, other_beat)
                )
            ):
                if other_beat.semantic_group_id == beat.semantic_group_id:
                    same_group.append(other)
                elif allow_cross_group:
                    cross_group.append(other)
            break
    return same_group or cross_group


def _pick_merge_survivor(
    survivors: Sequence[_SmartItemOutcome],
    open_beat: VisualBeat,
    normalized_speed: float,
) -> _SmartItemOutcome:
    """Choose which bordering shot absorbs the open beat's window.

    ``survivors`` arrives previous side first and holds at most two candidates.
    The survivor is the one with the most room left in its asset, because that is
    the clip that can be cut wider to cover the combined window without being
    re-selected — but "most room" only decides when the difference is real. Each
    side's merged window is the same length, yet one is measured as this beat's end
    minus the previous beat's start and the other as the next beat's end minus this
    beat's start, so the two headrooms routinely differ by a rounding error of about
    1e-15 seconds. Comparing them raw let that noise pick the survivor, and the
    later shot won every genuinely tied contest. So the previous side is kept unless
    a challenger is roomier by more than _MERGE_TIME_TOLERANCE_SECONDS — a
    microsecond of extra footage cannot change a cut, and preferring the previous
    side keeps the rewritten timeline in index order, since a survivor keeps its own
    index and absorbing forwards is what leaves the indexes ascending.

    A candidate that has lost its approved clip scores -inf, so it loses to any real
    neighbour and is only ever returned when it is the sole survivor — where the
    caller turns it into the same refusal it would have raised anyway.
    """

    def _headroom(candidate: _SmartItemOutcome) -> float:
        candidate_beat = candidate.beat
        candidate_winner = candidate.winner
        if candidate_beat is None or candidate_winner is None:
            return float("-inf")
        merged = _merged_visual_beat(candidate_beat, open_beat)
        needed = required_source_duration_for_timeline(
            merged.duration,
            normalized_speed,
        )
        return float(candidate_winner.duration) - needed

    survivor = survivors[0]
    survivor_headroom = _headroom(survivor)
    for challenger in survivors[1:]:
        challenger_headroom = _headroom(challenger)
        if challenger_headroom > survivor_headroom + _MERGE_TIME_TOLERANCE_SECONDS:
            survivor = challenger
            survivor_headroom = challenger_headroom
    return survivor


def _restamp_merged_source(
    *,
    winner: MaterialInfo,
    merged_beat: VisualBeat,
    segment: dict[str, Any],
    required_target_duration: float,
    required_source_duration: float,
    video_path: str,
) -> bool:
    """Point an already downloaded winner at the wider window it now has to cover.

    Returns ``False`` when the file on disk proves the wider window cannot stay
    inside one continuous shot of the source, leaving the winner untouched so the
    caller can fill the beat another way. A merge asks for the widest window this
    pipeline ever cuts — one clip covering two beats — which makes it the most
    likely to run past a cut, and it is the one place a cut is least acceptable,
    since the whole point of the merge is that these two beats become one shot.
    The file is already local, so the check costs nothing but an FFmpeg pass.

    An unmeasurable file keeps the window it was given: not being able to check
    must never be the reason a beat goes unfilled.
    """
    start = float(segment["source_start_time"])
    end = float(segment["source_end_time"])
    evidence: dict[str, Any] = {}
    cuts = shot_integrity.detect_shot_cuts(video_path) if video_path else None
    if cuts is None:
        evidence["shot_cuts"] = "unavailable"
    else:
        evidence["shot_cuts"] = cuts
        try:
            anchor_start = float(segment["verified_start_time"])
            anchor_end = float(segment["verified_end_time"])
        except (KeyError, TypeError, ValueError):
            anchor_start, anchor_end = start, end
        contained = shot_integrity.window_inside_one_shot(
            cuts,
            source_duration=max(float(winner.duration or 0.0), end),
            verified_start=anchor_start,
            verified_end=anchor_end,
            required_duration=end - start,
        )
        if contained is None:
            evidence["shot_containment"] = "no_shot_long_enough"
            logger.warning(
                "the neighbour's clip cannot cover the merged window inside one "
                f"shot: visual_beat={merged_beat.index}, cuts={cuts}, "
                f"window={start:.3f}..{end:.3f}"
            )
            return False
        shifted = abs(contained[0] - start)
        start, end = contained
        evidence["shot_containment"] = (
            "shifted" if shifted > 1e-3 else "already_inside"
        )
        if shifted > 1e-3:
            evidence["shifted_seconds"] = round(shifted, 3)

    winner.source_start_time = start
    winner.source_end_time = end
    source = dict(winner.source_info) if isinstance(winner.source_info, dict) else {}
    source["slot_index"] = merged_beat.index
    source["visual_beat_index"] = merged_beat.index
    source["semantic_group_id"] = int(merged_beat.semantic_group_id)
    source["required_target_duration"] = required_target_duration
    source["required_source_duration"] = required_source_duration
    segment_record = dict(segment)
    segment_record["source_start_time"] = start
    segment_record["source_end_time"] = end
    source["temporal_segment"] = segment_record
    source["local_clip_check"] = evidence
    winner.source_info = source
    return True


def _merge_unfillable_beats(
    *,
    outcomes: list[_SmartItemOutcome],
    merge_ceiling: int,
    resolved_specs: dict[str, VisualRequirementSpec],
    provider_chain: Sequence[tuple[str, Callable[..., List[MaterialInfo]]]],
    required_query_limit: int,
    video_aspect: VideoAspect,
    settings: dict[str, Any],
    candidate_limit: int,
    normalized_speed: float,
    twelvelabs_service,
    material_directory: str,
    used_asset_identities: set[tuple[str, str]],
    used_urls: set[str],
    analysis_budget: int,
    video_budget_exhausted: bool = False,
    video_analysis_budget: int = 0,
    item_name: str,
    item_log_name: str,
) -> _MergeResolution:
    """Let a neighbouring shot absorb the window of a beat nothing could fill.

    This is the last rung before the video fails, and it is deliberately the
    cheapest one: a neighbour's clip has already been verified against a
    requirement drawn from the adjacent stretch of the same narration, so
    extending that clip's source window buys no analysis and no tokens — only a
    new window out of footage that already passed. A fresh selection round is run
    only when the neighbour's asset is simply too short to cover both beats.

    A sibling shot of the open beat's own semantic group is always preferred,
    because it was written for the same moment. A neighbour from the adjacent group
    is accepted only when no such sibling exists, which is the ordinary case now
    that most spans yield a single beat, and is the difference between one weaker
    cut and no video at all.

    ``video_budget_exhausted`` marks the video as out of analysis money. It only
    disables that fresh selection round; the free window extension is unaffected,
    which is why a video out of budget can still finish instead of failing.
    """
    item_type = item_log_name.replace(" ", "_")
    result = _MergeResolution()
    for position, outcome in enumerate(outcomes):
        beat = outcome.beat
        if outcome.filled or outcome.absorbed_by or beat is None:
            continue

        def _refuse(decision: str, reason: str) -> None:
            result.verifier_runs.append(
                {
                    "visual_item_type": item_type,
                    "visual_item_index": beat.index,
                    "visual_requirement": outcome.requirement,
                    "final_decision": decision,
                    "reason": reason,
                }
            )
            result.failures.append(
                f"{item_name} {beat.index} could not be absorbed by a "
                f"neighbouring shot: {reason}"
            )

        if result.merged >= merge_ceiling:
            # Unreachable while selection stops the run as soon as more items are
            # open than the ceiling allows, which is where the ceiling actually
            # earns its keep: refusing there is what stops a lost video from
            # paying for every remaining beat. This stays as the backstop that
            # keeps the cap true if that fail-fast is ever relaxed.
            _refuse(
                "MERGE_CEILING_REACHED",
                f"this video already merged {result.merged} of its beats",
            )
            continue

        survivors = _adjacent_merge_survivors(outcomes, position)
        if not survivors:
            _refuse(
                "MERGE_NEIGHBOUR_UNAVAILABLE",
                "no filled shot borders it in the rewritten timeline",
            )
            continue

        survivor = _pick_merge_survivor(survivors, beat, normalized_speed)
        survivor_beat = survivor.beat
        survivor_winner = survivor.winner
        if survivor_beat is None or survivor_winner is None:
            _refuse(
                "MERGE_NEIGHBOUR_UNAVAILABLE",
                "the bordering shot lost its approved clip",
            )
            continue
        merged_beat = _merged_visual_beat(survivor_beat, beat)
        required_target_duration = float(merged_beat.duration)
        required_source_duration = required_source_duration_for_timeline(
            required_target_duration,
            normalized_speed,
        )

        merge_fill = ""
        if float(survivor_winner.duration) + _MERGE_TIME_TOLERANCE_SECONDS >= (
            required_source_duration
        ):
            # The approved asset is long enough, so nothing new is bought or
            # downloaded: the same file is simply cut wider.
            result.segmentation_calls += 1
            segment, segment_failure = _segment_smart_candidate(
                candidate=survivor_winner,
                visual_item=merged_beat,
                requirement=survivor.requirement,
                required_source_duration=required_source_duration,
                normalized_speed=normalized_speed,
                settings=settings,
                twelvelabs_service=twelvelabs_service,
                item_name=item_name,
                item_log_name=item_log_name,
            )
            if segment is None:
                _refuse("MERGE_SEGMENTATION_FAILED", segment_failure)
                continue
            if _restamp_merged_source(
                winner=survivor_winner,
                merged_beat=merged_beat,
                segment=segment,
                required_target_duration=required_target_duration,
                required_source_duration=required_source_duration,
                video_path=survivor.video_path,
            ):
                merge_fill = "neighbour_window_extended"
        if not merge_fill:
            # The free rung is unavailable: either the neighbour's asset is too
            # short for the combined window, or its shots are. Both mean the beat
            # can now only be filled by buying a fresh selection round.
            if video_budget_exhausted:
                # Filling this beat is the single most expensive thing left in the
                # ladder, spent on a beat that already failed once. A video that
                # has hit its ceiling refuses it and fails this beat instead,
                # which is what keeps the ceiling an actual bound rather than a
                # suggestion. Every free rung above has already been tried.
                _refuse(
                    "MERGE_ANALYSIS_BUDGET_EXHAUSTED",
                    "the neighbour's approved clip cannot cover both windows in "
                    "one continuous shot and this video has spent its analysis "
                    f"budget of {video_analysis_budget} analyzed candidates",
                )
                continue
            fresh = _reselect_for_merged_beat(
                merged_beat=merged_beat,
                resolved_specs=resolved_specs,
                provider_chain=provider_chain,
                required_query_limit=required_query_limit,
                required_source_duration=required_source_duration,
                required_target_duration=required_target_duration,
                video_aspect=video_aspect,
                settings=settings,
                candidate_limit=candidate_limit,
                normalized_speed=normalized_speed,
                twelvelabs_service=twelvelabs_service,
                material_directory=material_directory,
                used_asset_identities=used_asset_identities,
                used_urls=used_urls,
                analysis_budget=analysis_budget,
                item_name=item_name,
                item_log_name=item_log_name,
            )
            result.candidates_analyzed += fresh.candidates_analyzed
            result.source_seconds_analyzed += fresh.source_seconds_analyzed
            result.segmentation_calls += fresh.segmentation_calls
            result.verifier_runs.extend(fresh.verifier_runs)
            if not fresh.video_path or fresh.winner is None:
                _refuse(
                    "MERGE_RESELECTION_FAILED",
                    "; ".join(fresh.failures)
                    or "no candidate covered the combined window",
                )
                continue
            survivor.video_path = fresh.video_path
            survivor.winner = fresh.winner
            survivor.provider = fresh.provider
            survivor.query = fresh.query
            survivor.requirement = fresh.requirement
            _remember_material_identity(
                fresh.winner,
                used_asset_identities,
                used_urls,
            )
            merge_fill = "fresh_selection_round"

        survivor.beat = merged_beat
        survivor.absorbed_indexes = [*survivor.absorbed_indexes, beat.index]
        outcome.absorbed_by = merged_beat.index
        result.merged += 1
        merge_scope = (
            "same_semantic_group"
            if survivor_beat.semantic_group_id == beat.semantic_group_id
            else "adjacent_semantic_group"
        )
        logger.warning(
            "no material could fill a visual beat, so a neighbouring shot absorbed "
            f"its window: {item_log_name}={beat.index}, "
            f"merged_into={merged_beat.index}, scope={merge_scope}, "
            f"fill={merge_fill}, "
            f"merged_duration={required_target_duration:.3f}s"
        )
        merge_record = {
            "visual_item_type": item_type,
            "visual_item_index": beat.index,
            "visual_requirement": outcome.requirement,
            "final_decision": "UNFILLABLE_BEAT_MERGED",
            "merged_into_visual_item_index": merged_beat.index,
            "merged_target_duration": round(required_target_duration, 3),
            "merge_fill": merge_fill,
            # Which shot absorbed it: one written for the same span, or the
            # narration's next moment. Recorded on every merge rather than only the
            # crossing ones so a report can count the two strengths of rescue
            # without inferring anything from an absent key.
            "merge_scope": merge_scope,
            "reason": "; ".join(outcome.failures)[:240],
        }
        # Shots of one split span carry requirements of their own, so the clip now
        # covering this window was approved for a different requirement than the
        # one above. Recorded when the two differ, so a report reading this run
        # cannot mistake the rescue for an approval of the unfilled requirement.
        covering_requirement = (survivor.requirement or "").strip()
        if covering_requirement != (outcome.requirement or "").strip():
            merge_record["merged_into_visual_requirement"] = covering_requirement
        result.verifier_runs.append(merge_record)
    return result


@dataclass
class _MergeReselection:
    """What one fresh selection round for a merged window produced."""

    video_path: str = ""
    winner: MaterialInfo | None = None
    provider: str = ""
    query: str = ""
    requirement: str = ""
    verifier_runs: list[dict[str, Any]] = dataclass_field(default_factory=list)
    failures: list[str] = dataclass_field(default_factory=list)
    candidates_analyzed: int = 0
    source_seconds_analyzed: float = 0.0
    segmentation_calls: int = 0


def _reselect_for_merged_beat(
    *,
    merged_beat: VisualBeat,
    resolved_specs: dict[str, VisualRequirementSpec],
    provider_chain: Sequence[tuple[str, Callable[..., List[MaterialInfo]]]],
    required_query_limit: int,
    required_source_duration: float,
    required_target_duration: float,
    video_aspect: VideoAspect,
    settings: dict[str, Any],
    candidate_limit: int,
    normalized_speed: float,
    twelvelabs_service,
    material_directory: str,
    used_asset_identities: set[tuple[str, str]],
    used_urls: set[str],
    analysis_budget: int,
    item_name: str,
    item_log_name: str,
) -> _MergeReselection:
    """Search once for a clip long enough to cover a merged window.

    Only reached when the survivor's approved asset is shorter than the combined
    beats need. The requirement is unchanged — it is the survivor's own, the one its
    existing clip was already approved for — so this is one ordinary selection round
    asking for a longer clip, not a new reading of the narration.
    """
    result = _MergeReselection(requirement=merged_beat.visual_requirement)
    requirement_spec = resolved_specs.get(
        llm.normalize_visual_requirement(merged_beat.visual_requirement)
    )
    if requirement_spec is None and settings["fail_closed"]:
        result.failures.append(
            "the merged window has no verifiable requirement checklist"
        )
        return result
    planned_queries = [
        query for query in (merged_beat.search_queries or []) if str(query or "").strip()
    ]
    if not planned_queries:
        result.failures.append("the merged window has no searchable query")
        return result
    item_queries = _ordered_item_search_queries(
        merged_beat,
        planned_queries[0],
        required_query_limit,
    )
    logger.info(
        "searching once for a clip long enough to cover a merged visual beat: "
        f"{item_log_name}={merged_beat.index}, "
        f"required_source_duration={required_source_duration:.3f}s, "
        f"queries={len(item_queries)}"
    )
    selection = _select_smart_item_winner(
        visual_item=merged_beat,
        requirement=merged_beat.visual_requirement,
        requirement_spec=requirement_spec,
        item_queries=item_queries,
        provider_chain=provider_chain,
        required_source_duration=required_source_duration,
        video_aspect=video_aspect,
        settings=settings,
        candidate_limit=candidate_limit,
        normalized_speed=normalized_speed,
        twelvelabs_service=twelvelabs_service,
        item_asset_identities=set(used_asset_identities),
        item_urls=set(used_urls),
        item_name=item_name,
        item_log_name=item_log_name,
        analysis_budget=analysis_budget,
    )
    result.candidates_analyzed = selection.candidates_analyzed
    result.source_seconds_analyzed = selection.source_seconds_analyzed
    result.segmentation_calls = selection.segmentation_calls
    result.verifier_runs.extend(selection.verifier_runs)
    result.failures.extend(selection.failures)
    if selection.attempt is None:
        return result

    download = _download_smart_winner(
        attempt=selection.attempt,
        visual_item=merged_beat,
        requirement=selection.requirement,
        provider=selection.provider,
        query=selection.query,
        required_target_duration=required_target_duration,
        required_source_duration=required_source_duration,
        normalized_speed=normalized_speed,
        settings=settings,
        twelvelabs_service=twelvelabs_service,
        material_directory=material_directory,
        video_aspect=video_aspect,
        is_visual_beat=True,
        item_name=item_name,
        item_log_name=item_log_name,
    )
    result.segmentation_calls += download.segmentation_calls
    result.verifier_runs.extend(download.verifier_runs)
    result.failures.extend(download.failures)
    if download.video_path and download.winner is not None:
        result.video_path = download.video_path
        result.winner = download.winner
        result.provider = selection.provider
        result.query = selection.query
        result.requirement = selection.requirement
    return result


def _download_videos_by_script_order_smart(
    *,
    task_id: str,
    search_terms: List[str],
    visual_slots: Sequence[OrderedVisualItem] | None = None,
    visual_beats: Sequence[OrderedVisualItem] | None = None,
    search_videos: Callable[..., List[MaterialInfo]] | None = None,
    provider_searches: Sequence[tuple[str, Callable[..., List[MaterialInfo]]]]
    | None = None,
    video_aspect: VideoAspect,
    max_clip_duration: int,
    material_directory: str,
    clip_speed: float,
    twelvelabs_service,
    max_candidates_override: int | None = None,
    requirement_specs: dict[str, VisualRequirementSpec] | None = None,
    merged_beats_out: list[VisualBeat] | None = None,
) -> List[str]:
    """Select one best candidate for each generic ordered slot or beat.

    ``merged_beats_out`` is how a rewritten timeline gets back to the caller. It is
    filled only when an unfillable beat was absorbed by a neighbouring shot, because
    that is the only case where the beats the renderer must use are no longer the
    beats that were passed in. Left empty, the caller's own timeline still holds.
    """
    if visual_slots is not None and visual_beats is not None:
        raise ValueError("smart selection accepts visual slots or visual beats, not both")
    visual_items = list(visual_beats if visual_beats is not None else visual_slots or [])
    item_name = "Visual beat" if visual_beats is not None else "Visual slot"
    item_log_name = item_name.lower()
    provider_chain: list[tuple[str, Callable[..., List[MaterialInfo]]]] = list(
        provider_searches or []
    )
    if not provider_chain:
        if search_videos is None:
            raise ValueError("smart selection requires at least one provider search")
        provider_chain = [("pexels", search_videos)]
    if len(search_terms) != len(visual_items):
        logger.warning(
            f"smart ordered material matching requires one query per {item_log_name}: "
            f"queries={len(search_terms)}, items={len(visual_items)}"
        )
        return []
    for visual_item, search_query in zip(visual_items, search_terms):
        if not visual_item.visual_requirement.strip():
            raise SmartMaterialSelectionError(
                f"{item_name} {visual_item.index} has no visual requirement"
            )
        if (
            not visual_item.search_queries
            or search_query not in visual_item.search_queries
        ):
            raise SmartMaterialSelectionError(
                f"{item_name} {visual_item.index} has an inconsistent search query mapping"
            )

    settings = twelvelabs_service.candidate_selection_settings()
    normalized_speed = utils.normalize_clip_speed(clip_speed)
    used_asset_identities: set[tuple[str, str]] = set()
    used_urls: set[str] = set()
    outcomes: list[_SmartItemOutcome] = []
    semantic_verifier_runs: list[dict[str, Any]] = []
    total_candidates_analyzed = 0
    total_source_seconds_analyzed = 0.0
    segmentation_calls = 0
    # Merging rewrites the beat timeline, so it is available only where there is a
    # beat timeline to rewrite: fixed slots carry no semantic group to merge within,
    # and a duck-typed item cannot be recombined into a valid beat.
    merge_beats: list[VisualBeat] = (
        list(visual_items)  # type: ignore[arg-type]
        if visual_beats is not None
        and all(isinstance(item, VisualBeat) for item in visual_items)
        else []
    )
    merge_ceiling = max_merged_beats_per_video(len(merge_beats))
    if merge_ceiling:
        logger.info(
            "smart material unfillable beat merge ceiling: "
            f"max_merged_beats={merge_ceiling}"
        )
    if len(provider_chain) > 1:
        logger.info(
            "smart material provider cascade: "
            f"{' -> '.join(name for name, _ in provider_chain)}"
        )
    max_query_variants = max_query_variants_per_provider()
    if max_query_variants > 1:
        logger.info(
            "smart material query variants per provider: "
            f"max={max_query_variants}"
        )

    # The checklist normally arrives from the script stage, where it was computed
    # and persisted before a single stock request was made. Verification then gates
    # on exactly the plan the run was recorded with. Only requirements the script
    # stage could not resolve are decomposed here, which also keeps API and legacy
    # callers that pass nothing working exactly as before.
    resolved_specs: dict[str, VisualRequirementSpec] = dict(requirement_specs or {})
    unresolved_requirements: list[str] = []
    seen_unresolved: set[str] = set()
    for visual_item in visual_items:
        normalized = llm.normalize_visual_requirement(visual_item.visual_requirement)
        if normalized in resolved_specs or normalized in seen_unresolved:
            continue
        # Sibling shots of one semantic group share a requirement. The decomposer
        # de-duplicates internally too; collapsing here keeps the log count in
        # requirements rather than in beats, which is what actually gets requested.
        seen_unresolved.add(normalized)
        unresolved_requirements.append(visual_item.visual_requirement)
    if unresolved_requirements:
        if resolved_specs:
            logger.info(
                "decomposing visual requirements missing from the script-stage "
                f"checklist: supplied={len(resolved_specs)}, "
                f"missing={len(unresolved_requirements)}"
            )
        resolved_specs.update(
            llm.generate_visual_requirement_specs(unresolved_requirements)
        )
    missing_spec_items = [
        visual_item.index
        for visual_item in visual_items
        if llm.normalize_visual_requirement(visual_item.visual_requirement)
        not in resolved_specs
    ]
    if missing_spec_items and settings["fail_closed"]:
        # A requirement with no spec cannot be verified against, so this is a
        # per-item problem from here on: the item skips the stock search it could
        # never pass and goes straight to the recovery path, while items whose
        # requirements did decompose keep their full budget.
        logger.warning(
            "visual requirement decomposition produced no checklist for "
            f"{item_log_name} indexes {missing_spec_items}; no candidate will be "
            "requested for those requirements"
        )
    rewrite_enabled = is_requirement_rewrite_enabled()
    opening_shot_rewrite_enabled = is_opening_shot_rewrite_enabled()

    candidate_limit = settings["max_candidates"]
    if max_candidates_override is not None:
        candidate_limit = min(
            candidate_limit,
            max(1, int(max_candidates_override)),
        )
    # One unfillable item, left alone, will spend the full candidate cap on every
    # phrasing and every provider and then again on the rewrite. The budget caps
    # that per round; a healthy item settles well inside it and never notices.
    analysis_budget = analysis_budget_per_selection_round(candidate_limit)
    if analysis_budget:
        logger.info(
            "smart material per-round analysis budget: "
            f"max_analyzed_candidates={analysis_budget}"
        )
    # And a ceiling for the whole video, because the failure ladder means a run no
    # longer stops at its first unfillable item. This one is spent on recovery only.
    video_analysis_budget = analysis_budget_per_video(
        len(visual_items),
        candidate_limit,
    )
    if video_analysis_budget:
        logger.info(
            "smart material per-video analysis budget: "
            f"max_analyzed_candidates={video_analysis_budget}"
        )

    for item_position, (visual_item, search_query) in enumerate(
        zip(visual_items, search_terms)
    ):
        required_target_duration = float(visual_item.duration)
        required_source_duration = required_source_duration_for_timeline(
            required_target_duration,
            normalized_speed,
        )
        requirement_spec = resolved_specs.get(
            llm.normalize_visual_requirement(visual_item.visual_requirement)
        )
        item_queries = _ordered_item_search_queries(
            visual_item,
            search_query,
            max_query_variants,
        )

        winning_attempt: _SmartProviderAttempt | None = None
        winning_provider = ""
        winning_query = search_query
        selected_requirement = visual_item.visual_requirement
        attempt_failures: list[str] = []
        # Exclusions accumulate for the whole item, not for one attempt: a
        # candidate this item already rejected must not be analyzed again under
        # another phrasing. Identity is what grows here — the verdict cache is
        # keyed by provider asset, and search results carry no stable url beyond
        # it. Only the winner joins the timeline-wide sets, so later items stay
        # free to consider what this one turned down.
        item_asset_identities = set(used_asset_identities)
        item_urls = set(used_urls)

        if requirement_spec is None and settings["fail_closed"]:
            # Nothing is searched for a requirement no gate can approve; the item
            # is handed to the recovery path with its budget still unspent.
            semantic_verifier_runs.append(
                {
                    "visual_item_type": item_log_name.replace(" ", "_"),
                    "visual_item_index": visual_item.index,
                    "visual_requirement": visual_item.visual_requirement,
                    "final_decision": "DECOMPOSITION_FAILED",
                }
            )
            selection = _SmartItemSelection(
                requirement=visual_item.visual_requirement,
                failures=[
                    f"Visual requirement decomposition failed for {item_log_name} "
                    f"{visual_item.index}"
                ],
            )
        else:
            selection = _select_smart_item_winner(
                visual_item=visual_item,
                requirement=visual_item.visual_requirement,
                requirement_spec=requirement_spec,
                item_queries=item_queries,
                provider_chain=provider_chain,
                required_source_duration=required_source_duration,
                video_aspect=video_aspect,
                settings=settings,
                candidate_limit=candidate_limit,
                normalized_speed=normalized_speed,
                twelvelabs_service=twelvelabs_service,
                item_asset_identities=item_asset_identities,
                item_urls=item_urls,
                item_name=item_name,
                item_log_name=item_log_name,
                # Narrowed by whatever is left of the video's ceiling. Never below
                # one, so this item still gets its first look on the first provider
                # even in a video that has already overspent; what gets cut is the
                # phrasing-and-provider cascade behind that first look.
                analysis_budget=_effective_round_budget(
                    analysis_budget,
                    video_analysis_budget,
                    total_candidates_analyzed,
                ),
            )
        total_candidates_analyzed += selection.candidates_analyzed
        total_source_seconds_analyzed += selection.source_seconds_analyzed
        segmentation_calls += selection.segmentation_calls
        semantic_verifier_runs.extend(selection.verifier_runs)
        attempt_failures.extend(selection.failures)

        video_budget_exhausted = bool(video_analysis_budget) and (
            total_candidates_analyzed >= video_analysis_budget
        )
        if (
            selection.attempt is None
            and rewrite_enabled
            and item_position == 0
            and not opening_shot_rewrite_enabled
        ):
            # The opening shot is withheld from the rewrite rather than from the
            # search: every free rung above this one already ran. Position, not
            # index, decides, because the item that opens the video is the first
            # one on this timeline whatever its own numbering says.
            logger.warning(
                "withholding the requirement rewrite from the opening shot: "
                f"{item_log_name}={visual_item.index}, "
                f"requirement={visual_item.visual_requirement!r}"
            )
            attempt_failures.append(
                f"The requirement rewrite is withheld from the opening "
                f"{item_log_name} {visual_item.index}, because a re-described "
                f"opening shot can depict something the narration never promised"
            )
            semantic_verifier_runs.append(
                {
                    "visual_item_type": item_log_name.replace(" ", "_"),
                    "visual_item_index": visual_item.index,
                    "visual_requirement": visual_item.visual_requirement,
                    "final_decision": "OPENING_SHOT_REWRITE_WITHHELD",
                }
            )
        elif selection.attempt is None and rewrite_enabled and video_budget_exhausted:
            # The rewrite is the most expensive rung: a fresh requirement means a
            # fresh search and a fresh page of analyses for an item that has already
            # proved hard. It is also the rung the free rungs can substitute for, so
            # a video out of budget skips it and lets the merge carry the window.
            logger.warning(
                "smart material per-video analysis budget spent; skipping the "
                f"requirement rewrite for {item_log_name} {visual_item.index}: "
                f"analyzed={total_candidates_analyzed}, "
                f"budget={video_analysis_budget}"
            )
            attempt_failures.append(
                f"Per-video analysis budget of {video_analysis_budget} analyzed "
                f"candidates reached before the requirement rewrite for "
                f"{item_log_name} {visual_item.index}"
            )
            semantic_verifier_runs.append(
                {
                    "visual_item_index": visual_item.index,
                    "visual_requirement": visual_item.visual_requirement,
                    "final_decision": "VIDEO_ANALYSIS_BUDGET_EXHAUSTED",
                    "candidates_analyzed": total_candidates_analyzed,
                    "video_analysis_budget": video_analysis_budget,
                }
            )
        elif selection.attempt is None and rewrite_enabled:
            recovery = _rewrite_requirement_and_reselect(
                visual_item=visual_item,
                failed_requirement=visual_item.visual_requirement,
                failure_summary="; ".join(attempt_failures),
                query_limit=max_query_variants,
                provider_chain=provider_chain,
                required_source_duration=required_source_duration,
                video_aspect=video_aspect,
                settings=settings,
                candidate_limit=candidate_limit,
                normalized_speed=normalized_speed,
                twelvelabs_service=twelvelabs_service,
                item_asset_identities=item_asset_identities,
                item_urls=item_urls,
                item_name=item_name,
                item_log_name=item_log_name,
                analysis_budget=_effective_round_budget(
                    analysis_budget,
                    video_analysis_budget,
                    total_candidates_analyzed,
                ),
            )
            total_candidates_analyzed += recovery.candidates_analyzed
            total_source_seconds_analyzed += recovery.source_seconds_analyzed
            segmentation_calls += recovery.segmentation_calls
            semantic_verifier_runs.extend(recovery.verifier_runs)
            attempt_failures.extend(recovery.failures)
            if recovery.attempt is not None:
                selection = recovery

        if selection.attempt is not None:
            winning_attempt = selection.attempt
            winning_provider = selection.provider
            winning_query = selection.query
            selected_requirement = selection.requirement

        outcome = _SmartItemOutcome(
            visual_item=visual_item,
            requirement=selected_requirement,
            beat=merge_beats[item_position] if merge_beats else None,
        )
        outcomes.append(outcome)

        def _leave_open(reasons: list[str]) -> bool:
            """Keep this item open for a merge, or report that the video is lost.

            An item stays open only while a merge could still rescue it: within the
            merge ceiling, and with a filled shot bordering it in the planned
            timeline. Otherwise the run stops here rather than paying full price for
            every remaining item of a video that is already lost.
            """
            outcome.failures.extend(reasons)
            open_items = sum(
                1 for recorded in outcomes if not recorded.filled
            )
            if (
                merge_beats
                and open_items <= merge_ceiling
                and _has_mergeable_neighbour(merge_beats, item_position)
            ):
                logger.warning(
                    "no material could fill a visual item; leaving it open for a "
                    f"neighbouring shot to absorb: {item_log_name}={visual_item.index}, "
                    f"open={open_items}, ceiling={merge_ceiling}"
                )
                return True
            _persist_material_sources(
                task_id,
                _records_from_outcomes(outcomes),
                semantic_verifier_runs,
            )
            raise SmartMaterialSelectionError(
                "; ".join(
                    reason
                    for recorded in outcomes
                    for reason in recorded.failures
                )
            )

        if winning_attempt is None:
            _leave_open(attempt_failures)
            continue

        winner = winning_attempt.winner
        segment = winning_attempt.segment
        if winner is None or segment is None:
            # Unreachable through _SmartProviderAttempt.succeeded; kept so a
            # future refactor cannot silently render an unbound source window.
            raise SmartMaterialSelectionError(
                f"{item_name} {visual_item.index} produced an incomplete "
                "smart selection result"
            )

        download = _download_smart_winner(
            attempt=winning_attempt,
            visual_item=visual_item,
            requirement=selected_requirement,
            provider=winning_provider,
            query=winning_query,
            required_target_duration=required_target_duration,
            required_source_duration=required_source_duration,
            normalized_speed=normalized_speed,
            settings=settings,
            twelvelabs_service=twelvelabs_service,
            material_directory=material_directory,
            video_aspect=video_aspect,
            is_visual_beat=visual_beats is not None,
            item_name=item_name,
            item_log_name=item_log_name,
        )
        segmentation_calls += download.segmentation_calls
        semantic_verifier_runs.extend(download.verifier_runs)

        if not download.video_path or download.winner is None:
            # A verified item whose every approved candidate failed to transfer is
            # just as open as one nothing could verify, and a neighbour can absorb it
            # on the same terms.
            _leave_open(download.failures)
            continue
        outcome.provider = winning_provider
        outcome.query = winning_query
        outcome.video_path = download.video_path
        outcome.winner = download.winner
        _remember_material_identity(
            download.winner,
            used_asset_identities,
            used_urls,
        )

    if any(not outcome.filled for outcome in outcomes):
        merge = _merge_unfillable_beats(
            outcomes=outcomes,
            merge_ceiling=merge_ceiling,
            resolved_specs=resolved_specs,
            provider_chain=provider_chain,
            required_query_limit=max_query_variants,
            video_aspect=video_aspect,
            settings=settings,
            candidate_limit=candidate_limit,
            normalized_speed=normalized_speed,
            twelvelabs_service=twelvelabs_service,
            material_directory=material_directory,
            used_asset_identities=used_asset_identities,
            used_urls=used_urls,
            analysis_budget=_effective_round_budget(
                analysis_budget,
                video_analysis_budget,
                total_candidates_analyzed,
            ),
            # The free rung — extending a neighbour's approved clip — stays available
            # whatever the budget says, because it buys nothing. Only the fresh
            # search for a merged window is refused once the video is out of money.
            video_budget_exhausted=bool(video_analysis_budget)
            and total_candidates_analyzed >= video_analysis_budget,
            video_analysis_budget=video_analysis_budget,
            item_name=item_name,
            item_log_name=item_log_name,
        )
        total_candidates_analyzed += merge.candidates_analyzed
        total_source_seconds_analyzed += merge.source_seconds_analyzed
        segmentation_calls += merge.segmentation_calls
        semantic_verifier_runs.extend(merge.verifier_runs)
        still_open = [
            outcome
            for outcome in outcomes
            if not outcome.filled and not outcome.absorbed_by
        ]
        if still_open:
            _persist_material_sources(
                task_id,
                _records_from_outcomes(outcomes),
                semantic_verifier_runs,
            )
            raise SmartMaterialSelectionError(
                "; ".join(
                    [
                        reason
                        for outcome in still_open
                        for reason in outcome.failures
                    ]
                    + merge.failures
                )
            )

    video_paths = [
        outcome.video_path for outcome in outcomes if outcome.filled
    ]
    material_sources = _records_from_outcomes(outcomes)
    if merge_beats and merged_beats_out is not None:
        merged_timeline = [
            outcome.beat
            for outcome in outcomes
            if outcome.filled and outcome.beat is not None
        ]
        if len(merged_timeline) != len(merge_beats):
            # Only handed back when it actually differs, so a caller that merged
            # nothing keeps using the timeline it validated at the script stage.
            validate_merged_beat_timeline(merged_timeline)
            merged_beats_out.clear()
            merged_beats_out.extend(merged_timeline)
            logger.warning(
                "the visual beat timeline was rewritten by merging: "
                f"planned_beats={len(merge_beats)}, "
                f"rendered_beats={len(merged_timeline)}"
            )

    logger.info(
        "TwelveLabs smart selection usage: "
        f"candidates_analyzed={total_candidates_analyzed}, "
        f"video_analysis_budget={video_analysis_budget}, "
        f"source_seconds_analyzed={total_source_seconds_analyzed:.3f}, "
        f"segmentation_calls={segmentation_calls}"
    )
    logger.success(f"downloaded {len(video_paths)} smart ordered videos")
    _persist_material_sources(
        task_id,
        material_sources,
        semantic_verifier_runs,
    )
    return video_paths


def _download_videos_by_script_order(
    task_id: str,
    search_terms: List[str],
    search_videos,
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    visual_slots: list[VisualSlot] | None = None,
    visual_beats: list[VisualBeat] | None = None,
    clip_speed: float = 1.0,
    provider_searches: Sequence[tuple[str, Callable[..., List[MaterialInfo]]]]
    | None = None,
    requirement_specs: dict[str, VisualRequirementSpec] | None = None,
    merged_beats_out: list[VisualBeat] | None = None,
) -> List[str]:
    """
    按脚本文案顺序下载素材。

    默认下载逻辑会把所有关键词的候选素材合并成一个大列表；旧的顺序模式
    则按轮次遍历全部关键词，素材不够时会从脚本开头重新开始。这里先根据
    音频时长计算实际需要的镜头数，再把关键词只沿时间线向前分配。一个
    关键词可以连续占据多个镜头，但已经进入后半段后不会再跳回开头。

    顺序匹配是显式的质量模式：如果某个时间段没有足够的独立候选，宁可
    让素材阶段失败，也不把另一个叙事段落的画面强行塞进该位置。
    """
    logger.info("downloading videos with script-order material matching")
    # Imported lazily to avoid the module-level cycle: twelvelabs uses the API
    # key rotator defined in this module.
    from app.services import twelvelabs as twelvelabs_service

    if (
        visual_beats or visual_slots
    ) and twelvelabs_service.is_smart_visual_matching_enabled():
        # Beats own the timeline once S4 produced them; slots remain the input
        # for older tasks and for the plain ordered path.
        smart_items: dict[str, Any] = (
            {"visual_beats": list(visual_beats)}
            if visual_beats
            else {"visual_slots": list(visual_slots or [])}
        )
        return _download_videos_by_script_order_smart(
            task_id=task_id,
            search_terms=search_terms,
            search_videos=search_videos,
            provider_searches=provider_searches,
            video_aspect=video_aspect,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            clip_speed=clip_speed,
            twelvelabs_service=twelvelabs_service,
            requirement_specs=requirement_specs,
            merged_beats_out=merged_beats_out,
            **smart_items,
        )

    semantic_qa_enabled = twelvelabs_service.is_clip_qa_enabled()
    minimum_candidate_duration = (
        max(max_clip_duration, _MIN_SEMANTIC_QA_DURATION)
        if semantic_qa_enabled
        else max_clip_duration
    )
    candidate_groups = []
    valid_video_urls = set()
    found_duration = 0.0

    for search_term in search_terms:
        video_items = search_videos(
            search_term=search_term,
            minimum_duration=minimum_candidate_duration,
            video_aspect=video_aspect,
        )
        logger.info(f"found {len(video_items)} videos for '{search_term}'")

        term_items = []
        for item in video_items:
            if item.url in valid_video_urls:
                continue
            term_items.append(item)
            valid_video_urls.add(item.url)
            found_duration += item.duration

        candidate_groups.append((search_term, term_items))

    logger.info(
        f"found total ordered video candidates: {sum(len(items) for _, items in candidate_groups)}, "
        f"required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )

    required_clip_count = max(
        1,
        math.ceil(max(0.0, float(audio_duration)) / max_clip_duration),
    )
    term_plan = _build_script_order_term_plan(
        term_count=len(candidate_groups),
        required_clip_count=required_clip_count,
    )
    logger.info(
        "ordered material timeline plan: "
        f"clips={required_clip_count}, term_indexes={term_plan}"
    )

    required_by_term = [0] * len(candidate_groups)
    for term_index in term_plan:
        required_by_term[term_index] += 1

    insufficient_terms = [
        {
            "term": candidate_groups[index][0],
            "required": required,
            "available": len(candidate_groups[index][1]),
        }
        for index, required in enumerate(required_by_term)
        if len(candidate_groups[index][1]) < required
    ]
    if insufficient_terms:
        logger.warning(
            "script-order material coverage is incomplete; refusing unrelated "
            f"fallback clips: {insufficient_terms}"
        )
        return []

    video_paths = []
    material_sources: list[dict[str, Any]] = []
    candidate_offsets = [0] * len(candidate_groups)
    total_duration = 0.0

    for term_index in term_plan:
        search_term, term_items = candidate_groups[term_index]
        saved_for_slot = False
        while candidate_offsets[term_index] < len(term_items):
            item = term_items[candidate_offsets[term_index]]
            candidate_offsets[term_index] += 1
            try:
                semantic_qa = None
                if semantic_qa_enabled:
                    semantic_qa = twelvelabs_service.evaluate_clip_match(
                        video_url=item.url,
                        visual_query=search_term,
                    )
                    if semantic_qa is None:
                        if twelvelabs_service.clip_qa_fail_closed():
                            logger.warning(
                                "rejecting ordered candidate because semantic QA "
                                f"is unavailable: term={search_term!r}"
                            )
                            continue
                        logger.warning(
                            "semantic QA unavailable; allowing candidate because "
                            f"fail-closed is disabled: term={search_term!r}"
                        )
                    elif not semantic_qa.get("accepted", False):
                        logger.info(
                            "rejecting ordered candidate after semantic QA: "
                            f"term={search_term!r}, score={semantic_qa.get('score')}, "
                            f"reason={semantic_qa.get('reason')!r}"
                        )
                        continue

                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                if semantic_qa is not None:
                    source_info = dict(source_info)
                    source_info["semantic_qa"] = semantic_qa
                    item.source_info = source_info
                logger.info(
                    f"downloading ordered {item.provider} video for {search_term!r}: "
                    f"asset_id={source_info.get('asset_id') or 'unknown'}"
                )
                saved_video_path = save_video(
                    video_url=item.url,
                    save_dir=material_directory,
                    video_aspect=video_aspect,
                )
                if saved_video_path:
                    logger.info(f"video saved: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    saved_for_slot = True
                    try:
                        material_sources.append(
                            _material_source_record(item, saved_video_path)
                        )
                    except Exception as source_error:
                        logger.warning(
                            "failed to prepare ordered material source record: "
                            f"provider={item.provider}, "
                            f"error={type(source_error).__name__}, "
                            f"detail={source_error}"
                        )
                    total_duration += min(max_clip_duration, item.duration)
                    break
            except Exception as e:
                logger.error(
                    "failed to download ordered material video: "
                    f"provider={item.provider}, error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, item.url)}"
                )

        if not saved_for_slot:
            logger.warning(
                "script-order material slot has no valid candidate; refusing "
                f"cross-segment fallback: term={search_term!r}"
            )
            _persist_material_sources(task_id, material_sources)
            return []

    logger.success(
        f"downloaded {len(video_paths)} ordered videos, "
        f"planned duration: {total_duration} seconds"
    )
    _persist_material_sources(task_id, material_sources)
    return video_paths


if __name__ == "__main__":
    download_videos(
        "test123", ["Money Exchange Medium"], audio_duration=100, source="pixabay"
    )
