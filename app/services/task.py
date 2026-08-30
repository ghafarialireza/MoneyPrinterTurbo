import math
import os
import re
import socket
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from functools import partial
from os import path
from uuid import uuid4

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import (
    NarrationOverlap,
    NarrationSlot,
    NarrationTimingQuality,
    NarrationTimingSource,
    RenderSegment,
    SemanticVisualSpan,
    TimedNarrationUnit,
    VideoConcatMode,
    VideoParams,
    VisualBeat,
    VisualRequirementSpec,
    VisualSlot,
    VISUAL_BEAT_RAPID_CUT_SECONDS,
)
from app.services import bgm as bgm_service
from app.services import (
    elevenlabs_music,
    llm,
    loomloom,
    material,
    sonilo,
    subtitle,
    task_artifacts,
    twelvelabs,
    video,
    voice,
)
from app.services import upload_post
from app.services import state as sm
from app.utils import file_security, utils


# 发布请求最长可等待数分钟，不能继续占用视频生成任务的并发名额。
# 固定大小的线程池将发布吞吐限制在可控范围内，同时让视频产物生成后
# 立即进入完成状态。
_cross_post_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mpt-cross-post",
)
_cross_post_max_pending_tasks = max(
    1,
    int(config.app.get("upload_post_max_pending_tasks", 10)),
)
_cross_post_slots = threading.BoundedSemaphore(_cross_post_max_pending_tasks)
_cross_post_registry_lock = threading.RLock()
_cross_post_futures: dict[str, Future] = {}
_cross_post_process_owner = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
_ACTIVE_CROSS_POST_STATES = {
    const.CROSS_POST_STATE_PENDING,
    const.CROSS_POST_STATE_PROCESSING,
}
_CROSS_POST_STATE_WRITE_ATTEMPTS = 3
_SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS = 240
# Read-only spoken context sent when a shot has to be described from its own
# words. Generous enough to hold the sentences a split span was cut from, and
# capped because it is interpolated into the prompt verbatim.
_SHOT_RESCUE_NARRATION_CONTEXT_MAX_CHARS = 1200
_SEMANTIC_SPAN_OUTPUT_FIELDS = {
    "start_unit",
    "end_unit_exclusive",
    "visual_requirement",
}
_VISUAL_BEAT_PREFERRED_MIN_SECONDS = 2.0
_VISUAL_BEAT_RAPID_CUT_SECONDS = VISUAL_BEAT_RAPID_CUT_SECONDS
_VISUAL_BEAT_PREFERRED_MAX_SECONDS = 5.0
_VISUAL_BEAT_LONG_SPLIT_TARGET_SECONDS = 4.0
_VISUAL_BEAT_TIME_TOLERANCE_SECONDS = 1e-6
_CROSS_POST_STATE_RETRY_DELAY_SECONDS = 0.1
_LOOMLOOM_STATE_WRITE_ATTEMPTS = 3
_LOOMLOOM_STATE_RETRY_DELAY_SECONDS = 0.1
_INTERRUPTED_CROSS_POST_ERROR = (
    "cross-posting was interrupted before the process completed"
)
# 视频配乐服务只需实现 ``is_enabled`` 和 ``generate_bgm``。供应商差异集中在
# 文件扩展名、领域异常和 WebUI 警告代码；任务编排、0 音量短路及失败降级
# 全部复用同一路径，避免后续新增供应商时维护多份相似流程。
_VIDEO_MUSIC_PROVIDERS = {
    "sonilo": {
        "service": sonilo,
        "error_type": sonilo.SoniloError,
        "suffix": ".m4a",
        "warning_code": "sonilo_bgm_failed",
        "display_name": "Sonilo",
    },
    "elevenlabs": {
        "service": elevenlabs_music,
        "error_type": elevenlabs_music.ElevenLabsMusicError,
        "suffix": ".mp3",
        "warning_code": "elevenlabs_bgm_failed",
        "display_name": "ElevenLabs",
    },
}


def _get_video_music_prompt(params: VideoParams) -> str:
    """
    读取当前视频配乐供应商实际使用的提示词。

    新任务统一使用供应商无关字段；旧 Sonilo CLI 参数和历史任务仍可能只有
    ``sonilo_bgm_prompt``，因此仅在 Sonilo 通用字段为空时读取旧字段。
    """
    prompt = str(params.video_music_prompt or "").strip()
    if params.bgm_type == "sonilo" and not prompt:
        prompt = str(params.sonilo_bgm_prompt or "").strip()
    return prompt


def is_task_busy(task: dict | None) -> bool:
    """判断任务是否仍在生成或发布，供所有删除入口复用。"""
    if not task:
        return False

    state = task.get("state")
    try:
        state = int(state)
    except (TypeError, ValueError):
        pass

    # 视频生成和跨平台发布都可能继续读取任务目录。统一视为忙碌状态，
    # 可以避免 API 与 WebUI 分别维护规则后出现一个允许删除、另一个禁止
    # 删除的不一致行为。
    return (
        state == const.TASK_STATE_PROCESSING
        or task.get("cross_post_state") in _ACTIVE_CROSS_POST_STATES
    )


def _register_cross_post_future(task_id: str, future: Future) -> None:
    """登记当前进程持有的发布 Future，供启动恢复和测试判断真实运行状态。"""
    with _cross_post_registry_lock:
        _cross_post_futures[task_id] = future


def _unregister_cross_post_future(task_id: str, future: Future | None = None) -> None:
    """仅移除匹配的 Future，避免旧回调误删同任务后续注册的新工作。"""
    with _cross_post_registry_lock:
        current = _cross_post_futures.get(task_id)
        if current is None or (future is not None and current is not future):
            return
        _cross_post_futures.pop(task_id, None)


def _is_cross_post_active_in_process(task_id: str) -> bool:
    """判断当前进程是否仍持有未结束的发布任务。"""
    with _cross_post_registry_lock:
        future = _cross_post_futures.get(task_id)
        return future is not None and not future.done()


def _is_windows_process_alive(process_id: int) -> bool:
    """通过只读 Win32 API 判断进程状态，避免用 os.kill 误终止进程。"""
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # ctypes 默认把未声明的返回值当作 32 位 int。Windows 64 位进程句柄可能
    # 因此被截断，必须显式声明 Win32 函数签名后再调用。
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        process_id,
    )
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code == error_invalid_parameter:
            return False
        if error_code == error_access_denied:
            # 进程存在但当前用户无查询权限时，必须保守地视为存活，避免错误
            # 回收其它账户正在执行的发布任务。
            return True
        logger.warning(
            "failed to open cross-post owner process on Windows, "
            f"process_id: {process_id}, error_code: {error_code}"
        )
        return True

    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            error_code = ctypes.get_last_error()
            logger.warning(
                "failed to read cross-post owner process state on Windows, "
                f"process_id: {process_id}, error_code: {error_code}"
            )
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _is_cross_post_owner_alive(owner: str | None) -> bool:
    """判断持久化发布任务的本机进程是否仍存在。"""
    if not owner:
        return False

    try:
        hostname, process_id_text, _ = owner.split(":", 2)
        process_id = int(process_id_text)
    except (TypeError, ValueError):
        logger.warning(f"invalid cross-post owner metadata: {owner}")
        return False

    # 无法可靠探测其它主机上的进程。共享 Redis 的多主机部署中必须保守地
    # 视为仍在运行，避免当前节点误删另一节点正在读取的视频文件。
    if hostname != socket.gethostname():
        return True

    # 当前进程内是否仍有真实发布工作，已经由 Future 注册表准确判断。运行到
    # 这里说明注册表中没有对应 Future，即使 owner 与当前进程完全一致，也应
    # 视为已中断；这可以覆盖终态写入持续失败、Future 已结束的场景。
    if process_id == os.getpid():
        return False

    # Windows 的 os.kill(pid, 0) 与 POSIX 语义不同，可能直接终止目标进程。
    # 使用只申请查询权限的 Win32 API，不向目标进程发送任何信号。
    if os.name == "nt":
        return _is_windows_process_alive(process_id)

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        logger.warning(
            f"failed to inspect cross-post owner process, owner: {owner}, error: {exc}"
        )
        return True
    return True


def _mark_task_failed(
    task_id: str,
    stage: str,
    error: str,
    details: dict | None = None,
) -> dict:
    """记录结构化失败信息，并保留任务失败前已经到达的进度。"""
    existing_task = None
    try:
        existing_task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.warning(f"failed to read task state before failure update: {exc}")

    # 具体服务函数通常比编排层拥有更准确的错误原因。后续的空结果检查
    # 不能再用通用文案覆盖它，否则 API 调用方仍然只能看到模糊信息。
    if (
        existing_task
        and existing_task.get("state") == const.TASK_STATE_FAILED
        and existing_task.get("error")
    ):
        return existing_task

    message = str(error or "unknown task error").strip()
    progress = int((existing_task or {}).get("progress", 0) or 0)
    logger.error(f"task failed, task_id: {task_id}, stage: {stage}, error: {message}")
    failure = {
        "task_id": task_id,
        "state": const.TASK_STATE_FAILED,
        "progress": progress,
        "failed_stage": stage,
        "error": message,
    }
    # 某些外部任务已经创建了可用于恢复或排障的远端 ID。失败状态需要保留
    # 这些非敏感字段，但不能允许调用方覆盖统一的状态、进度和错误结构。
    failure_details = {
        key: value for key, value in dict(details or {}).items() if key not in failure
    }
    failure.update(failure_details)
    sm.state.update_task(
        task_id,
        state=failure["state"],
        progress=failure["progress"],
        failed_stage=failure["failed_stage"],
        error=failure["error"],
        **failure_details,
    )
    return failure


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    if not video_script:
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
        )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        _mark_task_failed(task_id, "script", "failed to generate video script")
        return None

    return video_script


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if not video_terms:
        # 开启素材按文案顺序匹配后，关键词本身也必须按脚本叙事顺序生成；
        # 否则后续即使顺序下载和顺序拼接，也只能复用一组全局主题词，
        # 无法改善“后面内容的画面提前出现”的问题。
        video_terms = llm.generate_terms(
            video_subject=params.video_subject,
            video_script=video_script,
            amount=8 if params.match_materials_to_script else 5,
            match_script_order=params.match_materials_to_script,
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        _mark_task_failed(
            task_id,
            "terms",
            "failed to generate video search terms",
        )
        return None

    # 可选的 TwelveLabs Marengo 语义重排：未启用时返回原顺序，无任何副作用。
    # 顺序匹配模式下关键词顺序本身就是脚本叙事顺序，必须保持原样，故跳过。
    if not params.match_materials_to_script:
        video_terms = twelvelabs.rerank_terms_by_subject(
            video_subject=params.video_subject,
            search_terms=video_terms,
        )

    return video_terms


def resolve_narration_timing_source(
    params: VideoParams,
    sub_maker,
) -> NarrationTimingSource:
    """Describe where subtitle timing came from without overstating estimates."""
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    if subtitle_provider == "whisper":
        return "whisper"

    voice_name = voice.parse_voice_name(params.voice_name or "")
    if sub_maker is not None and voice.is_azure_v2_voice(voice_name):
        return "azure_tts_boundary"

    estimated_voice = any(
        (
            voice.is_siliconflow_voice(voice_name),
            voice.is_gemini_voice(voice_name),
            voice.is_mimo_voice(voice_name),
            voice.is_minimax_voice(voice_name),
            voice.is_elevenlabs_voice(voice_name),
            voice.is_chatterbox_voice(voice_name),
            voice.is_no_voice(voice_name),
        )
    )
    if sub_maker is None or estimated_voice:
        return "estimated"
    return "edge_tts_boundary"


def resolve_timed_narration_timing_source(
    params: VideoParams,
    sub_maker,
) -> NarrationTimingSource | None:
    """Resolve the strongest timing carried by the TTS result itself.

    This intentionally does not inspect ``subtitle_provider``. Display subtitles
    may use Whisper while an Edge/Azure TTS result still carries more direct word
    boundaries that should remain available as an independent timing artifact.
    """
    if sub_maker is None:
        return None

    voice_name = voice.parse_voice_name(params.voice_name or "")
    if voice.is_azure_v2_voice(voice_name):
        return "azure_tts_boundary"
    if any(
        (
            voice.is_siliconflow_voice(voice_name),
            voice.is_gemini_voice(voice_name),
            voice.is_mimo_voice(voice_name),
            voice.is_minimax_voice(voice_name),
            voice.is_elevenlabs_voice(voice_name),
            voice.is_chatterbox_voice(voice_name),
            voice.is_no_voice(voice_name),
        )
    ):
        return "estimated"
    return "edge_tts_boundary"


def _timing_quality(
    timing_source: NarrationTimingSource,
) -> NarrationTimingQuality:
    if timing_source in {"edge_tts_boundary", "azure_tts_boundary"}:
        return "boundary"
    if timing_source == "whisper":
        return "speech_recognition"
    return "estimated"


def _parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = (int(part) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid SRT timestamp: {value!r}")
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def _normalize_narration_text(value: str) -> str:
    normalized = utils.normalize_script_for_subtitle_matching(value or "")
    return re.sub(r"[_\W]+", "", normalized, flags=re.UNICODE).casefold()


def build_narration_slots(
    subtitle_path: str,
    audio_duration: float,
    timing_source: NarrationTimingSource,
    expected_script: str = "",
) -> list[NarrationSlot]:
    """Parse and validate the existing SRT as the canonical narration timeline."""
    if not math.isfinite(audio_duration) or audio_duration <= 0:
        raise ValueError("narration timeline requires a positive audio duration")

    subtitle_items = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_items:
        raise ValueError(
            "missing narration timeline: subtitle SRT is empty or unavailable"
        )

    narration_slots: list[NarrationSlot] = []
    previous_start = -1.0
    previous_end = -1.0
    for slot_index, (_, timestamp_line, raw_text) in enumerate(subtitle_items, start=1):
        timestamp_parts = timestamp_line.split(" --> ")
        if len(timestamp_parts) != 2:
            raise ValueError(
                f"narration slot {slot_index} has an invalid timestamp range"
            )
        start_time = _parse_srt_timestamp(timestamp_parts[0])
        end_time = _parse_srt_timestamp(timestamp_parts[1])
        text = " ".join(str(raw_text or "").split())

        if not text:
            raise ValueError(f"narration slot {slot_index} has empty text")
        if end_time <= start_time:
            raise ValueError(
                f"narration slot {slot_index} must have end_time > start_time"
            )
        if start_time < previous_start or end_time < previous_end:
            raise ValueError(
                f"narration slot {slot_index} timestamps are not ascending"
            )
        if end_time > audio_duration + 0.001:
            raise ValueError(
                f"narration slot {slot_index} ends after the audio duration: "
                f"end={end_time:.3f}, audio={audio_duration:.3f}"
            )

        narration_slots.append(
            NarrationSlot(
                index=slot_index,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                text=text,
                timing_source=timing_source,
            )
        )
        previous_start = start_time
        previous_end = end_time

    expected_text = _normalize_narration_text(expected_script)
    actual_text = "".join(
        _normalize_narration_text(slot.text) for slot in narration_slots
    )
    if expected_text and actual_text != expected_text:
        expected_lines = utils.split_string_by_punctuations(
            utils.normalize_script_for_subtitle_matching(expected_script)
        )
        missing_lines = [
            line.strip()
            for line in expected_lines
            if _normalize_narration_text(line)
            and _normalize_narration_text(line) not in actual_text
        ]
        missing_preview = missing_lines[:3] or ["unmatched narration text"]
        raise ValueError(
            "missing narration in subtitle timeline: " + " | ".join(missing_preview)
        )

    return narration_slots


def associate_timed_units_with_narration_slots(
    timed_units: list[TimedNarrationUnit],
    narration_slots: list[NarrationSlot],
) -> list[TimedNarrationUnit]:
    """Associate units by exact normalized text position, never fuzzy matching."""
    if not timed_units:
        return []
    if not narration_slots:
        raise ValueError("timed narration units require narration slots")

    unit_ranges: list[tuple[int, int, TimedNarrationUnit]] = []
    unit_cursor = 0
    for unit in timed_units:
        normalized = voice.normalize_timing_alignment_text(unit.text)
        unit_end = unit_cursor + len(normalized)
        unit_ranges.append((unit_cursor, unit_end, unit))
        unit_cursor = unit_end

    slot_ranges: list[tuple[int, int, NarrationSlot]] = []
    slot_cursor = 0
    for narration_slot in narration_slots:
        normalized = voice.normalize_timing_alignment_text(narration_slot.text)
        slot_end = slot_cursor + len(normalized)
        slot_ranges.append((slot_cursor, slot_end, narration_slot))
        slot_cursor = slot_end

    unit_text = "".join(
        voice.normalize_timing_alignment_text(unit.text) for unit in timed_units
    )
    slot_text = "".join(
        voice.normalize_timing_alignment_text(slot.text) for slot in narration_slots
    )
    if unit_cursor != slot_cursor or unit_text != slot_text:
        raise ValueError("timed narration units do not align with narration slots")

    slot_position = 0
    for unit_start, unit_end, unit in unit_ranges:
        while (
            slot_position < len(slot_ranges)
            and unit_start >= slot_ranges[slot_position][1]
        ):
            slot_position += 1
        if slot_position >= len(slot_ranges):
            raise ValueError("timed narration unit is outside narration slots")
        slot_start, slot_end, narration_slot = slot_ranges[slot_position]
        if unit_start >= slot_start and unit_end <= slot_end:
            unit.source_narration_slot_index = narration_slot.index
        else:
            # A coarse provider cue can genuinely cross a sentence/SRT boundary.
            # Keep its timing and text intact, but do not claim one source slot.
            unit.source_narration_slot_index = None
    return timed_units


def validate_semantic_visual_span_specs(
    raw_specs,
    unit_count: int,
) -> list[dict]:
    """Validate complete unit coverage without trusting any LLM-authored text."""
    if unit_count <= 0:
        raise ValueError("semantic span validation requires narration units")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("semantic span response must be a non-empty JSON array")

    validated: list[dict] = []
    expected_start = 0
    for position, raw_spec in enumerate(raw_specs, start=1):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"semantic span {position} must be an object")
        if set(raw_spec) != _SEMANTIC_SPAN_OUTPUT_FIELDS:
            raise ValueError(f"semantic span {position} contains unsupported fields")

        start_unit = raw_spec["start_unit"]
        end_unit_exclusive = raw_spec["end_unit_exclusive"]
        if (
            isinstance(start_unit, bool)
            or not isinstance(start_unit, int)
            or isinstance(end_unit_exclusive, bool)
            or not isinstance(end_unit_exclusive, int)
        ):
            raise ValueError("semantic span unit indexes must be integers")
        if start_unit != expected_start:
            raise ValueError("semantic spans contain a gap, overlap, or reordering")
        if start_unit < 0 or end_unit_exclusive > unit_count:
            raise ValueError("semantic span unit index is outside narration units")
        if start_unit >= end_unit_exclusive:
            raise ValueError("semantic span must contain at least one narration unit")

        visual_requirement = raw_spec["visual_requirement"]
        if not isinstance(visual_requirement, str) or not visual_requirement.strip():
            raise ValueError("semantic span visual_requirement must be non-empty")
        visual_requirement = visual_requirement.strip()
        if len(visual_requirement) > _SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS:
            raise ValueError("semantic span visual_requirement is too long")

        validated.append(
            {
                "start_unit": start_unit,
                "end_unit_exclusive": end_unit_exclusive,
                "visual_requirement": visual_requirement,
            }
        )
        expected_start = end_unit_exclusive

    if expected_start != unit_count:
        raise ValueError("semantic spans do not cover every narration unit")
    return validated


def reconstruct_semantic_spoken_text(
    narration_script: str,
    timed_units: list[TimedNarrationUnit],
    start_unit: int,
    end_unit_exclusive: int,
) -> str:
    """Slice source narration by S1 offsets; never use LLM-authored source text."""
    if not (0 <= start_unit < end_unit_exclusive <= len(timed_units)):
        raise ValueError("semantic spoken-text range is outside narration units")

    first = timed_units[start_unit]
    if first.script_start_char is None:
        raise ValueError("semantic spoken-text range has no source offset")
    start_char = 0 if start_unit == 0 else first.script_start_char
    if end_unit_exclusive == len(timed_units):
        end_char = len(narration_script)
    else:
        end_char = timed_units[end_unit_exclusive].script_start_char
        if end_char is None:
            raise ValueError("semantic spoken-text range has no source offset")
    if start_char < 0 or end_char <= start_char or end_char > len(narration_script):
        raise ValueError("semantic spoken-text source offsets are invalid")

    spoken_text = narration_script[start_char:end_char].strip()
    expected = "".join(
        voice.normalize_timing_alignment_text(unit.text)
        for unit in timed_units[start_unit:end_unit_exclusive]
    )
    if not spoken_text or voice.normalize_timing_alignment_text(spoken_text) != expected:
        raise ValueError("semantic spoken text does not match its source units")
    return spoken_text


def _semantic_span_slot_indexes(
    timed_units: list[TimedNarrationUnit],
    narration_slots: list[NarrationSlot],
    start_unit: int,
    end_unit_exclusive: int,
) -> list[int]:
    normalized_units = [
        voice.normalize_timing_alignment_text(unit.text) for unit in timed_units
    ]
    normalized_slots = [
        voice.normalize_timing_alignment_text(slot.text) for slot in narration_slots
    ]
    unit_lengths = [len(text) for text in normalized_units]
    slot_lengths = [len(text) for text in normalized_slots]
    if "".join(normalized_units) == "".join(normalized_slots):
        span_start = sum(unit_lengths[:start_unit])
        span_end = sum(unit_lengths[:end_unit_exclusive])
        result: list[int] = []
        slot_start = 0
        for narration_slot, slot_length in zip(narration_slots, slot_lengths):
            slot_end = slot_start + slot_length
            if slot_start < span_end and slot_end > span_start:
                result.append(narration_slot.index)
            slot_start = slot_end
        if result:
            return result

    return list(
        dict.fromkeys(
            unit.source_narration_slot_index
            for unit in timed_units[start_unit:end_unit_exclusive]
            if unit.source_narration_slot_index is not None
        )
    )


def _semantic_span_timing(
    units: list[TimedNarrationUnit] | list[SemanticVisualSpan],
) -> tuple[NarrationTimingSource, NarrationTimingQuality]:
    """Collapse the timing provenance of several ordered pieces into one pair.

    Accepts timed units or already-built spans: merging spans has to answer the
    same question about the pieces it absorbs, and the answer must degrade the
    same way -- one shared source or "estimated", and the weakest quality wins.
    """
    timing_sources = {unit.timing_source for unit in units}
    timing_source: NarrationTimingSource = (
        next(iter(timing_sources)) if len(timing_sources) == 1 else "estimated"
    )
    qualities = {unit.timing_quality for unit in units}
    if "estimated" in qualities:
        timing_quality: NarrationTimingQuality = "estimated"
    elif "speech_recognition" in qualities:
        timing_quality = "speech_recognition"
    else:
        timing_quality = "boundary"
    return timing_source, timing_quality


def _bounded_fallback_visual_requirement(text: str) -> str:
    compacted = " ".join(str(text or "").split())
    if len(compacted) <= _SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS:
        return compacted
    prefix = compacted[: _SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS - 3]
    word_boundary = prefix.rfind(" ")
    if word_boundary > _SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS // 2:
        prefix = prefix[:word_boundary]
    return prefix.rstrip() + "..."


def build_semantic_visual_spans_from_specs(
    narration_script: str,
    timed_units: list[TimedNarrationUnit],
    narration_slots: list[NarrationSlot],
    raw_specs,
) -> list[SemanticVisualSpan]:
    specs = validate_semantic_visual_span_specs(raw_specs, len(timed_units))
    spans: list[SemanticVisualSpan] = []
    for index, spec in enumerate(specs, start=1):
        start_unit = spec["start_unit"]
        end_unit_exclusive = spec["end_unit_exclusive"]
        source_units = timed_units[start_unit:end_unit_exclusive]
        timing_source, timing_quality = _semantic_span_timing(source_units)
        spans.append(
            SemanticVisualSpan(
                index=index,
                start_unit=start_unit,
                end_unit_exclusive=end_unit_exclusive,
                spoken_text=reconstruct_semantic_spoken_text(
                    narration_script,
                    timed_units,
                    start_unit,
                    end_unit_exclusive,
                ),
                visual_requirement=spec["visual_requirement"],
                source_narration_slot_indexes=_semantic_span_slot_indexes(
                    timed_units,
                    narration_slots,
                    start_unit,
                    end_unit_exclusive,
                ),
                start_time=source_units[0].start_time,
                end_time=source_units[-1].end_time,
                timing_source=timing_source,
                timing_quality=timing_quality,
                grouping_source="llm",
            )
        )
    return spans


def build_narration_slot_semantic_fallback(
    narration_script: str,
    timed_units: list[TimedNarrationUnit],
    narration_slots: list[NarrationSlot],
) -> list[SemanticVisualSpan]:
    """Build deterministic slot-led spans, preserving indivisible coarse units."""
    if not narration_slots:
        return []
    if not timed_units:
        return [
            SemanticVisualSpan(
                index=index,
                start_unit=None,
                end_unit_exclusive=None,
                spoken_text=slot.text,
                visual_requirement=_bounded_fallback_visual_requirement(slot.text),
                source_narration_slot_indexes=[slot.index],
                start_time=slot.start_time,
                end_time=slot.end_time,
                timing_source=slot.timing_source,
                timing_quality=_timing_quality(slot.timing_source),
                grouping_source="narration_slot_fallback",
            )
            for index, slot in enumerate(narration_slots, start=1)
        ]

    normalized_units = [
        voice.normalize_timing_alignment_text(unit.text) for unit in timed_units
    ]
    normalized_slots = [
        voice.normalize_timing_alignment_text(slot.text) for slot in narration_slots
    ]
    unit_lengths = [len(text) for text in normalized_units]
    slot_lengths = [len(text) for text in normalized_slots]
    if "".join(normalized_units) != "".join(normalized_slots):
        raise ValueError("narration-slot fallback text does not align with timing units")

    slot_boundaries: set[int] = set()
    slot_cursor = 0
    for slot_length in slot_lengths[:-1]:
        slot_cursor += slot_length
        slot_boundaries.add(slot_cursor)

    cuts = [0]
    unit_cursor = 0
    for unit_index, unit_length in enumerate(unit_lengths, start=1):
        unit_cursor += unit_length
        if unit_cursor in slot_boundaries:
            cuts.append(unit_index)
    cuts.append(len(timed_units))

    spans: list[SemanticVisualSpan] = []
    for start_unit, end_unit_exclusive in zip(cuts, cuts[1:]):
        source_units = timed_units[start_unit:end_unit_exclusive]
        timing_source, timing_quality = _semantic_span_timing(source_units)
        spoken_text = reconstruct_semantic_spoken_text(
            narration_script,
            timed_units,
            start_unit,
            end_unit_exclusive,
        )
        spans.append(
            SemanticVisualSpan(
                index=len(spans) + 1,
                start_unit=start_unit,
                end_unit_exclusive=end_unit_exclusive,
                spoken_text=spoken_text,
                visual_requirement=_bounded_fallback_visual_requirement(spoken_text),
                source_narration_slot_indexes=_semantic_span_slot_indexes(
                    timed_units,
                    narration_slots,
                    start_unit,
                    end_unit_exclusive,
                ),
                start_time=source_units[0].start_time,
                end_time=source_units[-1].end_time,
                timing_source=timing_source,
                timing_quality=timing_quality,
                grouping_source="narration_slot_fallback",
            )
        )
    return spans


def _consolidate_repaired_spans(
    narration_script: str,
    timed_units: list[TimedNarrationUnit],
    spans: list[SemanticVisualSpan],
    requirements: dict[int, str],
) -> list[SemanticVisualSpan]:
    """Rebuild the timeline around the lines that have visible content.

    Every span whose line has no filmable requirement is absorbed by the nearest
    span that has one -- the previous one, or the first one for a run of leading
    lines. This is the free version of the material stage's unfillable-beat
    merge: it happens before a single stock request, so nothing is spent proving
    that "Patient." has no footage, and the narration timing never moves because
    the absorbing span simply covers the absorbed window too.

    Returns an empty list when no line has visible content, which cannot be
    consolidated into anything and must be handled by the caller.
    """
    groups: list[tuple[str, list[SemanticVisualSpan]]] = []
    leading: list[SemanticVisualSpan] = []
    for span in spans:
        requirement = requirements.get(span.index)
        if requirement:
            groups.append((requirement, [span]))
        elif groups:
            groups[-1][1].append(span)
        else:
            leading.append(span)
    if not groups:
        return []
    if leading:
        groups[0] = (groups[0][0], leading + groups[0][1])

    merged: list[SemanticVisualSpan] = []
    for position, (requirement, members) in enumerate(groups, start=1):
        first = members[0]
        last = members[-1]
        start_unit = first.start_unit
        end_unit_exclusive = last.end_unit_exclusive
        if timed_units and start_unit is not None and end_unit_exclusive is not None:
            spoken_text = reconstruct_semantic_spoken_text(
                narration_script,
                timed_units,
                start_unit,
                end_unit_exclusive,
            )
            timing_source, timing_quality = _semantic_span_timing(
                timed_units[start_unit:end_unit_exclusive]
            )
        else:
            spoken_text = " ".join(
                member.spoken_text.strip()
                for member in members
                if member.spoken_text.strip()
            )
            timing_source, timing_quality = _semantic_span_timing(members)
        merged.append(
            SemanticVisualSpan(
                index=position,
                start_unit=start_unit,
                end_unit_exclusive=end_unit_exclusive,
                spoken_text=spoken_text,
                visual_requirement=requirement,
                source_narration_slot_indexes=list(
                    dict.fromkeys(
                        slot_index
                        for member in members
                        for slot_index in member.source_narration_slot_indexes
                    )
                ),
                start_time=first.start_time,
                end_time=last.end_time,
                timing_source=timing_source,
                timing_quality=timing_quality,
                grouping_source="narration_slot_repaired",
            )
        )
    return merged


def repair_fallback_visual_requirements(
    narration_script: str,
    timed_units: list[TimedNarrationUnit],
    spans: list[SemanticVisualSpan],
) -> list[SemanticVisualSpan]:
    """Replace spoken narration lines with requirements a clip can satisfy.

    The slot-led fallback has one span per narration line, so its
    ``visual_requirement`` is the spoken line itself. Handing that to stock
    search and to candidate verification asks the pipeline to find footage of a
    sentence: a line like "Patient." can never be filled, yet it still buys a
    full search and a full round of candidate analysis on every phrasing of
    every provider before the failure ladder gives up on it. One extra
    provider-neutral call is much cheaper than that, and the lines it reports as
    having no visible content of their own cost nothing at all -- a neighbour
    absorbs their window.

    Returns the original spans unchanged when the repair produced nothing
    usable, so the failure stays visible in ``grouping_source`` instead of
    looking like a planned timeline.
    """
    if not spans:
        return spans

    try:
        repaired_requirements = llm.generate_narration_visual_requirements(
            narration_text=narration_script,
            narration_lines=[
                {"index": span.index, "spoken_text": span.spoken_text}
                for span in spans
            ],
        )
    except Exception as exc:
        logger.warning(
            "narration visual requirement repair failed: "
            f"error={type(exc).__name__}"
        )
        return spans
    if not repaired_requirements:
        logger.warning(
            "narration visual requirement repair returned nothing usable; "
            f"visual requirements are still spoken narration: lines={len(spans)}"
        )
        return spans

    requirements: dict[int, str] = {}
    over_long = 0
    for span in spans:
        answer = repaired_requirements.get(span.index)
        requirement = " ".join(str(answer or "").split())
        if not requirement:
            continue
        # The same ceiling the LLM grouping path validates against. An answer
        # over it is not adopted, because a bloated requirement is carried into
        # the checklist and into every adjudication prompt of that beat.
        if len(requirement) > _SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS:
            over_long += 1
            continue
        requirements[span.index] = requirement
    if over_long:
        logger.warning(
            "repaired visual requirements exceeded the span limit and were "
            f"treated as unfilmable lines: count={over_long}"
        )

    repaired_spans = _consolidate_repaired_spans(
        narration_script,
        timed_units,
        spans,
        requirements,
    )
    if not repaired_spans:
        logger.warning(
            "no narration line has camera-visible content of its own; keeping "
            f"the spoken slot-led requirements: lines={len(spans)}"
        )
        return spans

    logger.info(
        "narration visual requirements repaired: "
        f"lines={len(spans)}, filmable_lines={len(requirements)}, "
        f"spans={len(repaired_spans)}"
    )
    return repaired_spans


def generate_semantic_visual_spans(
    narration_script: str,
    timed_units: list[TimedNarrationUnit],
    narration_slots: list[NarrationSlot],
) -> list[SemanticVisualSpan]:
    """Use one provider-neutral LLM call, then fail wholly to slot-led spans."""
    if timed_units:
        try:
            raw_specs = llm.generate_semantic_visual_span_specs(
                narration_text=narration_script,
                timed_units=[
                    {
                        "text": unit.text,
                        "source_narration_slot_index": (
                            unit.source_narration_slot_index
                        ),
                    }
                    for unit in timed_units
                ],
            )
            if raw_specs is not None:
                return build_semantic_visual_spans_from_specs(
                    narration_script,
                    timed_units,
                    narration_slots,
                    raw_specs,
                )
        except Exception as exc:
            logger.warning(
                "semantic visual grouping rejected; using narration-slot fallback: "
                f"error={type(exc).__name__}"
            )

    try:
        fallback_spans = build_narration_slot_semantic_fallback(
            narration_script,
            timed_units,
            narration_slots,
        )
        fallback_units = timed_units
    except ValueError as exc:
        logger.warning(
            "timed narration fallback unavailable; using coarse narration slots: "
            f"reason={exc}"
        )
        fallback_spans = build_narration_slot_semantic_fallback(
            narration_script,
            [],
            narration_slots,
        )
        fallback_units = []

    # The fallback owns the timeline shape; it must not also decide what a shot
    # has to show. Its requirement is the spoken line, so it is repaired before
    # anything downstream can mistake narration for a description of footage.
    return repair_fallback_visual_requirements(
        narration_script,
        fallback_units,
        fallback_spans,
    )


def semantic_visual_requirements_are_spoken_narration(
    semantic_visual_spans: list[SemanticVisualSpan],
) -> bool:
    """True when no span in the timeline describes footage.

    ``narration_slot_fallback`` survives only when semantic grouping failed and
    the repair could not turn a single spoken line into a visible situation.
    Every requirement in the timeline is then the narration itself, so the beat
    path would search stock catalogs for sentences and pay the verifier to
    reject real footage for lines like "Patient." -- a whole video of guaranteed
    rejections. The proven fixed-slot timeline is worth more than that.
    """
    return bool(semantic_visual_spans) and all(
        span.grouping_source == "narration_slot_fallback"
        for span in semantic_visual_spans
    )


def _validate_semantic_spans_for_visual_beats(
    semantic_visual_spans: list[SemanticVisualSpan],
    timed_units: list[TimedNarrationUnit],
    audio_duration: float,
) -> None:
    if not semantic_visual_spans:
        raise ValueError("visual beats require semantic visual spans")

    uses_unit_ranges = semantic_visual_spans[0].start_unit is not None
    expected_unit = 0
    previous_end = 0.0
    for position, span in enumerate(semantic_visual_spans, start=1):
        if span.index != position:
            raise ValueError("semantic visual span indexes must be sequential")
        if not span.spoken_text.strip() or not span.visual_requirement.strip():
            raise ValueError("semantic visual spans require visible text metadata")
        if not math.isfinite(span.start_time) or not math.isfinite(span.end_time):
            raise ValueError("semantic visual span timing must be finite")
        if span.start_time < 0 or span.end_time <= span.start_time:
            raise ValueError("semantic visual span timing is invalid")
        if span.end_time > audio_duration + _VISUAL_BEAT_TIME_TOLERANCE_SECONDS:
            raise ValueError("semantic visual span exceeds the audio duration")
        if (
            position > 1
            and span.start_time
            < previous_end - _VISUAL_BEAT_TIME_TOLERANCE_SECONDS
        ):
            raise ValueError("semantic visual spans overlap or are reordered")

        has_start_unit = span.start_unit is not None
        has_end_unit = span.end_unit_exclusive is not None
        if has_start_unit != has_end_unit or has_start_unit != uses_unit_ranges:
            raise ValueError("semantic visual span unit ranges are inconsistent")
        if uses_unit_ranges:
            start_unit = span.start_unit
            end_unit_exclusive = span.end_unit_exclusive
            if start_unit is None or end_unit_exclusive is None:
                raise ValueError("semantic visual span unit range is missing")
            if (
                start_unit != expected_unit
                or start_unit < 0
                or end_unit_exclusive <= start_unit
                or end_unit_exclusive > len(timed_units)
            ):
                raise ValueError("semantic visual span unit coverage is invalid")
            source_units = timed_units[start_unit:end_unit_exclusive]
            if not math.isclose(
                span.start_time,
                source_units[0].start_time,
                rel_tol=0.0,
                abs_tol=_VISUAL_BEAT_TIME_TOLERANCE_SECONDS,
            ) or not math.isclose(
                span.end_time,
                source_units[-1].end_time,
                rel_tol=0.0,
                abs_tol=_VISUAL_BEAT_TIME_TOLERANCE_SECONDS,
            ):
                raise ValueError("semantic visual span timing does not match its units")
            expected_unit = end_unit_exclusive
        previous_end = span.end_time

    if uses_unit_ranges and expected_unit != len(timed_units):
        raise ValueError("semantic visual spans do not cover every timing unit")


def _desired_visual_beat_shot_count(duration: float) -> int:
    if duration <= (
        _VISUAL_BEAT_PREFERRED_MAX_SECONDS
        + _VISUAL_BEAT_TIME_TOLERANCE_SECONDS
    ):
        return 1
    minimum_count = math.ceil(duration / _VISUAL_BEAT_PREFERRED_MAX_SECONDS)
    target_count = math.floor(
        duration / _VISUAL_BEAT_LONG_SPLIT_TARGET_SECONDS + 0.5
    )
    return max(2, minimum_count, target_count)


def _visual_beat_unit_cut_candidates(
    span: SemanticVisualSpan,
    timed_units: list[TimedNarrationUnit],
    group_start: float,
    group_end: float,
) -> list[tuple[int, float]]:
    if span.start_unit is None or span.end_unit_exclusive is None:
        return []

    candidates: list[tuple[int, float]] = []
    previous_time = group_start
    for unit_index in range(span.start_unit + 1, span.end_unit_exclusive):
        cut_time = timed_units[unit_index].start_time
        if (
            not math.isfinite(cut_time)
            or cut_time <= group_start + _VISUAL_BEAT_TIME_TOLERANCE_SECONDS
            or cut_time >= group_end - _VISUAL_BEAT_TIME_TOLERANCE_SECONDS
            or cut_time <= previous_time + _VISUAL_BEAT_TIME_TOLERANCE_SECONDS
        ):
            continue
        candidates.append((unit_index, cut_time))
        previous_time = cut_time
    return candidates


def _choose_balanced_visual_beat_cuts(
    candidates: list[tuple[int, float]],
    group_start: float,
    group_end: float,
    desired_shot_count: int,
) -> list[tuple[int, float]]:
    """Choose deterministic, balanced cuts only at real timing-unit starts."""
    desired_shot_count = min(desired_shot_count, len(candidates) + 1)
    for shot_count in range(desired_shot_count, 1, -1):
        # state: last candidate position -> (squared timing error, chosen positions)
        states: dict[int, tuple[float, tuple[int, ...]]] = {-1: (0.0, ())}
        for cut_number in range(1, shot_count):
            ideal_time = group_start + (
                (group_end - group_start) * cut_number / shot_count
            )
            remaining_segments = shot_count - cut_number
            next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
            for previous_position, (cost, chosen_positions) in states.items():
                previous_time = (
                    group_start
                    if previous_position < 0
                    else candidates[previous_position][1]
                )
                for candidate_position in range(
                    previous_position + 1,
                    len(candidates),
                ):
                    cut_time = candidates[candidate_position][1]
                    if (
                        cut_time - previous_time
                        < _VISUAL_BEAT_RAPID_CUT_SECONDS
                        - _VISUAL_BEAT_TIME_TOLERANCE_SECONDS
                    ):
                        continue
                    if (
                        group_end - cut_time
                        < remaining_segments * _VISUAL_BEAT_RAPID_CUT_SECONDS
                        - _VISUAL_BEAT_TIME_TOLERANCE_SECONDS
                    ):
                        continue
                    candidate_state = (
                        cost + (cut_time - ideal_time) ** 2,
                        chosen_positions + (candidate_position,),
                    )
                    current_state = next_states.get(candidate_position)
                    if current_state is None or candidate_state < current_state:
                        next_states[candidate_position] = candidate_state
            states = next_states
            if not states:
                break

        valid_states = [
            state
            for candidate_position, state in states.items()
            if candidate_position >= 0
            and group_end - candidates[candidate_position][1]
            >= _VISUAL_BEAT_RAPID_CUT_SECONDS
            - _VISUAL_BEAT_TIME_TOLERANCE_SECONDS
        ]
        if valid_states:
            _, chosen_positions = min(valid_states)
            return [candidates[position] for position in chosen_positions]
    return []


def _visual_beat_timing(
    span: SemanticVisualSpan,
    source_units: list[TimedNarrationUnit],
) -> tuple[NarrationTimingSource, NarrationTimingQuality]:
    if not source_units:
        return span.timing_source, span.timing_quality

    unit_source, unit_quality = _semantic_span_timing(source_units)
    timing_source: NarrationTimingSource = (
        unit_source if unit_source == span.timing_source else "estimated"
    )
    qualities = {unit_quality, span.timing_quality}
    if "estimated" in qualities:
        timing_quality: NarrationTimingQuality = "estimated"
    elif "speech_recognition" in qualities:
        timing_quality = "speech_recognition"
    else:
        timing_quality = "boundary"
    return timing_source, timing_quality


def _build_visual_beats_from_valid_spans(
    narration_script: str,
    semantic_visual_spans: list[SemanticVisualSpan],
    timed_units: list[TimedNarrationUnit],
    audio_duration: float,
    source_semantic_spans_available: bool,
) -> list[VisualBeat]:
    visual_beats: list[VisualBeat] = []
    for group_position, span in enumerate(semantic_visual_spans):
        group_start = 0.0 if group_position == 0 else span.start_time
        group_end = (
            semantic_visual_spans[group_position + 1].start_time
            if group_position + 1 < len(semantic_visual_spans)
            else audio_duration
        )
        if (
            group_end - group_start
            <= _VISUAL_BEAT_TIME_TOLERANCE_SECONDS
        ):
            raise ValueError("semantic visual group has no usable timeline duration")

        candidates = _visual_beat_unit_cut_candidates(
            span,
            timed_units,
            group_start,
            group_end,
        )
        cuts = _choose_balanced_visual_beat_cuts(
            candidates,
            group_start,
            group_end,
            _desired_visual_beat_shot_count(group_end - group_start),
        )
        unit_boundaries = (
            [span.start_unit]
            + [unit_index for unit_index, _ in cuts]
            + [span.end_unit_exclusive]
            if span.start_unit is not None
            and span.end_unit_exclusive is not None
            else [None, None]
        )
        time_boundaries = [group_start] + [cut_time for _, cut_time in cuts] + [group_end]
        split_succeeded = bool(cuts)

        if split_succeeded:
            try:
                spoken_parts = [
                    reconstruct_semantic_spoken_text(
                        narration_script,
                        timed_units,
                        start_unit,
                        end_unit_exclusive,
                    )
                    for start_unit, end_unit_exclusive in zip(
                        unit_boundaries,
                        unit_boundaries[1:],
                    )
                    if start_unit is not None and end_unit_exclusive is not None
                ]
                if len(spoken_parts) != len(time_boundaries) - 1:
                    raise ValueError("visual beat text partition is incomplete")
            except ValueError:
                # Never invent text or a sub-unit boundary merely to satisfy an
                # editing preference. A coarse/unalignable long concept remains
                # one semantically correct shot.
                cuts = []
                unit_boundaries = [span.start_unit, span.end_unit_exclusive]
                time_boundaries = [group_start, group_end]
                spoken_parts = [span.spoken_text]
                split_succeeded = False
        else:
            spoken_parts = [span.spoken_text]

        source_span_duration = span.end_time - span.start_time
        if split_succeeded:
            duration_policy = "long_span_split"
        elif (
            source_span_duration
            < _VISUAL_BEAT_PREFERRED_MIN_SECONDS
            - _VISUAL_BEAT_TIME_TOLERANCE_SECONDS
        ):
            duration_policy = "short_semantic_preserved"
        else:
            duration_policy = "semantic_original"

        for shot_index, (start_time, end_time) in enumerate(
            zip(time_boundaries, time_boundaries[1:]),
            start=1,
        ):
            start_unit = unit_boundaries[shot_index - 1]
            end_unit_exclusive = unit_boundaries[shot_index]
            source_units = (
                timed_units[start_unit:end_unit_exclusive]
                if start_unit is not None and end_unit_exclusive is not None
                else []
            )
            timing_source, timing_quality = _visual_beat_timing(
                span,
                source_units,
            )
            source_narration_slot_indexes = (
                _semantic_span_slot_indexes(
                    timed_units,
                    [],
                    start_unit,
                    end_unit_exclusive,
                )
                if split_succeeded
                and start_unit is not None
                and end_unit_exclusive is not None
                else span.source_narration_slot_indexes
            )
            duration = end_time - start_time
            visual_beats.append(
                VisualBeat(
                    index=len(visual_beats) + 1,
                    semantic_group_id=group_position + 1,
                    shot_index=shot_index,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    spoken_text=spoken_parts[shot_index - 1],
                    visual_requirement=span.visual_requirement,
                    source_semantic_span_index=(
                        span.index if source_semantic_spans_available else None
                    ),
                    source_narration_slot_indexes=list(
                        source_narration_slot_indexes
                    ),
                    start_unit=start_unit,
                    end_unit_exclusive=end_unit_exclusive,
                    timing_source=timing_source,
                    timing_quality=timing_quality,
                    duration_policy=duration_policy,
                    rapid_cut=(
                        duration
                        < _VISUAL_BEAT_RAPID_CUT_SECONDS
                        - _VISUAL_BEAT_TIME_TOLERANCE_SECONDS
                    ),
                )
            )
    return visual_beats


def _rescue_unrefined_shot_requirements(
    groups: dict[int, list[VisualBeat]],
    parents: dict[int, str],
) -> int:
    """Describe a shot from its own words when the parent could not be divided.

    A shot that keeps its span's requirement is the pathology this whole stage
    exists to remove, not a neutral outcome: the parent names several moments at
    once, so the checklist makes every one of them mandatory and no single clip
    can satisfy all of them. In task 3f0f2b07 two of eleven shots stayed on the
    parent, one of them failed outright, and its rewrite came back as "A
    motionless stone" -- the parent's entire causal action dropped.

    This asks a different and much easier question than the split did: not which
    part of the parent belongs to this shot, but what a camera would see while
    this line is spoken. It does not raise the run's LLM bill, because the beats
    it repairs are exactly the ones that would otherwise reach the rewrite rung
    and pay for a call there anyway -- except this one happens before any stock
    search or clip analysis is paid for. Failure stays free: the shot keeps the
    parent and behaves as it did before this stage existed.
    """
    leftovers: list[VisualBeat] = []
    context_parts: list[str] = []
    for group_id, group in groups.items():
        unrefined = [
            beat for beat in group if beat.visual_requirement == parents[group_id]
        ]
        if not unrefined:
            continue
        leftovers.extend(unrefined)
        # Read-only context, so a dependent fragment such as "solid rock, and
        # nothing about the stone" can be resolved against the sentence it was
        # cut from. Only spans that still need help contribute, and the join is
        # capped because it is sent verbatim in the prompt.
        context_parts.extend(
            " ".join(str(beat.spoken_text or "").split()) for beat in group
        )
    if not leftovers:
        return 0

    context = " ".join(part for part in context_parts if part)[
        :_SHOT_RESCUE_NARRATION_CONTEXT_MAX_CHARS
    ]
    try:
        answers = llm.generate_narration_visual_requirements(
            context,
            [
                {"index": beat.index, "spoken_text": beat.spoken_text}
                for beat in leftovers
            ],
        )
    except Exception as exc:
        logger.warning(
            "shot requirement rescue failed: "
            f"error={type(exc).__name__}, shots={len(leftovers)}"
        )
        return 0
    if not answers:
        logger.warning(
            "shot requirement rescue returned nothing usable; those shots keep "
            f"their span requirement: shots={len(leftovers)}"
        )
        return 0

    rescued = 0
    for group_id, group in groups.items():
        taken = {beat.visual_requirement.casefold() for beat in group}
        for beat in group:
            if beat.visual_requirement != parents[group_id]:
                continue
            requirement = " ".join(str(answers.get(beat.index) or "").split())
            if not requirement:
                continue
            if len(requirement) > _SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS:
                continue
            key = requirement.casefold()
            if key in taken:
                # Either the parent again or a sibling's text. Adopting it would
                # undo the de-duplication this stage exists to create, and the
                # parent is at least still a distinct string.
                continue
            taken.add(key)
            beat.visual_requirement = requirement
            rescued += 1
    logger.info(
        f"shot requirement rescue completed: shots={len(leftovers)}, "
        f"rescued={rescued}"
    )
    return rescued


def refine_split_span_shot_requirements(visual_beats: list[VisualBeat]) -> int:
    """Give each shot of a split span a requirement of its own.

    A span long enough to be split becomes several shots, and every shot
    inherits the span's single ``visual_requirement`` verbatim. That requirement
    describes the whole span, so it usually names several moments at once -- a
    drip, a canyon, a thousand years. Three consequences follow, and all three
    are paid for on every run: the beats deduplicate into one stock search and
    buy near-identical clips, the requirement checklist turns every moment into a
    mandatory fact that no single clip can show, and so each of those shots
    exhausts its candidates and drops into the failure ladder.

    One batched provider-neutral call splits the parent across its shots, using
    each shot's own spoken text to decide which part is that shot's. Any shot
    the split could not narrow is then described from its own spoken line
    instead, because keeping the parent is the pathology above and not a safe
    default. Failure of both is free: the shot keeps the parent requirement and
    behaves exactly as it did before this stage existed.

    Returns the number of beats whose requirement was replaced by either route,
    and mutates those beats in place so both the search queries and the
    checklist -- built afterwards -- read the refined text without knowing about
    this stage.
    """
    groups: dict[int, list[VisualBeat]] = {}
    for beat in visual_beats:
        if beat.duration_policy != "long_span_split":
            continue
        groups.setdefault(beat.semantic_group_id, []).append(beat)

    requestable: dict[int, list[VisualBeat]] = {}
    for group_id, group in groups.items():
        # One shot already owns its span's requirement, so there is nothing to
        # divide. Shots that do not share one parent are left alone as well,
        # because then the parent sent with the request would not be the parent
        # of every shot in it.
        if len(group) < 2:
            continue
        if len({beat.visual_requirement for beat in group}) != 1:
            continue
        requestable[group_id] = sorted(group, key=lambda beat: beat.shot_index)
    if not requestable:
        return 0

    # Captured before anything is mutated, so a shot that still holds this text
    # afterwards is exactly a shot the split could not narrow.
    parents = {
        group_id: group[0].visual_requirement
        for group_id, group in requestable.items()
    }
    try:
        answers = llm.generate_shot_visual_requirements(
            [
                {
                    "span_requirement": group[0].visual_requirement,
                    "shots": [
                        {
                            "shot_id": beat.index,
                            "spoken_text": beat.spoken_text,
                        }
                        for beat in group
                    ],
                }
                for group in requestable.values()
            ]
        )
    except Exception as exc:
        logger.warning(
            "shot visual requirement split failed: "
            f"error={type(exc).__name__}, spans={len(requestable)}"
        )
        return 0
    if answers is None:
        # The documented "this stage is unavailable" signal: no batch produced a
        # parseable payload. Asking a second question of the same provider in
        # that state would only pay for another unparseable answer, so the shots
        # keep their span requirement exactly as they did before.
        logger.warning(
            "shot visual requirement split is unavailable; every shot "
            f"of a split span keeps its span requirement: spans={len(requestable)}"
        )
        return 0
    if not answers:
        # Answered, but narrowed nothing. That is a statement about the parent
        # being hard to divide, not about the provider, so the rescue below still
        # has a real chance and every shot is a leftover.
        logger.warning(
            "shot visual requirement split narrowed no shot: "
            f"spans={len(requestable)}"
        )

    refined = 0
    over_long = 0
    duplicated = 0
    for group in requestable.values():
        claimed: set[str] = set()
        for beat in group:
            requirement = " ".join(str(answers.get(beat.index) or "").split())
            if not requirement:
                continue
            # The ceiling the LLM grouping path validates spans against. A
            # bloated requirement is carried into the checklist and into every
            # adjudication prompt of this beat, so it is not adopted.
            if len(requirement) > _SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS:
                over_long += 1
                continue
            # Identical siblings are the exact pathology this stage removes, and
            # the query cache keys on the group plus the requirement, so a
            # repeated answer would collapse back into one search. The later
            # shot keeps the parent instead: still a distinct string, so both
            # shots are searched separately.
            key = requirement.casefold()
            if key in claimed:
                duplicated += 1
                continue
            claimed.add(key)
            if requirement == beat.visual_requirement:
                continue
            beat.visual_requirement = requirement
            refined += 1

    if over_long:
        logger.warning(
            "shot visual requirements exceeded the span limit and were "
            f"discarded: count={over_long}"
        )
    if duplicated:
        logger.warning(
            "shot visual requirements repeated a sibling answer and were "
            f"discarded: count={duplicated}"
        )

    rescued = _rescue_unrefined_shot_requirements(requestable, parents)
    logger.info(
        "split span shot requirements refined: "
        f"spans={len(requestable)}, "
        f"shots={sum(len(group) for group in requestable.values())}, "
        f"refined_shots={refined}, rescued_shots={rescued}"
    )
    return refined + rescued


def _validate_complete_visual_beat_timeline(
    visual_beats: list[VisualBeat],
    audio_duration: float,
) -> None:
    if not visual_beats:
        raise ValueError("visual beat timeline is empty")
    if visual_beats[0].start_time != 0.0:
        raise ValueError("visual beat timeline must start at zero")
    if visual_beats[-1].end_time != audio_duration:
        raise ValueError("visual beat timeline must end at the audio duration")

    for position, beat in enumerate(visual_beats, start=1):
        if beat.index != position:
            raise ValueError("visual beat indexes must be sequential")
        if not math.isfinite(beat.start_time) or not math.isfinite(beat.end_time):
            raise ValueError("visual beat timing must be finite")
        if beat.duration <= _VISUAL_BEAT_TIME_TOLERANCE_SECONDS:
            raise ValueError("visual beat duration must be positive")
        if not math.isclose(
            beat.duration,
            beat.end_time - beat.start_time,
            rel_tol=0.0,
            abs_tol=_VISUAL_BEAT_TIME_TOLERANCE_SECONDS,
        ):
            raise ValueError("visual beat duration is inconsistent")
        if position < len(visual_beats) and not math.isclose(
            beat.end_time,
            visual_beats[position].start_time,
            rel_tol=0.0,
            abs_tol=_VISUAL_BEAT_TIME_TOLERANCE_SECONDS,
        ):
            raise ValueError("visual beat timeline contains a gap or overlap")

    if not math.isclose(
        math.fsum(beat.duration for beat in visual_beats),
        audio_duration,
        rel_tol=0.0,
        abs_tol=_VISUAL_BEAT_TIME_TOLERANCE_SECONDS,
    ):
        raise ValueError("visual beat durations do not cover the audio duration")


def build_visual_beats(
    narration_script: str,
    semantic_visual_spans: list[SemanticVisualSpan] | None,
    timed_units: list[TimedNarrationUnit],
    narration_slots: list[NarrationSlot],
    audio_duration: float,
) -> list[VisualBeat]:
    """Build a gapless variable shot timeline without changing semantics."""
    if not math.isfinite(audio_duration) or audio_duration <= 0:
        raise ValueError("visual beat timeline requires a positive audio duration")

    if semantic_visual_spans:
        try:
            _validate_semantic_spans_for_visual_beats(
                semantic_visual_spans,
                timed_units,
                audio_duration,
            )
            visual_beats = _build_visual_beats_from_valid_spans(
                narration_script,
                semantic_visual_spans,
                timed_units,
                audio_duration,
                source_semantic_spans_available=True,
            )
            _validate_complete_visual_beat_timeline(visual_beats, audio_duration)
            return visual_beats
        except ValueError as exc:
            logger.warning(
                "semantic visual beats unavailable; using narration-slot fallback: "
                f"error={type(exc).__name__}"
            )

    if not narration_slots:
        return []

    fallback_attempts = (timed_units, []) if timed_units else ([],)
    for fallback_units in fallback_attempts:
        try:
            fallback_spans = build_narration_slot_semantic_fallback(
                narration_script,
                fallback_units,
                narration_slots,
            )
            _validate_semantic_spans_for_visual_beats(
                fallback_spans,
                fallback_units,
                audio_duration,
            )
            visual_beats = _build_visual_beats_from_valid_spans(
                narration_script,
                fallback_spans,
                fallback_units,
                audio_duration,
                source_semantic_spans_available=False,
            )
            _validate_complete_visual_beat_timeline(visual_beats, audio_duration)
            return visual_beats
        except ValueError as exc:
            logger.warning(
                "narration-slot visual beats unavailable: "
                f"error={type(exc).__name__}"
            )
    return []


def build_visual_slots(
    narration_slots: list[NarrationSlot],
    audio_duration: float,
    video_clip_duration: float,
) -> list[VisualSlot]:
    """Project narration onto the renderer's existing fixed-duration timeline."""
    if not narration_slots:
        raise ValueError("visual timeline requires narration slots")
    if not math.isfinite(audio_duration) or audio_duration <= 0:
        raise ValueError("visual timeline requires a positive audio duration")
    if not math.isfinite(video_clip_duration) or video_clip_duration <= 0:
        raise ValueError("visual timeline requires a positive clip duration")

    visual_slots: list[VisualSlot] = []
    slot_count = math.ceil(audio_duration / video_clip_duration)
    for zero_based_index in range(slot_count):
        start_time = zero_based_index * video_clip_duration
        end_time = min(start_time + video_clip_duration, audio_duration)
        overlapping = [
            narration
            for narration in narration_slots
            if narration.start_time < end_time and narration.end_time > start_time
        ]
        if not overlapping:
            raise ValueError(
                f"visual slot {zero_based_index + 1} has no overlapping narration"
            )

        narration_overlaps = [
            NarrationOverlap(
                narration_slot_index=narration.index,
                overlap_start_time=max(start_time, narration.start_time),
                overlap_end_time=min(end_time, narration.end_time),
                overlap_duration=(
                    min(end_time, narration.end_time)
                    - max(start_time, narration.start_time)
                ),
            )
            for narration in overlapping
        ]
        maximum_overlap = max(
            overlap.overlap_duration for overlap in narration_overlaps
        )
        primary_candidates = [
            narration
            for narration, overlap in zip(overlapping, narration_overlaps)
            if math.isclose(
                overlap.overlap_duration,
                maximum_overlap,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]
        visual_midpoint = (start_time + end_time) / 2
        midpoint_candidates = [
            narration
            for narration in primary_candidates
            if narration.start_time <= visual_midpoint < narration.end_time
        ]
        if midpoint_candidates:
            primary_candidates = midpoint_candidates
        primary_narration = min(
            primary_candidates,
            key=lambda narration: (
                abs(
                    ((narration.start_time + narration.end_time) / 2)
                    - visual_midpoint
                ),
                narration.index,
            ),
        )

        timing_sources = {slot.timing_source for slot in overlapping}
        timing_source = (
            next(iter(timing_sources)) if len(timing_sources) == 1 else "estimated"
        )
        visual_slots.append(
            VisualSlot(
                index=zero_based_index + 1,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                narration_slot_indexes=[slot.index for slot in overlapping],
                narration_text=" ".join(slot.text for slot in overlapping),
                primary_narration_slot_index=primary_narration.index,
                primary_narration_text=primary_narration.text,
                visual_requirement=primary_narration.text,
                narration_overlaps=narration_overlaps,
                search_queries=[],
                timing_source=timing_source,
                timing_quality=_timing_quality(timing_source),
            )
        )
    return visual_slots


def generate_visual_slot_search_queries(
    params: VideoParams,
    visual_slots: list[VisualSlot],
) -> list[str]:
    """Attach one indexed query to every visual slot and return legacy flat terms."""
    slot_payload = [
        {
            "slot_index": slot.index,
            "start_time": slot.start_time,
            "end_time": slot.end_time,
            "visual_requirement": slot.visual_requirement,
        }
        for slot in visual_slots
    ]
    queries_by_slot = llm.generate_visual_slot_queries(
        video_subject=params.video_subject,
        visual_slots=slot_payload,
        queries_per_slot=1,
    )

    for slot in visual_slots:
        queries = queries_by_slot.get(slot.index, [])
        if not queries:
            raise ValueError(f"visual slot {slot.index} has no search query")
        slot.search_queries = list(queries)
    return [slot.search_queries[0] for slot in visual_slots]


def generate_visual_beat_search_queries(
    visual_beats: list[VisualBeat],
    queries_per_beat: int = 1,
) -> list[str]:
    """Generate once per semantic group/requirement and copy to sibling shots."""
    if not visual_beats:
        return []

    representatives: dict[tuple[int, str], VisualBeat] = {}
    beat_group_keys: dict[int, tuple[int, str]] = {}
    for beat in visual_beats:
        requirement = " ".join(beat.visual_requirement.split()).strip()
        if not requirement:
            raise ValueError(f"visual beat {beat.index} has no visual requirement")
        group_key = (beat.semantic_group_id, requirement.casefold())
        representatives.setdefault(group_key, beat)
        beat_group_keys[beat.index] = group_key

    representative_payload = [
        {
            "slot_index": beat.index,
            "start_time": beat.start_time,
            "end_time": beat.end_time,
            "visual_requirement": beat.visual_requirement,
        }
        for beat in representatives.values()
    ]
    # VisualBeat queries must not inherit whole-video or neighboring narration
    # context. The existing provider-neutral query abstraction therefore gets
    # only representative visual requirements and an intentionally empty subject.
    queries_by_representative = llm.generate_visual_slot_queries(
        video_subject="",
        visual_slots=representative_payload,
        queries_per_slot=queries_per_beat,
    )

    queries_by_group: dict[tuple[int, str], list[str]] = {}
    for group_key, representative in representatives.items():
        queries = queries_by_representative.get(representative.index, [])
        if not queries:
            raise ValueError(
                f"visual beat group {representative.semantic_group_id} "
                "has no search query"
            )
        queries_by_group[group_key] = list(queries)

    for beat in visual_beats:
        beat.search_queries = list(queries_by_group[beat_group_keys[beat.index]])
    return [beat.search_queries[0] for beat in visual_beats]


def generate_visual_requirement_checklist(
    visual_beats: list[VisualBeat],
) -> dict[str, VisualRequirementSpec]:
    """Decompose every beat requirement while still in the script stage.

    The checklist is the contract candidate verification gates on, so it belongs
    to the script timeline rather than to the material stage: it is written to
    the task manifest before a single stock request is made, so the plan can be
    inspected before anything is downloaded or analyzed, and material selection
    then verifies against exactly the checklist this run was planned with.

    Keys are normalized requirements. A requirement the provider could not
    decompose is simply absent; no synthetic checklist is invented here, and the
    material stage keeps deciding what a missing checklist means for that beat.
    """
    if not visual_beats:
        return {}

    requirements = [beat.visual_requirement for beat in visual_beats]
    specs = llm.generate_visual_requirement_specs(requirements)
    missing_beats = [
        beat.index
        for beat in visual_beats
        if llm.normalize_visual_requirement(beat.visual_requirement) not in specs
    ]
    if missing_beats:
        logger.warning(
            "visual requirement checklist is incomplete: "
            f"beats={len(visual_beats)}, decomposed={len(specs)}, "
            f"missing_beat_indexes={missing_beats}"
        )
    else:
        logger.info(
            "visual requirement checklist ready: "
            f"beats={len(visual_beats)}, unique_requirements={len(specs)}"
        )
    return specs


def _visual_beat_records(
    visual_beats: list[VisualBeat] | None,
) -> list[dict[str, object]]:
    """The persisted shape of one beat timeline.

    Written twice: once by the script stage for the timeline it planned, and again
    after material selection if a merge rewrote it. One definition keeps those two
    writes from drifting into two different schemas for the same field.
    """
    return [
        {
            "index": beat.index,
            "semantic_group_id": beat.semantic_group_id,
            "shot_index": beat.shot_index,
            "start_time": beat.start_time,
            "end_time": beat.end_time,
            "duration": beat.duration,
            "spoken_text": beat.spoken_text,
            "visual_requirement": beat.visual_requirement,
            "source_semantic_span_index": beat.source_semantic_span_index,
            "source_narration_slot_indexes": (
                beat.source_narration_slot_indexes
            ),
            "start_unit": beat.start_unit,
            "end_unit_exclusive": beat.end_unit_exclusive,
            "timing_source": beat.timing_source,
            "timing_quality": beat.timing_quality,
            "duration_policy": beat.duration_policy,
            "rapid_cut": beat.rapid_cut,
            "search_queries": beat.search_queries,
        }
        for beat in (visual_beats or [])
    ]


def persist_narration_timeline(
    task_id: str,
    narration_slots: list[NarrationSlot],
    visual_slots: list[VisualSlot],
    video_terms: list[str],
    timed_narration_units: list[TimedNarrationUnit] | None = None,
    semantic_visual_spans: list[SemanticVisualSpan] | None = None,
    visual_beats: list[VisualBeat] | None = None,
    visual_requirement_specs: dict[str, VisualRequirementSpec] | None = None,
) -> None:
    narration_records = [
        {
            "index": slot.index,
            "start_time": slot.start_time,
            "end_time": slot.end_time,
            "duration": slot.duration,
            "text": slot.text,
            "timing_source": slot.timing_source,
        }
        for slot in narration_slots
    ]
    visual_records = [
        {
            "index": slot.index,
            "start_time": slot.start_time,
            "end_time": slot.end_time,
            "duration": slot.duration,
            "narration_slot_indexes": slot.narration_slot_indexes,
            "narration_text": slot.narration_text,
            "primary_narration_slot_index": slot.primary_narration_slot_index,
            "primary_narration_text": slot.primary_narration_text,
            "visual_requirement": slot.visual_requirement,
            "narration_overlaps": [
                {
                    "narration_slot_index": overlap.narration_slot_index,
                    "overlap_start_time": overlap.overlap_start_time,
                    "overlap_end_time": overlap.overlap_end_time,
                    "overlap_duration": overlap.overlap_duration,
                }
                for overlap in slot.narration_overlaps
            ],
            "search_queries": slot.search_queries,
            "timing_source": slot.timing_source,
            "timing_quality": slot.timing_quality,
        }
        for slot in visual_slots
    ]
    timed_unit_records = [
        {
            "index": unit.index,
            "text": unit.text,
            "start_time": unit.start_time,
            "end_time": unit.end_time,
            "duration": unit.duration,
            "source_narration_slot_index": unit.source_narration_slot_index,
            "timing_source": unit.timing_source,
            "timing_quality": unit.timing_quality,
            "source_boundary_type": unit.source_boundary_type,
            "script_start_char": unit.script_start_char,
            "script_end_char": unit.script_end_char,
        }
        for unit in (timed_narration_units or [])
    ]
    semantic_span_records = [
        {
            "index": span.index,
            "start_unit": span.start_unit,
            "end_unit_exclusive": span.end_unit_exclusive,
            "spoken_text": span.spoken_text,
            "visual_requirement": span.visual_requirement,
            "source_narration_slot_indexes": span.source_narration_slot_indexes,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "timing_source": span.timing_source,
            "timing_quality": span.timing_quality,
            "grouping_source": span.grouping_source,
        }
        for span in (semantic_visual_spans or [])
    ]
    visual_beat_records = _visual_beat_records(visual_beats)
    # The checklist is written per unique normalized requirement because sibling
    # shots of one semantic group share it. ``missing`` is listed explicitly: it
    # is the difference between "this beat will be verified" and "this beat has
    # no gate", and it must be visible before any download starts.
    requirement_spec_records = [
        {
            "normalized_requirement": normalized,
            "spec": llm.visual_requirement_spec_to_dict(spec),
        }
        for normalized, spec in sorted((visual_requirement_specs or {}).items())
    ]
    missing_requirement_beats = [
        beat.index
        for beat in (visual_beats or [])
        if visual_requirement_specs is not None
        and llm.normalize_visual_requirement(beat.visual_requirement)
        not in visual_requirement_specs
    ]
    task_artifacts.patch_script_data(
        task_id,
        timeline_schema_version=2,
        search_terms=video_terms,
        narration_slots=narration_records,
        visual_slots=visual_records,
        timed_narration_units=timed_unit_records,
        semantic_visual_spans=semantic_span_records,
        visual_beats=visual_beat_records,
        visual_requirement_specs=requirement_spec_records,
        visual_requirement_specs_missing_beat_indexes=missing_requirement_beats,
    )


def save_script_data(task_id, video_script, video_terms, params):
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }
    task_artifacts.write_script_data(task_id, script_data)


def resolve_custom_audio_file(task_id: str, custom_audio_file: str | None) -> str:
    requested_file = (custom_audio_file or "").strip()
    if not requested_file:
        return ""

    task_dir = utils.task_dir(task_id)
    try:
        return file_security.resolve_path_within_directory(
            task_dir,
            requested_file,
        )
    except ValueError as exc:
        task_dir_error = exc

    server_audio_file = path.realpath(
        requested_file
        if path.isabs(requested_file)
        else path.join(utils.root_dir(), requested_file)
    )
    if not path.isabs(requested_file):
        project_root = path.realpath(utils.root_dir())
        try:
            if path.commonpath([project_root, server_audio_file]) != project_root:
                raise ValueError(
                    "relative custom audio paths must stay within the project directory"
                )
        except ValueError as exc:
            raise ValueError(
                "custom audio file must be task-local or an existing server-side file"
            ) from exc

    if not path.isfile(server_audio_file):
        raise ValueError(
            "custom audio file does not exist or is not a file"
        ) from task_dir_error

    return server_audio_file


def _resolve_reusable_voice_preview(
    task_id: str,
    params,
    video_script: str,
    voice_preview: dict | None,
) -> tuple[str, float, object] | None:
    """
    校验并解析 WebUI 提交的完整试听缓存。

    该载荷不是公开 API 参数，只能来自当前进程的 WebUI。即便如此，后台任务
    仍重新核对文案和全部配音参数，并限制音频位于当前任务目录；任何不一致都
    回退普通 TTS，不让过期试听污染正式成片。
    """
    if not voice_preview:
        return None

    expected_values = {
        "script": str(video_script or "").strip(),
        "voice_name": params.voice_name,
        "voice_rate": float(params.voice_rate),
        "voice_volume": float(params.voice_volume),
    }
    if not math.isclose(float(params.voice_volume), 1.0) or any(
        voice_preview.get(key) != value for key, value in expected_values.items()
    ):
        logger.info(
            f"skip stale voice preview cache, task_id: {task_id}, "
            "reason: voice parameters changed"
        )
        return None

    preview_file = path.realpath(str(voice_preview.get("audio_file") or ""))
    task_root = path.realpath(utils.task_dir(task_id))
    try:
        preview_is_task_local = path.commonpath([task_root, preview_file]) == task_root
    except ValueError:
        preview_is_task_local = False

    duration = voice_preview.get("duration")
    sub_maker = voice_preview.get("sub_maker")
    if (
        not preview_is_task_local
        or not path.isfile(preview_file)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
        or sub_maker is None
    ):
        logger.warning(
            f"skip invalid voice preview cache, task_id: {task_id}, "
            f"audio_file: {preview_file or '<empty>'}"
        )
        return None

    logger.info(
        f"using full voice preview audio, task_id: {task_id}, duration: {duration:.2f}s"
    )
    return preview_file, math.ceil(duration), sub_maker


def generate_audio(task_id, params, video_script, voice_preview=None):
    """
    Generate audio for the video script.
    If a custom audio file is provided, it will be used directly.
    There will be no subtitle maker object returned in this case.
    Otherwise, TTS will be used to generate the audio.
    Returns:
        - audio_file: path to the generated or provided audio file
        - audio_duration: duration of the audio in seconds
        - sub_maker: subtitle maker object if TTS is used, None otherwise
    """
    logger.info("\n\n## generating audio")
    # /audio 和 /subtitle 请求模型不包含 custom_audio_file，
    # 这里统一做兼容读取，避免直调接口时抛属性错误。
    requested_custom_audio_file = getattr(params, "custom_audio_file", None)
    try:
        custom_audio_file = resolve_custom_audio_file(
            task_id, requested_custom_audio_file
        )
    except ValueError as exc:
        _mark_task_failed(
            task_id,
            "audio",
            f"invalid custom audio file: {exc}",
        )
        return None, None, None

    if not custom_audio_file:
        reusable_preview = _resolve_reusable_voice_preview(
            task_id,
            params,
            video_script,
            voice_preview,
        )
        if reusable_preview:
            return reusable_preview

        logger.info("no custom audio file provided, using TTS to generate audio.")
        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        try:
            sub_maker = voice.tts(
                text=video_script,
                voice_name=voice.parse_voice_name(params.voice_name),
                voice_rate=params.voice_rate,
                voice_file=audio_file,
            )
        except voice.TTSServiceError as exc:
            _mark_task_failed(task_id, "audio", str(exc))
            return None, None, None
        if sub_maker is None:
            _mark_task_failed(
                task_id,
                "audio",
                "failed to synthesize audio; verify the selected voice and TTS connectivity",
            )
            return None, None, None
        audio_duration = math.ceil(voice.get_audio_duration(sub_maker))
        if audio_duration == 0:
            _mark_task_failed(task_id, "audio", "generated audio duration is zero")
            return None, None, None
        return audio_file, audio_duration, sub_maker
    else:
        logger.info(f"using custom audio file: {custom_audio_file}")
        audio_duration = voice.get_audio_duration(custom_audio_file)
        if audio_duration == 0:
            _mark_task_failed(
                task_id,
                "audio",
                "custom audio duration is zero",
            )
            return None, None, None
        return custom_audio_file, audio_duration, None


def generate_subtitle(
    task_id,
    params,
    video_script,
    sub_maker,
    audio_file,
    force_timeline: bool = False,
):
    """
    Generate subtitle for the video script.
    If subtitle display is disabled, ordered matching can still force creation of
    an internal SRT timeline. Without a subtitle maker, only Whisper can create it.
    Returns:
        - subtitle_path: path to the generated subtitle file
    """
    logger.info("\n\n## generating subtitle")
    if not params.subtitle_enabled and not force_timeline:
        return ""

    if force_timeline and not params.subtitle_enabled:
        logger.info(
            "subtitle display is disabled; generating internal narration timeline "
            "for ordered material matching"
        )

    subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    if not subtitle_provider:
        logger.info("subtitle provider is empty, skip subtitle generation")
        return ""

    if sub_maker is None and subtitle_provider != "whisper":
        # 自定义音频不会经过 TTS，因此没有 Edge/Azure 等 TTS 返回的
        # sub_maker 时间轴。只有 Whisper 可以直接从音频文件转写字幕；
        # 其他字幕提供方继续保持原有行为，避免生成错误的空时间轴。
        logger.warning(
            "subtitle maker is missing, skip subtitle generation for provider: "
            f"{subtitle_provider}"
        )
        return ""

    if subtitle_provider == "edge":
        voice.create_subtitle(
            text=video_script, sub_maker=sub_maker, subtitle_file=subtitle_path
        )
        if not os.path.exists(subtitle_path):
            # Edge 字幕偶尔会因为时间轴与文案无法匹配而没有产出文件。这里不能
            # 自动切换到 Whisper，否则首次失败会在用户不知情的情况下下载数 GB
            # 的模型。只有显式配置 Whisper 时才允许加载模型，Edge 失败则保留
            # 无字幕视频并记录原因，避免意外的网络和磁盘开销。
            logger.warning(
                "edge subtitle generation did not produce a subtitle file; "
                "skip subtitles without falling back to whisper"
            )
            return ""

    if subtitle_provider == "whisper":
        subtitle.create(audio_file=audio_file, subtitle_file=subtitle_path)
        logger.info("\n\n## correcting subtitle")
        subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    return subtitle_path


def get_video_materials(
    task_id,
    params,
    video_terms,
    audio_duration,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    visual_slots: list[VisualSlot] | None = None,
    visual_beats: list[VisualBeat] | None = None,
    visual_requirement_specs: dict[str, VisualRequirementSpec] | None = None,
    merged_beats_out: list[VisualBeat] | None = None,
):
    if params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            _mark_task_failed(
                task_id,
                "materials",
                "no valid local video materials were found",
            )
            return None
        return [material_info.url for material_info in materials]
    elif params.video_source == "loomloom":
        if not isinstance(
            loomloom_video_request, loomloom.LoomLoomConfirmedVideoRequest
        ):
            _mark_task_failed(
                task_id,
                "materials",
                "LoomLoom video generation requires a confirmed quote",
            )
            return None

        request = loomloom_video_request
        logger.info(
            "\n\n## generating "
            f"{len(request.batch.input_rows)} video materials with LoomLoom"
        )
        run_id = ""
        try:
            request.validate()
            backend = loomloom.LoomLoomVideoBackend(request.settings)
            execution = backend.execute(
                request.batch,
                client_request_id=request.client_request_id,
                listing_version_id=request.listing_version_id,
                confirm=True,
            )
            run_id = execution.run_id
            # execute 返回即表示付费任务已经由远端接受。必须先把 run ID 写入
            # 进程日志，即使 Redis 等状态后端随后不可用，运维人员仍能凭日志
            # 在胜算云侧定位任务，不能让唯一标识只存在于局部变量中。
            logger.info(
                "LoomLoom paid video run created: "
                f"task_id={task_id}, run_id={run_id}, "
                f"listing_version_id={request.listing_version_id}"
            )
            # 付费任务一旦创建就立即记录远端 ID。即使后续轮询超时，日志和任务
            # 状态仍能帮助用户或平台支持人员定位并找回已经生成的产物。状态后端
            # 故障只能降低可观测性，不能中断已经开始计费的远端任务和产物下载。
            _record_loomloom_run_reference(
                task_id=task_id,
                run_id=run_id,
                listing_version_id=request.listing_version_id,
            )
            backend.wait_for_run(run_id)
            return list(
                backend.download_video_results(
                    run_id,
                    utils.task_dir(task_id),
                )
            )
        except (loomloom.LoomLoomError, ValueError) as exc:
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details={
                    "loomloom_run_id": run_id,
                    "loomloom_listing_version_id": request.listing_version_id,
                },
            )
            return None
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        # A beat timeline carries its own per-beat queries, and smart selection
        # requires exactly one query per visual item. Using the slot queries here
        # would pair every beat with the wrong search term.
        beat_terms = (
            [beat.search_queries[0] for beat in visual_beats] if visual_beats else []
        )
        # 顺序匹配模式只在用户显式开启时生效。这里强制素材下载按关键词顺序
        # 轮询，避免某个早期关键词下载太多素材，把后续脚本主题挤出最终时间线。
        downloaded_videos = material.download_videos(
            task_id=task_id,
            search_terms=beat_terms or video_terms,
            source=params.video_source,
            video_aspect=params.video_aspect,
            video_concat_mode=(
                VideoConcatMode.sequential
                if params.match_materials_to_script
                else params.video_concat_mode
            ),
            # Ordered outputs share one narration timeline. Multiplying by
            # video_count downloads extra late-script clips that sequential
            # composition never reaches and can distort the term allocation.
            audio_duration=(
                audio_duration
                if params.match_materials_to_script
                else audio_duration * params.video_count
            ),
            max_clip_duration=params.video_clip_duration,
            match_script_order=params.match_materials_to_script,
            visual_slots=None if visual_beats else visual_slots,
            visual_beats=visual_beats or None,
            clip_speed=params.video_clip_speed,
            # Computed in the script stage so verification uses the checklist
            # this run was planned and persisted with.
            requirement_specs=visual_requirement_specs or None,
            # Filled only if an unfillable beat had to be absorbed by a
            # neighbouring shot, in which case the renderer must use the rewritten
            # timeline.
            merged_beats_out=merged_beats_out,
        )
        if not downloaded_videos:
            _mark_task_failed(
                task_id,
                "materials",
                f"failed to download video materials from {params.video_source}",
            )
            return None
        return downloaded_videos


def _record_loomloom_run_reference(
    *, task_id: str, run_id: str, listing_version_id: str
) -> bool | None:
    """
    尽最大努力保存已创建的付费 LoomLoom Run，不让状态故障中断远端任务。

    返回 True 表示保存成功，False 表示任务记录已经不存在，None 表示状态后端
    在有限重试后仍不可用。调用方无论得到哪种结果都应继续轮询和下载，因为
    execute 已经产生外部付费副作用，停止本地流程只会让产物更难找回。
    """
    fields = {
        "loomloom_run_id": run_id,
        "loomloom_listing_version_id": listing_version_id,
    }
    for attempt in range(1, _LOOMLOOM_STATE_WRITE_ATTEMPTS + 1):
        try:
            updated = sm.state.patch_task(task_id, **fields)
        except Exception as exc:
            if attempt >= _LOOMLOOM_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    "failed to persist LoomLoom paid run after retries: "
                    f"task_id={task_id}, run_id={run_id}, attempts={attempt}, "
                    f"error={exc}"
                )
                return None
            logger.warning(
                "retry LoomLoom paid run state update: "
                f"task_id={task_id}, run_id={run_id}, attempt={attempt}, "
                f"error={exc}"
            )
            time.sleep(_LOOMLOOM_STATE_RETRY_DELAY_SECONDS)
            continue

        if updated is False:
            logger.warning(
                "could not persist LoomLoom paid run because task is missing: "
                f"task_id={task_id}, run_id={run_id}"
            )
        return updated

    return None


def generate_final_videos(
    task_id,
    params,
    downloaded_videos,
    audio_file,
    subtitle_path,
    audio_duration,
    source_ranges: list[tuple[float, float]] | None = None,
    render_segments: list[RenderSegment] | None = None,
):
    final_video_paths = []
    combined_video_paths = []
    warnings = []
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_requested = (
        video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    # 多视频生成默认会打散素材以增加差异；但“按文案顺序匹配素材”追求的是
    # 时间线稳定性和可解释性，所以开启后所有输出都使用顺序拼接。
    if params.match_materials_to_script:
        video_concat_mode = VideoConcatMode.sequential
    elif params.video_count == 1:
        video_concat_mode = params.video_concat_mode
    else:
        video_concat_mode = VideoConcatMode.random
    video_transition_mode = params.video_transition_mode

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
            clip_speed=params.video_clip_speed,
            source_ranges=source_ranges,
            render_segments=render_segments,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        # 视频配乐模式先明确禁用默认 BGM 解析，避免旧任务残留的 bgm_file 被
        # 误用。只有音量大于 0 才生成代理并调用付费 API；0 音量统一跳过。
        bgm_file_override = "" if video_music_provider else None
        if video_music_requested:
            service = video_music_provider["service"]
            display_name = video_music_provider["display_name"]
            warning_code = video_music_provider["warning_code"]
            generated_bgm_path = path.join(
                utils.task_dir(task_id),
                (f"{params.bgm_type}-bgm-{index}{video_music_provider['suffix']}"),
            )
            try:
                service.generate_bgm(
                    video_path=combined_video_path,
                    output_path=generated_bgm_path,
                    video_duration=audio_duration,
                    prompt=_get_video_music_prompt(params),
                )
                bgm_file_override = generated_bgm_path
            except video_music_provider["error_type"] as exc:
                # 视频、旁白和字幕都已生成时，第三方配乐临时失败不应浪费整条
                # 任务。当前视频明确禁用 BGM，并把降级结果返回 WebUI 提醒用户。
                logger.warning(
                    f"{display_name} BGM generation failed: task_id={task_id}, "
                    f"video_index={index}, error={exc}"
                )
                bgm_file_override = ""
                warnings.append({"code": warning_code, "video_index": index})

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        bgm_mix_succeeded = video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
            bgm_file_override=bgm_file_override,
        )
        if (
            video_music_provider is not None
            and bgm_file_override
            and not bgm_mix_succeeded
        ):
            # 第三方已成功返回并通过 FFmpeg 校验，但 MoviePy 最终混音仍可能
            # 因运行环境失败。视频服务会保留无 BGM 成片；API 生成失败时
            # override 为空，因此不会重复追加警告。
            warnings.append(
                {
                    "code": video_music_provider["warning_code"],
                    "video_index": index,
                }
            )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)

    return final_video_paths, combined_video_paths, warnings


def _patch_cross_post_state(task_id: str, **kwargs) -> bool | None:
    """安全更新发布字段；短暂状态后端故障时有限重试。"""
    for attempt in range(1, _CROSS_POST_STATE_WRITE_ATTEMPTS + 1):
        try:
            return sm.state.patch_task(task_id, **kwargs)
        except Exception as exc:
            # Redis 短暂断连不应让任务永久停留在 pending/processing。发布状态
            # 写入频率很低，这里使用固定次数和短等待即可覆盖瞬时故障，同时
            # 避免后台线程无限阻塞。最后一次失败保留完整堆栈便于定位。
            if attempt >= _CROSS_POST_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    f"failed to update cross-post state after retries, "
                    f"task_id: {task_id}, fields: {', '.join(kwargs)}, "
                    f"attempts: {attempt}, error: {exc}"
                )
                return None

            logger.warning(
                f"retry cross-post state update, task_id: {task_id}, "
                f"fields: {', '.join(kwargs)}, attempt: {attempt}, error: {exc}"
            )
            time.sleep(_CROSS_POST_STATE_RETRY_DELAY_SECONDS)

    return None


def _record_cross_post_failure(
    task_id: str,
    error: Exception,
    results: list[dict] | None = None,
) -> None:
    """尽最大努力保存发布失败；状态后端不可用时由日志保留诊断信息。"""
    updated = _patch_cross_post_state(
        task_id,
        cross_post_state=const.CROSS_POST_STATE_FAILED,
        cross_post_results=results or None,
        cross_post_error=str(error),
        cross_post_owner=None,
    )
    if updated is False:
        logger.warning(f"discard cross-post failure for missing task: {task_id}")


def _ensure_cross_post_terminal_state(task_id: str) -> None:
    """Future 结束后把仍处于活动态的任务收敛为失败。"""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        # 此处已经是 Future 的最终回调，没有后续同步调用方可以处理异常。
        # 状态后端恢复后，下一次进程启动仍会通过恢复逻辑处理遗留状态。
        logger.exception(
            f"failed to verify final cross-post state, task_id: {task_id}, error: {exc}"
        )
        return

    if not task or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES:
        return

    logger.warning(
        f"cross-post worker ended without terminal state, task_id: {task_id}, "
        f"state: {task.get('cross_post_state')}"
    )
    _record_cross_post_failure(
        task_id,
        RuntimeError("cross-post worker ended without persisting a terminal state"),
        task.get("cross_post_results"),
    )


def recover_interrupted_cross_posts(page_size: int = 100) -> int | None:
    """
    将进程重启后无法恢复的发布任务标记为失败。

    跨平台发布使用当前进程内的线程池，不是持久化任务队列。进程启动时，
    Redis 中残留的 pending/processing 不会自动继续执行；如果继续把它们视为
    运行中，用户将永久无法删除任务。这里分页扫描状态，只处理当前进程没有
    对应 Future 的活动记录，并保留已经生成的视频结果。
    """
    recovered = 0
    page = 1

    while True:
        try:
            tasks, total = sm.state.get_all_tasks(page, page_size)
        except Exception as exc:
            logger.exception(f"failed to recover interrupted cross-post tasks: {exc}")
            return None

        for task in tasks:
            task_id = str(task.get("task_id") or "")
            if (
                not task_id
                or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES
                or _is_cross_post_active_in_process(task_id)
                or _is_cross_post_owner_alive(task.get("cross_post_owner"))
            ):
                continue

            updated = _patch_cross_post_state(
                task_id,
                cross_post_state=const.CROSS_POST_STATE_FAILED,
                cross_post_error=_INTERRUPTED_CROSS_POST_ERROR,
                cross_post_owner=None,
            )
            if updated is True:
                recovered += 1

        if page * page_size >= total or not tasks:
            break
        page += 1

    if recovered:
        logger.warning(f"recovered interrupted cross-post tasks: {recovered}")
    return recovered


def _run_cross_post(
    task_id: str,
    video_paths: tuple[str, ...],
    video_subject: str,
    video_script: str,
    video_language: str,
    platforms: tuple[str, ...],
    youtube_privacy_status: str,
) -> None:
    """后台执行跨平台发布，并只补充发布相关的任务字段。"""
    results = []
    try:
        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_PROCESSING,
            cross_post_error=None,
            cross_post_owner=_cross_post_process_owner,
        )
        if state_updated is not True:
            # False 表示任务已删除，None 表示状态后端暂时不可用。两种情况都
            # 不应继续调用第三方接口，否则用户无法查询或控制这次发布。
            if state_updated is False:
                logger.warning(f"skip cross-post for missing task: {task_id}")
            else:
                _record_cross_post_failure(
                    task_id,
                    RuntimeError("failed to persist cross-post processing state"),
                )
            return

        logger.info(
            f"cross-post started, task_id: {task_id}, platforms: {', '.join(platforms)}"
        )
        youtube_extra = None
        if any(platform.startswith("youtube") for platform in platforms):
            metadata = llm.generate_social_metadata(
                video_subject=video_subject,
                video_script=video_script,
                language=video_language or "",
                platform="youtube_shorts",
            )
            youtube_extra = {
                "youtube_title": metadata.get("title", video_subject),
                "youtube_description": metadata.get("caption", ""),
                "tags": metadata.get("hashtags", []),
                "privacyStatus": youtube_privacy_status,
                "containsSyntheticMedia": True,
            }

        for video_path in video_paths:
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=video_subject or "Check out this video! #shorts #viral",
                platforms=list(platforms),
                youtube_extra=youtube_extra,
            )
            if not isinstance(result, dict):
                result = {
                    "success": False,
                    "error": "Upload-Post returned an invalid response",
                }
            results.append(result)

        failures = [result for result in results if not result.get("success")]
        if failures:
            error_messages = [
                str(
                    result.get("error")
                    or result.get("message")
                    or "unknown upload error"
                )
                for result in failures
            ]
            cross_post_state = const.CROSS_POST_STATE_FAILED
            cross_post_error = "; ".join(error_messages)
            logger.warning(
                f"cross-post completed with failures, task_id: {task_id}, "
                f"failed: {len(failures)}, total: {len(results)}"
            )
        else:
            cross_post_state = const.CROSS_POST_STATE_COMPLETE
            cross_post_error = None
            logger.success(
                f"cross-post completed, task_id: {task_id}, videos: {len(results)}"
            )

        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=cross_post_state,
            cross_post_results=results,
            cross_post_error=cross_post_error,
            cross_post_owner=None,
        )
        if state_updated is False:
            logger.warning(f"discard cross-post result for missing task: {task_id}")
        elif state_updated is None:
            # 上传已经结束但结果没有持久化时，不能继续保留 processing。
            # 失败状态写入会再次经过有限重试，至少让调用方得到明确终态。
            _record_cross_post_failure(
                task_id,
                RuntimeError("failed to persist final cross-post result"),
                results,
            )
    except Exception as exc:
        # 发布失败只影响发布状态，不能反向覆盖已经完成的视频任务。
        # 异常原文写入任务状态，API 调用方无需访问服务端日志也能定位问题。
        logger.exception(f"cross-post failed, task_id: {task_id}, error: {exc}")
        _record_cross_post_failure(task_id, exc, results)


def _run_cross_post_with_slot(*args) -> None:
    """执行发布任务，并确保成功、失败或异常时都会归还队列容量。"""
    try:
        _run_cross_post(*args)
    except Exception as exc:
        # _run_cross_post 已处理预期异常；这里是最后一道保护，避免未来新增
        # 逻辑抛出的异常只保存在无人读取的 Future 中。
        task_id = str(args[0]) if args else "unknown"
        logger.exception(f"cross-post worker crashed, task_id: {task_id}, error: {exc}")
        if args:
            _record_cross_post_failure(task_id, exc)
    finally:
        _cross_post_slots.release()


def _finalize_cross_post_future(task_id: str, future: Future) -> None:
    """清理 Future 注册，并确保取消、异常和状态写入失败都能收敛。"""
    _unregister_cross_post_future(task_id, future)

    try:
        error = future.exception()
    except CancelledError:
        logger.warning(f"cross-post future was cancelled, task_id: {task_id}")
        # Future 在开始执行前被取消时，worker 的 finally 不会运行，因此需要
        # 在回调中归还队列容量，并把持久化状态改为失败。
        _cross_post_slots.release()
        _record_cross_post_failure(
            task_id,
            RuntimeError("cross-post job was cancelled before execution"),
        )
        return
    except Exception as exc:
        logger.exception(
            f"failed to inspect cross-post future, task_id: {task_id}, error: {exc}"
        )
        _ensure_cross_post_terminal_state(task_id)
        return

    if error is not None:
        logger.error(
            f"cross-post future failed, task_id: {task_id}, "
            f"error: {type(error).__name__}: {error}"
        )

    _ensure_cross_post_terminal_state(task_id)


def _schedule_cross_post(
    task_id: str,
    video_paths: list[str],
    params: VideoParams,
    video_script: str,
    platforms: list[str],
    youtube_privacy_status: str,
) -> str | None:
    """提交后台发布任务；成功返回 None，调度失败返回可查询的错误原因。"""
    if not _cross_post_slots.acquire(blocking=False):
        error = "cross-post queue is full; publishing was skipped"
        logger.warning(
            f"skip cross-post because queue is full, task_id: {task_id}, "
            f"capacity: {_cross_post_max_pending_tasks}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=error,
            cross_post_owner=None,
        )
        return error

    try:
        future = _cross_post_executor.submit(
            _run_cross_post_with_slot,
            task_id,
            tuple(video_paths),
            params.video_subject or "",
            video_script,
            params.video_language or "",
            tuple(platforms),
            youtube_privacy_status,
        )
        _register_cross_post_future(task_id, future)
        future.add_done_callback(partial(_finalize_cross_post_future, task_id))
    except RuntimeError as exc:
        _unregister_cross_post_future(task_id)
        _cross_post_slots.release()
        logger.exception(
            f"failed to schedule cross-post, task_id: {task_id}, error: {exc}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=f"failed to schedule cross-post: {exc}",
            cross_post_owner=None,
        )
        return f"failed to schedule cross-post: {exc}"

    return None


def _run_pipeline(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)

    if (
        stop_at in {"materials", "video"}
        and material.supports_smart_visual_matching(params.video_source)
        and params.match_materials_to_script
        and twelvelabs.visual_matching_requested()
    ):
        configuration_error = twelvelabs.validate_smart_visual_matching_configuration()
        if configuration_error:
            return _mark_task_failed(
                task_id,
                "preflight",
                configuration_error,
            )

    # 只有完整成片流程需要视频配乐供应商。尽早阻止缺少 Key 的完整任务，避免
    # 先消耗 LLM、TTS 和素材服务额度；中间产物接口仍可独立使用。
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_enabled = (
        stop_at == "video"
        and video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    if video_music_enabled:
        service = video_music_provider["service"]
        display_name = video_music_provider["display_name"]
        if not service.is_enabled():
            return _mark_task_failed(
                task_id,
                "preflight",
                f"{display_name} background music requires an API key",
            )

        # WebUI 会限制输入长度，但 API、CLI 和历史任务可以绕过前端控件。
        # 在生成脚本、配音和素材之前按供应商上限再次校验，避免完整视频合成后
        # 才由第三方请求拒绝。服务层仍保留同一校验，作为直接调用时的最后防线。
        music_prompt = _get_video_music_prompt(params)
        max_prompt_length = int(getattr(service, "MAX_PROMPT_LENGTH", 0) or 0)
        if max_prompt_length and len(music_prompt) > max_prompt_length:
            return _mark_task_failed(
                task_id,
                "preflight",
                (f"{display_name} music prompt exceeds {max_prompt_length} characters"),
            )

        # 供应商可以选择提供不计费的账号前置检查。检查函数只应抛出确定性
        # 错误；网络波动或权限范围无法确认时由服务层记录警告并继续实际生成。
        validate_access = getattr(service, "validate_generation_access", None)
        if callable(validate_access):
            try:
                validate_access()
            except video_music_provider["error_type"] as exc:
                return _mark_task_failed(task_id, "preflight", str(exc))

    # 1. Generate script
    video_script = generate_script(task_id, params)
    if not video_script or "Error: " in video_script:
        error = (
            video_script.removeprefix("Error: ").strip()
            if isinstance(video_script, str) and "Error: " in video_script
            else "failed to generate video script"
        )
        return _mark_task_failed(task_id, "script", error)

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    ordered_timeline_enabled = (
        params.match_materials_to_script and params.video_source != "local"
    )
    visual_slots: list[VisualSlot] | None = None
    visual_beats: list[VisualBeat] = []
    # ``None`` means "no checklist was requested for this run"; an empty mapping
    # means "requested and nothing could be decomposed". The manifest keeps those
    # two apart, so stay with ``None`` until the gate below actually fires.
    visual_requirement_specs: dict[str, VisualRequirementSpec] | None = None
    # Smart matching drives the renderer from the variable beat timeline; every
    # other configuration keeps the fixed-slot path untouched.
    smart_matching_requested = (
        material.supports_smart_visual_matching(params.video_source)
        and params.match_materials_to_script
        and twelvelabs.visual_matching_requested()
    )

    # 2. Generate terms. Ordered matching must wait for the narration timeline;
    # the normal mode intentionally keeps its existing whole-script behavior.
    video_terms = []
    if params.video_source != "local" and not ordered_timeline_enabled:
        video_terms = generate_terms(task_id, params, video_script)
        if not video_terms:
            return _mark_task_failed(
                task_id,
                "terms",
                "failed to generate video search terms",
            )

    save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms" and not ordered_timeline_enabled:
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # 3. Generate audio
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id,
        params,
        video_script,
        voice_preview=voice_preview,
    )
    if not audio_file:
        return _mark_task_failed(
            task_id,
            "audio",
            "failed to prepare narration audio",
        )

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id,
        params,
        video_script,
        sub_maker,
        audio_file,
        force_timeline=ordered_timeline_enabled,
    )

    material_audio_duration = audio_duration
    if ordered_timeline_enabled:
        if not subtitle_path:
            return _mark_task_failed(
                task_id,
                "narration_timeline",
                "ordered material matching requires a valid narration subtitle timeline",
            )

        exact_audio_duration = voice.get_audio_duration(audio_file)
        if not math.isfinite(exact_audio_duration) or exact_audio_duration <= 0:
            exact_audio_duration = float(audio_duration)
        material_audio_duration = exact_audio_duration
        timing_source = resolve_narration_timing_source(params, sub_maker)
        timed_narration_units: list[TimedNarrationUnit] = []
        timed_unit_source = resolve_timed_narration_timing_source(params, sub_maker)
        if timed_unit_source is not None:
            try:
                timed_narration_units = voice.extract_timed_narration_units(
                    sub_maker=sub_maker,
                    narration_text=video_script,
                    timing_source=timed_unit_source,
                    audio_duration=exact_audio_duration,
                )
            except ValueError as exc:
                # S1 is an additive timing artifact. A provider text transform that
                # cannot be aligned exactly must not alter the existing subtitle or
                # fixed-slot behavior; later semantic stages can use their explicit
                # coarse fallback instead of consuming untrusted word timing.
                logger.warning(
                    "timed narration units unavailable: "
                    f"source={timed_unit_source}, reason={exc}"
                )
        try:
            narration_slots = build_narration_slots(
                subtitle_path=subtitle_path,
                audio_duration=exact_audio_duration,
                timing_source=timing_source,
                expected_script=video_script,
            )
            visual_slots = build_visual_slots(
                narration_slots=narration_slots,
                audio_duration=exact_audio_duration,
                video_clip_duration=params.video_clip_duration,
            )
        except ValueError as exc:
            return _mark_task_failed(task_id, "narration_timeline", str(exc))

        if timed_narration_units:
            try:
                timed_narration_units = associate_timed_units_with_narration_slots(
                    timed_narration_units,
                    narration_slots,
                )
            except ValueError as exc:
                # This is optional S1 metadata. Exact association failure must not
                # make the established NarrationSlot/VisualSlot path fail.
                logger.warning(f"timed narration slot association unavailable: {exc}")
                timed_narration_units = []

        semantic_visual_spans = generate_semantic_visual_spans(
            narration_script=video_script,
            timed_units=timed_narration_units,
            narration_slots=narration_slots,
        )
        visual_beats = build_visual_beats(
            narration_script=video_script,
            semantic_visual_spans=semantic_visual_spans,
            timed_units=timed_narration_units,
            narration_slots=narration_slots,
            audio_duration=exact_audio_duration,
        )
        # Checked before beat queries and before the checklist, because both are
        # paid calls and neither can produce anything usable from a requirement
        # that is really a spoken sentence.
        if visual_beats and semantic_visual_requirements_are_spoken_narration(
            semantic_visual_spans
        ):
            logger.warning(
                "visual beat requirements are spoken narration because semantic "
                "grouping and its repair both failed; falling back to the fixed "
                "visual slot timeline instead of searching for sentences"
            )
            visual_beats = []

        if stop_at == "subtitle":
            persist_narration_timeline(
                task_id=task_id,
                narration_slots=narration_slots,
                visual_slots=visual_slots,
                video_terms=[],
                timed_narration_units=timed_narration_units,
                semantic_visual_spans=semantic_visual_spans,
                visual_beats=visual_beats,
            )
        else:
            # Before the beat queries and before the checklist, because both key
            # on the requirement text: refining it afterwards would leave the
            # shots of a split span sharing one search and one checklist entry.
            # Gated on the beat timeline alone rather than on the verification
            # credentials, because a distinct query per shot is what buys
            # distinct footage whether or not candidates are ever adjudicated.
            # Not run on the `stop_at == "subtitle"` path above, which never
            # searches for anything and so must not pay for this.
            if visual_beats:
                refine_split_span_shot_requirements(visual_beats)
            try:
                # Ask for as many phrasings as material selection is allowed to
                # try on one provider. Fewer may come back; that only costs a
                # fallback, while asking for one guarantees the beat has nothing
                # to retry with before the cascade jumps to a thinner catalog.
                generate_visual_beat_search_queries(
                    visual_beats,
                    queries_per_beat=material.max_query_variants_per_provider(),
                )
            except ValueError as exc:
                # Without one query per beat there is nothing to search per beat,
                # so drop back to the fixed-slot timeline instead of failing the
                # task. The run still gets ordered, script-matched materials.
                logger.warning(
                    f"visual beat search queries unavailable: {exc}; "
                    "falling back to the fixed visual slot timeline"
                )
                visual_beats = []
            try:
                video_terms = generate_visual_slot_search_queries(
                    params=params,
                    visual_slots=visual_slots,
                )
            except ValueError as exc:
                return _mark_task_failed(task_id, "terms", str(exc))
            # The checklist gate is deliberately the credential-aware one that
            # material selection itself uses. The looser beat-timeline gate would
            # decompose requirements for runs that can never verify a candidate,
            # and `smart_matching_requested` keeps sources that never reach smart
            # selection from paying for the decomposition at all.
            if (
                visual_beats
                and smart_matching_requested
                and twelvelabs.is_smart_visual_matching_enabled()
            ):
                visual_requirement_specs = generate_visual_requirement_checklist(
                    visual_beats
                )
            persist_narration_timeline(
                task_id=task_id,
                narration_slots=narration_slots,
                visual_slots=visual_slots,
                video_terms=video_terms,
                timed_narration_units=timed_narration_units,
                semantic_visual_spans=semantic_visual_spans,
                visual_beats=visual_beats,
                visual_requirement_specs=visual_requirement_specs,
            )

        if stop_at == "terms":
            sm.state.update_task(
                task_id,
                state=const.TASK_STATE_COMPLETE,
                progress=100,
                terms=video_terms,
            )
            return {"script": video_script, "terms": video_terms}

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # Beats replace slots only when S4 actually produced a complete timeline with
    # queries. Anything less keeps the proven fixed-slot renderer path.
    use_visual_beats = smart_matching_requested and bool(visual_beats)
    if smart_matching_requested and not use_visual_beats:
        logger.warning(
            "smart visual matching is running on the fixed slot timeline because "
            "no usable visual beat timeline was produced"
        )

    # 5. Get video materials
    # Selection is allowed to rewrite the beat timeline: when no provider,
    # phrasing, or rewritten requirement can fill a beat, the shot beside it absorbs
    # its window — a sibling of the same semantic group where one exists, otherwise
    # the shot of the adjacent group. The narration timing never moves, so this list
    # is either empty or a shorter timeline covering the same span.
    merged_visual_beats: list[VisualBeat] = []
    try:
        downloaded_videos = get_video_materials(
            task_id,
            params,
            video_terms,
            material_audio_duration,
            loomloom_video_request=loomloom_video_request,
            visual_slots=visual_slots,
            visual_beats=visual_beats if use_visual_beats else None,
            visual_requirement_specs=(
                visual_requirement_specs if use_visual_beats else None
            ),
            merged_beats_out=merged_visual_beats if use_visual_beats else None,
        )
    except material.SmartMaterialSelectionError as exc:
        return _mark_task_failed(task_id, "materials", str(exc))
    if not downloaded_videos:
        return _mark_task_failed(
            task_id,
            "materials",
            "failed to prepare video materials",
        )

    # From here on the merged timeline is the only true one: the downloaded
    # material records were written against it, so the renderer must bind to it
    # rather than to the timeline the script stage planned.
    if use_visual_beats and merged_visual_beats:
        logger.warning(
            "visual beat timeline was rewritten by merges: "
            f"planned={len(visual_beats)}, rendering={len(merged_visual_beats)}"
        )
        visual_beats = merged_visual_beats
        # Rewritten in the task file too. The planned timeline is still recoverable
        # from the merge records in ``semantic_verifier_runs``, and leaving a beat
        # list behind that names shots the video does not contain would make the
        # artifact disagree with both the render and its own material records.
        task_artifacts.patch_script_data(
            task_id,
            visual_beats=_visual_beat_records(visual_beats),
        )

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    source_ranges: list[tuple[float, float]] | None = None
    render_segments: list[RenderSegment] | None = None
    if smart_matching_requested:
        try:
            if use_visual_beats:
                # The beat path owns the whole timeline: every beat renders once,
                # in order, for exactly its own duration.
                render_segments = material.load_render_segments(
                    task_id,
                    downloaded_videos,
                    visual_beats,
                    clip_speed=params.video_clip_speed,
                    audio_duration=material_audio_duration,
                )
            else:
                source_ranges = material.load_selected_source_ranges(
                    task_id,
                    downloaded_videos,
                )
        except (OSError, ValueError) as exc:
            return _mark_task_failed(task_id, "video", str(exc))

    # 仅完整视频生成流程才需要处理视频拼接模式；
    # 这样可以避免 /subtitle 和 /audio 这类请求访问不存在的字段。
    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 6. Generate final videos
    final_video_paths, combined_video_paths, generation_warnings = (
        generate_final_videos(
            task_id,
            params,
            downloaded_videos,
            audio_file,
            subtitle_path,
            audio_duration,
            source_ranges=source_ranges,
            render_segments=render_segments,
        )
    )

    if not final_video_paths:
        return _mark_task_failed(
            task_id,
            "video",
            "failed to generate final video",
        )

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    # 7. 先完成视频生成任务，再按需提交跨平台发布。第三方上传可能耗时
    # 数分钟，不应阻塞视频结果返回，也不能反向影响已经生成的成片。
    cross_post_enabled = (
        upload_post.upload_post_service.is_configured()
        and upload_post.upload_post_service.auto_upload
    )
    platforms = (
        list(upload_post.upload_post_service.platforms) if cross_post_enabled else []
    )
    should_cross_post = cross_post_enabled and bool(platforms)
    if cross_post_enabled and not platforms:
        logger.warning(
            f"skip cross-post because no platforms are configured, task_id: {task_id}"
        )
    cross_post_state = const.CROSS_POST_STATE_PENDING if should_cross_post else None

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "cross_post_state": cross_post_state,
        "cross_post_results": None,
        "cross_post_error": None,
        "cross_post_owner": _cross_post_process_owner if should_cross_post else None,
        "warnings": generation_warnings or None,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )

    if should_cross_post:
        scheduling_error = _schedule_cross_post(
            task_id=task_id,
            video_paths=final_video_paths,
            params=params,
            video_script=video_script,
            platforms=platforms,
            youtube_privacy_status=(
                upload_post.upload_post_service.youtube_privacy_status
            ),
        )
        # 队列满或线程池关闭属于同步可知的调度失败。任务状态已经由调度函数
        # 更新，这里同步修正返回快照，避免调用方收到与后续查询不一致的 pending。
        if scheduling_error:
            kwargs["cross_post_state"] = const.CROSS_POST_STATE_FAILED
            kwargs["cross_post_error"] = scheduling_error
            kwargs["cross_post_owner"] = None

    return kwargs


def start(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
):
    """执行任务流水线，并确保未预期异常也会转换成可查询的失败状态。"""
    try:
        return _run_pipeline(
            task_id,
            params,
            stop_at=stop_at,
            voice_preview=voice_preview,
            loomloom_video_request=loomloom_video_request,
        )
    except Exception as exc:
        logger.exception(
            f"unexpected task pipeline failure, task_id: {task_id}, error: {exc}"
        )
        return _mark_task_failed(
            task_id,
            "pipeline",
            f"{type(exc).__name__}: {exc}",
        )


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="金钱的作用",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
