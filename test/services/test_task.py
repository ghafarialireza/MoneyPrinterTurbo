import unittest
import os
import shutil
import sys
import tempfile
from concurrent.futures import Future
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import task as tm
from app.models.schema import (
    CriticalVisualFact,
    MaterialInfo,
    NarrationOverlap,
    NarrationSlot,
    RenderSegment,
    SemanticVisualSpan,
    TimedNarrationUnit,
    VideoParams,
    VisualBeat,
    VisualRequirementSpec,
    VisualSlot,
)
from app.services.state import MemoryState, RedisState
from app.utils import utils

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


def _requirement_spec(requirement):
    """A minimal, source-grounded decomposition of one visual requirement."""
    return VisualRequirementSpec(
        schema_version="visual-requirement-spec-v1",
        generator_provider="test-provider",
        generator_model="test-model",
        original_requirement=requirement,
        subjects=["subject"],
        primary_action=None,
        objects=[],
        required_relations=[],
        required_context=[],
        required_visible_state=[],
        optional_attributes=[],
        critical_visual_facts=[
            CriticalVisualFact(
                id="f1",
                fact=requirement,
                mandatory=True,
                direct_evidence_needed=True,
                evidence_description="The defining requirement is directly visible",
                basis_type="explicit",
                basis_quote=requirement,
            )
        ],
        ambiguity_notes=[],
    )


class TestTaskService(unittest.TestCase):
    def setUp(self):
        # 发布 Future 注册表是进程级状态。测试间清理可以避免某个模拟 Future
        # 影响后续恢复测试，同时不会触碰真正线程池中的生产任务。
        with tm._cross_post_registry_lock:
            tm._cross_post_futures.clear()

    def tearDown(self):
        with tm._cross_post_registry_lock:
            tm._cross_post_futures.clear()

    @staticmethod
    def _semantic_units(script, tokens, slot_indexes=None, boundary_type="WordBoundary"):
        units = []
        cursor = 0
        slot_indexes = slot_indexes or [1] * len(tokens)
        for index, (token, slot_index) in enumerate(
            zip(tokens, slot_indexes),
            start=1,
        ):
            start_char = script.find(token, cursor)
            if start_char < 0:
                raise AssertionError(f"test token is not sequentially present: {token}")
            end_char = start_char + len(token)
            start_time = 0.2 + (index - 1) * 0.4
            units.append(
                TimedNarrationUnit(
                    index=index,
                    text=token,
                    start_time=start_time,
                    end_time=start_time + 0.25,
                    duration=0.25,
                    timing_source="edge_tts_boundary",
                    timing_quality="boundary",
                    source_narration_slot_index=slot_index,
                    source_boundary_type=boundary_type,
                    script_start_char=start_char,
                    script_end_char=end_char,
                )
            )
            cursor = end_char
        return units

    @staticmethod
    def _timed_units_with_ranges(script, entries):
        """Build exact-offset units from (text, start, end, slot) entries."""
        units = []
        cursor = 0
        for index, (text, start_time, end_time, slot_index) in enumerate(
            entries,
            start=1,
        ):
            start_char = script.find(text, cursor)
            if start_char < 0:
                raise AssertionError(f"test unit is not sequentially present: {text}")
            end_char = start_char + len(text)
            units.append(
                TimedNarrationUnit(
                    index=index,
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    timing_source="edge_tts_boundary",
                    timing_quality="boundary",
                    source_narration_slot_index=slot_index,
                    source_boundary_type="WordBoundary",
                    script_start_char=start_char,
                    script_end_char=end_char,
                )
            )
            cursor = end_char
        return units

    @staticmethod
    def _semantic_spans_from_ranges(script, units, ranges):
        spans = []
        for index, (start_unit, end_unit_exclusive, requirement) in enumerate(
            ranges,
            start=1,
        ):
            source_units = units[start_unit:end_unit_exclusive]
            spans.append(
                SemanticVisualSpan(
                    index=index,
                    start_unit=start_unit,
                    end_unit_exclusive=end_unit_exclusive,
                    spoken_text=tm.reconstruct_semantic_spoken_text(
                        script,
                        units,
                        start_unit,
                        end_unit_exclusive,
                    ),
                    visual_requirement=requirement,
                    source_narration_slot_indexes=list(
                        dict.fromkeys(
                            unit.source_narration_slot_index
                            for unit in source_units
                            if unit.source_narration_slot_index is not None
                        )
                    ),
                    start_time=source_units[0].start_time,
                    end_time=source_units[-1].end_time,
                    timing_source="edge_tts_boundary",
                    timing_quality="boundary",
                    grouping_source="llm",
                )
            )
        return spans

    def test_is_task_busy_covers_generation_and_cross_posting(self):
        """删除入口必须同时识别视频生成和跨平台发布的活跃状态。"""
        busy_tasks = (
            {"state": tm.const.TASK_STATE_PROCESSING},
            {
                "state": tm.const.TASK_STATE_COMPLETE,
                "cross_post_state": tm.const.CROSS_POST_STATE_PENDING,
            },
            {
                "state": tm.const.TASK_STATE_COMPLETE,
                "cross_post_state": tm.const.CROSS_POST_STATE_PROCESSING,
            },
        )
        for task in busy_tasks:
            with self.subTest(task=task):
                self.assertTrue(tm.is_task_busy(task))

        self.assertFalse(
            tm.is_task_busy(
                {
                    "state": tm.const.TASK_STATE_COMPLETE,
                    "cross_post_state": tm.const.CROSS_POST_STATE_COMPLETE,
                }
            )
        )
        self.assertFalse(tm.is_task_busy(None))

    def test_generate_script_forwards_advanced_prompt_options(self):
        """
        任务生成入口和 WebUI/API 共用 VideoParams。这里验证自动生成文案时，
        高级提示词参数会继续传到 LLM 服务层，避免只在 /scripts 接口生效。
        """
        params = VideoParams(
            video_subject="咖啡",
            video_script="",
            video_language="zh-CN",
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )

        with patch.object(
            tm.llm, "generate_script", return_value="生成的文案"
        ) as generate:
            result = tm.generate_script("task-id", params)

        self.assertEqual(result, "生成的文案")
        generate.assert_called_once_with(
            video_subject="咖啡",
            language="zh-CN",
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )

    def test_generate_final_videos_forwards_clip_speed(self):
        """任务编排层必须把用户选择的画面速度传给视频合成服务。"""
        params = VideoParams(
            video_subject="test",
            video_count=1,
            video_clip_speed=1.25,
        )

        with (
            patch.object(tm.video, "combine_videos") as combine_videos,
            patch.object(tm.video, "generate_video"),
            patch.object(tm.sm.state, "update_task"),
        ):
            tm.generate_final_videos(
                task_id="clip-speed-task",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(combine_videos.call_args.kwargs["clip_speed"], 1.25)

    def test_generate_final_videos_uses_generated_sonilo_music(self):
        """Sonilo 必须针对每条拼接后的视频生成配乐，并传给最终混音。"""
        params = VideoParams(
            video_subject="test",
            video_count=1,
            bgm_type="sonilo",
            sonilo_bgm_prompt="warm acoustic",
        )

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.sonilo,
                "generate_bgm",
                side_effect=lambda **kwargs: kwargs["output_path"],
            ) as generate_bgm,
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            _, _, warnings = tm.generate_final_videos(
                task_id="sonilo-task",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(warnings, [])
        self.assertEqual(generate_bgm.call_args.kwargs["video_duration"], 5)
        self.assertEqual(generate_bgm.call_args.kwargs["prompt"], "warm acoustic")
        self.assertTrue(
            generate_video.call_args.kwargs["bgm_file_override"].endswith(
                "sonilo-bgm-1.m4a"
            )
        )

    def test_generate_final_videos_uses_generated_elevenlabs_music(self):
        """ElevenLabs 应复用视频配乐编排，并使用通用风格提示词。"""
        params = VideoParams(
            video_subject="test",
            video_count=1,
            bgm_type="elevenlabs",
            video_music_prompt="gentle documentary",
        )

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.elevenlabs_music,
                "generate_bgm",
                side_effect=lambda **kwargs: kwargs["output_path"],
            ) as generate_bgm,
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            _, _, warnings = tm.generate_final_videos(
                task_id="elevenlabs-task",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(warnings, [])
        self.assertEqual(generate_bgm.call_args.kwargs["video_duration"], 5)
        self.assertEqual(generate_bgm.call_args.kwargs["prompt"], "gentle documentary")
        self.assertTrue(
            generate_video.call_args.kwargs["bgm_file_override"].endswith(
                "elevenlabs-bgm-1.mp3"
            )
        )

    def test_generate_final_videos_falls_back_on_elevenlabs_failure(self):
        """ElevenLabs 暂时失败时必须保留无配乐视频和结构化警告。"""
        params = VideoParams(video_subject="test", bgm_type="elevenlabs")

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.elevenlabs_music,
                "generate_bgm",
                side_effect=tm.elevenlabs_music.ElevenLabsMusicError(
                    "temporary outage"
                ),
            ),
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="elevenlabs-fallback",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(
            warnings,
            [{"code": "elevenlabs_bgm_failed", "video_index": 1}],
        )
        self.assertEqual(generate_video.call_args.kwargs["bgm_file_override"], "")

    def test_generate_final_videos_falls_back_without_bgm_on_sonilo_failure(self):
        """第三方配乐失败时应完成视频并返回可见警告，而不是丢弃所有产物。"""
        params = VideoParams(video_subject="test", bgm_type="sonilo")

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.sonilo,
                "generate_bgm",
                side_effect=tm.sonilo.SoniloError("temporary outage"),
            ),
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="sonilo-fallback",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(warnings, [{"code": "sonilo_bgm_failed", "video_index": 1}])
        self.assertEqual(generate_video.call_args.kwargs["bgm_file_override"], "")

    def test_generate_final_videos_skips_sonilo_when_volume_is_zero(self):
        """0 音量必须完全跳过 Sonilo 生成，并显式禁用残留背景音乐。"""
        params = VideoParams(
            video_subject="test",
            bgm_type="sonilo",
            bgm_volume=0.0,
            bgm_file="stale-custom-bgm.mp3",
        )

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(tm.sonilo, "generate_bgm") as generate_bgm,
            patch.object(tm.video, "generate_video", return_value=True) as generate,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="sonilo-zero-volume",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(warnings, [])
        generate_bgm.assert_not_called()
        self.assertEqual(generate.call_args.kwargs["bgm_file_override"], "")

    def test_generate_final_videos_warns_when_sonilo_mix_fails(self):
        """Sonilo 生成成功但最终混音失败时，任务必须保留视频并返回警告。"""
        params = VideoParams(video_subject="test", bgm_type="sonilo")

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.sonilo,
                "generate_bgm",
                side_effect=lambda **kwargs: kwargs["output_path"],
            ),
            patch.object(tm.video, "generate_video", return_value=False) as generate,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="sonilo-mix-fallback",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(warnings, [{"code": "sonilo_bgm_failed", "video_index": 1}])
        self.assertTrue(generate.call_args.kwargs["bgm_file_override"].endswith(".m4a"))

    def test_start_rejects_missing_sonilo_key_before_costly_pipeline_steps(self):
        """完整任务缺少 Sonilo Key 时不能先调用 LLM、TTS 或素材服务。"""
        params = VideoParams(video_subject="test", bgm_type="sonilo")
        state = MemoryState()
        with (
            patch.object(tm.sonilo, "is_enabled", return_value=False),
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm, "get_video_materials") as get_materials,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("missing-sonilo-key", params)

        generate_script.assert_not_called()
        generate_audio.assert_not_called()
        get_materials.assert_not_called()
        failed_task = state.get_task("missing-sonilo-key")
        self.assertEqual(result, failed_task)
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "preflight")
        self.assertIn("API key", failed_task["error"])

    def test_start_does_not_require_sonilo_key_when_volume_is_zero(self):
        """0 音量不会使用 Sonilo，因此缺少 Key 时仍应进入正常任务流水线。"""
        params = VideoParams(
            video_subject="test",
            bgm_type="sonilo",
            bgm_volume=0.0,
        )
        state = MemoryState()
        with (
            patch.object(tm.sonilo, "is_enabled", return_value=False),
            patch.object(tm, "generate_script", return_value="") as generate_script,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("zero-volume-without-key", params)

        generate_script.assert_called_once_with("zero-volume-without-key", params)
        self.assertEqual(result["failed_stage"], "script")

    def test_loomloom_material_failure_keeps_remote_run_id(self):
        """远端运行已创建后失败，任务状态必须保留 LoomLoom run ID。"""
        params = VideoParams(video_subject="AI 办公", video_source="loomloom")
        settings = tm.loomloom.LoomLoomSettings(
            base_url="https://example.test/loom/v1",
            api_token="test-token",
            market_listing_id=tm.loomloom.DEFAULT_SCRIPT_MARKET_LISTING_ID,
        )
        batch = tm.loomloom.LoomLoomVideoBatch(
            input_rows=(
                {
                    "scenePrompt": "office worker",
                    "aspectRatio": "9:16",
                    "sceneIndex": "1",
                },
            ),
        )
        request = tm.loomloom.LoomLoomConfirmedVideoRequest(
            settings=settings,
            batch=batch,
            listing_version_id="version-1",
            client_request_id="mpt-video-1",
        )
        backend = MagicMock()
        backend.execute.return_value = tm.loomloom.LoomLoomExecution(
            run_id="run-1",
            transaction_id="transaction-1",
            transaction_status="running",
            listing_version_id="version-1",
        )
        backend.wait_for_run.side_effect = tm.loomloom.LoomLoomRunError(
            "remote run timeout"
        )
        state = MemoryState()
        state.update_task(
            "loomloom-material-timeout",
            state=tm.const.TASK_STATE_PROCESSING,
            progress=40,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.loomloom,
                "LoomLoomVideoBackend",
                return_value=backend,
            ),
        ):
            result = tm.get_video_materials(
                "loomloom-material-timeout",
                params,
                ["office worker"],
                10,
                loomloom_video_request=request,
            )

        self.assertIsNone(result)
        failed_task = state.get_task("loomloom-material-timeout")
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "materials")
        self.assertEqual(failed_task["loomloom_run_id"], "run-1")
        self.assertEqual(failed_task["loomloom_listing_version_id"], "version-1")

    def test_loomloom_state_failure_does_not_abandon_paid_remote_run(self):
        """状态后端不可用时仍需等待并下载已经开始计费的远端任务。"""
        params = VideoParams(video_subject="AI 办公", video_source="loomloom")
        settings = tm.loomloom.LoomLoomSettings(
            base_url="https://example.test/loom/v1",
            api_token="test-token",
            market_listing_id=tm.loomloom.DEFAULT_VIDEO_MARKET_LISTING_ID,
        )
        request = tm.loomloom.LoomLoomConfirmedVideoRequest(
            settings=settings,
            batch=tm.loomloom.LoomLoomVideoBatch(
                input_rows=(
                    {
                        "scenePrompt": "office worker",
                        "aspectRatio": "9:16",
                        "sceneIndex": "1",
                    },
                )
            ),
            listing_version_id="version-1",
            client_request_id="mpt-video-state-failure",
        )
        backend = MagicMock()
        backend.execute.return_value = tm.loomloom.LoomLoomExecution(
            run_id="paid-run-1",
            transaction_id="transaction-1",
            transaction_status="running",
            listing_version_id="version-1",
        )
        backend.download_video_results.return_value = ("clip.mp4",)
        unavailable_state = MagicMock()
        unavailable_state.patch_task.side_effect = RuntimeError("Redis unavailable")

        with (
            patch.object(tm.sm, "state", unavailable_state),
            patch.object(
                tm.loomloom,
                "LoomLoomVideoBackend",
                return_value=backend,
            ),
            patch.object(tm.time, "sleep") as sleep,
        ):
            result = tm.get_video_materials(
                "loomloom-state-failure",
                params,
                ["office worker"],
                10,
                loomloom_video_request=request,
            )

        self.assertEqual(result, ["clip.mp4"])
        self.assertEqual(
            unavailable_state.patch_task.call_count,
            tm._LOOMLOOM_STATE_WRITE_ATTEMPTS,
        )
        self.assertEqual(
            sleep.call_count,
            tm._LOOMLOOM_STATE_WRITE_ATTEMPTS - 1,
        )
        backend.wait_for_run.assert_called_once_with("paid-run-1")
        backend.download_video_results.assert_called_once()

    def test_mark_task_failed_preserves_a_specific_service_failure(self):
        """服务层已记录具体错误时，编排层不能再用通用错误覆盖它。"""
        state = MemoryState()
        state.update_task(
            "specific-service-failure",
            state=tm.const.TASK_STATE_FAILED,
            progress=40,
            failed_stage="materials",
            error="remote run timed out",
            loomloom_run_id="run-1",
        )

        with patch.object(tm.sm, "state", state):
            result = tm._mark_task_failed(
                "specific-service-failure",
                "materials",
                "failed to prepare video materials",
            )

        self.assertEqual(result["error"], "remote run timed out")
        self.assertEqual(result["loomloom_run_id"], "run-1")

    def test_start_rejects_missing_elevenlabs_key_before_pipeline_steps(self):
        """完整任务缺少 ElevenLabs Key 时必须在任何付费步骤前失败。"""
        params = VideoParams(video_subject="test", bgm_type="elevenlabs")
        state = MemoryState()
        with (
            patch.object(tm.elevenlabs_music, "is_enabled", return_value=False),
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("missing-elevenlabs-key", params)

        generate_script.assert_not_called()
        generate_audio.assert_not_called()
        self.assertEqual(result["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("ElevenLabs", result["error"])

    def test_start_rejects_free_elevenlabs_plan_before_pipeline_steps(self):
        """已确认的免费套餐不能先消耗 LLM、TTS 或素材服务额度。"""
        params = VideoParams(video_subject="test", bgm_type="elevenlabs")
        state = MemoryState()
        with (
            patch.object(tm.elevenlabs_music, "is_enabled", return_value=True),
            patch.object(
                tm.elevenlabs_music,
                "validate_generation_access",
                side_effect=(
                    tm.elevenlabs_music.ElevenLabsPaidPlanRequiredError(
                        "ElevenLabs Music API requires a paid plan"
                    )
                ),
            ) as validate_access,
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("free-elevenlabs-plan", params)

        validate_access.assert_called_once_with()
        generate_script.assert_not_called()
        generate_audio.assert_not_called()
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("paid plan", result["error"])

    def test_start_rejects_oversized_elevenlabs_prompt_before_account_check(self):
        """API/CLI 绕过 WebUI 时，超长提示词也必须在昂贵步骤前被拒绝。"""
        params = VideoParams(
            video_subject="test",
            bgm_type="elevenlabs",
            video_music_prompt="x" * 1001,
        )
        state = MemoryState()
        with (
            patch.object(tm.elevenlabs_music, "is_enabled", return_value=True),
            patch.object(
                tm.elevenlabs_music, "validate_generation_access"
            ) as validate_access,
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("oversized-elevenlabs-prompt", params)

        validate_access.assert_not_called()
        generate_script.assert_not_called()
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("1000", result["error"])

    def test_generate_terms_uses_script_order_mode_when_enabled(self):
        """
        默认模式不受影响；只有用户显式开启素材按文案顺序匹配时，任务层才
        要求 LLM 生成有序关键词，并适当增加关键词数量以覆盖更多脚本片段。
        """
        params = VideoParams(
            video_subject="城市通勤",
            video_script="",
            match_materials_to_script=True,
        )

        with patch.object(
            tm.llm, "generate_terms", return_value=["city", "train"]
        ) as generate:
            result = tm.generate_terms("task-id", params, "先城市，再地铁")

        self.assertEqual(result, ["city", "train"])
        generate.assert_called_once_with(
            video_subject="城市通勤",
            video_script="先城市，再地铁",
            amount=8,
            match_script_order=True,
        )

    def test_valid_srt_builds_typed_narration_slots(self):
        srt = (
            "1\n00:00:00,000 --> 00:00:02,800\nSentence A\n\n"
            "2\n00:00:02,800 --> 00:00:06,500\nSentence B\n\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_path = Path(tmp_dir) / "subtitle.srt"
            subtitle_path.write_text(srt, encoding="utf-8")
            slots = tm.build_narration_slots(
                subtitle_path=str(subtitle_path),
                audio_duration=6.5,
                timing_source="edge_tts_boundary",
                expected_script="Sentence A. Sentence B.",
            )

        self.assertEqual(len(slots), 2)
        self.assertIsInstance(slots[0], NarrationSlot)
        self.assertEqual(slots[0].index, 1)
        self.assertEqual(slots[0].start_time, 0.0)
        self.assertEqual(slots[0].end_time, 2.8)
        self.assertEqual(slots[0].duration, 2.8)
        self.assertEqual(slots[0].text, "Sentence A")
        self.assertEqual(slots[0].timing_source, "edge_tts_boundary")

    def test_narration_slots_reject_zero_range_and_missing_narration(self):
        invalid_cases = {
            "zero range": (
                "1\n00:00:00,000 --> 00:00:00,000\nSentence A\n\n",
                "Sentence A.",
                "end_time > start_time",
            ),
            "missing narration": (
                "1\n00:00:00,000 --> 00:00:01,000\nSentence A\n\n",
                "Sentence A. Sentence B.",
                "missing narration",
            ),
            "empty text": (
                "1\n00:00:00,000 --> 00:00:01,000\n\n",
                "",
                "empty text|empty or unavailable",
            ),
            "after audio": (
                "1\n00:00:00,000 --> 00:00:03,000\nSentence A\n\n",
                "Sentence A.",
                "ends after the audio duration",
            ),
            "not ascending": (
                "1\n00:00:01,000 --> 00:00:02,000\nSentence A\n\n"
                "2\n00:00:00,500 --> 00:00:03,000\nSentence B\n\n",
                "Sentence A. Sentence B.",
                "not ascending",
            ),
        }
        for name, (srt, script, error_text) in invalid_cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp_dir:
                subtitle_path = Path(tmp_dir) / "subtitle.srt"
                subtitle_path.write_text(srt, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error_text):
                    tm.build_narration_slots(
                        subtitle_path=str(subtitle_path),
                        audio_duration=2.0,
                        timing_source="estimated",
                        expected_script=script,
                    )

    def test_visual_slots_include_overlapping_narration_and_short_last_slot(self):
        narration_slots = [
            NarrationSlot(1, 0.0, 2.8, 2.8, "Sentence A", "whisper"),
            NarrationSlot(2, 2.8, 6.5, 3.7, "Sentence B", "whisper"),
            NarrationSlot(3, 6.5, 10.1, 3.6, "Sentence C", "whisper"),
        ]

        visual_slots = tm.build_visual_slots(
            narration_slots=narration_slots,
            audio_duration=10.1,
            video_clip_duration=4,
        )

        self.assertEqual(len(visual_slots), 3)
        self.assertIsInstance(visual_slots[0], VisualSlot)
        self.assertEqual(visual_slots[0].narration_slot_indexes, [1, 2])
        self.assertEqual(visual_slots[0].narration_text, "Sentence A Sentence B")
        self.assertEqual(visual_slots[0].primary_narration_slot_index, 1)
        self.assertEqual(visual_slots[0].primary_narration_text, "Sentence A")
        self.assertEqual(visual_slots[0].visual_requirement, "Sentence A")
        overlap_durations = [
            overlap.overlap_duration
            for overlap in visual_slots[0].narration_overlaps
        ]
        self.assertAlmostEqual(overlap_durations[0], 2.8)
        self.assertAlmostEqual(overlap_durations[1], 1.2)
        self.assertEqual(visual_slots[1].narration_slot_indexes, [2, 3])
        self.assertEqual(visual_slots[2].start_time, 8.0)
        self.assertEqual(visual_slots[2].end_time, 10.1)
        self.assertAlmostEqual(visual_slots[2].duration, 2.1)
        self.assertEqual(visual_slots[0].timing_quality, "speech_recognition")

    def test_visual_slot_primary_narration_tie_uses_midpoint_then_index(self):
        narration_slots = [
            NarrationSlot(1, 0.0, 2.0, 2.0, "First scene", "whisper"),
            NarrationSlot(2, 2.0, 4.0, 2.0, "Second scene", "whisper"),
        ]

        slot = tm.build_visual_slots(narration_slots, 4.0, 4.0)[0]

        self.assertEqual(slot.narration_text, "First scene Second scene")
        self.assertEqual(slot.primary_narration_slot_index, 2)
        self.assertEqual(slot.visual_requirement, "Second scene")

        same_midpoint = [
            NarrationSlot(7, 0.0, 4.0, 4.0, "Higher index", "whisper"),
            NarrationSlot(3, 0.0, 4.0, 4.0, "Lower index", "whisper"),
        ]
        stable_slot = tm.build_visual_slots(same_midpoint, 4.0, 4.0)[0]
        self.assertEqual(stable_slot.primary_narration_slot_index, 3)

    def test_visual_slot_exact_boundary_does_not_inherit_adjacent_narration(self):
        narration_slots = [
            NarrationSlot(1, 0.0, 4.0, 4.0, "Opening scene", "whisper"),
            NarrationSlot(2, 4.0, 8.0, 4.0, "Following scene", "whisper"),
        ]

        visual_slots = tm.build_visual_slots(narration_slots, 8.0, 4.0)

        self.assertEqual(visual_slots[0].narration_slot_indexes, [1])
        self.assertEqual(visual_slots[0].visual_requirement, "Opening scene")
        self.assertEqual(visual_slots[1].narration_slot_indexes, [2])
        self.assertEqual(visual_slots[1].visual_requirement, "Following scene")

    def test_visual_slot_selects_largest_of_three_overlaps(self):
        narration_slots = [
            NarrationSlot(1, 0.0, 1.0, 1.0, "Brief opening", "whisper"),
            NarrationSlot(2, 1.0, 3.5, 2.5, "Dominant action", "whisper"),
            NarrationSlot(3, 3.5, 4.0, 0.5, "Brief ending", "whisper"),
        ]

        slot = tm.build_visual_slots(narration_slots, 4.0, 4.0)[0]

        self.assertEqual(slot.narration_slot_indexes, [1, 2, 3])
        self.assertEqual(slot.primary_narration_slot_index, 2)
        self.assertEqual(slot.visual_requirement, "Dominant action")
        self.assertEqual(
            [overlap.narration_slot_index for overlap in slot.narration_overlaps],
            [1, 2, 3],
        )

    def test_coffee_slot_keeps_context_but_uses_largest_overlap_requirement(self):
        picking = (
            "A farm worker reaches between the leaves and hand-picks the ripe "
            "cherries into a basket."
        )
        drying = (
            "Pale coffee beans spread across drying beds and sit under the warm sun."
        )
        narration_slots = [
            NarrationSlot(1, 0.0, 4.0, 4.0, "Coffee farm introduction.", "whisper"),
            NarrationSlot(2, 4.0, 9.2, 5.2, picking, "whisper"),
            NarrationSlot(3, 9.2, 14.0, 4.8, drying, "whisper"),
        ]
        visual_slots = tm.build_visual_slots(narration_slots, 14.0, 4.0)
        coffee_slot = visual_slots[2]

        self.assertEqual(coffee_slot.start_time, 8.0)
        self.assertEqual(coffee_slot.end_time, 12.0)
        self.assertEqual(coffee_slot.narration_slot_indexes, [2, 3])
        self.assertEqual(coffee_slot.narration_text, f"{picking} {drying}")
        self.assertEqual(coffee_slot.primary_narration_slot_index, 3)
        self.assertEqual(coffee_slot.visual_requirement, drying)
        self.assertAlmostEqual(
            coffee_slot.narration_overlaps[0].overlap_duration, 1.2
        )
        self.assertAlmostEqual(
            coffee_slot.narration_overlaps[1].overlap_duration, 2.8
        )

        params = VideoParams(
            video_subject="coffee production",
            match_materials_to_script=True,
        )
        with patch.object(
            tm.llm,
            "generate_visual_slot_queries",
            return_value={slot.index: [f"query {slot.index}"] for slot in visual_slots},
        ) as generate:
            tm.generate_visual_slot_search_queries(params, visual_slots)

        sent_slot = generate.call_args.kwargs["visual_slots"][2]
        self.assertEqual(sent_slot["visual_requirement"], drying)
        self.assertNotIn("narration_text", sent_slot)

    def test_visual_slot_queries_stay_attached_to_their_indexes(self):
        narration_slots = [
            NarrationSlot(
                index=index,
                start_time=(index - 1) * 4,
                end_time=index * 4,
                duration=4,
                text=f"Narration for slot {index}",
                timing_source="edge_tts_boundary",
            )
            for index in range(1, 6)
        ]
        visual_slots = tm.build_visual_slots(narration_slots, 20, 4)
        returned_queries = {
            index: [f"visible query {index}"] for index in reversed(range(1, 6))
        }
        params = VideoParams(
            video_subject="railway repair",
            match_materials_to_script=True,
            video_terms=["stale whole-script keyword"],
        )

        with patch.object(
            tm.llm,
            "generate_visual_slot_queries",
            return_value=returned_queries,
        ) as generate:
            terms = tm.generate_visual_slot_search_queries(params, visual_slots)

        self.assertEqual(terms, [f"visible query {index}" for index in range(1, 6)])
        self.assertEqual(visual_slots[4].search_queries, ["visible query 5"])
        sent_slots = generate.call_args.kwargs["visual_slots"]
        self.assertEqual(sent_slots[4]["slot_index"], 5)
        self.assertEqual(sent_slots[4]["visual_requirement"], "Narration for slot 5")

    def test_estimated_tts_timing_is_explicitly_marked_estimated(self):
        params = VideoParams(
            video_subject="test",
            voice_name="gemini:Zephyr-Female",
        )
        with patch.object(
            tm.config,
            "app",
            dict(tm.config.app, subtitle_provider="edge"),
        ):
            source = tm.resolve_narration_timing_source(params, object())

        self.assertEqual(source, "estimated")

    def test_timeline_artifact_contains_slot_text_queries_and_timing_quality(self):
        narration_slots = [NarrationSlot(1, 0.0, 3.0, 3.0, "Sentence A", "estimated")]
        timed_units = [
            TimedNarrationUnit(
                index=1,
                text="Sentence A",
                start_time=0.2,
                end_time=2.8,
                duration=2.6,
                timing_source="estimated",
                timing_quality="estimated",
                source_narration_slot_index=1,
                source_boundary_type="EstimatedScriptSegment",
                script_start_char=0,
                script_end_char=10,
            )
        ]
        visual_slots = [
            VisualSlot(
                index=1,
                start_time=0.0,
                end_time=3.0,
                duration=3.0,
                narration_slot_indexes=[1],
                narration_text="Sentence A",
                primary_narration_slot_index=1,
                primary_narration_text="Sentence A",
                visual_requirement="Sentence A",
                narration_overlaps=[NarrationOverlap(1, 0.0, 3.0, 3.0)],
                search_queries=["visible subject action"],
                timing_source="estimated",
                timing_quality="estimated",
            )
        ]

        with patch.object(tm.task_artifacts, "patch_script_data") as persist:
            tm.persist_narration_timeline(
                task_id="timeline-artifact",
                narration_slots=narration_slots,
                visual_slots=visual_slots,
                video_terms=["visible subject action"],
                timed_narration_units=timed_units,
            )

        saved_visual = persist.call_args.kwargs["visual_slots"][0]
        self.assertEqual(saved_visual["narration_text"], "Sentence A")
        self.assertEqual(saved_visual["primary_narration_slot_index"], 1)
        self.assertEqual(saved_visual["primary_narration_text"], "Sentence A")
        self.assertEqual(saved_visual["visual_requirement"], "Sentence A")
        self.assertEqual(
            saved_visual["narration_overlaps"],
            [
                {
                    "narration_slot_index": 1,
                    "overlap_start_time": 0.0,
                    "overlap_end_time": 3.0,
                    "overlap_duration": 3.0,
                }
            ],
        )
        self.assertEqual(saved_visual["search_queries"], ["visible subject action"])
        self.assertEqual(saved_visual["timing_source"], "estimated")
        self.assertEqual(saved_visual["timing_quality"], "estimated")
        self.assertEqual(persist.call_args.kwargs["timeline_schema_version"], 2)
        self.assertEqual(
            persist.call_args.kwargs["timed_narration_units"],
            [
                {
                    "index": 1,
                    "text": "Sentence A",
                    "start_time": 0.2,
                    "end_time": 2.8,
                    "duration": 2.6,
                    "source_narration_slot_index": 1,
                    "timing_source": "estimated",
                    "timing_quality": "estimated",
                    "source_boundary_type": "EstimatedScriptSegment",
                    "script_start_char": 0,
                    "script_end_char": 10,
                }
            ],
        )

    def test_timed_units_associate_by_order_and_leave_crossing_cue_unassigned(self):
        narration_slots = [
            NarrationSlot(1, 0.0, 1.0, 1.0, "Bright red", "edge_tts_boundary"),
            NarrationSlot(2, 1.0, 2.0, 1.0, "coffee cherries", "edge_tts_boundary"),
        ]
        timed_units = [
            TimedNarrationUnit(
                1,
                "Bright",
                0.1,
                0.5,
                0.4,
                "edge_tts_boundary",
                "boundary",
                source_boundary_type="WordBoundary",
                script_start_char=0,
                script_end_char=6,
            ),
            TimedNarrationUnit(
                2,
                "red coffee",
                0.6,
                1.4,
                0.8,
                "edge_tts_boundary",
                "boundary",
                source_boundary_type="SentenceBoundary",
                script_start_char=7,
                script_end_char=17,
            ),
            TimedNarrationUnit(
                3,
                "cherries",
                1.5,
                1.9,
                0.4,
                "edge_tts_boundary",
                "boundary",
                source_boundary_type="WordBoundary",
                script_start_char=18,
                script_end_char=26,
            ),
        ]

        associated = tm.associate_timed_units_with_narration_slots(
            timed_units,
            narration_slots,
        )

        self.assertEqual(
            [unit.source_narration_slot_index for unit in associated],
            [1, None, 2],
        )

    def test_timed_unit_association_rejects_equal_length_different_text(self):
        narration_slots = [
            NarrationSlot(1, 0.0, 1.0, 1.0, "red", "edge_tts_boundary")
        ]
        timed_units = [
            TimedNarrationUnit(
                1,
                "bed",
                0.1,
                0.9,
                0.8,
                "edge_tts_boundary",
                "boundary",
                source_boundary_type="WordBoundary",
                script_start_char=0,
                script_end_char=3,
            )
        ]

        with self.assertRaisesRegex(ValueError, "do not align"):
            tm.associate_timed_units_with_narration_slots(
                timed_units,
                narration_slots,
            )

    def test_old_script_json_without_timed_units_remains_readable(self):
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            tm.utils,
            "task_dir",
            return_value=tmp_dir,
        ):
            legacy_payload = {
                "script": "Legacy narration.",
                "narration_slots": [],
                "visual_slots": [],
            }
            tm.task_artifacts.write_script_data("legacy", legacy_payload)

            self.assertEqual(
                tm.task_artifacts.read_script_data("legacy"),
                legacy_payload,
            )

            timed_unit = TimedNarrationUnit(
                1,
                "Legacy narration",
                0.1,
                0.9,
                0.8,
                "estimated",
                "estimated",
                source_boundary_type="EstimatedScriptSegment",
                script_start_char=0,
                script_end_char=16,
            )
            tm.persist_narration_timeline(
                task_id="legacy",
                narration_slots=[],
                visual_slots=[],
                video_terms=[],
                timed_narration_units=[timed_unit],
            )
            upgraded = tm.task_artifacts.read_script_data("legacy")
            self.assertEqual(upgraded["script"], "Legacy narration.")
            self.assertEqual(upgraded["timeline_schema_version"], 2)
            self.assertEqual(upgraded["timed_narration_units"][0]["text"], "Legacy narration")

    def test_one_visible_concept_stays_one_semantic_span_across_punctuation(self):
        cases = (
            "The worker removes damaged boards from the wall.",
            (
                "The beans enter the roaster. They tumble continuously. "
                "They slowly turn brown."
            ),
            "Coffee beans tumble inside the drum.\n\nThey slowly turn brown.",
            "The worker carefully removes three damaged boards, from the wall.",
        )
        for script in cases:
            tokens = [token.strip(".,") for token in script.split()]
            units = self._semantic_units(script, tokens)
            slots = [
                NarrationSlot(
                    1,
                    0.0,
                    units[-1].end_time,
                    units[-1].end_time,
                    script,
                    "edge_tts_boundary",
                )
            ]
            specs = [
                {
                    "start_unit": 0,
                    "end_unit_exclusive": len(units),
                    "visual_requirement": "Continuous visible process",
                }
            ]

            with self.subTest(script=script):
                spans = tm.build_semantic_visual_spans_from_specs(
                    script,
                    units,
                    slots,
                    specs,
                )
                self.assertEqual(len(spans), 1)
                self.assertEqual(spans[0].spoken_text, script)
                self.assertEqual(spans[0].start_unit, 0)
                self.assertEqual(spans[0].end_unit_exclusive, len(units))

    def test_one_sentence_can_form_two_true_semantic_visual_spans(self):
        script = "The worker removes damaged boards and installs fresh insulation."
        tokens = [
            "The",
            "worker",
            "removes",
            "damaged",
            "boards",
            "and",
            "installs",
            "fresh",
            "insulation",
        ]
        units = self._semantic_units(script, tokens)
        slots = [
            NarrationSlot(1, 0.0, 4.0, 4.0, script, "edge_tts_boundary")
        ]
        spans = tm.build_semantic_visual_spans_from_specs(
            script,
            units,
            slots,
            [
                {
                    "start_unit": 0,
                    "end_unit_exclusive": 5,
                    "visual_requirement": "Worker removing damaged boards",
                },
                {
                    "start_unit": 5,
                    "end_unit_exclusive": 9,
                    "visual_requirement": "Worker installing fresh insulation",
                },
            ],
        )

        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].spoken_text, "The worker removes damaged boards")
        self.assertEqual(spans[1].spoken_text, "and installs fresh insulation.")
        self.assertEqual(spans[0].end_time, units[4].end_time)
        self.assertEqual(spans[1].start_time, units[5].start_time)

    def test_repeated_words_and_non_visual_clause_keep_exact_unit_coverage(self):
        repeated_script = (
            "Workers pick coffee cherries while other workers sort cherries."
        )
        repeated_tokens = [
            "Workers",
            "pick",
            "coffee",
            "cherries",
            "while",
            "other",
            "workers",
            "sort",
            "cherries",
        ]
        repeated_units = self._semantic_units(repeated_script, repeated_tokens)
        repeated_spans = tm.build_semantic_visual_spans_from_specs(
            repeated_script,
            repeated_units,
            [NarrationSlot(1, 0.0, 4.0, 4.0, repeated_script, "edge_tts_boundary")],
            [
                {
                    "start_unit": 0,
                    "end_unit_exclusive": 4,
                    "visual_requirement": "Workers picking coffee cherries",
                },
                {
                    "start_unit": 4,
                    "end_unit_exclusive": 9,
                    "visual_requirement": "Workers sorting coffee cherries",
                },
            ],
        )
        self.assertEqual(repeated_spans[0].spoken_text, "Workers pick coffee cherries")
        self.assertEqual(
            repeated_spans[1].spoken_text,
            "while other workers sort cherries.",
        )
        self.assertLess(
            repeated_units[3].script_start_char,
            repeated_units[8].script_start_char,
        )

        abstract_script = (
            "The beans dry in the sun, improving flavor before roasting."
        )
        abstract_tokens = [token.strip(".,") for token in abstract_script.split()]
        abstract_units = self._semantic_units(abstract_script, abstract_tokens)
        abstract_spans = tm.build_semantic_visual_spans_from_specs(
            abstract_script,
            abstract_units,
            [NarrationSlot(1, 0.0, 4.0, 4.0, abstract_script, "edge_tts_boundary")],
            [
                {
                    "start_unit": 0,
                    "end_unit_exclusive": 9,
                    "visual_requirement": "Coffee beans drying in sunlight",
                },
                {
                    "start_unit": 9,
                    "end_unit_exclusive": 10,
                    "visual_requirement": "Coffee beans roasting",
                },
            ],
        )
        self.assertIn("improving flavor before", abstract_spans[0].spoken_text)
        self.assertNotEqual(
            abstract_spans[0].visual_requirement.lower(),
            "improving flavor",
        )
        self.assertEqual(abstract_spans[-1].end_unit_exclusive, len(abstract_units))

    def test_semantic_span_validator_rejects_untrusted_range_shapes(self):
        valid_requirement = "Visible action"
        invalid_cases = {
            "gap": [
                {"start_unit": 0, "end_unit_exclusive": 1, "visual_requirement": valid_requirement},
                {"start_unit": 2, "end_unit_exclusive": 4, "visual_requirement": valid_requirement},
            ],
            "overlap": [
                {"start_unit": 0, "end_unit_exclusive": 3, "visual_requirement": valid_requirement},
                {"start_unit": 2, "end_unit_exclusive": 4, "visual_requirement": valid_requirement},
            ],
            "reordered": [
                {"start_unit": 2, "end_unit_exclusive": 4, "visual_requirement": valid_requirement},
                {"start_unit": 0, "end_unit_exclusive": 2, "visual_requirement": valid_requirement},
            ],
            "invalid index": [
                {"start_unit": 0, "end_unit_exclusive": 5, "visual_requirement": valid_requirement}
            ],
            "incomplete final coverage": [
                {"start_unit": 0, "end_unit_exclusive": 3, "visual_requirement": valid_requirement}
            ],
            "boolean index": [
                {"start_unit": False, "end_unit_exclusive": 4, "visual_requirement": valid_requirement}
            ],
            "zero length": [
                {"start_unit": 0, "end_unit_exclusive": 0, "visual_requirement": valid_requirement}
            ],
            "empty requirement": [
                {"start_unit": 0, "end_unit_exclusive": 4, "visual_requirement": " "}
            ],
            "oversized requirement": [
                {
                    "start_unit": 0,
                    "end_unit_exclusive": 4,
                    "visual_requirement": "x" * 241,
                }
            ],
            "fabricated timestamp": [
                {
                    "start_unit": 0,
                    "end_unit_exclusive": 4,
                    "visual_requirement": valid_requirement,
                    "start_time": 9.9,
                }
            ],
            "fabricated unit ids": [
                {
                    "start_unit": 0,
                    "end_unit_exclusive": 4,
                    "visual_requirement": valid_requirement,
                    "unit_ids": [0, 1, 2, 3],
                }
            ],
        }
        for name, specs in invalid_cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                tm.validate_semantic_visual_span_specs(specs, 4)

    def test_semantic_llm_failure_and_missing_units_use_slot_fallback(self):
        script = "Boards are removed. Insulation is installed."
        tokens = ["Boards", "are", "removed", "Insulation", "is", "installed"]
        units = self._semantic_units(script, tokens, [1, 1, 1, 2, 2, 2])
        slots = [
            NarrationSlot(1, 0.0, 1.4, 1.4, "Boards are removed", "edge_tts_boundary"),
            NarrationSlot(2, 1.4, 2.8, 1.4, "Insulation is installed", "edge_tts_boundary"),
        ]

        with patch.object(
            tm.llm,
            "generate_semantic_visual_span_specs",
            return_value=None,
        ), patch.object(
            # Grouping failing is what makes the fallback run; the repair failing
            # too is what makes the fallback's spoken requirements survive. This
            # test is about the shape of that timeline, so both are pinned off.
            tm.llm,
            "generate_narration_visual_requirements",
            return_value=None,
        ):
            fallback = tm.generate_semantic_visual_spans(script, units, slots)

        self.assertEqual(len(fallback), 2)
        self.assertTrue(
            all(span.grouping_source == "narration_slot_fallback" for span in fallback)
        )
        self.assertEqual(
            [(span.start_unit, span.end_unit_exclusive) for span in fallback],
            [(0, 3), (3, 6)],
        )

        with patch.object(
            tm.llm,
            "generate_semantic_visual_span_specs",
        ) as semantic_llm, patch.object(
            tm.llm,
            "generate_narration_visual_requirements",
            return_value=None,
        ):
            no_units = tm.generate_semantic_visual_spans(script, [], slots)

        semantic_llm.assert_not_called()
        self.assertEqual(len(no_units), 2)
        self.assertIsNone(no_units[0].start_unit)
        self.assertEqual(no_units[1].source_narration_slot_indexes, [2])

    def test_coarse_unit_is_indivisible_and_multilingual_units_are_supported(self):
        coarse_script = "Boards are removed. Insulation is installed."
        coarse_units = self._semantic_units(
            coarse_script,
            ["Boards are removed. Insulation is installed"],
            [None],
            boundary_type="SentenceBoundary",
        )
        coarse_slots = [
            NarrationSlot(1, 0.0, 1.0, 1.0, "Boards are removed", "edge_tts_boundary"),
            NarrationSlot(2, 1.0, 2.0, 1.0, "Insulation is installed", "edge_tts_boundary"),
        ]
        with patch.object(
            tm.llm,
            "generate_semantic_visual_span_specs",
            return_value=[
                {"start_unit": 0, "end_unit_exclusive": 1, "visual_requirement": "Removing boards"},
                {"start_unit": 1, "end_unit_exclusive": 2, "visual_requirement": "Installing insulation"},
            ],
        ), patch.object(
            # Two spans over one indivisible unit is invalid, so this reaches the
            # slot-led fallback. Its requirement repair is pinned off: the subject
            # here is unit indivisibility, not what the shot has to show.
            tm.llm,
            "generate_narration_visual_requirements",
            return_value=None,
        ):
            coarse_fallback = tm.generate_semantic_visual_spans(
                coarse_script,
                coarse_units,
                coarse_slots,
            )
        self.assertEqual(len(coarse_fallback), 1)
        self.assertEqual((coarse_fallback[0].start_unit, coarse_fallback[0].end_unit_exclusive), (0, 1))
        self.assertEqual(coarse_fallback[0].source_narration_slot_indexes, [1, 2])

        multilingual_script = "العامل يزيل الألواح.\n工人挑选咖啡豆。"
        multilingual_units = self._semantic_units(
            multilingual_script,
            ["العامل يزيل الألواح", "工人挑选咖啡豆"],
            [1, 2],
            boundary_type="SentenceBoundary",
        )
        multilingual_slots = [
            NarrationSlot(1, 0.0, 0.8, 0.8, "العامل يزيل الألواح", "edge_tts_boundary"),
            NarrationSlot(2, 0.8, 1.5, 0.7, "工人挑选咖啡豆", "edge_tts_boundary"),
        ]
        multilingual_spans = tm.build_semantic_visual_spans_from_specs(
            multilingual_script,
            multilingual_units,
            multilingual_slots,
            [
                {"start_unit": 0, "end_unit_exclusive": 1, "visual_requirement": "عامل يزيل ألواحاً"},
                {"start_unit": 1, "end_unit_exclusive": 2, "visual_requirement": "工人挑选咖啡豆"},
            ],
        )
        self.assertEqual(multilingual_spans[0].spoken_text, "العامل يزيل الألواح.")
        self.assertEqual(multilingual_spans[1].spoken_text, "工人挑选咖啡豆。")

    def _abstract_narration_fixture(self):
        """A narration whose second and third lines show nothing on their own.

        "Patient." and "Not loud." are the shape of the lines that made every
        beat of a recorded run unfillable: they are true sentences of the
        narration and complete nonsense as a description of footage.
        """
        script = (
            "Patient. Rain falls on a dry field. "
            "Not loud. A green shoot breaks the soil."
        )
        tokens = [
            "Patient",
            "Rain",
            "falls",
            "on",
            "a",
            "dry",
            "field",
            "Not",
            "loud",
            "A",
            "green",
            "shoot",
            "breaks",
            "the",
            "soil",
        ]
        units = self._semantic_units(script, tokens, [1] + [2] * 6 + [3] * 2 + [4] * 6)
        slots = [
            NarrationSlot(1, 0.0, 0.6, 0.6, "Patient", "edge_tts_boundary"),
            NarrationSlot(
                2, 0.6, 3.0, 2.4, "Rain falls on a dry field", "edge_tts_boundary"
            ),
            NarrationSlot(3, 3.0, 3.7, 0.7, "Not loud", "edge_tts_boundary"),
            NarrationSlot(
                4, 3.7, 6.4, 2.7, "A green shoot breaks the soil", "edge_tts_boundary"
            ),
        ]
        return script, units, slots

    def test_a_line_with_nothing_to_show_is_absorbed_by_a_filmable_neighbour(self):
        script, units, slots = self._abstract_narration_fixture()
        repaired = {
            1: "",
            2: "Heavy rain falls on cracked dry earth",
            4: "A green seedling pushes up through dark soil",
        }

        with patch.object(
            tm.llm,
            "generate_semantic_visual_span_specs",
            return_value=None,
        ), patch.object(
            tm.llm,
            "generate_narration_visual_requirements",
            return_value=repaired,
        ) as repair:
            spans = tm.generate_semantic_visual_spans(script, units, slots)

        self.assertEqual(repair.call_args.kwargs["narration_text"], script)
        self.assertEqual(
            [line["index"] for line in repair.call_args.kwargs["narration_lines"]],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            repair.call_args.kwargs["narration_lines"][2]["spoken_text"],
            "Not loud.",
        )
        # Line 1 came back empty and line 3 never came back at all. Both mean the
        # same thing -- the line has no visible content of its own -- and both
        # must be absorbed here, for free, instead of buying a search and a round
        # of candidate analysis apiece before the failure ladder gives up.
        self.assertEqual(len(spans), 2)
        self.assertEqual([span.index for span in spans], [1, 2])
        self.assertTrue(
            all(span.grouping_source == "narration_slot_repaired" for span in spans)
        )
        self.assertEqual(
            [span.visual_requirement for span in spans],
            [repaired[2], repaired[4]],
        )
        self.assertEqual(
            [(span.start_unit, span.end_unit_exclusive) for span in spans],
            [(0, 9), (9, 15)],
        )
        # The absorbing span carries the absorbed narration and its slots, so no
        # narration is dropped from the timeline and none of it is searched for.
        self.assertEqual(
            spans[0].spoken_text,
            script[: script.index("A green")].strip(),
        )
        self.assertNotIn("Patient", spans[0].visual_requirement)
        self.assertEqual(spans[0].source_narration_slot_indexes, [1, 2, 3])
        self.assertEqual(spans[1].source_narration_slot_indexes, [4])
        self.assertEqual(spans[0].start_time, units[0].start_time)
        self.assertEqual(spans[0].end_time, units[8].end_time)
        self.assertEqual(spans[1].start_time, units[9].start_time)
        self.assertEqual(spans[1].end_time, units[-1].end_time)

    def test_a_repaired_timeline_is_a_valid_input_for_the_beat_stage(self):
        script, units, slots = self._abstract_narration_fixture()
        repaired = {
            2: "Heavy rain falls on cracked dry earth",
            4: "A green seedling pushes up through dark soil",
        }

        with patch.object(
            tm.llm,
            "generate_semantic_visual_span_specs",
            return_value=None,
        ), patch.object(
            tm.llm,
            "generate_narration_visual_requirements",
            return_value=repaired,
        ):
            spans = tm.generate_semantic_visual_spans(script, units, slots)
        audio_duration = 6.4
        beats = tm.build_visual_beats(script, spans, units, slots, audio_duration)

        # Consolidation rewrote the span indexes and unit ranges, so the beat
        # stage's own validation is the proof that it stayed a legal timeline.
        self.assertTrue(beats)
        self.assertEqual(
            {beat.visual_requirement for beat in beats},
            set(repaired.values()),
        )
        self.assertEqual(beats[0].start_time, 0.0)
        self.assertEqual(beats[-1].end_time, audio_duration)
        for previous, following in zip(beats, beats[1:]):
            self.assertEqual(previous.end_time, following.start_time)

    def test_an_overlong_repaired_requirement_is_treated_as_unfilmable(self):
        script, units, slots = self._abstract_narration_fixture()
        bloated = "A dry field " + "with cracked earth everywhere " * 12
        self.assertGreater(len(bloated), tm._SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS)

        with patch.object(
            tm.llm,
            "generate_semantic_visual_span_specs",
            return_value=None,
        ), patch.object(
            tm.llm,
            "generate_narration_visual_requirements",
            return_value={
                2: bloated,
                4: "A green seedling pushes up through dark soil",
            },
        ):
            spans = tm.generate_semantic_visual_spans(script, units, slots)

        # A requirement over the span limit is carried into the checklist and into
        # every adjudication prompt of its beat, so it is refused here and the
        # line is absorbed like any other line with nothing to show.
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].visual_requirement,
            "A green seedling pushes up through dark soil",
        )
        self.assertEqual((spans[0].start_unit, spans[0].end_unit_exclusive), (0, 15))
        self.assertEqual(spans[0].grouping_source, "narration_slot_repaired")
        self.assertEqual(spans[0].spoken_text, script)
        self.assertEqual(spans[0].source_narration_slot_indexes, [1, 2, 3, 4])

    def test_a_narration_with_nothing_filmable_keeps_its_spoken_requirements(self):
        script, units, slots = self._abstract_narration_fixture()

        with patch.object(
            tm.llm,
            "generate_semantic_visual_span_specs",
            return_value=None,
        ), patch.object(
            tm.llm,
            "generate_narration_visual_requirements",
            return_value={1: "", 2: "   ", 3: "", 4: ""},
        ):
            nothing_visible = tm.generate_semantic_visual_spans(script, units, slots)

        # Absorbing every line into nothing is not a timeline. The spoken
        # requirements survive so the failure stays visible in the provenance
        # instead of looking like a planned set of shots.
        self.assertEqual(len(nothing_visible), 4)
        self.assertTrue(
            all(
                span.grouping_source == "narration_slot_fallback"
                for span in nothing_visible
            )
        )
        self.assertEqual(nothing_visible[0].visual_requirement, "Patient.")
        self.assertTrue(
            tm.semantic_visual_requirements_are_spoken_narration(nothing_visible)
        )

        with patch.object(
            tm.llm,
            "generate_semantic_visual_span_specs",
            return_value=None,
        ), patch.object(
            tm.llm,
            "generate_narration_visual_requirements",
            side_effect=RuntimeError("provider is unreachable"),
        ):
            unavailable = tm.generate_semantic_visual_spans(script, units, slots)

        # A provider outage must read as "repair unavailable", never as "this
        # narration has nothing to show", and it must not fail the run.
        self.assertEqual(
            [span.visual_requirement for span in unavailable],
            [span.visual_requirement for span in nothing_visible],
        )
        self.assertTrue(
            tm.semantic_visual_requirements_are_spoken_narration(unavailable)
        )
        # An empty timeline has no spoken requirement to protect against, and a
        # repaired one describes footage, so neither triggers the guard.
        self.assertFalse(tm.semantic_visual_requirements_are_spoken_narration([]))

    def test_realistic_coffee_semantic_spans_own_exact_text_and_derived_timing(self):
        lines = [
            "Bright red coffee cherries grow on the plant.",
            "Workers pick the ripe cherries by hand.",
            "The beans are spread out to dry in the sun.",
            "Coffee beans roast inside a heated drum.",
        ]
        script = "\n".join(lines)
        token_groups = [
            ["Bright", "red", "coffee", "cherries", "grow", "on", "the", "plant"],
            ["Workers", "pick", "the", "ripe", "cherries", "by", "hand"],
            ["The", "beans", "are", "spread", "out", "to", "dry", "in", "the", "sun"],
            ["Coffee", "beans", "roast", "inside", "a", "heated", "drum"],
        ]
        tokens = [token for group in token_groups for token in group]
        slot_indexes = [
            slot_index
            for slot_index, group in enumerate(token_groups, start=1)
            for _ in group
        ]
        units = self._semantic_units(script, tokens, slot_indexes)
        slots = [
            NarrationSlot(
                index,
                units[sum(len(group) for group in token_groups[: index - 1])].start_time,
                units[sum(len(group) for group in token_groups[:index]) - 1].end_time,
                1.0,
                line,
                "edge_tts_boundary",
            )
            for index, line in enumerate(lines, start=1)
        ]
        boundaries = [0, 8, 15, 25, 32]
        requirements = [
            "Ripe coffee cherries growing on coffee plants",
            "Worker hand-picking ripe coffee cherries",
            "Coffee beans drying in sunlight",
            "Coffee beans roasting inside a heated drum",
        ]
        raw_specs = [
            {
                "start_unit": start,
                "end_unit_exclusive": end,
                "visual_requirement": requirement,
            }
            for start, end, requirement in zip(
                boundaries,
                boundaries[1:],
                requirements,
            )
        ]
        spans = tm.build_semantic_visual_spans_from_specs(
            script,
            units,
            slots,
            raw_specs,
        )

        self.assertEqual([span.source_narration_slot_indexes for span in spans], [[1], [2], [3], [4]])
        self.assertIn("pick the ripe cherries", spans[1].spoken_text)
        self.assertNotIn("dry", spans[1].spoken_text)
        self.assertIn("spread out to dry", spans[2].spoken_text)
        self.assertNotIn("roast", spans[2].spoken_text)
        self.assertIn("roast inside", spans[3].spoken_text)
        for span, start, end in zip(spans, boundaries, boundaries[1:]):
            self.assertEqual(span.start_time, units[start].start_time)
            self.assertEqual(span.end_time, units[end - 1].end_time)
            self.assertEqual(span.grouping_source, "llm")

    def test_visual_beats_follow_realistic_coffee_semantic_timing(self):
        lines = [
            "Coffee cherries grow on the plant.",
            "Workers pick ripe cherries by hand.",
            "Coffee beans dry in the sun.",
            "Coffee beans roast inside a heated drum.",
        ]
        script = "\n".join(lines)
        units = self._timed_units_with_ranges(
            script,
            [
                (lines[0], 0.0, 3.1, 1),
                (lines[1], 3.2, 5.9, 2),
                (lines[2], 6.0, 9.8, 3),
                (lines[3], 10.0, 12.8, 4),
            ],
        )
        requirements = [
            "Coffee cherries growing on coffee plants",
            "Worker hand-picking ripe coffee cherries",
            "Coffee beans drying in sunlight",
            "Coffee beans roasting inside a heated drum",
        ]
        spans = self._semantic_spans_from_ranges(
            script,
            units,
            [
                (index, index + 1, requirement)
                for index, requirement in enumerate(requirements)
            ],
        )

        beats = tm.build_visual_beats(script, spans, units, [], 12.8)

        self.assertEqual(len(beats), 4)
        self.assertEqual(
            [(beat.start_time, beat.end_time) for beat in beats],
            [(0.0, 3.2), (3.2, 6.0), (6.0, 10.0), (10.0, 12.8)],
        )
        self.assertEqual(
            [beat.visual_requirement for beat in beats],
            requirements,
        )
        self.assertEqual([beat.semantic_group_id for beat in beats], [1, 2, 3, 4])
        self.assertEqual(
            [beat.duration_policy for beat in beats],
            ["semantic_original"] * 4,
        )
        self.assertAlmostEqual(sum(beat.duration for beat in beats), 12.8)
        self.assertNotIn(4.0, [beat.end_time for beat in beats])

    def test_visual_beat_queries_use_only_requirements_and_reuse_siblings(self):
        requirements = [
            "Coffee cherries growing on a plant",
            "Coffee beans drying in sunlight",
            "Coffee beans drying in sunlight",
        ]
        beats = [
            VisualBeat(
                index=index,
                semantic_group_id=1 if index == 1 else 2,
                shot_index=1 if index < 3 else 2,
                start_time=float(index - 1),
                end_time=float(index),
                duration=1.0,
                spoken_text=(
                    "Neighbor context mentions picking and roasting, but it must "
                    "not enter the search query."
                ),
                visual_requirement=requirement,
                source_semantic_span_index=1 if index == 1 else 2,
                source_narration_slot_indexes=[index],
                start_unit=index - 1,
                end_unit_exclusive=index,
                timing_source="edge_tts_boundary",
                timing_quality="boundary",
                duration_policy="short_semantic_preserved",
                rapid_cut=True,
            )
            for index, requirement in enumerate(requirements, start=1)
        ]

        def fake_queries(**kwargs):
            self.assertEqual(kwargs["video_subject"], "")
            payload = kwargs["visual_slots"]
            self.assertEqual(len(payload), 2)
            self.assertEqual(
                [item["visual_requirement"] for item in payload],
                requirements[:2],
            )
            self.assertNotIn("Neighbor context", str(payload))
            return {
                1: ["coffee cherries growing"],
                2: ["coffee beans drying sun"],
            }

        with patch.object(
            tm.llm,
            "generate_visual_slot_queries",
            side_effect=fake_queries,
        ) as generate:
            flat_queries = tm.generate_visual_beat_search_queries(beats)

        generate.assert_called_once()
        self.assertEqual(
            flat_queries,
            [
                "coffee cherries growing",
                "coffee beans drying sun",
                "coffee beans drying sun",
            ],
        )
        self.assertEqual(beats[1].search_queries, beats[2].search_queries)
        self.assertIsNot(beats[1].search_queries, beats[2].search_queries)

    def test_beat_queries_request_as_many_phrasings_as_selection_may_try(self):
        # Material selection tries a beat's alternative phrasings on the current
        # provider before it changes catalog. Asking the script stage for exactly
        # one phrasing would leave it nothing to retry with.
        beats = [
            VisualBeat(
                index=index,
                semantic_group_id=1,
                shot_index=index,
                start_time=float(index - 1),
                end_time=float(index),
                duration=1.0,
                spoken_text="A worker digs a hole.",
                visual_requirement="A worker digs a hole.",
                source_semantic_span_index=1,
                source_narration_slot_indexes=[index],
                start_unit=index - 1,
                end_unit_exclusive=index,
                timing_source="edge_tts_boundary",
                timing_quality="boundary",
                duration_policy="semantic_original",
                rapid_cut=True,
            )
            for index in (1, 2)
        ]
        phrasings = ["worker digging hole", "shovel breaking soil", "hands in earth"]

        # The count travels from the real config knob through the real material
        # predicate, so this test fails if either side stops agreeing.
        with (
            patch.dict(tm.config.app, {"smart_material_max_query_variants": 3}),
            patch.object(
                tm.llm,
                "generate_visual_slot_queries",
                return_value={1: phrasings},
            ) as generate,
        ):
            flat_queries = tm.generate_visual_beat_search_queries(
                beats,
                queries_per_beat=tm.material.max_query_variants_per_provider(),
            )

        self.assertEqual(generate.call_args.kwargs["queries_per_slot"], 3)
        # The flat return stays the planned query per beat; the alternates travel
        # on the beats themselves, so they reach the manifest and selection.
        self.assertEqual(flat_queries, [phrasings[0], phrasings[0]])
        self.assertEqual(beats[0].search_queries, phrasings)
        self.assertEqual(beats[1].search_queries, phrasings)

    # The requirement three shots of one split span really shared in task
    # 93b80e04, where all three searched one phrase and bought one clip.
    DRIP_SPAN_REQUIREMENT = (
        "A slow water drip carving a canyon through solid rock over time"
    )

    def _split_span_beats(self, shots=3, group_id=1, policy="long_span_split"):
        spoken = [
            "It starts with a single drop of water.",
            "Over a thousand years that same slow drip carves",
            "solid rock, and nothing about the stone ever fought back.",
        ]
        return [
            VisualBeat(
                index=index,
                semantic_group_id=group_id,
                shot_index=index,
                start_time=float(index - 1),
                end_time=float(index),
                duration=1.0,
                spoken_text=spoken[(index - 1) % len(spoken)],
                visual_requirement=self.DRIP_SPAN_REQUIREMENT,
                source_semantic_span_index=group_id,
                source_narration_slot_indexes=[index],
                start_unit=index - 1,
                end_unit_exclusive=index,
                timing_source="edge_tts_boundary",
                timing_quality="boundary",
                duration_policy=policy,
                rapid_cut=False,
            )
            for index in range(1, shots + 1)
        ]

    def _refine(self, beats, *, split, rescue=None):
        """Run the stage with both of its provider calls under control.

        The rescue call is patched even in the cases that must never reach it,
        because an unpatched call here would bill the live provider from the
        test suite.
        """
        with patch.object(
            tm.llm, "generate_shot_visual_requirements", **split
        ) as split_call, patch.object(
            tm.llm,
            "generate_narration_visual_requirements",
            **({"return_value": {}} if rescue is None else rescue),
        ) as rescue_call:
            refined = tm.refine_split_span_shot_requirements(beats)
        return refined, split_call, rescue_call

    def test_each_shot_of_a_split_span_gets_its_own_requirement_and_query(self):
        beats = self._split_span_beats()
        answers = {
            1: "A single drop of water falling into stillness",
            2: "Water tracing a groove down a rock face",
            3: "A deep canyon cut into bare stone",
        }

        refined, split, rescue = self._refine(beats, split={"return_value": answers})

        self.assertEqual(refined, 3)
        # No shot was left on the parent, so the rescue must not be paid for.
        rescue.assert_not_called()
        self.assertEqual([beat.visual_requirement for beat in beats], list(
            answers.values()
        ))
        # Siblings are answered in one request, because the requirement being
        # divided is theirs jointly and they have to come back different.
        sent = split.call_args.args[0]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["span_requirement"], self.DRIP_SPAN_REQUIREMENT)
        self.assertEqual(
            [shot["shot_id"] for shot in sent[0]["shots"]], [1, 2, 3]
        )
        self.assertEqual(
            sent[0]["shots"][0]["spoken_text"],
            "It starts with a single drop of water.",
        )

        # The payoff: beat queries deduplicate on group plus requirement, so
        # before this stage these three shots shared one search and one clip.
        with patch.object(
            tm.llm,
            "generate_visual_slot_queries",
            return_value={1: ["drop of water"], 2: ["water on rock"], 3: ["canyon"]},
        ) as generate:
            flat_queries = tm.generate_visual_beat_search_queries(beats)

        self.assertEqual(len(generate.call_args.kwargs["visual_slots"]), 3)
        self.assertEqual(flat_queries, ["drop of water", "water on rock", "canyon"])

    def test_a_repeated_sibling_answer_keeps_the_parent_requirement(self):
        beats = self._split_span_beats()
        answers = {
            1: "A single drop of water",
            2: "a  SINGLE   drop of water",
            3: "A deep canyon cut into bare stone",
        }

        refined, _, rescue = self._refine(beats, split={"return_value": answers})

        # Identical siblings are the pathology being removed, so the repeat is
        # dropped rather than adopted — and the parent is still a distinct
        # string, so this shot is searched separately anyway.
        self.assertEqual(refined, 2)
        self.assertEqual(beats[1].visual_requirement, self.DRIP_SPAN_REQUIREMENT)
        self.assertEqual(
            len({beat.visual_requirement for beat in beats}), 3
        )
        # Still on the parent still means unfillable, so that one shot — and
        # only that one — is offered to the rescue.
        self.assertEqual([line["index"] for line in rescue.call_args.args[1]], [2])

    def test_an_over_long_shot_requirement_keeps_the_parent(self):
        beats = self._split_span_beats()
        over_long = "z" * (tm._SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS + 1)

        refined, _, rescue = self._refine(
            beats,
            split={"return_value": {1: over_long, 2: "Water tracing a groove", 3: ""}},
        )

        # A bloated requirement is carried into the checklist and into every
        # adjudication prompt of that beat, so it is not adopted.
        self.assertEqual(refined, 1)
        self.assertEqual(beats[0].visual_requirement, self.DRIP_SPAN_REQUIREMENT)
        self.assertEqual(beats[1].visual_requirement, "Water tracing a groove")
        self.assertEqual(beats[2].visual_requirement, self.DRIP_SPAN_REQUIREMENT)
        # The discarded answer and the deliberately empty one are both still
        # unfillable shots, so both are offered to the rescue.
        self.assertEqual([line["index"] for line in rescue.call_args.args[1]], [1, 3])

    def test_a_failed_split_leaves_every_shot_on_its_span_requirement(self):
        # An unavailable provider is not asked a second question, because that
        # signal means no batch parsed at all. A provider that answered and
        # narrowed nothing is asked, because that says the parent was hard to
        # divide, not that the provider is unusable.
        for label, kwargs, offered in (
            ("unavailable", {"return_value": None}, []),
            ("nothing to narrow", {"return_value": {}}, [1, 2, 3]),
            ("raised", {"side_effect": RuntimeError("provider exploded")}, []),
        ):
            with self.subTest(outcome=label):
                beats = self._split_span_beats()

                refined, _, rescue = self._refine(beats, split=kwargs)

                # Failing open costs nothing: the shots behave exactly as they
                # did before this stage existed.
                self.assertEqual(refined, 0)
                self.assertEqual(
                    [beat.visual_requirement for beat in beats],
                    [self.DRIP_SPAN_REQUIREMENT] * 3,
                )
                self.assertEqual(
                    [
                        line["index"]
                        for call in rescue.call_args_list
                        for line in call.args[1]
                    ],
                    offered,
                )

    def test_a_shot_the_split_could_not_narrow_is_described_from_its_own_line(self):
        beats = self._split_span_beats()

        refined, _, rescue = self._refine(
            beats,
            split={"return_value": {2: "Water tracing a groove down a rock face"}},
            rescue={
                "return_value": {
                    1: "A single drop of water falling",
                    3: "A canyon cut into bare stone",
                }
            },
        )

        # Inheriting a multi-event parent is what made beats 4 and 10 of task
        # 3f0f2b07 unfillable and sent one of them into a rewrite that dropped
        # the action entirely, so a skipped shot is described from its own
        # spoken line rather than left on the parent.
        self.assertEqual(refined, 3)
        self.assertEqual(
            [beat.visual_requirement for beat in beats],
            [
                "A single drop of water falling",
                "Water tracing a groove down a rock face",
                "A canyon cut into bare stone",
            ],
        )
        context, lines = rescue.call_args.args
        self.assertEqual([line["index"] for line in lines], [1, 3])
        # The whole span's words are the read-only context, because a shot's own
        # line is often a fragment that cannot be resolved on its own.
        self.assertIn("It starts with a single drop of water.", context)
        self.assertIn("nothing about the stone ever fought back", context)

    def test_a_rescued_requirement_is_held_to_the_same_rules_as_a_split_one(self):
        sibling = "Water tracing a groove down a rock face"
        cases = {
            "the parent verbatim": self.DRIP_SPAN_REQUIREMENT,
            "a sibling's refined text": sibling,
            "over the span character limit": "z"
            * (tm._SEMANTIC_VISUAL_REQUIREMENT_MAX_CHARS + 1),
            "an empty answer": "",
        }
        for label, answer in cases.items():
            with self.subTest(rescue=label):
                beats = self._split_span_beats(shots=2)

                refined, _, _ = self._refine(
                    beats,
                    split={"return_value": {2: sibling}},
                    rescue={"return_value": {1: answer}},
                )

                # A rescue that would collapse two shots back into one search,
                # or bloat every adjudication prompt of this beat, is no better
                # than the parent it would replace.
                self.assertEqual(refined, 1)
                self.assertEqual(
                    beats[0].visual_requirement, self.DRIP_SPAN_REQUIREMENT
                )
                self.assertEqual(beats[1].visual_requirement, sibling)

    def test_only_a_split_span_with_more_than_one_shot_is_ever_sent(self):
        cases = {
            "one shot": self._split_span_beats(shots=1),
            "not split": self._split_span_beats(policy="semantic_original"),
            "no beats": [],
        }
        for label, beats in cases.items():
            with self.subTest(timeline=label):
                refined, split, rescue = self._refine(beats, split={})

                split.assert_not_called()
                rescue.assert_not_called()
                self.assertEqual(refined, 0)

    def test_two_split_spans_are_divided_without_crossing_each_other(self):
        beats = self._split_span_beats(shots=2, group_id=1)
        second = self._split_span_beats(shots=2, group_id=2)
        for offset, beat in enumerate(second, start=3):
            beat.index = offset
            beat.visual_requirement = "Snow gathering on a slope until it releases"
        beats.extend(second)

        refined, split, rescue = self._refine(
            beats,
            split={
                "return_value": {
                    1: "A drop falling",
                    2: "A groove in rock",
                    3: "Snow settling flake by flake",
                    4: "A slope releasing in an avalanche",
                }
            },
        )

        self.assertEqual(refined, 4)
        rescue.assert_not_called()
        sent = split.call_args.args[0]
        self.assertEqual(
            [[shot["shot_id"] for shot in span["shots"]] for span in sent],
            [[1, 2], [3, 4]],
        )
        # Each span carries its own parent, so a shot can never be narrowed
        # against a requirement that belongs to a different part of the script.
        self.assertEqual(
            [span["span_requirement"] for span in sent],
            [
                self.DRIP_SPAN_REQUIREMENT,
                "Snow gathering on a slope until it releases",
            ],
        )

    def test_visual_beats_assign_initial_interspan_and_trailing_silence(self):
        script = "Cherries grow. Workers sort beans."
        units = self._timed_units_with_ranges(
            script,
            [
                ("Cherries grow", 0.3, 2.0, 1),
                ("Workers sort beans", 2.3, 9.7, 2),
            ],
        )
        spans = self._semantic_spans_from_ranges(
            script,
            units,
            [
                (0, 1, "Coffee cherries growing"),
                (1, 2, "Workers sorting coffee beans"),
            ],
        )

        beats = tm.build_visual_beats(script, spans, units, [], 10.2)

        self.assertEqual(len(beats), 2)
        self.assertEqual((beats[0].start_time, beats[0].end_time), (0.0, 2.3))
        self.assertEqual((beats[1].start_time, beats[1].end_time), (2.3, 10.2))
        self.assertEqual(beats[0].end_time, beats[1].start_time)
        self.assertAlmostEqual(sum(beat.duration for beat in beats), 10.2)

    def test_visual_beats_preserve_short_and_very_short_distinct_concepts(self):
        script = "A match ignites. A door slams. Smoke fills the room."
        units = self._timed_units_with_ranges(
            script,
            [
                ("A match ignites", 0.0, 1.7, 1),
                ("A door slams", 1.7, 2.5, 2),
                ("Smoke fills the room", 2.5, 5.0, 3),
            ],
        )
        requirements = [
            "A match igniting",
            "A door slamming shut",
            "Smoke filling a room",
        ]
        spans = self._semantic_spans_from_ranges(
            script,
            units,
            [
                (index, index + 1, requirement)
                for index, requirement in enumerate(requirements)
            ],
        )

        beats = tm.build_visual_beats(script, spans, units, [], 5.0)

        self.assertEqual(len(beats), 3)
        self.assertEqual([beat.visual_requirement for beat in beats], requirements)
        self.assertEqual(beats[0].duration_policy, "short_semantic_preserved")
        self.assertFalse(beats[0].rapid_cut)
        self.assertEqual(beats[1].duration_policy, "short_semantic_preserved")
        self.assertTrue(beats[1].rapid_cut)
        self.assertEqual(beats[1].semantic_group_id, 2)

    def test_visual_beats_split_long_concepts_at_balanced_unit_boundaries(self):
        cases = (
            (
                8.0,
                [0.0, 2.0, 4.0, 6.0],
                [1.8, 3.8, 5.8, 8.0],
                [4.0],
            ),
            (
                11.4,
                [0.0, 1.9, 3.8, 5.7, 7.6, 9.5],
                [1.7, 3.6, 5.5, 7.4, 9.3, 11.4],
                [3.8, 7.6],
            ),
        )
        tokens = ["Coffee", "beans", "keep", "roasting", "inside", "drum"]
        for audio_duration, starts, ends, expected_cuts in cases:
            selected_tokens = tokens[: len(starts)]
            script = " ".join(selected_tokens) + "."
            units = self._timed_units_with_ranges(
                script,
                [
                    (token, start, end, 1)
                    for token, start, end in zip(selected_tokens, starts, ends)
                ],
            )
            spans = self._semantic_spans_from_ranges(
                script,
                units,
                [(0, len(units), "Coffee beans roasting inside a roaster")],
            )

            with self.subTest(audio_duration=audio_duration):
                beats = tm.build_visual_beats(
                    script,
                    spans,
                    units,
                    [],
                    audio_duration,
                )
                self.assertEqual(
                    [beat.end_time for beat in beats[:-1]],
                    expected_cuts,
                )
                self.assertEqual(
                    [beat.start_time for beat in beats[1:]],
                    expected_cuts,
                )
                self.assertTrue(
                    all(
                        beat.duration_policy == "long_span_split"
                        for beat in beats
                    )
                )
                self.assertEqual(
                    {beat.semantic_group_id for beat in beats},
                    {1},
                )
                self.assertEqual(
                    {beat.visual_requirement for beat in beats},
                    {"Coffee beans roasting inside a roaster"},
                )
                valid_boundaries = {unit.start_time for unit in units[1:]}
                self.assertTrue(set(expected_cuts).issubset(valid_boundaries))

    def test_visual_beats_keep_multiple_concepts_in_one_sentence_separate(self):
        script = "The mechanic removes the wheel and installs a tire before lowering the car."
        units = self._timed_units_with_ranges(
            script,
            [
                ("The mechanic removes the wheel", 0.0, 2.0, 1),
                ("and installs a tire", 2.0, 4.0, 1),
                ("before lowering the car", 4.0, 6.0, 1),
            ],
        )
        requirements = [
            "Mechanic removing a damaged wheel",
            "Mechanic installing a new tire",
            "Mechanic lowering the car",
        ]
        spans = self._semantic_spans_from_ranges(
            script,
            units,
            [
                (index, index + 1, requirement)
                for index, requirement in enumerate(requirements)
            ],
        )

        beats = tm.build_visual_beats(script, spans, units, [], 6.0)

        self.assertEqual(len(beats), 3)
        self.assertEqual([beat.visual_requirement for beat in beats], requirements)
        self.assertEqual([beat.semantic_group_id for beat in beats], [1, 2, 3])

    def test_visual_beats_keep_one_multisentence_semantic_identity_when_split(self):
        lines = [
            "The beans enter the roaster.",
            "They tumble continuously.",
            "They slowly turn brown.",
        ]
        script = "\n".join(lines)
        units = self._timed_units_with_ranges(
            script,
            [
                (lines[0], 0.0, 2.2, 1),
                (lines[1], 2.3, 4.5, 2),
                (lines[2], 4.6, 7.0, 3),
            ],
        )
        requirement = "Coffee beans roasting inside a roaster"
        spans = self._semantic_spans_from_ranges(
            script,
            units,
            [(0, 3, requirement)],
        )

        beats = tm.build_visual_beats(script, spans, units, [], 7.0)

        self.assertEqual(len(beats), 2)
        self.assertEqual({beat.semantic_group_id for beat in beats}, {1})
        self.assertEqual({beat.visual_requirement for beat in beats}, {requirement})
        self.assertEqual([beat.shot_index for beat in beats], [1, 2])
        self.assertIn("enter the roaster", beats[0].spoken_text)
        self.assertIn("turn brown", beats[1].spoken_text)

    def test_visual_beat_timing_quality_is_conservative_and_deterministic(self):
        script = "Coffee beans roast inside the drum."
        units = self._timed_units_with_ranges(
            script,
            [
                ("Coffee beans", 0.0, 1.8, 1),
                ("roast inside", 2.0, 3.8, 1),
                ("the drum", 4.0, 8.0, 1),
            ],
        )
        units[1].timing_source = "estimated"
        units[1].timing_quality = "estimated"
        spans = self._semantic_spans_from_ranges(
            script,
            units,
            [(0, 3, "Coffee beans roasting inside a drum")],
        )
        spans[0].timing_source = "estimated"
        spans[0].timing_quality = "estimated"

        first = tm.build_visual_beats(script, spans, units, [], 8.0)
        second = tm.build_visual_beats(script, spans, units, [], 8.0)

        self.assertEqual(first, second)
        self.assertTrue(all(beat.timing_quality == "estimated" for beat in first))
        self.assertTrue(all(beat.timing_source == "estimated" for beat in first))

    def test_visual_beats_use_narration_fallback_then_leave_legacy_available(self):
        script = "Boards are removed. Insulation is installed."
        units = self._timed_units_with_ranges(
            script,
            [
                ("Boards are removed", 0.0, 2.0, 1),
                ("Insulation is installed", 2.0, 4.0, 2),
            ],
        )
        slots = [
            NarrationSlot(1, 0.0, 2.0, 2.0, "Boards are removed", "edge_tts_boundary"),
            NarrationSlot(2, 2.0, 4.0, 2.0, "Insulation is installed", "edge_tts_boundary"),
        ]

        fallback = tm.build_visual_beats(script, None, units, slots, 4.0)
        invalid_semantic = SemanticVisualSpan(
            index=2,
            start_unit=0,
            end_unit_exclusive=2,
            spoken_text=script,
            visual_requirement="Invalid reordered semantic metadata",
            source_narration_slot_indexes=[1, 2],
            start_time=0.0,
            end_time=4.0,
            timing_source="edge_tts_boundary",
            timing_quality="boundary",
            grouping_source="llm",
        )
        invalid_fallback = tm.build_visual_beats(
            script,
            [invalid_semantic],
            units,
            slots,
            4.0,
        )
        legacy = tm.build_visual_beats(script, None, [], [], 4.0)

        self.assertEqual(len(fallback), 2)
        self.assertTrue(
            all(beat.source_semantic_span_index is None for beat in fallback)
        )
        self.assertEqual([beat.semantic_group_id for beat in fallback], [1, 2])
        self.assertEqual(
            [beat.visual_requirement for beat in invalid_fallback],
            [beat.visual_requirement for beat in fallback],
        )
        self.assertEqual(legacy, [])

    def test_visual_beats_persist_additively_and_old_manifests_stay_readable(self):
        beat = VisualBeat(
            index=1,
            semantic_group_id=1,
            shot_index=1,
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
            spoken_text="Coffee beans roast.",
            visual_requirement="Coffee beans roasting",
            source_semantic_span_index=1,
            source_narration_slot_indexes=[1],
            start_unit=0,
            end_unit_exclusive=1,
            timing_source="edge_tts_boundary",
            timing_quality="boundary",
            duration_policy="semantic_original",
            rapid_cut=False,
            search_queries=["coffee beans roasting"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            tm.utils,
            "task_dir",
            return_value=tmp_dir,
        ):
            old_payload = {"script": "Coffee beans roast.", "visual_slots": []}
            tm.task_artifacts.write_script_data("beats", old_payload)
            self.assertEqual(
                tm.task_artifacts.read_script_data("beats"),
                old_payload,
            )
            tm.persist_narration_timeline(
                task_id="beats",
                narration_slots=[],
                visual_slots=[],
                video_terms=[],
                visual_beats=[beat],
            )
            persisted = tm.task_artifacts.read_script_data("beats")

        self.assertEqual(persisted["timeline_schema_version"], 2)
        self.assertEqual(persisted["visual_beats"][0]["semantic_group_id"], 1)
        self.assertEqual(
            persisted["visual_beats"][0]["duration_policy"],
            "semantic_original",
        )
        self.assertEqual(
            persisted["visual_beats"][0]["search_queries"],
            ["coffee beans roasting"],
        )
        self.assertNotIn("api_key", persisted["visual_beats"][0])

    def test_semantic_visual_spans_persist_round_trip_without_raw_llm_data(self):
        semantic_span = SemanticVisualSpan(
            index=1,
            start_unit=0,
            end_unit_exclusive=1,
            spoken_text="Coffee beans roast.",
            visual_requirement="Coffee beans roasting",
            source_narration_slot_indexes=[1],
            start_time=0.2,
            end_time=1.1,
            timing_source="edge_tts_boundary",
            timing_quality="boundary",
            grouping_source="llm",
        )
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            tm.utils,
            "task_dir",
            return_value=tmp_dir,
        ):
            tm.task_artifacts.write_script_data("semantic", {"script": "Coffee beans roast."})
            tm.persist_narration_timeline(
                task_id="semantic",
                narration_slots=[],
                visual_slots=[],
                video_terms=[],
                semantic_visual_spans=[semantic_span],
            )
            persisted = tm.task_artifacts.read_script_data("semantic")

        self.assertEqual(persisted["timeline_schema_version"], 2)
        self.assertEqual(persisted["semantic_visual_spans"][0]["start_unit"], 0)
        self.assertEqual(
            persisted["semantic_visual_spans"][0]["spoken_text"],
            "Coffee beans roast.",
        )
        self.assertNotIn("raw_response", persisted["semantic_visual_spans"][0])

    def test_start_stops_before_materials_when_term_provider_fails(self):
        """
        关键词 Provider 失败后，任务必须立即结束，不能继续生成音频或下载素材。

        这里从任务入口覆盖完整的错误传播路径，避免未来只修服务层返回类型，
        却又在任务编排层把空列表转换成其它真值后继续执行外部请求。
        """
        params = VideoParams(
            video_subject="startup story",
            video_script="A short startup story.",
        )
        state = MemoryState()

        with (
            patch.object(
                tm.llm,
                "_generate_response",
                return_value="Error: invalid API key",
            ),
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm, "get_video_materials") as get_video_materials,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("term-provider-error", params)

        generate_audio.assert_not_called()
        get_video_materials.assert_not_called()
        failed_task = state.get_task("term-provider-error")
        self.assertEqual(result, failed_task)
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "terms")
        self.assertTrue(failed_task["error"])

    def test_generate_audio_uses_custom_file_inside_task_directory(self):
        task_id = "test-custom-audio-safe"
        task_dir = utils.task_dir(task_id)
        custom_audio_file = os.path.join(task_dir, "custom-audio.mp3")
        with open(custom_audio_file, "wb") as audio:
            audio.write(b"fake audio")

        params = VideoParams(
            video_subject="custom audio",
            video_script="",
            custom_audio_file=custom_audio_file,
            voice_name="test-voice",
        )

        try:
            with (
                patch.object(tm.voice, "tts") as tts,
                patch.object(tm.voice, "get_audio_duration", return_value=7),
            ):
                audio_file, audio_duration, sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(audio_file, os.path.realpath(custom_audio_file))
        self.assertEqual(audio_duration, 7)
        self.assertIsNone(sub_maker)
        tts.assert_not_called()

    def test_generate_audio_accepts_server_side_custom_file(self):
        task_id = "test-custom-audio-server-side"
        task_dir = utils.task_dir(task_id)

        with tempfile.NamedTemporaryFile(suffix=".mp3") as server_audio:
            server_audio.write(b"fake audio")
            server_audio.flush()
            params = VideoParams(
                video_subject="custom audio",
                video_script="",
                custom_audio_file=server_audio.name,
                voice_name="test-voice",
            )

            try:
                with (
                    patch.object(tm.voice, "tts") as tts,
                    patch.object(tm.voice, "get_audio_duration", return_value=6),
                ):
                    audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                        task_id, params, "script"
                    )
            finally:
                shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(audio_file, os.path.realpath(server_audio.name))
        self.assertEqual(audio_duration, 6)
        self.assertIsNone(result_sub_maker)
        tts.assert_not_called()

    def test_generate_audio_rejects_missing_custom_file_without_tts(self):
        task_id = "test-custom-audio-missing"
        task_dir = utils.task_dir(task_id)
        missing_audio_file = os.path.join(task_dir, "missing.mp3")
        params = VideoParams(
            video_subject="custom audio",
            video_script="",
            custom_audio_file=missing_audio_file,
            voice_name="test-voice",
        )
        state = MemoryState()

        try:
            with (
                patch.object(tm.voice, "tts") as tts,
                patch.object(tm.sm, "state", state),
            ):
                audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertIsNone(audio_file)
        self.assertIsNone(audio_duration)
        self.assertIsNone(result_sub_maker)
        tts.assert_not_called()
        failed_task = state.get_task(task_id)
        self.assertEqual(failed_task["failed_stage"], "audio")
        self.assertIn("does not exist", failed_task["error"])

    def test_generate_audio_exposes_classified_tts_voice_error(self):
        task_id = "test-classified-tts-error"
        params = VideoParams(
            video_subject="voice failure",
            voice_name="invalid-edge-voice",
        )
        state = MemoryState()
        safe_message = (
            "Edge TTS voice problem; verify the selected voice and its language"
        )

        try:
            with (
                patch.object(
                    tm.voice,
                    "tts",
                    side_effect=tm.voice.TTSServiceError("voice", safe_message),
                ),
                patch.object(tm.sm, "state", state),
            ):
                result = tm.generate_audio(task_id, params, "script")
        finally:
            shutil.rmtree(utils.task_dir(task_id), ignore_errors=True)

        self.assertEqual(result, (None, None, None))
        failed_task = state.get_task(task_id)
        self.assertEqual(failed_task["failed_stage"], "audio")
        self.assertEqual(failed_task["error"], safe_message)

    def test_generate_subtitle_uses_whisper_for_custom_audio_without_sub_maker(self):
        """
        自定义音频不会经过 TTS，所以没有 sub_maker。
        Whisper 可以直接从音频文件转写，此时不能被 sub_maker 为空的保护逻辑提前跳过。
        """
        task_id = "test-custom-audio-whisper-subtitle"
        task_dir = utils.task_dir(task_id)
        audio_file = os.path.join(task_dir, "custom-audio.mp3")
        Path(audio_file).write_bytes(b"fake audio")
        params = VideoParams(
            video_subject="custom audio",
            video_script="Hello world.",
            subtitle_enabled=True,
        )

        def fake_whisper_create(audio_file, subtitle_file):
            Path(subtitle_file).write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n\n",
                encoding="utf-8",
            )

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="whisper"),
                ),
                patch.object(
                    tm.subtitle, "create", side_effect=fake_whisper_create
                ) as create,
                patch.object(tm.subtitle, "correct") as correct,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Hello world.",
                    sub_maker=None,
                    audio_file=audio_file,
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertTrue(subtitle_path.endswith("subtitle.srt"))
        create.assert_called_once_with(
            audio_file=audio_file, subtitle_file=subtitle_path
        )
        correct.assert_called_once_with(
            subtitle_file=subtitle_path, video_script="Hello world."
        )

    def test_generate_subtitle_skips_edge_provider_without_sub_maker(self):
        """
        Edge 字幕依赖 TTS 返回的 sub_maker 时间轴。
        自定义音频缺少该对象时应继续跳过，避免产生不可信的字幕时间轴。
        """
        task_id = "test-custom-audio-edge-no-submaker"
        task_dir = utils.task_dir(task_id)
        audio_file = os.path.join(task_dir, "custom-audio.mp3")
        Path(audio_file).write_bytes(b"fake audio")
        params = VideoParams(
            video_subject="custom audio",
            video_script="Hello world.",
            subtitle_enabled=True,
        )

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="edge"),
                ),
                patch.object(tm.voice, "create_subtitle") as create_subtitle,
                patch.object(tm.subtitle, "create") as whisper_create,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Hello world.",
                    sub_maker=None,
                    audio_file=audio_file,
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(subtitle_path, "")
        create_subtitle.assert_not_called()
        whisper_create.assert_not_called()

    def test_ordered_matching_builds_internal_timeline_when_subtitles_are_hidden(self):
        task_id = "test-hidden-subtitle-timeline"
        task_dir = utils.task_dir(task_id)
        params = VideoParams(
            video_subject="hidden subtitles",
            video_script="Sentence A.",
            subtitle_enabled=False,
            match_materials_to_script=True,
        )
        sub_maker = object()

        def fake_create_subtitle(text, sub_maker, subtitle_file):
            Path(subtitle_file).write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nSentence A\n\n",
                encoding="utf-8",
            )

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="edge"),
                ),
                patch.object(
                    tm.voice,
                    "create_subtitle",
                    side_effect=fake_create_subtitle,
                ) as create_subtitle,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Sentence A.",
                    sub_maker=sub_maker,
                    audio_file="audio.mp3",
                    force_timeline=params.match_materials_to_script,
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertTrue(subtitle_path.endswith("subtitle.srt"))
        create_subtitle.assert_called_once()

    def test_generate_subtitle_does_not_fallback_to_whisper_when_edge_fails(self):
        """
        Edge 没有生成字幕文件时应保留无字幕结果，不能自动下载 Whisper 模型。

        该场景可能由 TTS 时间轴与原始文案无法匹配触发。自动回退会让未选择
        Whisper 的用户意外下载数 GB 模型，因此必须验证 Whisper 完全不会被调用。
        """
        task_id = "test-edge-subtitle-without-output"
        task_dir = utils.task_dir(task_id)
        params = VideoParams(
            video_subject="edge subtitle",
            video_script="Hello world.",
            subtitle_enabled=True,
        )
        sub_maker = object()

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="edge"),
                ),
                patch.object(tm.voice, "create_subtitle") as create_subtitle,
                patch.object(tm.subtitle, "create") as whisper_create,
                patch.object(tm.subtitle, "correct") as whisper_correct,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Hello world.",
                    sub_maker=sub_maker,
                    audio_file=os.path.join(task_dir, "audio.mp3"),
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(subtitle_path, "")
        create_subtitle.assert_called_once()
        whisper_create.assert_not_called()
        whisper_correct.assert_not_called()

    def test_non_ordered_terms_path_keeps_existing_behavior(self):
        params = VideoParams(
            video_subject="Coffee",
            video_script="Coffee beans are roasted.",
            match_materials_to_script=False,
        )

        with (
            patch.object(tm, "generate_script", return_value=params.video_script),
            patch.object(tm, "generate_terms", return_value=["coffee beans"]) as terms,
            patch.object(tm, "save_script_data"),
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm.llm, "generate_visual_slot_queries") as slot_queries,
            patch.object(tm.sm.state, "update_task"),
        ):
            result = tm.start("non-ordered-terms", params, stop_at="terms")

        self.assertEqual(result["terms"], ["coffee beans"])
        terms.assert_called_once_with("non-ordered-terms", params, params.video_script)
        generate_audio.assert_not_called()
        slot_queries.assert_not_called()

    def test_ordered_pipeline_generates_queries_after_internal_timeline(self):
        params = VideoParams(
            video_subject="Railway",
            video_script="Workers inspect railway tracks.",
            subtitle_enabled=False,
            match_materials_to_script=True,
            video_clip_duration=4,
        )
        narration_slots = [
            NarrationSlot(
                1,
                0.0,
                4.0,
                4.0,
                "Workers inspect railway tracks",
                "edge_tts_boundary",
            )
        ]
        visual_slots = [
            VisualSlot(
                index=1,
                start_time=0.0,
                end_time=4.0,
                duration=4.0,
                narration_slot_indexes=[1],
                narration_text="Workers inspect railway tracks",
                primary_narration_slot_index=1,
                primary_narration_text="Workers inspect railway tracks",
                visual_requirement="Workers inspect railway tracks",
                narration_overlaps=[NarrationOverlap(1, 0.0, 4.0, 4.0)],
                search_queries=[],
                timing_source="edge_tts_boundary",
                timing_quality="boundary",
            )
        ]
        timed_units = [
            TimedNarrationUnit(
                index=1,
                text="Workers inspect railway tracks",
                start_time=0.2,
                end_time=3.8,
                duration=3.6,
                timing_source="edge_tts_boundary",
                timing_quality="boundary",
                source_boundary_type="SentenceBoundary",
                script_start_char=0,
                script_end_char=30,
            )
        ]
        events = []

        def fake_audio(*args, **kwargs):
            events.append("audio")
            return "audio.mp3", 4, object()

        def fake_subtitle(*args, **kwargs):
            events.append("subtitle")
            return "subtitle.srt"

        def fake_narration(*args, **kwargs):
            events.append("narration_slots")
            return narration_slots

        def fake_visual(*args, **kwargs):
            events.append("visual_slots")
            return visual_slots

        def fake_queries(*args, **kwargs):
            events.append("slot_queries")
            visual_slots[0].search_queries = ["workers inspecting railway tracks"]
            return ["workers inspecting railway tracks"]

        def fake_beat_queries(beats, queries_per_beat=1):
            events.append("beat_queries")
            beats[0].search_queries = ["workers inspecting railway tracks"]
            return ["workers inspecting railway tracks"]

        with (
            patch.object(tm, "generate_script", return_value=params.video_script),
            patch.object(tm, "generate_terms") as legacy_terms,
            # This test is about stage order, not about the checklist. Without this
            # patch the checklist gate reads the real config and the script stage
            # sends a live decomposition request to the configured LLM provider.
            patch.object(
                tm.twelvelabs, "is_smart_visual_matching_enabled", return_value=False
            ),
            patch.object(tm, "save_script_data"),
            patch.object(tm, "generate_audio", side_effect=fake_audio),
            patch.object(
                tm, "generate_subtitle", side_effect=fake_subtitle
            ) as subtitles,
            patch.object(tm.voice, "get_audio_duration", return_value=4.0),
            patch.object(
                tm.voice,
                "extract_timed_narration_units",
                return_value=timed_units,
            ) as extract_timing,
            patch.object(
                tm.llm,
                "generate_semantic_visual_span_specs",
                return_value=[
                    {
                        "start_unit": 0,
                        "end_unit_exclusive": 1,
                        "visual_requirement": "Workers inspecting railway tracks",
                    }
                ],
            ) as semantic_grouping,
            patch.object(tm, "build_narration_slots", side_effect=fake_narration),
            patch.object(tm, "build_visual_slots", side_effect=fake_visual),
            patch.object(
                tm,
                "generate_visual_beat_search_queries",
                side_effect=fake_beat_queries,
            ),
            patch.object(
                tm,
                "generate_visual_slot_search_queries",
                side_effect=fake_queries,
            ),
            patch.object(tm, "persist_narration_timeline") as persist_timeline,
            patch.object(tm.sm.state, "update_task"),
        ):
            result = tm.start("ordered-timeline", params, stop_at="terms")

        self.assertEqual(
            events,
            [
                "audio",
                "subtitle",
                "narration_slots",
                "visual_slots",
                "beat_queries",
                "slot_queries",
            ],
        )
        self.assertEqual(result["terms"], ["workers inspecting railway tracks"])
        legacy_terms.assert_not_called()
        self.assertTrue(subtitles.call_args.kwargs["force_timeline"])
        extract_timing.assert_called_once()
        semantic_grouping.assert_called_once()
        persisted_units = persist_timeline.call_args.kwargs["timed_narration_units"]
        self.assertIs(persisted_units[0], timed_units[0])
        self.assertEqual(persisted_units[0].source_narration_slot_index, 1)
        persisted_spans = persist_timeline.call_args.kwargs["semantic_visual_spans"]
        self.assertEqual(persisted_spans[0].spoken_text, params.video_script)
        self.assertEqual(persisted_spans[0].visual_requirement, "Workers inspecting railway tracks")
        persisted_beats = persist_timeline.call_args.kwargs["visual_beats"]
        self.assertEqual(len(persisted_beats), 1)
        self.assertEqual((persisted_beats[0].start_time, persisted_beats[0].end_time), (0.0, 4.0))
        self.assertEqual(
            persisted_beats[0].visual_requirement,
            "Workers inspecting railway tracks",
        )
        self.assertEqual(
            persisted_beats[0].search_queries,
            ["workers inspecting railway tracks"],
        )

    def test_smart_material_failure_is_exposed_as_material_stage_error(self):
        params = VideoParams(
            video_subject="Railway",
            video_script="Workers inspect railway tracks.",
            subtitle_enabled=False,
            match_materials_to_script=True,
            video_clip_duration=4,
        )
        narration_slots = [
            NarrationSlot(
                1,
                0.0,
                4.0,
                4.0,
                "Workers inspect railway tracks",
                "edge_tts_boundary",
            )
        ]
        visual_slots = [
            VisualSlot(
                index=1,
                start_time=0.0,
                end_time=4.0,
                duration=4.0,
                narration_slot_indexes=[1],
                narration_text="Workers inspect railway tracks",
                primary_narration_slot_index=1,
                primary_narration_text="Workers inspect railway tracks",
                visual_requirement="Workers inspect railway tracks",
                narration_overlaps=[NarrationOverlap(1, 0.0, 4.0, 4.0)],
                search_queries=["workers inspecting railway tracks"],
                timing_source="edge_tts_boundary",
                timing_quality="boundary",
            )
        ]
        state = MemoryState()

        with (
            patch.object(
                tm.twelvelabs, "visual_matching_requested", return_value=False
            ),
            patch.object(tm, "generate_script", return_value=params.video_script),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 4.0, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(tm.voice, "get_audio_duration", return_value=4.0),
            # Without timed units the semantic stage skips grouping and builds the
            # slot-led fallback, whose spoken requirements are then repaired by a
            # second provider call. Both are pinned off so this test cannot depend
            # on a provider being reachable, or bill one.
            patch.object(tm.voice, "extract_timed_narration_units", return_value=[]),
            patch.object(
                tm.llm, "generate_narration_visual_requirements", return_value=None
            ),
            patch.object(tm, "build_narration_slots", return_value=narration_slots),
            patch.object(tm, "build_visual_slots", return_value=visual_slots),
            # A patched narration slot is enough for `build_visual_beats` to reach
            # its fallback and produce a beat, and the beat timeline then asks the
            # configured LLM provider for phrasings. Unpatched that is a real,
            # billable request inside a unit test about a material-stage failure.
            patch.object(tm, "generate_visual_beat_search_queries", return_value=[]),
            patch.object(
                tm,
                "generate_visual_slot_search_queries",
                return_value=["workers inspecting railway tracks"],
            ),
            patch.object(tm, "persist_narration_timeline"),
            patch.object(
                tm,
                "get_video_materials",
                side_effect=tm.material.SmartMaterialSelectionError(
                    "TwelveLabs quota is exhausted"
                ),
            ),
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("smart-material-failure", params, stop_at="materials")

        failed_task = state.get_task("smart-material-failure")
        self.assertEqual(result, failed_task)
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "materials")
        self.assertEqual(failed_task["error"], "TwelveLabs quota is exhausted")

    @staticmethod
    def _railway_beat_timeline():
        """One narration slot, one fixed visual slot and two variable beats."""
        narration_slots = [
            NarrationSlot(
                1,
                0.0,
                5.6,
                5.6,
                "Workers inspect railway tracks",
                "edge_tts_boundary",
            )
        ]
        visual_slots = [
            VisualSlot(
                index=1,
                start_time=0.0,
                end_time=5.6,
                duration=5.6,
                narration_slot_indexes=[1],
                narration_text="Workers inspect railway tracks",
                primary_narration_slot_index=1,
                primary_narration_text="Workers inspect railway tracks",
                visual_requirement="Workers inspect railway tracks",
                narration_overlaps=[NarrationOverlap(1, 0.0, 5.6, 5.6)],
                search_queries=[],
                timing_source="edge_tts_boundary",
                timing_quality="boundary",
            )
        ]
        visual_beats = [
            VisualBeat(
                index=index,
                semantic_group_id=index,
                shot_index=index,
                start_time=(index - 1) * 2.8,
                end_time=index * 2.8,
                duration=2.8,
                spoken_text="Workers inspect railway tracks",
                visual_requirement=requirement,
                source_semantic_span_index=index,
                source_narration_slot_indexes=[1],
                start_unit=index - 1,
                end_unit_exclusive=index,
                timing_source="edge_tts_boundary",
                timing_quality="boundary",
                duration_policy="semantic_original",
                rapid_cut=False,
                search_queries=[],
            )
            for index, requirement in enumerate(
                (
                    "Workers walk along the railway track",
                    "A worker tightens a rail bolt",
                ),
                start=1,
            )
        ]
        return narration_slots, visual_slots, visual_beats

    @staticmethod
    def _assign_beat_queries(visual_beats, queries_per_beat=1):
        """Stand in for the LLM query stage, including the phrasings it was asked for.

        The primary phrasing keeps its stable name so existing assertions on the
        search terms still read the same; the alternates only exist so a caller
        can prove the pipeline requested more than one.
        """
        variants = max(1, int(queries_per_beat))
        for position, beat in enumerate(visual_beats, start=1):
            beat.search_queries = [f"beat {position} query"] + [
                f"beat {position} query variant {variant}"
                for variant in range(2, variants + 1)
            ]
        return [beat.search_queries[0] for beat in visual_beats]

    def _beat_pipeline_patchers(
        self,
        *,
        video_script,
        narration_slots,
        visual_slots,
        visual_beats,
        beat_queries,
        audio_duration=5.6,
    ):
        """Patch every stage that runs before material selection in a beat task.

        Returned as a list instead of a nested ``with`` block: the beat pipeline
        needs more patches than CPython allows statically nested blocks, so the
        callers enter them through an ``ExitStack``.
        """
        return [
            patch.object(
                tm.twelvelabs, "visual_matching_requested", return_value=True
            ),
            # The script stage decomposes the visual-requirement checklist behind
            # this credential-aware gate. Left unpatched it reads the developer's
            # real config, and on a machine with TwelveLabs keys and clip QA on it
            # calls the configured LLM provider for real — a billable network
            # request inside a unit test. Tests that are about the checklist
            # re-patch this after the shared patchers so their own value wins.
            patch.object(
                tm.twelvelabs, "is_smart_visual_matching_enabled", return_value=False
            ),
            patch.object(tm, "generate_script", return_value=video_script),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", audio_duration, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(tm.voice, "get_audio_duration", return_value=audio_duration),
            patch.object(tm.voice, "extract_timed_narration_units", return_value=[]),
            patch.object(tm, "build_narration_slots", return_value=narration_slots),
            patch.object(tm, "build_visual_slots", return_value=visual_slots),
            patch.object(tm, "generate_semantic_visual_spans", return_value=[]),
            patch.object(tm, "build_visual_beats", return_value=visual_beats),
            # Same hazard as the checklist gate above: this stage runs on the
            # real beat timeline, so a fixture with split-span shots would call
            # the configured provider for real. Tests about the split call the
            # function directly instead.
            patch.object(
                tm, "refine_split_span_shot_requirements", return_value=0
            ),
            patch.object(
                tm, "generate_visual_beat_search_queries", **beat_queries
            ),
            patch.object(
                tm, "generate_visual_slot_search_queries", return_value=["slot query"]
            ),
            patch.object(tm, "persist_narration_timeline"),
            patch.object(
                tm.upload_post.upload_post_service, "is_configured", return_value=False
            ),
            patch.object(tm.sm.state, "update_task"),
        ]

    def test_visual_beat_timeline_reaches_the_renderer_as_render_segments(self):
        # pixabay also proves the smart path is no longer pexels-only.
        params = VideoParams(
            video_subject="Railway",
            video_script="Workers inspect railway tracks.",
            subtitle_enabled=False,
            match_materials_to_script=True,
            video_clip_duration=4,
            video_source="pixabay",
        )
        narration_slots, visual_slots, visual_beats = self._railway_beat_timeline()
        winners = ["D:/task/first.mp4", "D:/task/second.mp4"]
        segments = [
            RenderSegment(
                index=1,
                file_path=winners[0],
                source_start=1.0,
                source_end=3.8,
                target_start=0.0,
                target_end=2.8,
                target_duration=2.8,
                playback_speed=1.0,
                visual_beat_index=1,
                semantic_group_id=1,
                provider="pixabay",
            ),
            RenderSegment(
                index=2,
                file_path=winners[1],
                source_start=4.0,
                source_end=6.8,
                target_start=2.8,
                target_end=5.6,
                target_duration=2.8,
                playback_speed=1.0,
                visual_beat_index=2,
                semantic_group_id=2,
                provider="coverr",
            ),
        ]

        with ExitStack() as stack:
            for patcher in self._beat_pipeline_patchers(
                video_script=params.video_script,
                narration_slots=narration_slots,
                visual_slots=visual_slots,
                visual_beats=visual_beats,
                beat_queries={"side_effect": self._assign_beat_queries},
            ):
                stack.enter_context(patcher)
            stack.enter_context(
                patch.object(
                    tm.twelvelabs,
                    "validate_smart_visual_matching_configuration",
                    return_value="",
                )
            )
            download = stack.enter_context(
                patch.object(tm.material, "download_videos", return_value=winners)
            )
            load_segments = stack.enter_context(
                patch.object(
                    tm.material, "load_render_segments", return_value=segments
                )
            )
            load_ranges = stack.enter_context(
                patch.object(tm.material, "load_selected_source_ranges")
            )
            generate_final = stack.enter_context(
                patch.object(
                    tm,
                    "generate_final_videos",
                    return_value=(["final.mp4"], ["combined.mp4"], []),
                )
            )
            result = tm.start("beat-render", params)

        self.assertEqual(result["videos"], ["final.mp4"])
        # Beats carry their own queries; pairing them with the slot queries would
        # search for the wrong thing in every beat.
        self.assertEqual(
            download.call_args.kwargs["search_terms"],
            ["beat 1 query", "beat 2 query"],
        )
        self.assertEqual(download.call_args.kwargs["visual_beats"], visual_beats)
        self.assertIsNone(download.call_args.kwargs["visual_slots"])
        self.assertTrue(download.call_args.kwargs["match_script_order"])
        load_ranges.assert_not_called()
        load_segments.assert_called_once()
        self.assertEqual(load_segments.call_args.args[1], winners)
        self.assertEqual(load_segments.call_args.args[2], visual_beats)
        self.assertEqual(load_segments.call_args.kwargs["audio_duration"], 5.6)
        self.assertEqual(generate_final.call_args.kwargs["render_segments"], segments)
        self.assertIsNone(generate_final.call_args.kwargs["source_ranges"])

    def test_beat_query_failure_keeps_the_fixed_slot_render_path(self):
        params = VideoParams(
            video_subject="Railway",
            video_script="Workers inspect railway tracks.",
            subtitle_enabled=False,
            match_materials_to_script=True,
            video_clip_duration=4,
        )
        narration_slots, visual_slots, visual_beats = self._railway_beat_timeline()
        selected_ranges = [(1.0, 5.0)]

        with ExitStack() as stack:
            for patcher in self._beat_pipeline_patchers(
                video_script=params.video_script,
                narration_slots=narration_slots,
                visual_slots=visual_slots,
                visual_beats=visual_beats,
                beat_queries={
                    "side_effect": ValueError("beat 2 has no usable search query")
                },
            ):
                stack.enter_context(patcher)
            stack.enter_context(
                patch.object(
                    tm.twelvelabs,
                    "validate_smart_visual_matching_configuration",
                    return_value="",
                )
            )
            download = stack.enter_context(
                patch.object(
                    tm.material, "download_videos", return_value=["D:/task/slot.mp4"]
                )
            )
            load_segments = stack.enter_context(
                patch.object(tm.material, "load_render_segments")
            )
            load_ranges = stack.enter_context(
                patch.object(
                    tm.material,
                    "load_selected_source_ranges",
                    return_value=selected_ranges,
                )
            )
            generate_final = stack.enter_context(
                patch.object(
                    tm,
                    "generate_final_videos",
                    return_value=(["final.mp4"], ["combined.mp4"], []),
                )
            )
            result = tm.start("beat-query-failure", params)

        # Losing the per-beat queries degrades to the proven fixed-slot timeline
        # instead of failing the task.
        self.assertEqual(result["videos"], ["final.mp4"])
        self.assertEqual(download.call_args.kwargs["search_terms"], ["slot query"])
        self.assertIsNone(download.call_args.kwargs["visual_beats"])
        self.assertEqual(download.call_args.kwargs["visual_slots"], visual_slots)
        load_segments.assert_not_called()
        load_ranges.assert_called_once()
        self.assertEqual(
            generate_final.call_args.kwargs["source_ranges"], selected_ranges
        )
        self.assertIsNone(generate_final.call_args.kwargs["render_segments"])

    def test_spoken_visual_requirements_never_reach_the_paid_beat_stages(self):
        params = VideoParams(
            video_subject="Railway",
            video_script="Workers inspect railway tracks.",
            subtitle_enabled=False,
            match_materials_to_script=True,
            video_clip_duration=4,
        )
        narration_slots, visual_slots, visual_beats = self._railway_beat_timeline()
        spoken_spans = [
            SemanticVisualSpan(
                index=1,
                start_unit=None,
                end_unit_exclusive=None,
                spoken_text="Workers inspect railway tracks",
                visual_requirement="Workers inspect railway tracks",
                source_narration_slot_indexes=[1],
                start_time=0.0,
                end_time=5.6,
                timing_source="edge_tts_boundary",
                timing_quality="boundary",
                grouping_source="narration_slot_fallback",
            )
        ]

        with ExitStack() as stack:
            for patcher in self._beat_pipeline_patchers(
                video_script=params.video_script,
                narration_slots=narration_slots,
                visual_slots=visual_slots,
                visual_beats=visual_beats,
                beat_queries={"return_value": []},
            ):
                stack.enter_context(patcher)
            spans = stack.enter_context(
                patch.object(
                    tm, "generate_semantic_visual_spans", return_value=spoken_spans
                )
            )
            # Both credential gates are opened on purpose: the guard, not a
            # missing key, has to be what keeps the paid stages out of this run.
            stack.enter_context(
                patch.object(
                    tm.twelvelabs,
                    "is_smart_visual_matching_enabled",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch.object(
                    tm.twelvelabs,
                    "validate_smart_visual_matching_configuration",
                    return_value="",
                )
            )
            beat_query_stage = stack.enter_context(
                patch.object(
                    tm, "generate_visual_beat_search_queries", return_value=[]
                )
            )
            checklist = stack.enter_context(
                patch.object(tm.llm, "generate_visual_requirement_specs")
            )
            persist = stack.enter_context(
                patch.object(tm, "persist_narration_timeline")
            )
            download = stack.enter_context(
                patch.object(
                    tm.material, "download_videos", return_value=["D:/task/slot.mp4"]
                )
            )
            load_segments = stack.enter_context(
                patch.object(tm.material, "load_render_segments")
            )
            stack.enter_context(
                patch.object(
                    tm.material,
                    "load_selected_source_ranges",
                    return_value=[(1.0, 5.0)],
                )
            )
            generate_final = stack.enter_context(
                patch.object(
                    tm,
                    "generate_final_videos",
                    return_value=(["final.mp4"], ["combined.mp4"], []),
                )
            )
            result = tm.start("spoken-requirement-guard", params)

        # The beat timeline was built and then dropped, so neither paid script
        # stage is asked to do anything: no phrasings for sentences, no checklist
        # decomposition of sentences.
        spans.assert_called_once()
        self.assertEqual(beat_query_stage.call_args.args[0], [])
        checklist.assert_not_called()
        self.assertEqual(persist.call_args.kwargs["visual_beats"], [])
        self.assertIsNone(persist.call_args.kwargs["visual_requirement_specs"])
        # The failed grouping stays legible in the persisted provenance instead
        # of being erased along with the beats it produced.
        self.assertEqual(
            persist.call_args.kwargs["semantic_visual_spans"], spoken_spans
        )
        # And the run still renders, on the proven fixed-slot path.
        self.assertEqual(result["videos"], ["final.mp4"])
        self.assertEqual(download.call_args.kwargs["search_terms"], ["slot query"])
        self.assertIsNone(download.call_args.kwargs["visual_beats"])
        load_segments.assert_not_called()
        self.assertIsNone(generate_final.call_args.kwargs["render_segments"])

    def test_the_merged_beat_channel_reaches_material_selection(self):
        """选择阶段重写后的时间线必须有一条回传通道，否则渲染仍绑定旧时间线。"""
        params = VideoParams(
            video_subject="Railway",
            video_source="pexels",
            match_materials_to_script=True,
        )
        _, _, visual_beats = self._railway_beat_timeline()
        self._assign_beat_queries(visual_beats)
        merged_beats_out: list[VisualBeat] = []

        with patch.object(
            tm.material, "download_videos", return_value=["D:/task/first.mp4"]
        ) as download:
            result = tm.get_video_materials(
                "merge-channel",
                params,
                ["fallback term"],
                5.6,
                visual_beats=visual_beats,
                merged_beats_out=merged_beats_out,
            )

        self.assertEqual(result, ["D:/task/first.mp4"])
        # Selection appends to this list in place, so it has to be the caller's own
        # object. A copy would carry every merge it recorded into a local that the
        # orchestrator never reads, and the renderer would bind to the timeline the
        # script stage planned instead of the one the downloads were made for.
        self.assertIs(download.call_args.kwargs["merged_beats_out"], merged_beats_out)

    def test_a_merged_timeline_is_what_the_renderer_and_the_artifact_receive(self):
        # Merging is the last resort before a video fails, so its whole value
        # depends on the shorter timeline actually being adopted: the material
        # records were written against it, and the renderer pairs clips to beats
        # by position.
        params = VideoParams(
            video_subject="Railway",
            video_script="Workers inspect railway tracks.",
            subtitle_enabled=False,
            match_materials_to_script=True,
            video_clip_duration=4,
        )
        narration_slots, visual_slots, visual_beats = self._railway_beat_timeline()
        # Only shots of one semantic group may absorb each other, because that is
        # what makes the neighbour's approved clip valid for the open window. The
        # fixture has to agree with that rule or this scenario could not occur.
        visual_beats = [
            replace(
                beat,
                semantic_group_id=1,
                shot_index=position,
                visual_requirement="Workers walk along the railway track",
            )
            for position, beat in enumerate(visual_beats, start=1)
        ]
        planned = list(visual_beats)
        # The second beat could not be filled, so the first one absorbed its window.
        merged_beat = replace(
            planned[0],
            end_time=5.6,
            duration=5.6,
            end_unit_exclusive=2,
            duration_policy="unfillable_beat_merged",
        )

        def _merge_while_downloading(**kwargs):
            kwargs["merged_beats_out"].append(merged_beat)
            return ["D:/task/first.mp4"]

        with ExitStack() as stack:
            for patcher in self._beat_pipeline_patchers(
                video_script=params.video_script,
                narration_slots=narration_slots,
                visual_slots=visual_slots,
                visual_beats=visual_beats,
                beat_queries={"side_effect": self._assign_beat_queries},
            ):
                stack.enter_context(patcher)
            stack.enter_context(
                patch.object(
                    tm.twelvelabs,
                    "validate_smart_visual_matching_configuration",
                    return_value="",
                )
            )
            download = stack.enter_context(
                patch.object(
                    tm.material,
                    "download_videos",
                    side_effect=_merge_while_downloading,
                )
            )
            load_segments = stack.enter_context(
                patch.object(tm.material, "load_render_segments", return_value=[])
            )
            patch_script = stack.enter_context(
                patch.object(tm.task_artifacts, "patch_script_data")
            )
            stack.enter_context(
                patch.object(
                    tm,
                    "generate_final_videos",
                    return_value=(["final.mp4"], ["combined.mp4"], []),
                )
            )
            result = tm.start("beat-merge-adoption", params)

        self.assertEqual(result["videos"], ["final.mp4"])
        # Two beats were planned and one clip was downloaded; binding the renderer
        # to the planned pair is exactly the length mismatch that would abort here.
        self.assertEqual(len(planned), 2)
        self.assertIs(
            load_segments.call_args.args[2],
            download.call_args.kwargs["merged_beats_out"],
        )
        self.assertEqual(load_segments.call_args.args[2], [merged_beat])
        # The narration span is unchanged, so the renderer still fills the audio.
        self.assertEqual(load_segments.call_args.kwargs["audio_duration"], 5.6)
        # script.json is rewritten too: a beat list naming a shot the video does
        # not contain would disagree with its own material records.
        persisted = patch_script.call_args.kwargs["visual_beats"]
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["end_time"], 5.6)
        self.assertEqual(persisted[0]["duration_policy"], "unfillable_beat_merged")

    def test_the_pipeline_requests_the_configured_number_of_beat_phrasings(self):
        # The script stage is the only place a beat can get alternative phrasings,
        # so if the pipeline asks for one, the per-provider retry in material
        # selection has nothing to spend and the whole knob is dead weight.
        params = VideoParams(
            video_subject="Railway",
            video_script="Workers inspect railway tracks.",
            subtitle_enabled=False,
            match_materials_to_script=True,
            video_clip_duration=4,
        )
        narration_slots, visual_slots, visual_beats = self._railway_beat_timeline()

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(tm.config.app, {"smart_material_max_query_variants": 4})
            )
            for patcher in self._beat_pipeline_patchers(
                video_script=params.video_script,
                narration_slots=narration_slots,
                visual_slots=visual_slots,
                visual_beats=visual_beats,
                beat_queries={"side_effect": self._assign_beat_queries},
            ):
                stack.enter_context(patcher)
            # Re-patched after the shared patchers so this mock is the one the
            # pipeline reaches, and its call can be inspected here.
            beat_queries = stack.enter_context(
                patch.object(
                    tm,
                    "generate_visual_beat_search_queries",
                    side_effect=self._assign_beat_queries,
                )
            )
            stack.enter_context(
                patch.object(
                    tm.twelvelabs,
                    "validate_smart_visual_matching_configuration",
                    return_value="",
                )
            )
            download = stack.enter_context(
                patch.object(
                    tm.material,
                    "download_videos",
                    return_value=["D:/task/first.mp4", "D:/task/second.mp4"],
                )
            )
            stack.enter_context(
                patch.object(tm.material, "load_render_segments", return_value=[])
            )
            stack.enter_context(
                patch.object(tm.material, "load_selected_source_ranges")
            )
            stack.enter_context(
                patch.object(
                    tm,
                    "generate_final_videos",
                    return_value=(["final.mp4"], ["combined.mp4"], []),
                )
            )
            result = tm.start("beat-query-variants", params)

        self.assertEqual(result["videos"], ["final.mp4"])
        self.assertEqual(beat_queries.call_args.kwargs["queries_per_beat"], 4)
        # Only the planned phrasing is searched first; the alternates ride along on
        # the beats for material selection to fall back to.
        self.assertEqual(
            download.call_args.kwargs["search_terms"],
            ["beat 1 query", "beat 2 query"],
        )
        self.assertEqual(len(visual_beats[0].search_queries), 4)

    def test_a_source_without_a_catalog_never_enters_the_smart_render_path(self):
        params = VideoParams(
            video_subject="Railway",
            video_script="Workers inspect railway tracks.",
            subtitle_enabled=False,
            match_materials_to_script=True,
            video_clip_duration=4,
            video_source="loomloom",
        )
        narration_slots, visual_slots, visual_beats = self._railway_beat_timeline()
        self.assertFalse(tm.material.supports_smart_visual_matching("loomloom"))

        with ExitStack() as stack:
            for patcher in self._beat_pipeline_patchers(
                video_script=params.video_script,
                narration_slots=narration_slots,
                visual_slots=visual_slots,
                visual_beats=visual_beats,
                beat_queries={"side_effect": self._assign_beat_queries},
            ):
                stack.enter_context(patcher)
            preflight = stack.enter_context(
                patch.object(
                    tm.twelvelabs, "validate_smart_visual_matching_configuration"
                )
            )
            stack.enter_context(
                patch.object(tm, "get_video_materials", return_value=["clip.mp4"])
            )
            load_segments = stack.enter_context(
                patch.object(tm.material, "load_render_segments")
            )
            load_ranges = stack.enter_context(
                patch.object(tm.material, "load_selected_source_ranges")
            )
            generate_final = stack.enter_context(
                patch.object(
                    tm,
                    "generate_final_videos",
                    return_value=(["final.mp4"], ["combined.mp4"], []),
                )
            )
            result = tm.start("no-catalog-source", params)

        self.assertEqual(result["videos"], ["final.mp4"])
        # A source with no searchable catalog must not even reach the smart
        # matching preflight, let alone the beat render contract.
        preflight.assert_not_called()
        load_segments.assert_not_called()
        load_ranges.assert_not_called()
        self.assertIsNone(generate_final.call_args.kwargs["render_segments"])
        self.assertIsNone(generate_final.call_args.kwargs["source_ranges"])

    def _run_beat_pipeline_for_checklist(self, *, smart_verification_enabled):
        """Run a beat task and report what the checklist stage did.

        Returns the recorded call order plus the decomposition, persistence and
        download mocks, so each test can assert on the part it cares about.
        """
        params = VideoParams(
            video_subject="Railway",
            video_script="Workers inspect railway tracks.",
            subtitle_enabled=False,
            match_materials_to_script=True,
            video_clip_duration=4,
            video_source="pexels",
        )
        narration_slots, visual_slots, visual_beats = self._railway_beat_timeline()
        checklist = {
            tm.llm.normalize_visual_requirement(beat.visual_requirement): object()
            for beat in visual_beats
        }
        order: list[str] = []

        with ExitStack() as stack:
            for patcher in self._beat_pipeline_patchers(
                video_script=params.video_script,
                narration_slots=narration_slots,
                visual_slots=visual_slots,
                visual_beats=visual_beats,
                beat_queries={"side_effect": self._assign_beat_queries},
            ):
                stack.enter_context(patcher)
            stack.enter_context(
                patch.object(
                    tm.twelvelabs,
                    "validate_smart_visual_matching_configuration",
                    return_value="",
                )
            )
            stack.enter_context(
                patch.object(
                    tm.twelvelabs,
                    "is_smart_visual_matching_enabled",
                    return_value=smart_verification_enabled,
                )
            )
            decompose = stack.enter_context(
                patch.object(
                    tm.llm,
                    "generate_visual_requirement_specs",
                    side_effect=lambda requirements: (
                        order.append("decompose") or checklist
                    ),
                )
            )
            # Entered after the shared patchers so this mock, not theirs, is the
            # one the pipeline calls and the one this test can inspect.
            persist = stack.enter_context(
                patch.object(
                    tm,
                    "persist_narration_timeline",
                    side_effect=lambda **kwargs: order.append("persist"),
                )
            )
            download = stack.enter_context(
                patch.object(
                    tm.material,
                    "download_videos",
                    side_effect=lambda **kwargs: (
                        order.append("download") or ["D:/task/first.mp4"]
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    tm.material, "load_render_segments", return_value=[]
                )
            )
            stack.enter_context(
                patch.object(
                    tm,
                    "generate_final_videos",
                    return_value=(["final.mp4"], ["combined.mp4"], []),
                )
            )
            tm.start("beat-checklist", params)

        return order, decompose, persist, download, visual_beats, checklist

    def test_requirement_checklist_is_planned_and_persisted_before_any_download(self):
        (
            order,
            decompose,
            persist,
            download,
            visual_beats,
            checklist,
        ) = self._run_beat_pipeline_for_checklist(smart_verification_enabled=True)

        # The point of moving the checklist into the script stage: it is decided
        # and written to the manifest before a single stock request is paid for.
        self.assertEqual(order, ["decompose", "persist", "download"])
        self.assertEqual(
            decompose.call_args.args[0],
            [beat.visual_requirement for beat in visual_beats],
        )
        self.assertEqual(
            persist.call_args.kwargs["visual_requirement_specs"], checklist
        )
        # Same object in both stages, so verification cannot gate on a second,
        # differently decomposed checklist.
        self.assertIs(download.call_args.kwargs["requirement_specs"], checklist)

    def test_checklist_is_skipped_when_verification_cannot_run(self):
        (
            order,
            decompose,
            persist,
            download,
            _visual_beats,
            _checklist,
        ) = self._run_beat_pipeline_for_checklist(smart_verification_enabled=False)

        # Without credentials no candidate can be verified, so paying an LLM
        # provider to decompose the timeline would buy nothing.
        decompose.assert_not_called()
        self.assertEqual(order, ["persist", "download"])
        self.assertIsNone(persist.call_args.kwargs["visual_requirement_specs"])
        self.assertIsNone(download.call_args.kwargs["requirement_specs"])

    def test_persisted_checklist_names_the_beats_it_could_not_decompose(self):
        beats = [
            VisualBeat(
                index=index,
                semantic_group_id=index,
                shot_index=1,
                start_time=float(index - 1),
                end_time=float(index),
                duration=1.0,
                spoken_text="Slow change becomes sudden change.",
                visual_requirement=requirement,
                source_semantic_span_index=index,
                source_narration_slot_indexes=[index],
                start_unit=index - 1,
                end_unit_exclusive=index,
                timing_source="edge_tts_boundary",
                timing_quality="boundary",
                duration_policy="semantic_original",
                rapid_cut=False,
                search_queries=[f"query {index}"],
            )
            for index, requirement in enumerate(
                ("A crack widens in dry soil", "A dam wall collapses"), start=1
            )
        ]
        resolved = {
            tm.llm.normalize_visual_requirement(
                beats[0].visual_requirement
            ): _requirement_spec(beats[0].visual_requirement)
        }

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(tm.utils, "task_dir", return_value=tmp_dir),
            patch.object(
                tm.llm, "generate_visual_requirement_specs", return_value=resolved
            ),
        ):
            tm.task_artifacts.write_script_data("checklist", {"script": "Slow change."})
            checklist = tm.generate_visual_requirement_checklist(beats)
            tm.persist_narration_timeline(
                task_id="checklist",
                narration_slots=[],
                visual_slots=[],
                video_terms=[],
                visual_beats=beats,
                visual_requirement_specs=checklist,
            )
            persisted = tm.task_artifacts.read_script_data("checklist")

        # A requirement the provider could not decompose stays absent instead of
        # being replaced by an invented spec, and the gap is named explicitly.
        self.assertEqual(checklist, resolved)
        self.assertEqual(
            persisted["visual_requirement_specs_missing_beat_indexes"], [2]
        )
        self.assertEqual(len(persisted["visual_requirement_specs"]), 1)
        record = persisted["visual_requirement_specs"][0]
        self.assertEqual(
            record["normalized_requirement"],
            tm.llm.normalize_visual_requirement(beats[0].visual_requirement),
        )
        self.assertEqual(
            record["spec"]["original_requirement"], beats[0].visual_requirement
        )
        self.assertNotIn("api_key", record["spec"])

    def test_start_returns_each_intermediate_result(self):
        """
        API 的 script、terms、audio、subtitle 和 materials 模式共用同一条任务
        流水线。每个提前停止点都要返回对应产物，同时不能误执行后续阶段。
        """
        expected_results = {
            "script": {"script": "generated script"},
            "terms": {
                "script": "generated script",
                "terms": ["coffee", "morning"],
            },
            "audio": {"audio_file": "audio.mp3", "audio_duration": 5},
            "subtitle": {"subtitle_path": "subtitle.srt"},
            "materials": {"materials": ["clip.mp4"]},
        }

        for stop_at, expected in expected_results.items():
            with self.subTest(stop_at=stop_at):
                params = VideoParams(video_subject="Coffee")
                with (
                    patch.object(
                        tm, "generate_script", return_value="generated script"
                    ),
                    patch.object(
                        tm,
                        "generate_terms",
                        return_value=["coffee", "morning"],
                    ),
                    patch.object(tm, "save_script_data"),
                    patch.object(
                        tm,
                        "generate_audio",
                        return_value=("audio.mp3", 5, object()),
                    ),
                    patch.object(
                        tm,
                        "generate_subtitle",
                        return_value="subtitle.srt",
                    ),
                    patch.object(
                        tm,
                        "get_video_materials",
                        return_value=["clip.mp4"],
                    ),
                    patch.object(tm, "generate_final_videos") as generate_final,
                    patch.object(tm.sm.state, "update_task"),
                ):
                    result = tm.start(
                        f"intermediate-{stop_at}", params, stop_at=stop_at
                    )

                self.assertEqual(result, expected)
                generate_final.assert_not_called()

    def test_start_completes_video_without_cross_posting(self):
        """
        完整任务在自动发布未配置时仍应稳定完成，并把所有中间产物写入最终
        状态。这里还覆盖 API 可能传入字符串拼接模式的兼容转换。
        """
        params = VideoParams(video_subject="Coffee")
        params.video_concat_mode = "sequential"

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(
                tm,
                "get_video_materials",
                return_value=["clip.mp4"],
            ),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(["final.mp4"], ["combined.mp4"], []),
            ),
            patch.object(
                tm.upload_post.upload_post_service,
                "is_configured",
                return_value=False,
            ),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
            patch.object(tm.sm.state, "update_task") as update_task,
        ):
            result = tm.start("complete-video", params)

        self.assertEqual(result["videos"], ["final.mp4"])
        self.assertEqual(result["combined_videos"], ["combined.mp4"])
        self.assertEqual(result["cross_post_results"], None)
        self.assertEqual(params.video_concat_mode, tm.VideoConcatMode.sequential)
        cross_post.assert_not_called()
        update_task.assert_called_with(
            "complete-video",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            **result,
        )

    def test_start_marks_pipeline_failures(self):
        """
        音频、素材和最终视频任一关键产物缺失时都必须进入失败状态，不能把
        不完整任务误报为完成。三个场景复用相同 mock，仅替换故障阶段。
        """
        failure_cases = {
            "audio": (
                (None, None, None),
                ["clip.mp4"],
                (["final.mp4"], ["combined.mp4"], []),
            ),
            "materials": (
                ("audio.mp3", 5, object()),
                None,
                (["final.mp4"], ["combined.mp4"], []),
            ),
            "video": (("audio.mp3", 5, object()), ["clip.mp4"], ([], [], [])),
        }

        for stage, failure_results in failure_cases.items():
            with self.subTest(stage=stage):
                audio_result, materials_result, videos_result = failure_results
                params = VideoParams(video_subject="Coffee")
                state = MemoryState()
                with (
                    patch.object(
                        tm, "generate_script", return_value="generated script"
                    ),
                    patch.object(tm, "generate_terms", return_value=["coffee"]),
                    patch.object(tm, "save_script_data"),
                    patch.object(tm, "generate_audio", return_value=audio_result),
                    patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
                    patch.object(
                        tm,
                        "get_video_materials",
                        return_value=materials_result,
                    ),
                    patch.object(
                        tm,
                        "generate_final_videos",
                        return_value=videos_result,
                    ),
                    patch.object(tm.sm, "state", state),
                ):
                    result = tm.start(f"failed-{stage}", params)

                failed_task = state.get_task(f"failed-{stage}")
                self.assertEqual(result, failed_task)
                self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
                self.assertEqual(failed_task["failed_stage"], stage)
                self.assertTrue(failed_task["error"])

    def test_start_records_unexpected_pipeline_exception(self):
        """未预期异常也必须结束任务，并向 API 暴露原始异常类型和信息。"""
        params = VideoParams(video_subject="Coffee")
        state = MemoryState()

        with (
            patch.object(
                tm,
                "generate_script",
                side_effect=RuntimeError("provider connection reset"),
            ),
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("unexpected-failure", params)

        failed_task = state.get_task("unexpected-failure")
        self.assertEqual(result, failed_task)
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "pipeline")
        self.assertEqual(
            failed_task["error"],
            "RuntimeError: provider connection reset",
        )

    def test_start_generates_youtube_metadata_for_each_cross_post(self):
        """
        自动发布到 YouTube 时只生成一次元数据，但要把同一份字段传给每个
        成片，并在任务结果中保留每次上传成功或失败的独立结果。
        """
        params = VideoParams(
            video_subject="Coffee",
            video_language="en",
        )
        metadata = {
            "title": "Morning Coffee",
            "caption": "A better morning.",
            "hashtags": ["coffee", "shorts"],
        }
        service = tm.upload_post.upload_post_service
        state = MemoryState()

        def run_immediately(function, *args):
            future = Future()
            try:
                function(*args)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)
            return future

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(
                tm,
                "get_video_materials",
                return_value=["clip.mp4"],
            ),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(
                    ["final-1.mp4", "final-2.mp4"],
                    ["combined-1.mp4", "combined-2.mp4"],
                    [],
                ),
            ),
            patch.object(service, "is_configured", return_value=True),
            patch.object(service, "auto_upload", True),
            patch.object(service, "platforms", ["youtube"]),
            patch.object(service, "youtube_privacy_status", "unlisted"),
            patch.object(
                tm.llm,
                "generate_social_metadata",
                return_value=metadata,
            ) as generate_metadata,
            patch.object(
                tm.upload_post,
                "cross_post_video",
                side_effect=[
                    {"success": True},
                    {"success": False, "error": "upload failed"},
                ],
            ) as cross_post,
            patch.object(tm.sm, "state", state),
            patch.object(
                tm._cross_post_executor,
                "submit",
                side_effect=run_immediately,
            ),
        ):
            result = tm.start("youtube-cross-post", params)

        generate_metadata.assert_called_once_with(
            video_subject="Coffee",
            video_script="generated script",
            language="en",
            platform="youtube_shorts",
        )
        expected_extra = {
            "youtube_title": "Morning Coffee",
            "youtube_description": "A better morning.",
            "tags": ["coffee", "shorts"],
            "privacyStatus": "unlisted",
            "containsSyntheticMedia": True,
        }
        self.assertEqual(cross_post.call_count, 2)
        for call in cross_post.call_args_list:
            self.assertEqual(call.kwargs["youtube_extra"], expected_extra)
            self.assertEqual(call.kwargs["platforms"], ["youtube"])

        # start() 返回的是视频完成时的稳定快照；后台发布结果通过任务查询获取。
        self.assertEqual(result["cross_post_state"], tm.const.CROSS_POST_STATE_PENDING)
        self.assertIsNone(result["cross_post_results"])
        published_task = state.get_task("youtube-cross-post")
        self.assertEqual(published_task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(
            published_task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED
        )
        self.assertEqual(
            published_task["cross_post_results"],
            [
                {"success": True},
                {"success": False, "error": "upload failed"},
            ],
        )
        self.assertEqual(published_task["cross_post_error"], "upload failed")

    def test_start_returns_before_cross_post_worker_runs(self):
        """视频任务完成时只提交发布工作，不能在生成线程中同步上传。"""
        params = VideoParams(video_subject="Coffee")
        service = tm.upload_post.upload_post_service
        state = MemoryState()
        submitted = []

        def capture_submission(function, *args):
            submitted.append((function, args))
            return MagicMock(spec=Future)

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(tm, "get_video_materials", return_value=["clip.mp4"]),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(["final.mp4"], ["combined.mp4"], []),
            ),
            patch.object(service, "is_configured", return_value=True),
            patch.object(service, "auto_upload", True),
            patch.object(service, "platforms", ["tiktok"]),
            patch.object(service, "youtube_privacy_status", "private"),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
            patch.object(tm.sm, "state", state),
            patch.object(
                tm._cross_post_executor,
                "submit",
                side_effect=capture_submission,
            ) as submit,
        ):
            result = tm.start("deferred-cross-post", params)

        submit.assert_called_once()
        cross_post.assert_not_called()
        self.assertEqual(result["videos"], ["final.mp4"])
        self.assertEqual(result["cross_post_state"], tm.const.CROSS_POST_STATE_PENDING)
        completed_task = state.get_task("deferred-cross-post")
        self.assertEqual(completed_task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(completed_task["progress"], 100)

        worker, worker_args = submitted[0]
        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.upload_post,
                "cross_post_video",
                return_value={"success": True, "request_id": "upload-1"},
            ),
        ):
            worker(*worker_args)

        published_task = state.get_task("deferred-cross-post")
        self.assertEqual(published_task["videos"], ["final.mp4"])
        self.assertEqual(
            published_task["cross_post_state"], tm.const.CROSS_POST_STATE_COMPLETE
        )

    def test_cross_post_worker_failure_does_not_change_video_completion(self):
        """发布线程异常只能更新发布状态，不能破坏已完成的视频结果。"""
        state = MemoryState()
        state.update_task(
            "cross-post-worker-failure",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.llm,
                "generate_social_metadata",
                side_effect=RuntimeError("metadata provider unavailable"),
            ),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
        ):
            tm._run_cross_post(
                "cross-post-worker-failure",
                ("final.mp4",),
                "Coffee",
                "A short coffee story.",
                "en",
                ("youtube",),
                "private",
            )

        cross_post.assert_not_called()
        task = state.get_task("cross-post-worker-failure")
        self.assertEqual(task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("metadata provider unavailable", task["cross_post_error"])

    def test_start_returns_cross_post_scheduling_failure(self):
        """同步调度失败必须同时体现在任务状态和 start() 返回快照中。"""
        params = VideoParams(video_subject="Coffee")
        service = tm.upload_post.upload_post_service
        state = MemoryState()

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(tm, "get_video_materials", return_value=["clip.mp4"]),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(["final.mp4"], ["combined.mp4"], []),
            ),
            patch.object(service, "is_configured", return_value=True),
            patch.object(service, "auto_upload", True),
            patch.object(service, "platforms", ["tiktok"]),
            patch.object(service, "youtube_privacy_status", "private"),
            patch.object(tm.sm, "state", state),
            patch.object(tm._cross_post_slots, "acquire", return_value=False),
            patch.object(tm._cross_post_executor, "submit") as submit,
        ):
            result = tm.start("cross-post-queue-full-result", params)

        submit.assert_not_called()
        self.assertEqual(result["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("queue is full", result["cross_post_error"])
        persisted_task = state.get_task("cross-post-queue-full-result")
        self.assertEqual(
            persisted_task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED
        )
        self.assertEqual(
            persisted_task["cross_post_error"],
            result["cross_post_error"],
        )

    def test_cross_post_schedule_failure_is_recorded_separately(self):
        """线程池拒绝新任务时应保留成片，并提供可查询的发布错误。"""
        state = MemoryState()
        slots = MagicMock()
        slots.acquire.return_value = True
        state.update_task(
            "cross-post-schedule-failure",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm, "_cross_post_slots", slots),
            patch.object(
                tm._cross_post_executor,
                "submit",
                side_effect=RuntimeError("executor is shutting down"),
            ),
        ):
            scheduling_error = tm._schedule_cross_post(
                task_id="cross-post-schedule-failure",
                video_paths=["final.mp4"],
                params=VideoParams(video_subject="Coffee"),
                video_script="A short coffee story.",
                platforms=["tiktok"],
                youtube_privacy_status="private",
            )

        slots.release.assert_called_once_with()
        self.assertIn("executor is shutting down", scheduling_error)
        task = state.get_task("cross-post-schedule-failure")
        self.assertEqual(task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("executor is shutting down", task["cross_post_error"])

    def test_cross_post_worker_always_releases_queue_slot(self):
        """发布工作异常退出时也必须归还容量，避免后续发布永久被拒绝。"""
        slots = MagicMock()
        state = MemoryState()
        state.update_task(
            "task-id",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm, "_cross_post_slots", slots),
            patch.object(tm.sm, "state", state),
            patch.object(
                tm,
                "_run_cross_post",
                side_effect=RuntimeError("worker crashed"),
            ),
        ):
            tm._run_cross_post_with_slot("task-id")

        slots.release.assert_called_once_with()
        task = state.get_task("task-id")
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("worker crashed", task["cross_post_error"])

    def test_cross_post_state_backend_failure_is_logged_and_skips_upload(self):
        """首次状态写入失败时不能静默退出，也不能继续消耗发布额度。"""
        state = MagicMock()
        state.patch_task.side_effect = RuntimeError("redis unavailable")

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
            patch.object(tm.logger, "exception") as log_exception,
            patch.object(tm.time, "sleep") as sleep,
        ):
            tm._run_cross_post(
                "state-backend-failure",
                ("final.mp4",),
                "Coffee",
                "A short coffee story.",
                "en",
                ("tiktok",),
                "private",
            )

        cross_post.assert_not_called()
        self.assertEqual(state.patch_task.call_count, 6)
        self.assertEqual(sleep.call_count, 4)
        self.assertEqual(log_exception.call_count, 2)
        self.assertTrue(
            all(
                "redis unavailable" in call.args[0]
                for call in log_exception.call_args_list
            )
        )

    def test_cross_post_state_update_retries_transient_backend_failure(self):
        """状态后端短暂失败一次后应继续发布，并最终保存完成状态。"""

        class FlakyMemoryState(MemoryState):
            def __init__(self):
                super().__init__()
                self.patch_calls = 0

            def patch_task(self, task_id, **kwargs):
                self.patch_calls += 1
                if self.patch_calls == 1:
                    raise RuntimeError("temporary redis outage")
                return super().patch_task(task_id, **kwargs)

        state = FlakyMemoryState()
        state.update_task(
            "transient-state-failure",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.upload_post,
                "cross_post_video",
                return_value={"success": True, "request_id": "upload-1"},
            ) as cross_post,
            patch.object(tm.time, "sleep") as sleep,
        ):
            tm._run_cross_post(
                "transient-state-failure",
                ("final.mp4",),
                "Coffee",
                "A short coffee story.",
                "en",
                ("tiktok",),
                "private",
            )

        sleep.assert_called_once_with(tm._CROSS_POST_STATE_RETRY_DELAY_SECONDS)
        cross_post.assert_called_once()
        task = state.get_task("transient-state-failure")
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_COMPLETE)
        self.assertIsNone(task["cross_post_error"])

    def test_recover_interrupted_cross_posts_preserves_active_future(self):
        """启动恢复只处理遗留状态，当前进程仍持有的发布任务不能被误伤。"""
        state = MemoryState()
        for task_id in (
            "stale-pending",
            "active-processing",
            "inactive-current-owner",
            "remote-processing",
            "already-complete",
        ):
            cross_post_state = {
                "stale-pending": tm.const.CROSS_POST_STATE_PENDING,
                "active-processing": tm.const.CROSS_POST_STATE_PROCESSING,
                "inactive-current-owner": tm.const.CROSS_POST_STATE_PROCESSING,
                "remote-processing": tm.const.CROSS_POST_STATE_PROCESSING,
                "already-complete": tm.const.CROSS_POST_STATE_COMPLETE,
            }[task_id]
            state.update_task(
                task_id,
                state=tm.const.TASK_STATE_COMPLETE,
                progress=100,
                videos=["final.mp4"],
                cross_post_state=cross_post_state,
                cross_post_owner=(
                    "another-host:123:remote"
                    if task_id == "remote-processing"
                    else (
                        tm._cross_post_process_owner
                        if task_id == "inactive-current-owner"
                        else None
                    )
                ),
            )

        active_future = Future()
        tm._register_cross_post_future("active-processing", active_future)
        with patch.object(tm.sm, "state", state):
            recovered = tm.recover_interrupted_cross_posts(page_size=1)

        self.assertEqual(recovered, 2)
        stale_task = state.get_task("stale-pending")
        self.assertEqual(
            stale_task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED
        )
        self.assertEqual(
            stale_task["cross_post_error"], tm._INTERRUPTED_CROSS_POST_ERROR
        )
        self.assertEqual(
            state.get_task("active-processing")["cross_post_state"],
            tm.const.CROSS_POST_STATE_PROCESSING,
        )
        self.assertEqual(
            state.get_task("inactive-current-owner")["cross_post_state"],
            tm.const.CROSS_POST_STATE_FAILED,
        )
        self.assertEqual(
            state.get_task("remote-processing")["cross_post_state"],
            tm.const.CROSS_POST_STATE_PROCESSING,
        )
        self.assertEqual(
            state.get_task("already-complete")["cross_post_state"],
            tm.const.CROSS_POST_STATE_COMPLETE,
        )
        active_future.set_result(None)

    def test_cross_post_owner_uses_future_registry_for_current_process(self):
        """当前进程无活动 Future 时，同 PID 的新旧 owner 都应视为中断。"""
        stale_owner = f"{tm.socket.gethostname()}:{tm.os.getpid()}:old-instance"

        self.assertFalse(tm._is_cross_post_owner_alive(stale_owner))
        self.assertFalse(tm._is_cross_post_owner_alive(tm._cross_post_process_owner))

    def test_cross_post_owner_detection_handles_process_boundaries(self):
        """所有者探测应覆盖旧记录、其它主机和本机进程异常边界。"""
        hostname = tm.socket.gethostname()

        self.assertFalse(tm._is_cross_post_owner_alive(None))
        self.assertFalse(tm._is_cross_post_owner_alive("invalid-owner"))
        self.assertTrue(tm._is_cross_post_owner_alive("another-host:123:instance"))

        with (
            patch.object(tm.os, "name", "posix"),
            patch.object(tm.os, "kill", side_effect=ProcessLookupError),
        ):
            self.assertFalse(
                tm._is_cross_post_owner_alive(f"{hostname}:987654:dead-instance")
            )
        with (
            patch.object(tm.os, "name", "posix"),
            patch.object(tm.os, "kill", side_effect=PermissionError),
        ):
            self.assertTrue(
                tm._is_cross_post_owner_alive(f"{hostname}:987654:restricted")
            )
        with (
            patch.object(tm.os, "name", "posix"),
            patch.object(tm.os, "kill", side_effect=OSError("inspection failed")),
            patch.object(tm.logger, "warning") as log_warning,
        ):
            self.assertTrue(tm._is_cross_post_owner_alive(f"{hostname}:987654:unknown"))
        self.assertIn("inspection failed", log_warning.call_args.args[0])

        with (
            patch.object(tm.os, "name", "nt"),
            patch.object(tm, "_is_windows_process_alive", return_value=True) as probe,
        ):
            self.assertTrue(tm._is_cross_post_owner_alive(f"{hostname}:987654:windows"))
        probe.assert_called_once_with(987654)

    @unittest.skipUnless(os.name == "nt", "Windows process API test")
    def test_windows_process_probe_is_read_only_and_detects_liveness(self):
        """Windows CI 应真实验证只读进程探测，不允许回退到 os.kill。"""
        self.assertTrue(tm._is_windows_process_alive(os.getpid()))
        self.assertFalse(tm._is_windows_process_alive(2_147_483_647))

    def test_cross_post_terminal_check_converts_active_state_to_failure(self):
        """worker 已结束但状态仍活动时，最终回调必须补写失败终态。"""
        state = MemoryState()
        state.update_task(
            "unfinished-cross-post",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PROCESSING,
        )

        with patch.object(tm.sm, "state", state):
            tm._ensure_cross_post_terminal_state("unfinished-cross-post")

        task = state.get_task("unfinished-cross-post")
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("without persisting", task["cross_post_error"])

    def test_cross_post_recovery_reports_state_backend_failure(self):
        """启动恢复读取状态失败时应返回 None，允许 WebUI 后续 rerun 重试。"""
        state = MagicMock()
        state.get_all_tasks.side_effect = RuntimeError("redis unavailable")

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm.logger, "exception") as log_exception,
        ):
            recovered = tm.recover_interrupted_cross_posts()

        self.assertIsNone(recovered)
        self.assertIn("redis unavailable", log_exception.call_args.args[0])

    def test_cancelled_cross_post_future_releases_slot_and_records_failure(self):
        """排队 Future 被取消时也必须释放容量并写入失败终态。"""
        state = MemoryState()
        state.update_task(
            "cancelled-cross-post",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )
        slots = MagicMock()
        future = Future()
        tm._register_cross_post_future("cancelled-cross-post", future)
        self.assertTrue(future.cancel())

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm, "_cross_post_slots", slots),
        ):
            tm._finalize_cross_post_future("cancelled-cross-post", future)

        slots.release.assert_called_once_with()
        self.assertFalse(tm._is_cross_post_active_in_process("cancelled-cross-post"))
        task = state.get_task("cancelled-cross-post")
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("cancelled", task["cross_post_error"])

    @unittest.skipUnless(
        os.getenv("MPT_TEST_REDIS_HOST"),
        "MPT_TEST_REDIS_HOST not set",
    )
    def test_real_redis_recovers_interrupted_cross_post_state(self):
        """真实 Redis 中的遗留发布状态必须在恢复后保留视频并进入失败终态。"""
        state = RedisState(
            host=os.environ["MPT_TEST_REDIS_HOST"],
            port=int(os.getenv("MPT_TEST_REDIS_PORT", "6379")),
            db=int(os.getenv("MPT_TEST_REDIS_DB", "15")),
        )
        task_id = f"ci-cross-post-recovery-{uuid4()}"
        state.update_task(
            task_id,
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PROCESSING,
            cross_post_owner="",
        )

        try:
            with patch.object(tm.sm, "state", state):
                recovered = tm.recover_interrupted_cross_posts(page_size=10)

            self.assertGreaterEqual(recovered, 1)
            task = state.get_task(task_id)
            self.assertEqual(task["videos"], ["final.mp4"])
            self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
            self.assertEqual(task["cross_post_error"], tm._INTERRUPTED_CROSS_POST_ERROR)
        finally:
            state.delete_task(task_id)

    def test_cross_post_future_exception_is_observed(self):
        """线程池自身抛出的异常必须进入日志，不能留在无人读取的 Future 中。"""
        future = Future()
        future.set_exception(RuntimeError("executor worker failed"))

        with patch.object(tm.logger, "error") as log_error:
            tm._finalize_cross_post_future("future-failure", future)

        log_error.assert_called_once()
        self.assertIn("executor worker failed", log_error.call_args.args[0])

    def test_cross_post_queue_full_rejects_only_publishing(self):
        """发布队列满载时必须保留成片，并且不能继续向线程池提交任务。"""
        state = MemoryState()
        state.update_task(
            "cross-post-queue-full",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm._cross_post_slots,
                "acquire",
                return_value=False,
            ),
            patch.object(tm._cross_post_executor, "submit") as submit,
        ):
            scheduling_error = tm._schedule_cross_post(
                task_id="cross-post-queue-full",
                video_paths=["final.mp4"],
                params=VideoParams(video_subject="Coffee"),
                video_script="A short coffee story.",
                platforms=["tiktok"],
                youtube_privacy_status="private",
            )

        submit.assert_not_called()
        self.assertIn("queue is full", scheduling_error)
        task = state.get_task("cross-post-queue-full")
        self.assertEqual(task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("queue is full", task["cross_post_error"])

    @unittest.skipUnless(
        RUN_INTEGRATION_TESTS,
        "MPT_RUN_INTEGRATION_TESTS not set",
    )
    def test_task_local_materials(self):
        task_id = "00000000-0000-0000-0000-000000000000"
        video_materials = []
        for i in range(1, 4):
            video_materials.append(
                MaterialInfo(
                    provider="local",
                    url=os.path.join(resources_dir, f"{i}.png"),
                    duration=0,
                )
            )

        params = VideoParams(
            video_subject="金钱的作用",
            video_script="金钱不仅是交换媒介，更是社会资源的分配工具。它能满足基本生存需求，如食物和住房，也能提供教育、医疗等提升生活品质的机会。拥有足够的金钱意味着更多选择权，比如职业自由或创业可能。但金钱的作用也有边界，它无法直接购买幸福、健康或真诚的人际关系。过度追逐财富可能导致价值观扭曲，忽视精神层面的需求。理想的状态是理性看待金钱，将其作为实现目标的工具而非终极目的。",
            video_terms="money importance, wealth and society, financial freedom, money and happiness, role of money",
            video_aspect="9:16",
            video_concat_mode="random",
            video_transition_mode="None",
            video_clip_duration=3,
            video_count=1,
            video_source="local",
            video_materials=video_materials,
            video_language="",
            voice_name="zh-CN-XiaoxiaoNeural-Female",
            voice_volume=1.0,
            voice_rate=1.0,
            bgm_type="random",
            bgm_file="",
            bgm_volume=0.2,
            subtitle_enabled=True,
            subtitle_position="bottom",
            custom_position=70.0,
            font_name="MicrosoftYaHeiBold.ttc",
            text_fore_color="#FFFFFF",
            text_background_color=True,
            font_size=60,
            stroke_color="#000000",
            stroke_width=1.5,
            n_threads=2,
            paragraph_number=1,
        )
        result = tm.start(task_id=task_id, params=params)
        print(result)


if __name__ == "__main__":
    unittest.main()
