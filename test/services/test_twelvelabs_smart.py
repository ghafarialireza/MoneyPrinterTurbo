import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect, VisualSlot
from app.services import material, task, twelvelabs


def _payload(score=0.8, **overrides):
    payload = {
        "match": True,
        "required_action_visible": True,
        "subject_clearly_visible": True,
        "scores": {
            "semantic_match": score,
            "action_match": score,
            "subject_visibility": score,
            "visual_quality": score,
        },
        "quality_flags": {
            "severe_blur": False,
            "dominant_text_or_logo": False,
            "bad_orientation": False,
            "awkward_or_unusable_framing": False,
        },
        "visible_summary": "The requested action is clearly visible.",
        "reason": "Visible evidence matches the narration.",
    }
    payload.update(overrides)
    return payload


def _evaluation(score, *, accepted=True):
    return {
        "accepted": accepted,
        "overall_score": score,
        "reason": f"score {score}",
        "_cache_hit": False,
        "_api_call": True,
    }


def _candidate(index, *, duration=12, width=1080, height=1920, url=None):
    return MaterialInfo(
        provider="pexels",
        url=url or f"https://videos.example/{index}.mp4",
        duration=duration,
        source_info={
            "provider": "pexels",
            "asset_id": str(index),
            "rendition": {"id": str(index), "width": width, "height": height},
        },
    )


def _visual_slot():
    return VisualSlot(
        index=1,
        start_time=0.0,
        end_time=4.0,
        duration=4.0,
        narration_slot_indexes=[1],
        narration_text="A worker removes rotten wooden boards.",
        search_queries=["worker removing rotten boards"],
        timing_source="edge_tts_boundary",
        timing_quality="boundary",
    )


class TestSmartTwelveLabsSelection(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app.update(
            {
                "twelvelabs_api_keys": ["test-key"],
                "twelvelabs_clip_qa": True,
                "twelvelabs_clip_qa_min_score": 0.70,
                "twelvelabs_clip_qa_fail_closed": True,
            }
        )

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_structured_response_uses_application_weights_and_hard_gates(self):
        payload = _payload()
        payload["scores"] = {
            "semantic_match": 1.0,
            "action_match": 0.5,
            "subject_visibility": 0.75,
            "visual_quality": 0.4,
        }

        result = twelvelabs._parse_candidate_response(payload, 0.70)

        self.assertAlmostEqual(result["overall_score"], 0.71)
        self.assertTrue(result["accepted"])
        payload["required_action_visible"] = False
        rejected = twelvelabs._parse_candidate_response(payload, 0.0)
        self.assertFalse(rejected["accepted"])

    def test_structured_schema_uses_only_twelvelabs_supported_number_constraints(self):
        serialized = json.dumps(twelvelabs._CANDIDATE_RESPONSE_SCHEMA)
        score_properties = twelvelabs._CANDIDATE_RESPONSE_SCHEMA["properties"][
            "scores"
        ]["properties"]

        self.assertNotIn("additionalProperties", serialized)
        for score_schema in score_properties.values():
            self.assertEqual(score_schema, {"type": "number"})

    def test_search_query_is_only_a_retrieval_hint_in_prompt(self):
        prompt = twelvelabs._candidate_prompt(
            slot_index=5,
            slot_duration=4.0,
            narration_text="A worker removes rotten boards.",
            search_query="wood construction",
        )

        self.assertIn("ACTUAL NARRATION REQUIREMENT", prompt)
        self.assertIn("A worker removes rotten boards.", prompt)
        self.assertIn("only to retrieve", prompt)
        self.assertIn("related topic alone is not enough", prompt)

    def test_strong_first_batch_evaluates_all_five_then_stops(self):
        candidates = [_candidate(index) for index in range(1, 11)]
        scores = {str(index): 0.75 for index in range(1, 11)}
        scores["4"] = 0.93

        def evaluate(**kwargs):
            return _evaluation(scores[kwargs["asset_id"]])

        with patch.object(
            twelvelabs, "evaluate_candidate", side_effect=evaluate
        ) as call:
            winner, stats = twelvelabs.select_best_candidate(
                candidates=candidates,
                slot_index=1,
                slot_duration=4,
                narration_text="exact narration",
                search_query="retrieval hint",
                batch_size=5,
                max_candidates=15,
                minimum_score=0.7,
                strong_early_stop_score=0.9,
                concurrency=5,
            )

        self.assertEqual(call.call_count, 5)
        self.assertEqual(winner.source_info["asset_id"], "4")
        self.assertTrue(stats["early_stopped"])
        for recorded_call in call.call_args_list:
            self.assertEqual(recorded_call.kwargs["narration_text"], "exact narration")
            self.assertEqual(recorded_call.kwargs["search_query"], "retrieval hint")

    def test_weak_first_batch_continues_and_selects_global_best(self):
        candidates = [_candidate(index) for index in range(1, 11)]
        scores = {str(index): 0.72 for index in range(1, 11)}
        scores["1"] = 0.80
        scores["8"] = 0.89

        with patch.object(
            twelvelabs,
            "evaluate_candidate",
            side_effect=lambda **kwargs: _evaluation(scores[kwargs["asset_id"]]),
        ) as call:
            winner, stats = twelvelabs.select_best_candidate(
                candidates=candidates,
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
                batch_size=5,
                max_candidates=15,
                minimum_score=0.7,
                strong_early_stop_score=0.9,
                concurrency=5,
            )

        self.assertEqual(call.call_count, 10)
        self.assertEqual(winner.source_info["asset_id"], "8")
        self.assertEqual(stats["batches_processed"], 2)
        self.assertNotEqual(winner.source_info["asset_id"], "1")

    def test_never_analyzes_more_than_fifteen_candidates(self):
        candidates = [_candidate(index) for index in range(1, 21)]
        with patch.object(
            twelvelabs,
            "evaluate_candidate",
            return_value=_evaluation(0.75),
        ) as call:
            _, stats = twelvelabs.select_best_candidate(
                candidates=candidates,
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
                batch_size=5,
                max_candidates=15,
                strong_early_stop_score=0.9,
            )

        self.assertEqual(call.call_count, 15)
        self.assertEqual(stats["candidates_evaluated"], 15)

    def test_malformed_response_and_api_error_fail_closed(self):
        with patch.object(
            twelvelabs,
            "_sync_candidate_analysis",
            return_value=("not json", "direct_url"),
        ):
            malformed = twelvelabs.evaluate_candidate(
                asset_id="malformed",
                video_url="https://videos.example/malformed.mp4",
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
            )
        with patch.object(
            twelvelabs,
            "_sync_candidate_analysis",
            side_effect=RuntimeError("secret test-key"),
        ):
            api_error = twelvelabs.evaluate_candidate(
                asset_id="api-error",
                video_url="https://videos.example/api-error.mp4",
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
            )

        self.assertFalse(malformed["accepted"])
        self.assertFalse(api_error["accepted"])
        self.assertNotIn("test-key", api_error["reason"])

    def test_truncated_structured_response_fails_closed(self):
        response = SimpleNamespace(data=_payload(0.99), finish_reason="length")
        with patch.object(
            twelvelabs,
            "_sync_candidate_analysis",
            return_value=(response, "direct_url"),
        ):
            result = twelvelabs.evaluate_candidate(
                asset_id="truncated",
                video_url="https://videos.example/truncated.mp4",
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
            )

        self.assertFalse(result["accepted"])
        self.assertIn("malformed", result["reason"])

    def test_candidate_cache_hit_avoids_duplicate_api_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    twelvelabs, "_candidate_cache_dir", return_value=Path(temp_dir)
                ),
                patch.object(
                    twelvelabs,
                    "_sync_candidate_analysis",
                    return_value=(_payload(0.82), "direct_url"),
                ) as analyze,
            ):
                first = twelvelabs.evaluate_candidate(
                    asset_id="cached",
                    video_url="https://videos.example/cached.mp4",
                    slot_index=1,
                    slot_duration=4,
                    narration_text="same narration",
                    search_query="query one",
                )
                second = twelvelabs.evaluate_candidate(
                    asset_id="cached",
                    video_url="https://videos.example/cached.mp4",
                    slot_index=2,
                    slot_duration=4,
                    narration_text="same narration",
                    search_query="different retrieval hint",
                )

        self.assertEqual(analyze.call_count, 1)
        self.assertFalse(first["_cache_hit"])
        self.assertTrue(second["_cache_hit"])

    def test_cache_miss_for_different_narration_calls_api_again(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    twelvelabs, "_candidate_cache_dir", return_value=Path(temp_dir)
                ),
                patch.object(
                    twelvelabs,
                    "_sync_candidate_analysis",
                    return_value=(_payload(0.82), "direct_url"),
                ) as analyze,
            ):
                for narration in ("first narration", "second narration"):
                    twelvelabs.evaluate_candidate(
                        asset_id="same-asset",
                        video_url="https://videos.example/same.mp4",
                        slot_index=1,
                        slot_duration=4,
                        narration_text=narration,
                        search_query="query",
                    )

        self.assertEqual(analyze.call_count, 2)

    def test_direct_url_analysis_does_not_create_asset(self):
        stub_types = type(sys)("twelvelabs.types")
        stub_types.AnalyzePromptV2 = lambda **kwargs: kwargs
        stub_types.SyncResponseFormat = lambda **kwargs: kwargs
        stub_types.VideoContext_AssetId = lambda **kwargs: kwargs
        stub_types.VideoContext_Url = lambda **kwargs: kwargs
        client = MagicMock()
        client.analyze.return_value = SimpleNamespace(data=_payload())

        with (
            patch.dict(sys.modules, {"twelvelabs.types": stub_types}),
            patch.object(twelvelabs, "_client", return_value=client),
        ):
            _, input_method = twelvelabs._sync_candidate_analysis(
                "https://videos.example/direct.mp4", "prompt"
            )

        self.assertEqual(input_method, "direct_url")
        client.assets.create.assert_not_called()

    def test_asset_fallback_is_used_only_for_explicit_direct_url_failure(self):
        class DirectUrlError(RuntimeError):
            status_code = 400

        stub_types = type(sys)("twelvelabs.types")
        stub_types.AnalyzePromptV2 = lambda **kwargs: kwargs
        stub_types.SyncResponseFormat = lambda **kwargs: kwargs
        stub_types.VideoContext_AssetId = lambda **kwargs: kwargs
        stub_types.VideoContext_Url = lambda **kwargs: kwargs
        client = MagicMock()
        client.analyze.side_effect = [
            DirectUrlError("failed to fetch video URL"),
            SimpleNamespace(data=_payload()),
        ]
        client.assets.create.return_value = SimpleNamespace(
            id="asset-1", status="ready"
        )

        with (
            patch.dict(sys.modules, {"twelvelabs.types": stub_types}),
            patch.object(twelvelabs, "_client", return_value=client),
        ):
            _, input_method = twelvelabs._sync_candidate_analysis(
                "https://videos.example/fallback.mp4", "prompt"
            )

        self.assertEqual(input_method, "asset_fallback")
        client.assets.create.assert_called_once_with(
            method="url", url="https://videos.example/fallback.mp4"
        )
        client.assets.delete.assert_called_once_with("asset-1", force=True)

        client.reset_mock()
        client.analyze.side_effect = DirectUrlError("invalid JSON schema")
        with (
            patch.dict(sys.modules, {"twelvelabs.types": stub_types}),
            patch.object(twelvelabs, "_client", return_value=client),
            self.assertRaises(DirectUrlError),
        ):
            twelvelabs._sync_candidate_analysis(
                "https://videos.example/no-fallback.mp4", "prompt"
            )
        client.assets.create.assert_not_called()

    def test_temporal_parser_selects_middle_match_and_rejects_ungated_segments(self):
        response = SimpleNamespace(
            result=SimpleNamespace(
                finish_reason="stop",
                data=json.dumps(
                    {
                        "best_visual_match": [
                            {
                                "start_time": 0,
                                "end_time": 4,
                                "metadata": {
                                    "match_quality": 0.72,
                                    "action_visible": False,
                                    "subject_visible": True,
                                    "description": "Related subject, wrong action",
                                },
                            },
                            {
                                "start_time": 9,
                                "end_time": 13,
                                "metadata": {
                                    "match_quality": 0.95,
                                    "action_visible": True,
                                    "subject_visible": True,
                                    "description": "The required action occurs",
                                },
                            },
                        ]
                    }
                ),
            )
        )

        segment = twelvelabs._parse_temporal_segments(
            response,
            source_duration=20,
            requested_source_duration=4,
        )

        self.assertEqual(segment["source_start_time"], 9.0)
        self.assertEqual(segment["source_end_time"], 13.0)
        response.result.data = json.dumps(
            {
                "best_visual_match": [
                    {
                        "start_time": 0,
                        "end_time": 4,
                        "metadata": {
                            "match_quality": 0.99,
                            "action_visible": False,
                            "subject_visible": True,
                        },
                    }
                ]
            }
        )
        self.assertIsNone(
            twelvelabs._parse_temporal_segments(
                response,
                source_duration=20,
                requested_source_duration=4,
            )
        )

    def test_missing_key_is_a_clear_preflight_error(self):
        config.app["twelvelabs_api_keys"] = []
        error = twelvelabs.validate_smart_visual_matching_configuration()
        self.assertIn("API key is missing", error)

    def test_api_key_is_absent_from_safe_error_and_material_artifact(self):
        secret = "test-key-should-not-appear"
        safe_reason = twelvelabs._safe_api_failure_reason(RuntimeError(secret))
        candidate = _candidate(1)
        candidate.source_info["api_key"] = secret
        candidate.semantic_evaluation = {
            "provider": "twelvelabs",
            "overall_score": 0.8,
            "api_key": secret,
        }
        serialized = json.dumps(
            material._material_source_record(candidate, "winner.mp4")
        )

        self.assertNotIn(secret, safe_reason)
        self.assertNotIn(secret, serialized)


class TestSmartMaterialPipeline(unittest.TestCase):
    def test_cheap_filters_precede_twelvelabs_and_long_clips_are_deprioritized(self):
        items = [
            _candidate("short", duration=12),
            _candidate("long", duration=45),
            _candidate("wrong-orientation", width=1920, height=1080),
            _candidate("low-resolution", width=360, height=640),
            _candidate("too-short", duration=3),
            _candidate("bad-url", url="http://videos.example/bad.mp4"),
            _candidate("duplicate", url="https://videos.example/short.mp4"),
        ]
        items[-1].source_info["asset_id"] = "short"

        prepared = material._prepare_twelvelabs_candidates(
            items,
            video_aspect=VideoAspect.portrait,
            required_source_duration=4,
            preferred_max_source_duration=30,
        )

        self.assertEqual(
            [item.source_info["asset_id"] for item in prepared],
            ["short", "long"],
        )

    def test_only_winner_is_segmented_downloaded_and_persisted(self):
        candidates = [_candidate(1), _candidate(2)]
        slot = _visual_slot()
        selector = MagicMock()

        def select_best(**kwargs):
            winner = kwargs["candidates"][1]
            winner.overall_score = 0.94
            winner.semantic_evaluation = {
                "provider": "twelvelabs",
                "accepted": True,
                "overall_score": 0.94,
            }
            return winner, {
                "api_candidates_analyzed": 2,
                "source_seconds_analyzed": 24,
            }

        selector.side_effect = select_best
        segmenter = MagicMock(
            return_value={
                "source_start_time": 6.5,
                "source_end_time": 10.5,
                "description": "required action",
            }
        )
        service = SimpleNamespace(
            candidate_selection_settings=lambda: {
                "batch_size": 5,
                "max_candidates": 15,
                "minimum_score": 0.7,
                "strong_early_stop_score": 0.9,
                "preferred_max_source_duration": 30,
                "concurrency": 5,
                "fail_closed": True,
            },
            select_best_candidate=selector,
            segment_winner=segmenter,
        )

        with (
            patch.object(
                material,
                "save_video",
                return_value="D:/task/winner.mp4",
            ) as save,
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ) as persist,
        ):
            paths = material._download_videos_by_script_order_smart(
                task_id="smart-task",
                search_terms=["worker removing rotten boards"],
                visual_slots=[slot],
                search_videos=lambda **kwargs: candidates,
                video_aspect=VideoAspect.portrait,
                max_clip_duration=4,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )

        self.assertEqual(paths, ["D:/task/winner.mp4"])
        self.assertEqual(segmenter.call_count, 1)
        self.assertEqual(
            segmenter.call_args.kwargs["video_url"],
            "https://videos.example/2.mp4",
        )
        self.assertEqual(save.call_count, 1)
        records = persist.call_args.kwargs["material_sources"]
        self.assertEqual(records[0]["source_start_time"], 6.5)
        self.assertEqual(records[0]["source_end_time"], 10.5)
        self.assertEqual(records[0]["slot_index"], 1)

    def test_no_accepted_winner_does_not_cross_segment_fallback(self):
        candidate = _candidate(1)
        slot = _visual_slot()
        service = SimpleNamespace(
            candidate_selection_settings=lambda: {
                "batch_size": 5,
                "max_candidates": 15,
                "minimum_score": 0.7,
                "strong_early_stop_score": 0.9,
                "preferred_max_source_duration": 30,
                "concurrency": 5,
                "fail_closed": True,
            },
            select_best_candidate=MagicMock(
                return_value=(
                    None,
                    {
                        "api_candidates_analyzed": 1,
                        "source_seconds_analyzed": 12,
                    },
                )
            ),
            segment_winner=MagicMock(),
        )
        with (
            patch.object(material, "save_video") as save,
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ),
            self.assertRaisesRegex(
                material.SmartMaterialSelectionError,
                "cross-segment fallback was not used",
            ),
        ):
            material._download_videos_by_script_order_smart(
                task_id="no-fallback",
                search_terms=["worker removing rotten boards"],
                visual_slots=[slot],
                search_videos=lambda **kwargs: [candidate],
                video_aspect=VideoAspect.portrait,
                max_clip_duration=4,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )

        service.segment_winner.assert_not_called()
        save.assert_not_called()

    def test_twelvelabs_disabled_keeps_legacy_ordered_path(self):
        with (
            patch.object(
                twelvelabs, "is_smart_visual_matching_enabled", return_value=False
            ),
            patch.object(material, "_download_videos_by_script_order_smart") as smart,
        ):
            result = material._download_videos_by_script_order(
                task_id="legacy",
                search_terms=["query"],
                search_videos=lambda **kwargs: [],
                video_aspect=VideoAspect.portrait,
                audio_duration=4,
                max_clip_duration=4,
                material_directory="",
                visual_slots=[_visual_slot()],
            )

        self.assertEqual(result, [])
        smart.assert_not_called()

    def test_ordered_matching_off_does_not_enter_smart_path(self):
        candidate = _candidate(1)
        with (
            patch.dict(config.app, {"material_directory": ""}),
            patch.object(material, "search_videos_pexels", return_value=[candidate]),
            patch.object(
                material.material_cache, "load_material_search_cache", return_value=None
            ),
            patch.object(material.material_cache, "save_material_search_cache"),
            patch.object(material, "save_video", return_value="D:/task/legacy.mp4"),
            patch.object(material, "_download_videos_by_script_order_smart") as smart,
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ),
        ):
            result = material.download_videos(
                task_id="unordered",
                search_terms=["query"],
                source="pexels",
                video_aspect=VideoAspect.portrait,
                audio_duration=4,
                max_clip_duration=4,
                match_script_order=False,
                visual_slots=[_visual_slot()],
            )

        self.assertEqual(result, ["D:/task/legacy.mp4"])
        smart.assert_not_called()

    def test_saved_ranges_are_loaded_in_download_order(self):
        payload = {
            "material_sources": [
                {
                    "local_file": "second.mp4",
                    "source_start_time": 8.0,
                    "source_end_time": 12.0,
                },
                {
                    "local_file": "first.mp4",
                    "source_start_time": 1.5,
                    "source_end_time": 5.5,
                },
            ]
        }
        with patch.object(
            material.task_artifacts, "read_script_data", return_value=payload
        ):
            ranges = material.load_selected_source_ranges(
                "task-id",
                ["D:/task/first.mp4", "D:/task/second.mp4"],
            )

        self.assertEqual(ranges, [(1.5, 5.5), (8.0, 12.0)])

    def test_generate_final_videos_passes_ranges_to_renderer(self):
        params = SimpleNamespace(
            bgm_type="",
            bgm_volume=0,
            match_materials_to_script=True,
            video_count=1,
            video_concat_mode=task.VideoConcatMode.random,
            video_transition_mode=None,
            video_aspect=VideoAspect.portrait,
            video_clip_duration=4,
            n_threads=2,
            video_clip_speed=1.0,
        )
        selected_ranges = [(11.4, 15.2)]
        with (
            patch.object(task.utils, "task_dir", return_value="D:/task"),
            patch.object(task.video, "combine_videos") as combine,
            patch.object(task.video, "generate_video", return_value=True),
            patch.object(task.sm.state, "update_task"),
        ):
            task.generate_final_videos(
                "task-id",
                params,
                ["D:/task/winner.mp4"],
                "D:/task/audio.mp3",
                "",
                4.0,
                source_ranges=selected_ranges,
            )

        self.assertEqual(combine.call_args.kwargs["source_ranges"], selected_ranges)
