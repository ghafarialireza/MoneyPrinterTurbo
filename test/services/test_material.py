import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import material


class TestMaterialTlsVerification(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)

    def test_search_pexels_uses_tls_verification_by_default(self):
        """
        默认路径必须开启 TLS 校验，避免素材 API key 和返回的素材 URL
        在公共网络或不可信代理环境中被中间人攻击截获或篡改。
        """
        config.app["pexels_api_keys"] = ["pexels-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "videos": [
                    {
                        "id": 321,
                        "url": "https://www.pexels.com/video/example-321/?token=drop",
                        "image": "https://images.pexels.com/videos/321/preview.jpg?size=large",
                        "duration": 8,
                        "user": {
                            "id": 654,
                            "name": "Pexels Creator",
                            "url": "https://www.pexels.com/@creator/?key=drop",
                        },
                        "video_files": [
                            {
                                "id": 987,
                                "width": 1080,
                                "height": 1920,
                                "link": "https://example.com/video.mp4",
                            }
                        ],
                    }
                ]
            }
        )

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ) as get:
            results = material.search_videos_pexels("cat", minimum_duration=1)

        self.assertEqual(len(results), 1)
        self.assertTrue(get.call_args.kwargs["verify"])
        self.assertEqual(results[0].source_info["asset_id"], "321")
        self.assertEqual(
            results[0].source_info["source_page"],
            "https://www.pexels.com/video/example-321/",
        )
        self.assertEqual(
            results[0].source_info["creator"]["profile_page"],
            "https://www.pexels.com/@creator/",
        )
        self.assertEqual(results[0].source_info["rendition"]["id"], "987")
        self.assertEqual(results[0].provider_asset_id, "321")
        self.assertEqual(
            results[0].preview_url,
            "https://images.pexels.com/videos/321/preview.jpg",
        )
        self.assertEqual(results[0].width, 1080)
        self.assertEqual(results[0].height, 1920)
        self.assertEqual(results[0].orientation, "portrait")
        self.assertEqual(results[0].rendition_id, "987")
        self.assertEqual(results[0].search_query, "cat")
        self.assertEqual(results[0].query_attempt, 1)
        self.assertEqual(
            results[0].source_page_url,
            "https://www.pexels.com/video/example-321/",
        )
        self.assertIn("per_page=80", get.call_args.args[0])

    def test_search_pexels_accepts_best_hd_rendition_without_exact_full_hd(self):
        config.app["pexels_api_keys"] = ["pexels-key"]
        fake_response = SimpleNamespace(
            json=lambda: {
                "videos": [
                    {
                        "id": 321,
                        "url": "https://www.pexels.com/video/example-321/",
                        "duration": 8,
                        "user": {"id": 654, "name": "Pexels Creator"},
                        "video_files": [
                            {
                                "id": 1,
                                "width": 360,
                                "height": 640,
                                "link": "https://example.com/low.mp4",
                            },
                            {
                                "id": 2,
                                "width": 720,
                                "height": 1280,
                                "link": "https://example.com/hd.mp4",
                            },
                            {
                                "id": 3,
                                "width": 1280,
                                "height": 720,
                                "link": "https://example.com/wrong-orientation.mp4",
                            },
                        ],
                    }
                ]
            }
        )

        with patch("app.services.material.requests.get", return_value=fake_response):
            results = material.search_videos_pexels("railway", minimum_duration=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/hd.mp4")
        self.assertEqual(
            results[0].source_info["rendition"],
            {"id": "2", "width": 720, "height": 1280},
        )

    def test_search_pixabay_allows_explicit_tls_disable_for_proxy(self):
        """
        少数企业代理会使用自签证书。该场景必须显式配置关闭 TLS 校验，
        不能再由代码硬编码默认关闭。
        """
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.app["tls_verify"] = False
        config.proxy.clear()

        fake_response = SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            text="",
            json=lambda: {
                "hits": [
                    {
                        "duration": 8,
                        "videos": {
                            "large": {
                                "width": 1920,
                                "height": 1080,
                                "url": "https://example.com/video.mp4",
                            }
                        },
                    }
                ]
            },
        )

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ) as get:
            results = material.search_videos_pixabay(
                "cat",
                minimum_duration=1,
                video_aspect=material.VideoAspect.landscape,
            )

        self.assertEqual(len(results), 1)
        self.assertFalse(get.call_args.kwargs["verify"])

    def test_remote_searches_only_return_requested_orientation(self):
        """
        两个素材源都必须只返回目标方向的素材，避免竖屏任务混入横屏素材后
        通过 letterbox 产生明显黑边。Pexels 使用远端参数并在本地校验，
        Pixabay 使用响应尺寸做本地过滤。
        """
        config.app["pexels_api_keys"] = ["pexels-key"]
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.proxy.clear()

        pexels_response = SimpleNamespace(
            json=lambda: {
                "videos": [
                    {
                        "id": 1,
                        "duration": 8,
                        "video_files": [
                            {
                                "id": 11,
                                "width": 1920,
                                "height": 1080,
                                "link": "https://example.com/landscape.mp4",
                            }
                        ],
                    },
                    {
                        "id": 2,
                        "duration": 8,
                        "video_files": [
                            {
                                "id": 22,
                                "width": 1080,
                                "height": 1920,
                                "link": "https://example.com/portrait.mp4",
                            }
                        ],
                    },
                ]
            }
        )
        pixabay_response = SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            text="",
            json=lambda: {
                "hits": [
                    {
                        "id": 1,
                        "duration": 8,
                        "videos": {
                            "large": {
                                "width": 1920,
                                "height": 1080,
                                "url": "https://example.com/landscape.mp4",
                            }
                        },
                    },
                    {
                        "id": 2,
                        "duration": 8,
                        "videos": {
                            "large": {
                                "width": 1080,
                                "height": 1920,
                                "url": "https://example.com/portrait.mp4",
                            }
                        },
                    },
                ]
            },
        )
        with patch(
            "app.services.material.requests.get",
            return_value=pexels_response,
        ) as get:
            pexels_results = material.search_videos_pexels(
                "city",
                minimum_duration=1,
                video_aspect=material.VideoAspect.portrait,
            )
            pexels_url = get.call_args.args[0]
        with patch(
            "app.services.material.requests.get",
            return_value=pixabay_response,
        ):
            pixabay_results = material.search_videos_pixabay(
                "city",
                minimum_duration=1,
                video_aspect=material.VideoAspect.portrait,
            )
        self.assertIn("/v1/videos/search?", pexels_url)
        self.assertIn("orientation=portrait", pexels_url)
        for results in (pexels_results, pixabay_results):
            self.assertEqual(
                [item.url for item in results],
                ["https://example.com/portrait.mp4"],
            )

    def test_video_aspect_matching_rejects_unknown_dimensions(self):
        """无法确认方向的素材不能进入严格的横竖屏候选列表。"""
        self.assertTrue(
            material._matches_video_aspect(
                1080,
                1920,
                material.VideoAspect.portrait,
            )
        )
        self.assertFalse(
            material._matches_video_aspect(
                1920,
                1080,
                material.VideoAspect.portrait,
            )
        )
        self.assertTrue(
            material._matches_video_aspect(
                None,
                None,
                material.VideoAspect.portrait,
                is_vertical=True,
            )
        )
        self.assertFalse(
            material._matches_video_aspect(
                None,
                None,
                material.VideoAspect.portrait,
            )
        )
        self.assertTrue(
            material._matches_video_aspect(
                1080,
                1080,
                material.VideoAspect.square,
            )
        )
        self.assertFalse(
            material._matches_video_aspect(
                1080,
                1920,
                material.VideoAspect.square,
            )
        )

    def test_square_search_preserves_crop_compatible_materials(self):
        """
        Pixabay 很少提供原生方形视频。方形输出必须继续接受可裁剪的
        横屏素材，否则选择该来源时会在搜索阶段直接得到空列表。
        """
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.proxy.clear()
        pixabay_response = SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            text="",
            json=lambda: {
                "hits": [
                    {
                        "id": 1,
                        "duration": 8,
                        "videos": {
                            "large": {
                                "width": 1920,
                                "height": 1080,
                                "url": "https://example.com/pixabay-landscape.mp4",
                            }
                        },
                    }
                ]
            },
        )
        with patch(
            "app.services.material.requests.get",
            return_value=pixabay_response,
        ):
            pixabay_results = material.search_videos_pixabay(
                "city",
                minimum_duration=1,
                video_aspect=material.VideoAspect.square,
            )

        self.assertEqual(
            [item.url for item in pixabay_results],
            ["https://example.com/pixabay-landscape.mp4"],
        )

    def test_search_pixabay_does_not_log_api_key(self):
        config.app["pixabay_api_keys"] = ["pixabay-secret-key"]
        config.proxy.clear()

        fake_response = SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            text="",
            json=lambda: {"hits": []},
        )

        with (
            patch("app.services.material.requests.get", return_value=fake_response),
            patch("app.services.material.logger.info") as log,
        ):
            material.search_videos_pixabay("cat", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertNotIn("pixabay-secret-key", logged_messages)

    def test_search_pixabay_reports_cloudflare_challenge(self):
        """
        Cloudflare Challenge 返回的是 HTML，不是 Pixabay API 的 JSON。
        应直接说明服务端拦截原因，避免用户只看到没有上下文的 JSON 解析错误。
        """
        config.app["pixabay_api_keys"] = ["pixabay-secret-key"]
        config.proxy.clear()

        fake_response = SimpleNamespace(
            status_code=429,
            headers={
                "content-type": "text/html; charset=UTF-8",
                "cf-mitigated": "challenge",
                "cf-ray": "test-ray",
            },
            text="<html><title>Just a moment...</title></html>",
        )

        with (
            patch("app.services.material.requests.get", return_value=fake_response),
            patch("app.services.material.logger.error") as log,
        ):
            results = material.search_videos_pixabay("nature", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertEqual(results, [])
        self.assertIn("Cloudflare challenge", logged_messages)
        self.assertIn("cf_ray=test-ray", logged_messages)
        self.assertNotIn("pixabay-secret-key", logged_messages)
        self.assertNotIn("Just a moment", logged_messages)

    def test_search_pixabay_reports_api_rate_limit(self):
        """
        Pixabay 自身的 429 限流与 Cloudflare HTML Challenge 是不同问题。
        保留 Retry-After 可以帮助用户判断何时重试，同时不记录响应正文。
        """
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.proxy.clear()

        fake_response = SimpleNamespace(
            status_code=429,
            headers={
                "content-type": "text/plain; charset=UTF-8",
                "retry-after": "60",
            },
            text="API rate limit exceeded",
        )

        with (
            patch("app.services.material.requests.get", return_value=fake_response),
            patch("app.services.material.logger.error") as log,
        ):
            results = material.search_videos_pixabay("nature", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertEqual(results, [])
        self.assertIn("API rate limit exceeded", logged_messages)
        self.assertIn("retry_after=60", logged_messages)

    def test_search_pixabay_reports_non_json_response(self):
        """
        即使状态码为 200，上游代理也可能返回登录页或其他非 JSON 内容。
        该场景应记录响应类型，而不是向外暴露底层 JSONDecodeError。
        """
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.proxy.clear()

        def raise_invalid_json():
            raise ValueError("Expecting value: line 1 column 1")

        fake_response = SimpleNamespace(
            status_code=200,
            headers={"content-type": "text/plain"},
            text="unexpected response",
            json=raise_invalid_json,
        )

        with (
            patch("app.services.material.requests.get", return_value=fake_response),
            patch("app.services.material.logger.error") as log,
        ):
            results = material.search_videos_pixabay("nature", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertEqual(results, [])
        self.assertIn("unexpected non-JSON response", logged_messages)
        self.assertNotIn("Expecting value", logged_messages)

    def test_search_pixabay_redacts_api_key_from_network_error(self):
        """
        requests 的连接异常可能回显完整请求 URL。异常详情仍应保留用于排查，
        但 URL 查询参数中的 Pixabay API Key 必须在写入日志前脱敏。
        """
        api_key = "pixabay-secret-key"
        config.app["pixabay_api_keys"] = [api_key]
        config.proxy.clear()
        error = requests.ConnectionError(
            f"request failed for https://pixabay.com/api/videos/?q=nature&key={api_key}"
        )

        with (
            patch("app.services.material.requests.get", side_effect=error),
            patch("app.services.material.logger.error") as log,
        ):
            results = material.search_videos_pixabay("nature", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertEqual(results, [])
        self.assertIn("ConnectionError", logged_messages)
        self.assertIn("key=***", logged_messages)
        self.assertNotIn(api_key, logged_messages)

    def test_search_pixabay_redacts_proxy_credentials_from_network_error(self):
        """
        代理连接异常可能回显含认证信息的完整代理 URL。日志应保留异常类型，
        但不能把代理用户名和密码持久化到日志文件。
        """
        proxy_url = "http://proxy-user:proxy-password@proxy.example.com:8080"
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.proxy.clear()
        config.proxy["http"] = proxy_url
        error = requests.exceptions.ProxyError(
            f"failed to connect to proxy {proxy_url}"
        )

        with (
            patch("app.services.material.requests.get", side_effect=error),
            patch("app.services.material.logger.error") as log,
        ):
            results = material.search_videos_pixabay("nature", minimum_duration=1)

        logged_messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertEqual(results, [])
        self.assertIn("ProxyError", logged_messages)
        self.assertNotIn("proxy-user", logged_messages)
        self.assertNotIn("proxy-password", logged_messages)

    def test_save_video_uses_tls_verification_by_default(self):
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(content=b"fake-video")

        class FakeVideoFileClip:
            duration = 1
            fps = 24

            def __init__(self, path):
                self.path = path

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "app.services.material.requests.get", return_value=fake_response
                ) as get,
                patch("app.services.material.VideoFileClip", FakeVideoFileClip),
            ):
                video_path = material.save_video(
                    "https://example.com/video.mp4?token=abc", save_dir=temp_dir
                )

            self.assertTrue(os.path.exists(video_path))
            self.assertTrue(get.call_args.kwargs["verify"])

    def test_save_video_rejects_wrong_orientation_after_decode(self):
        config.app.pop("tls_verify", None)
        config.proxy.clear()
        fake_response = SimpleNamespace(content=b"fake-landscape-video")

        class FakeLandscapeVideoFileClip:
            duration = 5
            fps = 30
            w = 1920
            h = 1080

            def __init__(self, path):
                self.path = path

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("app.services.material.requests.get", return_value=fake_response),
                patch(
                    "app.services.material.VideoFileClip", FakeLandscapeVideoFileClip
                ),
            ):
                video_path = material.save_video(
                    "https://example.com/landscape.mp4",
                    save_dir=temp_dir,
                    video_aspect=material.VideoAspect.portrait,
                )

            self.assertEqual(video_path, "")
            self.assertEqual(os.listdir(temp_dir), [])

    def test_download_videos_accepts_plain_string_concat_mode(self):
        """
        download_videos 可能被服务层或测试直接传入字符串模式，而不是
        VideoConcatMode 枚举。这里用空搜索词避免真实网络请求，只验证
        字符串 "random" 不会再因为访问 `.value` 抛 AttributeError。
        """
        result = material.download_videos(
            task_id="string-concat-mode",
            search_terms=[],
            video_concat_mode="random",
        )

        self.assertEqual(result, [])

    def test_material_source_record_uses_public_whitelist(self):
        """
        任务清单只应包含可追溯的公开字段，不能写入签名参数、下载地址、
        调用方传入的额外字段或本机绝对路径。
        """
        item = material.MaterialInfo(
            provider="pixabay",
            url="https://cdn.example.com/video.mp4?token=secret",
            duration=12,
            source_info={
                "provider": "pixabay",
                "search_term": "city",
                "asset_id": 123,
                "source_page": "https://pixabay.com/videos/city-123/?key=secret",
                "creator": {
                    "id": 456,
                    "name": "Creator",
                    "profile_page": "https://pixabay.com/users/creator/?token=secret",
                    "email": "private@example.com",
                },
                "rendition": {
                    "id": "large",
                    "width": 1920,
                    "height": 1080,
                    "download_url": "https://cdn.example.com/private",
                },
                "api_key": "must-not-persist",
            },
        )

        record = material._material_source_record(
            item,
            "/Users/example/private/task/vid-123.mp4",
        )
        serialized = str(record)

        self.assertEqual(record["local_file"], "vid-123.mp4")
        self.assertEqual(
            record["source_page"],
            "https://pixabay.com/videos/city-123/",
        )
        self.assertEqual(
            record["creator"]["profile_page"],
            "https://pixabay.com/users/creator/",
        )
        self.assertEqual(
            record["rendition"],
            {"id": "large", "width": 1920, "height": 1080},
        )
        self.assertEqual(record["provider"], "pixabay")
        self.assertEqual(record["provider_asset_id"], "123")
        self.assertEqual(record["asset_id"], "123")
        self.assertNotIn("secret", serialized)
        self.assertNotIn("/Users/example", serialized)
        self.assertNotIn("private@example.com", serialized)

    def test_material_source_record_keeps_the_evidence_behind_the_window(self):
        """The manifest has to say how good the match was and what played.

        ``source_start_time``/``source_end_time`` are the padded window the
        renderer plays; on their own they cannot answer why a beat looked wrong,
        because nothing records how much of that window the model actually
        described or what the local probes measured on the file. Both are kept
        here, trimmed: a reel can have dozens of cuts and the manifest only needs
        enough of them to explain why the window moved.
        """
        item = material.MaterialInfo(
            provider="pinterest",
            url="https://v.pinimg.com/videos/clip.mp4",
            duration=15,
            source_start_time=7.6,
            source_end_time=10.988,
            source_info={
                "provider": "pinterest",
                "asset_id": "998877",
                "temporal_segment": {
                    "source_start_time": 7.6,
                    "source_end_time": 10.988,
                    "verified_start_time": 7.013,
                    "verified_end_time": 8.9,
                    "padded_seconds": 1.501,
                    "match_quality": 0.8123,
                    "action_visible": True,
                    "subject_visible": False,
                    "description": "hands shaping clay\non a wheel",
                    "raw_provider_payload": {"token": "must-not-persist"},
                },
                "local_clip_check": {
                    "shot_cuts": [round(0.5 * step, 3) for step in range(1, 61)],
                    "shot_containment": "shifted",
                    "shifted_seconds": 0.587,
                    "burned_in_overlay": {
                        "zone": "bottom",
                        "density": 0.1494,
                        "middle_density": 0.0252,
                        "transitions": 5.3,
                        "frames_sampled": 8,
                    },
                },
            },
        )

        record = material._material_source_record(item, "/tmp/task/vid-998877.mp4")

        segment = record["temporal_segment"]
        self.assertEqual(segment["match_quality"], 0.8123)
        self.assertEqual(segment["verified_start_time"], 7.013)
        self.assertEqual(segment["verified_end_time"], 8.9)
        self.assertEqual(segment["padded_seconds"], 1.501)
        self.assertTrue(segment["action_visible"])
        self.assertFalse(segment["subject_visible"])
        # A control character in a model's prose must not reach the manifest.
        self.assertEqual(segment["description"], "hands shaping clay on a wheel")
        self.assertNotIn("must-not-persist", str(record))

        check = record["local_clip_check"]
        self.assertEqual(check["shot_cut_count"], 60)
        self.assertEqual(len(check["shot_cuts"]), 24)
        self.assertEqual(check["shot_containment"], "shifted")
        self.assertEqual(check["shifted_seconds"], 0.587)
        self.assertEqual(check["burned_in_overlay"]["density"], 0.1494)
        self.assertEqual(check["burned_in_overlay"]["zone"], "bottom")

    def test_material_source_record_survives_probes_that_could_not_run(self):
        """An unmeasurable clip records that, and records nothing it did not measure."""
        item = material.MaterialInfo(
            provider="pexels",
            url="https://videos.example/clip.mp4",
            duration=9,
            source_info={
                "provider": "pexels",
                "asset_id": "42",
                "temporal_segment": {"source_start_time": 0.0, "source_end_time": 3.0},
                "local_clip_check": {
                    "shot_cuts": "unavailable",
                    "burned_in_overlay": "unavailable",
                },
            },
        )

        record = material._material_source_record(item, "/tmp/task/vid-42.mp4")

        self.assertNotIn("temporal_segment", record)
        self.assertEqual(
            record["local_clip_check"],
            {"shot_cuts": "unavailable", "burned_in_overlay": "unavailable"},
        )

    def test_normalized_window_restates_padding_against_the_duration_it_used(self):
        """Padding is only meaningful next to the length the window was cut for.

        ``padded_seconds`` arrives computed against the length the segmentation
        call asked for. Whenever a window is re-derived against a different
        length, that number is stale, and a stale one under-reports how much of
        the shipped window nothing ever described.
        """
        normalized = material._normalize_selected_source_range(
            {
                "source_start_time": 3.0,
                "source_end_time": 9.0,
                "verified_start_time": 5.0,
                "verified_end_time": 7.0,
                "padded_seconds": 4.0,
            },
            source_duration=20.0,
            required_source_duration=5.6,
        )

        self.assertEqual(normalized["padded_seconds"], 3.6)
        self.assertAlmostEqual(
            normalized["source_end_time"] - normalized["source_start_time"], 5.6
        )
        # The described interval is untouched: only the window around it moved.
        self.assertEqual(normalized["verified_start_time"], 5.0)
        self.assertEqual(normalized["verified_end_time"], 7.0)

    def test_normalized_window_leaves_padding_alone_when_nothing_was_verified(self):
        """A segment from the fail-open path has no verified interval to measure."""
        normalized = material._normalize_selected_source_range(
            {"source_start_time": 0.0, "source_end_time": 8.0},
            source_duration=8.0,
            required_source_duration=3.0,
        )

        self.assertNotIn("padded_seconds", normalized)

    def test_download_videos_distributes_terms_monotonically_in_script_order(self):
        """
        三个镜头覆盖两个关键词时，第一个关键词连续覆盖前两个镜头，随后
        进入第二个关键词；不能在进入后段后再次回到开头。
        """
        search_results = {
            "opening city": [
                material.MaterialInfo(
                    provider="pexels",
                    url="https://v.example/a1.mp4",
                    duration=3,
                    source_info={
                        "provider": "pexels",
                        "search_term": "opening city",
                        "asset_id": "a1",
                    },
                ),
                material.MaterialInfo(
                    provider="pexels",
                    url="https://v.example/a2.mp4",
                    duration=3,
                    source_info={
                        "provider": "pexels",
                        "search_term": "opening city",
                        "asset_id": "a2",
                    },
                ),
            ],
            "middle office": [
                material.MaterialInfo(
                    provider="pexels",
                    url="https://v.example/b1.mp4",
                    duration=3,
                    source_info={
                        "provider": "pexels",
                        "search_term": "middle office",
                        "asset_id": "b1",
                    },
                ),
                material.MaterialInfo(
                    provider="pexels",
                    url="https://v.example/b2.mp4",
                    duration=3,
                    source_info={
                        "provider": "pexels",
                        "search_term": "middle office",
                        "asset_id": "b2",
                    },
                ),
            ],
        }
        downloaded_urls = []

        def fake_search(search_term, minimum_duration, video_aspect):
            return search_results[search_term]

        def fake_save_video(video_url, save_dir="", video_aspect=None):
            downloaded_urls.append(video_url)
            return f"/tmp/{video_url.rsplit('/', 1)[-1]}"

        with (
            patch.dict(
                config.app,
                {
                    "material_directory": "",
                    "twelvelabs_clip_qa": False,
                },
            ),
            patch.object(material, "search_videos_pexels", side_effect=fake_search),
            patch.object(material, "save_video", side_effect=fake_save_video),
            patch.object(
                material.material_cache,
                "load_material_search_cache",
                return_value=None,
            ),
            patch.object(material.material_cache, "save_material_search_cache"),
            patch.object(
                material.task_artifacts,
                "patch_script_data",
                return_value=True,
            ) as patch_script,
        ):
            result = material.download_videos(
                task_id="ordered-materials",
                search_terms=["opening city", "middle office"],
                source="pexels",
                audio_duration=7,
                max_clip_duration=3,
                match_script_order=True,
            )

        self.assertEqual(
            downloaded_urls,
            [
                "https://v.example/a1.mp4",
                "https://v.example/a2.mp4",
                "https://v.example/b1.mp4",
            ],
        )
        self.assertEqual(result, ["/tmp/a1.mp4", "/tmp/a2.mp4", "/tmp/b1.mp4"])
        recorded_sources = patch_script.call_args.kwargs["material_sources"]
        self.assertEqual(
            [source["asset_id"] for source in recorded_sources],
            ["a1", "a2", "b1"],
        )
        self.assertEqual(
            [source["local_file"] for source in recorded_sources],
            ["a1.mp4", "a2.mp4", "b1.mp4"],
        )

    def test_script_order_semantic_qa_rejects_candidate_then_uses_same_term(self):
        candidates = [
            material.MaterialInfo(
                provider="pexels",
                url="https://v.example/unrelated.mp4",
                duration=5,
                source_info={"provider": "pexels", "asset_id": "bad"},
            ),
            material.MaterialInfo(
                provider="pexels",
                url="https://v.example/ballast.mp4",
                duration=5,
                source_info={"provider": "pexels", "asset_id": "good"},
            ),
        ]
        minimum_durations = []

        def fake_search(search_term, minimum_duration, video_aspect):
            minimum_durations.append(minimum_duration)
            return candidates

        qa_results = [
            {
                "provider": "twelvelabs",
                "accepted": False,
                "score": 0.2,
                "reason": "A train is visible but the ballast is not.",
            },
            {
                "provider": "twelvelabs",
                "accepted": True,
                "score": 0.91,
                "reason": "Ballast stones are clearly visible under the rails.",
            },
        ]

        with (
            patch.dict(
                config.app,
                {
                    "material_directory": "",
                    "twelvelabs_clip_qa_fail_closed": True,
                },
            ),
            patch.object(material, "search_videos_pexels", side_effect=fake_search),
            patch.object(
                material,
                "save_video",
                return_value="/tmp/ballast.mp4",
            ) as save,
            patch.object(
                material.material_cache,
                "load_material_search_cache",
                return_value=None,
            ),
            patch.object(material.material_cache, "save_material_search_cache"),
            patch.object(
                material.task_artifacts,
                "patch_script_data",
                return_value=True,
            ) as patch_script,
            patch(
                "app.services.twelvelabs.is_clip_qa_enabled",
                return_value=True,
            ),
            patch(
                "app.services.twelvelabs.evaluate_clip_match",
                side_effect=qa_results,
            ) as evaluate,
        ):
            result = material.download_videos(
                task_id="semantic-qa",
                search_terms=["railway ballast closeup"],
                source="pexels",
                audio_duration=3,
                max_clip_duration=3,
                match_script_order=True,
            )

        self.assertEqual(result, ["/tmp/ballast.mp4"])
        self.assertEqual(minimum_durations, [4])
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(
            save.call_args.kwargs["video_url"],
            "https://v.example/ballast.mp4",
        )
        recorded = patch_script.call_args.kwargs["material_sources"][0]
        self.assertEqual(recorded["asset_id"], "good")
        self.assertEqual(recorded["semantic_qa"]["score"], 0.91)
        self.assertTrue(recorded["semantic_qa"]["accepted"])

    def test_script_order_semantic_qa_failure_rejects_in_quality_mode(self):
        item = material.MaterialInfo(
            provider="pexels",
            url="https://v.example/unverified.mp4",
            duration=5,
            source_info={"provider": "pexels", "asset_id": "unverified"},
        )

        with (
            patch.dict(
                config.app,
                {
                    "material_directory": "",
                    "twelvelabs_clip_qa_fail_closed": True,
                },
            ),
            patch.object(material, "search_videos_pexels", return_value=[item]),
            patch.object(material, "save_video") as save,
            patch.object(
                material.material_cache,
                "load_material_search_cache",
                return_value=None,
            ),
            patch.object(material.material_cache, "save_material_search_cache"),
            patch.object(
                material.task_artifacts,
                "patch_script_data",
                return_value=True,
            ),
            patch(
                "app.services.twelvelabs.is_clip_qa_enabled",
                return_value=True,
            ),
            patch(
                "app.services.twelvelabs.evaluate_clip_match",
                return_value=None,
            ),
        ):
            result = material.download_videos(
                task_id="semantic-qa-unavailable",
                search_terms=["railway ballast closeup"],
                source="pexels",
                audio_duration=3,
                max_clip_duration=3,
                match_script_order=True,
            )

        self.assertEqual(result, [])
        save.assert_not_called()

    def test_script_order_term_plan_never_rewinds_for_long_narration(self):
        self.assertEqual(
            material._build_script_order_term_plan(
                term_count=8,
                required_clip_count=14,
            ),
            [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7],
        )

    def test_script_order_term_plan_samples_script_end_when_clips_are_few(self):
        self.assertEqual(
            material._build_script_order_term_plan(
                term_count=8,
                required_clip_count=2,
            ),
            [0, 7],
        )

    def test_material_source_persistence_failure_does_not_break_download(self):
        """辅助任务记录失败时，已经下载成功的素材仍应正常返回给成片主流程。"""
        item = material.MaterialInfo(
            provider="pexels",
            url="https://v.example/a1.mp4",
            duration=5,
            source_info={"provider": "pexels", "asset_id": "a1"},
        )

        with (
            patch.dict(config.app, {"material_directory": ""}),
            patch.object(material, "search_videos_pexels", return_value=[item]),
            patch.object(material, "save_video", return_value="/tmp/a1.mp4"),
            patch.object(
                material.material_cache,
                "load_material_search_cache",
                return_value=None,
            ),
            patch.object(material.material_cache, "save_material_search_cache"),
            patch.object(
                material.task_artifacts,
                "patch_script_data",
                side_effect=OSError("disk unavailable"),
            ),
            patch.object(material.logger, "warning") as warning,
        ):
            result = material.download_videos(
                task_id="persist-failure",
                search_terms=["city"],
                source="pexels",
                audio_duration=1,
                max_clip_duration=5,
            )

        self.assertEqual(result, ["/tmp/a1.mp4"])
        self.assertTrue(warning.called)


class TestPinterestMaterialSearch(unittest.TestCase):
    """``search_videos_pinterest`` as the cascade's first, keyless provider."""

    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)

    @staticmethod
    def _record(**overrides):
        record = {
            "pin_id": "9001",
            "pin_url": "https://www.pinterest.com/pin/9001/",
            "duration": 8.9,
            "poster": "https://i.pinimg.com/9001.jpg?size=large",
            "creator": {
                "id": "77",
                "name": "Pinner",
                "profile_page": "https://www.pinterest.com/pinner/?token=drop",
            },
            "renditions": [
                {
                    "id": "V_720P",
                    "link": "https://v.pinimg.com/videos/9001/720p.mp4",
                    "width": 1080,
                    "height": 1920,
                }
            ],
        }
        record.update(overrides)
        return record

    def test_a_pin_becomes_a_material_with_provenance_and_no_query_string(self):
        config.proxy.clear()
        config.app.pop("tls_verify", None)

        with patch.object(
            material.pinterest, "search_video_pins", return_value=[self._record()]
        ) as search:
            results = material.search_videos_pinterest("sunrise", minimum_duration=3)

        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item.provider, "pinterest")
        self.assertEqual(item.url, "https://v.pinimg.com/videos/9001/720p.mp4")
        # Floored, never rounded: 8.9 must not be reported as a 9 second source.
        self.assertEqual(item.duration, 8)
        self.assertEqual(item.provider_asset_id, "9001")
        self.assertEqual(item.width, 1080)
        self.assertEqual(item.height, 1920)
        self.assertEqual(item.orientation, "portrait")
        self.assertEqual(item.rendition_id, "V_720P")
        self.assertEqual(item.search_query, "sunrise")
        self.assertEqual(item.query_attempt, 1)
        self.assertEqual(item.source_page_url, "https://www.pinterest.com/pin/9001/")
        self.assertEqual(item.preview_url, "https://i.pinimg.com/9001.jpg")
        self.assertEqual(
            item.source_info["creator"]["profile_page"],
            "https://www.pinterest.com/pinner/",
        )
        self.assertEqual(item.source_info["provider"], "pinterest")
        self.assertEqual(item.source_info["search_term"], "sunrise")
        self.assertTrue(search.call_args.kwargs["verify"])

    def test_the_search_runs_through_the_configured_proxy(self):
        config.proxy.clear()
        config.proxy.update({"https": "http://127.0.0.1:7890"})

        with patch.object(
            material.pinterest, "search_video_pins", return_value=[]
        ) as search:
            material.search_videos_pinterest("sunrise", minimum_duration=3)

        self.assertEqual(
            search.call_args.kwargs["proxies"], {"https": "http://127.0.0.1:7890"}
        )

    def test_a_pin_shorter_than_the_beat_needs_is_dropped(self):
        with patch.object(
            material.pinterest,
            "search_video_pins",
            return_value=[self._record(duration=4.0)],
        ):
            results = material.search_videos_pinterest("sunrise", minimum_duration=6)

        self.assertEqual(results, [])

    def test_a_pin_with_an_unknown_duration_is_dropped_rather_than_guessed(self):
        # The render timeline sizes a beat's window from this number, so an
        # unknown length cannot be treated as "probably long enough".
        with patch.object(
            material.pinterest,
            "search_video_pins",
            return_value=[self._record(duration=None)],
        ):
            results = material.search_videos_pinterest("sunrise", minimum_duration=1)

        self.assertEqual(results, [])

    def test_renditions_face_the_same_resolution_gate_as_pexels(self):
        low_resolution = self._record(
            renditions=[
                {
                    "id": "V_360P",
                    "link": "https://v.pinimg.com/videos/9001/360p.mp4",
                    "width": 360,
                    "height": 640,
                }
            ]
        )

        with patch.object(
            material.pinterest, "search_video_pins", return_value=[low_resolution]
        ):
            results = material.search_videos_pinterest("sunrise", minimum_duration=1)

        self.assertEqual(results, [])

    def test_a_landscape_rendition_is_refused_for_a_portrait_video(self):
        landscape = self._record(
            renditions=[
                {
                    "id": "V_1080P",
                    "link": "https://v.pinimg.com/videos/9001/1080p.mp4",
                    "width": 1920,
                    "height": 1080,
                }
            ]
        )

        with patch.object(
            material.pinterest, "search_video_pins", return_value=[landscape]
        ):
            portrait = material.search_videos_pinterest(
                "sunrise", minimum_duration=1, video_aspect=material.VideoAspect.portrait
            )
            wide = material.search_videos_pinterest(
                "sunrise",
                minimum_duration=1,
                video_aspect=material.VideoAspect.landscape,
            )

        self.assertEqual(portrait, [])
        self.assertEqual(len(wide), 1)
        self.assertEqual(wide[0].orientation, "landscape")

    def test_a_search_outage_returns_empty_and_says_why(self):
        # The provider interface conflates "nothing found" with "could not ask",
        # so the actionable cause has to reach the log at error level or a
        # rate limit reads as an absent concept.
        with patch.object(
            material.pinterest,
            "search_video_pins",
            side_effect=material.pinterest.PinterestSearchError(
                "pinterest rate limit exceeded: status=429, retry_after=30"
            ),
        ):
            with patch.object(material.logger, "error") as error:
                results = material.search_videos_pinterest("sunrise", minimum_duration=1)

        self.assertEqual(results, [])
        self.assertTrue(error.called)
        message = error.call_args.args[0]
        self.assertIn("rate limit", message)
        self.assertIn("next provider in the cascade", message)

    def test_an_unexpected_error_is_contained_rather_than_raised(self):
        with patch.object(
            material.pinterest, "search_video_pins", side_effect=RuntimeError("boom")
        ):
            with patch.object(material.logger, "error") as error:
                results = material.search_videos_pinterest("sunrise", minimum_duration=1)

        self.assertEqual(results, [])
        self.assertTrue(error.called)

    def test_pins_that_all_fail_the_gates_are_logged_apart_from_no_results(self):
        with patch.object(
            material.pinterest,
            "search_video_pins",
            return_value=[self._record(duration=2.0)],
        ):
            with patch.object(material.logger, "info") as info:
                results = material.search_videos_pinterest("sunrise", minimum_duration=9)

        self.assertEqual(results, [])
        messages = [call.args[0] for call in info.call_args_list]
        self.assertTrue(
            any("none passed the material gates" in message for message in messages)
        )

    def test_pinterest_is_the_first_provider_and_needs_no_key(self):
        self.assertEqual(
            material._SMART_PROVIDER_CASCADE_ORDER,
            ("pinterest", "pexels", "pixabay"),
        )
        self.assertIsNone(material._STOCK_VIDEO_PROVIDER_API_KEYS["pinterest"])
        self.assertIs(
            material._remote_search_function("pinterest"),
            material.search_videos_pinterest,
        )


if __name__ == "__main__":
    unittest.main()
