import json
import os
import sys
import tempfile
import tomllib
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.llm_provider import (
    DEFAULT_LLM_PROVIDER_ID,
    LLM_PROVIDER_REGISTRY,
    LLM_PROVIDERS,
    get_llm_provider,
    normalize_provider_override,
)
from app.models.schema import VideoScriptRequest, VideoSocialMetadataRequest
from app.services import llm

RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class TestScriptPromptOptions(unittest.TestCase):
    def test_normalize_text_response_preserves_internal_newlines(self):
        result = llm._normalize_text_response(
            "\n  First line\nSecond line\n\nThird paragraph  \n",
            "openai",
        )

        self.assertEqual(result, "First line\nSecond line\n\nThird paragraph")

    def test_normalize_text_response_removes_think_blocks(self):
        """
        reasoning 模型可能返回 `<think>...</think>`。脚本生成链路必须只保留
        最终正文，避免思考过程进入字幕和配音。
        """
        result = llm._normalize_text_response(
            "<think>\nI should reason here.\n</think>\n测试成功",
            "minimax",
        )

        self.assertEqual(result, "测试成功")

    def test_normalize_text_response_rejects_think_only_response(self):
        """
        如果模型只返回思考块而没有最终答案，应视为空内容，触发重试或明确错误。
        """
        with self.assertRaises(ValueError):
            llm._normalize_text_response("<think>hidden reasoning</think>", "minimax")

    def test_normalize_text_response_removes_unclosed_think_block(self):
        """
        某些网关可能因为截断只返回未闭合的 `<think>`。这种内容同样不能
        进入最终脚本；如果清理后没有正文，就应该按空响应处理。
        """
        with self.assertRaises(ValueError):
            llm._normalize_text_response("<think>hidden reasoning", "minimax")

    def test_build_script_prompt_appends_advanced_requirements(self):
        """
        高级文案要求只作为附加约束，不替换默认系统提示词。
        这样普通用户不配置时仍然走稳定默认规则，高级用户也能细化风格。
        """
        prompt = llm.build_script_prompt(
            video_subject="咖啡",
            language="zh-CN",
            paragraph_number=3,
            video_script_prompt="语气轻松，面向程序员",
        )

        self.assertIn("# Role: Video Script Generator", prompt)
        self.assertIn("- video subject: 咖啡", prompt)
        self.assertIn("- number of paragraphs: 3", prompt)
        self.assertIn("- language: zh-CN", prompt)
        self.assertIn("# Additional User Requirements:", prompt)
        self.assertIn("语气轻松，面向程序员", prompt)

    def test_custom_system_prompt_keeps_runtime_context(self):
        """
        自定义 system prompt 会替换默认脚本规则，但视频主题、语言、段落数
        仍由服务层统一追加，避免高级用户漏写必要上下文。
        """
        prompt = llm.build_script_prompt(
            video_subject="露营",
            language="en",
            paragraph_number=2,
            custom_system_prompt="Only write cinematic narration.",
        )

        self.assertNotIn("# Role: Video Script Generator", prompt)
        self.assertIn("Only write cinematic narration.", prompt)
        self.assertIn("- video subject: 露营", prompt)
        self.assertIn("- number of paragraphs: 2", prompt)
        self.assertIn("- language: en", prompt)

    def test_generate_script_sends_custom_prompt_to_llm(self):
        captured = {}

        def fake_generate_response(prompt):
            captured["prompt"] = prompt
            return "第一段。\n\n第二段。"

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_script(
                video_subject="咖啡",
                language="zh-CN",
                paragraph_number=2,
                video_script_prompt="开头更有悬念",
            )

        self.assertEqual(result, "第一段。\n\n第二段。")
        self.assertIn("- number of paragraphs: 2", captured["prompt"])
        self.assertIn("开头更有悬念", captured["prompt"])

    def test_generate_script_reuses_submitted_config_snapshot(self):
        """WebUI 后台任务结束后应用新配置，不能改变正在重试的模型请求。"""
        captured = {}
        app_config = {
            "llm_provider": "openai",
            "openai_api_key": "snapshot-key",
            "openai_model_name": "snapshot-model",
        }

        def fake_generate_response(prompt, app_config=None):
            captured["prompt"] = prompt
            captured["app_config"] = app_config
            return "Snapshot response"

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_script(
                video_subject="Snapshot test",
                app_config=app_config,
            )

        self.assertEqual(result, "Snapshot response")
        self.assertIs(captured["app_config"], app_config)
        self.assertEqual(captured["app_config"]["openai_api_key"], "snapshot-key")

    def test_generate_terms_can_request_script_ordered_keywords(self):
        """
        按文案顺序匹配素材依赖 LLM 返回有序关键词。这里不调用真实模型，
        只验证服务层会把“按脚本叙事顺序输出”的约束写入 prompt，避免
        后续素材下载虽然顺序化，但关键词仍然是全局无序主题词。
        """
        captured = {}

        def fake_generate_response(prompt):
            captured["prompt"] = prompt
            return '["opening city", "middle office", "final sunset"]'

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_terms(
                video_subject="startup story",
                video_script="First city. Then office. Finally sunset.",
                amount=3,
                match_script_order=True,
            )

        self.assertEqual(result, ["opening city", "middle office", "final sunset"])
        self.assertIn("chronological stock-video search terms", captured["prompt"])
        self.assertIn("same order as the script narration", captured["prompt"])
        self.assertIn("concrete, camera-visible shot", captured["prompt"])
        self.assertIn("return exactly 3 items", captured["prompt"])

    def test_generate_ordered_terms_retries_wrong_timeline_count(self):
        responses = iter(
            [
                '["opening city", "final sunset"]',
                '["opening city", "middle office", "final sunset"]',
            ]
        )

        with patch.object(
            llm, "_generate_response", side_effect=lambda _: next(responses)
        ):
            result = llm.generate_terms(
                video_subject="startup story",
                video_script="First city. Then office. Finally sunset.",
                amount=3,
                match_script_order=True,
            )

        self.assertEqual(result, ["opening city", "middle office", "final sunset"])

    def test_generate_ordered_terms_validates_count_after_regex_recovery(self):
        responses = iter(
            [
                'Here are the terms: ["opening city", "final sunset"]',
                'Result: ["opening city", "middle office", "final sunset"]',
            ]
        )

        with patch.object(
            llm, "_generate_response", side_effect=lambda _: next(responses)
        ):
            result = llm.generate_terms(
                video_subject="startup story",
                video_script="First city. Then office. Finally sunset.",
                amount=3,
                match_script_order=True,
            )

        self.assertEqual(result, ["opening city", "middle office", "final sunset"])

    def test_generate_visual_slot_queries_uses_indexed_slot_narration(self):
        captured = {}
        slots = [
            {
                "slot_index": 1,
                "start_time": 0.0,
                "end_time": 4.0,
                "visual_requirement": "Workers inspect the railway tracks.",
            },
            {
                "slot_index": 5,
                "start_time": 16.0,
                "end_time": 20.0,
                "visual_requirement": "Workers remove damaged wooden sleepers.",
            },
        ]

        def fake_generate_response(prompt):
            captured["prompt"] = prompt
            # Deliberately return slot 5 first to prove association is index-based.
            return (
                '[{"slot_index": 5, "queries": ["workers replacing railway sleepers"]}, '
                '{"slot_index": 1, "queries": ["workers inspecting railway tracks"]}]'
            )

        with patch.object(
            llm,
            "_generate_response",
            side_effect=fake_generate_response,
        ):
            result = llm.generate_visual_slot_queries(
                video_subject="railway maintenance",
                visual_slots=slots,
                queries_per_slot=1,
            )

        self.assertEqual(
            result[5], ["workers replacing railway sleepers"]
        )
        self.assertEqual(result[1], ["workers inspecting railway tracks"])
        self.assertIn("Workers inspect the railway tracks.", captured["prompt"])
        self.assertIn(
            "Workers remove damaged wooden sleepers.", captured["prompt"]
        )
        self.assertIn(
            "Derive each query only from that slot's visual_requirement",
            captured["prompt"],
        )

    def test_generate_visual_slot_queries_supports_future_multiple_queries(self):
        response = (
            '[{"slot_index": 1, "queries": '
            '["railway ballast closeup", "workers spreading track ballast"]}]'
        )
        with patch.object(llm, "_generate_response", return_value=response):
            result = llm.generate_visual_slot_queries(
                video_subject="railway ballast",
                visual_slots=[
                    {
                        "slot_index": 1,
                        "start_time": 0.0,
                        "end_time": 4.0,
                        "visual_requirement": "Workers spread ballast under the rails.",
                    }
                ],
                queries_per_slot=2,
            )

        self.assertEqual(len(result[1]), 2)

    def test_fewer_phrasings_than_requested_are_kept_instead_of_rejected(self):
        # Alternative phrasings are fallbacks material selection tries in order.
        # Rejecting the slot because two arrived instead of three would throw away
        # the phrasing that works and drop the whole beat timeline.
        captured = {}

        def fake_generate_response(prompt, app_config=None):
            captured["prompt"] = prompt
            return (
                '[{"slot_index": 1, "queries": '
                '["worker digging hole", "Worker Digging Hole", "shovel in soil"]},'
                ' {"slot_index": 2, "queries": ["rain flooding field"]}]'
            )

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_visual_slot_queries(
                video_subject="slow change in nature",
                visual_slots=[
                    {
                        "slot_index": 1,
                        "start_time": 0.0,
                        "end_time": 4.0,
                        "visual_requirement": "A worker digs a hole.",
                    },
                    {
                        "slot_index": 2,
                        "start_time": 4.0,
                        "end_time": 8.0,
                        "visual_requirement": "Rain floods a dry field.",
                    },
                ],
                queries_per_slot=3,
            )

        # A case-only repeat buys a second search and no new candidates.
        self.assertEqual(result[1], ["worker digging hole", "shovel in soil"])
        self.assertEqual(result[2], ["rain flooding field"])
        self.assertIn("Order each slot's queries as fallbacks", captured["prompt"])

    def test_extra_phrasings_beyond_the_request_are_truncated(self):
        response = (
            '[{"slot_index": 1, "queries": '
            '["worker digging hole", "shovel in soil", "hands in wet earth"]}]'
        )
        with patch.object(llm, "_generate_response", return_value=response):
            result = llm.generate_visual_slot_queries(
                video_subject="slow change in nature",
                visual_slots=[
                    {
                        "slot_index": 1,
                        "start_time": 0.0,
                        "end_time": 4.0,
                        "visual_requirement": "A worker digs a hole.",
                    }
                ],
                queries_per_slot=2,
            )

        self.assertEqual(result[1], ["worker digging hole", "shovel in soil"])

    def test_a_slot_with_no_usable_query_is_still_rejected(self):
        response = '[{"slot_index": 1, "queries": ["   "]}]'
        with patch.object(llm, "_generate_response", return_value=response):
            result = llm.generate_visual_slot_queries(
                video_subject="slow change in nature",
                visual_slots=[
                    {
                        "slot_index": 1,
                        "start_time": 0.0,
                        "end_time": 4.0,
                        "visual_requirement": "A worker digs a hole.",
                    }
                ],
                queries_per_slot=2,
            )

        self.assertEqual(result, {})

    def test_a_single_query_request_does_not_ask_for_ordered_fallbacks(self):
        captured = {}

        def fake_generate_response(prompt, app_config=None):
            captured["prompt"] = prompt
            return '[{"slot_index": 1, "queries": ["worker digging hole"]}]'

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            llm.generate_visual_slot_queries(
                video_subject="slow change in nature",
                visual_slots=[
                    {
                        "slot_index": 1,
                        "start_time": 0.0,
                        "end_time": 4.0,
                        "visual_requirement": "A worker digs a hole.",
                    }
                ],
                queries_per_slot=1,
            )

        self.assertNotIn("fallbacks", captured["prompt"])

    def test_semantic_grouping_uses_selected_provider_path_and_stable_unit_ids(self):
        captured = {}
        app_config = {
            "llm_provider": "openai",
            "openai_api_key": "test-only-key",
            "openai_model_name": "test-model",
        }

        def fake_generate_response(prompt, app_config=None):
            captured["prompt"] = prompt
            captured["app_config"] = app_config
            return (
                '[{"start_unit":0,"end_unit_exclusive":2,'
                '"visual_requirement":"Worker removing boards"}]'
            )

        with patch.object(
            llm,
            "_generate_response",
            side_effect=fake_generate_response,
        ) as generate:
            result = llm.generate_semantic_visual_span_specs(
                narration_text="Worker removes boards.",
                timed_units=[
                    {"text": "Worker", "source_narration_slot_index": 1},
                    {"text": "removes boards", "source_narration_slot_index": 1},
                ],
                app_config=app_config,
            )

        self.assertEqual(result[0]["start_unit"], 0)
        self.assertEqual(result[0]["end_unit_exclusive"], 2)
        self.assertEqual(generate.call_count, 1)
        self.assertIs(captured["app_config"], app_config)
        self.assertIn('0|1|"Worker"', captured["prompt"])
        self.assertIn('1|1|"removes boards"', captured["prompt"])
        self.assertIn("SMALLEST number of spans", captured["prompt"])
        self.assertIn("Do not return timestamps", captured["prompt"])

    def test_semantic_grouping_malformed_json_fails_after_bounded_retry(self):
        """畸形 JSON 必须在有限次重试后失败，绝不能返回猜测出的分组。"""
        with patch.object(
            llm,
            "_generate_response",
            return_value="not json",
        ) as generate:
            result = llm.generate_semantic_visual_span_specs(
                narration_text="Worker removes boards.",
                timed_units=[{"text": "Worker removes boards"}],
            )

        self.assertIsNone(result)
        self.assertEqual(generate.call_count, llm._MAX_STRUCTURED_RESPONSE_ATTEMPTS)

    def test_semantic_grouping_recovers_a_chatty_array_reply(self):
        """语义分组的根是数组；被散文包裹时必须整体恢复，而不是只取第一个对象。"""
        payload = (
            '[{"start_unit":0,"end_unit_exclusive":1,"visual_requirement":"Worker"},'
            '{"start_unit":1,"end_unit_exclusive":2,"visual_requirement":"Boards"}]'
        )

        with patch.object(
            llm,
            "_generate_response",
            return_value=f"Here are the spans:\n```json\n{payload}\n```\nDone.",
        ) as generate:
            result = llm.generate_semantic_visual_span_specs(
                narration_text="Worker removes boards.",
                timed_units=[
                    {"text": "Worker", "source_narration_slot_index": 1},
                    {"text": "removes boards", "source_narration_slot_index": 1},
                ],
            )

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["visual_requirement"], "Boards")

    def test_generate_terms_returns_empty_list_on_provider_error(self):
        """
        Provider 错误必须保持 generate_terms 的 List[str] 返回契约。

        非空的 ``Error: ...`` 字符串在 Python 中是真值；如果直接返回，任务层
        会把它当成有效关键词，素材下载层随后还可能逐字符发起搜索请求。
        """
        with patch.object(
            llm,
            "_generate_response",
            return_value="Error: invalid API key",
        ):
            result = llm.generate_terms(
                video_subject="startup story",
                video_script="A short startup story.",
            )

        self.assertEqual(result, [])
        self.assertIsInstance(result, list)

    def test_video_script_request_rejects_invalid_advanced_options(self):
        """
        API 请求模型需要限制高级 prompt 参数，避免外部调用绕过 WebUI
        传入异常段落数或超长提示词，导致模型成本和结果不可控。
        """
        with self.assertRaises(ValidationError):
            VideoScriptRequest(video_subject="咖啡", paragraph_number=0)

        with self.assertRaises(ValidationError):
            VideoScriptRequest(
                video_subject="咖啡",
                video_script_prompt="x" * (llm.MAX_SCRIPT_PROMPT_LENGTH + 1),
            )


class TestLLMConnection(unittest.TestCase):
    def test_connection_sends_one_minimal_request(self):
        """连接测试只发送一次固定最小请求，不触发脚本生成重试。"""
        with (
            patch.object(llm, "_generate_response", return_value="OK") as generate,
            patch.object(llm, "perf_counter", side_effect=[10.0, 10.25]),
        ):
            result = llm.test_connection()

        generate.assert_called_once_with(prompt="Reply with exactly: OK")
        self.assertEqual(result, (True, "", 0.25))

    def test_connection_returns_provider_error(self):
        """Provider 返回错误时应保留可诊断信息，并报告本次请求耗时。"""
        with (
            patch.object(
                llm,
                "_generate_response",
                return_value="Error: invalid API key",
            ),
            patch.object(llm, "perf_counter", side_effect=[20.0, 20.5]),
        ):
            result = llm.test_connection()

        self.assertEqual(result, (False, "invalid API key", 0.5))

    def test_connection_rejects_empty_response(self):
        """极端情况下的空响应应显示明确错误，而不是误报连接成功。"""
        with (
            patch.object(llm, "_generate_response", return_value=""),
            patch.object(llm, "perf_counter", side_effect=[30.0, 31.0]),
        ):
            result = llm.test_connection()

        self.assertEqual(result, (False, "LLM returned an empty response", 1.0))


class TestLiteLLMProvider(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_current_default_model_names(self):
        """WebUI 与服务层必须共享同一组默认模型，避免展示值和请求值漂移。"""
        self.assertEqual(get_llm_provider("openai").default_model, "gpt-5.5")
        self.assertEqual(get_llm_provider("aimlapi").default_model, "openai/gpt-5-5")
        self.assertEqual(get_llm_provider("deepseek").default_model, "deepseek-v4-pro")
        self.assertEqual(
            get_llm_provider("modelscope").default_model, "ZhipuAI/GLM-5.2"
        )
        self.assertEqual(
            get_llm_provider("gemini").default_model, "gemini-3.1-pro-preview"
        )
        pollinations = get_llm_provider("pollinations")
        self.assertEqual(pollinations.default_model, "openai-fast")
        self.assertEqual(
            pollinations.default_base_url,
            "https://gen.pollinations.ai/v1",
        )
        self.assertTrue(pollinations.requires_api_key)
        self.assertEqual(pollinations.adapter, "openai_compatible")

    def test_provider_defaults_are_not_persisted_as_user_overrides(self):
        """默认值只用于运行和展示，只有不同值才应写入用户配置。"""
        self.assertEqual(
            normalize_provider_override("gpt-5.5", "gpt-5.5"),
            "",
        )
        self.assertEqual(
            normalize_provider_override("  gpt-5.5  ", "gpt-5.5"),
            "",
        )
        self.assertEqual(
            normalize_provider_override("gpt-5.6-custom", "gpt-5.5"),
            "gpt-5.6-custom",
        )

    def test_provider_registry_has_unique_stable_ids(self):
        """Registry 是 Provider 列表的唯一数据源，ID 必须唯一且默认项存在。"""
        provider_ids = [provider.provider_id for provider in LLM_PROVIDER_REGISTRY]

        self.assertEqual(len(provider_ids), len(set(provider_ids)))
        self.assertEqual(len(provider_ids), len(LLM_PROVIDERS))
        self.assertIn(DEFAULT_LLM_PROVIDER_ID, LLM_PROVIDERS)

    def test_provider_registry_preserves_product_group_order(self):
        """下拉顺序按推荐、原厂、聚合平台、本地部署和其它服务排列。"""
        self.assertEqual(
            [provider.provider_id for provider in LLM_PROVIDER_REGISTRY],
            [
                "moonshot",
                "openai",
                "gemini",
                "deepseek",
                "qwen",
                "azure",
                "volcengine",
                "grok",
                "minimax",
                "mimo",
                "shengsuanyun",
                "cloudflare",
                "modelscope",
                "aihubmix",
                "aimlapi",
                "evolink",
                "ollama",
                "oneapi",
                "litellm",
                "groq",
                "pollinations",
            ],
        )
        self.assertEqual(
            get_llm_provider("gemini").default_label,
            "Google Gemini",
        )
        self.assertEqual(
            get_llm_provider("azure").default_label,
            "Microsoft Azure OpenAI",
        )
        shengsuanyun = get_llm_provider("shengsuanyun")
        self.assertEqual(
            shengsuanyun.api_key_url,
            "https://www.shengsuanyun.com/?from=CH_XUQ4OTSK",
        )
        self.assertEqual(
            shengsuanyun.default_model,
            "deepseek/deepseek-v4-flash",
        )

    def test_provider_registry_uses_conventional_locale_and_config_keys(self):
        """统一命名规则可避免 WebUI 为每个 Provider 增加硬编码映射。"""
        for provider in LLM_PROVIDER_REGISTRY:
            self.assertEqual(
                provider.label_key,
                f"llm_provider_label.{provider.provider_id}",
            )
            self.assertEqual(
                provider.tips_key,
                f"llm_provider_tips.{provider.provider_id}",
            )
            self.assertEqual(
                provider.config_key("api_key"),
                f"{provider.provider_id}_api_key",
            )

    def test_registry_replaces_deprecated_provider_models(self):
        """历史默认模型应自动迁移，避免升级后继续使用已移除的接入语义。"""
        cloudflare = get_llm_provider("cloudflare")
        gemini = get_llm_provider("gemini")

        self.assertEqual(
            cloudflare.resolve_model_name("@cf/meta/llama-3.1-8b-instruct"),
            "openai/gpt-4.1-mini",
        )
        self.assertEqual(
            gemini.resolve_model_name("gemini-pro"),
            "gemini-3.1-pro-preview",
        )
        self.assertEqual(
            cloudflare.resolve_model_name("anthropic/claude-sonnet-4-5"),
            "anthropic/claude-sonnet-4-5",
        )

        pollinations = get_llm_provider("pollinations")
        self.assertEqual(
            pollinations.resolve_model_name("default"),
            "openai-fast",
        )
        self.assertEqual(
            pollinations.resolve_base_url("https://text.pollinations.ai/openai"),
            "https://gen.pollinations.ai/v1",
        )
        self.assertEqual(
            pollinations.resolve_base_url("https://example.com/v1"),
            "https://example.com/v1",
        )

    def test_provider_tip_templates_accept_registry_defaults(self):
        """所有语言的 Provider 提示模板都必须能安全注入 Registry 默认值。"""
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        for locale_file in i18n_dir.glob("*.json"):
            translations = json.loads(locale_file.read_text(encoding="utf-8"))[
                "Translation"
            ]
            for provider in LLM_PROVIDER_REGISTRY:
                tips = translations.get(provider.tips_key, "")
                if not tips:
                    continue
                rendered = tips.format(
                    api_key_url=provider.api_key_url,
                    default_model=provider.default_model,
                    default_base_url=provider.default_base_url,
                    docker_hint="",
                    **{
                        f"default_{field.config_suffix}": field.default_value
                        for field in provider.extra_fields
                    },
                )
                self.assertNotIn("{default_model}", rendered)
                self.assertNotIn("{default_base_url}", rendered)

    def test_primary_provider_tips_use_consistent_structure(self):
        """中英文配置说明统一展示 API Key、Base URL 和模型名称。"""
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        for language in ("zh", "en"):
            translations = json.loads(
                (i18n_dir / f"{language}.json").read_text(encoding="utf-8")
            )["Translation"]
            for provider in LLM_PROVIDER_REGISTRY:
                tips = translations[provider.tips_key]
                self.assertTrue(tips.startswith("##### "), provider.provider_id)
                self.assertIn("**API Key**", tips, provider.provider_id)
                self.assertIn("**Base Url**", tips, provider.provider_id)
                self.assertIn("**Model Name**", tips, provider.provider_id)

        zh_kimi_tips = json.loads((i18n_dir / "zh.json").read_text(encoding="utf-8"))[
            "Translation"
        ]["llm_provider_tips.moonshot"]
        self.assertIn("推荐理由：", zh_kimi_tips)
        self.assertIn("视频创作链路匹配", zh_kimi_tips)

    def test_required_api_key_providers_have_clickable_entry_points(self):
        """需要密钥的 Provider 必须提供统一申请入口，避免 WebUI 只给出文字。"""
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        locale_translations = {
            locale_file.stem: json.loads(locale_file.read_text(encoding="utf-8"))[
                "Translation"
            ]
            for locale_file in i18n_dir.glob("*.json")
        }

        for provider in LLM_PROVIDER_REGISTRY:
            if provider.requires_api_key:
                self.assertTrue(provider.api_key_url, provider.provider_id)
                self.assertTrue(
                    provider.api_key_url.startswith("https://"),
                    provider.provider_id,
                )
                for language, translations in locale_translations.items():
                    tips_template = translations.get(provider.tips_key, "")
                    if not tips_template:
                        continue
                    tips = tips_template.format(
                        api_key_url=provider.api_key_url,
                        default_model=provider.default_model,
                        default_base_url=provider.default_base_url,
                        docker_hint="",
                        **{
                            f"default_{field.config_suffix}": field.default_value
                            for field in provider.extra_fields
                        },
                    )
                    api_key_line = next(
                        line for line in tips.splitlines() if "**API Key**" in line
                    )
                    self.assertIn("](", api_key_line, provider.provider_id)
                    self.assertIn(
                        f"]({provider.api_key_url})",
                        api_key_line,
                        f"{language}: {provider.provider_id}",
                    )

    def test_example_config_does_not_duplicate_registry_defaults(self):
        """示例配置只保存用户覆盖值，默认模型和地址由 Registry 唯一维护。"""
        config_path = Path(__file__).parent.parent.parent / "config.example.toml"
        app_config = tomllib.loads(config_path.read_text(encoding="utf-8"))["app"]

        for provider in LLM_PROVIDER_REGISTRY:
            if provider.default_model:
                self.assertEqual(
                    app_config.get(provider.config_key("model_name"), ""),
                    "",
                    provider.provider_id,
                )
            if provider.default_base_url:
                self.assertEqual(
                    app_config.get(provider.config_key("base_url"), ""),
                    "",
                    provider.provider_id,
                )
            for field in provider.extra_fields:
                if field.default_value:
                    self.assertEqual(
                        app_config.get(provider.config_key(field.config_suffix), ""),
                        "",
                        provider.provider_id,
                    )

    def test_removed_ernie_provider_is_unsupported(self):
        """移除 ERNIE 后，遗留配置应返回明确错误，不再发起旧 OAuth 请求。"""
        config.app["llm_provider"] = "ernie"

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("unsupported llm provider", result)

    def test_pollinations_requires_api_key_before_request(self):
        """新统一 API 要求鉴权，缺少 Key 时不得发送匿名生成请求。"""
        config.app.update(
            {
                "llm_provider": "pollinations",
                "pollinations_api_key": "",
                "pollinations_base_url": "",
                "pollinations_model_name": "",
            }
        )

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("api_key is not set", result)

    def test_pollinations_uses_unified_openai_compatible_api(self):
        """历史地址和模型名应自动迁移，并通过统一 Chat Completions API 调用。"""
        config.app.update(
            {
                "llm_provider": "pollinations",
                "pollinations_api_key": "pollinations-test-key",
                "pollinations_base_url": "https://text.pollinations.ai/openai/",
                "pollinations_model_name": "default",
            }
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\npollinations")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="pollinations-test-key",
            base_url="https://gen.pollinations.ai/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai-fast",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\npollinations")

    def test_gemini_uses_google_genai_client(self):
        """Gemini 适配器应通过新版 SDK 的统一 Client 发起内容生成请求。"""
        config.app.update(
            {
                "llm_provider": "gemini",
                "gemini_api_key": "gemini-test-key",
                "gemini_base_url": "",
                "gemini_model_name": "gemini-test-model",
            }
        )
        captured = {}

        class FakeModels:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(text="hello\ngemini")

        class FakeClient:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.models = FakeModels()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                captured["closed"] = True

        with patch("google.genai.Client", FakeClient):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "hello\ngemini")
        self.assertEqual(
            captured["client_kwargs"],
            {"api_key": "gemini-test-key", "http_options": None},
        )
        self.assertEqual(captured["model"], "gemini-test-model")
        self.assertEqual(captured["contents"], "Say hello")
        self.assertEqual(
            captured["config"].max_output_tokens, llm._DEFAULT_MAX_OUTPUT_TOKENS
        )
        self.assertTrue(captured["closed"])

    def test_cloudflare_requires_account_id_before_request(self):
        """Cloudflare 缺少 Account ID 时应在本地失败，不发送无效请求。"""
        config.app.update(
            {
                "llm_provider": "cloudflare",
                "cloudflare_api_key": "test-token",
                "cloudflare_account_id": "",
                "cloudflare_model_name": "",
            }
        )

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("account_id is not set", result)

    def test_cloudflare_uses_ai_gateway_openai_endpoint(self):
        """Cloudflare Provider 必须走 AI Gateway，不再调用 Workers AI 接口。"""
        config.app.update(
            {
                "llm_provider": "cloudflare",
                "cloudflare_api_key": "cloudflare-token",
                "cloudflare_account_id": "account-123",
                "cloudflare_gateway_id": "",
                "cloudflare_model_name": "",
            }
        )

        fake_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="gateway\nresponse")
                )
            ]
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return fake_response

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="cloudflare-token",
            base_url=(
                "https://api.cloudflare.com/client/v4/accounts/account-123/ai/v1"
            ),
            default_headers={"cf-aig-gateway-id": "default"},
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai/gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "gateway\nresponse")

    def _use_litellm_provider(self, model_name="openai/gpt-4o-mini"):
        config.app["llm_provider"] = "litellm"
        config.app["litellm_model_name"] = model_name

    def test_litellm_provider_returns_normalized_text(self):
        """
        验证 LiteLLM provider 的主路径不依赖真实网络和私有 API key。

        这里用 fake module 注入 `sys.modules`，直接覆盖动态 import 的
        `litellm.completion()`，确保测试稳定覆盖 `_generate_response()` 里的
        litellm 分支。
        """
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace()

        def _completion(**kwargs):
            self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
            self.assertEqual(
                kwargs["messages"], [{"role": "user", "content": "Say hello"}]
            )
            self.assertTrue(kwargs["drop_params"])
            message = types.SimpleNamespace(content="hello\nworld")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

        fake_litellm.completion = _completion

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "hello\nworld")

    def test_litellm_provider_uses_registry_default_model(self):
        self._use_litellm_provider(model_name="")

        fake_litellm = types.SimpleNamespace()

        def _completion(**kwargs):
            self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
            message = types.SimpleNamespace(content="default model")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

        fake_litellm.completion = _completion

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertEqual(result, "default model")

    def test_litellm_provider_handles_empty_response(self):
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: types.SimpleNamespace(choices=[])
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("returned empty response", result)

    def test_litellm_provider_handles_empty_message(self):
        """
        某些 OpenAI-compatible 网关在内容过滤或安全拦截时会返回
        HTTP 200，但 `choices[0].message` 为 None。这里必须返回
        可诊断的错误，而不是抛出 AttributeError。
        """
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=None)]
            )
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("returned empty message", result)

    def test_sanitize_error_message_redacts_url_credentials_and_query_tokens(self):
        message = (
            "request failed for "
            "https://myuser:mypassword@proxy.example.com/v1/chat"
            "?api_key=secret-key&token=secret-token&safe=value"
        )

        result = llm._sanitize_error_message(message)

        self.assertIn("https://***:***@proxy.example.com", result)
        self.assertIn("api_key=***", result)
        self.assertIn("token=***", result)
        self.assertIn("safe=value", result)
        self.assertNotIn("myuser", result)
        self.assertNotIn("mypassword", result)
        self.assertNotIn("secret-key", result)
        self.assertNotIn("secret-token", result)

    def test_openai_provider_error_redacts_embedded_base_url_credentials(self):
        """
        自定义 OpenAI-compatible base_url 可能包含代理网关的 user:pass。
        SDK 抛错时常会把 URL 带回异常信息，这里验证最终返回给 WebUI/API 的
        `Error:` 文案不会泄露这些凭据。
        """
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = "groq-key"
        config.app["groq_model_name"] = "llama-3.3-70b-versatile"
        config.app["groq_base_url"] = (
            "https://myuser:mypassword@proxy.example.com/openai/v1"
        )

        class FakeCompletions:
            def create(self, **kwargs):
                raise RuntimeError(
                    "connection failed: "
                    "https://myuser:mypassword@proxy.example.com/openai/v1"
                    "?access_token=secret-token"
                )

        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )

        with patch.object(llm, "OpenAI", return_value=fake_client):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("https://***:***@proxy.example.com", result)
        self.assertIn("access_token=***", result)
        self.assertNotIn("myuser", result)
        self.assertNotIn("mypassword", result)
        self.assertNotIn("secret-token", result)

    def test_openai_provider_still_uses_existing_path(self):
        config.app["llm_provider"] = "openai"
        config.app["openai_api_key"] = ""
        config.app["openai_base_url"] = "https://api.openai.com/v1"
        config.app["openai_model_name"] = "gpt-4o-mini"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def _use_qwen_provider(self):
        config.app["llm_provider"] = "qwen"
        config.app["qwen_api_key"] = "qwen-key"
        config.app["qwen_model_name"] = "qwen-max"

    def _patch_dashscope_generation(self, response):
        class FakeGenerationResponse(dict):
            pass

        fake_response = FakeGenerationResponse(response)
        fake_response.status_code = response.get("status_code", 200)
        fake_dashscope = types.SimpleNamespace(
            api_key="",
            Generation=types.SimpleNamespace(call=lambda **kwargs: fake_response),
        )
        fake_dashscope_response = types.SimpleNamespace(
            GenerationResponse=FakeGenerationResponse
        )

        return patch.dict(
            sys.modules,
            {
                "dashscope": fake_dashscope,
                "dashscope.api_entities": types.SimpleNamespace(),
                "dashscope.api_entities.dashscope_response": fake_dashscope_response,
            },
        )

    def test_qwen_provider_reads_chat_choices_content(self):
        """
        DashScope chat 模式会把文本放在 `output.choices[0].message.content`。
        这里覆盖 issue #966 报告的 `output.text is None` 场景，避免再次触发
        `'NoneType' object has no attribute 'replace'`。
        """
        self._use_qwen_provider()
        response = {
            "output": {
                "text": None,
                "choices": [{"message": {"content": "你好\n世界"}}],
            }
        }

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "你好\n世界")

    def test_qwen_provider_falls_back_to_output_text(self):
        """保留旧 DashScope completion 响应结构的兼容路径。"""
        self._use_qwen_provider()
        response = {"output": {"text": "旧格式\n响应"}}

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "旧格式\n响应")

    def test_qwen_provider_reports_empty_text(self):
        """Qwen 空响应应返回可诊断错误，而不是底层 AttributeError。"""
        self._use_qwen_provider()
        response = {
            "output": {"text": None, "choices": [{"message": {"content": None}}]}
        }

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertIn("Error:", result)
        self.assertIn("returned empty text content", result)
        self.assertNotIn("NoneType", result)

    def test_qwen_provider_reports_empty_choices(self):
        """Qwen chat 响应 choices 为空时应返回明确错误。"""
        self._use_qwen_provider()
        response = {"output": {"text": None, "choices": []}}

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertIn("Error:", result)
        self.assertIn("returned empty choices", result)
        self.assertNotIn("NoneType", result)

    def test_aihubmix_provider_uses_openai_compatible_client(self):
        """
        AIHubMix 是 OpenAI-compatible 网关。这里用 fake OpenAI client
        验证独立 Provider 会使用 Registry 中的默认地址和模型，避免真实网络
        或私有 API Key 影响测试稳定性。
        """
        config.app["llm_provider"] = "aihubmix"
        config.app["aihubmix_api_key"] = "aihubmix-key"
        config.app["aihubmix_base_url"] = ""
        config.app["aihubmix_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\naihubmix")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="aihubmix-key",
            base_url="https://aihubmix.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-5.4-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\naihubmix")

    def test_aimlapi_provider_uses_openai_compatible_client(self):
        config.app["llm_provider"] = "aimlapi"
        config.app["aimlapi_api_key"] = "aimlapi-key"
        config.app["aimlapi_base_url"] = ""
        config.app["aimlapi_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\naimlapi")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="aimlapi-key",
            base_url="https://api.aimlapi.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai/gpt-5-5",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\naimlapi")

    def test_evolink_provider_uses_openai_compatible_client(self):
        """
        EvoLink exposes OpenAI-compatible Chat Completions at direct.evolink.ai.
        The provider should keep its own default endpoint and model instead of
        requiring users to overload the generic OpenAI settings.
        """
        config.app["llm_provider"] = "evolink"
        config.app["evolink_api_key"] = "evolink-key"
        config.app["evolink_base_url"] = ""
        config.app["evolink_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nevolink")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="evolink-key",
            base_url="https://direct.evolink.ai/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nevolink")

    def test_volcengine_provider_uses_openai_compatible_client(self):
        """
        VolcEngine Ark 暴露 OpenAI-compatible Chat Completions。
        这里用 fake OpenAI client 覆盖 provider 默认地址和默认模型，
        避免真实网络或私有 API key 影响测试稳定性。
        """
        config.app["llm_provider"] = "volcengine"
        config.app["volcengine_api_key"] = "volcengine-key"
        config.app["volcengine_base_url"] = ""
        config.app["volcengine_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nvolcengine")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="volcengine-key",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "doubao-seed-2-1-turbo-260628",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nvolcengine")

    def test_grok_provider_still_uses_existing_path(self):
        config.app["llm_provider"] = "grok"
        config.app["grok_api_key"] = ""
        config.app["grok_base_url"] = "https://api.x.ai/v1"
        config.app["grok_model_name"] = "grok-4.3"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def test_groq_provider_requires_api_key(self):
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = ""
        config.app["groq_base_url"] = "https://api.groq.com/openai/v1"
        config.app["groq_model_name"] = "llama-3.3-70b-versatile"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def test_groq_provider_uses_default_base_url(self):
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = "groq-test-key"
        config.app["groq_base_url"] = ""
        config.app["groq_model_name"] = "llama-3.3-70b-versatile"

        fake_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="hello\ngroq")
                )
            ]
        )
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="groq-test-key",
            base_url="https://api.groq.com/openai/v1",
        )
        self.assertEqual(result, "hello\ngroq")

    def _use_ollama_provider(self, base_url=""):
        config.app["llm_provider"] = "ollama"
        config.app["ollama_api_key"] = ""
        config.app["ollama_base_url"] = base_url
        config.app["ollama_model_name"] = "llama3"

    def _assert_ollama_base_url(self, expected_base_url: str):
        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nollama")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="ollama",
            base_url=expected_base_url,
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "llama3",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nollama")

    def test_ollama_default_base_url_uses_localhost_outside_container(self):
        """
        普通本机运行时，Ollama 默认仍然使用 localhost，避免影响已有用户。
        """
        self._use_ollama_provider()

        with patch.object(config, "is_running_in_container", return_value=False):
            self._assert_ollama_base_url("http://localhost:11434/v1")

    def test_ollama_default_base_url_uses_host_gateway_inside_container(self):
        """
        容器内运行时，localhost 指向容器自身；默认改为 host.docker.internal，
        方便 Docker Desktop 用户访问宿主机上的 Ollama。
        """
        self._use_ollama_provider()

        with (
            patch.object(config, "is_running_in_container", return_value=True),
            patch.object(config, "_can_resolve_hostname", return_value=True),
        ):
            self._assert_ollama_base_url("http://host.docker.internal:11434/v1")

    def test_ollama_default_base_url_falls_back_to_container_gateway(self):
        """
        原生 Linux Docker 里不一定能解析 host.docker.internal。此时使用容器
        默认网关作为兜底地址，比直接返回不可解析的 hostname 更稳。
        """
        self._use_ollama_provider()

        with (
            patch.object(config, "is_running_in_container", return_value=True),
            patch.object(config, "_can_resolve_hostname", return_value=False),
            patch.object(
                config, "get_container_default_gateway_ip", return_value="172.17.0.1"
            ),
        ):
            self._assert_ollama_base_url("http://172.17.0.1:11434/v1")

    def test_ollama_explicit_base_url_takes_precedence(self):
        """
        用户手动配置的 ollama_base_url 优先级最高，不受容器检测影响。
        """
        self._use_ollama_provider(base_url="http://ollama:11434/v1")

        with patch.object(config, "is_running_in_container", return_value=True):
            self._assert_ollama_base_url("http://ollama:11434/v1")

    def test_mimo_provider_uses_openai_compatible_client(self):
        """
        MiMo 官方接口兼容 OpenAI Chat Completions 协议。这里用 fake OpenAI
        client 验证 provider 会使用 MiMo 独立配置和默认 base_url，不依赖
        真实网络或私有 API Key。
        """
        config.app["llm_provider"] = "mimo"
        config.app["mimo_api_key"] = "mimo-key"
        config.app["mimo_base_url"] = ""
        config.app["mimo_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nmimo")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="mimo-key",
            base_url="https://api.xiaomimimo.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nmimo")

    def test_azure_provider_uses_azure_client_directly(self):
        """
        Azure OpenAI 的鉴权、endpoint 和 api-version 都由 AzureOpenAI 客户端处理。
        这个测试覆盖 issue #892：azure 分支必须直接调用 AzureOpenAI 创建的客户端，
        不能继续落入普通 OpenAI-compatible 分支，否则会丢失 Azure 专用请求配置。
        """
        config.app["llm_provider"] = "azure"
        config.app["azure_api_key"] = "azure-key"
        config.app["azure_base_url"] = "https://example.openai.azure.com"
        config.app["azure_model_name"] = "gpt-4o-mini"
        config.app["azure_api_version"] = "2024-02-15-preview"

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nazure")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "AzureOpenAI", return_value=fake_client) as azure_client,
            patch.object(llm, "OpenAI") as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        azure_client.assert_called_once_with(
            api_key="azure-key",
            api_version="2024-02-15-preview",
            azure_endpoint="https://example.openai.azure.com",
        )
        openai_client.assert_not_called()
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hello\nazure")

    def test_unsupported_provider_returns_clear_error(self):
        config.app["llm_provider"] = "g" + "4f"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("unsupported llm provider", result)


class TestRuntimeEnvironmentDetection(unittest.TestCase):
    def test_container_detection_ignores_plain_linux_cgroup_file(self):
        """
        普通 Linux 也有 /proc/1/cgroup，不能因为文件存在就判定为容器。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cgroup_path = Path(tmp_dir) / "cgroup"
            cgroup_path.write_text("0::/init.scope\n", encoding="utf-8")

            self.assertFalse(
                config.is_running_in_container(
                    dockerenv_path=str(Path(tmp_dir) / "missing-dockerenv"),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(cgroup_path),
                )
            )

    def test_container_detection_accepts_dockerenv_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dockerenv_path = Path(tmp_dir) / ".dockerenv"
            dockerenv_path.write_text("", encoding="utf-8")

            self.assertTrue(
                config.is_running_in_container(
                    dockerenv_path=str(dockerenv_path),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(Path(tmp_dir) / "missing-cgroup"),
                )
            )

    def test_container_detection_accepts_cgroup_container_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cgroup_path = Path(tmp_dir) / "cgroup"
            cgroup_path.write_text(
                "0::/system.slice/docker-abcdef.scope\n",
                encoding="utf-8",
            )

            self.assertTrue(
                config.is_running_in_container(
                    dockerenv_path=str(Path(tmp_dir) / "missing-dockerenv"),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(cgroup_path),
                )
            )

    def test_container_gateway_ip_decodes_default_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            route_path = Path(tmp_dir) / "route"
            route_path.write_text(
                "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
                "eth0\t00000000\t010011AC\t0003\t0\t0\t0\t00000000\t0\t0\t0\n",
                encoding="utf-8",
            )

            self.assertEqual(
                config.get_container_default_gateway_ip(str(route_path)),
                "172.17.0.1",
            )

    def test_container_gateway_ip_ignores_missing_default_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            route_path = Path(tmp_dir) / "route"
            route_path.write_text(
                "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
                "eth0\t0011AC0A\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n",
                encoding="utf-8",
            )

            self.assertEqual(
                config.get_container_default_gateway_ip(str(route_path)), ""
            )


class TestSocialMetadata(unittest.TestCase):
    """通用短视频发布文案元数据生成。"""

    def test_build_prompt_auto_language_uses_source_language(self):
        """
        language 默认 auto 时，不应该固定成某个国家或语种，而是让模型
        跟随视频主题和脚本的语言，扩大 API 适用范围。
        """
        prompt = llm.build_social_metadata_prompt(
            video_subject="上海一日游",
            video_script="今天带你快速看完上海经典路线。",
            language="auto",
            platform="tiktok",
        )

        self.assertIn("TikTok", prompt)
        self.assertIn("Use the same language as the video subject and script", prompt)
        self.assertIn("上海一日游", prompt)
        self.assertIn("array of exactly 5 strings", prompt)

    def test_build_prompt_accepts_explicit_language(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="Coffee tips",
            language="en-US",
            platform="youtube_shorts",
        )

        self.assertIn("YouTube Shorts", prompt)
        self.assertIn('Write "title" and "caption" in this language: en-US', prompt)
        self.assertIn("array of exactly 3 strings", prompt)

    def test_unknown_platform_falls_back_to_tiktok(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="x",
            platform="unsupported-platform",
        )

        self.assertIn("TikTok", prompt)

    def test_normalize_hashtags_from_string_dedupes_and_clamps(self):
        tags = llm._normalize_hashtags("#fyp fyp, trending #Trending viral", count=2)

        self.assertEqual(tags, ["#fyp", "#trending"])

    def test_normalize_hashtags_from_list_keeps_unicode_letters(self):
        tags = llm._normalize_hashtags(
            ["上海 旅行", "#việt nam", "  ", "@bad!chars"], count=5
        )

        self.assertEqual(tags, ["#上海旅行", "#việtnam", "#badchars"])

    def test_parse_social_metadata_recovers_embedded_json(self):
        raw = 'Sure: {"title":"T","caption":"C","hashtags":["#x"]} thanks'
        result = llm._parse_social_metadata(raw, "tiktok")

        self.assertEqual(result["title"], "T")
        self.assertEqual(result["caption"], "C")
        self.assertEqual(result["hashtags"], ["#x"])

    def test_parse_social_metadata_requires_title_or_caption(self):
        with self.assertRaises(ValueError):
            llm._parse_social_metadata('{"hashtags":["#x"]}', "tiktok")

    def test_generate_social_metadata_uses_llm_response(self):
        payload = (
            '{"title":"上海一日游","caption":"收藏这条路线，下次直接出发！",'
            '"hashtags":["#上海","#旅行","#shorts"]}'
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            result = llm.generate_social_metadata(
                video_subject="上海一日游",
                video_script="今天带你快速看完上海经典路线。",
                language="zh-CN",
                platform="tiktok",
            )

        self.assertEqual(result["title"], "上海一日游")
        self.assertEqual(result["caption"], "收藏这条路线，下次直接出发！")
        self.assertEqual(result["hashtags"], ["#上海", "#旅行", "#shorts"])

    def test_generate_social_metadata_falls_back_to_generic_hashtags(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            result = llm.generate_social_metadata(
                video_subject="Coffee tips",
                video_script="Save these three coffee tips.",
                platform="instagram_reels",
            )

        self.assertEqual(result["title"], "Coffee tips")
        self.assertEqual(result["caption"], "Save these three coffee tips.")
        self.assertEqual(len(result["hashtags"]), 8)
        self.assertEqual(result["hashtags"][0], "#shorts")

    def test_request_model_defaults_to_auto_language_tiktok(self):
        body = VideoSocialMetadataRequest(video_subject="Test")

        self.assertEqual(body.language, "auto")
        self.assertEqual(body.platform, "tiktok")

    def test_request_model_rejects_oversized_social_metadata_fields(self):
        """
        外部 API 不能接受无限长的脚本和语言参数，否则会直接放大 LLM
        token 成本。schema 层先拦截，服务层再做内部调用兜底。
        """
        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x" * 501)

        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x", video_script="x" * 8001)

        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x", language="x" * 65)

    def test_build_prompt_clamps_direct_service_inputs(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="x" * 600,
            video_script="y" * 9000,
            language="en",
        )

        self.assertIn("x" * llm.MAX_SOCIAL_SUBJECT_LENGTH, prompt)
        self.assertNotIn("x" * (llm.MAX_SOCIAL_SUBJECT_LENGTH + 1), prompt)
        self.assertIn("y" * llm.MAX_SOCIAL_SCRIPT_LENGTH, prompt)
        self.assertNotIn("y" * (llm.MAX_SOCIAL_SCRIPT_LENGTH + 1), prompt)

    def test_social_metadata_endpoint_response_shape(self):
        from fastapi.testclient import TestClient

        from app.asgi import app

        request_body = {
            "video_subject": "Tokyo coffee shops",
            "video_script": "Three quiet coffee shops for your next Tokyo morning.",
            "language": "en",
            "platform": "youtube_shorts",
        }
        llm_response = (
            '{"title":"3 Quiet Tokyo Coffee Shops",'
            '"caption":"Save these spots for your next Tokyo morning.",'
            '"hashtags":["#Tokyo","#Coffee","#Shorts"]}'
        )

        with patch.object(llm, "_generate_response", return_value=llm_response):
            response = TestClient(app).post(
                "/api/v1/social-metadata",
                json=request_body,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": 200,
                "message": "success",
                "data": {
                    "title": "3 Quiet Tokyo Coffee Shops",
                    "caption": "Save these spots for your next Tokyo morning.",
                    "hashtags": ["#Tokyo", "#Coffee", "#Shorts"],
                },
            },
        )


class TestGeneralSemanticVerifierLLM(unittest.TestCase):
    @staticmethod
    def _spec_response(requirement: str, **overrides) -> str:
        spec = {
            "requirement_id": 0,
            "original_requirement": requirement,
            "subjects": ["Worker"],
            "primary_action": "installs",
            "objects": ["tire"],
            "required_relations": [],
            "required_context": [],
            "required_visible_state": [],
            "optional_attributes": [],
            "critical_visual_facts": [
                {
                    "id": "f1",
                    "fact": requirement,
                    "mandatory": True,
                    "direct_evidence_needed": True,
                    "evidence_description": (
                        "The worker visibly installs the tire."
                    ),
                    "basis_type": "logically_necessary",
                    "basis_quote": requirement,
                }
            ],
            "ambiguity_notes": [],
        }
        spec.update(overrides)
        return json.dumps({"specs": [spec]})

    def test_requirement_decomposition_is_cached_per_normalized_unique_text(self):
        requirement = "Worker installs a new tire"
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    llm,
                    "_visual_requirement_spec_cache_dir",
                    return_value=Path(temp_dir),
                ),
                patch.object(
                    llm,
                    "_selected_llm_identity",
                    return_value=("test-provider", "test-model"),
                ),
                patch.object(
                    llm,
                    "_generate_response",
                    return_value=self._spec_response(requirement),
                ) as generate,
            ):
                first = llm.generate_visual_requirement_specs(
                    [requirement, f"  {requirement}  "]
                )
                second = llm.generate_visual_requirement_specs([requirement])

        normalized = llm.normalize_visual_requirement(requirement)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(set(first), {normalized})
        self.assertEqual(set(second), {normalized})
        self.assertEqual(first[normalized].generator_provider, "test-provider")

    def test_malformed_decomposition_does_not_invent_a_spec(self):
        with (
            patch.object(
                llm,
                "_selected_llm_identity",
                return_value=("test-provider", "test-model"),
            ),
            patch.object(llm, "_load_visual_requirement_spec_cache", return_value=None),
            patch.object(llm, "_generate_response", return_value='{"specs": []}'),
        ):
            result = llm.generate_visual_requirement_specs(
                ["Worker installs a new tire"]
            )

        self.assertEqual(result, {})

    @classmethod
    def _specs_response(cls, requirements: list[str]) -> str:
        """Build one well-formed batch response for the requested requirements."""
        specs = []
        for requirement_id, requirement in enumerate(requirements):
            spec = json.loads(cls._spec_response(requirement))["specs"][0]
            spec["requirement_id"] = requirement_id
            specs.append(spec)
        return json.dumps({"specs": specs})

    @staticmethod
    def _requested_requirements(prompt: str) -> list[str]:
        payload = json.loads(prompt.rsplit("Inputs:", 1)[1].strip())
        return [item["visual_requirement"] for item in payload]

    @staticmethod
    def _batch_requirements() -> list[str]:
        # Every requirement keeps the words the spec fields are grounded in, so
        # the strict validator stays in play while the batch size is exercised.
        return [
            f"Worker installs a new tire {suffix}"
            for suffix in ("one", "two", "three", "four", "five", "six")
        ]

    def test_decomposition_is_requested_in_small_batches(self):
        """整条时间线一次请求会被 Provider 截断，因此必须分批请求。"""
        requirements = self._batch_requirements()
        batch_sizes: list[int] = []

        def fake_generate(prompt, app_config=None):
            requested = self._requested_requirements(prompt)
            batch_sizes.append(len(requested))
            return self._specs_response(requested)

        with (
            patch.object(
                llm, "_selected_llm_identity", return_value=("test-provider", "test-model")
            ),
            patch.object(llm, "_load_visual_requirement_spec_cache", return_value=None),
            patch.object(llm, "_save_visual_requirement_spec_cache"),
            patch.object(llm, "_generate_response", side_effect=fake_generate),
        ):
            resolved = llm.generate_visual_requirement_specs(requirements)

        self.assertEqual(batch_sizes, [llm._VISUAL_REQUIREMENT_BATCH_SIZE, 2])
        self.assertEqual(
            set(resolved),
            {llm.normalize_visual_requirement(value) for value in requirements},
        )

    def test_one_unusable_batch_does_not_discard_the_usable_ones(self):
        """单个批次不可用时，已解析成功的批次必须保留。"""
        requirements = self._batch_requirements()
        batch_sizes: list[int] = []

        def fake_generate(prompt, app_config=None):
            requested = self._requested_requirements(prompt)
            batch_sizes.append(len(requested))
            if len(requested) == llm._VISUAL_REQUIREMENT_BATCH_SIZE:
                return self._specs_response(requested)
            return "the provider explained itself instead of answering"

        with (
            patch.object(
                llm, "_selected_llm_identity", return_value=("test-provider", "test-model")
            ),
            patch.object(llm, "_load_visual_requirement_spec_cache", return_value=None),
            patch.object(llm, "_save_visual_requirement_spec_cache"),
            patch.object(llm, "_generate_response", side_effect=fake_generate),
        ):
            resolved = llm.generate_visual_requirement_specs(requirements)

        self.assertEqual(
            batch_sizes,
            [llm._VISUAL_REQUIREMENT_BATCH_SIZE]
            + [2] * llm._MAX_STRUCTURED_RESPONSE_ATTEMPTS,
        )
        self.assertEqual(
            set(resolved),
            {
                llm.normalize_visual_requirement(value)
                for value in requirements[: llm._VISUAL_REQUIREMENT_BATCH_SIZE]
            },
        )

    def test_structured_payload_is_recovered_from_a_chatty_reply(self):
        """Provider 在 JSON 前后添加解释时也必须能解析。"""
        requirement = "Worker installs a new tire"
        chatty = (
            "Sure, here is the decomposition:\n```json\n"
            f"{self._spec_response(requirement)}\n"
            "```\nLet me know if you need changes."
        )

        with (
            patch.object(
                llm, "_selected_llm_identity", return_value=("test-provider", "test-model")
            ),
            patch.object(llm, "_load_visual_requirement_spec_cache", return_value=None),
            patch.object(llm, "_save_visual_requirement_spec_cache"),
            patch.object(llm, "_generate_response", return_value=chatty) as generate,
        ):
            resolved = llm.generate_visual_requirement_specs([requirement])

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(
            set(resolved), {llm.normalize_visual_requirement(requirement)}
        )

    def test_unparsable_structured_response_is_retried_before_giving_up(self):
        """空响应（被截断或被拒答）必须重试一次后才放弃。"""
        with (
            patch.object(
                llm, "_selected_llm_identity", return_value=("test-provider", "test-model")
            ),
            patch.object(llm, "_load_visual_requirement_spec_cache", return_value=None),
            patch.object(llm, "_generate_response", return_value="") as generate,
        ):
            resolved = llm.generate_visual_requirement_specs(
                ["Worker installs a new tire"]
            )

        self.assertEqual(resolved, {})
        self.assertEqual(generate.call_count, llm._MAX_STRUCTURED_RESPONSE_ATTEMPTS)

    def test_provider_unavailability_is_not_retried(self):
        """Provider 不可用时 _generate_response 已内部重试，这里不再重复请求。"""
        with (
            patch.object(
                llm, "_selected_llm_identity", return_value=("test-provider", "test-model")
            ),
            patch.object(llm, "_load_visual_requirement_spec_cache", return_value=None),
            patch.object(
                llm, "_generate_response", return_value="Error: provider exploded"
            ) as generate,
        ):
            resolved = llm.generate_visual_requirement_specs(
                ["Worker installs a new tire"]
            )

        self.assertEqual(resolved, {})
        self.assertEqual(generate.call_count, 1)

    def test_configured_output_ceiling_overrides_the_default(self):
        """显式上限优先；无效值回落到默认值，避免再次截断结构化响应。"""
        self.assertEqual(
            llm._resolved_max_output_tokens({"llm_max_output_tokens": 4096}), 4096
        )
        self.assertEqual(
            llm._resolved_max_output_tokens({}), llm._DEFAULT_MAX_OUTPUT_TOKENS
        )
        for invalid in ("", "abc", 0, -5, None):
            self.assertEqual(
                llm._resolved_max_output_tokens({"llm_max_output_tokens": invalid}),
                llm._DEFAULT_MAX_OUTPUT_TOKENS,
            )

    def test_json_payload_recovery_accepts_only_complete_documents(self):
        """截断的响应必须保持不可解析，不能被"恢复"成其中一个内层对象。"""
        payload = self._spec_response("Worker installs a new tire")
        array_payload = json.dumps([{"a": 1}, {"b": 2}])

        for raw, expected in (
            (payload, payload),
            (f"```json\n{payload}\n```", payload),
            (f"Here you go:\n{payload}\nHope that helps.", payload),
            (array_payload, array_payload),
            (f"```\n{array_payload}\n```", array_payload),
            (f"here: {array_payload} done", array_payload),
        ):
            with self.subTest(recover=raw[:40]):
                self.assertEqual(
                    json.loads(llm._extract_json_payload(raw)), json.loads(expected)
                )

        for raw in (
            "",
            "   ",
            "I cannot help with that.",
            payload[: len(payload) // 2],
            # A truncated array still contains a complete first object; returning
            # it would look like a successful parse of a partial timeline.
            array_payload[:12],
            f"```json\n{payload[: len(payload) // 2]}",
        ):
            with self.subTest(reject=raw[:40]):
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(llm._extract_json_payload(raw))

    def test_response_diagnostic_reports_length_and_a_bounded_preview(self):
        """空响应、截断响应与拒答必须在日志中可区分。"""
        self.assertEqual(
            llm._response_diagnostic(""),
            "response_length=0, response_preview=''",
        )

        long_diagnostic = llm._response_diagnostic("x" * 5000)
        self.assertIn("response_length=5000", long_diagnostic)
        self.assertIn("x" * llm._LOGGED_RESPONSE_PREVIEW_LENGTH, long_diagnostic)
        self.assertNotIn(
            "x" * (llm._LOGGED_RESPONSE_PREVIEW_LENGTH + 1), long_diagnostic
        )

        multiline = llm._response_diagnostic("line one\nline two")
        self.assertIn("line one line two", multiline)
        self.assertNotIn("\n", multiline)

    def test_unsupported_mandatory_environment_is_rejected_by_validator(self):
        requirement = "Worker installs a new tire"
        raw = json.loads(
            self._spec_response(
                requirement,
                required_context=["sunny outdoor workshop"],
                critical_visual_facts=[
                    {
                        "id": "f1",
                        "fact": "Worker installs a new tire in sunny weather",
                        "mandatory": True,
                        "direct_evidence_needed": True,
                        "evidence_description": "The installation happens in sun.",
                        "basis_type": "explicit",
                        "basis_quote": requirement,
                    }
                ],
            )
        )["specs"][0]

        with self.assertRaisesRegex(ValueError, "source-grounded|unsupported"):
            llm._validate_visual_requirement_spec_item(
                raw,
                requirement_id=0,
                original_requirement=requirement,
                provider_id="test-provider",
                model_name="test-model",
            )

    def test_action_requires_a_defining_logically_necessary_fact(self):
        requirement = "Worker installs a new tire"
        raw = json.loads(self._spec_response(requirement))["specs"][0]
        raw["critical_visual_facts"][0]["basis_type"] = "explicit"

        with self.assertRaisesRegex(
            ValueError,
            "defining logically necessary evidence fact",
        ):
            llm._validate_visual_requirement_spec_item(
                raw,
                requirement_id=0,
                original_requirement=requirement,
                provider_id="test-provider",
                model_name="test-model",
            )

    def test_logically_necessary_action_fact_may_use_grounded_paraphrase(self):
        requirement = "Worker installs a new tire"
        raw = json.loads(self._spec_response(requirement))["specs"][0]
        raw["critical_visual_facts"][0].update(
            {
                "fact": (
                    "The worker visibly fits and secures the new tire into its "
                    "installed position rather than only holding it"
                ),
                "basis_quote": requirement,
            }
        )

        spec = llm._validate_visual_requirement_spec_item(
            raw,
            requirement_id=0,
            original_requirement=requirement,
            provider_id="test-provider",
            model_name="test-model",
        )

        self.assertEqual(
            spec.critical_visual_facts[0].basis_type,
            "logically_necessary",
        )

    def test_continuous_visible_state_does_not_require_invented_action(self):
        requirement = "Coffee beans are dark brown inside the roaster"
        raw = json.loads(
            self._spec_response(
                requirement,
                subjects=["Coffee beans"],
                primary_action=None,
                objects=["roaster"],
                required_visible_state=["Coffee beans are dark brown"],
                critical_visual_facts=[
                    {
                        "id": "f1",
                        "fact": requirement,
                        "mandatory": True,
                        "direct_evidence_needed": False,
                        "evidence_description": "Dark brown beans are visible.",
                        "basis_type": "explicit",
                        "basis_quote": requirement,
                    }
                ],
            )
        )["specs"][0]

        spec = llm._validate_visual_requirement_spec_item(
            raw,
            requirement_id=0,
            original_requirement=requirement,
            provider_id="test-provider",
            model_name="test-model",
        )

        self.assertIsNone(spec.primary_action)
        self.assertFalse(spec.critical_visual_facts[0].direct_evidence_needed)

    def test_text_adjudicator_cannot_modify_supplied_fact_status(self):
        requirement = "Worker installs a new tire"
        raw = json.loads(self._spec_response(requirement))["specs"][0]
        spec = llm._validate_visual_requirement_spec_item(
            raw,
            requirement_id=0,
            original_requirement=requirement,
            provider_id="test-provider",
            model_name="test-model",
        )
        candidate = {
            "candidate_id": "pexels:123",
            "observed_facts": {
                "critical_fact_evidence": [
                    {
                        "fact_id": "f1",
                        "status": "OBSERVED",
                        "evidence": "The installation itself is directly visible.",
                    }
                ]
            },
        }
        accepted_response = json.dumps(
            {
                "decisions": [
                    {
                        "candidate_id": "pexels:123",
                        "decision": "ACCEPT",
                        "mandatory_fact_results": [
                            {"fact_id": "f1", "status": "OBSERVED"}
                        ],
                        "missing_or_contradictory_facts": [],
                        "reason": "All mandatory evidence is directly observed.",
                    }
                ]
            }
        )
        modified_response = accepted_response.replace(
            '"status": "OBSERVED"',
            '"status": "NOT_OBSERVED"',
        )

        # Both calls must actually reach the adjudicator: this test is about
        # rejecting a tampered response, and the verdict cache would otherwise
        # answer the second call from the first one's stored ACCEPT.
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    llm,
                    "_semantic_adjudication_cache_dir",
                    return_value=Path(temp_dir),
                ),
                patch.object(
                    llm, "_load_semantic_adjudication_cache", return_value=None
                ),
            ):
                with patch.object(
                    llm, "_generate_response", return_value=accepted_response
                ):
                    accepted = llm.adjudicate_visual_candidates(spec, [candidate])
                with patch.object(
                    llm, "_generate_response", return_value=modified_response
                ):
                    rejected_as_malformed = llm.adjudicate_visual_candidates(
                        spec, [candidate]
                    )

        self.assertEqual(accepted["pexels:123"].decision, "ACCEPT")
        self.assertEqual(rejected_as_malformed, {})


class TestSemanticAdjudicationCache(unittest.TestCase):
    """The paid observation behind a verdict is already cached; the verdict was not.

    Re-running the same script therefore used to cost nothing at the video model and
    full token price at the adjudicator. These tests pin down the two things that
    make reuse safe: the key must cover everything the verdict depends on, and a
    stored verdict must be re-checked against the evidence of the run reusing it.
    """

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.cache_dir = Path(temp_dir.name)
        for patcher in (
            patch.object(
                llm,
                "_semantic_adjudication_cache_dir",
                return_value=self.cache_dir,
            ),
            patch.object(
                llm,
                "_selected_llm_identity",
                return_value=("test-provider", "test-model"),
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _spec(requirement: str = "Worker installs a new tire"):
        raw = json.loads(
            TestGeneralSemanticVerifierLLM._spec_response(requirement)
        )["specs"][0]
        return llm._validate_visual_requirement_spec_item(
            raw,
            requirement_id=0,
            original_requirement=requirement,
            provider_id="test-provider",
            model_name="test-model",
        )

    @staticmethod
    def _candidate(
        candidate_id: str,
        *,
        status: str = "OBSERVED",
        evidence: str = "The installation itself is directly visible.",
    ) -> dict:
        return {
            "candidate_id": candidate_id,
            "observed_facts": {
                "critical_fact_evidence": [
                    {"fact_id": "f1", "status": status, "evidence": evidence}
                ]
            },
        }

    @staticmethod
    def _decision(
        candidate_id: str,
        *,
        decision: str = "ACCEPT",
        status: str = "OBSERVED",
        reason: str = "All mandatory evidence is directly observed.",
        missing: list | None = None,
    ) -> dict:
        return {
            "candidate_id": candidate_id,
            "decision": decision,
            "mandatory_fact_results": [{"fact_id": "f1", "status": status}],
            "missing_or_contradictory_facts": list(missing or []),
            "reason": reason,
        }

    @staticmethod
    def _response(*decisions: dict) -> str:
        return json.dumps({"decisions": list(decisions)})

    def _digest(self, spec, candidate: dict, **overrides) -> str:
        params = {
            "candidate_id": candidate["candidate_id"],
            "observed_facts": candidate["observed_facts"],
            "requirement_spec_digest": llm.visual_requirement_spec_digest(spec),
            "provider_id": "test-provider",
            "model_name": "test-model",
        }
        params.update(overrides)
        return llm._semantic_adjudication_cache_digest(
            params["candidate_id"],
            params["observed_facts"],
            params["requirement_spec_digest"],
            params["provider_id"],
            params["model_name"],
        )

    def _adjudicate(self, spec, candidates: list[dict], response: str | None = None):
        """Run one adjudication, returning the verdicts and the provider mock."""
        with patch.object(
            llm, "_generate_response", return_value=response
        ) as generate:
            verdicts = llm.adjudicate_visual_candidates(spec, candidates)
        return verdicts, generate

    def test_a_repeated_verdict_is_reused_without_paying_for_it_again(self):
        spec = self._spec()
        candidate = self._candidate("pexels:1")

        first, first_generate = self._adjudicate(
            spec, [candidate], self._response(self._decision("pexels:1"))
        )
        second, second_generate = self._adjudicate(spec, [candidate])

        self.assertEqual(first_generate.call_count, 1)
        second_generate.assert_not_called()
        self.assertEqual(first["pexels:1"].decision, "ACCEPT")
        self.assertEqual(second["pexels:1"].decision, "ACCEPT")
        self.assertEqual(second["pexels:1"].reason, first["pexels:1"].reason)
        self.assertEqual(
            second["pexels:1"].mandatory_fact_results[0].status, "OBSERVED"
        )

    def test_a_rejection_is_cached_too(self):
        # This is where the money actually is: a failing beat exhausts its whole
        # candidate budget, so its rejections are the bulk of the spend.
        spec = self._spec()
        candidate = self._candidate(
            "pexels:1",
            status="NOT_OBSERVED",
            evidence="Only a parked car is visible.",
        )
        rejection = self._response(
            self._decision(
                "pexels:1",
                decision="REJECT",
                status="NOT_OBSERVED",
                reason="The defining installation is never shown.",
                missing=["f1"],
            )
        )

        first, first_generate = self._adjudicate(spec, [candidate], rejection)
        second, second_generate = self._adjudicate(spec, [candidate])

        self.assertEqual(first_generate.call_count, 1)
        second_generate.assert_not_called()
        self.assertEqual(second["pexels:1"].decision, "REJECT")
        self.assertEqual(second["pexels:1"].missing_or_contradictory_facts, ["f1"])

    def test_only_the_uncached_candidates_are_sent_to_the_provider(self):
        spec = self._spec()
        cached = self._candidate("pexels:1")
        fresh = self._candidate(
            "pexels:2", evidence="A second clip also shows the tire being secured."
        )

        self._adjudicate(spec, [cached], self._response(self._decision("pexels:1")))
        both, generate = self._adjudicate(
            spec, [cached, fresh], self._response(self._decision("pexels:2"))
        )

        self.assertEqual(generate.call_count, 1)
        prompt = generate.call_args.args[0]
        self.assertIn("pexels:2", prompt)
        self.assertNotIn("pexels:1", prompt)
        self.assertEqual(set(both), {"pexels:1", "pexels:2"})

    def test_a_verdict_is_not_reused_once_the_evidence_changes(self):
        spec = self._spec()
        original = self._candidate("pexels:1")
        reobserved = self._candidate(
            "pexels:1", evidence="On review only the wheel is held, never fitted."
        )

        self._adjudicate(spec, [original], self._response(self._decision("pexels:1")))
        second, generate = self._adjudicate(
            spec,
            [reobserved],
            self._response(
                self._decision(
                    "pexels:1",
                    decision="UNCERTAIN",
                    reason="Holding the wheel does not establish the fitting.",
                )
            ),
        )

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(second["pexels:1"].decision, "UNCERTAIN")

    def test_a_verdict_is_not_reused_once_the_requirement_changes(self):
        candidate = self._candidate("pexels:1")
        first_spec = self._spec()
        # Still grounded in "installs", so this stays a valid spec — only the
        # requirement it was decomposed from differs.
        other_spec = self._spec("Worker installs a new tire at the roadside")

        self.assertNotEqual(
            llm.visual_requirement_spec_digest(first_spec),
            llm.visual_requirement_spec_digest(other_spec),
        )

        self._adjudicate(
            first_spec, [candidate], self._response(self._decision("pexels:1"))
        )
        _, generate = self._adjudicate(
            other_spec, [candidate], self._response(self._decision("pexels:1"))
        )

        self.assertEqual(generate.call_count, 1)

    def test_a_stored_accept_cannot_outlive_the_evidence_that_earned_it(self):
        # The file name is a hash of the evidence, so this can only happen through
        # a hand-edited or corrupted cache. It must still never be honored.
        spec = self._spec()
        candidate = self._candidate("pexels:1", status="NOT_OBSERVED")
        digest = self._digest(spec, candidate)
        llm._semantic_adjudication_cache_path(digest).write_text(
            json.dumps(
                {
                    "version": llm._SEMANTIC_ADJUDICATION_CACHE_FORMAT_VERSION,
                    "schema_version": llm.SEMANTIC_ADJUDICATION_SCHEMA_VERSION,
                    "decision": self._decision("pexels:1", status="OBSERVED"),
                }
            ),
            encoding="utf-8",
        )

        verdicts, generate = self._adjudicate(
            spec,
            [candidate],
            self._response(
                self._decision(
                    "pexels:1",
                    decision="REJECT",
                    status="NOT_OBSERVED",
                    reason="The mandatory fact is not observed.",
                    missing=["f1"],
                )
            ),
        )

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(verdicts["pexels:1"].decision, "REJECT")

    def test_a_superseded_schema_version_is_not_honored(self):
        spec = self._spec()
        candidate = self._candidate("pexels:1")
        digest = self._digest(spec, candidate)
        llm._semantic_adjudication_cache_path(digest).write_text(
            json.dumps(
                {
                    "version": llm._SEMANTIC_ADJUDICATION_CACHE_FORMAT_VERSION,
                    "schema_version": "semantic-adjudication-from-older-rules",
                    "decision": self._decision("pexels:1"),
                }
            ),
            encoding="utf-8",
        )

        _, generate = self._adjudicate(
            spec, [candidate], self._response(self._decision("pexels:1"))
        )

        self.assertEqual(generate.call_count, 1)

    def test_the_key_covers_the_model_that_reached_the_verdict(self):
        spec = self._spec()
        candidate = self._candidate("pexels:1")
        baseline = self._digest(spec, candidate)

        self.assertNotEqual(
            baseline, self._digest(spec, candidate, provider_id="other-provider")
        )
        self.assertNotEqual(
            baseline, self._digest(spec, candidate, model_name="other-model")
        )
        self.assertNotEqual(
            baseline,
            self._digest(spec, self._candidate("pexels:2")),
        )

    def test_a_damaged_cache_file_costs_one_request_not_correctness(self):
        spec = self._spec()
        candidate = self._candidate("pexels:1")
        llm._semantic_adjudication_cache_path(
            self._digest(spec, candidate)
        ).write_text("{ this is not json", encoding="utf-8")

        verdicts, generate = self._adjudicate(
            spec, [candidate], self._response(self._decision("pexels:1"))
        )

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(verdicts["pexels:1"].decision, "ACCEPT")

    def test_an_unreachable_cache_costs_tokens_never_correctness(self):
        spec = self._spec()
        candidate = self._candidate("pexels:1")

        with patch.object(
            llm,
            "_semantic_adjudication_cache_digest",
            side_effect=OSError("storage is unavailable"),
        ):
            verdicts, generate = self._adjudicate(
                spec, [candidate], self._response(self._decision("pexels:1"))
            )

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(verdicts["pexels:1"].decision, "ACCEPT")

    def test_a_malformed_batch_leaves_no_verdict_behind(self):
        spec = self._spec()
        good = self._candidate("pexels:1")
        bad = self._candidate(
            "pexels:2", evidence="A second angle on the same fitting."
        )
        # One valid decision and one that tampers with the supplied status. The whole
        # response is refused, so neither may be stored.
        response = self._response(
            self._decision("pexels:1"),
            self._decision("pexels:2", status="NOT_OBSERVED"),
        )

        verdicts, _ = self._adjudicate(spec, [good, bad], response)

        self.assertEqual(verdicts, {})
        self.assertEqual(sorted(self.cache_dir.iterdir()), [])

    def test_a_verdict_survives_a_cache_that_cannot_be_written(self):
        spec = self._spec()
        candidate = self._candidate("pexels:1")

        with patch.object(
            llm,
            "_save_semantic_adjudication_cache",
            side_effect=OSError("read-only storage"),
        ):
            verdicts, _ = self._adjudicate(
                spec, [candidate], self._response(self._decision("pexels:1"))
            )

        self.assertEqual(verdicts["pexels:1"].decision, "ACCEPT")


class TestAlternativeVisualRequirements(unittest.TestCase):
    """A rewritten requirement is the last thing standing between one unfillable
    beat and a failed video, so an ungrounded rewrite is worse than none at all:
    it would quietly render a different scene than the narration asked for."""

    NARRATION = "Workers once inspected every railway track by hand"
    FAILED = "Workers inspecting railway tracks at sunrise with lanterns"
    ALTERNATIVE = "Two workers walking along railway tracks"
    BASIS = "inspected every railway track"

    def _item(self, **overrides) -> dict:
        item = {
            "item_index": 4,
            "narration_text": self.NARRATION,
            "failed_requirement": self.FAILED,
            "problem": "no candidate passed semantic verification",
        }
        item.update(overrides)
        return item

    def _alternative(self, **overrides) -> dict:
        entry = {
            "item_id": 0,
            "visual_requirement": self.ALTERNATIVE,
            "narration_basis": self.BASIS,
        }
        entry.update(overrides)
        return entry

    @staticmethod
    def _response(alternatives: list) -> str:
        return json.dumps({"alternatives": alternatives})

    @staticmethod
    def _requested_items(prompt: str) -> list[dict]:
        return json.loads(prompt.rsplit("Inputs:", 1)[1].strip())

    def _run(self, response: str, items: list | None = None):
        """Run one rewrite request against a canned provider reply."""
        with patch.object(llm, "_generate_response", return_value=response) as generate:
            resolved = llm.generate_alternative_visual_requirements(
                [self._item()] if items is None else items
            )
        return resolved, generate

    def test_grounded_alternative_is_accepted_under_the_callers_index(self):
        response = self._response(
            [
                self._alternative(
                    visual_requirement="  Two workers   walking\nalong railway tracks "
                )
            ]
        )

        resolved, generate = self._run(response)

        self.assertEqual(
            resolved,
            {
                4: {
                    "visual_requirement": self.ALTERNATIVE,
                    "narration_basis": self.BASIS,
                }
            },
        )
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(
            self._requested_items(generate.call_args[0][0]),
            [
                {
                    "item_id": 0,
                    "spoken_text": self.NARRATION,
                    "rejected_visual_requirement": self.FAILED,
                    "why_it_failed": "no candidate passed semantic verification",
                }
            ],
        )

    def test_an_ungrounded_proposal_is_dropped_without_re_asking(self):
        """每个被丢弃的提案都不得再次请求 Provider，否则一个坏 Beat 会被反复计费。"""
        cases = {
            "narration_basis is not a quote of this line": self._alternative(
                narration_basis="workers wearing bright hard hats"
            ),
            "narration_basis words are not contiguous": self._alternative(
                narration_basis="workers inspected track"
            ),
            "narration_basis is a single word": self._alternative(
                narration_basis="railway"
            ),
            "the rejected wording is returned again": self._alternative(
                visual_requirement=(
                    "  workers INSPECTING railway tracks at sunrise with lanterns "
                )
            ),
            "the alternative carries no searchable letters": self._alternative(
                visual_requirement="کارگران روی ریل راه‌آهن"
            ),
        }
        for label, alternative in cases.items():
            with self.subTest(label):
                resolved, generate = self._run(self._response([alternative]))

                self.assertEqual(resolved, {})
                self.assertEqual(generate.call_count, 1)

    def test_an_unusable_item_is_never_sent_to_the_provider(self):
        long_text = "word " * 200
        cases = {
            "no items at all": [],
            "item_index is not an integer": [self._item(item_index="4")],
            "item_index is a bool": [self._item(item_index=True)],
            "the beat has no spoken text": [self._item(narration_text="   ")],
            "the spoken text is too long": [self._item(narration_text=long_text)],
            "the rejected requirement is too long": [
                self._item(failed_requirement=long_text)
            ],
        }
        for label, items in cases.items():
            with self.subTest(label):
                resolved, generate = self._run("unused", items=items)

                self.assertEqual(resolved, {})
                generate.assert_not_called()

    def test_a_malformed_batch_is_discarded_without_re_asking(self):
        cases = {
            "item_id is outside the batch": self._response(
                [self._alternative(item_id=7)]
            ),
            "item_id is not an integer": self._response(
                [self._alternative(item_id="0")]
            ),
            "item_id is a bool": self._response([self._alternative(item_id=True)]),
            "more alternatives than inputs": self._response(
                [self._alternative(), self._alternative(item_id=1)]
            ),
            "alternatives is not an array": json.dumps({"alternatives": {}}),
            "an alternative is not an object": self._response(["Two workers walking"]),
            "visual_requirement is missing": self._response(
                [{"item_id": 0, "narration_basis": self.BASIS}]
            ),
            "narration_basis is missing": self._response(
                [{"item_id": 0, "visual_requirement": self.ALTERNATIVE}]
            ),
        }
        for label, response in cases.items():
            with self.subTest(label):
                resolved, generate = self._run(response)

                self.assertEqual(resolved, {})
                self.assertEqual(generate.call_count, 1)

    def test_a_duplicated_item_id_discards_even_the_well_formed_sibling(self):
        """item_id 重复意味着无法判断哪个 Beat 对应哪个提案，整批必须丢弃。"""
        items = [self._item(item_index=4), self._item(item_index=9)]

        resolved, generate = self._run(
            self._response([self._alternative(), self._alternative()]),
            items=items,
        )

        self.assertEqual(resolved, {})
        self.assertEqual(generate.call_count, 1)

    def test_an_omitted_item_stays_unfilled_while_its_sibling_is_answered(self):
        items = [self._item(item_index=4), self._item(item_index=9)]

        resolved, generate = self._run(
            self._response([self._alternative(item_id=1)]), items=items
        )

        self.assertEqual(set(resolved), {9})
        self.assertEqual(resolved[9]["visual_requirement"], self.ALTERNATIVE)
        requested = self._requested_items(generate.call_args[0][0])
        self.assertEqual([entry["item_id"] for entry in requested], [0, 1])

    def test_requests_are_batched_and_one_bad_batch_keeps_the_good_one(self):
        items = [
            self._item(item_index=index, failed_requirement=f"{self.FAILED} {index}")
            for index in range(llm._VISUAL_REQUIREMENT_BATCH_SIZE + 2)
        ]
        batch_sizes: list[int] = []

        def fake_generate(prompt, app_config=None):
            requested = self._requested_items(prompt)
            batch_sizes.append(len(requested))
            if len(requested) < llm._VISUAL_REQUIREMENT_BATCH_SIZE:
                return "the provider explained itself instead of answering"
            return self._response(
                [self._alternative(item_id=entry["item_id"]) for entry in requested]
            )

        with patch.object(llm, "_generate_response", side_effect=fake_generate):
            resolved = llm.generate_alternative_visual_requirements(items)

        self.assertEqual(
            batch_sizes,
            [llm._VISUAL_REQUIREMENT_BATCH_SIZE]
            + [2] * llm._MAX_STRUCTURED_RESPONSE_ATTEMPTS,
        )
        self.assertEqual(set(resolved), set(range(llm._VISUAL_REQUIREMENT_BATCH_SIZE)))

    def test_a_long_failure_diagnosis_is_bounded_but_keeps_the_item(self):
        resolved, generate = self._run(
            self._response([self._alternative()]),
            items=[self._item(problem="rejected " * 200)],
        )

        requested = self._requested_items(generate.call_args[0][0])
        self.assertEqual(
            len(requested[0]["why_it_failed"]), llm._MAX_STRUCTURED_TEXT_LENGTH
        )
        self.assertEqual(set(resolved), {4})

    def test_provider_unavailability_is_not_retried_per_beat(self):
        resolved, generate = self._run("Error: provider is unavailable")

        self.assertEqual(resolved, {})
        self.assertEqual(generate.call_count, 1)

    def test_narration_grounding_requires_a_contiguous_quote(self):
        cases = [
            ("an exact quote", self.NARRATION, self.BASIS, True),
            (
                "punctuation and case differ",
                self.NARRATION,
                "Inspected, EVERY railway track!",
                True,
            ),
            ("the whole spoken line", self.NARRATION, self.NARRATION, True),
            ("scattered words", self.NARRATION, "workers inspected track", False),
            ("a single word", self.NARRATION, "railway", False),
            ("empty narration", "", self.BASIS, False),
            ("empty quote", self.NARRATION, "   ", False),
            (
                "a quote in the narration's own language",
                "کارگران ریل راه آهن را بازرسی می‌کردند",
                "ریل راه آهن",
                True,
            ),
            (
                "a quote absent from the narration's own language",
                "کارگران ریل راه آهن را بازرسی می‌کردند",
                "برداشت گیلاس",
                False,
            ),
        ]
        for label, narration, quote, expected in cases:
            with self.subTest(label):
                self.assertIs(
                    llm._narration_contains_quote(narration, quote), expected
                )


class TestNarrationVisualRequirementRepair(unittest.TestCase):
    """The call that turns spoken narration lines into filmable requirements.

    It runs only when semantic grouping already failed, so its own failures must
    stay legible: a line the provider cannot describe visually has to come back
    absent, never as an invented scene and never as the spoken line itself.
    """

    @staticmethod
    def _lines(count):
        return [
            {"index": index, "spoken_text": f"Narration line {index}."}
            for index in range(1, count + 1)
        ]

    def test_an_empty_answer_marks_a_line_as_having_no_visible_content(self):
        payload = (
            '[{"index":1,"visual_requirement":"A seed cracking open in dark soil"},'
            '{"index":2,"visual_requirement":""},'
            '{"index":3,"visual_requirement":"  "}]'
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            result = llm.generate_narration_visual_requirements(
                narration_text="A seed cracks open. Patient. Not loud.",
                narration_lines=self._lines(3),
            )

        # Absent, not empty-stringed: the caller attaches such a line to a
        # neighbour, and an empty requirement would reach stock search instead.
        self.assertEqual(result, {1: "A seed cracking open in dark soil"})

    def test_an_unusable_object_never_discards_the_lines_that_came_back(self):
        payload = (
            '[{"index":1,"visual_requirement":"Roots pushing down through soil"},'
            '{"index":"2","visual_requirement":"Leaves reaching up"},'
            '{"index":3,"visual_requirement":42},'
            '{"index":4,"visual_requirement":"Forest canopy at dawn",'
            '"confidence":0.9},'
            '{"index":9,"visual_requirement":"A line that was never requested"},'
            '{"index":1,"visual_requirement":"A duplicate answer"}]'
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            result = llm.generate_narration_visual_requirements(
                narration_text="Roots push down.",
                narration_lines=self._lines(4),
            )

        self.assertEqual(result, {1: "Roots pushing down through soil"})

    def test_the_prompt_carries_every_requested_line_and_forbids_invention(self):
        captured = {}

        def fake_generate_response(prompt, app_config=None):
            captured["prompt"] = prompt
            return '[{"index":1,"visual_requirement":"A single water drop"}]'

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ) as generate:
            result = llm.generate_narration_visual_requirements(
                narration_text="It starts with a single drop of water. One drop.",
                narration_lines=[
                    {"index": 1, "spoken_text": "It starts with a single drop."},
                    {"index": 2, "spoken_text": "One drop."},
                ],
            )

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(result, {1: "A single water drop"})
        self.assertIn('1|"It starts with a single drop."', captured["prompt"])
        self.assertIn('2|"One drop."', captured["prompt"])
        # The narration is context for resolving "One drop.", and inventing a
        # subject is the failure mode that makes a requirement unfillable.
        self.assertIn("It starts with a single drop of water.", captured["prompt"])
        self.assertIn("Add no fact the narration does not support", captured["prompt"])
        self.assertIn("must return an empty string", captured["prompt"])

    def test_provider_failure_reads_as_unavailable_not_as_nothing_visible(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: provider unavailable"
        ):
            unavailable = llm.generate_narration_visual_requirements(
                narration_text="Roots push down.",
                narration_lines=self._lines(2),
            )

        # None and {} mean different things to the caller: one keeps the spoken
        # requirements for the pipeline to reject, the other would consolidate a
        # timeline around the claim that no line is filmable.
        self.assertIsNone(unavailable)

        with patch.object(llm, "_generate_response", return_value="[]") as generate:
            nothing_visible = llm.generate_narration_visual_requirements(
                narration_text="Patient. Not loud.",
                narration_lines=self._lines(2),
            )

        self.assertEqual(nothing_visible, {})
        self.assertEqual(generate.call_count, 1)

    def test_a_long_narration_is_asked_for_in_bounded_batches(self):
        line_count = llm._NARRATION_REQUIREMENT_BATCH_SIZE + 3
        requested_batches = []

        def fake_generate_response(prompt, app_config=None):
            indexes = [
                int(line.split("|", 1)[0])
                for line in prompt.splitlines()
                if line and line[0].isdigit() and "|" in line
            ]
            requested_batches.append(indexes)
            return json.dumps(
                [
                    {"index": index, "visual_requirement": f"Visible situation {index}"}
                    for index in indexes
                ]
            )

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ) as generate:
            result = llm.generate_narration_visual_requirements(
                narration_text="A long narration.",
                narration_lines=self._lines(line_count),
            )

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(
            len(requested_batches[0]),
            llm._NARRATION_REQUIREMENT_BATCH_SIZE,
        )
        self.assertEqual(len(requested_batches[1]), 3)
        self.assertEqual(len(result), line_count)
        self.assertEqual(result[line_count], f"Visible situation {line_count}")

    def test_a_failed_batch_keeps_the_batches_that_answered(self):
        line_count = llm._NARRATION_REQUIREMENT_BATCH_SIZE + 2
        responses = []

        def fake_generate_response(prompt, app_config=None):
            if not responses:
                responses.append("first")
                return json.dumps(
                    [{"index": 1, "visual_requirement": "A visible first situation"}]
                )
            return "not json"

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_narration_visual_requirements(
                narration_text="A long narration.",
                narration_lines=self._lines(line_count),
            )

        self.assertEqual(result, {1: "A visible first situation"})

    def test_a_line_without_spoken_text_is_never_sent(self):
        with patch.object(llm, "_generate_response") as generate:
            result = llm.generate_narration_visual_requirements(
                narration_text="   ",
                narration_lines=[
                    {"index": 1, "spoken_text": "   "},
                    {"index": True, "spoken_text": "A boolean is not a line index."},
                ],
            )

        generate.assert_not_called()
        self.assertIsNone(result)


class TestShotVisualRequirementSplit(unittest.TestCase):
    """The call that divides one span requirement across the shots it became.

    A span long enough to be split describes several moments at once, and every
    shot inherited that whole description. This call narrows each shot to the
    moment its own spoken text is about, so its failures must leave the parent
    requirement standing rather than replace it with something narrower than the
    narration supports.
    """

    PARENT = "A slow water drip carving a canyon through solid rock over time"

    @staticmethod
    def _spans(count, shots_per_span=2):
        return [
            {
                "span_requirement": f"Parent requirement {span_index}",
                "shots": [
                    {
                        "shot_id": (span_index - 1) * shots_per_span + shot_offset,
                        "spoken_text": (
                            f"Spoken fragment {span_index}.{shot_offset}"
                        ),
                    }
                    for shot_offset in range(1, shots_per_span + 1)
                ],
            }
            for span_index in range(1, count + 1)
        ]

    def _real_split_span(self):
        """The span from task 93b80e04, whose three shots all inherited PARENT."""
        return [
            {
                "span_requirement": self.PARENT,
                "shots": [
                    {
                        "shot_id": 1,
                        "spoken_text": "It starts with a single drop of water.",
                    },
                    {
                        "shot_id": 2,
                        "spoken_text": "Over a thousand years that drip carves",
                    },
                    {
                        "shot_id": 3,
                        "spoken_text": "solid rock, and the stone never fought back.",
                    },
                ],
            }
        ]

    def test_each_shot_is_asked_for_the_moment_its_own_words_describe(self):
        captured = {}

        def fake_generate_response(prompt, app_config=None):
            captured["prompt"] = prompt
            return (
                '[{"shot_id":1,"visual_requirement":"A single drop falling"},'
                '{"shot_id":2,"visual_requirement":"  Water   tracing a groove "},'
                '{"shot_id":3,"visual_requirement":"A canyon cut into bare stone"}]'
            )

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ) as generate:
            result = llm.generate_shot_visual_requirements(self._real_split_span())

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(
            result,
            {
                1: "A single drop falling",
                2: "Water tracing a groove",
                3: "A canyon cut into bare stone",
            },
        )
        # The parent is the boundary of the answer, and each shot's own words are
        # what decides which part of it belongs to that shot.
        self.assertIn(f'1|"{self.PARENT}"', captured["prompt"])
        self.assertIn(
            '1|1|"It starts with a single drop of water."', captured["prompt"]
        )
        self.assertIn(
            "introduce a subject the parent does not contain", captured["prompt"]
        )
        self.assertIn("must describe different moments", captured["prompt"])

    def test_a_shot_adding_no_visible_moment_comes_back_absent(self):
        payload = (
            '[{"shot_id":1,"visual_requirement":"A single drop falling"},'
            '{"shot_id":2,"visual_requirement":""},'
            '{"shot_id":3,"visual_requirement":"   "}]'
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            result = llm.generate_shot_visual_requirements(self._real_split_span())

        # Absent, not empty: the caller leaves such a shot on the parent
        # requirement, while an empty string would reach stock search.
        self.assertEqual(result, {1: "A single drop falling"})

    def test_an_unusable_object_never_discards_the_shots_that_came_back(self):
        payload = (
            '[{"shot_id":1,"visual_requirement":"A single drop falling"},'
            '{"shot_id":"2","visual_requirement":"Water tracing a groove"},'
            '{"shot_id":3,"visual_requirement":17},'
            '{"shot_id":2,"visual_requirement":"A canyon","span_id":1},'
            '{"shot_id":99,"visual_requirement":"A shot that was never requested"},'
            '{"shot_id":1,"visual_requirement":"A duplicate answer"}]'
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            result = llm.generate_shot_visual_requirements(self._real_split_span())

        self.assertEqual(result, {1: "A single drop falling"})

    def test_provider_failure_reads_as_unavailable_not_as_nothing_to_narrow(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: provider unavailable"
        ):
            unavailable = llm.generate_shot_visual_requirements(
                self._real_split_span()
            )

        # None and {} mean different things: one says the stage never ran, the
        # other says the provider found nothing in the parent worth narrowing.
        self.assertIsNone(unavailable)

        with patch.object(llm, "_generate_response", return_value="[]") as generate:
            nothing_to_narrow = llm.generate_shot_visual_requirements(
                self._real_split_span()
            )

        self.assertEqual(nothing_to_narrow, {})
        self.assertEqual(generate.call_count, 1)

    def test_many_split_spans_are_asked_for_in_bounded_batches(self):
        span_count = llm._SHOT_REQUIREMENT_SPANS_PER_BATCH + 2
        requested_batches = []

        def fake_generate_response(prompt, app_config=None):
            # Two blocks in this prompt start with a digit: parents carry two
            # fields and shots carry three, so they are told apart by width.
            shot_ids = []
            parents = set()
            for line in prompt.splitlines():
                if not line or not line[0].isdigit() or "|" not in line:
                    continue
                fields = line.split("|")
                if len(fields) == 3:
                    shot_ids.append(int(fields[1]))
                elif len(fields) == 2:
                    parents.add(int(fields[0]))
            requested_batches.append((sorted(parents), shot_ids))
            return json.dumps(
                [
                    {"shot_id": shot_id, "visual_requirement": f"Moment {shot_id}"}
                    for shot_id in shot_ids
                ]
            )

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ) as generate:
            result = llm.generate_shot_visual_requirements(self._spans(span_count))

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(
            len(requested_batches[0][0]),
            llm._SHOT_REQUIREMENT_SPANS_PER_BATCH,
        )
        self.assertEqual(len(requested_batches[1][0]), 2)
        # Every shot of a span is answered in the same request as its siblings,
        # because the whole point is that they must differ from each other.
        self.assertEqual(len(result), span_count * 2)
        self.assertEqual(requested_batches[1][1], [9, 10, 11, 12])
        self.assertEqual(result[12], "Moment 12")

    def test_a_failed_batch_keeps_the_spans_that_answered(self):
        span_count = llm._SHOT_REQUIREMENT_SPANS_PER_BATCH + 1
        answered = []

        def fake_generate_response(prompt, app_config=None):
            if not answered:
                answered.append("first")
                return json.dumps(
                    [{"shot_id": 1, "visual_requirement": "A single drop falling"}]
                )
            return "not json at all"

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_shot_visual_requirements(self._spans(span_count))

        # A failed batch costs only its own spans, which keep the parent and
        # behave exactly as they did before this stage existed.
        self.assertEqual(result, {1: "A single drop falling"})

    def test_a_span_with_one_shot_is_never_sent(self):
        over_long_parent = "y" * (llm._MAX_STRUCTURED_TEXT_LENGTH + 1)
        with patch.object(llm, "_generate_response") as generate:
            result = llm.generate_shot_visual_requirements(
                [
                    {
                        "span_requirement": self.PARENT,
                        "shots": [{"shot_id": 1, "spoken_text": "One drop."}],
                    },
                    {"span_requirement": "   ", "shots": self._spans(1)[0]["shots"]},
                    {
                        "span_requirement": over_long_parent,
                        "shots": self._spans(1)[0]["shots"],
                    },
                ]
            )

        # A span that owns its requirement has no question to answer, so asking
        # would spend a request on nothing.
        generate.assert_not_called()
        self.assertIsNone(result)


FOUNDRY_KEY = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY", "")
FOUNDRY_BASE = "https://amanrai-test-resource.services.ai.azure.com/anthropic"
FOUNDRY_MODEL = "azure_ai/claude-sonnet-4-6"


@unittest.skipUnless(
    RUN_INTEGRATION_TESTS and FOUNDRY_KEY,
    "MPT_RUN_INTEGRATION_TESTS and ANTHROPIC_FOUNDRY_API_KEY not set",
)
class TestLiteLLMLiveIntegration(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["llm_provider"] = "litellm"
        config.app["litellm_model_name"] = FOUNDRY_MODEL
        os.environ["AZURE_AI_API_KEY"] = FOUNDRY_KEY
        os.environ["AZURE_AI_API_BASE"] = FOUNDRY_BASE

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_live_litellm_completion(self):
        result = llm._generate_response("What is 2+2? Reply with just the number.")

        self.assertNotIn("Error:", result)
        self.assertIn("4", result)


if __name__ == "__main__":
    unittest.main()
