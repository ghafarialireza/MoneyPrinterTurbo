import hashlib
import json
import logging
import re
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.schema import (
    CriticalVisualFact,
    MandatoryFactResult,
    ObservedVisualFacts,
    SemanticAdjudication,
    VisualRequirementSpec,
)
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider
from app.utils import utils

_max_retries = 5
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]*:[^/\s?#@]*@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)
VISUAL_REQUIREMENT_SPEC_SCHEMA_VERSION = "visual-requirement-spec-v2"
# Part of the adjudication cache key. A stored decision is only reusable while the
# rules that produced it still hold, so bump this whenever the decision rules in
# adjudicate_visual_candidates change — otherwise a run would keep honoring
# verdicts reached under rules it no longer applies.
SEMANTIC_ADJUDICATION_SCHEMA_VERSION = "semantic-adjudication-v1"
_VISUAL_REQUIREMENT_CACHE_FORMAT_VERSION = 2
_VISUAL_REQUIREMENT_CACHE_LOCKS = tuple(threading.Lock() for _ in range(64))
_SEMANTIC_ADJUDICATION_CACHE_FORMAT_VERSION = 1
_SEMANTIC_ADJUDICATION_CACHE_LOCKS = tuple(threading.Lock() for _ in range(64))
_MAX_REQUIREMENT_TEXT_LENGTH = 500
_MAX_REQUIREMENT_LIST_ITEMS = 12
_MAX_CRITICAL_VISUAL_FACTS = 8
_MAX_STRUCTURED_TEXT_LENGTH = 500
# One decomposition spec is a large object (up to _MAX_CRITICAL_VISUAL_FACTS
# nested fact objects). Asking for every requirement of a timeline in a single
# response produced ~10k tokens of JSON, which providers truncate mid-object and
# we then discard whole. Small batches keep each response parsable, and the
# per-requirement cache makes the extra requests free on the next run.
_VISUAL_REQUIREMENT_BATCH_SIZE = 4
# One repaired narration line is a single short string, so batches can be much
# larger than a decomposition batch. The bound exists because the model has to
# keep every requested line index straight in one answer, not because of size.
_NARRATION_REQUIREMENT_BATCH_SIZE = 25
# Shots of one span must be answered together, because the whole point is that
# they differ from each other. Batching is therefore by span, and this bounds how
# many spans share a request: a failed batch costs only those spans, which keep
# their parent requirement and behave exactly as they did before this stage
# existed.
_SHOT_REQUIREMENT_SPANS_PER_BATCH = 4
# Providers that require an explicit output ceiling used to get 2048 tokens,
# which silently truncated every multi-object structured response.
_DEFAULT_MAX_OUTPUT_TOKENS = 8192
_MAX_STRUCTURED_RESPONSE_ATTEMPTS = 2
_LOGGED_RESPONSE_PREVIEW_LENGTH = 200
_TIMESTAMP_EVIDENCE_RE = re.compile(
    r"(?:\b\d{1,2}:\d{2}(?::\d{2})?\b|\b\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds)\b)",
    re.IGNORECASE,
)
_CRITICAL_FACT_BOILERPLATE_WORDS = {
    "a",
    "an",
    "are",
    "be",
    "directly",
    "is",
    "shown",
    "the",
    "visible",
}

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.
""".strip()


def _normalize_text_response(content, llm_provider: str) -> str:
    # 不同 LLM SDK 在异常或被拦截场景下，可能返回 None、空字符串，
    # 甚至返回非字符串对象。这里统一做兜底校验，避免后续直接调用
    # `.replace()` 时抛出 `NoneType` 之类的属性错误。
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    # MiniMax M3、DeepSeek R1 这类 reasoning 模型可能会把内部推理包在
    # `<think>...</think>` 中返回。视频脚本和关键词只需要最终可朗读文本，
    # 如果不在服务层统一清理，WebUI、字幕和配音都会把思考过程当正文处理。
    content = _THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    # Leading/trailing whitespace was already stripped above. Preserve internal
    # single and double newlines because generated scripts use them for semantic
    # line breaks and paragraph boundaries consumed by the subtitle timeline.
    return content


def _sanitize_error_message(error: object) -> str:
    """
    清理返回给 WebUI/API 的错误信息，避免自定义 base_url 中的凭据泄露。

    一些 OpenAI-compatible SDK 会把请求 URL 原样拼进异常信息。如果用户为了
    代理网关配置了 `https://user:pass@example.com/v1`，直接返回 `str(e)`
    就会把密码暴露给页面、API 调用方或后续日志。这里仅处理错误文案，不改变
    实际请求地址，避免影响正常调用链路。
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    return message


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI 兼容接口在异常场景下，可能返回没有 choices、
    # 或者 choices/message/content 为空的响应对象。
    # 这里统一做结构校验，避免出现 `NoneType is not subscriptable`
    # 这类底层属性访问错误。
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """兼容 dict 和 SDK 响应对象的字段读取。"""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _extract_qwen_generation_text(response) -> str:
    """
    从 DashScope Generation 响应中提取文本。

    Qwen 使用 `messages` 调用时返回的是 chat 结构：
    `output.choices[0].message.content`；旧 completion 形态才会返回
    `output.text`。这里两个路径都兼容，避免 `output.text` 为 None 时
    继续 `.replace()` 触发不可诊断的 AttributeError。
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen returned an empty choices list")
            raise ValueError("[qwen] returned empty choices")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _resolved_max_output_tokens(runtime_app_config) -> int:
    """Output-token ceiling for providers whose SDK needs an explicit limit.

    Only a few adapters (currently Gemini) require the caller to state a maximum;
    the OpenAI-compatible ones default to the model maximum. The previous
    hardcoded 2048 was below the size of a single structured response, so the
    semantic grouping and requirement decomposition calls came back truncated and
    were discarded as unparsable. Operators can still lower or raise the ceiling
    through ``llm_max_output_tokens``.
    """
    try:
        configured = int(runtime_app_config.get("llm_max_output_tokens", 0) or 0)
    except (TypeError, ValueError):
        configured = 0
    return configured if configured > 0 else _DEFAULT_MAX_OUTPUT_TOKENS


def _generate_response(prompt: str, app_config=None) -> str:
    try:
        # WebUI 在视频生成期间允许用户准备下一条文案。调用方可以传入提交瞬间
        # 的配置快照，确保模型请求重试期间不会因为后台任务结束并应用新配置，
        # 而切换到另一个 Provider、Base URL 或模型。
        runtime_app_config = app_config if app_config is not None else config.app
        llm_provider = str(
            runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
        ).lower()
        provider = get_llm_provider(llm_provider)
        if provider is None:
            raise ValueError(f"{llm_provider}: unsupported llm provider")

        logger.info(f"llm provider: {llm_provider}")
        api_key = runtime_app_config.get(provider.config_key("api_key"), "")
        configured_model = runtime_app_config.get(provider.config_key("model_name"), "")
        model_name = provider.resolve_model_name(configured_model)
        if configured_model and model_name != configured_model:
            logger.warning(
                f"{llm_provider} model '{configured_model}' is deprecated, "
                f"fallback to '{model_name}'"
            )
        configured_base_url = runtime_app_config.get(
            provider.config_key("base_url"), ""
        )
        base_url = provider.resolve_base_url(configured_base_url)
        if configured_base_url and configured_base_url.strip().rstrip("/") in {
            url.rstrip("/") for url in provider.deprecated_base_urls
        }:
            logger.warning(
                f"{llm_provider} base URL '{configured_base_url}' is deprecated, "
                f"fallback to '{base_url}'"
            )
        adapter = provider.adapter
        api_version = ""

        # Ollama 的默认地址依赖当前是否运行在容器中，无法作为静态 Registry
        # 值保存；Registry 仍负责模型和必填规则，运行环境差异在这里解析。
        if llm_provider == "ollama":
            api_key = "ollama"
            if not base_url:
                base_url = config.get_default_ollama_base_url()

        if adapter == "azure":
            api_version = runtime_app_config.get(
                provider.config_key("api_version"), "2024-02-15-preview"
            )

        extra_values = {
            field.config_suffix: (
                runtime_app_config.get(provider.config_key(field.config_suffix), "")
                or field.default_value
            )
            for field in provider.extra_fields
        }

        if provider.requires_api_key and not api_key:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if provider.requires_model_name and not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if provider.requires_base_url and not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )

        for field in provider.extra_fields:
            if field.required and not extra_values[field.config_suffix]:
                raise ValueError(
                    f"{llm_provider}: {field.config_suffix} is not set, "
                    "please set it in the config.toml file."
                )

        if adapter == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            dashscope.api_key = api_key
            response = dashscope.Generation.call(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, GenerationResponse):
                    status_code = response.status_code
                    if status_code != 200:
                        raise Exception(
                            f'[{llm_provider}] returned an error response: "{response}"'
                        )

                    return _extract_qwen_generation_text(response)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}"'
                    )
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        if adapter == "gemini":
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(base_url=base_url) if base_url else None
            generation_config = types.GenerateContentConfig(
                temperature=0.5,
                top_p=1,
                top_k=1,
                max_output_tokens=_resolved_max_output_tokens(runtime_app_config),
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            )

            try:
                # 新版 google-genai 通过统一 Client 暴露模型服务。上下文管理器
                # 会在请求结束后关闭底层 HTTP 连接，避免频繁生成时积累连接资源。
                with genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                ) as client:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                generated_text = response.text
            except (AttributeError, IndexError, ValueError) as e:
                logger.warning(f"gemini returned invalid response content: {str(e)}")
                raise ValueError(f"[{llm_provider}] returned invalid response content")

            return _normalize_text_response(generated_text, llm_provider)

        if adapter == "cloudflare_ai_gateway":
            account_id = extra_values["account_id"]
            gateway_id = extra_values["gateway_id"]
            # Cloudflare 当前推荐的 AI Gateway REST API 兼容 OpenAI SDK。
            # Account ID 用于构造统一端点，Gateway ID 通过请求头选择；这里
            # 不再调用 Workers AI 的 /ai/run/{model} 专用接口。
            client = OpenAI(
                api_key=api_key,
                base_url=(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
                ),
                default_headers={"cf-aig-gateway-id": gateway_id},
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "litellm":
            import litellm

            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                )

            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                drop_params=True,
            )

            if not response:
                raise ValueError(f"[{llm_provider}] returned empty response")
            if not getattr(response, "choices", None):
                raise ValueError(f"[{llm_provider}] returned empty response")

            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "azure":
            # Azure OpenAI SDK 使用 `azure_endpoint` 和 `api_version` 生成专用请求地址，
            # 不能继续复用下面普通 OpenAI-compatible 的 `base_url` 初始化逻辑。
            # 这里在 Azure 分支内完成请求并立即返回，避免客户端被后续 fallback
            # 覆盖，导致用户配置的 Azure 凭证通过校验但实际请求没有被使用。
            logger.info(f"requesting azure chat completion, model: {model_name}")
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
            )
            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        if adapter == "modelscope":
            content = ""
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                stream=True,
            )
            if response:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content

                if not content.strip():
                    raise ValueError("Empty content in stream response")

                return _normalize_text_response(content, llm_provider)
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        if response:
            if isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            else:
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                    f"connection and try again."
                )
        else:
            raise Exception(
                f"[{llm_provider}] returned an empty response, please check your network connection and try again."
            )

    except Exception as e:
        return f"Error: {_sanitize_error_message(e)}"


def test_connection() -> tuple[bool, str, float]:
    """
    使用当前 Provider 配置发起一次最小请求，验证实际生成链路是否可用。

    连接测试直接复用 `_generate_response()`，因此会覆盖 API Key、Base URL、
    模型名称和 Provider 专用字段，但不会进入脚本生成的重试逻辑，也不会发送
    用户的视频主题或文案。返回值依次为成功状态、错误信息和请求耗时。
    """
    started_at = perf_counter()
    response = _generate_response(prompt="Reply with exactly: OK")
    elapsed = perf_counter() - started_at

    if not response:
        error_message = "LLM returned an empty response"
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    if response.startswith("Error:"):
        error_message = response.removeprefix("Error:").strip()
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    logger.info(f"llm connection test succeeded, elapsed: {elapsed:.2f}s")
    return True, "", elapsed


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API 层已经用 Pydantic 做长度校验；这里继续兜底，是为了保护
    # WebUI 或内部服务直接调用 generate_script 时不会把超长提示词发送给模型，
    # 避免 token 成本异常和请求失败。
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # WebUI 和 API 都会限制范围；这里兜底处理内部调用，避免异常参数直接扩大
        # LLM 生成成本或生成空结果。
        logger.warning(
            f"script paragraph_number is out of range and will be clamped: {value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # 将“脚本生成规则”和“运行时上下文”分开拼接。这样高级用户即使覆盖默认
    # system prompt，也不会漏掉视频主题、语言、段落数这些每次生成都必须带上的参数。
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    app_config=None,
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    final_script = ""
    logger.info(
        "generating video script: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt=prompt)
            else:
                response = _generate_response(prompt=prompt, app_config=app_config)
            if response:
                final_script = format_response(response)
            else:
                logging.error("gpt returned an empty response")

            # Some upstream providers may return quota errors as plain text.
            if final_script and "当日额度已消耗完" in final_script:
                raise ValueError(final_script)

            if final_script:
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if "Error: " in final_script:
        logger.error(f"failed to generate video script: {final_script}")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding markdown code fence from an LLM response.

    Non-OpenAI providers (Claude, Gemini, …) frequently wrap JSON output in a
    ```json … ``` fence even when asked to return raw JSON. Removing it lets the
    first json.loads() succeed instead of falling through to the regex recovery
    path (and spuriously logging a warning). Mirrors the DOTALL handling already
    used in _parse_social_metadata().
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json_payload(text: str) -> str:
    """Best-effort recovery of the JSON document inside a chatty response.

    Providers sometimes prepend a sentence, append a note, or wrap the payload in
    a fence. ``generate_visual_slot_queries`` already recovered from that with a
    local regex, which is exactly why it kept working while the structured calls
    that lacked recovery failed on the same provider. This centralizes the
    behavior so every structured call benefits.
    """
    stripped = _strip_code_fence(text)
    try:
        json.loads(stripped)
        return stripped
    except (json.JSONDecodeError, ValueError):
        pass
    # The document starts at the first structural character; anything before it is
    # prose. ``raw_decode`` accepts one complete value there and ignores trailing
    # prose, so a truncated response stays unparsable instead of being "recovered"
    # as one of its inner objects, which would parse cleanly and be silently wrong.
    starts = [
        position
        for position in (stripped.find("{"), stripped.find("["))
        if position >= 0
    ]
    if not starts:
        return stripped
    start = min(starts)
    try:
        _, end = json.JSONDecoder().raw_decode(stripped, start)
    except (json.JSONDecodeError, ValueError):
        return stripped
    return stripped[start:end]


def _response_diagnostic(response: str) -> str:
    """Log-safe description of a response we could not parse.

    Logging only the exception class name made these failures undiagnosable: a
    truncated payload, an empty payload and a refusal all reported
    ``JSONDecodeError``. The preview is provider output, never a request, so it
    cannot contain configured credentials.
    """
    text = (response or "").strip()
    preview = re.sub(r"\s+", " ", text[:_LOGGED_RESPONSE_PREVIEW_LENGTH])
    return f"response_length={len(text)}, response_preview={preview!r}"


def _generate_structured_response(
    prompt: str,
    *,
    purpose: str,
    app_config=None,
    attempts: int = _MAX_STRUCTURED_RESPONSE_ATTEMPTS,
) -> Any:
    """Ask the selected provider for JSON, with recovery and a bounded retry.

    Returns the parsed payload, or ``None`` when every attempt failed. Provider
    unavailability is not retried: ``_generate_response`` already exhausts its own
    transport retries and returns a sanitized ``Error: `` string.
    """
    for attempt in range(1, max(1, attempts) + 1):
        if app_config is None:
            response = _generate_response(prompt)
        else:
            response = _generate_response(prompt, app_config=app_config)
        if response.startswith("Error: "):
            logger.warning(
                f"{purpose} provider is unavailable: "
                f"{response[:_LOGGED_RESPONSE_PREVIEW_LENGTH]}"
            )
            return None
        try:
            return json.loads(_extract_json_payload(response))
        except Exception as exc:
            logger.warning(
                f"{purpose} returned unusable structured data "
                f"(attempt {attempt}/{max(1, attempts)}): "
                f"error={type(exc).__name__}: {exc}, {_response_diagnostic(response)}"
            )
    return None


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    app_config=None,
) -> List[str]:
    if match_script_order:
        goal = (
            f"Generate exactly {amount} chronological stock-video search terms, "
            "one for each consecutive visual slot in the video script."
        )
        term_shape_rule = (
            "2. each search term must be a concrete, camera-visible shot of 2-6 "
            "English words; include the main subject and avoid abstract concepts, "
            "emotions, metaphors, camera transitions, or narration summaries."
        )
        ordering_rule = (
            "6. keep the terms in the same order as the script narration; "
            "earlier terms must describe earlier visual moments.\n"
            "7. every item must describe only its own consecutive script segment; "
            "do not repeat an opening visual after moving to a later topic.\n"
            f"8. return exactly {amount} items, including the final script moment."
        )
        # 有序关键词模式下，示例数量要和 amount 保持一致，避免模型被固定
        # 的 4 个示例误导，导致长文案只返回少量关键词，影响素材覆盖度。
        example_terms = [
            "opening visual topic",
            *[f"script visual topic {index}" for index in range(2, max(amount, 1))],
            "final visual topic",
        ]
        output_example = json.dumps(example_terms[:amount], ensure_ascii=False)
    else:
        goal = (
            f"Generate {amount} search terms for stock videos, depending on the "
            "subject of a video."
        )
        ordering_rule = ""
        term_shape_rule = (
            "2. each search term should consist of 1-3 words, always add the main "
            "subject of the video."
        )
        output_example = (
            '["search term 1", "search term 2", "search term 3",'
            '"search term 4", "search term 5"]'
        )

    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
{goal}

## Constrains:
1. the search terms are to be returned as a json-array of strings.
{term_shape_rule}
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.
{ordering_rule}

## Output Example:
{output_example}

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}, match_script_order: {match_script_order}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        # Never let a partially parsed value from an earlier attempt satisfy a
        # later retry whose response could not be decoded.
        search_terms = []
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                # generate_terms 的公开返回类型是 List[str]。如果把 Provider 的
                # 错误文案原样返回，下游只做空值判断时会把非空字符串误认为成功，
                # 素材下载循环还会按字符遍历错误文案，产生无意义的外部请求。
                # 这里统一返回空列表，让任务编排层在真实故障位置立即结束任务。
                logger.error(f"failed to generate video terms: {response}")
                return []
            search_terms = json.loads(_strip_code_fence(response))
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.error("response is not a list of strings.")
                continue
            search_terms = [term.strip() for term in search_terms if term.strip()]
            if match_script_order and len(search_terms) != amount:
                logger.warning(
                    "ordered video terms count mismatch: "
                    f"expected={amount}, received={len(search_terms)}"
                )
                search_terms = []
                continue

        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response, re.DOTALL)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as e:
                        # 这里保留重试流程，但必须记录 LLM 返回的非标准 JSON，
                        # 否则后续排查搜索词为空时无法定位
                        # 是模型格式问题还是解析逻辑问题。
                        logger.warning(f"failed to generate video terms: {str(e)}")

        # The regex recovery path above must obey exactly the same type/count
        # contract as the primary JSON path. Previously it could accept the
        # wrong number of ordered terms and bypass the timeline guarantee.
        if search_terms:
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.warning("recovered video terms are not a list of strings")
                search_terms = []
            else:
                search_terms = [term.strip() for term in search_terms if term.strip()]
                if match_script_order and len(search_terms) != amount:
                    logger.warning(
                        "recovered ordered video terms count mismatch: "
                        f"expected={amount}, received={len(search_terms)}"
                    )
                    search_terms = []

        if search_terms and len(search_terms) > 0:
            break
        if i < _max_retries:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


def generate_visual_slot_queries(
    video_subject: str,
    visual_slots: list[dict],
    queries_per_slot: int = 1,
    app_config=None,
) -> dict[int, list[str]]:
    """Generate indexed stock-video queries from each slot's visual requirement."""
    if not visual_slots:
        return {}

    queries_per_slot = max(1, int(queries_per_slot))
    slot_payload = [
        {
            "slot_index": int(slot["slot_index"]),
            "start_time": float(slot["start_time"]),
            "end_time": float(slot["end_time"]),
            "visual_requirement": str(slot["visual_requirement"]).strip(),
        }
        for slot in visual_slots
    ]
    expected_indexes = {slot["slot_index"] for slot in slot_payload}
    output_example = [
        {
            "slot_index": slot["slot_index"],
            "queries": [
                f"camera visible query {query_index + 1}"
                for query_index in range(queries_per_slot)
            ],
        }
        for slot in slot_payload
    ]
    # With more than one query per slot the extra phrasings are ordered fallbacks
    # for the same shot, tried on the same provider before another catalog is
    # asked. Near-duplicates would waste a search and an analysis round, so the
    # ordering and the "same scene" constraint are stated explicitly.
    fallback_rule = (
        ""
        if queries_per_slot == 1
        else (
            "\n8. Order each slot's queries as fallbacks for the same shot: the most "
            "literal phrasing first, then progressively simpler or more generic "
            "phrasings of the same visible subject and action. Fallbacks must differ "
            "in wording, and none of them may describe a different scene."
        )
    )
    prompt = f"""
# Role: Visual Slot Stock-Video Search Query Generator

## Goal
Generate Pexels stock-video search queries for the indexed visual slots below.

## Rules
1. Return a JSON array only, with exactly one object for every supplied slot_index.
2. Each object must contain slot_index and exactly {queries_per_slot} queries.
3. Derive each query only from that slot's visual_requirement. Never move an action,
   subject, or scene from one slot to another.
4. Preserve the narration's main visible subject and action.
5. Every query must describe a concrete, camera-visible shot suitable for stock footage.
6. Avoid abstract concepts, emotions, metaphors, narration summaries, camera
   transitions, and editorial instructions.
7. Use concise English queries of 2-6 words.{fallback_rule}

## Video Subject
{video_subject}

## Visual Slots
{json.dumps(slot_payload, ensure_ascii=False)}

## Output Example
{json.dumps(output_example, ensure_ascii=False)}
""".strip()

    def parse_slot_queries(raw_response: str) -> dict[int, list[str]]:
        parsed = json.loads(_strip_code_fence(raw_response))
        if not isinstance(parsed, list):
            raise ValueError("visual slot query response must be a JSON array")

        queries_by_slot: dict[int, list[str]] = {}
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("each visual slot query item must be an object")
            slot_index = item.get("slot_index")
            if isinstance(slot_index, bool):
                raise ValueError("slot_index must be an integer")
            try:
                slot_index = int(slot_index)
            except (TypeError, ValueError) as exc:
                raise ValueError("slot_index must be an integer") from exc
            if slot_index not in expected_indexes or slot_index in queries_by_slot:
                raise ValueError(f"unexpected or duplicate slot_index: {slot_index}")

            queries = item.get("queries")
            if not isinstance(queries, list) or not all(
                isinstance(query, str) for query in queries
            ):
                raise ValueError(f"slot {slot_index} queries must be strings")
            queries = [query.strip() for query in queries if query.strip()]
            # Alternate phrasings are a bonus, not a contract: material selection
            # tries them in order and stops at the first winner. Rejecting a slot
            # because the provider returned two phrasings instead of three would
            # throw away the one phrasing that works, so accept what arrived,
            # collapse repeats, and keep at most what was asked for.
            deduplicated: list[str] = []
            seen_queries: set[str] = set()
            for query in queries:
                normalized_query = query.casefold()
                if normalized_query in seen_queries:
                    continue
                seen_queries.add(normalized_query)
                deduplicated.append(query)
            queries = deduplicated[:queries_per_slot]
            if not queries:
                raise ValueError(f"slot {slot_index} must contain at least one query")
            if not all(re.search(r"[A-Za-z]", query) for query in queries):
                raise ValueError(f"slot {slot_index} queries must be English")
            queries_by_slot[slot_index] = queries

        if set(queries_by_slot) != expected_indexes:
            missing = sorted(expected_indexes - set(queries_by_slot))
            raise ValueError(f"visual slot query response is missing slots: {missing}")
        if queries_per_slot > 1:
            short_slots = sorted(
                slot_index
                for slot_index, slot_queries in queries_by_slot.items()
                if len(slot_queries) < queries_per_slot
            )
            if short_slots:
                logger.warning(
                    "visual slot queries returned fewer phrasings than requested: "
                    f"requested={queries_per_slot}, slots={short_slots}"
                )
        return queries_by_slot

    response = ""
    for attempt in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                logger.error(f"failed to generate visual slot queries: {response}")
                return {}
            queries_by_slot = parse_slot_queries(response)
            logger.success(
                f"completed visual slot queries: slots={len(queries_by_slot)}"
            )
            return queries_by_slot
        except Exception as exc:
            logger.warning(f"failed to generate visual slot queries: {str(exc)}")
            if response:
                match = re.search(r"\[.*]", response, re.DOTALL)
                if match:
                    try:
                        queries_by_slot = parse_slot_queries(match.group())
                        logger.success(
                            "completed visual slot queries after response recovery: "
                            f"slots={len(queries_by_slot)}"
                        )
                        return queries_by_slot
                    except Exception as recovery_exc:
                        logger.warning(
                            "failed to recover visual slot queries: "
                            f"{str(recovery_exc)}"
                        )
            if attempt < _max_retries - 1:
                logger.warning(
                    "failed to generate visual slot queries, trying again... "
                    f"{attempt + 1}"
                )

    return {}


def generate_semantic_visual_span_specs(
    narration_text: str,
    timed_units: list[dict],
    app_config=None,
) -> object | None:
    """Request one compact semantic grouping from the selected LLM provider.

    The response remains untrusted here. The timeline service performs the
    authoritative range/schema/coverage validation before constructing spans.
    """
    if not timed_units:
        return None

    unit_lines = []
    for zero_based_id, unit in enumerate(timed_units):
        unit_lines.append(
            "|".join(
                (
                    str(zero_based_id),
                    str(unit.get("source_narration_slot_index") or "-"),
                    json.dumps(str(unit.get("text") or ""), ensure_ascii=False),
                )
            )
        )

    prompt = f"""
# Role: Semantic Visual Span Grouper

## Goal
Identify only the points where the camera-visible meaning of this narration
genuinely changes. Prefer the SMALLEST number of spans needed for distinct
visible concepts.

## Authoritative Units
Each line is: zero_based_unit_id|narration_slot_hint|spoken_unit_text
Unit IDs and their order are authoritative. A range uses start_unit inclusive
and end_unit_exclusive. Never rewrite, omit, duplicate, or reorder source units.

{chr(10).join(unit_lines)}

## Read-only Narration Context
This preserves punctuation and paragraph hints. It is context only and must not
be copied into the response.
<narration>
{narration_text}
</narration>

## Semantic Rules
1. Start a new span only for a meaningful visible change in subject, physical
   action, object, location, environment, state, process, or event.
2. Do not split merely for punctuation, sentence/paragraph boundaries,
   adjectives, clauses, or conjunctions.
3. Keep a continuous visible process together even when described in multiple
   sentences.
4. Attach abstract or non-visible wording to the nearest valid visible concept;
   never create a nonsense standalone visual for an abstract phrase.
5. A multi-word unit is indivisible. Boundaries may occur only between unit IDs.
6. visual_requirement must be concise, concrete, camera-visible, and must
   preserve the real subject/action without adding facts.

## Strict Output
Return one JSON array only. Every object must contain exactly:
- start_unit: integer
- end_unit_exclusive: integer
- visual_requirement: non-empty string

Do not return timestamps, durations, spoken_text, search queries, explanations,
provider data, or any other fields.

Example shape only:
[
  {{"start_unit": 0, "end_unit_exclusive": 8,
    "visual_requirement": "Ripe coffee cherries growing on coffee plants"}}
]
""".strip()

    try:
        parsed = _generate_structured_response(
            prompt,
            purpose="semantic visual grouping",
            app_config=app_config,
        )
        if parsed is None:
            return None
        return parsed
    except Exception as exc:
        logger.warning(
            "semantic visual grouping returned unusable structured data: "
            f"error={type(exc).__name__}: {exc}"
        )
        return None


def _validated_narration_visual_requirements(
    parsed: object,
    requested_indexes: set[int],
) -> dict[int, str]:
    """Keep the well-formed answers and drop the rest.

    A requested line is absent from the result when the provider left it out,
    answered it with an empty string, or answered it with something malformed.
    The caller treats all three the same way -- that line has no visual
    requirement of its own -- so one bad object never discards the lines that
    came back correctly.
    """
    if not isinstance(parsed, list):
        raise ValueError("narration visual requirements must be a JSON array")
    if len(parsed) > 2 * max(1, len(requested_indexes)):
        raise ValueError("narration visual requirements returned too many objects")

    requirements: dict[int, str] = {}
    rejected = 0
    for item in parsed:
        if not isinstance(item, dict) or set(item) - {"index", "visual_requirement"}:
            rejected += 1
            continue
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            rejected += 1
            continue
        if index not in requested_indexes or index in requirements:
            rejected += 1
            continue
        raw_requirement = item.get("visual_requirement")
        if isinstance(raw_requirement, str) and not raw_requirement.strip():
            # The documented way to say "this line has no visible content of its
            # own". It is an answer, not a provider failure.
            continue
        try:
            requirements[index] = _bounded_structured_text(
                raw_requirement,
                "visual_requirement",
            )
        except ValueError:
            rejected += 1
    if rejected:
        logger.warning(
            "narration visual requirement objects were unusable: "
            f"rejected={rejected}, accepted={len(requirements)}"
        )
    return requirements


def generate_narration_visual_requirements(
    narration_text: str,
    narration_lines: list[dict],
    app_config=None,
) -> dict[int, str] | None:
    """Rewrite spoken narration lines as requirements a camera could record.

    Used when semantic grouping failed and the timeline fell back to one span
    per narration line: the spoken line is then standing in for a visual
    requirement, and a line such as "Patient." cannot be filled by any stock
    clip. Returns the accepted requirements keyed by the caller's line index.
    A line that has no visible content of its own is deliberately absent rather
    than given an invented scene, so the caller can attach it to a neighbour.

    Returns ``None`` only when no batch produced a parseable payload at all,
    which the caller must treat as "the repair is unavailable" and not as "the
    narration has nothing visible in it".
    """
    requested = [
        {"index": line.get("index"), "spoken_text": str(line.get("spoken_text") or "")}
        for line in narration_lines
        if isinstance(line.get("index"), int)
        and not isinstance(line.get("index"), bool)
        and str(line.get("spoken_text") or "").strip()
    ]
    if not requested:
        return None

    requirements: dict[int, str] = {}
    parsed_any_batch = False
    for batch_start in range(0, len(requested), _NARRATION_REQUIREMENT_BATCH_SIZE):
        batch = requested[
            batch_start : batch_start + _NARRATION_REQUIREMENT_BATCH_SIZE
        ]
        line_block = chr(10).join(
            "|".join(
                (
                    str(line["index"]),
                    json.dumps(line["spoken_text"], ensure_ascii=False),
                )
            )
            for line in batch
        )
        prompt = f"""
# Role: Narration Line to Filmable Visual Requirement

## Goal
For each narration line below, describe what a real stock video clip would have
to show while that line is spoken. The result is used to search stock catalogs
and to verify a candidate clip against it, so it must describe visible things
and never the meaning of the words.

## Authoritative Lines
Each line is: line_index|spoken_text
Line indexes are authoritative. Answer every line exactly once and never
rewrite, merge, split, reorder, or invent lines.

{line_block}

## Read-only Narration Context
Use this only to resolve what a short or dependent line refers to. It must not
be copied into the response.
<narration>
{narration_text}
</narration>

## Rules
1. visual_requirement must be concrete and camera-visible: a subject, action,
   object, location, environment, or state that a camera can record.
2. Add no fact the narration does not support. Resolve pronouns and elliptical
   lines from the context above instead of inventing a new subject.
3. Never request on-screen text, captions, logos, brands, or named real people.
4. A line with no visible content of its own -- a judgement, a feeling, an
   abstract statement, a connective phrase -- must return an empty string.
   Never invent a nonsense visual for such a line.
5. Describe one visible situation in under 20 words.

## Strict Output
Return one JSON array only. Every object must contain exactly:
- index: integer, one of the line indexes above
- visual_requirement: string, empty only for a line with no visible content

Do not return timestamps, durations, spoken text, search queries, explanations,
provider data, or any other fields.

Example shape only:
[
  {{"index": 1,
    "visual_requirement": "A single water drop falling into still water"}},
  {{"index": 2, "visual_requirement": ""}}
]
""".strip()

        parsed = _generate_structured_response(
            prompt,
            purpose="narration visual requirement repair",
            app_config=app_config,
        )
        if parsed is None:
            continue
        parsed_any_batch = True
        try:
            requirements.update(
                _validated_narration_visual_requirements(
                    parsed,
                    {line["index"] for line in batch},
                )
            )
        except ValueError as exc:
            logger.warning(
                "narration visual requirement repair returned an unusable batch: "
                f"reason={exc}"
            )
    if not parsed_any_batch:
        return None
    return requirements


def _validated_shot_visual_requirements(
    parsed: object,
    requested_shot_ids: set[int],
) -> dict[int, str]:
    """Keep the well-formed shot answers and drop the rest.

    A requested shot is absent from the result when the provider left it out,
    answered it with an empty string, or answered it with something malformed.
    The caller treats all three the same way -- that shot keeps its parent
    span's requirement -- so one bad object never discards the shots that came
    back correctly.
    """
    if not isinstance(parsed, list):
        raise ValueError("shot visual requirements must be a JSON array")
    if len(parsed) > 2 * max(1, len(requested_shot_ids)):
        raise ValueError("shot visual requirements returned too many objects")

    requirements: dict[int, str] = {}
    rejected = 0
    for item in parsed:
        if not isinstance(item, dict) or set(item) - {
            "shot_id",
            "visual_requirement",
        }:
            rejected += 1
            continue
        shot_id = item.get("shot_id")
        if isinstance(shot_id, bool) or not isinstance(shot_id, int):
            rejected += 1
            continue
        if shot_id not in requested_shot_ids or shot_id in requirements:
            rejected += 1
            continue
        raw_requirement = item.get("visual_requirement")
        if isinstance(raw_requirement, str) and not raw_requirement.strip():
            # The documented way to say "this shot cannot be narrowed any
            # further". It is an answer, not a provider failure.
            continue
        try:
            requirements[shot_id] = _bounded_structured_text(
                raw_requirement,
                "visual_requirement",
            )
        except ValueError:
            rejected += 1
    if rejected:
        logger.warning(
            "shot visual requirement objects were unusable: "
            f"rejected={rejected}, accepted={len(requirements)}"
        )
    return requirements


def _requestable_shot_requirement_spans(spans: list[dict]) -> list[dict]:
    """Spans worth asking about: a usable parent plus two or more shots."""
    requestable: list[dict] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        parent = " ".join(str(span.get("span_requirement") or "").split())
        if not parent or len(parent) > _MAX_STRUCTURED_TEXT_LENGTH:
            continue
        shots: list[dict] = []
        seen_shot_ids: set[int] = set()
        for shot in span.get("shots") or []:
            if not isinstance(shot, dict):
                continue
            shot_id = shot.get("shot_id")
            if isinstance(shot_id, bool) or not isinstance(shot_id, int):
                continue
            spoken_text = " ".join(str(shot.get("spoken_text") or "").split())
            if not spoken_text or shot_id in seen_shot_ids:
                continue
            if len(spoken_text) > _MAX_STRUCTURED_TEXT_LENGTH:
                continue
            seen_shot_ids.add(shot_id)
            shots.append({"shot_id": shot_id, "spoken_text": spoken_text})
        # A span with one shot already owns its requirement, so asking about it
        # would spend a request on a question that has no answer.
        if len(shots) < 2:
            continue
        requestable.append({"span_requirement": parent, "shots": shots})
    return requestable


def generate_shot_visual_requirements(
    spans: list[dict],
    app_config=None,
) -> dict[int, str] | None:
    """Give each shot of a split span one visible moment of its own.

    A long semantic span is cut into several consecutive shots, and every shot
    used to inherit the span's entire requirement. Three shots then searched for
    the same multi-event scene, so provider de-duplication had to buy three
    near-duplicate clips, and each one was verified against facts that belong to
    the other two. This narrows the parent requirement to the moment each shot's
    own words describe.

    Returns the accepted requirements keyed by the caller's shot id. A shot that
    cannot be narrowed is deliberately absent rather than given an invented
    scene, so the caller simply keeps the parent requirement for it.

    Returns ``None`` only when no batch produced a parseable payload at all,
    which the caller must treat as "this stage is unavailable" and never as "no
    shot could be narrowed".
    """
    requestable = _requestable_shot_requirement_spans(spans)
    if not requestable:
        return None

    requirements: dict[int, str] = {}
    parsed_any_batch = False
    for batch_start in range(
        0,
        len(requestable),
        _SHOT_REQUIREMENT_SPANS_PER_BATCH,
    ):
        batch = requestable[
            batch_start : batch_start + _SHOT_REQUIREMENT_SPANS_PER_BATCH
        ]
        parent_block = chr(10).join(
            "|".join(
                (
                    str(span_position),
                    json.dumps(span["span_requirement"], ensure_ascii=False),
                )
            )
            for span_position, span in enumerate(batch, start=1)
        )
        shot_block = chr(10).join(
            "|".join(
                (
                    str(span_position),
                    str(shot["shot_id"]),
                    json.dumps(shot["spoken_text"], ensure_ascii=False),
                )
            )
            for span_position, span in enumerate(batch, start=1)
            for shot in span["shots"]
        )
        prompt = f"""
# Role: Split One Visual Requirement Across Its Shots

## Goal
A long spoken passage was cut into consecutive shots. Each shot needs its own
description of what a real stock video clip would have to show while that shot's
words are spoken. The result is used to search stock catalogs and to verify a
candidate clip against it, so it must describe visible things only.

## Parent Requirements
Each line is: span_id|parent_requirement
This is the whole passage's visual requirement, and it is the boundary for that
span's shots: a shot requirement must be a narrower part of it.

{parent_block}

## Shots
Each line is: span_id|shot_id|spoken_text
Shot ids are authoritative. Answer every shot exactly once and never rewrite,
merge, split, reorder, or invent shots. Shots of one span are in spoken order.

{shot_block}

## Rules
1. visual_requirement must be ONE camera-visible moment: a single subject and
   action that a camera can record in one continuous shot.
2. Stay inside the parent requirement of that shot's span. Narrow it, and never
   introduce a subject the parent does not contain.
3. Prefer the part of the parent that this shot's own spoken_text describes.
   Shots are sentence fragments, so read the parent to resolve them.
4. Shots of the same span must describe different moments. Never return the
   same requirement for two shots of one span.
5. Never request on-screen text, captions, logos, brands, or named real people.
6. A shot whose words add no visible moment of their own must return an empty
   string. Never invent a nonsense visual to fill it.
7. Describe one visible situation in under 20 words.

## Strict Output
Return one JSON array only. Every object must contain exactly:
- shot_id: integer, one of the shot ids above
- visual_requirement: string, empty only for a shot that cannot be narrowed

Do not return span ids, timestamps, durations, spoken text, search queries,
explanations, provider data, or any other fields.

Example shape only:
[
  {{"shot_id": 1,
    "visual_requirement": "A single water drop falling onto bare stone"}},
  {{"shot_id": 2,
    "visual_requirement": "Water running over a grooved rock face"}},
  {{"shot_id": 3, "visual_requirement": ""}}
]
""".strip()

        parsed = _generate_structured_response(
            prompt,
            purpose="shot visual requirement split",
            app_config=app_config,
        )
        if parsed is None:
            continue
        parsed_any_batch = True
        try:
            requirements.update(
                _validated_shot_visual_requirements(
                    parsed,
                    {
                        shot["shot_id"]
                        for span in batch
                        for shot in span["shots"]
                    },
                )
            )
        except ValueError as exc:
            logger.warning(
                "shot visual requirement split returned an unusable batch: "
                f"reason={exc}"
            )
    if not parsed_any_batch:
        return None
    return requirements


def normalize_visual_requirement(value: str) -> str:
    """Canonical text identity shared by requirement batching and cache lookups."""
    return " ".join(str(value or "").split()).strip().casefold()


def _semantic_words(value: str) -> str:
    return " ".join(
        re.sub(r"[^\w]+", " ", str(value or "").casefold(), flags=re.UNICODE).split()
    )


def _selected_llm_identity(app_config=None) -> tuple[str, str]:
    runtime_app_config = app_config if app_config is not None else config.app
    provider_id = str(
        runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
    ).lower()
    provider = get_llm_provider(provider_id)
    if provider is None:
        return provider_id, ""
    configured_model = runtime_app_config.get(provider.config_key("model_name"), "")
    return provider_id, provider.resolve_model_name(configured_model)


def _bounded_structured_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    text = " ".join(value.split()).strip()
    if not text or len(text) > _MAX_STRUCTURED_TEXT_LENGTH:
        raise ValueError(f"{field_name} is empty or too long")
    return text


def _bounded_structured_string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > _MAX_REQUIREMENT_LIST_ITEMS:
        raise ValueError(f"{field_name} must be a bounded string array")
    return [
        _bounded_structured_text(item, f"{field_name} item")
        for item in value
    ]


def visual_requirement_spec_to_dict(spec: VisualRequirementSpec) -> dict[str, Any]:
    """Return the validated, credential-free representation used by other services."""
    return asdict(spec)


def visual_requirement_spec_digest(spec: VisualRequirementSpec) -> str:
    payload = json.dumps(
        visual_requirement_spec_to_dict(spec),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _visual_requirement_spec_cache_dir() -> Path:
    return Path(utils.storage_dir("cache_visual_requirement_specs", create=True))


def _visual_requirement_spec_cache_digest(
    normalized_requirement: str,
    provider_id: str,
    model_name: str,
) -> str:
    payload = json.dumps(
        {
            "normalized_requirement_hash": hashlib.sha256(
                normalized_requirement.encode("utf-8")
            ).hexdigest(),
            "provider": provider_id,
            "model": model_name,
            "schema_version": VISUAL_REQUIREMENT_SPEC_SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _visual_requirement_spec_cache_path(
    normalized_requirement: str,
    provider_id: str,
    model_name: str,
) -> Path:
    digest = _visual_requirement_spec_cache_digest(
        normalized_requirement,
        provider_id,
        model_name,
    )
    return _visual_requirement_spec_cache_dir() / f"{digest}.json"


def _visual_requirement_spec_cache_lock(
    normalized_requirement: str,
    provider_id: str,
    model_name: str,
) -> threading.Lock:
    digest = _visual_requirement_spec_cache_digest(
        normalized_requirement,
        provider_id,
        model_name,
    )
    return _VISUAL_REQUIREMENT_CACHE_LOCKS[
        int(digest[:8], 16) % len(_VISUAL_REQUIREMENT_CACHE_LOCKS)
    ]


_VISUAL_REQUIREMENT_SPEC_ITEM_FIELDS = {
    "requirement_id",
    "original_requirement",
    "subjects",
    "primary_action",
    "objects",
    "required_relations",
    "required_context",
    "required_visible_state",
    "optional_attributes",
    "critical_visual_facts",
    "ambiguity_notes",
}
_CRITICAL_VISUAL_FACT_FIELDS = {
    "id",
    "fact",
    "mandatory",
    "direct_evidence_needed",
    "evidence_description",
    "basis_type",
    "basis_quote",
}


def _validate_visual_requirement_spec_item(
    item: object,
    *,
    requirement_id: int,
    original_requirement: str,
    provider_id: str,
    model_name: str,
) -> VisualRequirementSpec:
    if not isinstance(item, dict) or set(item) != _VISUAL_REQUIREMENT_SPEC_ITEM_FIELDS:
        raise ValueError("visual requirement spec fields are invalid")
    if isinstance(item.get("requirement_id"), bool) or item.get(
        "requirement_id"
    ) != requirement_id:
        raise ValueError("visual requirement spec id is invalid")

    original = _bounded_structured_text(
        item.get("original_requirement"), "original_requirement"
    )
    if normalize_visual_requirement(original) != normalize_visual_requirement(
        original_requirement
    ):
        raise ValueError("visual requirement spec rewrote the source requirement")

    primary_action_value = item.get("primary_action")
    if primary_action_value is None:
        primary_action = None
    else:
        primary_action = _bounded_structured_text(
            primary_action_value, "primary_action"
        )
        action_words = _semantic_words(primary_action)
        if not action_words or action_words not in _semantic_words(original):
            raise ValueError("primary_action is not grounded in the source requirement")

    raw_facts = item.get("critical_visual_facts")
    if (
        not isinstance(raw_facts, list)
        or not raw_facts
        or len(raw_facts) > _MAX_CRITICAL_VISUAL_FACTS
    ):
        raise ValueError("critical_visual_facts must be a bounded non-empty array")

    critical_facts: list[CriticalVisualFact] = []
    fact_ids: set[str] = set()
    source_words = _semantic_words(original)
    source_word_set = set(source_words.split())
    for fact_index, raw_fact in enumerate(raw_facts, start=1):
        if not isinstance(raw_fact, dict) or set(raw_fact) != _CRITICAL_VISUAL_FACT_FIELDS:
            raise ValueError("critical visual fact fields are invalid")
        fact_id = _bounded_structured_text(raw_fact.get("id"), "critical fact id")
        if fact_id != f"f{fact_index}" or fact_id in fact_ids:
            raise ValueError("critical visual fact ids must be unique and sequential")
        fact_ids.add(fact_id)
        mandatory = raw_fact.get("mandatory")
        direct_evidence_needed = raw_fact.get("direct_evidence_needed")
        if not isinstance(mandatory, bool) or not isinstance(
            direct_evidence_needed, bool
        ):
            raise ValueError("critical fact flags must be booleans")
        basis_type = raw_fact.get("basis_type")
        if basis_type not in {"explicit", "logically_necessary"}:
            raise ValueError("critical fact basis_type is invalid")
        basis_quote = _bounded_structured_text(
            raw_fact.get("basis_quote"), "critical fact basis_quote"
        )
        quote_words = _semantic_words(basis_quote)
        if not quote_words or quote_words not in source_words:
            raise ValueError("critical fact basis_quote is not from the requirement")
        if basis_type == "logically_necessary":
            if primary_action is None or _semantic_words(primary_action) not in quote_words:
                raise ValueError(
                    "logically necessary facts must cite the requested action phrase"
                )
        fact_text = _bounded_structured_text(raw_fact.get("fact"), "critical fact")
        unsupported_fact_words = set(_semantic_words(fact_text).split()) - (
            source_word_set | _CRITICAL_FACT_BOILERPLATE_WORDS
        )
        if mandatory and basis_type == "explicit" and unsupported_fact_words:
            raise ValueError(
                "mandatory critical facts must use only source-grounded wording"
            )
        if basis_type == "logically_necessary" and not direct_evidence_needed:
            raise ValueError(
                "logically necessary action facts require direct visual evidence"
            )
        critical_facts.append(
            CriticalVisualFact(
                id=fact_id,
                fact=fact_text,
                mandatory=mandatory,
                direct_evidence_needed=direct_evidence_needed,
                evidence_description=_bounded_structured_text(
                    raw_fact.get("evidence_description"),
                    "critical fact evidence_description",
                ),
                basis_type=basis_type,
                basis_quote=basis_quote,
            )
        )

    if not any(fact.mandatory for fact in critical_facts):
        raise ValueError("at least one critical visual fact must be mandatory")
    if primary_action is not None and not any(
        fact.mandatory
        and fact.direct_evidence_needed
        and fact.basis_type == "logically_necessary"
        for fact in critical_facts
    ):
        raise ValueError(
            "an action requirement needs a defining logically necessary evidence fact"
        )

    subjects = _bounded_structured_string_list(item.get("subjects"), "subjects")
    objects = _bounded_structured_string_list(item.get("objects"), "objects")
    required_relations = _bounded_structured_string_list(
        item.get("required_relations"), "required_relations"
    )
    required_context = _bounded_structured_string_list(
        item.get("required_context"), "required_context"
    )
    required_visible_state = _bounded_structured_string_list(
        item.get("required_visible_state"), "required_visible_state"
    )
    for field_name, values in (
        ("subjects", subjects),
        ("objects", objects),
        ("required_relations", required_relations),
        ("required_context", required_context),
        ("required_visible_state", required_visible_state),
    ):
        for value in values:
            if not set(_semantic_words(value).split()).issubset(source_word_set):
                raise ValueError(f"{field_name} contains unsupported source details")

    return VisualRequirementSpec(
        schema_version=VISUAL_REQUIREMENT_SPEC_SCHEMA_VERSION,
        generator_provider=provider_id,
        generator_model=model_name,
        original_requirement=original,
        subjects=subjects,
        primary_action=primary_action,
        objects=objects,
        required_relations=required_relations,
        required_context=required_context,
        required_visible_state=required_visible_state,
        optional_attributes=_bounded_structured_string_list(
            item.get("optional_attributes"), "optional_attributes"
        ),
        critical_visual_facts=critical_facts,
        ambiguity_notes=_bounded_structured_string_list(
            item.get("ambiguity_notes"), "ambiguity_notes"
        ),
    )


def _load_visual_requirement_spec_cache(
    original_requirement: str,
    provider_id: str,
    model_name: str,
) -> VisualRequirementSpec | None:
    normalized = normalize_visual_requirement(original_requirement)
    try:
        payload = json.loads(
            _visual_requirement_spec_cache_path(
                normalized, provider_id, model_name
            ).read_text(encoding="utf-8")
        )
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _VISUAL_REQUIREMENT_CACHE_FORMAT_VERSION
            or payload.get("schema_version")
            != VISUAL_REQUIREMENT_SPEC_SCHEMA_VERSION
            or payload.get("provider") != provider_id
            or payload.get("model") != model_name
            or not isinstance(payload.get("spec"), dict)
        ):
            return None
        raw_spec = dict(payload["spec"])
        raw_spec["requirement_id"] = 0
        for field in ("schema_version", "generator_provider", "generator_model"):
            raw_spec.pop(field, None)
        return _validate_visual_requirement_spec_item(
            raw_spec,
            requirement_id=0,
            original_requirement=original_requirement,
            provider_id=provider_id,
            model_name=model_name,
        )
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning(
            "failed to read visual requirement spec cache: "
            f"error={type(exc).__name__}"
        )
        return None


def _save_visual_requirement_spec_cache(spec: VisualRequirementSpec) -> None:
    normalized = normalize_visual_requirement(spec.original_requirement)
    cache_path = _visual_requirement_spec_cache_path(
        normalized,
        spec.generator_provider,
        spec.generator_model,
    )
    temp_path: Path | None = None
    try:
        payload = {
            "version": _VISUAL_REQUIREMENT_CACHE_FORMAT_VERSION,
            "schema_version": VISUAL_REQUIREMENT_SPEC_SCHEMA_VERSION,
            "provider": spec.generator_provider,
            "model": spec.generator_model,
            "spec": visual_requirement_spec_to_dict(spec),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, ensure_ascii=False, separators=(",", ":"))
            temp_file.flush()
        temp_path.replace(cache_path)
        temp_path = None
    except Exception as exc:
        logger.warning(
            "failed to write visual requirement spec cache: "
            f"error={type(exc).__name__}"
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _validated_requirement_specs(
    parsed: Any,
    *,
    chunk: list[str],
    provider_id: str,
    model_name: str,
) -> list[VisualRequirementSpec]:
    """Validate one decomposition batch and return specs ordered like ``chunk``.

    Raises on any structural problem so the caller can drop this batch alone.
    Requirement ids are batch-local, which is what keeps a malformed batch from
    invalidating the batches that parsed cleanly.
    """
    if not isinstance(parsed, dict) or set(parsed) != {"specs"}:
        raise ValueError("visual requirement decomposition root is invalid")
    raw_specs = parsed.get("specs")
    if not isinstance(raw_specs, list) or len(raw_specs) != len(chunk):
        raise ValueError("visual requirement decomposition count is invalid")
    by_id: dict[int, dict[str, Any]] = {}
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("visual requirement spec must be an object")
        raw_id = raw_spec.get("requirement_id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise ValueError("visual requirement spec id must be an integer")
        if raw_id in by_id or not 0 <= raw_id < len(chunk):
            raise ValueError("visual requirement spec id is duplicate or unknown")
        by_id[raw_id] = raw_spec
    if set(by_id) != set(range(len(chunk))):
        raise ValueError("visual requirement decomposition is incomplete")
    return [
        _validate_visual_requirement_spec_item(
            by_id[requirement_id],
            requirement_id=requirement_id,
            original_requirement=requirement,
            provider_id=provider_id,
            model_name=model_name,
        )
        for requirement_id, requirement in enumerate(chunk)
    ]


def generate_visual_requirement_specs(
    visual_requirements: list[str],
    app_config=None,
) -> dict[str, VisualRequirementSpec]:
    """Decompose all unique requirements with the selected provider, in batches.

    Requirements are requested in small batches because one spec is a large
    object; a single request for a whole timeline produced a response the provider
    truncated, and the entire timeline then had no specs. A batch that fails is
    dropped on its own, so the batches that parsed cleanly still return specs.

    Returned keys are normalized requirements. Missing keys mean the provider
    response was unavailable or failed strict source-grounding validation; no
    synthetic fallback spec is invented.
    """
    unique_requirements: list[str] = []
    seen: set[str] = set()
    for value in visual_requirements:
        requirement = " ".join(str(value or "").split()).strip()
        normalized = normalize_visual_requirement(requirement)
        if (
            not normalized
            or len(requirement) > _MAX_REQUIREMENT_TEXT_LENGTH
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        unique_requirements.append(requirement)
    if not unique_requirements:
        return {}

    provider_id, model_name = _selected_llm_identity(app_config)
    resolved: dict[str, VisualRequirementSpec] = {}
    uncached: list[str] = []
    try:
        for requirement in unique_requirements:
            normalized = normalize_visual_requirement(requirement)
            lock = _visual_requirement_spec_cache_lock(
                normalized, provider_id, model_name
            )
            with lock:
                cached = _load_visual_requirement_spec_cache(
                    requirement, provider_id, model_name
                )
            if cached is None:
                uncached.append(requirement)
            else:
                resolved[normalized] = cached

        if not uncached:
            return resolved

        failed_batches = 0
        for chunk_start in range(0, len(uncached), _VISUAL_REQUIREMENT_BATCH_SIZE):
            chunk = uncached[chunk_start : chunk_start + _VISUAL_REQUIREMENT_BATCH_SIZE]
            request_items = [
                {"requirement_id": index, "visual_requirement": requirement}
                for index, requirement in enumerate(chunk)
            ]
            prompt = f"""
# Role: General Visual Requirement Decomposer

## Goal
Convert each immutable visual requirement into only the camera-observable facts
needed to distinguish the requested scene/action from merely related footage.

## Critical safety rules
1. Never add a mandatory fact merely because it is common, plausible, attractive,
   or useful in a typical scene.
2. A mandatory fact is allowed only when it is explicitly stated by the source
   requirement, or logically necessary for the requested action/relation to be true.
3. Clothing, weather, location, camera angle, colors, and background details are
   optional unless explicitly required by the source wording.
4. basis_quote must be an exact quote from visual_requirement. For a
   logically_necessary fact it must include the exact requested action phrase.
   Mandatory fact wording and every required field must use only words grounded in
   visual_requirement. Put any merely plausible additions in optional_attributes.
5. Keep optional details in optional_attributes; they can never become hard gates.
6. Set direct_evidence_needed=true only when the defining action/relation/event
   itself must be visible. Related subjects or objects alone are not evidence.
7. When primary_action is not null, include at least one mandatory,
   direct_evidence_needed, logically_necessary fact that operationalizes the exact
   visible event/change/relation which makes that action true. Merely restating the
   action label is insufficient: distinguish it from preparation, handling,
   possession, inspection, aftermath, or a nearby related action. New wording is
   allowed in this one kind of fact only when it describes that logically necessary
   defining evidence; never use it to add typical environmental details.
8. A continuous visible state with no requested action must not receive an invented
   action gate.
9. Preserve requirement_id and original wording. Do not output timestamps.

## Strict JSON output
Return one object with exactly one key, specs. specs must contain exactly one object
per input. Every spec object must contain exactly:
requirement_id, original_requirement, subjects, primary_action, objects,
required_relations, required_context, required_visible_state, optional_attributes,
critical_visual_facts, ambiguity_notes.

primary_action is a source-grounded string or null. All plural fields are arrays of
strings. critical_visual_facts is a non-empty array of at most
{_MAX_CRITICAL_VISUAL_FACTS} objects, with sequential IDs f1, f2, ... and exactly:
id, fact, mandatory, direct_evidence_needed, evidence_description, basis_type,
basis_quote. basis_type is explicit or logically_necessary.

Inputs:
{json.dumps(request_items, ensure_ascii=False)}
""".strip()
            parsed = _generate_structured_response(
                prompt,
                purpose="visual requirement decomposition",
                app_config=app_config,
            )
            if parsed is None:
                failed_batches += 1
                continue
            try:
                specs = _validated_requirement_specs(
                    parsed,
                    chunk=chunk,
                    provider_id=provider_id,
                    model_name=model_name,
                )
            except Exception as exc:
                failed_batches += 1
                logger.warning(
                    "visual requirement decomposition batch is unusable: "
                    f"batch_start={chunk_start}, size={len(chunk)}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                continue
            for requirement, spec in zip(chunk, specs):
                normalized = normalize_visual_requirement(requirement)
                resolved[normalized] = spec
                with _visual_requirement_spec_cache_lock(
                    normalized, provider_id, model_name
                ):
                    _save_visual_requirement_spec_cache(spec)

        cached_count = len(unique_requirements) - len(uncached)
        logger.info(
            "visual requirement decomposition completed: "
            f"unique={len(unique_requirements)}, requested={len(uncached)}, "
            f"generated={len(resolved) - cached_count}, cached={cached_count}, "
            f"failed_batches={failed_batches}"
        )
        return resolved
    except Exception as exc:
        logger.warning(
            "visual requirement decomposition returned unusable structured data: "
            f"error={type(exc).__name__}: {exc}"
        )
        return resolved


def _narration_contains_quote(narration_text: str, quote: str) -> bool:
    """True when the quote is a contiguous word sequence of the narration."""
    narration_words = _semantic_words(narration_text)
    quote_words = _semantic_words(quote)
    if not narration_words or not quote_words:
        return False
    if len(quote_words.split()) < 2 and len(narration_words.split()) >= 2:
        return False
    return f" {quote_words} " in f" {narration_words} "


def _validated_alternative_requirements(
    parsed: object,
    chunk: list[dict],
) -> dict[int, dict[str, str]]:
    """Keep only proposals that are grounded in the beat's own spoken text."""
    payload = parsed.get("alternatives") if isinstance(parsed, dict) else parsed
    if not isinstance(payload, list):
        raise ValueError("alternatives must be an array")
    if len(payload) > len(chunk):
        raise ValueError("alternatives contain more items than requested")

    accepted: dict[int, dict[str, str]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("each alternative must be an object")
        item_id = entry.get("item_id")
        if not isinstance(item_id, int) or isinstance(item_id, bool):
            raise ValueError("item_id must be an integer")
        if not 0 <= item_id < len(chunk) or item_id in accepted:
            raise ValueError(f"item_id {item_id} is unknown or duplicated")
        item = chunk[item_id]
        requirement = _bounded_structured_text(
            entry.get("visual_requirement"), "visual_requirement"
        )
        narration_basis = _bounded_structured_text(
            entry.get("narration_basis"), "narration_basis"
        )
        if not re.search(r"[A-Za-z]", requirement):
            logger.warning(
                "alternative visual requirement is not searchable: "
                f"item_index={item['item_index']}"
            )
            continue
        if normalize_visual_requirement(requirement) == normalize_visual_requirement(
            item["failed_requirement"]
        ):
            logger.warning(
                "alternative visual requirement repeats the rejected one: "
                f"item_index={item['item_index']}"
            )
            continue
        if not _narration_contains_quote(item["narration_text"], narration_basis):
            logger.warning(
                "alternative visual requirement is not grounded in the narration: "
                f"item_index={item['item_index']}"
            )
            continue
        accepted[item_id] = {
            "visual_requirement": requirement,
            "narration_basis": narration_basis,
        }
    return accepted


def generate_alternative_visual_requirements(
    items: list[dict],
    app_config=None,
) -> dict[int, dict[str, str]]:
    """Describe the same narration a second way when the first way found nothing.

    A beat becomes unfillable either because its requirement never decomposed or
    because no candidate satisfied it. Both are failures of the *wording*, not of
    the narration, so the fix is to re-describe the same spoken line in plainer,
    more common visual terms instead of abandoning the video.

    Each item needs `item_index`, `narration_text`, and `failed_requirement`.
    Returned keys are the caller's `item_index`; a missing key means the caller
    must keep treating that beat as unfilled. Every proposal must carry a
    `narration_basis` quote which is verified here to be a real fragment of that
    beat's spoken text, because an ungrounded rewrite would quietly replace the
    requested scene with a different one.
    """
    requests: list[dict] = []
    for item in items or []:
        item_index = item.get("item_index") if isinstance(item, dict) else None
        if not isinstance(item_index, int) or isinstance(item_index, bool):
            continue
        narration_text = " ".join(str(item.get("narration_text") or "").split())
        failed_requirement = " ".join(str(item.get("failed_requirement") or "").split())
        if not narration_text or len(narration_text) > _MAX_REQUIREMENT_TEXT_LENGTH:
            continue
        if len(failed_requirement) > _MAX_REQUIREMENT_TEXT_LENGTH:
            continue
        requests.append(
            {
                "item_index": item_index,
                "narration_text": narration_text,
                "failed_requirement": failed_requirement,
                "problem": " ".join(str(item.get("problem") or "").split())[
                    :_MAX_STRUCTURED_TEXT_LENGTH
                ],
            }
        )
    if not requests:
        return {}

    resolved: dict[int, dict[str, str]] = {}
    for chunk_start in range(0, len(requests), _VISUAL_REQUIREMENT_BATCH_SIZE):
        chunk = requests[chunk_start : chunk_start + _VISUAL_REQUIREMENT_BATCH_SIZE]
        request_items = [
            {
                "item_id": batch_local_id,
                "spoken_text": item["narration_text"],
                "rejected_visual_requirement": item["failed_requirement"],
                "why_it_failed": item["problem"],
            }
            for batch_local_id, item in enumerate(chunk)
        ]
        prompt = f"""
# Role: Alternative Visual Requirement Author

## Goal
For each item, write ONE different way to show the SAME spoken line. The rejected
requirement could not be satisfied by real stock footage, so the alternative must
be easier to film and easier to find, without changing what the line says.

## Rules
1. Depict the same spoken line. Never introduce a subject, action, object, event,
   place, or claim that the spoken text does not support.
2. Choose the most ordinary camera-visible moment that still fits the line: one
   concrete subject performing one concrete visible action or in one visible state.
3. Simpler and more common wins. Drop counts, named places, brands, on-screen text,
   specific weather, rare compound scenes, and any detail the spoken text does not
   state.
4. Do not reuse the rejected requirement, and do not merely reorder its words or
   swap synonyms; change which visible moment is shown.
5. Never describe abstract meaning, emotions, metaphors, camera moves, transitions,
   edits, or narration summaries.
6. narration_basis must be an exact contiguous quote copied from that item's
   spoken_text, in its original language, at least two words long. It is the proof
   that the alternative still describes this line.
7. visual_requirement must be one short concrete English phrase, because it is used
   for stock search and footage comparison. No timestamps and no lists.

## Strict JSON output
Return one object with exactly one key, alternatives. alternatives must contain at
most one object per input, and each object must contain exactly:
item_id, visual_requirement, narration_basis.
Omit an item entirely rather than inventing an ungrounded alternative for it.

Inputs:
{json.dumps(request_items, ensure_ascii=False)}
""".strip()
        parsed = _generate_structured_response(
            prompt,
            purpose="alternative visual requirement",
            app_config=app_config,
        )
        if parsed is None:
            continue
        try:
            accepted = _validated_alternative_requirements(parsed, chunk)
        except Exception as exc:
            logger.warning(
                "alternative visual requirement batch is unusable: "
                f"batch_start={chunk_start}, size={len(chunk)}, "
                f"error={type(exc).__name__}: {exc}"
            )
            continue
        for batch_local_id, alternative in accepted.items():
            resolved[chunk[batch_local_id]["item_index"]] = alternative

    logger.info(
        "alternative visual requirements completed: "
        f"requested={len(requests)}, generated={len(resolved)}"
    )
    return resolved


def _observed_facts_to_dict(value: ObservedVisualFacts | dict[str, Any]) -> dict:
    if isinstance(value, ObservedVisualFacts):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise ValueError("observed facts are invalid")


_ADJUDICATION_ITEM_FIELDS = {
    "candidate_id",
    "decision",
    "mandatory_fact_results",
    "missing_or_contradictory_facts",
    "reason",
}


def _validated_adjudication(
    item: object,
    *,
    mandatory_ids: set[str],
    source_statuses: dict[str, dict[str, str]],
) -> SemanticAdjudication:
    """Check one adjudication record against the evidence that was actually supplied.

    This runs over cached decisions as well as fresh ones, which is what makes the
    cache safe to trust: a stored verdict is re-checked against the statuses of the
    run reusing it, so an entry written under different evidence — or edited by
    hand — cannot smuggle an ACCEPT into a beat.
    """
    if not isinstance(item, dict) or set(item) != _ADJUDICATION_ITEM_FIELDS:
        raise ValueError("semantic adjudication fields are invalid")
    candidate_id = str(item.get("candidate_id") or "").strip()
    if candidate_id not in source_statuses:
        raise ValueError("semantic adjudication candidate id is invalid")
    decision = item.get("decision")
    if decision not in {"ACCEPT", "REJECT", "UNCERTAIN"}:
        raise ValueError("semantic adjudication decision is invalid")
    raw_fact_results = item.get("mandatory_fact_results")
    if not isinstance(raw_fact_results, list):
        raise ValueError("mandatory_fact_results must be an array")
    fact_results: list[MandatoryFactResult] = []
    seen_fact_ids: set[str] = set()
    for raw_result in raw_fact_results:
        if not isinstance(raw_result, dict) or set(raw_result) != {
            "fact_id",
            "status",
        }:
            raise ValueError("mandatory fact result fields are invalid")
        fact_id = raw_result.get("fact_id")
        status = raw_result.get("status")
        if (
            fact_id not in mandatory_ids
            or fact_id in seen_fact_ids
            or status != source_statuses[candidate_id].get(fact_id)
        ):
            raise ValueError("adjudicator modified supplied fact evidence")
        seen_fact_ids.add(fact_id)
        fact_results.append(MandatoryFactResult(fact_id=fact_id, status=status))
    if seen_fact_ids != mandatory_ids:
        raise ValueError("adjudication omitted mandatory facts")
    if decision == "ACCEPT" and any(
        result.status != "OBSERVED" for result in fact_results
    ):
        raise ValueError("adjudicator accepted a failed mandatory fact")
    missing = item.get("missing_or_contradictory_facts")
    if not isinstance(missing, list) or any(
        not isinstance(fact_id, str) or fact_id not in mandatory_ids
        for fact_id in missing
    ):
        raise ValueError("adjudication missing fact IDs are invalid")
    reason = _bounded_structured_text(item.get("reason"), "adjudication reason")
    if _TIMESTAMP_EVIDENCE_RE.search(reason):
        raise ValueError("adjudication reason invented or repeated a timestamp")
    return SemanticAdjudication(
        candidate_id=candidate_id,
        decision=decision,
        mandatory_fact_results=fact_results,
        missing_or_contradictory_facts=list(dict.fromkeys(missing)),
        reason=reason,
    )


def _semantic_adjudication_cache_dir() -> Path:
    return Path(utils.storage_dir("cache_semantic_adjudication", create=True))


def _semantic_adjudication_cache_digest(
    candidate_id: str,
    observed_facts: dict[str, Any],
    requirement_spec_digest: str,
    provider_id: str,
    model_name: str,
) -> str:
    payload = json.dumps(
        {
            "candidate_id": candidate_id,
            "observed_facts_hash": hashlib.sha256(
                json.dumps(
                    observed_facts,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "requirement_spec_digest": str(requirement_spec_digest or ""),
            "provider": provider_id,
            "model": model_name,
            "schema_version": SEMANTIC_ADJUDICATION_SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _semantic_adjudication_cache_path(digest: str) -> Path:
    return _semantic_adjudication_cache_dir() / f"{digest}.json"


def _semantic_adjudication_cache_lock(digest: str) -> threading.Lock:
    return _SEMANTIC_ADJUDICATION_CACHE_LOCKS[
        int(digest[:8], 16) % len(_SEMANTIC_ADJUDICATION_CACHE_LOCKS)
    ]


def _load_semantic_adjudication_cache(
    digest: str,
    *,
    mandatory_ids: set[str],
    source_statuses: dict[str, dict[str, str]],
) -> SemanticAdjudication | None:
    """Return a stored verdict for this exact evidence, or None to pay for a fresh one.

    Every failure mode — absent file, unreadable file, superseded format, a record
    that no longer squares with the supplied evidence — returns None, so the worst a
    damaged cache can do is cost one normal request.
    """
    try:
        payload = json.loads(
            _semantic_adjudication_cache_path(digest).read_text(encoding="utf-8")
        )
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _SEMANTIC_ADJUDICATION_CACHE_FORMAT_VERSION
            or payload.get("schema_version") != SEMANTIC_ADJUDICATION_SCHEMA_VERSION
            or not isinstance(payload.get("decision"), dict)
        ):
            return None
        return _validated_adjudication(
            payload["decision"],
            mandatory_ids=mandatory_ids,
            source_statuses=source_statuses,
        )
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning(
            "failed to reuse a cached semantic adjudication: "
            f"error={type(exc).__name__}"
        )
        return None


def _save_semantic_adjudication_cache(
    digest: str,
    adjudication: SemanticAdjudication,
) -> None:
    cache_path = _semantic_adjudication_cache_path(digest)
    temp_path: Path | None = None
    try:
        payload = {
            "version": _SEMANTIC_ADJUDICATION_CACHE_FORMAT_VERSION,
            "schema_version": SEMANTIC_ADJUDICATION_SCHEMA_VERSION,
            "decision": asdict(adjudication),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, ensure_ascii=False, separators=(",", ":"))
            temp_file.flush()
        temp_path.replace(cache_path)
        temp_path = None
    except Exception as exc:
        logger.warning(
            f"failed to write semantic adjudication cache: error={type(exc).__name__}"
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def adjudicate_visual_candidates(
    requirement_spec: VisualRequirementSpec,
    candidates: list[dict[str, Any]],
    app_config=None,
) -> dict[str, SemanticAdjudication]:
    """Batch text-only adjudication over immutable, already-observed candidate facts."""
    if not candidates:
        return {}
    mandatory_ids = {
        fact.id for fact in requirement_spec.critical_visual_facts if fact.mandatory
    }
    safe_candidates: list[dict[str, Any]] = []
    source_statuses: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in source_statuses:
            return {}
        facts = _observed_facts_to_dict(candidate.get("observed_facts"))
        evidence_items = facts.get("critical_fact_evidence")
        if not isinstance(evidence_items, list):
            return {}
        status_map: dict[str, str] = {}
        for item in evidence_items:
            if not isinstance(item, dict):
                return {}
            fact_id = item.get("fact_id")
            status = item.get("status")
            if fact_id in status_map or status not in {
                "OBSERVED",
                "NOT_OBSERVED",
                "CONTRADICTED",
                "UNCERTAIN",
            }:
                return {}
            status_map[fact_id] = status
        if not mandatory_ids.issubset(status_map):
            return {}
        source_statuses[candidate_id] = status_map
        safe_candidates.append(
            {"candidate_id": candidate_id, "observed_facts": facts}
        )

    # The paid video observation is already cached, but the verdict drawn from it was
    # not, so re-running the same script used to cost nothing at the video model and
    # full price again here. Identical evidence judged by the same model under the
    # same rules yields the same verdict, so it is looked up rather than re-bought.
    cache_digests: dict[str, str] = {}
    results: dict[str, SemanticAdjudication] = {}
    pending: list[dict[str, Any]] = list(safe_candidates)
    try:
        provider_id, model_name = _selected_llm_identity(app_config)
        spec_digest = visual_requirement_spec_digest(requirement_spec)
        pending = []
        for candidate in safe_candidates:
            digest = _semantic_adjudication_cache_digest(
                candidate["candidate_id"],
                candidate["observed_facts"],
                spec_digest,
                provider_id,
                model_name,
            )
            cache_digests[candidate["candidate_id"]] = digest
            with _semantic_adjudication_cache_lock(digest):
                cached = _load_semantic_adjudication_cache(
                    digest,
                    mandatory_ids=mandatory_ids,
                    source_statuses=source_statuses,
                )
            if cached is None:
                pending.append(candidate)
            else:
                results[candidate["candidate_id"]] = cached
    except Exception as exc:
        # A cache that cannot be consulted costs tokens, never correctness.
        logger.warning(
            "semantic adjudication cache unavailable: "
            f"error={type(exc).__name__}: {exc}"
        )
        cache_digests = {}
        results = {}
        pending = list(safe_candidates)
    if not pending:
        logger.info(
            "semantic adjudication reused every verdict from cache: "
            f"candidates={len(results)}"
        )
        return results
    if results:
        logger.info(
            "semantic adjudication partially cached: "
            f"reused={len(results)}, requested={len(pending)}"
        )

    prompt = f"""
# Role: Strict Text-Only Visual Evidence Adjudicator

You receive one immutable VisualRequirementSpec and structured ObservedVisualFacts.
Reason only over supplied facts. Never alter the requirement or observations, invent
visual evidence, add facts, or invent timestamps.

Decision rules:
- ACCEPT only if every mandatory critical fact has direct OBSERVED evidence and no
  supplied contradiction or unresolved uncertainty undermines the core meaning.
- REJECT if any mandatory fact is NOT_OBSERVED or CONTRADICTED, or supplied facts
  directly show a different defining action/relation.
- UNCERTAIN if no mandatory fact is disproved but any mandatory fact or defining
  relation remains ambiguous, occluded, partial, inferred, or insufficient.
- Related subject matter never substitutes for the requested action/relation.
- Treat supplied status labels as immutable records, not proof by themselves. Check
  whether the accompanying evidence text actually entails the exact critical fact.
  For a defining action, generic handling/movement, preparation, possession, or an
  aftermath does not establish the required change or source/target relation.
- If an OBSERVED label is supported only by evidence for a related but different
  action, REJECT without changing that supplied label and list the affected fact ID.
- Use that REJECT rule only when the supplied observations directly establish the
  different/conflicting action or a sufficiently complete visible sequence. When
  intent, attention, a defining transition, or an occluded event simply cannot be
  determined from the supplied evidence, return UNCERTAIN rather than REJECT.

Return raw JSON with exactly one key, decisions. Return one decision per candidate.
Each decision must contain exactly: candidate_id, decision,
mandatory_fact_results, missing_or_contradictory_facts, reason.
decision is ACCEPT, REJECT, or UNCERTAIN. mandatory_fact_results must repeat every
mandatory fact ID exactly once with the unchanged supplied status. Do not include
timestamps in reason.

VisualRequirementSpec:
{json.dumps(visual_requirement_spec_to_dict(requirement_spec), ensure_ascii=False)}

Candidates:
{json.dumps(pending, ensure_ascii=False)}
""".strip()
    try:
        parsed = _generate_structured_response(
            prompt,
            purpose="semantic adjudication",
            app_config=app_config,
        )
        if parsed is None:
            return {}
        if not isinstance(parsed, dict) or set(parsed) != {"decisions"}:
            raise ValueError("semantic adjudication root is invalid")
        raw_decisions = parsed.get("decisions")
        if not isinstance(raw_decisions, list) or len(raw_decisions) != len(pending):
            raise ValueError("semantic adjudication count is invalid")
        pending_ids = {candidate["candidate_id"] for candidate in pending}
        fresh: dict[str, SemanticAdjudication] = {}
        for item in raw_decisions:
            adjudication = _validated_adjudication(
                item,
                mandatory_ids=mandatory_ids,
                source_statuses=source_statuses,
            )
            if (
                adjudication.candidate_id not in pending_ids
                or adjudication.candidate_id in fresh
            ):
                raise ValueError("semantic adjudication candidate id is invalid")
            fresh[adjudication.candidate_id] = adjudication
        results.update(fresh)
        if set(results) != set(source_statuses):
            raise ValueError("semantic adjudication is incomplete")
        # Written only after the whole response validated, so a partly malformed
        # batch never leaves a verdict behind for the next run to trust. Persisting
        # is a saving, not a result, so a failure here must never discard verdicts
        # the caller already paid for.
        try:
            for candidate_id, adjudication in fresh.items():
                digest = cache_digests.get(candidate_id)
                if not digest:
                    continue
                with _semantic_adjudication_cache_lock(digest):
                    _save_semantic_adjudication_cache(digest, adjudication)
        except Exception as exc:
            logger.warning(
                "failed to store semantic adjudication verdicts: "
                f"error={type(exc).__name__}: {exc}"
            )
        return results
    except Exception as exc:
        logger.warning(
            "semantic adjudication returned unusable structured data: "
            f"error={type(exc).__name__}: {exc}"
        )
        return {}


# =============================================================================
# Social publishing metadata
#
# 根据视频主题和脚本生成发布到短视频平台时常用的 title、caption 和 hashtags。
# 这块能力只复用现有 LLM provider，不接入任何外部发布服务，也不影响视频生成主链路。
# =============================================================================

# 不同平台的文案长度和 hashtag 数量偏好不同。这里使用保守上限，避免模型返回
# 过长内容后调用方还需要二次裁剪。
SOCIAL_PLATFORMS = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}
DEFAULT_SOCIAL_PLATFORM = "tiktok"
DEFAULT_SOCIAL_LANGUAGE = "auto"
MAX_SOCIAL_SUBJECT_LENGTH = 500
MAX_SOCIAL_SCRIPT_LENGTH = 8000
MAX_SOCIAL_LANGUAGE_LENGTH = 64

SOCIAL_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "instagram_reels": "Instagram Reels",
    "facebook_reels": "Facebook Reels",
}

# LLM 不可用时的通用兜底标签。这里故意不绑定某个国家或语种，保证 API
# 对中文、英文、越南语等不同场景都能返回可用结构。
DEFAULT_SOCIAL_HASHTAGS = [
    "#shorts",
    "#viral",
    "#trending",
    "#fyp",
    "#video",
    "#reels",
    "#creator",
    "#content",
]


def _resolve_social_platform(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    return value if value in SOCIAL_PLATFORMS else DEFAULT_SOCIAL_PLATFORM


def _normalize_social_language(language: str | None) -> str:
    value = (language or DEFAULT_SOCIAL_LANGUAGE).strip()
    if len(value) > MAX_SOCIAL_LANGUAGE_LENGTH:
        logger.warning(
            "social metadata language is too long and will be truncated to "
            f"{MAX_SOCIAL_LANGUAGE_LENGTH} characters."
        )
        value = value[:MAX_SOCIAL_LANGUAGE_LENGTH]
    return value or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API 层会限制长度；这里继续兜底，是为了保护内部调用或未来 WebUI
    # 直接调用时不会把超长内容发送给模型，避免 token 成本异常。
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _social_language_instruction(language: str | None) -> str:
    language = _normalize_social_language(language)
    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        return (
            "Use the same language as the video subject and script. If the subject "
            "and script use different languages, prefer the script language."
        )

    return f'Write "title" and "caption" in this language: {language}.'


def _clamp_text(text, max_length: int) -> str:
    value = ("" if text is None else str(text)).strip()
    if max_length and len(value) > max_length:
        return value[:max_length].rstrip()
    return value


def _normalize_hashtags(raw, count: int) -> List[str]:
    """
    将 LLM 返回的 hashtag 统一整理成 `#tag` 格式。

    LLM 可能返回字符串、数组、带空格的词组、重复标签或包含标点的内容。
    这里集中清洗，可以让接口响应结构稳定，也避免平台发布时出现空标签、
    重复标签或不符合常见格式的 hashtag。
    """
    if isinstance(raw, str):
        candidates = re.split(r"[\s,]+", raw)
    elif isinstance(raw, (list, tuple)):
        # 数组里的每一项视为一个完整标签，因此 "du lich" 会变成
        # "#dulich"，而不是拆成两个标签。
        candidates = [str(entry) for entry in raw]
    else:
        candidates = []

    seen = set()
    result: List[str] = []
    for item in candidates:
        tag = re.sub(r"[^\w]", "", item, flags=re.UNICODE)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"#{tag}")
        if count and len(result) >= count:
            break
    return result


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    platform = _resolve_social_platform(platform)
    spec = SOCIAL_PLATFORMS[platform]
    label = SOCIAL_PLATFORM_LABELS.get(platform, platform)
    language_instruction = _social_language_instruction(language)

    prompt = f"""
# Role: Short-Video Social Media Copywriter

## Goal
Write engaging publishing metadata for a short video that will be posted on {label}.

## Constraints
1. Respond ONLY with a single valid minified JSON object. No markdown, no code fences, no commentary.
2. The JSON must contain exactly these keys: "title", "caption", "hashtags".
3. "title": a catchy hook, at most {spec["title_max"]} characters.
4. "caption": an engaging description that ends with a call to action, at most {spec["caption_max"]} characters. Do not put hashtags inside the caption.
5. "hashtags": a JSON array of exactly {spec["hashtag_count"]} strings. Each must start with "#", contain no spaces, and be relevant to the topic and to {label}.
6. {language_instruction}

## Output Example
{{"title":"...","caption":"...","hashtags":["#example","#video"]}}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
    return prompt


def _parse_social_metadata(response: str, platform: str) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]

    data = None
    try:
        data = json.loads(_strip_code_fence(response))
    except Exception:
        # 部分模型会在 JSON 外层包一段说明文字或 markdown fence。
        # API 调用方只需要稳定结构，所以这里尝试提取第一个 JSON object。
        match = re.search(r"\{.*\}", response or "", re.DOTALL)
        if match:
            data = json.loads(match.group())

    if not isinstance(data, dict):
        raise ValueError("social metadata response is not a JSON object")

    title = _clamp_text(data.get("title", ""), spec["title_max"])
    caption = _clamp_text(data.get("caption", ""), spec["caption_max"])
    hashtags = _normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"])

    if not title and not caption:
        raise ValueError("social metadata response is missing both title and caption")

    return {"title": title, "caption": caption, "hashtags": hashtags}


def _fallback_social_metadata(
    video_subject: str, video_script: str, platform: str
) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]
    subject = (video_subject or "").strip()
    script = (video_script or "").strip()

    title = subject
    if not title and script:
        # 没有主题时，用脚本第一句兜底生成 title，避免接口返回空标题。
        title = re.split(r"(?<=[.!?。！？])\s+", script)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    生成短视频发布文案元数据。

    返回结构固定为 `{"title": str, "caption": str, "hashtags": List[str]}`。
    如果 LLM 不可用或返回格式异常，会降级为通用启发式结果，保证 API
    调用方始终拿到可展示、可发布前编辑的数据结构。
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(f"generating social metadata: platform={platform}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and "Error: " in response:
                logger.error(f"failed to generate social metadata: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completed: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"failed to parse social metadata: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"failed to generate social metadata, trying again... {i + 1}"
            )

    logger.warning("falling back to heuristic social metadata")
    return _fallback_social_metadata(video_subject, video_script, platform)


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
