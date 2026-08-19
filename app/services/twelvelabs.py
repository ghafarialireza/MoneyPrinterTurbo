"""
TwelveLabs (https://twelvelabs.io) integration — optional, opt-in helpers.

This module wraps two TwelveLabs models so MoneyPrinterTurbo can make better
use of the stock/B-roll footage it downloads:

  * Marengo (multimodal embeddings, 512-dim) — used to *semantically reorder*
    the LLM-generated search terms against the video subject, so that when the
    timeline budget runs out the most on-topic footage is the footage that made
    it in (instead of whatever the LLM happened to list first).

  * Pegasus (video understanding) — used to QA / describe a generated clip from
    a public URL, e.g. to sanity-check that a downloaded clip actually matches
    the script before it ships.

The integration is fully opt-in and non-breaking:
  * If `twelvelabs_api_keys` is not configured, every public function here is a
    no-op that returns its input unchanged (or None), so default behavior is
    identical to a build without TwelveLabs.
  * The `twelvelabs` SDK is imported lazily, so the dependency is only required
    when the feature is actually used.

Config (config.toml, [app] section):
    twelvelabs_api_keys = ["tlk_xxx"]   # required to enable
    twelvelabs_rerank_terms = true      # opt-in: reorder search terms by relevance
    twelvelabs_clip_qa = true           # opt-in: reject mismatched clips
    twelvelabs_clip_qa_min_score = 0.70 # match-confidence threshold
    twelvelabs_clip_qa_fail_closed = true # reject candidates when QA is unavailable
    twelvelabs_marengo_model = "marengo3.0"   # optional override
    twelvelabs_pegasus_model = "pegasus1.5"   # optional override

Configure a TwelveLabs API key from the TwelveLabs dashboard (https://twelvelabs.io) to enable this optional integration.
"""

import json
import hashlib
import math
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Sequence

from packaging.version import Version

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo
from app.services import material
from app.utils import utils

DEFAULT_MARENGO_MODEL = "marengo3.0"
DEFAULT_PEGASUS_MODEL = "pegasus1.5"
MINIMUM_SDK_VERSION = Version("1.3.0")
# Pegasus requires max_tokens in [512, 98304]; 512 is plenty for a one-line QA.
_PEGASUS_MIN_MAX_TOKENS = 512
_PEGASUS_TEMPORAL_MIN_MAX_TOKENS = 2048
DEFAULT_CLIP_QA_MIN_SCORE = 0.70
DEFAULT_CANDIDATE_BATCH_SIZE = 5
DEFAULT_MAX_CANDIDATES_PER_SLOT = 15
DEFAULT_STRONG_EARLY_STOP_SCORE = 0.90
DEFAULT_PREFERRED_MAX_SOURCE_DURATION = 30.0
DEFAULT_CANDIDATE_CONCURRENCY = 5
EVALUATION_SCHEMA_VERSION = "smart-pexels-candidate-v1"
_CACHE_FORMAT_VERSION = 1
_CACHE_LOCKS = tuple(threading.Lock() for _ in range(128))

_CANDIDATE_SCORE_WEIGHTS = {
    "semantic_match": 0.35,
    "action_match": 0.30,
    "subject_visibility": 0.20,
    "visual_quality": 0.15,
}

_CANDIDATE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "match": {"type": "boolean"},
        "required_action_visible": {"type": "boolean"},
        "subject_clearly_visible": {"type": "boolean"},
        "scores": {
            "type": "object",
            "properties": {
                "semantic_match": {"type": "number"},
                "action_match": {"type": "number"},
                "subject_visibility": {"type": "number"},
                "visual_quality": {"type": "number"},
            },
            "required": list(_CANDIDATE_SCORE_WEIGHTS),
        },
        "quality_flags": {
            "type": "object",
            "properties": {
                "severe_blur": {"type": "boolean"},
                "dominant_text_or_logo": {"type": "boolean"},
                "bad_orientation": {"type": "boolean"},
                "awkward_or_unusable_framing": {"type": "boolean"},
            },
            "required": [
                "severe_blur",
                "dominant_text_or_logo",
                "bad_orientation",
                "awkward_or_unusable_framing",
            ],
        },
        "visible_summary": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": [
        "match",
        "required_action_visible",
        "subject_clearly_visible",
        "scores",
        "quality_flags",
        "visible_summary",
        "reason",
    ],
}


def is_enabled() -> bool:
    """True only when at least one TwelveLabs API key is configured."""
    keys = config.app.get("twelvelabs_api_keys")
    return bool(keys)


def is_clip_qa_enabled() -> bool:
    """True when both TwelveLabs credentials and per-clip QA are enabled."""
    return is_enabled() and bool(config.app.get("twelvelabs_clip_qa", False))


def visual_matching_requested() -> bool:
    """Whether the user explicitly enabled smart TwelveLabs material matching."""
    return bool(config.app.get("twelvelabs_clip_qa", False))


def is_smart_visual_matching_enabled() -> bool:
    """True only when smart matching is requested and credentials are present."""
    return visual_matching_requested() and is_enabled()


def clip_qa_fail_closed() -> bool:
    """Whether an unavailable/malformed QA result must reject the candidate."""
    return bool(config.app.get("twelvelabs_clip_qa_fail_closed", True))


def _clip_qa_min_score(value: Any = None) -> float:
    if value is None:
        value = config.app.get(
            "twelvelabs_clip_qa_min_score", DEFAULT_CLIP_QA_MIN_SCORE
        )
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        logger.warning(
            f"invalid twelvelabs_clip_qa_min_score; using {DEFAULT_CLIP_QA_MIN_SCORE}"
        )
        return DEFAULT_CLIP_QA_MIN_SCORE


def _bounded_int(config_key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(config.app.get(config_key, default))
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


def _bounded_float(
    config_key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(config.app.get(config_key, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return min(maximum, max(minimum, value))


def candidate_selection_settings() -> dict[str, Any]:
    """Return normalized, bounded settings used by the smart selector."""
    batch_size = _bounded_int(
        "twelvelabs_candidate_batch_size",
        DEFAULT_CANDIDATE_BATCH_SIZE,
        1,
        5,
    )
    return {
        "batch_size": batch_size,
        "max_candidates": _bounded_int(
            "twelvelabs_max_candidates_per_slot",
            DEFAULT_MAX_CANDIDATES_PER_SLOT,
            1,
            15,
        ),
        "minimum_score": _clip_qa_min_score(),
        "strong_early_stop_score": _bounded_float(
            "twelvelabs_strong_early_stop_score",
            DEFAULT_STRONG_EARLY_STOP_SCORE,
            0.0,
            1.0,
        ),
        "preferred_max_source_duration": _bounded_float(
            "twelvelabs_preferred_max_source_duration",
            DEFAULT_PREFERRED_MAX_SOURCE_DURATION,
            4.0,
            600.0,
        ),
        "concurrency": min(
            batch_size,
            _bounded_int(
                "twelvelabs_candidate_concurrency",
                DEFAULT_CANDIDATE_CONCURRENCY,
                1,
                5,
            ),
        ),
        "fail_closed": clip_qa_fail_closed(),
    }


def validate_smart_visual_matching_configuration() -> str | None:
    """Return a user-facing preflight error without exposing credentials."""
    if not visual_matching_requested():
        return None
    if not is_enabled():
        return "TwelveLabs visual matching is enabled but its API key is missing"
    try:
        from importlib.metadata import version

        sdk_version = Version(version("twelvelabs"))
    except Exception:
        return (
            "TwelveLabs visual matching requires the optional twelvelabs SDK "
            "version 1.3.0"
        )
    if sdk_version < MINIMUM_SDK_VERSION:
        return (
            "TwelveLabs visual matching requires twelvelabs SDK 1.3.0 or newer; "
            f"installed version is {sdk_version}"
        )
    return None


def _client():
    # Lazy import + rotated key reuse mirrors the other providers in
    # material.py (get_api_key rotates across configured keys).
    from twelvelabs import TwelveLabs

    api_key = material.get_api_key("twelvelabs_api_keys")
    return TwelveLabs(api_key=api_key)


def _api_status_code(exc: Exception) -> int | None:
    for name in ("status_code", "status", "code"):
        value = getattr(exc, name, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    response = getattr(exc, "response", None)
    try:
        return int(getattr(response, "status_code", None))
    except (TypeError, ValueError):
        return None


def _safe_api_failure_reason(exc: Exception) -> str:
    """Classify an SDK failure without logging its request or API key."""
    status = _api_status_code(exc)
    type_name = type(exc).__name__
    if status == 429 or "RateLimit" in type_name:
        return "TwelveLabs quota or rate limit exhausted"
    if status in {401, 403} or type_name in {
        "AuthenticationError",
        "PermissionDeniedError",
    }:
        return "TwelveLabs authentication or permission failed"
    if status is not None:
        return f"TwelveLabs API error (HTTP {status})"
    return f"TwelveLabs API unavailable ({type_name})"


def _is_direct_url_failure(exc: Exception) -> bool:
    """Only source-validation failures justify creating a temporary Asset."""
    if _api_status_code(exc) not in {400, 422}:
        return False
    # A generic 400 can also mean an invalid prompt/schema. Uploading the same
    # video in that situation wastes quota and cannot fix the request. Inspect
    # the SDK error privately (never log it) and fall back only when the service
    # explicitly identifies the URL/video source as the failed input.
    detail_parts = [str(exc)]
    response = getattr(exc, "response", None)
    if response is not None:
        detail_parts.append(str(getattr(response, "text", "") or ""))
    detail = " ".join(detail_parts).lower()
    source_failure_phrases = (
        "video url",
        "video_url",
        "video source",
        "media source",
        "video context",
        "invalid url",
        "unable to fetch",
        "failed to fetch",
        "unable to download",
        "failed to download",
        "could not download",
        "cannot access video",
        "could not access video",
    )
    return any(phrase in detail for phrase in source_failure_phrases)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_text(text: str, model: Optional[str] = None) -> Optional[List[float]]:
    """
    Return a 512-dim Marengo text embedding, or None on failure / when disabled.

    Cached so repeated terms across a session don't re-hit the API.
    """
    if not is_enabled() or not text or not text.strip():
        return None
    model = model or config.app.get("twelvelabs_marengo_model", DEFAULT_MARENGO_MODEL)
    try:
        # lru_cache only memoizes successful returns; a raised exception is not
        # cached, so a transient API error never poisons the cache.
        return _embed_text_cached(text.strip(), model)
    except Exception as e:  # noqa: BLE001 - never break the pipeline on TL errors
        logger.warning(f"TwelveLabs embed_text failed, skipping rerank: {e}")
        return None


@lru_cache(maxsize=512)
def _embed_text_cached(text: str, model: str) -> List[float]:
    client = _client()
    resp = client.embed.create(model_name=model, text=text)
    # SDK aliases the raw JSON 'float' vector key to `float_`.
    return list(resp.text_embedding.segments[0].float_)


def rerank_terms_by_subject(
    video_subject: str,
    search_terms: List[str],
    model: Optional[str] = None,
) -> List[str]:
    """
    Reorder `search_terms` so the terms most semantically relevant to
    `video_subject` come first (Marengo cosine similarity).

    Opt-in: only runs when TwelveLabs is enabled AND
    `twelvelabs_rerank_terms` is truthy. Falls back to the original order on
    any failure, so it can never make the pipeline worse.
    """
    if not is_enabled() or not config.app.get("twelvelabs_rerank_terms"):
        return search_terms
    if not video_subject or len(search_terms) < 2:
        return search_terms

    subject_vec = embed_text(video_subject, model)
    if subject_vec is None:
        return search_terms

    scored = []
    for term in search_terms:
        vec = embed_text(term, model)
        if vec is None:
            # If any term can't be embedded, don't risk a partial reorder.
            return search_terms
        scored.append((term, _cosine(subject_vec, vec)))

    ranked = [term for term, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    logger.info(
        f"TwelveLabs Marengo reranked {len(ranked)} search terms by relevance "
        f"to subject '{video_subject}': {ranked}"
    )
    return ranked


def _candidate_cache_dir() -> Path:
    return Path(utils.storage_dir("cache_twelvelabs_candidate", create=True))


def _candidate_cache_digest(asset_id: str, narration_text: str) -> str:
    payload = json.dumps(
        {
            "asset_id": str(asset_id),
            "narration_hash": hashlib.sha256(
                narration_text.strip().encode("utf-8")
            ).hexdigest(),
            "model": DEFAULT_PEGASUS_MODEL,
            "schema_version": EVALUATION_SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_cache_path(asset_id: str, narration_text: str) -> Path:
    return _candidate_cache_dir() / (
        f"{_candidate_cache_digest(asset_id, narration_text)}.json"
    )


def _candidate_cache_lock(asset_id: str, narration_text: str) -> threading.Lock:
    digest = _candidate_cache_digest(asset_id, narration_text)
    return _CACHE_LOCKS[int(digest[:8], 16) % len(_CACHE_LOCKS)]


def _load_candidate_evaluation_cache(
    asset_id: str,
    narration_text: str,
) -> dict[str, Any] | None:
    try:
        cache_path = _candidate_cache_path(asset_id, narration_text)
        with cache_path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _CACHE_FORMAT_VERSION
            or payload.get("model") != DEFAULT_PEGASUS_MODEL
            or payload.get("schema_version") != EVALUATION_SCHEMA_VERSION
            or not isinstance(payload.get("result"), dict)
        ):
            return None
        return dict(payload["result"])
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning(
            f"failed to read TwelveLabs candidate cache: error={type(exc).__name__}"
        )
        return None


def _save_candidate_evaluation_cache(
    asset_id: str,
    narration_text: str,
    result: dict[str, Any],
) -> None:
    temp_path: Path | None = None
    try:
        cache_path = _candidate_cache_path(asset_id, narration_text)
        persisted_result = {
            key: value for key, value in result.items() if not key.startswith("_")
        }
        payload = {
            "version": _CACHE_FORMAT_VERSION,
            "model": DEFAULT_PEGASUS_MODEL,
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "result": persisted_result,
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
            os.fsync(temp_file.fileno())
        os.replace(temp_path, cache_path)
        temp_path = None
    except Exception as exc:
        logger.warning(
            f"failed to write TwelveLabs candidate cache: error={type(exc).__name__}"
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _clean_text(value: Any, maximum_length: int) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()
    return text[:maximum_length]


def _score_component(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def _candidate_is_accepted(
    result: dict[str, Any],
    minimum_score: float,
) -> bool:
    flags = result.get("quality_flags")
    overall_score = _score_component(result.get("overall_score"))
    return bool(
        result.get("match") is True
        and result.get("required_action_visible") is True
        and result.get("subject_clearly_visible") is True
        and isinstance(flags, dict)
        and not any(value is True for value in flags.values())
        and overall_score is not None
        and overall_score >= minimum_score
    )


def _candidate_prompt(
    *,
    slot_index: int,
    slot_duration: float,
    narration_text: str,
    search_query: str,
) -> str:
    return f"""
You are evaluating one stock-video candidate for visual use in narration slot {slot_index}.

ACTUAL NARRATION REQUIREMENT:
{narration_text.strip()}

SLOT DURATION: {slot_duration:.3f} seconds
RETRIEVAL SEARCH QUERY: {search_query.strip()}

The search query was used only to retrieve this candidate. It is a hint, not
the acceptance requirement. Judge only the ACTUAL NARRATION REQUIREMENT against
VISIBLE VIDEO CONTENT. A related topic alone is not enough. If the narration
specifies an action, that exact action must be visibly occurring. If it does not
specify an action, required_action_visible means the required scene or state is
visibly satisfied. Reject severe blur, dominant text/logo, bad orientation, and
awkward or unusable framing. Scores must be numbers from 0.0 to 1.0.
""".strip()


def _parse_candidate_response(
    response: Any,
    minimum_score: float,
) -> dict[str, Any] | None:
    """Validate official structured JSON; regex remains a legacy fallback only."""
    if isinstance(response, dict):
        payload = response
    elif isinstance(response, str) and response.strip():
        value = response.strip()
        try:
            payload = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            match = re.search(r"\{.*\}", value, re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group())
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
    else:
        return None

    if not isinstance(payload, dict):
        return None
    boolean_fields = (
        "match",
        "required_action_visible",
        "subject_clearly_visible",
    )
    if any(not isinstance(payload.get(field), bool) for field in boolean_fields):
        return None

    raw_scores = payload.get("scores")
    raw_flags = payload.get("quality_flags")
    if not isinstance(raw_scores, dict) or not isinstance(raw_flags, dict):
        return None

    scores: dict[str, float] = {}
    for field in _CANDIDATE_SCORE_WEIGHTS:
        score = _score_component(raw_scores.get(field))
        if score is None:
            return None
        scores[field] = round(score, 4)

    flag_names = tuple(
        _CANDIDATE_RESPONSE_SCHEMA["properties"]["quality_flags"]["required"]
    )
    if any(not isinstance(raw_flags.get(field), bool) for field in flag_names):
        return None
    quality_flags = {field: raw_flags[field] for field in flag_names}
    overall_score = round(
        sum(
            scores[field] * weight for field, weight in _CANDIDATE_SCORE_WEIGHTS.items()
        ),
        4,
    )
    result = {
        "provider": "twelvelabs",
        "model": DEFAULT_PEGASUS_MODEL,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "match": payload["match"],
        "required_action_visible": payload["required_action_visible"],
        "subject_clearly_visible": payload["subject_clearly_visible"],
        "scores": scores,
        "overall_score": overall_score,
        "quality_flags": quality_flags,
        "visible_summary": _clean_text(payload.get("visible_summary"), 500),
        "reason": _clean_text(payload.get("reason"), 500),
    }
    result["accepted"] = _candidate_is_accepted(result, minimum_score)
    return result


def _failed_candidate_evaluation(
    reason: str,
    *,
    api_call: bool,
) -> dict[str, Any]:
    return {
        "provider": "twelvelabs",
        "model": DEFAULT_PEGASUS_MODEL,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "accepted": False,
        "match": False,
        "required_action_visible": False,
        "subject_clearly_visible": False,
        "scores": {field: 0.0 for field in _CANDIDATE_SCORE_WEIGHTS},
        "overall_score": 0.0,
        "quality_flags": {
            field: False
            for field in _CANDIDATE_RESPONSE_SCHEMA["properties"]["quality_flags"][
                "required"
            ]
        },
        "visible_summary": "",
        "reason": _clean_text(reason, 500),
        "_cache_hit": False,
        "_api_call": api_call,
    }


def _wait_for_asset_ready(client, asset, timeout_seconds: float = 180.0):
    asset_id = str(getattr(asset, "id", "") or "")
    if not asset_id:
        raise RuntimeError("TwelveLabs temporary asset has no ID")
    deadline = time.monotonic() + timeout_seconds
    current = asset
    while str(getattr(current, "status", "") or "") == "processing":
        if time.monotonic() >= deadline:
            raise TimeoutError("TwelveLabs temporary asset processing timed out")
        time.sleep(2.0)
        current = client.assets.retrieve(asset_id)
    if str(getattr(current, "status", "") or "") != "ready":
        raise RuntimeError("TwelveLabs temporary asset processing failed")
    return current


def _delete_temporary_asset(client, asset_id: str | None) -> None:
    if not asset_id:
        return
    try:
        client.assets.delete(asset_id, force=True)
    except Exception as exc:
        logger.warning(
            "failed to delete temporary TwelveLabs asset: "
            f"asset_id={asset_id}, error={type(exc).__name__}"
        )


def _sync_candidate_analysis(
    video_url: str,
    prompt: str,
) -> tuple[Any, str]:
    from twelvelabs.types import (
        AnalyzePromptV2,
        SyncResponseFormat,
        VideoContext_AssetId,
        VideoContext_Url,
    )

    client = _client()

    def analyze(video_context):
        return client.analyze(
            model_name=DEFAULT_PEGASUS_MODEL,
            video=video_context,
            prompt_v_2=AnalyzePromptV2(input_text=prompt),
            response_format=SyncResponseFormat(
                type="json_schema",
                json_schema=_CANDIDATE_RESPONSE_SCHEMA,
            ),
            max_tokens=_PEGASUS_MIN_MAX_TOKENS,
        )

    try:
        response = analyze(VideoContext_Url(url=video_url))
        return response, "direct_url"
    except Exception as direct_error:
        if not _is_direct_url_failure(direct_error):
            raise

    # A temporary Asset is only created after a concrete 400/422 direct-URL
    # failure. It is removed after the fallback analysis completes.
    asset_id: str | None = None
    try:
        asset = client.assets.create(method="url", url=video_url)
        asset_id = str(getattr(asset, "id", "") or "") or None
        asset = _wait_for_asset_ready(client, asset)
        asset_id = str(getattr(asset, "id", "") or asset_id or "") or None
        if not asset_id:
            raise RuntimeError("TwelveLabs temporary asset has no ID")
        response = analyze(VideoContext_AssetId(asset_id=asset_id))
        return response, "asset_fallback"
    finally:
        _delete_temporary_asset(client, asset_id)


def evaluate_candidate(
    *,
    asset_id: str,
    video_url: str,
    slot_index: int,
    slot_duration: float,
    narration_text: str,
    search_query: str,
    minimum_score: float | None = None,
) -> dict[str, Any]:
    """Score one candidate with structured Pegasus 1.5 output and disk cache."""
    narration_text = str(narration_text or "").strip()
    search_query = str(search_query or "").strip()
    asset_id = str(asset_id or "").strip()
    video_url = str(video_url or "").strip()
    if not all((asset_id, video_url, narration_text, search_query)):
        return _failed_candidate_evaluation(
            "candidate metadata or narration requirement is missing",
            api_call=False,
        )
    threshold = _clip_qa_min_score(minimum_score)
    cache_lock = _candidate_cache_lock(asset_id, narration_text)
    with cache_lock:
        cached = _load_candidate_evaluation_cache(asset_id, narration_text)
        if cached is not None:
            cached["accepted"] = _candidate_is_accepted(cached, threshold)
            cached["_cache_hit"] = True
            cached["_api_call"] = False
            return cached

        prompt = _candidate_prompt(
            slot_index=slot_index,
            slot_duration=slot_duration,
            narration_text=narration_text,
            search_query=search_query,
        )
        try:
            response, input_method = _sync_candidate_analysis(video_url, prompt)
        except Exception as exc:
            reason = _safe_api_failure_reason(exc)
            logger.warning(
                "TwelveLabs candidate analysis failed: "
                f"slot={slot_index}, asset_id={asset_id}, reason={reason}"
            )
            return _failed_candidate_evaluation(reason, api_call=True)

        if getattr(response, "finish_reason", None) == "length":
            result = None
        else:
            response_data = getattr(response, "data", response)
            result = _parse_candidate_response(response_data, threshold)
        if result is None:
            logger.warning(
                "TwelveLabs candidate analysis returned malformed structured data: "
                f"slot={slot_index}, asset_id={asset_id}"
            )
            return _failed_candidate_evaluation(
                "malformed TwelveLabs structured response",
                api_call=True,
            )
        result["analysis_input"] = input_method
        result["_cache_hit"] = False
        result["_api_call"] = True
        _save_candidate_evaluation_cache(asset_id, narration_text, result)
        return result


def select_best_candidate(
    *,
    candidates: Sequence[MaterialInfo],
    slot_index: int,
    slot_duration: float,
    narration_text: str,
    search_query: str,
    batch_size: int | None = None,
    max_candidates: int | None = None,
    minimum_score: float | None = None,
    strong_early_stop_score: float | None = None,
    concurrency: int | None = None,
) -> tuple[MaterialInfo | None, dict[str, Any]]:
    """Evaluate complete bounded batches and choose the global best candidate."""
    settings = candidate_selection_settings()
    batch_size = min(5, max(1, int(batch_size or settings["batch_size"])))
    max_candidates = min(
        15,
        max(1, int(max_candidates or settings["max_candidates"])),
    )
    concurrency = min(
        batch_size,
        5,
        max(1, int(concurrency or settings["concurrency"])),
    )
    threshold = _clip_qa_min_score(minimum_score)
    early_stop = (
        settings["strong_early_stop_score"]
        if strong_early_stop_score is None
        else min(1.0, max(0.0, float(strong_early_stop_score)))
    )
    bounded_candidates = list(candidates[:max_candidates])
    evaluated: list[tuple[int, MaterialInfo, dict[str, Any]]] = []
    api_candidates_analyzed = 0
    cached_candidates_used = 0
    source_seconds_analyzed = 0.0
    batches_processed = 0

    for batch_start in range(0, len(bounded_candidates), batch_size):
        batch = bounded_candidates[batch_start : batch_start + batch_size]
        batch_number = (batch_start // batch_size) + 1
        batches_processed += 1
        batch_results: list[tuple[int, MaterialInfo, dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=min(concurrency, len(batch))) as executor:
            futures = {}
            for offset, candidate in enumerate(batch):
                source = (
                    candidate.source_info
                    if isinstance(candidate.source_info, dict)
                    else {}
                )
                candidate_index = batch_start + offset
                asset_id = str(source.get("asset_id") or "")
                future = executor.submit(
                    evaluate_candidate,
                    asset_id=asset_id,
                    video_url=candidate.url,
                    slot_index=slot_index,
                    slot_duration=slot_duration,
                    narration_text=narration_text,
                    search_query=search_query,
                    minimum_score=threshold,
                )
                futures[future] = (candidate_index, candidate, asset_id)

            for future in as_completed(futures):
                candidate_index, candidate, asset_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = _failed_candidate_evaluation(
                        _safe_api_failure_reason(exc),
                        api_call=True,
                    )
                if result.get("_cache_hit"):
                    cached_candidates_used += 1
                if result.get("_api_call"):
                    api_candidates_analyzed += 1
                    source_seconds_analyzed += float(candidate.duration)
                logger.info(
                    "TwelveLabs candidate result: "
                    f"slot={slot_index}, asset_id={asset_id or 'unknown'}, "
                    f"duration={candidate.duration}, batch={batch_number}, "
                    f"accepted={bool(result.get('accepted'))}, "
                    f"overall_score={float(result.get('overall_score') or 0):.4f}, "
                    f"reason={_clean_text(result.get('reason'), 240)!r}"
                )
                batch_results.append((candidate_index, candidate, result))

        batch_results.sort(key=lambda item: item[0])
        evaluated.extend(batch_results)
        accepted_so_far = [
            item for item in evaluated if item[2].get("accepted") is True
        ]
        if accepted_so_far:
            best_so_far = max(
                accepted_so_far,
                key=lambda item: (float(item[2]["overall_score"]), -item[0]),
            )
            if float(best_so_far[2]["overall_score"]) >= early_stop:
                break

    accepted = [item for item in evaluated if item[2].get("accepted") is True]
    api_failure_reason = None
    for _, _, result in evaluated:
        reason = str(result.get("reason") or "")
        if reason.startswith(
            (
                "TwelveLabs quota",
                "TwelveLabs authentication",
                "TwelveLabs API error",
                "TwelveLabs API unavailable",
                "malformed TwelveLabs",
            )
        ):
            api_failure_reason = reason
            break
    winner: MaterialInfo | None = None
    winner_result: dict[str, Any] | None = None
    if accepted:
        _, winner, winner_result = max(
            accepted,
            key=lambda item: (float(item[2]["overall_score"]), -item[0]),
        )
        winner.semantic_evaluation = {
            key: value
            for key, value in winner_result.items()
            if not key.startswith("_")
        }
        winner.overall_score = float(winner_result["overall_score"])

    stats = {
        "candidates_considered": len(bounded_candidates),
        "candidates_evaluated": len(evaluated),
        "api_candidates_analyzed": api_candidates_analyzed,
        "cached_candidates_used": cached_candidates_used,
        "source_seconds_analyzed": round(source_seconds_analyzed, 3),
        "batches_processed": batches_processed,
        "early_stopped": len(evaluated) < len(bounded_candidates),
        "winner_evaluation": winner_result,
        "api_failure_reason": api_failure_reason,
    }
    return winner, stats


def _temporal_response_format(narration_text: str):
    from twelvelabs.types import AsyncResponseFormat, SegmentDefinition, SegmentField

    return AsyncResponseFormat(
        type="segment_definitions",
        segment_time_format="seconds",
        segment_definitions=[
            SegmentDefinition(
                id="best_visual_match",
                description=(
                    "Find only continuous video intervals that visibly satisfy this "
                    f"narration requirement: {narration_text.strip()}"
                ),
                fields=[
                    SegmentField(
                        name="match_quality",
                        type="number",
                        description=(
                            "Visual match quality for the narration requirement from "
                            "0.0 to 1.0"
                        ),
                    ),
                    SegmentField(
                        name="action_visible",
                        type="boolean",
                        description=(
                            "Whether the required action or, when no action is stated, "
                            "the required visible scene is actually shown"
                        ),
                    ),
                    SegmentField(
                        name="subject_visible",
                        type="boolean",
                        description="Whether the required subject is clearly visible",
                    ),
                    SegmentField(
                        name="description",
                        type="string",
                        description="Brief visible evidence for this interval",
                    ),
                ],
            )
        ],
    )


def _create_temporal_task(
    client,
    *,
    video_context,
    narration_text: str,
    requested_source_duration: float,
):
    segment_duration = max(2.0, float(requested_source_duration))
    return client.analyze_async.tasks.create(
        video=video_context,
        model_name=DEFAULT_PEGASUS_MODEL,
        analysis_mode="time_based_metadata",
        max_tokens=_PEGASUS_TEMPORAL_MIN_MAX_TOKENS,
        min_segment_duration=segment_duration,
        max_segment_duration=segment_duration,
        response_format=_temporal_response_format(narration_text),
    )


def _wait_for_temporal_task(
    client,
    task_id: str,
    *,
    timeout_seconds: float = 300.0,
):
    deadline = time.monotonic() + timeout_seconds
    while True:
        result = client.analyze_async.tasks.retrieve(task_id)
        status = str(getattr(result, "status", "") or "")
        if status == "ready":
            return result
        if status == "failed":
            raise RuntimeError("TwelveLabs temporal segmentation failed")
        if time.monotonic() >= deadline:
            raise TimeoutError("TwelveLabs temporal segmentation timed out")
        time.sleep(2.0)


def _parse_temporal_segments(
    response: Any,
    *,
    source_duration: float,
    requested_source_duration: float,
) -> dict[str, Any] | None:
    task_result = getattr(response, "result", None)
    if task_result is None or getattr(task_result, "finish_reason", None) == "length":
        return None
    raw_data = getattr(task_result, "data", None)
    if not isinstance(raw_data, str) or not raw_data.strip():
        return None
    try:
        payload = json.loads(raw_data)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    segments = payload.get("best_visual_match") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        return None

    valid_segments = []
    for position, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        try:
            start_time = float(segment.get("start_time"))
            end_time = float(segment.get("end_time"))
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(start_time)
            or not math.isfinite(end_time)
            or end_time <= start_time
        ):
            continue
        metadata = segment.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        match_quality = _score_component(metadata.get("match_quality"))
        if match_quality is None:
            match_quality = 0.0
        action_visible = metadata.get("action_visible") is True
        subject_visible = metadata.get("subject_visible") is True
        valid_segments.append(
            (
                action_visible and subject_visible,
                match_quality,
                -position,
                start_time,
                end_time,
                metadata,
            )
        )
    if not valid_segments:
        return None

    matched, match_quality, _, start_time, end_time, metadata = max(valid_segments)
    if not matched or match_quality < _clip_qa_min_score():
        return None
    source_duration = max(0.0, float(source_duration))
    start_time = min(source_duration, max(0.0, start_time))
    end_time = min(source_duration, max(0.0, end_time))
    if end_time <= start_time:
        return None

    # Normalize the returned interval to the exact amount of source time the
    # renderer needs. Short final slots may request less than the official
    # two-second segmentation minimum, so trim safely around the detected center.
    requested = min(source_duration, max(0.0, float(requested_source_duration)))
    if requested <= 0:
        return None
    center = (start_time + end_time) / 2.0
    normalized_start = max(0.0, center - (requested / 2.0))
    normalized_end = normalized_start + requested
    if normalized_end > source_duration:
        normalized_end = source_duration
        normalized_start = max(0.0, normalized_end - requested)
    if normalized_end <= normalized_start:
        return None
    return {
        "source_start_time": round(normalized_start, 3),
        "source_end_time": round(normalized_end, 3),
        "match_quality": round(match_quality, 4),
        "action_visible": metadata.get("action_visible") is True,
        "subject_visible": metadata.get("subject_visible") is True,
        "description": _clean_text(metadata.get("description"), 500),
    }


def segment_winner(
    *,
    video_url: str,
    narration_text: str,
    slot_duration: float,
    source_duration: float,
    clip_speed: float = 1.0,
) -> dict[str, Any] | None:
    """Run one async time-based segmentation call for the selected winner."""
    from twelvelabs.types import VideoContext_AssetId, VideoContext_Url

    requested_source_duration = float(slot_duration) * float(clip_speed)
    if requested_source_duration <= 0 or float(source_duration) < 4.0:
        return None
    client = _client()
    asset_id: str | None = None
    input_method = "direct_url"
    try:
        try:
            task = _create_temporal_task(
                client,
                video_context=VideoContext_Url(url=video_url),
                narration_text=narration_text,
                requested_source_duration=requested_source_duration,
            )
        except Exception as direct_error:
            if not _is_direct_url_failure(direct_error):
                raise
            asset = client.assets.create(method="url", url=video_url)
            asset_id = str(getattr(asset, "id", "") or "") or None
            asset = _wait_for_asset_ready(client, asset)
            asset_id = str(getattr(asset, "id", "") or asset_id or "") or None
            if not asset_id:
                raise RuntimeError("TwelveLabs temporary asset has no ID")
            input_method = "asset_fallback"
            task = _create_temporal_task(
                client,
                video_context=VideoContext_AssetId(asset_id=asset_id),
                narration_text=narration_text,
                requested_source_duration=requested_source_duration,
            )

        task_id = str(getattr(task, "task_id", "") or "")
        if not task_id:
            raise RuntimeError("TwelveLabs temporal task has no ID")
        response = _wait_for_temporal_task(client, task_id)
        result = _parse_temporal_segments(
            response,
            source_duration=source_duration,
            requested_source_duration=requested_source_duration,
        )
        if result is not None:
            result["analysis_input"] = input_method
        return result
    except Exception as exc:
        logger.warning(
            "TwelveLabs winner segmentation failed: "
            f"reason={_safe_api_failure_reason(exc)}"
        )
        return None
    finally:
        _delete_temporary_asset(client, asset_id)


def analyze_clip(
    video_url: str,
    prompt: str = "Describe what happens in this video in one sentence.",
    model: Optional[str] = None,
    max_tokens: int = _PEGASUS_MIN_MAX_TOKENS,
) -> Optional[str]:
    """
    QA / describe a clip from a public URL with Pegasus, returning the model's
    text answer (or None when disabled / on failure).

    Notes (TwelveLabs API constraints):
      * Pegasus needs a publicly reachable URL (or an uploaded asset), not a
        bare local path; the analyzed window must be >= 4s.
      * max_tokens must be >= 512 for this model.
    """
    if not is_enabled() or not video_url:
        return None
    # Smart material selection is intentionally locked to Pegasus 1.5. Allowing
    # a stale config value to select 1.2 would silently disable V1.3 contracts.
    model = DEFAULT_PEGASUS_MODEL
    try:
        from twelvelabs.types import AnalyzePromptV2, VideoContext_Url

        client = _client()
        resp = client.analyze(
            model_name=model,
            video=VideoContext_Url(url=video_url),
            prompt_v_2=AnalyzePromptV2(input_text=prompt),
            max_tokens=max(max_tokens, _PEGASUS_MIN_MAX_TOKENS),
        )
        return resp.data
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"TwelveLabs analyze_clip failed: reason={_safe_api_failure_reason(e)}"
        )
        return None


def _parse_clip_qa_response(
    response: str,
    min_score: float,
) -> Optional[dict[str, Any]]:
    """Parse and validate the deliberately tiny Pegasus QA JSON contract."""
    if not isinstance(response, str) or not response.strip():
        return None

    value = response.strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)

    try:
        payload = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        match = re.search(r"\{.*?\}", value, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group())
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    if not isinstance(payload, dict) or not isinstance(payload.get("match"), bool):
        return None
    try:
        score = min(1.0, max(0.0, float(payload.get("score"))))
    except (TypeError, ValueError):
        return None

    reason = str(payload.get("reason") or "").strip()
    reason = re.sub(r"[\x00-\x1f\x7f]+", " ", reason)[:240].strip()
    accepted = payload["match"] and score >= min_score
    return {
        "provider": "twelvelabs",
        "accepted": accepted,
        "score": round(score, 4),
        "reason": reason,
    }


def evaluate_clip_match(
    video_url: str,
    visual_query: str,
    min_score: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Judge whether a public stock clip visibly matches one timeline query.

    ``None`` means the optional QA service could not produce a trustworthy
    result. The caller decides whether that is allowed through
    ``twelvelabs_clip_qa_fail_closed``.
    """
    if (
        not is_clip_qa_enabled()
        or not isinstance(video_url, str)
        or not video_url.strip()
        or not isinstance(visual_query, str)
        or not visual_query.strip()
    ):
        return None

    threshold = _clip_qa_min_score(min_score)
    prompt = f"""
You are a strict stock-footage quality gate.
Decide whether the visible content of this video clearly shows this requested shot:
{visual_query.strip()}

Reject clips that are merely loosely related, dominated by text/logos, visually
sideways or upside down, or do not clearly show the requested subject/action.
Return ONLY one minified JSON object with exactly these keys:
{{"match":true,"score":0.0,"reason":"short visual evidence"}}
""".strip()
    response = analyze_clip(video_url=video_url.strip(), prompt=prompt)
    result = _parse_clip_qa_response(response, threshold)
    if result is None:
        logger.warning(
            f"TwelveLabs clip QA returned an unusable result: query={visual_query!r}"
        )
        return None

    logger.info(
        "TwelveLabs clip QA: "
        f"accepted={result['accepted']}, score={result['score']}, "
        f"query={visual_query!r}, reason={result['reason']!r}"
    )
    return result
