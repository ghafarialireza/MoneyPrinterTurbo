"""Pinterest video-pin discovery for smart visual matching.

Pinterest is the only searchable provider in this pipeline that needs no
credential. The pin search its own web client performs is a public JSON
resource, so there is no API key to configure and no key rotation to do — which
is why ``_STOCK_VIDEO_PROVIDER_API_KEYS`` has to tolerate a provider with no
config key at all.

That convenience comes with three properties the licensed stock APIs do not
have, and every design choice below follows from one of them.

**It is not a documented API.** The endpoint answers because a browser would
call it, so a rate limit, a bot check or a layout change shows up as an HTTP
status or a missing field rather than as a documented error object. Reading any
of those as "the catalog has nothing for this concept" is the bug class this
repository has already fixed twice (a provider quota error reported as a
grounding failure, and an API failure read as unrelated footage). So this module
raises :class:`PinterestSearchError` for every transport-level and protocol-level
problem and returns an empty list only when Pinterest genuinely answered with no
video pins.

**Its media URLs expire.** Pin video files live on a CDN that hands out
short-lived links, so a search result is perishable in a way a Pexels link is
not. This module therefore reports the URL it was given and nothing more; the
caller is responsible for keeping the search cache short enough that an expired
link is never replayed.

**Its results are user-pinned reposts.** Duration, resolution and orientation
are whatever the original uploader produced, so the quality gates matter more
here than anywhere else. Renditions are handed back in the same
``{"id", "link", "width", "height"}`` shape Pexels uses precisely so that
``_select_best_video_rendition`` — the existing short-edge and orientation
ranking — is the one that judges them, instead of a second, more forgiving
copy of that logic growing here.

The reference implementation this was learned from (``pinterest_video_finder``)
fetches full pin details for every result before using it. That is right for a
tool a human drives one query at a time, and wrong here: smart matching runs a
search per phrasing, per provider, per beat, so an extra request per result
would multiply into hundreds of round trips for one video. Detail lookups are
kept as a bounded rescue for the case where a search page carries no playable
rendition at all, which is the only case where they can change the outcome.
"""

from __future__ import annotations

import json
from typing import Any, Iterable
from urllib.parse import urlencode, urlsplit

import requests
from loguru import logger

_BASE_URL = "https://www.pinterest.com"
_SEARCH_RESOURCE_URL = f"{_BASE_URL}/resource/BaseSearchResource/get/"
_PIN_RESOURCE_URL = f"{_BASE_URL}/resource/PinResource/get/"
# The resource endpoints answer a browser, so they are called the way a browser
# calls them. The PWS handler header is what selects the server-side route.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_SEARCH_PWS_HANDLER = "www/[username]/search/pins.js"
_PIN_PWS_HANDLER = "www/[username]/pin.js"
# Pinterest ignores larger page sizes and answers with its own default, so
# asking for more than this silently wastes the request.
_MAX_PAGE_SIZE = 25
# Only consulted when a whole search page produced no playable rendition. Eight
# lookups cost about as much as a single extra search page and are the
# difference between "this provider works" and "this provider always returns
# nothing" if search results ever stop carrying the video list inline.
_MAX_DETAIL_LOOKUPS_PER_PAGE = 8


class PinterestSearchError(RuntimeError):
    """Pinterest could not answer the search, as opposed to answering with nothing.

    Raised for connection failures, rate limits, bot checks, non-JSON bodies and
    unsuccessful resource statuses. The distinction matters: an empty result may
    legitimately end a beat's search on this provider, while a failure must
    never be allowed to look like evidence that a concept does not exist.
    """


def _resource_params(options: dict[str, Any], source_url: str) -> dict[str, str]:
    """Build the query string the Pinterest resource endpoints expect."""
    return {
        "source_url": source_url,
        "data": json.dumps({"options": options, "context": {}}),
    }


def _looks_like_bot_check(response: requests.Response) -> bool:
    """Report whether an HTML body was returned where JSON was expected.

    Pinterest answers an unauthenticated resource call it does not trust with a
    challenge or a login wall rendered as HTML. Detecting that by content type
    keeps the failure honest instead of letting a JSON decode error be reported
    as a malformed catalog response.
    """
    headers = getattr(response, "headers", {}) or {}
    content_type = str(headers.get("content-type", "")).lower()
    return "json" not in content_type and "text/html" in content_type


def _call_resource(
    url: str,
    *,
    options: dict[str, Any],
    source_url: str,
    handler: str,
    session: requests.Session | None,
    timeout: tuple[int, int],
    proxies: Any,
    verify: Any,
) -> dict[str, Any]:
    """Call one Pinterest resource endpoint and return its ``data`` payload."""
    headers = {
        "User-Agent": _BROWSER_USER_AGENT,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "X-Pinterest-PWS-Handler": handler,
        "X-Requested-With": "XMLHttpRequest",
    }
    getter = session.get if session is not None else requests.get
    try:
        response = getter(
            url,
            params=_resource_params(options, source_url),
            headers=headers,
            proxies=proxies,
            verify=verify,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise PinterestSearchError(
            f"pinterest request failed: error={type(exc).__name__}"
        ) from exc

    status_code = int(getattr(response, "status_code", 200))
    response_headers = getattr(response, "headers", {}) or {}
    if status_code == 429:
        retry_after = response_headers.get("retry-after")
        raise PinterestSearchError(
            f"pinterest rate limit exceeded: status=429, "
            f"retry_after={retry_after or 'unknown'}"
        )
    if status_code in (401, 403):
        raise PinterestSearchError(
            "pinterest refused the request, which usually means the endpoint "
            f"requires a session from this network: status={status_code}"
        )
    if status_code >= 400:
        raise PinterestSearchError(f"pinterest request failed: status={status_code}")
    if _looks_like_bot_check(response):
        raise PinterestSearchError(
            "pinterest answered with HTML instead of JSON, which usually means a "
            f"bot check or a login wall: status={status_code}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        content_type = str(response_headers.get("content-type", "")) or "unknown"
        raise PinterestSearchError(
            "pinterest returned a body that is not JSON: "
            f"status={status_code}, content_type={content_type}"
        ) from exc

    if not isinstance(payload, dict):
        raise PinterestSearchError("pinterest returned an unsupported payload shape")

    resource_response = payload.get("resource_response")
    if not isinstance(resource_response, dict):
        raise PinterestSearchError("pinterest response carried no resource_response")
    if str(resource_response.get("status", "")).lower() != "success":
        message = resource_response.get("message")
        if not message:
            error = resource_response.get("error")
            message = error.get("message") if isinstance(error, dict) else None
        raise PinterestSearchError(
            f"pinterest resource call was unsuccessful: detail={message or 'unknown'}"
        )
    return resource_response


def _is_mp4_url(value: Any) -> str | None:
    """Return the URL only when it points at a progressive MP4 file.

    HLS variants are dropped on purpose. The shared downloader is a plain HTTP
    GET, so a playlist would be written to disk as an ``.mp4`` containing text
    and would only be caught later, when decoding fails and the beat has already
    spent its analysis budget. Rejecting it here costs one candidate; accepting
    it costs a beat.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    if not parsed.path.lower().endswith(".mp4"):
        return None
    return candidate


_MIN_PLAUSIBLE_DURATION_SECONDS = 1.0
_MAX_PLAUSIBLE_DURATION_SECONDS = 3600.0


def _duration_seconds(value: Any) -> float | None:
    """Convert a Pinterest rendition duration in milliseconds into seconds.

    The unit is milliseconds, and it is read as milliseconds unconditionally
    rather than guessed at from magnitude. A magnitude heuristic looks harmless
    until a genuine 900 ms clip is read as 900 seconds, and over-reporting a
    source duration is the one error this timeline cannot absorb: the beat would
    pass ``required_source_duration_for_timeline``, win selection, and only then
    turn out to be too short to cover its own window. Under-reporting merely
    loses a candidate to the next provider.

    ``None`` is returned for anything unusable, deliberately in preference to
    zero, so that "unknown" and "known to be very short" stay distinguishable.
    If Pinterest ever changed the unit, every pin would normalize below the
    plausibility floor and the search log would report ``usable=0`` rather than
    quietly selecting clips against invented lengths.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    seconds = numeric / 1000.0
    if (
        seconds < _MIN_PLAUSIBLE_DURATION_SECONDS
        or seconds > _MAX_PLAUSIBLE_DURATION_SECONDS
    ):
        return None
    return seconds


def _video_lists(pin: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield every ``video_list`` mapping a pin can carry.

    A plain video pin exposes one under ``videos``; an idea pin spreads its
    clips across ``story_pin_data`` pages and blocks. Both shapes appear in
    search results, so both are read.
    """
    videos = pin.get("videos")
    if isinstance(videos, dict):
        video_list = videos.get("video_list") or videos.get("videoUrls")
        if isinstance(video_list, dict):
            yield video_list

    story = pin.get("story_pin_data")
    if not isinstance(story, dict):
        return
    pages = story.get("pages")
    if not isinstance(pages, list):
        return
    for page in pages:
        if not isinstance(page, dict):
            continue
        blocks = page.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            video = block.get("video")
            if not isinstance(video, dict):
                continue
            video_list = video.get("video_list")
            if isinstance(video_list, dict):
                yield video_list


def _renditions_and_duration(
    pin: dict[str, Any],
) -> tuple[list[dict[str, Any]], float | None]:
    """Collect playable renditions in the shape the shared ranking expects.

    ``link`` rather than ``url`` is not an accident: it is the key
    ``_select_best_video_rendition`` reads, so Pinterest candidates are ranked
    for orientation and short edge by exactly the same code that ranks Pexels
    candidates, and a future change to those gates cannot apply to one provider
    and miss the other.
    """
    renditions: list[dict[str, Any]] = []
    seen_links: set[str] = set()
    longest: float | None = None

    for video_list in _video_lists(pin):
        for rendition_id, rendition in video_list.items():
            if not isinstance(rendition, dict):
                continue
            duration = _duration_seconds(rendition.get("duration"))
            if duration is not None and (longest is None or duration > longest):
                longest = duration
            link = _is_mp4_url(rendition.get("url"))
            if not link or link in seen_links:
                continue
            try:
                width = int(rendition.get("width") or 0)
                height = int(rendition.get("height") or 0)
            except (TypeError, ValueError):
                continue
            if width <= 0 or height <= 0:
                continue
            seen_links.add(link)
            renditions.append(
                {
                    "id": str(rendition_id),
                    "link": link,
                    "width": width,
                    "height": height,
                }
            )
    return renditions, longest


def _poster_url(pin: dict[str, Any]) -> str | None:
    for video_list in _video_lists(pin):
        for rendition in video_list.values():
            if isinstance(rendition, dict):
                thumbnail = rendition.get("thumbnail")
                if isinstance(thumbnail, str) and thumbnail.strip():
                    return thumbnail.strip()
    images = pin.get("images")
    if isinstance(images, dict):
        for size_key in ("orig", "736x", "474x", "236x"):
            entry = images.get(size_key)
            if isinstance(entry, dict):
                url = entry.get("url")
                if isinstance(url, str) and url.strip():
                    return url.strip()
    return None


def _creator(pin: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("pinner", "native_creator", "closeup_attribution"):
        candidate = pin.get(key)
        if isinstance(candidate, dict) and (
            candidate.get("username") or candidate.get("full_name")
        ):
            username = candidate.get("username")
            return {
                "id": candidate.get("id"),
                "name": candidate.get("full_name") or username,
                "profile_url": f"{_BASE_URL}/{username}/" if username else None,
            }
    return None


def normalize_video_pin(pin: Any) -> dict[str, Any] | None:
    """Reduce one raw pin to the fields material selection needs, or ``None``.

    ``None`` means the pin cannot be used at all — no id, or no progressive MP4
    with usable dimensions. Discarding it here keeps a pin that could never be
    downloaded out of the candidate count, so the analysis budget is spent on
    clips that could actually fill a beat.
    """
    if not isinstance(pin, dict):
        return None
    pin_id = str(pin.get("id") or "").strip()
    if not pin_id:
        return None
    renditions, duration = _renditions_and_duration(pin)
    if not renditions:
        return None
    return {
        "pin_id": pin_id,
        "pin_url": f"{_BASE_URL}/pin/{pin_id}/",
        "renditions": renditions,
        "duration": duration,
        "poster": _poster_url(pin),
        "creator": _creator(pin),
        "title": str(pin.get("grid_title") or pin.get("title") or "").strip() or None,
        "description": str(pin.get("description") or "").strip() or None,
    }


def _search_page(
    query: str,
    *,
    page_size: int,
    bookmark: str | None,
    session: requests.Session | None,
    timeout: tuple[int, int],
    proxies: Any,
    verify: Any,
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch one page of video pins and the bookmark that follows it."""
    options: dict[str, Any] = {
        "query": query,
        "scope": "videos",
        "page_size": max(1, min(_MAX_PAGE_SIZE, int(page_size))),
        "bookmarks": [bookmark] if bookmark else [],
    }
    resource_response = _call_resource(
        _SEARCH_RESOURCE_URL,
        options=options,
        source_url=f"/search/videos/?{urlencode({'q': query})}",
        handler=_SEARCH_PWS_HANDLER,
        session=session,
        timeout=timeout,
        proxies=proxies,
        verify=verify,
    )
    data = resource_response.get("data")
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        results = []
    bookmark_value = resource_response.get("bookmark")
    next_bookmark = None
    if isinstance(bookmark_value, str) and bookmark_value.strip():
        next_bookmark = bookmark_value.strip()
    return [pin for pin in results if isinstance(pin, dict)], next_bookmark


def _pin_detail(
    pin_id: str,
    *,
    session: requests.Session | None,
    timeout: tuple[int, int],
    proxies: Any,
    verify: Any,
) -> dict[str, Any] | None:
    """Fetch one pin's detailed record, or ``None`` when it cannot be read.

    A failure here is deliberately swallowed rather than raised. This is the
    bounded rescue path: the search itself already succeeded, so one unreadable
    pin should cost that pin and nothing else.
    """
    try:
        resource_response = _call_resource(
            _PIN_RESOURCE_URL,
            options={"id": pin_id, "field_set_key": "detailed"},
            source_url=f"/pin/{pin_id}/",
            handler=_PIN_PWS_HANDLER,
            session=session,
            timeout=timeout,
            proxies=proxies,
            verify=verify,
        )
    except PinterestSearchError as exc:
        logger.debug(f"pinterest pin detail unavailable: pin={pin_id}, detail={exc}")
        return None
    data = resource_response.get("data")
    return data if isinstance(data, dict) else None


def search_video_pins(
    query: str,
    *,
    limit: int = 50,
    max_pages: int = 3,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (30, 60),
    proxies: Any = None,
    verify: Any = True,
) -> list[dict[str, Any]]:
    """Search Pinterest for video pins and return normalized records.

    Raises :class:`PinterestSearchError` when Pinterest could not answer, and
    returns an empty list when it answered with nothing usable. Pages are
    followed only while they are still adding candidates, so a query with few
    results costs one request rather than ``max_pages``.
    """
    query = str(query or "").strip()
    if not query:
        return []
    limit = max(1, int(limit))
    max_pages = max(1, int(max_pages))

    records: list[dict[str, Any]] = []
    seen_pin_ids: set[str] = set()
    bookmark: str | None = None
    pins_seen = 0

    for page_index in range(max_pages):
        pins, bookmark = _search_page(
            query,
            # Always ask for a full page. Only some pins carry a progressive MP4
            # at a usable size, so requesting exactly the shortfall would
            # under-fill every page and spend an extra round trip to recover.
            page_size=_MAX_PAGE_SIZE,
            bookmark=bookmark,
            session=session,
            timeout=timeout,
            proxies=proxies,
            verify=verify,
        )
        if not pins:
            break
        pins_seen += len(pins)
        page_records = 0
        unusable: list[str] = []
        for pin in pins:
            pin_id = str(pin.get("id") or "").strip()
            if not pin_id or pin_id in seen_pin_ids:
                continue
            seen_pin_ids.add(pin_id)
            record = normalize_video_pin(pin)
            if record is None:
                unusable.append(pin_id)
                continue
            records.append(record)
            page_records += 1
            if len(records) >= limit:
                break

        if page_records == 0 and unusable:
            # The page carried pins but none of them exposed a playable file.
            # This is the only situation where a detail lookup can change the
            # outcome, so it is also the only situation that pays for one.
            for pin_id in unusable[:_MAX_DETAIL_LOOKUPS_PER_PAGE]:
                detail = _pin_detail(
                    pin_id,
                    session=session,
                    timeout=timeout,
                    proxies=proxies,
                    verify=verify,
                )
                record = normalize_video_pin(detail) if detail else None
                if record is not None:
                    records.append(record)
                    if len(records) >= limit:
                        break
            logger.debug(
                "pinterest search page had no inline video renditions, used "
                f"detail lookups: query={query!r}, page={page_index + 1}, "
                f"pins={len(unusable)}, recovered={len(records)}"
            )

        if len(records) >= limit or not bookmark:
            break

    logger.info(
        f"pinterest video search finished: term={query!r}, "
        f"pins_seen={pins_seen}, usable={len(records)}"
    )
    return records[:limit]
