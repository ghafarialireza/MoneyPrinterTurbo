"""Unit tests for the keyless Pinterest video search provider.

Pinterest is read through an undocumented public web resource, so these tests
pin down the two things that cannot be verified by running the pipeline: that a
failure is reported as the failure it is rather than as an absent concept, and
that a pin is never described as longer, cleaner or more playable than it is.
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import pinterest


def _response(
    payload=None,
    *,
    status_code=200,
    content_type="application/json",
    raw_body=None,
):
    """Build the minimal response surface ``_call_resource`` reads."""

    def _json():
        if raw_body is not None:
            raise ValueError("no json body")
        return payload

    return SimpleNamespace(
        status_code=status_code,
        headers={"content-type": content_type},
        json=_json,
        text=raw_body or "",
    )


def _success(data, bookmark=None):
    resource_response = {"status": "success", "data": data}
    if bookmark is not None:
        resource_response["bookmark"] = bookmark
    return _response({"resource_response": resource_response})


def _video_pin(pin_id, *, url=None, width=1080, height=1920, duration_ms=8000):
    """A plain video pin in the shape the search resource returns."""
    return {
        "id": pin_id,
        "grid_title": f"pin {pin_id}",
        "videos": {
            "video_list": {
                "V_720P": {
                    "url": url or f"https://v.pinimg.com/videos/{pin_id}/720p.mp4",
                    "width": width,
                    "height": height,
                    "duration": duration_ms,
                    "thumbnail": f"https://i.pinimg.com/{pin_id}.jpg",
                }
            }
        },
    }


class TestPinterestDurationReading(unittest.TestCase):
    """A pin's duration is milliseconds and is never inferred from magnitude."""

    def test_milliseconds_are_converted_to_seconds(self):
        self.assertEqual(pinterest._duration_seconds(8000), 8.0)
        self.assertEqual(pinterest._duration_seconds("15500"), 15.5)
        self.assertEqual(pinterest._duration_seconds(1000), 1.0)

    def test_a_sub_second_clip_is_unknown_rather_than_hundreds_of_seconds(self):
        # 900 is the case a magnitude heuristic gets wrong: read as "already
        # seconds" it becomes a 900 second source, which would pass the timeline
        # duration check, win selection, and then fail to cover its own window.
        self.assertIsNone(pinterest._duration_seconds(900))
        self.assertIsNone(pinterest._duration_seconds(0))
        self.assertIsNone(pinterest._duration_seconds(-8000))

    def test_an_implausible_length_is_refused(self):
        self.assertIsNone(pinterest._duration_seconds(3600001))

    def test_unusable_values_are_unknown_not_zero(self):
        for value in (None, True, False, "", "abc", [], {}):
            self.assertIsNone(pinterest._duration_seconds(value))


class TestPinterestRenditionUrls(unittest.TestCase):
    """Only a progressive MP4 can be handed to the shared downloader."""

    def test_a_progressive_mp4_is_accepted(self):
        self.assertEqual(
            pinterest._is_mp4_url("https://v.pinimg.com/videos/x/720p.mp4"),
            "https://v.pinimg.com/videos/x/720p.mp4",
        )

    def test_a_protocol_relative_url_is_upgraded_to_https(self):
        self.assertEqual(
            pinterest._is_mp4_url("//v.pinimg.com/videos/x/720p.mp4"),
            "https://v.pinimg.com/videos/x/720p.mp4",
        )

    def test_an_hls_playlist_is_refused(self):
        # ``save_video`` is a plain GET. A playlist would be written to disk as
        # an .mp4 full of text and only fail at decode time, after the beat had
        # already paid for its analysis.
        self.assertIsNone(
            pinterest._is_mp4_url("https://v.pinimg.com/videos/x/hls/master.m3u8")
        )

    def test_insecure_and_empty_urls_are_refused(self):
        for value in (
            "http://v.pinimg.com/videos/x/720p.mp4",
            "",
            "   ",
            None,
            42,
            "https:///720p.mp4",
        ):
            self.assertIsNone(pinterest._is_mp4_url(value))


class TestPinterestPinNormalization(unittest.TestCase):
    """A normalized record carries only what material selection can act on."""

    def test_a_plain_video_pin_becomes_one_record(self):
        record = pinterest.normalize_video_pin(_video_pin("123"))

        self.assertIsNotNone(record)
        self.assertEqual(record["pin_id"], "123")
        self.assertEqual(record["pin_url"], "https://www.pinterest.com/pin/123/")
        self.assertEqual(record["duration"], 8.0)
        self.assertEqual(len(record["renditions"]), 1)
        rendition = record["renditions"][0]
        # ``link``/``width``/``height`` are the keys the shared Pexels ranking
        # reads. Renaming them here would silently bypass the resolution and
        # orientation gates for this provider only.
        self.assertEqual(sorted(rendition), ["height", "id", "link", "width"])
        self.assertEqual(rendition["link"], "https://v.pinimg.com/videos/123/720p.mp4")
        self.assertEqual(rendition["width"], 1080)
        self.assertEqual(rendition["height"], 1920)

    def test_an_idea_pin_is_read_from_its_story_pages(self):
        pin = {
            "id": "555",
            "story_pin_data": {
                "pages": [
                    {"blocks": [{"nope": True}]},
                    {
                        "blocks": [
                            {
                                "video": {
                                    "video_list": {
                                        "V_HLSV4": {
                                            "url": "https://v.pinimg.com/a/master.m3u8",
                                            "width": 1080,
                                            "height": 1920,
                                            "duration": 12000,
                                        },
                                        "V_720P": {
                                            "url": "https://v.pinimg.com/a/720p.mp4",
                                            "width": 720,
                                            "height": 1280,
                                            "duration": 12000,
                                        },
                                    }
                                }
                            }
                        ]
                    },
                ]
            },
        }

        record = pinterest.normalize_video_pin(pin)

        self.assertIsNotNone(record)
        self.assertEqual(record["duration"], 12.0)
        self.assertEqual(
            [item["link"] for item in record["renditions"]],
            ["https://v.pinimg.com/a/720p.mp4"],
        )

    def test_a_pin_with_no_playable_file_is_dropped(self):
        # Dropping it here keeps a candidate that could never be downloaded out
        # of the analysis budget.
        hls_only = _video_pin("777", url="https://v.pinimg.com/a/master.m3u8")
        self.assertIsNone(pinterest.normalize_video_pin(hls_only))

        no_dimensions = _video_pin("778", width=0, height=0)
        self.assertIsNone(pinterest.normalize_video_pin(no_dimensions))

        self.assertIsNone(pinterest.normalize_video_pin({"id": "", "videos": {}}))
        self.assertIsNone(pinterest.normalize_video_pin(None))
        self.assertIsNone(pinterest.normalize_video_pin("not a pin"))

    def test_a_pin_without_a_readable_duration_reports_it_as_unknown(self):
        record = pinterest.normalize_video_pin(_video_pin("900", duration_ms=None))

        self.assertIsNotNone(record)
        self.assertIsNone(record["duration"])


class TestPinterestFailureClassification(unittest.TestCase):
    """Every outage has to name itself, so it is never read as "no footage"."""

    def _call(self, response):
        with patch.object(pinterest.requests, "get", return_value=response):
            return pinterest.search_video_pins("sprouting seedling")

    def test_a_rate_limit_names_itself_and_its_retry_hint(self):
        response = _response({"resource_response": {"status": "success"}})
        response.status_code = 429
        response.headers = {"content-type": "application/json", "retry-after": "30"}

        with self.assertRaises(pinterest.PinterestSearchError) as ctx:
            self._call(response)

        self.assertIn("rate limit", str(ctx.exception))
        self.assertIn("30", str(ctx.exception))

    def test_a_refusal_is_reported_as_a_refusal(self):
        for status in (401, 403):
            with self.assertRaises(pinterest.PinterestSearchError) as ctx:
                self._call(_response({}, status_code=status))
            self.assertIn(str(status), str(ctx.exception))

    def test_a_server_error_is_reported_with_its_status(self):
        with self.assertRaises(pinterest.PinterestSearchError) as ctx:
            self._call(_response({}, status_code=503))

        self.assertIn("503", str(ctx.exception))

    def test_an_html_body_is_reported_as_a_bot_check(self):
        response = _response(
            None, content_type="text/html; charset=utf-8", raw_body="<html>"
        )

        with self.assertRaises(pinterest.PinterestSearchError) as ctx:
            self._call(response)

        self.assertIn("HTML", str(ctx.exception))

    def test_a_non_json_body_is_reported_as_such(self):
        with self.assertRaises(pinterest.PinterestSearchError) as ctx:
            self._call(_response(None, raw_body="not json"))

        self.assertIn("not JSON", str(ctx.exception))

    def test_a_missing_resource_response_is_reported(self):
        with self.assertRaises(pinterest.PinterestSearchError):
            self._call(_response({"unexpected": True}))

        with self.assertRaises(pinterest.PinterestSearchError):
            self._call(_response(["not", "a", "mapping"]))

    def test_an_unsuccessful_resource_call_surfaces_its_message(self):
        response = _response(
            {
                "resource_response": {
                    "status": "failure",
                    "error": {"message": "unauthenticated"},
                }
            }
        )

        with self.assertRaises(pinterest.PinterestSearchError) as ctx:
            self._call(response)

        self.assertIn("unauthenticated", str(ctx.exception))

    def test_a_transport_error_is_wrapped_without_leaking_the_url(self):
        with patch.object(
            pinterest.requests,
            "get",
            side_effect=requests.ConnectionError("https://secret.example/x"),
        ):
            with self.assertRaises(pinterest.PinterestSearchError) as ctx:
                pinterest.search_video_pins("cat")

        message = str(ctx.exception)
        self.assertIn("ConnectionError", message)
        self.assertNotIn("secret.example", message)


class TestPinterestSearchPaging(unittest.TestCase):
    """Paging spends a request only while it is still adding candidates."""

    def test_the_search_request_asks_for_video_scope_and_a_full_page(self):
        response = _success({"results": [_video_pin("1")]})

        with patch.object(pinterest.requests, "get", return_value=response) as get:
            pinterest.search_video_pins("morning coffee", limit=1)

        self.assertEqual(get.call_count, 1)
        kwargs = get.call_args.kwargs
        self.assertEqual(
            kwargs["headers"]["X-Pinterest-PWS-Handler"],
            "www/[username]/search/pins.js",
        )
        options = json.loads(kwargs["params"]["data"])["options"]
        self.assertEqual(options["query"], "morning coffee")
        self.assertEqual(options["scope"], "videos")
        self.assertEqual(options["page_size"], 25)
        self.assertEqual(options["bookmarks"], [])
        self.assertEqual(
            parse_qs(urlsplit(kwargs["params"]["source_url"]).query)["q"],
            ["morning coffee"],
        )

    def test_proxy_and_tls_settings_are_forwarded_to_every_request(self):
        response = _success({"results": [_video_pin("1")]})
        proxies = {"https": "http://127.0.0.1:7890"}

        with patch.object(pinterest.requests, "get", return_value=response) as get:
            pinterest.search_video_pins(
                "cat", limit=1, proxies=proxies, verify=False
            )

        self.assertEqual(get.call_args.kwargs["proxies"], proxies)
        self.assertFalse(get.call_args.kwargs["verify"])

    def test_the_next_page_is_requested_with_the_returned_bookmark(self):
        pages = [
            _success({"results": [_video_pin("1")]}, bookmark="bm-1"),
            _success({"results": [_video_pin("2")]}, bookmark=None),
        ]

        with patch.object(pinterest.requests, "get", side_effect=pages) as get:
            records = pinterest.search_video_pins("cat", limit=10, max_pages=3)

        self.assertEqual([record["pin_id"] for record in records], ["1", "2"])
        self.assertEqual(get.call_count, 2)
        second_options = json.loads(get.call_args_list[1].kwargs["params"]["data"])
        self.assertEqual(second_options["options"]["bookmarks"], ["bm-1"])

    def test_a_repeated_pin_is_counted_once(self):
        pages = [
            _success({"results": [_video_pin("1")]}, bookmark="bm-1"),
            _success({"results": [_video_pin("1"), _video_pin("2")]}, bookmark="bm-2"),
        ]

        with patch.object(pinterest.requests, "get", side_effect=pages):
            records = pinterest.search_video_pins("cat", limit=10, max_pages=2)

        self.assertEqual([record["pin_id"] for record in records], ["1", "2"])

    def test_paging_stops_once_the_limit_is_reached(self):
        pages = [
            _success(
                {"results": [_video_pin("1"), _video_pin("2")]}, bookmark="bm-1"
            ),
            _success({"results": [_video_pin("3")]}, bookmark="bm-2"),
        ]

        with patch.object(pinterest.requests, "get", side_effect=pages) as get:
            records = pinterest.search_video_pins("cat", limit=2, max_pages=3)

        self.assertEqual(len(records), 2)
        self.assertEqual(get.call_count, 1)

    def test_an_empty_page_ends_the_search(self):
        pages = [_success({"results": []}, bookmark="bm-1")]

        with patch.object(pinterest.requests, "get", side_effect=pages) as get:
            records = pinterest.search_video_pins("cat", limit=10, max_pages=3)

        self.assertEqual(records, [])
        self.assertEqual(get.call_count, 1)

    def test_a_blank_query_never_reaches_the_network(self):
        with patch.object(pinterest.requests, "get") as get:
            self.assertEqual(pinterest.search_video_pins("   "), [])

        get.assert_not_called()


class TestPinterestDetailRescue(unittest.TestCase):
    """Per-pin detail lookups are a bounded rescue, not the normal path."""

    def test_no_detail_lookup_happens_when_the_page_was_usable(self):
        response = _success({"results": [_video_pin("1")]})

        with patch.object(pinterest.requests, "get", return_value=response) as get:
            pinterest.search_video_pins("cat", limit=1)

        self.assertEqual(get.call_count, 1)

    def test_a_page_with_no_playable_file_recovers_through_pin_details(self):
        # The reference project enriched every pin unconditionally with a sleep
        # between calls. Selection searches per phrasing, per provider, per beat,
        # so that cost is only worth paying when the page produced nothing.
        hls_only = _video_pin("1", url="https://v.pinimg.com/a/master.m3u8")
        pages = [
            _success({"results": [hls_only]}, bookmark=None),
            _success(_video_pin("1")),
        ]

        with patch.object(pinterest.requests, "get", side_effect=pages) as get:
            records = pinterest.search_video_pins("cat", limit=5, max_pages=1)

        self.assertEqual([record["pin_id"] for record in records], ["1"])
        self.assertEqual(get.call_count, 2)
        self.assertEqual(
            get.call_args_list[1].kwargs["headers"]["X-Pinterest-PWS-Handler"],
            "www/[username]/pin.js",
        )
        detail_options = json.loads(get.call_args_list[1].kwargs["params"]["data"])
        self.assertEqual(detail_options["options"]["id"], "1")
        self.assertEqual(detail_options["options"]["field_set_key"], "detailed")

    def test_the_rescue_is_capped_so_one_bad_page_cannot_fan_out(self):
        unusable = [
            _video_pin(str(index), url="https://v.pinimg.com/a/master.m3u8")
            for index in range(20)
        ]
        responses = [_success({"results": unusable}, bookmark=None)]
        responses.extend(
            _success({"nothing": "usable"})
            for _ in range(pinterest._MAX_DETAIL_LOOKUPS_PER_PAGE)
        )

        with patch.object(pinterest.requests, "get", side_effect=responses) as get:
            records = pinterest.search_video_pins("cat", limit=5, max_pages=1)

        self.assertEqual(records, [])
        self.assertEqual(get.call_count, 1 + pinterest._MAX_DETAIL_LOOKUPS_PER_PAGE)

    def test_a_failed_detail_lookup_costs_that_pin_and_nothing_else(self):
        hls_only = _video_pin("1", url="https://v.pinimg.com/a/master.m3u8")
        pages = [
            _success({"results": [hls_only]}, bookmark=None),
            _response({}, status_code=500),
        ]

        with patch.object(pinterest.requests, "get", side_effect=pages):
            records = pinterest.search_video_pins("cat", limit=5, max_pages=1)

        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
