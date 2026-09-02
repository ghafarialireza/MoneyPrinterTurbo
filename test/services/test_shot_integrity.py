"""Unit tests for the local pixel checks run on a downloaded clip.

These two gates exist because two defects reached a finished video: a beat whose
window played across a cut in the source, and a beat whose footage was itself an
advertisement for someone's channel. Both had passed every paid gate above them,
so the tests here pin down the parts that cannot be verified by watching a run.

The arithmetic tests fix the containment rules, including the one that matters
most: a window is only ever moved into a shot the segmenter actually looked at,
never into some other long shot elsewhere in the file. The end-to-end tests
encode real footage with FFmpeg — a two-shot clip and a clip with a frozen
high-contrast band across its lower third — and assert the verdicts, because a
detector that only ever sees mocked numbers proves nothing about pixels.

Every probe must also fail open. A missing binary or an unreadable file means
the check could not be performed, never that the clip is bad; the opposite would
turn a broken FFmpeg install into a video with no footage in it.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import shot_integrity
from app.utils import utils

_FIXTURE_WIDTH = 240
_FIXTURE_HEIGHT = 426
_FIXTURE_FPS = 15


def _numpy():
    try:
        import numpy
    except ImportError:  # pragma: no cover - numpy ships with MoviePy
        return None
    return numpy


def _ffmpeg_encodes() -> bool:
    """Whether this machine can encode the fixtures these tests need."""
    if _numpy() is None:
        return False
    try:
        completed = subprocess.run(
            [utils.get_ffmpeg_binary(), "-hide_banner", "-version"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _background(index: int, brightness: float, wavelength: float):
    """A frame of moving, softly noisy footage: static in nothing, edges nowhere."""
    numpy = _numpy()
    grid_y, grid_x = numpy.mgrid[0:_FIXTURE_HEIGHT, 0:_FIXTURE_WIDTH]
    frame = (
        brightness
        + 40 * numpy.sin(grid_x / wavelength + index * 0.35)
        + 30 * numpy.cos(grid_y / (wavelength * 1.4) - index * 0.25)
    )
    noise = numpy.random.default_rng(index).normal(
        0, 5, (_FIXTURE_HEIGHT, _FIXTURE_WIDTH)
    )
    return frame + noise


def _overlay_pattern():
    """A frozen blocky pattern across the lower fifth, standing in for burned-in text.

    Text is not being simulated here; what is being simulated is the measurable
    property text has and footage does not — pixels that never change while
    carrying hard edges, packed against one edge of the frame.
    """
    numpy = _numpy()
    mask = numpy.zeros((_FIXTURE_HEIGHT, _FIXTURE_WIDTH), bool)
    values = numpy.zeros((_FIXTURE_HEIGHT, _FIXTURE_WIDTH), numpy.float32)
    pattern = numpy.random.default_rng(11).integers(0, 2, (9, 15))
    block_height, block_width = 9, 16
    first_row = int(0.80 * _FIXTURE_HEIGHT)
    for row in range(pattern.shape[0]):
        for column in range(pattern.shape[1]):
            top = first_row + row * block_height
            left = column * block_width
            if top + block_height > _FIXTURE_HEIGHT:
                continue
            if left + block_width > _FIXTURE_WIDTH:
                continue
            mask[top : top + block_height, left : left + block_width] = True
            values[top : top + block_height, left : left + block_width] = (
                255.0 if pattern[row, column] else 0.0
            )
    return mask, values


def _write_clip(path: str, *, seconds=3.0, branded=False, cut_at=None) -> bool:
    numpy = _numpy()
    mask, values = _overlay_pattern()
    frames = []
    for index in range(int(round(seconds * _FIXTURE_FPS))):
        if cut_at is not None and index / _FIXTURE_FPS >= cut_at:
            frame = _background(index, 200, 11.0)
        else:
            frame = _background(index, 90, 37.0)
        if branded:
            frame[mask] = values[mask]
        frames.append(numpy.clip(frame, 0, 255).astype(numpy.uint8))
    completed = subprocess.run(
        [
            utils.get_ffmpeg_binary(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-s",
            f"{_FIXTURE_WIDTH}x{_FIXTURE_HEIGHT}",
            "-r",
            str(_FIXTURE_FPS),
            "-i",
            "-",
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            path,
        ],
        input=numpy.stack(frames).tobytes(),
        capture_output=True,
        timeout=120,
    )
    return completed.returncode == 0


class ShotCutArithmeticTest(unittest.TestCase):
    """The rules that turn scene detections into a usable window."""

    def test_merge_cuts_drops_the_leading_artifact_and_the_duplicate(self):
        """A detection at t≈0 has no previous shot, and one cut can report twice."""
        self.assertEqual(
            shot_integrity._merge_cuts([0.04, 0.12, 0.4, 0.5, 1.2]),
            [0.4, 1.2],
        )

    def test_shot_bounds_ignores_cuts_against_either_edge(self):
        """Splitting on a decode artifact would invent a shot with no frames in it."""
        self.assertEqual(
            shot_integrity.shot_bounds([0.05, 5.0, 9.95], 10.0),
            [(0.0, 5.0), (5.0, 10.0)],
        )

    def test_a_continuous_source_centres_the_window_on_what_was_verified(self):
        """With no cuts the only job is to keep the described moment in frame."""
        self.assertEqual(
            shot_integrity.window_inside_one_shot(
                [],
                source_duration=10.0,
                verified_start=4.0,
                verified_end=5.0,
                required_duration=3.0,
            ),
            (3.0, 6.0),
        )

    def test_a_window_straddling_a_cut_moves_into_the_described_shot(self):
        """The shot holding the midpoint of the verified interval is the right one.

        The guard keeps the window clear of the internal boundary at 2.5s, but
        not of the end of the file, where there is no transition frame to avoid.
        """
        self.assertEqual(
            shot_integrity.window_inside_one_shot(
                [2.5],
                source_duration=10.0,
                verified_start=2.0,
                verified_end=4.0,
                required_duration=3.0,
            ),
            (2.56, 5.56),
        )

    def test_a_shot_too_short_for_the_slot_is_refused_not_stretched(self):
        """The production defect: 3.39s asked of a shot that ends 2.33s in."""
        self.assertIsNone(
            shot_integrity.window_inside_one_shot(
                [2.33],
                source_duration=15.0,
                verified_start=0.006,
                verified_end=2.0,
                required_duration=3.388,
            )
        )

    def test_an_unrelated_long_shot_is_never_borrowed(self):
        """Reaching past the verified interval would ship footage nothing examined.

        The file has eighteen unbroken seconds after the cut, and they are not
        available: no gate ever looked at them.
        """
        self.assertIsNone(
            shot_integrity.window_inside_one_shot(
                [2.0],
                source_duration=20.0,
                verified_start=0.2,
                verified_end=1.8,
                required_duration=3.0,
            )
        )

    def test_a_nonsensical_required_duration_is_refused(self):
        for required in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(required=required):
                self.assertIsNone(
                    shot_integrity.window_inside_one_shot(
                        [],
                        source_duration=10.0,
                        verified_start=1.0,
                        verified_end=2.0,
                        required_duration=required,
                    )
                )

    def test_an_empty_verified_interval_is_refused(self):
        """Without a described moment there is nothing to anchor the window to."""
        self.assertIsNone(
            shot_integrity.window_inside_one_shot(
                [],
                source_duration=10.0,
                verified_start=4.0,
                verified_end=4.0,
                required_duration=2.0,
            )
        )


class OverlayDecisionTest(unittest.TestCase):
    """The decision rule, held against the numbers measured on real clips."""

    def test_the_branded_pin_is_called_branded(self):
        """The clip that shipped: a channel advertisement across the bottom third."""
        self.assertTrue(
            shot_integrity.overlay_is_burned_in(
                {"zone": "bottom", "density": 0.1494, "middle_density": 0.0252}
            )
        )

    def test_the_busiest_clean_clip_is_left_alone(self):
        """A static bucket rim is the strongest false signal in the whole cache."""
        self.assertFalse(
            shot_integrity.overlay_is_burned_in(
                {"zone": "bottom", "density": 0.0483, "middle_density": 0.0206}
            )
        )

    def test_a_locked_off_shot_is_separated_by_the_middle_zone(self):
        """A near-frozen macro shot is static and edgy everywhere, not just at an edge."""
        self.assertFalse(
            shot_integrity.overlay_is_burned_in(
                {"zone": "bottom", "density": 0.2201, "middle_density": 0.2053}
            )
        )

    def test_a_missing_or_malformed_measurement_accuses_nobody(self):
        for measurement in (None, {}, {"density": "x", "middle_density": 0.0}, []):
            with self.subTest(measurement=measurement):
                self.assertFalse(shot_integrity.overlay_is_burned_in(measurement))


class ProbeFailOpenTest(unittest.TestCase):
    """Whatever goes wrong locally, it must not be read as a bad candidate."""

    def test_a_file_too_small_to_be_video_is_not_probed_at_all(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tiny.mp4")
            with open(path, "wb") as handle:
                handle.write(b"\x00" * 1024)

            with patch.object(shot_integrity.subprocess, "run") as run:
                self.assertIsNone(shot_integrity.detect_shot_cuts(path))
                self.assertIsNone(
                    shot_integrity.measure_burned_in_overlay(
                        path, start=0.0, duration=2.0
                    )
                )

            run.assert_not_called()

    def test_a_missing_file_is_not_probed_at_all(self):
        self.assertIsNone(shot_integrity.detect_shot_cuts("/nonexistent/clip.mp4"))

    def test_a_failing_ffmpeg_reports_nothing_rather_than_no_cuts(self):
        """``None`` and ``[]`` are different answers and must not be confused."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "clip.mp4")
            with open(path, "wb") as handle:
                handle.write(b"\x00" * (64 * 1024))

            with patch.object(
                shot_integrity.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 1, b"", b"boom"),
            ):
                self.assertIsNone(shot_integrity.detect_shot_cuts(path))

            with patch.object(
                shot_integrity.subprocess,
                "run",
                side_effect=OSError("ffmpeg is not installed"),
            ):
                self.assertIsNone(shot_integrity.detect_shot_cuts(path))

            with patch.object(
                shot_integrity.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired([], 1),
            ):
                self.assertIsNone(shot_integrity.detect_shot_cuts(path))

    def test_the_scene_filter_escapes_the_comma_inside_gt(self):
        """Unescaped, FFmpeg reads it as a filter separator and detects nothing.

        The filter graph is passed as argv, not through a shell, so nothing else
        will escape it and the failure is silent: every clip reports zero cuts.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "clip.mp4")
            with open(path, "wb") as handle:
                handle.write(b"\x00" * (64 * 1024))

            with patch.object(
                shot_integrity.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], 0, b"", b"n:0 pts_time:1.5 \nn:1 pts_time:4.25 \n"
                ),
            ) as run:
                self.assertEqual(shot_integrity.detect_shot_cuts(path), [1.5, 4.25])

            command = run.call_args.args[0]
            filter_graph = command[command.index("-filter:v") + 1]
            self.assertIn("select=gt(scene\\,", filter_graph)
            self.assertNotIn("select=gt(scene,", filter_graph)

    def test_an_unmeasurable_clip_keeps_the_window_it_arrived_with(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tiny.mp4")
            with open(path, "wb") as handle:
                handle.write(b"\x00" * 1024)

            inspection = shot_integrity.inspect_downloaded_clip(
                path,
                source_duration=10.0,
                start_time=2.0,
                end_time=5.0,
            )

        self.assertFalse(inspection.rejected)
        self.assertEqual((inspection.start_time, inspection.end_time), (2.0, 5.0))
        self.assertEqual(inspection.evidence["shot_cuts"], "unavailable")
        self.assertEqual(inspection.evidence["burned_in_overlay"], "unavailable")


@unittest.skipUnless(_ffmpeg_encodes(), "FFmpeg or numpy is unavailable")
class RealFootageTest(unittest.TestCase):
    """The detectors, run against encoded frames rather than mocked numbers."""

    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        root = cls._directory.name
        cls.clean = os.path.join(root, "clean.mp4")
        cls.branded = os.path.join(root, "branded.mp4")
        cls.two_shot = os.path.join(root, "two_shot.mp4")
        written = (
            _write_clip(cls.clean),
            _write_clip(cls.branded, branded=True),
            _write_clip(cls.two_shot, cut_at=1.5),
        )
        if not all(written):  # pragma: no cover - depends on the local encoder
            cls._directory.cleanup()
            raise unittest.SkipTest("this FFmpeg build could not encode the fixtures")

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_continuous_footage_reports_no_cuts(self):
        self.assertEqual(shot_integrity.detect_shot_cuts(self.clean), [])

    def test_a_hard_cut_is_found_where_it_was_encoded(self):
        cuts = shot_integrity.detect_shot_cuts(self.two_shot)
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0], 1.5, delta=0.1)

    def test_clean_footage_measures_no_overlay(self):
        measurement = shot_integrity.measure_burned_in_overlay(
            self.clean, start=0.0, duration=2.0
        )
        self.assertIsNotNone(measurement)
        self.assertFalse(shot_integrity.overlay_is_burned_in(measurement))

    def test_a_frozen_edge_band_is_measured_and_refused(self):
        measurement = shot_integrity.measure_burned_in_overlay(
            self.branded, start=0.0, duration=2.0
        )
        self.assertEqual(measurement["zone"], "bottom")
        self.assertGreater(measurement["density"], 0.09)
        self.assertTrue(shot_integrity.overlay_is_burned_in(measurement))

    def test_the_branded_clip_is_rejected_end_to_end(self):
        inspection = shot_integrity.inspect_downloaded_clip(
            self.branded,
            source_duration=3.0,
            start_time=0.0,
            end_time=2.0,
        )

        self.assertTrue(inspection.rejected)
        self.assertIn("burned-in", inspection.rejection_reason)
        self.assertEqual(inspection.evidence["shot_cuts"], [])

    def test_a_slot_longer_than_the_described_shot_is_rejected_end_to_end(self):
        """Exactly the beat that shipped: 2s asked of a shot 1.5s long."""
        inspection = shot_integrity.inspect_downloaded_clip(
            self.two_shot,
            source_duration=3.0,
            start_time=0.0,
            end_time=2.0,
            verified_start=0.0,
            verified_end=1.0,
        )

        self.assertTrue(inspection.rejected)
        self.assertEqual(
            inspection.evidence["shot_containment"], "no_shot_long_enough"
        )

    def test_a_window_padded_across_a_cut_is_pulled_back_not_dropped(self):
        """The described moment is in the second shot, so the window moves there."""
        inspection = shot_integrity.inspect_downloaded_clip(
            self.two_shot,
            source_duration=3.0,
            start_time=1.2,
            end_time=2.2,
            verified_start=1.6,
            verified_end=2.2,
        )

        self.assertFalse(inspection.rejected)
        self.assertGreater(inspection.start_time, 1.5)
        self.assertAlmostEqual(
            inspection.end_time - inspection.start_time, 1.0, places=3
        )
        self.assertEqual(inspection.evidence["shot_containment"], "shifted")
        self.assertGreater(inspection.evidence["shifted_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
