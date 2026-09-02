import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from app.config import config
from app.models.schema import (
    CriticalVisualFact,
    MaterialInfo,
    MandatoryFactResult,
    NarrationOverlap,
    SemanticAdjudication,
    VideoAspect,
    VisualRequirementSpec,
    VisualBeat,
    VisualSlot,
)
from app.services import material, task, twelvelabs


def _requirement_spec(requirement="Worker removes damaged boards"):
    return VisualRequirementSpec(
        schema_version="visual-requirement-spec-v1",
        generator_provider="test-provider",
        generator_model="test-model",
        original_requirement=requirement,
        subjects=["worker"],
        primary_action=None,
        objects=["boards"],
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


def _payload(score=0.8, *, spec=None, statuses=None, **overrides):
    spec = spec or _requirement_spec()
    statuses = statuses or {
        fact.id: "OBSERVED" for fact in spec.critical_visual_facts
    }
    payload = {
        "video_summary": "The requested action is clearly visible.",
        "observed_subjects": ["worker"],
        "observed_actions": [
            {
                "actor": "worker",
                "action": "performs requested action",
                "object": "boards",
                "source_or_target": "work area",
                "directness": "DIRECTLY_OBSERVED",
                "evidence": "The action itself is visible.",
            }
        ],
        "observed_objects": ["boards"],
        "observed_relations": ["worker acts on boards"],
        "observed_context": ["work area"],
        "visible_state": ["worker and boards are visible"],
        "critical_fact_evidence": [
            {
                "fact_id": fact.id,
                "status": statuses[fact.id],
                "evidence": f"Evidence for {fact.id}",
            }
            for fact in spec.critical_visual_facts
        ],
        "missing_required_facts": [
            fact.id
            for fact in spec.critical_visual_facts
            if fact.mandatory and statuses[fact.id] == "NOT_OBSERVED"
        ],
        "contradictory_facts": [
            fact.id
            for fact in spec.critical_visual_facts
            if statuses[fact.id] == "CONTRADICTED"
        ],
        "uncertainty": [
            fact.id
            for fact in spec.critical_visual_facts
            if statuses[fact.id] == "UNCERTAIN"
        ],
        "inference_warnings": [],
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
    }
    payload.update(overrides)
    return payload


def _evaluation(score, *, accepted=True):
    return {
        "accepted": False,
        "eligible_for_adjudication": accepted,
        "overall_score": score,
        "reason": f"score {score}",
        "observed_facts": {
            "critical_fact_evidence": [
                {"fact_id": "f1", "status": "OBSERVED", "evidence": "visible"}
            ]
        },
        "critical_gate": {
            "decision": "PASS" if accepted else "REJECT",
            "mandatory_fact_results": [
                {"fact_id": "f1", "status": "OBSERVED"}
            ],
            "missing_fact_ids": [],
            "contradictory_fact_ids": [],
            "uncertain_fact_ids": [],
        },
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
        primary_narration_slot_index=1,
        primary_narration_text="A worker removes rotten wooden boards.",
        visual_requirement="A worker removes rotten wooden boards.",
        narration_overlaps=[NarrationOverlap(1, 0.0, 4.0, 4.0)],
        search_queries=["worker removing rotten boards"],
        timing_source="edge_tts_boundary",
        timing_quality="boundary",
    )


def _visual_beat(
    *,
    index=1,
    semantic_group_id=1,
    duration=2.8,
    requirement="Coffee beans drying in sunlight",
    query="coffee beans drying sun",
    start_time=None,
):
    # Beats are contiguous by default. ``start_time`` is only overridden by tests
    # that need to prove a broken timeline is rejected.
    start_time = (index - 1) * duration if start_time is None else float(start_time)
    return VisualBeat(
        index=index,
        semantic_group_id=semantic_group_id,
        shot_index=index,
        start_time=start_time,
        end_time=start_time + duration,
        duration=duration,
        spoken_text=(
            "Workers previously picked cherries; later the beans will roast."
        ),
        visual_requirement=requirement,
        source_semantic_span_index=semantic_group_id,
        source_narration_slot_indexes=[index],
        start_unit=index - 1,
        end_unit_exclusive=index,
        timing_source="edge_tts_boundary",
        timing_quality="boundary",
        duration_policy="semantic_original",
        rapid_cut=duration < 1.5,
        search_queries=[query],
    )


def _patch_requirement_rewrite(test_case):
    """Make the requirement-rewrite recovery path inert for one test.

    Selection now asks the configured LLM for an alternative wording whenever an
    item cannot be filled, so any test that lets an item fail would otherwise send
    a real, billable request to whatever provider the developer has configured.
    Returning nothing reproduces the previous behavior of failing immediately, so
    existing expectations still describe what the code does. Tests that exercise
    the recovery path patch it themselves with a grounded alternative.
    """
    patcher = patch.object(
        material.llm,
        "generate_alternative_visual_requirements",
        return_value={},
    )
    mock = patcher.start()
    test_case.addCleanup(patcher.stop)
    return mock


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
        self.adjudicator_patcher = patch.object(
            twelvelabs.llm,
            "adjudicate_visual_candidates",
            side_effect=self._accept_observed_candidates,
        )
        self.adjudicator = self.adjudicator_patcher.start()

    @staticmethod
    def _accept_observed_candidates(requirement_spec, candidates, app_config=None):
        return {
            candidate["candidate_id"]: SemanticAdjudication(
                candidate_id=candidate["candidate_id"],
                decision="ACCEPT",
                mandatory_fact_results=[
                    MandatoryFactResult(fact_id=fact.id, status="OBSERVED")
                    for fact in requirement_spec.critical_visual_facts
                    if fact.mandatory
                ],
                missing_or_contradictory_facts=[],
                reason="All mandatory direct evidence is observed.",
            )
            for candidate in candidates
        }

    def tearDown(self):
        self.adjudicator_patcher.stop()
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_structured_response_uses_application_weights_and_hard_gates(self):
        spec = _requirement_spec()
        payload = _payload(spec=spec)
        payload["scores"] = {
            "semantic_match": 1.0,
            "action_match": 0.5,
            "subject_visibility": 0.75,
            "visual_quality": 0.4,
        }

        result = twelvelabs._parse_candidate_response(payload, 0.70, spec)

        self.assertAlmostEqual(result["overall_score"], 0.71)
        self.assertEqual(result["critical_gate"]["decision"], "PASS")
        self.assertTrue(result["eligible_for_adjudication"])
        self.assertFalse(result["accepted"])
        payload = _payload(spec=spec, statuses={"f1": "NOT_OBSERVED"}, score=1.0)
        rejected = twelvelabs._parse_candidate_response(payload, 0.0, spec)
        self.assertEqual(rejected["critical_gate"]["decision"], "REJECT")
        self.assertFalse(rejected["accepted"])
        self.assertFalse(rejected["eligible_for_adjudication"])

    def test_structured_schema_uses_only_twelvelabs_supported_number_constraints(self):
        schema = twelvelabs._candidate_response_schema(_requirement_spec())
        serialized = json.dumps(schema)
        score_properties = schema["properties"]["scores"]["properties"]

        self.assertNotIn("additionalProperties", serialized)
        for score_schema in score_properties.values():
            self.assertEqual(score_schema, {"type": "number"})

    def test_search_query_is_only_a_retrieval_hint_in_prompt(self):
        requirement = "A worker removes rotten boards."
        prompt = twelvelabs._candidate_prompt(_requirement_spec(requirement))

        self.assertIn(requirement, prompt)
        self.assertIn("what is directly visible", prompt)
        self.assertIn("retrieval metadata is provided", prompt)
        self.assertNotIn("wood construction", prompt)

    def test_known_high_score_false_positive_fails_critical_action_gate(self):
        requirement = "Worker hand-picking ripe coffee cherries"
        spec = _requirement_spec(requirement)
        result = twelvelabs._parse_candidate_response(
            _payload(
                1.0,
                spec=spec,
                statuses={"f1": "NOT_OBSERVED"},
                video_summary=(
                    "A worker handles already-picked cherries inside a basket."
                ),
                observed_actions=[
                    {
                        "actor": "worker",
                        "action": "handles",
                        "object": "coffee cherries",
                        "source_or_target": "basket",
                        "directness": "DIRECTLY_OBSERVED",
                        "evidence": "Hands move cherries already inside the basket.",
                    }
                ],
            ),
            0.7,
            spec,
        )

        self.assertEqual(result["overall_score"], 1.0)
        self.assertEqual(result["critical_gate"]["decision"], "REJECT")
        self.assertFalse(result["eligible_for_adjudication"])
        self.assertFalse(result["accepted"])

    def test_known_correct_positive_can_pass_gate_and_adjudication(self):
        requirement = "Worker hand-picking ripe coffee cherries"
        spec = _requirement_spec(requirement)
        parsed = twelvelabs._parse_candidate_response(
            _payload(0.92, spec=spec),
            0.7,
            spec,
        )
        candidate = _candidate("7116400")
        with patch.object(twelvelabs, "evaluate_candidate", return_value=parsed):
            winner, stats = twelvelabs.select_best_candidate(
                candidates=[candidate],
                slot_index=1,
                slot_duration=4,
                narration_text=requirement,
                search_query="coffee harvest",
                requirement_spec=spec,
            )

        self.assertIs(winner, candidate)
        self.assertEqual(winner.semantic_evaluation["critical_gate"]["decision"], "PASS")
        self.assertEqual(
            winner.semantic_evaluation["semantic_adjudication"]["decision"],
            "ACCEPT",
        )
        self.assertEqual(stats["adjudication_calls"], 1)

    def test_general_action_failures_are_gated_without_special_cases(self):
        cases = (
            (
                "Worker removes damaged boards",
                "worker touches and inspects boards",
                "NOT_OBSERVED",
                "REJECT",
            ),
            (
                "Mechanic installs a new tire",
                "mechanic measures the wheel hub",
                "NOT_OBSERVED",
                "REJECT",
            ),
            (
                "Cook flips a pancake",
                "the pan blocks the defining motion",
                "UNCERTAIN",
                "UNCERTAIN",
            ),
            (
                "Cook pours batter into a bowl",
                "cook stirs batter already in a bowl",
                "CONTRADICTED",
                "REJECT",
            ),
            (
                "Worker cuts a metal pipe",
                "worker holds a pipe without cutting",
                "NOT_OBSERVED",
                "REJECT",
            ),
        )
        for requirement, visible_action, status, expected_gate in cases:
            with self.subTest(requirement=requirement):
                spec = _requirement_spec(requirement)
                result = twelvelabs._parse_candidate_response(
                    _payload(
                        1.0,
                        spec=spec,
                        statuses={"f1": status},
                        video_summary=visible_action,
                    ),
                    0.0,
                    spec,
                )
                self.assertEqual(
                    result["critical_gate"]["decision"], expected_gate
                )
                self.assertFalse(result["eligible_for_adjudication"])
                self.assertFalse(result["accepted"])

    def test_missing_optional_attribute_does_not_fail_core_gate(self):
        spec = _requirement_spec("Worker installs a tire")
        spec.optional_attributes.append("bright workshop lighting")
        result = twelvelabs._parse_candidate_response(
            _payload(0.85, spec=spec),
            0.7,
            spec,
        )

        self.assertEqual(result["critical_gate"]["decision"], "PASS")
        self.assertTrue(result["eligible_for_adjudication"])
        candidate = _candidate("optional-missing")
        with patch.object(twelvelabs, "evaluate_candidate", return_value=result):
            winner, _ = twelvelabs.select_best_candidate(
                candidates=[candidate],
                slot_index=1,
                slot_duration=4,
                narration_text=spec.original_requirement,
                search_query="worker tire",
                requirement_spec=spec,
            )
        self.assertIs(winner, candidate)

    def test_high_score_failed_gate_cannot_trigger_early_stop(self):
        candidates = [_candidate(index) for index in range(1, 11)]

        def evaluate(**kwargs):
            if int(kwargs["asset_id"]) <= 5:
                result = _evaluation(1.0, accepted=False)
                result["critical_gate"]["decision"] = "REJECT"
                return result
            return _evaluation(0.92)

        with patch.object(
            twelvelabs, "evaluate_candidate", side_effect=evaluate
        ) as analyze:
            winner, stats = twelvelabs.select_best_candidate(
                candidates=candidates,
                slot_index=1,
                slot_duration=4,
                narration_text="Worker performs an action",
                search_query="retrieval hint",
                requirement_spec=_requirement_spec("Worker performs an action"),
                batch_size=5,
                strong_early_stop_score=0.9,
            )

        self.assertEqual(analyze.call_count, 10)
        self.assertEqual(winner.source_info["asset_id"], "6")
        self.assertEqual(stats["critical_gate_rejections"], 5)
        self.assertFalse(stats["early_stopped"])

    def test_uniformly_unrelated_footage_stops_before_the_next_batch(self):
        # The beat-level waste came from grinding through every page of a catalog
        # that never held the concept. One ranked list peaking far below the gate
        # is enough evidence: the pages behind it are ranked lower still.
        candidates = [_candidate(index) for index in range(1, 11)]

        with patch.object(
            twelvelabs,
            "evaluate_candidate",
            side_effect=lambda **kwargs: _evaluation(0.2, accepted=False),
        ) as analyze:
            winner, stats = twelvelabs.select_best_candidate(
                candidates=candidates,
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
                requirement_spec=_requirement_spec("narration"),
                batch_size=5,
                max_candidates=15,
                minimum_score=0.7,
                concurrency=5,
            )

        self.assertIsNone(winner)
        self.assertEqual(analyze.call_count, 5)
        self.assertTrue(stats["unrelated_footage"])
        self.assertTrue(stats["early_stopped"])
        self.assertEqual(stats["best_overall_score"], 0.2)
        self.adjudicator.assert_not_called()

    def test_footage_short_of_the_gate_but_close_is_still_explored(self):
        # "Unrelated" has to mean nowhere near, not merely short. Footage this
        # close is a wording problem, and a later page or phrasing can still win.
        candidates = [_candidate(index) for index in range(1, 11)]

        with patch.object(
            twelvelabs,
            "evaluate_candidate",
            side_effect=lambda **kwargs: _evaluation(0.6, accepted=False),
        ) as analyze:
            _, stats = twelvelabs.select_best_candidate(
                candidates=candidates,
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
                requirement_spec=_requirement_spec("narration"),
                batch_size=5,
                max_candidates=15,
                minimum_score=0.7,
                concurrency=5,
            )

        self.assertEqual(analyze.call_count, 10)
        self.assertFalse(stats["unrelated_footage"])

    def test_a_thin_sample_is_never_called_unrelated(self):
        candidates = [_candidate(index) for index in range(1, 3)]

        with patch.object(
            twelvelabs,
            "evaluate_candidate",
            side_effect=lambda **kwargs: _evaluation(0.1, accepted=False),
        ) as analyze:
            _, stats = twelvelabs.select_best_candidate(
                candidates=candidates,
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
                requirement_spec=_requirement_spec("narration"),
                batch_size=5,
                max_candidates=15,
                minimum_score=0.7,
                concurrency=5,
            )

        self.assertEqual(analyze.call_count, 2)
        self.assertFalse(stats["unrelated_footage"])

    def test_a_provider_failure_is_never_reported_as_unrelated_footage(self):
        # Zero scores from a quota error mean the observation never happened.
        # Reading that as "the catalog lacks this concept" would retire a
        # perfectly findable requirement on the strength of a billing problem.
        candidates = [_candidate(index) for index in range(1, 11)]

        def evaluate(**kwargs):
            result = _evaluation(0.0, accepted=False)
            result["reason"] = "TwelveLabs quota exceeded"
            return result

        with patch.object(twelvelabs, "evaluate_candidate", side_effect=evaluate):
            _, stats = twelvelabs.select_best_candidate(
                candidates=candidates,
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
                requirement_spec=_requirement_spec("narration"),
                batch_size=5,
                max_candidates=15,
                minimum_score=0.7,
                concurrency=5,
            )

        self.assertFalse(stats["unrelated_footage"])
        self.assertEqual(stats["api_failure_reason"], "TwelveLabs quota exceeded")

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
                requirement_spec=_requirement_spec("exact narration"),
                batch_size=5,
                max_candidates=15,
                minimum_score=0.7,
                strong_early_stop_score=0.9,
                concurrency=5,
            )

        self.assertEqual(call.call_count, 5)
        self.assertEqual(winner.source_info["asset_id"], "4")
        self.assertTrue(stats["early_stopped"])
        self.assertEqual(self.adjudicator.call_count, 1)
        self.assertEqual(
            len(self.adjudicator.call_args.args[1]),
            5,
        )
        for recorded_call in call.call_args_list:
            self.assertEqual(recorded_call.kwargs["provider"], "pexels")
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
                requirement_spec=_requirement_spec("narration"),
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
                requirement_spec=_requirement_spec("narration"),
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
                requirement_spec=_requirement_spec("narration"),
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
                requirement_spec=_requirement_spec("narration"),
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
                requirement_spec=_requirement_spec("narration"),
            )

        self.assertFalse(result["accepted"])
        # A truncated answer and an unparseable one are different failures with
        # different fixes, so they must never collapse into one reason string:
        # reporting truncation as "malformed" is what hid the token-budget bug.
        self.assertIn("truncated", result["reason"])
        self.assertNotIn("malformed", result["reason"])

    def test_a_discarded_response_names_the_rule_that_discarded_it(self):
        # Every discard here throws away an analysis that was already billed, and
        # "malformed" names the outcome while hiding which of two dozen checks
        # produced it. That is not a cosmetic complaint: it is how a token-budget
        # bug survived long enough to lose 12 candidates in one run.
        spec = _requirement_spec("narration")
        payload = _payload(0.9, spec=spec)
        payload["scores"] = {"semantic_match": 1.0}

        with patch.object(
            twelvelabs,
            "_sync_candidate_analysis",
            return_value=(payload, "direct_url"),
        ):
            result = twelvelabs.evaluate_candidate(
                asset_id="short-scores",
                video_url="https://videos.example/short-scores.mp4",
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
                requirement_spec=spec,
            )

        self.assertFalse(result["accepted"])
        self.assertIn("scores", result["reason"])
        # The prefix has to survive, because _analysis_api_failure_reason matches
        # this string by prefix to tell a provider-side non-verdict apart from a
        # content rejection.
        self.assertTrue(
            result["reason"].startswith("malformed TwelveLabs structured response")
        )

    def test_a_truncated_response_is_never_read_as_unrelated_footage(self):
        # A truncated answer is the strongest possible case of an observation that
        # never happened: the model was still writing when the budget ran out. Its
        # zero score says nothing about the footage, so it must not be able to
        # retire a search phrasing -- the same rule a quota error already gets.
        candidates = [_candidate(index) for index in range(1, 11)]

        def evaluate(**kwargs):
            result = _evaluation(0.0, accepted=False)
            result["reason"] = "truncated TwelveLabs structured response"
            return result

        with patch.object(twelvelabs, "evaluate_candidate", side_effect=evaluate):
            _, stats = twelvelabs.select_best_candidate(
                candidates=candidates,
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
                requirement_spec=_requirement_spec("narration"),
                batch_size=5,
                max_candidates=15,
                minimum_score=0.7,
                concurrency=5,
            )

        self.assertFalse(stats["unrelated_footage"])
        self.assertEqual(
            stats["api_failure_reason"], "truncated TwelveLabs structured response"
        )

    def test_an_unrequested_extra_field_does_not_discard_the_analysis(self):
        # Absent fields are fatal because the evidence they carried is genuinely
        # not there. An extra one is not: every field this parser uses it reads by
        # name, so a surplus sibling changes nothing, and discarding a complete
        # observation over it spends a candidate and can spend the phrasing too.
        spec = _requirement_spec("narration")
        payload = _payload(0.8, spec=spec)
        payload["confidence_note"] = "high confidence in the framing call"

        result = twelvelabs._parse_candidate_response(payload, 0.70, spec)

        self.assertIsNotNone(result)
        self.assertEqual(result["critical_gate"]["decision"], "PASS")
        self.assertTrue(result["eligible_for_adjudication"])

    def test_a_missing_field_still_discards_the_analysis(self):
        spec = _requirement_spec("narration")
        payload = _payload(0.8, spec=spec)
        del payload["visible_state"]

        result, rule = twelvelabs._parse_candidate_payload(payload, 0.70, spec)

        self.assertIsNone(result)
        self.assertIn("visible_state", rule)

    def test_a_restated_fact_list_is_derived_rather_than_used_to_discard(self):
        # missing_required_facts, contradictory_facts and uncertainty are a pure
        # function of critical_fact_evidence, which is validated fact by fact just
        # above. Comparing them to the model's own restatement was a cross-check
        # with no information in it whose penalty was discarding the whole billed
        # analysis, so they are derived and a disagreement is only logged.
        spec = _requirement_spec("narration")
        statuses = {fact.id: "OBSERVED" for fact in spec.critical_visual_facts}
        first_fact = spec.critical_visual_facts[0].id
        statuses[first_fact] = "CONTRADICTED"
        payload = _payload(0.8, spec=spec, statuses=statuses)
        payload["contradictory_facts"] = []
        payload["uncertainty"] = ["f99"]

        result = twelvelabs._parse_candidate_response(payload, 0.70, spec)

        self.assertIsNotNone(result)
        self.assertEqual(
            result["observed_facts"]["contradictory_facts"], [first_fact]
        )
        self.assertEqual(result["observed_facts"]["uncertainty"], [])
        self.assertEqual(result["critical_gate"]["decision"], "REJECT")

    def test_one_unjudged_candidate_does_not_void_its_adjudicated_siblings(self):
        # A batch holds up to five candidates that have each already passed the
        # whole critical evidence gate, which is the scarcest thing this pipeline
        # produces. An adjudicator that returns four verdicts instead of five used
        # to cost all five, and the search phrasing behind them.
        candidates = [_candidate(index) for index in range(1, 6)]
        spec = _requirement_spec("narration")

        def partial_adjudication(requirement_spec, batch, app_config=None):
            decisions = self._accept_observed_candidates(
                requirement_spec, batch, app_config=app_config
            )
            decisions.pop(batch[0]["candidate_id"], None)
            return decisions

        self.adjudicator.side_effect = partial_adjudication
        with patch.object(
            twelvelabs,
            "evaluate_candidate",
            side_effect=lambda **kwargs: _evaluation(
                0.95 if kwargs["asset_id"] == "1" else 0.80
            ),
        ):
            winner, stats = twelvelabs.select_best_candidate(
                candidates=candidates,
                slot_index=1,
                slot_duration=4,
                narration_text="narration",
                search_query="query",
                requirement_spec=spec,
                batch_size=5,
                max_candidates=5,
                minimum_score=0.7,
                strong_early_stop_score=0.99,
                concurrency=5,
            )

        # The unjudged candidate was the highest scoring one, so if it were still
        # poisoning the batch there would be no winner at all.
        self.assertIsNotNone(winner)
        self.assertEqual(winner.source_info["asset_id"], "2")
        self.assertEqual(stats["adjudication_failures"], 1)
        self.assertEqual(stats["candidates_adjudicated"], 4)

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
                    requirement_spec=_requirement_spec("same narration"),
                )
                second = twelvelabs.evaluate_candidate(
                    asset_id="cached",
                    video_url="https://videos.example/cached.mp4",
                    slot_index=2,
                    slot_duration=4,
                    narration_text="same narration",
                    search_query="different retrieval hint",
                    requirement_spec=_requirement_spec("same narration"),
                )

        self.assertEqual(analyze.call_count, 1)
        self.assertFalse(first["_cache_hit"])
        self.assertTrue(second["_cache_hit"])

    def test_candidate_cache_key_is_provider_and_requirement_aware(self):
        pexels_key = twelvelabs._candidate_cache_digest(
            "pexels",
            "123",
            "worker removes damaged boards",
        )
        pixabay_key = twelvelabs._candidate_cache_digest(
            "pixabay",
            "123",
            "worker removes damaged boards",
        )
        changed_requirement_key = twelvelabs._candidate_cache_digest(
            "pexels",
            "123",
            "worker paints undamaged boards",
        )

        self.assertNotEqual(pexels_key, pixabay_key)
        self.assertNotEqual(pexels_key, changed_requirement_key)

    def test_old_score_first_cache_version_is_not_reused(self):
        requirement = "Worker installs a new tire"
        spec = _requirement_spec(requirement)
        digest = twelvelabs.llm.visual_requirement_spec_digest(spec)
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                twelvelabs, "_candidate_cache_dir", return_value=Path(temp_dir)
            ):
                cache_path = twelvelabs._candidate_cache_path(
                    "pexels",
                    "123",
                    requirement,
                    digest,
                )
                cache_path.write_text(
                    json.dumps(
                        {
                            "version": 2,
                            "model": twelvelabs.DEFAULT_PEGASUS_MODEL,
                            "schema_version": "smart-stock-candidate-v2",
                            "result": {"accepted": True, "overall_score": 1.0},
                        }
                    ),
                    encoding="utf-8",
                )

                loaded = twelvelabs._load_candidate_evaluation_cache(
                    "pexels",
                    "123",
                    requirement,
                    digest,
                )

        self.assertIsNone(loaded)

    def test_candidate_cache_does_not_collide_across_providers(self):
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
                for provider in ("pexels", "pixabay"):
                    twelvelabs.evaluate_candidate(
                        provider=provider,
                        asset_id="123",
                        video_url=f"https://videos.example/{provider}.mp4",
                        slot_index=1,
                        slot_duration=4,
                        narration_text="same visual requirement",
                        search_query="same query",
                        requirement_spec=_requirement_spec(
                            "same visual requirement"
                        ),
                    )

        self.assertEqual(analyze.call_count, 2)

    def test_candidate_cache_excludes_secret_request_metadata(self):
        secret = "tlk-secret-must-not-persist"
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                twelvelabs,
                "_candidate_cache_dir",
                return_value=Path(temp_dir),
            ):
                twelvelabs._save_candidate_evaluation_cache(
                    "pexels",
                    "123",
                    secret,
                    {
                        **_evaluation(0.82),
                        "provider": "twelvelabs",
                        "api_key": secret,
                        "authorization": f"Bearer {secret}",
                    },
                )
                cache_contents = "".join(
                    path.read_text(encoding="utf-8")
                    for path in Path(temp_dir).glob("*.json")
                )

        self.assertNotIn(secret, cache_contents)
        self.assertNotIn("api_key", cache_contents)
        self.assertNotIn("authorization", cache_contents)

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
                        requirement_spec=_requirement_spec(narration),
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
                "https://videos.example/direct.mp4",
                "prompt",
                twelvelabs._candidate_response_schema(_requirement_spec()),
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
                "https://videos.example/fallback.mp4",
                "prompt",
                twelvelabs._candidate_response_schema(_requirement_spec()),
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
                "https://videos.example/no-fallback.mp4",
                "prompt",
                twelvelabs._candidate_response_schema(_requirement_spec()),
            )
        client.assets.create.assert_not_called()

    def test_candidate_analysis_gets_its_own_output_budget(self):
        """The evidence schema must not reuse the one-line QA token floor.

        The candidate schema has fourteen required fields, two of which are
        arrays of objects carrying free-text evidence. Sending it with the QA
        floor made Pegasus return finish_reason="length" on long or visually
        busy clips, and every one of those candidates was discarded without a
        verdict. Asserting the shape -- a dedicated, strictly larger budget that
        stays inside the documented range -- is what stops the constants being
        conflated again, because the failure is invisible to any test whose
        fixture response happens to be short.
        """
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
            twelvelabs._sync_candidate_analysis(
                "https://videos.example/budget.mp4",
                "prompt",
                twelvelabs._candidate_response_schema(_requirement_spec()),
            )

        max_tokens = client.analyze.call_args.kwargs["max_tokens"]
        self.assertEqual(max_tokens, twelvelabs._PEGASUS_CANDIDATE_MAX_TOKENS)
        self.assertGreater(max_tokens, twelvelabs._PEGASUS_MIN_MAX_TOKENS)
        # Pegasus rejects max_tokens outside [512, 98304] with a 400.
        self.assertGreaterEqual(max_tokens, 512)
        self.assertLessEqual(max_tokens, 98304)

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

    def test_short_beat_uses_supported_detection_window_then_exact_trim(self):
        client = MagicMock()
        with patch.object(
            twelvelabs,
            "_temporal_response_format",
            return_value="response-format",
        ):
            twelvelabs._create_temporal_task(
                client,
                video_context="video-context",
                narration_text="A match igniting",
                requested_source_duration=1.6,
            )

        create_kwargs = client.analyze_async.tasks.create.call_args.kwargs
        self.assertEqual(create_kwargs["min_segment_duration"], 2.0)
        self.assertEqual(create_kwargs["max_segment_duration"], 2.0)

        response = SimpleNamespace(
            result=SimpleNamespace(
                finish_reason="stop",
                data=json.dumps(
                    {
                        "best_visual_match": [
                            {
                                "start_time": 9.0,
                                "end_time": 11.0,
                                "metadata": {
                                    "match_quality": 0.95,
                                    "action_visible": True,
                                    "subject_visible": True,
                                    "description": "the match ignites",
                                },
                            }
                        ]
                    }
                ),
            )
        )
        segment = twelvelabs._parse_temporal_segments(
            response,
            source_duration=20.0,
            requested_source_duration=1.6,
        )

        self.assertAlmostEqual(
            segment["source_end_time"] - segment["source_start_time"],
            1.6,
        )
        self.assertGreater(segment["source_start_time"], 0.0)

    def test_temporal_retrieval_retries_transient_error_on_same_task_id(self):
        class RemoteProtocolError(Exception):
            pass

        ready = SimpleNamespace(status="ready", result=SimpleNamespace())
        client = MagicMock()
        client.analyze_async.tasks.retrieve.side_effect = [
            RemoteProtocolError("connection closed"),
            RemoteProtocolError("connection closed again"),
            ready,
        ]

        with patch.object(twelvelabs.time, "sleep") as sleep:
            result = twelvelabs._wait_for_temporal_task(
                client,
                "existing-task-id",
                timeout_seconds=30,
                backoff_seconds=0.1,
            )

        self.assertIs(result, ready)
        self.assertEqual(
            client.analyze_async.tasks.retrieve.call_args_list,
            [
                call("existing-task-id"),
                call("existing-task-id"),
                call("existing-task-id"),
            ],
        )
        self.assertEqual([item.args[0] for item in sleep.call_args_list], [0.1, 0.2])

    def test_temporal_retrieval_does_not_retry_auth_or_deterministic_error(self):
        class ApiError(Exception):
            def __init__(self, status_code):
                super().__init__(f"HTTP {status_code}")
                self.status_code = status_code

        for status_code, expected_category in ((401, "auth_quota"), (400, "service")):
            with self.subTest(status_code=status_code):
                client = MagicMock()
                client.analyze_async.tasks.retrieve.side_effect = ApiError(status_code)
                with (
                    patch.object(twelvelabs.time, "sleep") as sleep,
                    self.assertRaises(twelvelabs.TemporalSegmentationError) as raised,
                ):
                    twelvelabs._wait_for_temporal_task(
                        client,
                        "existing-task-id",
                        timeout_seconds=30,
                    )

                self.assertEqual(raised.exception.category, expected_category)
                client.analyze_async.tasks.retrieve.assert_called_once_with(
                    "existing-task-id"
                )
                sleep.assert_not_called()

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
    def setUp(self):
        self.requirement_patcher = patch.object(
            material.llm,
            "generate_visual_requirement_specs",
            side_effect=lambda requirements: {
                material.llm.normalize_visual_requirement(requirement): (
                    _requirement_spec(requirement)
                )
                for requirement in requirements
            },
        )
        self.requirement_generator = self.requirement_patcher.start()
        _patch_requirement_rewrite(self)

    def tearDown(self):
        self.requirement_patcher.stop()

    @staticmethod
    def _prepare(items):
        return material._prepare_twelvelabs_candidates(
            items,
            video_aspect=VideoAspect.portrait,
            required_source_duration=4,
            preferred_max_source_duration=30,
        )

    def test_same_numeric_id_from_different_providers_is_not_duplicate(self):
        pexels = _candidate("pexels-123", url="https://videos.example/p.mp4")
        pexels.source_info["asset_id"] = "123"
        pixabay = _candidate("pixabay-123", url="https://videos.example/x.mp4")
        pixabay.provider = "pixabay"
        pixabay.source_info.update({"provider": "pixabay", "asset_id": "123"})

        prepared = self._prepare([pexels, pixabay])

        self.assertEqual(prepared, [pexels, pixabay])

    def test_fail_closed_decomposition_failure_precedes_stock_search(self):
        search = MagicMock()
        service = SimpleNamespace(
            candidate_selection_settings=lambda: {
                "batch_size": 5,
                "max_candidates": 15,
                "minimum_score": 0.7,
                "strong_early_stop_score": 0.9,
                "preferred_max_source_duration": 30,
                "concurrency": 5,
                "fail_closed": True,
            }
        )
        self.requirement_generator.side_effect = None
        self.requirement_generator.return_value = {}

        with (
            patch.object(
                material.task_artifacts,
                "patch_script_data",
                return_value=True,
            ) as persist,
            self.assertRaisesRegex(
                material.SmartMaterialSelectionError,
                "decomposition failed",
            ),
        ):
            material._download_videos_by_script_order_smart(
                task_id="decomposition-failed",
                search_terms=["worker removing rotten boards"],
                visual_slots=[_visual_slot()],
                search_videos=search,
                video_aspect=VideoAspect.portrait,
                max_clip_duration=4,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )

        search.assert_not_called()
        self.assertEqual(
            persist.call_args.kwargs["semantic_verifier_runs"][0][
                "final_decision"
            ],
            "DECOMPOSITION_FAILED",
        )

    def test_same_provider_asset_id_is_duplicate(self):
        first = _candidate("first", url="https://videos.example/first.mp4")
        second = _candidate("second", url="https://videos.example/second.mp4")
        first.source_info["asset_id"] = "123"
        second.source_info["asset_id"] = "123"

        prepared = self._prepare([first, second])

        self.assertEqual(prepared, [first])

    def test_exact_normalized_url_is_duplicate_across_provider_metadata(self):
        pexels = _candidate(
            "pexels-url",
            url="https://VIDEOS.example:443/shared.mp4#preview-one",
        )
        pixabay = _candidate(
            "pixabay-url",
            url="https://videos.example/shared.mp4#preview-two",
        )
        pixabay.provider = "pixabay"
        pixabay.source_info.update(
            {"provider": "pixabay", "asset_id": "different-id"}
        )

        prepared = self._prepare([pexels, pixabay])

        self.assertEqual(prepared, [pexels])

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

        prepared = self._prepare(items)

        self.assertEqual(
            [item.source_info["asset_id"] for item in prepared],
            ["short", "long"],
        )

    def test_visual_beat_source_duration_uses_beat_duration_and_clip_speed(self):
        cases = (
            (1.7, 1.0, 1.7, 4),
            (2.8, 1.0, 2.8, 4),
            (4.6, 1.0, 4.6, 5),
            (4.0, 0.5, 2.0, 4),
            (4.0, 1.0, 4.0, 4),
            (4.0, 2.0, 8.0, 8),
        )
        for duration, speed, expected_source, expected_minimum in cases:
            with self.subTest(duration=duration, speed=speed):
                beat = _visual_beat(duration=duration)
                candidate = _candidate(f"{duration}-{speed}", duration=12)
                candidate.overall_score = 0.91
                candidate.semantic_evaluation = {
                    "provider": "twelvelabs",
                    "accepted": True,
                    "overall_score": 0.91,
                }
                selector = MagicMock(
                    return_value=(
                        candidate,
                        {
                            "api_candidates_analyzed": 1,
                            "source_seconds_analyzed": 12,
                        },
                    )
                )
                segmenter = MagicMock(
                    return_value={
                        "source_start_time": 1.0,
                        "source_end_time": 1.0 + expected_source,
                        "description": "coffee beans visibly drying",
                    }
                )
                service = SimpleNamespace(
                    TemporalSegmentationError=twelvelabs.TemporalSegmentationError,
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
                search = MagicMock(return_value=[candidate])

                with (
                    patch.object(
                        material,
                        "save_video",
                        return_value="D:/task/beat.mp4",
                    ),
                    patch.object(
                        material.task_artifacts,
                        "patch_script_data",
                        return_value=True,
                    ) as persist,
                ):
                    paths = material._download_videos_by_script_order_smart(
                        task_id="beat-duration",
                        search_terms=beat.search_queries,
                        visual_beats=[beat],
                        search_videos=search,
                        video_aspect=VideoAspect.portrait,
                        max_clip_duration=5,
                        material_directory="",
                        clip_speed=speed,
                        twelvelabs_service=service,
                        max_candidates_override=5,
                    )

                self.assertEqual(paths, ["D:/task/beat.mp4"])
                self.assertEqual(
                    material.required_source_duration_for_timeline(duration, speed),
                    expected_source,
                )
                self.assertEqual(
                    search.call_args.kwargs["minimum_duration"],
                    expected_minimum,
                )
                self.assertEqual(selector.call_args.kwargs["slot_duration"], duration)
                self.assertEqual(selector.call_args.kwargs["max_candidates"], 5)
                self.assertEqual(
                    selector.call_args.kwargs["narration_text"],
                    beat.visual_requirement,
                )
                self.assertNotIn(
                    "previously picked",
                    selector.call_args.kwargs["narration_text"],
                )
                self.assertEqual(segmenter.call_args.kwargs["slot_duration"], duration)
                self.assertEqual(segmenter.call_args.kwargs["clip_speed"], speed)
                self.assertEqual(
                    segmenter.call_args.kwargs["requested_source_duration"],
                    expected_source,
                )
                self.assertEqual(
                    segmenter.call_args.kwargs["narration_text"],
                    beat.visual_requirement,
                )
                record = persist.call_args.kwargs["material_sources"][0]
                self.assertEqual(record["visual_beat_index"], beat.index)
                self.assertEqual(record["semantic_group_id"], beat.semantic_group_id)
                self.assertEqual(record["search_term"], beat.search_queries[0])
                self.assertEqual(record["required_target_duration"], duration)
                self.assertEqual(record["required_source_duration"], expected_source)
                self.assertAlmostEqual(
                    record["source_end_time"] - record["source_start_time"],
                    expected_source,
                )

    def test_short_visual_beat_fail_closed_never_uses_zero_start_fallback(self):
        beat = _visual_beat(duration=1.6)
        candidate = _candidate("short-beat", duration=4)
        candidate.overall_score = 0.9
        service = SimpleNamespace(
            TemporalSegmentationError=twelvelabs.TemporalSegmentationError,
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
                    candidate,
                    {
                        "api_candidates_analyzed": 1,
                        "source_seconds_analyzed": 4,
                    },
                )
            ),
            segment_winner=MagicMock(
                return_value={
                    "source_start_time": 0.0,
                    "source_end_time": 1.0,
                    "description": "too short to satisfy the beat",
                }
            ),
        )

        with (
            patch.object(material, "save_video") as save,
            patch.object(
                material.task_artifacts,
                "patch_script_data",
                return_value=True,
            ),
            self.assertRaisesRegex(
                material.SmartMaterialSelectionError,
                "No valid TwelveLabs temporal segment",
            ),
        ):
            material._download_videos_by_script_order_smart(
                task_id="short-fail-closed",
                search_terms=beat.search_queries,
                visual_beats=[beat],
                search_videos=lambda **kwargs: [candidate],
                video_aspect=VideoAspect.portrait,
                max_clip_duration=5,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )

        save.assert_not_called()

    def test_sibling_visual_beats_remain_separate_and_do_not_repeat_winner(self):
        requirement = "Coffee beans roasting inside a heated drum"
        query = "coffee beans roasting drum"
        beats = [
            _visual_beat(
                index=1,
                semantic_group_id=7,
                duration=4.0,
                requirement=requirement,
                query=query,
            ),
            _visual_beat(
                index=2,
                semantic_group_id=7,
                duration=4.0,
                requirement=requirement,
                query=query,
            ),
        ]
        candidates = [_candidate(701), _candidate(702)]
        selector = MagicMock()

        def select_first(**kwargs):
            winner = kwargs["candidates"][0]
            winner.overall_score = 0.92
            winner.semantic_evaluation = {
                "provider": "twelvelabs",
                "accepted": True,
                "overall_score": 0.92,
            }
            return winner, {
                "api_candidates_analyzed": len(kwargs["candidates"]),
                "source_seconds_analyzed": sum(
                    item.duration for item in kwargs["candidates"]
                ),
            }

        selector.side_effect = select_first
        service = SimpleNamespace(
            TemporalSegmentationError=twelvelabs.TemporalSegmentationError,
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
            segment_winner=MagicMock(
                return_value={
                    "source_start_time": 2.0,
                    "source_end_time": 6.0,
                    "description": "coffee beans roasting",
                }
            ),
        )

        with (
            patch.object(
                material,
                "save_video",
                side_effect=["D:/task/beat-1.mp4", "D:/task/beat-2.mp4"],
            ),
            patch.object(
                material.task_artifacts,
                "patch_script_data",
                return_value=True,
            ) as persist,
        ):
            paths = material._download_videos_by_script_order_smart(
                task_id="sibling-beats",
                search_terms=[query, query],
                visual_beats=beats,
                search_videos=lambda **kwargs: candidates,
                video_aspect=VideoAspect.portrait,
                max_clip_duration=5,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )

        self.assertEqual(paths, ["D:/task/beat-1.mp4", "D:/task/beat-2.mp4"])
        self.assertEqual(selector.call_count, 2)
        self.assertEqual(selector.call_args_list[0].kwargs["candidates"], candidates)
        self.assertEqual(
            selector.call_args_list[1].kwargs["candidates"],
            [candidates[1]],
        )
        records = persist.call_args.kwargs["material_sources"]
        self.assertEqual(
            [record["visual_beat_index"] for record in records],
            [1, 2],
        )
        self.assertEqual(
            [record["provider_asset_id"] for record in records],
            ["701", "702"],
        )

    def test_coffee_visual_beats_keep_requirements_durations_and_provenance_isolated(self):
        specifications = [
            (3.2, "Coffee cherries growing on plants", "coffee cherries plant"),
            (2.8, "Worker hand-picking coffee cherries", "picking coffee cherries"),
            (4.0, "Coffee beans drying in sunlight", "coffee beans drying sun"),
            (2.8, "Coffee beans roasting in a drum", "coffee beans roasting drum"),
        ]
        beats = [
            _visual_beat(
                index=index,
                semantic_group_id=index,
                duration=duration,
                requirement=requirement,
                query=query,
            )
            for index, (duration, requirement, query) in enumerate(
                specifications,
                start=1,
            )
        ]
        candidates_by_query = {
            query: _candidate(index, duration=12)
            for index, (_, _, query) in enumerate(specifications, start=1)
        }
        selector_calls = []

        def select_candidate(**kwargs):
            selector_calls.append(kwargs)
            winner = kwargs["candidates"][0]
            winner.overall_score = 0.9
            winner.semantic_evaluation = {
                "provider": "twelvelabs",
                "accepted": True,
                "overall_score": 0.9,
            }
            return winner, {
                "api_candidates_analyzed": 1,
                "source_seconds_analyzed": winner.duration,
            }

        service = SimpleNamespace(
            TemporalSegmentationError=twelvelabs.TemporalSegmentationError,
            candidate_selection_settings=lambda: {
                "batch_size": 5,
                "max_candidates": 15,
                "minimum_score": 0.7,
                "strong_early_stop_score": 0.9,
                "preferred_max_source_duration": 30,
                "concurrency": 5,
                "fail_closed": True,
            },
            select_best_candidate=select_candidate,
            segment_winner=lambda **kwargs: {
                "source_start_time": 1.0,
                "source_end_time": 1.0 + kwargs["slot_duration"],
                "description": kwargs["narration_text"],
            },
        )

        with (
            patch.object(
                material,
                "save_video",
                side_effect=[f"D:/task/coffee-{index}.mp4" for index in range(1, 5)],
            ),
            patch.object(
                material.task_artifacts,
                "patch_script_data",
                return_value=True,
            ) as persist,
        ):
            paths = material._download_videos_by_script_order_smart(
                task_id="coffee-beats",
                search_terms=[beat.search_queries[0] for beat in beats],
                visual_beats=beats,
                search_videos=lambda search_term, **kwargs: [
                    candidates_by_query[search_term]
                ],
                video_aspect=VideoAspect.portrait,
                max_clip_duration=5,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )

        self.assertEqual(len(paths), 4)
        self.assertEqual(
            [call_args["narration_text"] for call_args in selector_calls],
            [requirement for _, requirement, _ in specifications],
        )
        self.assertNotIn("picking", selector_calls[2]["narration_text"].lower())
        self.assertNotIn("drying", selector_calls[3]["narration_text"].lower())
        self.assertEqual(
            [call_args["slot_duration"] for call_args in selector_calls],
            [duration for duration, _, _ in specifications],
        )
        records = persist.call_args.kwargs["material_sources"]
        self.assertEqual(
            [record["visual_beat_index"] for record in records],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            [record["required_target_duration"] for record in records],
            [duration for duration, _, _ in specifications],
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
                "candidate_evaluations": [
                    {
                        "provider": "pexels",
                        "provider_asset_id": "2",
                        "ranking_position": 1,
                        "observed_facts": {
                            "critical_fact_evidence": [
                                {
                                    "fact_id": "f1",
                                    "status": "OBSERVED",
                                    "evidence": "The required action is visible.",
                                }
                            ]
                        },
                        "critical_gate": {"decision": "PASS"},
                        "semantic_adjudication": {"decision": "ACCEPT"},
                        "overall_score": 0.94,
                        "accepted": True,
                    }
                ],
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
            selector.call_args.kwargs["narration_text"], slot.visual_requirement
        )
        self.assertEqual(
            segmenter.call_args.kwargs["narration_text"], slot.visual_requirement
        )
        self.assertEqual(
            segmenter.call_args.kwargs["video_url"],
            "https://videos.example/2.mp4",
        )
        self.assertEqual(save.call_count, 1)
        records = persist.call_args.kwargs["material_sources"]
        self.assertEqual(records[0]["source_start_time"], 6.5)
        self.assertEqual(records[0]["source_end_time"], 10.5)
        self.assertEqual(records[0]["slot_index"], 1)
        verifier_runs = persist.call_args.kwargs["semantic_verifier_runs"]
        self.assertEqual(verifier_runs[0]["visual_item_index"], 1)
        self.assertEqual(verifier_runs[0]["final_decision"], "ACCEPT")
        self.assertEqual(
            verifier_runs[0]["candidate_evaluations"][0]["critical_gate"][
                "decision"
            ],
            "PASS",
        )
        self.assertEqual(
            verifier_runs[0]["candidate_evaluations"][0][
                "semantic_adjudication"
            ]["decision"],
            "ACCEPT",
        )

    def test_coffee_slot_evaluates_and_segments_only_primary_visual_requirement(self):
        picking = (
            "A farm worker reaches between the leaves and hand-picks the ripe "
            "cherries into a basket."
        )
        drying = (
            "Pale coffee beans spread across drying beds and sit under the warm sun."
        )
        slot = VisualSlot(
            index=3,
            start_time=8.0,
            end_time=12.0,
            duration=4.0,
            narration_slot_indexes=[2, 3],
            narration_text=f"{picking} {drying}",
            primary_narration_slot_index=3,
            primary_narration_text=drying,
            visual_requirement=drying,
            narration_overlaps=[
                NarrationOverlap(2, 8.0, 9.2, 1.2),
                NarrationOverlap(3, 9.2, 12.0, 2.8),
            ],
            search_queries=["coffee beans drying beds"],
            timing_source="whisper",
            timing_quality="speech_recognition",
        )
        candidate = _candidate(31)
        candidate.overall_score = 0.91
        selector = MagicMock(
            return_value=(
                candidate,
                {
                    "api_candidates_analyzed": 1,
                    "source_seconds_analyzed": 12,
                },
            )
        )
        segmenter = MagicMock(
            return_value={
                "source_start_time": 3.0,
                "source_end_time": 7.0,
                "description": "coffee beans drying in the sun",
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
            patch.object(material, "save_video", return_value="D:/task/coffee.mp4"),
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ),
        ):
            result = material._download_videos_by_script_order_smart(
                task_id="coffee-primary-requirement",
                search_terms=["coffee beans drying beds"],
                visual_slots=[slot],
                search_videos=lambda **kwargs: [candidate],
                video_aspect=VideoAspect.portrait,
                max_clip_duration=4,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )

        self.assertEqual(result, ["D:/task/coffee.mp4"])
        self.assertEqual(selector.call_args.kwargs["narration_text"], drying)
        self.assertEqual(segmenter.call_args.kwargs["narration_text"], drying)
        self.assertNotIn(picking, selector.call_args.kwargs["narration_text"])

    def test_temporal_failure_messages_distinguish_no_match_service_and_auth(self):
        cases = [
            (None, "No valid TwelveLabs temporal segment"),
            (
                twelvelabs.TemporalSegmentationError(
                    "service", "TwelveLabs API unavailable (RemoteProtocolError)"
                ),
                "temporarily unavailable",
            ),
            (
                twelvelabs.TemporalSegmentationError(
                    "auth_quota", "TwelveLabs quota or rate limit exhausted"
                ),
                "quota or rate limit exhausted",
            ),
        ]

        for segment_result, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                candidate = _candidate(1)
                candidate.overall_score = 0.9
                segmenter = MagicMock()
                if isinstance(segment_result, Exception):
                    segmenter.side_effect = segment_result
                else:
                    segmenter.return_value = segment_result
                service = SimpleNamespace(
                    TemporalSegmentationError=twelvelabs.TemporalSegmentationError,
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
                            candidate,
                            {
                                "api_candidates_analyzed": 1,
                                "source_seconds_analyzed": 12,
                            },
                        )
                    ),
                    segment_winner=segmenter,
                )

                with (
                    patch.object(
                        material.task_artifacts,
                        "patch_script_data",
                        return_value=True,
                    ),
                    self.assertRaisesRegex(
                        material.SmartMaterialSelectionError,
                        expected_message,
                    ),
                ):
                    material._download_videos_by_script_order_smart(
                        task_id="temporal-failure",
                        search_terms=["worker removing rotten boards"],
                        visual_slots=[_visual_slot()],
                        search_videos=lambda **kwargs: [candidate],
                        video_aspect=VideoAspect.portrait,
                        max_clip_duration=4,
                        material_directory="",
                        clip_speed=1.0,
                        twelvelabs_service=service,
                    )

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

    def test_production_ordered_path_selects_visual_slots_only_once(self):
        slot = _visual_slot()
        with (
            patch.object(
                twelvelabs,
                "is_smart_visual_matching_enabled",
                return_value=True,
            ),
            patch.object(
                material,
                "_download_videos_by_script_order_smart",
                return_value=["D:/task/slot.mp4"],
            ) as smart,
        ):
            result = material._download_videos_by_script_order(
                task_id="production-slot-only",
                search_terms=slot.search_queries,
                search_videos=lambda **kwargs: [],
                video_aspect=VideoAspect.portrait,
                audio_duration=4,
                max_clip_duration=4,
                material_directory="",
                visual_slots=[slot],
                clip_speed=1.0,
            )

        self.assertEqual(result, ["D:/task/slot.mp4"])
        smart.assert_called_once()
        self.assertEqual(smart.call_args.kwargs["visual_slots"], [slot])
        self.assertNotIn("visual_beats", smart.call_args.kwargs)

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


class TestSmartProviderCascade(unittest.TestCase):
    """The provider chain smart matching may walk for a single visual item."""

    @staticmethod
    def _keys(**overrides):
        keys = {
            "pexels_api_keys": ["pexels-key"],
            "pixabay_api_keys": ["pixabay-key"],
            "smart_material_provider_cascade": True,
        }
        keys.update(overrides)
        return keys

    def test_selected_provider_always_leads_the_cascade(self):
        with patch.dict(config.app, self._keys()):
            self.assertEqual(
                material.smart_provider_chain("pinterest"),
                ["pinterest", "pexels", "pixabay"],
            )
            self.assertEqual(
                material.smart_provider_chain("pixabay"),
                ["pixabay", "pinterest", "pexels"],
            )
            self.assertEqual(
                material.smart_provider_chain("pexels"),
                ["pexels", "pinterest", "pixabay"],
            )
            # Coverr was retired as a paid provider and is no longer a
            # searchable stock source, so it never enters the cascade.
            self.assertEqual(material.smart_provider_chain("coverr"), [])

    def test_providers_without_an_api_key_are_skipped(self):
        with patch.dict(config.app, self._keys(pixabay_api_keys=[])):
            self.assertFalse(material.provider_has_api_key("pixabay"))
            self.assertEqual(
                material.smart_provider_chain("pexels"), ["pexels", "pinterest"]
            )

    def test_a_keyless_provider_is_never_treated_as_unconfigured(self):
        # Pinterest has no config key at all. The readiness check has to answer
        # "needs no credential" and "has a credential" the same way, or the one
        # provider that always works would be the only one dropped.
        with patch.dict(
            config.app,
            self._keys(pexels_api_keys=[], pixabay_api_keys=[]),
        ):
            self.assertTrue(material.provider_has_api_key("pinterest"))
            self.assertTrue(material.provider_has_api_key("PINTEREST"))
            self.assertFalse(material.provider_has_api_key("coverr"))
            self.assertFalse(material.provider_has_api_key(""))
            self.assertFalse(material.provider_has_api_key(None))

    def test_cascade_without_any_key_falls_through_to_the_keyless_provider(self):
        # This used to keep the unusable selected provider so the user would see
        # its "api key is not set" error. There is now a provider that needs no
        # key, so the run has somewhere to go and stopping would be a choice to
        # fail a task that can still succeed.
        with patch.dict(
            config.app,
            self._keys(
                pexels_api_keys=[],
                pixabay_api_keys=[],
            ),
        ):
            self.assertEqual(material.smart_provider_chain("pexels"), ["pinterest"])
            self.assertEqual(material.smart_provider_chain("pixabay"), ["pinterest"])

    def test_cascade_can_be_disabled_to_pin_the_selected_provider(self):
        with patch.dict(
            config.app, self._keys(smart_material_provider_cascade="off")
        ):
            self.assertFalse(material.is_provider_cascade_enabled())
            self.assertEqual(material.smart_provider_chain("pexels"), ["pexels"])

    def test_a_pinned_provider_without_a_key_still_reports_its_own_error(self):
        # With the cascade off there is nothing to fall through to, so the chain
        # must keep the provider the user selected. Returning an empty list here
        # would surface as "no candidates" instead of "your key is missing".
        with patch.dict(
            config.app,
            self._keys(
                pexels_api_keys=[],
                smart_material_provider_cascade="off",
            ),
        ):
            self.assertEqual(material.smart_provider_chain("pexels"), ["pexels"])

    def test_only_searchable_stock_providers_support_smart_matching(self):
        for provider in ("pinterest", "pexels", "Pixabay"):
            self.assertTrue(material.supports_smart_visual_matching(provider))
        for provider in ("", "local", "generated", None, "coverr", " coverr "):
            self.assertFalse(material.supports_smart_visual_matching(provider))
        self.assertEqual(material.smart_provider_chain("local"), [])
        self.assertEqual(material.smart_provider_chain("coverr"), [])

    def test_the_query_variant_cap_reads_as_a_positive_whole_number(self):
        # Both halves of the retry read this knob: the script stage decides how
        # many phrasings to generate and material selection decides how many to
        # spend, so a value it cannot parse must degrade to a working default
        # rather than to zero searches.
        # Read against a config that has no such key, so the built-in default is
        # asserted instead of whatever the local config.toml happens to hold.
        without_key = dict(config.app)
        without_key.pop("smart_material_max_query_variants", None)
        with patch.dict(config.app, without_key, clear=True):
            self.assertEqual(material.max_query_variants_per_provider(), 3)
        for value, expected in (
            (1, 1),
            (5, 5),
            ("2", 2),
            (" 4 ", 4),
            (0, 1),
            (-7, 1),
            ("many", 3),
            (None, 3),
        ):
            with self.subTest(value=value):
                with patch.dict(
                    config.app, {"smart_material_max_query_variants": value}
                ):
                    self.assertEqual(
                        material.max_query_variants_per_provider(), expected
                    )


class TestVisualBeatMaterialSelection(unittest.TestCase):
    """Per-beat provider cascade and the beat-to-source-window binding."""

    def setUp(self):
        self.requirement_patcher = patch.object(
            material.llm,
            "generate_visual_requirement_specs",
            side_effect=lambda requirements: {
                material.llm.normalize_visual_requirement(requirement): (
                    _requirement_spec(requirement)
                )
                for requirement in requirements
            },
        )
        self.requirement_patcher.start()
        _patch_requirement_rewrite(self)

    def tearDown(self):
        self.requirement_patcher.stop()

    @staticmethod
    def _settings(**overrides):
        settings = {
            "batch_size": 5,
            "max_candidates": 15,
            "minimum_score": 0.7,
            "strong_early_stop_score": 0.9,
            "preferred_max_source_duration": 30,
            "concurrency": 5,
            "fail_closed": True,
        }
        settings.update(overrides)
        return settings

    @staticmethod
    def _provider_candidate(provider, asset_id):
        candidate = _candidate(
            asset_id, url=f"https://videos.example/{provider}-{asset_id}.mp4"
        )
        candidate.provider = provider
        candidate.source_info.update({"provider": provider, "asset_id": asset_id})
        return candidate

    @staticmethod
    def _beat_record(beat, *, local_file, source_start, source_end, **overrides):
        record = {
            "provider": "pexels",
            "local_file": local_file,
            "visual_beat_index": beat.index,
            "semantic_group_id": beat.semantic_group_id,
            "source_start_time": source_start,
            "source_end_time": source_end,
        }
        record.update(overrides)
        return record

    def test_beat_falls_through_to_the_next_provider_after_a_reject(self):
        beat = _visual_beat()
        weak = self._provider_candidate("pexels", "weak")
        strong = self._provider_candidate("pixabay", "strong")
        searched: list[str] = []

        def pexels_search(**kwargs):
            searched.append("pexels")
            return [weak]

        def pixabay_search(**kwargs):
            searched.append("pixabay")
            return [strong]

        def select_best(**kwargs):
            candidate = kwargs["candidates"][0]
            stats = {
                "api_candidates_analyzed": 1,
                "source_seconds_analyzed": 12.0,
                "candidate_evaluations": [],
            }
            if candidate.provider == "pexels":
                return None, stats
            candidate.overall_score = 0.93
            return candidate, stats

        segmenter = MagicMock(
            return_value={
                "source_start_time": 5.0,
                "source_end_time": 8.0,
                "description": "beans drying under the sun",
            }
        )
        service = SimpleNamespace(
            candidate_selection_settings=self._settings,
            select_best_candidate=MagicMock(side_effect=select_best),
            segment_winner=segmenter,
        )

        with (
            patch.object(material, "save_video", return_value="D:/task/beat.mp4"),
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ) as persist,
        ):
            paths = material._download_videos_by_script_order_smart(
                task_id="cascade-beat",
                search_terms=[beat.search_queries[0]],
                visual_beats=[beat],
                provider_searches=[
                    ("pexels", pexels_search),
                    ("pixabay", pixabay_search),
                ],
                video_aspect=VideoAspect.portrait,
                max_clip_duration=4,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )

        self.assertEqual(paths, ["D:/task/beat.mp4"])
        self.assertEqual(searched, ["pexels", "pixabay"])
        # Only the winner is segmented, so the cascade costs one extra search,
        # not an extra temporal segmentation call.
        self.assertEqual(segmenter.call_count, 1)
        self.assertEqual(segmenter.call_args.kwargs["video_url"], strong.url)
        record = persist.call_args.kwargs["material_sources"][0]
        self.assertEqual(record["provider"], "pixabay")
        self.assertEqual(record["visual_beat_index"], 1)
        self.assertEqual(record["semantic_group_id"], 1)
        self.assertAlmostEqual(
            record["source_end_time"] - record["source_start_time"],
            beat.duration,
            places=3,
        )
        runs = persist.call_args.kwargs["semantic_verifier_runs"]
        self.assertEqual(
            [(run["stock_provider"], run["final_decision"]) for run in runs],
            [("pexels", "REJECT"), ("pixabay", "ACCEPT")],
        )

    def test_explicit_fail_open_does_not_pay_for_the_rest_of_the_cascade(self):
        beat = _visual_beat()
        weak = self._provider_candidate("pexels", "weak")
        searched: list[str] = []

        def pexels_search(**kwargs):
            searched.append("pexels")
            return [weak]

        def pixabay_search(**kwargs):
            searched.append("pixabay")
            return [self._provider_candidate("pixabay", "strong")]

        service = SimpleNamespace(
            candidate_selection_settings=lambda: self._settings(fail_closed=False),
            select_best_candidate=MagicMock(
                return_value=(
                    None,
                    {
                        "api_candidates_analyzed": 1,
                        "source_seconds_analyzed": 12.0,
                        "candidate_evaluations": [],
                    },
                )
            ),
            segment_winner=MagicMock(
                return_value={
                    "source_start_time": 5.0,
                    "source_end_time": 8.0,
                    "description": "fail-open fallback",
                }
            ),
        )

        with (
            patch.object(material, "save_video", return_value="D:/task/beat.mp4"),
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ) as persist,
        ):
            paths = material._download_videos_by_script_order_smart(
                task_id="fail-open-beat",
                search_terms=[beat.search_queries[0]],
                visual_beats=[beat],
                provider_searches=[
                    ("pexels", pexels_search),
                    ("pixabay", pixabay_search),
                ],
                video_aspect=VideoAspect.portrait,
                max_clip_duration=4,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )

        self.assertEqual(paths, ["D:/task/beat.mp4"])
        self.assertEqual(searched, ["pexels"])
        runs = persist.call_args.kwargs["semantic_verifier_runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["final_decision"], "FAIL_OPEN_FALLBACK")

    def test_every_provider_failing_names_each_provider_in_the_error(self):
        beat = _visual_beat()
        service = SimpleNamespace(
            candidate_selection_settings=self._settings,
            select_best_candidate=MagicMock(),
            segment_winner=MagicMock(),
        )

        with (
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ) as persist,
            self.assertRaisesRegex(
                material.SmartMaterialSelectionError,
                r"provider=pexels:.*provider=pixabay:",
            ),
        ):
            material._download_videos_by_script_order_smart(
                task_id="cascade-exhausted",
                search_terms=[beat.search_queries[0]],
                visual_beats=[beat],
                provider_searches=[
                    ("pexels", lambda **kwargs: []),
                    ("pixabay", lambda **kwargs: []),
                ],
                video_aspect=VideoAspect.portrait,
                max_clip_duration=4,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )

        service.select_best_candidate.assert_not_called()
        # Both providers report NO_CANDIDATES. The requirement-rewrite rung would
        # run next, but this beat is the one that opens the video, so it is withheld
        # and the beat is failed rather than re-described into something else.
        self.assertEqual(
            [
                run["final_decision"]
                for run in persist.call_args.kwargs["semantic_verifier_runs"]
            ],
            ["NO_CANDIDATES", "NO_CANDIDATES", "OPENING_SHOT_REWRITE_WITHHELD"],
        )

    def test_render_segments_join_records_to_beats_by_beat_index(self):
        first = _visual_beat(index=1)
        second = _visual_beat(index=2)
        payload = {
            "material_sources": [
                self._beat_record(
                    second,
                    local_file="second.mp4",
                    source_start=1.0,
                    source_end=3.8,
                ),
                self._beat_record(
                    first,
                    local_file="first.mp4",
                    source_start=5.1,
                    source_end=7.9,
                ),
            ]
        }

        with patch.object(
            material.task_artifacts, "read_script_data", return_value=payload
        ):
            segments = material.load_render_segments(
                "beat-task",
                ["D:/task/first.mp4", "D:/task/second.mp4"],
                [first, second],
                clip_speed=1.0,
                audio_duration=second.end_time,
            )

        self.assertEqual([segment.index for segment in segments], [1, 2])
        self.assertEqual([segment.visual_beat_index for segment in segments], [1, 2])
        self.assertEqual(segments[0].file_path, "D:/task/first.mp4")
        self.assertEqual(segments[1].file_path, "D:/task/second.mp4")
        self.assertAlmostEqual(segments[0].source_start, 5.1)
        self.assertAlmostEqual(segments[1].source_start, 1.0)
        self.assertAlmostEqual(segments[0].target_start, 0.0)
        self.assertAlmostEqual(segments[1].target_start, first.end_time)
        self.assertEqual(
            [segment.provider for segment in segments], ["pexels", "pexels"]
        )

    def test_render_segments_reject_records_that_predate_the_beat_timeline(self):
        beat = _visual_beat()
        record = self._beat_record(
            beat, local_file="first.mp4", source_start=5.1, source_end=7.9
        )
        record.pop("visual_beat_index")

        with (
            patch.object(
                material.task_artifacts,
                "read_script_data",
                return_value={"material_sources": [record]},
            ),
            self.assertRaisesRegex(ValueError, "predate the visual beat timeline"),
        ):
            material.load_render_segments("beat-task", ["D:/task/first.mp4"], [beat])

    def test_render_segments_reject_two_records_for_one_beat(self):
        beat = _visual_beat()
        payload = {
            "material_sources": [
                self._beat_record(
                    beat, local_file="first.mp4", source_start=5.1, source_end=7.9
                ),
                self._beat_record(
                    beat, local_file="other.mp4", source_start=0.0, source_end=2.8
                ),
            ]
        }

        with (
            patch.object(
                material.task_artifacts, "read_script_data", return_value=payload
            ),
            self.assertRaisesRegex(ValueError, "ambiguous for beat 1"),
        ):
            material.load_render_segments("beat-task", ["D:/task/first.mp4"], [beat])

    def test_render_segments_reject_a_record_pointing_at_another_file(self):
        beat = _visual_beat()
        payload = {
            "material_sources": [
                self._beat_record(
                    beat, local_file="other.mp4", source_start=5.1, source_end=7.9
                )
            ]
        }

        with (
            patch.object(
                material.task_artifacts, "read_script_data", return_value=payload
            ),
            self.assertRaisesRegex(ValueError, "points at 'other.mp4'"),
        ):
            material.load_render_segments("beat-task", ["D:/task/first.mp4"], [beat])

    def test_render_segments_reject_a_window_selected_for_another_duration(self):
        beat = _visual_beat()
        payload = {
            "material_sources": [
                self._beat_record(
                    beat, local_file="first.mp4", source_start=1.0, source_end=5.0
                )
            ]
        }

        with (
            patch.object(
                material.task_artifacts, "read_script_data", return_value=payload
            ),
            self.assertRaisesRegex(ValueError, "renders 4.000s at 1.00x"),
        ):
            material.load_render_segments("beat-task", ["D:/task/first.mp4"], [beat])

    def test_render_segments_reject_a_timeline_that_does_not_start_at_zero(self):
        beat = _visual_beat(start_time=0.4)
        payload = {
            "material_sources": [
                self._beat_record(
                    beat, local_file="first.mp4", source_start=5.1, source_end=7.9
                )
            ]
        }

        with (
            patch.object(
                material.task_artifacts, "read_script_data", return_value=payload
            ),
            self.assertRaisesRegex(ValueError, "must start at 0.0s"),
        ):
            material.load_render_segments("beat-task", ["D:/task/first.mp4"], [beat])

    def test_render_segments_reject_a_gapped_beat_timeline(self):
        first = _visual_beat(index=1)
        second = _visual_beat(index=2, start_time=3.5)
        payload = {
            "material_sources": [
                self._beat_record(
                    first,
                    local_file="first.mp4",
                    source_start=5.1,
                    source_end=7.9,
                ),
                self._beat_record(
                    second,
                    local_file="second.mp4",
                    source_start=1.0,
                    source_end=3.8,
                ),
            ]
        }

        with (
            patch.object(
                material.task_artifacts, "read_script_data", return_value=payload
            ),
            self.assertRaisesRegex(ValueError, "beat 2 leaves a gap or overlap"),
        ):
            material.load_render_segments(
                "beat-task",
                ["D:/task/first.mp4", "D:/task/second.mp4"],
                [first, second],
            )

    def test_render_segments_reject_a_timeline_shorter_than_the_audio(self):
        beat = _visual_beat()
        payload = {
            "material_sources": [
                self._beat_record(
                    beat, local_file="first.mp4", source_start=5.1, source_end=7.9
                )
            ]
        }

        with (
            patch.object(
                material.task_artifacts, "read_script_data", return_value=payload
            ),
            self.assertRaisesRegex(ValueError, "does not cover the narration audio"),
        ):
            material.load_render_segments(
                "beat-task",
                ["D:/task/first.mp4"],
                [beat],
                audio_duration=5.0,
            )

    def test_render_segments_scale_the_source_window_with_playback_speed(self):
        beat = _visual_beat(index=1, duration=4.0)
        payload = {
            "material_sources": [
                self._beat_record(
                    beat, local_file="fast.mp4", source_start=2.0, source_end=10.0
                )
            ]
        }

        with patch.object(
            material.task_artifacts, "read_script_data", return_value=payload
        ):
            segments = material.load_render_segments(
                "beat-task",
                ["D:/task/fast.mp4"],
                [beat],
                clip_speed=2.0,
                audio_duration=4.0,
            )

        self.assertEqual(segments[0].playback_speed, 2.0)
        self.assertAlmostEqual(segments[0].source_duration, 8.0)
        self.assertAlmostEqual(segments[0].target_duration, 4.0)


class TestScriptStageRequirementChecklist(unittest.TestCase):
    """The checklist is planned in the script stage; selection reuses it as-is.

    What matters here is that verification gates on the same decomposition the run
    was planned and persisted with, and that a run never pays twice for it.
    """

    def setUp(self):
        _patch_requirement_rewrite(self)

    @staticmethod
    def _settings(**overrides):
        settings = {
            "batch_size": 5,
            "max_candidates": 15,
            "minimum_score": 0.7,
            "strong_early_stop_score": 0.9,
            "preferred_max_source_duration": 30,
            "concurrency": 5,
            "fail_closed": True,
        }
        settings.update(overrides)
        return settings

    def _select_every_beat(self, beats, requirement_specs):
        """Run smart selection where every beat wins on the first provider."""
        used_specs: list[VisualRequirementSpec | None] = []

        def search(**kwargs):
            return [_candidate(kwargs["search_term"])]

        def select_best(**kwargs):
            used_specs.append(kwargs["requirement_spec"])
            candidate = kwargs["candidates"][0]
            candidate.overall_score = 0.95
            return candidate, {
                "api_candidates_analyzed": 1,
                "source_seconds_analyzed": 12.0,
                "candidate_evaluations": [],
            }

        service = SimpleNamespace(
            candidate_selection_settings=self._settings,
            select_best_candidate=MagicMock(side_effect=select_best),
            segment_winner=MagicMock(
                return_value={
                    "source_start_time": 5.0,
                    "source_end_time": 8.0,
                    "description": "the requested action",
                }
            ),
        )
        saved = 0

        def save_video(**kwargs):
            nonlocal saved
            saved += 1
            return f"D:/task/beat-{saved}.mp4"

        with (
            patch.object(material, "save_video", side_effect=save_video),
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ),
        ):
            paths = material._download_videos_by_script_order_smart(
                task_id="checklist-beats",
                search_terms=[beat.search_queries[0] for beat in beats],
                visual_beats=beats,
                provider_searches=[("pexels", search)],
                video_aspect=VideoAspect.portrait,
                max_clip_duration=4,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
                requirement_specs=requirement_specs,
            )

        self.assertEqual(len(paths), len(beats))
        return used_specs

    def test_supplied_checklist_is_not_requested_from_the_provider_again(self):
        beats = [
            _visual_beat(
                index=1,
                semantic_group_id=1,
                requirement="Worker digs a hole",
                query="worker digging hole",
            ),
            _visual_beat(
                index=2,
                semantic_group_id=2,
                requirement="Rain floods a dry field",
                query="rain flooding dry field",
            ),
        ]
        checklist = {
            material.llm.normalize_visual_requirement(beat.visual_requirement): (
                _requirement_spec(beat.visual_requirement)
            )
            for beat in beats
        }

        with patch.object(
            material.llm, "generate_visual_requirement_specs"
        ) as decompose:
            used_specs = self._select_every_beat(beats, checklist)

        decompose.assert_not_called()
        # Identity, not equality: the object the run was planned with is the object
        # verification gates on.
        self.assertIs(used_specs[0], checklist[list(checklist)[0]])
        self.assertEqual(
            [spec.original_requirement for spec in used_specs],
            [beat.visual_requirement for beat in beats],
        )

    def test_only_requirements_absent_from_the_checklist_are_decomposed(self):
        beats = [
            _visual_beat(
                index=1,
                semantic_group_id=1,
                requirement="Worker digs a hole",
                query="worker digging hole",
            ),
            _visual_beat(
                index=2,
                semantic_group_id=2,
                requirement="Rain floods a dry field",
                query="rain flooding dry field",
            ),
        ]
        planned = beats[0].visual_requirement
        checklist = {
            material.llm.normalize_visual_requirement(planned): _requirement_spec(
                planned
            )
        }

        with patch.object(
            material.llm,
            "generate_visual_requirement_specs",
            side_effect=lambda requirements: {
                material.llm.normalize_visual_requirement(requirement): (
                    _requirement_spec(requirement)
                )
                for requirement in requirements
            },
        ) as decompose:
            used_specs = self._select_every_beat(beats, checklist)

        decompose.assert_called_once_with([beats[1].visual_requirement])
        self.assertIs(used_specs[0], checklist[list(checklist)[0]])
        self.assertEqual(used_specs[1].original_requirement, beats[1].visual_requirement)

    def test_without_a_checklist_each_requirement_is_requested_once(self):
        # Sibling shots of one semantic group share a requirement. Callers that
        # supply no checklist (the API and the legacy path) must still work, and
        # must not pay per beat for a decomposition they only need per requirement.
        shared = "Worker digs a hole"
        beats = [
            _visual_beat(
                index=1,
                semantic_group_id=1,
                requirement=shared,
                query="worker digging hole",
            ),
            _visual_beat(
                index=2,
                semantic_group_id=1,
                requirement=shared,
                query="shovel breaking soil",
            ),
            _visual_beat(
                index=3,
                semantic_group_id=2,
                requirement="Rain floods a dry field",
                query="rain flooding dry field",
            ),
        ]

        with patch.object(
            material.llm,
            "generate_visual_requirement_specs",
            side_effect=lambda requirements: {
                material.llm.normalize_visual_requirement(requirement): (
                    _requirement_spec(requirement)
                )
                for requirement in requirements
            },
        ) as decompose:
            used_specs = self._select_every_beat(beats, None)

        decompose.assert_called_once_with([shared, beats[2].visual_requirement])
        self.assertIs(used_specs[0], used_specs[1])
        self.assertEqual(used_specs[2].original_requirement, beats[2].visual_requirement)


class TestPerProviderQueryVariants(unittest.TestCase):
    """A beat spends its alternative phrasings before the cascade changes catalog.

    A beat that finds nothing usable is more often phrased badly than missing from
    the catalog, and the script stage already generated alternative phrasings for
    it. These tests also hold the cost line: no phrasing may re-analyze a candidate
    this beat already rejected, and the number of phrasings stays bounded.
    """

    def setUp(self):
        self.requirement_patcher = patch.object(
            material.llm,
            "generate_visual_requirement_specs",
            side_effect=lambda requirements: {
                material.llm.normalize_visual_requirement(requirement): (
                    _requirement_spec(requirement)
                )
                for requirement in requirements
            },
        )
        self.requirement_patcher.start()
        self.addCleanup(self.requirement_patcher.stop)
        _patch_requirement_rewrite(self)

    @staticmethod
    def _settings(**overrides):
        settings = {
            "batch_size": 5,
            "max_candidates": 15,
            "minimum_score": 0.7,
            "strong_early_stop_score": 0.9,
            "preferred_max_source_duration": 30,
            "concurrency": 5,
            "fail_closed": True,
        }
        settings.update(overrides)
        return settings

    @staticmethod
    def _beat_with_phrasings(*queries):
        beat = _visual_beat(requirement="Worker digs a hole", query=queries[0])
        beat.search_queries = list(queries)
        return beat

    @staticmethod
    def _candidate_for(provider, key):
        candidate = _candidate(key, url=f"https://videos.example/{provider}-{key}.mp4")
        candidate.provider = provider
        candidate.source_info.update({"provider": provider, "asset_id": key})
        return candidate

    @staticmethod
    def _service(select_best, **overrides):
        service = SimpleNamespace(
            candidate_selection_settings=TestPerProviderQueryVariants._settings,
            select_best_candidate=MagicMock(side_effect=select_best),
            segment_winner=MagicMock(
                return_value={
                    "source_start_time": 5.0,
                    "source_end_time": 8.0,
                    "description": "the requested action",
                }
            ),
        )
        for name, value in overrides.items():
            setattr(service, name, value)
        return service

    @staticmethod
    def _stats(candidates):
        return {
            "api_candidates_analyzed": len(candidates),
            "source_seconds_analyzed": 12.0 * len(candidates),
            "candidate_evaluations": [
                {
                    "provider": candidate.provider,
                    "provider_asset_id": candidate.source_info["asset_id"],
                }
                for candidate in candidates
            ],
        }

    def _select(self, beat, provider_searches, service, *, variants=3):
        with (
            patch.dict(
                config.app,
                {"smart_material_max_query_variants": variants},
            ),
            patch.object(material, "save_video", return_value="D:/task/beat.mp4"),
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ) as persist,
        ):
            paths = material._download_videos_by_script_order_smart(
                task_id="query-variants",
                search_terms=[beat.search_queries[0]],
                visual_beats=[beat],
                provider_searches=provider_searches,
                video_aspect=VideoAspect.portrait,
                max_clip_duration=4,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )
        return paths, persist

    def test_the_next_phrasing_is_tried_before_the_next_provider(self):
        beat = self._beat_with_phrasings(
            "worker digging hole", "shovel breaking dry soil"
        )
        searched: list[tuple[str, str]] = []

        def provider_search(provider):
            def search(**kwargs):
                searched.append((provider, kwargs["search_term"]))
                return [self._candidate_for(provider, kwargs["search_term"])]

            return search

        def select_best(**kwargs):
            candidates = kwargs["candidates"]
            if kwargs["search_query"] == beat.search_queries[0]:
                return None, self._stats(candidates)
            candidates[0].overall_score = 0.93
            return candidates[0], self._stats(candidates)

        service = self._service(select_best)
        paths, persist = self._select(
            beat,
            [
                ("pexels", provider_search("pexels")),
                ("pixabay", provider_search("pixabay")),
            ],
            service,
        )

        self.assertEqual(paths, ["D:/task/beat.mp4"])
        # The second phrasing is spent on the provider that is already configured
        # and cached, before the thinner catalog is asked anything at all.
        self.assertEqual(
            searched,
            [
                ("pexels", beat.search_queries[0]),
                ("pexels", beat.search_queries[1]),
            ],
        )
        # Only the winner is segmented; another phrasing costs a search and an
        # analysis round, never an extra temporal segmentation call.
        self.assertEqual(service.segment_winner.call_count, 1)
        record = persist.call_args.kwargs["material_sources"][0]
        self.assertEqual(record["provider"], "pexels")
        # Provenance records the phrasing that actually won, not the planned one.
        self.assertEqual(record["search_term"], beat.search_queries[1])
        runs = persist.call_args.kwargs["semantic_verifier_runs"]
        self.assertEqual(
            [(run["search_query"], run["final_decision"]) for run in runs],
            [
                (beat.search_queries[0], "REJECT"),
                (beat.search_queries[1], "ACCEPT"),
            ],
        )

    def test_a_rejected_candidate_is_not_analyzed_again_under_the_next_phrasing(self):
        beat = self._beat_with_phrasings(
            "worker digging hole", "shovel breaking dry soil"
        )
        rejected = self._candidate_for("pexels", "already-rejected")
        fresh = self._candidate_for("pexels", "fresh")
        analyzed: list[list[str]] = []

        def pexels_search(**kwargs):
            if kwargs["search_term"] == beat.search_queries[0]:
                return [rejected]
            # A different phrasing of the same shot returns most of the same
            # catalog, which is exactly where a second bill would come from.
            return [rejected, fresh]

        def select_best(**kwargs):
            candidates = kwargs["candidates"]
            analyzed.append([candidate.url for candidate in candidates])
            if kwargs["search_query"] == beat.search_queries[0]:
                return None, self._stats(candidates)
            candidates[0].overall_score = 0.93
            return candidates[0], self._stats(candidates)

        service = self._service(select_best)
        paths, persist = self._select(beat, [("pexels", pexels_search)], service)

        self.assertEqual(paths, ["D:/task/beat.mp4"])
        # The verdict on the rejected asset is already paid for, so the second
        # phrasing only analyzes what the first one never looked at.
        self.assertEqual(analyzed, [[rejected.url], [fresh.url]])
        self.assertEqual(
            persist.call_args.kwargs["material_sources"][0]["asset_id"], "fresh"
        )

    def test_the_variant_cap_bounds_the_phrasings_per_provider(self):
        beat = self._beat_with_phrasings(
            "worker digging hole",
            "shovel breaking dry soil",
            "hands moving wet earth",
        )
        searched: list[str] = []

        def pexels_search(**kwargs):
            searched.append(kwargs["search_term"])
            return [self._candidate_for("pexels", kwargs["search_term"])]

        def select_best(**kwargs):
            return None, self._stats(kwargs["candidates"])

        service = self._service(select_best)
        with self.assertRaises(material.SmartMaterialSelectionError) as failure:
            self._select(beat, [("pexels", pexels_search)], service, variants=2)

        self.assertEqual(searched, beat.search_queries[:2])
        # The failure has to name the phrasings that were tried, or the operator
        # cannot tell a wording problem from an empty catalog.
        self.assertIn(f"query={beat.search_queries[0]!r}", str(failure.exception))
        self.assertIn(f"query={beat.search_queries[1]!r}", str(failure.exception))
        self.assertNotIn(beat.search_queries[2], str(failure.exception))

    def test_a_cap_of_one_spends_a_single_phrasing_across_the_cascade(self):
        beat = self._beat_with_phrasings(
            "worker digging hole", "shovel breaking dry soil"
        )
        searched: list[tuple[str, str]] = []

        def provider_search(provider):
            def search(**kwargs):
                searched.append((provider, kwargs["search_term"]))
                return [self._candidate_for(provider, kwargs["search_term"])]

            return search

        def select_best(**kwargs):
            return None, self._stats(kwargs["candidates"])

        service = self._service(select_best)
        with self.assertRaises(material.SmartMaterialSelectionError) as failure:
            self._select(
                beat,
                [
                    ("pexels", provider_search("pexels")),
                    ("pixabay", provider_search("pixabay")),
                ],
                service,
                variants=1,
            )

        # A cap of one is the previous behavior: one query for the whole cascade,
        # and a failure message that still reads as a per-provider report.
        self.assertEqual(
            searched,
            [
                ("pexels", beat.search_queries[0]),
                ("pixabay", beat.search_queries[0]),
            ],
        )
        self.assertNotIn("query=", str(failure.exception))
        self.assertIn("provider=pexels:", str(failure.exception))

    def test_uniformly_unrelated_footage_skips_this_provider_remaining_phrasings(self):
        # Rewording is the right rung when a catalog holds the concept and the
        # phrasing missed it. When the catalog's own ranking puts nothing close,
        # every rewording re-searches that same catalog and buys a pool that
        # overlaps almost completely -- which is how one beat burned ninety
        # analyses. A different catalog is different evidence, so the cascade
        # still moves on.
        beat = self._beat_with_phrasings(
            "worker digging hole",
            "shovel breaking dry soil",
            "hands moving wet earth",
        )
        searched: list[tuple[str, str]] = []

        def provider_search(provider):
            def search(**kwargs):
                searched.append((provider, kwargs["search_term"]))
                return [self._candidate_for(provider, kwargs["search_term"])]

            return search

        def select_best(**kwargs):
            candidates = kwargs["candidates"]
            if candidates[0].provider == "pexels":
                return None, {
                    **self._stats(candidates),
                    "unrelated_footage": True,
                }
            candidates[0].overall_score = 0.93
            return candidates[0], self._stats(candidates)

        service = self._service(select_best)
        paths, persist = self._select(
            beat,
            [
                ("pexels", provider_search("pexels")),
                ("pixabay", provider_search("pixabay")),
            ],
            service,
        )

        self.assertEqual(paths, ["D:/task/beat.mp4"])
        self.assertEqual(
            searched,
            [
                ("pexels", beat.search_queries[0]),
                ("pixabay", beat.search_queries[0]),
            ],
        )
        self.assertEqual(
            persist.call_args.kwargs["material_sources"][0]["provider"], "pixabay"
        )

    def test_both_catalogs_unrelated_fails_after_one_phrasing_each(self):
        # The ceiling for a concept neither catalog carries: two searches out of a
        # possible six, and a failure that still names the phrasing it judged so
        # the operator can tell an empty catalog from a bad wording.
        beat = self._beat_with_phrasings(
            "worker digging hole",
            "shovel breaking dry soil",
            "hands moving wet earth",
        )
        searched: list[tuple[str, str]] = []

        def provider_search(provider):
            def search(**kwargs):
                searched.append((provider, kwargs["search_term"]))
                return [self._candidate_for(provider, kwargs["search_term"])]

            return search

        def select_best(**kwargs):
            candidates = kwargs["candidates"]
            return None, {**self._stats(candidates), "unrelated_footage": True}

        service = self._service(select_best)
        with self.assertRaises(material.SmartMaterialSelectionError) as failure:
            self._select(
                beat,
                [
                    ("pexels", provider_search("pexels")),
                    ("pixabay", provider_search("pixabay")),
                ],
                service,
            )

        self.assertEqual(
            searched,
            [
                ("pexels", beat.search_queries[0]),
                ("pixabay", beat.search_queries[0]),
            ],
        )
        self.assertIn("provider=pexels", str(failure.exception))
        self.assertIn("provider=pixabay", str(failure.exception))
        self.assertIn(f"query={beat.search_queries[0]!r}", str(failure.exception))
        self.assertNotIn(beat.search_queries[1], str(failure.exception))


class TestApprovedAlternatePromotion(unittest.TestCase):
    """A verified winner that will not download must not cost the whole video.

    Verification already paid for a verdict on every candidate it analyzed, so a
    beat whose top-ranked asset fails to transfer can render an approved
    runner-up instead of aborting the render. The line these tests hold is that
    only *approved* candidates are promoted: a rejected one reaching the timeline
    this way would quietly undo the fail-closed policy.
    """

    def setUp(self):
        self.requirement_patcher = patch.object(
            material.llm,
            "generate_visual_requirement_specs",
            side_effect=lambda requirements: {
                material.llm.normalize_visual_requirement(requirement): (
                    _requirement_spec(requirement)
                )
                for requirement in requirements
            },
        )
        self.requirement_patcher.start()
        self.addCleanup(self.requirement_patcher.stop)
        _patch_requirement_rewrite(self)

    @staticmethod
    def _settings():
        return {
            "batch_size": 5,
            "max_candidates": 15,
            "minimum_score": 0.7,
            "strong_early_stop_score": 0.9,
            "preferred_max_source_duration": 30,
            "concurrency": 5,
            "fail_closed": True,
        }

    @staticmethod
    def _candidate_for(asset_id):
        candidate = _candidate(
            asset_id, url=f"https://videos.example/pexels-{asset_id}.mp4"
        )
        candidate.source_info.update({"provider": "pexels", "asset_id": asset_id})
        return candidate

    @staticmethod
    def _stats(verdicts):
        """Verification statistics for ``(candidate, accepted)`` pairs, best first."""
        return {
            "api_candidates_analyzed": len(verdicts),
            "source_seconds_analyzed": 12.0 * len(verdicts),
            "candidate_evaluations": [
                {
                    "provider": candidate.provider,
                    "provider_asset_id": candidate.source_info["asset_id"],
                    "accepted": accepted,
                    "ranking_position": position,
                    "overall_score": 0.93 if accepted else 0.41,
                }
                for position, (candidate, accepted) in enumerate(verdicts, start=1)
            ],
        }

    def _run(self, *, candidates, verdicts, winner, transfers, segments):
        beat = _visual_beat(
            requirement="Worker digs a hole", query="worker digging hole"
        )

        def select_best(**kwargs):
            winner.overall_score = 0.93
            return winner, self._stats(verdicts)

        service = SimpleNamespace(
            candidate_selection_settings=self._settings,
            select_best_candidate=MagicMock(side_effect=select_best),
            segment_winner=MagicMock(side_effect=list(segments)),
        )
        save_video = MagicMock(
            side_effect=lambda **kwargs: transfers.get(kwargs["video_url"], "")
        )
        error = None
        paths: list[str] = []
        with (
            patch.object(material, "save_video", save_video),
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ) as persist,
        ):
            try:
                paths = material._download_videos_by_script_order_smart(
                    task_id="promote-approved-alternate",
                    search_terms=[beat.search_queries[0]],
                    visual_beats=[beat],
                    provider_searches=[("pexels", lambda **kwargs: list(candidates))],
                    video_aspect=VideoAspect.portrait,
                    max_clip_duration=4,
                    material_directory="",
                    clip_speed=1.0,
                    twelvelabs_service=service,
                )
            except material.SmartMaterialSelectionError as exc:
                # Captured rather than raised, so a failing beat can still be
                # inspected for what it spent before it gave up.
                error = exc
        return SimpleNamespace(
            paths=paths,
            error=error,
            persist=persist,
            service=service,
            save_video=save_video,
            beat=beat,
        )

    def test_an_approved_runner_up_is_promoted_when_the_winner_will_not_download(self):
        winner = self._candidate_for("top-ranked")
        alternate = self._candidate_for("approved-runner-up")
        outcome = self._run(
            candidates=[winner, alternate],
            verdicts=[(winner, True), (alternate, True)],
            winner=winner,
            transfers={alternate.url: "D:/task/promoted.mp4"},
            segments=[
                {
                    "source_start_time": 5.0,
                    "source_end_time": 7.8,
                    "description": "the requested action",
                },
                {
                    "source_start_time": 1.0,
                    "source_end_time": 3.8,
                    "description": "the requested action, promoted asset",
                },
            ],
        )

        self.assertIsNone(outcome.error)
        # The beat still produces exactly one clip, so the render timeline stays
        # one record per beat and the video is not lost to a transport error.
        self.assertEqual(outcome.paths, ["D:/task/promoted.mp4"])
        record = outcome.persist.call_args.kwargs["material_sources"][0]
        self.assertEqual(record["asset_id"], "approved-runner-up")
        # The promoted asset renders its own verified window, never the window
        # that was computed for the asset that failed to transfer.
        self.assertEqual(record["source_start_time"], 1.0)
        self.assertEqual(record["source_end_time"], 3.8)
        self.assertEqual(record["visual_beat_index"], 1)
        # The promotion costs one source window, not a second round of analysis:
        # the runner-up's verdict was already bought and paid for.
        self.assertEqual(outcome.service.select_best_candidate.call_count, 1)
        self.assertEqual(
            [
                segmented.kwargs["video_url"]
                for segmented in outcome.service.segment_winner.call_args_list
            ],
            [winner.url, alternate.url],
        )
        runs = outcome.persist.call_args.kwargs["semantic_verifier_runs"]
        self.assertEqual(
            [run["final_decision"] for run in runs],
            ["ACCEPT", "WINNER_DOWNLOAD_SUBSTITUTED"],
        )
        # Provenance has to say the rendered asset was not the ranked winner, or
        # a later quality review cannot explain the clip it is looking at.
        self.assertEqual(runs[-1]["promoted_plan_position"], 2)
        self.assertEqual(runs[-1]["visual_item_index"], 1)

    def test_a_rejected_candidate_is_never_promoted_after_a_failed_download(self):
        winner = self._candidate_for("top-ranked")
        rejected = self._candidate_for("rejected")
        outcome = self._run(
            candidates=[winner, rejected],
            verdicts=[(winner, True), (rejected, False)],
            winner=winner,
            transfers={},
            segments=[
                {
                    "source_start_time": 5.0,
                    "source_end_time": 7.8,
                    "description": "the requested action",
                }
            ],
        )

        # Footage verification refused is not a fallback. The beat fails, and the
        # rejected asset is never segmented and never even requested.
        self.assertEqual(outcome.paths, [])
        self.assertIsNotNone(outcome.error)
        self.assertEqual(
            [
                transfer.kwargs["video_url"]
                for transfer in outcome.save_video.call_args_list
            ],
            [winner.url],
        )
        self.assertEqual(outcome.service.segment_winner.call_count, 1)
        self.assertNotIn(
            "WINNER_DOWNLOAD_SUBSTITUTED",
            [
                run.get("final_decision")
                for run in outcome.persist.call_args.kwargs["semantic_verifier_runs"]
            ],
        )

    def test_without_an_approved_alternate_the_failure_message_is_unchanged(self):
        winner = self._candidate_for("top-ranked")
        outcome = self._run(
            candidates=[winner],
            verdicts=[(winner, True)],
            winner=winner,
            transfers={},
            segments=[
                {
                    "source_start_time": 5.0,
                    "source_end_time": 7.8,
                    "description": "the requested action",
                }
            ],
        )

        # With nothing to promote the behavior is exactly the previous behavior,
        # down to the message an operator reads in the failed task.
        self.assertEqual(
            str(outcome.error),
            "The selected pexels winner for visual beat 1 could not be downloaded",
        )

    def test_alternates_are_ranked_deduplicated_and_limited_to_this_search(self):
        winner = self._candidate_for("top-ranked")
        second = self._candidate_for("second")
        third = self._candidate_for("third")
        rejected = self._candidate_for("rejected")
        alternates = material._approved_alternate_candidates(
            [winner, second, third, rejected],
            [
                {
                    "provider": "pexels",
                    "provider_asset_id": "top-ranked",
                    "accepted": True,
                    "ranking_position": 1,
                },
                {
                    "provider": "pexels",
                    "provider_asset_id": "third",
                    "accepted": True,
                    "ranking_position": 3,
                },
                {
                    "provider": "pexels",
                    "provider_asset_id": "rejected",
                    "accepted": False,
                    "ranking_position": 4,
                },
                {
                    "provider": "pexels",
                    "provider_asset_id": "second",
                    "accepted": True,
                    "ranking_position": 2,
                },
                # A repeated verdict must not become a repeated fallback.
                {
                    "provider": "pexels",
                    "provider_asset_id": "second",
                    "accepted": True,
                    "ranking_position": 2,
                },
                # A verdict for an asset this search never returned cannot be
                # downloaded, so it is not a fallback either.
                {
                    "provider": "pexels",
                    "provider_asset_id": "never-searched",
                    "accepted": True,
                    "ranking_position": 5,
                },
            ],
            winner,
        )

        self.assertEqual(
            [candidate.source_info["asset_id"] for candidate in alternates],
            ["second", "third"],
        )

        # Ranking metadata is provenance, not a contract. Without it the order
        # verification reported is still a usable order.
        unranked = material._approved_alternate_candidates(
            [winner, second, third],
            [
                {"provider": "pexels", "provider_asset_id": "third", "accepted": True},
                {"provider": "pexels", "provider_asset_id": "second", "accepted": True},
            ],
            winner,
        )
        self.assertEqual(
            [candidate.source_info["asset_id"] for candidate in unranked],
            ["third", "second"],
        )

    def test_the_promoted_asset_is_the_one_a_sibling_beat_must_not_reuse(self):
        first = _visual_beat(
            index=1, requirement="Worker digs a hole", query="worker digging hole"
        )
        second = _visual_beat(
            index=2, requirement="Worker digs a hole", query="worker digging hole"
        )
        analyzed: list[list[str]] = []
        transfers: list[str] = []

        def pexels_search(**kwargs):
            # Fresh objects per search, the way a provider client behaves.
            return [self._candidate_for("top-ranked"), self._candidate_for("runner-up")]

        def select_best(**kwargs):
            candidates = kwargs["candidates"]
            analyzed.append(
                [candidate.source_info["asset_id"] for candidate in candidates]
            )
            winner = candidates[0]
            winner.overall_score = 0.93
            return winner, self._stats(
                [(candidate, True) for candidate in candidates]
            )

        def save_video(**kwargs):
            transfers.append(kwargs["video_url"])
            # Only the first transfer fails, which is what a transient transport
            # error looks like rather than an unusable asset.
            return "" if len(transfers) == 1 else f"D:/task/{len(transfers)}.mp4"

        service = SimpleNamespace(
            candidate_selection_settings=self._settings,
            select_best_candidate=MagicMock(side_effect=select_best),
            segment_winner=MagicMock(
                return_value={
                    "source_start_time": 5.0,
                    "source_end_time": 7.8,
                    "description": "the requested action",
                }
            ),
        )
        with (
            patch.object(material, "save_video", MagicMock(side_effect=save_video)),
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ) as persist,
        ):
            paths = material._download_videos_by_script_order_smart(
                task_id="promote-then-dedupe",
                search_terms=[first.search_queries[0], second.search_queries[0]],
                visual_beats=[first, second],
                provider_searches=[("pexels", pexels_search)],
                video_aspect=VideoAspect.portrait,
                max_clip_duration=4,
                material_directory="",
                clip_speed=1.0,
                twelvelabs_service=service,
            )

        self.assertEqual(paths, ["D:/task/2.mp4", "D:/task/3.mp4"])
        # Deduplication has to follow the asset that was actually rendered. The
        # promoted runner-up is off the table for the sibling beat, while the
        # asset that failed to transfer was never used and stays available.
        self.assertEqual(analyzed, [["top-ranked", "runner-up"], ["top-ranked"]])
        records = persist.call_args.kwargs["material_sources"]
        self.assertEqual(
            [record["asset_id"] for record in records], ["runner-up", "top-ranked"]
        )


class TestUnfillableRequirementRewrite(unittest.TestCase):
    """An item nothing could fill gets one second reading of the same narration.

    A requirement is only one reading of what the narration says. When that reading
    cannot be decomposed, or no catalog holds footage that satisfies it, abandoning
    the video throws away work that a different reading of the same spoken line
    would have completed. These tests hold three lines: the narration itself is
    never replaced, exactly one alternative is tried, and a reading that could
    never be verified is not searched for at all.
    """

    ALTERNATIVE = "Two workers walking along railway tracks"
    ALTERNATIVE_QUERY = "workers walking railway tracks"
    ORIGINAL = "Workers inspecting railway tracks"
    ORIGINAL_QUERY = "railway track inspection"
    # Only used by the tests that need the item under test to not be the opening
    # shot: this one fills on its first phrasing and takes position 0 away from it.
    LEAD = "A train approaching a station platform"
    LEAD_QUERY = "train approaching station"

    @staticmethod
    def _settings(**overrides):
        settings = {
            "batch_size": 5,
            "max_candidates": 15,
            "minimum_score": 0.7,
            "strong_early_stop_score": 0.9,
            "preferred_max_source_duration": 30,
            "concurrency": 5,
            "fail_closed": True,
        }
        settings.update(overrides)
        return settings

    @staticmethod
    def _candidate_for(asset_id):
        candidate = _candidate(asset_id, url=f"https://videos.example/{asset_id}.mp4")
        candidate.source_info.update({"provider": "pexels", "asset_id": asset_id})
        return candidate

    def _grounded_alternative(self):
        return {
            "visual_requirement": self.ALTERNATIVE,
            # A real quote of the beat's spoken text; this is what proves the
            # alternative still describes the line the video promised.
            "narration_basis": "Workers previously picked cherries",
        }

    def _run(
        self,
        *,
        undecomposable=(),
        decomposition_outage=(),
        alternative=None,
        alternative_queries=(ALTERNATIVE_QUERY,),
        approved_queries=(),
        asset_id_for=None,
        rewrite_enabled=None,
        rewrite_opening_shot=True,
        lead_with_a_filled_beat=False,
        settings_overrides=None,
    ):
        # Every test here is about the rewrite mechanism, and the single beat they
        # use would otherwise be the opening shot, where the rewrite is withheld on
        # purpose. So the opening-shot protection is off by default in this harness
        # and switched on only by the tests that are about the protection itself.
        if lead_with_a_filled_beat:
            lead = _visual_beat(
                index=1,
                semantic_group_id=1,
                requirement=self.LEAD,
                query=self.LEAD_QUERY,
            )
            beat = _visual_beat(
                index=2,
                semantic_group_id=2,
                requirement=self.ORIGINAL,
                query=self.ORIGINAL_QUERY,
            )
            beats = [lead, beat]
        else:
            beat = _visual_beat(requirement=self.ORIGINAL, query=self.ORIGINAL_QUERY)
            beats = [beat]
        undecomposable = set(undecomposable)
        decomposition_outage = set(decomposition_outage)
        asset_id_for = asset_id_for or (lambda term: term.replace(" ", "-"))
        searches: list[str] = []
        analyzed: list[str] = []

        def decompose(requirements):
            if decomposition_outage.intersection(requirements):
                # What the real generator does when the provider refuses: no spec
                # and no verdict, only a note that nobody answered.
                material.llm._record_provider_unavailable()
            return {
                material.llm.normalize_visual_requirement(requirement): (
                    _requirement_spec(requirement)
                )
                for requirement in requirements
                if requirement not in undecomposable
            }

        def search(**kwargs):
            searches.append(kwargs["search_term"])
            return [self._candidate_for(asset_id_for(kwargs["search_term"]))]

        def select_best(**kwargs):
            analyzed.append(kwargs["narration_text"])
            candidate = kwargs["candidates"][0]
            accepted = kwargs["search_query"] in approved_queries
            stats = {
                "api_candidates_analyzed": 1,
                "source_seconds_analyzed": 12.0,
                "candidate_evaluations": [
                    {
                        "provider": candidate.provider,
                        "provider_asset_id": candidate.source_info["asset_id"],
                        "accepted": accepted,
                        "ranking_position": 1,
                        "overall_score": 0.95 if accepted else 0.41,
                    }
                ],
            }
            if not accepted:
                return None, stats
            candidate.overall_score = 0.95
            return candidate, stats

        service = SimpleNamespace(
            candidate_selection_settings=lambda: self._settings(
                **(settings_overrides or {})
            ),
            select_best_candidate=MagicMock(side_effect=select_best),
            segment_winner=MagicMock(
                return_value={
                    "source_start_time": 1.0,
                    "source_end_time": 3.8,
                    "description": "the requested action",
                }
            ),
        )
        rewrite = MagicMock(
            return_value={beat.index: alternative} if alternative else {}
        )
        query_generator = MagicMock(
            return_value={beat.index: list(alternative_queries)}
        )
        config_overrides = {
            "smart_material_rewrite_opening_shot": rewrite_opening_shot,
        }
        if rewrite_enabled is not None:
            config_overrides["smart_material_requirement_rewrite"] = rewrite_enabled
        error = None
        paths: list[str] = []
        with (
            patch.object(
                material.llm, "generate_visual_requirement_specs", side_effect=decompose
            ),
            patch.object(
                material.llm, "generate_alternative_visual_requirements", rewrite
            ),
            patch.object(material.llm, "generate_visual_slot_queries", query_generator),
            patch.object(material, "save_video", return_value="D:/task/clip.mp4"),
            patch.dict(config.app, config_overrides),
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ) as persist,
        ):
            try:
                paths = material._download_videos_by_script_order_smart(
                    task_id="rewrite-unfillable-requirement",
                    search_terms=[item.search_queries[0] for item in beats],
                    visual_beats=beats,
                    provider_searches=[("pexels", search)],
                    video_aspect=VideoAspect.portrait,
                    max_clip_duration=4,
                    material_directory="",
                    clip_speed=1.0,
                    twelvelabs_service=service,
                )
            except material.SmartMaterialSelectionError as exc:
                error = exc
        return SimpleNamespace(
            paths=paths,
            error=error,
            persist=persist,
            service=service,
            rewrite=rewrite,
            query_generator=query_generator,
            searches=searches,
            analyzed=analyzed,
            beat=beat,
            runs=persist.call_args.kwargs["semantic_verifier_runs"],
            decisions=[
                run["final_decision"]
                for run in persist.call_args.kwargs["semantic_verifier_runs"]
            ],
        )

    def test_a_requirement_that_never_decomposed_is_rewritten_before_anything_is_bought(
        self,
    ):
        outcome = self._run(
            undecomposable={self.ORIGINAL},
            alternative=self._grounded_alternative(),
            approved_queries=(self.ALTERNATIVE_QUERY,),
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.paths, ["D:/task/clip.mp4"])
        # The wording that could not be verified is never searched for: candidates
        # bought against it could not have passed any gate.
        self.assertEqual(outcome.searches, [self.ALTERNATIVE_QUERY])
        self.assertEqual(outcome.analyzed, [self.ALTERNATIVE])
        self.assertEqual(
            outcome.decisions,
            ["DECOMPOSITION_FAILED", "REQUIREMENT_REWRITTEN", "ACCEPT"],
        )
        rewritten = outcome.runs[1]
        self.assertEqual(rewritten["visual_requirement"], self.ORIGINAL)
        self.assertEqual(rewritten["alternative_visual_requirement"], self.ALTERNATIVE)
        self.assertEqual(
            rewritten["narration_basis"], "Workers previously picked cherries"
        )
        # The rewrite is asked to re-describe the spoken line, not the requirement,
        # which is the only reason the result can still be faithful to the video.
        requested = outcome.rewrite.call_args.args[0][0]
        self.assertEqual(requested["narration_text"], outcome.beat.spoken_text)
        self.assertEqual(requested["failed_requirement"], self.ORIGINAL)
        # The window that gets rendered is cut against the wording that was
        # actually verified, not against the abandoned one.
        self.assertEqual(
            outcome.service.segment_winner.call_args.kwargs["narration_text"],
            self.ALTERNATIVE,
        )
        record = outcome.persist.call_args.kwargs["material_sources"][0]
        self.assertEqual(record["search_term"], self.ALTERNATIVE_QUERY)
        self.assertEqual(record["visual_beat_index"], 1)

    def test_an_item_whose_candidates_were_all_rejected_is_rewritten_once(self):
        outcome = self._run(
            alternative=self._grounded_alternative(),
            approved_queries=(self.ALTERNATIVE_QUERY,),
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.paths, ["D:/task/clip.mp4"])
        # The planned reading gets its full chance first; the alternative is a
        # recovery, not a replacement for the script stage.
        self.assertEqual(
            outcome.searches, [self.ORIGINAL_QUERY, self.ALTERNATIVE_QUERY]
        )
        self.assertEqual(outcome.analyzed, [self.ORIGINAL, self.ALTERNATIVE])
        self.assertEqual(
            outcome.decisions, ["REJECT", "REQUIREMENT_REWRITTEN", "ACCEPT"]
        )
        self.assertEqual(outcome.rewrite.call_count, 1)

    def test_the_second_round_never_re_analyzes_a_candidate_the_first_round_judged(self):
        outcome = self._run(
            alternative=self._grounded_alternative(),
            # Both readings surface the same asset, which is exactly when a naive
            # retry would pay twice for one verdict.
            asset_id_for=lambda term: "shared-asset",
        )

        self.assertIsNotNone(outcome.error)
        self.assertEqual(
            outcome.decisions,
            ["REJECT", "REQUIREMENT_REWRITTEN", "NO_CANDIDATES"],
        )
        self.assertEqual(outcome.service.select_best_candidate.call_count, 1)
        self.assertEqual(outcome.analyzed, [self.ORIGINAL])
        self.assertIn("after metadata quality filters", str(outcome.error))

    def test_without_a_usable_alternative_the_item_fails_with_both_diagnoses(self):
        outcome = self._run(undecomposable={self.ORIGINAL}, alternative=None)

        self.assertIsNotNone(outcome.error)
        message = str(outcome.error)
        self.assertIn("decomposition failed for visual beat 1", message)
        self.assertIn("No alternative visual requirement was usable", message)
        self.assertEqual(outcome.searches, [])
        self.assertEqual(
            outcome.decisions,
            ["DECOMPOSITION_FAILED", "REQUIREMENT_REWRITE_UNAVAILABLE"],
        )
        self.assertTrue(outcome.runs[-1]["reason"])

    def test_an_alternative_that_cannot_be_decomposed_either_is_not_searched_for(self):
        outcome = self._run(
            undecomposable={self.ORIGINAL, self.ALTERNATIVE},
            alternative=self._grounded_alternative(),
        )

        self.assertIsNotNone(outcome.error)
        self.assertEqual(outcome.searches, [])
        self.assertEqual(
            outcome.decisions,
            ["DECOMPOSITION_FAILED", "REQUIREMENT_REWRITE_UNAVAILABLE"],
        )
        self.assertIn("could not be decomposed", outcome.runs[-1]["reason"])

    def test_a_provider_outage_is_not_reported_as_a_wording_problem(self):
        """Run e04d3f7e told the user to reword a requirement Gemini never judged.

        A 429 and an ungroundable requirement both leave the beat without a spec.
        Only one of them is the requirement's fault, and only one of them is worth
        the user's time to fix.
        """
        outcome = self._run(
            undecomposable={self.ORIGINAL, self.ALTERNATIVE},
            decomposition_outage={self.ALTERNATIVE},
            alternative=self._grounded_alternative(),
        )

        self.assertIsNotNone(outcome.error)
        self.assertEqual(outcome.searches, [])
        self.assertEqual(
            outcome.decisions,
            ["DECOMPOSITION_FAILED", "REQUIREMENT_REWRITE_UNAVAILABLE"],
        )
        reason = outcome.runs[-1]["reason"]
        self.assertIn("provider was unavailable", reason)
        self.assertNotIn("could not be decomposed", reason)

    def test_the_rewrite_can_be_switched_off_without_changing_anything_else(self):
        outcome = self._run(
            undecomposable={self.ORIGINAL},
            alternative=self._grounded_alternative(),
            rewrite_enabled=False,
        )

        self.assertIsNotNone(outcome.error)
        outcome.rewrite.assert_not_called()
        self.assertEqual(outcome.searches, [])
        self.assertEqual(outcome.decisions, ["DECOMPOSITION_FAILED"])
        self.assertIn("decomposition failed for visual beat 1", str(outcome.error))

    def test_the_opening_shot_is_never_rewritten(self):
        """Run bdfd478b opened on a cheese grater because of this exact path.

        Beat 1 asked for a plate of restaurant-style vegetables, could not be
        filled, was re-described as "vegetables cooking in a restaurant kitchen",
        and won a shot of cheese being grated — over a line saying the dish is not
        about butter. The rewrite is allowed to change what is depicted, so the one
        frame the whole video is judged on is the one place it must not run.
        """
        outcome = self._run(
            undecomposable={self.ORIGINAL},
            alternative=self._grounded_alternative(),
            approved_queries=(self.ALTERNATIVE_QUERY,),
            rewrite_opening_shot=False,
        )

        # The alternative wording would have filled this beat. It is refused
        # anyway: a failed opening shot is visible and fixable, a wrong one is not.
        self.assertIsNotNone(outcome.error)
        self.assertEqual(outcome.paths, [])
        outcome.rewrite.assert_not_called()
        self.assertEqual(outcome.searches, [])
        self.assertEqual(
            outcome.decisions,
            ["DECOMPOSITION_FAILED", "OPENING_SHOT_REWRITE_WITHHELD"],
        )
        # The operator has to be able to tell this apart from a plain outage.
        self.assertIn("withheld from the opening", str(outcome.error))

    def test_a_later_shot_is_still_rewritten_when_the_opening_shot_is_not(self):
        """The policy is about position, not about distrusting the rewrite.

        Same setting, same failure, same alternative — the only difference is that
        something else opens the video now, and the recovery runs again.
        """
        outcome = self._run(
            undecomposable={self.ORIGINAL},
            alternative=self._grounded_alternative(),
            approved_queries=(self.LEAD_QUERY, self.ALTERNATIVE_QUERY),
            rewrite_opening_shot=False,
            lead_with_a_filled_beat=True,
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.rewrite.call_count, 1)
        self.assertEqual(
            outcome.decisions,
            ["ACCEPT", "DECOMPOSITION_FAILED", "REQUIREMENT_REWRITTEN", "ACCEPT"],
        )
        self.assertNotIn("OPENING_SHOT_REWRITE_WITHHELD", outcome.decisions)

    def test_the_opening_shot_flag_is_opt_in_rather_than_opt_out(self):
        """Unlike the global rewrite flag, an unreadable value keeps the protection.

        The two flags default opposite ways, so they cannot share a parser: an
        unset or misspelled value has to leave the opening shot protected.
        """
        for value, expected in (
            ("on", True),
            ("1", True),
            ("TRUE", True),
            ("yes", True),
            (True, True),
            ("off", False),
            ("0", False),
            ("no", False),
            (False, False),
            ("maybe", False),
        ):
            with self.subTest(value=value):
                with patch.dict(
                    config.app, {"smart_material_rewrite_opening_shot": value}
                ):
                    self.assertEqual(
                        material.is_opening_shot_rewrite_enabled(), expected
                    )

    def test_an_unconfigured_run_protects_its_opening_shot(self):
        with patch.dict(config.app, {}, clear=False):
            config.app.pop("smart_material_rewrite_opening_shot", None)
            self.assertFalse(material.is_opening_shot_rewrite_enabled())

    def test_the_rewrite_flag_is_read_like_the_other_smart_material_flags(self):
        for value, expected in (
            ("off", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            (False, False),
            ("on", True),
            ("1", True),
            (True, True),
        ):
            with self.subTest(value=value):
                with patch.dict(
                    config.app, {"smart_material_requirement_rewrite": value}
                ):
                    self.assertEqual(
                        material.is_requirement_rewrite_enabled(), expected
                    )

    def test_the_default_is_on_so_an_unconfigured_run_can_still_recover(self):
        with patch.dict(config.app, {}, clear=False):
            config.app.pop("smart_material_requirement_rewrite", None)
            self.assertTrue(material.is_requirement_rewrite_enabled())

    def test_a_rewritten_requirement_uses_generated_queries_when_they_exist(self):
        beat = _visual_beat(requirement=self.ORIGINAL, query=self.ORIGINAL_QUERY)
        generator = MagicMock(
            return_value={beat.index: ["first phrasing", "second phrasing", "third"]}
        )

        with patch.object(material.llm, "generate_visual_slot_queries", generator):
            queries = material._alternative_item_search_queries(
                beat, self.ALTERNATIVE, beat.spoken_text, 2
            )

        self.assertEqual(queries, ["first phrasing", "second phrasing"])
        # The phrasings the script stage planned belong to the abandoned wording,
        # so the new reading must be given queries of its own.
        self.assertEqual(
            generator.call_args.kwargs["visual_slots"][0]["visual_requirement"],
            self.ALTERNATIVE,
        )
        self.assertEqual(generator.call_args.kwargs["video_subject"], beat.spoken_text)

    def test_a_short_alternative_is_its_own_query_and_a_long_one_is_not(self):
        beat = _visual_beat(requirement=self.ORIGINAL, query=self.ORIGINAL_QUERY)
        long_requirement = (
            "Two workers in orange vests walking slowly along wet railway tracks "
            "at sunrise"
        )

        with patch.object(
            material.llm,
            "generate_visual_slot_queries",
            side_effect=RuntimeError("provider is unavailable"),
        ):
            short_queries = material._alternative_item_search_queries(
                beat, self.ALTERNATIVE, beat.spoken_text, 3
            )
            long_queries = material._alternative_item_search_queries(
                beat, long_requirement, beat.spoken_text, 3
            )

        # A short concrete phrase behaves like a stock query; a full sentence is an
        # over-constrained query that returns nothing, so it is not worth a search.
        self.assertEqual(short_queries, [self.ALTERNATIVE])
        self.assertEqual(long_queries, [])


class TestPerRoundAnalysisBudget(unittest.TestCase):
    """One item nothing can satisfy must not be able to cost a whole run.

    Only a *successful* round stops early — the selector settles on its first
    strong candidate — so a healthy item is cheap by construction. A round that
    can never be satisfied is the opposite: it spends the full candidate cap on
    every phrasing of every provider, and analysis is the part that is metered.
    These tests hold the two lines that make a ceiling safe rather than merely
    cheap: it never denies an item its first look, and it never starves the
    alternative-wording round that is the item's best remaining chance.
    """

    REQUIREMENT = "Worker inspects a rail joint"

    def setUp(self):
        self.requirement_patcher = patch.object(
            material.llm,
            "generate_visual_requirement_specs",
            side_effect=lambda requirements: {
                material.llm.normalize_visual_requirement(requirement): (
                    _requirement_spec(requirement)
                )
                for requirement in requirements
            },
        )
        self.requirement_patcher.start()
        self.addCleanup(self.requirement_patcher.stop)
        self.rewrite = _patch_requirement_rewrite(self)
        # This class measures the rewrite round's budget using a single beat, which
        # is therefore also the opening shot, where the rewrite is withheld on
        # purpose. Enable it explicitly so the round under test actually runs; the
        # policy itself is covered by TestUnfillableRequirementRewrite.
        opening_shot_rewrite = patch.dict(
            config.app, {"smart_material_rewrite_opening_shot": True}
        )
        opening_shot_rewrite.start()
        self.addCleanup(opening_shot_rewrite.stop)

    @staticmethod
    def _settings(**overrides):
        settings = {
            "batch_size": 5,
            "max_candidates": 15,
            "minimum_score": 0.7,
            "strong_early_stop_score": 0.9,
            "preferred_max_source_duration": 30,
            "concurrency": 5,
            "fail_closed": True,
        }
        settings.update(overrides)
        return settings

    @staticmethod
    def _beat(*queries):
        beat = _visual_beat(
            requirement=TestPerRoundAnalysisBudget.REQUIREMENT, query=queries[0]
        )
        beat.search_queries = list(queries)
        return beat

    @staticmethod
    def _candidate_for(provider, term):
        # Distinct per provider *and* per phrasing: an asset this item already had
        # a verdict on is excluded from later phrasings, so overlapping pages would
        # measure the exclusion set rather than the budget.
        asset_id = f"{provider}-{term.replace(' ', '-')}"
        candidate = _candidate(asset_id, url=f"https://videos.example/{asset_id}.mp4")
        candidate.provider = provider
        candidate.source_info.update({"provider": provider, "asset_id": asset_id})
        return candidate

    @staticmethod
    def _stats(candidates, *, accepted=False):
        return {
            "api_candidates_analyzed": len(candidates),
            "source_seconds_analyzed": 12.0 * len(candidates),
            "candidate_evaluations": [
                {
                    "provider": candidate.provider,
                    "provider_asset_id": candidate.source_info["asset_id"],
                    "accepted": accepted,
                    "ranking_position": position + 1,
                    "overall_score": 0.95 if accepted else 0.41,
                }
                for position, candidate in enumerate(candidates)
            ],
        }

    def _service(self, select_best):
        return SimpleNamespace(
            candidate_selection_settings=self._settings,
            select_best_candidate=MagicMock(side_effect=select_best),
            segment_winner=MagicMock(
                return_value={
                    "source_start_time": 1.0,
                    "source_end_time": 3.8,
                    "description": "the requested action",
                }
            ),
        )

    def _select(self, beat, provider_searches, service, *, budget, variants=3):
        error = None
        paths: list[str] = []
        with (
            patch.dict(
                config.app,
                {
                    "smart_material_max_query_variants": variants,
                    "smart_material_max_analyzed_candidates_per_round": budget,
                },
            ),
            patch.object(material, "save_video", return_value="D:/task/beat.mp4"),
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ) as persist,
        ):
            try:
                paths = material._download_videos_by_script_order_smart(
                    task_id="analysis-budget",
                    search_terms=[beat.search_queries[0]],
                    visual_beats=[beat],
                    provider_searches=provider_searches,
                    video_aspect=VideoAspect.portrait,
                    max_clip_duration=4,
                    material_directory="",
                    clip_speed=1.0,
                    twelvelabs_service=service,
                )
            except material.SmartMaterialSelectionError as exc:
                error = exc
        return SimpleNamespace(
            paths=paths,
            error=error,
            persist=persist,
            service=service,
            runs=persist.call_args.kwargs["semantic_verifier_runs"],
        )

    def _rejecting_cascade(self, beat, budget, *, searched):
        def provider_search(provider):
            def search(**kwargs):
                searched.append((provider, kwargs["search_term"]))
                return [self._candidate_for(provider, kwargs["search_term"])]

            return search

        def select_best(**kwargs):
            return None, self._stats(kwargs["candidates"])

        service = self._service(select_best)
        return self._select(
            beat,
            [
                ("pexels", provider_search("pexels")),
                ("pixabay", provider_search("pixabay")),
            ],
            service,
            budget=budget,
        )

    def test_a_round_that_spent_its_budget_buys_no_further_phrasing_or_provider(self):
        beat = self._beat("rail joint inspection", "worker checking rail", "rail bolts")
        searched: list[tuple[str, str]] = []

        result = self._rejecting_cascade(beat, 2, searched=searched)

        # Two analyses were allowed, so the third phrasing is never requested and
        # the second catalog is never opened. Without the ceiling this beat would
        # have bought six searches and six rounds of analysis.
        self.assertEqual(
            searched,
            [
                ("pexels", beat.search_queries[0]),
                ("pexels", beat.search_queries[1]),
            ],
        )
        self.assertEqual(result.service.select_best_candidate.call_count, 2)
        self.assertIsNotNone(result.error)
        # The operator has to be able to tell "we stopped paying" apart from
        # "the catalog is empty", or the wrong knob gets turned next.
        self.assertIn(
            "analysis budget of 2 analyzed candidates reached for visual beat 1",
            str(result.error),
        )
        # And it has to survive into the manifest, not only into the exception,
        # because the run that gets diagnosed later is the persisted one.
        cutoff = [
            run
            for run in result.runs
            if run["final_decision"] == "ANALYSIS_BUDGET_EXHAUSTED"
        ]
        self.assertEqual(len(cutoff), 1)
        self.assertEqual(cutoff[0]["candidates_analyzed"], 2)
        self.assertEqual(cutoff[0]["analysis_budget"], 2)

    def test_a_budget_of_zero_leaves_the_full_cascade_exactly_as_it_was(self):
        beat = self._beat("rail joint inspection", "worker checking rail", "rail bolts")
        searched: list[tuple[str, str]] = []

        result = self._rejecting_cascade(beat, 0, searched=searched)

        # Zero is the documented escape hatch, not a cap of nothing: every phrasing
        # is spent on the first provider before the cascade changes catalog.
        self.assertEqual(
            searched,
            [("pexels", query) for query in beat.search_queries]
            + [("pixabay", query) for query in beat.search_queries],
        )
        self.assertNotIn("analysis budget", str(result.error))

    def test_an_item_is_never_denied_its_first_look(self):
        beat = self._beat("rail joint inspection", "worker checking rail")
        searched: list[str] = []

        def pexels_search(**kwargs):
            searched.append(kwargs["search_term"])
            return [self._candidate_for("pexels", kwargs["search_term"])]

        def select_best(**kwargs):
            candidate = kwargs["candidates"][0]
            candidate.overall_score = 0.95
            return candidate, self._stats(kwargs["candidates"], accepted=True)

        service = self._service(select_best)
        # A budget below one analysis is the harshest setting an operator can
        # choose; it still may not turn a fillable beat into a failed video.
        result = self._select(beat, [("pexels", pexels_search)], service, budget=1)

        self.assertEqual(result.paths, ["D:/task/beat.mp4"])
        self.assertEqual(searched, [beat.search_queries[0]])
        self.assertEqual(
            [run["final_decision"] for run in result.runs], ["ACCEPT"]
        )

    def test_the_alternative_wording_round_is_budgeted_separately(self):
        beat = self._beat("rail joint inspection", "worker checking rail", "rail bolts")
        alternative = "Two workers walking along railway tracks"
        alternative_query = "workers walking railway tracks"
        searched: list[str] = []

        def pexels_search(**kwargs):
            searched.append(kwargs["search_term"])
            return [self._candidate_for("pexels", kwargs["search_term"])]

        def select_best(**kwargs):
            candidates = kwargs["candidates"]
            if kwargs["search_query"] != alternative_query:
                return None, self._stats(candidates)
            candidates[0].overall_score = 0.95
            return candidates[0], self._stats(candidates, accepted=True)

        service = self._service(select_best)
        self.rewrite.return_value = {
            beat.index: {
                "visual_requirement": alternative,
                "narration_basis": "Workers previously picked cherries",
            }
        }
        with patch.object(
            material.llm,
            "generate_visual_slot_queries",
            return_value={beat.index: [alternative_query]},
        ):
            result = self._select(
                beat, [("pexels", pexels_search)], service, budget=2
            )

        # The planned round spent the whole budget and was cut off, yet the rewrite
        # still got to search: the ceiling is per round precisely so that the
        # recovery most likely to work is not paid for by the failure before it.
        self.assertEqual(
            searched,
            [beat.search_queries[0], beat.search_queries[1], alternative_query],
        )
        self.assertEqual(result.paths, ["D:/task/beat.mp4"])
        record = result.persist.call_args.kwargs["material_sources"][0]
        self.assertEqual(record["search_term"], alternative_query)
        # The successful video still records that the first round was cut short,
        # so a ceiling that is quietly costing quality stays visible.
        self.assertEqual(
            [run["final_decision"] for run in result.runs],
            [
                "REJECT",
                "REJECT",
                "ANALYSIS_BUDGET_EXHAUSTED",
                "REQUIREMENT_REWRITTEN",
                "ACCEPT",
            ],
        )

    def test_the_budget_reads_as_a_whole_number_of_analyses(self):
        # Read against a config with no such key so the built-in default is
        # asserted instead of whatever the local config.toml happens to hold.
        without_key = dict(config.app)
        without_key.pop("smart_material_max_analyzed_candidates_per_round", None)
        with patch.dict(config.app, without_key, clear=True):
            # The default tracks the per-search cap, so lowering that cap for cost
            # reasons lowers the ceiling with it instead of leaving it stranded.
            self.assertEqual(material.analysis_budget_per_selection_round(15), 75)
            self.assertEqual(material.analysis_budget_per_selection_round(8), 40)
            self.assertEqual(material.analysis_budget_per_selection_round(0), 5)
        for value, expected in (
            (0, 0),
            (4, 4),
            ("20", 20),
            (" 6 ", 6),
            # A cap below nothing has no reading that leaves a run able to select
            # anything, and neither has a word; both degrade to the default rather
            # than to an unbounded or unusable run.
            (-3, 75),
            ("lots", 75),
            (None, 75),
            ("", 75),
        ):
            with self.subTest(value=value):
                with patch.dict(
                    config.app,
                    {"smart_material_max_analyzed_candidates_per_round": value},
                ):
                    self.assertEqual(
                        material.analysis_budget_per_selection_round(15), expected
                    )


class TestUnfillableBeatMerge(unittest.TestCase):
    """A beat nothing could fill is absorbed by the shot beside it.

    This is the last rung before the video fails, and its entire justification is
    cost: the neighbour's clip has already been approved for the adjacent stretch
    of the same narration, so it is cut wider rather than replaced. A sibling of the
    open beat's own semantic group is the strongest form of that, because the two
    share a requirement; the shot of the next group is the weaker form, taken only
    when no sibling exists. These tests hold both lines. A merge normally buys one
    segmentation call and nothing else, it never reaches across a group boundary
    while a sibling is available, the timeline it leaves behind still covers the
    narration without a seam, and a video no merge could rescue stops paying
    immediately instead of analyzing every beat that is left.
    """

    GROUP_REQUIREMENT = "Coffee beans drying in sunlight"

    @staticmethod
    def _settings(**overrides):
        settings = {
            "batch_size": 5,
            "max_candidates": 15,
            "minimum_score": 0.7,
            "strong_early_stop_score": 0.9,
            "preferred_max_source_duration": 30,
            "concurrency": 5,
            "fail_closed": True,
        }
        settings.update(overrides)
        return settings

    @staticmethod
    def _asset_id(term):
        return term.replace(" ", "-")

    @classmethod
    def _candidate_for(cls, term, duration=12):
        asset_id = cls._asset_id(term)
        candidate = _candidate(
            asset_id,
            duration=duration,
            url=f"https://videos.example/{asset_id}.mp4",
        )
        candidate.source_info.update({"provider": "pexels", "asset_id": asset_id})
        return candidate

    def _beats(self, count, *, group_ids=None, requirements=None, duration=2.8):
        # Siblings of one semantic group share the requirement by construction in
        # S4, which is why a sibling's approved clip can stand in unchanged. Shots
        # of different groups do not, so ``requirements`` lets a test give the
        # neighbouring group its own requirement and keep the cross-group cases
        # honest about what the covering clip was actually approved for.
        group_ids = list(group_ids or [1] * count)
        requirements = list(requirements or [self.GROUP_REQUIREMENT] * count)
        return [
            _visual_beat(
                index=position,
                semantic_group_id=group_ids[position - 1],
                duration=duration,
                requirement=requirements[position - 1],
                query=f"beans query {position}",
            )
            for position in range(1, count + 1)
        ]

    def _run(
        self,
        *,
        beats,
        approved_queries,
        merge_ceiling=1,
        cross_group_merge=True,
        durations=None,
        undownloadable=(),
        segment_failures_after=None,
        clip_speed=1.0,
    ):
        durations = durations or {}
        searches: list[str] = []
        analyzed: list[str] = []
        segmented: list[dict] = []
        saved: list[str] = []
        merged_beats: list[VisualBeat] = []

        def decompose(requirements):
            return {
                material.llm.normalize_visual_requirement(requirement): (
                    _requirement_spec(requirement)
                )
                for requirement in requirements
            }

        def search(**kwargs):
            term = kwargs["search_term"]
            searches.append(term)
            # A real catalog answers with more than one clip, and the second one is
            # what a merge falls back to when the winner is too short to be widened.
            return [
                self._candidate_for(term, durations.get(term, 12)),
                self._candidate_for(f"{term} spare", 20),
            ]

        def select_best(**kwargs):
            analyzed.append(kwargs["search_query"])
            if kwargs["search_query"] not in approved_queries:
                return None, {
                    "api_candidates_analyzed": len(kwargs["candidates"]),
                    "source_seconds_analyzed": 12.0,
                    "candidate_evaluations": [],
                }
            candidate = kwargs["candidates"][0]
            candidate.overall_score = 0.95
            return candidate, {
                "api_candidates_analyzed": len(kwargs["candidates"]),
                "source_seconds_analyzed": 12.0,
                "candidate_evaluations": [
                    {
                        "provider": candidate.provider,
                        "provider_asset_id": candidate.source_info["asset_id"],
                        "accepted": True,
                        "ranking_position": 1,
                        "overall_score": 0.95,
                    }
                ],
            }

        def segment(**kwargs):
            segmented.append(kwargs)
            if (
                segment_failures_after is not None
                and len(segmented) > segment_failures_after
            ):
                return None
            # Offering the whole asset lets the production normalizer choose the
            # window, so the test never hand-computes a source range.
            return {
                "source_start_time": 0.0,
                "source_end_time": float(kwargs["source_duration"]),
                "description": "the requested action",
            }

        def save(**kwargs):
            url = str(kwargs["video_url"])
            if url in undownloadable:
                return ""
            saved.append(url)
            return f"D:/task/{Path(url).name}"

        service = SimpleNamespace(
            candidate_selection_settings=lambda: self._settings(),
            select_best_candidate=MagicMock(side_effect=select_best),
            segment_winner=MagicMock(side_effect=segment),
        )
        error = None
        paths: list[str] = []
        with (
            patch.object(
                material.llm, "generate_visual_requirement_specs", side_effect=decompose
            ),
            # Rung 2 is exercised by its own suite; neutralizing it here keeps these
            # assertions about the merge and keeps the run off the live LLM.
            patch.object(
                material.llm,
                "generate_alternative_visual_requirements",
                MagicMock(return_value={}),
            ),
            patch.object(
                material.llm, "generate_visual_slot_queries", MagicMock(return_value={})
            ),
            patch.object(material, "save_video", side_effect=save),
            patch.dict(
                config.app,
                {
                    "smart_material_max_merged_beats": merge_ceiling,
                    "smart_material_cross_group_merge": cross_group_merge,
                    "smart_material_requirement_rewrite": False,
                    # Pinned so a local config cannot decide how much this run may
                    # analyze before the merge is even reached.
                    "smart_material_max_analyzed_candidates_per_round": 75,
                },
            ),
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ) as persist,
        ):
            try:
                paths = material._download_videos_by_script_order_smart(
                    task_id="merge-unfillable-beat",
                    search_terms=[beat.search_queries[0] for beat in beats],
                    visual_beats=list(beats),
                    provider_searches=[("pexels", search)],
                    video_aspect=VideoAspect.portrait,
                    max_clip_duration=4,
                    material_directory="",
                    clip_speed=clip_speed,
                    twelvelabs_service=service,
                    merged_beats_out=merged_beats,
                )
            except material.SmartMaterialSelectionError as exc:
                error = exc
        runs = persist.call_args.kwargs["semantic_verifier_runs"]
        return SimpleNamespace(
            paths=paths,
            error=error,
            merged=merged_beats,
            records=persist.call_args.kwargs["material_sources"],
            runs=runs,
            decisions=[run["final_decision"] for run in runs],
            merges=[
                run for run in runs if run["final_decision"] == "UNFILLABLE_BEAT_MERGED"
            ],
            searches=searches,
            analyzed=analyzed,
            segmented=segmented,
            saved=saved,
            service=service,
        )

    def test_a_video_that_needed_no_merge_keeps_the_timeline_it_planned(self):
        beats = self._beats(2)
        outcome = self._run(
            beats=beats,
            approved_queries={"beans query 1", "beans query 2"},
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(len(outcome.paths), 2)
        # An untouched run must not hand a timeline back at all, or the caller
        # would swap a validated timeline for a rebuilt copy of itself.
        self.assertEqual(outcome.merged, [])
        self.assertEqual(outcome.merges, [])
        self.assertEqual(len(outcome.segmented), 2)

    def test_the_previous_sibling_absorbs_an_unfillable_beat_without_buying_analysis(
        self,
    ):
        beats = self._beats(2)
        outcome = self._run(beats=beats, approved_queries={"beans query 1"})

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.paths, ["D:/task/beans-query-1.mp4"])
        # One clip, one record, one beat: the cut between the two shots is gone
        # while the narration window they covered together is untouched.
        self.assertEqual(len(outcome.merged), 1)
        merged = outcome.merged[0]
        self.assertEqual(merged.index, 1)
        self.assertAlmostEqual(merged.start_time, 0.0)
        self.assertAlmostEqual(merged.end_time, 5.6)
        self.assertAlmostEqual(merged.duration, 5.6)
        self.assertEqual(merged.duration_policy, "unfillable_beat_merged")
        self.assertFalse(merged.rapid_cut)
        self.assertEqual(merged.semantic_group_id, 1)
        self.assertEqual(merged.source_narration_slot_indexes, [1, 2])
        self.assertEqual(
            merged.spoken_text,
            f"{beats[0].spoken_text} {beats[1].spoken_text}",
        )
        # The whole point of merging into a sibling: the second beat's window is
        # cut out of footage that was already judged, so no new analysis is bought.
        self.assertEqual(outcome.service.select_best_candidate.call_count, 2)
        self.assertEqual(outcome.saved, ["https://videos.example/beans-query-1.mp4"])
        self.assertEqual(len(outcome.segmented), 2)
        widened = outcome.segmented[-1]
        self.assertEqual(widened["video_url"], "https://videos.example/beans-query-1.mp4")
        self.assertAlmostEqual(widened["slot_duration"], 5.6)
        self.assertAlmostEqual(widened["requested_source_duration"], 5.6)
        self.assertEqual(len(outcome.records), 1)
        record = outcome.records[0]
        self.assertEqual(record["visual_beat_index"], 1)
        self.assertAlmostEqual(record["required_target_duration"], 5.6)
        self.assertAlmostEqual(
            record["source_end_time"] - record["source_start_time"], 5.6, places=3
        )
        self.assertEqual(
            outcome.merges,
            [
                {
                    "visual_item_type": "visual_beat",
                    "visual_item_index": 2,
                    "visual_requirement": self.GROUP_REQUIREMENT,
                    "final_decision": "UNFILLABLE_BEAT_MERGED",
                    "merged_into_visual_item_index": 1,
                    "merged_target_duration": 5.6,
                    "merge_fill": "neighbour_window_extended",
                    "merge_scope": "same_semantic_group",
                    "reason": outcome.merges[0]["reason"],
                }
            ],
        )
        # The reason a beat was retired has to survive in the provenance, because
        # the finished video no longer shows that a shot was planned there.
        self.assertIn("visual beat 2", outcome.merges[0]["reason"])

    def test_the_next_sibling_absorbs_a_first_beat_nothing_could_fill(self):
        outcome = self._run(beats=self._beats(2), approved_queries={"beans query 2"})

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.paths, ["D:/task/beans-query-2.mp4"])
        self.assertEqual(len(outcome.merged), 1)
        merged = outcome.merged[0]
        # The survivor keeps its own index even though it is now the first beat:
        # its persisted record was written under that index.
        self.assertEqual(merged.index, 2)
        self.assertAlmostEqual(merged.start_time, 0.0)
        self.assertAlmostEqual(merged.end_time, 5.6)
        self.assertEqual(outcome.records[0]["visual_beat_index"], 2)
        self.assertEqual(outcome.merges[0]["merged_into_visual_item_index"], 2)
        self.assertEqual(outcome.merges[0]["visual_item_index"], 1)

    def test_two_consecutive_unfillable_beats_collapse_into_one_survivor(self):
        outcome = self._run(
            beats=self._beats(3),
            approved_queries={"beans query 1"},
            merge_ceiling=2,
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.paths, ["D:/task/beans-query-1.mp4"])
        self.assertEqual(len(outcome.merged), 1)
        merged = outcome.merged[0]
        self.assertEqual(merged.index, 1)
        self.assertAlmostEqual(merged.end_time, 8.4)
        self.assertAlmostEqual(merged.duration, 8.4)
        self.assertEqual(merged.source_narration_slot_indexes, [1, 2, 3])
        # The second merge has to see the already widened survivor, otherwise the
        # third beat would look like it borders nothing.
        self.assertEqual(
            [run["merged_into_visual_item_index"] for run in outcome.merges], [1, 1]
        )
        self.assertEqual([run["visual_item_index"] for run in outcome.merges], [2, 3])
        self.assertAlmostEqual(outcome.segmented[-1]["slot_duration"], 8.4)
        self.assertEqual(outcome.saved, ["https://videos.example/beans-query-1.mp4"])

    def test_a_video_no_merge_could_rescue_stops_before_paying_for_the_rest(self):
        outcome = self._run(
            beats=self._beats(3),
            approved_queries=set(),
            merge_ceiling=0,
        )

        self.assertIsNotNone(outcome.error)
        # Merging switched off restores the old policy exactly: the first
        # unfillable beat ends the run, and beats 2 and 3 are never searched for.
        self.assertEqual(outcome.searches, ["beans query 1"])
        self.assertEqual(outcome.analyzed, ["beans query 1"])
        self.assertEqual(outcome.merged, [])
        self.assertEqual(outcome.merges, [])
        self.assertEqual(outcome.records, [])

    def test_more_open_beats_than_the_ceiling_allows_ends_the_run_at_that_point(self):
        outcome = self._run(
            beats=self._beats(3),
            approved_queries={"beans query 2"},
            merge_ceiling=1,
        )

        self.assertIsNotNone(outcome.error)
        # Beat 1 is allowed to stay open, beat 3 pushes the run past the ceiling,
        # and the video is abandoned there rather than merged down to one shot.
        self.assertEqual(
            outcome.searches, ["beans query 1", "beans query 2", "beans query 3"]
        )
        self.assertEqual(outcome.merged, [])
        self.assertEqual(outcome.merges, [])
        self.assertIn("visual beat 1", str(outcome.error))
        self.assertIn("visual beat 3", str(outcome.error))
        # A beat that did get its clip still leaves its provenance behind, so a
        # failed run can be diagnosed from script.json.
        self.assertEqual([record["visual_beat_index"] for record in outcome.records], [2])

    def test_a_lone_shot_of_its_group_is_absorbed_by_the_shot_beside_it(self):
        # The shape that made this rung necessary. Once the span grouper began
        # emitting one single-event requirement per span, most spans became a single
        # beat, so a beat that fails is usually the only shot of its group -- and the
        # same-group rule meant the merge could not fire at all and the whole video
        # was lost. A measured render died exactly here.
        beats = self._beats(
            3,
            group_ids=[1, 2, 3],
            requirements=[
                "Coffee cherries being picked by hand",
                self.GROUP_REQUIREMENT,
                "Roasted beans pouring into a grinder",
            ],
        )
        outcome = self._run(
            beats=beats,
            approved_queries={"beans query 1", "beans query 3"},
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(
            outcome.paths, ["D:/task/beans-query-1.mp4", "D:/task/beans-query-3.mp4"]
        )
        self.assertEqual(len(outcome.merged), 2)
        merged = outcome.merged[0]
        self.assertEqual(merged.index, 1)
        self.assertAlmostEqual(merged.start_time, 0.0)
        self.assertAlmostEqual(merged.end_time, 5.6)
        # The weaker rescue is labelled as such: the covering clip was verified
        # against the adjacent moment of the narration, not this one.
        self.assertEqual(merged.duration_policy, "unfillable_beat_cross_group_merged")
        self.assertEqual(merged.semantic_group_id, 1)
        self.assertEqual(merged.source_narration_slot_indexes, [1, 2])
        # Still the cheapest rung: one clip per surviving beat, no analysis bought
        # for the window that was absorbed.
        self.assertEqual(outcome.service.select_best_candidate.call_count, 3)
        self.assertEqual(
            outcome.saved,
            [
                "https://videos.example/beans-query-1.mp4",
                "https://videos.example/beans-query-3.mp4",
            ],
        )
        self.assertEqual(outcome.merges[0]["merge_scope"], "adjacent_semantic_group")
        self.assertEqual(outcome.merges[0]["merged_into_visual_item_index"], 1)
        self.assertEqual(outcome.merges[0]["visual_requirement"], self.GROUP_REQUIREMENT)
        # A report reading this run must not mistake the rescue for an approval of
        # the requirement that went unfilled.
        self.assertEqual(
            outcome.merges[0]["merged_into_visual_requirement"],
            "Coffee cherries being picked by hand",
        )

    def test_a_sibling_of_the_same_group_outranks_a_roomier_shot_of_the_next_group(self):
        # Both sides border the open beat, and the survivor is otherwise chosen by
        # how much room its asset has left -- so a long cross-group clip would win on
        # headroom alone. The sibling was written for this very moment, so it has to
        # win regardless.
        beats = self._beats(3, group_ids=[1, 1, 2])
        outcome = self._run(
            beats=beats,
            approved_queries={"beans query 1", "beans query 3"},
            durations={"beans query 1": 12.0, "beans query 3": 20.0},
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.merges[0]["merged_into_visual_item_index"], 1)
        self.assertEqual(outcome.merges[0]["merge_scope"], "same_semantic_group")
        self.assertEqual(outcome.merged[0].duration_policy, "unfillable_beat_merged")
        self.assertAlmostEqual(outcome.merged[0].end_time, 5.6)

    def test_two_equally_roomy_neighbours_hand_the_window_to_the_previous_shot(self):
        # Both neighbours are filled, both border the open beat, and both assets are
        # the same length, so nothing real separates them. Their merged windows are
        # the same length too -- but each is measured from a different pair of
        # timeline stamps, so the two headrooms differ by a rounding error of roughly
        # 1e-15 seconds. Ranking them raw let that noise choose, and the later shot
        # won every time. The previous shot has to win: a survivor keeps its own
        # index, so absorbing forwards keeps the rewritten timeline in index order.
        outcome = self._run(
            beats=self._beats(3),
            approved_queries={"beans query 1", "beans query 3"},
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.merges[0]["merged_into_visual_item_index"], 1)
        self.assertEqual([beat.index for beat in outcome.merged], [1, 3])
        self.assertAlmostEqual(outcome.merged[0].start_time, 0.0)
        self.assertAlmostEqual(outcome.merged[0].end_time, 5.6)
        self.assertAlmostEqual(outcome.merged[1].start_time, 5.6)

    def test_a_clearly_roomier_next_shot_absorbs_rather_than_the_previous_one(self):
        # The tolerance that keeps rounding noise out of that choice must not harden
        # into "always the previous shot". Both sides can cover the combined window
        # here, but the previous asset would have only a fraction of a second to
        # spare while the next one has plenty, and room left over is exactly what
        # decides whether a later merge on that same survivor stays free.
        outcome = self._run(
            beats=self._beats(3),
            approved_queries={"beans query 1", "beans query 3"},
            durations={"beans query 1": 6.0, "beans query 3": 20.0},
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.merges[0]["merged_into_visual_item_index"], 3)
        self.assertEqual(outcome.merges[0]["merge_fill"], "neighbour_window_extended")
        self.assertEqual([beat.index for beat in outcome.merged], [1, 3])
        # The previous shot keeps its own window untouched.
        self.assertAlmostEqual(outcome.merged[0].end_time, 2.8)
        self.assertEqual(outcome.merged[0].duration_policy, "semantic_original")
        self.assertAlmostEqual(outcome.merged[1].start_time, 2.8)
        self.assertAlmostEqual(outcome.merged[1].end_time, 8.4)

    def test_a_survivor_that_crossed_a_boundary_keeps_saying_so(self):
        # A survivor can absorb on both sides. The policy names the weakest claim any
        # of those absorptions made, so a later same-group merge cannot quietly
        # upgrade a beat that is already covering a neighbouring group's window.
        beats = self._beats(3, group_ids=[2, 1, 1])
        outcome = self._run(
            beats=beats,
            approved_queries={"beans query 2"},
            merge_ceiling=2,
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.paths, ["D:/task/beans-query-2.mp4"])
        self.assertEqual(len(outcome.merged), 1)
        self.assertAlmostEqual(outcome.merged[0].duration, 8.4)
        self.assertEqual(
            [run["merge_scope"] for run in outcome.merges],
            ["adjacent_semantic_group", "same_semantic_group"],
        )
        self.assertEqual(
            outcome.merged[0].duration_policy, "unfillable_beat_cross_group_merged"
        )

    def test_a_beat_that_nothing_borders_is_refused_rather_than_guessed_at(self):
        outcome = self._run(
            beats=self._beats(2),
            approved_queries=set(),
            merge_ceiling=2,
        )

        self.assertIsNotNone(outcome.error)
        # Both beats stayed open in the hope of a merge, and neither can rescue the
        # other: a merge needs footage that already passed, not another open window.
        self.assertIn("MERGE_NEIGHBOUR_UNAVAILABLE", outcome.decisions)
        self.assertIn(
            "no filled shot borders it in the rewritten timeline", str(outcome.error)
        )
        self.assertEqual(outcome.merged, [])
        self.assertEqual(outcome.merges, [])
        self.assertEqual(outcome.records, [])

    def test_switching_off_the_cross_group_rescue_fails_the_video_instead(self):
        outcome = self._run(
            beats=self._beats(3, group_ids=[1, 2, 3]),
            approved_queries={"beans query 1", "beans query 3"},
            merge_ceiling=1,
            cross_group_merge=False,
        )

        self.assertIsNotNone(outcome.error)
        # With the rescue off, a lone shot can never be absorbed, so the run must
        # stop at the beat that failed rather than pay for beat 3 to find out.
        self.assertEqual(outcome.searches, ["beans query 1", "beans query 2"])
        self.assertEqual(outcome.merged, [])
        self.assertEqual(outcome.merges, [])

    def test_a_sibling_clip_too_short_for_the_merged_window_is_reselected_once(self):
        outcome = self._run(
            beats=self._beats(2),
            approved_queries={"beans query 1"},
            durations={"beans query 1": 4.0},
        )

        self.assertIsNone(outcome.error)
        # The approved asset covered one beat but cannot cover both, so the merge
        # falls back to one ordinary selection round for a longer clip.
        self.assertEqual(outcome.merges[0]["merge_fill"], "fresh_selection_round")
        self.assertEqual(outcome.paths, ["D:/task/beans-query-1-spare.mp4"])
        self.assertEqual(
            outcome.searches, ["beans query 1", "beans query 2", "beans query 1"]
        )
        # The requirement is not rewritten for a reselection: it is the group's own,
        # which both beats already shared.
        self.assertEqual(
            outcome.service.select_best_candidate.call_args.kwargs["narration_text"],
            self.GROUP_REQUIREMENT,
        )
        self.assertEqual(len(outcome.records), 1)
        record = outcome.records[0]
        self.assertEqual(record["visual_beat_index"], 1)
        self.assertEqual(record["local_file"], "beans-query-1-spare.mp4")
        self.assertAlmostEqual(record["required_target_duration"], 5.6)
        self.assertAlmostEqual(
            record["source_end_time"] - record["source_start_time"], 5.6, places=3
        )
        self.assertEqual(len(outcome.merged), 1)
        self.assertAlmostEqual(outcome.merged[0].duration, 5.6)

    def test_a_failed_segmentation_refuses_the_merge_instead_of_guessing_a_window(self):
        outcome = self._run(
            beats=self._beats(2),
            approved_queries={"beans query 1"},
            segment_failures_after=1,
        )

        self.assertIsNotNone(outcome.error)
        self.assertIn("could not be absorbed by a neighbouring shot", str(outcome.error))
        self.assertIn("MERGE_SEGMENTATION_FAILED", outcome.decisions)
        self.assertEqual(outcome.merged, [])
        self.assertEqual(outcome.merges, [])
        # A refused merge must leave the survivor exactly as it was, still cut for
        # its own beat, rather than half-restamped for a window it never got.
        self.assertEqual(len(outcome.records), 1)
        record = outcome.records[0]
        self.assertAlmostEqual(record["required_target_duration"], 2.8)
        self.assertAlmostEqual(
            record["source_end_time"] - record["source_start_time"], 2.8, places=3
        )

    def test_a_merged_window_is_bought_in_source_seconds_at_the_playback_speed(self):
        outcome = self._run(
            beats=self._beats(2),
            approved_queries={"beans query 1"},
            clip_speed=2.0,
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.merges[0]["merge_fill"], "neighbour_window_extended")
        # A merged beat obeys the same speed rule as any other: 5.6s of timeline at
        # 2x needs 11.2s of source, not 5.6s stretched.
        self.assertAlmostEqual(
            outcome.segmented[-1]["requested_source_duration"], 11.2
        )
        record = outcome.records[0]
        self.assertAlmostEqual(record["required_source_duration"], 11.2)
        self.assertAlmostEqual(
            record["source_end_time"] - record["source_start_time"], 11.2, places=3
        )

    def test_the_merged_timeline_renders_as_one_gap_free_segment(self):
        outcome = self._run(beats=self._beats(2), approved_queries={"beans query 1"})

        with patch.object(
            material.task_artifacts,
            "read_script_data",
            return_value={"material_sources": outcome.records},
        ):
            segments = material.load_render_segments(
                "merge-unfillable-beat",
                outcome.paths,
                outcome.merged,
                clip_speed=1.0,
                audio_duration=5.6,
            )

        # This is the assertion the whole rung exists for: the renderer accepts the
        # rewritten timeline, covers the narration to its end, and plays the widened
        # window at normal speed.
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.visual_beat_index, 1)
        self.assertAlmostEqual(segment.target_start, 0.0)
        self.assertAlmostEqual(segment.target_end, 5.6)
        self.assertAlmostEqual(segment.target_duration, 5.6)
        self.assertAlmostEqual(segment.source_end - segment.source_start, 5.6, places=3)
        self.assertEqual(segment.playback_speed, 1.0)
        self.assertEqual(segment.file_path, "D:/task/beans-query-1.mp4")

    def test_a_merged_timeline_must_still_start_at_zero_and_never_overlap(self):
        beats = self._beats(3)
        # Dropping the middle beat without giving its window to anybody is exactly
        # the bug this validator exists to catch.
        with self.assertRaises(ValueError) as gap:
            material.validate_merged_beat_timeline([beats[0], beats[2]])
        self.assertIn("gap or overlap", str(gap.exception))
        with self.assertRaises(ValueError) as start:
            material.validate_merged_beat_timeline([beats[1], beats[2]])
        self.assertIn("must start at zero", str(start.exception))
        with self.assertRaises(ValueError) as order:
            material.validate_merged_beat_timeline([beats[0], beats[0]])
        self.assertIn("must increase", str(order.exception))
        with self.assertRaises(ValueError):
            material.validate_merged_beat_timeline([])
        # Non-contiguous indexes are the normal state of a merged timeline and must
        # be accepted, which is the one rule the script-stage validator cannot share.
        material.validate_merged_beat_timeline(
            [material._merged_visual_beat(beats[0], beats[1]), beats[2]]
        )

    def test_the_merge_ceiling_reads_as_a_whole_number_of_beats(self):
        without_key = dict(config.app)
        without_key.pop("smart_material_max_merged_beats", None)
        with patch.dict(config.app, without_key, clear=True):
            # A third of the beats: enough to survive a few unlucky requirements,
            # not enough to quietly collapse a fourteen-shot edit into three.
            self.assertEqual(material.max_merged_beats_per_video(14), 4)
            self.assertEqual(material.max_merged_beats_per_video(3), 1)
            # Too few beats for a third of one to exist means no merging, which is
            # the pre-merge behavior.
            self.assertEqual(material.max_merged_beats_per_video(2), 0)
            self.assertEqual(material.max_merged_beats_per_video(0), 0)
        for value, expected in (
            (0, 0),
            (2, 2),
            ("3", 3),
            (" 1 ", 1),
            # Trading visual variety for completion is a deliberate choice, so a
            # ceiling above a third is honored as written rather than clamped.
            (9, 9),
            # A negative or unreadable ceiling has no reading that is safer than
            # the default, and reading it as "unlimited" would be the worst one.
            (-1, 4),
            ("many", 4),
            (None, 4),
            ("", 4),
        ):
            with self.subTest(value=value):
                with patch.dict(
                    config.app, {"smart_material_max_merged_beats": value}
                ):
                    self.assertEqual(material.max_merged_beats_per_video(14), expected)


class TestPerVideoAnalysisBudget(unittest.TestCase):
    """One damaged video must not be able to cost a whole quota.

    The per-round ceiling bounds a single item; nothing bounded a run until now,
    and the failure ladder made that gap matter: a beat nothing can fill no longer
    stops the video, so several hard beats can each spend two full rounds before
    the run ends. These tests hold the one line that makes such a ceiling safe
    rather than merely cheap — it is spent only on *recovery*, so every remaining
    beat still gets its first look and the free rungs of the ladder still finish
    the video, while the expensive rungs are abandoned instead of bought.
    """

    GROUP_REQUIREMENT = "Coffee beans drying in sunlight"

    @staticmethod
    def _settings(**overrides):
        settings = {
            "batch_size": 5,
            "max_candidates": 15,
            "minimum_score": 0.7,
            "strong_early_stop_score": 0.9,
            "preferred_max_source_duration": 30,
            "concurrency": 5,
            "fail_closed": True,
        }
        settings.update(overrides)
        return settings

    def _beats(self, count, *, duration=2.8):
        # One semantic group, three phrasings each: the group is what makes the
        # merge rung legal, and the phrasings are what the ceiling has to be able
        # to abandon.
        beats = []
        for position in range(1, count + 1):
            beat = _visual_beat(
                index=position,
                semantic_group_id=1,
                duration=duration,
                requirement=self.GROUP_REQUIREMENT,
                query=f"beans query {position}",
            )
            beat.search_queries = [
                f"beans query {position}",
                f"beans alt {position}",
                f"beans extra {position}",
            ]
            beats.append(beat)
        return beats

    @staticmethod
    def _candidate_for(provider, term, duration=12):
        # Distinct per provider and per phrasing: an asset this item already has a
        # verdict on is skipped, so overlapping pages would measure the exclusion
        # set instead of the budget.
        asset_id = f"{provider}-{term}".replace(" ", "-")
        candidate = _candidate(
            asset_id,
            duration=duration,
            url=f"https://videos.example/{asset_id}.mp4",
        )
        candidate.provider = provider
        candidate.source_info.update({"provider": provider, "asset_id": asset_id})
        return candidate

    def _run(
        self,
        *,
        beats,
        approved_queries,
        video_budget,
        merge_ceiling=1,
        durations=None,
        rewrite=None,
        providers=("pexels", "pixabay"),
    ):
        durations = durations or {}
        searches: list[tuple[str, str]] = []
        segmented: list[dict] = []
        merged_beats: list[VisualBeat] = []

        def decompose(requirements):
            return {
                material.llm.normalize_visual_requirement(requirement): (
                    _requirement_spec(requirement)
                )
                for requirement in requirements
            }

        def provider_search(provider):
            def search(**kwargs):
                term = kwargs["search_term"]
                searches.append((provider, term))
                # A real catalog answers with more than one clip, and the spare is
                # what a merge falls back to when the winner is too short to widen.
                return [
                    self._candidate_for(provider, term, durations.get(term, 12)),
                    self._candidate_for(provider, f"{term} spare", 20),
                ]

            return search

        def select_best(**kwargs):
            candidates = kwargs["candidates"]
            stats = {
                "api_candidates_analyzed": len(candidates),
                "source_seconds_analyzed": 12.0 * len(candidates),
                "candidate_evaluations": [],
            }
            if kwargs["search_query"] not in approved_queries:
                return None, stats
            winner = candidates[0]
            winner.overall_score = 0.95
            stats["candidate_evaluations"] = [
                {
                    "provider": winner.provider,
                    "provider_asset_id": winner.source_info["asset_id"],
                    "accepted": True,
                    "ranking_position": 1,
                    "overall_score": 0.95,
                }
            ]
            return winner, stats

        def segment(**kwargs):
            segmented.append(kwargs)
            return {
                "source_start_time": 0.0,
                "source_end_time": float(kwargs["source_duration"]),
                "description": "the requested action",
            }

        service = SimpleNamespace(
            candidate_selection_settings=lambda: self._settings(),
            select_best_candidate=MagicMock(side_effect=select_best),
            segment_winner=MagicMock(side_effect=segment),
        )
        rewrite_mock = MagicMock(return_value={})
        queries_mock = MagicMock(return_value={})
        if rewrite:
            # Grounded in the same narration, which is what the rewrite rung is
            # allowed to do; here it would succeed, so a skipped rewrite is
            # visibly the ceiling's doing and not a failed recovery.
            rewrite_mock = MagicMock(
                return_value={
                    beat.index: {
                        "visual_requirement": rewrite["requirement"],
                        "narration_basis": beat.spoken_text,
                    }
                    for beat in beats
                }
            )
            queries_mock = MagicMock(
                return_value={beat.index: [rewrite["query"]] for beat in beats}
            )
        error = None
        paths: list[str] = []
        with (
            patch.object(
                material.llm,
                "generate_visual_requirement_specs",
                side_effect=decompose,
            ),
            patch.object(
                material.llm, "generate_alternative_visual_requirements", rewrite_mock
            ),
            patch.object(material.llm, "generate_visual_slot_queries", queries_mock),
            patch.object(
                material,
                "save_video",
                side_effect=lambda **kwargs: (
                    f"D:/task/{Path(str(kwargs['video_url'])).name}"
                ),
            ),
            patch.dict(
                config.app,
                {
                    "smart_material_max_query_variants": 3,
                    # Pinned high so the per-round ceiling cannot be what cuts a
                    # run here; everything these tests observe is per-video.
                    "smart_material_max_analyzed_candidates_per_round": 75,
                    "smart_material_max_analyzed_candidates_per_video": video_budget,
                    "smart_material_max_merged_beats": merge_ceiling,
                    "smart_material_requirement_rewrite": bool(rewrite),
                },
            ),
            patch.object(
                material.task_artifacts, "patch_script_data", return_value=True
            ) as persist,
        ):
            try:
                paths = material._download_videos_by_script_order_smart(
                    task_id="per-video-analysis-budget",
                    search_terms=[beat.search_queries[0] for beat in beats],
                    visual_beats=list(beats),
                    provider_searches=[
                        (provider, provider_search(provider))
                        for provider in providers
                    ],
                    video_aspect=VideoAspect.portrait,
                    max_clip_duration=4,
                    material_directory="",
                    clip_speed=1.0,
                    twelvelabs_service=service,
                    merged_beats_out=merged_beats,
                )
            except material.SmartMaterialSelectionError as exc:
                error = exc
        runs = persist.call_args.kwargs["semantic_verifier_runs"]
        return SimpleNamespace(
            paths=paths,
            error=error,
            merged=merged_beats,
            records=persist.call_args.kwargs["material_sources"],
            runs=runs,
            decisions=[run["final_decision"] for run in runs],
            merges=[
                run for run in runs if run["final_decision"] == "UNFILLABLE_BEAT_MERGED"
            ],
            searches=searches,
            segmented=segmented,
            service=service,
            rewrite=rewrite_mock,
        )

    def _record(self, outcome, decision):
        matching = [run for run in outcome.runs if run["final_decision"] == decision]
        self.assertEqual(len(matching), 1, outcome.decisions)
        return matching[0]

    def test_a_beat_still_gets_its_first_look_after_the_video_is_out_of_budget(self):
        beats = self._beats(2)
        # Four analyses is two searches at two candidates each: beat 1 burns the
        # whole budget without filling, which is exactly the state in which a
        # ceiling could do damage if it were checked at the wrong place.
        outcome = self._run(
            beats=beats, approved_queries={"beans query 2"}, video_budget=4
        )

        self.assertIsNone(outcome.error)
        # Beat 2 searched once and was filled. Had the ceiling been applied before
        # first looks instead of before recovery, this beat would never have been
        # searched and the video would have failed with money still unspent.
        self.assertEqual(
            outcome.searches,
            [
                ("pexels", "beans query 1"),
                ("pexels", "beans alt 1"),
                ("pexels", "beans query 2"),
            ],
        )
        self.assertEqual(outcome.paths, ["D:/task/pexels-beans-query-2.mp4"])
        # Beat 1's round was cut with the narrowed number, not the pinned 75, which
        # is the proof that the video's ceiling reached the selector at all.
        self.assertEqual(
            self._record(outcome, "ANALYSIS_BUDGET_EXHAUSTED")["analysis_budget"], 4
        )

    def test_the_free_merge_rung_still_finishes_a_video_with_no_budget_left(self):
        outcome = self._run(
            beats=self._beats(2), approved_queries={"beans query 2"}, video_budget=4
        )

        # Extending an already-approved clip buys no analysis, so refusing it once
        # the money is gone would throw the video away for nothing. Beat 1's window
        # is absorbed and the narration is still covered end to end.
        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.merges[0]["merge_fill"], "neighbour_window_extended")
        self.assertEqual(len(outcome.merged), 1)
        self.assertAlmostEqual(outcome.merged[0].end_time, 5.6)
        self.assertEqual(outcome.merged[0].duration_policy, "unfillable_beat_merged")
        self.assertEqual(outcome.service.select_best_candidate.call_count, 3)

    def test_a_spent_video_abandons_the_rest_of_the_cascade_instead_of_buying_it(self):
        outcome = self._run(
            beats=self._beats(2), approved_queries=set(), video_budget=4
        )

        self.assertIsNotNone(outcome.error)
        # Beat 2 got its one look and then nothing: no second phrasing, no second
        # catalog. Nine searches and nine pages of analysis were left unbought.
        self.assertEqual(
            outcome.searches,
            [
                ("pexels", "beans query 1"),
                ("pexels", "beans alt 1"),
                ("pexels", "beans query 2"),
            ],
        )
        cutoffs = [
            run
            for run in outcome.runs
            if run["final_decision"] == "ANALYSIS_BUDGET_EXHAUSTED"
        ]
        self.assertEqual([run["analysis_budget"] for run in cutoffs], [4, 1])

    def test_without_a_ceiling_the_same_run_buys_every_phrasing_and_provider(self):
        # The falsifier for the test above: with the ceiling removed, the identical
        # scenario spends every phrasing of both catalogs on both beats. Twelve
        # searches instead of three is the size of what the ceiling is saving.
        outcome = self._run(
            beats=self._beats(2),
            approved_queries=set(),
            video_budget=0,
            merge_ceiling=1,
        )

        self.assertIsNotNone(outcome.error)
        self.assertEqual(len(outcome.searches), 12)
        self.assertNotIn("ANALYSIS_BUDGET_EXHAUSTED", outcome.decisions)

    def test_the_requirement_rewrite_is_refused_when_the_budget_is_gone(self):
        rewrite = {
            "requirement": "Roasted coffee beans falling into a bag",
            "query": "roasted beans falling bag",
        }
        outcome = self._run(
            beats=self._beats(2),
            approved_queries={"beans query 1", rewrite["query"]},
            video_budget=4,
            rewrite=rewrite,
        )

        # The rewrite is the most expensive rung — a new requirement means a new
        # search and a new page of analyses for a beat that already failed once —
        # and it is the one rung the free merge can substitute for. So it is the
        # first thing a spent video gives up, and the LLM is not even asked.
        self.assertEqual(outcome.rewrite.call_count, 0)
        exhausted = self._record(outcome, "VIDEO_ANALYSIS_BUDGET_EXHAUSTED")
        self.assertEqual(exhausted["visual_item_index"], 2)
        self.assertEqual(exhausted["candidates_analyzed"], 4)
        self.assertEqual(exhausted["video_analysis_budget"], 4)
        # Giving up the rewrite must not give up the video: the sibling absorbs the
        # window for free and the run still ends with a renderable timeline.
        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.paths, ["D:/task/pexels-beans-query-1.mp4"])
        self.assertEqual(outcome.merges[0]["merge_fill"], "neighbour_window_extended")

    def test_the_rewrite_is_still_bought_while_the_video_has_budget_left(self):
        rewrite = {
            "requirement": "Roasted coffee beans falling into a bag",
            "query": "roasted beans falling bag",
        }
        outcome = self._run(
            beats=self._beats(2),
            approved_queries={"beans query 1", rewrite["query"]},
            video_budget=0,
            rewrite=rewrite,
        )

        # Same scenario, ceiling removed: the rewrite runs and fills beat 2, so the
        # refusal above is the ceiling's decision and not a broken recovery path.
        self.assertEqual(outcome.rewrite.call_count, 1)
        self.assertIsNone(outcome.error)
        self.assertEqual(len(outcome.paths), 2)
        self.assertEqual(outcome.merges, [])
        self.assertNotIn("VIDEO_ANALYSIS_BUDGET_EXHAUSTED", outcome.decisions)

    def test_a_merge_that_would_have_to_buy_a_fresh_round_is_refused(self):
        outcome = self._run(
            beats=self._beats(2),
            approved_queries={"beans query 2"},
            video_budget=4,
            # The survivor's approved clip covers its own 2.8s beat but not the
            # 5.6s merged window, so this merge is the paid variety.
            durations={"beans query 2": 4.0},
        )

        # A fresh selection round for a merged window is a whole new page of
        # analyses, and it is the last thing left that could still cost money after
        # the ceiling is reached. Refusing it is what makes the bound real.
        self.assertIsNotNone(outcome.error)
        refusal = self._record(outcome, "MERGE_ANALYSIS_BUDGET_EXHAUSTED")
        self.assertEqual(refusal["visual_item_index"], 1)
        self.assertIn("spent its analysis budget of 4", refusal["reason"])
        self.assertEqual(outcome.merges, [])
        self.assertEqual(outcome.merged, [])
        # Nothing new was searched or analyzed after the refusal.
        self.assertEqual(
            outcome.searches,
            [
                ("pexels", "beans query 1"),
                ("pexels", "beans alt 1"),
                ("pexels", "beans query 2"),
            ],
        )

    def test_with_budget_left_that_same_merge_buys_its_fresh_round(self):
        outcome = self._run(
            beats=self._beats(2),
            approved_queries={"beans query 2"},
            video_budget=0,
            durations={"beans query 2": 4.0},
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.merges[0]["merge_fill"], "fresh_selection_round")
        self.assertEqual(outcome.paths, ["D:/task/pexels-beans-query-2-spare.mp4"])
        self.assertNotIn("MERGE_ANALYSIS_BUDGET_EXHAUSTED", outcome.decisions)

    def test_a_healthy_video_never_reaches_the_ceiling(self):
        beats = self._beats(3)
        # 90 is what an unset key derives for three beats at fifteen candidates.
        self.assertEqual(material.analysis_budget_per_video(3, 15), 90)
        outcome = self._run(
            beats=beats,
            approved_queries={beat.search_queries[0] for beat in beats},
            video_budget=90,
            merge_ceiling=1,
        )

        self.assertIsNone(outcome.error)
        self.assertEqual(len(outcome.paths), 3)
        self.assertEqual(outcome.merged, [])
        # Three searches, six analyses: a video whose beats settle on their first
        # look spends a fifteenth of the default ceiling, which is why the default
        # can be loose enough to leave a hard beat its full ladder.
        self.assertEqual(len(outcome.searches), 3)
        self.assertEqual(outcome.service.select_best_candidate.call_count, 3)
        self.assertEqual(
            [decision for decision in outcome.decisions if "BUDGET" in decision], []
        )

    def test_the_per_video_budget_reads_as_a_whole_number_of_analyses(self):
        without_key = dict(config.app)
        without_key.pop("smart_material_max_analyzed_candidates_per_video", None)
        with patch.dict(config.app, without_key, clear=True):
            # Two full candidate pages per beat: a healthy video spends well under
            # one, so this only bites a run that is already paying for failures.
            self.assertEqual(material.analysis_budget_per_video(14, 15), 420)
            self.assertEqual(material.analysis_budget_per_video(3, 15), 90)
            # Nonsense inputs still leave a budget an item can spend, because a
            # ceiling of zero would read as "unlimited" further down.
            self.assertEqual(material.analysis_budget_per_video(0, 0), 2)
        for value, expected in (
            (0, 0),
            (200, 200),
            ("200", 200),
            (" 6 ", 6),
            # A negative or unreadable ceiling has no reading safer than the
            # default, and reading it as "unlimited" would be the worst one.
            (-5, 420),
            ("plenty", 420),
            (None, 420),
            ("", 420),
        ):
            with self.subTest(value=value):
                with patch.dict(
                    config.app,
                    {"smart_material_max_analyzed_candidates_per_video": value},
                ):
                    self.assertEqual(
                        material.analysis_budget_per_video(14, 15), expected
                    )

    def test_an_exhausted_video_budget_never_reads_as_no_ceiling_at_all(self):
        # Zero means "no ceiling" everywhere else in this module, so the one number
        # a spent video must never produce is zero. One is both safe and exactly
        # the intended behavior: the first look happens, the cascade does not.
        self.assertEqual(material._effective_round_budget(75, 420, 420), 1)
        self.assertEqual(material._effective_round_budget(75, 420, 900), 1)
        self.assertEqual(material._effective_round_budget(0, 420, 420), 1)
        # A video with room left is bounded by whichever ceiling is tighter.
        self.assertEqual(material._effective_round_budget(75, 420, 0), 75)
        self.assertEqual(material._effective_round_budget(75, 420, 400), 20)
        self.assertEqual(material._effective_round_budget(0, 420, 100), 320)
        # And with no per-video ceiling the per-round number passes through
        # untouched, including its own "no ceiling" reading.
        self.assertEqual(material._effective_round_budget(75, 0, 900), 75)
        self.assertEqual(material._effective_round_budget(0, 0, 900), 0)

