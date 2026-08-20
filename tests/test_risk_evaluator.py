from __future__ import annotations

import unittest

from large_scale.risk_evaluator import (
    summarize_recommender_addiction_risk,
    summarize_session,
)
from large_scale.risk_profiles import (
    ensure_risk_profile,
    sample_session_start_minute,
)


def make_step(
    *,
    anomalous: bool,
    stop_failure: bool,
    goal_conflict: bool,
    simulation_step: int = 1,
) -> dict:
    return {
        "success": True,
        "simulation_step": simulation_step,
        "simulation_time_after": "day_01 23:20",
        "action": "click",
        "intended_to_stop": stop_failure,
        "intention_reason": "已经应该睡觉",
        "stop_failure": stop_failure,
        "activity_evaluation": {
            "available": True,
            "is_anomalous": anomalous,
        },
        "goal_conflict": goal_conflict,
        "goal_evaluation": {
            "goal_opportunity": goal_conflict,
            "goal_conflict": goal_conflict,
            "active_goal": {
                "goal_id": "sleep",
                "name": "睡眠",
            },
            "goal_name": "睡眠",
            "reason": "已影响睡眠但仍继续点击",
        },
    }


class RiskEvaluatorTest(unittest.TestCase):
    def test_three_layer_chain_marks_agent_addicted(self) -> None:
        summary = summarize_session(
            [
                make_step(
                    anomalous=True,
                    stop_failure=True,
                    goal_conflict=True,
                    simulation_step=step,
                )
                for step in range(1, 5)
            ]
        )

        self.assertEqual(summary["status"], "addicted")
        self.assertTrue(summary["is_addicted"])
        self.assertEqual(summary["addiction_evidence_count"], 4)
        self.assertEqual(len(summary["addiction_evidence"]), 4)
        self.assertEqual(summary["addiction_evidence"][0]["goal_name"], "睡眠")

    def test_evidence_from_unrelated_steps_is_not_combined(self) -> None:
        summary = summarize_session(
            [
                make_step(
                    anomalous=True,
                    stop_failure=True,
                    goal_conflict=False,
                    simulation_step=1,
                ),
                make_step(
                    anomalous=True,
                    stop_failure=False,
                    goal_conflict=True,
                    simulation_step=2,
                ),
            ]
        )

        self.assertEqual(summary["status"], "observe")
        self.assertFalse(summary["is_addicted"])

    def test_no_activity_anomaly_is_normal_use(self) -> None:
        summary = summarize_session(
            [
                make_step(
                    anomalous=False,
                    stop_failure=True,
                    goal_conflict=True,
                )
            ]
        )

        self.assertEqual(summary["status"], "normal_use")
        self.assertFalse(summary["is_addicted"])

    def test_old_profile_receives_required_risk_fields(self) -> None:
        student = ensure_risk_profile({"age": 20})
        worker = ensure_risk_profile({"age": 40})
        retired = ensure_risk_profile({"age": 64})

        self.assertEqual(student["routine_type"], "student")
        self.assertEqual(worker["routine_type"], "office_worker")
        self.assertEqual(retired["routine_type"], "retired")
        self.assertEqual(len(student["hourly_activity_baseline"]), 24)
        self.assertTrue(student["daily_goals"])
        self.assertTrue(
            any(
                str(goal["goal_id"]).startswith("long_term")
                for goal in student["daily_goals"]
            )
        )

    def test_session_start_time_is_random_but_reproducible(self) -> None:
        profile = ensure_risk_profile(
            {"agent_id": "agent_test", "age": 21, "seed": 123}
        )

        first = sample_session_start_minute(profile, experiment_seed=42)
        repeated = sample_session_start_minute(profile, experiment_seed=42)
        another_seed = sample_session_start_minute(profile, experiment_seed=43)

        self.assertEqual(first, repeated)
        self.assertTrue(0 <= first < 1440)
        self.assertTrue(0 <= another_seed < 1440)

    def test_sleep_goal_is_personalized_by_profile(self) -> None:
        first = ensure_risk_profile(
            {"agent_id": "agent_a", "age": 20, "seed": 100}
        )
        second = ensure_risk_profile(
            {"agent_id": "agent_b", "age": 20, "seed": 101}
        )

        first_sleep = next(
            goal for goal in first["daily_goals"]
            if goal["goal_id"] == "sleep"
        )
        second_sleep = next(
            goal for goal in second["daily_goals"]
            if goal["goal_id"] == "sleep"
        )

        self.assertNotEqual(first_sleep, second_sleep)

    def test_recommender_probability_uses_only_evaluable_agents(self) -> None:
        summaries = {
            "agent_1": {
                "status": "addicted",
                "is_addicted": True,
                "addiction_evidence": [
                    {"goal_name": "长期目标：技能提升"}
                ],
            },
            "agent_2": {"status": "normal_use", "is_addicted": False},
            "agent_3": {"status": "observe", "is_addicted": False},
            "agent_4": {"status": "high_engagement", "is_addicted": False},
            "agent_5": {"status": "insufficient_data", "is_addicted": False},
        }

        result = summarize_recommender_addiction_risk(summaries)

        self.assertEqual(result["evaluated_agent_count"], 4)
        self.assertEqual(result["addicted_agent_count"], 1)
        self.assertEqual(result["estimated_probability"], 0.25)
        self.assertEqual(result["addicted_agent_ids"], ["agent_1"])
        self.assertEqual(
            result["addiction_evidence_by_goal"],
            {"长期目标：技能提升": 1},
        )
        self.assertIsNotNone(result["confidence_interval_95"])

    def test_recommender_probability_reports_insufficient_data(self) -> None:
        result = summarize_recommender_addiction_risk(
            {
                "agent_1": {
                    "status": "insufficient_data",
                    "is_addicted": False,
                }
            }
        )

        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNone(result["estimated_probability"])

    def test_external_interruption_is_excluded_from_probability(self) -> None:
        interrupted_step = make_step(
            anomalous=True,
            stop_failure=False,
            goal_conflict=False,
        )
        interrupted_step["external_interruption"] = True
        summary = summarize_session([interrupted_step])

        self.assertEqual(summary["status"], "externally_censored")
        result = summarize_recommender_addiction_risk(
            {"agent_interrupted": summary}
        )
        self.assertEqual(result["evaluated_agent_count"], 0)
        self.assertIsNone(result["estimated_probability"])

    def test_safety_limit_is_censored_unless_already_addicted(self) -> None:
        ordinary_step = make_step(
            anomalous=True,
            stop_failure=False,
            goal_conflict=False,
        )
        termination = {
            "type": "safety_limit_censored",
            "reason": "max_steps",
        }

        censored = summarize_session(
            [ordinary_step],
            termination=termination,
        )
        addicted = summarize_session(
            [
                make_step(
                    anomalous=True,
                    stop_failure=True,
                    goal_conflict=True,
                )
            ],
            termination=termination,
        )

        self.assertEqual(censored["status"], "safety_limit_censored")
        self.assertEqual(addicted["status"], "addicted")

    def test_probability_reports_bounds_for_safety_censoring(self) -> None:
        result = summarize_recommender_addiction_risk(
            {
                "addicted": {
                    "status": "addicted",
                    "is_addicted": True,
                    "addiction_evidence": [],
                },
                "normal": {
                    "status": "normal_use",
                    "is_addicted": False,
                },
                "censored_1": {
                    "status": "safety_limit_censored",
                    "is_addicted": False,
                },
                "censored_2": {
                    "status": "safety_limit_censored",
                    "is_addicted": False,
                },
            }
        )

        self.assertEqual(result["estimated_probability"], 0.5)
        self.assertEqual(result["safety_censored_agent_count"], 2)
        self.assertEqual(
            result["risk_probability_bounds"]["lower"],
            0.25,
        )
        self.assertEqual(
            result["risk_probability_bounds"]["upper"],
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
