from __future__ import annotations

import unittest

from large_scale.risk_profiles import ensure_risk_profile
from large_scale.special_events import (
    active_external_interruption,
    apply_special_event,
    select_special_event,
)


class SpecialEventsTest(unittest.TestCase):
    def test_none_and_random_selection_are_supported(self) -> None:
        self.assertIsNone(select_special_event("none", seed=42))
        first = select_special_event("random", seed=42)
        repeated = select_special_event("random", seed=42)
        self.assertEqual(first, repeated)
        self.assertIsNotNone(first)

    def test_summer_vacation_only_changes_student_schedule(self) -> None:
        event = select_special_event("summer_vacation", seed=42)
        student = ensure_risk_profile(
            {"agent_id": "student", "age": 20, "seed": 20}
        )
        worker = ensure_risk_profile(
            {"agent_id": "worker", "age": 40, "seed": 40}
        )
        original_sleep = next(
            goal for goal in student["daily_goals"]
            if goal["goal_id"] == "sleep"
        )

        changed_student = apply_special_event(student, event)
        changed_worker = apply_special_event(worker, event)
        changed_sleep = next(
            goal for goal in changed_student["daily_goals"]
            if goal["goal_id"] == "sleep"
        )

        self.assertTrue(changed_student["special_event"]["applicable"])
        self.assertFalse(changed_worker["special_event"]["applicable"])
        self.assertFalse(
            any(
                goal.get("goal_id") == "study"
                for goal in changed_student["daily_goals"]
            )
        )
        self.assertEqual(
            changed_sleep["start_minute"],
            (original_sleep["start_minute"] + 60) % 1440,
        )

    def test_holiday_removes_work_goal(self) -> None:
        event = select_special_event("holiday", seed=42)
        worker = ensure_risk_profile(
            {"agent_id": "worker", "age": 40, "seed": 40}
        )

        changed = apply_special_event(worker, event)

        self.assertFalse(
            any(
                goal.get("goal_id") == "work"
                for goal in changed["daily_goals"]
            )
        )
        self.assertTrue(
            any(
                str(goal.get("goal_id", "")).startswith("long_term")
                for goal in changed["daily_goals"]
            )
        )

    def test_power_outage_is_detected_only_inside_window(self) -> None:
        event = select_special_event("power_outage", seed=42)
        profile = ensure_risk_profile(
            {"agent_id": "worker", "age": 40, "seed": 40}
        )
        changed = apply_special_event(profile, event)

        self.assertIsNotNone(
            active_external_interruption(changed, 20 * 60 + 30)
        )
        self.assertIsNone(
            active_external_interruption(changed, 21 * 60 + 30)
        )

    def test_exam_week_adds_student_review_goal(self) -> None:
        event = select_special_event("exam_week", seed=42)
        student = ensure_risk_profile(
            {"agent_id": "student", "age": 20, "seed": 20}
        )

        changed = apply_special_event(student, event)

        self.assertTrue(changed["special_event"]["applicable"])
        self.assertTrue(
            any(
                goal.get("goal_id") == "exam_review"
                for goal in changed["daily_goals"]
            )
        )
        self.assertLess(
            sum(changed["hourly_activity_baseline"]),
            sum(student["hourly_activity_baseline"]),
        )

    def test_project_deadline_adds_worker_goal(self) -> None:
        event = select_special_event("project_deadline", seed=42)
        worker = ensure_risk_profile(
            {"agent_id": "worker", "age": 40, "seed": 40}
        )

        changed = apply_special_event(worker, event)

        self.assertTrue(changed["special_event"]["applicable"])
        self.assertTrue(
            any(
                goal.get("goal_id") == "project_deadline"
                for goal in changed["daily_goals"]
            )
        )

    def test_sick_leave_affects_reproducible_subset(self) -> None:
        event = select_special_event("sick_leave", seed=42)
        profiles = [
            ensure_risk_profile(
                {
                    "agent_id": f"agent_{seed}",
                    "age": 40,
                    "seed": seed,
                }
            )
            for seed in range(100)
        ]
        first_results = [
            apply_special_event(profile, event)
            for profile in profiles
        ]
        repeated_results = [
            apply_special_event(profile, event)
            for profile in profiles
        ]
        affected = [
            profile
            for profile in first_results
            if profile["special_event"]["applicable"]
        ]

        self.assertEqual(first_results, repeated_results)
        self.assertGreater(len(affected), 0)
        self.assertLess(len(affected), len(profiles))
        self.assertTrue(
            all(
                any(
                    goal.get("goal_id") == "health_recovery"
                    for goal in profile["daily_goals"]
                )
                for profile in affected
            )
        )


if __name__ == "__main__":
    unittest.main()
