from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import random
from typing import Any


OFFICE_ACTIVITY_BASELINE = [
    0.05, 0.03, 0.02, 0.02, 0.02, 0.03,
    0.10, 0.15, 0.08, 0.05, 0.05, 0.08,
    0.20, 0.15, 0.08, 0.08, 0.10, 0.20,
    0.35, 0.50, 0.65, 0.55, 0.25, 0.10,
]

STUDENT_ACTIVITY_BASELINE = [
    0.04, 0.02, 0.02, 0.02, 0.02, 0.03,
    0.08, 0.12, 0.06, 0.04, 0.04, 0.06,
    0.12, 0.10, 0.06, 0.06, 0.10, 0.20,
    0.40, 0.58, 0.72, 0.62, 0.30, 0.12,
]

RETIRED_ACTIVITY_BASELINE = [
    0.04, 0.02, 0.02, 0.02, 0.03, 0.05,
    0.10, 0.12, 0.10, 0.10, 0.12, 0.15,
    0.18, 0.16, 0.14, 0.15, 0.18, 0.24,
    0.32, 0.38, 0.42, 0.38, 0.22, 0.10,
]


def _routine_for_age(age: int) -> str:
    if age <= 24:
        return "student"
    if age >= 61:
        return "retired"
    return "office_worker"


def _stable_seed(
    profile: dict[str, Any],
    salt: str,
) -> int:
    identity = str(
        profile.get("seed")
        or profile.get("agent_id")
        or profile.get("age")
        or "anonymous"
    )
    digest = sha256(
        f"{identity}:{salt}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _bounded_minute(value: int) -> int:
    return max(0, min(1439, value))


def _default_goals(
    profile: dict[str, Any],
    routine_type: str,
) -> list[dict[str, Any]]:
    rng = random.Random(
        _stable_seed(profile, "daily-goals-v2")
    )

    if routine_type == "student":
        sleep_start = _bounded_minute(
            1410 + rng.randint(-35, 25)
        )
        sleep_end = _bounded_minute(
            450 + rng.randint(-25, 35)
        )
    elif routine_type == "retired":
        sleep_start = _bounded_minute(
            1335 + rng.randint(-30, 30)
        )
        sleep_end = _bounded_minute(
            390 + rng.randint(-25, 25)
        )
    else:
        sleep_start = _bounded_minute(
            1380 + rng.randint(-35, 35)
        )
        sleep_end = _bounded_minute(
            420 + rng.randint(-25, 25)
        )

    sleep_goal = {
        "goal_id": "sleep",
        "name": "睡眠",
        "start_minute": sleep_start,
        "end_minute": sleep_end,
        "priority": 3,
        "tolerance_minutes": 15,
    }

    if routine_type == "student":
        study_start = _bounded_minute(
            510 + rng.randint(-30, 30)
        )
        study_end = _bounded_minute(
            1020 + rng.randint(-30, 30)
        )
        long_term_start = _bounded_minute(
            1140 + rng.randint(-40, 40)
        )
        return [
            {
                "goal_id": "study",
                "name": "上课或学习",
                "start_minute": study_start,
                "end_minute": study_end,
                "priority": 3,
                "tolerance_minutes": 10,
            },
            {
                "goal_id": "long_term_learning",
                "name": "长期目标：自主学习",
                "start_minute": long_term_start,
                "end_minute": _bounded_minute(
                    long_term_start + 100
                ),
                "priority": 3,
                "tolerance_minutes": 15,
            },
            sleep_goal,
        ]

    if routine_type == "retired":
        health_start = _bounded_minute(
            450 + rng.randint(-35, 35)
        )
        return [
            {
                "goal_id": "long_term_health",
                "name": "长期目标：健康锻炼",
                "start_minute": health_start,
                "end_minute": _bounded_minute(
                    health_start + 60
                ),
                "priority": 3,
                "tolerance_minutes": 15,
            },
            sleep_goal,
        ]

    work_start = _bounded_minute(
        540 + rng.randint(-35, 35)
    )
    work_end = _bounded_minute(
        work_start + 510 + rng.randint(-30, 30)
    )
    long_term_start = _bounded_minute(
        1170 + rng.randint(-45, 45)
    )

    return [
        {
            "goal_id": "work",
            "name": "工作",
            "start_minute": work_start,
            "end_minute": work_end,
            "priority": 3,
            "tolerance_minutes": 10,
        },
        {
            "goal_id": "long_term_development",
            "name": "长期目标：技能提升",
            "start_minute": long_term_start,
            "end_minute": _bounded_minute(
                long_term_start + 90
            ),
            "priority": 3,
            "tolerance_minutes": 15,
        },
        sleep_goal,
    ]


def sample_session_start_minute(
    profile: dict[str, Any],
    experiment_seed: int,
    day_number: int = 1,
) -> int:
    """按个人正常活动分布抽取可复现的会话开始时间。"""

    baseline = profile.get("hourly_activity_baseline")
    if not isinstance(baseline, list) or len(baseline) != 24:
        weights = [1.0] * 24
    else:
        weights = []
        for value in baseline:
            try:
                weight = float(value)
            except (TypeError, ValueError):
                weight = 0.0
            # 保留很小的非零概率，使异常时段仍可能被抽到。
            weights.append(max(0.01, weight))

    rng = random.Random(
        _stable_seed(
            profile,
            f"session-start:{experiment_seed}:{day_number}",
        )
    )
    hour = rng.choices(
        range(24),
        weights=weights,
        k=1,
    )[0]
    minute = rng.randrange(60)
    return int(hour * 60 + minute)


def ensure_risk_profile(
    profile: dict[str, Any],
) -> dict[str, Any]:
    """为旧画像补齐单会话风险判断所需的字段。"""

    enriched = deepcopy(profile)

    try:
        age = int(enriched.get("age", 35))
    except (TypeError, ValueError):
        age = 35

    routine_type = str(
        enriched.get("routine_type")
        or _routine_for_age(age)
    )
    enriched["routine_type"] = routine_type

    baseline = enriched.get("hourly_activity_baseline")
    used_default = False
    if not isinstance(baseline, list) or len(baseline) != 24:
        if routine_type == "student":
            baseline = STUDENT_ACTIVITY_BASELINE
        elif routine_type == "retired":
            baseline = RETIRED_ACTIVITY_BASELINE
        else:
            baseline = OFFICE_ACTIVITY_BASELINE
        enriched["hourly_activity_baseline"] = list(baseline)
        used_default = True

    goals = enriched.get("daily_goals")
    if not isinstance(goals, list) or not goals:
        enriched["daily_goals"] = _default_goals(
            enriched,
            routine_type,
        )
        used_default = True

    enriched["risk_profile_source"] = (
        enriched.get("risk_profile_source")
        or (
            "personalized_deterministic_default"
            if used_default
            else "profile"
        )
    )

    return enriched
