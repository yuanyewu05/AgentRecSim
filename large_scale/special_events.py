from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import random
from typing import Any


SPECIAL_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "summer_vacation": {
        "event_id": "summer_vacation",
        "event_name": "学生暑假",
        "description": "学生当天处于暑假，不需要上课，作息和使用分布放宽",
        "applicable_routines": ["student"],
        "event_type": "schedule_override",
    },
    "holiday": {
        "event_id": "holiday",
        "event_name": "节假日",
        "description": "学生和上班族当天放假，不执行日常上课或工作目标",
        "applicable_routines": ["student", "office_worker"],
        "event_type": "schedule_override",
    },
    "power_outage": {
        "event_id": "power_outage",
        "event_name": "停电",
        "description": "当天20:00至21:00服务不可用，正在使用的会话会被外部中断",
        "applicable_routines": ["student", "office_worker", "retired"],
        "event_type": "external_interruption",
        "unavailable_start_minute": 1200,
        "unavailable_end_minute": 1260,
    },
    "exam_week": {
        "event_id": "exam_week",
        "event_name": "考试周",
        "description": "学生处于考试周，增加高优先级复习目标并减少娱乐时间",
        "applicable_routines": ["student"],
        "event_type": "schedule_override",
    },
    "project_deadline": {
        "event_id": "project_deadline",
        "event_name": "项目截止日",
        "description": "上班族面临项目截止，增加高优先级加班目标并减少娱乐时间",
        "applicable_routines": ["office_worker"],
        "event_type": "schedule_override",
    },
    "sick_leave": {
        "event_id": "sick_leave",
        "event_name": "生病休息",
        "description": "部分用户当天生病，暂停工作或学习并改变睡眠和活动模式",
        "applicable_routines": ["student", "office_worker", "retired"],
        "applicability_probability": 0.35,
        "event_type": "schedule_override",
    },
}

SPECIAL_EVENT_CHOICES = (
    "none",
    "random",
    *SPECIAL_EVENT_CATALOG.keys(),
)


def select_special_event(
    requested_event: str,
    seed: int,
) -> dict[str, Any] | None:
    """选择本次运行使用的唯一特殊事件。"""

    if requested_event == "none":
        return None

    if requested_event == "random":
        rng = random.Random(seed)
        event_id = rng.choice(
            list(SPECIAL_EVENT_CATALOG)
        )
        return deepcopy(SPECIAL_EVENT_CATALOG[event_id])

    if requested_event not in SPECIAL_EVENT_CATALOG:
        raise ValueError(
            f"未知特殊事件：{requested_event}"
        )

    return deepcopy(
        SPECIAL_EVENT_CATALOG[requested_event]
    )


def _shift_baseline(
    baseline: list[Any],
    multipliers: dict[int, float],
) -> list[float]:
    shifted: list[float] = []
    for hour, value in enumerate(baseline):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = 0.01
        multiplier = multipliers.get(hour, 1.0)
        shifted.append(
            round(
                max(0.01, min(0.95, numeric_value * multiplier)),
                3,
            )
        )
    return shifted


def _shift_goal_minutes(
    goal: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    shifted = deepcopy(goal)
    shifted["start_minute"] = (
        int(shifted["start_minute"]) + offset
    ) % 1440
    shifted["end_minute"] = (
        int(shifted["end_minute"]) + offset
    ) % 1440
    return shifted


def _event_applies_to_profile(
    profile: dict[str, Any],
    event: dict[str, Any],
) -> tuple[bool, float | None]:
    routine_type = str(
        profile.get("routine_type", "unknown")
    )
    if routine_type not in event.get("applicable_routines", []):
        return False, None

    probability = event.get("applicability_probability")
    if probability is None:
        return True, None

    identity = str(
        profile.get("seed")
        or profile.get("agent_id")
        or "anonymous"
    )
    digest = sha256(
        f"{identity}:{event['event_id']}:applicability".encode("utf-8")
    ).digest()
    roll = int.from_bytes(digest[:8], "big") / float(2**64)
    return roll < float(probability), round(roll, 6)


def apply_special_event(
    profile: dict[str, Any],
    event: dict[str, Any] | None,
) -> dict[str, Any]:
    """把本次运行的特殊事件应用到一个Agent当天的画像。"""

    enriched = deepcopy(profile)

    if event is None:
        enriched["special_event"] = {
            "event_id": "none",
            "event_name": "无特殊事件",
            "applicable": False,
            "effects": [],
        }
        return enriched

    routine_type = str(
        enriched.get("routine_type", "unknown")
    )
    applicable, applicability_roll = _event_applies_to_profile(
        enriched,
        event,
    )
    context = deepcopy(event)
    context["applicable"] = applicable
    if event.get("applicability_probability") is not None:
        context["applicability_probability"] = float(
            event["applicability_probability"]
        )
        context["applicability_roll"] = applicability_roll
    context["effects"] = []

    if not applicable:
        if (
            routine_type in event.get("applicable_routines", [])
            and event.get("applicability_probability") is not None
        ):
            context["reason"] = "该用户未被本次部分用户事件抽中"
        else:
            context["reason"] = (
                f"事件不适用于{routine_type}画像"
            )
        enriched["special_event"] = context
        return enriched

    goals = enriched.get("daily_goals", [])
    if not isinstance(goals, list):
        goals = []

    baseline = enriched.get("hourly_activity_baseline", [])
    if not isinstance(baseline, list) or len(baseline) != 24:
        baseline = [0.1] * 24

    event_id = str(event["event_id"])

    if event_id == "summer_vacation":
        enriched["daily_goals"] = [
            _shift_goal_minutes(goal, 60)
            if (
                isinstance(goal, dict)
                and goal.get("goal_id") == "sleep"
            )
            else deepcopy(goal)
            for goal in goals
            if not (
                isinstance(goal, dict)
                and goal.get("goal_id") == "study"
            )
        ]
        enriched["hourly_activity_baseline"] = _shift_baseline(
            baseline,
            {
                **{hour: 1.8 for hour in range(9, 18)},
                **{hour: 1.25 for hour in range(18, 24)},
            },
        )
        context["effects"] = [
            "暂停上课目标",
            "白天和晚间使用概率提高",
            "睡眠时间推迟60分钟",
        ]

    elif event_id == "holiday":
        enriched["daily_goals"] = [
            _shift_goal_minutes(goal, 30)
            if (
                isinstance(goal, dict)
                and goal.get("goal_id") == "sleep"
            )
            else deepcopy(goal)
            for goal in goals
            if not (
                isinstance(goal, dict)
                and goal.get("goal_id") in {"work", "study"}
            )
        ]
        enriched["hourly_activity_baseline"] = _shift_baseline(
            baseline,
            {
                **{hour: 1.5 for hour in range(9, 18)},
                **{hour: 1.15 for hour in range(18, 24)},
            },
        )
        context["effects"] = [
            "暂停工作或上课目标",
            "白天使用概率提高",
            "睡眠时间推迟30分钟",
        ]

    elif event_id == "power_outage":
        context["effects"] = [
            "20:00至21:00服务不可用",
            "会话可能被外部强制中断",
        ]

    elif event_id == "exam_week":
        review_start = 1110
        enriched["daily_goals"] = [
            deepcopy(goal)
            for goal in goals
            if not (
                isinstance(goal, dict)
                and str(goal.get("goal_id", "")).startswith("long_term")
            )
        ] + [
            {
                "goal_id": "exam_review",
                "name": "考试周高优先级复习",
                "start_minute": review_start,
                "end_minute": 1350,
                "priority": 3,
                "tolerance_minutes": 5,
            }
        ]
        enriched["hourly_activity_baseline"] = _shift_baseline(
            baseline,
            {hour: 0.55 for hour in range(8, 24)},
        )
        context["effects"] = [
            "增加18:30至22:30高优先级复习目标",
            "暂停普通长期学习安排",
            "减少娱乐使用概率",
        ]

    elif event_id == "project_deadline":
        work_goals = [
            goal
            for goal in goals
            if isinstance(goal, dict)
            and goal.get("goal_id") == "work"
        ]
        normal_work_end = max(
            [
                int(goal.get("end_minute", 1080))
                for goal in work_goals
            ]
            or [1080]
        )
        deadline_start = max(1080, normal_work_end)
        enriched["daily_goals"] = [
            deepcopy(goal)
            for goal in goals
            if not (
                isinstance(goal, dict)
                and str(goal.get("goal_id", "")).startswith("long_term")
            )
        ] + [
            {
                "goal_id": "project_deadline",
                "name": "项目截止日高优先级工作",
                "start_minute": deadline_start,
                "end_minute": min(1439, deadline_start + 240),
                "priority": 3,
                "tolerance_minutes": 5,
            }
        ]
        enriched["hourly_activity_baseline"] = _shift_baseline(
            baseline,
            {hour: 0.50 for hour in range(9, 24)},
        )
        context["effects"] = [
            "增加晚间高优先级项目工作目标",
            "暂停普通技能提升安排",
            "减少娱乐使用概率",
        ]

    elif event_id == "sick_leave":
        changed_goals: list[dict[str, Any]] = []
        for goal in goals:
            if not isinstance(goal, dict):
                continue
            goal_id = str(goal.get("goal_id", ""))
            if goal_id in {"work", "study"} or goal_id.startswith(
                "long_term"
            ):
                continue
            if goal_id == "sleep":
                sleep_goal = deepcopy(goal)
                sleep_goal["start_minute"] = (
                    int(sleep_goal["start_minute"]) - 60
                ) % 1440
                sleep_goal["end_minute"] = (
                    int(sleep_goal["end_minute"]) + 90
                ) % 1440
                changed_goals.append(sleep_goal)
            else:
                changed_goals.append(deepcopy(goal))
        changed_goals.append(
            {
                "goal_id": "health_recovery",
                "name": "生病恢复与休息",
                "start_minute": 540,
                "end_minute": 1260,
                "priority": 3,
                "tolerance_minutes": 30,
            }
        )
        enriched["daily_goals"] = changed_goals
        enriched["hourly_activity_baseline"] = _shift_baseline(
            baseline,
            {hour: 0.65 for hour in range(24)},
        )
        context["effects"] = [
            "暂停工作、上课和普通长期目标",
            "增加高优先级恢复休息目标",
            "睡眠提前60分钟并延长90分钟",
            "整体娱乐使用概率降低",
        ]

    enriched["special_event"] = context
    return enriched


def minute_is_in_window(
    minute_of_day: int,
    start_minute: int,
    end_minute: int,
) -> bool:
    if start_minute == end_minute:
        return False
    if start_minute < end_minute:
        return start_minute <= minute_of_day < end_minute
    return minute_of_day >= start_minute or minute_of_day < end_minute


def active_external_interruption(
    profile: dict[str, Any],
    simulation_minute: int,
) -> dict[str, Any] | None:
    """返回当前时刻生效的外部强制中断事件。"""

    event = profile.get("special_event")
    if not isinstance(event, dict):
        return None
    if not event.get("applicable"):
        return None
    if event.get("event_type") != "external_interruption":
        return None

    try:
        start_minute = int(event["unavailable_start_minute"])
        end_minute = int(event["unavailable_end_minute"])
    except (KeyError, TypeError, ValueError):
        return None

    if minute_is_in_window(
        simulation_minute % 1440,
        start_minute,
        end_minute,
    ):
        return deepcopy(event)
    return None
