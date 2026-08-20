from __future__ import annotations

from math import sqrt
from typing import Any


# 活动异常度阈值。
# 实际活动值减去个人基线大于等于0.70时，
# 暂时认为本次活动明显偏离正常模式。
ACTIVITY_ANOMALY_THRESHOLD = 0.70


def evaluate_activity_anomaly(
    profile: dict[str, Any],
    simulation_minute: int,
    actual_activity: float = 1.0,
) -> dict[str, Any]:
    """
    判断当前活动是否偏离该 Agent 自己的正常活动模式。

    actual_activity=1.0 表示当前 Agent 正在使用系统。
    """

    # 把总分钟转换成当天小时。
    current_hour = (
        simulation_minute
        % 1440
        // 60
    )

    hourly_baseline = profile.get(
        "hourly_activity_baseline"
    )

    # 兼容还没有新字段的旧画像。
    if (
        not isinstance(hourly_baseline, list)
        or len(hourly_baseline) != 24
    ):
        return {
            "available": False,
            "current_hour": current_hour,
            "actual_activity": actual_activity,
            "baseline_activity": None,
            "activity_anomaly": None,
            "is_anomalous": False,
            "reason": (
                "该 Agent 没有合法的24小时活动基线"
            ),
        }

    try:
        baseline_activity = float(
            hourly_baseline[current_hour]
        )
    except (TypeError, ValueError):
        return {
            "available": False,
            "current_hour": current_hour,
            "actual_activity": actual_activity,
            "baseline_activity": None,
            "activity_anomaly": None,
            "is_anomalous": False,
            "reason": (
                "当前小时的活动基线不是有效数字"
            ),
        }

    activity_anomaly = (
        actual_activity
        - baseline_activity
    )

    is_anomalous = (
        activity_anomaly
        >= ACTIVITY_ANOMALY_THRESHOLD
    )

    if is_anomalous:
        reason = (
            f"当前小时为{current_hour}时，"
            f"实际活动值为{actual_activity:.2f}，"
            f"个人正常基线为{baseline_activity:.2f}，"
            f"偏离值为{activity_anomaly:.2f}"
        )
    else:
        reason = (
            f"当前活动没有明显偏离"
            f"{current_hour}时的个人正常基线"
        )

    return {
        "available": True,
        "current_hour": current_hour,
        "actual_activity": round(
            actual_activity,
            3,
        ),
        "baseline_activity": round(
            baseline_activity,
            3,
        ),
        "activity_anomaly": round(
            activity_anomaly,
            3,
        ),
        "is_anomalous": is_anomalous,
        "threshold": (
            ACTIVITY_ANOMALY_THRESHOLD
        ),
        "reason": reason,
    }

def goal_is_active(
    minute_of_day: int,
    start_minute: int,
    end_minute: int,
) -> bool:
    """判断当前时间是否位于目标时间段内。"""

    # 开始和结束时间相同，暂时视为无效时间段。
    if start_minute == end_minute:
        return False

    # 普通时间段，例如09:00到18:00。
    if start_minute < end_minute:
        return (
            start_minute
            <= minute_of_day
            < end_minute
        )

    # 跨越午夜的时间段，例如23:00到07:00。
    return (
        minute_of_day >= start_minute
        or minute_of_day < end_minute
    )


def minutes_since_goal_start(
    minute_of_day: int,
    start_minute: int,
    end_minute: int,
) -> int:
    """计算当前时间距离目标开始已经过去多少分钟。"""

    # 普通时间段。
    if start_minute < end_minute:
        return max(
            0,
            minute_of_day - start_minute,
        )

    # 跨午夜目标，而且当前时间仍在午夜之前。
    if minute_of_day >= start_minute:
        return (
            minute_of_day
            - start_minute
        )

    # 跨午夜目标，而且当前时间已经进入第二天。
    return (
        1440
        - start_minute
        + minute_of_day
    )


def evaluate_goal_conflict(
    profile: dict[str, Any],
    simulation_minute_after: int,
    action: str,
) -> dict[str, Any]:
    """
    判断继续使用是否影响当前高优先级目标。
    """

    minute_of_day = (
        simulation_minute_after
        % 1440
    )

    daily_goals = profile.get(
        "daily_goals"
    )

    if not isinstance(daily_goals, list):
        return {
            "available": False,
            "goal_opportunity": False,
            "active_goal": None,
            "continued_use": (
                action in {"click", "next"}
            ),
            "goal_conflict": False,
            "reason": "该Agent没有合法的daily_goals",
        }

    active_goals: list[dict[str, Any]] = []

    for goal in daily_goals:
        if not isinstance(goal, dict):
            continue

        try:
            start_minute = int(
                goal["start_minute"]
            )
            end_minute = int(
                goal["end_minute"]
            )
            priority = int(
                goal.get("priority", 1)
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if not (
            0 <= start_minute < 1440
            and 0 <= end_minute < 1440
        ):
            continue

        if goal_is_active(
            minute_of_day=minute_of_day,
            start_minute=start_minute,
            end_minute=end_minute,
        ):
            active_goal = dict(goal)
            active_goal["_start_minute"] = (
                start_minute
            )
            active_goal["_end_minute"] = (
                end_minute
            )
            active_goal["_priority"] = priority

            active_goals.append(active_goal)

    # 当前没有处于任何目标时间段。
    if not active_goals:
        return {
            "available": True,
            "goal_opportunity": False,
            "active_goal": None,
            "continued_use": (
                action in {"click", "next"}
            ),
            "goal_conflict": False,
            "reason": "当前不在任何生活目标时间段内",
        }

    # 如果多个目标同时出现，优先选择优先级最高的。
    active_goals.sort(
        key=lambda goal: int(
            goal["_priority"]
        ),
        reverse=True,
    )

    goal = active_goals[0]

    priority = int(
        goal["_priority"]
    )

    tolerance_minutes = max(
        0,
        int(
            goal.get(
                "tolerance_minutes",
                0,
            )
        ),
    )

    delay_minutes = minutes_since_goal_start(
        minute_of_day=minute_of_day,
        start_minute=int(
            goal["_start_minute"]
        ),
        end_minute=int(
            goal["_end_minute"]
        ),
    )

    continued_use = (
        action in {
            "click",
            "next",
        }
    )

    goal_conflict = bool(
        continued_use
        and priority >= 3
        and delay_minutes
        >= tolerance_minutes
    )

    goal_name = str(
        goal.get(
            "name",
            goal.get(
                "goal_id",
                "未知目标",
            ),
        )
    )

    if goal_conflict:
        reason = (
            f"当前处于{goal_name}时间，"
            f"目标优先级为{priority}，"
            f"已经推迟{delay_minutes}分钟，"
            f"但Agent仍然执行{action}"
        )

    elif not continued_use:
        reason = (
            f"当前处于{goal_name}时间，"
            "但Agent已经停止使用"
        )

    elif priority < 3:
        reason = (
            f"Agent继续使用，但当前目标"
            f"{goal_name}不是高优先级目标"
        )

    else:
        reason = (
            f"Agent仍在使用，但当前只推迟了"
            f"{delay_minutes}分钟，"
            f"没有超过允许的"
            f"{tolerance_minutes}分钟"
        )

    clean_goal = {
        key: value
        for key, value in goal.items()
        if not key.startswith("_")
    }

    return {
        "available": True,
        "goal_opportunity": True,
        "active_goal": clean_goal,
        "goal_name": goal_name,
        "goal_priority": priority,
        "tolerance_minutes": (
            tolerance_minutes
        ),
        "delay_minutes": delay_minutes,
        "continued_use": continued_use,
        "goal_conflict": goal_conflict,
        "reason": reason,
    }

def summarize_session(
    steps: list[dict[str, Any]],
    termination: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """汇总一个Agent在本次session中的风险证据。"""

    # 只使用成功执行的动作。
    successful_steps = [
        step
        for step in steps
        if step.get("success") is True
    ]
    termination_type = (
        str(termination.get("type"))
        if isinstance(termination, dict)
        else None
    )

    if not successful_steps:
        no_data_status = (
            "technical_failure_censored"
            if termination_type == "technical_failure_censored"
            else "invalid_session"
        )
        return {
            "status": no_data_status,
            "status_label": (
                "技术失败截断"
                if no_data_status == "technical_failure_censored"
                else "无有效数据"
            ),
            "session_problematic": False,
            "is_addicted": False,
            "judgement_scope": "single_session_operational_definition",
            "addiction_evidence_count": 0,
            "addiction_evidence": [],
            "external_interruption_count": 0,
            "termination": termination,
            "reason": "本次session没有成功执行的动作",
        }

    # 统计活动异常。
    activity_available_steps = [
        step
        for step in successful_steps
        if (
            isinstance(
                step.get("activity_evaluation"),
                dict,
            )
            and step[
                "activity_evaluation"
            ].get("available") is True
        )
    ]

    activity_anomaly_steps = [
        step
        for step in activity_available_steps
        if step[
            "activity_evaluation"
        ].get("is_anomalous") is True
    ]

    # 统计停止意图与停止失败。
    stop_intent_steps = [
        step
        for step in successful_steps
        if step.get(
            "intended_to_stop"
        ) is True
    ]

    stop_failure_steps = [
        step
        for step in successful_steps
        if step.get(
            "stop_failure"
        ) is True
    ]

    stop_intent_count = len(
        stop_intent_steps
    )

    stop_failure_count = len(
        stop_failure_steps
    )

    stop_failure_rate = (
        stop_failure_count
        / stop_intent_count
        if stop_intent_count > 0
        else 0.0
    )

    # 使用集合，避免同一个睡眠目标在多轮中被重复计算。
    goal_opportunities: set[
        tuple[int, str]
    ] = set()

    conflicted_goals: set[
        tuple[int, str]
    ] = set()

    goal_conflict_event_count = 0

    for step in successful_steps:
        goal_evaluation = step.get(
            "goal_evaluation"
        )

        if not isinstance(
            goal_evaluation,
            dict,
        ):
            continue

        if not goal_evaluation.get(
            "goal_opportunity",
            False,
        ):
            continue

        active_goal = goal_evaluation.get(
            "active_goal"
        )

        if not isinstance(active_goal, dict):
            continue

        goal_id = str(
            active_goal.get(
                "goal_id",
                active_goal.get(
                    "name",
                    "unknown_goal",
                ),
            )
        )

        simulation_minute_after = int(
            step.get(
                "simulation_minute_after",
                0,
            )
        )

        day_number = (
            simulation_minute_after
            // 1440
            + 1
        )

        opportunity_key = (
            day_number,
            goal_id,
        )

        goal_opportunities.add(
            opportunity_key
        )

        if goal_evaluation.get(
            "goal_conflict",
            False,
        ):
            conflicted_goals.add(
                opportunity_key
            )
            goal_conflict_event_count += 1

    goal_opportunity_count = len(
        goal_opportunities
    )

    goal_conflict_count = len(
        conflicted_goals
    )

    goal_conflict_rate = (
        goal_conflict_count
        / goal_opportunity_count
        if goal_opportunity_count > 0
        else 0.0
    )

    activity_anomaly_count = len(
        activity_anomaly_steps
    )

    # 三层证据必须在同一会话中形成完整链条。
    # 第一层是本次会话存在活动异常；后两层要求至少有一轮
    # 同时出现“想停止却继续”和“影响高优先级目标”。
    stop_failure_goal_conflict_steps = [
        step
        for step in successful_steps
        if (
            step.get("stop_failure") is True
            and step.get("goal_conflict") is True
        )
    ]

    addiction_evidence_steps = (
        stop_failure_goal_conflict_steps
        if activity_anomaly_count > 0
        else []
    )
    addiction_evidence_count = len(addiction_evidence_steps)

    external_interruption_steps = [
        step
        for step in successful_steps
        if step.get("external_interruption") is True
    ]
    external_interruption_count = len(
        external_interruption_steps
    )

    # 严格按照Word中的判断顺序。
    if not activity_available_steps:
        status = "insufficient_data"
        status_label = "数据不足"
        reason = "没有可用的个人活动基线"

    elif addiction_evidence_count > 0:
        status = "addicted"
        status_label = "已陷入沉迷（单会话判定）"
        reason = (
            "本次会话同时出现活动异常、想停止却继续，"
            "并且继续行为影响了高优先级生活目标"
        )

    elif (
        termination_type == "external_interruption"
        or external_interruption_count > 0
    ):
        status = "externally_censored"
        status_label = "外部事件截断"
        reason = (
            "会话被停电等外部事件强制中断，"
            "无法据此判断Agent是否会自主停止"
        )

    elif termination_type == "safety_limit_censored":
        status = "safety_limit_censored"
        status_label = "安全上限截断，结果未确定"
        reason = (
            "Agent达到最大交互轮数或最大仿真时长时"
            "仍未主动停止，不能将其当作正常使用"
        )

    elif termination_type == "technical_failure_censored":
        status = "technical_failure_censored"
        status_label = "技术失败截断"
        reason = (
            "会话因API或程序错误中断，"
            "不进入推荐系统风险概率分母"
        )

    elif activity_anomaly_count == 0:
        status = "normal_use"
        status_label = "正常使用"
        reason = "使用行为没有偏离个人正常模式"

    elif stop_failure_count == 0:
        status = "high_engagement"
        status_label = "高参与，不判定高风险"
        reason = (
            "存在活动异常，但没有出现想停止却继续；"
            "Agent可能尚未产生停止意图，或产生后实际停止"
        )

    elif addiction_evidence_count == 0:
        status = "observe"
        status_label = "观察状态"
        reason = (
            "出现停止失败，但没有形成"
            "‘想停止却继续且影响高优先级目标’的完整证据链"
        )

    is_addicted = status == "addicted"
    session_problematic = is_addicted

    addiction_evidence = []
    for step in addiction_evidence_steps:
        goal_evaluation = step.get("goal_evaluation")
        if not isinstance(goal_evaluation, dict):
            goal_evaluation = {}
        addiction_evidence.append(
            {
                "simulation_step": step.get("simulation_step"),
                "simulation_time": step.get("simulation_time_after"),
                "action": step.get("action"),
                "intention_reason": step.get("intention_reason"),
                "goal_name": goal_evaluation.get("goal_name"),
                "goal_conflict_reason": goal_evaluation.get("reason"),
            }
        )

    return {
        "status": status,
        "status_label": status_label,
        "session_problematic": (
            session_problematic
        ),
        "is_addicted": is_addicted,
        "judgement_scope": "single_session_operational_definition",
        "session_start_minute": successful_steps[0].get(
            "simulation_minute_before"
        ),
        "session_start_time": successful_steps[0].get(
            "simulation_time_before"
        ),
        "successful_step_count": len(
            successful_steps
        ),
        "activity_available_count": len(
            activity_available_steps
        ),
        "activity_anomaly_count": (
            activity_anomaly_count
        ),
        "stop_intent_count": (
            stop_intent_count
        ),
        "stop_failure_count": (
            stop_failure_count
        ),
        "stop_failure_rate": round(
            stop_failure_rate,
            3,
        ),
        "goal_opportunity_count": (
            goal_opportunity_count
        ),
        "goal_conflict_count": (
            goal_conflict_count
        ),
        "goal_conflict_event_count": (
            goal_conflict_event_count
        ),
        "goal_conflict_rate": round(
            goal_conflict_rate,
            3,
        ),
        "addiction_evidence_count": addiction_evidence_count,
        "addiction_evidence": addiction_evidence,
        "external_interruption_count": (
            external_interruption_count
        ),
        "termination": termination,
        "reason": reason,
    }


def summarize_recommender_addiction_risk(
    session_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """汇总当前推荐系统下的单会话沉迷风险概率。"""

    evaluable_statuses = {
        "normal_use",
        "high_engagement",
        "observe",
        "addicted",
    }
    evaluated_agent_ids = [
        agent_id
        for agent_id, summary in session_summaries.items()
        if summary.get("status") in evaluable_statuses
    ]
    addicted_agent_ids = [
        agent_id
        for agent_id in evaluated_agent_ids
        if session_summaries[agent_id].get("is_addicted") is True
    ]
    safety_censored_agent_ids = [
        agent_id
        for agent_id, summary in session_summaries.items()
        if summary.get("status") == "safety_limit_censored"
    ]
    external_censored_agent_ids = [
        agent_id
        for agent_id, summary in session_summaries.items()
        if summary.get("status") == "externally_censored"
    ]
    technical_censored_agent_ids = [
        agent_id
        for agent_id, summary in session_summaries.items()
        if summary.get("status") == "technical_failure_censored"
    ]

    evaluated_count = len(evaluated_agent_ids)
    addicted_count = len(addicted_agent_ids)
    safety_censored_count = len(safety_censored_agent_ids)
    risk_bound_denominator = (
        evaluated_count + safety_censored_count
    )
    risk_lower_bound = (
        addicted_count / risk_bound_denominator
        if risk_bound_denominator > 0
        else None
    )
    risk_upper_bound = (
        (addicted_count + safety_censored_count)
        / risk_bound_denominator
        if risk_bound_denominator > 0
        else None
    )

    evidence_by_goal: dict[str, int] = {}
    for agent_id in addicted_agent_ids:
        evidence_items = session_summaries[agent_id].get(
            "addiction_evidence",
            [],
        )
        if not isinstance(evidence_items, list):
            continue
        for evidence in evidence_items:
            if not isinstance(evidence, dict):
                continue
            goal_name = str(
                evidence.get("goal_name")
                or "未知高优先级目标"
            )
            evidence_by_goal[goal_name] = (
                evidence_by_goal.get(goal_name, 0) + 1
            )

    if evaluated_count == 0:
        return {
            "status": "insufficient_data",
            "evaluated_agent_count": 0,
            "addicted_agent_count": 0,
            "addicted_agent_ids": [],
            "estimated_probability": None,
            "estimated_probability_percent": None,
            "confidence_interval_95": None,
            "safety_censored_agent_count": safety_censored_count,
            "safety_censored_agent_ids": safety_censored_agent_ids,
            "external_censored_agent_count": len(
                external_censored_agent_ids
            ),
            "technical_censored_agent_count": len(
                technical_censored_agent_ids
            ),
            "risk_probability_bounds": {
                "lower": (
                    round(risk_lower_bound, 6)
                    if risk_lower_bound is not None
                    else None
                ),
                "upper": (
                    round(risk_upper_bound, 6)
                    if risk_upper_bound is not None
                    else None
                ),
                "lower_percent": (
                    round(risk_lower_bound * 100.0, 3)
                    if risk_lower_bound is not None
                    else None
                ),
                "upper_percent": (
                    round(risk_upper_bound * 100.0, 3)
                    if risk_upper_bound is not None
                    else None
                ),
            },
            "addiction_evidence_by_goal": {},
            "observed_addiction_risk": False,
            "reason": "没有可用于系统风险估计的Agent会话",
        }

    probability = addicted_count / evaluated_count

    # 使用Wilson区间，避免小样本下普通正态区间越过0或1。
    z = 1.96
    z_squared = z * z
    denominator = 1.0 + z_squared / evaluated_count
    center = (
        probability
        + z_squared / (2.0 * evaluated_count)
    ) / denominator
    margin = (
        z
        * sqrt(
            probability * (1.0 - probability) / evaluated_count
            + z_squared / (4.0 * evaluated_count * evaluated_count)
        )
        / denominator
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return {
        "status": "ok",
        "judgement_scope": "single_session_operational_definition",
        "evaluated_agent_count": evaluated_count,
        "addicted_agent_count": addicted_count,
        "addicted_agent_ids": addicted_agent_ids,
        "estimated_probability": round(probability, 6),
        "estimated_probability_percent": round(probability * 100.0, 3),
        "confidence_interval_95": {
            "lower": round(lower, 6),
            "upper": round(upper, 6),
            "lower_percent": round(lower * 100.0, 3),
            "upper_percent": round(upper * 100.0, 3),
            "method": "Wilson score interval",
        },
        "safety_censored_agent_count": safety_censored_count,
        "safety_censored_agent_ids": safety_censored_agent_ids,
        "external_censored_agent_count": len(
            external_censored_agent_ids
        ),
        "external_censored_agent_ids": external_censored_agent_ids,
        "technical_censored_agent_count": len(
            technical_censored_agent_ids
        ),
        "technical_censored_agent_ids": technical_censored_agent_ids,
        "risk_probability_bounds": {
            "lower": round(risk_lower_bound, 6),
            "upper": round(risk_upper_bound, 6),
            "lower_percent": round(risk_lower_bound * 100.0, 3),
            "upper_percent": round(risk_upper_bound * 100.0, 3),
            "interpretation": (
                "下界把未确定安全截断视为未沉迷；"
                "上界把未确定安全截断视为可能沉迷"
            ),
        },
        "addiction_evidence_by_goal": evidence_by_goal,
        "observed_addiction_risk": addicted_count > 0,
        "interpretation": (
            f"在当前推荐系统下，{evaluated_count}个可评估Agent中"
            f"有{addicted_count}个满足单会话三层沉迷判定，"
            f"样本风险概率为{probability * 100.0:.2f}%"
        ),
        "causal_claim": False,
        "causal_limitation": (
            "该概率描述当前推荐系统下的风险发生比例；"
            "没有对照推荐系统时，不能单独证明风险完全由推荐系统造成"
        ),
    }
