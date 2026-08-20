from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_WEIGHTS = {
    "click_rate": 0.30,
    "session_length": 0.40,
    "action_transitions": 0.30,
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"找不到文件：{path}"
        ) from None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON格式错误：{path}：{exc}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            f"JSON根节点必须是对象：{path}"
        )
    return value


def _load_histogram(
    metric: dict[str, Any],
) -> dict[int, int]:
    raw_histogram = metric.get("histogram")
    if not isinstance(raw_histogram, dict):
        raise ValueError("基线缺少histogram")

    histogram: dict[int, int] = {}
    for raw_key, raw_count in raw_histogram.items():
        key = int(raw_key)
        count = int(raw_count)
        if count < 0:
            raise ValueError("histogram计数不能为负数")
        histogram[key] = count

    if sum(histogram.values()) <= 0:
        raise ValueError("histogram没有有效样本")
    return histogram


def _percentile_rank(
    value: float,
    histogram: dict[int, int],
) -> float:
    """使用mid-rank计算数值在基线中的百分位。"""

    total = sum(histogram.values())
    below = sum(
        count
        for key, count in histogram.items()
        if key < value
    )
    equal = sum(
        count
        for key, count in histogram.items()
        if key == value
    )
    percentile = (
        below + 0.5 * equal
    ) / total * 100.0
    return round(percentile, 3)


def _centrality_from_percentile(
    percentile: float,
) -> float:
    """50百分位得1分，两端逐渐降到0分。"""

    return max(
        0.0,
        1.0 - abs(percentile - 50.0) / 50.0,
    )


def _normalized_distribution(
    counts: Counter[str],
    keys: set[str],
) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {key: 0.0 for key in keys}
    return {
        key: counts.get(key, 0) / total
        for key in keys
    }


def _transition_similarity(
    observed_counts: Counter[str],
    baseline_probabilities: dict[str, float],
) -> float:
    """以Jensen-Shannon距离计算0到1的分布相似度。"""

    keys = (
        set(observed_counts)
        | set(baseline_probabilities)
    )
    if not keys or sum(observed_counts.values()) <= 0:
        return 0.0

    observed = _normalized_distribution(
        observed_counts,
        keys,
    )
    baseline_total = sum(
        max(0.0, float(value))
        for value in baseline_probabilities.values()
    )
    if baseline_total <= 0:
        raise ValueError(
            "基线action_transitions概率无效"
        )
    baseline = {
        key: max(
            0.0,
            float(baseline_probabilities.get(key, 0.0)),
        ) / baseline_total
        for key in keys
    }

    midpoint = {
        key: (observed[key] + baseline[key]) / 2.0
        for key in keys
    }

    def kl_divergence(
        left: dict[str, float],
        right: dict[str, float],
    ) -> float:
        return sum(
            left[key]
            * math.log(left[key] / right[key])
            for key in keys
            if left[key] > 0 and right[key] > 0
        )

    js_divergence = 0.5 * (
        kl_divergence(observed, midpoint)
        + kl_divergence(baseline, midpoint)
    )
    js_distance = math.sqrt(
        max(0.0, js_divergence) / math.log(2.0)
    )
    return round(
        max(0.0, min(1.0, 1.0 - js_distance)),
        6,
    )


def _read_agent_actions(
    events_path: Path,
) -> tuple[dict[str, list[str]], int, int]:
    agent_actions: dict[str, list[str]] = defaultdict(list)
    event_count = 0
    ignored_event_count = 0

    try:
        event_file = events_path.open(
            "r",
            encoding="utf-8",
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"找不到Agent事件文件：{events_path}"
        ) from None

    with event_file:
        for line_number, raw_line in enumerate(
            event_file,
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"events第{line_number}行JSON错误：{exc}"
                ) from exc

            event_count += 1
            if event.get("success") is not True:
                ignored_event_count += 1
                continue

            agent_id = str(
                event.get("agent_id", "")
            )
            action = str(
                event.get("action", "")
            )
            if not agent_id or action not in {
                "click",
                "next",
                "stop",
            }:
                ignored_event_count += 1
                continue

            agent_actions[agent_id].append(action)

    return (
        dict(agent_actions),
        event_count,
        ignored_event_count,
    )


def _evaluate_agent(
    agent_id: str,
    actions: list[str],
    click_histogram: dict[int, int],
    click_rate_scale: int,
    length_histogram: dict[int, int],
    baseline_transitions: dict[str, float],
    weights: dict[str, float],
) -> dict[str, Any]:
    interactions = [
        action
        for action in actions
        if action in {"click", "next"}
    ]
    if not interactions:
        return {
            "agent_id": agent_id,
            "status": "insufficient_data",
            "reason": "没有click或next动作",
        }

    click_count = interactions.count("click")
    next_count = interactions.count("next")
    click_rate = click_count / len(interactions)
    session_length = len(interactions)

    comparison_actions = list(actions)
    stop_inferred = False
    if not comparison_actions or comparison_actions[-1] != "stop":
        comparison_actions.append("stop")
        stop_inferred = True

    transition_counts: Counter[str] = Counter(
        f"{left}->{right}"
        for left, right in zip(
            comparison_actions,
            comparison_actions[1:],
        )
    )

    click_rate_percentile = _percentile_rank(
        round(click_rate * click_rate_scale),
        click_histogram,
    )
    session_length_percentile = _percentile_rank(
        session_length,
        length_histogram,
    )
    click_rate_similarity = (
        _centrality_from_percentile(
            click_rate_percentile
        )
    )
    session_length_similarity = (
        _centrality_from_percentile(
            session_length_percentile
        )
    )
    transition_similarity = _transition_similarity(
        transition_counts,
        baseline_transitions,
    )

    score = 100.0 * (
        weights["click_rate"]
        * click_rate_similarity
        + weights["session_length"]
        * session_length_similarity
        + weights["action_transitions"]
        * transition_similarity
    )

    return {
        "agent_id": agent_id,
        "status": "ok",
        "interaction_count": session_length,
        "click_count": click_count,
        "next_count": next_count,
        "stop_observed": "stop" in actions,
        "stop_inferred_for_comparison": stop_inferred,
        "click_rate": round(click_rate, 6),
        "click_rate_percentile": click_rate_percentile,
        "click_rate_similarity": round(
            click_rate_similarity,
            6,
        ),
        "session_length": session_length,
        "session_length_percentile": (
            session_length_percentile
        ),
        "session_length_similarity": round(
            session_length_similarity,
            6,
        ),
        "action_transition_counts": dict(
            sorted(transition_counts.items())
        ),
        "action_transition_similarity": (
            transition_similarity
        ),
        "behavioral_realism_score": round(
            score,
            3,
        ),
    }


def evaluate_events_file(
    events_path: Path,
    baseline_path: Path,
    output_path: Path | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """比较Agent事件与KuaiSAR基线并输出离线报告。"""

    baseline = _load_json(baseline_path)
    if baseline.get("schema_version") != 1:
        raise ValueError(
            "只支持schema_version=1的KuaiSAR基线"
        )

    metrics = baseline.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("KuaiSAR基线缺少metrics")

    click_metric = metrics.get("click_rate")
    length_metric = metrics.get(
        "session_length_actions"
    )
    transition_metric = metrics.get(
        "action_transitions"
    )
    if not all(
        isinstance(metric, dict)
        for metric in (
            click_metric,
            length_metric,
            transition_metric,
        )
    ):
        raise ValueError("KuaiSAR基线指标不完整")

    click_histogram = _load_histogram(
        click_metric
    )
    length_histogram = _load_histogram(
        length_metric
    )
    click_rate_scale = int(
        click_metric.get("histogram_scale", 1000)
    )
    baseline_transitions = transition_metric.get(
        "probabilities"
    )
    if not isinstance(baseline_transitions, dict):
        raise ValueError(
            "KuaiSAR基线缺少动作转移概率"
        )

    score_weights = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        score_weights.update(weights)
    if any(
        score_weights.get(key, 0.0) < 0
        for key in DEFAULT_WEIGHTS
    ):
        raise ValueError("真实性评分权重不能为负数")
    weight_total = sum(
        score_weights[key]
        for key in DEFAULT_WEIGHTS
    )
    if weight_total <= 0:
        raise ValueError("真实性评分权重总和必须大于0")
    score_weights = {
        key: score_weights[key] / weight_total
        for key in DEFAULT_WEIGHTS
    }

    (
        agent_actions,
        event_count,
        ignored_event_count,
    ) = _read_agent_actions(events_path)

    agent_reports = [
        _evaluate_agent(
            agent_id=agent_id,
            actions=actions,
            click_histogram=click_histogram,
            click_rate_scale=click_rate_scale,
            length_histogram=length_histogram,
            baseline_transitions={
                str(key): float(value)
                for key, value
                in baseline_transitions.items()
            },
            weights=score_weights,
        )
        for agent_id, actions in sorted(
            agent_actions.items()
        )
    ]

    valid_reports = [
        report
        for report in agent_reports
        if report.get("status") == "ok"
    ]
    aggregate: dict[str, Any]
    if valid_reports:
        aggregate = {
            "valid_agent_count": len(valid_reports),
            "mean_click_rate": round(
                mean(
                    report["click_rate"]
                    for report in valid_reports
                ),
                6,
            ),
            "mean_click_rate_percentile": round(
                mean(
                    report["click_rate_percentile"]
                    for report in valid_reports
                ),
                3,
            ),
            "mean_session_length": round(
                mean(
                    report["session_length"]
                    for report in valid_reports
                ),
                3,
            ),
            "mean_session_length_percentile": round(
                mean(
                    report[
                        "session_length_percentile"
                    ]
                    for report in valid_reports
                ),
                3,
            ),
            "mean_action_transition_similarity": round(
                mean(
                    report[
                        "action_transition_similarity"
                    ]
                    for report in valid_reports
                ),
                6,
            ),
            "mean_behavioral_realism_score": round(
                mean(
                    report[
                        "behavioral_realism_score"
                    ]
                    for report in valid_reports
                ),
                3,
            ),
        }
    else:
        aggregate = {
            "valid_agent_count": 0,
            "reason": "没有可评估的Agent行为",
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "offline_behavioral_realism",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "events_file": str(events_path.resolve()),
        "baseline_file": str(baseline_path.resolve()),
        "baseline_source": baseline.get("source", {}),
        "event_count": event_count,
        "ignored_event_count": ignored_event_count,
        "score_weights": {
            key: round(value, 6)
            for key, value in score_weights.items()
        },
        "aggregate": aggregate,
        "agents": agent_reports,
        "interpretation": (
            "该分数只衡量Agent行为统计与KuaiSAR"
            "短视频日志的相似程度，不表示心理真实性，"
            "也不参与沉迷风险判断"
        ),
        "limitations": baseline.get(
            "limitations",
            [],
        ),
    }

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "将Agent events.jsonl与KuaiSAR基线离线比较"
        )
    )
    parser.add_argument(
        "--events",
        type=Path,
        required=True,
        help="Agent实验生成的events.jsonl",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="build_kuaisar_baseline.py生成的基线JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("realism_report.json"),
        help="真实性评估报告输出路径",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    report = evaluate_events_file(
        events_path=args.events,
        baseline_path=args.baseline,
        output_path=args.output,
    )
    print(
        "真实性评估完成："
        f"{args.output.resolve()}"
    )
    aggregate = report["aggregate"]
    if aggregate.get("valid_agent_count", 0) > 0:
        print(
            "平均行为真实性得分："
            f"{aggregate['mean_behavioral_realism_score']}"
        )


if __name__ == "__main__":
    main()
