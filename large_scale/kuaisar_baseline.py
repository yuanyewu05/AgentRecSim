from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLICK_RATE_SCALE = 1000


@dataclass
class _OpenSession:
    last_timestamp_ms: int
    interaction_count: int = 0
    click_count: int = 0
    previous_action: str | None = None


def _parse_timestamp_ms(raw_value: str) -> int:
    try:
        return int(float(raw_value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"无效timestamp：{raw_value!r}"
        ) from exc


def _parse_click(raw_value: str) -> int:
    try:
        value = int(float(raw_value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"无效click：{raw_value!r}"
        ) from exc

    if value not in {0, 1}:
        raise ValueError(
            f"click必须是0或1，实际为{value}"
        )

    return value


def _finalize_session(
    session: _OpenSession,
    length_histogram: Counter[int],
    click_rate_histogram: Counter[int],
    transition_counts: Counter[str],
) -> None:
    if session.interaction_count <= 0:
        return

    length_histogram[
        session.interaction_count
    ] += 1

    click_rate = (
        session.click_count
        / session.interaction_count
    )
    click_rate_bin = round(
        click_rate * CLICK_RATE_SCALE
    )
    click_rate_histogram[
        click_rate_bin
    ] += 1

    if session.previous_action is not None:
        transition_counts[
            f"{session.previous_action}->stop"
        ] += 1


def _sorted_histogram(
    histogram: Counter[int],
) -> dict[str, int]:
    return {
        str(key): int(histogram[key])
        for key in sorted(histogram)
    }


def _transition_probabilities(
    counts: Counter[str],
) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}

    return {
        key: round(value / total, 8)
        for key, value in sorted(counts.items())
    }


def build_kuaisar_baseline(
    rec_inter_path: Path,
    output_path: Path,
    session_gap_minutes: int = 30,
    delimiter: str | None = None,
) -> dict[str, Any]:
    """从KuaiSAR rec_inter文件生成离线行为基线。"""

    if session_gap_minutes <= 0:
        raise ValueError(
            "session_gap_minutes必须大于0"
        )

    if not rec_inter_path.is_file():
        raise FileNotFoundError(
            f"找不到KuaiSAR推荐日志：{rec_inter_path}"
        )

    if delimiter is None:
        delimiter = (
            "\t"
            if rec_inter_path.suffix.lower() == ".tsv"
            else ","
        )

    gap_ms = session_gap_minutes * 60 * 1000
    length_histogram: Counter[int] = Counter()
    click_rate_histogram: Counter[int] = Counter()
    transition_counts: Counter[str] = Counter()
    row_count = 0
    user_count = 0

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp_file = tempfile.NamedTemporaryFile(
        prefix="kuaisar_sort_",
        suffix=".sqlite3",
        dir=output_path.parent,
        delete=False,
    )
    temp_database_path = Path(temp_file.name)
    temp_file.close()
    connection: sqlite3.Connection | None = None

    try:
        print(
            "第1阶段：读取CSV并写入临时排序数据库"
        )
        connection = sqlite3.connect(
            temp_database_path
        )
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE events ("
            "user_id TEXT NOT NULL, "
            "timestamp_ms INTEGER NOT NULL, "
            "click INTEGER NOT NULL, "
            "source_order INTEGER NOT NULL)"
        )

        insert_batch: list[
            tuple[str, int, int, int]
        ] = []
        with rec_inter_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as input_file:
            reader = csv.DictReader(
                input_file,
                delimiter=delimiter,
            )
            required_fields = {
                "user_id",
                "timestamp",
                "click",
            }
            available_fields = set(
                reader.fieldnames or []
            )
            missing_fields = (
                required_fields - available_fields
            )
            if missing_fields:
                raise ValueError(
                    "rec_inter缺少字段："
                    + ", ".join(
                        sorted(missing_fields)
                    )
                )

            connection.execute("BEGIN")
            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                user_id = str(row["user_id"])
                try:
                    timestamp_ms = (
                        _parse_timestamp_ms(
                            row["timestamp"]
                        )
                    )
                    click = _parse_click(
                        row["click"]
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"第{row_number}行：{exc}"
                    ) from exc

                row_count += 1
                insert_batch.append(
                    (
                        user_id,
                        timestamp_ms,
                        click,
                        row_count,
                    )
                )

                if len(insert_batch) >= 100_000:
                    connection.executemany(
                        "INSERT INTO events VALUES (?, ?, ?, ?)",
                        insert_batch,
                    )
                    insert_batch.clear()

                if row_count % 1_000_000 == 0:
                    print(
                        "已导入KuaiSAR推荐日志："
                        f"{row_count:,}行"
                    )

            if insert_batch:
                connection.executemany(
                    "INSERT INTO events VALUES (?, ?, ?, ?)",
                    insert_batch,
                )
            connection.commit()

        print(
            "第2阶段：按user_id和timestamp建立磁盘索引"
        )
        connection.execute(
            "CREATE INDEX events_user_time_idx "
            "ON events(user_id, timestamp_ms, source_order)"
        )
        connection.commit()

        print(
            "第3阶段：按时间顺序重建会话并统计基线"
        )
        current_user_id: str | None = None
        session: _OpenSession | None = None
        sorted_row_count = 0

        cursor = connection.execute(
            "SELECT user_id, timestamp_ms, click "
            "FROM events "
            "ORDER BY user_id, timestamp_ms, source_order"
        )
        for user_id, timestamp_ms, click in cursor:
            sorted_row_count += 1
            user_id = str(user_id)
            timestamp_ms = int(timestamp_ms)
            click = int(click)

            if user_id != current_user_id:
                if session is not None:
                    _finalize_session(
                        session,
                        length_histogram,
                        click_rate_histogram,
                        transition_counts,
                    )
                current_user_id = user_id
                user_count += 1
                session = _OpenSession(
                    last_timestamp_ms=timestamp_ms
                )
            elif (
                session is not None
                and timestamp_ms
                - session.last_timestamp_ms
                > gap_ms
            ):
                _finalize_session(
                    session,
                    length_histogram,
                    click_rate_histogram,
                    transition_counts,
                )
                session = _OpenSession(
                    last_timestamp_ms=timestamp_ms
                )

            if session is None:
                raise RuntimeError(
                    "内部错误：会话没有初始化"
                )

            action = (
                "click"
                if click == 1
                else "next"
            )
            if session.previous_action is not None:
                transition_counts[
                    f"{session.previous_action}->{action}"
                ] += 1

            session.interaction_count += 1
            session.click_count += click
            session.previous_action = action
            session.last_timestamp_ms = timestamp_ms

            if sorted_row_count % 1_000_000 == 0:
                print(
                    "已统计排序后的推荐日志："
                    f"{sorted_row_count:,}行"
                )

        if session is not None:
            _finalize_session(
                session,
                length_histogram,
                click_rate_histogram,
                transition_counts,
            )
    finally:
        if connection is not None:
            connection.close()
        if temp_database_path.is_file():
            temp_database_path.unlink()

    session_count = sum(
        length_histogram.values()
    )
    if session_count <= 0:
        raise ValueError(
            "KuaiSAR日志中没有可用会话"
        )

    baseline: dict[str, Any] = {
        "schema_version": 1,
        "baseline_name": "KuaiSAR recommendation behavior",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": {
            "dataset": "KuaiSAR",
            "domain": "short_video",
            "input_file": str(
                rec_inter_path.resolve()
            ),
            "row_count": row_count,
            "user_count": user_count,
            "session_count": session_count,
        },
        "sessionization": {
            "gap_minutes": session_gap_minutes,
            "rule": (
                "同一用户相邻推荐事件间隔超过阈值时"
                "开始新会话"
            ),
            "recommendation_session_id_available": False,
        },
        "action_mapping": {
            "click=1": "click",
            "click=0": "next_approximation",
            "session_end": "stop_inferred_for_transitions",
        },
        "metrics": {
            "click_rate": {
                "definition": (
                    "每个会话click=1数量除以推荐事件数量"
                ),
                "histogram_scale": CLICK_RATE_SCALE,
                "histogram": _sorted_histogram(
                    click_rate_histogram
                ),
                "sample_count": session_count,
            },
            "session_length_actions": {
                "definition": (
                    "每个重建会话中的推荐事件数量，"
                    "不包含推断的stop"
                ),
                "histogram": _sorted_histogram(
                    length_histogram
                ),
                "sample_count": session_count,
            },
            "action_transitions": {
                "definition": (
                    "click/next近似动作与会话末尾推断stop"
                    "之间的总体转移分布"
                ),
                "counts": {
                    key: int(value)
                    for key, value in sorted(
                        transition_counts.items()
                    )
                },
                "probabilities": (
                    _transition_probabilities(
                        transition_counts
                    )
                ),
            },
        },
        "limitations": [
            "KuaiSAR是短视频场景，当前Agent是电影推荐场景",
            "click=0并不等同于用户主动next",
            "推荐会话和stop均由时间间隔规则推断",
            "该基线只用于离线行为相似度评估，不用于Agent决策",
        ],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            baseline,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return baseline


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "一次性离线生成KuaiSAR真人行为基线JSON"
        )
    )
    parser.add_argument(
        "--rec-inter",
        type=Path,
        required=True,
        help="KuaiSAR的rec_inter.csv或rec_inter.tsv路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kuaisar_baseline.json"),
        help="输出基线JSON路径",
    )
    parser.add_argument(
        "--session-gap-minutes",
        type=int,
        default=30,
        help="重建推荐会话使用的空闲间隔分钟数",
    )
    parser.add_argument(
        "--delimiter",
        choices=["comma", "tab"],
        default=None,
        help="不指定时根据.csv或.tsv自动判断",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    delimiter = None
    if args.delimiter == "comma":
        delimiter = ","
    elif args.delimiter == "tab":
        delimiter = "\t"

    baseline = build_kuaisar_baseline(
        rec_inter_path=args.rec_inter,
        output_path=args.output,
        session_gap_minutes=(
            args.session_gap_minutes
        ),
        delimiter=delimiter,
    )
    print(
        "KuaiSAR基线生成完成："
        f"{args.output.resolve()}"
    )
    print(
        "用户数："
        f"{baseline['source']['user_count']}"
    )
    print(
        "会话数："
        f"{baseline['source']['session_count']}"
    )


if __name__ == "__main__":
    main()
