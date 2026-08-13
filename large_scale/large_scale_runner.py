from __future__ import annotations

import random
import argparse
import asyncio
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from large_scale.llm_policy import LLMPolicy
from large_scale.profile_updater import (
    initialize_dynamic_profile,
    update_dynamic_profile,
)
from large_scale.websim_env import (
    AgentAction,
    AgentState,
    WebSimEnvironment,
)


# 项目根目录：
# C:\Users\wyy05\Desktop\D8EAX
BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_PROFILES_PATH = BASE_DIR / "profiles.jsonl"
DEFAULT_RESULT_DIR = BASE_DIR / "large_scale_runs"


def load_profiles(
    profiles_path: Path,
    count: int,
    selection_mode: str = "sequential",
    start_index: int = 0,
    sample_seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    从profiles.jsonl读取用户画像。

    selection_mode:
    - sequential：从start_index开始连续读取
    - random：从整个画像库无放回随机选择count个

    sample_seed:
    - 相同画像文件和相同seed会得到相同的一组画像
    - 为None时，每次运行通常得到不同结果
    """

    if not profiles_path.exists():
        raise FileNotFoundError(
            f"没有找到用户画像文件：{profiles_path}"
        )

    if count <= 0:
        raise ValueError(
            "count必须大于0"
        )

    if selection_mode not in {
        "sequential",
        "random",
    }:
        raise ValueError(
            "selection_mode必须是"
            "sequential或random"
        )

    all_profiles: list[dict[str, Any]] = []

    with profiles_path.open(
        "r",
        encoding="utf-8",
    ) as profile_file:
        for line_index, line in enumerate(
            profile_file
        ):
            line = line.strip()

            if not line:
                continue

            profile = json.loads(line)

            if not isinstance(profile, dict):
                raise ValueError(
                    f"画像文件第{line_index + 1}行"
                    "不是JSON对象"
                )

            # 没有agent_id时，根据原始行号生成。
            profile.setdefault(
                "agent_id",
                f"agent_{line_index:06d}",
            )

            # 始终保存它在画像库中的真实位置。
            profile["_profile_index"] = line_index

            all_profiles.append(profile)

    if count > len(all_profiles):
        raise ValueError(
            f"要求选择{count}个画像，"
            f"但画像库只有{len(all_profiles)}个"
        )

    if selection_mode == "random":
        # 创建独立随机数生成器，
        # 避免影响程序中其他随机逻辑。
        random_generator = random.Random(
            sample_seed
        )

        selected_profiles = (
            random_generator.sample(
                all_profiles,
                k=count,
            )
        )

        # 排序只为方便查看日志。
        # 被选中的画像集合仍然是随机的。
        selected_profiles.sort(
            key=lambda profile: int(
                profile["_profile_index"]
            )
        )

        return selected_profiles

    # 保留原来的连续选择模式。
    end_index = start_index + count

    selected_profiles = [
        profile
        for profile in all_profiles
        if (
            start_index
            <= int(profile["_profile_index"])
            < end_index
        )
    ]

    if len(selected_profiles) != count:
        raise ValueError(
            f"要求从索引{start_index}开始"
            f"连续读取{count}个画像，"
            f"实际读取到{len(selected_profiles)}个"
        )

    return selected_profiles


def create_agent_states(
    profiles: list[dict[str, Any]],
) -> list[AgentState]:
    """根据画像创建相互独立的逻辑Agent状态。"""

    states: list[AgentState] = []

    for profile in profiles:
        profile_index = int(
            profile["_profile_index"]
        )

        agent_id = str(
            profile.get(
                "agent_id",
                f"agent_{profile_index:06d}",
            )
        )

        seed = int(
            profile.get(
                "seed",
                42 + profile_index,
            )
        )

        state = AgentState(
            agent_id=agent_id,
            profile_index=profile_index,
            profile=profile,
            seed=seed,
        )

        # 根据原始用户画像，生成该Agent独立的动态画像。
        initialize_dynamic_profile(state)

        states.append(state)

    return states


def can_go_next(
    state: AgentState,
    cards: list[dict[str, Any]],
) -> bool:
    """判断当前Agent是否还有下一页。"""

    if state.mode == "random":
        return True

    if not cards:
        return False

    page_size = len(cards)

    next_page_start = (
        state.page_index + 1
    ) * page_size

    return next_page_start < len(
        state.rec_ids
    )


def format_visible_cards(
    cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """整理Agent当前看到的电影卡片。"""

    visible_cards: list[dict[str, Any]] = []

    for index, card in enumerate(cards):
        item_id = str(
            card.get("id")
            or card.get("item_id")
            or ""
        )

        visible_cards.append(
            {
                "index": index,
                "item_id": item_id,
                "title": str(
                    card.get(
                        "title",
                        "",
                    )
                ),
                "description": str(
                    card.get(
                        "description",
                        card.get(
                            "desc",
                            "",
                        ),
                    )
                ),
                "rating": str(
                    card.get(
                        "rating_value",
                        card.get(
                            "rating",
                            "",
                        ),
                    )
                ),
                "heat": str(
                    card.get(
                        "rating_count",
                        card.get(
                            "heat",
                            "",
                        ),
                    )
                ),
            }
        )

    return visible_cards


async def process_one_agent(
    state: AgentState,
    simulation_step: int,
    cards: list[dict[str, Any]],
    environment: WebSimEnvironment,
    policy: LLMPolicy,
    run_id: str,
    dataset: str,
    model: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """让一个逻辑Agent完成一轮真实大模型决策。"""

    # 保存动作发生之前的历史。
    history_before = list(
        state.history
    )

    visible_cards = format_visible_cards(
        cards
    )

    # 保存本轮决策前的动态画像。
    dynamic_profile_before = deepcopy(
        state.dynamic_profile
    )

    # 点击动作对应的电影卡片。
    selected_card: dict[str, Any] | None = None

    # 保存本轮画像具体发生了什么变化。
    profile_update: dict[str, Any] = {}

    started_at = time.perf_counter()

    try:
        # 使用CAMEL和云雾大模型作出决策。
        # 取得当前动态耐心。
        current_patience = float(
            state.dynamic_profile.get(
                "current_patience",
                1.0,
            )
        )

        consecutive_next = int(
            state.dynamic_profile.get(
                "consecutive_next",
                0,
            )
        )

        boredom = float(
            state.dynamic_profile.get(
                "boredom",
                0.0,
            )
        )

        # 动态画像达到停止条件时，
        # 本地程序直接决定stop，不再浪费一次云雾API请求。
        if current_patience <= 0.05:
            action = AgentAction(
                action="stop",
                reason=(
                    "当前耐心已经接近耗尽，"
                    "用户停止继续浏览"
                ),
            )

        elif boredom >= 0.80:
            action = AgentAction(
                action="stop",
                reason=(
                    "当前无聊程度已经很高，"
                    "用户停止继续浏览"
                ),
            )

        elif consecutive_next >= 8:
            action = AgentAction(
                action="stop",
                reason=(
                    "用户已经连续翻页多次，"
                    "决定结束本次浏览"
                ),
            )

        else:
            # 没有达到硬性停止条件时，
            # 再交给CAMEL和云雾大模型自主决策。
            action = await policy.decide(
                state=state,
                cards=cards,
                next_enabled=can_go_next(
                    state,
                    cards,
                ),
            )

        # 大模型或本地规则选择click时，
        # 找到它实际选择的电影卡片。
        if action.action == "click":
            selected_card = next(
                (
                    card
                    for card in cards
                    if str(
                    card.get("id")
                    or card.get("item_id")
                    or ""
                )
                       == str(action.item_id)
                ),
                None,
            )

        # 真正执行click、next或stop。
        next_cards = environment.step(
            state=state,
            action=action,
        )

        # 动作成功执行后，立即更新该Agent的动态画像。
        # 动作成功执行后，立即更新该Agent的动态画像。
        profile_update = update_dynamic_profile(
            state=state,
            action=action,
            selected_card=selected_card,
        )

        # 点击以后，动态画像已经学习了本次电影的类型。
        # 现在使用最新的interest_weights重新排列推荐列表。
        if action.action == "click":
            next_cards = environment.rerank_by_preferences(
                state
            )

        success = True
        error_message = None

        success = True
        error_message = None

    except Exception as error:
        # LLMPolicy内部已经重试过。
        # 最终仍失败时停止该Agent并记录错误。
        state.stopped = True

        action = AgentAction(
            action="stop",
            reason=(
                "大模型请求连续失败，"
                "程序停止该Agent"
            ),
        )

        next_cards = []
        success = False

        error_message = (
            f"{type(error).__name__}: {error}"
        )

    # 无论成功或失败，都保存本轮结束时的画像。
    dynamic_profile_after = deepcopy(
        state.dynamic_profile
    )

    latency_seconds = (
        time.perf_counter()
        - started_at
    )

    event = {
        "run_id": run_id,
        "simulation_step": simulation_step,
        "agent_id": state.agent_id,
        "profile_index": state.profile_index,
        "profile_number": (
            state.profile_index + 1
        ),
        "group": state.profile.get(
            "group",
            "Unknown",
        ),
        "dataset": dataset,
        "model": model,
        "policy": "camel_yunwu_llm",
        "visible_cards": visible_cards,
        "history_before": history_before,

        # 本轮行为前后的动态画像。
        "dynamic_profile_before": dynamic_profile_before,
        "profile_update": profile_update,
        "dynamic_profile_after": dynamic_profile_after,

        "action": action.action,
        "item_id": action.item_id,
        "reason": action.reason,
        "history_after": list(
            state.history
        ),
        "history_length": len(
            state.history
        ),
        "success": success,
        "error": error_message,
        "latency_seconds": round(
            latency_seconds,
            3,
        ),
    }

    return event, next_cards


def build_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "CAMEL+云雾大模型的WebSim"
            "无浏览器大规模Agent运行器"
        )
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="从第几个画像开始，索引从0开始",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="本次运行的逻辑Agent总数",
    )

    parser.add_argument(
        "--profile-selection",
        type=str,
        choices=[
            "sequential",
            "random",
        ],
        default="sequential",
        help=(
            "画像选择模式："
            "sequential表示从start-index连续选择；"
            "random表示从整个画像库随机选择"
        ),
    )

    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help=(
            "随机选择画像时使用的随机种子；"
            "相同种子可复现相同画像集合"
        ),
    )

    parser.add_argument(
        "--profiles",
        type=Path,
        default=DEFAULT_PROFILES_PATH,
        help="profiles.jsonl路径",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ml1m",
        help="WebSim数据集",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="poprec",
        help="推荐模型",
    )

    parser.add_argument(
        "--track",
        type=int,
        default=1,
        help="每个Agent最多交互多少轮",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help=(
            "一次提交到异步队列的Agent数量，"
            "不等于真实API并发数"
        ),
    )

    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="同时调用云雾API的最大Agent数量",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="单次大模型决策的最大尝试次数",
    )

    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="单次云雾API请求的超时秒数",
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="实验结果目录",
    )

    return parser


async def async_main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.start_index < 0:
        raise ValueError(
            "--start-index不能小于0"
        )

    if args.count <= 0:
        raise ValueError(
            "--count必须大于0"
        )

    if args.track <= 0:
        raise ValueError(
            "--track必须大于0"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size必须大于0"
        )

    if args.max_concurrency <= 0:
        raise ValueError(
            "--max-concurrency必须大于0"
        )

    if args.max_retries <= 0:
        raise ValueError(
            "--max-retries必须大于0"
        )

    if args.request_timeout <= 0:
        raise ValueError(
            "--request-timeout必须大于0"
        )

    profiles_path = args.profiles

    if not profiles_path.is_absolute():
        profiles_path = (
            BASE_DIR / profiles_path
        )

    result_dir = args.result_dir

    if not result_dir.is_absolute():
        result_dir = (
            BASE_DIR / result_dir
        )

    profiles = load_profiles(
        profiles_path=profiles_path,
        count=args.count,
        selection_mode=(
            args.profile_selection
        ),
        start_index=args.start_index,
        sample_seed=args.sample_seed,
    )

    selected_profile_indices = [
        int(profile["_profile_index"])
        for profile in profiles
    ]

    selected_agent_ids = [
        str(profile["agent_id"])
        for profile in profiles
    ]

    states = create_agent_states(
        profiles
    )

    print("=" * 65)
    print("WebSim CAMEL+云雾AI大规模Agent实验")
    print(f"Agent数量：{args.count}")
    print(
        f"画像选择模式："
        f"{args.profile_selection}"
    )

    if args.profile_selection == "random":
        print(
            f"随机种子："
            f"{args.sample_seed}"
        )

        print(
            "选中的画像索引："
            + ", ".join(
                str(index)
                for index
                in selected_profile_indices
            )
        )
    else:
        print(
            f"开始索引："
            f"{args.start_index}"
        )

        print(
            f"画像范围："
            f"第{args.start_index + 1}个"
            f"至第"
            f"{args.start_index + args.count}个"
        )
    print(f"数据集：{args.dataset}")
    print(f"推荐模型：{args.model}")
    print(f"交互轮数：{args.track}")
    print(f"任务批大小：{args.batch_size}")
    print(
        f"云雾API最大并发："
        f"{args.max_concurrency}"
    )
    print(
        f"单次请求超时："
        f"{args.request_timeout}秒"
    )
    print("=" * 65)

    print("正在加载无浏览器WebSim环境……")

    environment = WebSimEnvironment(
        dataset=args.dataset,
        model=args.model,
    )

    print("正在创建大模型并发控制器……")

    policy = LLMPolicy(
        max_concurrency=args.max_concurrency,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
    )

    print("正在初始化Agent状态……")

    observations: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for state in states:
        observations[state.agent_id] = (
            environment.reset(state)
        )

    run_id = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_llm_large"
    )

    run_directory = (
        result_dir / run_id
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    events_path = (
        run_directory
        / "events.jsonl"
    )

    summary_path = (
        run_directory
        / "summary.json"
    )

    memory_path = (
        run_directory
        / "memory.json"
    )

    scheduler_log_path = (
        run_directory
        / "scheduler.log"
    )

    def scheduler_log(
        message: str,
    ) -> None:
        """同时写入终端和scheduler.log。"""

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        line = (
            f"{timestamp} | {message}"
        )

        print(line)

        with scheduler_log_path.open(
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                line + "\n"
            )

    # 创建汇总版memory结构。
    memory: dict[str, Any] = {
        "run_id": run_id,
        "config": {
            "selection_mode": (
    args.profile_selection
),
"start_index": (
    args.start_index
    if args.profile_selection
    == "sequential"
    else None
),
"sample_seed": args.sample_seed,
"selected_profile_indices": (
    selected_profile_indices
),
"selected_agent_ids": (
    selected_agent_ids
),
"agent_count": args.count,
            "dataset": args.dataset,
            "model": args.model,
            "track": args.track,
            "batch_size": args.batch_size,
            "max_api_concurrency": (
                args.max_concurrency
            ),
            "max_retries": (
                args.max_retries
            ),
            "policy": "camel_yunwu_llm",
        },
        "agents": {},
    }

    for state in states:
        clean_profile = {
            key: value
            for key, value
            in state.profile.items()
            if key != "_profile_index"
        }

        memory["agents"][
            state.agent_id
        ] = {
            "agent_id": state.agent_id,
            "profile_index": (
                state.profile_index
            ),
            "profile_number": (
                state.profile_index + 1
            ),
            "profile": clean_profile,
            "steps": [],
        }

    scheduler_log(
        f"实验开始 | "
        f"Agent数量={args.count} | "
        f"画像索引={args.start_index}～"
        f"{args.start_index + args.count - 1}"
    )

    scheduler_log(
        f"运行配置 | "
        f"track={args.track} | "
        f"batch_size={args.batch_size} | "
        f"max_api_concurrency="
        f"{args.max_concurrency} | "
        f"max_retries={args.max_retries}"
    )

    scheduler_log(
        f"环境配置 | "
        f"dataset={args.dataset} | "
        f"model={args.model} | "
        f"policy=camel_yunwu_llm"
    )

    started_at = time.perf_counter()

    event_count = 0
    click_count = 0
    next_count = 0
    stop_count = 0
    failed_count = 0

    with events_path.open(
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as event_file:

        for simulation_step in range(
            1,
            args.track + 1,
        ):
            step_started_at = (
                time.perf_counter()
            )

            active_states = [
                state
                for state in states
                if not state.stopped
            ]

            if not active_states:
                scheduler_log(
                    "所有Agent均已停止，"
                    "实验提前结束"
                )
                break

            scheduler_log(
                f"第{simulation_step}轮开始 | "
                f"活跃Agent={len(active_states)}"
            )

            for batch_start in range(
                0,
                len(active_states),
                args.batch_size,
            ):
                batch_states = active_states[
                    batch_start:
                    batch_start + args.batch_size
                ]

                batch_number = (
                    batch_start
                    // args.batch_size
                    + 1
                )

                batch_first_agent = (
                    batch_states[0].agent_id
                )

                batch_last_agent = (
                    batch_states[-1].agent_id
                )

                scheduler_log(
                    f"第{simulation_step}轮 "
                    f"第{batch_number}批开始 | "
                    f"Agent={batch_first_agent}～"
                    f"{batch_last_agent} | "
                    f"任务数={len(batch_states)} | "
                    f"API最大并发="
                    f"{args.max_concurrency}"
                )

                # 创建本批异步任务。
                # LLMPolicy内部的工作池限制真实API并发数量。
                tasks = [
                    process_one_agent(
                        state=state,
                        simulation_step=(
                            simulation_step
                        ),
                        cards=observations.get(
                            state.agent_id,
                            [],
                        ),
                        environment=environment,
                        policy=policy,
                        run_id=run_id,
                        dataset=args.dataset,
                        model=args.model,
                    )
                    for state in batch_states
                ]

                results = await asyncio.gather(
                    *tasks
                )

                for state, result in zip(
                    batch_states,
                    results,
                ):
                    event, next_cards = (
                        result
                    )

                    observations[
                        state.agent_id
                    ] = next_cards

                    # 原始事件流。
                    event_file.write(
                        json.dumps(
                            event,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    # 按Agent整理到memory.json。
                    memory[
                        "agents"
                    ][state.agent_id][
                        "steps"
                    ].append(
                        {
                            "simulation_step": (
                                event[
                                    "simulation_step"
                                ]
                            ),
                            "visible_cards": (
                                event[
                                    "visible_cards"
                                ]
                            ),
                            "history_before": (
                                event[
                                    "history_before"
                                ]
                            ),
                            "dynamic_profile_before": (
                                event[
                                    "dynamic_profile_before"
                                ]
                            ),
                            "profile_update": (
                                event[
                                    "profile_update"
                                ]
                            ),
                            "dynamic_profile_after": (
                                event[
                                    "dynamic_profile_after"
                                ]
                            ),
                            "action": (
                                event[
                                    "action"
                                ]
                            ),
                            "item_id": (
                                event[
                                    "item_id"
                                ]
                            ),
                            "reason": (
                                event[
                                    "reason"
                                ]
                            ),
                            "history_after": (
                                event[
                                    "history_after"
                                ]
                            ),
                            "success": (
                                event[
                                    "success"
                                ]
                            ),
                            "error": (
                                event[
                                    "error"
                                ]
                            ),
                            "latency_seconds": (
                                event[
                                    "latency_seconds"
                                ]
                            ),
                        }
                    )

                    event_count += 1

                    if not event["success"]:
                        failed_count += 1

                    if event["action"] == "click":
                        click_count += 1

                    elif event["action"] == "next":
                        next_count += 1

                    elif event["action"] == "stop":
                        stop_count += 1

                # 把当前批事件立即写入硬盘。
                event_file.flush()

                completed = min(
                    batch_start
                    + len(batch_states),
                    len(active_states),
                )

                batch_success_count = sum(
                    1
                    for event, _ in results
                    if event["success"]
                )

                batch_failed_count = (
                    len(results)
                    - batch_success_count
                )

                scheduler_log(
                    f"第{simulation_step}轮 "
                    f"第{batch_number}批完成 | "
                    f"成功={batch_success_count} | "
                    f"失败={batch_failed_count} | "
                    f"本轮进度={completed}/"
                    f"{len(active_states)}"
                )

            step_elapsed = (
                time.perf_counter()
                - step_started_at
            )

            active_agent_count = sum(
                1
                for state in states
                if not state.stopped
            )

            scheduler_log(
                f"第{simulation_step}轮完成 | "
                f"累计事件={event_count} | "
                f"仍活跃Agent="
                f"{active_agent_count} | "
                f"本轮耗时="
                f"{step_elapsed:.2f}秒"
            )

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    events_per_second = (
        event_count / elapsed_seconds
        if elapsed_seconds > 0
        else 0
    )

    summary = {
        "run_id": run_id,
        "selection_mode": (
            args.profile_selection
        ),
        "start_index": (
            args.start_index
            if args.profile_selection
               == "sequential"
            else None
        ),
        "sample_seed": args.sample_seed,
        "selected_profile_indices": (
            selected_profile_indices
        ),
        "selected_agent_ids": (
            selected_agent_ids
        ),
        "agent_count": args.count,
        "dataset": args.dataset,
        "model": args.model,
        "track": args.track,
        "batch_size": args.batch_size,
        "max_api_concurrency": (
            args.max_concurrency
        ),
        "max_retries": (
            args.max_retries
        ),
        "policy": "camel_yunwu_llm",
        "event_count": event_count,
        "click_count": click_count,
        "next_count": next_count,
        "stop_count": stop_count,
        "failed_count": failed_count,
        "elapsed_seconds": round(
            elapsed_seconds,
            3,
        ),
        "events_per_second": round(
            events_per_second,
            3,
        ),
        "events_file": str(
            events_path
        ),
        "summary_file": str(
            summary_path
        ),
        "memory_file": str(
            memory_path
        ),
        "scheduler_log_file": str(
            scheduler_log_path
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    memory["summary"] = summary

    memory_path.write_text(
        json.dumps(
            memory,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scheduler_log(
        f"实验完成 | "
        f"Agent数量={args.count} | "
        f"AI决策事件={event_count} | "
        f"点击={click_count} | "
        f"翻页={next_count} | "
        f"停止={stop_count} | "
        f"失败={failed_count} | "
        f"总耗时={elapsed_seconds:.2f}秒"
    )

    scheduler_log(
        f"结果目录：{run_directory}"
    )

    print("\n" + "=" * 65)
    print("CAMEL+云雾AI大规模Agent实验完成")
    print(f"Agent数量：{args.count}")
    print(f"AI决策事件数：{event_count}")
    print(f"点击次数：{click_count}")
    print(f"翻页次数：{next_count}")
    print(f"停止次数：{stop_count}")
    print(f"失败次数：{failed_count}")
    print(f"总耗时：{elapsed_seconds:.2f}秒")
    print(
        f"处理速度："
        f"{events_per_second:.3f}"
        f"次AI决策/秒"
    )
    print(f"结果目录：{run_directory}")
    print("=" * 65)


def main() -> None:
    asyncio.run(
        async_main()
    )


if __name__ == "__main__":
    main()