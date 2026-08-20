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
from large_scale.risk_evaluator import (
    evaluate_activity_anomaly,
    evaluate_goal_conflict,
    summarize_recommender_addiction_risk,
    summarize_session,
)
from large_scale.realism_evaluator import (
    evaluate_events_file,
)
from large_scale.risk_profiles import (
    ensure_risk_profile,
    sample_session_start_minute,
)
from large_scale.special_events import (
    SPECIAL_EVENT_CHOICES,
    active_external_interruption,
    apply_special_event,
    select_special_event,
)
from large_scale.profile_updater import (
    initialize_dynamic_profile,
    update_dynamic_profile,
)
from large_scale.websim_env import (
    AgentAction,
    AgentState,
    WebSimEnvironment,
)


# 项目根目录。
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

            profile = ensure_risk_profile(profile)

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
    session_time_seed: int,
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

        session_start_minute = sample_session_start_minute(
            profile=profile,
            experiment_seed=session_time_seed,
            day_number=1,
        )

        state = AgentState(
            agent_id=agent_id,
            profile_index=profile_index,
            profile=profile,
            seed=seed,
            session_start_minute=session_start_minute,
            simulation_minute=session_start_minute,
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

def format_simulation_time(
    total_minutes: int,
) -> str:
    """把仿真分钟转换成第几天和具体时间。"""

    day_number = total_minutes // 1440 + 1
    minute_of_day = total_minutes % 1440

    hour = minute_of_day // 60
    minute = minute_of_day % 60

    return (
        f"day_{day_number:02d} "
        f"{hour:02d}:{minute:02d}"
    )

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

    # 保存本轮动作执行前的仿真时间。
    simulation_minute_before = int(
        state.simulation_minute
    )

    # 根据当前时间和个人24小时基线，
    # 计算本轮活动异常度。
    activity_evaluation = (
        evaluate_activity_anomaly(
            profile=state.profile,
            simulation_minute=(
                simulation_minute_before
            ),
            actual_activity=1.0,
        )
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

    # 外部事件（例如停电）可能强制中断会话。
    external_interruption = active_external_interruption(
        profile=state.profile,
        simulation_minute=simulation_minute_before,
    )

    started_at = time.perf_counter()

    try:
        # 只有停电等真实外部事件可以强制中断。
        # 耐心、无聊和连续翻页仅作为画像信号交给大模型，
        # 不再由本地阈值替Agent决定stop。
        if external_interruption is not None:
            action = AgentAction(
                action="stop",
                reason=(
                    f"{external_interruption.get('event_name', '外部事件')}"
                    "导致推荐系统暂时不可用，会话被迫中断"
                ),
                intended_to_stop=False,
                intention_reason="用户没有主动停止，由外部事件中断",
            )

        else:
            # 正常情况下始终由Agent自主选择click、next或stop。
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

    # 保存动作执行后的仿真时间。
    simulation_minute_after = int(
        state.simulation_minute
    )

    # 判断本轮行为是否影响高优先级目标。
    goal_evaluation = evaluate_goal_conflict(
        profile=state.profile,
        simulation_minute_after=(
            simulation_minute_after
        ),
        action=action.action,
    )

    # 判断是否出现“想停止却继续”。
    #
    # 只有动作成功执行，并且Agent已经产生停止意图，
    # 但实际动作仍然是click或next，才算停止失败。
    stop_failure = bool(
        success
        and action.intended_to_stop
        and action.action in {
            "click",
            "next",
        }
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
        "simulation_minute_before": (
            simulation_minute_before
        ),
        "simulation_minute_after": (
            simulation_minute_after
        ),
        "simulation_time_before": (
            format_simulation_time(
                simulation_minute_before
            )
        ),
        "simulation_time_after": (
            format_simulation_time(
                simulation_minute_after
            )
        ),
        "current_hour": (
                simulation_minute_before
                % 1440
                // 60
        ),
        "activity_evaluation": (
            activity_evaluation
        ),
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
        "policy": "camel_bailian_qwen_llm",
        "visible_cards": visible_cards,
        "history_before": history_before,

        # 本轮行为前后的动态画像。
        "dynamic_profile_before": dynamic_profile_before,
        "profile_update": profile_update,
        "dynamic_profile_after": dynamic_profile_after,

                # Agent动作前的停止意图。
        "intended_to_stop": bool(
            action.intended_to_stop
        ),
        "intention_reason": (
            action.intention_reason
        ),

        # Agent最终实际执行的动作。
        "action": action.action,
        "item_id": action.item_id,
        "reason": action.reason,

        # 是否出现“想停止却继续”。
        "stop_failure": stop_failure,
        # 当前生活目标和目标冲突结果。
        "goal_evaluation": (
            goal_evaluation
        ),
        "goal_conflict": bool(
            goal_evaluation.get(
                "goal_conflict",
                False,
            )
        ),
        "special_event": deepcopy(
            state.profile.get("special_event")
        ),
        "external_interruption": (
            external_interruption is not None
        ),
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
            "CAMEL+阿里云百炼大模型的WebSim"
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
        default=None,
        help=(
            "可选的实验轮数上限；不填写时进入Agent自主运行模式，"
            "由max-auto-steps和max-session-minutes负责安全兜底"
        ),
    )

    parser.add_argument(
        "--max-auto-steps",
        type=int,
        default=50,
        help=(
            "未填写track时，每个Agent自主运行的最大安全交互轮数；"
            "仅用于防止无限运行，不要求Agent必须运行到该轮数"
        ),
    )

    parser.add_argument(
        "--max-session-minutes",
        type=int,
        default=120,
        help=(
            "每个Agent单次会话最多推进多少个仿真分钟；"
            "达到后标记为安全截断"
        ),
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
        help="同时调用阿里云百炼API的最大Agent数量",
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
        help="单次阿里云百炼API请求的超时秒数",
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="实验结果目录",
    )

    parser.add_argument(
        "--realism-baseline",
        type=Path,
        default=None,
        help=(
            "可选的KuaiSAR基线JSON；指定后在实验结束时"
            "离线生成realism_report.json"
        ),
    )

    parser.add_argument(
        "--special-event",
        type=str,
        choices=SPECIAL_EVENT_CHOICES,
        default="none",
        help=(
            "特殊事件情景：none不使用；random随机选择；"
            "也可指定summer_vacation、holiday、power_outage、"
            "exam_week、project_deadline或sick_leave"
        ),
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

    if args.track is not None and args.track <= 0:
        raise ValueError(
            "--track必须大于0"
        )

    if args.max_auto_steps <= 0:
        raise ValueError(
            "--max-auto-steps必须大于0"
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

    if args.max_session_minutes <= 0:
        raise ValueError(
            "--max-session-minutes必须大于0"
        )

    autonomous_mode = args.track is None
    execution_mode = (
        "autonomous_until_stop"
        if autonomous_mode
        else "explicit_track_limit"
    )
    effective_max_steps = (
        args.max_auto_steps
        if autonomous_mode
        else args.track
    )

    realism_baseline_path = (
        args.realism_baseline
    )
    if realism_baseline_path is not None:
        if not realism_baseline_path.is_absolute():
            realism_baseline_path = (
                BASE_DIR / realism_baseline_path
            )
        if not realism_baseline_path.is_file():
            raise FileNotFoundError(
                "找不到KuaiSAR真实性基线："
                f"{realism_baseline_path}"
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

    session_time_seed = (
        args.sample_seed
        if args.sample_seed is not None
        else random.SystemRandom().randrange(
            0,
            2**63,
        )
    )

    selected_special_event = select_special_event(
        requested_event=args.special_event,
        seed=session_time_seed,
    )
    profiles = [
        apply_special_event(
            profile,
            selected_special_event,
        )
        for profile in profiles
    ]
    special_event_affected_agent_count = sum(
        1
        for profile in profiles
        if profile.get("special_event", {}).get("applicable") is True
    )
    special_event_summary: dict[str, Any] = {
        "mode": args.special_event,
        "selected_event_id": (
            selected_special_event.get("event_id")
            if selected_special_event is not None
            else "none"
        ),
        "selected_event_name": (
            selected_special_event.get("event_name")
            if selected_special_event is not None
            else "无特殊事件"
        ),
        "description": (
            selected_special_event.get("description")
            if selected_special_event is not None
            else None
        ),
        "affected_agent_count": (
            special_event_affected_agent_count
        ),
    }

    states = create_agent_states(
        profiles,
        session_time_seed=session_time_seed,
    )

    print("=" * 65)
    print("WebSim CAMEL+阿里云百炼大规模Agent实验")
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
    print(
        "运行模式："
        + (
            "Agent自主运行直到主动停止"
            if autonomous_mode
            else "指定track轮数上限"
        )
    )
    print(f"最大安全交互轮数：{effective_max_steps}")
    print(
        "最大安全仿真时长："
        f"{args.max_session_minutes}分钟"
    )
    print(
        "会话开始时间：按个人24小时活动基线随机抽取"
    )
    print(f"会话时间随机种子：{session_time_seed}")
    if selected_special_event is None:
        print("特殊事件：无")
    else:
        print(
            "特殊事件："
            f"{selected_special_event['event_name']} "
            f"({selected_special_event['event_id']})"
        )
        print(
            "特殊事件适用Agent数量："
            f"{special_event_affected_agent_count}"
        )
    print(f"任务批大小：{args.batch_size}")
    print(
        f"阿里云百炼API最大并发："
        f"{args.max_concurrency}"
    )
    print(
        f"单次请求超时："
        f"{args.request_timeout}秒"
    )
    if realism_baseline_path is not None:
        print(
            "实验结束后执行KuaiSAR离线真实性评估："
            f"{realism_baseline_path}"
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

    addiction_report_path = (
        run_directory
        / "addiction_report.json"
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
"session_time_seed": session_time_seed,
"session_start_sampling": "profile_activity_weighted_random",
"special_event": special_event_summary,
"selected_profile_indices": (
    selected_profile_indices
),
"selected_agent_ids": (
    selected_agent_ids
),
"agent_count": args.count,
            "dataset": args.dataset,
            "model": args.model,
            "execution_mode": execution_mode,
            "requested_track": args.track,
            "max_auto_steps": args.max_auto_steps,
            "effective_max_steps": effective_max_steps,
            "max_session_minutes": args.max_session_minutes,
            "batch_size": args.batch_size,
            "max_api_concurrency": (
                args.max_concurrency
            ),
            "max_retries": (
                args.max_retries
            ),
            "policy": "camel_bailian_qwen_llm",
            "realism_baseline": (
                str(realism_baseline_path)
                if realism_baseline_path is not None
                else None
            ),
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
            "session_start_minute": state.session_start_minute,
            "session_start_time": format_simulation_time(
                state.session_start_minute
            ),
            "termination": None,
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
        f"mode={execution_mode} | "
        f"max_steps={effective_max_steps} | "
        f"max_session_minutes={args.max_session_minutes} | "
        f"batch_size={args.batch_size} | "
        f"max_api_concurrency="
        f"{args.max_concurrency} | "
        f"max_retries={args.max_retries}"
    )

    scheduler_log(
        f"环境配置 | "
        f"dataset={args.dataset} | "
        f"model={args.model} | "
        f"policy=camel_bailian_qwen_llm"
    )

    started_at = time.perf_counter()

    event_count = 0
    click_count = 0
    next_count = 0
    stop_count = 0
    failed_count = 0
    termination_by_agent: dict[str, dict[str, Any]] = {}

    with events_path.open(
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as event_file:

        for simulation_step in range(
            1,
            effective_max_steps + 1,
        ):
            step_started_at = (
                time.perf_counter()
            )

            # 仿真时长是第二重安全上限。
            for state in states:
                if state.stopped:
                    continue
                elapsed_simulation_minutes = (
                    state.simulation_minute
                    - state.session_start_minute
                )
                if elapsed_simulation_minutes >= args.max_session_minutes:
                    state.stopped = True
                    termination_by_agent[state.agent_id] = {
                        "type": "safety_limit_censored",
                        "reason": "max_session_minutes",
                        "limit": args.max_session_minutes,
                        "observed_steps": state.step,
                        "observed_simulation_minutes": (
                            elapsed_simulation_minutes
                        ),
                    }

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
                            "simulation_minute_before": (
                                event[
                                    "simulation_minute_before"
                                ]
                            ),
                            "simulation_minute_after": (
                                event[
                                    "simulation_minute_after"
                                ]
                            ),
                            "simulation_time_before": (
                                event[
                                    "simulation_time_before"
                                ]
                            ),
                            "simulation_time_after": (
                                event[
                                    "simulation_time_after"
                                ]
                            ),
                            "current_hour": (
                                event[
                                    "current_hour"
                                ]
                            ),
                            "activity_evaluation": (
                                event[
                                    "activity_evaluation"
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
                            "intended_to_stop": (
                                event[
                                    "intended_to_stop"
                                ]
                            ),
                            "intention_reason": (
                                event[
                                    "intention_reason"
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
                            "stop_failure": (
                                event[
                                    "stop_failure"
                                ]
                            ),
                            "goal_evaluation": (
                                event[
                                    "goal_evaluation"
                                ]
                            ),
                            "goal_conflict": (
                                event[
                                    "goal_conflict"
                                ]
                            ),
                            "special_event": (
                                event["special_event"]
                            ),
                            "external_interruption": (
                                event["external_interruption"]
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
                        termination_by_agent[state.agent_id] = {
                            "type": "technical_failure_censored",
                            "reason": event.get("error"),
                            "observed_steps": state.step,
                            "observed_simulation_minutes": (
                                state.simulation_minute
                                - state.session_start_minute
                            ),
                        }

                    if event["action"] == "click":
                        click_count += 1

                    elif event["action"] == "next":
                        next_count += 1

                    elif event["action"] == "stop":
                        stop_count += 1
                        if event.get("external_interruption") is True:
                            termination_by_agent[state.agent_id] = {
                                "type": "external_interruption",
                                "reason": event.get("reason"),
                                "observed_steps": state.step,
                                "observed_simulation_minutes": (
                                    state.simulation_minute
                                    - state.session_start_minute
                                ),
                            }
                        elif event["success"]:
                            termination_by_agent[state.agent_id] = {
                                "type": "natural_stop",
                                "reason": event.get("reason"),
                                "observed_steps": state.step,
                                "observed_simulation_minutes": (
                                    state.simulation_minute
                                    - state.session_start_minute
                                ),
                            }

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

    # 运行完最大轮数仍未主动停止的Agent属于右截断，
    # 不能被当成正常使用。
    for state in states:
        if state.agent_id in termination_by_agent:
            continue
        elapsed_simulation_minutes = (
            state.simulation_minute
            - state.session_start_minute
        )
        termination_by_agent[state.agent_id] = {
            "type": "safety_limit_censored",
            "reason": (
                "max_session_minutes"
                if elapsed_simulation_minutes >= args.max_session_minutes
                else "max_steps"
            ),
            "limit": (
                args.max_session_minutes
                if elapsed_simulation_minutes >= args.max_session_minutes
                else effective_max_steps
            ),
            "observed_steps": state.step,
            "observed_simulation_minutes": (
                elapsed_simulation_minutes
            ),
        }

    # 汇总每个Agent在本次session中的风险证据。
    session_summaries: dict[
        str,
        dict[str, Any],
    ] = {}

    for (
        agent_id,
        agent_memory,
    ) in memory["agents"].items():
        agent_steps = agent_memory.get(
            "steps",
            [],
        )

        termination = termination_by_agent.get(agent_id)
        agent_memory["termination"] = termination

        session_summary = summarize_session(
            agent_steps,
            termination=termination,
        )

        # 保存到该Agent自己的memory中。
        agent_memory[
            "session_summary"
        ] = session_summary

        # 同时保存一份用于总体summary。
        session_summaries[
            agent_id
        ] = session_summary

    single_session_warning_count = sum(
        1
        for session_summary
        in session_summaries.values()
        if session_summary.get(
            "session_problematic"
        ) is True
    )

    recommender_addiction_risk = (
        summarize_recommender_addiction_risk(
            session_summaries
        )
    )
    addicted_agent_ids = list(
        recommender_addiction_risk.get(
            "addicted_agent_ids",
            [],
        )
    )
    addicted_agent_count = int(
        recommender_addiction_risk.get(
            "addicted_agent_count",
            0,
        )
    )
    externally_censored_agent_ids = [
        agent_id
        for agent_id, session_summary
        in session_summaries.items()
        if session_summary.get("status") == "externally_censored"
    ]
    safety_censored_agent_ids = [
        agent_id
        for agent_id, session_summary
        in session_summaries.items()
        if session_summary.get("status") == "safety_limit_censored"
    ]
    technical_censored_agent_ids = [
        agent_id
        for agent_id, session_summary
        in session_summaries.items()
        if session_summary.get("status") == "technical_failure_censored"
    ]
    natural_stop_agent_ids = [
        agent_id
        for agent_id, termination
        in termination_by_agent.items()
        if termination.get("type") == "natural_stop"
    ]
    special_event_summary = {
        **special_event_summary,
        "externally_censored_agent_count": len(
            externally_censored_agent_ids
        ),
        "externally_censored_agent_ids": (
            externally_censored_agent_ids
        ),
    }

    addiction_report = {
        "schema_version": 1,
        "report_type": "recommendation_system_addiction_risk",
        "run_id": run_id,
        "recommendation_system": {
            "dataset": args.dataset,
            "model": args.model,
            "policy": "camel_bailian_qwen_llm",
            "session_start_sampling": (
                "profile_activity_weighted_random"
            ),
            "session_time_seed": session_time_seed,
            "special_event": special_event_summary,
            "execution_mode": execution_mode,
            "safety_limits": {
                "requested_track": args.track,
                "max_auto_steps": args.max_auto_steps,
                "effective_max_steps": effective_max_steps,
                "max_session_minutes": args.max_session_minutes,
            },
        },
        "judgement_scope": "single_session_operational_definition",
        "operational_definition": (
            "同一会话中存在活动异常，并且至少有一轮同时出现"
            "Agent想停止却继续和继续行为影响高优先级生活目标"
        ),
        "clinical_diagnosis": False,
        "probability_condition": (
            special_event_summary["selected_event_id"]
        ),
        "special_event": special_event_summary,
        "agent_count": args.count,
        "termination_summary": {
            "natural_stop_count": len(natural_stop_agent_ids),
            "safety_limit_censored_count": len(
                safety_censored_agent_ids
            ),
            "external_interruption_count": len(
                externally_censored_agent_ids
            ),
            "technical_failure_censored_count": len(
                technical_censored_agent_ids
            ),
        },
        "recommendation_system_addiction_risk": (
            recommender_addiction_risk
        ),
        "addicted_agent_count": addicted_agent_count,
        "addicted_agent_ids": addicted_agent_ids,
        "agents": {
            agent_id: session_summary
            for agent_id, session_summary
            in session_summaries.items()
        },
    }
    addiction_report_path.write_text(
        json.dumps(
            addiction_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    realism_evaluation: dict[str, Any] = {
        "status": "not_requested"
    }
    if realism_baseline_path is not None:
        realism_report_path = (
            run_directory / "realism_report.json"
        )
        scheduler_log(
            "开始执行KuaiSAR离线行为真实性评估"
        )
        try:
            realism_report = evaluate_events_file(
                events_path=events_path,
                baseline_path=(
                    realism_baseline_path
                ),
                output_path=realism_report_path,
            )
            realism_evaluation = {
                "status": "ok",
                "baseline_file": str(
                    realism_baseline_path
                ),
                "report_file": str(
                    realism_report_path
                ),
                "aggregate": realism_report.get(
                    "aggregate",
                    {},
                ),
            }
            scheduler_log(
                "KuaiSAR离线真实性评估完成 | "
                "平均得分="
                f"{realism_evaluation['aggregate'].get('mean_behavioral_realism_score')}"
            )
        except Exception as exc:
            realism_evaluation = {
                "status": "error",
                "baseline_file": str(
                    realism_baseline_path
                ),
                "report_file": str(
                    realism_report_path
                ),
                "error": str(exc),
            }
            realism_report_path.write_text(
                json.dumps(
                    realism_evaluation,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            scheduler_log(
                "KuaiSAR离线真实性评估失败 | "
                f"{exc}"
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
        "session_time_seed": session_time_seed,
        "session_start_sampling": (
            "profile_activity_weighted_random"
        ),
        "session_start_times": {
            state.agent_id: format_simulation_time(
                state.session_start_minute
            )
            for state in states
        },
        "special_event": special_event_summary,
        "selected_profile_indices": (
            selected_profile_indices
        ),
        "selected_agent_ids": (
            selected_agent_ids
        ),
        "agent_count": args.count,
        "dataset": args.dataset,
        "model": args.model,
        "execution_mode": execution_mode,
        "track": args.track,
        "max_auto_steps": args.max_auto_steps,
        "effective_max_steps": effective_max_steps,
        "max_session_minutes": args.max_session_minutes,
        "batch_size": args.batch_size,
        "max_api_concurrency": (
            args.max_concurrency
        ),
        "max_retries": (
            args.max_retries
        ),
        "policy": "camel_bailian_qwen_llm",
        "event_count": event_count,
        "click_count": click_count,
        "next_count": next_count,
        "stop_count": stop_count,
        "failed_count": failed_count,
        "single_session_warning_count": (
            single_session_warning_count
        ),
        "addicted_agent_count": addicted_agent_count,
        "addicted_agent_ids": addicted_agent_ids,
        "addiction_report_file": str(addiction_report_path),
        "recommendation_system_addiction_risk": (
            recommender_addiction_risk
        ),
        "termination_summary": {
            "natural_stop_count": len(natural_stop_agent_ids),
            "natural_stop_agent_ids": natural_stop_agent_ids,
            "safety_limit_censored_count": len(
                safety_censored_agent_ids
            ),
            "safety_limit_censored_agent_ids": (
                safety_censored_agent_ids
            ),
            "external_interruption_count": len(
                externally_censored_agent_ids
            ),
            "technical_failure_censored_count": len(
                technical_censored_agent_ids
            ),
        },
        "session_summaries": (
            session_summaries
        ),
        "realism_evaluation": (
            realism_evaluation
        ),
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

    probability_for_log = recommender_addiction_risk.get(
        "estimated_probability_percent"
    )
    scheduler_log(
        "推荐系统沉迷风险估计完成 | "
        "概率="
        + (
            f"{probability_for_log}%"
            if probability_for_log is not None
            else "数据不足"
        )
        + " | "
        f"判定数量={addicted_agent_count}/"
        f"{recommender_addiction_risk.get('evaluated_agent_count')} | "
        "Agent="
        + (
            ", ".join(addicted_agent_ids)
            if addicted_agent_ids
            else "无"
        )
    )

    scheduler_log(
        f"结果目录：{run_directory}"
    )

    print("\n" + "=" * 65)
    print("CAMEL+阿里云百炼大规模Agent实验完成")
    print(f"Agent数量：{args.count}")
    print(f"AI决策事件数：{event_count}")
    print(f"点击次数：{click_count}")
    print(f"翻页次数：{next_count}")
    print(f"停止次数：{stop_count}")
    print(f"失败次数：{failed_count}")
    print(
        "本次特殊事件："
        f"{special_event_summary['selected_event_name']}"
    )
    print(
        "外部事件截断Agent数量："
        f"{len(externally_censored_agent_ids)}"
    )
    print(f"自然停止Agent数量：{len(natural_stop_agent_ids)}")
    print(
        "安全上限截断Agent数量："
        f"{len(safety_censored_agent_ids)}"
    )
    print(
        "技术失败截断Agent数量："
        f"{len(technical_censored_agent_ids)}"
    )
    probability_percent = recommender_addiction_risk.get(
        "estimated_probability_percent"
    )
    confidence_interval = recommender_addiction_risk.get(
        "confidence_interval_95"
    )
    if probability_percent is None:
        print("推荐系统沉迷风险概率：数据不足")
    else:
        print(
            "推荐系统沉迷风险概率："
            f"{probability_percent:.3f}% "
            f"({addicted_agent_count}/"
            f"{recommender_addiction_risk['evaluated_agent_count']})"
        )
        if isinstance(confidence_interval, dict):
            print(
                "95%置信区间："
                f"{confidence_interval['lower_percent']:.3f}%～"
                f"{confidence_interval['upper_percent']:.3f}%"
            )
        evidence_by_goal = recommender_addiction_risk.get(
            "addiction_evidence_by_goal",
            {},
        )
        if isinstance(evidence_by_goal, dict):
            print(
                "沉迷风险涉及的目标："
                + (
                    ", ".join(
                        f"{name}={count}次"
                        for name, count
                        in evidence_by_goal.items()
                    )
                    if evidence_by_goal
                    else "无"
                )
            )
    risk_bounds = recommender_addiction_risk.get(
        "risk_probability_bounds",
        {},
    )
    if (
        isinstance(risk_bounds, dict)
        and risk_bounds.get("lower_percent") is not None
        and risk_bounds.get("upper_percent") is not None
    ):
        print(
            "考虑安全截断后的风险范围："
            f"{risk_bounds['lower_percent']:.3f}%～"
            f"{risk_bounds['upper_percent']:.3f}%"
        )
    print(
        "作为系统风险证据的沉迷Agent数量："
        f"{addicted_agent_count}"
    )
    print(
        "作为系统风险证据的沉迷Agent："
        + (
            ", ".join(addicted_agent_ids)
            if addicted_agent_ids
            else "无"
        )
    )
    print(f"推荐系统沉迷风险报告：{addiction_report_path}")
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
