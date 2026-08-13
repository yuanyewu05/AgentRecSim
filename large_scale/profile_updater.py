from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from large_scale.websim_env import AgentAction, AgentState

from large_scale.genre_utils import (
    normalize_labels as normalize_genre_labels,
)


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """把数值限制在0到1之间。"""

    return max(
        minimum,
        min(maximum, float(value)),
    )


def to_float(
    value: Any,
    default: float,
) -> float:
    """安全地把画像字段转换为浮点数。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_labels(value: Any) -> list[str]:
    """
    把喜欢、厌恶和电影类型统一转换成标准标签列表。

    实际的拆分、英文到中文转换、去重工作，
    统一交给genre_utils.py处理。

    保留这个函数名称，是为了不用修改本文件中
    其他调用normalize_labels()的位置。
    """

    return normalize_genre_labels(value)


def extract_card_labels(
    card: dict[str, Any] | None,
) -> list[str]:
    """
    从电影卡片中读取类型。

    兼容可能出现的字段：
    genres、genre、categories、category、tags。
    """

    if not card:
        return []

    raw_labels = (
        card.get("genres")
        or card.get("genre")
        or card.get("categories")
        or card.get("category")
        or card.get("tags")
        or []
    )

    return normalize_labels(raw_labels)

def smooth_trait_update(
    current_value: float,
    target_value: float,
    learning_rate: float = 0.08,
) -> float:
    """
    把动态属性缓慢推向本轮行为所反映的目标值。

    learning_rate越大，单次行为对画像的影响越强。
    """
    return round(
        clamp(
            current_value * (1.0 - learning_rate)
            + target_value * learning_rate
        ),
        4,
    )


def extract_popularity_signal(
    card: dict[str, Any] | None,
) -> float | None:
    """
    根据电影的rating_count，
    计算0到1之间的热门程度。
    """
    if not card:
        return None

    raw_count = card.get("rating_count")

    # 兼容旧卡片中可能出现的heat字段。
    if raw_count is None:
        raw_count = card.get("heat")

    try:
        rating_count = max(
            0.0,
            float(raw_count),
        )
    except (TypeError, ValueError):
        return None

    # 采用平滑饱和转换：
    #
    # 0票      -> 0.00
    # 1000票   -> 0.50
    # 5000票   -> 0.83
    # 10000票  -> 0.91
    #
    # 热度继续上升时会逐渐接近1，
    # 但不会超过1。
    return clamp(
        rating_count
        / (rating_count + 1000.0)
    )


def extract_rating_signal(
    card: dict[str, Any] | None,
) -> float | None:
    """
    根据电影评分，
    计算0到1之间的高评分程度。
    """
    if not card:
        return None

    raw_rating = card.get("rating_value")

    # 兼容旧卡片中可能出现的rating字段。
    if raw_rating is None:
        raw_rating = card.get("rating")

    try:
        rating_value = float(raw_rating)
    except (TypeError, ValueError):
        return None

    # 0分通常表示缺失评分，
    # 不据此修改rating_sensitivity。
    if rating_value <= 0.0:
        return None

    # 兼容少量可能使用10分制的数据。
    if rating_value > 5.0:
        rating_value = rating_value / 2.0

    # 当前MovieLens评分为5分制。
    return clamp(
        rating_value / 5.0
    )


def get_last_clicked_labels(
    dynamic_profile: dict[str, Any],
) -> list[str]:
    """
    读取当前动作发生前，
    最近一次点击电影的类型。
    """
    recent_actions = dynamic_profile.get(
        "recent_actions",
        [],
    )

    if not isinstance(recent_actions, list):
        return []

    # 从后往前寻找最近一次click。
    for record in reversed(recent_actions):
        if not isinstance(record, dict):
            continue

        if record.get("action") != "click":
            continue

        return normalize_labels(
            record.get(
                "selected_labels",
                [],
            )
        )

    return []


def label_similarity(
    first_labels: list[str],
    second_labels: list[str],
) -> float | None:
    """
    使用Jaccard相似度，
    衡量两次点击的电影类型有多相似。

    完全不相似：0
    完全相同：1
    """
    first_set = set(first_labels)
    second_set = set(second_labels)

    if not first_set or not second_set:
        return None

    union = first_set | second_set
    intersection = first_set & second_set

    return (
        len(intersection)
        / len(union)
    )


def initialize_dynamic_profile(
    state: AgentState,
) -> None:
    """
    根据profiles.jsonl中的固定画像，
    初始化该Agent自己的动态画像。
    """

    # 避免重复初始化。
    if state.dynamic_profile:
        return

    likes = normalize_labels(
        state.profile.get(
            "likes",
            [],
        )
    )

    dislikes = normalize_labels(
        state.profile.get(
            "dislikes",
            [],
        )
    )

    interest_weights: dict[str, float] = {}

    # 初始喜欢项设为较高兴趣。
    for label in likes:
        interest_weights[label] = 0.80

    # 初始厌恶项设为较低兴趣。
    for label in dislikes:
        interest_weights[label] = 0.10

    base_patience = clamp(
        to_float(
            state.profile.get("patience"),
            0.50,
        )
    )

    state.dynamic_profile = {
        # 每种兴趣当前的动态权重。
        "interest_weights": interest_weights,

        # 以下属性从原始画像复制，
        # 后续可以根据行为缓慢变化。
        "exploration_rate": clamp(
            to_float(
                state.profile.get(
                    "exploration_rate"
                ),
                0.30,
            )
        ),
        "popularity_bias": clamp(
            to_float(
                state.profile.get(
                    "popularity_bias"
                ),
                0.50,
            )
        ),
        "rating_sensitivity": clamp(
            to_float(
                state.profile.get(
                    "rating_sensitivity"
                ),
                0.50,
            )
        ),
        "novelty_preference": clamp(
            to_float(
                state.profile.get(
                    "novelty_preference"
                ),
                0.50,
            )
        ),
        "repeat_aversion": clamp(
            to_float(
                state.profile.get(
                    "repeat_aversion"
                ),
                0.50,
            )
        ),

        # 原始耐心不变，用于记录初始人格。
        "baseline_patience": base_patience,

        # 当前耐心会随着翻页、点击而变化。
        "current_patience": base_patience,

        # 当前心理状态。
        "satisfaction": 0.50,
        "boredom": 0.00,

        # 行为状态。
        "consecutive_next": 0,
        "click_count": 0,
        "next_count": 0,
        "stop_count": 0,

        # 只保留最近几次动作。
        "recent_actions": [],
    }


def get_base_interest(
    state: AgentState,
    label: str,
) -> float:
    """取得某个兴趣在原始画像中的基础值。"""

    likes = set(
        normalize_labels(
            state.profile.get(
                "likes",
                [],
            )
        )
    )

    dislikes = set(
        normalize_labels(
            state.profile.get(
                "dislikes",
                [],
            )
        )
    )

    if label in likes:
        return 0.80

    if label in dislikes:
        return 0.10

    return 0.50


def build_changed_fields(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """记录本轮具体修改了哪些动态属性。"""

    changes: dict[str, Any] = {}

    all_keys = set(before) | set(after)

    for key in all_keys:
        old_value = before.get(key)
        new_value = after.get(key)

        if old_value != new_value:
            changes[key] = {
                "before": old_value,
                "after": new_value,
            }

    return changes


def update_dynamic_profile(
    state: AgentState,
    action: AgentAction,
    selected_card: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    根据本轮实际动作实时更新动态画像。

    click：
    增强点击内容相关兴趣，
    提高满意度，降低无聊程度。

    next：
    降低耐心和满意度，
    增加无聊程度和连续翻页次数。

    stop：
    记录停止，并进一步降低当前耐心。
    """

    initialize_dynamic_profile(state)

    before = deepcopy(
        state.dynamic_profile
    )

    dynamic = state.dynamic_profile

    interests: dict[str, float] = (
        dynamic.setdefault(
            "interest_weights",
            {},
        )
    )

    # 所有动态兴趣都轻微回归原始画像，
    # 防止某个兴趣无限增强或无限减弱。
    for label in list(interests):
        current_value = to_float(
            interests.get(label),
            0.50,
        )

        base_value = get_base_interest(
            state,
            label,
        )

        interests[label] = round(
            clamp(
                current_value * 0.98
                + base_value * 0.02
            ),
            4,
        )

    selected_labels: list[str] = []

    if action.action == "click":
        selected_labels = extract_card_labels(
            selected_card
        )

        # 点击代表正反馈，增强相关类型兴趣。
        for label in selected_labels:
            current_value = to_float(
                interests.get(
                    label,
                    get_base_interest(
                        state,
                        label,
                    ),
                ),
                0.50,
            )

            interests[label] = round(
                clamp(
                    current_value * 0.88
                    + 1.00 * 0.12
                ),
                4,
            )

        dynamic["click_count"] = (
            int(
                dynamic.get(
                    "click_count",
                    0,
                )
            )
            + 1
        )

        dynamic["consecutive_next"] = 0

        dynamic["satisfaction"] = round(
            clamp(
                to_float(
                    dynamic.get(
                        "satisfaction"
                    ),
                    0.50,
                )
                + 0.08
            ),
            4,
        )

        dynamic["boredom"] = round(
            clamp(
                to_float(
                    dynamic.get("boredom"),
                    0.00,
                )
                - 0.08
            ),
            4,
        )

        dynamic["current_patience"] = round(
            clamp(
                to_float(
                    dynamic.get(
                        "current_patience"
                    ),
                    0.50,
                )
                + 0.03
            ),
            4,
        )

        # 点击原始likes之外的内容，
        # 说明该用户发生了一定探索。
        original_likes = set(
            normalize_labels(
                state.profile.get(
                    "likes",
                    [],
                )
            )
        )

        novel_click = bool(
            selected_labels
            and any(
                label not in original_likes
                for label in selected_labels
            )
        )

        # ==================================================
        # 1. 动态更新热门偏好 popularity_bias
        # ==================================================
        # 点击高热度电影：
        # popularity_bias向高处变化。
        #
        # 点击低热度电影：
        # popularity_bias向低处变化。
        popularity_signal = (
            extract_popularity_signal(
                selected_card
            )
        )

        if popularity_signal is not None:
            old_popularity_bias = to_float(
                dynamic.get(
                    "popularity_bias"
                ),
                0.50,
            )

            dynamic["popularity_bias"] = (
                smooth_trait_update(
                    current_value=old_popularity_bias,
                    target_value=popularity_signal,
                )
            )

        # ==================================================
        # 2. 动态更新评分敏感度 rating_sensitivity
        # ==================================================
        # 点击高评分电影：
        # rating_sensitivity向高处变化。
        #
        # 点击较低评分电影：
        # rating_sensitivity向低处变化。
        rating_signal = extract_rating_signal(
            selected_card
        )

        if rating_signal is not None:
            old_rating_sensitivity = to_float(
                dynamic.get(
                    "rating_sensitivity"
                ),
                0.50,
            )

            dynamic["rating_sensitivity"] = (
                smooth_trait_update(
                    current_value=(
                        old_rating_sensitivity
                    ),
                    target_value=rating_signal,
                )
            )

        # ==================================================
        # 3. 动态更新新颖偏好 novelty_preference
        # ==================================================
        # 点击原始likes之外的类型，
        # 说明用户愿意尝试新内容，
        # novelty_preference向1靠近。
        #
        # 只点击原本就喜欢的类型，
        # novelty_preference向0靠近。
        novelty_target = (
            1.0
            if novel_click
            else 0.0
        )

        old_novelty_preference = to_float(
            dynamic.get(
                "novelty_preference"
            ),
            0.50,
        )

        dynamic["novelty_preference"] = (
            smooth_trait_update(
                current_value=(
                    old_novelty_preference
                ),
                target_value=novelty_target,
            )
        )

        # ==================================================
        # 4. 动态更新重复厌恶 repeat_aversion
        # ==================================================
        # 先取得上一次点击电影的类型。
        previous_clicked_labels = (
            get_last_clicked_labels(
                dynamic
            )
        )

        # 计算本次点击与上次点击的类型相似度。
        similarity = label_similarity(
            previous_clicked_labels,
            selected_labels,
        )

        # 第一次点击没有上一次点击，
        # 因此不更新repeat_aversion。
        if similarity is not None:
            # 相似度越高，用户仍然愿意点击，
            # 说明用户越不讨厌重复内容。
            #
            # 完全相同：
            # similarity = 1
            # repeat_target = 0
            #
            # 完全不同：
            # similarity = 0
            # repeat_target = 1
            repeat_target = (
                    1.0 - similarity
            )

            old_repeat_aversion = to_float(
                dynamic.get(
                    "repeat_aversion"
                ),
                0.50,
            )

            dynamic["repeat_aversion"] = (
                smooth_trait_update(
                    current_value=(
                        old_repeat_aversion
                    ),
                    target_value=repeat_target,
                )
            )

        exploration_target = (
            1.0
            if novel_click
            else 0.0
        )

        old_exploration = to_float(
            dynamic.get(
                "exploration_rate"
            ),
            0.30,
        )

        dynamic["exploration_rate"] = round(
            clamp(
                old_exploration * 0.97
                + exploration_target * 0.03
            ),
            4,
        )

    elif action.action == "next":
        dynamic["next_count"] = (
            int(
                dynamic.get(
                    "next_count",
                    0,
                )
            )
            + 1
        )

        dynamic["consecutive_next"] = (
            int(
                dynamic.get(
                    "consecutive_next",
                    0,
                )
            )
            + 1
        )

        dynamic["satisfaction"] = round(
            clamp(
                to_float(
                    dynamic.get(
                        "satisfaction"
                    ),
                    0.50,
                )
                - 0.04
            ),
            4,
        )

        dynamic["boredom"] = round(
            clamp(
                to_float(
                    dynamic.get("boredom"),
                    0.00,
                )
                + 0.08
            ),
            4,
        )

        dynamic["current_patience"] = round(
            clamp(
                to_float(
                    dynamic.get(
                        "current_patience"
                    ),
                    0.50,
                )
                - 0.05
            ),
            4,
        )

        # 用户仍然选择继续寻找，
        # 探索倾向略微增加。
        dynamic["exploration_rate"] = round(
            clamp(
                to_float(
                    dynamic.get(
                        "exploration_rate"
                    ),
                    0.30,
                )
                + 0.01
            ),
            4,
        )

    elif action.action == "stop":
        dynamic["stop_count"] = (
            int(
                dynamic.get(
                    "stop_count",
                    0,
                )
            )
            + 1
        )

        dynamic["current_patience"] = round(
            clamp(
                to_float(
                    dynamic.get(
                        "current_patience"
                    ),
                    0.50,
                )
                - 0.10
            ),
            4,
        )

        dynamic["boredom"] = round(
            clamp(
                to_float(
                    dynamic.get("boredom"),
                    0.00,
                )
                + 0.04
            ),
            4,
        )

    action_record = {
        "action": action.action,
        "item_id": action.item_id,
        "reason": action.reason,
        "selected_labels": selected_labels,
    }

    recent_actions = dynamic.setdefault(
        "recent_actions",
        [],
    )

    recent_actions.append(action_record)

    # 防止以后Prompt越来越长。
    dynamic["recent_actions"] = (
        recent_actions[-5:]
    )

    after = deepcopy(
        state.dynamic_profile
    )

    update_record = {
        "action": action.action,
        "item_id": action.item_id,
        "selected_labels": selected_labels,
        "changed_fields": build_changed_fields(
            before,
            after,
        ),
        "dynamic_profile_before": before,
        "dynamic_profile_after": after,
    }

    state.action_history.append(
        update_record
    )

    return update_record