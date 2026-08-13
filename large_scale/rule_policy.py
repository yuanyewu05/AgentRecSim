from __future__ import annotations

import random
import re
from typing import Any

from large_scale.websim_env import AgentAction, AgentState


# 将画像中的中文偏好转换成电影数据中可能出现的英文关键词。
KEYWORD_MAP = {
    "科幻": ["sci-fi", "science fiction"],
    "动作": ["action"],
    "冒险": ["adventure"],
    "悬疑": ["mystery", "thriller"],
    "惊悚": ["thriller", "horror"],
    "恐怖": ["horror"],
    "喜剧": ["comedy"],
    "爱情": ["romance"],
    "剧情": ["drama"],
    "犯罪": ["crime"],
    "动画": ["animation"],
    "家庭": ["family", "children"],
    "奇幻": ["fantasy"],
    "战争": ["war"],
    "纪录片": ["documentary"],
    "音乐": ["musical", "music"],
}


def card_text(card: dict[str, Any]) -> str:
    """把电影卡片中可用的文字拼接起来，便于匹配画像偏好。"""

    values = [
        card.get("title", ""),
        card.get("description", ""),
        card.get("desc", ""),
        card.get("genres", ""),
        card.get("genre", ""),
    ]

    return " ".join(str(value) for value in values).lower()


def expand_keywords(values: list[Any]) -> list[str]:
    """同时保留原关键词和中英文映射关键词。"""

    results: list[str] = []

    for value in values:
        keyword = str(value).strip().lower()

        if not keyword:
            continue

        results.append(keyword)

        for mapped_keyword in KEYWORD_MAP.get(keyword, []):
            results.append(mapped_keyword.lower())

    return results


def parse_number(value: Any) -> float:
    """从评分、热度等字段中提取数字。"""

    if value is None:
        return 0.0

    if isinstance(value, int | float):
        return float(value)

    match = re.search(r"-?\d+(?:\.\d+)?", str(value))

    if match is None:
        return 0.0

    return float(match.group())


class RulePolicy:
    """万级Agent使用的轻量级画像决策策略。"""

    def score_card(
        self,
        profile: dict[str, Any],
        card: dict[str, Any],
    ) -> float:
        text = card_text(card)

        likes = expand_keywords(profile.get("likes", []))
        dislikes = expand_keywords(profile.get("dislikes", []))

        score = 0.0

        # 匹配喜欢的类型，提高得分。
        for keyword in likes:
            if keyword in text:
                score += 2.0

        # 匹配不喜欢的类型，降低得分。
        for keyword in dislikes:
            if keyword in text:
                score -= 3.0

        rating = parse_number(
            card.get("rating_value", card.get("rating"))
        )

        popularity = parse_number(
            card.get(
                "rating_count",
                card.get("popularity", card.get("heat")),
            )
        )

        rating_sensitivity = float(
            profile.get("rating_sensitivity", 0.5) or 0.5
        )

        popularity_bias = float(
            profile.get("popularity_bias", 0.5) or 0.5
        )

        score += rating_sensitivity * rating
        score += popularity_bias * min(popularity / 1000.0, 5.0)

        return score

    def decide(
        self,
        state: AgentState,
        cards: list[dict[str, Any]],
        rng: random.Random,
    ) -> AgentAction:
        """根据画像和当前卡片选择click、next或stop。"""

        if not cards:
            return AgentAction(
                action="stop",
                reason="当前没有可选电影",
            )

        exploration_rate = float(
            state.profile.get("exploration_rate", 0.1) or 0.1
        )

        exploration_rate = max(
            0.0,
            min(1.0, exploration_rate),
        )

        # 按画像探索率随机尝试一张电影。
        if rng.random() < exploration_rate:
            selected_card = rng.choice(cards)

            return AgentAction(
                action="click",
                item_id=str(selected_card["id"]),
                reason="按照用户探索率随机尝试电影",
            )

        scored_cards = [
            (
                self.score_card(state.profile, card),
                card,
            )
            for card in cards
        ]

        best_score, best_card = max(
            scored_cards,
            key=lambda result: result[0],
        )

        # 已进入推荐页，但全部候选匹配度很低时翻页。
        if (
            state.mode == "recommend"
            and best_score <= 0
            and rng.random() < 0.35
        ):
            return AgentAction(
                action="next",
                reason="当前推荐与用户画像匹配度较低",
            )

        return AgentAction(
            action="click",
            item_id=str(best_card["id"]),
            reason=f"当前电影画像匹配得分最高：{best_score:.3f}",
        )