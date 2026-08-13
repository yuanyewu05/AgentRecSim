from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# 直接使用现有 app.py 中已经跑通的推荐系统核心。
# 不启动浏览器，也不依赖 Flask session。
from app import PAGE_SIZE, TOPK, get_recommender, unique_keep_order

from large_scale.genre_utils import (
    normalize_labels as normalize_genre_labels,
)


ActionType = Literal["click", "next", "stop"]


@dataclass(slots=True)
class AgentState:
    """单个虚拟用户的独立状态。"""

    agent_id: str
    profile_index: int
    profile: dict[str, Any]
    seed: int

    # 用户已经点击过的物品。
    history: list[str] = field(default_factory=list)

    # 当前推荐模型产生的完整推荐列表。
    rec_ids: list[str] = field(default_factory=list)

    # 当前推荐列表翻到了第几页，从0开始。
    page_index: int = 0

    # random表示初始随机页，recommend表示模型推荐页。
    mode: str = "random"

    # Agent是否已经停止。
    stopped: bool = False

    # 已经执行的交互轮数。
    step: int = 0

    # 运行过程中实时变化的用户画像。
    dynamic_profile: dict[str, Any] = field(default_factory=dict)

    # 保存每一轮行为和画像变化。
    action_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class AgentAction:
    """Agent向WebSim环境提交的动作。"""

    action: ActionType
    item_id: str | None = None
    reason: str = ""


class WebSimEnvironment:
    """不使用Playwright的WebSim批量仿真环境。"""

    def __init__(
        self,
        dataset: str = "ml1m",
        model: str = "poprec",
    ) -> None:
        self.dataset = dataset

        # get_recommender会复用你现有的推荐系统。
        self.recommender = get_recommender(dataset)

        # 防止模型名称不存在。
        self.model = self.recommender.normalize_model_name(model)

    def _cards_from_ids(
        self,
        item_ids: list[str],
    ) -> list[dict[str, Any]]:
        """把物品ID转换成电影卡片信息。"""

        return [
            self.recommender.movie_card(str(item_id))
            for item_id in item_ids
        ]

    def rerank_by_preferences(
        self,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        """
        根据当前Agent已经学习到的interest_weights，
        对现有推荐候选列表state.rec_ids重新排序。

        注意：
        这个函数不重新调用推荐算法。
        它只调整推荐算法已经生成的候选电影顺序。
        """

        # 只有正式推荐列表才需要重排序。
        if (
            state.mode != "recommend"
            or not state.rec_ids
        ):
            return self.observe(state)

        # 读取当前Agent已经学习到的兴趣权重。
        raw_interests = state.dynamic_profile.get(
            "interest_weights",
            {},
        )

        if not isinstance(raw_interests, dict):
            return self.observe(state)

        interests: dict[str, float] = {}

        # 把所有权重安全地转换为float。
        for label, value in raw_interests.items():
            try:
                weight = float(value)
            except (TypeError, ValueError):
                continue

            normalized_labels = normalize_genre_labels(
                [label]
            )

            if not normalized_labels:
                continue

            normalized_label = normalized_labels[0]

            # 如果旧数据同时存在“动作”和“Action”，
            # 统一后保留其中较高的权重。
            interests[normalized_label] = max(
                interests.get(normalized_label, 0.0),
                weight,
            )

        if not interests:
            return self.observe(state)

        def get_movie_labels(
                item_id: str,
        ) -> list[str]:
            """
            读取电影genres，并统一转换成中文标准标签。

            例如：
                Action          -> 动作
                Adventure       -> 冒险
                Science Fiction -> 科幻
            """

            card = self.recommender.movie_card(
                str(item_id)
            )

            return normalize_genre_labels(
                card.get("genres") or []
            )

        def preference_score(
            item_id: str,
        ) -> float:
            """
            计算一部电影与当前动态兴趣画像的匹配分。

            已经学习过的类型使用实际权重；
            尚未学习过的类型使用中性值0.50。
            """

            labels = get_movie_labels(item_id)

            if not labels:
                return 0.50

            label_weights = [
                interests.get(label, 0.50)
                for label in labels
            ]

            return (
                sum(label_weights)
                / len(label_weights)
            )

        # enumerate保存电影原来的位置。
        # 当两部电影偏好分相同时，维持原推荐顺序。
        indexed_rec_ids = list(
            enumerate(state.rec_ids)
        )

        indexed_rec_ids.sort(
            key=lambda pair: (
                -preference_score(pair[1]),
                pair[0],
            )
        )

        state.rec_ids = [
            item_id
            for _, item_id in indexed_rec_ids
        ]

        # 重排序以后从推荐列表第一页开始展示。
        state.page_index = 0

        return self.observe(state)

    def _current_page_ids(
        self,
        state: AgentState,
    ) -> list[str]:
        """从完整推荐列表中取出当前页的4张卡片。"""

        start = state.page_index * PAGE_SIZE
        end = start + PAGE_SIZE

        return state.rec_ids[start:end]

    def reset(
        self,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        """重置一个Agent，并返回初始随机推荐卡片。"""

        item_ids = self.recommender.random_movie_ids(
            PAGE_SIZE,
            avoid_last_same=False,
        )

        state.history.clear()
        state.rec_ids = [
            str(item_id)
            for item_id in item_ids
        ]
        state.page_index = 0
        state.mode = "random"
        state.stopped = False
        state.step = 0

        return self._cards_from_ids(state.rec_ids)

    def observe(
        self,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        """读取Agent当前能够看到的推荐卡片。"""

        if state.stopped:
            return []

        return self._cards_from_ids(
            self._current_page_ids(state)
        )

    def step(
        self,
        state: AgentState,
        action: AgentAction,
    ) -> list[dict[str, Any]]:
        """执行一个Agent动作，并返回下一轮推荐卡片。"""

        if state.stopped:
            return []

        state.step += 1

        # 停止交互。
        if action.action == "stop":
            state.stopped = True
            return []

        # 点击某一张推荐卡片。
        if action.action == "click":
            if not action.item_id:
                raise ValueError(
                    "click动作必须提供item_id"
                )

            item_id = str(action.item_id)

            state.history.append(item_id)

            # 与现有app.py保持一致，只保留最近20次点击。
            state.history = state.history[-20:]

            rec_ids = self.recommender.recommend_ids(
                state.history,
                model_name=self.model,
                topk=TOPK,
            )

            state.rec_ids = unique_keep_order(
                [
                    str(rec_id)
                    for rec_id in rec_ids
                ]
            )

            state.page_index = 0
            state.mode = "recommend"

            return self.observe(state)

        # 翻到下一页。
        if action.action == "next":
            if state.mode == "random":
                # 初始随机页本身没有完整推荐列表，
                # 因此重新获取一组随机卡片。
                item_ids = self.recommender.random_movie_ids(
                    PAGE_SIZE,
                    avoid_last_same=False,
                )

                state.rec_ids = [
                    str(item_id)
                    for item_id in item_ids
                ]
                state.page_index = 0

                return self.observe(state)

            state.page_index += 1

            page_ids = self._current_page_ids(state)

            # 已经没有下一页。
            if not page_ids:
                state.stopped = True
                return []

            return self._cards_from_ids(page_ids)

        raise ValueError(
            f"不支持的动作：{action.action}"
        )