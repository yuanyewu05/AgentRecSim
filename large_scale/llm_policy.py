from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dotenv import load_dotenv

from large_scale.websim_env import AgentAction, AgentState

from large_scale.genre_utils import (
    normalize_labels as normalize_genre_labels,
)


# 项目根目录：
# C:\Users\wyy05\Desktop\D8EAX
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def required_env(name: str) -> str:
    """读取必要的环境变量。"""

    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"缺少环境变量{name}，"
            f"请检查项目根目录中的.env文件。"
        )

    return value


def get_agent_text(response: Any) -> str:
    """读取不同CAMEL版本返回的文本内容。"""

    messages = getattr(response, "msgs", None)

    if messages:
        content = getattr(messages[0], "content", None)

        if content:
            return str(content).strip()

    message = getattr(response, "msg", None)

    if message:
        content = getattr(message, "content", None)

        if content:
            return str(content).strip()

    raise RuntimeError(
        "CAMEL返回了响应，但没有找到有效文本。"
    )


def parse_json_object(text: str) -> dict[str, Any]:
    """从大模型回复中提取JSON对象。"""

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        cleaned = re.sub(
            r"```$",
            "",
            cleaned,
        ).strip()

    try:
        result = json.loads(cleaned)

    except json.JSONDecodeError:
        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if match is None:
            raise ValueError(
                f"大模型没有返回有效JSON：\n{text}"
            )

        result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError(
            "大模型返回的结果不是JSON对象。"
        )

    return result


class LLMPolicy:
    """使用CAMEL和云雾大模型进行决策。

    每次API调用都会创建一个全新的CAMEL ChatAgent，
    不复用上一名虚拟用户使用过的本地对话对象。
    Semaphore只负责限制同时发出的API请求数量。
    """

    def __init__(
            self,
            max_concurrency: int = 4,
            max_retries: int = 3,
            request_timeout: float = 120.0,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError(
                "max_concurrency必须大于0"
            )

        if max_retries <= 0:
            raise ValueError(
                "max_retries必须大于0"
            )

        if request_timeout <= 0:
            raise ValueError(
                "request_timeout必须大于0"
            )

        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.request_timeout = float(
            request_timeout
        )

        # 只限制API请求的真实并发数。
        # 这里不保存、也不复用任何ChatAgent对象。
        self.request_semaphore = asyncio.Semaphore(
            max_concurrency
        )

    def _create_worker(self) -> ChatAgent:
        """为单次API调用创建一个全新的CAMEL Agent。"""

        model = ModelFactory.create(
            model_platform=(
                ModelPlatformType.OPENAI_COMPATIBLE_MODEL
            ),
            model_type=required_env(
                "YUNWU_MODEL"
            ),
            url=required_env(
                "YUNWU_BASE_URL"
            ).rstrip("/"),
            api_key=required_env(
                "YUNWU_API_KEY"
            ),
            model_config_dict={
                "temperature": 0.2,
            },
        )

        system_message = """
你是WebSim电影推荐实验中的虚拟用户决策Agent。

每次请求都会提供：
1. 本次请求唯一的request_id；
2. 当前虚拟用户的完整画像；
3. 当前页面可见的电影卡片；
4. 该用户最近的点击历史；
5. 下一页是否可用。

你必须严格按照本次请求中的用户画像进行决策。
不同请求可能属于不同虚拟用户，不得混淆用户身份。
request_id只用于技术校验，不得参与推荐决策。
返回JSON时必须原样返回本次请求的request_id。

click不要求电影完美符合用户的全部偏好。
只要当前页面存在一部具有明确吸引点、
整体符合程度为正、且没有明显触犯dislikes的电影，
真实用户就可能选择click。

只有当当前所有电影都缺乏有效吸引点时，
才应选择next。
不要把next作为不确定情况下的默认动作。

你只能返回一个JSON对象，不得使用Markdown，
也不得在JSON前后添加其他解释。

选择电影时：

{
  "request_id": "原样返回本次请求的request_id",
  "action": "click",
  "index": 0,
  "reason": "这部电影符合该用户的科幻和动作偏好"
}

进入下一页时：

{
  "request_id": "原样返回本次请求的request_id",
  "action": "next",
  "index": null,
  "reason": "当前电影均不符合该用户偏好"
}

停止时：

{
  "request_id": "原样返回本次请求的request_id",
  "action": "stop",
  "index": null,
  "reason": "该用户已经完成足够的浏览行为"
}

index从0开始。
""".strip()

        return ChatAgent(
            system_message=system_message,
            model=model,
        )

    @staticmethod
    def _format_cards(
        cards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """把WebSim卡片整理为大模型需要的字段。"""

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
                    "title": card.get(
                        "title",
                        "",
                    ),
                    "description": card.get(
                        "description",
                        card.get("desc", ""),
                    ),

                    # 把电影类型传给云雾大模型，并统一为中文。
                    "genres": normalize_genre_labels(
                        card.get("genres") or []
                    ),

                    "rating": card.get(
                        "rating_value",
                        card.get("rating", ""),
                    ),
                    "heat": card.get(
                        "rating_count",
                        card.get("heat", ""),
                    ),
                }
            )

        return visible_cards

    async def decide(
        self,
        state: AgentState,
        cards: list[dict[str, Any]],
        next_enabled: bool,
    ) -> AgentAction:
        """异步调用CAMEL和云雾API作出一次决策。"""

        if not cards:
            return AgentAction(
                action="stop",
                reason="当前没有可选电影",
            )

        visible_cards = self._format_cards(cards)

        # 为本次逻辑决策生成唯一编号。
        # 编号只用于核对响应归属，不参与推荐判断。
        request_id = (
            f"{state.agent_id}"
            f"-step-{state.step + 1}"
            f"-{uuid.uuid4().hex}"
        )

        # 固定画像保持不变，并去掉程序内部字段。
        base_profile = {
            key: value
            for key, value in state.profile.items()
            if key != "_profile_index"
        }

        observation = {
            "request_id": request_id,
            "agent_id": state.agent_id,
            "profile_index": state.profile_index,
            "step": state.step + 1,

            # 用户最初生成的固定画像，实验过程中不修改。
            "base_profile": base_profile,

            # 根据前几轮行为实时变化的当前画像。
            "dynamic_profile": state.dynamic_profile,

            # 当前页面上实际可见的电影。
            "visible_cards": visible_cards,

            # 该用户最近点击过的电影。
            "recent_clicked_item_ids": state.history[-10:],

            # 最近几轮做过的动作。
            "recent_actions": state.dynamic_profile.get(
                "recent_actions",
                [],
            )[-5:],

            # 当前是否还能翻到下一页。
            "next_enabled": next_enabled,
        }

        prompt = f"""
        你正在模拟电影推荐系统中的一个真实虚拟用户。

        下面是该用户当前完整的 observation：

        {json.dumps(
            observation,
            ensure_ascii=False,
            indent=2,
        )}

        请根据固定画像、动态画像、点击历史和当前电影，
        决定本轮执行 click、next 或 stop。

        【固定画像说明】

        base_profile 表示该用户长期、稳定的身份和基础偏好，
        包括：

        - age：年龄；
        - group：用户群体；
        - likes：长期喜欢的内容；
        - dislikes：长期厌恶的内容；
        - exploration_rate：基础探索倾向；
        - popularity_bias：基础热门内容偏好；
        - rating_sensitivity：基础评分敏感度；
        - novelty_preference：基础新颖内容偏好；
        - patience：基础耐心；
        - repeat_aversion：基础重复厌恶程度。

        固定画像是用户的初始人格，不代表当前状态完全没有变化。

        【动态画像说明】

        dynamic_profile 表示该用户经过前几轮交互后，
        当前实时变化的兴趣和心理状态。

        请重点参考：

        - interest_weights：
          当前对各种内容的兴趣强度，数值越高越喜欢；

        - exploration_rate：
          当前尝试原有偏好之外内容的倾向；

        - popularity_bias：
          当前对热门内容的重视程度；

        - rating_sensitivity：
          当前对高评分内容的重视程度；

        - novelty_preference：
          当前对新颖、未接触内容的偏好；

        - repeat_aversion：
          当前对重复、相似内容的厌恶程度；

        - current_patience：
          当前继续浏览的耐心，越低越可能停止；

        - satisfaction：
          当前满意度，越高说明前面的推荐体验越好；

        - boredom：
          当前无聊程度，越高越可能翻页或停止；

        - consecutive_next：
          已经连续翻页的次数，次数较多时应考虑耐心下降；

        - recent_actions：
          用户最近执行过的行为。

        动态画像比固定画像更能反映用户此时此刻的状态，
        但两者都必须参考。

                【动作规则】

        1. click

        click表示用户愿意进一步查看或观看某部电影，
        不要求该电影完美符合用户的全部偏好。

        当某部电影满足以下任意一种情况，
        并且没有明显触犯用户的dislikes时，
        应优先考虑click：

        - 电影类型命中用户的likes；
        - 电影类型命中当前较高的interest_weights；
        - 电影包含至少一个用户明确喜欢的核心类型；
        - 电影评分较高，并且用户当前rating_sensitivity较高；
        - 电影热度较高，并且用户当前popularity_bias较高；
        - 电影具有新的类型，并且当前novelty_preference
          或exploration_rate较高；
        - 电影与用户最近点击的内容相似，
          并且当前repeat_aversion较低；
        - 电影同时具有两个或以上中等程度的正向信号。

        一个明确的核心类型匹配，
        或两个中等程度的正向信号，
        就可以构成合理的click理由。

        不要要求电影同时满足类型、评分、热度、
        新颖性等全部条件。

        如果当前有多部可以点击的电影，
        选择总体吸引力最高的一部。

        只有电影明显触犯dislikes，
        或几乎不存在任何正向匹配信号时，
        才不应点击。

        选择click时：

        - index必须是当前visible_cards中的整数索引；
        - index从0开始；
        - 不能选择不存在的索引。

        2. next

        只有在以下条件同时成立时，
        才选择next：

        - 当前所有电影都没有明确的核心偏好匹配；
        - 当前所有电影都没有两个以上的中等正向信号；
        - 当前不存在一部总体吸引力为正的电影；
        - next_enabled为true。

        不要把next作为不确定时的默认动作。

        如果某一部电影虽然不是完美匹配，
        但具有一个明确吸引点且没有触犯dislikes，
        应优先选择click，而不是next。

        选择next时：

        - index必须为null。

        3. stop

        出现以下情况时可以选择 stop：

        - current_patience 已经很低；
        - boredom 已经很高；
        - 连续翻页次数较多；
        - 当前没有合适内容；
        - 用户已经不愿意继续浏览；
        - next_enabled 为 false，并且当前也没有值得点击的电影。

        选择 stop 时：

        - index 必须为 null。

        【重要限制】

        - request_id必须原样返回，不允许修改；
        - request_id不得参与用户偏好或推荐判断；
        - 只能返回 click、next 或 stop；
        - next_enabled 为 false 时，禁止返回 next；
        - 不要返回 Markdown；
        - 不要返回代码块；
        - 不要返回解释性前缀；
        - 只能返回一个合法 JSON 对象。

        严格使用以下格式：

        {{
          "request_id": "{request_id}",
          "action": "click 或 next 或 stop",
          "index": 0 或 null,
          "reason": "用一句中文说明为什么做出这个决定"
        }}
                """.strip()

        last_error: Exception | None = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            # 每次尝试都重新创建ChatAgent。
            # 即使发生重试，也不会复用上一次失败请求的对象。
            worker: ChatAgent | None = None

            try:
                # 同一个API Key允许并发使用；Semaphore只限制数量。
                async with self.request_semaphore:
                    worker = self._create_worker()





                    response = await asyncio.wait_for(
                        worker.astep(prompt),
                        timeout=self.request_timeout,
                    )

    



                raw_response = get_agent_text(
                    response
                )

                decision = parse_json_object(
                    raw_response
                )

                returned_request_id = str(
                    decision.get(
                        "request_id",
                        "",
                    )
                ).strip()

                if returned_request_id != request_id:
                    raise ValueError(
                        "响应request_id不匹配："
                        f"期望{request_id}，"
                        f"实际{returned_request_id or '空'}"
                    )


                action = str(
                    decision.get(
                        "action",
                        "",
                    )
                ).strip().lower()

                index = decision.get("index")

                reason = str(
                    decision.get(
                        "reason",
                        "",
                    )
                ).strip()

                if action not in {
                    "click",
                    "next",
                    "stop",
                }:
                    raise ValueError(
                        f"无效action：{action}"
                    )

                if action == "click":
                    # 部分模型可能把0返回成字符串"0"。
                    if (
                        isinstance(index, str)
                        and index.isdigit()
                    ):
                        index = int(index)

                    if not isinstance(index, int):
                        raise ValueError(
                            "click动作的index必须是整数"
                        )

                    if (
                        index < 0
                        or index >= len(visible_cards)
                    ):
                        raise ValueError(
                            f"无效index={index}，"
                            f"当前有{len(visible_cards)}张卡片"
                        )

                    selected_item_id = (
                        visible_cards[index][
                            "item_id"
                        ]
                    )

                    return AgentAction(
                        action="click",
                        item_id=selected_item_id,
                        reason=reason,
                    )

                if action == "next":
                    if next_enabled:
                        return AgentAction(
                            action="next",
                            reason=reason,
                        )

                    # 下一页不可用时，与原浏览器模式一样，
                    # 回退点击第一张卡片。
                    return AgentAction(
                        action="click",
                        item_id=(
                            visible_cards[0][
                                "item_id"
                            ]
                        ),
                        reason=(
                            "大模型请求下一页，"
                            "但下一页当前不可用，"
                            "因此回退选择第一张卡片。"
                        ),
                    )

                return AgentAction(
                    action="stop",
                    reason=reason,
                )

            except Exception as error:
                last_error = error

                if attempt >= self.max_retries:
                    break

                wait_seconds = min(
                    2 ** (attempt - 1),
                    8,
                )

                print(
                    f"Agent {state.agent_id} "
                    f"第{attempt}次大模型请求失败："
                    f"{type(error).__name__}: {error}"
                )

                print(
                    f"{wait_seconds}秒后重试……"
                )

                # 重试等待发生在Semaphore外，
                # 不占用宝贵的API并发名额。
                await asyncio.sleep(
                    wait_seconds
                )

            finally:
                # 本次创建的ChatAgent到这里结束生命周期，
                # 不放回任何工作池，也不交给其他虚拟用户。
                if worker is not None:
                    try:
                        worker.reset()
                    except Exception:
                        pass

        raise RuntimeError(
            f"Agent {state.agent_id} "
            f"连续{self.max_retries}次"
            f"调用大模型失败：{last_error}"
        )
