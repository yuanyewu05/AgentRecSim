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


# 项目根目录。
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(
    BASE_DIR / ".env",
    override=True,
)


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
    """使用CAMEL和阿里云百炼大模型进行决策。

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
                "DASHSCOPE_MODEL"
            ),
            url=required_env(
                "DASHSCOPE_BASE_URL"
            ).rstrip("/"),
            api_key=required_env(
                "DASHSCOPE_API_KEY"
            ),
            model_config_dict={
                # 降低随机性，让模型更稳定地返回JSON。
                "temperature": 0.2,

                # 关闭千问思考模式。
                # Agent决策只需要最终JSON，不需要输出思考过程。
                "extra_body": {
                    "enable_thinking": False,
                },
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

click不要求电影完美符合用户的全部偏好，
但必须同时存在至少两个相互独立、明确且足够强的内容证据。
其中至少一个证据必须来自核心兴趣或当前高权重兴趣；
另一个证据必须来自不同的具体内容维度。
评分、热度和“没有命中dislikes”只能增强内容证据，
不能单独充当第二个证据。
同一偏好下的多个相近类型标签只能合并算作一个证据。
仅仅命中一个likes标签、总体符合程度为正、评分较高、
热度较高，或者没有触犯dislikes，都不足以选择click。
likes表示长期偏好，不表示用户看到同类内容就一定会点击。
连续出现相似内容或最近已经点击同类内容时，
应结合repeat_aversion提高再次点击的门槛。
不同电影、不同IP或细分风格不同，并不自动表示没有重复感。
最近连续click两次后，必须有明显的新内容价值才可再次click；
最近连续click三次或更多时，若next可用，
除非出现与画像高度契合且明显稀缺的内容，否则应选择next。

next是正常的探索行为。
当页面中没有电影达到该用户的明确点击标准，
但用户仍愿意继续浏览时，应选择next；
不要求所有电影都完全不符合偏好。

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
            # 百炼官方规格：qwen3.7-flash上下文长度为1M。
            # 显式设置后，CAMEL不会再把它当作未知模型
            # 并使用999_999_999的默认占位值。
            token_limit=1_000_000,
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

                    # 把电影类型传给百炼大模型，并统一为中文。
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
        """异步调用CAMEL和阿里云百炼API作出一次决策。"""

        if not cards:
            return AgentAction(
                action="stop",
                reason="当前没有可选电影",
                intended_to_stop=True,
                intention_reason=(
                    "没有可继续浏览的推荐内容"
                ),
            )

        visible_cards = self._format_cards(cards)

        # 为本次逻辑决策生成唯一编号。
        # 编号只用于核对响应归属，不参与推荐判断。
        request_id = (
            f"{state.agent_id}"
            f"-step-{state.step + 1}"
            # 8位随机后缀已经足够区分并发请求，
            # 同时降低大模型复制长字符串时出错的概率。
            f"-{uuid.uuid4().hex[:8]}"
        )

        # 固定画像保持不变，并去掉程序内部字段。
        base_profile = {
            key: value
            for key, value in state.profile.items()
            if key != "_profile_index"
        }

        # 当前仿真总分钟数。
        simulation_minute = int(
            state.simulation_minute
        )

        # 当前是仿真的第几天。
        simulation_day = (
            simulation_minute
            // 1440
            + 1
        )

        # 当前时间在当天对应的分钟数。
        minute_of_day = (
            simulation_minute
            % 1440
        )

        current_hour = (
            minute_of_day
            // 60
        )

        current_minute = (
            minute_of_day
            % 60
        )

        # 方便大模型理解的时间文本。
        current_time = (
            f"day_{simulation_day:02d} "
            f"{current_hour:02d}:"
            f"{current_minute:02d}"
        )

        observation = {
            "request_id": request_id,
            "agent_id": state.agent_id,
            "profile_index": state.profile_index,
            "step": state.step + 1,

            # 当前仿真时间。
            "simulation_minute": (
                simulation_minute
            ),
            "simulation_day": simulation_day,
            "minute_of_day": minute_of_day,
            "current_hour": current_hour,
            "current_time": current_time,

            # 当前 Agent 的生活作息和重要目标。
            "routine_type": state.profile.get(
                "routine_type",
                "unknown",
            ),
            "daily_goals": state.profile.get(
                "daily_goals",
                [],
            ),
            "special_event": state.profile.get(
                "special_event",
                {
                    "event_id": "none",
                    "event_name": "无特殊事件",
                },
            ),

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

                【仿真时间与生活目标】

        observation中的时间字段表示当前仿真时间：

        - current_time：方便阅读的当前时间；
        - minute_of_day：当前时间在当天对应的分钟数；
        - daily_goals：该用户每天需要完成的重要目标；
        - priority：目标优先级，3表示高优先级；
        - tolerance_minutes：允许目标被推迟的分钟数。

        special_event表示当天是否存在暑假、节假日或停电等
        特殊情景。你必须依据其中的applicable和effects理解
        当天作息；已经被事件暂停的目标不会出现在daily_goals中。
        特殊事件只改变当天实际环境，不能凭空添加未提供的影响。

        判断是否应该停止浏览时，必须考虑当前时间是否已经
        进入睡眠、工作、学习或其他高优先级目标的时间范围。

        对于跨越午夜的目标，例如：

        start_minute=1380
        end_minute=420

        表示该目标从23:00持续到第二天07:00。

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

                【停止意图与实际动作必须分开判断】

        你必须先判断用户是否产生了停止意图，
        然后再判断用户最终实际执行的动作。

        intended_to_stop表示用户内心是否认为自己
        现在应该停止，或是否已经产生了停止浏览的计划。

        出现以下情况时，可以将intended_to_stop设为true：

        - 已经进入睡眠、工作、学习等高优先级目标时间；
        - 当前耐心很低；
        - 当前无聊或疲劳程度较高；
        - 已经浏览较长时间；
        - 用户认为继续浏览会影响自己的重要目标。

        intended_to_stop和最终action允许不一致。

        例如，用户知道已经应该睡觉，
        因此intended_to_stop为true，
        但当前推荐内容很有吸引力，
        最终仍然可能执行click或next。

        这种情况表示用户想停止却仍然继续，
        不要为了让两个字段一致而强制选择stop。

        intention_reason需要用一句中文说明用户为什么
        产生或没有产生停止意图。

                【动作决策顺序】

        你必须按照以下顺序进行判断。
        
        第一步：先判断用户是否还愿意继续浏览。
        
        不要先寻找当前页面中有没有一部可以点击的电影。
        
        即使当前页面存在符合偏好的电影，
        真实用户也可能因为浏览时间较长、
        耐心下降、已经获得足够内容、
        感到疲劳或无聊而选择stop。
        
        满意度高不代表用户愿意无限继续使用。
        每一次浏览、点击和翻页都会消耗用户的时间和注意力。
        
        应重点结合：
        
        - current_patience
        - boredom
        - satisfaction
        - step
        - consecutive_next
        - recent_actions
        
        判断当前用户是否仍有继续浏览的意愿。
        
        当出现以下情况时，应认真考虑stop：
        
        - current_patience低于约0.25；
        - boredom高于约0.55；
        - 已经进行了较多轮交互；
        - 用户已经连续浏览了较长时间；
        - 最近多次操作后继续浏览的边际收益已经较低；
        - 当前用户虽然仍能找到感兴趣内容，
          但已经没有明显继续浏览的动力。
        
        stop是正常行为，
        不能只在完全没有内容可看时才stop。
        
        第二步：
        
        只有当判断用户仍然愿意继续浏览时，
        才在click和next之间选择。
        
        1. click
        
        只有当前页面存在具有较强吸引力的电影时，
        才选择click。不要把“可能喜欢”误判为“会立即点击”。
        
        在选择click之前，必须逐项完成以下点击门槛检查：

        - 至少存在两个相互独立的明确内容证据；
        - 第一个证据必须是核心likes或当前高interest_weights
          与电影的具体内容明显匹配；
        - 第二个证据必须来自不同的具体内容维度，例如特殊主题、
          叙事设定、人物关系、新颖体验或与当前状态特别契合；
        - 评分和热度只是辅助信息，不能独立作为第二个证据；
        - 动作加冒险、剧情加文艺等相近类型标签，若都来自同一
          长期偏好，只能合并算作一个兴趣证据；
        - 如果最近已经点击过同类内容，必须根据repeat_aversion
          提高门槛；重复题材本身不能作为第二个证据；
        - dislikes、疲劳、无聊或目标冲突等负向证据，
          会削弱正向证据，不能忽略。

        以下情况都不足以单独构成click理由：

        - 只命中一个likes或普通题材标签；
        - 只有评分高或热度高；
        - 只是没有命中dislikes；
        - 只是总体上“可能会喜欢”；
        - 当前页面中它仅仅比其他电影稍好。

        【连续点击修正规则】

        判断recent_actions末尾连续出现了多少次click：

        - 连续0到1次click：使用上述普通点击门槛；
        - 连续2次click：再次click必须具有明显的新内容价值，
          不能只依靠类型匹配、评分、热度或换了电影/IP；
        - 连续3次或更多click：当next_enabled为true时，
          除非当前电影同时高度契合核心偏好、具有明显稀缺性，
          并且足以克服当前repeat_aversion，否则应选择next；
        - “不同IP”“不同作品”“细分风格略有不同”不能自动证明
          内容不重复，也不能用来绕过连续点击门槛。

        连续点击修正规则不是固定点击配额，仍须结合用户画像；
        它用于表现真实用户注意力消耗、选择性和内容饱和感。

        likes描述的是长期偏好，不代表用户看到同类电影
        就必然点击。若上述点击门槛没有完整满足，
        且用户仍愿意浏览，应选择next。
        
        2. next
        
        用户仍想继续浏览，
        但当前页面没有足够吸引他的电影时，
        选择next。
        
        next是正常的探索行为，
        不需要当前所有电影都完全不符合偏好。
        
        3. stop
        
        用户已经不愿继续浏览时选择stop。
        
        选择stop时，
        即使当前页面仍然有一些匹配内容也是合理的。

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
          "intended_to_stop": true,
          "intention_reason": "用一句中文说明停止意图",
          "action": "click",
          "index": 0,
          "reason": "用一句中文说明最终动作原因"
        }}
                字段要求：

        - intended_to_stop只能是JSON布尔值true或false，
          不能写成字符串"true"或"false"；
        - intention_reason不能为空；
        - action只能是click、next或stop；
        - action为click时，index必须是有效整数；
        - action为next或stop时，index必须是null；
        - 即使intended_to_stop为true，
          action仍然允许是click或next。
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

                # 读取停止意图。
                intended_to_stop = decision.get(
                    "intended_to_stop"
                )

                if not isinstance(
                    intended_to_stop,
                    bool,
                ):
                    raise ValueError(
                        "intended_to_stop必须是"
                        "JSON布尔值true或false"
                    )

                intention_reason = str(
                    decision.get(
                        "intention_reason",
                        "",
                    )
                ).strip()

                if not intention_reason:
                    raise ValueError(
                        "intention_reason不能为空"
                    )

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
                        intended_to_stop=(
                            intended_to_stop
                        ),
                        intention_reason=(
                            intention_reason
                        ),
                    )

                if action == "next":
                    if next_enabled:
                        return AgentAction(
                            action="next",
                            reason=reason,
                            intended_to_stop=(
                                intended_to_stop
                            ),
                            intention_reason=(
                                intention_reason
                            ),
                        )

                    return AgentAction(
                        action="stop",
                        reason=(
                            "用户希望继续查看下一页，"
                            "但当前已经没有更多推荐内容，"
                            "因此结束本次浏览。"
                        ),
                        intended_to_stop=(
                            intended_to_stop
                        ),
                        intention_reason=(
                            intention_reason
                        ),
                    )

                return AgentAction(
                    action="stop",
                    reason=reason,
                    intended_to_stop=(
                        intended_to_stop
                    ),
                    intention_reason=(
                        intention_reason
                    ),
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
