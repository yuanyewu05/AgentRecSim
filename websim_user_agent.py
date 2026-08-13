from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dotenv import load_dotenv
from playwright.sync_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


BASE_DIR = Path(__file__).resolve().parent

TASK_PATH = BASE_DIR / "agent_task.yaml"
PROFILE_PATH = BASE_DIR / "profiles.jsonl"
RUNS_DIR = BASE_DIR / "agent_runs"

load_dotenv(BASE_DIR / ".env")


def required_env(name: str) -> str:
    """读取必要的环境变量。"""

    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}，请检查项目根目录中的 .env 文件。"
        )

    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"没有找到配置文件：{path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误：{path}")

    return data


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"没有找到 JSON 文件：{path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON 文件格式错误：{path}")

    return data

def load_profile_from_jsonl(
    path: Path,
    profile_index: int,
) -> dict[str, Any]:
    """读取 profiles.jsonl 中指定编号的用户画像。"""

    if not path.exists():
        raise FileNotFoundError(
            f"没有找到用户画像文件：{path}"
        )

    if profile_index < 0:
        raise ValueError(
            "profile_index 不能小于 0"
        )

    with path.open("r", encoding="utf-8") as file:
        for current_index, line in enumerate(file):
            if current_index != profile_index:
                continue

            line = line.strip()

            if not line:
                raise ValueError(
                    f"第 {profile_index} 行画像为空"
                )

            profile = json.loads(line)

            if not isinstance(profile, dict):
                raise ValueError(
                    f"第 {profile_index} 个画像不是 JSON 对象"
                )

            return profile

    raise IndexError(
        f"profiles.jsonl 中不存在编号为 "
        f"{profile_index} 的用户画像"
    )


def create_decision_agent(profile: dict[str, Any]) -> ChatAgent:
    """创建使用云雾 API 的 CAMEL Agent。"""

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=required_env("YUNWU_MODEL"),
        url=required_env("YUNWU_BASE_URL").rstrip("/"),
        api_key=required_env("YUNWU_API_KEY"),
        model_config_dict={
            "temperature": 0.2,
        },
    )

    system_message = f"""
你是一个正在使用电影推荐系统的虚拟用户 Agent。

你的固定用户画像是：

{json.dumps(profile, ensure_ascii=False, indent=2)}

你需要像这个画像代表的真人一样行动。

每轮你会看到：
1. 当前页面上的推荐卡片；
2. 最近几轮历史选择；
3. “下一页”按钮是否可用。

你的任务：
1. 根据用户画像选择最感兴趣的项目。
2. 保持长期偏好基本稳定。
3. 不要只根据评分选择。
4. 不要总是选择第一张卡片。
5. 已经点击过的项目尽量不要重复点击。
6. 当前页面没有合适项目时，可以选择下一页。
7. 行为必须符合用户画像，而不是充当推荐助手。

你只能返回一个 JSON 对象，不得返回 Markdown，不得在 JSON
前后添加解释。

选择卡片时：

{{
  "action": "click",
  "index": 0,
  "reason": "该项目符合用户对科幻和悬疑内容的偏好"
}}

进入下一页时：

{{
  "action": "next",
  "index": null,
  "reason": "当前项目均不符合用户偏好"
}}

停止实验时：

{{
  "action": "stop",
  "index": null,
  "reason": "已经完成足够的浏览行为"
}}

index 从 0 开始。
""".strip()

    return ChatAgent(
        system_message=system_message,
        model=model,
    )


def get_agent_text(response: Any) -> str:
    """兼容不同 CAMEL 版本的响应格式。"""

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

    raise RuntimeError("CAMEL 返回了响应，但没有找到有效文本。")


def parse_json_object(text: str) -> dict[str, Any]:
    """从模型回复中提取 JSON。"""

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

        if not match:
            raise ValueError(
                f"模型没有返回有效 JSON：\n{text}"
            )

        result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError("模型返回的结果不是 JSON 对象。")

    return result


def read_cards(page: Page) -> list[dict[str, Any]]:
    """读取当前页面上的推荐卡片。"""

    card_locators = page.locator(
        ".card[data-movie-id]"
    )

    cards: list[dict[str, Any]] = []

    for index in range(card_locators.count()):
        card = card_locators.nth(index)

        title_locator = card.locator(".title")
        description_locator = card.locator(".desc")
        rating_locator = card.locator(
            ".rating-group .metric-value"
        )
        heat_locator = card.locator(
            ".heat-group .metric-value"
        )

        title = (
            title_locator.inner_text().strip()
            if title_locator.count() > 0
            else ""
        )

        description = (
            description_locator.inner_text().strip()
            if description_locator.count() > 0
            else ""
        )

        rating = (
            rating_locator.inner_text().strip()
            if rating_locator.count() > 0
            else ""
        )

        heat = (
            heat_locator.inner_text().strip()
            if heat_locator.count() > 0
            else ""
        )

        cards.append(
            {
                "index": index,
                "item_id": (
                    card.get_attribute("data-movie-id")
                    or ""
                ),
                "title": title,
                "description": description,
                "rating": rating,
                "heat": heat,
            }
        )

    return cards


def decide_action(
    agent: ChatAgent,
    cards: list[dict[str, Any]],
    history: list[dict[str, Any]],
    next_enabled: bool,
    step: int,
) -> dict[str, Any]:
    """让 CAMEL Agent 对当前推荐页面作出决策。"""

    observation = {
        "step": step,
        "visible_cards": cards,
        "next_enabled": next_enabled,
        "recent_history": history[-6:],
    }

    prompt = f"""
这是第 {step} 轮推荐页面的 observation：

{json.dumps(observation, ensure_ascii=False, indent=2)}

请根据固定用户画像作出本轮行为决策。

只能返回规定格式的 JSON。
""".strip()

    response = agent.step(prompt)
    raw_response = get_agent_text(response)

    decision = parse_json_object(raw_response)

    action = str(
        decision.get("action", "")
    ).strip().lower()

    index = decision.get("index")
    reason = str(
        decision.get("reason", "")
    ).strip()

    if action not in {"click", "next", "stop"}:
        raise ValueError(
            f"Agent 返回了无效 action：{action}"
        )

    if action == "click":
        if not isinstance(index, int):
            raise ValueError(
                "click 行为的 index 必须是整数。"
            )

        if index < 0 or index >= len(cards):
            raise ValueError(
                f"Agent 返回的 index={index} 无效，"
                f"当前共有 {len(cards)} 张卡片。"
            )

    if action == "next" and not next_enabled:
        # 下一页不可用时，退回点击第一张卡片。
        action = "click"
        index = 0
        reason = (
            "Agent 请求下一页，但下一页当前不可用，"
            "因此回退选择第一张卡片。"
        )

    return {
        "action": action,
        "index": index,
        "reason": reason,
        "raw_response": raw_response,
    }


def wait_for_cards(page: Page) -> None:
    """等待页面加载出推荐卡片。"""

    page.wait_for_function(
        """
        () => document.querySelectorAll(
            ".card[data-movie-id]"
        ).length > 0
        """,
        timeout=15000,
    )


def wait_for_cards_to_change(
    page: Page,
    previous_ids: list[str],
) -> None:
    """等待推荐页面发生变化。"""

    try:
        page.wait_for_function(
            """
            previousIds => {
                const currentIds = Array.from(
                    document.querySelectorAll(
                        ".card[data-movie-id]"
                    )
                ).map(
                    card => card.dataset.movieId || ""
                );

                return JSON.stringify(currentIds)
                    !== JSON.stringify(previousIds);
            }
            """,
            arg=previous_ids,
            timeout=15000,
        )

    except PlaywrightTimeoutError:
        status_text = page.locator(
            "#statusText"
        ).inner_text()

        raise RuntimeError(
            "点击后推荐卡片没有发生变化。"
            f"当前页面状态：{status_text}"
        )


def save_memory(
    memory_path: Path,
    memory: dict[str, Any],
) -> None:
    memory_path.write_text(
        json.dumps(
            memory,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run() -> None:
    parser = argparse.ArgumentParser(
        description="运行 WebSim 虚拟用户 Agent"
    )

    parser.add_argument(
        "--profile-index",
        type=int,
        default=0,
        help="profiles.jsonl 中的用户编号，从 0 开始",
    )

    parser.add_argument(
        "--profiles",
        type=str,
        default="profiles.jsonl",
        help="批量用户画像文件路径",
    )

    args = parser.parse_args()

    task = load_yaml(TASK_PATH)

    profiles_path = BASE_DIR / args.profiles

    profile = load_profile_from_jsonl(
        path=profiles_path,
        profile_index=args.profile_index,
    )

    agent_id = str(
        profile.get(
            "agent_id",
            f"agent_{args.profile_index:06d}",
        )
    )

    websim_url = str(
        task.get(
            "websim_url",
            "http://127.0.0.1:19001/",
        )
    )

    dataset = str(
        task.get("dataset", "ml1m")
    )

    model_name = str(
        task.get("model", "poprec")
    )

    track = int(
        task.get("track", 5)
    )

    headless = bool(
        task.get("headless", False)
    )

    slow_mo_ms = int(
        task.get("slow_mo_ms", 300)
    )

    delay_min = int(
        task.get("human_delay_min_ms", 1000)
    )

    delay_max = int(
        task.get("human_delay_max_ms", 2500)
    )

    if track <= 0:
        raise ValueError("track 必须大于 0。")

    if delay_min > delay_max:
        delay_min, delay_max = delay_max, delay_min

    agent = create_decision_agent(profile)

    run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + agent_id
    )

    run_dir = RUNS_DIR / run_id
    screenshot_dir = run_dir / "screenshots"

    screenshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    memory_path = run_dir / "memory.json"
    summary_path = run_dir / "summary.txt"

    memory: dict[str, Any] = {
        "run_id": run_id,
        "profile": profile,
        "task": task,
        "started_at": datetime.now().isoformat(),
        "steps": [],
    }

    print("=" * 60)
    print("WebSim 最小虚拟用户 Agent")
    print(f"用户编号：{agent_id}")
    print(f"用户群体：{profile.get('group', 'Unknown')}")
    print(f"喜欢：{profile.get('likes', [])}")
    print(f"讨厌：{profile.get('dislikes', [])}")
    print(f"探索率：{profile.get('exploration_rate', 0)}")
    print(f"数据集：{dataset}")
    print(f"推荐模型：{model_name}")
    print(f"轨迹长度：{track}")
    print(f"输出目录：{run_dir}")
    print("=" * 60)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=headless,
            slow_mo=slow_mo_ms,
        )

        context = browser.new_context(
            viewport={
                "width": 1440,
                "height": 1000,
            }
        )

        page = context.new_page()

        try:
            print("\n正在打开 WebSim……")

            page.goto(
                websim_url,
                wait_until="networkidle",
                timeout=30000,
            )

            wait_for_cards(page)

            # 选择数据集。
            page.locator(
                "#datasetSelect"
            ).select_option(dataset)

            page.wait_for_timeout(800)

            # 选择推荐模型。
            page.locator(
                "#algoSelect"
            ).select_option(model_name)

            page.wait_for_timeout(800)

            # 随机重置。
            page.locator(
                "#resetBtn"
            ).click()

            wait_for_cards(page)
            page.wait_for_timeout(800)

            for step in range(1, track + 1):
                cards = read_cards(page)

                if not cards:
                    print("当前页面没有推荐卡片，实验停止。")
                    break

                next_enabled = not page.locator(
                    "#nextBtn"
                ).is_disabled()

                screenshot_path = (
                    screenshot_dir
                    / f"step_{step:03d}.png"
                )

                page.screenshot(
                    path=str(screenshot_path),
                    full_page=True,
                )

                print(f"\n第 {step} 轮观察：")

                for card in cards:
                    print(
                        f"[{card['index']}] "
                        f"{card['title']} "
                        f"| 评分={card['rating']} "
                        f"| 热度={card['heat']}"
                    )

                decision = decide_action(
                    agent=agent,
                    cards=cards,
                    history=memory["steps"],
                    next_enabled=next_enabled,
                    step=step,
                )

                print(
                    f"Agent 行为：{decision['action']} "
                    f"index={decision['index']}"
                )

                print(
                    f"Agent 理由：{decision['reason']}"
                )

                record: dict[str, Any] = {
                    "step": step,
                    "timestamp": datetime.now().isoformat(),
                    "observation": {
                        "cards": cards,
                        "next_enabled": next_enabled,
                        "screenshot": str(screenshot_path),
                    },
                    "decision": decision,
                }

                previous_ids = [
                    str(card["item_id"])
                    for card in cards
                ]

                action = decision["action"]

                if action == "stop":
                    record["result"] = {
                        "status": "agent_stopped"
                    }

                    memory["steps"].append(record)
                    save_memory(memory_path, memory)
                    break

                if action == "next":
                    page.locator(
                        "#nextBtn"
                    ).click()

                    wait_for_cards_to_change(
                        page,
                        previous_ids,
                    )

                    record["result"] = {
                        "status": "next_page_clicked"
                    }

                elif action == "click":
                    selected_index = int(
                        decision["index"]
                    )

                    selected_card = cards[
                        selected_index
                    ]

                    page.locator(
                        ".card[data-movie-id]"
                    ).nth(
                        selected_index
                    ).locator(
                        ".card-main"
                    ).click()

                    wait_for_cards_to_change(
                        page,
                        previous_ids,
                    )

                    record["result"] = {
                        "status": "card_clicked",
                        "selected_card": selected_card,
                    }

                memory["steps"].append(record)
                save_memory(memory_path, memory)

                # 模拟真人的阅读和反应时间。
                page.wait_for_timeout(
                    random.randint(
                        delay_min,
                        delay_max,
                    )
                )

        finally:
            memory["finished_at"] = (
                datetime.now().isoformat()
            )

            save_memory(
                memory_path,
                memory,
            )

            summary_lines = [
                f"run_id={run_id}",
                f"profile={args.profile_index + 1}",
                f"dataset={dataset}",
                f"model={model_name}",
                f"track_target={track}",
                f"completed_steps={len(memory['steps'])}",
                f"memory_file={memory_path}",
                f"screenshot_dir={screenshot_dir}",
            ]

            summary_path.write_text(
                "\n".join(summary_lines),
                encoding="utf-8",
            )

            print("\n实验运行结束。")
            print(f"memory：{memory_path}")
            print(f"summary：{summary_path}")
            print(f"截图目录：{screenshot_dir}")

            page.wait_for_timeout(1500)

            context.close()
            browser.close()


if __name__ == "__main__":
    try:
        run()

    except Exception as error:
        print("\nAgent 运行失败：")
        print(
            f"{type(error).__name__}: {error}"
        )

        raise