from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from large_scale.risk_profiles import ensure_risk_profile


BASE_DIR = Path(__file__).resolve().parent


COHORTS: dict[str, dict[str, Any]] = {
    "scifi_action": {
        "weight": 0.20,
        "likes": [
            "科幻",
            "动作",
            "悬疑",
            "冒险",
            "反转",
            "快节奏",
        ],
        "dislikes": [
            "纯爱情",
            "儿童向",
            "节奏特别慢",
        ],
        "age_range": (18, 35),
        "exploration": (0.10, 0.30),
        "popularity": (0.40, 0.85),
        "novelty": (0.30, 0.70),
    },
    "thriller_horror": {
        "weight": 0.14,
        "likes": [
            "悬疑",
            "惊悚",
            "恐怖",
            "犯罪",
            "反转",
        ],
        "dislikes": [
            "儿童向",
            "轻松喜剧",
            "纯爱情",
        ],
        "age_range": (18, 45),
        "exploration": (0.15, 0.35),
        "popularity": (0.30, 0.75),
        "novelty": (0.35, 0.80),
    },
    "romance_drama": {
        "weight": 0.17,
        "likes": [
            "爱情",
            "剧情",
            "青春",
            "家庭",
            "情感",
        ],
        "dislikes": [
            "重口味恐怖",
            "纯动作",
            "血腥",
        ],
        "age_range": (18, 55),
        "exploration": (0.08, 0.25),
        "popularity": (0.45, 0.90),
        "novelty": (0.20, 0.60),
    },
    "family_animation": {
        "weight": 0.13,
        "likes": [
            "动画",
            "家庭",
            "喜剧",
            "奇幻",
            "温馨",
        ],
        "dislikes": [
            "血腥",
            "恐怖",
            "沉重犯罪",
        ],
        "age_range": (20, 60),
        "exploration": (0.08, 0.22),
        "popularity": (0.50, 0.95),
        "novelty": (0.15, 0.55),
    },
    "arthouse": {
        "weight": 0.12,
        "likes": [
            "文艺",
            "剧情",
            "历史",
            "传记",
            "慢节奏",
        ],
        "dislikes": [
            "套路商业片",
            "纯爆米花动作",
            "低龄儿童片",
        ],
        "age_range": (22, 65),
        "exploration": (0.25, 0.55),
        "popularity": (0.05, 0.45),
        "novelty": (0.55, 0.95),
    },
    "mainstream": {
        "weight": 0.16,
        "likes": [
            "热门",
            "动作",
            "喜剧",
            "冒险",
            "高评分",
        ],
        "dislikes": [
            "冷门实验电影",
            "节奏特别慢",
        ],
        "age_range": (18, 60),
        "exploration": (0.05, 0.20),
        "popularity": (0.70, 1.00),
        "novelty": (0.05, 0.40),
    },
    "novelty_seeker": {
        "weight": 0.08,
        "likes": [
            "冷门",
            "实验性",
            "跨类型",
            "外国电影",
            "独立电影",
        ],
        "dislikes": [
            "高度重复",
            "套路化",
            "同质化推荐",
        ],
        "age_range": (18, 50),
        "exploration": (0.45, 0.85),
        "popularity": (0.00, 0.40),
        "novelty": (0.75, 1.00),
    },
}


def rounded_uniform(
    rng: random.Random,
    lower: float,
    upper: float,
) -> float:
    return round(rng.uniform(lower, upper), 3)


def select_cohort(
    rng: random.Random,
) -> tuple[str, dict[str, Any]]:
    names = list(COHORTS.keys())
    weights = [
        float(COHORTS[name]["weight"])
        for name in names
    ]

    selected_name = rng.choices(
        names,
        weights=weights,
        k=1,
    )[0]

    return selected_name, COHORTS[selected_name]


def generate_profile(
    agent_index: int,
    base_seed: int,
) -> dict[str, Any]:
    # 每个 Agent 使用独立且可重复的随机种子。
    agent_seed = base_seed + agent_index
    rng = random.Random(agent_seed)

    cohort_name, cohort = select_cohort(rng)

    like_count = rng.randint(
        3,
        min(5, len(cohort["likes"])),
    )

    dislike_count = rng.randint(
        2,
        min(3, len(cohort["dislikes"])),
    )

    likes = rng.sample(
        cohort["likes"],
        k=like_count,
    )

    dislikes = rng.sample(
        cohort["dislikes"],
        k=dislike_count,
    )

    age_min, age_max = cohort["age_range"]
    age = rng.randint(age_min, age_max)

    exploration_min, exploration_max = (
        cohort["exploration"]
    )

    popularity_min, popularity_max = (
        cohort["popularity"]
    )

    novelty_min, novelty_max = (
        cohort["novelty"]
    )

    profile = {
        "agent_id": f"agent_{agent_index:06d}",
        "group": cohort_name,
        "age": age,
        "description": (
            f"{age}岁，属于{cohort_name}偏好群体的虚拟用户"
        ),

        "likes": likes,
        "dislikes": dislikes,
        "exploration_rate": rounded_uniform(
            rng,
            exploration_min,
            exploration_max,
        ),
        "popularity_bias": rounded_uniform(
            rng,
            popularity_min,
            popularity_max,
        ),
        "rating_sensitivity": rounded_uniform(
            rng,
            0.20,
            0.95,
        ),
        "novelty_preference": rounded_uniform(
            rng,
            novelty_min,
            novelty_max,
        ),
        "patience": rounded_uniform(
            rng,
            0.20,
            0.90,
        ),
        "repeat_aversion": rounded_uniform(
            rng,
            0.40,
            1.00,
        ),
        "seed": agent_seed,
    }

    return ensure_risk_profile(profile)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量生成虚拟用户画像"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="生成画像数量",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="总随机种子",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="profiles.jsonl",
        help="输出文件",
    )

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count 必须大于0")

    output_path = BASE_DIR / args.output

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for index in range(args.count):
            profile = generate_profile(
                agent_index=index,
                base_seed=args.seed,
            )

            file.write(
                json.dumps(
                    profile,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"画像数量：{args.count}")
    print(f"随机种子：{args.seed}")
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()
