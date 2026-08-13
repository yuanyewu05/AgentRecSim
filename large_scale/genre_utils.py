from __future__ import annotations

import re
from typing import Any


# 英文电影类型 → 项目统一使用的中文类型。
#
# 左边全部使用小写，是为了兼容：
# Action、ACTION、action 等不同写法。
GENRE_LABEL_MAP: dict[str, str] = {
    # 动作与冒险
    "action": "动作",
    "adventure": "冒险",

    # 动画、儿童与家庭
    "animation": "动画",
    "animated": "动画",
    "children": "儿童",
    "children's": "儿童",
    "childrens": "儿童",
    "family": "家庭",

    # 喜剧与剧情
    "comedy": "喜剧",
    "drama": "剧情",

    # 科幻与奇幻
    "science fiction": "科幻",
    "sci-fi": "科幻",
    "sci fi": "科幻",
    "fantasy": "奇幻",

    # 悬疑、犯罪与惊悚
    "mystery": "悬疑",
    "crime": "犯罪",
    "thriller": "惊悚",
    "horror": "恐怖",

    # 爱情
    "romance": "爱情",

    # 其他常见电影类型
    "documentary": "纪录片",
    "history": "历史",
    "historical": "历史",
    "biography": "传记",
    "war": "战争",
    "western": "西部",
    "musical": "歌舞",
    "music": "音乐",
    "film-noir": "黑色电影",
    "film noir": "黑色电影",
    "tv movie": "电视电影",
}


def canonicalize_label(label: Any) -> str:
    """
    把一个标签转换成项目统一使用的名称。

    例如：
        Action          -> 动作
        Adventure       -> 冒险
        Science Fiction -> 科幻
        Animation       -> 动画

    不在映射表中的标签保持原样。

    因此用户画像里的：
        温馨
        反转
        节奏特别慢
        纯爱情

    不会被删除或错误修改。
    """

    # 先把输入统一转换成字符串。
    cleaned = str(label).strip()

    if not cleaned:
        return ""

    # 把连续空格压缩成一个空格。
    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    # casefold用于不区分英文大小写。
    lookup_key = cleaned.casefold()

    return GENRE_LABEL_MAP.get(
        lookup_key,
        cleaned,
    )


def normalize_labels(value: Any) -> list[str]:
    """
    把字符串、列表或其他格式统一转换成标签列表，
    同时完成英文电影类型到中文类型的转换。

    示例一：
        "Action,Adventure,Science Fiction"

    返回：
        ["动作", "冒险", "科幻"]

    示例二：
        ["Animation", "Family", "Comedy"]

    返回：
        ["动画", "家庭", "喜剧"]

    示例三：
        ["家庭", "温馨", "动画"]

    返回：
        ["家庭", "温馨", "动画"]
    """

    if value is None:
        return []

    raw_parts: list[Any] = []

    if isinstance(value, str):
        # 兼容逗号、竖线、顿号、分号、斜杠等分隔符。
        raw_parts = re.split(
            r"[|,、;/]+",
            value,
        )

    elif isinstance(
        value,
        (list, tuple, set),
    ):
        # 列表中的每一项也可能包含多个标签，
        # 例如 ["Action|Adventure", "Sci-Fi"]。
        for item in value:
            if isinstance(item, str):
                raw_parts.extend(
                    re.split(
                        r"[|,、;/]+",
                        item,
                    )
                )
            else:
                raw_parts.append(item)

    else:
        raw_parts = [value]

    normalized: list[str] = []
    seen: set[str] = set()

    for raw_label in raw_parts:
        label = canonicalize_label(raw_label)

        if not label:
            continue

        # 去掉重复标签，同时保持原来的顺序。
        if label in seen:
            continue

        seen.add(label)
        normalized.append(label)

    return normalized