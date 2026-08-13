from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
AGENT_SCRIPT = BASE_DIR / "websim_user_agent.py"
LOG_DIR = BASE_DIR / "batch_logs"


def run_one_agent(
    profile_index: int,
    profiles_file: str,
) -> dict[str, Any]:
    """启动一个独立的 Agent 子进程。"""

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        LOG_DIR
        / f"agent_{profile_index:06d}.log"
    )

    command = [
        sys.executable,
        str(AGENT_SCRIPT),
        "--profile-index",
        str(profile_index),
        "--profiles",
        profiles_file,
    ]

    # 保证子进程的中文日志使用 UTF-8。
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    started_at = time.perf_counter()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )

    elapsed_seconds = round(
        time.perf_counter() - started_at,
        2,
    )

    return {
        "profile_index": profile_index,
        "return_code": result.returncode,
        "elapsed_seconds": elapsed_seconds,
        "log_path": str(log_path),
        "success": result.returncode == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="并发运行多个 WebSim Agent"
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="第一个用户画像编号",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="本次运行的 Agent 总数",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="同时运行的最大 Agent 数量",
    )

    parser.add_argument(
        "--profiles",
        type=str,
        default="profiles.jsonl",
        help="用户画像文件",
    )

    args = parser.parse_args()

    if args.start_index < 0:
        raise ValueError(
            "--start-index 不能小于 0"
        )

    if args.count <= 0:
        raise ValueError(
            "--count 必须大于 0"
        )

    if args.concurrency <= 0:
        raise ValueError(
            "--concurrency 必须大于 0"
        )

    if not AGENT_SCRIPT.exists():
        raise FileNotFoundError(
            f"没有找到 Agent 程序：{AGENT_SCRIPT}"
        )

    profiles_path = BASE_DIR / args.profiles

    if not profiles_path.exists():
        raise FileNotFoundError(
            f"没有找到画像文件：{profiles_path}"
        )

    profile_indexes = list(
        range(
            args.start_index,
            args.start_index + args.count,
        )
    )

    print("=" * 60)
    print("WebSim Agent 并发运行器")
    print(f"画像编号：{profile_indexes}")
    print(f"Agent 总数：{args.count}")
    print(f"最大并发：{args.concurrency}")
    print(f"日志目录：{LOG_DIR}")
    print("=" * 60)

    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = {
            executor.submit(
                run_one_agent,
                profile_index,
                args.profiles,
            ): profile_index
            for profile_index in profile_indexes
        }

        for future in as_completed(futures):
            profile_index = futures[future]

            try:
                result = future.result()
            except Exception as error:
                result = {
                    "profile_index": profile_index,
                    "success": False,
                    "return_code": -1,
                    "elapsed_seconds": 0,
                    "log_path": "",
                    "error": (
                        f"{type(error).__name__}: {error}"
                    ),
                }

            results.append(result)

            status = (
                "成功"
                if result["success"]
                else "失败"
            )

            print(
                f"Agent {profile_index:06d}："
                f"{status}，"
                f"耗时 {result['elapsed_seconds']} 秒"
            )

            if result.get("log_path"):
                print(
                    f"  日志：{result['log_path']}"
                )

    success_count = sum(
        1
        for result in results
        if result["success"]
    )

    failure_count = len(results) - success_count

    print("\n" + "=" * 60)
    print("并发实验结束")
    print(f"成功：{success_count}")
    print(f"失败：{failure_count}")
    print("=" * 60)

    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()