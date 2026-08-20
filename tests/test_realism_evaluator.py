from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from large_scale.kuaisar_baseline import (
    build_kuaisar_baseline,
)
from large_scale.realism_evaluator import (
    evaluate_events_file,
)


class RealismPipelineTest(unittest.TestCase):
    def test_build_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rec_path = root / "rec_inter.csv"
            baseline_path = root / "kuaisar_baseline.json"
            events_path = root / "events.jsonl"
            report_path = root / "realism_report.json"

            with rec_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "user_id",
                        "timestamp",
                        "click",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "user_id": "1",
                            "timestamp": "0",
                            "click": "1",
                        },
                        {
                            "user_id": "1",
                            "timestamp": "60000",
                            "click": "0",
                        },
                        {
                            "user_id": "1",
                            "timestamp": "2000000",
                            "click": "1",
                        },
                        {
                            "user_id": "2",
                            "timestamp": "0",
                            "click": "0",
                        },
                        {
                            "user_id": "2",
                            "timestamp": "120000",
                            "click": "1",
                        },
                    ]
                )

            baseline = build_kuaisar_baseline(
                rec_inter_path=rec_path,
                output_path=baseline_path,
                session_gap_minutes=30,
            )
            self.assertEqual(
                baseline["source"]["session_count"],
                3,
            )

            events = [
                {
                    "agent_id": "agent_000000",
                    "success": True,
                    "action": "click",
                },
                {
                    "agent_id": "agent_000000",
                    "success": True,
                    "action": "next",
                },
                {
                    "agent_id": "agent_000000",
                    "success": True,
                    "action": "stop",
                },
            ]
            events_path.write_text(
                "".join(
                    json.dumps(event) + "\n"
                    for event in events
                ),
                encoding="utf-8",
            )

            report = evaluate_events_file(
                events_path=events_path,
                baseline_path=baseline_path,
                output_path=report_path,
            )
            agent_report = report["agents"][0]
            self.assertEqual(
                agent_report["status"],
                "ok",
            )
            self.assertEqual(
                agent_report["session_length"],
                2,
            )
            self.assertEqual(
                agent_report["click_rate"],
                0.5,
            )
            self.assertGreaterEqual(
                agent_report[
                    "behavioral_realism_score"
                ],
                0.0,
            )
            self.assertLessEqual(
                agent_report[
                    "behavioral_realism_score"
                ],
                100.0,
            )
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
