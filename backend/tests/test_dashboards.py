import tempfile
import unittest
from pathlib import Path

from dapta.dashboards import (
    _nvbench_split,
    materialize_dashboard_dataset,
    validate_dashboard_dataset,
)
from dapta.data import read_jsonl


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "research/adapter-assets/sources.lock.json"
CONTRACT = ROOT / "research/adapter-assets/skills/csv-to-dashboard"


def database_for(split):
    for index in range(1000):
        value = f"database-{index}"
        if _nvbench_split(value) == split:
            return value
    raise AssertionError(split)


def nvbench_row(db_id, chart="Bar"):
    return {
        "db_id": db_id,
        "chart": chart,
        "nl_queries": [f"Build a {chart.lower()} chart of sales by region."],
        "vis_obj": {
            "x_name": "region",
            "y_name": "sales",
            "x_data": [["East", "West"]],
            "y_data": [[10, 20]],
            "classify": [],
        },
    }


class DashboardDatasetTests(unittest.TestCase):
    def test_materializes_remote_only_sources_without_split_leakage(self):
        records = [
            {"key": "train-case", **nvbench_row(database_for("train"))},
            {"key": "dev-case", **nvbench_row(database_for("dev"), "Line")},
        ]
        plotbench = [{
            "id": 7,
            "data_csv": "region,sales\nEast,10\nWest,20\n",
            "task__plot_description": "Plot sales by region.",
            "task__plot_style": "Use a bar chart.",
            "code_plot": "df.plot.bar(x='region', y='sales')",
        }]
        with tempfile.TemporaryDirectory() as temp:
            dataset = materialize_dashboard_dataset(
                Path(temp),
                source_manifest=SOURCES,
                contract_dir=CONTRACT,
                nvbench_records=records,
                plotbench_rows=plotbench,
                allow_local_test=True,
            )
            report = validate_dashboard_dataset(
                dataset,
                contract_dir=CONTRACT,
                source_manifest=SOURCES,
            )

            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(report["counts"], {"train": 1, "dev": 1, "test": 1})
            train = read_jsonl(dataset / "train.jsonl")[0]
            self.assertEqual(train["images"], [])
            self.assertIn("dashboard.html", train["completion"]["dashboard_code"])
            self.assertIn("cleaned.csv", train["completion"]["cleaning_code"])
            test = read_jsonl(dataset / "test.jsonl")[0]
            self.assertEqual(test["metadata"]["source_id"], "dashboard-pandasplotbench")
            self.assertTrue((dataset / "test.sha256").is_file())

    def test_local_download_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "remote worker"):
                materialize_dashboard_dataset(
                    Path(temp), source_manifest=SOURCES, contract_dir=CONTRACT
                )


if __name__ == "__main__":
    unittest.main()
