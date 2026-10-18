from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.request import urlopen

from jsonschema import Draft202012Validator

from .data import REQUIRED_SPLITS, freeze_test, read_jsonl, sha256, validate_dataset


PREPROCESSING_VERSION = "dashboard-nvbench-plotbench-v1"
SOURCE_IDS = {
    "dashboard-nvbench",
    "dashboard-pandasplotbench",
    "dashboard-build-dashboard-skill",
}


@dataclass(frozen=True)
class SourceSpec:
    id: str
    source: str
    revision: str


def load_dashboard_sources(manifest_path: Path) -> dict[str, SourceSpec]:
    manifest = _read_object(manifest_path)
    if manifest.get("data_residency") != "remote-only":
        raise ValueError("dashboard sources must use remote-only data residency")
    sources = {
        item["id"]: SourceSpec(item["id"], item["source"], item["revision"])
        for item in manifest.get("sources", [])
        if item.get("id") in SOURCE_IDS
    }
    if set(sources) != SOURCE_IDS:
        raise ValueError("dashboard sources are missing from the source manifest")
    for source in sources.values():
        if not _is_commit(source.revision):
            raise ValueError(f"source revision is not an immutable commit: {source.id}")
    return sources


def materialize_dashboard_dataset(
    job_dir: Path,
    *,
    source_manifest: Path,
    contract_dir: Path,
    nvbench_records: Iterable[Mapping[str, Any]] | None = None,
    plotbench_rows: Iterable[Mapping[str, Any]] | None = None,
    max_rows_per_split: int | None = None,
    allow_local_test: bool = False,
) -> Path:
    if not allow_local_test and os.environ.get("DAPTA_REMOTE_WORKER") != "1":
        raise RuntimeError("dashboard datasets may only be materialized on a Dapta remote worker")
    if max_rows_per_split is not None and max_rows_per_split < 1:
        raise ValueError("max_rows_per_split must be positive")

    source_manifest = source_manifest.resolve()
    contract_dir = contract_dir.resolve()
    sources = load_dashboard_sources(source_manifest)
    schema = _read_object(contract_dir / "output_schema.json")
    Draft202012Validator.check_schema(schema)
    skill_text = (contract_dir / "SKILL.md").read_text(encoding="utf-8")

    job_dir = job_dir.resolve()
    dataset_dir = job_dir / "dashboard-dataset"
    cache_dir = job_dir / "source-cache"
    if dataset_dir.exists():
        raise FileExistsError(dataset_dir)
    dataset_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    rows: dict[str, list[dict[str, Any]]] = {split: [] for split in REQUIRED_SPLITS}

    try:
        nvbench = sources["dashboard-nvbench"]
        records = nvbench_records if nvbench_records is not None else _download_nvbench(nvbench)
        for key, record in _iter_nvbench(records):
            target_split = _nvbench_split(str(record.get("db_id", "")))
            if max_rows_per_split is not None and len(rows[target_split]) >= max_rows_per_split:
                if len(rows["train"]) >= max_rows_per_split and len(rows["dev"]) >= max_rows_per_split:
                    break
                continue
            for query_index, query in enumerate(record.get("nl_queries", [])):
                if max_rows_per_split is not None and len(rows[target_split]) >= max_rows_per_split:
                    break
                rows[target_split].append(
                    _nvbench_row(
                        key,
                        query_index,
                        str(query),
                        record,
                        nvbench,
                        target_split,
                        skill_text,
                    )
                )

        plotbench = sources["dashboard-pandasplotbench"]
        benchmark = plotbench_rows if plotbench_rows is not None else _load_plotbench(plotbench, cache_dir)
        for index, row in enumerate(benchmark):
            if max_rows_per_split is not None and index >= max_rows_per_split:
                break
            rows["test"].append(_plotbench_row(index, row, plotbench, skill_text))

        for split in REQUIRED_SPLITS:
            if not rows[split]:
                raise ValueError(f"dashboard materializer produced an empty {split} split")
            _write_jsonl(dataset_dir / f"{split}.jsonl", rows[split])
        frozen_test_sha = freeze_test(dataset_dir)
        counts = validate_dataset(dataset_dir)
        manifest = {
            "preprocessing_version": PREPROCESSING_VERSION,
            "source_manifest_sha256": sha256(source_manifest),
            "contract_sha256": {
                name: sha256(contract_dir / name)
                for name in ("SKILL.md", "output_schema.json", "evaluation_plan.json")
            },
            "sources": {
                source.id: {"repository": source.source, "revision": source.revision}
                for source in sources.values()
            },
            "counts": counts,
            "split_sha256": {
                split: sha256(dataset_dir / f"{split}.jsonl") for split in REQUIRED_SPLITS
            },
            "frozen_test_sha256": frozen_test_sha,
        }
        (dataset_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = validate_dashboard_dataset(
            dataset_dir,
            contract_dir=contract_dir,
            source_manifest=source_manifest,
        )
        if not report["passed"]:
            raise ValueError(f"dashboard dataset failed validation: {report['errors'][0]}")
    except Exception:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return dataset_dir


def validate_dashboard_dataset(
    dataset_dir: Path,
    *,
    contract_dir: Path,
    source_manifest: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    counts: dict[str, int] = {}
    try:
        counts = validate_dataset(dataset_dir)
    except Exception as exc:
        errors.append(str(exc))
    try:
        sources = load_dashboard_sources(source_manifest)
        schema = _read_object(contract_dir / "output_schema.json")
        validator = Draft202012Validator(schema)
        manifest = _read_object(dataset_dir / "manifest.json")
        if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("preprocessing version mismatch")
        if manifest.get("source_manifest_sha256") != sha256(source_manifest):
            raise ValueError("source manifest hash mismatch")
        expected_sources = {
            source.id: {"repository": source.source, "revision": source.revision}
            for source in sources.values()
        }
        if manifest.get("sources") != expected_sources:
            raise ValueError("pinned source revisions mismatch")
        train_databases: set[str] = set()
        dev_databases: set[str] = set()
        for split in REQUIRED_SPLITS:
            split_rows = read_jsonl(dataset_dir / f"{split}.jsonl")
            if manifest.get("counts", {}).get(split) != len(split_rows):
                raise ValueError(f"manifest count mismatch: {split}")
            if manifest.get("split_sha256", {}).get(split) != sha256(dataset_dir / f"{split}.jsonl"):
                raise ValueError(f"manifest hash mismatch: {split}")
            for index, row in enumerate(split_rows, start=1):
                completion = row.get("completion")
                schema_errors = list(validator.iter_errors(completion))
                if schema_errors:
                    raise ValueError(f"{split}:{index} output schema: {schema_errors[0].message}")
                source_id = row.get("metadata", {}).get("source_id")
                if split == "test" and source_id != "dashboard-pandasplotbench":
                    raise ValueError("test contains non-PandasPlotBench data")
                if split != "test" and source_id != "dashboard-nvbench":
                    raise ValueError(f"{split} contains non-nvBench data")
                database = row.get("metadata", {}).get("db_id")
                if split == "train":
                    train_databases.add(database)
                elif split == "dev":
                    dev_databases.add(database)
        if train_databases & dev_databases:
            raise ValueError("nvBench database leakage across train and dev")
    except Exception as exc:
        errors.append(str(exc))
    return {
        "schema_version": 1,
        "validator_version": "dashboard-validator-v1",
        "passed": not errors,
        "counts": counts,
        "errors": errors,
        "hashes": {
            "splits": {
                split: sha256(dataset_dir / f"{split}.jsonl")
                for split in REQUIRED_SPLITS
                if (dataset_dir / f"{split}.jsonl").is_file()
            }
        },
    }


def _download_nvbench(source: SourceSpec) -> Iterable[Mapping[str, Any]]:
    url = (
        "https://raw.githubusercontent.com/TsinghuaDatabaseGroup/nvBench/"
        f"{source.revision}/NVBench.json"
    )
    with urlopen(url, timeout=120) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("nvBench root must be a JSON object")
    return value.items()


def _load_plotbench(source: SourceSpec, cache_dir: Path) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    return load_dataset(
        source.source,
        revision=source.revision,
        split="test",
        streaming=True,
        cache_dir=str(cache_dir),
        token=os.environ.get("HF_TOKEN"),
    )


def _iter_nvbench(records: Iterable[Mapping[str, Any]]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for index, item in enumerate(records):
        if isinstance(item, tuple) and len(item) == 2:
            key, record = item
        else:
            record = item
            key = record.get("key", index)
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid nvBench row: {key}")
        yield str(key), record


def _nvbench_row(
    key: str,
    query_index: int,
    query: str,
    record: Mapping[str, Any],
    source: SourceSpec,
    split: str,
    skill_text: str,
) -> dict[str, Any]:
    headers, table_rows, series_field = _table_from_vis(record.get("vis_obj", {}))
    clean_csv = _csv_text(headers, table_rows)
    dirty_csv, cleaning_code, corruption = _corrupt_csv(clean_csv, f"{key}:{query_index}")
    dashboard_code = _dashboard_code(
        str(record.get("chart", "bar")), headers[0], headers[1], series_field
    )
    return {
        "id": f"dashboard-nvbench:{key}:{query_index}",
        "images": [],
        "system": skill_text,
        "prompt": f"{query}\n\nInput CSV:\n{dirty_csv}",
        "completion": {
            "cleaning_code": cleaning_code,
            "cleaned_csv": clean_csv,
            "dashboard_code": dashboard_code,
        },
        "metadata": {
            "source_id": source.id,
            "source_revision": source.revision,
            "source_key": key,
            "db_id": str(record.get("db_id", "")),
            "target_split": split,
            "chart": str(record.get("chart", "")),
            "corruption": corruption,
        },
    }


def _plotbench_row(
    index: int,
    row: Mapping[str, Any],
    source: SourceSpec,
    skill_text: str,
) -> dict[str, Any]:
    data_csv = str(row.get("data_csv", "")).strip()
    if not data_csv:
        raise ValueError(f"PandasPlotBench row {index} has no CSV")
    data_csv += "\n"
    prompt = " ".join(
        value.strip()
        for value in (
            str(row.get("task__plot_description", "")),
            str(row.get("task__plot_style", "")),
        )
        if value.strip()
    )
    code = str(row.get("code_plot", "")).strip()
    if not prompt or not code:
        raise ValueError(f"PandasPlotBench row {index} is incomplete")
    return {
        "id": f"dashboard-pandasplotbench:{row.get('id', index)}",
        "images": [],
        "system": skill_text,
        "prompt": f"{prompt}\n\nInput CSV:\n{data_csv}",
        "completion": {
            "cleaning_code": _cleaning_code("none"),
            "cleaned_csv": data_csv,
            "dashboard_code": code,
        },
        "metadata": {
            "source_id": source.id,
            "source_revision": source.revision,
            "source_split": "test",
            "target_split": "test",
            "corruption": "none",
        },
    }


def _table_from_vis(vis: Any) -> tuple[list[str], list[list[Any]], str | None]:
    if not isinstance(vis, Mapping):
        raise ValueError("nvBench row has no visualization object")
    x_name = str(vis.get("x_name") or "x")
    y_name = str(vis.get("y_name") or "y")
    x_groups = vis.get("x_data") or []
    y_groups = vis.get("y_data") or []
    if not isinstance(x_groups, list) or not isinstance(y_groups, list) or not x_groups or not y_groups:
        raise ValueError("nvBench visualization has no tabular data")
    classifications = vis.get("classify") or []
    rows: list[list[Any]] = []
    grouped = len(x_groups) > 1 or len(y_groups) > 1
    for group_index, (xs, ys) in enumerate(zip(x_groups, y_groups)):
        if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
            raise ValueError("nvBench x/y data lengths do not match")
        label = _classification_label(classifications, group_index)
        for x_value, y_value in zip(xs, ys):
            rows.append([x_value, y_value, label] if grouped else [x_value, y_value])
    if not rows:
        raise ValueError("nvBench visualization table is empty")
    return ([x_name, y_name, "series"] if grouped else [x_name, y_name], rows, "series" if grouped else None)


def _classification_label(value: Any, index: int) -> str:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, list) and index < len(value):
        return str(value[index])
    return f"series-{index + 1}"


def _csv_text(headers: list[str], rows: list[list[Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue()


def _corrupt_csv(clean_csv: str, key: str) -> tuple[str, str, str]:
    parsed = list(csv.reader(io.StringIO(clean_csv)))
    corruption = ("whitespace", "duplicate", "currency")[
        int(hashlib.sha256(key.encode()).hexdigest(), 16) % 3
    ]
    if corruption == "currency" and all(_is_number(row[1]) for row in parsed[1:]):
        for row in parsed[1:]:
            row[1] = f"${float(row[1]):,.2f}"
    elif corruption == "currency":
        corruption = "whitespace"
    if corruption == "whitespace":
        parsed[0] = [f" {value} " for value in parsed[0]]
        for row in parsed[1:]:
            row[:] = [f" {value} " if isinstance(value, str) else value for value in row]
    elif corruption == "duplicate":
        parsed.append(list(parsed[1]))
    return _csv_text(parsed[0], parsed[1:]), _cleaning_code(corruption), corruption


def _cleaning_code(corruption: str) -> str:
    lines = ["import pandas as pd", "df = pd.read_csv('input.csv')"]
    if corruption == "whitespace":
        lines += [
            "df.columns = df.columns.str.strip()",
            "for col in df.select_dtypes(include='object'): df[col] = df[col].str.strip()",
        ]
    elif corruption == "duplicate":
        lines.append("df = df.drop_duplicates()")
    elif corruption == "currency":
        lines.append("df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1].str.replace(r'[$,]', '', regex=True))")
    lines.append("df.to_csv('cleaned.csv', index=False)")
    return "\n".join(lines) + "\n"


def _dashboard_code(chart: str, x_field: str, y_field: str, series_field: str | None) -> str:
    chart = chart.casefold()
    function = "scatter" if "scatter" in chart else "line" if "line" in chart else "pie" if "pie" in chart else "bar"
    if function == "pie":
        call = f"px.pie(df, names={x_field!r}, values={y_field!r})"
    else:
        color = f", color={series_field!r}" if series_field else ""
        call = f"px.{function}(df, x={x_field!r}, y={y_field!r}{color})"
    lines = [
        "import pandas as pd",
        "import plotly.express as px",
        "df = pd.read_csv('cleaned.csv')",
        f"fig = {call}",
    ]
    if "stacked" in chart:
        lines.append("fig.update_layout(barmode='stack')")
    lines.append("fig.write_html('dashboard.html', include_plotlyjs=True)")
    return "\n".join(lines) + "\n"


def _nvbench_split(db_id: str) -> str:
    digest = hashlib.sha256(f"20261017:{db_id}".encode()).digest()
    return "dev" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _is_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sources", type=Path, default=Path("research/adapter-assets/sources.lock.json")
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("research/adapter-assets/skills/csv-to-dashboard"),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    dataset = materialize_dashboard_dataset(
        args.job_dir,
        source_manifest=args.sources,
        contract_dir=args.contract,
        max_rows_per_split=args.limit,
    )
    report = validate_dashboard_dataset(
        dataset,
        contract_dir=args.contract,
        source_manifest=args.sources,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(dataset), "counts": report["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
