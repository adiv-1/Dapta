---
name: csv-to-dashboard
description: Clean an uploaded CSV and turn it into a self-contained dashboard when a user repeatedly needs the same analysis workflow.
source: anthropics/knowledge-work-plugins/data/skills/build-dashboard/SKILL.md
source_revision: 33eeda1e8fd856455b05479d13a0f85b5f803744
license: Apache-2.0
---

# CSV to Dashboard

Return only JSON matching `output_schema.json`.

1. Inspect the CSV columns and the requested analysis.
2. Produce deterministic pandas cleaning code. It must read `input.csv` and write `cleaned.csv` without network access.
3. Include the exact cleaned CSV contents in `cleaned_csv`.
4. Produce Plotly Python code that reads `cleaned.csv` and writes a self-contained `dashboard.html`.
5. Use the requested fields, aggregation, chart type, labels, and grouping. Do not invent rows or business metrics.
6. Keep source values unless an explicitly recoverable formatting defect is being repaired. Remove only exact duplicate rows.

The public build-dashboard skill is the workflow reference. This profile narrows its output so training and evaluation can be deterministic.
