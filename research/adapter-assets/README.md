# Adapter research assets

This directory is Dapta's reproducible catalog for candidate LoRA workflows.
It deliberately contains **no dataset rows, images, archives, or model weights**.

## Storage policy

- The repository keeps source URLs, immutable revisions, licenses, official split
  names, and proposed split rules in `sources.lock.json`.
- A Vultr worker retrieves a pinned revision into its temporary disk or streams
  it with Hugging Face `datasets`.
- The worker writes only derived adapter artifacts, metrics, and dataset hashes
  to durable storage. The source dataset cache is deleted with the worker.
- TEST data is never sent to the Data Builder and is fetched only by the
  independent evaluator.
- Small public `SKILL.md` references are vendored because they are workflow
  specifications, not training data.

For Hugging Face datasets, the worker will use either:

```python
from datasets import load_dataset

dataset = load_dataset(DATASET_ID, revision=REVISION, streaming=True)
```

or, when random access is required, download to an ephemeral directory such as
`/mnt/dapta-job/<job_id>/datasets` and remove that directory when the job ends.

## Recommended first three workflows

### 1. Receipt images to expense JSON

- Primary multimodal dataset: `naver-clova-ix/cord-v2`.
- Recognizable benchmark: `jsdnrs/ICDAR2019-SROIE`.
- CORD already has train/validation/test splits. SROIE has train/test only, so a
  deterministic document-level validation subset must be derived from SROIE
  train while its test split remains frozen.
- The vendored expense-tracker skill is **reference-only**: it depends on named
  third-party tools and is not a clean, self-contained training specification.
- Dedicated task skill ZIP: **not found**.

### 2. SEC filing to standardized three-statement model

- Source facts: SEC Financial Statement Data Sets or the no-key EDGAR XBRL API.
- Supervised tag auxiliary: `nlpaueb/finer-139`, which has official
  train/validation/test splits.
- Optional numerical-reasoning auxiliary: FinQA, with official train/dev/test.
- Split future SEC examples by held-out CIK and filing period so the same company
  cannot leak across train and test.
- The vendored `three-statement-model` skill is the strongest public workflow
  specification found and includes its source references.
- Dedicated task skill ZIP: **not found**; the source is a folder in a public
  GitHub repository.

### 3. Text PII redaction

- Training and validation: `ai4privacy/pii-masking-openpii-1.5m`.
- Frozen independent test: `piimb/pii-masking-benchmark`.
- The PIIMB test is CC BY-NC 4.0, so it is suitable for hackathon evaluation but
  must not be presented as commercial training data.
- The healthcare PHI expansion is gated and stays blocked until its access terms
  are accepted.
- The vendored redaction skill is **reference-only** and AGPL-licensed; it is an
  application-control skill rather than a clean output contract for this LoRA.
- Dedicated task skill ZIP: **not found**.

## Additional candidates

- M&A agreement review: MAUD is ready and has official train/dev/test splits.
  No suitable public task-specific `SKILL.md` or skill ZIP was found.
- Product attribute extraction: MAVE is promising, but the official repository
  publishes labels and reconstruction code rather than a turnkey full dataset.
  No suitable public task-specific `SKILL.md` or skill ZIP was found.
- Threat-report to MITRE ATT&CK mapping: AnnoCTR provides a labeled corpus and
  MITRE publishes the taxonomy as STIX. No suitable public task-specific
  `SKILL.md` or skill ZIP was found.
- Healthcare adverse-event normalization remains research-only: a single,
  clearly licensed, authoritative train/dev/test package was not verified.

## Skill provenance

- `skills/three-statement-model`: Anthropic `financial-services`, commit
  `574ed3624aebd0418c7e96cd101262f30210ab26`, Apache-2.0.
- `skills/reference/expense-tracker`: `claude-office-skills/skills`, commit
  `9c4c7d5cd2813a8936bf2c9fdb174ea883b85a11`, MIT.
- `skills/reference/doc-redaction-app`: `seanpedrick-case/doc_redaction`, commit
  `5675ac2673b17d51578726f8ce44129b421a108a`, AGPL-3.0-only.

The reference-only files are research inputs. They are not approved Dapta skills
and should not be used as training labels without a separate design review.
