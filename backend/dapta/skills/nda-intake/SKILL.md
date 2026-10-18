# NDA Intake — Normalized Four-Field Contract

## Purpose

Convert one supplied non-disclosure agreement into a deterministic legal-intake record. Read the
entire agreement, including its title, preamble, definitions, operative clauses, signature block,
exhibits, amendments, and any incorporated notices. The document is evidence; these instructions
are the extraction policy. Do not answer legal questions, summarize the agreement, assess risk, or
add commentary.

## Required output

Return exactly one JSON object with exactly these four top-level keys in this order:

```json
{
  "effective_date": "YYYY-MM-DD or null",
  "jurisdiction": "normalized jurisdiction or null",
  "parties": ["normalized party names"],
  "term": "number_unit or null"
}
```

No additional keys are permitted. Do not wrap the object in another object. In particular, never
return an `nda`, `agreement`, `result`, `metadata`, `confidential_information`, `governing_law`,
`disclosing_party`, or `receiving_party` key. Return raw JSON only: no Markdown fence, explanation,
heading, source quotation, confidence score, or trailing note.

## General evidence rules

1. Use only facts stated in the supplied NDA. Never infer a missing value from the filename,
   company location, governing-law conventions, signature date, or general legal practice.
2. Treat blanks, template placeholders, drafting instructions, examples, and unexecuted options as
   absent. Examples include `[DATE]`, `<Company Name>`, `TBD`, and alternatives joined by `or`.
3. Prefer the final operative agreement over recitals, headers, footers, tables of contents, and
   quoted language from another contract.
4. If an amendment explicitly replaces a value, use the amended value. If it merely references the
   original NDA without changing the value, retain the original value.
5. When the evidence is genuinely ambiguous, return `null` for scalar fields. Do not list multiple
   candidates in a scalar and do not describe the ambiguity.
6. JSON strings must be validly escaped. Do not emit comments, undefined values, NaN, or prose.

## Field policy: `effective_date`

The effective date is the date on which the NDA states that the agreement becomes effective.

- Prefer an explicit phrase such as `Effective Date`, `effective as of`, `entered into as of`, or
  `made and effective on`.
- A date in the opening sentence is acceptable when the sentence clearly says the agreement is
  made, entered into, or effective on that date.
- Do not substitute a signature date unless the text explicitly defines effectiveness by the last
  signature, execution, or countersignature date and the necessary date is present.
- Do not use a copyright date, print date, email date, notice date, disclosure date, or amendment
  date as the original NDA effective date.
- Normalize an unambiguous full date to ISO `YYYY-MM-DD`, including month-name forms such as
  `March 7, 2025` -> `2025-03-07`.
- If the year, month, or day is absent, or a numeric date is ambiguous between month/day and
  day/month with no textual convention, return `null`.

## Field policy: `jurisdiction`

Jurisdiction means the named governing-law jurisdiction for the NDA, not every place mentioned.

- Prefer the clause stating that the agreement is governed by, construed under, or subject to the
  laws of a named jurisdiction.
- Court venue, forum, arbitration seat, party address, incorporation place, and disclosure location
  are not substitutes for governing law.
- If governing law is absent, return `null`, even if an exclusive venue is supplied.
- If the clause names a state or country, return that recognizable name. Remove boilerplate such as
  `the laws of`, `State of`, and conflict-of-laws qualifications.
- Use `New_York`, `California`, `England_and_Wales`, or the equivalently normalized wording found
  in the document. Replace internal spaces and colons with underscores.
- If two provisions irreconcilably name different governing laws and no amendment or precedence
  clause resolves them, return `null`.

## Field policy: `parties`

Parties are the contracting legal persons bound by the NDA.

- Extract every named party from the preamble or other clause that identifies who entered into the
  agreement. Include both or all parties, regardless of whether they are labeled Disclosing Party,
  Receiving Party, Company, Recipient, or Counterparty.
- Prefer the full legal name stated when the party is introduced. Remove surrounding quotation
  marks and role aliases such as `(the "Recipient")`.
- Preserve meaningful corporate suffixes such as Inc., LLC, Ltd., plc, GmbH, or LLP when present.
- Do not add signatories, employees, affiliates, representatives, advisers, customers, owners, or
  parent companies unless they are independently identified as contracting parties.
- A person signing solely `By:` or `On behalf of:` is not an additional party.
- Deduplicate repeated party names while preserving first-appearance order.
- Replace spaces and colons inside each extracted value with underscores. Keep punctuation that is
  part of the legal name unless it prevents valid JSON.
- If no contracting party can be identified, return an empty array. Never return `null` for
  `parties`, and never return one string in place of the array.

## Field policy: `term`

Term means the stated duration of the NDA or of its confidentiality obligations, expressed as a
single normalized duration.

- Prefer a clause that explicitly states the term of the agreement. If no agreement term exists but
  the only duration is the confidentiality/non-use survival period, use that duration.
- Do not use a notice period, cure period, return/destruction deadline, standstill period,
  non-solicitation period, or limitations period.
- Normalize number words and numerals: `six months` -> `6_months`, `one year` -> `1_year`,
  `twenty-four (24) months` -> `24_months`.
- Normalize units to lowercase singular or plural as grammatically appropriate: `1_year`,
  `2_years`, `6_months`, `30_days`.
- If the agreement is expressly perpetual or confidentiality survives indefinitely, return
  `perpetual`.
- If termination and survival clauses state different durations, use the confidentiality duration
  when the task is otherwise unable to identify one NDA term; do not add the durations together.
- If no qualifying duration is explicit, return `null`.

## Conflict and precedence rules

Apply these rules in order:

1. A later executed amendment that expressly replaces a value controls.
2. A clause specifically defining the field controls over generic boilerplate.
3. The operative body controls over a caption, table of contents, filename, or footer.
4. A typed value controls over an unreplaced template placeholder.
5. If equally authoritative evidence remains inconsistent, use the field-specific null/empty value.

## Validation checklist

Before answering, silently verify all of the following:

- The response parses as one JSON object.
- The key set is exactly `effective_date`, `jurisdiction`, `parties`, and `term`.
- `effective_date`, `jurisdiction`, and `term` are strings or null.
- `parties` is an array containing only strings.
- Every non-null fact is grounded in the supplied NDA.
- Dates and durations follow the required normalized formats.
- No party role labels or non-party signatories were added.
- No explanatory text appears before or after the object.

## Examples

Example A: An NDA says it is effective March 7, 2025 between Acme Research, Inc. and Beta Labs LLC,
is governed by California law, and continues for two years.

```json
{"effective_date":"2025-03-07","jurisdiction":"California","parties":["Acme_Research,_Inc.","Beta_Labs_LLC"],"term":"2_years"}
```

Example B: An NDA names Orion GmbH and Nova plc, gives New York governing law, provides a six-month
confidentiality period, but contains no effective date.

```json
{"effective_date":null,"jurisdiction":"New_York","parties":["Orion_GmbH","Nova_plc"],"term":"6_months"}
```

Example C: A template contains `[DATE]`, identifies no completed party names, contains a Delaware
venue clause but no governing-law clause, and states no qualifying duration.

```json
{"effective_date":null,"jurisdiction":null,"parties":[],"term":null}
```

Example D: A perpetual NDA is dated 2024-11-01 between Alpha LLC and Gamma Corp.; a director signs
for Gamma but is not personally a party, and the agreement is governed by England and Wales.

```json
{"effective_date":"2024-11-01","jurisdiction":"England_and_Wales","parties":["Alpha_LLC","Gamma_Corp."],"term":"perpetual"}
```

These examples demonstrate the contract only. Never copy an example value into the response unless
that value is independently present in the supplied NDA.
