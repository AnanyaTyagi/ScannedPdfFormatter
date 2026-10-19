# Evaluation results

## verapdf_results.csv

Per-document veraPDF 1.28.2 results (PDF/UA-1 profile).

Columns:
- `batch` — source folder (Batch1–7, Flyers1–3)
- `group` — `well_encoded` or `flyers`
- `filename` — original source PDF filename
- `tagged_filename` — tagged output filename (format: `BatchN__original.pdf`)
- `compliant` — True/False
- `error_count` — number of "fail" tokens in veraPDF text output

## corpus_manifest.csv

Columns:
- `batch`, `group`, `filename` — same as above
- `pages` — page count of the source PDF

## verapdf_errors_raw.csv

Per-file, per-rule veraPDF failure details (non-compliant documents only; compliant documents have a single row with empty clause/description).

Columns:
- `filename` — tagged PDF filename
- `batch`, `group` — same as above
- `compliant` — True/False
- `clause` — ISO 14289-1 clause number (e.g. `7.21.8`)
- `description` — rule description text
- `failed_checks` — number of individual check failures for this rule in this document

## verapdf_error_summary.csv

Aggregated failure counts across all 106 non-compliant documents.

Columns:
- `clause` — ISO 14289-1 clause number
- `description` — rule description
- `documents_affected` — count of non-compliant documents failing this rule
- `pct_of_noncompliant` — percentage of the 106 non-compliant documents
- `total_failed_checks` — sum of individual check failures across all documents

Top failure modes: clause 7.21.8 `.notdef` glyph (98.1% of failures), clause 7.21.7 ToUnicode CMap (22.6%), clause 7.1 artifact/tagged content (1.9%), clause 7.21.5 glyph widths (1.9%), clauses 7.18.1/7.18.5 annotation metadata (0.9%).
