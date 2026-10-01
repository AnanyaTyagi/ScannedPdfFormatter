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
