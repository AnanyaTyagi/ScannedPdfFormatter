---
title: PDF Accessibility Formatter
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# TagMyPDF

Converts scanned (OCR'd) PDFs into PDF/UA-compliant accessible PDFs with structure tagging.

## What it does

Takes a scanned PDF with an invisible OCR text layer and outputs a tagged PDF with `<H1>`, `<H2>`, `<P>`, `<Figure>`, `<Caption>`, `<LI>`, and `<Table>` structure elements, validated against ISO 14289-1 (PDF/UA-1).

## Pipeline

1. **ProbingFile.py** — renders pages to PNG at 300 DPI
2. **DetectLayoutFixed.py** — runs YOLOv8 (DocLayNet weights) to detect region types per page
3. **ExportLines.py** — extracts OCR word-level text from the PDF
4. **FuseLinesFixed_fixed.py** — assigns structure tags using YOLO regions + font heuristics
5. **TableLists.py** — post-processes table and list structure
6. **TagPDFFinal.py** — writes the PDF/UA structure tree using pikepdf

Run via `TagMyPDF2.py` or through the Streamlit web app (`app.py`).

## Running

```bash
python TagMyPDF2.py input.pdf
```

Output is `output.pdf` in the working directory.

## Evaluation results

Evaluated on 258 scanned archival PDFs from the [De la Colección: Farmworker Heritage](https://digitalcollections.sjlibrary.org/) collection.
Validation used veraPDF 1.28.2 with the PDF/UA-1 profile.

| Group | Documents | Pass | Compliance |
|---|---|---|---|
| Well-encoded (Batch1–7) | 74 | 65 | 87.8% |
| Flyers (Flyers1–3) | 184 | 88 | 47.8% |
| **Overall** | **258** | **153** | **59.3%** |

Per-batch breakdown:

| Batch | Pass/Total | Rate |
|---|---|---|
| Batch1 | 9/10 | 90% |
| Batch2 | 4/6 | 67% |
| Batch3 | 11/12 | 92% |
| Batch4 | 9/11 | 82% |
| Batch5 | 10/12 | 83% |
| Batch6 | 18/19 | 95% |
| Batch7 | 4/4 | 100% |
| Flyers1 | 26/56 | 46% |
| Flyers2 | 22/51 | 43% |
| Flyers3 | 40/77 | 52% |

The compliance split reflects source font encoding quality, not document complexity. Flyers1–3 contain embedded fonts with missing Unicode character mapping tables (ISO 14289-1 clause 7.21.8 violations), which the tagging pipeline cannot fix. Batch1–7 documents have well-encoded fonts and achieve 67–100% compliance.

Speed: 142.9 pages/minute on a MacBook Pro M4 (0.42 seconds/page average, 3,272 pages total).

Raw veraPDF results and corpus manifest are in `results/`.

## Failure modes

The dominant failure (99% of non-compliant documents) is source font encoding: embedded fonts lacking Unicode character maps produce `.notdef` glyph violations regardless of tagging quality. Secondary failures are reading order ambiguity in multi-column layouts and missing table structural markup.

## Requirements

See `requirements.txt`. YOLOv8 weights (`yolov8n-doclaynet.pt`) must be present in the working directory.
