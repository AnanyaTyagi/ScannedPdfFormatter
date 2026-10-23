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

Takes a scanned PDF (using its existing text layer, or Tesseract OCR on pages without one) and outputs a tagged PDF with `<H1>`, `<H2>`, `<P>`, `<Figure>`, `<Caption>`, `<LI>`, and `<Table>` structure elements, validated against ISO 14289-1 (PDF/UA-1).

## Pipeline

1. **ProbingFile.py** — renders pages to PNG at 300 DPI
2. **DetectLayoutFixed.py** — runs YOLOv8 (DocLayNet weights) to detect region types per page
3. **ExportLines.py** — extracts OCR word-level text from the PDF
4. **FuseLinesFixed_fixed.py** — assigns structure tags using YOLO regions + font heuristics
5. **TableLists.py** — post-processes table and list structure
6. **TagPDFFinal.py** — writes the PDF/UA structure tree using pikepdf

Run via `TagMyPDF2.py` or through the Streamlit web app (`app.py`).

## Running locally

### Requirements

- macOS or Linux (Windows users: use Docker, below)
- Python 3.12 (the version used in the Dockerfile; PyTorch 2.5 does not support newer Python versions)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) with English and Spanish language data
- [Git LFS](https://git-lfs.com/), which stores the YOLO weights `yolov8n-doclaynet.pt`
- Optional, for validation: [veraPDF](https://verapdf.org/software/) (the evaluation used 1.28.2)

### Option A: setup script

```bash
git lfs install
git clone https://github.com/AnanyaTyagi/ScannedPdfFormatter.git
cd ScannedPdfFormatter
./setup.sh
```

`setup.sh` installs Tesseract and Git LFS (Homebrew on macOS, apt on Debian/Ubuntu), creates a Python 3.12 virtual environment in `.venv`, and installs `requirements.txt`.

### Option B: manual setup

```bash
# System packages
brew install python@3.12 tesseract tesseract-lang git-lfs          # macOS
# sudo apt-get install python3.12 python3.12-venv git-lfs \
#     tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa libgl1 libglib2.0-0   # Ubuntu/Debian

git lfs install
git clone https://github.com/AnanyaTyagi/ScannedPdfFormatter.git
cd ScannedPdfFormatter
git lfs pull                       # yolov8n-doclaynet.pt should be about 6 MB, not a small text file

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Check OCR is available; if pytesseract cannot run, image-only pages are tagged without text
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
tesseract --list-langs             # should list eng and spa
```

### Tag a PDF

```bash
source .venv/bin/activate
python TagMyPDF2.py path/to/input.pdf
```

The tagged PDF is written to `output.pdf` in the current directory. Intermediate files go to `debug/`, `layout_out/`, `lines_out/`, `fused_tags/` and `structures/` in the current directory; these folders and `output.pdf` are overwritten on every run, so copy `output.pdf` elsewhere before processing the next file.

To process a folder of PDFs:

```bash
mkdir -p tagged
for f in path/to/pdfs/*.pdf; do
    python TagMyPDF2.py "$f" && cp output.pdf "tagged/$(basename "$f")"
done
```

Options: `--dpi 300` (render resolution), `--weights yolov8n-doclaynet.pt`, `--labelmap labelmap.json`.

### Validate with veraPDF

```bash
verapdf --flavour ua1 output.pdf                       # one file
verapdf --flavour ua1 --format xml -r tagged/ > report.xml   # a folder
```

`isCompliant="true"` in the report means the file passes PDF/UA-1.

### Option C: Docker

```bash
docker build -t tagmypdf .

# Web interface at http://localhost:7860
docker run --rm -p 7860:7860 tagmypdf

# Command line, on a PDF in the current directory
docker run --rm -v "$PWD":/data -w /data tagmypdf python /app/TagMyPDF2.py /data/input.pdf
```

The command-line form writes `output.pdf` to the current directory.

## Evaluation results

Evaluated on 257 scanned archival PDFs from the [De la Colección: Farmworker Heritage](https://digitalcollections.sjlibrary.org/) collection.
Validation used veraPDF 1.28.2 with the PDF/UA-1 profile.

| Group | Documents | Pass | Compliance |
|---|---|---|---|
| Well-encoded (Batch1–7) | 73 | 63 | 86.3% |
| Flyers (Flyers1–3) | 184 | 88 | 47.8% |
| **Overall** | **257** | **151** | **58.8%** |

Per-batch breakdown:

| Batch | Pass/Total | Rate |
|---|---|---|
| Batch1 | 9/10 | 90% |
| Batch2 | 4/6 | 67% |
| Batch3 | 11/12 | 92% |
| Batch4 | 8/10 | 80% |
| Batch5 | 10/12 | 83% |
| Batch6 | 17/19 | 89% |
| Batch7 | 4/4 | 100% |
| Flyers1 | 26/56 | 46% |
| Flyers2 | 22/51 | 43% |
| Flyers3 | 40/77 | 52% |

The compliance split reflects source font encoding quality, not document complexity. Flyers1–3 contain embedded fonts with missing Unicode character mapping tables (ISO 14289-1 clause 7.21.8 violations), which the tagging pipeline cannot fix. Batch1–7 documents have well-encoded fonts and achieve 67–100% compliance.

Speed: 93.1 pages/minute on a MacBook Pro M4 with 16 GB RAM (0.64 seconds/page average, 3,262 pages total, including OCR on image-only pages).

Raw veraPDF results and corpus manifest are in `results/`.

## Failure modes

The dominant failure (104 of 106 non-compliant documents, 98%) is source font encoding: embedded fonts lacking Unicode character maps produce `.notdef` glyph violations regardless of tagging quality. The remaining failures are one flyer with inconsistent glyph widths (clause 7.21.5) and one document with tagged content inside an Artifact (clause 7.1); one document also has link annotations without alternate descriptions (clauses 7.18.1/7.18.5). No table-structure or reading-order failures occur in this corpus.

## License

GNU Affero General Public License v3.0 (AGPL-3.0). See `LICENSE`. This matches the licenses of the Ultralytics and PyMuPDF dependencies.
