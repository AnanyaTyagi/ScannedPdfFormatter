# ScannedPdfFormatter

Convert **scanned OCR PDFs** into **properly tagged, accessible PDFs/HTML**.
Adds semantic structure (headings, paragraphs, lists) so documents work better with screen readers and pass common accessibility checkers.

## Features

* 🧠 Extracts text and layout from PDFs using **PyMuPDF (fitz)**.
* 🏷️ Heuristically infers structure (body text size, headings, lists) and emits semantic **HTML**.
* 📄 Renders to **PDF** via WeasyPrint or **PrinceXML** (recommended for PDF/UA conformance).
* 🖼️ **Facsimile mode**: preserves original page imagery with selectable text overlay.
* 🔎 CLI tools that print per-page stats (lines, body size, H1/H2, list items) to help tune rules.

## Quick start

```bash
# 1) Clone and enter
git clone https://github.com/AnanyaTyagi/ScannedPdfFormatter.git
cd ScannedPdfFormatter

# 2) Python env (3.9+ recommended; works on 3.13)
python -m venv .venv
source .venv/bin/activate

# 3) Install Python deps
python -m pip install --upgrade pip
python -m pip install pymupdf regex numpy weasyprint
```

> **macOS note (WeasyPrint native libs)**
> WeasyPrint needs Cairo/Pango stack. Install with Homebrew:
>
> ```bash
> brew install cairo pango gdk-pixbuf libffi gobject-introspection pkg-config
> ```
>
> If WeasyPrint still can’t find `libgobject-2.0`, prefer **PrinceXML** for PDF output (works great and supports PDF/UA).

## Commands

### 1) Produce structured HTML (and optionally a PDF)

```bash
# HTML only
python format_pdf.py input.pdf out.html

# HTML + PDF (WeasyPrint; requires Cairo/Pango)
python format_pdf.py input.pdf out.html --pdf out.pdf

# HTML + PDF via PrinceXML (recommended)
python format_pdf.py input.pdf out.html --pdf out.pdf --engine prince
```

### 2) Facsimile mode (keeps page image, overlays selectable text)

```bash
python format_pdf.py input.pdf out_facsimile.html \
  --pdf out_facsimile.pdf --engine prince --mode facsimile
```

### 3) Low-level structure probe (debug/metrics)

```bash
python step3_structure_from_pdf.py input.pdf out.html
# prints per-page: lines, estimated body size, counts of h1/h2/li, etc.
```

## Example output (console)

```
Page 1: lines=8 body=11.1 h1=0 h2=0 li=0
Page 4: lines=13 body=10.6 h1=0 h2=2 li=1
...
Wrote out.html with 10 pages
```

## How it works (high level)

1. **Extraction**: uses `fitz.Page.get_text("text"|"dict")` to gather text spans and coordinates.
2. **Analysis**: estimates **body font size**, detects **headings** (size/weight deltas), **lists** (bullet/number patterns via `regex`), and block grouping.
3. **Emission**: writes semantic **HTML** (h1/h2/p/ul/li) with simple CSS.
4. **Rendering**: optional conversion to PDF via **WeasyPrint** or **PrinceXML**.

## Installation details

### Python dependencies

* `pymupdf` (a.k.a. `fitz`)
* `regex`
* `numpy`
* `weasyprint` *(optional, for PDF via Cairo/Pango)*
* PrinceXML *(optional, recommended for robust PDF/UA)*

### macOS system packages (for WeasyPrint)

Install via Homebrew:

```bash
brew install cairo pango gdk-pixbuf libffi gobject-introspection pkg-config
```

If you see:

```
OSError: cannot load library 'libgobject-2.0-0'
```

either ensure the Homebrew libs are visible to the dynamic loader **or** use:

```bash
python format_pdf.py input.pdf out.html --pdf out.pdf --engine prince
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'frontend'` after `import fitz`**
You likely installed the wrong package (`fitz`). Uninstall and install **PyMuPDF**:

```bash
python -m pip uninstall -y fitz
python -m pip install pymupdf
```

**WeasyPrint native lib errors (gobject/cairo/pango)**
Install Homebrew packages above, or switch to `--engine prince`.

**`FileNotFoundError: no such file: 'input.pdf'`**
Check the path, working directory, and filename.

**Git push rejected (remote has README)**

```bash
git fetch origin
git pull --rebase origin main
git push -u origin main
```

## Accessibility goals

* Generate **semantically tagged** output to work better with screen readers.
* Improve results in automated checkers (e.g., PAC, Acrobat’s checker).
* Provide a foundation for adding landmarks, alt text hooks, and tables of contents.

## Roadmap

* Heading-level tuning rules per font/weight.
* Table detection and tagging.
* Landmark roles and language metadata.
* Image alt-text stubs + OCR text mapping.

## Project layout

```
format_pdf.py                   # Main CLI for HTML/PDF output
step3_structure_from_pdf.py     # Structure analysis / HTML prototype
OCRScanner.py                   # OCR helper (future integration/usage)
Fascimile_fromatter.py          # Facsimile utilities (typo kept from original)
Test.pdf / input_ocr.pdf        # Sample inputs (optional)
out*.html / out*.pdf            # Generated artifacts (ignored in .gitignore)
```

## .gitignore (suggested)

```gitignore
# macOS / IDE
.DS_Store
.idea/

# Python
.venv/
__pycache__/
*.pyc

# Outputs
out*.html
out*.pdf
*.pdf
*.log
```

