#!/usr/bin/env bash
# Local setup for TagMyPDF (macOS or Debian/Ubuntu Linux).
# Installs Tesseract OCR (English + Spanish), creates a Python 3.12
# virtual environment in .venv, and installs the Python dependencies.
set -euo pipefail

cd "$(dirname "$0")"

# 1. System packages
if [[ "$(uname)" == "Darwin" ]]; then
    command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }
    brew install python@3.12 tesseract tesseract-lang git-lfs
elif command -v apt-get >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv git-lfs \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa libgl1 libglib2.0-0
else
    echo "Unsupported system: install Python 3.12, Tesseract (eng+spa) and git-lfs manually, or use Docker."
    exit 1
fi

# 2. Model weights (stored with Git LFS)
git lfs install
git lfs pull
if [[ $(wc -c < yolov8n-doclaynet.pt) -lt 1000000 ]]; then
    echo "yolov8n-doclaynet.pt is still an LFS pointer; run 'git lfs pull' and retry."
    exit 1
fi

# 3. Python environment
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 4. Sanity check: OCR must be available, or image-only pages produce no text
.venv/bin/python -c "import pytesseract; print('Tesseract', pytesseract.get_tesseract_version())"
tesseract --list-langs | grep -q '^spa$' || echo "Warning: Spanish OCR data (spa) not found."

echo
echo "Setup complete. Run:  .venv/bin/python TagMyPDF2.py path/to/input.pdf"
