# !/usr/bin/env python3
"""
ExportLines_FIXED_v2.py
FIXED: Safer Unicode cleaning that preserves word boundaries
INTEGRATED: Google Vision OCR (toggle-enabled)
"""
import os, sys, json
from collections import Counter, defaultdict
from statistics import median, StatisticsError
import unicodedata
from google_ocr_module import run_google_ocr
import fitz  # PyMuPDF

# --- Optional OCR deps ---
try:
    import pytesseract
    from pytesseract import Output
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

DPI_FOR_OCR = 300
USE_GOOGLE_OCR = True  # Toggle this to enable Google Vision OCR instead of pytesseract


def clean_text_for_pdf(text):
    if not text:
        return text
    text = text.replace('\x00', '').replace('\u0000', '').replace('\ufeff', '').replace('\ufffe', '')
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if category in ('Cc', 'Cf', 'Cs', 'Co', 'Cn'):
            if char in ('\n', '\t', '\r'):
                cleaned.append(char)
            continue
        cleaned.append(char)
    return ''.join(cleaned)


def dedupe_lines(lines, y_bucket=2.0):
    groups = {}
    order = []
    def norm_text(t): return " ".join((t or "").split())
    def height(bbox): return float(bbox[3]) - float(bbox[1]) if bbox and len(bbox) == 4 else 1e9
    for ln in lines:
        txt = norm_text(ln.get("text", ""))
        if not txt:
            key = ("", id(ln))
            groups[key] = ln
            order.append(key)
            continue
        bbox = ln.get("bbox") or [0, 0, 0, 0]
        y_mid = 0.5 * (float(bbox[1]) + float(bbox[3])) if len(bbox) == 4 else 0.0
        y_key = round(y_mid / y_bucket)
        key = (txt, y_key)
        if key not in groups:
            groups[key] = ln
            order.append(key)
        else:
            old = groups[key]
            if height(ln.get("bbox")) < height(old.get("bbox")):
                groups[key] = ln
    return [groups[k] for k in order]


def page_lines_from_google_ocr(image_path):
    results = run_google_ocr(image_path)
    lines = []
    for r in results:
        txt = clean_text_for_pdf(r['text'])
        if not txt:
            continue
        lines.append({
            'text': txt,
            'bbox': r['bbox'],
            'size': 10.0,
            'bold': False
        })
    return lines


def page_lines_from_pytesseract(page):
    if not OCR_AVAILABLE:
        return []
    scale = DPI_FOR_OCR / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    data = pytesseract.image_to_data(img, lang='eng+spa', output_type=Output.DICT)
    lines_dict = {}
    n = len(data['text'])
    for i in range(n):
        txt = data['text'][i].strip()
        if not txt or data['conf'][i] < 30:
            continue
        txt = clean_text_for_pdf(txt)
        if not txt:
            continue
        line_num = data['line_num'][i]
        block_num = data['block_num'][i]
        key = (block_num, line_num)
        if key not in lines_dict:
            lines_dict[key] = {
                'words': [], 'x0': float('inf'), 'y0': float('inf'), 'x1': float('-inf'), 'y1': float('-inf')
            }
        l = lines_dict[key]
        l['words'].append(txt)
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        x0 = x * 72.0 / DPI_FOR_OCR
        y0 = y * 72.0 / DPI_FOR_OCR
        x1 = (x + w) * 72.0 / DPI_FOR_OCR
        y1 = (y + h) * 72.0 / DPI_FOR_OCR
        l['x0'] = min(l['x0'], x0)
        l['y0'] = min(l['y0'], y0)
        l['x1'] = max(l['x1'], x1)
        l['y1'] = max(l['y1'], y1)
    ocr_lines = []
    for key, l in lines_dict.items():
        txt = ' '.join(l['words'])
        txt = clean_text_for_pdf(txt)
        if not txt:
            continue
        ocr_lines.append({
            'text': txt,
            'bbox': [l['x0'], l['y0'], l['x1'], l['y1']],
            'size': 10.0,
            'bold': False
        })
    return ocr_lines


def estimate_body_size(lines):
    if not lines:
        return 10.0
    sizes = [ln.get('size', 10.0) for ln in lines if ln.get('size', 0) > 0]
    try:
        return median(sizes)
    except StatisticsError:
        return 10.0


def main(pdf_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    print(f"Processing: {pdf_path}")
    print(f"  Pages: {len(doc)}")
    print(f"  OCR available: {OCR_AVAILABLE}")

    for pnum, page in enumerate(doc, start=1):
        image_path = os.path.join(out_dir, f"page_{pnum:03d}.png")
        scale = DPI_FOR_OCR / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(image_path)

        if USE_GOOGLE_OCR:
            lines = page_lines_from_google_ocr(image_path)
        else:
            lines = page_lines_from_pytesseract(page)

        lines = dedupe_lines(lines)
        body_size = estimate_body_size(lines)
        output = {'page': pnum, 'body_size': body_size, 'lines': lines}
        out_path = os.path.join(out_dir, f'page_{pnum:03d}.lines.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  Page {pnum}: {len(lines)} lines, body_size={body_size:.1f}pt")

    print(f"✅ Done. Lines exported to: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ExportLines_FIXED_v2.py input.pdf output_dir")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
