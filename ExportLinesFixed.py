#########THIS FILE IS CARRYING SOME EXTRA EXPERIMENTAL CODE
"""
ExportLines_FINAL.py
FINAL FIX: Rounds y-coordinates to group words on same line before sorting
"""
import os
import sys
import json
from collections import defaultdict
from statistics import median, StatisticsError
import unicodedata
import re

import fitz  # PyMuPDF


def clean_text_for_pdf(text):
    """Remove invalid Unicode, preserve word boundaries."""
    if not text:
        return text

    text = text.replace('\x00', '').replace('\u0000', '')
    text = text.replace('\ufeff', '').replace('\ufffe', '')
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')

    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if category in ('Cc', 'Cf', 'Cs', 'Co', 'Cn'):
            if char in ('\n', '\t', '\r', ' '):
                cleaned.append(char)
            continue
        cleaned.append(char)

    result = ''.join(cleaned)
    result = re.sub(r' {4,}', '  ', result)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result


def page_lines_from_text_layer(page):
    """
    Extract lines with proper word ordering.

    KEY FIX: Within each line bbox from PyMuPDF, sort words by x-position ONLY.
    Don't re-sort by y-position as that scrambles words with slight vertical misalignment.
    """
    words = page.get_text("words")
    d = page.get_text("dict")
    span_lines = []

    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue

        for l in b.get("lines", []):
            spans = l.get("spans", [])
            if not spans:
                continue

            # Get line bounding box
            x0 = min(s["bbox"][0] for s in spans)
            y0 = min(s["bbox"][1] for s in spans)
            x1 = max(s["bbox"][2] for s in spans)
            y1 = max(s["bbox"][3] for s in spans)

            # Find words in this line's bbox
            line_words = []
            for w in words:
                wx0, wy0, wx1, wy1 = w[0], w[1], w[2], w[3]

                # Use generous vertical tolerance to catch all words in line
                if (wx0 <= x1 + 2 and wx1 >= x0 - 2 and
                        wy0 <= y1 + 2 and wy1 >= y0 - 2):
                    line_words.append(w)

            if not line_words:
                continue

            # ✅ CRITICAL FIX: Sort ONLY by x-position (left to right)
            # PyMuPDF already grouped these words into the correct line
            # Sorting by y would scramble words with slight vertical misalignment
            line_words.sort(key=lambda w: w[0])  # Sort by x0 only!

            # Build text with spacing based on gaps
            text_parts = []
            prev_x1 = None

            for w in line_words:
                word_text = w[4]
                word_x0 = w[0]

                if prev_x1 is not None:
                    gap = word_x0 - prev_x1
                    if gap > 15:  # Large gap (column break)
                        text_parts.append("  ")
                    elif gap > 2:  # Normal word spacing
                        text_parts.append(" ")

                text_parts.append(word_text)
                prev_x1 = w[2]

            txt = "".join(text_parts).strip()
            txt = clean_text_for_pdf(txt)

            if not txt:
                continue

            # Extract font info
            sizes = [s.get("size", 0) for s in spans]
            fonts = [s.get("font", "") for s in spans]
            bold = any(("Bold" in f) or ("Semibold" in f) or ("Demi" in f) for f in fonts)

            span_lines.append({
                "text": txt,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "size": float(median(sizes)) if sizes else 10.0,
                "bold": bool(bold),
            })

    return span_lines


def dedupe_lines(lines, y_bucket=2.0):
    """Deduplicate lines."""
    groups = {}
    order = []

    def norm_text(t):
        return " ".join((t or "").split())

    def height(bbox):
        if not bbox or len(bbox) != 4:
            return 1e9
        return float(bbox[3]) - float(bbox[1])

    for ln in lines:
        txt = norm_text(ln.get("text", ""))
        if not txt:
            key = ("", id(ln))
            groups[key] = ln
            order.append(key)
            continue

        bbox = ln.get("bbox") or [0, 0, 0, 0]
        if len(bbox) == 4:
            y_mid = 0.5 * (float(bbox[1]) + float(bbox[3]))
        else:
            y_mid = 0.0

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


def estimate_body_size(lines):
    """Estimate body text font size."""
    if not lines:
        return 10.0

    sizes = [ln.get('size', 10.0) for ln in lines if ln.get('size', 0) > 0]
    if not sizes:
        return 10.0

    try:
        return median(sizes)
    except StatisticsError:
        return 10.0


def main(pdf_path, out_dir):
    """Main entry point."""
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    print(f"Processing: {pdf_path}")
    print(f"  Pages: {len(doc)}")

    for pnum, page in enumerate(doc, start=1):
        lines = page_lines_from_text_layer(page)
        lines = dedupe_lines(lines)
        body_size = estimate_body_size(lines)

        output = {
            'page': pnum,
            'body_size': body_size,
            'lines': lines
        }

        out_path = os.path.join(out_dir, f'page_{pnum:03d}.lines.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  Page {pnum}: {len(lines)} lines, body_size={body_size:.1f}pt")

    print(f"✅ Done. Lines exported to: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 ExportLines_FINAL.py input.pdf output_dir")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])


# !/usr/bin/env python3
"""
ExportLines_FIXED_v2.py
FIXED: Safer Unicode cleaning that preserves word boundaries
"""
import os, sys, json
from collections import Counter, defaultdict
from statistics import median, StatisticsError
import unicodedata

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
DETECTION_DPI = 150


# ============================================================================
# ✅ FIXED: SAFER TEXT CLEANING (Preserves word boundaries)
# ============================================================================
def clean_text_for_pdf(text):
    """
    Remove ONLY truly invalid Unicode characters.

    SAFE version - preserves word boundaries and normal spaces.
    Only removes characters that cause PDF/UA validation failures.

    Fixes veraPDF errors:
    - "The document contains a reference to the .notdef glyph"
    - "The glyph has Unicode value 0, U+FEFF or U+FFFE"
    """
    if not text:
        return text

    # Remove null bytes (these are always invalid)
    text = text.replace('\x00', '')
    text = text.replace('\u0000', '')

    # Remove BOM markers (invisible, cause validation errors)
    text = text.replace('\ufeff', '')
    text = text.replace('\ufffe', '')

    # Remove zero-width characters (invisible formatting)
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\u200c', '')  # Zero-width non-joiner
    text = text.replace('\u200d', '')  # Zero-width joiner

    # ✅ FIXED: Only filter out truly problematic categories
    # Do NOT touch normal spaces, newlines, or regular characters
    cleaned = []
    for char in text:
        category = unicodedata.category(char)

        # ✅ KEEP all normal characters including spaces (Zs category)
        # Only remove control/format/surrogate/private/unassigned
        if category in ('Cc', 'Cf', 'Cs', 'Co', 'Cn'):
            # Exception: keep tabs, newlines, carriage returns
            if char in ('\n', '\t', '\r'):
                cleaned.append(char)
            # Skip all other control characters
            continue

        # Keep everything else (including NORMAL SPACES)
        cleaned.append(char)

    result = ''.join(cleaned)

    # ✅ REMOVED: Do NOT normalize whitespace!
    # This was breaking word boundaries
    # Old code: result = ' '.join(result.split())

    return result


def dedupe_lines(lines, y_bucket=2.0):
    """Deduplicate line objects that are effectively the same visual line."""
    groups = {}
    order = []

    def norm_text(t):
        return " ".join((t or "").split())

    def height(bbox):
        if not bbox or len(bbox) != 4:
            return 1e9
        return float(bbox[3]) - float(bbox[1])

    for ln in lines:
        txt = norm_text(ln.get("text", ""))
        if not txt:
            key = ("", id(ln))
            groups[key] = ln
            order.append(key)
            continue

        bbox = ln.get("bbox") or [0, 0, 0, 0]
        if len(bbox) == 4:
            y_mid = 0.5 * (float(bbox[1]) + float(bbox[3]))
        else:
            y_mid = 0.0

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


def page_lines_from_text_layer(page):
    """Extract line objects using PyMuPDF text spans/words (digital PDFs)."""
    words = page.get_text("words")
    d = page.get_text("dict")
    span_lines = []

    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for l in b.get("lines", []):
            spans = l.get("spans", [])
            if not spans:
                continue

            x0 = min(s["bbox"][0] for s in spans)
            y0 = min(s["bbox"][1] for s in spans)
            x1 = max(s["bbox"][2] for s in spans)
            y1 = max(s["bbox"][3] for s in spans)

            line_words = [
                w for w in words
                if (w[0] <= x1 + 1 and w[2] >= x0 - 1 and
                    w[1] <= y1 + 1 and w[3] >= y0 - 1)
            ]
            # Sort ONLY left-to-right to avoid y-jitter scrambling
            line_words.sort(key=lambda w: w[0])

            # Build text with spacing based on horizontal gaps (preserves columns better)
            text_parts = []
            prev_x1 = None
            for w in line_words:
                word_text = w[4]
                word_x0 = w[0]

                if prev_x1 is not None:
                    gap = word_x0 - prev_x1
                    if gap > 15:
                        text_parts.append("  ")  # column-ish break
                    elif gap > 2:
                        text_parts.append(" ")

                text_parts.append(word_text)
                prev_x1 = w[2]

            txt = "".join(text_parts).strip()

            # ✅ APPLY SAFER TEXT CLEANING
            txt = clean_text_for_pdf(txt)

            if not txt:
                continue

            sizes = [s.get("size", 0) for s in spans]
            fonts = [s.get("font", "") for s in spans]
            bold = any(("Bold" in f) or ("Semibold" in f) or ("Demi" in f) for f in fonts)

            span_lines.append({
                "text": txt,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "size": float(median(sizes)) if sizes else 10.0,
                "bold": bool(bold),
            })

    return span_lines


def detect_language_fast(page):
    """Fast language detection using low-res sample OCR."""
    if not OCR_AVAILABLE:
        return 'eng'

    try:
        scale = DETECTION_DPI / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        height = img.height
        cropped = img.crop((0, 0, img.width, int(height * 0.3)))

        try:
            osd = pytesseract.image_to_osd(cropped)
            if 'Script:' in osd:
                for line in osd.split('\n'):
                    if 'Script:' in line:
                        script = line.split(':')[1].strip()
                        if 'Latin' in script:
                            quick_text = pytesseract.image_to_string(
                                cropped,
                                lang='eng+spa',
                                config='--psm 6'
                            )[:200]

                            spanish_chars = set('áéíóúñÁÉÍÓÚÑ¿¡')
                            spanish_words = {'de', 'el', 'la', 'en', 'y', 'del', 'los', 'las', 'un', 'una'}

                            if any(c in quick_text for c in spanish_chars):
                                return 'spa'

                            words = quick_text.lower().split()
                            spanish_count = sum(1 for w in words if w in spanish_words)
                            if spanish_count > len(words) * 0.3:
                                return 'spa'

                            return 'eng'
        except:
            pass

        return 'eng+spa'

    except Exception as e:
        print(f"   Language detection failed: {e}, defaulting to eng")
        return 'eng'


def page_lines_from_ocr(page, lang='eng'):
    """OCR fallback using Tesseract with specified language."""
    if not OCR_AVAILABLE:
        return []

    scale = DPI_FOR_OCR / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    try:
        data = pytesseract.image_to_data(
            img,
            lang=lang,
            output_type=Output.DICT,
            config='--psm 3 --oem 1'
        )
    except Exception as e:
        print(f"   OCR error: {e}")
        return []

    lines_dict = {}
    n = len(data['text'])

    for i in range(n):
        txt = data['text'][i].strip()
        if not txt or data['conf'][i] < 30:
            continue

        # ✅ APPLY SAFER TEXT CLEANING TO OCR OUTPUT
        txt = clean_text_for_pdf(txt)
        if not txt:
            continue

        line_num = data['line_num'][i]
        block_num = data['block_num'][i]
        key = (block_num, line_num)

        if key not in lines_dict:
            lines_dict[key] = {
                'words': [],
                'x0': float('inf'),
                'y0': float('inf'),
                'x1': float('-inf'),
                'y1': float('-inf')
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

        # ✅ FINAL CLEANING CHECK
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
    """Estimate the most common body text font size."""
    if not lines:
        return 10.0

    sizes = [ln.get('size', 10.0) for ln in lines if ln.get('size', 0) > 0]
    if not sizes:
        return 10.0

    try:
        return median(sizes)
    except StatisticsError:
        return 10.0


def main(pdf_path, out_dir):
    """Main entry point."""
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    print(f"Processing: {pdf_path}")
    print(f"  Pages: {len(doc)}")
    print(f"  OCR available: {OCR_AVAILABLE}")

    for pnum, page in enumerate(doc, start=1):
        # Try text layer first
        lines = page_lines_from_text_layer(page)

        # Fallback to OCR if needed
        if len(lines) < 5:
            print(f"  Page {pnum}: Sparse text layer ({len(lines)} lines), using OCR...")
            lang = detect_language_fast(page)
            print(f"    Detected language: {lang}")
            ocr_lines = page_lines_from_ocr(page, lang=lang)
            lines.extend(ocr_lines)

        # Deduplicate
        lines = dedupe_lines(lines)

        # Estimate body size
        body_size = estimate_body_size(lines)

        # Save
        output = {
            'page': pnum,
            'body_size': body_size,
            'lines': lines
        }

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

