# #!/usr/bin/env python3
# import os, sys, json
# from collections import Counter, defaultdict
# from statistics import median, StatisticsError
#
# import fitz  # PyMuPDF
#
# # --- Optional OCR deps (only used if no text layer is present) ---
# try:
#     import pytesseract
#     from pytesseract import Output
#     from PIL import Image
#     OCR_AVAILABLE = True
# except Exception:
#     OCR_AVAILABLE = False
#
# DPI_FOR_OCR = 300  # render resolution for OCR fallback
#
#
# def dedupe_lines(lines, y_bucket=2.0):
#     """
#     Deduplicate line objects that are effectively the same visual line.
#
#     Group by:
#       - normalized text (whitespace collapsed)
#       - y-position bucket (so tiny y differences still match)
#
#     Keep the 'better' line per group:
#       - prefer *shorter* bbox height (tighter box -> more likely real text line,
#         avoids giant block-level or hidden OCR boxes)
#     """
#     groups = {}
#     order = []
#
#     def norm_text(t):
#         return " ".join((t or "").split())
#
#     def height(bbox):
#         if not bbox or len(bbox) != 4:
#             # treat missing/zero bbox as "very large" so real boxes win
#             return 1e9
#         return float(bbox[3]) - float(bbox[1])
#
#     for ln in lines:
#         txt = norm_text(ln.get("text", ""))
#         if not txt:
#             # keep weird empty lines individually (or drop them if you want)
#             key = ("", id(ln))
#             groups[key] = ln
#             order.append(key)
#             continue
#
#         bbox = ln.get("bbox") or [0, 0, 0, 0]
#         if len(bbox) == 4:
#             y_mid = 0.5 * (float(bbox[1]) + float(bbox[3]))
#         else:
#             y_mid = 0.0
#
#         # bucket y to tolerate small coordinate differences
#         y_key = round(y_mid / y_bucket)
#
#         key = (txt, y_key)
#         if key not in groups:
#             groups[key] = ln
#             order.append(key)
#         else:
#             old = groups[key]
#             # keep the tighter box = *smaller* height
#             if height(ln.get("bbox")) < height(old.get("bbox")):
#                 groups[key] = ln
#
#     # preserve approximate reading order
#     return [groups[k] for k in order]
#
#
# def page_lines_from_text_layer(page):
#     """Extract line objects using PyMuPDF text spans/words (digital PDFs)."""
#     words = page.get_text("words")  # x0,y0,x1,y1,text,block,line,word
#     d = page.get_text("dict")
#     span_lines = []
#
#     for b in d.get("blocks", []):
#         if b.get("type") != 0:  # text blocks only
#             continue
#         for l in b.get("lines", []):
#             spans = l.get("spans", [])
#             if not spans:
#                 continue
#
#             x0 = min(s["bbox"][0] for s in spans)
#             y0 = min(s["bbox"][1] for s in spans)
#             x1 = max(s["bbox"][2] for s in spans)
#             y1 = max(s["bbox"][3] for s in spans)
#
#             # collect words overlapping this span-bbox
#             line_words = [
#                 w for w in words
#                 if (w[0] <= x1 + 1 and w[2] >= x0 - 1 and
#                     w[1] <= y1 + 1 and w[3] >= y0 - 1)
#             ]
#             line_words.sort(key=lambda w: (w[1], w[0]))
#             txt = " ".join(w[4] for w in line_words).strip()
#             if not txt:
#                 continue
#
#             sizes = [s.get("size", 0) for s in spans]
#             fonts = [s.get("font", "") for s in spans]
#             bold = any(("Bold" in f) or ("Semibold" in f) or ("Demi" in f) for f in fonts)
#
#             span_lines.append({
#                 "text": txt,
#                 "bbox": [float(x0), float(y0), float(x1), float(y1)],
#                 "size": float(median(sizes)) if sizes else 10.0,
#                 "bold": bool(bold),
#             })
#
#     return span_lines
#
#
# def page_lines_from_ocr(page):
#     """OCR fallback using Tesseract; returns line objects approximated from OCR data."""
#     if not OCR_AVAILABLE:
#         return []
#
#     # Render at DPI_FOR_OCR to get a crisp bitmap for OCR
#     scale = DPI_FOR_OCR / 72.0
#     mat = fitz.Matrix(scale, scale)
#     pix = page.get_pixmap(matrix=mat, alpha=False)
#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#
#     data = pytesseract.image_to_data(img, output_type=Output.DICT)  # word-level boxes
#     n = len(data["text"])
#     if n == 0:
#         return []
#
#     # Group words into lines using (block_num, par_num, line_num)
#     groups = defaultdict(list)
#     for i in range(n):
#         txt = (data["text"][i] or "").strip()
#         conf = data.get("conf", ["-1"] * n)[i]
#         try:
#             conf = float(conf)
#         except Exception:
#             conf = -1.0
#
#         # skip empty/very low confidence noise
#         if not txt or conf < 0:
#             continue
#
#         key = (
#             data.get("block_num", [0] * n)[i],
#             data.get("par_num",   [0] * n)[i],
#             data.get("line_num",  [0] * n)[i],
#         )
#
#         left = data["left"][i]
#         top  = data["top"][i]
#         w    = data["width"][i]
#         h    = data["height"][i]
#         groups[key].append((left, top, w, h, txt))
#
#     # Convert pixel coords -> PDF points, build line objects
#     lines = []
#     for _, words in groups.items():
#         if not words:
#             continue
#
#         # order by reading order
#         words.sort(key=lambda x: (x[1], x[0]))
#         x0 = min(l for (l, t, w, h, txt) in words)
#         y0 = min(t for (l, t, w, h, txt) in words)
#         x1 = max(l + w for (l, t, w, h, txt) in words)
#         y1 = max(t + h for (l, t, w, h, txt) in words)
#
#         # scale back to PDF point coords
#         x0p, y0p, x1p, y1p = (x0 / scale, y0 / scale, x1 / scale, y1 / scale)
#         text = " ".join(txt for (_, _, _, _, txt) in words)
#
#         # Use line height in points as a proxy for font size
#         approx_size = max(8.0, (y1 - y0) / scale * 0.8)
#
#         lines.append({
#             "text": text.strip(),
#             "bbox": [float(x0p), float(y0p), float(x1p), float(y1p)],
#             "size": float(approx_size),
#             "bold": False,  # OCR cannot reliably flag bold; keep False
#         })
#
#     return lines
#
#
# def compute_body_size(lines):
#     """Robust body font size estimation with safe fallback."""
#     longs = [round(l["size"], 1) for l in lines if len(l["text"]) > 20]
#     if not longs:
#         longs = [round(l["size"], 1) for l in lines]
#
#     if not longs:
#         return 10.0
#
#     cnt = Counter(longs)
#     top = cnt.most_common(5)
#     try:
#         vals = [top[k][0] for k in range(min(5, len(top)))]
#         return float(median(vals))
#     except (StatisticsError, IndexError):
#         return float(median(longs)) if longs else 10.0
#
#
# def main(pdf_path, out_dir="lines_out"):
#     os.makedirs(out_dir, exist_ok=True)
#     doc = fitz.open(pdf_path)
#
#     for i, page in enumerate(doc, start=1):
#         # 1) Try digital text layer first
#         lines_text = page_lines_from_text_layer(page)
#
#         # 2) If text layer has anything, ignore OCR completely
#         if lines_text:
#             lines = lines_text
#         else:
#             print(f"⚠️  Page {i}: No text layer — using OCR fallback")
#             ocr_lines = page_lines_from_ocr(page)
#             if not ocr_lines and not OCR_AVAILABLE:
#                 print(f"❌  Page {i}: OCR not available (install tesseract + pytesseract + pillow)")
#             lines = ocr_lines
#
#         # 3) Dedupe lines (removes duplicate text or giant block vs tight line)
#         lines = dedupe_lines(lines)
#
#         # 4) Build page JSON
#         if not lines:
#             page_obj = {
#                 "file": os.path.basename(pdf_path),
#                 "page": i,
#                 "body_size": 10.0,
#                 "lines": []
#             }
#         else:
#             body = compute_body_size(lines)
#             page_obj = {
#                 "file": os.path.basename(pdf_path),
#                 "page": i,
#                 "body_size": float(body),
#                 "lines": lines
#             }
#
#         op = os.path.join(out_dir, f"page_{i:03d}.lines.json")
#         with open(op, "w", encoding="utf-8") as f:
#             json.dump(page_obj, f, indent=2)
#         print("wrote:", op)
#
#
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python ExportLines.py input.pdf [out_dir]")
#         sys.exit(1)
#
#     pdf = sys.argv[1]
#     out_dir = sys.argv[2] if len(sys.argv) > 2 else "lines_out"
#     main(pdf, out_dir)


# !/usr/bin/env python3
"""
ExportLines.py - EasyOCR Version
Python 3.12 compatible, excellent accuracy, automatic multi-language support
"""
import os, sys, json
from collections import defaultdict
from statistics import median

import fitz  # PyMuPDF

# --- OCR with EasyOCR ---
try:
    import easyocr
    import cv2
    import numpy as np
    from PIL import Image

    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

DPI_FOR_OCR = 300

# Initialize EasyOCR (only once, reused for all pages)
_easyocr_reader = None


def get_easyocr_reader(languages=['en']):
    """
    Initialize EasyOCR reader with specified languages.

    Common language codes:
    - 'en': English
    - 'es': Spanish
    - 'fr': French
    - 'de': German
    - 'it': Italian
    - 'pt': Portuguese
    - 'ru': Russian
    - 'ch_sim': Chinese Simplified
    - 'ja': Japanese
    - 'ko': Korean
    - 'ar': Arabic

    Full list: https://www.jaided.ai/easyocr/
    """
    global _easyocr_reader
    if _easyocr_reader is None and EASYOCR_AVAILABLE:
        print(f"[EasyOCR] Initializing with languages: {languages}")
        _easyocr_reader = easyocr.Reader(
            languages,
            gpu=False,  # Set to True if you have CUDA GPU
            verbose=False
        )
    return _easyocr_reader


def preprocess_image_for_ocr(pil_img):
    """Light preprocessing for EasyOCR (it handles most internally)"""
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Light denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Slight contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Convert back to RGB (EasyOCR expects RGB)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return rgb


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

            sizes = [s.get("size", 0) for s in spans if s.get("size", 0) > 0]
            approx_size = median(sizes) if sizes else 12.0

            txt = " ".join(s["text"] for s in spans)
            span_lines.append({
                "text": txt,
                "bbox": [x0, y0, x1, y1],
                "size": approx_size
            })

    return span_lines


def page_lines_from_easyocr(page, languages=['en', 'es', 'fr', 'de']):
    """
    EasyOCR-based text extraction.

    Advantages:
    - Python 3.12 compatible ✅
    - Excellent accuracy (85-90%)
    - Automatic multi-language detection
    - Handles rotated text well
    - No language pack installation
    - Clean, simple API

    Args:
        page: PyMuPDF page object
        languages: List of language codes (e.g., ['en', 'es', 'fr'])
    """
    if not EASYOCR_AVAILABLE:
        print("[OCR] EasyOCR not available, skipping OCR")
        return []

    reader = get_easyocr_reader(languages)
    if reader is None:
        return []

    # Render page to image
    scale = DPI_FOR_OCR / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Light preprocessing
    img_np = preprocess_image_for_ocr(img)

    try:
        # EasyOCR returns: [([box_points], text, confidence), ...]
        results = reader.readtext(
            img_np,
            detail=1,  # Return detailed results (box, text, confidence)
            paragraph=False,  # Return individual text blocks, not paragraphs
            min_size=10,  # Minimum text size to detect
            text_threshold=0.5,  # Text detection confidence threshold
            low_text=0.4,  # Lower bound for text regions
            link_threshold=0.4,  # Link text boxes threshold
            canvas_size=2560,  # Max image dimension
            mag_ratio=1.0  # Image magnification ratio
        )

        if not results:
            return []

        # Sort by Y-coordinate (top to bottom), then X (left to right)
        results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))

        # Group into lines (boxes with similar Y-coordinates)
        lines = []
        current_line = []
        current_y = None
        y_threshold = 15.0 / scale  # Lines within this distance are grouped

        for box_points, text, confidence in results:
            # box_points is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [p[0] for p in box_points]
            y_coords = [p[1] for p in box_points]
            x0, y0 = min(x_coords), min(y_coords)
            x1, y1 = max(x_coords), max(y_coords)

            # Convert pixel coords to PDF points
            x0p, y0p = x0 / scale, y0 / scale
            x1p, y1p = x1 / scale, y1 / scale

            # Skip low confidence results
            if confidence < 0.5:  # 50% threshold
                continue

            # Group into lines by Y-coordinate
            if current_y is None or abs(y0p - current_y) < y_threshold:
                current_line.append({
                    'text': text,
                    'bbox': [x0p, y0p, x1p, y1p],
                    'confidence': confidence
                })
                current_y = y0p
            else:
                # Finalize current line
                if current_line:
                    lines.append(merge_line_boxes(current_line))
                # Start new line
                current_line = [{
                    'text': text,
                    'bbox': [x0p, y0p, x1p, y1p],
                    'confidence': confidence
                }]
                current_y = y0p

        # Add last line
        if current_line:
            lines.append(merge_line_boxes(current_line))

        return lines

    except Exception as e:
        print(f"[OCR] EasyOCR error: {e}")
        return []


def merge_line_boxes(boxes):
    """Merge multiple text boxes into a single line object"""
    if not boxes:
        return None

    # Sort boxes left to right
    boxes = sorted(boxes, key=lambda b: b['bbox'][0])

    # Merge text with spaces
    text = " ".join(b['text'] for b in boxes)

    # Calculate bounding box
    x0 = min(b['bbox'][0] for b in boxes)
    y0 = min(b['bbox'][1] for b in boxes)
    x1 = max(b['bbox'][2] for b in boxes)
    y1 = max(b['bbox'][3] for b in boxes)

    # Calculate average confidence
    avg_conf = sum(b['confidence'] for b in boxes) / len(boxes)

    # Estimate font size from height
    height = y1 - y0
    approx_size = max(8.0, height * 0.8)

    return {
        'text': text,
        'bbox': [x0, y0, x1, y1],
        'size': approx_size,
        'confidence': avg_conf
    }


def page_lines(page, languages=['en', 'es', 'fr', 'de']):
    """
    Main entry: prefer text layer if present, else use EasyOCR.

    Args:
        page: PyMuPDF page object
        languages: List of language codes for OCR
    """
    lines = page_lines_from_text_layer(page)
    if lines:
        return dedupe_lines(lines)

    # Fallback to EasyOCR
    ocr_lines = page_lines_from_easyocr(page, languages)
    return dedupe_lines(ocr_lines)


def export_lines_json(pdf_path, out_dir, languages=['en', 'es']):
    """
    Export lines for each page to JSON files.

    Args:
        pdf_path: Path to input PDF
        out_dir: Output directory for JSON files
        languages: List of language codes for OCR
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    # SAVE PAGE COUNT BEFORE PROCESSING
    num_pages = len(doc)

    print(f"Processing {num_pages} pages with {'EasyOCR' if EASYOCR_AVAILABLE else 'text extraction only'}...")
    if EASYOCR_AVAILABLE:
        print(f"OCR languages: {languages}")

    for i, page in enumerate(doc, start=1):
        print(f"  Page {i}/{num_pages}...", end="\r")
        lines = page_lines(page, languages)
        out_path = os.path.join(out_dir, f"page_{i:03d}.lines.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"page": i, "lines": lines}, f, indent=2, ensure_ascii=False)

    doc.close()
    # USE SAVED num_pages INSTEAD OF len(doc)
    print(f"\nExported {num_pages} page(s) to {out_dir}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python ExportLines.py input.pdf output_lines_dir [languages]")
        print("  languages (optional): Comma-separated list (e.g., 'en,es,fr')")
        print("  Default: en,es,fr,de")
        sys.exit(1)

    pdf_in = sys.argv[1]
    lines_dir = sys.argv[2]

    # Parse languages
    if len(sys.argv) > 3:
        languages = [lang.strip() for lang in sys.argv[3].split(',')]
    else:
        languages = ['en', 'es', 'fr', 'de']  # Default: English, Spanish, French, German

    if not EASYOCR_AVAILABLE:
        print("WARNING: EasyOCR not installed. Install with: pip install easyocr")
        print("Falling back to text layer extraction only (no OCR for scanned pages)")

    export_lines_json(pdf_in, lines_dir, languages)


if __name__ == "__main__":
    main()