#!/usr/bin/env python3
import os, sys, json
from collections import Counter, defaultdict
from statistics import median, StatisticsError

import fitz  # PyMuPDF

# --- Optional OCR deps (only used if no text layer is present) ---
try:
    import pytesseract
    from pytesseract import Output
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

DPI_FOR_OCR = 300  # render resolution for OCR fallback


def dedupe_lines(lines, y_bucket=2.0):
    """
    Deduplicate line objects that are effectively the same visual line.

    Group by:
      - normalized text (whitespace collapsed)
      - y-position bucket (so tiny y differences still match)

    Keep the 'better' line per group:
      - prefer *shorter* bbox height (tighter box -> more likely real text line,
        avoids giant block-level or hidden OCR boxes)
    """
    groups = {}
    order = []

    def norm_text(t):
        return " ".join((t or "").split())

    def height(bbox):
        if not bbox or len(bbox) != 4:
            # treat missing/zero bbox as "very large" so real boxes win
            return 1e9
        return float(bbox[3]) - float(bbox[1])

    for ln in lines:
        txt = norm_text(ln.get("text", ""))
        if not txt:
            # keep weird empty lines individually (or drop them if you want)
            key = ("", id(ln))
            groups[key] = ln
            order.append(key)
            continue

        bbox = ln.get("bbox") or [0, 0, 0, 0]
        if len(bbox) == 4:
            y_mid = 0.5 * (float(bbox[1]) + float(bbox[3]))
        else:
            y_mid = 0.0

        # bucket y to tolerate small coordinate differences
        y_key = round(y_mid / y_bucket)

        key = (txt, y_key)
        if key not in groups:
            groups[key] = ln
            order.append(key)
        else:
            old = groups[key]
            # keep the tighter box = *smaller* height
            if height(ln.get("bbox")) < height(old.get("bbox")):
                groups[key] = ln

    # preserve approximate reading order
    return [groups[k] for k in order]


def page_lines_from_text_layer(page):
    """Extract line objects using PyMuPDF text spans/words (digital PDFs)."""
    words = page.get_text("words")  # x0,y0,x1,y1,text,block,line,word
    d = page.get_text("dict")
    span_lines = []

    for b in d.get("blocks", []):
        if b.get("type") != 0:  # text blocks only
            continue
        for l in b.get("lines", []):
            spans = l.get("spans", [])
            if not spans:
                continue

            x0 = min(s["bbox"][0] for s in spans)
            y0 = min(s["bbox"][1] for s in spans)
            x1 = max(s["bbox"][2] for s in spans)
            y1 = max(s["bbox"][3] for s in spans)

            # collect words overlapping this span-bbox
            line_words = [
                w for w in words
                if (w[0] <= x1 + 1 and w[2] >= x0 - 1 and
                    w[1] <= y1 + 1 and w[3] >= y0 - 1)
            ]
            line_words.sort(key=lambda w: (w[1], w[0]))
            txt = " ".join(w[4] for w in line_words).strip()
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


def page_lines_from_ocr(page):
    """OCR fallback using Tesseract; returns line objects approximated from OCR data."""
    if not OCR_AVAILABLE:
        return []

    # Render at DPI_FOR_OCR to get a crisp bitmap for OCR
    scale = DPI_FOR_OCR / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    data = pytesseract.image_to_data(img, output_type=Output.DICT)  # word-level boxes
    n = len(data["text"])
    if n == 0:
        return []

    # Group words into lines using (block_num, par_num, line_num)
    groups = defaultdict(list)
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = data.get("conf", ["-1"] * n)[i]
        try:
            conf = float(conf)
        except Exception:
            conf = -1.0

        # skip empty/very low confidence noise
        if not txt or conf < 0:
            continue

        key = (
            data.get("block_num", [0] * n)[i],
            data.get("par_num",   [0] * n)[i],
            data.get("line_num",  [0] * n)[i],
        )

        left = data["left"][i]
        top  = data["top"][i]
        w    = data["width"][i]
        h    = data["height"][i]
        groups[key].append((left, top, w, h, txt))

    # Convert pixel coords -> PDF points, build line objects
    lines = []
    for _, words in groups.items():
        if not words:
            continue

        # order by reading order
        words.sort(key=lambda x: (x[1], x[0]))
        x0 = min(l for (l, t, w, h, txt) in words)
        y0 = min(t for (l, t, w, h, txt) in words)
        x1 = max(l + w for (l, t, w, h, txt) in words)
        y1 = max(t + h for (l, t, w, h, txt) in words)

        # scale back to PDF point coords
        x0p, y0p, x1p, y1p = (x0 / scale, y0 / scale, x1 / scale, y1 / scale)
        text = " ".join(txt for (_, _, _, _, txt) in words)

        # Use line height in points as a proxy for font size
        approx_size = max(8.0, (y1 - y0) / scale * 0.8)

        lines.append({
            "text": text.strip(),
            "bbox": [float(x0p), float(y0p), float(x1p), float(y1p)],
            "size": float(approx_size),
            "bold": False,  # OCR cannot reliably flag bold; keep False
        })

    return lines


def compute_body_size(lines):
    """Robust body font size estimation with safe fallback."""
    longs = [round(l["size"], 1) for l in lines if len(l["text"]) > 20]
    if not longs:
        longs = [round(l["size"], 1) for l in lines]

    if not longs:
        return 10.0

    cnt = Counter(longs)
    top = cnt.most_common(5)
    try:
        vals = [top[k][0] for k in range(min(5, len(top)))]
        return float(median(vals))
    except (StatisticsError, IndexError):
        return float(median(longs)) if longs else 10.0


def main(pdf_path, out_dir="lines_out"):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc, start=1):
        # 1) Try digital text layer first
        lines_text = page_lines_from_text_layer(page)

        # 2) If text layer has anything, ignore OCR completely
        if lines_text:
            lines = lines_text
        else:
            print(f"⚠️  Page {i}: No text layer — using OCR fallback")
            ocr_lines = page_lines_from_ocr(page)
            if not ocr_lines and not OCR_AVAILABLE:
                print(f"❌  Page {i}: OCR not available (install tesseract + pytesseract + pillow)")
            lines = ocr_lines

        # 3) Dedupe lines (removes duplicate text or giant block vs tight line)
        lines = dedupe_lines(lines)

        # 4) Build page JSON
        if not lines:
            page_obj = {
                "file": os.path.basename(pdf_path),
                "page": i,
                "body_size": 10.0,
                "lines": []
            }
        else:
            body = compute_body_size(lines)
            page_obj = {
                "file": os.path.basename(pdf_path),
                "page": i,
                "body_size": float(body),
                "lines": lines
            }

        op = os.path.join(out_dir, f"page_{i:03d}.lines.json")
        with open(op, "w", encoding="utf-8") as f:
            json.dump(page_obj, f, indent=2)
        print("wrote:", op)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ExportLines.py input.pdf [out_dir]")
        sys.exit(1)

    pdf = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "lines_out"
    main(pdf, out_dir)
