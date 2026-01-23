# # !/usr/bin/env python3
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
#
#     OCR_AVAILABLE = True
# except Exception:
#     OCR_AVAILABLE = False
#
# DPI_FOR_OCR = 200  # Reduced from 300 for faster processing
# DETECTION_DPI = 150  # Even lower DPI for language detection
#
#
# def dedupe_lines(lines, y_bucket=2.0):
#     """
#     Deduplicate line objects that are effectively the same visual line.
#     """
#     groups = {}
#     order = []
#
#     def norm_text(t):
#         return " ".join((t or "").split())
#
#     def height(bbox):
#         if not bbox or len(bbox) != 4:
#             return 1e9
#         return float(bbox[3]) - float(bbox[1])
#
#     for ln in lines:
#         txt = norm_text(ln.get("text", ""))
#         if not txt:
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
#         y_key = round(y_mid / y_bucket)
#         key = (txt, y_key)
#
#         if key not in groups:
#             groups[key] = ln
#             order.append(key)
#         else:
#             old = groups[key]
#             if height(ln.get("bbox")) < height(old.get("bbox")):
#                 groups[key] = ln
#
#     return [groups[k] for k in order]
#
#
# def page_lines_from_text_layer(page):
#     """Extract line objects using PyMuPDF text spans/words (digital PDFs)."""
#     words = page.get_text("words")
#     d = page.get_text("dict")
#     span_lines = []
#
#     for b in d.get("blocks", []):
#         if b.get("type") != 0:
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
# def detect_language_fast(page):
#     """
#     Fast language detection using low-res sample OCR.
#     Returns best guess language code.
#     """
#     if not OCR_AVAILABLE:
#         return 'eng'
#
#     try:
#         # Render at very low DPI for fast detection
#         scale = DETECTION_DPI / 72.0
#         mat = fitz.Matrix(scale, scale)
#         pix = page.get_pixmap(matrix=mat, alpha=False)
#
#         # Only OCR a small section (top 30% of page)
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         height = img.height
#         cropped = img.crop((0, 0, img.width, int(height * 0.3)))
#
#         # Quick OSD (Orientation and Script Detection) to detect language
#         try:
#             osd = pytesseract.image_to_osd(cropped)
#             # Parse script from OSD output
#             if 'Script:' in osd:
#                 for line in osd.split('\n'):
#                     if 'Script:' in line:
#                         script = line.split(':')[1].strip()
#                         if 'Latin' in script:
#                             # For Latin scripts, try quick detection with common languages
#                             quick_text = pytesseract.image_to_string(
#                                 cropped,
#                                 lang='eng+spa',
#                                 config='--psm 6'
#                             )[:200]
#
#                             # Simple heuristic
#                             spanish_chars = set('áéíóúñÁÉÍÓÚÑ¿¡')
#                             spanish_words = {'de', 'el', 'la', 'en', 'y', 'del', 'los', 'las', 'un', 'una'}
#
#                             if any(c in quick_text for c in spanish_chars):
#                                 return 'spa'
#
#                             words = quick_text.lower().split()
#                             spanish_count = sum(1 for w in words if w in spanish_words)
#                             if spanish_count > len(words) * 0.3:
#                                 return 'spa'
#
#                             return 'eng'
#         except:
#             pass
#
#         # Fallback: Try both English and Spanish
#         return 'eng+spa'
#
#     except Exception as e:
#         print(f"   Language detection failed: {e}, defaulting to eng")
#         return 'eng'
#
#
# def page_lines_from_ocr(page, lang='eng'):
#     """
#     OCR fallback using Tesseract with specified language.
#     Optimized for speed.
#     """
#     if not OCR_AVAILABLE:
#         return []
#
#     # Render at optimized DPI
#     scale = DPI_FOR_OCR / 72.0
#     mat = fitz.Matrix(scale, scale)
#     pix = page.get_pixmap(matrix=mat, alpha=False)
#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#
#     try:
#         # Use detected language with optimized config
#         data = pytesseract.image_to_data(
#             img,
#             lang=lang,
#             output_type=Output.DICT,
#             config='--psm 3 --oem 1'  # Faster: PSM 3 (auto), OEM 1 (LSTM only)
#         )
#     except pytesseract.TesseractError as e:
#         print(f"⚠️  OCR with lang={lang} failed, trying English: {e}")
#         try:
#             data = pytesseract.image_to_data(
#                 img,
#                 lang='eng',
#                 output_type=Output.DICT,
#                 config='--psm 3 --oem 1'
#             )
#         except Exception as e2:
#             print(f"❌ OCR failed completely: {e2}")
#             return []
#
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
#         if not txt or conf < 30:  # Increased threshold to filter more noise
#             continue
#
#         key = (
#             data.get("block_num", [0] * n)[i],
#             data.get("par_num", [0] * n)[i],
#             data.get("line_num", [0] * n)[i],
#         )
#
#         left = data["left"][i]
#         top = data["top"][i]
#         w = data["width"][i]
#         h = data["height"][i]
#         groups[key].append((left, top, w, h, txt))
#
#     # Convert pixel coords -> PDF points, build line objects
#     lines = []
#     for _, words in groups.items():
#         if not words:
#             continue
#
#         words.sort(key=lambda x: (x[1], x[0]))
#         x0 = min(l for (l, t, w, h, txt) in words)
#         y0 = min(t for (l, t, w, h, txt) in words)
#         x1 = max(l + w for (l, t, w, h, txt) in words)
#         y1 = max(t + h for (l, t, w, h, txt) in words)
#
#         x0p, y0p, x1p, y1p = (x0 / scale, y0 / scale, x1 / scale, y1 / scale)
#         text = " ".join(txt for (_, _, _, _, txt) in words)
#
#         approx_size = max(8.0, (y1 - y0) / scale * 0.8)
#
#         lines.append({
#             "text": text.strip(),
#             "bbox": [float(x0p), float(y0p), float(x1p), float(y1p)],
#             "size": float(approx_size),
#             "bold": False,
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
#     # Detect language once for the whole document (use first page as sample)
#     detected_lang = None
#
#     for i, page in enumerate(doc, start=1):
#         # 1) Try digital text layer first
#         lines_text = page_lines_from_text_layer(page)
#
#         # 2) If text layer has anything, ignore OCR completely
#         if lines_text:
#             lines = lines_text
#         else:
#             # Detect language only once (on first OCR page)
#             if detected_lang is None:
#                 print(f"⚠️  Page {i}: No text layer — detecting language...")
#                 detected_lang = detect_language_fast(page)
#                 print(f"   Detected: {detected_lang}")
#
#             print(f"⚠️  Page {i}: OCR with lang={detected_lang}")
#             ocr_lines = page_lines_from_ocr(page, lang=detected_lang)
#
#             if not ocr_lines and not OCR_AVAILABLE:
#                 print(f"❌ Page {i}: OCR not available")
#             lines = ocr_lines
#
#         # 3) Dedupe lines
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
#             json.dump(page_obj, f, indent=2, ensure_ascii=False)
#         print(f"wrote: {op}")
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
ExportLines_PATCHED.py
FIXED: Added Unicode/text cleaning to prevent .notdef glyph errors
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
DETECTION_DPI = 150  # For language detection


# ============================================================================
# ✅ NEW: TEXT CLEANING FUNCTION (Fixes .notdef glyph + invalid Unicode errors)
# ============================================================================
def clean_text_for_pdf(text):
    """
    Remove invalid Unicode characters that cause PDF/UA validation failures.

    Fixes veraPDF errors:
    - "The document contains a reference to the .notdef glyph"
    - "The glyph has Unicode value 0, U+FEFF or U+FFFE"

    Removes:
    - Null bytes (\x00, U+0000)
    - Byte Order Marks (\ufeff, U+FEFF)
    - Invalid Unicode (\ufffe, U+FFFE)
    - Control characters (except newline/tab)
    - Zero-width spaces and other invisible formatting
    """
    if not text:
        return text

    # Remove null bytes
    text = text.replace('\x00', '')
    text = text.replace('\u0000', '')

    # Remove BOM
    text = text.replace('\ufeff', '')
    text = text.replace('\ufffe', '')

    # Remove zero-width spaces and joiners
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\u200c', '')  # Zero-width non-joiner
    text = text.replace('\u200d', '')  # Zero-width joiner
    text = text.replace('\ufeff', '')  # Zero-width no-break space (BOM)

    # Filter out problematic Unicode categories
    cleaned = []
    for char in text:
        # Keep newlines, tabs, and normal printable characters
        if char in ('\n', '\t', '\r'):
            cleaned.append(char)
            continue

        # Get Unicode category
        category = unicodedata.category(char)

        # Skip control characters (Cc), format characters (Cf),
        # surrogates (Cs), private use (Co), unassigned (Cn)
        if category in ('Cc', 'Cf', 'Cs', 'Co', 'Cn'):
            continue

        # Keep everything else (letters, numbers, punctuation, symbols, spaces)
        cleaned.append(char)

    result = ''.join(cleaned)

    # Final cleanup: normalize whitespace
    result = ' '.join(result.split())

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
            line_words.sort(key=lambda w: (w[1], w[0]))
            txt = " ".join(w[4] for w in line_words).strip()

            # ✅ APPLY TEXT CLEANING
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

        # ✅ APPLY TEXT CLEANING TO OCR OUTPUT
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
        print("Usage: python ExportLines_PATCHED.py input.pdf output_dir")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])