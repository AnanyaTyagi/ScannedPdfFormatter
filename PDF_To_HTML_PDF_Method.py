import sys, json, math
import fitz  # from pymupdf
import regex as re  # or: import re
import numpy as np
from collections import Counter
import string


BULLET_RE = re.compile(r'^\s*(?:[\-\u2022\*\u25E6]|[0-9]+\.|[A-Za-z]\))\s+')
TABLE_SPLIT_RE = re.compile(r'\s{2,}')  # 2+ spaces ~ column gap
PUNCT_RIGHT = set(",.;:!?)]}”’\"")

def assemble_from_words(words, x0, y0, x1, y1, y_overlap_tol=2.0):
    """
    Build text for a line by collecting page 'words' that overlap this line's bbox.
    This produces correct spaces even when each word is a separate span.
    """
    # pick words that vertically overlap the line and horizontally intersect its bbox
    sel = [w for w in words
           if (w["y0"] <= y1 + y_overlap_tol and w["y1"] >= y0 - y_overlap_tol)
           and (w["x1"] >= x0 - 1 and w["x0"] <= x1 + 1)]
    sel.sort(key=lambda w: (w["x0"], w["y0"]))

    # join with spaces, but avoid a space before right punctuation
    out = []
    for i, w in enumerate(sel):
        if i > 0:
            if not (w["text"] and w["text"][0] in PUNCT_RIGHT):
                out.append(" ")
        out.append(w["text"])
    return "".join(out).strip()

def page_to_lines(page):
    # 1) collect words (with boxes)
    wtuples = page.get_text("words")  # (x0, y0, x1, y1, text, block_no, line_no, word_no)
    words = [{"x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3], "text": w[4],
              "block": w[5], "line": w[6], "wno": w[7]} for w in wtuples]

    # 2) collect spans (for font size / bold)
    d = page.get_text("dict")
    spans = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t = s.get("text","").strip()
                if not t:
                    continue
                bbox = s.get("bbox",[0,0,0,0])
                spans.append({
                    "text": t,
                    "font": s.get("font",""),
                    "size": float(s.get("size",0) or 0),
                    "x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3],
                })
    spans.sort(key=lambda s: (round(s["y0"],2), round(s["x0"],2)))

    # 3) group spans into lines (will use words to build spaced text)
    return group_spans_into_lines(spans, words, y_tol=3.0)


def join_spans_with_spaces(spans_in_line):
    spans_in_line = sorted(spans_in_line, key=lambda t: t["x0"])
    widths = [(s["x1"] - s["x0"]) / max(1, len(s["text"])) for s in spans_in_line if s["text"].strip()]
    avg_char_width = float(np.median(widths)) if widths else 3.0

    # ↓ TUNE THIS if you still get glued words:
    gap_threshold = avg_char_width * 0.55   # try 0.55–0.75

    buf = []
    for k, s in enumerate(spans_in_line):
        if k > 0:
            prev = spans_in_line[k-1]
            gap = s["x0"] - prev["x1"]

            # add a space only when the gap is meaningfully large
            if gap > gap_threshold:
                # avoid space before punctuation (e.g., "word ,")
                if not (s["text"] and s["text"][0] in PUNCT_RIGHT):
                    buf.append(" ")
        buf.append(s["text"])
    return "".join(buf)


def group_spans_into_lines(spans, words, y_tol=3.0):
    lines = []
    cur = []
    last_y = None
    for s in spans:
        y = s["y0"]
        if last_y is None or abs(y - last_y) <= y_tol:
            cur.append(s)
            last_y = y if last_y is None else (last_y + y)/2
        else:
            lines.append(cur)
            cur = [s]
            last_y = y
    if cur:
        lines.append(cur)

    out = []
    for ln in lines:
        # bbox for the line (from spans)
        x0 = min(t["x0"] for t in ln)
        y0 = min(t["y0"] for t in ln)
        x1 = max(t["x1"] for t in ln)
        y1 = max(t["y1"] for t in ln)

        # TEXT: build from page 'words' (correct spacing). Fallback to span-join if empty.
        txt = assemble_from_words(words, x0, y0, x1, y1)
        if not txt:
            # fallback – very rare
            txt = " ".join(t["text"] for t in sorted(ln, key=lambda t: t["x0"])).strip()
        if not txt:
            continue

        sizes = [t["size"] for t in ln]
        fonts = [t["font"] for t in ln]
        bold = any(("Bold" in f) or ("Semibold" in f) or ("Demi" in f) for f in fonts)
        out.append({
            "text": txt,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "size": float(np.median(sizes)),
            "bold": bold,
            "fonts": fonts,
        })
    return out


def infer_body_font_size(all_lines):
    sizes = [round(l["size"], 1) for l in all_lines if l["text"] and len(l["text"]) > 20]
    if not sizes:
        sizes = [round(l["size"], 1) for l in all_lines]
    freq = Counter(sizes)
    common = [s for s, _ in freq.most_common(5)]
    return float(np.median(common)) if common else (float(np.median(sizes)) if sizes else 10.0)

def looks_table_line(text: str) -> bool:
    return len(TABLE_SPLIT_RE.split(text.strip())) >= 3

def classify_lines(lines, body_size):
    tagged = []
    for i, ln in enumerate(lines):
        text = ln["text"]
        size = float(ln.get("size", 0.0))
        is_list = bool(BULLET_RE.match(text))
        letters_only = re.sub(r'[^A-Za-z]', '', text)
        is_all_caps = bool(letters_only) and text.upper() == text
        is_short = len(text) <= 100

        tag = "p"
        if (size >= body_size * 1.5 and is_short) \
           or (size >= body_size * 1.25 and ln.get("bold") and is_short) \
           or (is_all_caps and size >= body_size * 1.15 and is_short):
            tag = "h1" if size >= body_size * 1.8 else "h2"
        elif is_list:
            tag = "li"
        else:
            if looks_table_line(text) and i + 1 < len(lines) and looks_table_line(lines[i + 1]["text"]):
                tag = "table_row"

        tagged.append({**ln, "tag": tag})
    return tagged

def group_tables(tagged):
    out = []
    i = 0
    while i < len(tagged):
        if tagged[i]["tag"] != "table_row":
            out.append(tagged[i]); i += 1; continue
        block = []
        while i < len(tagged) and tagged[i]["tag"] == "table_row":
            block.append(tagged[i]); i += 1
        table = []
        max_cols = 0
        for row in block:
            cells = [c.strip() for c in TABLE_SPLIT_RE.split(row["text"].strip()) if c.strip() != ""]
            table.append(cells)
            max_cols = max(max_cols, len(cells))
        table = [r + [""]*(max_cols - len(r)) for r in table]
        out.append({"tag": "table", "rows": table})
    return out

def escape_html(s: str) -> str:
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))

def to_html(tagged_or_tables):
    html = []
    in_list = False
    for item in tagged_or_tables:
        if isinstance(item, dict) and item.get("tag") == "table":
            if in_list:
                html.append("</ul>"); in_list=False
            rows = item["rows"]
            html.append("<table>")
            for r in rows:
                html.append("<tr>" + "".join(f"<td>{escape_html(c)}</td>" for c in r) + "</tr>")
            html.append("</table>")
            continue

        tag = item["tag"]
        text = escape_html(item["text"])
        if tag == "li":
            if not in_list:
                html.append("<ul>")
                in_list = True
            html.append(f"<li>{text}</li>")
        else:
            if in_list:
                html.append("</ul>")
                in_list = False
            if tag in ("h1","h2"):
                html.append(f"<{tag}>{text}</{tag}>")
            else:
                html.append(f"<p>{text}</p>")
    if in_list:
        html.append("</ul>")
    return "\n".join(html)

def main(pdf_path, out_html_path):
    doc = fitz.open(pdf_path)
    page_html = []
    for pnum, page in enumerate(doc, start=1):
        lines = page_to_lines(page)
        if not lines:
            t = (page.get_text("text") or "").strip()
            lines = [{"text": ln, "size": 0.0, "bold": False} for ln in t.splitlines() if ln.strip()]
        body_size = infer_body_font_size(lines)
        tagged = classify_lines(lines, body_size)
        tagged = group_tables(tagged)
        html = to_html(tagged)
        page_html.append(f"<section data-page='{pnum}'><h3>Page {pnum}</h3>\n{html}\n</section>")
        # debug:
        print(f"Page {pnum}: lines={len(lines)} body={body_size:.1f} "
              f"h1={sum(1 for t in tagged if isinstance(t, dict) and t.get('tag')=='h1')} "
              f"h2={sum(1 for t in tagged if isinstance(t, dict) and t.get('tag')=='h2')} "
              f"li={sum(1 for t in tagged if isinstance(t, dict) and t.get('tag')=='li')}")

    full = f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<title>Structured from {escape_html(pdf_path)}</title>
<body>
{'\n<hr/>\n'.join(page_html)}
</body></html>"""
    with open(out_html_path, "w", encoding="utf-8") as w:
        w.write(full)
    print(f"Wrote {out_html_path} with {len(page_html)} pages")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python PDF_To_HTML_PDF_Method.py input.pdf out_Initial.html")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
