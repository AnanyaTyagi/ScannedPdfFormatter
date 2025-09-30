import sys
import fitz  # PyMuPDF
import numpy as np
try:
    import regex as re   # switch to: import re   if you prefer
except Exception:
    import re
from collections import Counter
from typing import List, Dict, Any

# -------- Heuristics / regexes --------
BULLET_RE = re.compile(r'^\s*(?:[\-\u2022\*\u25E6]|[0-9]+\.|[A-Za-z]\))\s+')
TABLE_SPLIT_RE = re.compile(r'\s{2,}')  # 2+ spaces ~ column gap
PUNCT_RIGHT = set(",.;:!?)]}”’\"")

# --------- Core helpers ----------
def assemble_from_words(words, x0, y0, x1, y1, y_overlap_tol=2.0):
    """
    Build text for a line by collecting page 'words' that overlap the line bbox.
    Produces correct spacing even when each word is a separate span.
    """
    sel = [w for w in words
           if (w["y0"] <= y1 + y_overlap_tol and w["y1"] >= y0 - y_overlap_tol)
           and (w["x1"] >= x0 - 1 and w["x0"] <= x1 + 1)]
    sel.sort(key=lambda w: (w["x0"], w["y0"]))
    out = []
    for i, w in enumerate(sel):
        if i > 0 and not (w["text"] and w["text"][0] in PUNCT_RIGHT):
            out.append(" ")
        out.append(w["text"])
    return "".join(out).strip()

def page_to_lines(page) -> List[Dict[str, Any]]:
    """
    Extract lines for ONE page: use words for text spacing, spans for font/bold/coords.
    """
    # words (for spacing)
    wtuples = page.get_text("words")  # (x0, y0, x1, y1, text, block, line, wno)
    words = [{"x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3], "text": w[4],
              "block": w[5], "line": w[6], "wno": w[7]} for w in wtuples]

    # spans (for font metrics)
    d = page.get_text("dict")
    spans = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:  # text only
            continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t = (s.get("text") or "").strip()
                if not t:
                    continue
                x0, y0, x1, y1 = (s.get("bbox") or [0,0,0,0])
                spans.append({
                    "text": t,
                    "font": s.get("font",""),
                    "size": float(s.get("size") or 0),
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                })

    # group spans into lines by baseline proximity
    spans.sort(key=lambda s: (round(s["y0"],2), round(s["x0"],2)))
    lines = group_spans_into_lines(spans, words, y_tol=3.0)
    return lines

def group_spans_into_lines(spans, words, y_tol=3.0):
    grouped = []
    cur = []
    last_y = None
    for s in spans:
        y = s["y0"]
        if last_y is None or abs(y - last_y) <= y_tol:
            cur.append(s)
            last_y = y if last_y is None else (last_y + y)/2
        else:
            grouped.append(cur)
            cur = [s]
            last_y = y
    if cur:
        grouped.append(cur)

    out = []
    for ln in grouped:
        x0 = min(t["x0"] for t in ln); y0 = min(t["y0"] for t in ln)
        x1 = max(t["x1"] for t in ln); y1 = max(t["y1"] for t in ln)

        txt = assemble_from_words(words, x0, y0, x1, y1)
        if not txt:
            # rare fallback: join spans with a space
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

def dehyphenate_lines(lines):
    """
    Join a trailing hyphen line + next line starting lowercase: 'environ-' + 'ment' -> 'environment'
    """
    out = []
    for i, ln in enumerate(lines):
        if out and out[-1]["text"].endswith("-") and ln["text"] and ln["text"][0].islower():
            out[-1]["text"] = out[-1]["text"][:-1] + ln["text"]
        else:
            out.append(ln)
    return out

def infer_body_font_size(lines):
    sizes = [round(l["size"], 1) for l in lines if l["text"] and len(l["text"]) > 20]
    if not sizes:
        sizes = [round(l["size"], 1) for l in lines]
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

def to_html(tagged_or_tables, page_num: int) -> str:
    html = [f"<section data-page='{page_num}'><h3>Page {page_num}</h3>"]
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
                html.append("</ul>"); in_list = False
            if tag in ("h1","h2"):
                html.append(f"<{tag}>{text}</{tag}>")
            else:
                html.append(f"<p>{text}</p>")
    if in_list: html.append("</ul>")
    html.append("</section>")
    return "\n".join(html)

# --------- Orchestrator ----------
def process_pdf_to_html(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    sections = []
    for pnum, page in enumerate(doc, start=1):
        lines = page_to_lines(page)
        lines = dehyphenate_lines(lines)  # optional but helpful
        body = infer_body_font_size(lines)
        tagged = classify_lines(lines, body)
        tagged = group_tables(tagged)
        sections.append(to_html(tagged, pnum))
        print(f"Page {pnum}: lines={len(lines)} body={body:.1f} "
              f"h1={sum(1 for t in tagged if isinstance(t, dict) and t.get('tag')=='h1')} "
              f"h2={sum(1 for t in tagged if isinstance(t, dict) and t.get('tag')=='h2')} "
              f"li={sum(1 for t in tagged if isinstance(t, dict) and t.get('tag')=='li')}")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Structured from {escape_html(pdf_path)}</title>
<style>
  @page {{ size: A4; margin: 20mm; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; line-height: 1.35; }}
  h1 {{ font-size: 1.6em; margin: 0.8em 0 0.3em; }}
  h2 {{ font-size: 1.35em; margin: 0.7em 0 0.25em; }}
  p  {{ margin: 0.25em 0 0.25em; }}
  ul {{ margin: 0.2em 0 0.4em 1.2em; }}
  table {{ border-collapse: collapse; margin: 0.5em 0; width: 100%; }}
  td {{ border: 1px solid #ccc; padding: 4px; vertical-align: top; }}
  section {{ break-inside: avoid-page; page-break-inside: avoid; margin-bottom: 1em; }}
  h3 {{ color: #666; font-weight: 600; font-size: 0.95em; margin: 0.2em 0 0.5em; }}
</style>
</head>
<body>
<main role="main">
{'\n<hr/>\n'.join(sections)}
</main>
</body>
</html>"""
    return html

import shutil, subprocess, os

def write_html_and_optional_pdf(html: str, out_html: str, out_pdf: str = None, engine: str = "auto"):
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote HTML: {os.path.abspath(out_html)}")

    if not out_pdf:
        return

    # prefer prince if available or if engine=prince
    prince = shutil.which("prince")
    if engine in ("auto","prince") and prince:
        try:
            subprocess.check_call([prince, out_html, "-o", out_pdf])
            print(f"Wrote PDF (Prince): {os.path.abspath(out_pdf)}")
            return
        except subprocess.CalledProcessError as e:
            print("Prince failed:", e)

    if engine in ("auto","weasy"):
        try:
            from weasyprint import HTML
            HTML(string=html, base_url=".").write_pdf(out_pdf)
            print(f"Wrote PDF (WeasyPrint): {os.path.abspath(out_pdf)}")
        except Exception as e:
            print("Could not create PDF with WeasyPrint:", e)
            print("Install Prince and run: prince", out_html, "-o", out_pdf)
# -------------- CLI --------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python format_pdf.py input.pdf out_Initial.html [--pdf out.pdf]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    out_html = sys.argv[2]
    out_pdf = None
    if len(sys.argv) >= 5 and sys.argv[3] == "--pdf":
        out_pdf = sys.argv[4]

    html = process_pdf_to_html(pdf_path)
    write_html_and_optional_pdf(html, out_html, out_pdf)

if __name__ == "__main__":
    main()
