#!/usr/bin/env python3
import os, sys, json
import fitz  # PyMuPDF
from collections import Counter
from statistics import median

def page_lines(page):
    # words for spacing
    words = page.get_text("words")  # x0,y0,x1,y1,text,block,line,word
    # spans for font metrics
    d = page.get_text("dict")
    span_lines = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:  # text only
            continue
        for l in b.get("lines", []):
            spans = l.get("spans", [])
            if not spans: continue
            # bbox from spans
            x0 = min(s["bbox"][0] for s in spans)
            y0 = min(s["bbox"][1] for s in spans)
            x1 = max(s["bbox"][2] for s in spans)
            y1 = max(s["bbox"][3] for s in spans)
            # text from words overlapping this bbox
            line_words = [w for w in words if w[0] <= x1+1 and w[2] >= x0-1 and w[1] <= y1+1 and w[3] >= y0-1]
            line_words.sort(key=lambda w: (w[1], w[0]))
            txt = " ".join(w[4] for w in line_words).strip()
            if not txt: continue
            sizes = [s.get("size", 0) for s in spans]
            fonts = [s.get("font","") for s in spans]
            bold = any(("Bold" in f) or ("Semibold" in f) or ("Demi" in f) for f in fonts)
            span_lines.append({
                "text": txt,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "size": float(median(sizes)) if sizes else 10.0,
                "bold": bool(bold),
            })
    return span_lines

def main(pdf_path):
    out_dir = "lines_out"
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        lines = page_lines(page)
        # body size heuristic
        longs = [round(l["size"],1) for l in lines if len(l["text"])>20]
        if not longs: longs = [round(l["size"],1) for l in lines]
        body = median(Counter(longs).most_common(5)[k][0] for k in range(min(5,len(Counter(longs)))))
        out = {
            "file": os.path.basename(pdf_path),
            "page": i,
            "body_size": float(body) if body else 10.0,
            "lines": lines
        }
        op = os.path.join(out_dir, f"page_{i:03d}.lines.json")
        with open(op, "w", encoding="utf-8") as f: json.dump(out, f, indent=2)
        print("wrote:", op)

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python 04_export_lines.py Test.pdf"); sys.exit(1)
    main(sys.argv[1])
