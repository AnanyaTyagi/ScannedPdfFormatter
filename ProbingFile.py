import sys, os, json
try:
    import fitz
except ImportError:
    import sys
    sys.path.insert(0, '/home/adminuser/venv/lib/python3.13/site-packages')
    import fitz
pdf_path = sys.argv[1]
doc = fitz.open(pdf_path)
os.makedirs("debug", exist_ok=True)

report = {"file": os.path.basename(pdf_path), "pages": []}

for pnum, page in enumerate(doc, start=1):
    # text layer check
    words = page.get_text("words")  # list of (x0,y0,x1,y1,text,block,line,wno)
    has_text = len(words) > 5
    char_count = sum(len(w[4]) for w in words)

    # basic font stats from spans
    sizes = []
    d = page.get_text("dict")
    for b in d.get("blocks", []):
        if b.get("type") != 0: continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                if s.get("text", "").strip():
                    sizes.append(round(float(s.get("size", 0) or 0), 1))
    sizes.sort()
    body_guess = sizes[len(sizes)//2] if sizes else 0.0

    # render a low-res preview PNG (150 dpi)
    zoom = 150/72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    out_png = f"debug/preview_p{pnum:03d}.png"
    pix.save(out_png)

    report["pages"].append({
        "page": pnum,
        "has_text_layer": has_text,
        "word_count": len(words),
        "char_count": char_count,
        "median_font_size": body_guess,
        "preview_png": out_png
    })

print(json.dumps(report, indent=2))
