import sys, regex as re, numpy as np
import fitz
from collections import Counter

BULLET_RE = re.compile(r'^\s*(?:[-\u2022*\u25E6]|[0-9]+\.|[A-Za-z]\))\s+')
ONLY_LETTERS = re.compile(r'[^A-Za-z]+')

def join_line_from_words(page, x0,y0,x1,y1, y_tol=2.0):
    words = page.get_text("words")
    sel = [w for w in words if (w[1] <= y1 + y_tol and w[3] >= y0 - y_tol) and (w[2] >= x0-1 and w[0] <= x1+1)]
    sel.sort(key=lambda w: (w[0], w[1]))
    out=[]
    for i,w in enumerate(sel):
        if i>0 and not (w[4] and w[4][0] in ",.;:!?)]}”’\""):
            out.append(" ")
        out.append(w[4])
    return "".join(out).strip()

def page_lines(page):
    spans=[]
    d=page.get_text("dict")
    for b in d.get("blocks", []):
        if b.get("type")!=0: continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t=(s.get("text") or "").strip()
                if not t: continue
                x0,y0,x1,y1 = s.get("bbox", [0,0,0,0])
                spans.append({"x0":x0,"y0":y0,"x1":x1,"y1":y1,"size":float(s.get("size") or 0)})
    spans.sort(key=lambda s:(round(s["y0"],2), s["x0"]))
    # group by baseline
    lines=[]; cur=[]; last_y=None
    for s in spans:
        y=s["y0"]
        if last_y is None or abs(y-last_y)<=3.0:
            cur.append(s); last_y = y if last_y is None else (last_y+y)/2
        else:
            lines.append(cur); cur=[s]; last_y=y
    if cur: lines.append(cur)

    out=[]
    for ln in lines:
        x0=min(t["x0"] for t in ln); y0=min(t["y0"] for t in ln)
        x1=max(t["x1"] for t in ln); y1=max(t["y1"] for t in ln)
        text = join_line_from_words(page, x0,y0,x1,y1)
        if not text: continue
        size = float(np.median([t["size"] for t in ln]))
        out.append({"text":text,"x0":x0,"y0":y0,"x1":x1,"y1":y1,"size":size})
    return out

def classify(lines):
    sizes=[round(l["size"],1) for l in lines if len(l["text"])>20] or [round(l["size"],1) for l in lines]
    body = float(np.median(sizes)) if sizes else 10.0
    tagged=[]
    for ln in lines:
        txt = ln["text"]
        size= ln["size"]
        is_list = bool(BULLET_RE.match(txt))
        letters = ONLY_LETTERS.sub("", txt)
        is_caps = bool(letters) and txt.upper()==txt
        is_short = len(txt)<=100

        tag="P"
        if (size>=body*1.5 and is_short) or (is_caps and size>=body*1.15 and is_short):
            tag="H1" if size>=body*1.8 else "H2"
        elif is_list:
            tag="LI"
        tagged.append({**ln,"tag":tag, "body":body})
    return tagged

def main(pdf):
    doc=fitz.open(pdf)
    for pnum,page in enumerate(doc, start=1):
        lines = page_lines(page)
        tagged = classify(lines)
        print(f"\n--- Page {pnum} ---")
        for t in tagged[:60]:  # show first 60 lines
            print(f"{t['tag']:>2} | {t['size']:.1f} | {t['text'][:100]}")
    print("\nTip: cross-check with debug/preview_pXXX.png")

if __name__=="__main__":
    main(sys.argv[1])
