#!/usr/bin/env python3
import os, sys, json, glob

MIN_COVER = 0.35  # line coverage by region

def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    x0=max(ax0,bx0); y0=max(ay0,by0); x1=min(ax1,bx1); y1=min(ay1,by1)
    if x1<=x0 or y1<=y0: return 0.0
    inter=(x1-x0)*(y1-y0); area=(ax1-ax0)*(ay1-ay0)
    return inter/max(1e-9,area)

def tag_line(ln, region, body):
    txt = ln["text"].strip()
    size = float(ln.get("size",10.0))
    bold = bool(ln.get("bold",False))
    letters = "".join([c for c in txt if c.isalpha()])
    is_all_caps = bool(letters) and txt.upper()==txt
    is_short = len(txt)<=100

    if region and region["cls"]=="title":
        if size >= body*1.8 and is_short: return "H1"
        if (size >= body*1.4 and is_short) or (size>=body*1.25 and bold): return "H2"
    if region and region["cls"]=="list":
        return "LI"
    if region and region["cls"]=="table":
        return "TableRowCandidate"
    if region and region["cls"]=="figure":
        return "FigureCaptionCandidate" if "fig" in txt.lower() or "figure" in txt.lower() else "P"
    if is_all_caps and size>=body*1.2 and is_short:
        return "H2"
    return "P"

def main(pdf_path, layout_dir, lines_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pages = sorted(glob.glob(os.path.join(lines_dir, "page_*.lines.json")))
    for lp in pages:
        with open(lp,"r",encoding="utf-8") as f: L=json.load(f)
        pnum = L["page"]; body=L.get("body_size",10.0); lines=L["lines"]
        layoutp = os.path.join(layout_dir, f"page_{pnum:03d}.layout.json")
        regions=[]
        if os.path.exists(layoutp):
            with open(layoutp,"r",encoding="utf-8") as f: R=json.load(f)
            regions = R.get("regions",[])
        tagged=[]
        for ln in lines:
            best=None; best_cov=0.0
            for reg in regions:
                cov = iou(ln["bbox"], reg["bbox"])
                if cov>best_cov:
                    best, best_cov = reg, cov
            use_reg = best if best_cov>=MIN_COVER else None
            tag = tag_line(ln, use_reg, body)
            tagged.append({
                "tag": tag,
                "text": ln["text"],
                "bbox": ln["bbox"],
                "size": ln["size"],
                "bold": ln["bold"],
                "region": use_reg
            })
        outp = os.path.join(out_dir, f"page_{pnum:03d}.tags.json")
        with open(outp,"w",encoding="utf-8") as f: json.dump({"page":pnum,"tags":tagged}, f, indent=2)
        print("wrote:", outp)

if __name__=="__main__":
    if len(sys.argv)<5:
        print("Usage: python 04b_fuse_layout.py Test.pdf layout_out lines_out fused_tags"); sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
