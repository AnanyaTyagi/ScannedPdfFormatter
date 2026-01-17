#!/usr/bin/env python3
import os, sys, json, argparse, glob
from pathlib import Path

import PIL.Image as Image
import pikepdf

# ultralytics is required: pip install ultralytics
from ultralytics import YOLO

# ---- Map rich DocLayNet labels to 5 canonical region types ----
DOC2CANON = {
    "Title": "title",
    "Section-header": "section_header",
    "Text": "text",
    "List-item": "list_item",
    "Table": "table",
    "Picture": "picture",

    "Caption": "caption",
    "Footnote": "footnote",
    "Formula": "formula",
    "Page-footer": "page_footer",
    "Page-header": "page_header",
}


# If you ever use PubLayNet weights, map them directly
PUBL2CANON = {
    "text": "text",
    "title": "title",
    "list": "list",
    "table": "table",
    "figure": "figure",
}

def load_labelmap(path):
    """
    Load a labelmap (id->name). Accepts either:
      {"0":"Caption","1":"Footnote",...}  (DocLayNet style)
    or  {"0":"text","1":"title",...}      (PubLayNet style)
    """
    with open(path, "r", encoding="utf-8") as f:
        lm = json.load(f)

    # Normalize keys to int
    fixed = {}
    for k, v in lm.items():
        try:
            idx = int(k)
        except Exception:
            continue
        fixed[idx] = str(v)
    return fixed

def get_pdf_page_size_pts(pdf, page_index: int):
    """
    Return (width, height) in PDF points for the given page.
    Works even if /MediaBox elements are Decimals or pikepdf.Objects.
    """
    page = pdf.pages[page_index]
    page_obj = page.obj

    # Prefer MediaBox; fall back to CropBox; finally Letter defaults
    mb = page_obj.get('/MediaBox') or page_obj.get('/CropBox')
    if not mb:
        return 612.0, 792.0  # 8.5in x 11in

    # mb is a pikepdf.Array; convert safely to list of floats
    try:
        coords = [float(x) for x in list(mb)[:4]]
    except Exception:
        # ultra-defensive: index one by one
        coords = [float(mb[0]), float(mb[1]), float(mb[2]), float(mb[3])]

    x0, y0, x1, y1 = coords
    return (x1 - x0), (y1 - y0)

# def to_pdf_bbox(xyxy, img_w, img_h, pdf_w, pdf_h):
#     """
#     Convert image pixel bbox (x0,y0,x1,y1) with origin top-left (image space)
#     to PDF points with origin bottom-left (PDF space).
#     """
#     x0, y0, x1, y1 = [float(v) for v in xyxy]
#     sx = pdf_w / float(img_w)
#     sy = pdf_h / float(img_h)
#
#     # convert and flip Y
#     px0 = x0 * sx
#     px1 = x1 * sx
#     py0 = pdf_h - (y1 * sy)
#     py1 = pdf_h - (y0 * sy)
#
#     # ensure well-ordered
#     if px1 < px0: px0, px1 = px1, px0
#     if py1 < py0: py0, py1 = py1, py0
#
#     return [px0, py0, px1, py1]


def to_pdf_bbox(xyxy, img_w, img_h, pdf_w, pdf_h):
    """
    Convert image pixel bbox (x0,y0,x1,y1) with origin TOP-LEFT (image space)
    to PDF points with origin ALSO TOP-LEFT (to match lines_out).
    """
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    sx = pdf_w / float(img_w)
    sy = pdf_h / float(img_h)

    # keep Y top-left: no flip
    px0 = x0 * sx
    px1 = x1 * sx
    py0 = y0 * sy
    py1 = y1 * sy

    if px1 < px0:
        px0, px1 = px1, px0
    if py1 < py0:
        py0, py1 = py1, py0

    return [px0, py0, px1, py1]

def choose_normalizer(names_dict):
    """
    Pick the right normalizer based on model names.
    If it looks like DocLayNet, use DOC2CANON; else PubLayNet-ish, use PUBL2CANON.
    """
    lower_names = {v.lower() for v in names_dict.values()}
    # Heuristic: DocLayNet includes many labels like "caption", "footnote", "section-header"
    if {"caption", "footnote", "formula", "section-header", "page-header", "page-footer"} & lower_names:
        return DOC2CANON
    # Fallback assume PubLayNet-like
    return PUBL2CANON

def main():
    ap = argparse.ArgumentParser(description="Detect layout regions with YOLO and write PDF-space bboxes.")
    ap.add_argument("debug_dir", help="Directory with preview PNGs from ProbingFile.py (e.g. debug)")
    ap.add_argument("out_dir", help="Output directory for layout JSON files")
    ap.add_argument("dpi", type=int, help="Render DPI used for previews (only for bookkeeping)")
    ap.add_argument("--pdf", required=True, help="Path to original PDF for accurate page sizes")
    ap.add_argument("--labelmap", required=True, help="Path to labelmap.json matching the model")
    ap.add_argument("--weights", default="yolov8n-doclaynet.pt", help="YOLOv8 weights path")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--debugpng", action="store_true", help="Also save debug overlay PNGs")
    args = ap.parse_args()

    debug_dir = Path(args.debug_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load labelmap & model
    labelmap = load_labelmap(args.labelmap)
    model = YOLO(args.weights)
    print(f"Loaded YOLO weights: {args.weights}")

    # Determine which normalizer we should use based on model.names
    # but fallback to labelmap if names are missing
    model_names = model.names if hasattr(model, "names") else labelmap
    normalizer = choose_normalizer(model_names)

    # Open the PDF to fetch exact page sizes (points)
    pdf = pikepdf.open(args.pdf)
    num_pages = len(pdf.pages)

    # Find preview images in debug dir
    # We expect files like debug/preview_p001.png
    img_paths = sorted(debug_dir.glob("preview_p*.png"))
    if not img_paths:
        print(f"ERROR: No preview images found in {debug_dir}")
        sys.exit(1)

    # Build mapping pnum->image path
    page_images = {}
    for p in img_paths:
        # extract page number
        # preview_p001.png -> 1
        stem = p.stem
        # expected stem: "preview_pXYZ"
        try:
            pnum = int(stem.split("_p")[-1])
        except Exception:
            continue
        page_images[pnum] = p

    written = 0

    for pnum in range(1, num_pages + 1):
        if pnum not in page_images:
            # No image for this page (skip)
            layout = {
                "file": os.path.basename(args.pdf),
                "image": None,
                "page": pnum,
                "dpi": args.dpi,
                "width": None,
                "height": None,
                "regions": []
            }
            with open(out_dir / f"page_{pnum:03d}.layout.json", "w", encoding="utf-8") as f:
                json.dump(layout, f, indent=2)
            continue

        img_path = page_images[pnum]
        im = Image.open(img_path).convert("RGB")
        img_w, img_h = im.size

        pdf_w, pdf_h = get_pdf_page_size_pts(pdf, pnum - 1)

        # Run YOLO on the page preview
        results = model.predict(source=str(img_path), conf=args.conf, verbose=False)[0]

        regions = []
        # For drawing, collect shapes if requested
        draw_boxes = []

        # Each detection
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                # Get class name from labelmap or model.names
                raw_name = None
                if cls_idx in labelmap:
                    raw_name = labelmap[cls_idx]
                elif hasattr(model, "names") and cls_idx in model.names:
                    raw_name = model.names[cls_idx]
                else:
                    raw_name = str(cls_idx)

                # Normalize to canonical classes we use downstream
                norm = normalizer.get(raw_name, None)
                if norm is None:
                    # skip unknown labels
                    continue

                xyxy = boxes.xyxy[i].cpu().tolist()  # [x0,y0,x1,y1] in image pixels
                bbox_pdf = to_pdf_bbox(xyxy, img_w, img_h, pdf_w, pdf_h)

                regions.append({
                    "cls": norm,
                    "score": conf,
                    "bbox": bbox_pdf
                })

                if args.debugpng:
                    draw_boxes.append((xyxy, raw_name, conf))

        layout = {
            "file": os.path.basename(args.pdf),
            "image": img_path.name,
            "page": pnum,
            "dpi": args.dpi,
            "width": pdf_w,
            "height": pdf_h,
            "regions": regions
        }

        # Write JSON
        out_json = out_dir / f"page_{pnum:03d}.layout.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2)

        written += 1
        print(f"Wrote {out_json}  (regions={len(regions)})")

        # Optional: draw PNG overlays for visual debugging
        if args.debugpng and draw_boxes:
            try:
                from PIL import ImageDraw, ImageFont
                im_dbg = im.copy()
                dr = ImageDraw.Draw(im_dbg)
                for (xyxy, name, conf) in draw_boxes:
                    x0,y0,x1,y1 = xyxy
                    dr.rectangle([x0,y0,x1,y1], outline=(0,255,0), width=2)
                    label = f"{name} {conf:.2f}"
                    dr.text((x0+3,y0+3), label, fill=(0,255,0))
                im_dbg.save(out_dir / f"page_{pnum:03d}.detected.png")
            except Exception:
                pass

    print(f"Done. Layout JSONs in: {out_dir}")

if __name__ == "__main__":
    main()

