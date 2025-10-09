#!/usr/bin/env python3
import os, sys, re, glob, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

# optional: page size lookup from the original PDF
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from ultralytics import YOLO

# --------- CONFIG ---------
# Default label map for a doc-layout model
DEFAULT_LABEL_MAP = {
    0: "text",
    1: "title",
    2: "list",
    3: "table",
    4: "figure",
}

# Colors for debug overlay
COLORS = {
    "title": (220, 20, 60),   # crimson
    "text": (30,144,255),     # dodgerblue
    "list": (60,179,113),     # mediumseagreen
    "table": (255,165,0),     # orange
    "figure": (147,112,219),  # mediumpurple
}


def load_page_size(pdf_path, page_index_1based):
    """Return (width_pts, height_pts) for page, or (None, None) if not available."""
    if not (fitz and pdf_path and os.path.exists(pdf_path)):
        return None, None
    try:
        doc = fitz.open(pdf_path)
        p = page_index_1based - 1
        if 0 <= p < len(doc):
            r = doc[p].rect
            return float(r.width), float(r.height)
    except Exception:
        pass
    return None, None


def parse_page_num_from_name(path):
    """
    Try to read 'pNNN' or '_pageNNN' from filename. Examples:
      debug/preview_p001.png  -> 1
      debug/foo_page012.png   -> 12
    """
    name = os.path.basename(path)
    m = re.search(r'(?:_p|_page)?(\d{1,4})', name)
    if m:
        return int(m.group(1))
    # fallback: enumerate later if needed
    return None


def boxes_image_to_pdf(xyxy_px, zoom):
    """Convert [x0,y0,x1,y1] in pixels -> points using pdf = px / zoom."""
    return (xyxy_px / zoom).tolist()


def draw_boxes_debug(img, regions, zoom, out_path):
    """Draw boxes back on the image (in image pixels) and save."""
    d = ImageDraw.Draw(img)
    for r in regions:
        x0, y0, x1, y1 = [v * zoom for v in r["bbox"]]  # back to px
        color = COLORS.get(r["cls"], (255, 0, 0))
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f'{r["cls"]} {r["score"]:.2f}'
        d.text((x0 + 3, y0 + 3), label, fill=color)
    img.save(out_path)


def main():
    if len(sys.argv) < 4:
        print("Usage: python 03B_detect_layout_yolo.py <debug_dir> <out_dir> <dpi> "
              "[--pdf path/to/original.pdf] [--weights path/to/model.pt] "
              "[--labelmap path/to/labelmap.json] [--debugpng]")
        sys.exit(1)

    debug_dir = sys.argv[1]
    out_dir = sys.argv[2]
    dpi = int(sys.argv[3])

    pdf_path = None
    weights = "/Users/AnanyaTyagi/PycharmProjects/ScannedPdfFormatter/yolov8n-doclaynet.pt"

    label_map = DEFAULT_LABEL_MAP.copy()
    make_dbg_png = False

    # Parse optional flags
    if "--pdf" in sys.argv:
        pdf_path = sys.argv[sys.argv.index("--pdf") + 1]
    if "--weights" in sys.argv:
        weights = sys.argv[sys.argv.index("--weights") + 1]
    if "--labelmap" in sys.argv:
        lm_path = sys.argv[sys.argv.index("--labelmap") + 1]
        with open(lm_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
            # Ensure keys are ints
            label_map = {int(k): v for k, v in label_map.items()}
    if "--debugpng" in sys.argv:
        make_dbg_png = True

    os.makedirs(out_dir, exist_ok=True)

    # Collect images (sorted)
    imgs = sorted(glob.glob(os.path.join(debug_dir, "*.png")))
    if not imgs:
        print(f"No PNGs found in {debug_dir}. Expected your preview images there.")
        sys.exit(1)

    # Load YOLO model
    model = YOLO(weights)
    print(f"Loaded YOLO weights: {weights}")

    zoom = dpi / 72.0  # px per point

    # Process all pages
    index = {"pages": [], "dpi": dpi, "weights": weights}
    for idx, img_path in enumerate(imgs, start=1):
        page_num = parse_page_num_from_name(img_path) or idx

        # Open image
        img = Image.open(img_path).convert("RGB")
        w_px, h_px = img.size

        # Run detector
        results = model.predict(
            source=img,
            verbose=False,
            imgsz=1280,
            conf=0.15,
            iou=0.6,
            max_det=300,
            agnostic_nms=False,
            half=True,
            device=0 if torch.cuda.is_available() else 'cpu',
            augment=True,
            classes=[0, 1, 2, 3, 4]
        )

        r = results[0]
        # Extract detections
        xyxy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        cls_ids = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros((0,), dtype=int)
        scores = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros((0,))

        # Convert to regions in PDF coords
        regions = []
        for bb, c, s in zip(xyxy, cls_ids, scores):
            name = label_map.get(int(c))
            if name is None:
                # If your weights are COCO (person/car/etc.), this will be None → skip
                continue
            pdf_bbox = boxes_image_to_pdf(bb, zoom)
            regions.append({
                "cls": name,
                "score": float(s),
                "bbox": [float(v) for v in pdf_bbox]  # [x0,y0,x1,y1] in PDF points
            })

        # Page size (optional)
        width_pts, height_pts = load_page_size(pdf_path, page_num) if pdf_path else (None, None)

        # Write JSON
        out_json = os.path.join(out_dir, f"page_{page_num:03d}.layout.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "file": os.path.basename(pdf_path) if pdf_path else None,
                "image": os.path.basename(img_path),
                "page": page_num,
                "dpi": dpi,
                "width": width_pts,
                "height": height_pts,
                "regions": regions
            }, f, indent=2)
        print(f"Wrote {out_json}  (regions={len(regions)})")

        # Optional debug overlay
        if make_dbg_png:
            dbg_out = os.path.join(out_dir, f"page_{page_num:03d}.det.png")
            draw_boxes_debug(img.copy(), [{"cls": r["cls"], "score": r["score"], "bbox": r["bbox"]} for r in regions], zoom, dbg_out)

        index["pages"].append({"page": page_num, "json": out_json})

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Done. Layout JSONs in: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
