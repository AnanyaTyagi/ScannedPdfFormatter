#!/usr/bin/env python3
import os, sys, re, json, glob

# ---------- simple helpers ----------
BULLET_CHARS = "•·●▪◦‣-–—*"
CAP_RE = re.compile(r'^[A-Z0-9 ,:\-()\/]+$')
H1_PATTERNS = [
    re.compile(r'^\s*(abstract|preface|introduction|conclusion|references)\s*$', re.I),
]
H2_PATTERNS = [
    re.compile(r'^\s*\d+[\.\)]\s+\S'),  # 1. Title / 1) Title
    re.compile(r'^\s*[IVXLC]+\.\s+\S'),  # I. Title
    re.compile(r'^\s*[A-Z]\)\s+\S'),  # A) Title
]
CAPTION_PAT = re.compile(r'^\s*(figure|fig\.|table)\s*\d+[\.:)]', re.I)


def iou(a, b):
    # boxes: [x0,y0,x1,y1]
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
    union = area_a + area_b - inter if (area_a + area_b - inter) > 0 else 1e-9
    return inter / union


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def looks_like_heading(t):
    s = (t or "").strip()
    if not s: return False
    # short and ALL CAPS (common in scanned PDFs)
    if len(s) <= 80 and CAP_RE.match(s):
        return True
    # known section words
    for pat in H1_PATTERNS:
        if pat.match(s): return True
    return False


def looks_like_subheading(t):
    s = (t or "").strip()
    if not s: return False
    for pat in H2_PATTERNS:
        if pat.match(s): return True
    # title-case and relatively short can also be subheading
    if 3 <= len(s.split()) <= 12 and s[0].isupper() and not s.endswith('.'):
        return True
    return False


def looks_like_list_item(t):
    s = (t or "").strip()
    if not s: return False
    # bullet char at start
    if s and s[0] in BULLET_CHARS:
        return True
    # numbered / lettered lists
    if re.match(r'^\s*\d+[\.\)]\s', s): return True
    if re.match(r'^\s*[a-zA-Z][\.\)]\s', s): return True
    if re.match(r'^\s*[-–—]\s+', s): return True
    return False


def looks_like_caption(t):
    return CAPTION_PAT.match((t or "").strip()) is not None


# ---------- main promotion ----------
def promote_page_tags(tags_obj, layout_obj=None):
    """Return upgraded tags_obj (in-place modified copy)."""
    # Build region lists from layout if present
    fig_boxes = []
    table_boxes = []
    if layout_obj and isinstance(layout_obj.get("regions"), list):
        for r in layout_obj["regions"]:
            cls = (r.get("cls") or "").lower()
            bb = r.get("bbox") or [0, 0, 0, 0]
            if cls == "figure":
                fig_boxes.append(bb)
            elif cls == "table":
                table_boxes.append(bb)

    items = tags_obj.get("tags") or []
    for it in items:
        txt = it.get("text") or ""
        bbox = it.get("bbox") or [0, 0, 0, 0]
        tag = it.get("tag") or "P"

        # 1) Promote captions early
        if looks_like_caption(txt):
            it["tag"] = "FigureCaptionCandidate"
            continue

        # 2) Promote lists
        if looks_like_list_item(txt):
            it["tag"] = "LI"
            continue

        # 3) Promote headings/subheadings
        if looks_like_heading(txt):
            it["tag"] = "H1"
            continue
        if looks_like_subheading(txt):
            it["tag"] = "H2"
            continue

        # otherwise keep P
        it["tag"] = "P"

        # 4) Add region hint if overlapping a detected figure/table
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            best_iou_fig = max([iou(bbox, fb) for fb in fig_boxes], default=0.0)
            best_iou_tab = max([iou(bbox, tb) for tb in table_boxes], default=0.0)
            if best_iou_fig >= 0.20:
                it["region"] = {"cls": "figure", "score": best_iou_fig}
            elif best_iou_tab >= 0.20:
                it["region"] = {"cls": "table", "score": best_iou_tab}
            # else leave region as-is (or None)

    return tags_obj


def main(fused_dir, layout_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pages = sorted(glob.glob(os.path.join(fused_dir, "page_*.tags.json")))
    if not pages:
        print(f"No input pages found in {fused_dir}")
        sys.exit(1)

    for p in pages:
        base = os.path.basename(p)
        # try to load corresponding layout json (optional)
        num = re_search_num(base)
        layout = None
        if layout_dir and num is not None:
            candidate = os.path.join(layout_dir, f"page_{num:03d}.layout.json")
            layout = load_json(candidate, {})
        tags_obj = load_json(p, {})
        if not tags_obj:
            continue
        upgraded = promote_page_tags(tags_obj, layout)

        # Save with .promoted.json extension
        promoted_name = base.replace('.tags.json', '.promoted.json')
        save_json(os.path.join(out_dir, promoted_name), upgraded)
        print("promoted:", promoted_name)


def re_search_num(name):
    import re
    m = re.search(r'(\d{1,4})', name)
    return int(m.group(1)) if m else None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python promote_tags.py <fused_tags_in> <promoted_out> [layout_out]")
        sys.exit(1)
    fused_in = sys.argv[1]
    promoted_out = sys.argv[2]
    layout_out = sys.argv[3] if len(sys.argv) > 3 else None
    main(fused_in, layout_out, promoted_out)