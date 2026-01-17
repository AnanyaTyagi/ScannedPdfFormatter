# #!/usr/bin/env python3
# import os, sys, json, glob, re
#
# # ========== CONFIGURATION ==========
# MIN_COVER = 0.35  # line coverage by region
#
# # Pattern matching constants
# BULLET_CHARS = "•·●▪◦‣-–—*"
# CAP_RE = re.compile(r'^[A-Z0-9 ,:\-()\/]+$')
#
# H1_PATTERNS = [
#     re.compile(r'^\s*(abstract|preface|introduction|conclusion|references|acknowledgments?)\s*$', re.I),
# ]
#
# H2_PATTERNS = [
#     re.compile(r'^\s*\d+[\.\)]\s+\S'),  # 1. Title / 1) Title
#     re.compile(r'^\s*[IVXLC]+\.\s+\S'),  # I. Title
#     re.compile(r'^\s*[A-Z]\)\s+\S'),  # A) Title
# ]
#
# CAPTION_PAT = re.compile(r'^\s*(figure|fig\.|table|tbl\.?)\s*\d+[\.:)]', re.I)
#
#
# # ========== HELPER FUNCTIONS ==========
#
# def iou(a, b):
#     """Calculate Intersection over Union (line overlap with region)"""
#     ax0, ay0, ax1, ay1 = a
#     bx0, by0, bx1, by1 = b
#     x0 = max(ax0, bx0)
#     y0 = max(ay0, by0)
#     x1 = min(ax1, bx1)
#     y1 = min(ay1, by1)
#
#     if x1 <= x0 or y1 <= y0:
#         return 0.0
#
#     inter = (x1 - x0) * (y1 - y0)
#     area = (ax1 - ax0) * (ay1 - ay0)
#     return inter / max(1e-9, area)
#
#
# def looks_like_caption(txt):
#     """Detect figure/table captions: 'Figure 1:', 'Table 2.', etc."""
#     return CAPTION_PAT.match((txt or "").strip()) is not None
#
#
# def looks_like_list_item(txt):
#     """Detect list items by bullet chars or numbering"""
#     s = (txt or "").strip()
#     if not s:
#         return False
#
#     # Bullet character at start
#     if s[0] in BULLET_CHARS:
#         return True
#
#     # Numbered lists: 1. , 2) , etc.
#     if re.match(r'^\s*\d+[\.\)]\s', s):
#         return True
#
#     # Lettered lists: a. , b) , A. , B)
#     if re.match(r'^\s*[a-zA-Z][\.\)]\s', s):
#         return True
#
#     # Dash bullets with space: - Item, – Item, — Item
#     if re.match(r'^\s*[-–—]\s+', s):
#         return True
#
#     return False
#
#
# def looks_like_h1_pattern(txt):
#     """Detect H1 by known section names"""
#     s = (txt or "").strip()
#     if not s:
#         return False
#
#     for pat in H1_PATTERNS:
#         if pat.match(s):
#             return True
#     return False
#
#
# def looks_like_h2_pattern(txt):
#     """Detect H2 by numbering patterns"""
#     s = (txt or "").strip()
#     if not s:
#         return False
#
#     for pat in H2_PATTERNS:
#         if pat.match(s):
#             return True
#
#     # Title-case heuristic: 3-12 words, starts capital, no period
#     words = s.split()
#     if 3 <= len(words) <= 12 and s[0].isupper() and not s.endswith('.'):
#         return True
#
#     return False
#
#
# def is_short(txt):
#     """Check if text is short (likely heading, not paragraph)"""
#     return len((txt or "").strip()) <= 100
#
#
# def is_all_caps(txt):
#     """Check if text is ALL CAPS (with numbers/punctuation allowed)"""
#     s = (txt or "").strip()
#     if not s:
#         return False
#
#     letters = "".join([c for c in s if c.isalpha()])
#     return bool(letters) and CAP_RE.match(s)
#
# def tag_line_smart(ln, region, body):
#     """
#     Intelligent tagging with priority system:
#     1. High-confidence patterns (captions, explicit lists)
#     2. ML regions + font analysis (YOLO + typography)
#     3. Pattern-based headings (fallback when no region)
#     4. Font-only analysis (large text = heading)
#     5. All-caps heuristics
#     6. Default to P
#     """
#     txt = (ln.get("text") or "").strip()
#     size = float(ln.get("size", 10.0))
#     bold = bool(ln.get("bold", False))
#
#     # ===== PRIORITY 1: High-Confidence Patterns =====
#     # These are almost always correct, override everything
#
#     if looks_like_caption(txt):
#         return "FigureCaptionCandidate"
#
#     if looks_like_list_item(txt):
#         return "LI"
#
#     # ===== PRIORITY 2: ML Region + Font Analysis (Hybrid) =====
#     # Use YOLO detection + typography together
#
#     region_cls = None
#     if region:
#         # accept either "cls" (YOLO) or "label" (other layouts)
#         region_cls = (region.get("cls") or region.get("label") or "").lower()
#
#     if region_cls:
#         # --- TITLE / SECTION HEADER REGIONS ---
#         if region_cls in ("title", "section_header"):
#             # Very large text in title region → H1
#             if size >= body * 1.8 and is_short(txt):
#                 return "H1"
#
#             # Medium-large or bold in title region → H2
#             if (size >= body * 1.4 and is_short(txt)) or (size >= body * 1.25 and bold):
#                 return "H2"
#
#             # Title region but normal size → still H2 (trust the ML)
#             if is_short(txt):
#                 return "H2"
#
#         # --- CAPTION REGIONS ---
#         if region_cls == "caption":
#             # If it's in a caption region, treat as caption even if pattern is fuzzy
#             return "FigureCaptionCandidate"
#
#         # --- LIST REGIONS ---
#         if region_cls in ("list", "list_item"):
#             return "LI"
#
#         # --- TABLE REGIONS ---
#         if region_cls == "table":
#             return "TableRowCandidate"
#
#         # --- FIGURE / PICTURE REGIONS ---
#         if region_cls in ("figure", "picture"):
#             # Check if it's a caption within figure region
#             low = txt.lower()
#             if "figure" in low or "fig." in low or "table" in low or "tbl" in low:
#                 return "FigureCaptionCandidate"
#             # Otherwise just treat as body text near a figure
#             return "P"
#
#         # --- FOOTNOTE REGIONS ---
#         if region_cls == "footnote":
#             # Could be specialized later, but for now mark as paragraph-ish
#             return "P"
#
#         # Page header/footer: usually not structural headings in content
#         if region_cls in ("page_header", "page_footer"):
#             return "P"
#
#     # ===== PRIORITY 3: Pattern-Based Headings (No Region or ML Missed) =====
#
#     if looks_like_h1_pattern(txt):
#         return "H1"
#
#     if looks_like_h2_pattern(txt):
#         return "H2"
#
#     # ===== PRIORITY 4: Font-Based Analysis (Region-Agnostic) =====
#
#     # Very large text → H1
#     if size >= body * 1.8 and is_short(txt):
#         return "H1"
#
#     # Large text → H2
#     if size >= body * 1.4 and is_short(txt):
#         return "H2"
#
#     # Bold + moderately large → H2
#     if bold and size >= body * 1.25 and is_short(txt):
#         return "H2"
#
#     # ===== PRIORITY 5: All-Caps Heuristic =====
#
#     if is_all_caps(txt) and size >= body * 1.2 and is_short(txt):
#         return "H2"
#
#     # ===== PRIORITY 6: Default =====
#     return "P"
#
# # ========== MAIN PROCESSING ==========
#
# def main(pdf_path, layout_dir, lines_dir, out_dir):
#     os.makedirs(out_dir, exist_ok=True)
#
#     pages = sorted(glob.glob(os.path.join(lines_dir, "page_*.lines.json")))
#
#     if not pages:
#         print(f"No line files found in {lines_dir}")
#         sys.exit(1)
#
#     for lp in pages:
#         # Load lines data
#         with open(lp, "r", encoding="utf-8") as f:
#             L = json.load(f)
#
#         pnum = L["page"]
#         body = L.get("body_size", 10.0)
#         lines = L["lines"]
#
#         # Load corresponding layout data (optional)
#         layoutp = os.path.join(layout_dir, f"page_{pnum:03d}.layout.json")
#         regions = []
#
#         if os.path.exists(layoutp):
#             with open(layoutp, "r", encoding="utf-8") as f:
#                 R = json.load(f)
#             regions = R.get("regions", [])
#
#         # Tag each line
#         tagged = []
#
#         for ln in lines:
#             # Find best matching region
#             best = None
#             best_cov = 0.0
#
#             for reg in regions:
#                 cov = iou(ln["bbox"], reg["bbox"])
#                 if cov > best_cov:
#                     best, best_cov = reg, cov
#
#             # Use region only if coverage meets threshold
#             use_reg = best if best_cov >= MIN_COVER else None
#
#             # Smart tagging with ALL signals
#             tag = tag_line_smart(ln, use_reg, body)
#
#             # Build tagged line object
#             tagged.append({
#                 "tag": tag,
#                 "text": ln["text"],
#                 "bbox": ln["bbox"],
#                 "size": ln["size"],
#                 "bold": ln["bold"],
#                 "region": use_reg
#             })
#
#         # Save tagged output
#         outp = os.path.join(out_dir, f"page_{pnum:03d}.tags.json")
#         with open(outp, "w", encoding="utf-8") as f:
#             json.dump({"page": pnum, "tags": tagged}, f, indent=2)
#
#         print(f"Wrote: {outp} ({len(tagged)} lines tagged)")
#
#
# if __name__ == "__main__":
#     if len(sys.argv) < 5:
#         print("Usage: python FuseLines.py <pdf_path> <layout_dir> <lines_dir> <out_dir>")
#         sys.exit(1)
#
#     main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


# !/usr/bin/env python3
import os, sys, json, glob, re

# ========== CONFIGURATION ==========
MIN_COVER = 0.35  # line coverage by region

# Pattern matching constants
BULLET_CHARS = "•·●▪◦‣-–—*"
CAP_RE = re.compile(r'^[A-Z0-9 ,:\-()\/]+$')

H1_PATTERNS = [
    re.compile(r'^\s*(abstract|preface|introduction|conclusion|references|acknowledgments?)\s*$', re.I),
]

H2_PATTERNS = [
    re.compile(r'^\s*\d+[\.\)]\s+\S'),  # 1. Title / 1) Title
    re.compile(r'^\s*[IVXLC]+\.\s+\S'),  # I. Title
    re.compile(r'^\s*[A-Z]\)\s+\S'),  # A) Title
]

CAPTION_PAT = re.compile(r'^\s*(figure|fig\.|table|tbl\.?)\s*\d+[\.:)]', re.I)


# ========== HELPER FUNCTIONS ==========

def iou(a, b):
    """Calculate Intersection over Union (line overlap with region)"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    inter = (x1 - x0) * (y1 - y0)
    area = (ax1 - ax0) * (ay1 - ay0)
    return inter / max(1e-9, area)


def looks_like_caption(txt):
    """Detect figure/table captions: 'Figure 1:', 'Table 2.', etc."""
    return CAPTION_PAT.match((txt or "").strip()) is not None


def looks_like_list_item(txt):
    """Detect list items by bullet chars or numbering"""
    s = (txt or "").strip()
    if not s:
        return False

    # Bullet character at start
    if s[0] in BULLET_CHARS:
        return True

    # Numbered lists: 1. , 2) , etc.
    if re.match(r'^\s*\d+[\.\)]\s', s):
        return True

    # Lettered lists: a. , b) , A. , B)
    if re.match(r'^\s*[a-zA-Z][\.\)]\s', s):
        return True

    # Dash bullets with space: - Item, – Item, — Item
    if re.match(r'^\s*[-–—]\s+', s):
        return True

    return False


def looks_like_h1_pattern(txt):
    """Detect H1 by known section names"""
    s = (txt or "").strip()
    if not s:
        return False

    for pat in H1_PATTERNS:
        if pat.match(s):
            return True
    return False


def looks_like_h2_pattern(txt):
    """Detect H2 by numbering patterns"""
    s = (txt or "").strip()
    if not s:
        return False

    for pat in H2_PATTERNS:
        if pat.match(s):
            return True

    # Title-case heuristic: 3-12 words, starts capital, no period
    words = s.split()
    if 3 <= len(words) <= 12 and s[0].isupper() and not s.endswith('.'):
        return True

    return False


def is_short(txt):
    """Check if text is short (likely heading, not paragraph)"""
    return len((txt or "").strip()) <= 100


def is_all_caps(txt):
    """Check if text is ALL CAPS (with numbers/punctuation allowed)"""
    s = (txt or "").strip()
    if not s:
        return False

    letters = "".join([c for c in s if c.isalpha()])
    return bool(letters) and CAP_RE.match(s)


# ========== SMART TAGGING FUNCTION WITH SOURCE TRACKING ==========

def tag_line_smart(ln, region, body):
    """
    Intelligent tagging with priority system and SOURCE TRACKING.

    Returns: (tag, source)
    where source is one of:
    - "ML:region_type" (from YOLO/ML detection)
    - "heuristic:pattern" (from pattern matching)
    - "heuristic:font" (from font analysis)
    - "heuristic:caps" (from all-caps detection)
    - "default" (fallback to P)
    """
    txt = (ln.get("text") or "").strip()
    size = float(ln.get("size", 10.0))
    bold = bool(ln.get("bold", False))

    # ===== PRIORITY 1: High-Confidence Patterns =====

    if looks_like_caption(txt):
        return "FigureCaptionCandidate", "heuristic:caption_pattern"

    if looks_like_list_item(txt):
        return "LI", "heuristic:list_pattern"

    # ===== PRIORITY 2: ML Region + Font Analysis =====

    region_cls = None
    if region:
        region_cls = (region.get("cls") or region.get("label") or "").lower()

    if region_cls:
        # --- TITLE / SECTION HEADER REGIONS ---
        if region_cls in ("title", "section_header"):
            # Very large text in title region → H1
            if size >= body * 1.8 and is_short(txt):
                return "H1", f"ML:{region_cls}+font"

            # Medium-large or bold in title region → H2
            if (size >= body * 1.4 and is_short(txt)) or (size >= body * 1.25 and bold):
                return "H2", f"ML:{region_cls}+font"

            # Title region but normal size → still H2 (trust the ML)
            if is_short(txt):
                return "H2", f"ML:{region_cls}"

        # --- CAPTION REGIONS ---
        if region_cls == "caption":
            return "FigureCaptionCandidate", f"ML:{region_cls}"

        # --- LIST REGIONS ---
        if region_cls in ("list", "list_item"):
            return "LI", f"ML:{region_cls}"

        # --- TABLE REGIONS ---
        if region_cls == "table":
            return "TableRowCandidate", f"ML:{region_cls}"

        # --- FIGURE / PICTURE REGIONS ---
        if region_cls in ("figure", "picture"):
            low = txt.lower()
            if "figure" in low or "fig." in low or "table" in low or "tbl" in low:
                return "FigureCaptionCandidate", f"ML:{region_cls}+pattern"
            return "P", f"ML:{region_cls}"

        # --- FOOTNOTE REGIONS ---
        if region_cls == "footnote":
            return "P", f"ML:{region_cls}"

        # --- PAGE HEADER/FOOTER ---
        if region_cls in ("page_header", "page_footer"):
            return "P", f"ML:{region_cls}"

    # ===== PRIORITY 3: Pattern-Based Headings =====

    if looks_like_h1_pattern(txt):
        return "H1", "heuristic:h1_pattern"

    if looks_like_h2_pattern(txt):
        return "H2", "heuristic:h2_pattern"

    # ===== PRIORITY 4: Font-Based Analysis =====

    # Very large text → H1
    if size >= body * 1.8 and is_short(txt):
        return "H1", "heuristic:font_size"

    # Large text → H2
    if size >= body * 1.4 and is_short(txt):
        return "H2", "heuristic:font_size"

    # Bold + moderately large → H2
    if bold and size >= body * 1.25 and is_short(txt):
        return "H2", "heuristic:font_bold"

    # ===== PRIORITY 5: All-Caps Heuristic =====

    if is_all_caps(txt) and size >= body * 1.2 and is_short(txt):
        return "H2", "heuristic:all_caps"

    # ===== PRIORITY 6: Default =====
    return "P", "default"


# ========== MAIN PROCESSING ==========

def main(pdf_path, layout_dir, lines_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    pages = sorted(glob.glob(os.path.join(lines_dir, "page_*.lines.json")))

    if not pages:
        print(f"No line files found in {lines_dir}")
        sys.exit(1)

    # Track statistics
    stats = {
        'total_tags': 0,
        'ml_tags': 0,
        'heuristic_tags': 0,
        'source_breakdown': {},
        'tag_types': {}
    }

    for lp in pages:
        # Load lines data
        with open(lp, "r", encoding="utf-8") as f:
            L = json.load(f)

        pnum = L["page"]
        body = L.get("body_size", 10.0)
        lines = L["lines"]

        # Load corresponding layout data (optional)
        layoutp = os.path.join(layout_dir, f"page_{pnum:03d}.layout.json")
        regions = []

        if os.path.exists(layoutp):
            with open(layoutp, "r", encoding="utf-8") as f:
                R = json.load(f)
            regions = R.get("regions", [])

        # Tag each line
        tagged = []
        page_ml_count = 0
        page_heuristic_count = 0

        for ln in lines:
            # Find best matching region
            best = None
            best_cov = 0.0

            for reg in regions:
                cov = iou(ln["bbox"], reg["bbox"])
                if cov > best_cov:
                    best, best_cov = reg, cov

            # Use region only if coverage meets threshold
            use_reg = best if best_cov >= MIN_COVER else None

            # Smart tagging with source tracking
            tag, source = tag_line_smart(ln, use_reg, body)

            # Track statistics
            stats['total_tags'] += 1
            stats['tag_types'][tag] = stats['tag_types'].get(tag, 0) + 1
            stats['source_breakdown'][source] = stats['source_breakdown'].get(source, 0) + 1

            if source.startswith("ML:"):
                stats['ml_tags'] += 1
                page_ml_count += 1
            else:
                stats['heuristic_tags'] += 1
                page_heuristic_count += 1

            # Build tagged line object
            tagged.append({
                "tag": tag,
                "text": ln["text"],
                "bbox": ln["bbox"],
                "size": ln["size"],
                "bold": ln.get("bold", False),
                "italic": ln.get("italic", False),
                "region": use_reg,
                "source": source  # NEW: track source
            })

        # Save tagged output
        outp = os.path.join(out_dir, f"page_{pnum:03d}.tags.json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump({
                "page": pnum,
                "tags": tagged,
                "stats": {
                    "ml_tags": page_ml_count,
                    "heuristic_tags": page_heuristic_count,
                    "total": len(tagged)
                }
            }, f, indent=2)

        # Print page summary
        ml_pct = (page_ml_count / len(tagged) * 100) if tagged else 0
        print(f"Page {pnum}: {len(tagged)} tags ({page_ml_count} ML [{ml_pct:.1f}%], {page_heuristic_count} heuristic)")

    # Print overall statistics
    print(f"\n{'=' * 70}")
    print("TAGGING STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total tags: {stats['total_tags']}")
    print(f"  • ML-based: {stats['ml_tags']} ({stats['ml_tags'] / stats['total_tags'] * 100:.1f}%)")
    print(
        f"  • Heuristic-based: {stats['heuristic_tags']} ({stats['heuristic_tags'] / stats['total_tags'] * 100:.1f}%)")

    print(f"\nTag type distribution:")
    for tag, count in sorted(stats['tag_types'].items(), key=lambda x: -x[1]):
        print(f"  • {tag}: {count}")

    print(f"\nSource breakdown:")
    for source, count in sorted(stats['source_breakdown'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_tags'] * 100
        print(f"  • {source}: {count} ({pct:.1f}%)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python FuseLines.py <pdf_path> <layout_dir> <lines_dir> <out_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])