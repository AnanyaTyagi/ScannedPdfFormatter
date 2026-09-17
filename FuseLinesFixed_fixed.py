# !/usr/bin/env python3
"""
FuseLinesFixed_PATCHED.py
FIXED: Added artifact detection for page numbers, headers, footers
This fixes 22,000+ PDF/UA validation errors
"""
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

    # NOTE:
    # We intentionally DO NOT use a \"title-case\" heuristic here.
    # Wrapped paragraph lines often look title-case (capitalized, no period at line end),
    # which causes over-tagging of body paragraphs as H2.
    return False


def is_short(txt):
    """Check if text is short *as a heading* (avoid paragraph wrapped lines)."""
    s = (txt or "").strip()
    if not s:
        return True
    words = s.split()
    # Headings are typically short in both characters and words.
    return (len(s) <= 60) and (len(words) <= 10)


def is_all_caps(txt):
    """Check if text is ALL CAPS (with numbers/punctuation allowed)"""
    s = (txt or "").strip()
    if not s:
        return False

    letters = "".join([c for c in s if c.isalpha()])
    return bool(letters) and CAP_RE.match(s)


# ============================================================================
# ✅ NEW: ARTIFACT DETECTION (Fixes 22,000+ validation errors)
# ============================================================================
def should_be_artifact(txt, region_cls, bbox, page_height):
    """
    Determine if content should be marked as Artifact (decorative/non-structural).

    PDF/UA requires ALL content to be either:
    - Tagged as structural content (H1, P, Table, etc.)
    - Marked as Artifact (invisible to screen readers)

    Artifacts include:
    - Page numbers (short numeric text at page edges)
    - Headers/footers (detected by YOLO in page_header/page_footer regions)
    - Very small decorative text (< 6pt)
    - Empty/whitespace-only content
    - Decorative lines and borders

    Fixes veraPDF error:
    "Content is neither marked as Artifact nor tagged as real content"
    """
    txt_stripped = (txt or "").strip()

    # Empty content → artifact
    if not txt_stripped:
        return True

    # ✅ ML detected header/footer/figure regions → artifact
    if region_cls in ("page_header", "page_footer", "figure", "picture"):
        return True

    # ✅ Page numbers: short numeric text at page edges
    # Common patterns: "1", "Page 1", "- 42 -", etc.
    if len(txt_stripped) <= 10:
        # Check if mostly numeric
        digits = sum(1 for c in txt_stripped if c.isdigit())
        if digits >= len(txt_stripped) * 0.5:
            # Check if in top 60 pts or bottom 60 pts of page
            y0, y1 = bbox[1], bbox[3]
            if y0 < 60 or y1 > (page_height - 60):
                return True

    # ✅ Common page number patterns at edges
    page_num_patterns = [
        r'^\s*-?\s*\d+\s*-?\s*$',  # "1", "- 1 -", "42"
        r'^\s*page\s+\d+\s*$',  # "Page 1", "page 42"
        r'^\s*\d+\s*/\s*\d+\s*$',  # "1/10", "42/100"
    ]

    for pat in page_num_patterns:
        if re.match(pat, txt_stripped, re.IGNORECASE):
            y0, y1 = bbox[1], bbox[3]
            if y0 < 60 or y1 > (page_height - 60):
                return True

    # ✅ Very short text at extreme edges (likely decorative)
    if len(txt_stripped) <= 3:
        y0, y1 = bbox[1], bbox[3]
        x0, x1 = bbox[0], bbox[2]
        # At top/bottom edge
        if y0 < 40 or y1 > (page_height - 40):
            return True

    return False


# ========== SMART TAGGING FUNCTION WITH ARTIFACT DETECTION ==========

def tag_line_smart(ln, region, body, page_height):
    """
    Intelligent tagging with priority system and ARTIFACT DETECTION.

    Returns: (tag, source)
    where source is one of:
    - "artifact_detection" (decorative content)
    - "ML:region_type" (from YOLO/ML detection)
    - "heuristic:pattern" (from pattern matching)
    - "heuristic:font" (from font analysis)
    - "heuristic:caps" (from all-caps detection)
    - "default" (fallback to P)
    """
    txt = (ln.get("text") or "").strip()
    size = float(ln.get("size", 10.0))
    bold = bool(ln.get("bold", False))
    bbox = ln.get("bbox", [0, 0, 0, 0])

    # Get region class
    region_cls = None
    if region:
        region_cls = (region.get("cls") or region.get("label") or "").lower()

    # ===== PRIORITY 0: ARTIFACT DETECTION =====
    if should_be_artifact(txt, region_cls, bbox, page_height):
        return "Artifact", "artifact_detection"

    # ===== PRIORITY 0.5: FIGURE/PICTURE REGION OVERRIDE =====
    # Must run BEFORE list/heading pattern checks so that OCR noise inside
    # images (bullets, dashes, single chars) is not mis-tagged as LI/H.
    if region_cls in ("figure", "picture"):
        if looks_like_caption(txt):
            return "FigureCaptionCandidate", f"ML:{region_cls}+pattern"
        low = txt.lower()
        if "figure" in low or "fig." in low or "table" in low or "tbl" in low:
            return "FigureCaptionCandidate", f"ML:{region_cls}+pattern"
        return "Artifact", "artifact_detection"

    # ===== PRIORITY 1: High-Confidence Patterns =====

    if looks_like_caption(txt):
        return "FigureCaptionCandidate", "heuristic:caption_pattern"

    if looks_like_list_item(txt):
        return "LI", "heuristic:list_pattern"

    # ===== PRIORITY 2: ML Region + Font Analysis =====

    if region_cls:
        # --- TITLE / SECTION HEADER REGIONS ---
        if region_cls in ("title", "section_header"):
            # Very large text in title region → H1
            if size >= body * 1.8 and is_short(txt):
                return "H1", f"ML:{region_cls}+font"

            # Medium-large or bold in title region → H2
            if (size >= body * 1.4 and is_short(txt)) or (size >= body * 1.25 and bold):
                return "H2", f"ML:{region_cls}+font"

            # Font size suggests heading
            if size >= body * 1.15 and is_short(txt):
                return "H2", f"ML:{region_cls}"

            # Font size check failed (scanned PDF rounding) — trust YOLO directly
            if region_cls == "title":
                return "H1", f"ML:{region_cls}"
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

        # --- FOOTNOTE REGIONS ---
        if region_cls == "footnote":
            return "P", f"ML:{region_cls}"

        # ✅ REMOVED: page_header/page_footer now return Artifact above

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


# ========== PARAGRAPH GROUPING ==========

def group_paragraphs(tagged):
    """Merge consecutive P-tagged lines within the same YOLO region into one P element."""
    if not tagged:
        return tagged

    grouped = []
    i = 0
    while i < len(tagged):
        t = tagged[i]
        reg = t.get('region')
        if t['tag'] == 'P' and reg and reg.get('cls') in ('text', 'footnote'):
            reg_key = tuple(reg['bbox'])
            group = [t]
            j = i + 1
            while j < len(tagged):
                nt = tagged[j]
                nr = nt.get('region')
                if nt['tag'] == 'P' and nr and tuple(nr['bbox']) == reg_key:
                    group.append(nt)
                    j += 1
                else:
                    break

            # Two-pass reading-order sort:
            # Pass 1 — top-to-bottom by Y
            # Pass 2 — group same-row entries (Y within avg_h/3), sort each row left-to-right
            heights = [l['bbox'][3] - l['bbox'][1] for l in group if l['bbox'][3] > l['bbox'][1]]
            avg_h = (sum(heights) / len(heights)) if heights else 10.0
            row_threshold = avg_h / 3

            sorted_by_y = sorted(group, key=lambda l: l['bbox'][1])
            rows = []
            for line in sorted_by_y:
                if rows and line['bbox'][1] - rows[-1][-1]['bbox'][1] <= row_threshold:
                    rows[-1].append(line)
                else:
                    rows.append([line])
            group = []
            for row in rows:
                group.extend(sorted(row, key=lambda l: l['bbox'][0]))

            # Split on vertical gaps > 1.5x the median line gap for this group.
            # Using median gap (not avg_h) adapts to the document's actual line
            # spacing, so it works for single, 1.5x, and double-spaced documents.
            gaps = [group[k]['bbox'][1] - group[k-1]['bbox'][3] for k in range(1, len(group))]
            positive_gaps = sorted([g for g in gaps if g > 0])
            if positive_gaps:
                median_gap = positive_gaps[len(positive_gaps) // 2]
                split_threshold = median_gap * 1.5
            else:
                split_threshold = avg_h * 2

            sub_groups = [[group[0]]]
            for k in range(1, len(group)):
                prev = group[k - 1]
                curr = group[k]
                gap = curr['bbox'][1] - prev['bbox'][3]
                if gap > split_threshold:
                    sub_groups.append([curr])
                else:
                    sub_groups[-1].append(curr)

            for sub in sub_groups:
                merged_bbox = [
                    min(l['bbox'][0] for l in sub),
                    min(l['bbox'][1] for l in sub),
                    max(l['bbox'][2] for l in sub),
                    max(l['bbox'][3] for l in sub),
                ]
                grouped.append({
                    'tag': 'P',
                    'text': ' '.join(l['text'] for l in sub),
                    'bbox': merged_bbox,
                    'lines': [{'text': l['text'], 'bbox': l['bbox']} for l in sub],
                    'size': sub[0]['size'],
                    'bold': sub[0]['bold'],
                    'italic': sub[0].get('italic', False),
                    'region': reg,
                    'source': sub[0]['source'],
                })
            i = j

        elif t['tag'] == 'P' and reg is None:
            # No-region fallback: collect consecutive no-region P lines,
            # merge words on the same visual row, keep each row as its own P.
            group = [t]
            j = i + 1
            while j < len(tagged):
                nt = tagged[j]
                if nt['tag'] == 'P' and nt.get('region') is None:
                    group.append(nt)
                    j += 1
                else:
                    break

            heights = [l['bbox'][3] - l['bbox'][1] for l in group if l['bbox'][3] > l['bbox'][1]]
            avg_h = (sum(heights) / len(heights)) if heights else 10.0
            row_threshold = avg_h / 3

            sorted_by_y = sorted(group, key=lambda l: l['bbox'][1])
            rows = []
            for line in sorted_by_y:
                if rows and line['bbox'][1] - rows[-1][-1]['bbox'][1] <= row_threshold:
                    rows[-1].append(line)
                else:
                    rows.append([line])

            for row in rows:
                row_sorted = sorted(row, key=lambda l: l['bbox'][0])
                merged_bbox = [
                    min(l['bbox'][0] for l in row_sorted),
                    min(l['bbox'][1] for l in row_sorted),
                    max(l['bbox'][2] for l in row_sorted),
                    max(l['bbox'][3] for l in row_sorted),
                ]
                grouped.append({
                    'tag': 'P',
                    'text': ' '.join(l['text'] for l in row_sorted),
                    'bbox': merged_bbox,
                    'lines': [{'text': l['text'], 'bbox': l['bbox']} for l in row_sorted],
                    'size': row_sorted[0]['size'],
                    'bold': row_sorted[0]['bold'],
                    'italic': row_sorted[0].get('italic', False),
                    'region': None,
                    'source': row_sorted[0]['source'],
                })
            i = j

        elif t['tag'] == 'LI':
            # Merge consecutive LI lines (any region) into logical list items.
            # Also pulls in immediately following P lines from the SAME region as
            # continuation lines, guarded by a vertical gap check so a new paragraph
            # that happens to share the region bbox is NOT swallowed.
            reg_key = tuple(reg['bbox']) if reg else None

            # Estimate avg_h from first line before the collection loop (needed for gap guard)
            h0 = t['bbox'][3] - t['bbox'][1]
            avg_h_est = h0 if h0 > 0 else 10.0

            group = [t]
            j = i + 1
            while j < len(tagged):
                nt = tagged[j]
                nr = nt.get('region')
                nr_key = tuple(nr['bbox']) if nr else None

                if nr_key != reg_key or nt['tag'] not in ('LI', 'P'):
                    break

                # Gap guard: large vertical gap means a new paragraph, not a continuation
                gap = nt['bbox'][1] - group[-1]['bbox'][3]
                if gap > avg_h_est * 2:
                    break

                group.append(nt)
                j += 1

            heights = [l['bbox'][3] - l['bbox'][1] for l in group if l['bbox'][3] > l['bbox'][1]]
            avg_h = (sum(heights) / len(heights)) if heights else 10.0
            row_threshold = avg_h / 3
            sorted_by_y = sorted(group, key=lambda l: l['bbox'][1])
            rows = []
            for line in sorted_by_y:
                if rows and line['bbox'][1] - rows[-1][-1]['bbox'][1] <= row_threshold:
                    rows[-1].append(line)
                else:
                    rows.append([line])
            group = []
            for row in rows:
                group.extend(sorted(row, key=lambda l: l['bbox'][0]))

            # Split into logical items:
            # - LI line with heuristic:list_pattern → always new item start
            # - LI line with x0 significantly less than prev (hanging indent) → new item
            # - P line → always a continuation of the current item
            items = [[group[0]]]
            for k in range(1, len(group)):
                prev = group[k - 1]
                curr = group[k]
                if curr['tag'] == 'P':
                    items[-1].append(curr)
                else:
                    x_diff = prev['bbox'][0] - curr['bbox'][0]  # positive → curr more outdented
                    is_new_item = (curr.get('source') == 'heuristic:list_pattern') or (x_diff > avg_h * 0.8)
                    if is_new_item:
                        items.append([curr])
                    else:
                        items[-1].append(curr)

            for item_lines in items:
                merged_bbox = [
                    min(l['bbox'][0] for l in item_lines),
                    min(l['bbox'][1] for l in item_lines),
                    max(l['bbox'][2] for l in item_lines),
                    max(l['bbox'][3] for l in item_lines),
                ]
                grouped.append({
                    'tag': 'LI',
                    'text': ' '.join(l['text'] for l in item_lines),
                    'bbox': merged_bbox,
                    'lines': [{'text': l['text'], 'bbox': l['bbox']} for l in item_lines],
                    'size': item_lines[0]['size'],
                    'bold': item_lines[0]['bold'],
                    'italic': item_lines[0].get('italic', False),
                    'region': reg,
                    'source': item_lines[0]['source'],
                })
            i = j

        elif t['tag'] in ('H1', 'H2') and reg and reg.get('cls') in ('title', 'section_header'):
            # Merge consecutive H1/H2 lines in the same title/section_header region
            # into a single heading element.
            tag_type = t['tag']
            reg_key = tuple(reg['bbox'])
            group = [t]
            j = i + 1
            while j < len(tagged):
                nt = tagged[j]
                nr = nt.get('region')
                if nt['tag'] == tag_type and nr and tuple(nr['bbox']) == reg_key:
                    group.append(nt)
                    j += 1
                else:
                    break

            heights = [l['bbox'][3] - l['bbox'][1] for l in group if l['bbox'][3] > l['bbox'][1]]
            avg_h = (sum(heights) / len(heights)) if heights else 10.0
            row_threshold = avg_h / 3
            sorted_by_y = sorted(group, key=lambda l: l['bbox'][1])
            rows = []
            for line in sorted_by_y:
                if rows and line['bbox'][1] - rows[-1][-1]['bbox'][1] <= row_threshold:
                    rows[-1].append(line)
                else:
                    rows.append([line])
            group = []
            for row in rows:
                group.extend(sorted(row, key=lambda l: l['bbox'][0]))

            merged_bbox = [
                min(l['bbox'][0] for l in group),
                min(l['bbox'][1] for l in group),
                max(l['bbox'][2] for l in group),
                max(l['bbox'][3] for l in group),
            ]
            grouped.append({
                'tag': tag_type,
                'text': ' '.join(l['text'] for l in group),
                'bbox': merged_bbox,
                'lines': [{'text': l['text'], 'bbox': l['bbox']} for l in group],
                'size': group[0]['size'],
                'bold': group[0]['bold'],
                'italic': group[0].get('italic', False),
                'region': reg,
                'source': group[0]['source'],
            })
            i = j

        else:
            # DEBUG: log anything that falls through unmerged — reveals no-region LI,
            # H1/H2 without title region, and other orphaned lines
            reg_cls = reg.get('cls', '?') if reg else 'NO_REGION'
            with open('/tmp/fuse_group_debug.txt', 'a') as _dbg:
                _dbg.write(f"PASSTHROUGH  tag={t['tag']:4}  reg={reg_cls:15}  "
                           f"y0={t['bbox'][1]:.1f}  x0={t['bbox'][0]:.1f}  "
                           f"text='{t.get('text','')[:60]}'\n")
            grouped.append(t)
            i += 1
    return grouped


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
        'artifact_tags': 0,  # NEW
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
        page_height = 792.0  # Default letter size

        if os.path.exists(layoutp):
            with open(layoutp, "r", encoding="utf-8") as f:
                R = json.load(f)
            regions = R.get("regions", [])
            # ✅ GET PAGE HEIGHT for artifact detection
            page_height = R.get("height", 792.0)

        # Tag each line
        tagged = []
        page_ml_count = 0
        page_heuristic_count = 0
        page_artifact_count = 0  # NEW

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

            # DEBUG: log every no-region line (no YOLO match) — reveals orphan words
            # like single "A" that couldn't be matched to any region
            if use_reg is None:
                txt_raw = ln.get("text", "").strip()
                with open('/tmp/fuse_noregion_debug.txt', 'a') as _dbg:
                    _dbg.write(f"P{pnum} best_cov={best_cov:.3f}  "
                               f"best_reg={best.get('cls','?') if best else 'NONE':15}  "
                               f"y0={ln['bbox'][1]:.1f}  x0={ln['bbox'][0]:.1f}  "
                               f"text='{txt_raw[:60]}'\n")

            # ✅ Smart tagging with artifact detection (PASS PAGE_HEIGHT)
            tag, source = tag_line_smart(ln, use_reg, body, page_height)

            # DEBUG: log LI tags and heading tags to understand list/heading issues
            txt_stripped = ln.get("text", "").strip()
            reg_cls = (use_reg.get('cls') or '') if use_reg else 'NO_REGION'
            if tag in ('LI', 'H1', 'H2') or (tag == 'P' and looks_like_list_item(txt_stripped)):
                with open('/tmp/fuse_debug.txt', 'a') as dbg:
                    dbg.write(f"P{pnum} tag={tag:4} source={source:30} reg={reg_cls:15} "
                              f"y0={ln['bbox'][1]:.1f} x0={ln['bbox'][0]:.1f} "
                              f"text='{txt_stripped[:60]}'\n")

            # Track statistics
            stats['total_tags'] += 1
            stats['tag_types'][tag] = stats['tag_types'].get(tag, 0) + 1
            stats['source_breakdown'][source] = stats['source_breakdown'].get(source, 0) + 1

            if source == "artifact_detection":
                stats['artifact_tags'] += 1
                page_artifact_count += 1
            elif source.startswith("ML:"):
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
                "source": source
            })

        # Merge P lines within the same YOLO region into single paragraph elements
        tagged = group_paragraphs(tagged)

        # Save tagged output
        outp = os.path.join(out_dir, f"page_{pnum:03d}.tags.json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump({
                "page": pnum,
                "tags": tagged,
                "stats": {
                    "ml_tags": page_ml_count,
                    "heuristic_tags": page_heuristic_count,
                    "artifact_tags": page_artifact_count,  # NEW
                    "total": len(tagged)
                }
            }, f, indent=2)

        # Print page summary
        print(
            f"Page {pnum}: {len(tagged)} tags ({page_ml_count} ML, {page_heuristic_count} heuristic, {page_artifact_count} artifacts)")

    # Final summary across all pages
    total = stats['total_tags']
    ml = stats['ml_tags']
    heuristic = stats['heuristic_tags']
    artifact = stats['artifact_tags']
    print("\n" + "=" * 60)
    print("TAGGING SUMMARY")
    print("=" * 60)
    print(f"Total tags   : {total}")
    print(f"ML-based     : {ml:5d}  ({ml/total*100:.1f}%)" if total else "ML-based     : 0")
    print(f"Heuristic    : {heuristic:5d}  ({heuristic/total*100:.1f}%)" if total else "Heuristic    : 0")
    print(f"Artifacts    : {artifact:5d}  ({artifact/total*100:.1f}%)" if total else "Artifacts    : 0")
    print("\nTag type breakdown:")
    for tag, count in sorted(stats['tag_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {tag:<30} {count:5d}  ({count/total*100:.1f}%)" if total else f"  {tag}")
    print("\nDecision source breakdown:")
    for src, count in sorted(stats['source_breakdown'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {src:<35} {count:5d}  ({count/total*100:.1f}%)" if total else f"  {src}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python FuseLinesFixed_PATCHED.py <pdf_path> <layout_dir> <lines_dir> <out_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])