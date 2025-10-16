#!/usr/bin/env python3
import os, sys, json
import pikepdf
from pikepdf import Name, Dictionary, Array, String


# ---------------- ParentTree builder ----------------
class ParentTreeBuilder:
    def __init__(self):
        self.by_key = {}

    def ensure_page_key(self, key):
        self.by_key.setdefault(key, [])

    def add(self, key, mcid, struct_elem):
        arr = self.by_key.setdefault(key, [])
        if len(arr) <= mcid:
            arr.extend([None] * (mcid + 1 - len(arr)))
        arr[mcid] = struct_elem

    def build_numbertree_dict(self):
        nums = Array()
        for key in sorted(self.by_key.keys()):
            arr = Array()
            for se in self.by_key[key]:
                arr.append(se if se is not None else None)
            nums.append(key)
            nums.append(arr)
        nt = Dictionary()
        nt[Name('/Nums')] = nums
        return nt


# ---------------- helpers ----------------
def pdf_name(s):
    return Name(s if s.startswith('/') else '/' + s)


def as_str(s):
    return String(s) if not isinstance(s, String) else s


def make_indirect(pdf, obj):
    return pdf.make_indirect(obj)


def ensure_markinfo(pdf):
    root = pdf.Root
    mi = root.get('/MarkInfo', None)
    if mi is None:
        mi = Dictionary()
        root[Name('/MarkInfo')] = make_indirect(pdf, mi)
    mi[Name('/Marked')] = True


ROLEMAP = {
    'Document': 'Document', 'Part': 'Part', 'H1': 'H1', 'H2': 'H2', 'P': 'P',
    'L': 'L', 'LI': 'LI', 'Table': 'Table', 'TR': 'TR', 'TH': 'TH', 'TD': 'TD',
    'Figure': 'Figure', 'Caption': 'Caption', 'Span': 'Span'
}

TAG_TO_S = {
    'H1': 'H1', 'H2': 'H2', 'P': 'P', 'LI': 'LI', 'Table': 'Table', 'TR': 'TR',
    'TH': 'TH', 'TD': 'TD', 'Figure': 'Figure', 'Caption': 'Caption'
}


def ensure_struct_tree_root(pdf):
    root = pdf.Root
    if '/StructTreeRoot' in root:
        return root['/StructTreeRoot']
    st = Dictionary()
    st[Name('/Type')] = pdf_name('StructTreeRoot')
    rm = Dictionary()
    for k, v in ROLEMAP.items():
        rm[pdf_name(k)] = pdf_name(v)
    st[Name('/RoleMap')] = rm
    root[Name('/StructTreeRoot')] = make_indirect(pdf, st)
    return root['/StructTreeRoot']


def set_doc_metadata(pdf, title="Tagged Document", lang="en-US"):
    info = pdf.docinfo
    info[Name('/Title')] = as_str(title)
    pdf.Root[Name('/Lang')] = as_str(lang)


def make_selement(pdf, S, title_text=None, parent=None):
    el = Dictionary()
    el[Name('/Type')] = pdf_name('StructElem')
    el[Name('/S')] = pdf_name(S)
    if title_text:
        el[Name('/T')] = as_str(title_text)
    if parent is not None:
        el[Name('/P')] = parent
    return pdf.make_indirect(el)


def attach_k_ref(el, page_obj, mcid):
    kid = Dictionary()
    kid[Name('/Type')] = pdf_name('MCR')  # Marked Content Reference
    kid[Name('/MCID')] = mcid
    kid[Name('/Pg')] = page_obj
    if '/K' not in el:
        el[Name('/K')] = Array([kid])
    else:
        arr = el['/K'] if isinstance(el['/K'], Array) else Array([el['/K']])
        arr.append(kid)
        el[Name('/K')] = arr


# --------- bbox helpers + text drawing ---------
def _page_size_pts(page_obj):
    """Return (w, h) in PDF points from /MediaBox."""
    mb = page_obj.get('/MediaBox', None)
    if not mb or len(mb) < 4:
        return 612.0, 792.0
    x0, y0, x1, y1 = [float(m) for m in mb]
    return (x1 - x0), (y1 - y0)


def _tlbbox_to_pdf_coords(bbox_tl, page_h):
    """
    Convert top-left bbox [x0, y0, x1, y1] to PDF bottom-left coordinates.
    Returns (x, y, width, height, font_size)
    """
    if not bbox_tl or len(bbox_tl) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox_tl]
    if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
        return None

    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return None

    # PDF y-coordinate (bottom-left origin)
    x = x0
    y = page_h - y1

    # Font size matching the bbox height for better coverage
    # Using the full height helps Acrobat compute better highlight regions
    font_size = max(8.0, height)

    return x, y, width, height, font_size


def page_add_invisible_mcid_stream(pdf, page_obj, mcid, text, bbox=None, debug=False):
    """
    Add marked content with MCID and place invisible text at bbox.
    Uses Tr 3 (invisible) with proper positioning for highlight support.
    """
    # Ensure /Resources and /Font exist
    resources = page_obj.get('/Resources', None)
    if resources is None:
        resources = Dictionary()
        page_obj['/Resources'] = resources

    fonts = resources.get('/Font', None)
    if fonts is None:
        fonts = Dictionary()
        resources['/Font'] = fonts

    if '/F1' not in fonts:
        fdict = Dictionary({
            '/Type': pdf_name('Font'),
            '/Subtype': pdf_name('Type1'),
            '/BaseFont': pdf_name('Helvetica'),
        })
        try:
            fonts['/F1'] = pdf.make_indirect(fdict)
        except AttributeError:
            fonts['/F1'] = fdict

    # Get page dimensions
    _, page_h = _page_size_pts(page_obj)

    # Check if bbox is valid
    coords = _tlbbox_to_pdf_coords(bbox, page_h)

    if coords is None:
        if debug:
            print(f"  WARNING: Invalid bbox for MCID {mcid}: {bbox}")
        # Still add MCID marker but without positioned text
        content = f"""/Span << /MCID {int(mcid)} >> BDC
EMC
""".encode('utf-8')
    else:
        x, y, width, height, font_size = coords
        safe_text = (text or '').replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

        if debug:
            print(f"  MCID {mcid}: bbox={bbox} -> PDF coords=({x:.1f},{y:.1f}) size={font_size:.1f}")

        # Create invisible text at the exact bbox location
        # Tr 3 = invisible text rendering mode
        # Using actual text helps Acrobat compute highlight region
        content = f"""q
/Span << /MCID {int(mcid)} >> BDC
BT
/F1 {font_size:.2f} Tf
3 Tr
{x:.2f} {y:.2f} Td
({safe_text}) Tj
ET
EMC
Q
""".encode('utf-8')

    new_stream = pdf.make_stream(content)
    contents = page_obj.get('/Contents', None)
    if contents is None:
        page_obj[Name('/Contents')] = make_indirect(pdf, new_stream)
    elif isinstance(contents, Array):
        contents.append(make_indirect(pdf, new_stream))
    else:
        page_obj[Name('/Contents')] = Array([contents, make_indirect(pdf, new_stream)])


# ------------- tag attach helpers -------------
def attach_text_to_elem(pdf, se, page_obj, text, bbox, *, parenttree, page_key, mcid, debug=False):
    """Attach text content to a structure element with MCID."""
    se[Name('/ActualText')] = as_str(text)
    se[Name('/T')] = as_str(text)
    attach_k_ref(se, page_obj, mcid)
    page_add_invisible_mcid_stream(pdf, page_obj, mcid, text, bbox=bbox, debug=debug)
    parenttree.add(page_key, mcid, se)
    return mcid + 1


def tag_block(pdf, st_root, parent_se, page_obj, items, parenttree=None, page_key=None, start_mcid=0, debug=False):
    """Tag a block of items (paragraphs, list items, etc.)"""
    if '/StructParents' not in page_obj and page_key is not None:
        page_obj[Name('/StructParents')] = page_key
    mcid = int(start_mcid)

    for it in items:
        tag = it.get('tag', 'P')
        text = it.get('text', '')
        bbox = it.get('bbox', None)
        role = TAG_TO_S.get(tag, 'P')
        se = make_selement(pdf, role, parent=parent_se)
        se[Name('/ActualText')] = as_str(text)

        if '/K' not in parent_se:
            parent_se[Name('/K')] = Array([se])
        else:
            arr = parent_se['/K'] if isinstance(parent_se['/K'], Array) else Array([parent_se['/K']])
            arr.append(se)
            parent_se[Name('/K')] = arr

        attach_k_ref(se, page_obj, mcid)
        page_add_invisible_mcid_stream(pdf, page_obj, mcid, text, bbox=bbox, debug=debug)

        if parenttree is not None and page_key is not None:
            parenttree.add(page_key, mcid, se)

        mcid += 1
    return mcid


# ---------- JSON loading ----------
def load_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def load_tags_for_page(fused_dir, pnum):
    """Prefer promoted.json when present; otherwise use tags.json."""
    promoted = os.path.join(fused_dir, f'page_{pnum:03d}.promoted.json')
    normal = os.path.join(fused_dir, f'page_{pnum:03d}.tags.json')
    path = promoted if os.path.exists(promoted) else normal
    data = load_json(path, {}) or {}
    return data.get('tags') or []


# ---------------- main ----------------
def main(pdf_in, fused_dir, struct_dir, pdf_out, title="Tagged Reconstruction", lang="en-US", debug=False):
    pdf = pikepdf.open(pdf_in)
    set_doc_metadata(pdf, title=title, lang=lang)
    ensure_markinfo(pdf)

    st_root = ensure_struct_tree_root(pdf)

    doc_el = make_selement(pdf, 'Document', title_text=title, parent=st_root)
    st_root[Name('/K')] = Array([doc_el])

    part_el = make_selement(pdf, 'Part', parent=doc_el)
    doc_el[Name('/K')] = Array([part_el])

    parenttree = ParentTreeBuilder()
    next_key = 0

    num_pages = len(pdf.pages)
    print(f"Processing {num_pages} pages...")

    for i in range(num_pages):
        pnum = i + 1
        page = pdf.pages[i]
        page_obj = page.obj

        if debug:
            print(f"\n=== Page {pnum} ===")

        page_key = next_key
        next_key += 1
        page_obj[Name('/StructParents')] = page_key
        parenttree.ensure_page_key(page_key)

        next_mcid = 0

        # Load tags
        tags_raw = load_tags_for_page(fused_dir, pnum)
        if not tags_raw:
            if debug:
                print(f"  No tags found for page {pnum}")
            continue

        # Normalize tags
        tags = []
        bbox_count = 0
        for t in tags_raw:
            if not isinstance(t, dict):
                continue
            t.setdefault('tag', 'P')
            t.setdefault('text', '')
            t.setdefault('bbox', [0, 0, 0, 0])
            t.setdefault('region', None)
            bbox = t.get('bbox')
            if bbox and bbox != [0, 0, 0, 0]:
                bbox_count += 1
            tags.append(t)

        if debug:
            print(f"  Found {len(tags)} tags, {bbox_count} with valid bboxes")

        if not tags:
            continue

        page_part = make_selement(pdf, 'Part', title_text=f'Page {pnum}', parent=part_el)
        arr = part_el['/K'] if '/K' in part_el else Array()
        arr.append(page_part)
        part_el[Name('/K')] = arr

        # (A) Lists
        lists_json = load_json(os.path.join(struct_dir, f'page_{pnum:03d}.lists.json'), {}) or {}
        lists = lists_json.get('lists') or []
        li_texts = set()
        for lst in lists:
            if not isinstance(lst, dict):
                continue
            items = lst.get('items') or []
            if not items:
                continue
            L_el = make_selement(pdf, 'L', parent=page_part)
            arr2 = page_part['/K'] if '/K' in page_part else Array()
            arr2.append(L_el)
            page_part[Name('/K')] = arr2
            for item in items:
                if not item:
                    continue
                LI_el = make_selement(pdf, 'LI', parent=L_el)
                P_el = make_selement(pdf, 'P', parent=LI_el)
                LI_el[Name('/K')] = Array([P_el])
                arr3 = L_el['/K'] if '/K' in L_el else Array()
                arr3.append(LI_el)
                L_el[Name('/K')] = arr3
                li_texts.add(item)
                next_mcid = tag_block(
                    pdf, st_root, P_el, page_obj,
                    [{'tag': 'P', 'text': item, 'bbox': None}],
                    parenttree=parenttree, page_key=page_key,
                    start_mcid=next_mcid, debug=debug
                )

        # (B) Figures + captions
        def is_figure(t):
            r = t.get('region')
            return isinstance(r, dict) and r.get('cls') == 'figure'

        figures = [t for t in tags if is_figure(t)]
        caps = [t for t in tags if t.get('tag') == 'FigureCaptionCandidate']

        for f in figures:
            fig_el = make_selement(pdf, 'Figure', parent=page_part)
            cap_text = None
            cap_bbox = None
            fb = f.get('bbox') or [0, 0, 0, 0]
            if isinstance(fb, (list, tuple)) and len(fb) == 4:
                fy_bottom = fb[3]
                near = []
                for c in caps:
                    cb = c.get('bbox') or [0, 0, 0, 0]
                    if isinstance(cb, (list, tuple)) and len(cb) == 4:
                        if abs((cb[1] or 0) - fy_bottom) < 40:
                            near.append(c)
                if near:
                    cap_text = (near[0].get('text') or '').strip() or None
                    cap_bbox = near[0].get('bbox')

            if cap_text:
                cap_el = make_selement(pdf, 'Caption', parent=fig_el)
                fig_el[Name('/K')] = Array([cap_el])
                fig_el[Name('/Alt')] = as_str(cap_text)
                next_mcid = attach_text_to_elem(
                    pdf, cap_el, page_obj, cap_text, cap_bbox,
                    parenttree=parenttree, page_key=page_key, mcid=next_mcid, debug=debug
                )
            arr4 = page_part['/K'] if '/K' in page_part else Array()
            arr4.append(fig_el)
            page_part[Name('/K')] = arr4

        # (C) Tables
        tables_json = load_json(os.path.join(struct_dir, f'page_{pnum:03d}.tables.json'), {}) or {}
        for row in (tables_json.get('tables') or []):
            if not isinstance(row, dict):
                continue
            cells = row.get('cells') or []
            if not cells:
                continue
            table_el = make_selement(pdf, 'Table', parent=page_part)
            tr_el = make_selement(pdf, 'TR', parent=table_el)
            table_el[Name('/K')] = Array([tr_el])
            arr5 = page_part['/K'] if '/K' in page_part else Array()
            arr5.append(table_el)
            page_part[Name('/K')] = arr5
            for cell in cells:
                td_el = make_selement(pdf, 'TD', parent=tr_el)
                p_el = make_selement(pdf, 'P', parent=td_el)
                td_el[Name('/K')] = Array([p_el])
                arr6 = tr_el['/K'] if '/K' in tr_el else Array()
                arr6.append(td_el)
                tr_el[Name('/K')] = arr6
                next_mcid = tag_block(
                    pdf, st_root, p_el, page_obj,
                    [{'tag': 'P', 'text': str(cell) if cell is not None else '', 'bbox': None}],
                    parenttree=parenttree, page_key=page_key,
                    start_mcid=next_mcid, debug=debug
                )

        # (D) Headings & paragraphs
        consumed_texts = set([c.get('text', '') for c in caps if isinstance(c, dict)])
        consumed_texts.update(li_texts)
        text_items = []
        for t in tags:
            tagname = t.get('tag')
            if tagname in ('H1', 'H2', 'P'):
                tx = (t.get('text', '') or '').strip()
                if tx and tx not in consumed_texts:
                    text_items.append({'tag': tagname, 'text': tx, 'bbox': t.get('bbox')})

        if debug and text_items:
            print(f"  Processing {len(text_items)} text items (H1/H2/P)")

        for it in text_items:
            role = TAG_TO_S.get(it['tag'], 'P')
            se = make_selement(pdf, role, parent=page_part)
            kids = page_part['/K'] if '/K' in page_part else Array()
            kids.append(se)
            page_part[Name('/K')] = kids
            next_mcid = attach_text_to_elem(
                pdf, se, page_obj, it['text'], it.get('bbox'),
                parenttree=parenttree, page_key=page_key, mcid=next_mcid, debug=debug
            )

    # Build and attach ParentTree
    pt_dict = parenttree.build_numbertree_dict()
    st_root[Name('/ParentTree')] = make_indirect(pdf, pt_dict)
    st_root[Name('/ParentTreeNextKey')] = next_key
    ensure_markinfo(pdf)

    pdf.save(pdf_out, linearize=True)
    print(f"\nWrote tagged PDF: {os.path.abspath(pdf_out)}")
    print("Open in Adobe Acrobat and check View > Show/Hide > Navigation Panes > Tags")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python TagPikePDF.py input.pdf fused_tags structures output.pdf [--debug]")
        sys.exit(1)

    debug_mode = '--debug' in sys.argv
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], debug=debug_mode)