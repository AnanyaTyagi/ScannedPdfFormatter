#!/usr/bin/env python3
# TagPDFFinal.py - Simplified Version
# Creates tagged PDF with ActualText for screen readers
# Text overlay is always invisible (for structure only)

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
        nt[Name.Nums] = nums
        return nt

# ---------------- helpers ----------------
def pdf_name(s):
    return Name(s if isinstance(s, str) and s.startswith('/') else '/' + s)

def as_str(s):
    return String(s) if not isinstance(s, String) else s

def ensure_markinfo(pdf):
    root = pdf.Root
    mi = root.get(Name.MarkInfo, None)
    if mi is None:
        mi = Dictionary()
        root[Name.MarkInfo] = pdf.make_indirect(mi)
    mi[Name.Marked] = True
    return mi

# Role map
ROLEMAP = {
    'Document':'Document','Part':'Part',
    'H1':'H1','H2':'H2','H3':'H3','P':'P',
    'L':'L','LI':'LI','Lbl':'Lbl','LBody':'LBody',
    'Table':'Table','TR':'TR','TH':'TH','TD':'TD',
    'Figure':'Figure','Caption':'Caption','Span':'Span'
}

# Tag mapping
TAG_TO_S = {
    'H1': 'H1', 'H2': 'H2', 'H3': 'H3', 'P': 'P', 'LI': 'LI',
    'Table': 'Table', 'TR': 'TR', 'TH': 'TH', 'TD': 'TD',
    'Figure': 'Figure', 'Caption': 'Caption',
    'FigureCaptionCandidate': 'Caption',
    'TableRowCandidate': 'P',
}

def ensure_struct_tree_root(pdf):
    root = pdf.Root
    if Name.StructTreeRoot in root:
        return root[Name.StructTreeRoot]
    st = Dictionary()
    st[Name.Type] = Name.StructTreeRoot
    rm = Dictionary()
    for k, v in ROLEMAP.items():
        rm[pdf_name(k)] = pdf_name(v)
    st[Name.RoleMap] = rm
    root[Name.StructTreeRoot] = pdf.make_indirect(st)
    return root[Name.StructTreeRoot]

def set_doc_metadata(pdf, input_path, title=None, lang="en-US", author=None):
    if not title:
        title = os.path.splitext(os.path.basename(input_path))[0]

    if not hasattr(pdf, "docinfo") or pdf.docinfo is None:
        pdf.docinfo = Dictionary()
    pdf.docinfo[Name.Title] = as_str(title)
    pdf.docinfo[Name.Creator] = as_str("PDF Accessibility Tagger")
    pdf.docinfo[Name.Producer] = as_str("pikepdf")
    if author:
        pdf.docinfo[Name.Author] = as_str(author)

    pdf.Root[Name.Lang] = as_str(lang)

    vp = pdf.Root.get(Name.ViewerPreferences, None)
    if vp is None:
        vp = Dictionary()
        pdf.Root[Name.ViewerPreferences] = pdf.make_indirect(vp)
    vp[Name.DisplayDocTitle] = True

    xmp = f'''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{title}</rdf:li></rdf:Alt></dc:title>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
    meta = pdf.make_stream(xmp.encode("utf-8"))
    meta[Name.Type] = Name.Metadata
    meta[Name.Subtype] = Name.XML
    pdf.Root[Name.Metadata] = pdf.make_indirect(meta)

def create_page_bookmarks(pdf, fused_dir):
    """Create bookmarks from H1/H2 tags"""
    outlines = Dictionary()
    outlines[Name.Type] = Name.Outlines
    items = []

    for i, page in enumerate(pdf.pages, start=1):
        title_text = None
        for stem in ("promoted", "tags"):
            p = os.path.join(fused_dir, f'page_{i:03d}.{stem}.json')
            if os.path.exists(p):
                try:
                    data = json.load(open(p, "r", encoding="utf-8"))
                    for t in data.get("tags", []):
                        if t.get("tag") in ("H1","H2") and (t.get("text") or "").strip():
                            title_text = t["text"].strip()
                            break
                except Exception:
                    pass
            if title_text:
                break
        title = title_text or f"Page {i}"

        bm = Dictionary()
        bm[Name.Title] = as_str(title)
        bm[Name.Parent] = outlines
        bm[Name.Dest] = Array([page.obj, Name.Fit])
        items.append(pdf.make_indirect(bm))

    if items:
        outlines[Name.First] = items[0]
        outlines[Name.Last] = items[-1]
        outlines[Name.Count] = len(items)
        for idx, it in enumerate(items):
            if idx > 0:
                it[Name.Prev] = items[idx-1]
            if idx < len(items)-1:
                it[Name.Next] = items[idx+1]
    else:
        outlines[Name.Count] = 0
    pdf.Root[Name.Outlines] = pdf.make_indirect(outlines)

def make_selement(pdf, S, *, parent=None, title_text=None, actual_text=None):
    """Create structure element with ActualText for screen readers"""
    el = Dictionary()
    el[Name.Type] = Name.StructElem
    el[Name.S] = pdf_name(S)
    if parent is not None:
        el[Name.P] = parent
    if title_text:
        el[Name.T] = as_str(title_text)
    # ActualText is what JAWS reads
    if actual_text:
        el[Name.ActualText] = as_str(actual_text)
        if Name.T not in el:
            el[Name.T] = as_str(actual_text)
    return pdf.make_indirect(el)

def attach_k_ref(el, page_obj, mcid):
    """Link structure element to page content"""
    kid = Dictionary()
    kid[Name.Type] = Name.MCR
    kid[Name.Pg] = page_obj
    kid[Name.MCID] = mcid
    if Name.K not in el:
        el[Name.K] = Array([kid])
    else:
        arr = el[Name.K] if isinstance(el[Name.K], Array) else Array([el[Name.K]])
        arr.append(kid)
        el[Name.K] = arr

def page_add_invisible_mcid_stream(pdf, page_obj, mcid, text, *, bbox=None):
    """
    Add invisible text overlay for proper PDF structure.
    JAWS reads from ActualText, not from this overlay.
    """
    # Setup resources
    resources = page_obj.get(Name.Resources, None)
    if resources is None:
        resources = Dictionary()
        page_obj[Name.Resources] = resources

    # Font
    fonts = resources.get(Name.Font, None)
    if fonts is None:
        fonts = Dictionary()
        resources[Name.Font] = fonts
    if Name.F1 not in fonts:
        fnt = Dictionary()
        fnt[Name.Type] = Name.Font
        fnt[Name.Subtype] = Name.Type1
        fnt[Name.BaseFont] = Name.Helvetica
        fonts[Name.F1] = pdf.make_indirect(fnt)

    # Graphics state - ALWAYS INVISIBLE
    extg = resources.get(Name.ExtGState, None)
    if extg is None:
        extg = Dictionary()
        resources[Name.ExtGState] = extg
    if Name.GS1 not in extg:
        gs = Dictionary()
        gs[Name.Type] = Name.ExtGState
        gs[Name.ca] = 0.0  # Fill alpha = INVISIBLE
        gs[Name.CA] = 0.0  # Stroke alpha = INVISIBLE
        extg[Name.GS1] = pdf.make_indirect(gs)

    # Position
    mb = page_obj.get(Name.MediaBox, [0,0,612,792])
    page_h = float(mb[3]) - float(mb[1]) if len(mb) >= 4 else 792.0
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        x = x0
        y = page_h - y1
        font_size = max(8.0, (y1 - y0))
    else:
        x, y, font_size = 0.0, 0.0, 1.0

    # Escape special characters
    safe_text = (text or '').replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    # Create invisible text content stream
    content = f"""q
/Span << /MCID {int(mcid)} >> BDC
/GS1 gs
BT
/F1 {font_size:.2f} Tf
{x:.2f} {y:.2f} Td
({safe_text}) Tj
ET
EMC
Q
""".encode('utf-8')

    stream = pdf.make_indirect(pdf.make_stream(content))
    cur = page_obj.get(Name.Contents, None)
    if cur is None:
        page_obj[Name.Contents] = Array([stream])
    elif isinstance(cur, Array):
        fixed = Array()
        for s in cur:
            fixed.append(pdf.make_indirect(s) if not getattr(s, 'indirect', False) else s)
        fixed.append(stream)
        page_obj[Name.Contents] = fixed
    else:
        page_obj[Name.Contents] = Array([pdf.make_indirect(cur), stream])

def attach_text_to_elem(pdf, se, page_obj, text, bbox, *, parenttree, page_key, mcid):
    """Attach text to structure element"""
    attach_k_ref(se, page_obj, mcid)
    page_add_invisible_mcid_stream(pdf, page_obj, mcid, text, bbox=bbox)
    parenttree.add(page_key, mcid, se)
    return mcid + 1

# ---------------- JSON I/O ----------------
def load_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

def load_tags_for_page(tag_dir, pnum):
    promoted = os.path.join(tag_dir, f'page_{pnum:03d}.promoted.json')
    standard = os.path.join(tag_dir, f'page_{pnum:03d}.tags.json')
    path = promoted if os.path.exists(promoted) else standard
    data = load_json(path, {}) or {}
    return data.get('tags') or []

def normalize_heading_hierarchy(tags):
    has_h1 = any(t.get('tag') == 'H1' for t in tags)
    if not has_h1:
        for t in tags:
            if t.get('tag') == 'H2':
                t['tag'] = 'H1'
                break
    current = 0
    for t in tags:
        tg = t.get('tag','')
        if len(tg)==2 and tg[0]=='H' and tg[1].isdigit():
            lvl = int(tg[1])
            if lvl > current + 1:
                lvl = current + 1
                t['tag'] = f'H{lvl}'
            current = max(current, lvl)
    return tags

# ---------------- main ----------------
def main(pdf_in, fused_dir, struct_dir, pdf_out, title=None, lang="en-US", author=None):
    pdf = pikepdf.open(pdf_in)
    set_doc_metadata(pdf, input_path=pdf_in, title=title, lang=lang, author=author)
    ensure_markinfo(pdf)

    st_root = ensure_struct_tree_root(pdf)
    doc_el = make_selement(pdf, 'Document', parent=st_root, title_text=(title or None))
    st_root[Name.K] = Array([doc_el])
    part_el = make_selement(pdf, 'Part', parent=doc_el)
    doc_el[Name.K] = Array([part_el])

    parenttree = ParentTreeBuilder()
    next_key = 0

    for i, page in enumerate(pdf.pages):
        pnum = i + 1
        page_obj = page.obj

        page_key = next_key
        next_key += 1
        page_obj[Name.StructParents] = page_key
        parenttree.ensure_page_key(page_key)

        raw = load_tags_for_page(fused_dir, pnum)
        if not raw:
            continue

        tags = []
        for t in raw:
            if not isinstance(t, dict):
                continue
            t.setdefault('tag','P')
            t.setdefault('text','')
            t.setdefault('bbox',[0,0,0,0])
            tags.append(t)

        tags = normalize_heading_hierarchy(tags)
        tags = [t for t in tags if TAG_TO_S.get(t.get('tag', 'P'))]

        page_part = make_selement(pdf, 'Part', parent=part_el, title_text=f'Page {pnum}')
        kids = part_el.get(Name.K, Array())
        kids.append(page_part)
        part_el[Name.K] = kids

        next_mcid = 0
        for t in tags:
            role = TAG_TO_S.get(t.get('tag','P'), 'P')
            txt = (t.get('text') or '')
            bbox = t.get('bbox')
            # Create element with ActualText (JAWS reads this)
            se = make_selement(pdf, role, parent=page_part, actual_text=txt)
            arr = page_part.get(Name.K, Array())
            arr.append(se)
            page_part[Name.K] = arr

            next_mcid = attach_text_to_elem(
                pdf, se, page_obj, txt, bbox,
                parenttree=parenttree, page_key=page_key, mcid=next_mcid
            )

    pt_dict = parenttree.build_numbertree_dict()
    st_root[Name.ParentTree] = pdf.make_indirect(pt_dict)
    st_root[Name.ParentTreeNextKey] = next_key
    ensure_markinfo(pdf)

    create_page_bookmarks(pdf, fused_dir)

    pdf.save(pdf_out, linearize=True, compress_streams=True)
    print("Wrote tagged PDF:", os.path.abspath(pdf_out))

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python TagPDFFinal.py input.pdf promoted_tags_dir structures_dir output.pdf")
        sys.exit(1)

    in_pdf = sys.argv[1]
    tagsdir = sys.argv[2]
    struct = sys.argv[3]
    out_pdf = sys.argv[4]

    custom_title = os.environ.get("PDF_TITLE", None)
    main(in_pdf, tagsdir, struct, out_pdf, title=custom_title)

# import os, sys, json
# import pikepdf
# from pikepdf import Name, Dictionary, Array, String
#
# # ---------------- ParentTree builder ----------------
# class ParentTreeBuilder:
#     def __init__(self):
#         self.by_key = {}
#
#     def ensure_page_key(self, key):
#         self.by_key.setdefault(key, [])
#
#     def add(self, key, mcid, struct_elem):
#         arr = self.by_key.setdefault(key, [])
#         if len(arr) <= mcid:
#             arr.extend([None] * (mcid + 1 - len(arr)))
#         arr[mcid] = struct_elem
#
#     def build_numbertree_dict(self):
#         nums = Array()
#         for key in sorted(self.by_key.keys()):
#             arr = Array()
#             for se in self.by_key[key]:
#                 arr.append(se if se is not None else None)  # None => null
#             nums.append(key)
#             nums.append(arr)
#         nt = Dictionary()
#         nt[Name.Nums] = nums
#         return nt
#
# # ---------------- helpers ----------------
# def pdf_name(s):
#     return Name(s if isinstance(s, str) and s.startswith('/') else '/' + s)
#
# def as_str(s):
#     return String(s) if not isinstance(s, String) else s
#
# def ensure_markinfo(pdf):
#     root = pdf.Root
#     mi = root.get(Name.MarkInfo, None)
#     if mi is None:
#         mi = Dictionary()
#         root[Name.MarkInfo] = pdf.make_indirect(mi)
#     mi[Name.Marked] = True
#     return mi
#
# # Role map (standard)
# ROLEMAP = {
#     'Document':'Document','Part':'Part',
#     'H1':'H1','H2':'H2','H3':'H3','P':'P',
#     'L':'L','LI':'LI','Lbl':'Lbl','LBody':'LBody',
#     'Table':'Table','TR':'TR','TH':'TH','TD':'TD',
#     'Figure':'Figure','Caption':'Caption','Span':'Span'
# }
#
# # Tag mapping we expect from JSON (extended for hybrid tags)
# TAG_TO_S = {
#     'H1': 'H1',
#     'H2': 'H2',
#     'H3': 'H3',
#     'P':  'P',
#     'LI': 'LI',
#
#     'Table': 'Table',
#     'TR':    'TR',
#     'TH':    'TH',
#     'TD':    'TD',
#
#     'Figure':  'Figure',
#     'Caption': 'Caption',
#
#     # --- hybrid tags from FuseLinesSmart.py ---
#     # Turn figure caption candidates into real /Caption elements
#     'FigureCaptionCandidate': 'Caption',
#     # Table row candidates: for now tag as paragraph text
#     # (later you can wire them into real <Table>/<TR>/<TD> using tables.json)
#     'TableRowCandidate': 'P',
# }
#
# def ensure_struct_tree_root(pdf):
#     root = pdf.Root
#     if Name.StructTreeRoot in root:
#         return root[Name.StructTreeRoot]
#     st = Dictionary()
#     st[Name.Type] = Name.StructTreeRoot
#     rm = Dictionary()
#     for k, v in ROLEMAP.items():
#         rm[pdf_name(k)] = pdf_name(v)
#     st[Name.RoleMap] = rm
#     root[Name.StructTreeRoot] = pdf.make_indirect(st)
#     return root[Name.StructTreeRoot]
#
# def set_doc_metadata(pdf, input_path, title=None, lang="en-US", author=None):
#     # default title from input filename when not provided
#     if not title:
#         title = os.path.splitext(os.path.basename(input_path))[0]
#
#     if not hasattr(pdf, "docinfo") or pdf.docinfo is None:
#         pdf.docinfo = Dictionary()
#     pdf.docinfo[Name.Title]    = as_str(title)
#     pdf.docinfo[Name.Creator]  = as_str("PDF Accessibility Tagger")
#     pdf.docinfo[Name.Producer] = as_str("pikepdf")
#     if author:
#         pdf.docinfo[Name.Author] = as_str(author)
#
#     pdf.Root[Name.Lang] = as_str(lang)
#
#     vp = pdf.Root.get(Name.ViewerPreferences, None)
#     if vp is None:
#         vp = Dictionary()
#         pdf.Root[Name.ViewerPreferences] = pdf.make_indirect(vp)
#     vp[Name.DisplayDocTitle] = True
#
#     # (optional) XMP — safe to keep if you already had it
#     xmp = f'''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
# <x:xmpmeta xmlns:x="adobe:ns:meta/">
#  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
#   <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
#    <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{title}</rdf:li></rdf:Alt></dc:title>
#   </rdf:Description>
#  </rdf:RDF>
# </x:xmpmeta>
# <?xpacket end="w"?>'''
#     meta = pdf.make_stream(xmp.encode("utf-8"))
#     meta[Name.Type] = Name.Metadata
#     meta[Name.Subtype] = Name.XML
#     pdf.Root[Name.Metadata] = pdf.make_indirect(meta)
#
#
# def create_page_bookmarks(pdf, fused_dir):
#     """
#     Fix Acrobat 'Bookmarks - Failed' by creating /Outlines with real page destinations.
#     Uses first H1/H2 from each page's tags if available; else 'Page N'.
#     """
#     outlines = Dictionary()
#     outlines[Name.Type] = Name.Outlines
#     items = []
#
#     for i, page in enumerate(pdf.pages, start=1):
#         title_text = None
#         for stem in ("promoted", "tags"):
#             p = os.path.join(fused_dir, f'page_{i:03d}.{stem}.json')
#             if os.path.exists(p):
#                 try:
#                     data = json.load(open(p, "r", encoding="utf-8"))
#                     for t in data.get("tags", []):
#                         if t.get("tag") in ("H1","H2") and (t.get("text") or "").strip():
#                             title_text = t["text"].strip()
#                             break
#                 except Exception:
#                     pass
#             if title_text:
#                 break
#         title = title_text or f"Page {i}"
#
#         bm = Dictionary()
#         bm[Name.Title]  = as_str(title)
#         bm[Name.Parent] = outlines
#         bm[Name.Dest]   = Array([page.obj, Name.Fit])
#         items.append(pdf.make_indirect(bm))
#
#     if items:
#         outlines[Name.First] = items[0]
#         outlines[Name.Last]  = items[-1]
#         outlines[Name.Count] = len(items)
#         for idx, it in enumerate(items):
#             if idx > 0:
#                 it[Name.Prev] = items[idx-1]
#             if idx < len(items)-1:
#                 it[Name.Next] = items[idx+1]
#     else:
#         outlines[Name.Count] = 0
#     pdf.Root[Name.Outlines] = pdf.make_indirect(outlines)
#
# def make_selement(pdf, S, *, parent=None, title_text=None, actual_text=None):
#     el = Dictionary()
#     el[Name.Type] = Name.StructElem
#     el[Name.S] = pdf_name(S)
#     if parent is not None:
#         el[Name.P] = parent
#     if title_text:
#         el[Name.T] = as_str(title_text)
#     #Removed ActualText - causes text to show in tag tree
#     if actual_text:
#         el[Name.ActualText] = as_str(actual_text)
#         if Name.T not in el:
#             el[Name.T] = as_str(actual_text)
#     return pdf.make_indirect(el)
#
# def attach_k_ref(el, page_obj, mcid):
#     kid = Dictionary()
#     kid[Name.Type] = Name.MCR
#     kid[Name.Pg]   = page_obj
#     kid[Name.MCID] = mcid
#     if Name.K not in el:
#         el[Name.K] = Array([kid])
#     else:
#         arr = el[Name.K] if isinstance(el[Name.K], Array) else Array([el[Name.K]])
#         arr.append(kid)
#         el[Name.K] = arr
#
# def _utf16be_hex(s: str) -> str:
#     # hex string with UTF-16BE BOM (FE FF)
#     return '<FEFF' + (s or '').encode('utf-16-be').hex().upper() + '>'
#
#
# def page_add_invisible_mcid_stream(pdf, page_obj, mcid, text, *, bbox=None):
#     """
#     Add a /Span marked-content with fully transparent text placed at bbox.
#     Uses UTF-16BE hex string to avoid encoding issues (Acrobat error 18).
#     """
#     # Resources
#     resources = page_obj.get(Name.Resources, None)
#     if resources is None:
#         resources = Dictionary()
#         page_obj[Name.Resources] = resources
#
#     fonts = resources.get(Name.Font, None)
#     if fonts is None:
#         fonts = Dictionary()
#         resources[Name.Font] = fonts
#     if Name.F1 not in fonts:
#         fnt = Dictionary()
#         fnt[Name.Type] = Name.Font
#         fnt[Name.Subtype] = Name.Type1
#         fnt[Name.BaseFont] = Name.Helvetica
#         fonts[Name.F1] = pdf.make_indirect(fnt)
#
#     # ExtGState for full transparency
#     extg = resources.get(Name.ExtGState, None)
#     if extg is None:
#         extg = Dictionary()
#         resources[Name.ExtGState] = extg
#     if Name.GS1 not in extg:
#         gs = Dictionary()
#         gs[Name.Type] = Name.ExtGState
#         gs[Name.ca] = 1.0  # fill alpha = VISIBLE
#         gs[Name.CA] = 1.0  # stroke alpha = VISIBLE
#         extg[Name.GS1] = pdf.make_indirect(gs)
#
#     # Compute placement
#     mb = page_obj.get(Name.MediaBox, [0,0,612,792])
#     page_h = float(mb[3]) - float(mb[1]) if len(mb) >= 4 else 792.0
#     if bbox and len(bbox) == 4:
#         x0, y0, x1, y1 = [float(v) for v in bbox]
#         x = x0
#         y = page_h - y1
#         font_size = max(8.0, (y1 - y0))
#     else:
#         x, y, font_size = 0.0, 0.0, 1.0
#
#    # hex_text = _utf16be_hex(text or '')
#
#     # Escape special PDF characters
#     safe_text = (text or '').replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
#
#     content = f"""q
#     /Span << /MCID {int(mcid)} >> BDC
#     BT
#     /F1 {font_size:.2f} Tf
#     {x:.2f} {y:.2f} Td
#     ({safe_text}) Tj
#     ET
#     EMC
#     Q
#     """.encode('utf-8')
#
#     stream = pdf.make_indirect(pdf.make_stream(content))
#     cur = page_obj.get(Name.Contents, None)
#     if cur is None:
#         page_obj[Name.Contents] = Array([stream])
#     elif isinstance(cur, Array):
#         # ensure array of indirect streams
#         fixed = Array()
#         for s in cur:
#             fixed.append(pdf.make_indirect(s) if not getattr(s, 'indirect', False) else s)
#         fixed.append(stream)
#         page_obj[Name.Contents] = fixed
#     else:
#         page_obj[Name.Contents] = Array([pdf.make_indirect(cur), stream])
#
# def attach_text_to_elem(pdf, se, page_obj, text, bbox, *, parenttree, page_key, mcid):
#     attach_k_ref(se, page_obj, mcid)
#     page_add_invisible_mcid_stream(pdf, page_obj, mcid, text, bbox=bbox)
#     parenttree.add(page_key, mcid, se)
#     return mcid + 1
#
# # ---------------- JSON I/O ----------------
# def load_json(path, default=None):
#     try:
#         with open(path, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     except Exception:
#         return default
#
# def load_tags_for_page(tag_dir, pnum):
#     # Prefer promoted.json; otherwise tags.json
#     promoted = os.path.join(tag_dir, f'page_{pnum:03d}.promoted.json')
#     standard = os.path.join(tag_dir, f'page_{pnum:03d}.tags.json')
#     path = promoted if os.path.exists(promoted) else standard
#     data = load_json(path, {}) or {}
#     return data.get('tags') or []
#
# def normalize_heading_hierarchy(tags):
#     has_h1 = any(t.get('tag') == 'H1' for t in tags)
#     if not has_h1:
#         for t in tags:
#             if t.get('tag') == 'H2':
#                 t['tag'] = 'H1'
#                 break
#     current = 0
#     for t in tags:
#         tg = t.get('tag','')
#         if len(tg)==2 and tg[0]=='H' and tg[1].isdigit():
#             lvl = int(tg[1])
#             if lvl > current + 1:
#                 lvl = current + 1
#                 t['tag'] = f'H{lvl}'
#             current = max(current, lvl)
#     return tags
#
# # ---------------- main ----------------
# def main(pdf_in, fused_dir, struct_dir, pdf_out, title=None, lang="en-US", author=None):
#     pdf = pikepdf.open(pdf_in)
#     set_doc_metadata(pdf, input_path=pdf_in, title=title, lang=lang, author=author)
#     ensure_markinfo(pdf)
#
#     st_root = ensure_struct_tree_root(pdf)
#     # Document -> Part
#     doc_el  = make_selement(pdf, 'Document', parent=st_root, title_text=(title or None))
#     st_root[Name.K] = Array([doc_el])
#     part_el = make_selement(pdf, 'Part', parent=doc_el)
#     doc_el[Name.K] = Array([part_el])
#
#     parenttree = ParentTreeBuilder()
#     next_key = 0
#
#     for i, page in enumerate(pdf.pages):
#         pnum = i + 1
#         page_obj = page.obj
#
#         page_key = next_key
#         next_key += 1
#         page_obj[Name.StructParents] = page_key
#         parenttree.ensure_page_key(page_key)
#
#         # read tags
#         raw = load_tags_for_page(fused_dir, pnum)
#         if not raw:
#             continue
#
#         # normalize and sanitize
#         tags = []
#         for t in raw:
#             if not isinstance(t, dict):
#                 continue
#             t.setdefault('tag','P')
#             t.setdefault('text','')
#             t.setdefault('bbox',[0,0,0,0])
#             tags.append(t)
#
#         tags = normalize_heading_hierarchy(tags)
#
#         # only keep tags we know how to map to a structure role
#         tags = [t for t in tags if TAG_TO_S.get(t.get('tag', 'P'))]
#
#         # page part
#         page_part = make_selement(pdf, 'Part', parent=part_el, title_text=f'Page {pnum}')
#         kids = part_el.get(Name.K, Array())
#         kids.append(page_part)
#         part_el[Name.K] = kids
#
#         # tag everything as given (H1/H2/P/… + FigureCaptionCandidate, etc.)
#         next_mcid = 0
#         for t in tags:
#             role = TAG_TO_S.get(t.get('tag','P'), 'P')
#             txt  = (t.get('text') or '')
#             bbox = t.get('bbox')
#             se = make_selement(pdf, role, parent=page_part, actual_text=txt)
#             arr = page_part.get(Name.K, Array())
#             arr.append(se)
#             page_part[Name.K] = arr
#
#             next_mcid = attach_text_to_elem(
#                 pdf, se, page_obj, txt, bbox,
#                 parenttree=parenttree, page_key=page_key, mcid=next_mcid
#             )
#
#     # ParentTree + final touches
#     pt_dict = parenttree.build_numbertree_dict()
#     st_root[Name.ParentTree] = pdf.make_indirect(pt_dict)
#     st_root[Name.ParentTreeNextKey] = next_key
#     ensure_markinfo(pdf)
#
#     # Real bookmarks from tags/H1/H2 (or Page N)
#     create_page_bookmarks(pdf, fused_dir)
#
#     pdf.save(pdf_out, linearize=True, compress_streams=True)
#     print("Wrote tagged PDF:", os.path.abspath(pdf_out))
#
# if __name__ == "__main__":
#     if len(sys.argv) < 5:
#         print("Usage: python TagPikePDFFinal.py input.pdf promoted_tags_dir structures_dir output.pdf")
#         sys.exit(1)
#
#     in_pdf  = sys.argv[1]
#     tagsdir = sys.argv[2]
#     struct  = sys.argv[3]  # currently unused but kept for compatibility
#     out_pdf = sys.argv[4]
#
#     # Title defaults to input filename; you can pass a custom one via env if you want:
#     custom_title = os.environ.get("PDF_TITLE", None)
#
#     main(in_pdf, tagsdir, struct, out_pdf, title=custom_title)