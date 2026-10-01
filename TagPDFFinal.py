import os, sys, json
import unicodedata
import pikepdf
from pikepdf import Name, Dictionary, Array, String

# Helvetica glyph widths (units per 1000 em).  Used to compute natural text width
# so we can apply Tz (horizontal scale) and make the invisible MCID text fill the
# exact OCR bbox width → pink highlight box matches the visual line.
_HV_W = {
    ' ':278,'A':667,'B':667,'C':667,'D':722,'E':611,'F':556,'G':722,'H':722,
    'I':278,'J':500,'K':667,'L':611,'M':833,'N':722,'O':778,'P':667,'Q':778,
    'R':722,'S':667,'T':611,'U':722,'V':667,'W':944,'X':667,'Y':667,'Z':611,
    'a':556,'b':556,'c':500,'d':556,'e':556,'f':278,'g':556,'h':556,'i':222,
    'j':222,'k':500,'l':222,'m':833,'n':556,'o':556,'p':556,'q':556,'r':333,
    's':500,'t':278,'u':556,'v':500,'w':722,'x':500,'y':500,'z':500,
    '0':556,'1':556,'2':556,'3':556,'4':556,'5':556,'6':556,'7':556,'8':556,
    '9':556,'.':278,',':278,':':278,';':278,'!':333,'?':556,'-':333,
    '(':333,')':333,'[':278,']':278,'/':278,'\\':278,
}

def _hv_width(text, font_size):
    """Approximate width of `text` in Helvetica at `font_size` pt."""
    return sum(_HV_W.get(c, 556) for c in (text or '')) * font_size / 1000.0

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

# ---------------- FIX #3: WinAnsi-safe text helpers ----------------
def to_winansi_safe(text: str) -> str:
    """
    Force text into WinAnsi-compatible bytes (cp1252) to avoid .notdef glyph usage.
    Also strips Unicode values that are forbidden by veraPDF checks.
    """
    if not text:
        return ""

    # Normalize common punctuation that can cause encoding trouble
    text = (
        text.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
            .replace("–", "-").replace("—", "-")
            .replace("\u00a0", " ")
    )

    # Normalize to reduce weird combining sequences
    text = unicodedata.normalize("NFKD", text)

    # Encode to cp1252 and replace anything not representable
    text = text.encode("cp1252", "replace").decode("cp1252")

    # Remove forbidden codepoints explicitly (veraPDF flags these)
    text = text.replace("\x00", "").replace("\ufeff", "").replace("\ufffe", "")
    return text

def escape_pdf_literal_string(text: str) -> str:
    """
    Escape PDF literal string specials: backslash and parentheses.
    """
    return (
        text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
    )

# ---------------- FIX #1: Wrap existing page content as Artifact ----------------
def wrap_existing_page_content_as_artifact(pdf, page_obj):
    """
    veraPDF UA requires that ALL page content is either tagged or marked as Artifact.
    Your overlays are tagged, but the *original* PDF content isn't, so we wrap it:
        /Artifact BMC ... EMC
    """
    contents = page_obj.get(Name.Contents, None)
    if contents is None:
        return

    def wrap_stream(s):
        # Make sure it's indirect so we can read and replace safely
        s_ind = pdf.make_indirect(s) if not getattr(s, "indirect", False) else s
        data = bytes(s_ind.read_bytes())
        wrapped = b"/Artifact BMC\n" + data + b"\nEMC\n"
        return pdf.make_indirect(pdf.make_stream(wrapped))

    if isinstance(contents, Array):
        new_arr = Array()
        for s in contents:
            new_arr.append(wrap_stream(s))
        page_obj[Name.Contents] = new_arr
    else:
        page_obj[Name.Contents] = wrap_stream(contents)

# Role map
# ---------------- FIX #2: Remove self-mapping RoleMap to avoid circular mapping ----------------
# For standard tags (Document, P, H1, etc.) you don't need RoleMap at all.
ROLEMAP = {}

# Tag mapping with Artifact handling
TAG_TO_S = {
    'H1': 'H1', 'H2': 'H2', 'H3': 'H3', 'P': 'P', 'LI': 'LI',
    'Table': 'Table', 'TR': 'TR', 'TH': 'TH', 'TD': 'TD',
    'Figure': 'Figure', 'Caption': 'Caption',
    'FigureCaptionCandidate': 'Caption',
    'FigureArea': 'FigureArea',   # image bbox placeholder → MCID on <Figure>
    'TableRowCandidate': 'TR',
    'Artifact': None,  # ← Artifacts don't go in structure tree
}

def ensure_struct_tree_root(pdf):
    root = pdf.Root
    if Name.StructTreeRoot in root:
        return root[Name.StructTreeRoot]
    st = Dictionary()
    st[Name.Type] = Name.StructTreeRoot

    # Only write RoleMap if non-empty (we keep it empty to avoid circular mappings)
    if ROLEMAP:
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

    # Add PDF/UA identifier in XMP metadata
    xmp = f'''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{title}</rdf:li></rdf:Alt></dc:title>
  </rdf:Description>
  <rdf:Description xmlns:pdfuaid="http://www.aiim.org/pdfua/ns/id/" pdfuaid:part="1"/>
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
                    with open(p, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    for t in data.get("tags", []):
                        if t.get("tag") in ("H1", "H2") and (t.get("text") or "").strip():
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
                it[Name.Prev] = items[idx - 1]
            if idx < len(items) - 1:
                it[Name.Next] = items[idx + 1]
    else:
        outlines[Name.Count] = 0
    pdf.Root[Name.Outlines] = pdf.make_indirect(outlines)

def make_selement(pdf, S, *, parent=None, title_text=None, actual_text=None):
    """Create structure element"""
    el = Dictionary()
    el[Name.Type] = Name.StructElem
    el[Name.S] = pdf_name(S)
    if parent is not None:
        el[Name.P] = parent
    if title_text:
        el[Name.T] = as_str(title_text)
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
    """Add invisible text overlay"""
    resources = page_obj.get(Name.Resources, None)
    if resources is None:
        resources = Dictionary()
        page_obj[Name.Resources] = resources

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

    mb = page_obj.get(Name.MediaBox, [0, 0, 612, 792])
    page_h = float(mb[3]) - float(mb[1]) if len(mb) >= 4 else 792.0
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        x = x0
        bbox_height = y1 - y0
        bbox_width = x1 - x0
        y = page_h - y1  # baseline = bottom of visual bbox

        if not (text or '').strip():
            # FigureArea has no text — fill the bbox with spaces so Acrobat can
            # compute a highlight box when clicking <Figure>.
            # font_size scaled so Helvetica ascent (0.718 em) spans full bbox height.
            font_size = max(4.0, bbox_height / 0.718)
            # Use enough spaces so h_scale stays under 500 (PDF Tz limit).
            # Each space in Helvetica is 278/1000 em wide.
            space_w = 278 * font_size / 1000.0
            n_spaces = max(1, int((bbox_width * 500.0) // (space_w * 100.0 * 500.0 / 100.0)) + 1)
            # Simpler: n_spaces such that n_spaces * space_w * (h_scale/100) ≈ bbox_width
            # with h_scale ≤ 500, so n_spaces ≥ bbox_width / (space_w * 5.0)
            n_spaces = max(1, int(bbox_width / (space_w * 5.0)) + 1)
            text = ' ' * n_spaces
        else:
            # Use full bbox height as font size so the invisible glyph fills the visual line.
            font_size = max(4.0, bbox_height)

        # Horizontal scale: stretch/compress invisible text so its glyph bboxes
        # exactly cover the OCR bbox width → pink highlight matches visual line.
        natural_w = _hv_width(text, font_size)
        if natural_w > 0 and bbox_width > 0:
            h_scale = max(10.0, min(500.0, (bbox_width / natural_w) * 100.0))
        else:
            h_scale = 100.0

        # DEBUG: log first 200 MCIDs so we can check bbox vs font_size alignment
        try:
            if not hasattr(page_add_invisible_mcid_stream, '_dbg_count'):
                page_add_invisible_mcid_stream._dbg_count = 0
            if page_add_invisible_mcid_stream._dbg_count < 200:
                with open('/tmp/pinkbox_debug.txt', 'a') as _dbg:
                    _dbg.write(
                        f"mcid={mcid:4d}  bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})  "
                        f"bbox_h={bbox_height:.1f}  bbox_w={bbox_width:.1f}  "
                        f"font_size={font_size:.1f}  natural_w={natural_w:.1f}  "
                        f"h_scale={h_scale:.1f}  text={repr((text or '')[:50])}\n"
                    )
                page_add_invisible_mcid_stream._dbg_count += 1
        except Exception:
            pass
    else:
        x, y, font_size, h_scale = 0.0, 0.0, 1.0, 100.0

    # ---------------- FIX #3: sanitize text to WinAnsi + escape, and do not write UTF-8 strings ----------------
    safe_text = escape_pdf_literal_string(to_winansi_safe(text or ""))

    # Use text rendering mode 3 = invisible (still tagged & extractable).
    # Tz sets horizontal scale so the glyph bboxes fill the OCR bbox width.
    # IMPORTANT: encode stream as latin-1 so the PDF literal string stays single-byte.
    content = f"""q
/Span << /MCID {int(mcid)} >> BDC
BT
/F1 {font_size:.2f} Tf
3 Tr
{h_scale:.2f} Tz
{x:.2f} {y:.2f} Td
({safe_text}) Tj
ET
EMC
Q
"""
    content_bytes = content.encode("latin-1", errors="replace")

    stream = pdf.make_indirect(pdf.make_stream(content_bytes))
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
        tg = t.get('tag', '')
        if len(tg) == 2 and tg[0] == 'H' and tg[1].isdigit():
            lvl = int(tg[1])
            if lvl > current + 1:
                lvl = current + 1
                t['tag'] = f'H{lvl}'
            current = max(current, lvl)
    return tags

# ---------------- main ----------------
def main(pdf_in, fused_dir, struct_dir, pdf_out, title=None, lang="en-US", author=None):
    # Reset pink-box debug counter and log file for this run
    page_add_invisible_mcid_stream._dbg_count = 0
    try:
        open('/tmp/pinkbox_debug.txt', 'w').close()
        open('/tmp/tag_figure_debug.txt', 'w').close()
    except Exception:
        pass
    pdf = pikepdf.open(pdf_in)
    set_doc_metadata(pdf, input_path=pdf_in, title=title, lang=lang, author=author)
    ensure_markinfo(pdf)

    st_root = ensure_struct_tree_root(pdf)
    doc_el = make_selement(pdf, 'Document', parent=st_root, title_text=(title or None))
    st_root[Name.K] = Array([doc_el])
    part_el = make_selement(pdf, 'Part', parent=doc_el)

    # Initialize K array ONCE
    doc_el[Name.K] = Array([part_el])
    part_el[Name.K] = Array()  # Initialize empty, will append to it

    parenttree = ParentTreeBuilder()
    next_key = 0

    for i, page in enumerate(pdf.pages):
        pnum = i + 1
        page_obj = page.obj

        # ---------------- FIX #1: wrap ORIGINAL page content as Artifact before adding tagged overlays ----------------
        wrap_existing_page_content_as_artifact(pdf, page_obj)

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
            t.setdefault('tag', 'P')
            t.setdefault('text', '')
            t.setdefault('bbox', [0, 0, 0, 0])
            tags.append(t)

        tags = normalize_heading_hierarchy(tags)

        # Filter out artifacts AND unmapped tags
        tags = [t for t in tags if TAG_TO_S.get(t.get('tag', 'P')) is not None]

        # Create page part
        page_part = make_selement(pdf, 'Part', parent=part_el, title_text=f'Page {pnum}')

        # Append directly to existing array
        part_el[Name.K].append(page_part)

        # Initialize page_part K array ONCE
        page_part[Name.K] = Array()

        # Track current list/table/figure containers for wrapping
        next_mcid = 0
        current_list = None    # current <L> container
        current_table = None   # current <Table> container
        current_figure = None  # current <Figure> container (for consecutive Captions)

        for t in tags:
            tag = t.get('tag', 'P')
            role = TAG_TO_S.get(tag, 'P')

            # Skip artifacts (should be filtered above, but double-check)
            if role is None or tag == 'Artifact':
                continue

            txt = (t.get('text') or '')
            bbox = t.get('bbox')

            if role == 'LI':
                current_table = None
                current_figure = None
                if current_list is None:
                    current_list = make_selement(pdf, 'L', parent=page_part)
                    page_part[Name.K].append(current_list)
                    current_list[Name.K] = Array()
                se = make_selement(pdf, 'LI', parent=current_list, actual_text=txt)
                current_list[Name.K].append(se)

            elif role == 'TR':
                current_list = None
                current_figure = None
                if current_table is None:
                    current_table = make_selement(pdf, 'Table', parent=page_part)
                    page_part[Name.K].append(current_table)
                    current_table[Name.K] = Array()
                tr_elem = make_selement(pdf, 'TR', parent=current_table)
                current_table[Name.K].append(tr_elem)
                tr_elem[Name.K] = Array()
                # One TD per row (simplified; cell splitting can be added later)
                se = make_selement(pdf, 'TD', parent=tr_elem, actual_text=txt)
                tr_elem[Name.K].append(se)

            elif role == 'FigureArea':
                current_list = None
                current_table = None
                # Ensure a <Figure> container exists, then attach an MCID covering
                # the image bbox directly to it — clicking <Figure> in Acrobat will
                # highlight the image area on the page.
                with open('/tmp/tag_figure_debug.txt', 'a') as _tdbg:
                    _tdbg.write(f"[FigureArea] P{pnum} current_figure={'SET' if current_figure is not None else 'NONE'}  "
                                f"bbox={bbox}\n")
                if current_figure is None:
                    current_figure = make_selement(pdf, 'Figure', parent=page_part)
                    current_figure[Name.Alt] = as_str('Figure')
                    page_part[Name.K].append(current_figure)
                    current_figure[Name.K] = Array()
                    with open('/tmp/tag_figure_debug.txt', 'a') as _tdbg:
                        _tdbg.write(f"  -> created new Figure element\n")
                else:
                    with open('/tmp/tag_figure_debug.txt', 'a') as _tdbg:
                        _tdbg.write(f"  -> reusing existing Figure element\n")
                # Attach MCID directly to Figure (not a child element) so the image
                # area becomes the Figure's own content reference.
                se = current_figure
                sub_lines = t.get('lines') or [{'text': '', 'bbox': bbox}]
                for ln in sub_lines:
                    next_mcid = attach_text_to_elem(
                        pdf, se, page_obj, ln['text'], ln['bbox'],
                        parenttree=parenttree, page_key=page_key, mcid=next_mcid
                    )
                continue  # skip the generic attach block below

            elif role == 'Caption':
                current_list = None
                current_table = None
                # Consecutive Caption elements belong to the same figure.
                # Create a <Figure> wrapper on the first Caption and reuse it for
                # subsequent ones (e.g. a source-attribution line that follows).
                # /Alt on the Figure gives screen readers the image description.
                with open('/tmp/tag_figure_debug.txt', 'a') as _tdbg:
                    _tdbg.write(f"[Caption] P{pnum} current_figure={'SET' if current_figure is not None else 'NONE'}  "
                                f"text='{txt[:60]}'\n")
                if current_figure is None:
                    current_figure = make_selement(pdf, 'Figure', parent=page_part)
                    current_figure[Name.Alt] = as_str(txt)
                    page_part[Name.K].append(current_figure)
                    current_figure[Name.K] = Array()
                    with open('/tmp/tag_figure_debug.txt', 'a') as _tdbg:
                        _tdbg.write(f"  -> created new Figure element (from Caption)\n")
                se = make_selement(pdf, 'Caption', parent=current_figure, actual_text=txt)
                current_figure[Name.K].append(se)

            else:
                with open('/tmp/tag_figure_debug.txt', 'a') as _tdbg:
                    if current_figure is not None:
                        _tdbg.write(f"[RESET Figure] P{pnum} role={role} tag={tag}  "
                                    f"text='{txt[:60]}' -> current_figure=None\n")
                current_list = None
                current_table = None
                current_figure = None  # close figure context when non-Caption appears
                se = make_selement(pdf, role, parent=page_part, actual_text=txt)
                page_part[Name.K].append(se)

            # Attach to page content — one stream per line if lines are stored
            sub_lines = t.get('lines')
            if sub_lines:
                for ln in sub_lines:
                    next_mcid = attach_text_to_elem(
                        pdf, se, page_obj, ln['text'], ln['bbox'],
                        parenttree=parenttree, page_key=page_key, mcid=next_mcid
                    )
            else:
                next_mcid = attach_text_to_elem(
                    pdf, se, page_obj, txt, bbox,
                    parenttree=parenttree, page_key=page_key, mcid=next_mcid
                )

    # Finalize
    pt_dict = parenttree.build_numbertree_dict()
    st_root[Name.ParentTree] = pdf.make_indirect(pt_dict)
    st_root[Name.ParentTreeNextKey] = next_key
    ensure_markinfo(pdf)

    create_page_bookmarks(pdf, fused_dir)

    pdf.save(pdf_out, linearize=True, compress_streams=True)
    print("Wrote tagged PDF:", os.path.abspath(pdf_out))
    print("\nFixes applied:")
    print("  ✓ Original page content wrapped as Artifact (UA 7.1 #3)")
    print("  ✓ RoleMap removed to avoid circular mapping (UA 7.1 #6)")
    print("  ✓ Overlay text sanitized to WinAnsi + latin-1 to avoid .notdef (UA 7.21.7/7.21.8)")
    print("  ✓ List structure fixed (LI wrapped in L)")
    print("  ✓ Artifacts excluded from structure tree")
    print("  ✓ PDF/UA metadata identifier added")

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
