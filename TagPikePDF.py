#!/usr/bin/env python3
import os, sys, json
import pikepdf
from pikepdf import Name, Dictionary, Array, String

PDF_NULL = pikepdf.Object.new_null()
PDF_NULL = pikepdf.Object(None)

# ---------------- ParentTree builder ----------------
class ParentTreeBuilder:
    def __init__(self):
        # key (StructParents value per page) -> list where index=MCID and value=StructElem ref
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
                # if no StructElem for this MCID, insert a PDF null object
                arr.append(se if se is not None else PDF_NULL)
            nums.append(key)
            nums.append(arr)
        nt = Dictionary()
        nt[Name('/Nums')] = nums
        return nt


# ---------------- helpers ----------------
def pdf_name(s):
    # pikepdf.Name must start with '/'
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

# Minimal role map (standard roles)
ROLEMAP = {
    'Document':'Document',
    'Part':'Part',
    'H1':'H1',
    'H2':'H2',
    'P':'P',
    'L':'L',
    'LI':'LI',
    'Table':'Table',
    'TR':'TR',
    'TH':'TH',
    'TD':'TD',
    'Figure':'Figure',
    'Caption':'Caption',
    'Span':'Span'
}

# Map our tags to structure roles
TAG_TO_S = {
    'H1':'H1', 'H2':'H2', 'P':'P', 'LI':'LI',
    'Table':'Table', 'TR':'TR', 'TH':'TH', 'TD':'TD',
    'Figure':'Figure', 'Caption':'Caption'
}

def ensure_struct_tree_root(pdf):
    root = pdf.Root
    if '/StructTreeRoot' in root:
        return root['/StructTreeRoot']
    st = Dictionary()
    st[Name('/Type')] = pdf_name('StructTreeRoot')
    # RoleMap
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

def make_selement(pdf, S, title_text=None):
    el = Dictionary()
    el[Name('/Type')] = pdf_name('StructElem')
    el[Name('/S')] = pdf_name(S)
    if title_text:
        el[Name('/T')] = as_str(title_text)
    return pdf.make_indirect(el)   # <-- ensure indirect


def attach_k_ref(el, page_obj, mcid):
    # /K can be an array of dicts with /MCID and /Pg
    kid = Dictionary()
    kid[Name('/MCID')] = mcid
    kid[Name('/Pg')] = page_obj
    if '/K' not in el:
        el[Name('/K')] = Array([kid])
    else:
        arr = el['/K'] if isinstance(el['/K'], Array) else Array([el['/K']])
        arr.append(kid)
        el[Name('/K')] = arr

def page_add_invisible_mcid_stream(pdf, page_obj, mcid, text):
    """
    Append a content stream that adds a marked-content sequence with MCID
    and writes invisible text (Tr 3) so appearance is unchanged.
    """
    # Ensure /Resources dict exists
    # Ensure /Resources and /Font exist
    resources = page_obj.get('/Resources', None)
    if resources is None:
        resources = Dictionary()
        page_obj['/Resources'] = resources

    fonts = resources.get('/Font', None)
    if fonts is None:
        fonts = Dictionary()
        resources['/Font'] = fonts

    # Add a standard Type1 font (/Helvetica) as /F1 if missing
    if '/F1' not in fonts:
        fdict = Dictionary({
            '/Type': pdf_name('Font'),
            '/Subtype': pdf_name('Type1'),
            '/BaseFont': pdf_name('Helvetica'),
        })
        # Prefer an indirect object
        try:
            fonts['/F1'] = pdf.make_indirect(fdict)
        except AttributeError:
            # If older pikepdf without make_indirect, direct is usually fine:
            fonts['/F1'] = fdict

    # Escape backslashes & parens in literal string
    safe_text = (text or '').replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    # Build a small overlay stream (1pt font, invisible text mode Tr 3)
    content = f"""q
/Span << /MCID {int(mcid)} >> BDC
BT /F1 1 Tf 3 Tr 0 0 Td ({safe_text}) Tj ET
EMC
Q
""".encode('utf-8')

    # Append to /Contents (handle missing/single/array)
    new_stream = pdf.make_stream(content)
    contents = page_obj.get('/Contents', None)
    if contents is None:
        page_obj[Name('/Contents')] = make_indirect(pdf, new_stream)
    elif isinstance(contents, Array):
        contents.append(make_indirect(pdf, new_stream))
    else:
        page_obj[Name('/Contents')] = Array([contents, make_indirect(pdf, new_stream)])

def tag_block(pdf, st_root, parent_se, page_obj, items, parenttree=None, page_key=None):
    """
    Create child StructElems under parent_se, each referencing a /MCID on the page.
    Also record ParentTree entries so readers can navigate tags.
    """
    # Ensure the page has a StructParents key (should be set by caller)
    if '/StructParents' not in page_obj and page_key is not None:
        page_obj[Name('/StructParents')] = page_key

    mcid_counter = 0
    for it in items:
        tag = it.get('tag', 'P')
        text = it.get('text', '')
        role = TAG_TO_S.get(tag, 'P')

        se = make_indirect(pdf, make_selement(pdf, role))
        se[Name('/ActualText')] = as_str(text)

        # Attach StructElem under parent
        if '/K' not in parent_se:
            parent_se[Name('/K')] = Array([se])
        else:
            arr = parent_se['/K'] if isinstance(parent_se['/K'], Array) else Array([parent_se['/K']])
            arr.append(se)
            parent_se[Name('/K')] = arr

        # This StructElem's /K references page and MCID
        attach_k_ref(se, page_obj, mcid_counter)

        # Inject invisible MCID-marked text into page content
        page_add_invisible_mcid_stream(pdf, page_obj, mcid_counter, text)

        # Record ParentTree mapping
        if parenttree is not None and page_key is not None:
            parenttree.add(page_key, mcid_counter, se)

        mcid_counter += 1

def load_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

# ---------------- main ----------------
def main(pdf_in, fused_dir, struct_dir, pdf_out, title="Tagged Reconstruction", lang="en-US"):
    pdf = pikepdf.open(pdf_in)
    set_doc_metadata(pdf, title=title, lang=lang)
    ensure_markinfo(pdf)

    st_root = ensure_struct_tree_root(pdf)

    # Root: Document (indirect)
    doc_el = make_indirect(pdf, make_selement(pdf, 'Document', title_text=title))
    st_root[Name('/K')] = Array([doc_el])

    # Top-level Part (indirect)
    part_el = make_indirect(pdf, make_selement(pdf, 'Part'))
    doc_el[Name('/K')] = Array([part_el])

    # ParentTree
    parenttree = ParentTreeBuilder()
    next_key = 0

    # Process pages
    num_pages = len(pdf.pages)
    for i in range(num_pages):
        pnum = i + 1
        page = pdf.pages[i]
        page_obj = page.obj

        # give this page a unique StructParents key
        page_key = next_key
        next_key += 1
        page_obj[Name('/StructParents')] = page_key
        parenttree.ensure_page_key(page_key)

        # ----- Load & normalize tags -----
        tags_json = load_json(os.path.join(fused_dir, f'page_{pnum:03d}.tags.json'), {}) or {}
        tags_raw = tags_json.get('tags') or []
        if not tags_raw:
            continue

        tags = []
        for t in tags_raw:
            if not isinstance(t, dict):
                continue
            t.setdefault('tag', 'P')
            t.setdefault('text', '')
            t.setdefault('bbox', [0, 0, 0, 0])
            t.setdefault('region', None)
            tags.append(t)
        if not tags:
            continue

        # Make a page-level Part (indirect)
        page_part = make_indirect(pdf, make_selement(pdf, 'Part', title_text=f'Page {pnum}'))
        arr = part_el['/K'] if '/K' in part_el else Array()
        arr.append(page_part)
        part_el[Name('/K')] = arr

        # ----- (A) Lists -----
        lists_json = load_json(os.path.join(struct_dir, f'page_{pnum:03d}.lists.json'), {}) or {}
        lists = lists_json.get('lists') or []
        li_texts = set()

        for lst in lists:
            if not isinstance(lst, dict):
                continue
            items = lst.get('items') or []
            if not items:
                continue

            L_el = make_indirect(pdf, make_selement(pdf, 'L'))
            arr2 = page_part['/K'] if '/K' in page_part else Array()
            arr2.append(L_el)
            page_part[Name('/K')] = arr2

            for item in items:
                if not item:
                    continue
                LI_el = make_indirect(pdf, make_selement(pdf, 'LI'))
                P_el  = make_indirect(pdf, make_selement(pdf, 'P'))
                # LI -> P
                LI_el[Name('/K')] = Array([P_el])
                # L -> LI
                arr3 = L_el['/K'] if '/K' in L_el else Array()
                arr3.append(LI_el)
                L_el[Name('/K')] = arr3

                li_texts.add(item)
                # tag P with invisible MCID + ActualText
                tag_block(pdf, st_root, P_el, page_obj,
                          [{'tag':'P','text':item}],
                          parenttree=parenttree, page_key=page_key)

        # ----- (B) Figures + captions -----
        def is_figure(t):
            r = t.get('region')
            return isinstance(r, dict) and r.get('cls') == 'figure'

        figures = [t for t in tags if is_figure(t)]
        caps = [t for t in tags if t.get('tag') == 'FigureCaptionCandidate']

        for f in figures:
            fig_el = make_indirect(pdf, make_selement(pdf, 'Figure'))

            # Try to find a nearby caption below figure
            cap_text = None
            fb = f.get('bbox') or [0, 0, 0, 0]
            if isinstance(fb, (list, tuple)) and len(fb) == 4:
                fy_bottom = fb[3]
                candidates = []
                for c in caps:
                    cb = c.get('bbox') or [0, 0, 0, 0]
                    if isinstance(cb, (list, tuple)) and len(cb) == 4:
                        if abs((cb[1] or 0) - fy_bottom) < 40:
                            candidates.append(c)
                if candidates:
                    cap_text = (candidates[0].get('text') or '').strip() or None

            if cap_text:
                cap_el = make_indirect(pdf, make_selement(pdf, 'Caption'))
                fig_el[Name('/K')] = Array([cap_el])
                cap_el[Name('/ActualText')] = as_str(cap_text)
                fig_el[Name('/Alt')] = as_str(cap_text)  # optional alt
                # write MCID for caption text
                tag_block(pdf, st_root, cap_el, page_obj,
                          [{'tag':'P','text':cap_text}],
                          parenttree=parenttree, page_key=page_key)

            # attach fig under page_part
            arr4 = page_part['/K'] if '/K' in page_part else Array()
            arr4.append(fig_el)
            page_part[Name('/K')] = arr4

        # ----- (C) Tables (minimal) -----
        tables_json = load_json(os.path.join(struct_dir, f'page_{pnum:03d}.tables.json'), {}) or {}
        for row in (tables_json.get('tables') or []):
            if not isinstance(row, dict):
                continue
            cells = row.get('cells') or []
            if not cells:
                continue

            table_el = make_indirect(pdf, make_selement(pdf, 'Table'))
            tr_el = make_indirect(pdf, make_selement(pdf, 'TR'))
            table_el[Name('/K')] = Array([tr_el])

            # attach under page_part
            arr5 = page_part['/K'] if '/K' in page_part else Array()
            arr5.append(table_el)
            page_part[Name('/K')] = arr5

            for cell in cells:
                td_el = make_indirect(pdf, make_selement(pdf, 'TD'))
                p_el  = make_indirect(pdf, make_selement(pdf, 'P'))
                td_el[Name('/K')] = Array([p_el])
                # TR -> TD
                arr6 = tr_el['/K'] if '/K' in tr_el else Array()
                arr6.append(td_el)
                tr_el[Name('/K')] = arr6
                tag_block(pdf, st_root, p_el, page_obj,
                          [{'tag':'P','text':str(cell) if cell is not None else ''}],
                          parenttree=parenttree, page_key=page_key)

        # ----- (D) Headings & paragraphs (remaining) -----
        consumed_texts = set([c.get('text','') for c in caps if isinstance(c, dict)])
        # include list items
        consumed_texts.update(li_texts)

        text_items = []
        for t in tags:
            tagname = t.get('tag')
            if tagname in ('H1','H2','P'):
                tx = (t.get('text','') or '').strip()
                if tx and tx not in consumed_texts:
                    text_items.append({'tag': tagname, 'text': tx})

        for it in text_items:
            se = make_indirect(pdf, make_selement(pdf, TAG_TO_S.get(it['tag'], 'P')))
            arr7 = page_part['/K'] if '/K' in page_part else Array()
            arr7.append(se)
            page_part[Name('/K')] = arr7
            tag_block(pdf, st_root, se, page_obj,
                      [{'tag':'P','text': it['text']}],
                      parenttree=parenttree, page_key=page_key)

    # ----- Write ParentTree -----
    pt_dict = parenttree.build_numbertree_dict()
    st_root[Name('/ParentTree')] = make_indirect(pdf, pt_dict)
    st_root[Name('/ParentTreeNextKey')] = next_key

    # Save linearized
    pdf.save(pdf_out, linearize=True)
    print("Wrote tagged PDF:", os.path.abspath(pdf_out))

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python TagPikePDF.py Test.pdf fused_tags structures out_tagged.pdf")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
