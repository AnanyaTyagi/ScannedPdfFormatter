#!/usr/bin/env python3
import os, sys, json, glob

def main(pdf_in, fused_dir, struct_dir, pdf_out):
    try:
        import pdftron
        from pdftron.PDF import PDFDoc, ElementBuilder, ElementWriter, Struct, SDFDoc
        from pdftron.PDF.Struct import STree, SElement
        from pdftron.Common import PDFNetException
    except Exception:
        print("Apryse/PDFTron Python SDK not installed. Install: pip install pdftron")
        sys.exit(1)

    pdftron.PDFNet.Initialize()  # requires license env if applicable

    doc = PDFDoc(pdf_in)
    doc.InitSecurityHandler()

    st = STree.Create(doc)  # Structure Tree Root
    doc.GetRoot().Put("StructTreeRoot", st.GetSDFObj())

    # RoleMap (map custom roles to standard)
    role_map = st.GetRoleMap()
    for k,v in {
        "Document": "Document",
        "Part": "Part",
        "H1": "H1",
        "H2": "H2",
        "P": "P",
        "L": "L",
        "LI": "LI",
        "Table": "Table",
        "TR": "TR",
        "TH": "TH",
        "TD": "TD",
        "Figure": "Figure",
        "Caption": "Caption"
    }.items():
        role_map.PutName(k, v)

    # Root element: Document -> Part
    root = SElement.Create(st, "Document")
    st.SetRoot(root)
    part = SElement.Create(doc, "Part")
    root.AppendChild(part)

    # Simple artifact detector: repeated top/bottom lines
    def is_artifact(line):
        t=line["text"].strip()
        if not t: return False
        # very naive: page number-only or 1-2 words at extremes
        if len(t)<=3 and t.isdigit(): return True
        return False

    # iterate pages
    page_files = sorted(glob.glob(os.path.join(fused_dir, "page_*.tags.json")))
    for tp in page_files:
        with open(tp,"r",encoding="utf-8") as f: T=json.load(f)
        pnum=T["page"]; tags=T["tags"]
        page = doc.GetPage(pnum)
        if not page: continue

        eb = ElementBuilder()
        ew = ElementWriter()
        ew.Begin(page, ElementWriter.e_overlay)  # overlay: keep original appearance

        # container for page content
        pg_group = SElement.Create(doc, "Part")
        part.AppendChild(pg_group)

        # lists accumulator
        list_open=False; list_el=None

        for t in tags:
            if is_artifact(t):
                # mark artifact (we'll skip tagging and keep as visual-only)
                continue
            tag = t["tag"]; bbox=t["bbox"]; text=t["text"]

            if tag in ("H1","H2","P"):
                se = SElement.Create(doc, tag)
                pg_group.AppendChild(se)
                # Marked content (we place an MCID at bbox; here we add an invisible text mark)
                e = eb.CreateTextBegin()
                e.SetStructParentIndex(se.GetStructMCID())  # associate MCID
                ew.WriteElement(e); ew.WriteElement(eb.CreateTextEnd())
                # attach ActualText to improve text extraction
                se.SetActualText(text)

            elif tag=="LI":
                if not list_open:
                    list_el = SElement.Create(doc, "L")
                    pg_group.AppendChild(list_el)
                    list_open=True
                li = SElement.Create(doc, "LI")
                list_el.AppendChild(li)
                p = SElement.Create(doc, "P")
                li.AppendChild(p)
                p.SetActualText(text)
            else:
                # Table/Figure will be added from structures json below
                pass

        if list_open:
            list_open=False

        ew.End()

        # Tables: very light structure (rows/td from structures dir)
        tab_path = os.path.join(struct_dir, f"page_{pnum:03d}.tables.json")
        if os.path.exists(tab_path):
            with open(tab_path,"r",encoding="utf-8") as f: TJ=json.load(f)
            for ti, row in enumerate(TJ.get("tables",[]), start=1):
                table = SElement.Create(doc, "Table")
                pg_group.AppendChild(table)
                tr = SElement.Create(doc, "TR"); table.AppendChild(tr)
                # naive: treat all as TD, first row as TH if you prefer
                for cell in row["cells"]:
                    td = SElement.Create(doc, "TD")
                    tr.AppendChild(td)
                    td.SetActualText(cell)

        # Figures & captions: from fused tags (FigureCaptionCandidate)
        # Promote any "FigureCaptionCandidate" near figure region as Caption under Figure
        figs = [t for t in tags if t.get("region",{}).get("cls")=="figure"]
        caps = [t for t in tags if t["tag"]=="FigureCaptionCandidate"]
        for f in figs:
            fig = SElement.Create(doc,"Figure"); pg_group.AppendChild(fig)
            # Add alt text using caption if available (safe, non-hallucinatory)
            near = next((c for c in caps if c["bbox"][1] >= f["bbox"][3]-5 and c["bbox"][1] <= f["bbox"][3]+40), None)
            if near:
                cap = SElement.Create(doc,"Caption"); fig.AppendChild(cap)
                cap.SetActualText(near["text"])
                fig.SetAlt(near["text"])

    doc.Save(pdf_out, SDFDoc.e_linearized)
    pdftron.PDFNet.Terminate()
    print("wrote tagged PDF:", os.path.abspath(pdf_out))

if __name__=="__main__":
    if len(sys.argv)<5:
        print("Usage: python 06_tag_pdf_apryse.py Test.pdf fused_tags structures out_tagged.pdf"); sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
