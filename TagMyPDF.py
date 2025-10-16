#!/usr/bin/env python3
import os, sys, subprocess

def run(cmd):
    """Run a shell command and stream its output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Error running: {cmd}")
        sys.exit(result.returncode)

def main(pdf_path):
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    # Folder structure
    debug = "debug"
    layout_out = "layout_out"
    lines_out = "lines_out"
    fused_tags = "fused_tags"
    promoted_tags = "promoted_tags"
    structures = "structures"
    out_pdf = "out_tagged.pdf"

    # clean old outputs
    for d in [debug, layout_out, lines_out, fused_tags, promoted_tags, structures]:
        os.makedirs(d, exist_ok=True)

    print("\n=== Starting Accessible PDF Tagger ===")

    # Step 1: Probe
    run(f"python ProbingFile.py \"{pdf_path}\"")
    #
    # # Step 2: Heuristics (optional early cleanup)
    # run(f"python HeuristicsWithoutAI.py \"{pdf_path}\"")

    # Step 3: Layout detection
    run(f"python DetectLayout.py debug {layout_out} 150 --pdf \"{pdf_path}\" --labelmap labelmap.json --debugpng")

    # Step 4: Line export
    run(f"python ExportLines.py \"{pdf_path}\" {lines_out}")

    # Step 5: Fuse lines + layout
    run(f"python FuseLines.py \"{pdf_path}\" {layout_out} {lines_out} {fused_tags}")

    # Step 6: Detect tables/lists
    run(f"python TableLists.py {fused_tags} {structures}")

    # Step 7: Promote tags
    run(f"python promote_tags.py {fused_tags} {promoted_tags} {layout_out}")

    # Step 8: Build tagged PDF
    run(f"python TagPikePDF_fixed2.py \"{pdf_path}\" {promoted_tags} {structures} {out_pdf}")

    print(f"\n✅ Done! Final accessible PDF: {out_pdf}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python TagMyPDF.py <input.pdf>")
        sys.exit(1)
    main(sys.argv[1])
