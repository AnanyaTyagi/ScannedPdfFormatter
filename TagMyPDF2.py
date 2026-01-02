# #!/usr/bin/env python3
# import os, sys, subprocess, shlex, shutil, glob
#
# def q(p):  # shell-safe quoting
#     return shlex.quote(str(p))
#
# def run(cmd_list, check=True):
#     """Run a command list and stream output."""
#     cmd_str = " ".join(q(c) for c in cmd_list)
#     print(f"\n>>> {cmd_str}")
#     result = subprocess.run(cmd_str, shell=True)
#     if check and result.returncode != 0:
#         print(f"âŒ Error running: {cmd_str}")
#         sys.exit(result.returncode)
#     return result.returncode
#
# def ensure_nonempty_dir(path, pattern="*"):
#     files = glob.glob(os.path.join(path, pattern))
#     if not files:
#         print(f"âŒ Expected outputs in {path} (pattern {pattern}) but found none.")
#         sys.exit(2)
#
# def clean_dirs(*dirs):
#     for d in dirs:
#         if os.path.isdir(d):
#             shutil.rmtree(d, ignore_errors=True)
#             print(f"Cleaned up: {d}")
#         os.makedirs(d, exist_ok=True)
#
# def which_script(*candidates):
#     for c in candidates:
#         if os.path.exists(c):
#             return c
#     # Also try to locate in CWD without path case-sensitivity differences
#     lower = {f.lower(): f for f in os.listdir(".")}
#     for c in candidates:
#         name = os.path.basename(c).lower()
#         if name in lower:
#             return lower[name]
#     # If still not found, return the first (we'll let it fail clearly)
#     return candidates[0]
#
# def main(pdf_path, dpi="300", weights=None, labelmap="labelmap.json"):
#     py = sys.executable or "python"
#
#     # Resolve scripts (with sensible fallbacks)
#     probing = which_script("ProbingFile.py")
#     detect  = which_script("DetectLayoutFixed.py", "DetectLayout.py")
#     export  = which_script("ExportLines.py")
#     fuse    = which_script("FuseLinesFixed.py", "FuseLines.py")
#     tables  = which_script("TableLists.py")
#     tagger  = which_script("TagPDFFinal.py", "TagPikePDFProduction.py")
#
#     # Output folders
#     debug_dir      = "debug"
#     layout_out     = "layout_out"
#     lines_out      = "lines_out"
#     fused_tags     = "fused_tags"
#     structures     = "structures"
#     out_pdf        = "output.pdf"
#
#     # Clean old outputs
#     clean_dirs(debug_dir, layout_out, lines_out, fused_tags, structures)
#     if os.path.exists(out_pdf):
#         os.remove(out_pdf)
#         print(f"Cleaned up: {out_pdf}")
#
#     print("\n=== Starting Accessible PDF Tagger ===")
#
#     # 1) Probe
#     run([py, probing, pdf_path])
#     ensure_nonempty_dir(debug_dir, "preview_p*.png")
#
#     # 2) Layout detection (YOLO)
#     detect_args = [py, detect, debug_dir, layout_out, dpi, "--pdf", pdf_path, "--debugpng"]
#     if labelmap and os.path.exists(labelmap):
#         detect_args += ["--labelmap", labelmap]
#     if weights and os.path.exists(weights):
#         detect_args += ["--weights", weights]
#     run(detect_args)
#     ensure_nonempty_dir(layout_out, "page_*.layout.json")
#
#     # 3) Export lines
#     run([py, export, pdf_path, lines_out])
#     ensure_nonempty_dir(lines_out, "page_*.lines.json")
#
#     # 4) Fuse lines + layout
#     run([py, fuse, pdf_path, layout_out, lines_out, fused_tags])
#     ensure_nonempty_dir(fused_tags, "page_*.tags.json")
#
#     # 5) Tables / Lists detection
#     run([py, tables, fused_tags, structures])
#     # Not every doc has tables/lists; donâ€™t hard fail if empty.
#     # If you want to require, uncomment:
#     # ensure_nonempty_dir(structures, "page_*.tables.json")
#
#     # 6) Build the accessible tagged PDF
#     run([py, tagger, pdf_path, fused_tags, structures, out_pdf])
#
#     if not os.path.exists(out_pdf) or os.path.getsize(out_pdf) == 0:
#         print("âŒ Tagging step did not produce a valid PDF.")
#         sys.exit(3)
#
#     print(f"\nâœ… Done! Final accessible PDF: {out_pdf}")
#
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python TagMyPDF.py <input.pdf> [--dpi 300] [--weights yolov8n-doclaynet.pt] [--labelmap labelmap.json]")
#         sys.exit(1)
#
#     # simple arg parsing
#     pdf = sys.argv[1]
#     dpi = "300"
#     weights = None
#     labelmap = "labelmap.json"
#
#     if "--dpi" in sys.argv:
#         i = sys.argv.index("--dpi")
#         if i + 1 < len(sys.argv):
#             dpi = sys.argv[i + 1]
#     if "--weights" in sys.argv:
#         i = sys.argv.index("--weights")
#         if i + 1 < len(sys.argv):
#             weights = sys.argv[i + 1]
#     if "--labelmap" in sys.argv:
#         i = sys.argv.index("--labelmap")
#         if i + 1 < len(sys.argv):
#             labelmap = sys.argv[i + 1]
#
#     main(pdf, dpi=dpi, weights=weights, labelmap=labelmap)


# !/usr/bin/env python3
import os, sys, subprocess, shlex, shutil, glob
import shutil

def q(p):  # shell-safe quoting
    return shlex.quote(str(p))


def run(cmd_list, check=True):
    """Run a command list and stream output."""
    cmd_str = " ".join(q(c) for c in cmd_list)
    print(f"\n>>> {cmd_str}")
    result = subprocess.run(cmd_str, shell=True)
    if check and result.returncode != 0:
        print(f"❌ Error running: {cmd_str}")
        sys.exit(result.returncode)
    return result.returncode


def ensure_nonempty_dir(path, pattern="*"):
    files = glob.glob(os.path.join(path, pattern))
    if not files:
        print(f"❌ Expected outputs in {path} (pattern {pattern}) but found none.")
        sys.exit(2)


def clean_dirs(*dirs):
    for d in dirs:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"Cleaned up: {d}")
        os.makedirs(d, exist_ok=True)


def which_script(script_dir, *candidates):
    """
    FIXED: Find scripts in the script directory (where TagMyPDF2.py is located),
    not in the current working directory.
    """
    for c in candidates:
        # Check in script directory with absolute path
        full_path = os.path.join(script_dir, c)
        if os.path.exists(full_path):
            return full_path

        # Also try case-insensitive match
        try:
            files_in_dir = {f.lower(): f for f in os.listdir(script_dir)}
            name_lower = os.path.basename(c).lower()
            if name_lower in files_in_dir:
                return os.path.join(script_dir, files_in_dir[name_lower])
        except:
            pass

    # If still not found, return the first with full path (will fail clearly)
    return os.path.join(script_dir, candidates[0])


def main(pdf_path, dpi="300", weights=None, labelmap="labelmap.json"):
    # Find Python executable
    # For Streamlit Cloud
    if os.path.exists("/home/adminuser/venv/bin/python"):
        py = "/home/adminuser/venv/bin/python"
    else:
        py = sys.executable

    print(f"Using Python: {py}")

    # ═══════════════════════════════════════════════════════════════════
    # CRITICAL FIX: Get absolute path to script directory
    # ═══════════════════════════════════════════════════════════════════
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Resolve scripts with absolute paths
    probing = which_script(script_dir, "ProbingFile.py")
    detect = which_script(script_dir, "DetectLayoutFixed.py", "DetectLayout.py")
    export = which_script(script_dir, "ExportLines.py")
    fuse = which_script(script_dir, "FuseLinesFixed.py", "FuseLines.py")
    tables = which_script(script_dir, "TableLists.py")
    tagger = which_script(script_dir, "TagPDFFinal.py", "TagPikePDFProduction.py")

    # Convert labelmap and weights to absolute paths if provided
    # Convert labelmap to absolute path
    if labelmap and not os.path.isabs(labelmap):
        labelmap = os.path.join(script_dir, labelmap)

    # FIX: Auto-detect and set absolute path for weights
    if weights is None:
        # Check for common weight file names in script directory
        for weight_name in ["yolov8n-doclaynet.pt", "best.pt", "yolov8n.pt"]:
            weight_path = os.path.join(script_dir, weight_name)
            if os.path.exists(weight_path):
                weights = weight_path
                print(f"Found weights: {weights}")
                break
    elif not os.path.isabs(weights):
        weights = os.path.join(script_dir, weights)
    # Output folders (relative to current working directory - worker_0/)
    debug_dir = "debug"
    layout_out = "layout_out"
    lines_out = "lines_out"
    fused_tags = "fused_tags"
    structures = "structures"
    out_pdf = "output.pdf"

    # Clean old outputs
    clean_dirs(debug_dir, layout_out, lines_out, fused_tags, structures)
    if os.path.exists(out_pdf):
        os.remove(out_pdf)
        print(f"Cleaned up: {out_pdf}")

    print("\n=== Starting Accessible PDF Tagger ===")

    # 1) Probe
    run([py, probing, pdf_path])
    ensure_nonempty_dir(debug_dir, "preview_p*.png")

    # 2) Layout detection (YOLO)
    detect_args = [py, detect, debug_dir, layout_out, dpi, "--pdf", pdf_path, "--debugpng"]
    if labelmap and os.path.exists(labelmap):
        detect_args += ["--labelmap", labelmap]
    if weights and os.path.exists(weights):
        detect_args += ["--weights", weights]
    run(detect_args)
    ensure_nonempty_dir(layout_out, "page_*.layout.json")

    # 3) Export lines
    run([py, export, pdf_path, lines_out])
    ensure_nonempty_dir(lines_out, "page_*.lines.json")

    # 4) Fuse lines + layout
    run([py, fuse, pdf_path, layout_out, lines_out, fused_tags])
    ensure_nonempty_dir(fused_tags, "page_*.tags.json")

    # 5) Tables / Lists detection
    run([py, tables, fused_tags, structures])

    # 6) Build the accessible tagged PDF
    run([py, tagger, pdf_path, fused_tags, structures, out_pdf])

    if not os.path.exists(out_pdf) or os.path.getsize(out_pdf) == 0:
        print("❌ Tagging step did not produce a valid PDF.")
        sys.exit(3)

    print(f"\n✅ Done! Final accessible PDF: {out_pdf}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python TagMyPDF2.py <input.pdf> [--dpi 300] [--weights yolov8n-doclaynet.pt] [--labelmap labelmap.json]")
        sys.exit(1)

    # simple arg parsing
    pdf = sys.argv[1]
    dpi = "300"
    weights = None
    labelmap = "labelmap.json"

    if "--dpi" in sys.argv:
        i = sys.argv.index("--dpi")
        if i + 1 < len(sys.argv):
            dpi = sys.argv[i + 1]
    if "--weights" in sys.argv:
        i = sys.argv.index("--weights")
        if i + 1 < len(sys.argv):
            weights = sys.argv[i + 1]
    if "--labelmap" in sys.argv:
        i = sys.argv.index("--labelmap")
        if i + 1 < len(sys.argv):
            labelmap = sys.argv[i + 1]

    main(pdf, dpi=dpi, weights=weights, labelmap=labelmap)