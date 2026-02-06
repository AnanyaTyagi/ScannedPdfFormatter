#!/usr/bin/env python3
"""
RunPipeline.py
Simple pipeline runner with full diagnostics
"""
import os
import sys
import subprocess
import shutil
import glob
import json


def run_cmd(cmd, description):
    """Run command and show output."""
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"{'=' * 80}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print("\nOutput:")
        print(result.stdout)

    if result.stderr:
        print("\nErrors/Warnings:")
        print(result.stderr)

    if result.returncode != 0:
        print(f"\n❌ Command failed with return code: {result.returncode}")
        return False

    print(f"✅ Success!")
    return True


def setup_folders():
    """Create and clean output folders."""
    folders = ["debug", "layout_out", "lines_out", "fused_tags", "structures"]

    print("\nSetting up folders...")
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"  Cleaned: {folder}/")
        os.makedirs(folder, exist_ok=True)
        print(f"  Created: {folder}/")

    return folders


def verify_folder(folder, pattern="*.json"):
    """Check if folder has expected output."""
    if not os.path.exists(folder):
        print(f"❌ Folder not created: {folder}")
        return False

    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        print(f"⚠️  No files in {folder}/")
        all_files = os.listdir(folder)
        if all_files:
            print(f"   But found: {all_files[:5]}")
        return False

    print(f"✅ {folder}/ has {len(files)} files")

    # Show first file as sample
    if files:
        first_file = files[0]
        size = os.path.getsize(first_file)
        print(f"   Sample: {os.path.basename(first_file)} ({size:,} bytes)")

        # If JSON, show structure
        if first_file.endswith('.json'):
            try:
                with open(first_file, 'r') as f:
                    data = json.load(f)
                print(f"   Keys: {list(data.keys())}")
            except:
                pass

    return True


def main(pdf_path):
    """Run the complete pipeline."""

    print("=" * 80)
    print("PDF TAGGING PIPELINE")
    print("=" * 80)

    # Check PDF exists
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return False

    print(f"\nInput PDF: {pdf_path}")
    print(f"Size: {os.path.getsize(pdf_path):,} bytes")

    # Get page count
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        print(f"Pages: {page_count}")
    except Exception as e:
        print(f"⚠️  Could not read PDF: {e}")
        return False

    # Setup folders
    folders = setup_folders()

    # Step 1: Export lines
    if not run_cmd(
            [sys.executable, "ExportLines.py", pdf_path, "lines_out"],
            "STEP 1: Extracting text lines"
    ):
        return False

    if not verify_folder("lines_out", "page_*.lines.json"):
        print("\n❌ Line extraction failed - no output files")
        return False

    # Step 2: Layout detection
    if not run_cmd(
            [sys.executable, "DetectLayoutFixed.py", "debug", "layout_out", "300",
             "--pdf", pdf_path, "--labelmap", "labelmap.json",
             "--weights", "yolov8n-doclaynet.pt"],
            "STEP 2: Detecting document layout"
    ):
        return False

    if not verify_folder("layout_out", "page_*.layout.json"):
        print("\n❌ Layout detection failed - no output files")
        return False

    # Step 3: Fuse lines and layout
    if not run_cmd(
            [sys.executable, "FuseLinesFixed_fixed.py", pdf_path,
             "layout_out", "lines_out", "fused_tags"],
            "STEP 3: Fusing text lines with layout"
    ):
        return False

    if not verify_folder("fused_tags", "page_*.tags.json"):
        print("\n❌ Tag fusion failed - no output files")
        return False

    # Step 4: Create tagged PDF
    output_pdf = "output_tagged.pdf"
    if not run_cmd(
            [sys.executable, "TagPDFFinal.py", pdf_path,
             "fused_tags", "structures", output_pdf],
            "STEP 4: Creating tagged PDF"
    ):
        return False

    # Verify output
    if not os.path.exists(output_pdf):
        print(f"\n❌ Output PDF not created: {output_pdf}")
        return False

    output_size = os.path.getsize(output_pdf)
    if output_size == 0:
        print(f"\n❌ Output PDF is empty")
        return False

    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nOutput: {output_pdf}")
    print(f"Size: {output_size:,} bytes")

    # Summary
    print("\nGenerated files:")
    for folder in folders:
        if os.path.exists(folder):
            file_count = len(os.listdir(folder))
            print(f"  {folder:15} - {file_count} files")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 RunPipeline.py <input.pdf>")
        print("\nExample:")
        print("  python3 RunPipeline.py document.pdf")
        sys.exit(1)

    pdf = sys.argv[1]
    success = main(pdf)

    sys.exit(0 if success else 1)