#!/usr/bin/env python3
"""
DIAGNOSTIC: Test complete Force Retag workflow
This simulates exactly what your app does when Force Retag is enabled.
"""

import os
import sys
import shutil
import subprocess
import pikepdf
from pikepdf import Name


def is_pdf_already_tagged(pdf_path):
    """Check if PDF already has accessibility tags."""
    try:
        pdf = pikepdf.open(pdf_path)
        catalog = pdf.Root
        has_struct_tree = '/StructTreeRoot' in catalog
        pdf.close()
        return has_struct_tree
    except Exception as e:
        print(f"ERROR checking if tagged: {e}")
        return False


def test_complete_workflow(input_pdf_path):
    """Test the complete workflow: check → untag → tag"""

    print("=" * 80)
    print("COMPLETE WORKFLOW TEST")
    print("=" * 80)

    # Create temp directories
    test_dir = "/tmp/workflow_test"
    work_dir = os.path.join(test_dir, "work")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    for folder in ["debug", "fused_tags", "layout_out", "lines_out", "structures"]:
        os.makedirs(os.path.join(work_dir, folder), exist_ok=True)

    # Copy input PDF to test location
    test_pdf = os.path.join(test_dir, "test_input.pdf")
    shutil.copy(input_pdf_path, test_pdf)

    print(f"\n1. INPUT PDF: {os.path.basename(input_pdf_path)}")
    print(f"   Path: {test_pdf}")
    print(f"   Size: {os.path.getsize(test_pdf)} bytes")

    # Check if tagged
    print(f"\n2. CHECKING IF TAGGED...")
    is_tagged = is_pdf_already_tagged(test_pdf)
    print(f"   is_pdf_already_tagged() = {is_tagged}")

    if not is_tagged:
        print("   ⚠️ PDF is NOT tagged - Force Retag won't trigger untag")
        print("   ✓ Will proceed directly to tagging")
    else:
        print("   ✓ PDF IS tagged")
        print("   ✓ Force Retag SHOULD trigger untagging")

        # Untag it
        print(f"\n3. UNTAGGING...")
        try:
            from pikepdf import Array, Dictionary

            pdf = pikepdf.open(test_pdf)

            # Remove structures
            removed = []
            if Name.StructTreeRoot in pdf.Root:
                del pdf.Root[Name.StructTreeRoot]
                removed.append("StructTreeRoot")
            if Name.MarkInfo in pdf.Root:
                del pdf.Root[Name.MarkInfo]
                removed.append("MarkInfo")
            if Name.Lang in pdf.Root:
                del pdf.Root[Name.Lang]
                removed.append("Lang")

            # Clean pages
            for page in pdf.pages:
                if Name.StructParents in page.obj:
                    del page.obj[Name.StructParents]
                    removed.append("StructParents(page)")

            print(f"   Removed: {removed}")

            # Save
            temp_path = test_pdf + ".temp"
            pdf.save(temp_path)
            pdf.close()

            shutil.move(temp_path, test_pdf)

            # Verify
            still_tagged = is_pdf_already_tagged(test_pdf)
            if still_tagged:
                print(f"   ❌ ERROR: PDF still tagged after untag!")
            else:
                print(f"   ✅ Untag successful")

        except Exception as e:
            print(f"   ❌ Untag failed: {e}")
            import traceback
            traceback.print_exc()

    # Now run TagMyPDF2.py
    print(f"\n4. RUNNING TagMyPDF2.py...")

    # Copy to work dir as temp.pdf
    temp_pdf = os.path.join(work_dir, "temp.pdf")
    shutil.copy(test_pdf, temp_pdf)
    print(f"   Copied to: {temp_pdf}")

    # Find TagMyPDF2.py
    # Assume it's in current directory
    tagmypdf_script = "TagMyPDF2.py"
    if not os.path.exists(tagmypdf_script):
        tagmypdf_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TagMyPDF2.py")

    print(f"   Script: {tagmypdf_script}")
    print(f"   Exists: {os.path.exists(tagmypdf_script)}")

    if not os.path.exists(tagmypdf_script):
        print("   ❌ TagMyPDF2.py not found!")
        print("   Please run this script from the directory containing TagMyPDF2.py")
        return

    # Run it
    cmd = ["python", tagmypdf_script, "temp.pdf"]
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Working dir: {work_dir}")

    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )

        print(f"\n5. TagMyPDF2.py RESULTS:")
        print(f"   Return code: {result.returncode}")
        print(f"   Duration: ~instant" if result.returncode else "")

        if result.stdout:
            print(f"\n   STDOUT (first 20 lines):")
            for line in result.stdout.split('\n')[:20]:
                if line.strip():
                    print(f"     {line}")

        if result.stderr:
            print(f"\n   STDERR (first 20 lines):")
            for line in result.stderr.split('\n')[:20]:
                if line.strip():
                    print(f"     {line}")

        # Check for output
        output_pdf = os.path.join(work_dir, "output.pdf")
        print(f"\n6. OUTPUT CHECK:")
        print(f"   output.pdf exists: {os.path.exists(output_pdf)}")

        if os.path.exists(output_pdf):
            print(f"   output.pdf size: {os.path.getsize(output_pdf)} bytes")

            # Check if it's tagged
            is_output_tagged = is_pdf_already_tagged(output_pdf)
            print(f"   output.pdf is tagged: {is_output_tagged}")

            if is_output_tagged:
                print(f"\n✅ SUCCESS - Output PDF is properly tagged!")
            else:
                print(f"\n❌ PROBLEM - Output PDF is NOT tagged!")
        else:
            print(f"\n❌ PROBLEM - No output.pdf created!")
            print(f"\n   Checking work directory contents:")
            for item in os.listdir(work_dir):
                print(f"     - {item}")

        print("=" * 80)

    except subprocess.TimeoutExpired:
        print(f"   ❌ TIMEOUT after 60 seconds!")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_workflow.py <pdf_file>")
        print("Example: python test_workflow.py asian_festival.pdf")
        sys.exit(1)

    input_pdf = sys.argv[1]

    if not os.path.exists(input_pdf):
        print(f"ERROR: File not found: {input_pdf}")
        sys.exit(1)

    test_complete_workflow(input_pdf)