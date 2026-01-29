#!/usr/bin/env python3
"""
UntagPDF.py
Remove accessibility tags from tagged PDFs to enable retagging
"""
import sys
import pikepdf
from pikepdf import Name


def untag_pdf(input_path, output_path):
    """
    Remove all accessibility tags from a PDF.

    This removes:
    - StructTreeRoot (structure tree)
    - MarkInfo (marked content flag)
    - Tagged metadata
    - Parent tree
    - Role maps

    Returns the PDF to an untagged state for retagging.
    """
    try:
        pdf = pikepdf.open(input_path)

        # Track what was removed
        removed_items = []

        # Remove StructTreeRoot
        if Name.StructTreeRoot in pdf.Root:
            del pdf.Root[Name.StructTreeRoot]
            removed_items.append("StructTreeRoot")

        # Remove MarkInfo
        if Name.MarkInfo in pdf.Root:
            del pdf.Root[Name.MarkInfo]
            removed_items.append("MarkInfo")

        # Remove ParentTree references from pages
        for page in pdf.pages:
            page_obj = page.obj
            if Name.StructParents in page_obj:
                del page_obj[Name.StructParents]

        # Clean up page content streams (remove marked content tags)
        for page in pdf.pages:
            try:
                content = page.get_contents()
                if content:
                    # Note: This is a simplified cleanup
                    # Full cleanup would require parsing content streams
                    # For now, we just remove the structure tree references
                    pass
            except Exception:
                pass

        # Save the untagged PDF
        pdf.save(output_path, linearize=True)
        pdf.close()

        if removed_items:
            print(f"✅ Removed tags: {', '.join(removed_items)}")
        else:
            print("ℹ️  No tags found to remove")

        return True

    except Exception as e:
        print(f"❌ Error untagging PDF: {e}")
        return False


def is_pdf_tagged(pdf_path):
    """
    Check if a PDF has accessibility tags.
    Returns True if tagged, False otherwise.
    """
    try:
        pdf = pikepdf.open(pdf_path)

        # Check for StructTreeRoot
        has_struct_tree = Name.StructTreeRoot in pdf.Root

        # Check for MarkInfo
        has_mark_info = Name.MarkInfo in pdf.Root

        pdf.close()

        return has_struct_tree or has_mark_info

    except Exception as e:
        print(f"Warning: Could not check if PDF is tagged: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python UntagPDF.py input.pdf output.pdf")
        print("\nRemoves accessibility tags from a PDF to enable retagging")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2]

    # Check if PDF is tagged
    if not is_pdf_tagged(input_pdf):
        print(f"ℹ️  {input_pdf} is not tagged - no changes needed")
        sys.exit(0)

    print(f"Untagging: {input_pdf}")

    if untag_pdf(input_pdf, output_pdf):
        print(f"✅ Untagged PDF saved to: {output_pdf}")
        sys.exit(0)
    else:
        print(f"❌ Failed to untag PDF")
        sys.exit(1)