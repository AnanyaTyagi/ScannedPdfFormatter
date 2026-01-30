#!/usr/bin/env python3
"""
diagnose_coordinates.py
Test script to understand the coordinate systems in your pipeline
"""
import sys
import json
import fitz

if len(sys.argv) < 2:
    print("Usage: python diagnose_coordinates.py input.pdf")
    sys.exit(1)

pdf_path = sys.argv[1]
doc = fitz.open(pdf_path)
page = doc[0]  # First page

print("=" * 70)
print("COORDINATE SYSTEM DIAGNOSTIC")
print("=" * 70)

# Get page dimensions
mediabox = page.mediabox
print(f"\n📄 Page MediaBox: {mediabox}")
print(f"   Width: {mediabox.width} pts")
print(f"   Height: {mediabox.height} pts")

# Get first word from text layer
words = page.get_text("words")
if words:
    first_word = words[0]
    print(f"\n📝 First word from PyMuPDF:")
    print(f"   Text: '{first_word[4]}'")
    print(f"   Bbox: [{first_word[0]:.1f}, {first_word[1]:.1f}, {first_word[2]:.1f}, {first_word[3]:.1f}]")
    print(f"   x0={first_word[0]:.1f}, y0={first_word[1]:.1f}, x1={first_word[2]:.1f}, y1={first_word[3]:.1f}")
    
    # Analyze position
    y0 = first_word[1]
    y1 = first_word[3]
    page_height = mediabox.height
    
    print(f"\n🔍 Position Analysis:")
    print(f"   y0 = {y0:.1f}")
    print(f"   y1 = {y1:.1f}")
    print(f"   page_height = {page_height:.1f}")
    
    if y0 < page_height / 2:
        print(f"   ✅ y0 < page_height/2 → Word is in TOP half")
        print(f"   → PyMuPDF uses BOTTOM-LEFT origin (PDF standard)")
    else:
        print(f"   ⚠️  y0 > page_height/2 → Word is in BOTTOM half")
        print(f"   → This seems unusual!")

# Check what coordinate system lines.json uses
try:
    with open("lines_out/page_001.lines.json", "r") as f:
        lines_data = json.load(f)
    
    if lines_data.get("lines"):
        first_line = lines_data["lines"][0]
        print(f"\n📋 First line from lines.json:")
        print(f"   Text: '{first_line.get('text', '')[:50]}...'")
        print(f"   Bbox: {first_line.get('bbox')}")
        
        bbox = first_line.get('bbox', [0,0,0,0])
        if bbox[1] < page_height / 2:
            print(f"   → Likely BOTTOM-LEFT origin (y0 < page_height/2)")
        else:
            print(f"   → Likely TOP-LEFT origin (y0 > page_height/2)")
except:
    print("\n⚠️  Could not load lines.json (run ExportLines.py first)")

# Check layout.json
try:
    with open("layout_out/page_001.layout.json", "r") as f:
        layout_data = json.load(f)
    
    if layout_data.get("regions"):
        first_region = layout_data["regions"][0]
        print(f"\n🎯 First region from layout.json:")
        print(f"   Type: {first_region.get('cls')}")
        print(f"   Bbox: {first_region.get('bbox')}")
        
        bbox = first_region.get('bbox', [0,0,0,0])
        pdf_h = layout_data.get('height', page_height)
        if bbox[1] < pdf_h / 2:
            print(f"   → Likely BOTTOM-LEFT origin (y0 < height/2)")
        else:
            print(f"   → Likely TOP-LEFT origin (y0 > height/2)")
except:
    print("\n⚠️  Could not load layout.json (run DetectLayoutFixed.py first)")

print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print("PyMuPDF (fitz) uses: BOTTOM-LEFT origin (PDF standard)")
print("Your pipeline should: Match this coordinate system")
print("\nIf TagPDFFinal.py is FLIPPING coordinates (page_h - y),")
print("that means it expects TOP-LEFT input but receives BOTTOM-LEFT!")
print("=" * 70)

doc.close()
