#!/bin/bash

# ============================================================================
# ONE-COMMAND TEST SCRIPT
# ============================================================================
# This script tests Force Retag functionality end-to-end
#
# Usage: ./quick_test.sh /path/to/your/test.pdf
# ============================================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================================"
echo "FORCE RETAG TEST SCRIPT"
echo "============================================================================"

# Check if PDF provided
if [ $# -eq 0 ]; then
    echo -e "${RED}ERROR: No PDF file provided${NC}"
    echo "Usage: $0 /path/to/test.pdf"
    exit 1
fi

INPUT_PDF="$1"

if [ ! -f "$INPUT_PDF" ]; then
    echo -e "${RED}ERROR: File not found: $INPUT_PDF${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Input PDF: $INPUT_PDF"

# Create test directory
TEST_DIR="/tmp/force_retag_test_$$"
WORK_DIR="$TEST_DIR/work"
OUTPUT_DIR="$TEST_DIR/output"

mkdir -p "$WORK_DIR"/{debug,fused_tags,layout_out,lines_out,structures}
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}✓${NC} Created test directories: $TEST_DIR"

# Copy input PDF
TEST_PDF="$TEST_DIR/test_input.pdf"
cp "$INPUT_PDF" "$TEST_PDF"

# Step 1: Check if tagged
echo ""
echo "STEP 1: Checking if PDF is tagged..."
python3 - << EOF
import pikepdf
from pikepdf import Name

pdf = pikepdf.open('$TEST_PDF')
has_struct = Name.StructTreeRoot in pdf.Root
has_mark = Name.MarkInfo in pdf.Root
has_lang = Name.Lang in pdf.Root

print(f"  StructTreeRoot: {has_struct}")
print(f"  MarkInfo: {has_mark}")
print(f"  Lang: {has_lang}")

is_tagged = has_struct or has_mark or has_lang

if is_tagged:
    print("\n✓ PDF is TAGGED - Force Retag should trigger")
else:
    print("\n⚠ PDF is NOT TAGGED - Force Retag won't trigger untag")
    print("  (Will proceed directly to tagging)")

pdf.close()
exit(0 if is_tagged else 1)
EOF

IS_TAGGED=$?

# Step 2: Untag if tagged
if [ $IS_TAGGED -eq 0 ]; then
    echo ""
    echo "STEP 2: Untagging PDF..."
    python3 - << EOF
import pikepdf
from pikepdf import Name

pdf = pikepdf.open('$TEST_PDF')

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

for page in pdf.pages:
    if Name.StructParents in page.obj:
        del page.obj[Name.StructParents]
        if "StructParents" not in removed:
            removed.append("StructParents(pages)")

print(f"  Removed: {', '.join(removed)}")

pdf.save('$TEST_PDF')
pdf.close()

# Verify
pdf = pikepdf.open('$TEST_PDF')
still_tagged = Name.StructTreeRoot in pdf.Root or Name.MarkInfo in pdf.Root
pdf.close()

if still_tagged:
    print("\n✗ ERROR: PDF still tagged after untag!")
    exit(1)
else:
    print("\n✓ Untag successful")
    exit(0)
EOF
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Untag failed!${NC}"
        exit 1
    fi
else
    echo ""
    echo "STEP 2: Skipping untag (PDF not tagged)"
fi

# Step 3: Run TagMyPDF2.py
echo ""
echo "STEP 3: Running TagMyPDF2.py..."

# Copy to work dir
cp "$TEST_PDF" "$WORK_DIR/temp.pdf"

# Find TagMyPDF2.py
SCRIPT_PATH=""
for path in "./TagMyPDF2.py" "../TagMyPDF2.py" "/app/TagMyPDF2.py"; do
    if [ -f "$path" ]; then
        SCRIPT_PATH="$path"
        break
    fi
done

if [ -z "$SCRIPT_PATH" ]; then
    echo -e "${RED}✗ TagMyPDF2.py not found!${NC}"
    echo "  Searched: ./TagMyPDF2.py, ../TagMyPDF2.py, /app/TagMyPDF2.py"
    exit 1
fi

echo "  Script: $SCRIPT_PATH"

# Run it
cd "$WORK_DIR"
START_TIME=$(date +%s)

if python "$SCRIPT_PATH" temp.pdf > /tmp/tagmypdf_output.log 2>&1; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "  ${GREEN}✓ Completed in ${DURATION}s${NC}"
    
    # Check output
    if [ -f "output.pdf" ]; then
        echo -e "  ${GREEN}✓ output.pdf created${NC}"
        
        # Step 4: Verify output is tagged
        echo ""
        echo "STEP 4: Verifying output is tagged..."
        python3 - << EOF
import pikepdf
from pikepdf import Name

pdf = pikepdf.open('$WORK_DIR/output.pdf')
has_struct = Name.StructTreeRoot in pdf.Root
has_mark = Name.MarkInfo in pdf.Root

print(f"  StructTreeRoot: {has_struct}")
print(f"  MarkInfo: {has_mark}")

is_tagged = has_struct or has_mark

if is_tagged:
    print("\n✓ Output is TAGGED")
else:
    print("\n✗ Output is NOT TAGGED")

pdf.close()
exit(0 if is_tagged else 1)
EOF
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}============================================================================${NC}"
            echo -e "${GREEN}SUCCESS! Force Retag is working correctly!${NC}"
            echo -e "${GREEN}============================================================================${NC}"
            echo ""
            echo "Output saved to: $WORK_DIR/output.pdf"
            echo ""
            echo "To clean up: rm -rf $TEST_DIR"
        else
            echo ""
            echo -e "${RED}============================================================================${NC}"
            echo -e "${RED}FAILURE: Output PDF is not tagged!${NC}"
            echo -e "${RED}============================================================================${NC}"
            echo ""
            echo "TagMyPDF2.py ran but didn't tag the PDF properly."
            echo "Check logs at: /tmp/tagmypdf_output.log"
        fi
    else
        echo -e "  ${RED}✗ output.pdf NOT created${NC}"
        echo ""
        echo -e "${RED}============================================================================${NC}"
        echo -e "${RED}FAILURE: No output.pdf created${NC}"
        echo -e "${RED}============================================================================${NC}"
        echo ""
        echo "TagMyPDF2.py ran but didn't create output."
        echo "Check logs at: /tmp/tagmypdf_output.log"
        cat /tmp/tagmypdf_output.log
    fi
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "  ${RED}✗ Failed after ${DURATION}s${NC}"
    echo ""
    echo -e "${RED}============================================================================${NC}"
    echo -e "${RED}FAILURE: TagMyPDF2.py returned error${NC}"
    echo -e "${RED}============================================================================${NC}"
    echo ""
    echo "Error output:"
    cat /tmp/tagmypdf_output.log
fi

cd - > /dev/null

echo ""
echo "Test directory: $TEST_DIR"
echo "To clean up: rm -rf $TEST_DIR"
