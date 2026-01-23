import streamlit as st
import os
import subprocess
import time
import shutil
import zipfile
from pathlib import Path
import tempfile
import threading
import glob
from datetime import datetime

st.set_page_config(page_title="Accessible PDF Tagger", page_icon="📄", layout="wide")

st.title("📄 Accessible PDF Tagger")
st.write("Upload a ZIP file containing up to 100 PDFs to automatically tag them for accessibility compliance")

# Configuration
MAX_FILES = 100
TIMEOUT_PER_FILE = 900  # 15 minutes for OCR-heavy PDFs
CLEANUP_INTERVAL = 3600  # Run cleanup every hour (3600 seconds)
SESSION_MAX_AGE = 7200  # Delete sessions older than 2 hours (7200 seconds)


# ═══════════════════════════════════════════════════════════════════
# BUILT-IN CLEANUP SCHEDULER (No cron needed!)
# ═══════════════════════════════════════════════════════════════════
def is_pdf_already_tagged(pdf_path):
    """
    Check if PDF already has accessibility tags.
    Returns True if tagged, False if needs tagging.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)

        # Method 1: Check PyMuPDF's built-in flag
        if hasattr(doc, 'is_tagged') and doc.is_tagged:
            doc.close()
            return True

        # Method 2: Check for structure tree using pikepdf (more reliable)
        try:
            import pikepdf
            pdf = pikepdf.open(pdf_path)

            # Check if the PDF has a StructTreeRoot in the catalog
            catalog = pdf.Root
            has_struct_tree = '/StructTreeRoot' in catalog

            pdf.close()
            doc.close()

            if has_struct_tree:
                return True
        except Exception as e:
            print(f"Warning: Could not check with pikepdf: {e}")

        # Method 3: Check for marked content in the page stream
        # This detects tags even if StructTreeRoot is missing
        try:
            for page in doc:
                text = page.get_text("dict")
                # If page has any marked content operators, it's likely tagged
                page_content = page.read_contents().decode('latin-1', errors='ignore')
                if '/Artifact' in page_content or '/P' in page_content or '/Span' in page_content:
                    doc.close()
                    return True
        except Exception:
            pass

        doc.close()
        return False

    except Exception as e:
        # If we can't check, process it anyway to be safe
        print(f"Warning: Could not check if {pdf_path} is tagged: {e}")
        return False


def cleanup_old_sessions():
    """
    Background cleanup function that runs periodically.
    Removes session directories older than SESSION_MAX_AGE seconds.
    """
    while True:
        try:
            current_time = time.time()
            cleanup_count = 0

            # Find all pdf_batch directories
            session_dirs = glob.glob("/tmp/pdf_batch_*")

            for session_dir in session_dirs:
                try:
                    # Check directory age
                    dir_mtime = os.path.getmtime(session_dir)
                    age_seconds = current_time - dir_mtime

                    # If older than SESSION_MAX_AGE, delete it
                    if age_seconds > SESSION_MAX_AGE:
                        shutil.rmtree(session_dir, ignore_errors=True)
                        cleanup_count += 1
                        print(f"[CLEANUP] Removed old session: {session_dir} (age: {age_seconds / 60:.1f} min)")
                except Exception as e:
                    print(f"[CLEANUP] Error removing {session_dir}: {e}")

            if cleanup_count > 0:
                print(f"[CLEANUP] {datetime.now()}: Cleaned {cleanup_count} old session(s)")

        except Exception as e:
            print(f"[CLEANUP] Background cleanup error: {e}")

        # Sleep until next cleanup
        time.sleep(CLEANUP_INTERVAL)


# Start background cleanup thread (runs automatically when app starts)
if 'cleanup_thread_started' not in st.session_state:
    cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
    cleanup_thread.start()
    st.session_state.cleanup_thread_started = True
    print(f"[CLEANUP] Background cleanup thread started (runs every {CLEANUP_INTERVAL / 60:.0f} minutes)")

# ═══════════════════════════════════════════════════════════════════
# MULTI-USER SAFETY: Generate unique session ID for each user
# ═══════════════════════════════════════════════════════════════════
if 'session_id' not in st.session_state:
    import uuid

    st.session_state.session_id = str(uuid.uuid4())[:8]

SESSION_ID = st.session_state.session_id

# Sidebar configuration
keep_intermediate = False
show_terminal_output = False

# Optional: Add sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    show_terminal_output = st.checkbox("Show detailed errors", value=False)

# File uploader - Support both single PDF and ZIP
upload_type = st.radio("Upload type:", ["Single PDF", "ZIP file (batch)"], horizontal=True)

if upload_type == "Single PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
else:
    uploaded_file = st.file_uploader("Upload a ZIP file with PDFs", type="zip")


def process_single_pdf(pdf_path, output_dir, work_dir):
    """
    Process a single PDF file.
    """
    pdf_name = os.path.basename(pdf_path)

    # ✅ CHECK IF ALREADY TAGGED - SKIP IF YES
    if is_pdf_already_tagged(pdf_path):
        # Copy original to output (no processing needed)
        output_filename = pdf_name  # Keep original filename
        final_output = os.path.join(output_dir, output_filename)
        shutil.copy2(pdf_path, final_output)

        return {
            "status": "skipped",
            "name": pdf_name,
            "output": output_filename,
            "message": "Already tagged ✓"
        }

    try:
        # Clean work directory
        for folder in ["debug", "fused_tags", "layout_out", "lines_out", "structures"]:
            folder_path = os.path.join(work_dir, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path, ignore_errors=True)
            os.makedirs(folder_path, exist_ok=True)

        # Remove old output files
        output_pdf = os.path.join(work_dir, "output.pdf")
        if os.path.exists(output_pdf):
            os.remove(output_pdf)

        temp_pdf = os.path.join(work_dir, "temp.pdf")
        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)

        # Copy PDF to work directory
        shutil.copy2(pdf_path, temp_pdf)

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TagMyPDF2.py")

        result = subprocess.run(
            ["python", "-u", script_path, "temp.pdf"],
            env=env,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_PER_FILE
        )

        if result.returncode == 0 and os.path.exists(output_pdf):
            output_filename = pdf_name  # Keep original filename
            final_output = os.path.join(output_dir, output_filename)
            shutil.copy2(output_pdf, final_output)
            return {"status": "success", "name": pdf_name, "output": output_filename}
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            return {"status": "failed", "name": pdf_name, "error": f"Exit code: {result.returncode}",
                    "details": error_msg}

    except subprocess.TimeoutExpired:
        return {"status": "failed", "name": pdf_name, "error": f"Timeout (>{TIMEOUT_PER_FILE // 60} minutes)"}

    except Exception as e:
        return {"status": "failed", "name": pdf_name, "error": str(e)}


if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"📦 Uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
    st.caption(f"Session ID: {SESSION_ID}")

    if st.button("🚀 Process PDF" if upload_type == "Single PDF" else "🚀 Run Batch Tagging Pipeline", type="primary"):
        temp_dir = tempfile.mkdtemp(prefix=f"pdf_batch_{SESSION_ID}_")
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        work_dir = os.path.join(temp_dir, "worker")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)

        # Create work subdirectories
        for folder in ["debug", "fused_tags", "layout_out", "lines_out", "structures"]:
            os.makedirs(os.path.join(work_dir, folder), exist_ok=True)

        try:
            # Handle single PDF upload
            if upload_type == "Single PDF":
                pdf_path = os.path.join(input_dir, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())
                pdf_files = [pdf_path]

            # Handle ZIP upload
            else:
                st.info("📂 Extracting ZIP file...")
                zip_path = os.path.join(temp_dir, "upload.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.read())

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(input_dir)

                pdf_files = []
                for root, dirs, files in os.walk(input_dir):
                    for file in files:
                        if file.lower().endswith('.pdf') and not file.startswith('.'):
                            pdf_files.append(os.path.join(root, file))

            pdf_count = len(pdf_files)

            if pdf_count == 0:
                st.error("❌ No PDF files found in the ZIP archive")
            elif pdf_count > MAX_FILES:
                st.error(f"❌ Too many PDFs! Found {pdf_count} files, but maximum is {MAX_FILES}")
            else:
                st.success(f"✅ Found {pdf_count} PDF file(s) to process")

                progress_bar = st.progress(0)
                status_text = st.empty()
                results_placeholder = st.empty()

                results = []
                start_time = time.time()

                # Process PDFs sequentially
                for idx, pdf_path in enumerate(pdf_files, 1):
                    pdf_name = os.path.basename(pdf_path)
                    status_text.info(f"🔄 Processing {idx}/{pdf_count}: {pdf_name}")

                    result = process_single_pdf(pdf_path, output_dir, work_dir)
                    results.append(result)

                    progress_bar.progress(idx / pdf_count)

                    successful = [r for r in results if r["status"] == "success"]
                    skipped = [r for r in results if r["status"] == "skipped"]
                    failed = [r for r in results if r["status"] == "failed"]

                    with results_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total", pdf_count)
                        with col2:
                            st.metric("✅ Tagged", len(successful))
                        with col3:
                            st.metric("⏭️ Skipped", len(skipped))
                        with col4:
                            st.metric("❌ Failed", len(failed))

                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)

                successful = [r for r in results if r["status"] == "success"]
                skipped = [r for r in results if r["status"] == "skipped"]
                failed = [r for r in results if r["status"] == "failed"]

                status_text.success(f"✅ Processing complete! (took {minutes}m {seconds}s)")

                st.write("---")
                st.subheader("📊 Processing Summary")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Files", pdf_count)
                with col2:
                    st.metric("✅ Tagged", len(successful))
                with col3:
                    st.metric("⏭️ Skipped", len(skipped))
                with col4:
                    st.metric("❌ Failed", len(failed))

                if successful:
                    st.success(f"Successfully tagged {len(successful)} PDF(s)")
                    with st.expander("📄 View successful files", expanded=False):
                        for r in successful:
                            st.write(f"✓ {r['name']}")

                if skipped:
                    st.info(f"Skipped {len(skipped)} already-tagged PDF(s)")
                    with st.expander("⏭️ View skipped files", expanded=False):
                        for r in skipped:
                            st.write(f"⏭️ {r['name']}")

                if failed:
                    st.warning(f"Failed to tag {len(failed)} PDF(s)")
                    with st.expander("⚠️ View failed files", expanded=True):
                        for r in failed:
                            st.write(f"✗ **{r['name']}**")
                            st.caption(f"   Error: {r['error']}")
                            if 'details' in r and show_terminal_output:
                                with st.expander(f"Details for {r['name']}"):
                                    st.code(r['details'], language='text')

                if successful or skipped:
                    st.write("---")

                    # Single file: Direct download
                    if upload_type == "Single PDF" and len(successful + skipped) == 1:
                        result = (successful + skipped)[0]
                        file_path = os.path.join(output_dir, result['output'])

                        with open(file_path, "rb") as f:
                            pdf_bytes = f.read()

                        st.success("🎉 Your accessible PDF is ready!")

                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.download_button(
                                label="⬇️ Download Tagged PDF",
                                data=pdf_bytes,
                                file_name=result['output'],
                                mime="application/pdf",
                                use_container_width=True
                            )
                        with col2:
                            st.info(f"📄 File: {len(pdf_bytes) / (1024 * 1024):.2f} MB | ⏱️ Time: {minutes}m {seconds}s")

                    # Multiple files: ZIP download
                    else:
                        st.info("📦 Creating download package...")

                        output_zip_path = os.path.join(temp_dir, "tagged_pdfs.zip")
                        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            # Include both successful and skipped files
                            for r in successful + skipped:
                                file_path = os.path.join(output_dir, r['output'])
                                if os.path.exists(file_path):
                                    zipf.write(file_path, r['output'])

                        with open(output_zip_path, "rb") as f:
                            zip_bytes = f.read()

                        st.success("🎉 Your accessible PDFs are ready!")

                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.download_button(
                                label="⬇️ Download Tagged PDFs (ZIP)",
                                data=zip_bytes,
                                file_name="tagged_pdfs.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                        with col2:
                            st.info(
                                f"📦 Package: {len(zip_bytes) / (1024 * 1024):.2f} MB | ⏱️ Total time: {minutes}m {seconds}s")
                else:
                    st.error("❌ No files were successfully processed. Please check the errors above.")

        except zipfile.BadZipFile:
            st.error("❌ Invalid ZIP file. Please upload a valid ZIP archive.")

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            if show_terminal_output:
                st.exception(e)

        finally:
            if not keep_intermediate:
                try:
                    if os.path.exists(temp_dir):
                        time.sleep(0.5)
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as cleanup_error:
                    if show_terminal_output:
                        st.warning(f"Cleanup warning: {cleanup_error}")

# Instructions
st.write("---")
st.subheader("📋 How to Use")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **📤 Upload**
    1. Create a ZIP file with your PDFs
    2. Maximum 200 MB files per batch
    3. Files can be in subfolders

    **⚙️ Features**
    - Sequential processing (one at a time)
    - Auto-skip already-tagged PDFs
    - Automatic language detection (English/Spanish)
    - Safe for multiple users
    - Built-in automatic cleanup
    """)

with col2:
    st.markdown("""
    **▶️ Process**
    - Click "Run Batch Tagging Pipeline"
    - Monitor progress in real-time
    - View success/failure summary

    **📥 Download**
    - Download ZIP with tagged PDFs
    - Includes both new and skipped files
    - Old sessions cleaned automatically
    - No manual maintenance needed
    """)

st.write("---")
