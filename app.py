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
import logging
import sys
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# 🔧 LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/pdf_tagger.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("🚀 PDF TAGGER APPLICATION STARTING")
logger.info("=" * 80)

st.set_page_config(page_title="Accessible PDF Tagger", page_icon="📄", layout="wide")

st.title("📄 Accessible PDF Tagger")
st.write("Upload a ZIP file containing up to 100 PDFs to automatically tag them for accessibility compliance")

# Configuration
MAX_FILES = 100
TIMEOUT_PER_FILE = 900  # 15 minutes for OCR-heavy PDFs
CLEANUP_INTERVAL = 3600
SESSION_MAX_AGE = 7200

logger.info(f"Configuration loaded: MAX_FILES={MAX_FILES}, TIMEOUT={TIMEOUT_PER_FILE}s")


def is_pdf_already_tagged(pdf_path):
    """Check if PDF already has accessibility tags."""
    logger.info(f"Checking if PDF is tagged: {pdf_path}")
    try:
        import pikepdf
        pdf = pikepdf.open(pdf_path)
        catalog = pdf.Root
        has_struct_tree = '/StructTreeRoot' in catalog
        pdf.close()
        logger.info(f"PDF {os.path.basename(pdf_path)} tagged status: {has_struct_tree}")
        return has_struct_tree
    except Exception as e:
        logger.error(f"Error checking if {pdf_path} is tagged: {e}")
        return False


def untag_pdf(pdf_path):
    """
    Remove accessibility tags from a PDF in-place.
    Returns True if successful, False otherwise.
    """
    logger.info(f"🔄 Untagging PDF: {pdf_path}")
    try:
        import pikepdf
        from pikepdf import Name

        # Create a temporary file for the untagged version
        temp_path = pdf_path + ".temp"

        pdf = pikepdf.open(pdf_path)

        # Remove StructTreeRoot
        if Name.StructTreeRoot in pdf.Root:
            del pdf.Root[Name.StructTreeRoot]
            logger.info("  - Removed StructTreeRoot")

        # Remove MarkInfo
        if Name.MarkInfo in pdf.Root:
            del pdf.Root[Name.MarkInfo]
            logger.info("  - Removed MarkInfo")

        # Remove ParentTree references from pages
        page_count = 0
        for page in pdf.pages:
            page_obj = page.obj
            if Name.StructParents in page_obj:
                del page_obj[Name.StructParents]
                page_count += 1
        logger.info(f"  - Removed StructParents from {page_count} pages")

        # Save to temp file
        pdf.save(temp_path, linearize=True)
        pdf.close()

        # Replace original with untagged version
        shutil.move(temp_path, pdf_path)

        logger.info(f"✅ Successfully untagged: {os.path.basename(pdf_path)}")
        return True

    except Exception as e:
        logger.error(f"❌ Error untagging PDF {pdf_path}: {e}", exc_info=True)
        # Clean up temp file if it exists
        temp_path = pdf_path + ".temp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def cleanup_old_sessions():
    """Background cleanup function that runs periodically."""
    logger.info("🧹 Cleanup thread started")
    while True:
        try:
            current_time = time.time()
            cleanup_count = 0
            session_dirs = glob.glob("/tmp/pdf_batch_*")

            logger.debug(f"Found {len(session_dirs)} session directories to check")

            for session_dir in session_dirs:
                try:
                    dir_mtime = os.path.getmtime(session_dir)
                    age_seconds = current_time - dir_mtime

                    if age_seconds > SESSION_MAX_AGE:
                        shutil.rmtree(session_dir, ignore_errors=True)
                        cleanup_count += 1
                        logger.info(f"Cleaned up old session: {session_dir} (age: {age_seconds / 60:.1f} min)")
                except Exception as e:
                    logger.error(f"Error cleaning {session_dir}: {e}")

            if cleanup_count > 0:
                logger.info(f"[{datetime.now()}] Cleaned up {cleanup_count} old session(s)")

        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)

        time.sleep(CLEANUP_INTERVAL)


# Start background cleanup thread
if not any(thread.name == "CleanupThread" for thread in threading.enumerate()):
    cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True, name="CleanupThread")
    cleanup_thread.start()
    logger.info("✅ Background cleanup thread started")

# Generate unique session ID
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

SESSION_ID = st.session_state.session_id
logger.info(f"Session ID: {SESSION_ID}")

# ═══════════════════════════════════════════════════════════════════
# UI: UPLOAD OPTIONS
# ═══════════════════════════════════════════════════════════════════
upload_type = st.radio(
    "Upload Type",
    options=["Single PDF", "ZIP (Batch Processing)"],
    horizontal=True,
    help="Upload a single PDF or a ZIP file containing multiple PDFs"
)

uploaded_file = st.file_uploader(
    "Choose file",
    type=["pdf", "zip"] if upload_type == "ZIP (Batch Processing)" else ["pdf"],
    help="Maximum 100 PDFs per batch"
)

# ═══════════════════════════════════════════════════════════════════
# ✨ PROCESSING OPTIONS
# ═══════════════════════════════════════════════════════════════════
st.write("---")
st.subheader("⚙️ Processing Options")

col1, col2 = st.columns(2)

with col1:
    force_retag = st.checkbox(
        "🔄 Force Retag Already-Tagged PDFs",
        value=False,
        help="If checked, PDFs that are already tagged will be untagged first, then retagged with fresh tags. Use this if you want to update existing tags or if previous tagging had errors."
    )

with col2:
    show_terminal_output = st.checkbox(
        "🖥️ Show Terminal Output",
        value=False,
        help="Display detailed processing logs for debugging"
    )

# Show info about what force retag does
if force_retag:
    st.info(
        "ℹ️ **Force Retag Mode Enabled**: Already-tagged PDFs will be untagged first, then retagged. This ensures fresh, up-to-date accessibility tags.")
else:
    st.info("ℹ️ **Smart Skip Mode**: Already-tagged PDFs will be skipped automatically to save processing time.")


def process_single_pdf(pdf_path, output_dir, work_dir, force_retag=False):
    """
    Process a single PDF file through the tagging pipeline.

    Args:
        pdf_path: Path to input PDF
        output_dir: Directory for final tagged output
        work_dir: Working directory for intermediate files
        force_retag: If True, untag already-tagged PDFs before retagging
    """
    pdf_name = os.path.basename(pdf_path)
    logger.info("=" * 60)
    logger.info(f"📄 PROCESSING: {pdf_name}")
    logger.info("=" * 60)

    try:
        # Check if PDF is already tagged
        is_tagged = is_pdf_already_tagged(pdf_path)

        if is_tagged:
            if force_retag:
                # User wants to retag - untag it first
                logger.info(f"🔄 {pdf_name}: Already tagged - untagging first...")
                st.info(f"🔄 {pdf_name}: Already tagged - untagging first...")

                if not untag_pdf(pdf_path):
                    logger.error(f"❌ Failed to untag {pdf_name}")
                    return {"status": "failed", "name": pdf_name, "error": "Failed to untag PDF"}

                logger.info(f"✅ {pdf_name}: Untagged successfully, now retagging...")
                st.success(f"✅ {pdf_name}: Untagged successfully, now retagging...")
            else:
                # Skip already-tagged PDFs
                logger.info(f"⏭️ Skipping {pdf_name}: Already tagged")
                return {"status": "skipped", "name": pdf_name, "reason": "Already tagged"}

        # ✅ Clean work directory
        logger.info("🧹 Cleaning work directory...")
        for folder in ["debug", "fused_tags", "layout_out", "lines_out", "structures"]:
            folder_path = os.path.join(work_dir, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path, ignore_errors=True)
            os.makedirs(folder_path, exist_ok=True)
        logger.info("✅ Work directory cleaned")

        # ✅ Copy PDF to work directory as temp.pdf
        temp_pdf = os.path.join(work_dir, "temp.pdf")
        output_pdf = os.path.join(work_dir, "output.pdf")

        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)
        if os.path.exists(output_pdf):
            os.remove(output_pdf)

        logger.info(f"📋 Copying {pdf_name} to work directory as temp.pdf...")
        shutil.copy2(pdf_path, temp_pdf)
        logger.info(f"✅ Copied successfully (size: {os.path.getsize(temp_pdf)} bytes)")

        # ✅ Get TagMyPDF2.py script path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tagmypdf_script = os.path.join(script_dir, "TagMyPDF2.py")

        logger.info(f"🔍 Script location: {tagmypdf_script}")
        logger.info(f"🔍 Script exists: {os.path.exists(tagmypdf_script)}")
        logger.info(f"🔍 Work directory: {work_dir}")
        logger.info(f"🔍 temp.pdf exists: {os.path.exists(temp_pdf)}")

        # ✅ Run TagMyPDF2.py
        logger.info("🚀 Starting TagMyPDF2.py processing...")

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        cmd = ["python", "-u", tagmypdf_script, "temp.pdf"]
        logger.info(f"Command: {' '.join(cmd)}")

        start_time = time.time()

        result = subprocess.run(
            cmd,
            env=env,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_PER_FILE
        )

        elapsed = time.time() - start_time
        logger.info(f"⏱️ TagMyPDF2.py completed in {elapsed:.2f}s")
        logger.info(f"Return code: {result.returncode}")

        if result.stdout:
            logger.info("STDOUT:")
            for line in result.stdout.split('\n')[:50]:  # First 50 lines
                if line.strip():
                    logger.info(f"  {line}")

        if result.stderr:
            logger.warning("STDERR:")
            for line in result.stderr.split('\n')[:50]:  # First 50 lines
                if line.strip():
                    logger.warning(f"  {line}")

        # ✅ Check if output.pdf was created
        logger.info(f"🔍 Checking for output.pdf at: {output_pdf}")
        logger.info(f"🔍 output.pdf exists: {os.path.exists(output_pdf)}")

        if os.path.exists(output_pdf):
            logger.info(f"✅ output.pdf found (size: {os.path.getsize(output_pdf)} bytes)")

        if result.returncode == 0 and os.path.exists(output_pdf):
            output_filename = pdf_name
            final_output = os.path.join(output_dir, output_filename)
            shutil.copy2(output_pdf, final_output)

            logger.info(f"✅ SUCCESS: {pdf_name} tagged successfully")
            logger.info(f"📦 Output saved to: {final_output}")

            status_msg = "retagged" if is_tagged else "tagged"
            return {"status": "success", "name": pdf_name, "output": output_filename, "was_retagged": is_tagged}
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            logger.error(f"❌ FAILED: {pdf_name} - Exit code: {result.returncode}")
            logger.error(f"Error message: {error_msg}")
            return {"status": "failed", "name": pdf_name, "error": f"Exit code: {result.returncode}",
                    "details": error_msg}

    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ TIMEOUT: {pdf_name} exceeded {TIMEOUT_PER_FILE}s")
        return {"status": "failed", "name": pdf_name, "error": f"Timeout (>{TIMEOUT_PER_FILE // 60} minutes)"}

    except Exception as e:
        logger.error(f"💥 EXCEPTION: {pdf_name}", exc_info=True)
        return {"status": "failed", "name": pdf_name, "error": str(e)}


if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"📦 Uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
    st.caption(f"Session ID: {SESSION_ID}")

    logger.info(f"📤 File uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")

    button_label = "🚀 Process PDF" if upload_type == "Single PDF" else "🚀 Run Batch Tagging Pipeline"

    if st.button(button_label, type="primary"):
        logger.info("🎯 Processing button clicked!")
        logger.info(f"Upload type: {upload_type}")
        logger.info(f"Force retag: {force_retag}")

        temp_dir = tempfile.mkdtemp(prefix=f"pdf_batch_{SESSION_ID}_")
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        work_dir = os.path.join(temp_dir, "worker")

        logger.info(f"📁 Created temp directories:")
        logger.info(f"  temp_dir: {temp_dir}")
        logger.info(f"  input_dir: {input_dir}")
        logger.info(f"  output_dir: {output_dir}")
        logger.info(f"  work_dir: {work_dir}")

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)

        # Create work subdirectories
        for folder in ["debug", "fused_tags", "layout_out", "lines_out", "structures"]:
            os.makedirs(os.path.join(work_dir, folder), exist_ok=True)

        try:
            # Handle single PDF upload
            if upload_type == "Single PDF":
                logger.info("📄 Processing single PDF upload")
                pdf_path = os.path.join(input_dir, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())
                logger.info(f"✅ Saved to: {pdf_path}")
                pdf_files = [pdf_path]

            # Handle ZIP upload
            else:
                logger.info("📦 Processing ZIP upload")
                st.info("📂 Extracting ZIP file...")
                zip_path = os.path.join(temp_dir, "upload.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.read())
                logger.info(f"✅ ZIP saved to: {zip_path}")

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(input_dir)
                logger.info("✅ ZIP extracted")

                pdf_files = []
                for root, dirs, files in os.walk(input_dir):
                    for file in files:
                        if file.lower().endswith('.pdf') and not file.startswith('.'):
                            pdf_files.append(os.path.join(root, file))
                logger.info(f"Found {len(pdf_files)} PDF files in ZIP")

            pdf_count = len(pdf_files)

            if pdf_count == 0:
                logger.error("No PDF files found!")
                st.error("❌ No PDF files found in the ZIP archive")
            elif pdf_count > MAX_FILES:
                logger.error(f"Too many PDFs: {pdf_count} > {MAX_FILES}")
                st.error(f"❌ Too many PDFs! Found {pdf_count} files, but maximum is {MAX_FILES}")
            else:
                logger.info(f"✅ Processing {pdf_count} PDF file(s)")
                st.success(f"✅ Found {pdf_count} PDF file(s) to process")

                if force_retag:
                    logger.warning("⚠️ Force Retag Mode enabled")
                    st.warning("⚠️ Force Retag Mode: All PDFs will be retagged, even if already tagged")

                progress_bar = st.progress(0)
                status_text = st.empty()
                results_placeholder = st.empty()

                results = []
                start_time = time.time()

                # Process PDFs sequentially
                for idx, pdf_path in enumerate(pdf_files, 1):
                    pdf_name = os.path.basename(pdf_path)
                    logger.info(f"\n{'=' * 80}")
                    logger.info(f"Processing file {idx}/{pdf_count}: {pdf_name}")
                    logger.info(f"{'=' * 80}")
                    status_text.info(f"🔄 Processing {idx}/{pdf_count}: {pdf_name}")

                    # Pass force_retag flag to processing function
                    result = process_single_pdf(pdf_path, output_dir, work_dir, force_retag=force_retag)
                    results.append(result)

                    logger.info(f"Result: {result}")

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

                logger.info(f"\n{'=' * 80}")
                logger.info(f"PROCESSING COMPLETE")
                logger.info(f"Total time: {minutes}m {seconds}s")
                logger.info(f"Success: {len(successful)}, Skipped: {len(skipped)}, Failed: {len(failed)}")
                logger.info(f"{'=' * 80}\n")

                successful = [r for r in results if r["status"] == "success"]
                retagged = [r for r in successful if r.get("was_retagged", False)]
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
                    if retagged:
                        st.caption(f"({len(retagged)} retagged)")
                with col3:
                    st.metric("⏭️ Skipped", len(skipped))
                with col4:
                    st.metric("❌ Failed", len(failed))

                if successful:
                    st.success(f"Successfully tagged {len(successful)} PDF(s)")
                    if retagged:
                        st.info(f"♻️ {len(retagged)} PDF(s) were untagged and retagged")

                    with st.expander("📄 View successful files", expanded=False):
                        for r in successful:
                            icon = "♻️" if r.get("was_retagged", False) else "✓"
                            st.write(f"{icon} {r['name']}")

                if skipped:
                    st.info(f"Skipped {len(skipped)} already-tagged PDF(s)")
                    st.caption("💡 Enable 'Force Retag' to retag these files")
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

                # Download buttons
                st.write("---")
                st.subheader("📥 Download Results")

               # output_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]

                output_files = [f for f in os.listdir(output_dir) if Path(f).suffix.lower() == ".pdf"]

                logger.info(f"Output files ready: {output_files}")

                if len(output_files) == 1:
                    # Single file - direct download
                    output_file = output_files[0]
                    with open(os.path.join(output_dir, output_file), "rb") as f:
                        st.download_button(
                            label=f"📄 Download {output_file}",
                            data=f.read(),
                            file_name=output_file,
                            mime="application/pdf"
                        )

                elif len(output_files) > 1:
                    # Multiple files - create ZIP
                    zip_name = f"tagged_pdfs_{SESSION_ID}.zip"
                    zip_path = os.path.join(temp_dir, zip_name)

                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for pdf_file in output_files:
                            pdf_path = os.path.join(output_dir, pdf_file)
                            zipf.write(pdf_path, pdf_file)

                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label=f"📦 Download All ({len(output_files)} files)",
                            data=f.read(),
                            file_name=zip_name,
                            mime="application/zip"
                        )

        except Exception as e:
            logger.error(f"💥 CRITICAL ERROR", exc_info=True)
            st.error(f"❌ Unexpected error: {e}")
            if show_terminal_output:
                st.exception(e)

        finally:
            # Cleanup will happen automatically via background thread
            logger.info("Cleanup will be handled by background thread")

# Footer
st.write("---")
st.caption("💡 Tip: Already-tagged PDFs are automatically skipped unless you enable 'Force Retag'")
st.caption("🔧 Session cleanup happens automatically every hour")
st.caption("📋 View logs: docker logs -f pdf-tagger")

logger.info("Application ready and waiting for user input...")