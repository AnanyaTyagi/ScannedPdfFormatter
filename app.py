import streamlit as st
import os
import subprocess
import time
import shutil
import zipfile
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import glob
from datetime import datetime

st.set_page_config(page_title="Accessible PDF Tagger", page_icon="📄", layout="wide")

st.title("📄 Accessible PDF Tagger")
st.write("Upload a ZIP file containing up to 100 PDFs to automatically tag them for accessibility compliance")

# Configuration
MAX_FILES = 100
TIMEOUT_PER_FILE = 600  # 10 minutes
CLEANUP_INTERVAL = 3600  # Run cleanup every hour (3600 seconds)
SESSION_MAX_AGE = 7200  # Delete sessions older than 2 hours (7200 seconds)


# ═══════════════════════════════════════════════════════════════════
# BUILT-IN CLEANUP SCHEDULER (No cron needed!)
# ═══════════════════════════════════════════════════════════════════

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
parallel_processing = False
max_workers = 1
keep_intermediate = False
show_terminal_output = True

# Uncomment to enable parallel processing option:
# with st.sidebar:
#     st.header("⚙️ Settings")
#     parallel_processing = st.checkbox("Enable parallel processing", value=False)
#     if parallel_processing:
#         max_workers = st.slider("Parallel workers", min_value=2, max_value=4, value=2)
#     keep_intermediate = st.checkbox("Keep intermediate files", value=False)
#     show_terminal_output = st.checkbox("Show terminal output", value=False)

# File uploader
uploaded_file = st.file_uploader("Upload a ZIP file with PDFs", type="zip")


class WorkerPool:
    """
    Manages a pool of reusable worker directories.
    """

    def __init__(self, base_dir, num_workers, session_id):
        self.base_dir = base_dir
        self.num_workers = num_workers
        self.session_id = session_id
        self.workers = []
        self.locks = []

        for i in range(num_workers):
            worker_dir = os.path.join(base_dir, f"worker_{i}")
            os.makedirs(worker_dir, exist_ok=True)
            self.workers.append(worker_dir)
            self.locks.append(threading.Lock())

        for worker_dir in self.workers:
            for folder in ["debug", "fused_tags", "layout_out", "lines_out", "structures"]:
                os.makedirs(os.path.join(worker_dir, folder), exist_ok=True)

    def get_worker(self):
        for i, lock in enumerate(self.locks):
            if lock.acquire(blocking=False):
                return i, self.workers[i], lock
        return None, None, None

    def clean_worker(self, worker_dir):
        try:
            output_pdf = os.path.join(worker_dir, "output.pdf")
            if os.path.exists(output_pdf):
                os.remove(output_pdf)

            temp_pdf = os.path.join(worker_dir, "temp.pdf")
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)

            for folder in ["debug", "fused_tags", "layout_out", "lines_out", "structures"]:
                folder_path = os.path.join(worker_dir, folder)
                if os.path.exists(folder_path):
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        try:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                        except Exception:
                            pass
        except Exception as e:
            print(f"Warning: Failed to clean worker directory: {e}")

    def cleanup_all(self):
        for worker_dir in self.workers:
            try:
                if os.path.exists(worker_dir):
                    shutil.rmtree(worker_dir, ignore_errors=True)
            except Exception:
                pass


def process_single_pdf(pdf_path, output_dir, worker_pool, worker_idx=None):
    pdf_name = os.path.basename(pdf_path)
    work_dir = None
    lock = None

    try:
        if worker_idx is not None:
            work_dir = worker_pool.workers[worker_idx % worker_pool.num_workers]
            lock = worker_pool.locks[worker_idx % worker_pool.num_workers]
            lock.acquire()
        else:
            idx, work_dir, lock = worker_pool.get_worker()
            if work_dir is None:
                return {"status": "failed", "name": pdf_name, "error": "No workers available"}

        worker_pool.clean_worker(work_dir)

        temp_pdf = os.path.join(work_dir, "temp.pdf")
        shutil.copy2(pdf_path, temp_pdf)

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        # Get absolute path to TagMyPDF2.py (in project root)
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TagMyPDF2.py")

        result = subprocess.run(
            ["python", "-u", script_path, "temp.pdf"],
            env=env,
            cwd=work_dir,
            text=True,
            timeout=TIMEOUT_PER_FILE
        )
        # result = subprocess.run(
        #     ["python", "-u", "TagMyPDF2.py", "temp.pdf"],
        #     env=env,
        #     cwd=work_dir,
        #     capture_output=True,
        #     text=True,
        #     timeout=TIMEOUT_PER_FILE
        # )

        output_pdf = os.path.join(work_dir, "output.pdf")
        if result.returncode == 0 and os.path.exists(output_pdf):
            output_filename = f"tagged_{pdf_name}"
            final_output = os.path.join(output_dir, output_filename)
            shutil.copy2(output_pdf, final_output)
            return {"status": "success", "name": pdf_name, "output": output_filename}
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            return {"status": "failed", "name": pdf_name, "error": f"Exit code: {result.returncode}",
                    "details": error_msg}

    except subprocess.TimeoutExpired:
        return {"status": "failed", "name": pdf_name, "error": "Timeout (>10 minutes)"}

    except Exception as e:
        return {"status": "failed", "name": pdf_name, "error": str(e)}

    finally:
        if lock is not None:
            try:
                lock.release()
            except:
                pass


if uploaded_file:
    st.info(f"📦 Uploaded: {uploaded_file.name} ({uploaded_file.size / (1024 * 1024):.2f} MB)")
    st.caption(f"Session ID: {SESSION_ID}")

    if st.button("🚀 Run Batch Tagging Pipeline", type="primary"):
        temp_dir = tempfile.mkdtemp(prefix=f"pdf_batch_{SESSION_ID}_")
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        work_base_dir = os.path.join(temp_dir, "workers")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(work_base_dir, exist_ok=True)

        worker_pool = None

        try:
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

                num_workers = max_workers if parallel_processing else 1
                worker_pool = WorkerPool(work_base_dir, num_workers, SESSION_ID)
                st.info(f"🔧 Initialized {num_workers} worker director{'ies' if num_workers > 1 else 'y'}")

                progress_bar = st.progress(0)
                status_text = st.empty()
                results_placeholder = st.empty()

                results = []
                completed = 0

                start_time = time.time()

                if parallel_processing:
                    status_text.info(f"🔄 Processing {pdf_count} PDFs with {num_workers} parallel workers...")

                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        future_to_pdf = {}
                        for idx, pdf_path in enumerate(pdf_files):
                            future = executor.submit(process_single_pdf, pdf_path, output_dir, worker_pool, None)
                            future_to_pdf[future] = os.path.basename(pdf_path)

                        for future in as_completed(future_to_pdf):
                            pdf_name = future_to_pdf[future]
                            result = future.result()
                            results.append(result)
                            completed += 1

                            progress_bar.progress(completed / pdf_count)
                            status_text.info(f"🔄 Processing... ({completed}/{pdf_count} complete)")

                            successful = [r for r in results if r["status"] == "success"]
                            failed = [r for r in results if r["status"] == "failed"]

                            with results_placeholder.container():
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total", pdf_count)
                                with col2:
                                    st.metric("✅ Success", len(successful))
                                with col3:
                                    st.metric("❌ Failed", len(failed))

                else:
                    for idx, pdf_path in enumerate(pdf_files, 1):
                        pdf_name = os.path.basename(pdf_path)
                        status_text.info(f"🔄 Processing {idx}/{pdf_count}: {pdf_name}")

                        result = process_single_pdf(pdf_path, output_dir, worker_pool, idx)
                        results.append(result)

                        progress_bar.progress(idx / pdf_count)

                        successful = [r for r in results if r["status"] == "success"]
                        failed = [r for r in results if r["status"] == "failed"]

                        with results_placeholder.container():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total", pdf_count)
                            with col2:
                                st.metric("✅ Success", len(successful))
                            with col3:
                                st.metric("❌ Failed", len(failed))

                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)

                successful = [r for r in results if r["status"] == "success"]
                failed = [r for r in results if r["status"] == "failed"]

                status_text.success(f"✅ Processing complete! (took {minutes}m {seconds}s)")

                st.write("---")
                st.subheader("📊 Processing Summary")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Files", pdf_count)
                with col2:
                    st.metric("✅ Successful", len(successful))
                with col3:
                    st.metric("❌ Failed", len(failed))
                with col4:
                    st.metric("⚡ Avg Time", f"{elapsed_time / pdf_count:.1f}s")

                if successful:
                    st.success(f"Successfully tagged {len(successful)} PDF(s)")
                    with st.expander("📄 View successful files", expanded=False):
                        for r in successful:
                            st.write(f"✓ {r['name']}")

                if failed:
                    st.warning(f"Failed to tag {len(failed)} PDF(s)")
                    with st.expander("⚠️ View failed files", expanded=True):
                        for r in failed:
                            st.write(f"✗ **{r['name']}**")
                            st.caption(f"   Error: {r['error']}")
                            if 'details' in r and show_terminal_output:
                                with st.expander(f"Details for {r['name']}"):
                                    st.code(r['details'], language='text')

                if successful:
                    st.write("---")
                    st.info("📦 Creating download package...")

                    output_zip_path = os.path.join(temp_dir, "tagged_pdfs.zip")
                    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for r in successful:
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
            if worker_pool is not None:
                if not keep_intermediate:
                    st.info("🧹 Cleaning up worker directories...")
                    worker_pool.cleanup_all()
                else:
                    st.info(f"📁 Keeping intermediate files in: {work_base_dir}")

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
    2. Maximum 100 files per batch
    3. Files can be in subfolders

    **⚙️ Current Mode**
    - Sequential processing (1 file at a time)
    - Safe for multiple concurrent users
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
    - Old sessions cleaned automatically
    - No manual maintenance needed
    """)

st.write("---")
st.caption("Built with Streamlit • PDF/UA Accessibility Tagging • Self-Cleaning Architecture")