# # # save as app.py
# # import streamlit as st
# # import os, subprocess
# #
# # st.title("Accessible PDF Tagger")
# #
# # pdf = st.file_uploader("Upload a PDF", type="pdf")
# # if pdf:
# #     with open("temp.pdf", "wb") as f:
# #         f.write(pdf.read())
# #     if st.button("Run Tagging Pipeline"):
# #         result = subprocess.run(["python", "TagMyPDF.py", "temp.pdf"])
# #         if result.returncode == 0:
# #             st.success("✅ Tagged PDF ready!")
# #             st.download_button("Download output.pdf", open("out_tagged.pdf", "rb"), "application/pdf")
#
# # save as app.py
# # save as app.py
# # save as app.py
# save as app.py
# import streamlit as st
# import os
# import subprocess
# import time
# import shutil
#
# st.title("📄 Accessible PDF Tagger")
# st.write("Upload a PDF to automatically tag it for accessibility compliance")
#
# # File uploader
# pdf = st.file_uploader("Upload a PDF", type="pdf")
#
# if pdf:
#     # Display uploaded file info
#     st.info(f"📎 Uploaded: {pdf.name} ({pdf.size / 1024:.1f} KB)")
#
#     # Save the uploaded file
#     with open("temp.pdf", "wb") as f:
#         f.write(pdf.read())
#
#     # Run tagging button
#     if st.button("🚀 Run Tagging Pipeline", type="primary"):
#         # Create placeholders
#         status_placeholder = st.empty()
#         progress_bar = st.progress(0)
#
#         try:
#             # Clean up previous run folders
#             status_placeholder.info("🧹 Cleaning up previous outputs...")
#             folders_to_clean = [
#                 "debug",
#                 "fused_tags",
#                 "layout_out",
#                 "lines_out",
#                 "promoted_tags",
#                 "structures"
#             ]
#
#             import shutil
#
#             for folder in folders_to_clean:
#                 if os.path.exists(folder):
#                     try:
#                         shutil.rmtree(folder)
#                         print(f"Cleaned up: {folder}")
#                     except Exception as e:
#                         print(f"Warning: Could not remove {folder}: {e}")
#
#             # Also remove old output PDF if exists
#             if os.path.exists("out_tagged.pdf"):
#                 try:
#                     os.remove("out_tagged.pdf")
#                     print("Cleaned up: out_tagged.pdf")
#                 except Exception as e:
#                     print(f"Warning: Could not remove out_tagged.pdf: {e}")
#
#             time.sleep(0.3)  # Brief pause to show cleanup message
#             status_placeholder.info("⏳ Starting tagging pipeline...")
#
#             # Run with real-time terminal output (simple approach)
#             # This lets you see output in terminal immediately
#             env = os.environ.copy()
#             env['PYTHONUNBUFFERED'] = '1'
#
#             # Start the process
#             progress_bar.progress(20)
#             status_placeholder.info("🔄 Processing PDF (check terminal for detailed logs)...")
#
#             # Run the subprocess - output goes directly to terminal
#             result = subprocess.run(
#                 ["python", "-u", "TagMyPDF.py", "temp.pdf"],
#                 env=env,
#                 timeout=300  # 5 minute timeout
#             )
#
#             progress_bar.progress(90)
#
#             # Check results
#             if result.returncode == 0:
#                 status_placeholder.success("✅ PDF tagged successfully!")
#                 progress_bar.progress(100)
#
#                 # Check if output file exists
#                 output_file = "out_tagged.pdf"
#                 if os.path.exists(output_file):
#                     st.success("🎉 Your accessible PDF is ready!")
#
#                     # Read the output file
#                     with open(output_file, "rb") as f:
#                         pdf_bytes = f.read()
#
#                     # Download button
#                     st.download_button(
#                         label="⬇️ Download Tagged PDF",
#                         data=pdf_bytes,
#                         file_name=f"tagged_{pdf.name}",
#                         mime="application/pdf"
#                     )
#
#                     # Show output size
#                     st.info(f"Output file size: {len(pdf_bytes) / 1024:.1f} KB")
#
#                 else:
#                     st.error("❌ Output file not found. The tagging may have failed.")
#
#             else:
#                 status_placeholder.error(f"❌ Tagging failed with error code {result.returncode}")
#                 progress_bar.progress(0)
#                 st.error("Check your terminal for error details.")
#
#         except subprocess.TimeoutExpired:
#             status_placeholder.error("⏰ Processing timeout - the file may be too large")
#             progress_bar.progress(0)
#
#         except FileNotFoundError:
#             status_placeholder.error("❌ TagMyPDF.py script not found in the current directory")
#             progress_bar.progress(0)
#
#         except Exception as e:
#             status_placeholder.error(f"❌ An unexpected error occurred: {str(e)}")
#             progress_bar.progress(0)
#             st.exception(e)
#
#         finally:
#             # Cleanup temporary file
#             if os.path.exists("temp.pdf"):
#                 try:
#                     os.remove("temp.pdf")
#                 except:
#                     pass

# save as app.py - NO TIMEOUT VERSION for very large PDFs
import streamlit as st
import os
import subprocess
import time
import shutil

st.title("📄 Accessible PDF Tagger")
st.write("Upload a PDF to automatically tag it for accessibility compliance")

# File uploader
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    # Display uploaded file info
    st.info(f"📎 Uploaded: {pdf.name} ({pdf.size / 1024:.1f} KB)")

    # Save the uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    # Run tagging button
    if st.button("🚀 Run Tagging Pipeline", type="primary"):
        # Create placeholders
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        try:
            # Clean up previous run folders
            status_placeholder.info("🧹 Cleaning up previous outputs...")
            folders_to_clean = [
                "debug",
                "fused_tags",
                "layout_out",
                "lines_out",
                "promoted_tags",
                "structures"
            ]

            for folder in folders_to_clean:
                if os.path.exists(folder):
                    try:
                        shutil.rmtree(folder)
                        print(f"Cleaned up: {folder}")
                    except Exception as e:
                        print(f"Warning: Could not remove {folder}: {e}")

            # Also remove old output PDF if exists
            if os.path.exists("out_tagged.pdf"):
                try:
                    os.remove("out_tagged.pdf")
                    print("Cleaned up: out_tagged.pdf")
                except Exception as e:
                    print(f"Warning: Could not remove out_tagged.pdf: {e}")

            time.sleep(0.3)  # Brief pause to show cleanup message
            status_placeholder.info("⏳ Starting tagging pipeline...")

            # Run with real-time terminal output (simple approach)
            # This lets you see output in terminal immediately
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            # Start the process
            progress_bar.progress(20)
            status_placeholder.info("🔄 Processing PDF (check terminal for detailed logs)...")

            start_time = time.time()

            # Run the subprocess - NO TIMEOUT for very large files
            result = subprocess.run(
                ["python", "-u", "TagMyPDF.py", "temp.pdf"],
                env=env
                # No timeout! Process can run as long as needed
            )

            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            progress_bar.progress(90)

            # Check results
            if result.returncode == 0:
                status_placeholder.success(f"✅ PDF tagged successfully! (took {minutes}m {seconds}s)")
                progress_bar.progress(100)

                # Check if output file exists
                output_file = "out_tagged.pdf"
                if os.path.exists(output_file):
                    st.success("🎉 Your accessible PDF is ready!")

                    # Read the output file
                    with open(output_file, "rb") as f:
                        pdf_bytes = f.read()

                    # Download button
                    st.download_button(
                        label="⬇️ Download Tagged PDF",
                        data=pdf_bytes,
                        file_name=f"tagged_{pdf.name}",
                        mime="application/pdf"
                    )

                    # Show output size
                    st.info(f"Output file size: {len(pdf_bytes) / 1024:.1f} KB")
                    st.info(f"Processing time: {minutes} minutes {seconds} seconds")

                else:
                    st.error("❌ Output file not found. The tagging may have failed.")

            else:
                status_placeholder.error(f"❌ Tagging failed with error code {result.returncode}")
                progress_bar.progress(0)
                st.error("Check your terminal for error details.")

        except FileNotFoundError:
            status_placeholder.error("❌ TagMyPDF.py script not found in the current directory")
            progress_bar.progress(0)

        except Exception as e:
            status_placeholder.error(f"❌ An unexpected error occurred: {str(e)}")
            progress_bar.progress(0)
            st.exception(e)

        finally:
            # Cleanup temporary file
            if os.path.exists("temp.pdf"):
                try:
                    os.remove("temp.pdf")
                except:
                    pass
