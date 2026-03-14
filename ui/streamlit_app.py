import time
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DDR Report Generator", page_icon="🏠", layout="wide")
st.title("🏠 DDR Report Generator")
st.markdown(
    "Upload an **Inspection Report** and **Thermal Report** → "
    "generate a structured, citation-enforced **Detailed Diagnostic Report** "
    "using local RAG + Gemma3 (Ollama)."
)

try:
    h = requests.get(f"{API_URL}/health", timeout=3)
    if h.status_code == 200:
        st.success(f"API connected — {API_URL}")
    else:
        st.error(f"API returned status {h.status_code}")
except Exception:
    st.error(f"Cannot reach API at {API_URL}. Run: uvicorn app.api:app --reload")

st.divider()


def poll_job(job_id: str, status_text, progress_bar, poll_interval=3.0) -> dict:
    while True:
        try:
            r    = requests.get(f"{API_URL}/job/{job_id}", timeout=10)
            if r.status_code != 200:
                return {"status": "error", "error": f"Poll failed: {r.status_code}"}
            job     = r.json()
            status_text.markdown(f"⏳ **{job.get('step', 'Working…')}**")
            progress_bar.progress(max(job.get("percent", 0), 5))
            if job.get("status") in ("done", "error"):
                return job
        except Exception as e:
            status_text.markdown(f"Polling… ({e})")
        time.sleep(poll_interval)


tab1, tab2, tab3 = st.tabs(["📄 Upload & Ingest", "📋 Generate DDR Report", "📊 Pipeline Status"])


with tab1:
    st.header("Upload & Ingest Documents")
    st.markdown("Upload both PDFs. Text, tables, and images are extracted and indexed into FAISS.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Inspection Report")
        inspection_file = st.file_uploader("Upload Inspection PDF", type=["pdf"], key="insp")
        if inspection_file:
            st.success(f"✅ {inspection_file.name} ({inspection_file.size/1024:.1f} KB)")
    with col2:
        st.subheader("🌡️ Thermal Report")
        thermal_file = st.file_uploader("Upload Thermal PDF", type=["pdf"], key="therm")
        if thermal_file:
            st.success(f"✅ {thermal_file.name} ({thermal_file.size/1024:.1f} KB)")

    st.markdown("#### 🖼️ Image Captioning Mode")
    caption_mode = st.radio(
        "Choose how images are handled:",
        options=["none", "sample", "full"],
        index=0,
        horizontal=True,
        help=(
            "none — fastest, images referenced by page number (recommended)\n"
            "sample — captions first 10 images via BLIP (~1 min extra)\n"
            "full — captions ALL images (slow for large PDFs)"
        ),
    )
    mode_descriptions = {
        "none":   "⚡ **None** — Images extracted and referenced by page. No BLIP API calls. Recommended.",
        "sample": "🔬 **Sample** — First 10 images get BLIP captions (~1 min). Rest use page reference.",
        "full":   "🐢 **Full** — All images get BLIP captions. Can take 10+ minutes for 50 images.",
    }
    st.info(mode_descriptions[caption_mode])

    if st.button("🚀 Ingest Documents", type="primary", disabled=not (inspection_file and thermal_file)):
        with st.spinner("Uploading files…"):
            try:
                r = requests.post(
                    f"{API_URL}/ingest",
                    files={
                        "inspection_pdf": (inspection_file.name, inspection_file.getvalue(), "application/pdf"),
                        "thermal_pdf":    (thermal_file.name,    thermal_file.getvalue(),    "application/pdf"),
                    },
                    data={"caption_mode": caption_mode},
                    timeout=30,
                )
                if r.status_code != 200:
                    st.error(f"Failed to start ingestion: {r.text}")
                    st.stop()
                job_id = r.json()["job_id"]
            except Exception as e:
                st.error(f"Upload error: {e}")
                st.stop()

        st.info(f"Job `{job_id}` started — captioning mode: **{caption_mode}**")
        status_text  = st.empty()
        progress_bar = st.progress(5)
        job = poll_job(job_id, status_text, progress_bar)

        if job["status"] == "done":
            progress_bar.progress(100)
            status_text.empty()
            st.success("✅ Ingestion complete! Go to **Generate DDR Report** tab.")
            data = job.get("result", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Blocks",    data.get("total_blocks", 0))
            c2.metric("Vectors Indexed", data.get("indexed_vectors", 0))
            c3.metric("Caption Mode",    data.get("caption_mode", caption_mode))
            with st.expander("📋 Inspection details"):
                st.json(data.get("inspection", {}))
            with st.expander("🌡️ Thermal details"):
                st.json(data.get("thermal", {}))
        else:
            st.error(f"Ingestion failed: {job.get('error', 'Unknown error')}")

    elif not (inspection_file and thermal_file):
        st.info("Upload both PDFs above to enable ingestion.")


with tab2:
    st.header("Generate Detailed Diagnostic Report")
    st.markdown(
        "Runs **7 targeted RAG queries** — one per DDR section — plus a multi-pass "
        "area-wise section with a property-wide thermal summary."
    )
    st.info("⏱️ Generation takes 3–6 minutes with local Ollama (Gemma3).")

    if st.button("⚡ Generate DDR Report", type="primary"):
        with st.spinner("Starting generation…"):
            try:
                r = requests.post(f"{API_URL}/generate-ddr", timeout=15)
                if r.status_code == 400:
                    st.error("Pipeline not ready — ingest both PDFs first.")
                    st.stop()
                if r.status_code != 200:
                    st.error(f"Failed to start generation: {r.text}")
                    st.stop()
                job_id = r.json()["job_id"]
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.info(f"Job `{job_id}` started. Generating sections…")
        status_text  = st.empty()
        progress_bar = st.progress(5)
        job = poll_job(job_id, status_text, progress_bar)

        if job["status"] == "done":
            progress_bar.progress(100)
            status_text.empty()
            data   = job.get("result", {})
            report = data.get("report", "")

            st.success("✅ DDR Report generated successfully!")
            citation_counts = data.get("citation_counts", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("Sections",        len(data.get("sections", {})))
            m2.metric("Total Citations", sum(citation_counts.values()))
            m3.metric("Generated At",    data.get("timestamp", "")[:19].replace("T", " "))

            st.divider()
            st.markdown(report)
            st.divider()
            st.download_button("⬇️ Download DDR Report (.md)", data=report,
                               file_name="DDR_report.md", mime="text/markdown")

            with st.expander("🔍 Section-by-section breakdown"):
                sections = data.get("sections", {})
                labels = {
                    "summary":           "1. Property Issue Summary",
                    "area_observations": "2. Area-Wise Observations",
                    "root_cause":        "3. Probable Root Cause",
                    "severity":          "4. Severity Assessment",
                    "recommendations":   "5. Recommended Actions",
                    "additional_notes":  "6. Additional Notes",
                    "missing_info":      "7. Missing / Unclear Information",
                }
                for key, label in labels.items():
                    st.subheader(f"{label}  ({citation_counts.get(key, 0)} citations)")
                    st.markdown(sections.get(key, "Not Available"))
                    st.divider()
        else:
            st.error(f"Generation failed: {job.get('error', 'Unknown error')}")


with tab3:
    st.header("Pipeline Status")
    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        r = requests.get(f"{API_URL}/status", timeout=5)
        if r.status_code == 200:
            s        = r.json()
            is_ready = s.get("is_ready", False)
            st.success("✅ Pipeline ready") if is_ready else st.warning("⚠️ Not ready — ingest documents first")

            docs = s.get("ingested_docs", [])
            if docs:
                st.subheader("Ingested Documents")
                for d in docs:
                    st.markdown(f"- `{d}`")

            vs = s.get("vector_store", {})
            st.subheader("FAISS Vector Store")
            st.metric("Total Vectors", vs.get("total_vectors", 0))

            stats = s.get("ingest_stats", {})
            if stats:
                st.subheader("Document Breakdown")
                with st.expander("📋 Inspection"):
                    st.json(stats.get("inspection", {}))
                with st.expander("🌡️ Thermal"):
                    st.json(stats.get("thermal", {}))

            st.divider()
            if st.button("🗑️ Reset Pipeline", type="secondary"):
                requests.delete(f"{API_URL}/reset", timeout=10)
                st.success("Reset complete.")
                st.rerun()
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
