import logging
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pipeline.ddr_pipeline import DDRPipeline
from config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DDR Report Generator", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline = DDRPipeline()
jobs: dict[str, dict] = {}

# Auto-load existing index on startup — no re-ingest needed after code changes
if pipeline.retriever.load():
    pipeline.is_ready = True
    logger.info("Auto-loaded existing FAISS index.")


def _new_job(job_type: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"job_id": job_id, "type": job_type, "status": "running",
                    "step": "Starting…", "percent": 0, "result": None, "error": None}
    return job_id

def _update_job(job_id, step, percent=0):
    if job_id in jobs:
        jobs[job_id]["step"] = step
        jobs[job_id]["percent"] = percent

def _finish_job(job_id, result):
    if job_id in jobs:
        jobs[job_id].update({"status": "done", "step": "Complete", "percent": 100, "result": result})

def _fail_job(job_id, error):
    if job_id in jobs:
        jobs[job_id].update({"status": "error", "step": "Failed", "error": error})
    logger.error("[Job %s] %s", job_id, error)


def _run_ingest(job_id, insp_path, therm_path, caption_mode):
    try:
        result = pipeline.ingest(
            insp_path, therm_path,
            caption_mode=caption_mode,
            progress_cb=lambda s, p=0: _update_job(job_id, s, p),
        )
        if result.get("status") != "success":
            _fail_job(job_id, result.get("status", "Ingestion failed"))
        else:
            _finish_job(job_id, result)
    except Exception as e:
        _fail_job(job_id, str(e))
    finally:
        for p in [insp_path, therm_path]:
            Path(p).unlink(missing_ok=True)


def _run_generate(job_id):
    try:
        total = 7
        def section_cb(name, idx):
            _update_job(job_id, f"Generating: {name} ({idx}/{total})", int(idx / total * 100))
        result = pipeline.generate(save_to_disk=True, section_cb=section_cb)
        if result.get("status") != "success":
            _fail_job(job_id, result.get("message", "Generation failed"))
        else:
            _finish_job(job_id, {
                "report":          result["report"],
                "sections":        result["sections"],
                "citation_counts": result.get("citation_counts", {}),
                "timestamp":       result["timestamp"],
                "saved_to":        result.get("saved_to", ""),
            })
    except Exception as e:
        _fail_job(job_id, str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "DDR Report Generator"}


@app.post("/ingest")
async def ingest(
    inspection_pdf: UploadFile = File(...),
    thermal_pdf:    UploadFile = File(...),
    caption_mode:   str        = Form(default="none"),
):
    tmp_dir    = Path(tempfile.mkdtemp())
    insp_path  = str(tmp_dir / inspection_pdf.filename)
    therm_path = str(tmp_dir / thermal_pdf.filename)
    Path(insp_path).write_bytes(await inspection_pdf.read())
    Path(therm_path).write_bytes(await thermal_pdf.read())

    job_id = _new_job("ingest")
    threading.Thread(
        target=_run_ingest,
        args=(job_id, insp_path, therm_path, caption_mode),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "running", "caption_mode": caption_mode}


@app.post("/generate-ddr")
def generate_ddr():
    if not pipeline.is_ready:
        raise HTTPException(400, "Pipeline not ready. Ingest both PDFs first.")
    job_id = _new_job("generate")
    threading.Thread(target=_run_generate, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/job/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return jobs[job_id]


@app.get("/report")
def get_report():
    path = OUTPUT_DIR / "DDR_report.md"
    if not path.exists():
        raise HTTPException(404, "No report generated yet.")
    return FileResponse(str(path), media_type="text/markdown", filename="DDR_report.md")


@app.get("/status")
def get_status():
    return pipeline.get_status()


@app.delete("/reset")
def reset():
    pipeline.store.reset()
    pipeline.is_ready = False
    pipeline.ingested_docs = []
    pipeline.ingest_stats = {}
    jobs.clear()
    return {"status": "reset complete"}
