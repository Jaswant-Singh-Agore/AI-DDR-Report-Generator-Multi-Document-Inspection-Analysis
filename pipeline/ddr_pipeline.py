import logging
from pathlib import Path
from typing import Callable, Optional

from captioner.image_captioner import ImageCaptioner
from generator.answer_generator import AnswerGenerator
from generator.ddr_generator import DDRGenerator
from parser.pdf_parser import PDFParser
from retriever.retriever import DDRRetriever
from storage.faiss_store import FAISSStore
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


class DDRPipeline:

    def __init__(self):
        self.parser    = PDFParser()
        self.captioner = ImageCaptioner()
        self.store     = FAISSStore()
        self.retriever = DDRRetriever()
        self.generator = AnswerGenerator()
        self.ddr_gen   = DDRGenerator(self.retriever, self.generator)

        self.is_ready      = False
        self.ingested_docs: list[str] = []
        self.ingest_stats:  dict      = {}

    def ingest(
        self,
        inspection_pdf: str,
        thermal_pdf:    str,
        caption_mode:   str                = "none",
        progress_cb:    Optional[Callable] = None,
    ) -> dict:

        def _p(msg, pct=0):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg, pct)

        _p("Parsing inspection report…", 5)
        insp_result = self.parser.parse(inspection_pdf, source_label="inspection")
        _p("Parsing thermal report…", 15)
        therm_result = self.parser.parse(thermal_pdf, source_label="thermal")

        if insp_result.get("status") != "success":
            return {"status": f"Inspection parse failed: {insp_result.get('status')}"}
        if therm_result.get("status") != "success":
            return {"status": f"Thermal parse failed: {therm_result.get('status')}"}

        total_imgs = len(insp_result.get("images", [])) + len(therm_result.get("images", []))
        _p(f"Found {total_imgs} images — captioning mode: {caption_mode}", 20)

        for result_set in [insp_result, therm_result]:
            imgs = result_set.get("images", [])
            if imgs:
                _p(f"Processing {len(imgs)} images from {result_set.get('source', '')} report…", 30)
                result_set["images"] = self.captioner.caption_all(imgs, mode=caption_mode)

        _p("Merging all content blocks…", 45)
        all_blocks = self.parser.merge_all_blocks(insp_result, therm_result)

        _p("Building FAISS vector index…", 55)
        self.store.reset()
        index_result = self.store.build_index(all_blocks)

        if index_result.get("status") != "success":
            return {"status": "FAISS indexing failed"}

        _p("Loading retriever…", 92)
        self.retriever.load()
        self.is_ready = True

        self.ingested_docs = [insp_result["file_name"], therm_result["file_name"]]
        self.ingest_stats  = {
            "inspection": {
                "file":        insp_result["file_name"],
                "pages":       insp_result["total_pages"],
                "text_blocks": len(insp_result["text_blocks"]),
                "tables":      len(insp_result["tables"]),
                "images":      len(insp_result["images"]),
            },
            "thermal": {
                "file":        therm_result["file_name"],
                "pages":       therm_result["total_pages"],
                "text_blocks": len(therm_result["text_blocks"]),
                "tables":      len(therm_result["tables"]),
                "images":      len(therm_result["images"]),
            },
            "total_blocks":    len(all_blocks),
            "indexed_vectors": index_result.get("indexed_count", 0),
            "caption_mode":    caption_mode,
        }

        _p("Ingestion complete!", 100)
        return {"status": "success", **self.ingest_stats}

    def generate(
        self,
        save_to_disk: bool             = True,
        section_cb:   Optional[Callable] = None,
    ) -> dict:
        if not self.is_ready:
            if not self.retriever.load():
                return {"status": "error", "message": "No documents ingested. Call /ingest first."}
            self.is_ready = True

        result = self.ddr_gen.generate_full_report(section_cb=section_cb)

        if save_to_disk:
            out_path = OUTPUT_DIR / "DDR_report.md"
            out_path.write_text(result["report"], encoding="utf-8")
            result["saved_to"] = str(out_path)

        result["status"]        = "success"
        result["ingested_docs"] = self.ingested_docs
        return result

    def get_status(self) -> dict:
        return {
            "is_ready":      self.is_ready,
            "ingested_docs": self.ingested_docs,
            "ingest_stats":  self.ingest_stats,
            "vector_store":  self.retriever.get_stats(),
        }
