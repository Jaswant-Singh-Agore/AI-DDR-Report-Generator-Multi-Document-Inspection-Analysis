import logging
from pathlib import Path

import fitz
import pdfplumber

from config import IMAGE_DIR

logger = logging.getLogger(__name__)


class PDFParser:

    # Pages to skip — cover/title pages contain logos, not findings
    SKIP_PAGES       = {1, 2}
    MIN_PHOTO_WIDTH  = 200
    MIN_PHOTO_HEIGHT = 200
    MAX_ASPECT_RATIO = 4.0   # eliminates wide banner logos

    def parse(self, pdf_path: str, source_label: str) -> dict:
        path = Path(pdf_path)
        if not path.exists():
            logger.error("File not found: %s", pdf_path)
            return {"status": "error: file not found", "source": source_label}

        logger.info("Parsing [%s]: %s", source_label, path.name)
        text_blocks = self._extract_text(pdf_path, source_label)
        tables      = self._extract_tables(pdf_path, source_label)
        images      = self._extract_images(pdf_path, source_label)
        logger.info("[%s] text=%d  tables=%d  images=%d", source_label, len(text_blocks), len(tables), len(images))

        return {
            "status":      "success",
            "source":      source_label,
            "file_name":   path.name,
            "text_blocks": text_blocks,
            "tables":      tables,
            "images":      images,
            "total_pages": self._page_count(pdf_path),
        }

    def _extract_text(self, pdf_path: str, source: str) -> list[dict]:
        doc, blocks = fitz.open(pdf_path), []
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").strip()
            if text:
                blocks.append({"type": "text", "source": source, "page": page_num + 1, "content": text})
        doc.close()
        return blocks

    def _extract_tables(self, pdf_path: str, source: str) -> list[dict]:
        blocks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables():
                    if not table:
                        continue
                    rows    = [" | ".join(str(c).strip() if c else "" for c in row) for row in table]
                    content = "\n".join(rows)
                    blocks.append({"type": "table", "source": source, "page": page_num, "content": f"[TABLE]\n{content}"})
        return blocks

    def _extract_images(self, pdf_path: str, source: str) -> list[dict]:
        doc        = fitz.open(pdf_path)
        stem       = Path(pdf_path).stem.lower().replace(" ", "_")
        images     = []
        seen_xrefs = set()

        for page_num in range(len(doc)):
            if (page_num + 1) in self.SKIP_PAGES:
                continue

            for idx, img_info in enumerate(doc[page_num].get_images(full=True)):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base = doc.extract_image(xref)
                except Exception as e:
                    logger.warning("Skipping image xref=%d: %s", xref, e)
                    continue

                w, h = base.get("width", 0), base.get("height", 0)

                if w < self.MIN_PHOTO_WIDTH or h < self.MIN_PHOTO_HEIGHT:
                    continue
                if (w / h if h > 0 else 999) > self.MAX_ASPECT_RATIO:
                    continue

                # Skip logo-like images with uniform dark or white backgrounds
                try:
                    from PIL import Image as PILImage
                    import io
                    small  = PILImage.open(io.BytesIO(base["image"])).convert("RGB").resize((50, 50))
                    pixels = list(small.getdata())
                    total  = len(pixels)
                    if sum(1 for r, g, b in pixels if r < 40 and g < 40 and b < 40) / total > 0.40:
                        continue
                    if sum(1 for r, g, b in pixels if r > 220 and g > 220 and b > 220) / total > 0.55:
                        continue
                except Exception:
                    pass

                filename = f"{stem}_p{page_num + 1}_i{idx + 1}.{base.get('ext', 'png')}"
                filepath = IMAGE_DIR / filename
                filepath.write_bytes(base["image"])

                images.append({
                    "type":     "image",
                    "source":   source,
                    "page":     page_num + 1,
                    "filename": filename,
                    "filepath": str(filepath),
                    "width":    w,
                    "height":   h,
                    "content":  f"[IMAGE] {filename}",
                })

        doc.close()
        return images

    def _page_count(self, pdf_path: str) -> int:
        doc = fitz.open(pdf_path)
        n   = len(doc)
        doc.close()
        return n

    def merge_all_blocks(self, *parse_results: dict) -> list[dict]:
        all_blocks = []
        for result in parse_results:
            all_blocks += result.get("text_blocks", [])
            all_blocks += result.get("tables", [])
            all_blocks += result.get("images", [])
        return all_blocks
