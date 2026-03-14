import logging
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import HF_TOKEN, BLIP_MODEL

logger = logging.getLogger(__name__)

HF_API_URL          = f"https://api-inference.huggingface.co/models/{BLIP_MODEL}"
CAPTION_SAMPLE_SIZE = 10
MAX_CAPTION_WORKERS = 3


class ImageCaptioner:

    def __init__(self):
        if not HF_TOKEN:
            logger.warning("HF_TOKEN not set — image captioning disabled.")
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    def caption_image(self, filepath: str) -> str:
        if not HF_TOKEN:
            return f"[Image: {Path(filepath).name}]"
        try:
            image_data = Path(filepath).read_bytes()
        except Exception:
            return f"[Image: {Path(filepath).name}]"

        for _ in range(2):
            try:
                r = requests.post(HF_API_URL, headers=self.headers, data=image_data, timeout=30)
                if r.status_code == 503:
                    time.sleep(15)
                    continue
                if r.status_code == 200:
                    result = r.json()
                    if isinstance(result, list) and result:
                        caption = result[0].get("generated_text", "")
                        if caption:
                            return caption
            except Exception as e:
                logger.warning("Caption error %s: %s", Path(filepath).name, e)

        return f"[Image from report: {Path(filepath).name}]"

    def _fallback_description(self, block: dict) -> str:
        source = block.get("source", "report").capitalize()
        page   = block.get("page", "?")
        fname  = Path(block.get("filepath", block.get("filename", "image"))).stem
        return f"[Image from {source} Report, Page {page}: {fname}]"

    def caption_all(self, image_blocks: list[dict], mode: str = "none") -> list[dict]:
        """
        Caption images according to mode:
          none   — no API calls, uses page+filename (default, recommended)
          sample — captions first CAPTION_SAMPLE_SIZE images via BLIP
          full   — captions all images (slow for 30-50 image PDFs)
        """
        total = len(image_blocks)

        if mode == "none" or not HF_TOKEN:
            for block in image_blocks:
                desc = self._fallback_description(block)
                block["content"] = f"[IMAGE] {desc}"
                block["caption"] = desc
            return image_blocks

        to_caption = image_blocks[:CAPTION_SAMPLE_SIZE] if mode == "sample" else image_blocks
        to_skip    = image_blocks[CAPTION_SAMPLE_SIZE:] if mode == "sample" else []

        for block in to_skip:
            desc = self._fallback_description(block)
            block["content"] = f"[IMAGE] {desc}"
            block["caption"] = desc

        def _caption_one(block):
            caption = self.caption_image(block.get("filepath", ""))
            block["content"] = f"[IMAGE CAPTION] {caption}"
            block["caption"] = caption
            return block

        with ThreadPoolExecutor(max_workers=MAX_CAPTION_WORKERS) as pool:
            futures = {pool.submit(_caption_one, b): b for b in to_caption}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    block = futures[future]
                    desc  = self._fallback_description(block)
                    block["content"] = f"[IMAGE] {desc}"
                    block["caption"] = desc

        return image_blocks
