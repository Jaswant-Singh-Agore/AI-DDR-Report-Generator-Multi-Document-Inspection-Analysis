import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np
import requests

from config import FAISS_INDEX_PATH, METADATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN

logger = logging.getLogger(__name__)

OLLAMA_URL  = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM   = 768


class FAISSStore:
    """FAISS index with Ollama local embeddings (nomic-embed-text, 768-dim)."""

    def __init__(self):
        self.index:    faiss.IndexFlatIP | None = None
        self.metadata: list[dict]               = []

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        total   = len(texts)
        results = [None] * total
        done    = [0]

        def _embed_indexed(args):
            idx, text = args
            return idx, self._embed_one(text)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_embed_indexed, (i, t)): i for i, t in enumerate(texts)}
            for future in as_completed(futures):
                idx, vector  = future.result()
                results[idx] = vector
                done[0] += 1
                if done[0] % 100 == 0:
                    logger.info("  Embedded %d / %d…", done[0], total)

        return np.array(results, dtype=np.float32)

    def _embed_one(self, text: str) -> list[float]:
        text = text[:2000]
        for attempt in range(3):
            try:
                r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
                if r.status_code == 200:
                    return r.json()["embedding"]
                elif r.status_code == 404:
                    logger.error("Ollama model '%s' not found. Run: ollama pull %s", EMBED_MODEL, EMBED_MODEL)
                    return [0.0] * EMBED_DIM
                else:
                    time.sleep(2)
            except requests.exceptions.ConnectionError:
                logger.error("Cannot connect to Ollama. Run: ollama serve")
                time.sleep(3)
            except Exception as e:
                logger.warning("Embed error (attempt %d/3): %s", attempt + 1, e)
                time.sleep(2)

        logger.error("Embedding failed after 3 attempts — using zero vector.")
        return [0.0] * EMBED_DIM

    def build_index(self, blocks: list[dict]) -> dict:
        chunks = self._chunk_blocks(blocks)
        logger.info("Building FAISS index: %d chunks via Ollama…", len(chunks))

        embeddings = self.embed_texts([c["content"] for c in chunks])
        if embeddings.shape[0] == 0:
            return {"status": "error", "indexed_count": 0}

        faiss.normalize_L2(embeddings)
        self.index    = faiss.IndexFlatIP(EMBED_DIM)
        self.metadata = chunks
        self.index.add(embeddings)
        self._save()

        logger.info("FAISS index built: %d vectors.", self.index.ntotal)
        return {"status": "success", "indexed_count": self.index.ntotal}

    def search(self, query: str, top_k: int = 6, filter_source: str = None, filter_type: str = None) -> list[dict]:
        if self.index is None:
            if not self.load():
                return []

        q_embed = self.embed_texts([query])
        faiss.normalize_L2(q_embed)

        # Search large k when filters active — image chunks dominate thermal PDFs
        # and we need headroom to find text/table chunks after filtering
        total_vectors = self.index.ntotal
        k_search = min(total_vectors, max(top_k * 15, 80)) if (filter_source or filter_type) else min(total_vectors, top_k * 3)
        scores, indices = self.index.search(q_embed, k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx].copy()
            chunk["similarity_score"] = float(score)
            if filter_source and chunk.get("source") != filter_source:
                continue
            if filter_type and chunk.get("type") != filter_type:
                continue
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results

    def _save(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self) -> bool:
        if not Path(FAISS_INDEX_PATH).exists():
            return False
        try:
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH) as f:
                self.metadata = json.load(f)
            logger.info("Index loaded: %d vectors.", self.index.ntotal)
            return True
        except Exception as e:
            logger.error("Failed to load index: %s", e)
            return False

    def reset(self):
        self.index    = None
        self.metadata = []
        for p in [FAISS_INDEX_PATH, METADATA_PATH]:
            if Path(p).exists():
                Path(p).unlink()

    def get_stats(self) -> dict:
        if self.index is None:
            self.load()
        return {
            "total_vectors":  self.index.ntotal if self.index else 0,
            "embed_model":    EMBED_MODEL,
            "embed_provider": "Ollama (local)",
            "index_path":     FAISS_INDEX_PATH,
        }

    def _chunk_blocks(self, blocks: list[dict]) -> list[dict]:
        chunks = []
        for block in blocks:
            content = block.get("content", "").strip()
            btype   = block.get("type", "text")

            if len(content) < MIN_CHUNK_LEN:
                continue

            if btype in ("table", "image") or len(content) <= CHUNK_SIZE:
                chunks.append(block.copy())
                continue

            start = 0
            while start < len(content):
                end        = min(start + CHUNK_SIZE, len(content))
                chunk_text = content[start:end].strip()
                if len(chunk_text) >= MIN_CHUNK_LEN:
                    chunk            = block.copy()
                    chunk["content"] = chunk_text
                    chunks.append(chunk)
                start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks
