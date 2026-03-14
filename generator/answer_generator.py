import logging
import re
import requests

from config import OLLAMA_LLM_URL, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a building diagnostics engineer. Write factual, concise DDR sections.

STRICT RULES:
1. Use ONLY facts from the CONTEXT. Never invent anything.
2. Cite every fact with [1], [2] etc. referencing the numbered context.
3. Never write [N] — if you have no citation, write: Not Available
4. Client-friendly plain English. No technical jargon.
5. No duplicate observations.
6. NO report headers, dates, inspector names — just answer the task directly.
7. Stop immediately when the task is done. No padding."""


class AnswerGenerator:

    def generate(self, query: str, chunks: list[dict]) -> dict:
        if not chunks:
            return self._empty_response("No relevant context found.")

        context_block, citation_map = self._build_context(chunks)
        prompt  = f"{context_block}\n\nTASK:\n{query}"
        answer  = self._call_llm(prompt)

        return {
            "answer":         answer,
            "citations":      self._extract_citations(answer, citation_map),
            "citation_score": self._score_citations(answer, len(chunks)),
            "chunks_used":    len(chunks),
            "status":         "success",
        }

    def _build_context(self, chunks: list[dict]) -> tuple[str, dict]:
        lines        = ["CONTEXT (cite using [N]):"]
        citation_map = {}

        for i, chunk in enumerate(chunks, start=1):
            source  = chunk.get("source", "unknown").capitalize()
            page    = chunk.get("page", "?")
            ctype   = chunk.get("type", "text")
            content = " ".join(chunk.get("content", "").strip().split())[:600]

            lines.append(f"\n[{i}] {source} Report — Page {page} ({ctype}):\n{content}")
            citation_map[i] = {
                "citation_number": i,
                "source":          chunk.get("source", "unknown"),
                "page":            page,
                "type":            ctype,
                "content_preview": " ".join(content[:200].split())[:100],
            }

        return "\n".join(lines), citation_map

    def _call_llm(self, user_message: str) -> str:
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}"

        for attempt in range(3):
            try:
                response = requests.post(
                    OLLAMA_LLM_URL,
                    json={
                        "model":  LLM_MODEL,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": LLM_TEMPERATURE,
                            "num_predict": LLM_MAX_TOKENS,
                        },
                    },
                    timeout=180,
                )
                if response.status_code == 200:
                    return response.json().get("response", "").strip()
                elif response.status_code == 404:
                    logger.error("Ollama model '%s' not found. Run: ollama pull %s", LLM_MODEL, LLM_MODEL)
                    return "Not Available — LLM model not found in Ollama."
                else:
                    logger.error("Ollama error %d: %s", response.status_code, response.text[:200])

            except requests.exceptions.ConnectionError:
                logger.error("Cannot connect to Ollama. Run: ollama serve")
                return "Not Available — Ollama not running."
            except requests.exceptions.Timeout:
                logger.warning("LLM timeout (attempt %d/3)", attempt + 1)
            except Exception as e:
                logger.error("LLM call failed (attempt %d/3): %s", attempt + 1, e)

        return "Not Available — LLM generation failed."

    def _extract_citations(self, answer: str, citation_map: dict) -> list[dict]:
        cited_nums = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
        return [citation_map[n] for n in sorted(cited_nums) if n in citation_map]

    def _score_citations(self, answer: str, total_chunks: int) -> float:
        if total_chunks == 0:
            return 0.0
        cited = len(set(re.findall(r"\[(\d+)\]", answer)))
        return round(min(cited / total_chunks, 1.0), 3)

    def _empty_response(self, reason: str) -> dict:
        return {"answer": reason, "citations": [], "citation_score": 0.0, "chunks_used": 0, "status": "no_context"}
