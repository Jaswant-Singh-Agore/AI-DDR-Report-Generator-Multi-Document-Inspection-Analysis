"""
Orchestrates 7-section DDR report generation using targeted RAG queries.

Key design decisions:
- All LLM context uses text+table chunks only (never image filenames)
- Thermal summary generated ONCE at property level — not repeated per area
  (thermal PDF has no room labels, so per-area thermal is misleading)
- Images rotated per area so each area gets different supporting photos
"""

import logging
import re
from datetime import datetime
from pathlib import Path

from retriever.retriever import DDRRetriever
from generator.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


DDR_SECTION_CONFIGS = {
    "summary": {
        "query": "defects damage dampness cracks leakage plumbing issues observed in property",
        "filter_source": "inspection",
        "instruction": (
            "List 4-6 key property issues as bullet points (dash -)."
            " Format: - [Area]: [issue description] [N]."
            " Plain English. Cite every fact. No headers, no dates, no introductions."
        ),
    },
    "root_cause": {
        "query": "root cause reason why defects damage moisture cracks leakage occurred",
        "filter_source": "inspection",
        "instruction": (
            "List probable root causes as bullet points (dash -)."
            " Format: - [Defect]: likely caused by [reason] [N]."
            " Only use facts from context. No headers, no dates."
            " If cause is uncertain write: Probable cause — Not confirmed."
        ),
    },
    "severity": {
        "query": "severity urgency critical risk level damage extent inspection findings",
        "filter_source": None,   # both sources — inspection findings + thermal context
        "instruction": (
            "First line exactly: Overall Severity: Low OR Moderate OR High.\n"
            "Then 3-5 bullet points (dash -) justifying the rating with citations [N].\n"
            "No headers, no dates. Evidence only."
        ),
    },
    "recommendations": {
        "query": "recommended repairs corrective actions maintenance remedial work required",
        "filter_source": "inspection",
        "instruction": (
            "Write three groups with these exact headings:\n"
            "Immediate (within 4 weeks)\n"
            "Short-term (1 to 3 months)\n"
            "Long-term (3 to 12 months)\n"
            "One bullet per action. Plain homeowner language. Cite [N]. No other headers."
        ),
    },
    "additional_notes": {
        "query": "additional observations survey conditions limitations access restrictions",
        "filter_source": "inspection",
        "instruction": (
            "List 2-4 additional observations not covered elsewhere as bullet points (dash -).\n"
            "Cite [N]. No headers, no dates."
            " If nothing to add, write: No additional observations noted."
        ),
    },
    "missing_info": {
        "query": "missing unclear incomplete information gaps not available in inspection report",
        "filter_source": "inspection",
        "instruction": (
            "List only GENUINELY missing or unclear information as bullet points (dash -).\n"
            "Do NOT repeat findings already present in the report.\n"
            "If nothing is missing write: Not Available. No headers, no dates."
        ),
    },
}


def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return ""
    lines = ["\n*Citations:*"]
    for c in citations:
        num     = c.get("citation_number", "?")
        source  = c.get("source", "?").capitalize()
        page    = c.get("page", "?")
        ctype   = c.get("type", "")
        preview = c.get("content_preview", "")[:80]
        tag     = f" ({ctype})" if ctype else ""
        lines.append(f"  [{num}] {source} Report — Page {page}{tag}: {preview}…")
    return "\n".join(lines)


def _fetch_text_chunks(retriever, query, top_k=8, filter_source=None):
    """Retrieve text+table chunks only — never image filenames.
    Text is fetched at higher k than table so rich narrative content
    is not crowded out by a single dominant table chunk.
    """
    text_chunks  = retriever.search(query=query, top_k=top_k,
                                    filter_source=filter_source, filter_type="text")
    table_chunks = retriever.search(query=query, top_k=3,
                                    filter_source=filter_source, filter_type="table")
    seen, chunks = set(), []
    for c in text_chunks + table_chunks:
        key = c.get("content", "")[:80]
        if key not in seen:
            seen.add(key)
            chunks.append(c)
        if len(chunks) >= top_k:
            break
    return chunks


class DDRGenerator:

    def __init__(self, retriever: DDRRetriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator

    def generate_section(self, section_key: str) -> tuple[str, list[dict]]:
        cfg    = DDR_SECTION_CONFIGS[section_key]
        chunks = _fetch_text_chunks(self.retriever, cfg["query"], top_k=8,
                                    filter_source=cfg.get("filter_source"))
        result    = self.generator.generate(query=cfg["instruction"], chunks=chunks)
        answer    = result.get("answer", "Not Available")
        citations = result.get("citations", [])
        return answer + _format_citations(citations), citations

    def generate_thermal_summary(self) -> str:
        """
        Generate ONE honest thermal summary for the whole property.
        Thermal PDF has no room labels — repeating readings per area is misleading.
        """
        therm_chunks = _fetch_text_chunks(
            self.retriever,
            "thermal hotspot coldspot temperature emissivity readings date survey",
            top_k=10,
            filter_source="thermal",
        )

        if not therm_chunks:
            return "> *Thermal survey data not available in the provided documents.*"

        result = self.generator.generate(
            query=(
                "List the thermal imaging readings as bullet points.\n"
                "For each unique reading include: Hotspot, Coldspot, Emissivity, Date.\n"
                "Cite every reading with [N]. No intro text, no headers.\n"
                "Skip duplicate temperature combinations — list each unique reading once."
            ),
            chunks=therm_chunks,
        )
        answer    = result.get("answer", "Not Available")
        citations = _format_citations(result.get("citations", []))
        return answer + citations

    def generate_area_section(self) -> str:
        """
        Section 2 structure:
          - Property-Wide Thermal Survey (once, honest)
          - Per area: inspection findings + unique rotating images
        """
        # ── Thermal summary block ──────────────────────────────────────────────
        thermal_summary = self.generate_thermal_summary()
        thermal_block = (
            "### 📊 Property-Wide Thermal Survey\n\n"
            "> *Note: The thermal report provides temperature readings across the property. "
            "Individual readings cannot be mapped to specific rooms without thermal image captions.*\n\n"
            + thermal_summary + "\n"
        )

        # ── Area discovery ─────────────────────────────────────────────────────
        area_chunks = _fetch_text_chunks(
            self.retriever,
            "rooms areas locations bathroom bedroom hall kitchen parking inspection findings",
            top_k=10,
        )

        area_result = self.generator.generate(
            query=(
                "List every distinct room or area found in the inspection report.\n"
                "Return ONLY the area names, one per line. No descriptions, no numbers."
            ),
            chunks=area_chunks,
        )

        raw_areas = area_result.get("answer", "")
        areas = []
        for line in raw_areas.splitlines():
            line = re.sub(r"\[.*?\]", "", line).strip("- •*\t1234567890.)( ").strip()
            if line and 2 < len(line) < 50:
                areas.append(line)
        areas = list(dict.fromkeys(areas))

        BAD = {"flat no", "n/a", "na", "no", "yes", "point", "photo",
               "area", "not available", "flat", "103", "203", "report"}
        ROOM_KW = ["bathroom", "bedroom", "hall", "kitchen", "parking", "balcony",
                   "toilet", "lobby", "staircase", "terrace", "external", "passage",
                   "corridor", "living", "dining", "room", "wall"]

        areas = [
            a for a in areas
            if a.lower() not in BAD
            and not a.lower().startswith("flat no")
            and (len(a.split()) >= 2 or any(kw in a.lower() for kw in ROOM_KW))
        ]
        # Shorten names like "Hall of Flat No. 103" -> "Hall"
        cleaned = []
        for a in areas:
            short = a
            for marker in [" of Flat", " of flat", " No. ", " no. "]:
                if marker in short:
                    short = short[:short.index(marker)].strip()
            if short and len(short) > 2:
                cleaned.append(short)
        areas = list(dict.fromkeys(cleaned))

        if not areas:
            return thermal_block + "\n_No distinct areas identified from the inspection report._\n"

        # ── Load all images for rotation ───────────────────────────────────────
        all_images = self.retriever.search_images("thermal inspection photo image", top_k=60)
        seen_paths, unique_images = set(), []
        for img in all_images:
            fp = img.get("filepath", "")
            if fp and fp not in seen_paths:
                seen_paths.add(fp)
                unique_images.append(img)

        section_parts = [thermal_block]
        img_cursor = 0

        for area in areas:
            logger.info("  [Area] %s", area)

            insp_chunks = _fetch_text_chunks(
                self.retriever,
                f"defects damage observations findings issues {area}",
                top_k=6,
                filter_source="inspection",
            )

            insp_result = self.generator.generate(
                query=(
                    f"List the physical inspection findings for the {area} as bullet points.\n"
                    f"Each bullet: one observation with citation [N].\n"
                    f"Only findings relevant to {area}. No headers, no dates."
                ),
                chunks=insp_chunks,
            )

            insp_answer = insp_result.get("answer", "Not Available")
            insp_cites  = _format_citations(insp_result.get("citations", []))

            # Pick 2 different images per area, rotating through pool
            if unique_images:
                area_imgs = [unique_images[(img_cursor + i) % len(unique_images)] for i in range(2)]
                img_cursor = (img_cursor + 2) % len(unique_images)

                img_lines = ["**Supporting Images:**\n"]
                for chunk in area_imgs:
                    filepath = chunk.get("filepath", "")
                    page     = chunk.get("page", "?")
                    source   = chunk.get("source", "report").capitalize()
                    caption  = f"Thermal image — Page {page}"
                    if filepath:
                        rel = Path("extracted_images") / Path(filepath).name
                        img_lines.append(f"![{caption}]({rel})")
                    img_lines.append(f"*{source} Report — Page {page}*")
                image_block = "\n".join(img_lines)
            else:
                image_block = "**Supporting Image:** Image Not Available"

            section_parts.append(
                f"### {area}\n\n"
                f"**Observation from Inspection Report:**\n"
                f"{insp_answer}\n{insp_cites}\n\n"
                f"{image_block}\n"
            )

        return "\n---\n\n".join(section_parts)

    def generate_full_report(self, section_cb=None) -> dict:
        logger.info("[DDR] Starting full report generation…")
        sections, citation_counts = {}, {}
        section_keys = list(DDR_SECTION_CONFIGS.keys())
        total = len(section_keys) + 1

        for idx, key in enumerate(section_keys, start=1):
            logger.info("  → %s", key)
            if section_cb:
                section_cb(key, idx)
            text, citations = self.generate_section(key)
            sections[key]        = text
            citation_counts[key] = len(citations)

        logger.info("  → area_observations")
        if section_cb:
            section_cb("area_observations", total)
        sections["area_observations"] = self.generate_area_section()

        report = _assemble_report(sections)
        logger.info("[DDR] Complete.")

        return {
            "report":          report,
            "sections":        sections,
            "citation_counts": citation_counts,
            "timestamp":       datetime.now().isoformat(),
        }


def _assemble_report(sections: dict) -> str:
    now = datetime.now().strftime("%d %B %Y, %H:%M")
    return f"""# Detailed Diagnostic Report (DDR)

**Report Generated:** {now}
**System:** DDR Report Generator
**Embeddings:** nomic-embed-text (Ollama — local) | **Retrieval:** FAISS Semantic Search
**LLM:** gemma3:latest (Ollama — local)

---

## 1. PROPERTY ISSUE SUMMARY

{sections.get('summary', 'Not Available')}

---

## 2. AREA-WISE OBSERVATIONS

{sections.get('area_observations', 'Not Available')}

---

## 3. PROBABLE ROOT CAUSE

{sections.get('root_cause', 'Not Available')}

---

## 4. SEVERITY ASSESSMENT

{sections.get('severity', 'Not Available')}

---

## 5. RECOMMENDED ACTIONS

{sections.get('recommendations', 'Not Available')}

---

## 6. ADDITIONAL NOTES

{sections.get('additional_notes', 'Not Available')}

---

## 7. MISSING OR UNCLEAR INFORMATION

{sections.get('missing_info', 'Not Available')}

---

*All findings are derived exclusively from the ingested Inspection and Thermal reports.
Citations [N] reference specific pages retrieved during generation.
This report should be reviewed by a qualified professional before any remedial work begins.*
"""