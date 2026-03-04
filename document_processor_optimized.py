"""
Optimized Hierarchical Document Processor
Fast, section-aware chunking with minimal overhead
"""

import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from collections import defaultdict


class HierarchicalDocumentProcessor:
    """Process documents with optimized hierarchical structure extraction"""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", "; ", ", ", " ", ""],
            length_function=len,
        )

    def process_document(self, documents: List[Document], doc_name: str) -> Dict[str, Any]:
        full_text = "\n\n".join(d.page_content for d in documents)
        full_text = self._clean_text(full_text)
        structure = self.extract_structure(full_text)
        chunks = self._create_chunks(documents, structure, doc_name)
        metadata = self._extract_metadata(full_text, len(documents))
        return {
            "name": doc_name,
            "chunks": chunks,
            "structure": structure,
            "metadata": metadata,
            "raw_documents": documents,
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r'\f', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text.strip()

    def extract_structure(self, text: str) -> Dict[str, Any]:
        structure: Dict[str, Any] = {
            "sections": [],
            "hierarchy": defaultdict(list),
            "toc": [],
        }
        patterns = [
            (r'^(ARTICLE|SECTION|CHAPTER)\s+([IVXLCDM]+|\d+)[:\.\s]+(.+?)$', 'article', 1),
            (r'^(\d+\.(?:\d+\.)*)\s+([A-Z][^\n]{3,80})$', 'numbered', None),
            (r'^\s*\(([a-z]|[ivx]+)\)\s+(.+?)$', 'lettered', 3),
            (r'^([A-Z][A-Z\s]{10,60})$', 'header', 2),
        ]
        counter: Dict[str, int] = defaultdict(int)
        for line_num, line in enumerate(text.split('\n')):
            line = line.strip()
            if not line or len(line) < 4:
                continue
            for pattern, stype, level in patterns:
                m = re.match(pattern, line, re.IGNORECASE)
                if not m:
                    continue
                if stype == 'article':
                    sid, title, hlevel = m.group(2), m.group(3).strip(), level
                elif stype == 'numbered':
                    sid = m.group(1)
                    title = m.group(2).strip()
                    hlevel = len(sid.split('.'))
                elif stype == 'lettered':
                    sid, title, hlevel = m.group(1), m.group(2).strip()[:100], level
                else:
                    counter[stype] += 1
                    sid = f"H{counter[stype]}"
                    title, hlevel = line.strip(), level
                sec = {
                    "id": sid, "title": title, "type": stype,
                    "level": hlevel, "line_number": line_num, "full_text": line,
                }
                structure["sections"].append(sec)
                structure["hierarchy"][hlevel].append(sec)
                if hlevel <= 2:
                    structure["toc"].append({"id": sid, "title": title, "level": hlevel})
                break
        return structure

    def _create_chunks(self, documents, structure, doc_name):
        base_chunks = self.splitter.split_documents(documents)
        sections_lower = [
            (s["title"].lower(), s["id"], s) for s in structure["sections"]
        ]
        result = []
        total = len(base_chunks)
        for idx, chunk in enumerate(base_chunks):
            text_lower = chunk.page_content[:300].lower()
            sec_info: Dict[str, Any] = {}
            for title_l, sid, sec in sections_lower:
                if title_l in text_lower or sid in chunk.page_content[:150]:
                    sec_info = sec
                    break
            chunk.metadata.update({
                "source": doc_name,
                "chunk_index": idx,
                "total_chunks": total,
                "section": sec_info.get("title", "Unknown"),
                "section_id": sec_info.get("id", "N/A"),
                "hierarchy_level": sec_info.get("level", 0),
                "section_type": sec_info.get("type", "N/A"),
                "word_count": len(chunk.page_content.split()),
            })
            result.append(chunk)
        return result

    @staticmethod
    def _extract_metadata(full_text: str, total_pages: int) -> Dict[str, Any]:
        sample = full_text[:6000]
        dates = re.findall(
            r'\b(?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            sample, re.IGNORECASE
        )
        parties = re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+'
            r'(?:LLC|Inc\.|Corp\.|Corporation|Company|Ltd\.|Limited)\b', sample
        )
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', sample)
        return {
            "total_pages": total_pages,
            "total_length": len(full_text),
            "word_count": len(full_text.split()),
            "dates_found": list(set(dates))[:10],
            "parties": list(set(parties))[:10],
            "monetary_amounts": list(set(amounts))[:20],
        }
