"""
Optimized Document Comparison Engine
Fast comparisons with visual-ready output
"""

from typing import List, Dict, Any
from collections import defaultdict
import difflib


class DocumentComparator:

    def compare_structure(self, doc1: Dict, doc2: Dict) -> Dict[str, Any]:
        s1 = {s["title"].lower() for s in doc1.get("structure", {}).get("sections", [])}
        s2 = {s["title"].lower() for s in doc2.get("structure", {}).get("sections", [])}
        return {
            "doc1_name": doc1["name"],
            "doc2_name": doc2["name"],
            "doc1_sections": len(s1),
            "doc2_sections": len(s2),
            "common_section_titles": list(s1 & s2),
            "unique_to_doc1": list(s1 - s2),
            "unique_to_doc2": list(s2 - s1),
        }

    def compare_clauses(self, doc1: Dict, doc2: Dict) -> List[Dict[str, Any]]:
        from clause_extractor import ClauseExtractor
        ex = ClauseExtractor()
        c1 = ex.extract_clauses({doc1["name"]: doc1})
        c2 = ex.extract_clauses({doc2["name"]: doc2})
        all_types = set(c1) | set(c2)
        return [
            {
                "clause_type": ct,
                "doc1_name": doc1["name"],
                "doc2_name": doc2["name"],
                "doc1_count": len(c1.get(ct, [])),
                "doc2_count": len(c2.get(ct, [])),
                "doc1_summary": self._summarize(c1.get(ct, [])),
                "doc2_summary": self._summarize(c2.get(ct, [])),
            }
            for ct in all_types
        ]

    def compare_entities(self, doc1: Dict, doc2: Dict) -> Dict[str, Dict[str, List[str]]]:
        from knowledge_graph_optimized import KnowledgeGraphBuilder
        kg = KnowledgeGraphBuilder()
        def _by_type(doc):
            res: Dict[str, List[str]] = defaultdict(list)
            for e in kg.extract_entities(doc):
                res[e["type"]].append(e["text"])
            return dict(res)
        return {doc1["name"]: _by_type(doc1), doc2["name"]: _by_type(doc2)}

    def compare_content_similarity(
        self, doc1: Dict, doc2: Dict, threshold: float = 0.6
    ) -> Dict[str, Any]:
        chunks1 = [c.page_content for c in doc1["chunks"][:20]]
        chunks2 = [c.page_content for c in doc2["chunks"][:20]]
        similar = []
        for i, ch1 in enumerate(chunks1):
            for j, ch2 in enumerate(chunks2):
                sim = difflib.SequenceMatcher(None, ch1.lower()[:400], ch2.lower()[:400]).ratio()
                if sim >= threshold:
                    similar.append({
                        "doc1_chunk_index": i,
                        "doc2_chunk_index": j,
                        "similarity": sim,
                        "doc1_section": doc1["chunks"][i].metadata.get("section", "N/A"),
                        "doc2_section": doc2["chunks"][j].metadata.get("section", "N/A"),
                    })
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return {
            "doc1_name": doc1["name"],
            "doc2_name": doc2["name"],
            "total_doc1_chunks": len(doc1["chunks"]),
            "total_doc2_chunks": len(doc2["chunks"]),
            "similar_chunks_count": len(similar),
            "similarity_threshold": threshold,
            "similar_chunks": similar[:10],
        }

    @staticmethod
    def _summarize(clauses: List[Dict]) -> str:
        if not clauses:
            return "None found"
        return f"Found in {clauses[0].get('section', 'N/A')}"
