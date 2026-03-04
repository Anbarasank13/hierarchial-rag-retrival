"""
Optimized Knowledge Graph Builder
Fast entity extraction with clean Plotly-ready output
"""

import re
from typing import List, Dict, Any
import networkx as nx
from collections import defaultdict


class KnowledgeGraphBuilder:
    """Build compact, clear knowledge graphs"""

    MAX_ENTITIES = 50
    MAX_RELATIONSHIPS = 80
    CHUNKS_TO_SCAN = 12

    _PATTERNS: Dict[str, List[str]] = {
        "ORG": [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:LLC|Inc\.|Corp\.|Corporation|Company|Ltd\.|Limited)\b',
            r'\b(Party\s+[A-Z])\b',
        ],
        "DATE": [
            r'\b((?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{1,2},?\s+\d{4})\b',
        ],
        "MONEY": [r'(\$[\d,]+(?:\.\d{2})?)'],
        "CLAUSE": [r'\b((?:Article|Section|Clause)\s+[\dIVX]+(?:\.\d+)*)\b'],
    }

    def build_from_documents(self, documents: Dict[str, Any]) -> nx.DiGraph:
        graph = nx.DiGraph()
        all_entities: List[Dict] = []
        all_rels: List[tuple] = []

        for doc_name, doc_data in documents.items():
            ents = self.extract_entities(doc_data)
            rels = self._extract_relationships(doc_data, ents)
            all_entities.extend(ents)
            all_rels.extend(rels)

        # Keep top entities by frequency
        freq: Dict[str, int] = defaultdict(int)
        for e in all_entities:
            freq[e["text"]] += 1
        top_texts = {t for t, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:self.MAX_ENTITIES]}
        all_entities = [e for e in all_entities if e["text"] in top_texts]

        # Deduplicate relationships
        rel_set = list(set(all_rels))[:self.MAX_RELATIONSHIPS]

        self._populate_graph(graph, all_entities, rel_set)
        return graph

    def extract_entities(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = doc_data["chunks"][: self.CHUNKS_TO_SCAN]
        text = " ".join(c.page_content for c in chunks)
        entities: List[Dict] = []
        seen: set = set()

        for etype, pats in self._PATTERNS.items():
            for pat in pats:
                for m in re.finditer(pat, text, re.IGNORECASE):
                    val = (m.group(1) if m.lastindex else m.group(0)).strip()
                    if len(val) > 2 and val not in seen:
                        seen.add(val)
                        entities.append({"text": val, "type": etype, "source": doc_data["name"]})
        return entities

    def _extract_relationships(self, doc_data: Dict, entities: List[Dict]) -> List[tuple]:
        rels: List[tuple] = []
        for chunk in doc_data["chunks"][:5]:
            ct = chunk.page_content
            ents_here = [e["text"] for e in entities if e["text"] in ct][:6]
            for i, e1 in enumerate(ents_here):
                for e2 in ents_here[i + 1: i + 3]:
                    rels.append((e1, "RELATED_TO", e2))
        return rels

    def _populate_graph(self, graph: nx.DiGraph, entities, rels):
        for e in entities:
            if e["text"] not in graph:
                graph.add_node(e["text"], type=e["type"], source=e["source"])
        for e1, rel, e2 in rels:
            if e1 in graph and e2 in graph:
                graph.add_edge(e1, e2, relation=rel)

    def get_graph_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        type_counts: Dict[str, int] = defaultdict(int)
        for node in graph.nodes():
            type_counts[graph.nodes[node].get("type", "UNKNOWN")] += 1
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "entity_types": dict(type_counts),
        }
