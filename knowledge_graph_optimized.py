"""
Hierarchical Knowledge Graph Builder
Graph structure:  Document → Section → Chunk → Entity
Edges carry semantic relationship labels derived from co-occurrence context.
"""

import re
from typing import List, Dict, Any, Tuple
import networkx as nx
from collections import defaultdict


# ── Relation keyword → label map ─────────────────────────────────────────────
_REL_KEYWORDS: List[Tuple[str, str]] = [
    (r'\bpay(?:ment|s|able)?\b',          "PAYMENT_OF"),
    (r'\bterminat(?:e|ion|ed)\b',          "TERMINATES"),
    (r'\bsign(?:ed|atory|ature)?\b',       "SIGNED_BY"),
    (r'\bown(?:s|ed|ership)?\b',           "OWNED_BY"),
    (r'\bresponsib(?:le|ility)\b',         "RESPONSIBLE_FOR"),
    (r'\bconfidential(?:ity)?\b',          "BOUND_BY_CONFIDENTIALITY"),
    (r'\bindemnif(?:y|ication)\b',         "INDEMNIFIES"),
    (r'\bgovern(?:ed|ing|s)?\b',           "GOVERNED_BY"),
    (r'\beffective\b',                     "EFFECTIVE_ON"),
    (r'\bexpir(?:es?|ation|y)\b',          "EXPIRES_ON"),
    (r'\bdeliver(?:y|ed|able)?\b',         "DELIVERS"),
    (r'\blicens(?:e|ed|ing|or|ee)\b',      "LICENSES"),
    (r'\bemploy(?:ee|er|ment|ed)\b',       "EMPLOYED_BY"),
    (r'\bwarrant(?:y|ies|ed)\b',           "WARRANTS"),
]

_ENTITY_PATTERNS: Dict[str, List[str]] = {
    "ORG": [
        r'\b([A-Z][a-zA-Z&\s]{1,40}?)\s+(?:LLC|LLP|Inc\.?|Corp\.?|Corporation|Company|Co\.|Ltd\.?|Limited|Group|Associates|Partners|Services|Solutions|Technologies|Enterprises)\b',
        r'\b(Party\s+[A-Z])\b',
        r'(?i)(?:the\s+["\'])([A-Z][A-Za-z\s]{2,35})["\']',
    ],
    "PERSON": [
        r'\b((?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        r'(?:by|between|and|of)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)(?=\s*[,\(])',
    ],
    "DATE": [
        r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        r'\b(\d{4}-\d{2}-\d{2})\b',
    ],
    "MONEY": [
        r'(\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|k|M|B))?)',
        r'\b(USD\s*[\d,]+(?:\.\d{2})?)\b',
    ],
    "CLAUSE": [
        r'\b((?:Article|Section|Clause|Schedule|Exhibit|Annex)\s+[\dIVX]+(?:\.\d+)*(?:\([a-z]\))?)\b',
    ],
    "DURATION": [
        r'\b(\d+[\-\s](?:day|month|year|week)s?\s*(?:notice|period|term)?)\b',
    ],
    "PERCENTAGE": [
        r'\b(\d+(?:\.\d+)?%(?:\s*per\s+(?:annum|month|year))?)\b',
    ],
}


def _infer_relation(sentence: str) -> str:
    """Return the best relationship label found in a sentence, or RELATED_TO."""
    s = sentence.lower()
    for pattern, label in _REL_KEYWORDS:
        if re.search(pattern, s):
            return label
    return "RELATED_TO"


def _sentences_containing(text: str, entity: str) -> List[str]:
    """Return sentences from text that mention entity."""
    return [s.strip() for s in re.split(r'[.;]\s+', text) if entity.lower() in s.lower()]


class KnowledgeGraphBuilder:
    """
    Builds a 4-level hierarchical graph:

        DOCUMENT ──CONTAINS──▶ SECTION ──CONTAINS──▶ CHUNK ──HAS_ENTITY──▶ ENTITY
        ENTITY   ──<rel>──────▶ ENTITY   (co-occurrence with semantic label)
    """

    MAX_ENTITY_NODES  = 80
    MAX_ENTITY_EDGES  = 200
    CHUNKS_TO_SCAN    = 60   # scan all chunks for entity coverage

    def build_from_documents(self, documents: Dict[str, Any]) -> nx.DiGraph:
        graph = nx.DiGraph()

        for doc_name, doc_data in documents.items():
            self._add_document_subtree(graph, doc_name, doc_data)

        # After structural tree is built, add cross-entity semantic edges
        self._add_entity_edges(graph, documents)

        return graph

    # ── Level-1: Document node ────────────────────────────────────────────────

    def _add_document_subtree(self, graph: nx.DiGraph, doc_name: str, doc_data: Dict):
        doc_id = f"DOC::{doc_name}"
        graph.add_node(doc_id,
                       label=doc_name,
                       node_type="DOCUMENT",
                       level=0,
                       pages=doc_data.get("metadata", {}).get("total_pages", "?"))

        # Group chunks by their section metadata
        section_chunks: Dict[str, List] = defaultdict(list)
        for chunk in doc_data.get("chunks", []):
            sec = chunk.metadata.get("section", "Unknown")
            section_chunks[sec].append(chunk)

        # ── Level-2: Section nodes ────────────────────────────────────────────
        for sec_name, chunks in section_chunks.items():
            sec_id = f"SEC::{doc_name}::{sec_name}"
            hlevel = chunks[0].metadata.get("hierarchy_level", 1) if chunks else 1
            graph.add_node(sec_id,
                           label=sec_name,
                           node_type="SECTION",
                           level=hlevel,
                           doc=doc_name,
                           chunk_count=len(chunks))
            graph.add_edge(doc_id, sec_id, relation="CONTAINS")

            # ── Level-3: Chunk nodes ──────────────────────────────────────────
            for chunk in chunks:
                cidx   = chunk.metadata.get("chunk_index", 0)
                chunk_id = f"CHUNK::{doc_name}::{cidx}"
                wc     = chunk.metadata.get("word_count", len(chunk.page_content.split()))
                graph.add_node(chunk_id,
                               label=f"Chunk {cidx}",
                               node_type="CHUNK",
                               level=hlevel + 1,
                               doc=doc_name,
                               section=sec_name,
                               word_count=wc,
                               preview=chunk.page_content[:120])
                graph.add_edge(sec_id, chunk_id, relation="CONTAINS")

                # ── Level-4: Entity nodes from this chunk ─────────────────────
                ents = self._extract_entities_from_text(chunk.page_content, doc_name)
                for e in ents:
                    eid = f"ENT::{e['type']}::{e['text']}"
                    if eid not in graph:
                        graph.add_node(eid,
                                       label=e["text"],
                                       node_type=e["type"],
                                       level=hlevel + 2,
                                       source=doc_name)
                    graph.add_edge(chunk_id, eid, relation="HAS_ENTITY")

    # ── Cross-entity semantic edges ───────────────────────────────────────────

    def _add_entity_edges(self, graph: nx.DiGraph, documents: Dict[str, Any]):
        """For each chunk, find pairs of entities and connect with labelled edges."""
        edges_added = 0
        for doc_name, doc_data in documents.items():
            for chunk in doc_data.get("chunks", [])[:self.CHUNKS_TO_SCAN]:
                text  = chunk.page_content
                ents  = self._extract_entities_from_text(text, doc_name)
                eids  = [f"ENT::{e['type']}::{e['text']}" for e in ents if f"ENT::{e['type']}::{e['text']}" in graph]

                for i, e1 in enumerate(eids):
                    for e2 in eids[i + 1: i + 4]:
                        if e1 == e2 or graph.has_edge(e1, e2):
                            continue
                        # Find sentence(s) that contain both
                        lbl1 = graph.nodes[e1]["label"]
                        lbl2 = graph.nodes[e2]["label"]
                        sents = [s for s in re.split(r'[.;]\s+', text)
                                 if lbl1.lower() in s.lower() and lbl2.lower() in s.lower()]
                        rel = _infer_relation(sents[0]) if sents else "RELATED_TO"
                        graph.add_edge(e1, e2, relation=rel)
                        edges_added += 1
                        if edges_added >= self.MAX_ENTITY_EDGES:
                            return

    # ── Entity extraction helper ──────────────────────────────────────────────

    def _extract_entities_from_text(self, text: str, source: str) -> List[Dict]:
        entities: List[Dict] = []
        seen: set = set()
        for etype, pats in _ENTITY_PATTERNS.items():
            for pat in pats:
                for m in re.finditer(pat, text):
                    try:
                        val = (m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)).strip()
                    except IndexError:
                        val = m.group(0).strip()
                    val = re.sub(r'\s+', ' ', val)
                    if len(val) < 2 or len(val) > 60:
                        continue
                    if re.fullmatch(r'[\d\s\-\.\/,]+', val):
                        continue
                    key = val.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    entities.append({"text": val, "type": etype, "source": source})
        # Fallback: capitalised multi-word phrases
        if not entities:
            for val in re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text):
                key = val.lower()
                if key not in seen and 4 < len(val) < 50:
                    seen.add(key)
                    entities.append({"text": val, "type": "ENTITY", "source": source})
        return entities

    # ── Public helpers ────────────────────────────────────────────────────────

    def extract_entities(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compatibility shim used by comparison engine."""
        chunks = doc_data.get("chunks", [])[:self.CHUNKS_TO_SCAN]
        text   = " ".join(c.page_content for c in chunks)
        return self._extract_entities_from_text(text, doc_data.get("name", "unknown"))

    def get_graph_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        type_counts: Dict[str, int] = defaultdict(int)
        for node in graph.nodes():
            type_counts[graph.nodes[node].get("node_type", "UNKNOWN")] += 1
        return {
            "num_nodes":    graph.number_of_nodes(),
            "num_edges":    graph.number_of_edges(),
            "entity_types": dict(type_counts),
        }
