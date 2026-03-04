"""
Optimized Hybrid Retrieval Strategies
Reciprocal Rank Fusion for fast, accurate hybrid search
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict


class HybridRetriever:
    """Fast hybrid retriever using RRF to combine dense + sparse results"""

    def __init__(self, vector_store: FAISS, documents: List[Document], top_k: int = 4):
        self.vector_store = vector_store
        self.documents = documents
        self.top_k = top_k
        self._build_bm25()
        self._build_section_index()

    def _build_bm25(self):
        tokenized = [
            [t for t in doc.page_content.lower().split() if len(t) > 2]
            for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized)

    def _build_section_index(self):
        self.section_index: Dict[str, List[Document]] = defaultdict(list)
        for doc in self.documents:
            self.section_index[doc.metadata.get("section", "Unknown")].append(doc)

    # ── Public API ──────────────────────────────────────────────────────────

    def dense_search(self, query: str, k: int = None) -> List[Document]:
        k = k or self.top_k
        return self.vector_store.similarity_search(query, k=k)

    def sparse_search(self, query: str, k: int = None) -> List[Document]:
        k = k or self.top_k
        tokens = [t for t in query.lower().split() if len(t) > 2]
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_idx if scores[i] > 0]

    def hybrid_search(
        self, query: str, k: int = None,
        dense_weight: float = 0.65, sparse_weight: float = 0.35,
    ) -> List[Document]:
        """RRF-based hybrid search — fast and accurate"""
        k = k or self.top_k
        fetch = min(k * 2, 12)

        dense_res = self.dense_search(query, k=fetch)
        sparse_res = self.sparse_search(query, k=fetch)

        scores: Dict[str, float] = defaultdict(float)
        for rank, doc in enumerate(dense_res):
            scores[self._doc_id(doc)] += dense_weight / (60 + rank)
        for rank, doc in enumerate(sparse_res):
            scores[self._doc_id(doc)] += sparse_weight / (60 + rank)

        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
        id_map = {self._doc_id(d): d for d in self.documents}
        return [id_map[did] for did in sorted_ids[:k] if did in id_map]

    def hierarchical_search(self, query: str, k: int = None) -> List[Document]:
        k = k or self.top_k
        seed = self.dense_search(query, k=min(k, 4))
        if not seed:
            return self.hybrid_search(query, k=k)

        relevant_sections = {d.metadata.get("section", "Unknown") for d in seed}
        extra = []
        for sec in relevant_sections:
            extra.extend(self.section_index.get(sec, []))

        seen: set = set()
        combined: List[Document] = []
        for d in seed + extra:
            did = self._doc_id(d)
            if did not in seen:
                seen.add(did)
                combined.append(d)
        return combined[:k]

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _doc_id(self, doc: Document) -> str:
        return f"{doc.metadata.get('source','?')}_{doc.metadata.get('chunk_index',0)}"


class QueryExpander:
    def expand_query(self, query: str) -> List[str]:
        variations = [query]
        if "?" not in query:
            variations.append(query + "?")
        if query.endswith("?"):
            variations.append(query.rstrip("?"))
        return variations[:3]
