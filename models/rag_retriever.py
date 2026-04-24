"""
RAG Retriever

Retrieval-Augmented Generation retriever that combines multiple search strategies:
- Semantic search (vector similarity)
- Keyword search (BM25)
- Graph traversal (knowledge graph)
- Experience matching (similar situations)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter
import re

from models.vector_store import Document, VectorStore, EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    documents: List[Document]
    query: str
    scores: List[float]
    context: str  # Assembled context for LLM
    sources: Dict[str, int] = field(default_factory=dict)  # doc_type -> count


class BM25Scorer:
    """Simple BM25 implementation for keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0
        self.n_docs: int = 0
        self.doc_term_freqs: List[Dict[str, int]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def index(self, documents: List[str]) -> None:
        """Build index from documents."""
        self.n_docs = len(documents)
        self.doc_freqs = {}
        self.doc_lengths = []
        self.doc_term_freqs = []

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))

            term_freqs = Counter(tokens)
            self.doc_term_freqs.append(dict(term_freqs))

            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        total_len = sum(self.doc_lengths)
        self.avgdl = total_len / self.n_docs if self.n_docs > 0 else 0

    def score(self, query: str, doc_idx: int) -> float:
        """Score a document against a query."""
        if doc_idx >= len(self.doc_term_freqs):
            return 0.0

        query_tokens = self._tokenize(query)
        doc_tf = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        score = 0.0
        for term in query_tokens:
            if term not in self.doc_freqs:
                continue

            # IDF
            df = self.doc_freqs[term]
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

            # TF
            tf = doc_tf.get(term, 0)
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))

            score += idf * tf_component

        return score

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Search for top-k documents."""
        scores = []
        for i in range(self.n_docs):
            score = self.score(query, i)
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class RAGRetriever:
    """
    RAG retrieval system combining multiple search strategies.

    Features:
    - Semantic search via vector similarity
    - Keyword search via BM25
    - Hybrid search combining both
    - Experience matching for similar situations
    - Context assembly for LLM consumption
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        knowledge_graph=None,
        experience_store=None,
        max_context_tokens: int = 2000,
    ):
        self.store = vector_store
        self.embeddings = embedding_service
        self.kg = knowledge_graph
        self.exp_store = experience_store
        self.max_context_tokens = max_context_tokens

        # BM25 index for keyword search
        self._bm25_indices: Dict[str, BM25Scorer] = {}
        self._bm25_docs: Dict[str, List[Document]] = {}

    def retrieve_for_query(
        self,
        query: str,
        context: dict = None,
        top_k: int = 5,
        strategy: str = "hybrid",
        collection: str = None,
        semantic_weight: float = 0.6,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge for a query.

        Args:
            query: Search query
            context: Additional context (e.g., current state)
            top_k: Number of results to return
            strategy: 'semantic', 'keyword', 'hybrid', 'experience'
            collection: Specific collection to search (None = all)
            semantic_weight: Weight for semantic vs keyword in hybrid (0-1)

        Returns:
            RetrievalResult with documents and assembled context
        """
        documents = []
        scores = []

        if strategy == "semantic":
            documents, scores = self._semantic_search(query, top_k, collection)

        elif strategy == "keyword":
            documents, scores = self._keyword_search(query, top_k, collection)

        elif strategy == "hybrid":
            documents, scores = self._hybrid_search(query, top_k, collection, semantic_weight)

        elif strategy == "experience":
            documents, scores = self._experience_search(query, context, top_k)

        else:
            # Default to hybrid
            documents, scores = self._hybrid_search(query, top_k, collection, semantic_weight)

        # Assemble context
        context_text = self.assemble_context(query, documents, scores)
        sources = self._count_sources(documents)

        return RetrievalResult(
            documents=documents,
            query=query,
            scores=scores,
            context=context_text,
            sources=sources,
        )

    def retrieve_similar_experiences(
        self,
        state: dict,
        action_type: str,
        k: int = 3,
    ) -> List[dict]:
        """
        Find similar past experiences for the current situation.

        Args:
            state: Current penetration test state
            action_type: Type of action being considered
            k: Number of experiences to return

        Returns:
            List of experience dictionaries with similarity scores
        """
        if not self.exp_store:
            return []

        # Build query from state
        query_parts = [action_type]
        if state:
            if state.get("target"):
                query_parts.append(str(state["target"]))
            if state.get("services"):
                query_parts.extend(state["services"])
            if state.get("vulnerabilities"):
                query_parts.extend(state["vulnerabilities"])

        query = " ".join(query_parts)

        # Search in experience collection
        result = self.retrieve_for_query(
            query,
            context=state,
            top_k=k,
            strategy="semantic",
            collection="experiences",
        )

        experiences = []
        for doc, score in zip(result.documents, result.scores):
            exp = {
                "content": doc.content,
                "score": score,
                "metadata": doc.metadata,
                "doc_type": doc.doc_type,
            }
            experiences.append(exp)

        return experiences

    def retrieve_attack_patterns(
        self,
        vulnerability: str,
        target_config: dict = None,
        k: int = 5,
    ) -> List[dict]:
        """
        Retrieve attack patterns for a specific vulnerability.

        Args:
            vulnerability: CVE ID or vulnerability name
            target_config: Target configuration (OS, services, etc.)
            k: Number of patterns to return

        Returns:
            List of attack patterns with context
        """
        # Build query
        query = vulnerability
        if target_config:
            if target_config.get("os"):
                query += f" {target_config['os']}"
            if target_config.get("services"):
                query += " " + " ".join(target_config["services"])

        # Search in CVE and technique collections
        result = self.retrieve_for_query(
            query,
            context=target_config,
            top_k=k,
            strategy="hybrid",
            collection=None,  # Search all
        )

        patterns = []
        for doc, score in zip(result.documents, result.scores):
            pattern = {
                "id": doc.id,
                "content": doc.content,
                "score": score,
                "type": doc.doc_type,
                "metadata": doc.metadata,
            }
            patterns.append(pattern)

        return patterns

    def retrieve_tool_usage(
        self,
        tool_name: str,
        scenario: str = "",
    ) -> str:
        """
        Retrieve tool usage documentation for a scenario.

        Args:
            tool_name: Name of the tool
            scenario: Scenario description

        Returns:
            Tool usage documentation string
        """
        query = f"{tool_name} {scenario}".strip()

        result = self.retrieve_for_query(
            query,
            top_k=3,
            strategy="keyword",
            collection="tool_docs",
        )

        if result.documents:
            return result.context
        return ""

    def assemble_context(
        self,
        query: str,
        documents: List[Document],
        scores: List[float] = None,
        max_tokens: int = None,
    ) -> str:
        """
        Assemble retrieved documents into context for LLM.

        Args:
            query: Original query
            documents: Retrieved documents
            scores: Relevance scores
            max_tokens: Maximum tokens (uses instance default if None)

        Returns:
            Assembled context string
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens

        if not documents:
            return ""

        # Rough token estimation: ~4 chars per token for Chinese/English mix
        max_chars = max_tokens * 4

        context_parts = []
        current_chars = 0

        # Add header
        header = f"=== 相关知识检索结果 (查询: {query}) ===\n"
        context_parts.append(header)
        current_chars += len(header)

        # Group by type
        by_type: Dict[str, List[Tuple[Document, float]]] = {}
        for i, doc in enumerate(documents):
            score = scores[i] if scores and i < len(scores) else 0.0
            doc_type = doc.doc_type or "text"
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append((doc, score))

        # Type labels
        type_labels = {
            "cve": "CVE漏洞信息",
            "technique": "ATT&CK技术",
            "tool_doc": "工具文档",
            "experience": "经验记录",
            "exploit": "利用代码",
            "text": "一般信息",
        }

        # Add each type
        for doc_type, items in by_type.items():
            type_label = type_labels.get(doc_type, doc_type)
            type_header = f"\n--- {type_label} ---\n"

            if current_chars + len(type_header) > max_chars:
                break

            context_parts.append(type_header)
            current_chars += len(type_header)

            for doc, score in items:
                doc_text = f"\n[{doc.id}] (相关度: {score:.2f})\n{doc.content}\n"

                if current_chars + len(doc_text) > max_chars:
                    break

                context_parts.append(doc_text)
                current_chars += len(doc_text)

        return "".join(context_parts)

    def _semantic_search(
        self,
        query: str,
        k: int,
        collection: str = None,
    ) -> Tuple[List[Document], List[float]]:
        """Perform semantic search using vector similarity."""
        query_embedding = self.embeddings.embed_query(query)

        documents = self.store.search(
            query_embedding=query_embedding,
            k=k,
            collection=collection,
        )

        scores = [doc.score for doc in documents]
        return documents, scores

    def _keyword_search(
        self,
        query: str,
        k: int,
        collection: str = None,
    ) -> Tuple[List[Document], List[float]]:
        """Perform keyword search using BM25."""
        # Get all documents from collection(s)
        if collection:
            collections = [collection]
        else:
            stats = self.store.get_collection_stats()
            collections = list(stats.keys())

        all_docs = []
        for coll in collections:
            # Ensure BM25 index exists for this collection
            if coll not in self._bm25_indices or not self._bm25_docs.get(coll):
                self.build_bm25_index(coll)

            bm25 = self._bm25_indices.get(coll)
            bm25_docs = self._bm25_docs.get(coll, [])

            if bm25 and bm25_docs:
                hits = bm25.search(query, k)
                for idx, score in hits:
                    if idx < len(bm25_docs):
                        doc = bm25_docs[idx]
                        doc.score = score
                        all_docs.append((doc, score))

        if not all_docs:
            return self._semantic_search(query, k, collection)

        all_docs.sort(key=lambda x: x[1], reverse=True)
        documents = [d for d, _ in all_docs[:k]]
        scores = [s for _, s in all_docs[:k]]
        return documents, scores

    def _hybrid_search(
        self,
        query: str,
        k: int,
        collection: str = None,
        semantic_weight: float = 0.6,
    ) -> Tuple[List[Document], List[float]]:
        """
        Combine semantic and keyword search.

        Args:
            query: Search query
            k: Number of results
            collection: Collection to search
            semantic_weight: Weight for semantic (1-semantic_weight for keyword)
        """
        # Get more results from each to allow for fusion
        k_expanded = min(k * 2, 20)

        # Semantic search
        sem_docs, sem_scores = self._semantic_search(query, k_expanded, collection)

        # Normalize semantic scores
        if sem_scores:
            max_sem = max(sem_scores) if max(sem_scores) > 0 else 1.0
            sem_scores_norm = [s / max_sem for s in sem_scores]
        else:
            sem_scores_norm = []

        # Keyword search (simplified - uses semantic for now)
        kw_docs, kw_scores = self._keyword_search(query, k_expanded, collection)

        # Normalize keyword scores
        if kw_scores:
            max_kw = max(kw_scores) if max(kw_scores) > 0 else 1.0
            kw_scores_norm = [s / max_kw for s in kw_scores]
        else:
            kw_scores_norm = []

        # Combine using Reciprocal Rank Fusion (RRF)
        doc_scores: Dict[str, Tuple[Document, float]] = {}

        for i, doc in enumerate(sem_docs):
            rrf_score = 1.0 / (i + 60)  # RRF constant
            combined = rrf_score * semantic_weight
            if doc.id in doc_scores:
                existing_doc, existing_score = doc_scores[doc.id]
                doc_scores[doc.id] = (existing_doc, existing_score + combined)
            else:
                doc_scores[doc.id] = (doc, combined)

        for i, doc in enumerate(kw_docs):
            rrf_score = 1.0 / (i + 60)
            combined = rrf_score * (1 - semantic_weight)
            if doc.id in doc_scores:
                existing_doc, existing_score = doc_scores[doc.id]
                doc_scores[doc.id] = (existing_doc, existing_score + combined)
            else:
                doc_scores[doc.id] = (doc, combined)

        # Sort by combined score
        sorted_results = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)[:k]

        if sorted_results:
            documents = [r[0] for r in sorted_results]
            scores = [r[1] for r in sorted_results]
        else:
            documents = []
            scores = []

        return documents, scores

    def _experience_search(
        self,
        query: str,
        context: dict,
        k: int,
    ) -> Tuple[List[Document], List[float]]:
        """Search for similar experiences."""
        if not self.exp_store:
            return [], []

        # Build enhanced query from context
        query_parts = [query]
        if context:
            if context.get("target"):
                query_parts.append(str(context["target"]))
            if context.get("action_type"):
                query_parts.append(context["action_type"])

        enhanced_query = " ".join(query_parts)

        return self._semantic_search(enhanced_query, k, collection="experiences")

    def _count_sources(self, documents: List[Document]) -> Dict[str, int]:
        """Count documents by source type."""
        sources: Dict[str, int] = {}
        for doc in documents:
            doc_type = doc.doc_type or "unknown"
            sources[doc_type] = sources.get(doc_type, 0) + 1
        return sources

    def build_bm25_index(self, collection: str = None) -> None:
        """
        Build BM25 index for keyword search.

        Args:
            collection: Specific collection to index (None = all)
        """
        stats = self.store.get_collection_stats()
        collections = [collection] if collection else list(stats.keys())

        for coll in collections:
            if coll in self._bm25_indices and coll in self._bm25_docs:
                continue  # Already indexed

            # Retrieve documents by doing a broad semantic search to seed the index
            # Use a generic query to get representative docs
            query_embedding = self.embeddings.embed_query(collection or "all")
            docs = self.store.search(query_embedding, k=200, collection=coll)

            if not docs:
                continue

            bm25 = BM25Scorer()
            bm25.index([d.content for d in docs])

            self._bm25_indices[coll] = bm25
            self._bm25_docs[coll] = docs
            logger.debug(f"Built BM25 index for '{coll}': {len(docs)} documents")


# Global instance
_rag_retriever: Optional[RAGRetriever] = None


def get_rag_retriever(
    vector_store: VectorStore = None,
    embedding_service: EmbeddingService = None,
    knowledge_graph=None,
    experience_store=None,
) -> RAGRetriever:
    """Get or create global RAG retriever instance."""
    global _rag_retriever

    if _rag_retriever is None:
        if vector_store is None:
            from models.vector_store import get_vector_store
            vector_store = get_vector_store()
        if embedding_service is None:
            from models.vector_store import get_embedding_service
            embedding_service = get_embedding_service()

        _rag_retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_service=embedding_service,
            knowledge_graph=knowledge_graph,
            experience_store=experience_store,
        )

    return _rag_retriever
