"""Cross-encoder reranker for improving retrieval quality."""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class Reranker:
    """Cross-encoder based reranker for better relevance scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, documents: list[Document], top_k: int = 8
    ) -> list[Document]:
        """Rerank documents based on query relevance.
        
        Args:
            query: User's question
            documents: List of retrieved documents
            top_k: Number of top documents to return after reranking
            
        Returns:
            Reranked list of top_k documents
        """
        if not documents:
            return []

        # Create query-document pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]

        # Get relevance scores from cross-encoder
        scores = self.model.predict(pairs)

        # Sort documents by score (descending) and take top_k
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs[:top_k]]
