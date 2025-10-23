import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

load_dotenv()

class VectorRetriever:
    def __init__(self, cache_dir="./data/vector_stores"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = None
        # This embedding function is needed for loading and searching.
        # It should be the same model as in TranscriptEmbedder.
        self.embedding_function = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m"))
        self.top_k = int(os.getenv("TOP_K", 5))

    def build_index(self, embeddings: np.ndarray, metadata: list[dict], video_id: str):
        """Build FAISS index from embeddings and persist."""
        texts = [meta['text'] for meta in metadata]
        text_embedding_pairs = list(zip(texts, embeddings.tolist()))

        # from_embeddings requires an embedding function to be passed, but it's only used for query embeddings later.
        self.vector_store = FAISS.from_embeddings(text_embedding_pairs, self.embedding_function, metadatas=metadata)
        self.vector_store.save_local(self.cache_dir / video_id)

    def load_index(self, video_id: str):
        """Load existing index."""
        index_path = self.cache_dir / video_id
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found for {video_id}")
        # allow_dangerous_deserialization is needed for FAISS with pickle
        self.vector_store = FAISS.load_local(index_path, self.embedding_function, allow_dangerous_deserialization=True)

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve top-k chunks with metadata. The query is a string, which will be embedded."""
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        results = []
        for doc, score in results_with_scores:
            result = {"text": doc.page_content, "score": score}
            result.update(doc.metadata)
            results.append(result)
        return results
