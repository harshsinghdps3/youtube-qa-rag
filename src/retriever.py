import os
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document



class VectorRetriever:
    def __init__(self, cache_dir="./data/vector_stores"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.vector_store = None

    def build_index(self, documents: List[Document], embeddings, video_id: str):
        index_path = os.path.join(self.cache_dir, video_id)
        self.vector_store = FAISS.from_documents(documents, embeddings)
        self.vector_store.save_local(index_path)

    def load_index(self, video_id: str, embeddings):
        index_path = os.path.join(self.cache_dir, video_id)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index for video_id '{video_id}' not found.")
        self.vector_store = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

    def as_retriever(self, **kwargs):
        if not self.vector_store:
            raise Exception("Vector store not loaded. Please load an index first.")
        return self.vector_store.as_retriever(**kwargs)
      

