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
      

# minimal_test.py
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.documents import Document
import shutil

# Setup
vr = VectorRetriever("./temp_test")
emb = FakeEmbeddings(size=128)
docs = [Document(page_content=f"Doc {i}") for i in range(3)]

# Test
vr.build_index(docs, emb, "vid1")
assert vr.vector_store is not None, "Build failed"

vr.load_index("vid1", emb)
assert vr.vector_store is not None, "Load failed"

results = vr.vector_store.similarity_search("Doc", k=2)
assert len(results) == 2, f"Expected 2, got {len(results)}"

print("✅ All tests passed!")

