from llama_index.core import VectorStoreIndex, Document, StorageContext, QueryBundle
import os


class VectorRetriever:
    def __init__(self, cache_dir="./data/vector_stores"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.index = None

    def build_index(self, embeddings: list, metadata: list[dict], video_id: str):
        documents = [
            Document(
                text=m["text"],
                metadata={"start_time": m["start_time"], "end_time": m["end_time"]},
                embedding=emb,
            )
            for emb, m in zip(embeddings, metadata)
        ]

        self.index = VectorStoreIndex(
            documents, embed_model="local:BAAI/bge-base-en-v1.5"
        )
        self.index.storage_context.persist(persist_dir=f"{self.cache_dir}/{video_id}")

    def load_index(self, video_id: str):
        from llama_index.core import load_index_from_storage

        storage_context = StorageContext.from_defaults(
            persist_dir=f"{self.cache_dir}/{video_id}"
        )
        self.index = load_index_from_storage(
            storage_context, embed_model="local:BAAI/bge-base-en-v1.5"
        )

    def retrieve(self, query_embedding: list, top_k: int = 5) -> list[dict]:
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(QueryBundle(query_str="", embedding=query_embedding))

        results = []
        for node in nodes:
            results.append(
                {
                    "text": node.node.text,
                    "start_time": node.node.metadata["start_time"],
                    "end_time": node.node.metadata["end_time"],
                    "score": node.score,
                }
            )
        return results


# Test 1: Verify index creation
import os

retriever = VectorRetriever("./test_cache")
emb = [[0.1, 0.2], [0.3, 0.4]]
meta = [
    {"text": "test1", "start_time": 0, "end_time": 10},
    {"text": "test2", "start_time": 10, "end_time": 20},
]
retriever.build_index(emb, meta, "test_vid")
assert os.path.exists("./test_cache/test_vid/docstore.json")

# Test 2: Retrieval accuracy
query = [0.12, 0.22]  # Close to first embedding
results = retriever.retrieve(query, top_k=1)
print(results)
assert results[0]["text"] == "test1"
assert results[0]["score"] > 0.9  # High similarity expected
