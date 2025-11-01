from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings


class SimplifiedTranscriptEmbedder:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", chunk_size=512):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.embed_model = HuggingFaceEmbedding(model_name=self.model_name)

    def process(self, segments: list[dict]) -> tuple:
        # Convert segments to Documents
        documents = [
            Document(
                text=seg.get("text", "").strip(),
                metadata={
                    "start_time": seg.get("start", 0.0),
                    "duration": seg.get("duration", 0.0),
                },
            )
            for seg in segments
            if seg.get("text", "").strip()
        ]

        if not documents:
            return [], []

        # Parse and embed nodes
        node_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=64)
        nodes = node_parser.get_nodes_from_documents(documents)

        for node in nodes:
            node.embedding = self.embed_model.get_text_embedding(node.get_content())

        Settings.embed_model = self.embed_model
        VectorStoreIndex(nodes)

        # Extract embeddings and metadata
        embeddings = [node.embedding for node in nodes if node.embedding]
        chunks_metadata = [
            {
                "text": node.text,
                "start_time": node.metadata.get("start_time", 0.0),
                "end_time": node.metadata.get("start_time", 0.0)
                + node.metadata.get("duration", 0.0),
            }
            for node in nodes
            if node.embedding
        ]

        return embeddings, chunks_metadata


# Test
if __name__ == "__main__":
    test_segments = [
        {"text": "Test sentence one", "start": 0.0, "duration": 1.0},
        {"text": "Test sentence two", "start": 1.0, "duration": 1.0},
    ]

    embedder = SimplifiedTranscriptEmbedder()
    emb, meta = embedder.process(test_segments)

    print(f"Generated {len(emb)} embeddings")
    if emb:
        print(f"First embedding dimension: {len(emb[0])}")
