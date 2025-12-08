from langchain_huggingface import HuggingFaceEmbeddings


class TranscriptEmbedder:
    """Document embedder - embeds documents without prefix."""

    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},  # or 'cuda' if available
            encode_kwargs={"normalize_embeddings": True},
        )

    def get_model(self):
        return self.model


class BGEQueryEmbeddings:
    """Query embedder wrapper that adds 'query: ' prefix for BGE models.
    
    BGE models are trained with asymmetric retrieval - queries need a prefix
    for optimal performance while documents are embedded as-is.
    """

    def __init__(self, base_embeddings: HuggingFaceEmbeddings):
        self.base_embeddings = base_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the 'query: ' prefix."""
        return self.base_embeddings.embed_query(f"query: {text}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents without prefix (pass-through)."""
        return self.base_embeddings.embed_documents(texts)


