from langchain_huggingface import HuggingFaceEmbeddings


class TranscriptEmbedder:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},  # or 'cuda' if available
            encode_kwargs={"normalize_embeddings": True},
        )

    def get_model(self):
        return self.model


