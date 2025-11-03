from langchain_huggingface import HuggingFaceEmbeddings


class TranscriptEmbedder:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cuda"},  # or 'cuda' if available
            encode_kwargs={"normalize_embeddings": True},
        )

    def get_model(self):
        return self.model


try:
    embedder = TranscriptEmbedder()
    model = embedder.get_model()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error: {e}")
    
embedding = model.embed_query("test")
assert len(embedding) == 768, f"Expected 768 dims, got {len(embedding)}"


import numpy as np
emb_array = np.array(embedding)
magnitude = np.linalg.norm(emb_array)
assert 0.99 <= magnitude <= 1.01, f"Not normalized: {magnitude}"
print(f"✓ Normalized (magnitude={magnitude:.3f})")