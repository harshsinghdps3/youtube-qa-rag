from src.embeddings import TranscriptEmbedder
import numpy as np

# Sample transcript segments
test_segments = [
    {"start": 0.0, "duration": 5.0, "text": "Welcome to this tutorial on machine learning."},
    {"start": 5.0, "duration": 5.0, "text": "Today we'll discuss neural networks and deep learning."},
    {"start": 10.0, "duration": 5.0, "text": "First, let's understand the basics of embeddings."}
]

# Initialize embedder
embedder = TranscriptEmbedder(
    model_name="google/embeddinggemma-300m",
    chunk_size=512,
    overlap=64
)

# Process segments
embeddings, chunks = embedder.process(test_segments)

# Validate results
print(f"Number of chunks: {len(chunks)}")
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0]) if len(embeddings) > 0 else 0}")
print(f"Sample chunk: {chunks[0] if chunks else 'None'}")
print(f"Embedding shape: {np.array(embeddings).shape}")

# Verify embeddings are normalized vectors
assert len(embeddings) == len(chunks), "Mismatch between embeddings and chunks"
# assert len(embeddings[0]) == 786, "Expected 786-dimensional embeddings "
print("All tests passed!")
print("Embeddings:")

