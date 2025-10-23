import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class TranscriptEmbedder:
    def __init__(self, model_name=None, chunk_size=512, overlap=64):
        # Initialize text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("MAX_CHUNK_SIZE", chunk_size)),
            chunk_overlap=int(os.getenv("OVERLAP", overlap)),
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        )
        
        # Initialize embeddings using SentenceTransformer
        self.embeddings = SentenceTransformer(model_name or os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m"))
    
    def process(self, segments: list[dict], batch_size: int = 32) -> tuple[list, list[dict]]:
        """Convert segments to chunks with embeddings."""
        return self.process_with_timestamps(segments, batch_size=batch_size)   
    

    def process_with_timestamps(self, segments: list[dict], batch_size: int = 32) -> tuple[list, list[dict]]:
        """Enhanced version with timestamp tracking."""
        if not segments:
            return [], []

        # Filter out segments without text and build the full text and a character-to-segment map.
        full_text = ""
        char_to_segment_map = []
        valid_segments = []
        segment_index = 0
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            
            valid_segments.append(seg)
            
            if full_text:
                full_text += " "
                char_to_segment_map.append(segment_index - 1) # Space belongs to the previous segment.

            start_char_pos = len(full_text)
            full_text += text
            for _ in range(start_char_pos, len(full_text)):
                char_to_segment_map.append(segment_index)
            segment_index += 1

        if not full_text:
            return [], []

        # Split the text into chunks.
        text_chunks = self.splitter.split_text(full_text)

        # Map chunks back to timestamps.
        chunks_with_metadata = []
        current_pos = 0
        for chunk_text in text_chunks:
            start_char = full_text.find(chunk_text, current_pos)
            if start_char == -1:
                continue
            
            end_char = start_char + len(chunk_text)
            current_pos = end_char

            start_segment_idx = char_to_segment_map[start_char]
            end_segment_idx = char_to_segment_map[min(end_char - 1, len(char_to_segment_map) - 1)]

            start_time = valid_segments[start_segment_idx].get("start", 0.0)
            end_segment = valid_segments[end_segment_idx]
            end_time = end_segment.get("start", 0.0) + end_segment.get("duration", 0.0)

            chunks_with_metadata.append({
                "text": chunk_text,
                "start_time": start_time,
                "end_time": end_time,
            })

        if not chunks_with_metadata:
            return [], []

        # Generate embeddings for the chunks in batches.
        all_embeddings = []
        for i in range(0, len(chunks_with_metadata), batch_size):
            batch_chunks = [chunk["text"] for chunk in chunks_with_metadata[i:i+batch_size]]
            batch_embeddings = self.embeddings.encode(batch_chunks)
            all_embeddings.append(batch_embeddings)
        
        if not all_embeddings:
            return [], []

        embeddings = np.vstack(all_embeddings)

        return embeddings, chunks_with_metadata