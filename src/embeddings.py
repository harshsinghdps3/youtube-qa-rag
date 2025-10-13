import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


class TranscriptEmbedder:
    def __init__(self, model_name=None, chunk_size=512, overlap=64):
        # Initialize text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("MAX_CHUNK_SIZE", chunk_size)),
            chunk_overlap=int(os.getenv("OVERLAP", overlap)),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        )
    
    def process(self, segments: list[dict]) -> tuple[list, list[dict]]:
        """Convert segments to chunks with embeddings."""
        # Combine segments into text with metadata
        full_text = " ".join([seg["text"] for seg in segments])
        
        # Split using LangChain
        text_chunks = self.splitter.split_text(full_text)
        
        # Create chunk metadata (simplified)
        chunks = [{"text": chunk} for chunk in text_chunks]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(text_chunks)
        
        return embeddings, chunks
