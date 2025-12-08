import os
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import  PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import TranscriptEmbedder, BGEQueryEmbeddings
from .llm import LangChainLLM, OllamaLLM
from .reranker import Reranker
from .retriever import VectorRetriever
from .transcript import TranscriptRetriever


class RAGPipeline:
    def __init__(self):
        self.transcript_retriever = TranscriptRetriever()
        self.embedder = TranscriptEmbedder().get_model()
        # Query embedder with prefix for better retrieval
        self.query_embedder = BGEQueryEmbeddings(self.embedder)
        self.llm = OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self.langchain_llm = LangChainLLM(llm=self.llm)
        self.retriever = VectorRetriever()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=150
        )
        # Cross-encoder reranker for better relevance
        self.reranker = Reranker()

    def index_video(self, video_url: str) -> str:
        """Index the video using transcript and embeddings"""
        video_id = self.transcript_retriever.extract_video_id(video_url)

        try:
            self.retriever.load_index(video_id, self.embedder)
            print(f"✓ Index already exists for {video_id}")
            return video_id
        except FileNotFoundError:
            print(f"Creating new index for {video_id}...")

        # Fetch transcript
        segments = self.transcript_retriever.get_transcript(video_url)

        # Create documents with time-stamped metadata
        docs = [
            Document(
                page_content=seg["text"],
                metadata={"start": seg["start"], "duration": seg.get("duration", 0)},
            )
            for seg in segments
        ]

        # Split into chunks
        split_docs = self.text_splitter.split_documents(docs)

        print(f"Total {len(split_docs)} chunks created")

        # Build FAISS index
        self.retriever.build_index(split_docs, self.embedder, video_id)
        print(f"✓ Index saved for {video_id}")

        return video_id

    def answer_question(self, video_id: str, question: str) -> dict:
        """Answer the question using improved RAG with MMR + reranking"""
        try:
            self.retriever.load_index(video_id, self.embedder)
        except FileNotFoundError:
            return {
                "answer": f"Video {video_id} index does not exist. Please index first.",
                "sources": [],
                "question": question,
            }

        # Step 1: Add query prefix for BGE model optimization
        prefixed_query = f"query: {question}"

        # Step 2: MMR search for diverse candidates (fetch 30, keep 15)
        print(f"🔍 Performing MMR search...")
        mmr_docs = self.retriever.mmr_search(
            query=prefixed_query,
            k=15,
            fetch_k=30,
            lambda_mult=0.5  # Balance between relevance and diversity
        )

        # Step 3: Cross-encoder reranking (keep top 8)
        print(f"🎯 Reranking {len(mmr_docs)} candidates...")
        reranked_docs = self.reranker.rerank(question, mmr_docs, top_k=8)

        # Prompt for LangChain v1.0+ - Enhanced for better answers
        prompt_template = """You are a helpful video Q&A assistant. Answer the question using the provided video transcript excerpts.

Guidelines:
1. Every claim MUST include a time-stamped citation in format [MM:SS].
2. If the exact term is not mentioned, explain related concepts from the transcript.
3. Provide detailed, comprehensive answers - not just one-liners.
4. If truly no relevant information exists, clearly state that.

Video Transcript:
{context}

Question: {question}

Provide a detailed answer with citations :"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Format reranked documents as string
        def format_docs(docs):
            formatted = ""
            for i, doc in enumerate(docs, 1):
                start_time = self._format_timestamp(doc.metadata.get("start", 0))
                formatted += f"\n[{start_time}] {doc.page_content}"
            return formatted

        context = format_docs(reranked_docs)

        # Generate answer using LLM
        prompt = PROMPT.format(context=context, question=question)
        answer = self.langchain_llm.invoke(prompt)

        # Extract sources from reranked docs
        sources = []
        for doc in reranked_docs:
            start_time = doc.metadata.get("start", 0)
            duration = doc.metadata.get("duration", 0)
            end_time = start_time + duration
            sources.append(
                {
                    "text": doc.page_content[:150] + "...",
                    "timestamp": self._format_timestamp(start_time),
                    "start": start_time,
                    "end": end_time,
                }
            )

        return {"answer": answer, "sources": sources, "question": question}

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
