

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import TranscriptEmbedder
from .llm import LangChainLLM, OpenRouterLLM
from .retriever import VectorRetriever
from .transcript import TranscriptRetriever


class RAGPipeline:
    def __init__(self):
        self.transcript_retriever = TranscriptRetriever()
        self.embedder = TranscriptEmbedder().get_model()
        self.llm = OpenRouterLLM()
        self.langchain_llm = LangChainLLM(llm=self.llm)
        self.retriever = VectorRetriever()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )

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
        """Answer the question using RAG"""
        try:
            self.retriever.load_index(video_id, self.embedder)
        except FileNotFoundError:
            return {
                "answer": f"Video {video_id} index does not exist. Please index first.",
                "sources": [],
                "question": question,
            }

        # Setup retriever
        retriever = self.retriever.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 chunks
        )

        # Prompt for LangChain v1.0+
        prompt_template = """You are a precise video Q&A assistant. Answer the question using ONLY the provided video transcript excerpts.

Every claim MUST include a time-stamped citation in format [MM:SS].

Video Transcript:
{context}

Question: {question}

Answer with citations (Hindi/English mix allowed):"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Format documents as string
        def format_docs(docs):
            formatted = ""
            for i, doc in enumerate(docs, 1):
                start_time = self._format_timestamp(doc.metadata.get("start", 0))
                formatted += f"\n[{start_time}] {doc.page_content}"
            return formatted

        # Manual RAG chain
        rag_chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | PROMPT
            | self.langchain_llm
            | StrOutputParser()
        )

        # Generate answer
        answer = rag_chain.invoke(question)

        # Extract sources
        sources = []
        for doc in retriever.invoke(question):
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



