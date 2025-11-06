import streamlit as st
import time
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="YouTube Q&A with RAG", page_icon="🎥", layout="wide")

@st.cache_resource
def load_pipeline():
    """Cache pipeline initialization."""
    return RAGPipeline()

def main():
    st.title("🎥 YouTube Video Q&A Chatbot")
    st.markdown("Ask questions about any YouTube video and get answers with time-stamped citations.")

    pipeline = load_pipeline()

    # Video input
    video_url = st.text_input("Enter YouTube video URL:", placeholder="https://www.youtube.com/watch?v=...")

    if video_url:
        with st.spinner("Indexing video (first time only)..."):
            start = time.time()
            try:
                video_id = pipeline.index_video(video_url)
                index_time = time.time() - start
                st.success(f"✓ Video indexed in {index_time:.2f}s (cached for future queries)")

                # Question input
                question = st.text_input("Ask a question:", placeholder="What are the main points discussed?")

                if question:
                    with st.spinner("Generating answer..."):
                        qa_start = time.time()
                        result = pipeline.answer_question(video_id, question)
                        qa_time = time.time() - qa_start

                        st.markdown("### Answer")
                        st.markdown(result["answer"])

                        st.markdown(f"**Response time:** {qa_time:.2f}s")

                        # Show sources
                        with st.expander("📚 Source Citations"):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"**{i}. [{pipeline._format_timestamp(source['start'])} - {pipeline._format_timestamp(source['end'])}]**")
                                st.text(source["text"][:200] + "...")
                                st.markdown("---")

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()