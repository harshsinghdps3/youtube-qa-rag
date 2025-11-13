# 🎥 YouTube Video Q&A with RAG

This project is a web application built with Streamlit that allows you to "chat" with any YouTube video. It extracts the video's transcript, uses a Retrieval-Augmented Generation (RAG) pipeline to understand the content, and answers your questions using a local Large Language Model (LLM) powered by Ollama.

## ✨ Features

- **Chat with any YouTube Video**: Simply provide a YouTube video URL to get started.
- **Local LLM**: Uses a local LLM via Ollama, ensuring privacy and no API key costs.
- **Transcript Fallback**: If a video's built-in transcript is unavailable, it automatically falls back to transcribing the audio using OpenAI's Whisper model.
- **Efficient Caching**: Transcripts and vector indexes are cached locally, making subsequent queries on the same video instantaneous.
- **Timestamped Citations**: Answers are supported by timestamped excerpts from the video transcript, allowing you to verify the information.
- **Web Interface**: A simple and intuitive user interface powered by Streamlit.

## ⚙️ How It Works

The application follows a complete RAG pipeline to answer questions:

1.  **Transcript Retrieval**: When a YouTube URL is provided, the app first tries to fetch the pre-existing transcript using the `youtube-transcript-api`.
2.  **Whisper Fallback**: If no transcript is found, it downloads the video's audio using `yt-dlp` and transcribes it locally using `openai-whisper`.
3.  **Chunking**: The full transcript is split into smaller, overlapping text chunks using LangChain's `RecursiveCharacterTextSplitter`.
4.  **Embedding**: Each text chunk is converted into a numerical vector (embedding) using the `BAAI/bge-base-en-v1.5` sentence-transformer model.
5.  **Vector Indexing**: The embeddings are stored in a `FAISS` vector store, which allows for efficient similarity searches. This index is saved to disk for future use.
6.  **Question Answering**: When you ask a question:
    -   Your question is also converted into an embedding.
    -   The FAISS index is searched to find the most relevant transcript chunks (the "context").
    -   A prompt is constructed containing your question and the retrieved context.
    -   This prompt is sent to a local LLM (`qwen3:8b` via Ollama) to generate a final answer.
7.  **Display**: The generated answer and the source citations are displayed in the Streamlit web interface.

## 🛠️ Tech Stack

- **Application Framework**: Streamlit
- **Language**: Python
- **Core AI/ML**: LangChain, Sentence Transformers
- **LLM Provider**: Ollama (running `qwen3:8b` locally)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Transcript**: `youtube-transcript-api`, `openai-whisper`

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running on your system.

Make sure the Ollama server is running and you have pulled the required model:
```sh
ollama pull qwen3:8b
```

### Installation

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

Once the setup is complete, run the Streamlit app with the following command:

```sh
streamlit run app.py
```

Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).


